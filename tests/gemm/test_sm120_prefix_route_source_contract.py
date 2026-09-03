"""CPU-only source contracts for the SM120 fused-prefix A/B seam.

The seam spans three files: a C++ ladder header, the TVM-FFI prefix gate, and the
producer dispatcher. Python mirrors the ladder because the M537 native producer
reads a prepacked LoRA-down matrix that the cuBLASLt prefix does not, so a silent
disagreement between the two sides would corrupt results rather than slow them
down. These tests pin the agreement without a GPU.
"""

import re
from pathlib import Path

import pytest

from flashinfer.gemm.svdquant_sm120_routes import (
    sm120_producer_variants,
    PREFIX_ROUTE_OVERRIDE_ENV,
    SM120_PREFIX_ROUTE_LADDER,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LADDER_HEADER = REPO_ROOT / "include/flashinfer/gemm/svdquant_sm120_prefix_route.h"
PRODUCER_HEADER = (
    REPO_ROOT / "include/flashinfer/gemm/nvfp4_smooth_quantize_lora_down_sm120.cuh"
)
LINEAR_SOURCE = REPO_ROOT / "csrc/nvfp4_svdquant_gemm_cutlass_sm120.cu"

_LADDER_ENTRY = re.compile(
    r"\{\s*(\d+),\s*(\d+),\s*Route::k(\w+),\s*(true|false),\s*(true|false),"
    r"\s*(true|false)\s*\}"
)
_ROUTE_NAMES = {"Cublaslt": "cublaslt", "Native": "native", "NativeAlt": "native_alt"}
# The ladder lives in flashinfer::gemm; callers outside it must say so in full.
_QUALIFIED = "flashinfer::gemm::svdquant_sm120_prefix_route::"

# Mirrors KernelLaunchGeometry in the producer header.
_SF_VEC_SIZE = 16
_RANK = 32
_LARGE_M_DOWN_WARPS = _RANK // 8
_SM120_110SM_SM_COUNT = 110


def _parse_cpp_ladder() -> dict[tuple[int, int], tuple[str, frozenset[str]]]:
    text = LADDER_HEADER.read_text()
    body = text.split("inline constexpr LadderEntry kLadder[] = {", 1)[1].split("};", 1)[0]
    ladder: dict[tuple[int, int], tuple[str, frozenset[str]]] = {}
    for match in _LADDER_ENTRY.finditer(body):
        m, k, route, cublaslt, native, native_alt = match.groups()
        admitted = {
            name
            for name, allowed in (
                ("cublaslt", cublaslt),
                ("native", native),
                ("native_alt", native_alt),
            )
            if allowed == "true"
        }
        key = (int(m), int(k))
        assert key not in ladder, f"duplicate C++ ladder entry {key}"
        ladder[key] = (_ROUTE_NAMES[route], frozenset(admitted))
    return ladder


def _large_m_geometry(m: int, k: int, block_threads: int, tile_m: int, tile_k: int):
    """Recompute the producer header's compile-time launch contract in Python."""
    quant_threads = block_threads - _LARGE_M_DOWN_WARPS * 32
    sf_cols_per_tile = tile_k // _SF_VEC_SIZE
    padded_m = (m + 127) // 128 * 128
    grid_blocks = (padded_m + tile_m - 1) // tile_m
    prefetch_lines = k * _RANK * 2 // 128
    shared_bytes = (
        2 * tile_m * (tile_k + 8) * 2 + 2 * tile_k * 2 + 2 * (tile_m * sf_cols_per_tile if k >= 8192 else 1)
    )
    assert k % tile_k == 0
    assert tile_k % (4 * _SF_VEC_SIZE) == 0
    assert 0 < tile_m <= 80 and tile_m % 16 == 0
    assert block_threads % 32 == 0 and quant_threads > 0
    assert (tile_m * sf_cols_per_tile) % quant_threads == 0
    assert k < 8192 or grid_blocks * block_threads >= prefetch_lines
    assert shared_bytes <= 48 * 1024
    return grid_blocks, shared_bytes


def test_cpp_and_python_prefix_ladders_agree() -> None:
    # Given: the route decision is made in C++ and the L2T layout in Python.
    # When: both ladders are compared entry by entry.
    # Then: neither side can drift into reading a layout the other did not write.
    assert _parse_cpp_ladder() == dict(SM120_PREFIX_ROUTE_LADDER)


def test_both_sides_read_the_same_environment_variable() -> None:
    # Given: one diagnostic toggle drives both sides of the seam.
    # When: the C++ variable name is read out of the ladder header.
    # Then: it matches the name the Python route lookup consults.
    header = LADDER_HEADER.read_text()
    match = re.search(r'inline constexpr char kEnvName\[\] = "([^"]+)"', header)
    assert match is not None
    assert match.group(1) == PREFIX_ROUTE_OVERRIDE_ENV


def _parse_measured_gate() -> set[tuple[int, int]]:
    """Recover the literal cuBLASLt-prefix shape set from the TVM-FFI gate."""
    source = LINEAR_SOURCE.read_text()
    gate = source.split("bool const measured_cublaslt_lora_down =", 1)[1].split(";", 1)[0]
    groups = re.split(r"x\.size\(0\) == (\d+)", gate)[1:]
    shapes: set[tuple[int, int]] = set()
    for index in range(0, len(groups), 2):
        m = int(groups[index])
        for k in re.findall(r"x\.size\(1\) == (\d+)", groups[index + 1]):
            shapes.add((m, int(k)))
    return shapes


def test_measured_gate_matches_the_ladder_default_column() -> None:
    # Given: the seam only ever modifies the measured gate, never replaces it.
    # When: the literal gate is parsed and compared with the ladder's defaults.
    # Then: the two agree, which is also what the C++ reconciliation enforces.
    ladder_defaults = {
        key for key, (default, _) in SM120_PREFIX_ROUTE_LADDER.items() if default == "cublaslt"
    }
    assert _parse_measured_gate() == ladder_defaults


def test_linear_gate_applies_the_seam_through_the_reconciliation() -> None:
    # Given: a drifting gate would hand a prepacked matrix to the wrong prefix.
    # When: the TVM-FFI entry point is inspected.
    # Then: the measured gate feeds the reconciliation rather than being replaced.
    source = LINEAR_SOURCE.read_text()
    gate = source.split("bool const use_cublaslt_lora_down =", 1)[1].split(";", 1)[0]
    assert _QUALIFIED + "resolve_cublaslt_prefix" in gate
    assert "measured_cublaslt_lora_down" in gate
    assert '#include "flashinfer/gemm/svdquant_sm120_prefix_route.h"' in source
    # And: an override-selected shape can actually enter the cuBLASLt prefix.
    admission = source.split("void nvfp4_quantize_smooth_lora_down_cublaslt_sm120(", 1)[
        1
    ].split("cuBLASLt SM120 LoRA-down requires an admitted exact", 1)[0]
    assert _QUALIFIED + "uses_cublaslt_prefix" in admission


def test_linear_source_call_sites_are_fully_qualified() -> None:
    # Given: the TVM-FFI entry points live in namespace torch_ext, which the TU
    # opens only after closing flashinfer::gemm, so unqualified lookup of the
    # ladder namespace does not compile there.
    # When: every prefix-route reference in the TU is located.
    # Then: each one carries the full flashinfer::gemm:: qualification.
    source = LINEAR_SOURCE.read_text()
    assert source.index("}  // namespace gemm") < source.index("namespace torch_ext {")
    references = [
        match.start() for match in re.finditer(r"svdquant_sm120_prefix_route::", source)
    ]
    assert references, "the seam must still be wired into the TU"
    for start in references:
        assert source[start - len("flashinfer::gemm::") : start] == "flashinfer::gemm::"




def test_no_rung_admits_a_route_it_cannot_run() -> None:
    # Given: every alternate rung this ladder carried regressed on SM120 (110 SMs), and
    # every default is still the measured winner. The ladder is no longer closed
    # though: the cuBLASLt-routed shapes admit their native rung again, because
    # each of them now has a compiled native producer and a route nothing can
    # select cannot be re-measured.
    # When: each rung's admitted set is read.
    # Then: it always contains its own default, never contains the retired
    # native_alt, and any extra route it admits is one the C++ dispatcher can
    # actually reach -- admission is a claim that something will run.
    for key, (default, admitted) in SM120_PREFIX_ROUTE_LADDER.items():
        assert default in admitted, key
        assert "native_alt" not in admitted, key
        if "native" in admitted:
            m, k = key
            # Was: look for this shape's own launch_*_kernel<M, K, ...> in the
            # dispatcher body. The runtime-M/K conversion replaced the per-shape
            # instantiations with one launcher taking the geometry as arguments,
            # so a shape is backed when it computes a geometry, not when a
            # template bearing its numbers appears in the source.
            assert sm120_producer_variants(m, k), (
                f"{key} admits native but computes no producer geometry"
            )


def test_native_alt_grammar_survives_its_own_retirement() -> None:
    # Given: keeping the spelling stable across rounds is worth more than pruning
    # the enumerator, because campaign scripts are written against the grammar.
    # When: the parser and the admission check are read out of the ladder header.
    # Then: native_alt still parses, and is rejected by admission rather than syntax.
    header = LADDER_HEADER.read_text()
    assert 'if (text == "native_alt")' in header
    assert not any(
        "native_alt" in admitted for _, admitted in SM120_PREFIX_ROUTE_LADDER.values()
    )


@pytest.mark.parametrize(
    "tile_m,expected_blocks",
    [(32, 64), (16, 128)],
)
def test_row57_producer_underfill_was_real_but_not_the_cause(
    tile_m: int,
    expected_blocks: int,
) -> None:
    # Given: the underfill hypothesis for row 57 -- a 32-row tile leaves 64 blocks
    # on a 110-SM device -- was arithmetically sound and still lost by 7.7%.
    # When: both geometries are re-derived from the producer launch contract.
    # Then: each is legal and only the 16-row tile fills the device, which pins
    # the refutation to the K14336 per-block LoRA-down re-read, not to legality.
    grid_blocks, shared_bytes = _large_m_geometry(1935, 14336, 384, tile_m, 256)
    assert grid_blocks == expected_blocks
    assert shared_bytes <= 48 * 1024
    assert (grid_blocks >= _SM120_110SM_SM_COUNT) is (tile_m == 16)


def test_row57_is_the_only_underfilled_batch4_producer_grid() -> None:
    # Given: underfill is the one structural producer asymmetry in Batch 4.
    # When: every Batch-4 large-M producer grid is re-derived.
    # Then: only row 57's geometry leaves the device in a partial wave, so no
    # other row can be attributed to the same cause without new evidence.
    underfilled = {
        (m, k)
        for m, k, block_threads, tile_m, tile_k in (
            (1935, 14336, 384, 32, 256),
            (6913, 5120, 192, 32, 128),
            (6913, 5376, 192, 32, 128),
            (6913, 7168, 256, 32, 256),
            (6913, 14336, 384, 32, 256),
        )
        if _large_m_geometry(m, k, block_threads, tile_m, tile_k)[0] < _SM120_110SM_SM_COUNT
    }
    assert underfilled == {(1935, 14336)}
