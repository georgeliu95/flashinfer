"""CPU-only contracts for the SM120 linear route manifest.

Routes only. Tactics are not manifest data: they are measurements, and the
only thing allowed to produce one is the tuner;
``test_no_module_ships_a_shape_keyed_tactic_table`` is the guard that keeps it
that way. The tests after it hold the manifest against the C++ producer
dispatcher in both directions, because admission and instantiation live in
different languages and drifted apart once already.
"""

import ast
import re
from pathlib import Path
from types import MappingProxyType

import pytest

from flashinfer.gemm.svdquant_sm120_routes import (
    sm120_producer_variants,
    sm120_variant_packs_l2t,
    SM120_FAMILY_M537,
    SM120_FUSED_LINEAR_MK,
    SM120_LINEAR_ROUTE_ABI_OVERRIDES,
    SM120_LINEAR_ROUTE_V6_MKR,
    sm120_linear_route_abi_version,
)


def test_route_manifest_containers_are_immutable() -> None:
    # Given: routing metadata is shared by dispatch, tests, and tuning tools.
    # When: its public container types are inspected.
    # Then: no consumer can mutate process-wide route decisions in place.
    assert isinstance(SM120_LINEAR_ROUTE_ABI_OVERRIDES, MappingProxyType)
    assert isinstance(SM120_FUSED_LINEAR_MK, frozenset)
    assert isinstance(SM120_LINEAR_ROUTE_V6_MKR, frozenset)


@pytest.mark.parametrize(
    "m,k,rank,expected_version",
    [
        (256, 12288, 32, 5),
        (73984, 7168, 32, 6),
        (537, 5376, 32, 7),
        (82752, 5376, 32, 8),
        (537, 14336, 32, 9),
    ],
)
def test_route_abi_version_matches_manifest(
    m: int,
    k: int,
    rank: int,
    expected_version: int,
) -> None:
    # Given: one shape from each persisted route-ABI generation.
    # When: its current cache namespace is selected.
    actual_version = sm120_linear_route_abi_version(m, k, rank)
    # Then: only runner-set changes advance beyond the retained v5 base.
    assert actual_version == expected_version


def test_route_abi_override_manifest_has_no_base_version_entries() -> None:
    # Given: the compact mapping contains only exceptions to the v5 base.
    # When: every override version is inspected.
    override_versions = set(SM120_LINEAR_ROUTE_ABI_OVERRIDES.values())
    # Then: redundant v5 entries cannot drift from the default independently.
    assert override_versions == {6, 7, 8, 9}


def _m537_mixed_producer_shapes(header: str) -> set[tuple[int, int]]:
    """(M, K) whose native producer is the M537 mixed kernel, which indexes the
    prepacked L2T layout. M=537 now carries two producer families, so which
    layout a shape needs can no longer be inferred from its M."""
    body = header[
        header.index(
            "inline cudaError_t nvfp4_smooth_quantize_lora_down_sm120("
        ) : header.index("// Direct-only n16 diagnostic entry")
    ]
    return {
        (537, int(match.group(1)))
        for match in re.finditer(r"launch_m537_\w*kernel<\s*(\d+)", body)
    }


def test_the_two_formerly_pinned_problems_keep_their_fused_admission() -> None:
    # Given: 32760x1536x1536 and 32760x8960x1536 reached the fused runner only
    # because they carried a pinned tactic, and pinned prefixes were unioned into
    # admission separately from SM120_FUSED_LINEAR_MK.
    # When: the pins are deleted.
    # Then: their prefix has to survive in the admission set on its own, or the
    # tuner sees only Sm120CutlassLinearRunner for both -- measured at 19% on
    # 32760x1536x1536.
    assert (32760, 1536) in SM120_FUSED_LINEAR_MK


def _shape_keyed_tactic_tables(source: str) -> list[str]:
    """Names bound to a module-level mapping keyed by a four-int (m,n,k,rank)."""
    found: list[str] = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        else:
            continue
        value = node.value
        if value is None:
            continue
        # MappingProxyType({...}) and frozenset({...}) both wrap the literal.
        while isinstance(value, ast.Call) and value.args:
            value = value.args[0]
        if not isinstance(value, ast.Dict):
            continue
        for key in value.keys:
            if not isinstance(key, ast.Tuple) or len(key.elts) != 4:
                break
            if not all(
                isinstance(e, ast.Constant) and isinstance(e.value, int)
                for e in key.elts
            ):
                break
        else:
            if value.keys:
                found.extend(targets)
    return found


def test_no_module_ships_a_shape_keyed_tactic_table() -> None:
    # Given: a tactic is a measurement of one kernel on one device, and the only
    # thing entitled to produce one is the tuner. A cache is fine; a source
    # literal is not, because nothing re-checks it and nothing dates it. One went
    # stale by 4.02% and was found only by measuring the whole candidate field
    # (artifacts/selector-metric-20260821).
    # When: every module of the gemm package is parsed.
    # Then: no module-level mapping is keyed by an exact (M, N, K, rank) problem,
    # which is the shape such a table has to have to be looked up at dispatch.
    root = Path(__file__).resolve().parents[2]
    package = root / "flashinfer" / "gemm"
    # Every SVDQuant module under flashinfer/gemm. The CuTeDSL kernels this
    # guard used to also scan are gone with that route; what replaced them as
    # the tempting place to write such a table is the SM120 CUTLASS backend,
    # which owns the route manifest and the runner set.
    offenders: dict[str, list[str]] = {}
    scanned = sorted(package.rglob("*svdquant*.py"))
    covered = {path.name for path in scanned}
    for expected in (
        "svdquant_sm120_routes.py",
        "svdquant_sm120_cutlass.py",
        "gemm_svdquant.py",
    ):
        assert expected in covered, (
            f"{expected} is no longer scanned; the guard passes vacuously for it"
        )
    for path in scanned:
        names = _shape_keyed_tactic_tables(path.read_text())
        if names:
            offenders[str(path.relative_to(root))] = names
    assert offenders == {}, (
        f"a shape-keyed table is back in source: {offenders}. Tactics come from "
        "the tuner and persist through AutoTuner.save_configs / load_configs."
    )


def _has_native_producer(m: int, k: int) -> bool:
    """Can the producer actually launch this (M, K)?

    This used to be read off per-shape kernel instantiations in the C++ header
    (_native_producer_shapes below). The runtime-M/K conversion replaced those
    with one launcher that takes the geometry as arguments -- which is what let
    110 instantiations collapse to 34 -- so scanning for them now reports their
    absence rather than a coverage gap. What has to hold instead is that the
    shape computes a legal geometry at all.

    Reachability is deliberately NOT folded in here: the FFI still carries its
    own shape guard, and the tests assert against _ffi_guard_shapes() right
    beside this, so a shape that computes a geometry but cannot get through the
    entry point still fails.
    """
    return bool(sm120_producer_variants(m, k))


def _native_producer_shapes(header: str) -> set[tuple[int, int]]:
    """Every exact (M, K) the C++ producer dispatcher can actually launch.

    Read off the kernel instantiations rather than the ``if`` conditions: an
    instantiation is what has to exist for the launch to succeed, and its
    template arguments carry M and K directly.
    """
    body = header[
        header.index(
            "inline cudaError_t nvfp4_smooth_quantize_lora_down_sm120("
        ) : header.index("// Direct-only n16 diagnostic entry")
    ]
    shapes: set[tuple[int, int]] = set()
    for pattern in (
        r"launch_large_m_kernel<\s*(\d+)\s*,\s*(\d+)",
        r"launch_small_m_kernel<\s*(\d+)\s*,\s*(\d+)",
        r"nvfp4_smooth_quantize_lora_down_large_m_sm120_kernel<\s*(\d+)\s*,\s*(\d+)",
    ):
        for match in re.finditer(pattern, body):
            shapes.add((int(match.group(1)), int(match.group(2))))
    # These two launchers template on K alone; their M is fixed by the namespace.
    for match in re.finditer(r"launch_m512_kernel<\s*(\d+)", body):
        shapes.add((512, int(match.group(1))))
    for match in re.finditer(r"launch_m537_\w*kernel<\s*(\d+)", body):
        shapes.add((537, int(match.group(1))))
    # The base kernel takes no template arguments at all; its shape is the pair
    # of constants the namespace declares.
    shapes.add((64, 3072))
    return shapes


def _producer_header() -> str:
    root = Path(__file__).resolve().parents[2]
    return (
        root
        / "include"
        / "flashinfer"
        / "gemm"
        / "nvfp4_smooth_quantize_lora_down_sm120.cuh"
    ).read_text()


def _ffi_guard_shapes() -> set[tuple[int, int]]:
    """Every exact (M, K) the fused-prefix FFI entry point will accept.

    A third list, in a third language: ``nvfp4_quantize_smooth_lora_down_sm120_impl``
    rejects anything outside it before the dispatcher is ever reached, so a
    producer that exists and is admitted is still unreachable if this check does
    not name it.
    """
    root = Path(__file__).resolve().parents[2]
    source = (root / "csrc" / "nvfp4_smooth_quantize_sm100.cu").read_text()
    # Anchored on a whitespace-tolerant pattern rather than a literal slice of
    # the expression: where this condition wraps is clang-format's decision, not
    # a contract, and it moved once already when the file was first formatted.
    # What the test is actually about is which shapes the guard names.
    head = re.search(r"TVM_FFI_ICHECK\(\s*\(\(m\s*==", source)
    assert head is not None, "the SM120 K12 shape guard is no longer in the FFI source"
    start = head.start()
    end = source.index('<< "x must be one of the measured SM120 K12 shapes";', start)
    guard = source[start:end]
    shapes: set[tuple[int, int]] = set()
    # Clauses are `(m == A [|| m == B ...]) && k == C`, wrapped and line-broken.
    for clause in re.finditer(
        r"((?:m\s*==\s*\d+\s*(?:\|\||\s)*)+?)\)?\s*&&\s*k\s*==\s*(\d+)", guard
    ):
        k = int(clause.group(2))
        for m in re.findall(r"m\s*==\s*(\d+)", clause.group(1)):
            shapes.add((int(m), k))
    return shapes


def test_the_ffi_guard_admits_exactly_the_compiled_producers() -> None:
    # Given: the fused prefix is gated by three lists in three languages -- this
    # FFI check, the C++ producer dispatcher, and the Python admission set.
    # When: the FFI check is compared with the set of compiled producers.
    # Then: they name the same shapes. Adding a producer and its Python
    # admission while leaving this check alone produces a route the tuner will
    # pick and the FFI will refuse, which is how the six swept producers first
    # failed (artifacts/producer-gap-20260822).
    guard = _ffi_guard_shapes()
    unbacked = sorted(shape for shape in guard if not _has_native_producer(*shape))
    assert unbacked == [], f"{unbacked} pass the FFI check but have no producer"
    # The other direction. It used to read as "we compiled a producer for a
    # shape the FFI will not admit", which was a real waste when producers were
    # per-shape templates. Every legal (M, K) computes a geometry now, so that
    # difference would be every shape outside the guard and would say nothing.
    # Scoped to what this backend actually claims to support, the waste it
    # guarded against is still a thing worth refusing.
    claimed = set(SM120_FUSED_LINEAR_MK)
    unreachable = sorted(
        shape
        for shape in claimed
        if _has_native_producer(*shape) and shape not in guard
    )
    assert unreachable == [], f"{unreachable} have a producer the FFI check will reject"


def test_every_fused_admission_has_a_prefix_that_can_run() -> None:
    # Given: admitting (M, K) to SM120_FUSED_LINEAR_MK offers the fused runner to
    # the tuner, and the fused route reaches its prefix in one of two ways -- a
    # compiled native producer, or the cuBLASLt prefix the ladder can select.
    # When: the admission set is checked against both.
    # Then: nothing is admitted that has neither, which would make the C++
    # dispatcher return cudaErrorInvalidValue for a route the tuner can pick.
    # Reachable = the FFI lets it in AND the shape computes a geometry to launch.
    # The second term used to be the ladder's cuBLASLt column, for shapes that
    # had the prefix but no native producer. The prefix is a producer family now
    # and is offered exactly where the native families are, so a shape reachable
    # by one is reachable by the other and the ladder term has nothing to add.
    reachable = {shape for shape in _ffi_guard_shapes() if _has_native_producer(*shape)}
    orphans = sorted(set(SM120_FUSED_LINEAR_MK) - reachable)
    assert orphans == [], (
        f"{orphans} are admitted to the fused route but have no native producer "
        "and no cuBLASLt prefix rung"
    )


def test_every_native_producer_is_admitted() -> None:
    # Given: a producer instantiation costs compile time and binary size, and
    # earns nothing unless some shape can route to it.
    # When: the compiled set is compared with the admission set.
    # Then: no producer is stranded. This is the direction that actually drifted:
    # nineteen shapes ran the unfused prefix because their (M, K) had no
    # producer at all, which no test could see from the Python side alone.
    native = _native_producer_shapes(_producer_header())
    stranded = sorted(native - set(SM120_FUSED_LINEAR_MK))
    assert stranded == [], (
        f"{stranded} have a compiled producer that no shape can reach"
    )


@pytest.mark.parametrize(
    "m,k",
    [(512, 1536), (7800, 1536), (256, 3072), (6889, 3072), (9216, 3072), (16384, 3072)],
)
def test_swept_producers_are_admitted_and_route_native(m: int, k: int) -> None:
    # Given: six prefixes gained a producer because a full sweep of the legal
    # launch geometries measured the fused producer beating the unfused prefix
    # by 13.2% to 38.9% (artifacts/producer-gap-20260822).
    # When: each is resolved through the manifest.
    # Then: it is admitted, it routes native, and it does not claim the packed
    # L2T layout, which belongs to the M537 mixed producer alone.
    assert (m, k) in SM120_FUSED_LINEAR_MK
    assert _has_native_producer(m, k)
    assert (m, k) in _ffi_guard_shapes()
    # "routes native and does not claim the packed L2T layout" used to be two
    # table lookups. The producer is chosen per tactic now, so the same claim is
    # made about the variants themselves: none of these shapes offers the M537
    # family, which is the only one that reads L2T prepacked.
    variants = sm120_producer_variants(m, k)
    assert variants, f"({m}, {k}) computes no producer geometry"
    assert all(family != SM120_FAMILY_M537 for family, _, _ in variants)
    assert not any(sm120_variant_packs_l2t(m, k, v) for v in range(len(variants)))


@pytest.mark.parametrize(
    "m,k",
    [
        (4096, 3072),
        (1024, 3072),
        (64, 12288),
        (256, 12288),
        (512, 12288),
        (1024, 12288),
        (512, 5120),
        (537, 14336),
        (1935, 5376),
    ],
)
def test_prefixes_where_fusing_measured_slower_are_offered_anyway(
    m: int, k: int
) -> None:
    # Given: the sweep measured these prefixes too, and on every one the fused
    # producer was SLOWER than the prefix the shape pays today -- 2.9% to 33.3%
    # against the unfused prefix, 11.1% to 55.6% against the cuBLASLt one.
    # When: the manifest is checked.
    # Then: they are present regardless. Whether a candidate is worth running is
    # a measurement, and the only thing entitled to make it is the tuner; a
    # prefix left out of this table has no fused candidate for the tuner to
    # reject, and the source-side judgement that put it there never re-checks
    # itself. The measured costs live beside each dispatcher entry.
    assert (m, k) in SM120_FUSED_LINEAR_MK
    assert _has_native_producer(m, k)
    assert (m, k) in _ffi_guard_shapes()


def test_every_production_prefix_has_a_fused_producer() -> None:
    # Given: a shape whose (M, K) has no producer cannot offer the fused route
    # to the tuner at all -- not as a losing candidate, not as anything. The
    # coverage question is therefore separate from the performance question.
    # When: every registered production shape's prefix is looked up.
    # Then: all of them have one. This is the invariant that says the fused
    # implementation exists everywhere; which route each shape actually runs is
    # decided by measurement, per shape, per card.
    from flashinfer.testing.svdq_model_shapes import MODEL_SHAPE_CASES

    prefixes = {(case[0], case[2]) for case in MODEL_SHAPE_CASES}
    missing = sorted(p for p in prefixes if not _has_native_producer(*p))
    assert missing == [], f"{missing} have no fused producer"
    assert prefixes <= set(SM120_FUSED_LINEAR_MK)
    assert prefixes <= _ffi_guard_shapes()


# --- the producer axis of a packed tactic ------------------------------------
#
# A tactic is now (consumer row, producer variant) packed into one integer. The
# tests below hold the two halves of that grammar together: the ordering
# contract the packing rests on, its inertness where the axis is degenerate, and
# the one failure mode that is silent rather than loud.


def test_fused_linear_mk_is_the_benchmark_projection() -> None:
    """The named shape list must stay the benchmark matrix, not drift into a gate.

    SM120_FUSED_LINEAR_MK used to decide fused admission. It does not any more --
    _sm120_fused_linear_supported computes that from the launch geometry -- so
    what is left is a name for the shapes this backend was measured on. If the
    two ever diverge, the list has started encoding a decision again.
    """
    from flashinfer.testing.svdq_model_shapes import MODEL_SHAPE_CASES

    assert set(SM120_FUSED_LINEAR_MK) == {(m, k) for m, _, k in MODEL_SHAPE_CASES}
