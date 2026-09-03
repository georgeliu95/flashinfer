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
    PREFIX_ROUTE_OVERRIDE_ENV,
    SM120_TACTIC_ABI_VERSION,
    SM120_FUSED_LINEAR_MK,
    SM120_LINEAR_ROUTE_ABI_OVERRIDES,
    SM120_LINEAR_ROUTE_V6_MKR,
    SM120_PACKED_L2T_NATIVE_MK,
    SM120_PREFIX_ROUTE_LADDER,
    sm120_fused_prefix_route,
    sm120_fused_prefix_uses_packed_l2t,
    sm120_linear_route_abi_version,
    sm120_prefix_route_overrides,
    sm120_tactic_abi_version,
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






@pytest.mark.parametrize(
    "m,k,rank,expected",
    [
        (537, 5376, 32, True),
        (537, 5120, 32, False),
        (537, 5376, 64, False),
        (1935, 5376, 32, False),
    ],
)
def test_packed_l2t_is_a_fused_prefix_specific_contract(
    m: int,
    k: int,
    rank: int,
    expected: bool,
) -> None:
    # Given: M537 routes can use either cuBLASLt or the packed custom prefix.
    # When: the combined-prefix layout contract is selected.
    uses_packed_l2t = sm120_fused_prefix_uses_packed_l2t(m, k, rank)
    # Then: only the custom M537/K5376/rank-32 prefix requests packing.
    assert uses_packed_l2t is expected










def test_default_prefix_routes_reproduce_the_measured_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the ladder replaced a literal cuBLASLt-prefix shape list in C++.
    # When: the effective route is read with no override in the environment.
    # Then: exactly the seven recorded cuBLASLt shapes still take that prefix.
    monkeypatch.delenv(PREFIX_ROUTE_OVERRIDE_ENV, raising=False)
    cublaslt_shapes = {
        key for key in SM120_PREFIX_ROUTE_LADDER if sm120_fused_prefix_route(*key) == "cublaslt"
    }
    assert cublaslt_shapes == {
        (512, 5120),
        (537, 5120),
        (537, 7168),
        (537, 14336),
        (1935, 5120),
        (1935, 5376),
        (1935, 7168),
    }
    # And: every ladder default is itself an admitted route for that shape.
    for key, (default, admitted) in SM120_PREFIX_ROUTE_LADDER.items():
        assert default in admitted, key
    # And: a shape with no ladder entry keeps the native producer.
    assert sm120_fused_prefix_route(6913, 14336) == "native"
    assert sm120_prefix_route_overrides() == {}


def test_closed_ladder_admits_only_explicit_control_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: every alternate rung regressed on SM120 (110 SMs) in round 2, so each shape
    # is now pinned to its measured default and the ladder is closed.
    # When: an override spells out the defaults of two rungs explicitly.
    # Then: it parses and is a no-op, which is how a control arm stays expressible
    # without leaving a live path back to a refuted route.
    monkeypatch.setenv(
        PREFIX_ROUTE_OVERRIDE_ENV, "537:5120=cublaslt, 1935:14336=native"
    )
    assert sm120_fused_prefix_route(537, 5120) == "cublaslt"
    assert sm120_fused_prefix_route(1935, 14336) == "native"
    # And: unnamed neighbours are untouched.
    assert sm120_fused_prefix_route(537, 5376) == "native"
    assert sm120_fused_prefix_route(1935, 5376) == "cublaslt"
    assert sm120_fused_prefix_route(537, 7168) == "cublaslt"
    # And: the ladder itself is never mutated by an override.
    assert SM120_PREFIX_ROUTE_LADDER[(537, 5120)][0] == "cublaslt"


def test_refuted_rungs_are_measurable_again_without_moving_a_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the native rung lost on every cuBLASLt-routed shape, twice -- once
    # on a hand-picked geometry in round 2, and again in 2026-08 against the
    # fastest survivor of the producer's entire legal geometry field. A verdict
    # that firm is worth keeping as the default. It is not worth making
    # unrepeatable: each of these shapes now has a compiled native producer, and
    # a rung nothing can select is a kernel that exists only in the binary.
    # When: each refuted native rung is requested explicitly.
    # Then: it is accepted, and the default is unchanged, so re-measuring costs
    # an environment variable while shipping the loser still costs a code edit.
    for m, k in ((537, 5120), (537, 7168), (537, 14336),
                 (1935, 5120), (1935, 7168), (1935, 5376), (512, 5120)):
        monkeypatch.setenv(PREFIX_ROUTE_OVERRIDE_ENV, f"{m}:{k}=native")
        assert sm120_fused_prefix_route(m, k) == "native"
        monkeypatch.delenv(PREFIX_ROUTE_OVERRIDE_ENV)
        assert sm120_fused_prefix_route(m, k) == "cublaslt"
        assert SM120_PREFIX_ROUTE_LADDER[(m, k)][0] == "cublaslt"


def test_rungs_with_no_compiled_route_are_still_inadmissible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: admitting a rung is a claim that something can run it.
    # When: rungs whose implementation was retired or never built are requested.
    # Then: they are rejected by admission, not by syntax, so a stale campaign
    # script fails legibly rather than silently selecting a route that is gone.
    for spec in (
        "1935:14336=cublaslt",  # 448.51 us vs 424.22 us, refuted
        "1935:14336=native_alt",  # 456.70 us vs 424.22 us, geometry removed
        "537:5376=cublaslt",  # round-1 control for the settled rows 44/46 flips
    ):
        monkeypatch.setenv(PREFIX_ROUTE_OVERRIDE_ENV, spec)
        with pytest.raises(ValueError, match="does not admit"):
            sm120_fused_prefix_route(537, 5120)


@pytest.mark.parametrize(
    "spec",
    [
        "537:5120",  # no '='
        "537=native",  # not a full key
        "537:5376:32=native",  # too many key fields
        "537:x=native",  # non-integral key
        "537:5120=cutlass",  # unknown route
        "537:5120=",  # empty route
        "537:5120=native_alt",  # no alternate geometry compiled for this shape
        "1024:5120=native",  # shape with no A/B ladder
        "537:5120=cublaslt,537:5120=cublaslt",  # repeated shape
        "537:5120=cublaslt,",  # empty trailing entry
    ],
)
def test_prefix_route_override_rejects_malformed_specs(
    monkeypatch: pytest.MonkeyPatch,
    spec: str,
) -> None:
    # Given: a route seam can change numerics, not just speed.
    # When: a malformed or inadmissible override entry is supplied.
    # Then: the lookup raises instead of selecting an uncompiled route.
    monkeypatch.setenv(PREFIX_ROUTE_OVERRIDE_ENV, spec)
    with pytest.raises(ValueError):
        sm120_fused_prefix_route(537, 5120)


@pytest.mark.parametrize(
    "spec,m,k,expected",
    [
        (None, 537, 5376, True),
        (None, 537, 5120, False),
        (None, 537, 7168, False),
        (None, 1935, 14336, False),
        ("537:5376=native", 537, 5376, True),
        ("537:5120=cublaslt", 537, 5120, False),
    ],
)
def test_packed_l2t_layout_follows_the_effective_prefix_route(
    monkeypatch: pytest.MonkeyPatch,
    spec: str | None,
    m: int,
    k: int,
    expected: bool,
) -> None:
    # Given: the M537 native producer indexes L2T with the prepacked layout while
    # the cuBLASLt prefix consumes the row-major one.
    # When: the packing decision is read with the closed ladder, both with the
    # seam unset and with a rung's own default spelled out as a control arm.
    # Then: it follows the effective route, so neither side misreads L2T, and the
    # only packed Batch-4 shape left is row 45/47's native (537, 5376) producer.
    if spec is None:
        monkeypatch.delenv(PREFIX_ROUTE_OVERRIDE_ENV, raising=False)
    else:
        monkeypatch.setenv(PREFIX_ROUTE_OVERRIDE_ENV, spec)
    assert sm120_fused_prefix_uses_packed_l2t(m, k, 32) is expected
    # And: packing is still rank-32 only, whatever the route.
    assert sm120_fused_prefix_uses_packed_l2t(m, k, 64) is False


def test_prefix_route_containers_are_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the ladder and its overrides are shared process-wide.
    # When: their public containers are inspected.
    # Then: no caller can mutate another caller's route decisions in place.
    assert isinstance(SM120_PREFIX_ROUTE_LADDER, MappingProxyType)
    assert isinstance(SM120_PACKED_L2T_NATIVE_MK, frozenset)
    monkeypatch.setenv(PREFIX_ROUTE_OVERRIDE_ENV, "537:5120=cublaslt")
    overrides = sm120_prefix_route_overrides()
    assert isinstance(overrides, MappingProxyType)
    with pytest.raises(TypeError):
        overrides[(537, 7168)] = "native"  # type: ignore[index]


def _m537_mixed_producer_shapes(header: str) -> set[tuple[int, int]]:
    """(M, K) whose native producer is the M537 mixed kernel, which indexes the
    prepacked L2T layout. M=537 now carries two producer families, so which
    layout a shape needs can no longer be inferred from its M."""
    body = header[
        header.index("inline cudaError_t nvfp4_smooth_quantize_lora_down_sm120(") :
        header.index("// Direct-only n16 diagnostic entry")
    ]
    return {
        (537, int(match.group(1)))
        for match in re.finditer(r"launch_m537_\w*kernel<\s*(\d+)", body)
    }


def test_packed_l2t_matches_the_producers_that_read_the_packed_layout() -> None:
    # Given: handing a row-major L2T to a producer that indexes the prepacked one
    # is silent numerical corruption, not a crash, so the packing set has to track
    # which kernel family each native route actually reaches.
    # When: the set is compared against the M537-mixed instantiations in the
    # dispatcher, and against every rung that can route natively.
    # Then: they agree exactly. (537, 14336) is the case that broke the old
    # M-based rule: it routes natively now, and its producer is a small_m kernel
    # that reads row-major L2T, so it must NOT be packed.
    header = _producer_header()
    mixed = _m537_mixed_producer_shapes(header)
    assert set(SM120_PACKED_L2T_NATIVE_MK) == mixed, (
        f"packed set {sorted(SM120_PACKED_L2T_NATIVE_MK)} does not match the "
        f"M537-mixed producers {sorted(mixed)}"
    )
    for key, (_, admitted) in SM120_PREFIX_ROUTE_LADDER.items():
        if "native" not in admitted:
            continue
        assert _has_native_producer(*key), f"{key} admits native with no producer"
        assert (key in SM120_PACKED_L2T_NATIVE_MK) == (key in mixed), key


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
        header.index("inline cudaError_t nvfp4_smooth_quantize_lora_down_sm120(") :
        header.index("// Direct-only n16 diagnostic entry")
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
    start = source.index("TVM_FFI_ICHECK(((m ==")
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
    assert unbacked == [], (
        f"{unbacked} pass the FFI check but have no producer"
    )
    # The other direction. It used to read as "we compiled a producer for a
    # shape the FFI will not admit", which was a real waste when producers were
    # per-shape templates. Every legal (M, K) computes a geometry now, so that
    # difference would be every shape outside the guard and would say nothing.
    # Scoped to what this backend actually claims to support, the waste it
    # guarded against is still a thing worth refusing.
    claimed = set(SM120_FUSED_LINEAR_MK)
    unreachable = sorted(
        shape for shape in claimed
        if _has_native_producer(*shape) and shape not in guard
        and SM120_PREFIX_ROUTE_LADDER.get(shape, ("", ()))[0] != "cublaslt"
    )
    assert unreachable == [], (
        f"{unreachable} have a producer the FFI check will reject"
    )


def test_every_fused_admission_has_a_prefix_that_can_run() -> None:
    # Given: admitting (M, K) to SM120_FUSED_LINEAR_MK offers the fused runner to
    # the tuner, and the fused route reaches its prefix in one of two ways -- a
    # compiled native producer, or the cuBLASLt prefix the ladder can select.
    # When: the admission set is checked against both.
    # Then: nothing is admitted that has neither, which would make the C++
    # dispatcher return cudaErrorInvalidValue for a route the tuner can pick.
    # Reachable = the FFI lets it in AND the shape computes a geometry to launch.
    reachable = {
        shape for shape in _ffi_guard_shapes() if _has_native_producer(*shape)
    }
    cublaslt = {
        key for key, entry in SM120_PREFIX_ROUTE_LADDER.items() if entry[0] == "cublaslt"
    }
    orphans = sorted(set(SM120_FUSED_LINEAR_MK) - reachable - cublaslt)
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
    assert sm120_fused_prefix_route(m, k) == "native"
    assert not sm120_fused_prefix_uses_packed_l2t(m, k, 32)
    assert _has_native_producer(m, k)
    assert (m, k) in _ffi_guard_shapes()


@pytest.mark.parametrize(
    "m,k",
    [(4096, 3072), (1024, 3072), (64, 12288), (256, 12288), (512, 12288),
     (1024, 12288), (512, 5120), (537, 14336), (1935, 5376)],
)
def test_prefixes_where_fusing_measured_slower_are_offered_anyway(m: int, k: int) -> None:
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


