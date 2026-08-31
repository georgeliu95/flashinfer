"""Immutable SM120 SVDQuant linear route manifest.

Routes only. This module used to carry 51 measured tactics as a source literal
that bypassed the autotuner, and an A/B ladder for hand-driving them. Both are
gone: a tactic is a measurement, and a measurement belongs in something the
tuner produced -- `AutoTuner.save_configs` / `load_configs` -- not in a file
somebody edits. What remains here is what a tactic cannot be derived from: which
runners a shape may offer, which prefix carries which producer route, and the ABI
versions that name a persisted record.

`tests/gemm/test_svdquant_sm120_routes.py` holds a guard that fails if a
shape-keyed tactic table reappears in this package.
"""

import functools
import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, TypeAlias


LinearRouteKey: TypeAlias = tuple[int, int, int]

SM120_TACTIC_ABI_VERSION: Final = 3

# v4 is the packed (consumer row, producer variant) grammar. It is deliberately
# NOT a global bump. Where the producer axis is degenerate -- every shape off
# the nine-entry ladder -- a packed tactic is the same integer as the bare row
# it replaces and selects the same thing, so a v3 record still resolves
# correctly and throwing it away would cost a re-profile for nothing.
#
# The cost of getting that wrong is recorded in
# test_sm120_linear_op_name_keeps_v5_for_every_other_shape: an earlier global
# bump made every unrelated shape miss its record and re-profile, and one such
# re-tune picked a slower tactic and regressed that shape by 8.47%.
SM120_PACKED_TACTIC_ABI_VERSION: Final = 4
SM120_LINEAR_ROUTE_ABI_VERSION: Final = 5

# A route version changes only when the runner set for that exact (M, K, rank)
# changes. Unlisted shapes retain the v5 namespace and its persisted winner.
SM120_LINEAR_ROUTE_ABI_OVERRIDES: Final[Mapping[LinearRouteKey, int]] = (
    MappingProxyType(
        {
            # v6: screened MiniMax-H3 rows 50, 52, 59, 61, 68, and 70.
            (73984, 14336, 32): 6,
            (73984, 7168, 32): 6,
            (61056, 7168, 32): 6,
            (61056, 14336, 32): 6,
            (82752, 14336, 32): 6,
            (82752, 7168, 32): 6,
            # v7: the remaining Batch-4 routes with an exact fused prefix.
            (537, 5120, 32): 7,
            (537, 7168, 32): 7,
            (537, 5376, 32): 7,
            (1935, 5120, 32): 7,
            (1935, 7168, 32): 7,
            (1935, 14336, 32): 7,
            (6913, 5120, 32): 7,
            (6913, 7168, 32): 7,
            (6913, 14336, 32): 7,
            # v8: the previously unfused K5376 producer families.
            (1935, 5376, 32): 8,
            (6913, 5376, 32): 8,
            (73984, 5376, 32): 8,
            (61056, 5376, 32): 8,
            (82752, 5376, 32): 8,
            # v9: Batch-4 case 4 uses the independently screened cuBLASLt prefix.
            (537, 14336, 32): 9,
        }
    )
)


def _route_keys_for_version(version: int) -> frozenset[LinearRouteKey]:
    return frozenset(
        key
        for key, route_version in SM120_LINEAR_ROUTE_ABI_OVERRIDES.items()
        if route_version == version
    )


SM120_LINEAR_ROUTE_V6_MKR: Final = _route_keys_for_version(6)
SM120_LINEAR_ROUTE_V7_MKR: Final = _route_keys_for_version(7)
SM120_LINEAR_ROUTE_V8_MKR: Final = _route_keys_for_version(8)
SM120_LINEAR_ROUTE_V9_MKR: Final = _route_keys_for_version(9)


def sm120_linear_route_abi_version(m: int, k: int, rank: int) -> int:
    """Return the persistent route namespace for one exact linear shape."""
    return SM120_LINEAR_ROUTE_ABI_OVERRIDES.get(
        (m, k, rank), SM120_LINEAR_ROUTE_ABI_VERSION
    )


def sm120_tactic_abi_version(m: int, k: int) -> int:
    """Return the tactic namespace for one exact shape.

    Shape-keyed for the same reason the route ABI is: only the shapes whose
    candidate space actually changed should lose their persisted winners.
    """
    return (
        SM120_PACKED_TACTIC_ABI_VERSION
        if len(sm120_producer_variants(m, k)) > 1
        else SM120_TACTIC_ABI_VERSION
    )


PREFIX_ROUTE_OVERRIDE_ENV: Final = "SVDQ_SM120_PREFIX_ROUTE"

PrefixRoute: TypeAlias = str
PrefixRouteKey: TypeAlias = tuple[int, int]

# Mirror of the exact-shape ladder in
# include/flashinfer/gemm/svdquant_sm120_prefix_route.h: (default, admitted).
# The two sides must agree because the M537 native producer consumes a prepacked
# LoRA-down matrix while the cuBLASLt prefix consumes the row-major one; the
# packed layout below is chosen from the effective route, not from the shape.
#
# Every alternate rung this ladder once carried was measured on SM120 (110 SMs) in
# round 2 and refuted, so each shape now admits only its measured default. The
# C++ header records the per-rung medians. The table is retained for the
# gate/layout reconciliation, not as a live A/B instrument.
SM120_PREFIX_ROUTE_LADDER: Final[
    Mapping[PrefixRouteKey, tuple[PrefixRoute, frozenset[PrefixRoute]]]
] = MappingProxyType(
    {
        # Mirrors include/flashinfer/gemm/svdquant_sm120_prefix_route.h. Every
        # default is the measured winner and none of them move. The native rung
        # is admissible on the cuBLASLt-routed entries because each now has a
        # compiled native producer: selecting it still takes an explicit
        # SVDQ_SM120_PREFIX_ROUTE, but the losing arm can be re-measured instead
        # of re-argued. The margins are recorded in the C++ header.
        (512, 5120): ("cublaslt", frozenset({"cublaslt", "native"})),
        (537, 5120): ("cublaslt", frozenset({"cublaslt", "native"})),
        (537, 7168): ("cublaslt", frozenset({"cublaslt", "native"})),
        (537, 14336): ("cublaslt", frozenset({"cublaslt", "native"})),
        (537, 5376): ("native", frozenset({"native"})),
        (1935, 5120): ("cublaslt", frozenset({"cublaslt", "native"})),
        (1935, 7168): ("cublaslt", frozenset({"cublaslt", "native"})),
        (1935, 5376): ("cublaslt", frozenset({"cublaslt", "native"})),
        (1935, 14336): ("native", frozenset({"native"})),
    }
)

# Shapes whose native producer is the M537 mixed kernel, which indexes L2T with
# the prepacked layout. Row-major L2T on those routes would be misread.
SM120_PACKED_L2T_NATIVE_MK: Final[frozenset[PrefixRouteKey]] = frozenset(
    {(537, 5120), (537, 5376), (537, 7168)}
)

_NO_PREFIX_ROUTE_OVERRIDES: Final[Mapping[PrefixRouteKey, PrefixRoute]] = (
    MappingProxyType({})
)


def _parse_prefix_route_overrides(spec: str) -> Mapping[PrefixRouteKey, PrefixRoute]:
    """Parse ``m:k=route`` entries separated by commas.

    A/B-only diagnostic: the shape must appear in the ladder and the route must be
    admitted for that shape, so the seam can select between implementations that
    were already compiled for an exact shape but can never invent one.
    """
    overrides: dict[PrefixRouteKey, PrefixRoute] = {}
    for entry in spec.split(","):
        item = entry.strip()
        if not item:
            raise ValueError(f"{PREFIX_ROUTE_OVERRIDE_ENV} entry {item!r} is empty")
        key_text, separator, route = item.partition("=")
        if not separator:
            raise ValueError(
                f"{PREFIX_ROUTE_OVERRIDE_ENV} entry {item!r} must be m:k=route"
            )
        fields = key_text.split(":")
        if len(fields) != 2:
            raise ValueError(
                f"{PREFIX_ROUTE_OVERRIDE_ENV} entry {item!r} must name m:k"
            )
        try:
            key = (int(fields[0]), int(fields[1]))
        except ValueError as error:
            raise ValueError(
                f"{PREFIX_ROUTE_OVERRIDE_ENV} entry {item!r} is not integral"
            ) from error
        ladder = SM120_PREFIX_ROUTE_LADDER.get(key)
        if ladder is None:
            raise ValueError(
                f"{PREFIX_ROUTE_OVERRIDE_ENV} entry {item!r} names "
                f"{key}, which has no A/B ladder"
            )
        if route.strip() not in ladder[1]:
            raise ValueError(
                f"{PREFIX_ROUTE_OVERRIDE_ENV} entry {item!r} names a route that "
                f"{key} does not admit"
            )
        if key in overrides:
            raise ValueError(
                f"{PREFIX_ROUTE_OVERRIDE_ENV} entry {item!r} repeats {key}"
            )
        overrides[key] = route.strip()
    return MappingProxyType(overrides)


@functools.lru_cache(maxsize=8)
def _cached_prefix_route_overrides(spec: str) -> Mapping[PrefixRouteKey, PrefixRoute]:
    return _parse_prefix_route_overrides(spec)


def sm120_prefix_route_overrides() -> Mapping[PrefixRouteKey, PrefixRoute]:
    """Return the active ``SVDQ_SM120_PREFIX_ROUTE`` overrides (empty when unset)."""
    spec = os.environ.get(PREFIX_ROUTE_OVERRIDE_ENV)
    if not spec or not spec.strip():
        return _NO_PREFIX_ROUTE_OVERRIDES
    return _cached_prefix_route_overrides(spec)


def sm120_fused_prefix_route(m: int, k: int) -> PrefixRoute:
    """Return the effective fused-prefix implementation for one exact shape."""
    ladder = SM120_PREFIX_ROUTE_LADDER.get((m, k))
    if ladder is None:
        return "native"
    return sm120_prefix_route_overrides().get((m, k), ladder[0])


# --- the producer axis of a packed tactic ------------------------------------
#
# A tactic is a pair, packed the way nvfp4_svdquant_gemm_tactic_row already
# packs its own fields (kernel_id | splits << 8 | raster << 16 | swizzle << 24):
#
#     tactic = k3_row | producer_variant << SM120_PRODUCER_SHIFT
#
# k3_row is a row of kRuntimeTacticTableSm120 -- 0..82, so eight bits is ample --
# and producer_variant indexes sm120_producer_variants(m, k).
#
# Variant 0 is always the ladder default. That is what makes the change inert
# where the axis is degenerate: a packed tactic whose producer half is zero is
# numerically the same integer as the bare row it replaces, and selects the same
# producer that row selected before this axis existed.
SM120_PRODUCER_SHIFT: Final = 8
SM120_K3_ROW_MASK: Final = (1 << SM120_PRODUCER_SHIFT) - 1


def sm120_producer_variants(m: int, k: int) -> tuple[PrefixRoute, ...]:
    """Admitted producer routes for one exact shape, the default first.

    Default-first is a contract the packing depends on, not a convenience.
    Shapes outside the ladder resolve to the native producer and have exactly
    one variant, so the axis is degenerate for them.
    """
    ladder = SM120_PREFIX_ROUTE_LADDER.get((m, k))
    if ladder is None:
        return ("native",)
    default, admitted = ladder
    return (default, *sorted(admitted - {default}))


def sm120_pack_tactic(k3_row: int, producer_variant: int = 0) -> int:
    """Pack a (consumer row, producer variant) pair into one tactic integer."""
    if not 0 <= k3_row <= SM120_K3_ROW_MASK:
        raise ValueError(f"k3 row outside the packable range: {k3_row}")
    if producer_variant < 0:
        raise ValueError(f"negative producer variant: {producer_variant}")
    return k3_row | (producer_variant << SM120_PRODUCER_SHIFT)


def sm120_unpack_tactic(tactic: int) -> tuple[int, int]:
    """Return ``(k3_row, producer_variant)``.

    A negative tactic is the autotuner's "no selection" sentinel and passes
    through unchanged on the consumer side, with the default producer.
    """
    if tactic < 0:
        return tactic, 0
    return tactic & SM120_K3_ROW_MASK, tactic >> SM120_PRODUCER_SHIFT


# --- producer geometry axis -------------------------------------------------
#
# The large-M producer takes its block size and both tile extents as template
# arguments, so a geometry is a whole kernel instantiation rather than a runtime
# argument. Everything below mirrors
# include/flashinfer/gemm/nvfp4_smooth_quantize_lora_down_sm120.cuh; the mirror
# exists because Python has to know how many candidates a shape offers in order
# to build the tactic list, and C++ has to know which one an index names in
# order to launch it. Tests hold the two against each other.

# ORDER IS ABI -- see the same note on kGeometryLadder in the header.
SM120_PRODUCER_GEOMETRY_LADDER: Final[tuple[tuple[int, int, int], ...]] = (
    (192, 16, 512),
    (192, 32, 128),
    (192, 48, 128),
    (224, 48, 128),
    (256, 32, 256),
    (256, 80, 128),
    (288, 80, 128),
    (384, 32, 256),
)

# Budget, not a judgement: every admitted geometry is another instantiation in a
# single translation unit and another candidate the tuner profiles. Measured on
# SM120 (110 SMs), raising this from 1 to 4 took that unit from 6.9 s to 11.9 s to
# compile and its object from 700 KB to 1058 KB.
SM120_MAX_PRODUCER_GEOMETRIES: Final = 4

# The geometry each exact large-M shape is pinned to today, which is always
# variant 0 so an unselected shape runs what it has always run.
SM120_PINNED_PRODUCER_GEOMETRY: Final[Mapping[PrefixRouteKey, tuple[int, int, int]]] = (
    MappingProxyType({
        (1024, 12288): (192, 16, 512),
        (1935, 14336): (384, 32, 256),
        (4096, 3072): (384, 32, 256),
        (4096, 12288): (384, 32, 256),
        (6889, 3072): (224, 48, 128),
        (6889, 12288): (288, 80, 128),
        (6913, 5120): (192, 32, 128),
        (6913, 5376): (192, 32, 128),
        (6913, 7168): (256, 32, 256),
        (6913, 14336): (384, 32, 256),
        (7800, 1536): (224, 48, 128),
        (9216, 3072): (192, 48, 128),
        (9216, 12288): (288, 80, 128),
        (16384, 3072): (288, 80, 128),
        (16384, 12288): (288, 80, 128),
        (32760, 1536): (256, 32, 256),
        (61056, 5376): (256, 32, 256),
        (61056, 7168): (256, 32, 256),
        (61056, 14336): (256, 80, 128),
        (73984, 5376): (256, 32, 256),
        (73984, 7168): (256, 32, 256),
        (73984, 14336): (256, 80, 128),
        (82752, 5376): (256, 32, 256),
        (82752, 7168): (256, 32, 256),
        (82752, 14336): (256, 80, 128),
    })
)

_GEOMETRY_RANK: Final = 32
_GEOMETRY_SF_VEC: Final = 16
_GEOMETRY_ELEM_BYTES: Final = 2
_GEOMETRY_DOWN_WARPS: Final = 4
_GEOMETRY_SHARED_LIMIT: Final = 48 * 1024


def sm120_producer_geometry_is_valid(
    m: int, k: int, block_threads: int, tile_m: int, tile_k: int
) -> bool:
    """Mirror of ``geometry_is_valid`` in the producer header.

    Restates KernelLaunchGeometry's static_asserts as a predicate. It has to be
    a predicate rather than an assertion on both sides: enumeration must be able
    to skip a combination this shape cannot instantiate, and a static_assert on
    an unselected geometry is a build failure rather than a rejected candidate.
    """
    if tile_k <= 0 or k % tile_k:
        return False
    if tile_k % (4 * _GEOMETRY_SF_VEC):
        return False
    if not (0 < tile_m <= 80 and tile_m % 16 == 0):
        return False
    quant_threads = block_threads - _GEOMETRY_DOWN_WARPS * 32
    if block_threads % 32 or quant_threads <= 0:
        return False
    sf_cols_per_tile = tile_k // _GEOMETRY_SF_VEC
    if (tile_m * sf_cols_per_tile) % quant_threads:
        return False
    padded_m = (m + 127) // 128 * 128
    grid_blocks = (padded_m + tile_m - 1) // tile_m
    prefetch_lines = k * _GEOMETRY_RANK * _GEOMETRY_ELEM_BYTES // 128
    if not (k < 8192 or grid_blocks * block_threads >= prefetch_lines):
        return False
    shared_bytes = (
        2 * tile_m * (tile_k + 8) * _GEOMETRY_ELEM_BYTES
        + 2 * tile_k * _GEOMETRY_ELEM_BYTES
        + 2 * ((tile_m * sf_cols_per_tile) if k >= 8192 else 1)
    )
    return shared_bytes <= _GEOMETRY_SHARED_LIMIT


def sm120_producer_geometries(m: int, k: int) -> tuple[tuple[int, int, int], ...]:
    """Geometries one exact shape offers, its pinned one first.

    A shape with no large-M producer offers nothing here and gets an empty
    tuple, which is what makes the geometry axis degenerate for it.
    """
    pinned = SM120_PINNED_PRODUCER_GEOMETRY.get((m, k))
    if pinned is None:
        return ()
    offered = [pinned]
    for candidate in SM120_PRODUCER_GEOMETRY_LADDER:
        if len(offered) >= SM120_MAX_PRODUCER_GEOMETRIES:
            break
        if candidate == pinned:
            continue
        if sm120_producer_geometry_is_valid(m, k, *candidate):
            offered.append(candidate)
    return tuple(offered)


def sm120_producer_variant_count(m: int, k: int) -> int:
    """How many producer variants the tactic space offers for one shape.

    Routes first, then geometries appended. Appending is the ABI rule: a phase-1
    record holds a route index, and putting geometries in front of it would
    silently repoint every one of them.
    """
    routes = len(sm120_producer_variants(m, k))
    geometries = len(sm120_producer_geometries(m, k))
    return routes + max(0, geometries - 1)


def sm120_decode_producer_variant(m: int, k: int, variant: int) -> tuple[PrefixRoute, int]:
    """Split a producer variant into the route and the geometry it names.

    The two axes are disjoint in the manifest -- no shape admits more than one
    route AND carries a geometry ladder -- and a test pins that, because it is
    what makes this split unambiguous rather than a convention.
    """
    routes = sm120_producer_variants(m, k)
    if variant < len(routes):
        return sm120_tactic_producer_route(m, k, variant), 0
    return routes[0], variant - len(routes) + 1


def sm120_tactic_producer_route(m: int, k: int, producer_variant: int) -> PrefixRoute:
    """Route named by the producer half of a packed tactic.

    This is a mirror of ``resolve_cublaslt_prefix`` in
    ``include/flashinfer/gemm/svdquant_sm120_prefix_route.h`` and has to stay one,
    because the two run on the same call: C++ picks the prefix and Python picks
    the L2T layout it will read. Paraphrasing the rule instead of mirroring it is
    what broke the environment override -- variant 0 resolved to the ladder
    default here while C++ resolved it to the effective route, so an override on
    an M537 shape handed the row-major matrix to the prepacked producer.

    Variant 0, and any index the variant list does not hold, means the EFFECTIVE
    route, so the override keeps working exactly as it did before this axis
    existed. A variant above 0 names its route outright and the override does not
    reach it.
    """
    if producer_variant > 0:
        variants = sm120_producer_variants(m, k)
        if producer_variant < len(variants):
            return variants[producer_variant]
    return sm120_fused_prefix_route(m, k)


def sm120_fused_prefix_uses_packed_l2t(
    m: int,
    k: int,
    rank: int,
    route: PrefixRoute | None = None,
) -> bool:
    """Return whether the combined fused prefix requires the M537 L2T layout.

    ``route`` is the producer the caller is about to dispatch. It must be passed
    whenever a tactic selected the producer, because the layout and the dispatch
    have to describe the same route: the M537 native producer indexes L2T with
    the prepacked layout and the cuBLASLt prefix reads the row-major one, so a
    mismatched pair misreads the matrix rather than failing. Omitting it falls
    back to the ladder's effective route, which is correct only when nothing
    overrode it.
    """
    if rank != 32 or (m, k) not in SM120_PACKED_L2T_NATIVE_MK:
        return False
    effective = sm120_fused_prefix_route(m, k) if route is None else route
    return effective != "cublaslt"


SM120_FUSED_LINEAR_MK: Final[frozenset[tuple[int, int]]] = frozenset(
    {
        (64, 3072),
        (512, 3072),
        (512, 5120),
        (4096, 12288),
        (6889, 12288),
        (7800, 5120),
        (7800, 8960),
        (7800, 13824),
        (27280, 3072),
        (27280, 14336),
        # 32760x1536x1536 and 32760x8960x1536 reached the fused runner only
        # because they carried a pinned tactic, whose prefixes were unioned in
        # separately. Admission is route data, not a measured preference, so it
        # stays behind when the pins go. Without it the tuner sees only
        # Sm120CutlassLinearRunner for these two, which cost 19% on
        # 32760x1536x1536 when measured (artifacts/stage0-exposure-20260821).
        (32760, 1536),
        (32760, 5120),
        (32760, 8960),
        (32760, 13824),
        (73984, 7168),
        (61056, 7168),
        (61056, 14336),
        (73984, 14336),
        (82752, 14336),
        (82752, 7168),
        (537, 5120),
        (537, 7168),
        (537, 5376),
        (537, 14336),
        (1935, 5120),
        (1935, 5376),
        (1935, 7168),
        (1935, 14336),
        (6913, 5120),
        (6913, 5376),
        (73984, 5376),
        (61056, 5376),
        (82752, 5376),
        (6913, 7168),
        (6913, 14336),
        (75600, 5120),
        (75600, 13824),
        (9216, 12288),
        (16384, 12288),
        # Prefixes whose native producer was added after a full sweep of the
        # legal launch geometries showed the fused producer beating the unfused
        # prefix -- a quantize kernel plus a separate rank-32 torch.mm, which
        # streams the activation twice -- by 13.2% to 38.9%. The six further
        # prefixes swept alongside these are deliberately absent: fusing them
        # measured slower, so admitting them would cost time. The dispatcher
        # entries and the full measured table live in
        # include/flashinfer/gemm/nvfp4_smooth_quantize_lora_down_sm120.cuh.
        (512, 1536),
        (7800, 1536),
        (256, 3072),
        (6889, 3072),
        (9216, 3072),
        (16384, 3072),
        # Prefixes whose producer measured SLOWER than the prefix they pay
        # today. They are admitted anyway, so the fused route exists for every
        # production shape and the tuner is the thing that rejects it. Leaving
        # them out would mean this file had decided, once, on one card, that a
        # candidate was not worth measuring -- and nothing would ever re-check.
        # The measured numbers are in the dispatcher comment beside each one.
        (1024, 3072),
        (4096, 3072),
        (64, 12288),
        (256, 12288),
        (512, 12288),
        (1024, 12288),
    }
)
