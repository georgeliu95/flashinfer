"""CPU-only checks for SM120 linear-GEMM workspace sizing.

`_sm120_max_workspace_bytes` allocates one buffer per shape that must cover
every tactic the autotuner can select and no more. Sizing it over the wider
`can_implement` set reserved the O(splits*m*n*4B) Split-K reduction workspace on
shapes where `_sm120_tactic_exposed` prunes every Split-K row, which OOMed the
route autotuner on the largest MiniMax-H3 shapes. These tests pin both halves of
the invariant: never wider than the exposed set, never narrower than it.

No GPU and no JIT build: the backend module is a duck-typed fake.
"""

from types import SimpleNamespace

from flashinfer.gemm import gemm_svdquant

from flashinfer.gemm import svdquant_sm120_cutlass
from flashinfer.testing.svdq_model_shapes import MODEL_SHAPE_CASES


def _pack_row(kernel_id, splits=1, raster=0, swizzle=1, streamk=0):
    return (kernel_id | (splits << 8) | (raster << 16) | (swizzle << 24)
            | (streamk << 32))


# MiniMax-H3 row 69, the largest shape in the frozen benchmark matrix and one of
# the three that fell back to tactic=-1 after the route autotuner OOMed.
_H3_LARGEST = (82752, 28672, 5376, 32)

# The one MiniMax-H3 shape that does keep a Split-K row exposed: k >= 8192 and
# only 5*42 = 210 output tiles, so it clears both halves of the Split-K rule.
# Narrowing the sizing set must be a no-op here.
_H3_SPLITK_SHAPE = (537, 5376, 14336, 32)

# MiniMax-H3 row 52, newly admitted to the fused K12 producer. No combined
# tactic is pinned for it, so the buffer has to cover every K3 tactic the
# autotuner may select -- which is the same invariant as everywhere else, just
# on a shape that now has one more runner offering it.
_H3_ROW52 = (73984, 5376, 14336, 32)

# The 27 frozen MiniMax-H3 benchmark cases, mirroring the registry's own
# construction (see tests/gemm/test_svdq_model_shapes.py). Declared locally so
# this file stays readable, then cross-checked against the registry below.
_H3_TEXT_M = (537, 1935, 6913)
_H3_OMNI_M = (61056, 73984, 82752)
_H3_BLOCK_NK = ((21504, 5376), (5376, 7168), (28672, 5376), (5376, 14336))
_H3_SHAPES = frozenset(
    [(m, n, k) for m in (*_H3_TEXT_M, *_H3_OMNI_M) for n, k in _H3_BLOCK_NK]
    + [(m, 5376, 5120) for m in _H3_TEXT_M]
)

# Split-K on a legacy Stream-K kernel: implementable anywhere, exposed only on
# long-K problems with an underfilled grid, and by far the largest workspace.
_SPLITK_TACTIC = 1
_SPLITK_WORKSPACE_BYTES = 38 * 1024**3
_LEGACY_WORKSPACE_BYTES = 4 * 1024**2
_STREAMK_KERNEL_IDS = (2, 3, 6, 7, 10, 11, 14, 15)


def _fake_workspace_module(with_tactic_row=True):
    """Backend where only the Split-K row carries a multi-GB workspace.

    with_tactic_row=False models a backend compiled before the runtime tactic
    table was exported, where the exposure filter cannot run.
    """
    rows = {
        _SPLITK_TACTIC: _pack_row(2, splits=2, streamk=1),
    }
    module = SimpleNamespace(
        nvfp4_svdquant_gemm_tactic_num=lambda: 4,
        nvfp4_svdquant_gemm_can_implement=lambda m, n, k, rank, t: True,
        nvfp4_svdquant_gemm_workspace_size=lambda m, n, k, t: (
            _SPLITK_WORKSPACE_BYTES if t == _SPLITK_TACTIC
            else _LEGACY_WORKSPACE_BYTES
        ),
    )
    if with_tactic_row:
        module.nvfp4_svdquant_gemm_tactic_row = lambda t: rows.get(t, _pack_row(8))
    return module


def _patch_workspace_module(monkeypatch, module):
    monkeypatch.setattr(
        svdquant_sm120_cutlass, "_get_nvfp4_svdquant_module_for_device",
        lambda device: module)
    monkeypatch.setattr(svdquant_sm120_cutlass, "_SM120_WORKSPACE_BYTES_CACHE", {})


def test_sm120_workspace_skips_tactics_the_autotuner_cannot_select(monkeypatch):
    """Sizing ranges over the exposed set, not the wider can_implement set.

    Sizing over can_implement allocated the O(splits*m*n*4B) Split-K reduction
    buffer on shapes where _sm120_tactic_exposed already prunes every Split-K
    row, which OOMed the route autotuner on the largest H3 shapes.
    """
    _patch_workspace_module(monkeypatch, _fake_workspace_module())
    m, n, k, rank = _H3_LARGEST
    # Precondition: the expensive row is implementable but never selectable here.
    assert not svdquant_sm120_cutlass._sm120_tactic_exposed(
        m, n, k, _pack_row(2, splits=2, streamk=1))
    assert svdquant_sm120_cutlass._sm120_max_workspace_bytes(
        SimpleNamespace(index=0), m, n, k, rank) == _LEGACY_WORKSPACE_BYTES


def test_sm120_workspace_covers_every_selectable_tactic(monkeypatch):
    """The buffer must be a superset of what the autotuner can return.

    Under-sizing is caught fail-closed by the TVM_FFI_ICHECK_GE in
    csrc/nvfp4_svdquant_gemm_cutlass_sm120.cu, so this asserts the Python side
    never gets there.
    """
    module = _fake_workspace_module()
    _patch_workspace_module(monkeypatch, module)
    for m, n, k in [_H3_LARGEST[:3], _H3_SPLITK_SHAPE[:3], (512, 3072, 3072),
                    _H3_ROW52[:3]]:
        rank = 32
        allotted = svdquant_sm120_cutlass._sm120_max_workspace_bytes(
            SimpleNamespace(index=0), m, n, k, rank)
        selectable = svdquant_sm120_cutlass._nvfp4_svdquant_valid_tactics(
            module, m, n, k, rank)
        assert selectable, f"no tactic exposed for {(m, n, k)}"
        for t in selectable:
            assert allotted >= module.nvfp4_svdquant_gemm_workspace_size(
                m, n, k, t), f"tactic {t} would under-run the buffer at {(m, n, k)}"


def test_sm120_workspace_still_sizes_splitk_where_it_is_exposed(monkeypatch):
    """Narrowing must not starve the shapes Split-K legitimately wins on."""
    _patch_workspace_module(monkeypatch, _fake_workspace_module())
    m, n, k, rank = _H3_SPLITK_SHAPE
    assert svdquant_sm120_cutlass._sm120_tactic_exposed(
        m, n, k, _pack_row(2, splits=2, streamk=1))
    assert svdquant_sm120_cutlass._sm120_max_workspace_bytes(
        SimpleNamespace(index=0), m, n, k, rank) == _SPLITK_WORKSPACE_BYTES


def test_sm120_workspace_falls_back_to_can_implement_without_a_tactic_table(
    monkeypatch,
):
    """A backend too old to export tactic_row keeps the pre-narrowing sizing.

    _nvfp4_svdquant_valid_tactics degrades to the can_implement set there, so
    the buffer must stay wide rather than under-size a Split-K row the exposure
    filter was never able to prune.
    """
    _patch_workspace_module(
        monkeypatch, _fake_workspace_module(with_tactic_row=False))
    m, n, k, rank = _H3_LARGEST
    assert svdquant_sm120_cutlass._sm120_max_workspace_bytes(
        SimpleNamespace(index=0), m, n, k, rank) == _SPLITK_WORKSPACE_BYTES


def test_sm120_splitk_exposure_over_the_frozen_h3_matrix():
    """Which H3 shapes the narrowing actually changes -- no fakes, real rule.

    26 of the 27 frozen MiniMax-H3 cases expose no Split-K row at all, so sizing
    over can_implement was allocating a reduction buffer that could never be
    used; that includes all twelve large-M cases and all three that OOMed. Only
    (537, 5376, 14336) clears both `k >= 8192` and `output_tiles <= 220`, and
    there the sizing set is unchanged.
    """
    assert len(_H3_SHAPES) == 27
    assert _H3_SHAPES <= set(MODEL_SHAPE_CASES)
    splitk_rows = [_pack_row(kid, splits=sp, streamk=1)
                   for kid in _STREAMK_KERNEL_IDS for sp in (2, 4)]
    exposed = {(m, n, k) for m, n, k in _H3_SHAPES
               if any(svdquant_sm120_cutlass._sm120_tactic_exposed(m, n, k, row)
                      for row in splitk_rows)}
    assert exposed == {_H3_SPLITK_SHAPE[:3]}
    # The three route-autotuner OOM rows must be on the narrowed side.
    for shape in [(73984, 28672, 5376), (82752, 21504, 5376), (82752, 28672, 5376)]:
        assert shape in _H3_SHAPES and shape not in exposed


def test_sm120_row52_admission_does_not_widen_its_workspace():
    """Admitting row 52 to the fused route cannot re-OOM the route autotuner.

    The three shapes that OOMed did so because Split-K sizing was reserved
    where no Split-K row is exposed. Row 52 is on the narrowed side too, so the
    admission adds a runner without adding a byte of workspace.
    """
    m, n, k, rank = _H3_ROW52
    assert (m, n, k) in _H3_SHAPES
    splitk_rows = [_pack_row(kid, splits=sp, streamk=1)
                   for kid in _STREAMK_KERNEL_IDS for sp in (2, 4)]
    assert not any(
        svdquant_sm120_cutlass._sm120_tactic_exposed(m, n, k, row) for row in splitk_rows)
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(m, k, rank)
