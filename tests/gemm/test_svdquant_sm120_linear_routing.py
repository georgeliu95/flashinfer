"""Host-side contracts for SM120 full-linear runner selection."""

from types import SimpleNamespace

import pytest

from flashinfer.gemm import gemm_svdquant

from flashinfer.gemm import svdquant_sm120_cutlass


def test_table_miss_packs_m537_l2t_only_inside_fused_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shared tuner input stays row-major until the fused route is selected."""
    allocated = []
    native_calls = []
    chooser_inputs = []
    packed_l2t = SimpleNamespace(name="packed_l2t")

    def fake_empty(shape, **kwargs):
        tensor = SimpleNamespace(shape=tuple(shape), kwargs=kwargs)
        allocated.append(tensor)
        return tensor

    module = SimpleNamespace(
        nvfp4_svdquant_linear_sm120=lambda *args: native_calls.append(args),
    )
    monkeypatch.setattr(gemm_svdquant, "get_compute_capability", lambda device: (12, 0))
    monkeypatch.setattr(
        gemm_svdquant, "get_nvfp4_svdquant_sm120_module", lambda: module
    )
    monkeypatch.setattr(gemm_svdquant.torch, "empty", fake_empty)
    monkeypatch.setattr(
        gemm_svdquant, "_get_cache_buf", lambda *args, **kwargs: "workspace"
    )
    monkeypatch.setattr(gemm_svdquant, "_sm120_max_workspace_bytes", lambda *args: 0)
    monkeypatch.setattr(
        gemm_svdquant,
        "_sm120_cutedsl_fallback_device_info",
        lambda device: (110, False),
    )
    monkeypatch.setattr(
        gemm_svdquant,
        "_cached_sm120_m537_l2t",
        lambda l2t: packed_l2t,
    )
    monkeypatch.setattr(gemm_svdquant.AutoTuner, "get", staticmethod(lambda: "tuner"))

    runner = svdquant_sm120_cutlass._sm120_fused_linear_runner(False, "cuda:0")
    monkeypatch.setattr(
        gemm_svdquant, "_cached_sm120_linear_runners", lambda *args: [runner]
    )

    def choose_runner(tuner, runners, tuning_config, inputs, **kwargs):
        chooser_inputs.append(inputs)
        return runner, 81

    monkeypatch.setattr(gemm_svdquant, "_choose_sm120_linear_runner", choose_runner)

    x = SimpleNamespace(ndim=2, shape=(537, 5376), device="cuda:0")
    weight = SimpleNamespace(ndim=2, shape=(5376, 2688))
    raw_l2t = SimpleNamespace(ndim=2, shape=(5376, 32))

    out = gemm_svdquant.svdquant_linear(
        x,
        weight,
        "weight_sf",
        "alpha",
        "pre_quant_scale",
        raw_l2t,
        "l1_scaled",
        "global_scale",
        bias="bias",
        enable_pdl=False,
    )

    assert out is allocated[-1]
    assert chooser_inputs[0][5] is raw_l2t
    assert native_calls[0][3] is packed_l2t
