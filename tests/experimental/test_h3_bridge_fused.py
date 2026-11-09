"""Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from flashinfer.experimental.h3_bridge import fused

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_plan_includes_all_channels_and_time() -> None:
    # Given the deployed upscaler's entire activation domain.
    # When planning the reduction, then every T/H/W position belongs to its group.
    plan = fused.GroupNormPlan.from_shape((1, 512, 42, 48, 84), groups=32, eps=1e-5)
    assert plan.group_elements == 16 * 42 * 48 * 84
    assert plan.parts == (plan.group_elements + 4095) // 4096


@pytest.mark.parametrize(
    "shape,groups,eps",
    [
        ((1, 7, 2, 3, 4), 2, 1e-5),
        ((1, 8, 0, 3, 4), 2, 1e-5),
        ((1, 8, 3, 4), 2, 1e-5),
        ((1, 8, 2, 3, 4), 0, 1e-5),
        ((1, 8, 2, 3, 4), 2, 0.0),
        ((1, 8, 2, 3, 4), 2, float("nan")),
    ],
)
def test_invalid_plan_is_rejected(
    shape: tuple[int, ...], groups: int, eps: float
) -> None:
    # Given invalid metadata, when planning, then reject before GPU work.
    with pytest.raises(fused.BridgeFusionError):
        fused.GroupNormPlan.from_shape(shape, groups=groups, eps=eps)


@CUDA
@pytest.mark.parametrize("channels,groups", [(32, 8), (94, 2), (752, 16)])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("modulation", [False, True])
def test_group_norm_matches_full_domain(
    channels: int, groups: int, channels_last: bool, modulation: bool
) -> None:
    # Given batch-specific, temporally shifted data including 47-channel groups.
    torch.manual_seed(1101)
    x = torch.randn((2, channels, 3, 5, 7), device="cuda", dtype=torch.bfloat16)
    x += torch.arange(3, device="cuda", dtype=torch.bfloat16).view(1, 1, 3, 1, 1) * 3
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last_3d)
    weight = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn_like(weight)
    scale = torch.randn_like(weight) if modulation else None
    shift = torch.randn_like(weight) if modulation else None
    expected = F.group_norm(x, groups, weight, bias, 1e-5)
    if scale is not None and shift is not None:
        expected = expected * (1 + scale.view(1, channels, 1, 1, 1)) + shift.view(
            1, channels, 1, 1, 1
        )
    expected = F.silu(expected)
    # When fusing normalization, modulation and activation.
    actual = fused.group_norm_silu(
        x, weight, bias, groups=groups, eps=1e-5, scale=scale, shift=shift
    )
    # Then match the full-domain reference and emit convolution-ready layout.
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.008)
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)


@CUDA
@pytest.mark.parametrize(
    "shape,groups,eps",
    [((1, 512, 42, 48, 84), 32, 1e-5), ((1, 752, 17, 24, 42), 16, 1e-6)],
)
def test_group_norm_real_reduction_sizes(
    shape: tuple[int, ...], groups: int, eps: float
) -> None:
    # Given real multi-part reduction extents and BF16 affine parameters.
    torch.manual_seed(1101)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last_3d
    )
    weight = torch.randn(shape[1], device="cuda", dtype=torch.bfloat16)
    bias = torch.randn_like(weight)
    expected = F.silu(F.group_norm(x, groups, weight, bias, eps))
    # When reducing, then retain the complete group rather than per-part statistics.
    actual = fused.group_norm_silu(x, weight, bias, groups=groups, eps=eps)
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.008)


@CUDA
def test_modulation_preserves_bf16_intermediates() -> None:
    # Given values distinguishing BF16 eager arithmetic from one FP32 expression.
    torch.manual_seed(1101)
    x = torch.randn((2, 32, 3, 5, 7), device="cuda", dtype=torch.bfloat16)
    scale = torch.randn((1, 32, 1, 1, 1), device="cuda", dtype=torch.bfloat16)
    shift = torch.randn_like(scale)
    expected = F.silu(x * (1 + scale) + shift)
    unrounded = F.silu(x.float() * (1 + scale.float()) + shift.float()).bfloat16()
    # When fused, then intermediate rounding remains observable.
    actual = fused.modulate_silu(x, scale, shift)
    torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.008)
    assert not torch.equal(expected, unrounded)
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)


@CUDA
def test_silu_multiply_accepts_strided_gate() -> None:
    # Given channel-chunk views with gaps in the channels-last row stride.
    torch.manual_seed(1101)
    projection = torch.randn(
        (2, 94, 3, 5, 7), device="cuda", dtype=torch.bfloat16
    ).contiguous(memory_format=torch.channels_last_3d)
    value, gate = projection.chunk(2, dim=1)
    expected = value * F.silu(gate)
    # When fusing, then honor strides and the BF16 SiLU rounding boundary.
    actual = fused.silu_multiply(value, gate)
    torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.008)
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)


@CUDA
def test_add_and_silu_keep_channels_last() -> None:
    # Given differently laid-out BF16 values, when fused, then preserve results/layout.
    x = torch.randn((1, 47, 3, 5, 7), device="cuda", dtype=torch.bfloat16)
    y = torch.randn_like(x).contiguous(memory_format=torch.channels_last_3d)
    output = fused.add(x, y)
    assert torch.equal(output, x + y)
    activated = fused.silu(x)
    torch.testing.assert_close(activated, F.silu(x), atol=0.015625, rtol=0.008)
    assert output.is_contiguous(memory_format=torch.channels_last_3d)
    assert activated.is_contiguous(memory_format=torch.channels_last_3d)


def test_cpu_input_is_rejected() -> None:
    # Given CPU data, when invoking the backend, then reject before kernel import.
    x = torch.ones((1, 8, 2, 3, 4), dtype=torch.bfloat16)
    with pytest.raises(fused.BridgeFusionError, match="CUDA BF16"):
        fused.add(x, x)


@CUDA
@pytest.mark.parametrize("invalid", ["dtype", "grad", "shape", "empty"])
def test_pointwise_invalid_contracts(invalid: str) -> None:
    # Given an unsupported pointwise operand contract, when called, then reject.
    x = torch.ones((1, 8, 2, 3, 4), device="cuda", dtype=torch.bfloat16)
    bad = {
        "dtype": x.float(),
        "grad": x.clone().requires_grad_(),
        "shape": x[:, :4],
        "empty": x[:, :, :0],
    }[invalid]
    with pytest.raises(fused.BridgeFusionError):
        fused.add(bad, x)


@CUDA
@pytest.mark.parametrize(
    "invalid", ["affine", "affine_stride", "input_stride", "missing_shift", "scale"]
)
def test_group_norm_invalid_contracts(invalid: str) -> None:
    # Given a valid norm request with one malformed boundary field.
    x = torch.ones((1, 8, 2, 3, 4), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(8, device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros_like(weight)
    inputs = x[:, :, :, :, ::2] if invalid == "input_stride" else x
    affine = {
        "affine": weight[:4],
        "affine_stride": torch.ones(16, device="cuda", dtype=torch.bfloat16)[::2],
    }.get(invalid, weight)
    scale = (
        weight
        if invalid == "missing_shift"
        else weight[:4]
        if invalid == "scale"
        else None
    )
    shift = weight if invalid == "scale" else None
    # When entering the backend, then fail before launch.
    with pytest.raises(fused.BridgeFusionError):
        fused.group_norm_silu(
            inputs, affine, bias, groups=2, eps=1e-5, scale=scale, shift=shift
        )


@CUDA
def test_current_stream_and_warm_graph() -> None:
    # Given a warmed chain on a nondefault stream, with mutable graph inputs.
    x = torch.randn((1, 94, 3, 5, 7), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(94, device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros_like(weight)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def operation() -> torch.Tensor:
        normalized = fused.group_norm_silu(x, weight, bias, groups=2, eps=1e-5)
        return fused.silu(fused.add(normalized, normalized))

    with torch.cuda.stream(stream):
        for _ in range(3):
            operation()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = operation()
    # When replaying after changing input on that stream.
    with torch.cuda.stream(stream):
        x.fill_(2)
        graph.replay()
    stream.synchronize()
    expected = F.silu(F.silu(F.group_norm(x, 2, weight, bias, 1e-5)) * 2)
    # Then all kernels respect the active stream and graph's live input storage.
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.008)


@CUDA
def test_multi_device_guard_restores_current_device() -> None:
    # Given inputs on GPU1 while GPU0 is current.
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    with torch.cuda.device(0):
        x = torch.ones((1, 8, 2, 3, 4), device="cuda:1", dtype=torch.bfloat16)
        # When launching, then use GPU1's stream and restore GPU0 afterward.
        result = fused.add(x, x)
        torch.cuda.synchronize(1)
        assert torch.cuda.current_device() == 0
        assert torch.equal(result, x * 2)
        with pytest.raises(fused.BridgeFusionError):
            fused.add(x, x.to("cuda:0"))
