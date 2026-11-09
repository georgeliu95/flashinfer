# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU qualification of the rounded candidate against real FlashInfer APIs."""

from __future__ import annotations

import pytest
import torch

from flashinfer.diffusion_ops import h3_bridge as epi
from flashinfer.experimental.h3_bridge.bias_epilogue import EpilogueError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _cl(shape: tuple[int, ...], *, offset: int = 0) -> torch.Tensor:
    values = torch.randn(shape, device="cuda", dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last_3d
    )
    if offset:
        backing = torch.empty(
            values.numel() + offset, device="cuda", dtype=values.dtype
        )
        view = backing[offset:].as_strided(values.shape, values.stride())
        view.copy_(values)
        return view
    return values


def _exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)


@pytest.mark.parametrize(
    "shape", [(1, 512, 42, 48, 84), (2, 7, 3, 5, 9), (3, 1, 1, 2, 3)]
)
@pytest.mark.parametrize("offset", [0, 1])
def test_bias_residual_matches_torch_bias_then_existing_add(
    shape: tuple[int, ...], offset: int
) -> None:
    from flashinfer.diffusion_ops.h3_bridge import add

    # Given BF16 CL convolution output, per-channel bias and residual.
    torch.manual_seed(121)
    conv, residual = _cl(shape, offset=offset), _cl(shape, offset=offset)
    bias_backing = torch.randn(shape[1] + offset, device="cuda", dtype=torch.bfloat16)
    bias = bias_backing[offset:]
    expected = add(conv + bias.view(1, -1, 1, 1, 1), residual)
    # When fusing the bias and residual additions.
    actual = epi.add_bias_residual(conv, bias, residual)
    # Then preserve the intermediate BF16 rounding, including offset pointers.
    _exact(actual, expected)


@pytest.mark.parametrize(
    "shape", [(1, 3008, 17, 24, 42), (2, 14, 3, 5, 9), (3, 2, 1, 2, 3)]
)
@pytest.mark.parametrize("offset", [0, 1])
def test_bias_gate_matches_torch_bias_then_existing_silu_multiply(
    shape: tuple[int, ...], offset: int
) -> None:
    from flashinfer.diffusion_ops.h3_bridge import silu_multiply

    # Given the actual interleaved CL channels containing value and gate halves.
    torch.manual_seed(122)
    conv = _cl(shape, offset=offset)
    bias_backing = torch.randn(shape[1] + offset, device="cuda", dtype=torch.bfloat16)
    bias = bias_backing[offset:]
    value, gate = (conv + bias.view(1, -1, 1, 1, 1)).chunk(2, dim=1)
    expected = silu_multiply(value, gate)
    # When fusing bias, BF16-rounded SiLU and multiplication.
    actual = epi.bias_silu_multiply(conv, bias)
    # Then exactly match the maintained arithmetic and emit compact CL output.
    _exact(actual, expected)


def test_bias_residual_keeps_observable_bf16_rounding_boundary() -> None:
    # Given a half-ULP bias at1.0 and a cancelling residual.
    conv = torch.ones((2, 7, 1, 1, 3), device="cuda", dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last_3d
    )
    bias = torch.full((7,), 1 / 256, device="cuda", dtype=torch.bfloat16)
    residual = -conv
    expected = (conv + bias.view(1, -1, 1, 1, 1)) + residual
    unrounded = (
        conv.float() + bias.float().view(1, -1, 1, 1, 1) + residual.float()
    ).bfloat16()
    assert not torch.equal(expected, unrounded)
    # When fusing, then retain that required intermediate rounding.
    _exact(epi.add_bias_residual(conv, bias, residual), expected)


def test_fresh_pointers_on_nondefault_stream() -> None:
    from flashinfer.diffusion_ops.h3_bridge import add, silu_multiply

    # Given repeated calls with fresh offset inputs on a nondefault stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for seed in (1, 2):
            torch.manual_seed(seed)
            conv, residual = (
                _cl((2, 7, 3, 5, 9), offset=1),
                _cl((2, 7, 3, 5, 9), offset=1),
            )
            gate = _cl((2, 14, 3, 5, 9), offset=1)
            bias = torch.randn(14, device="cuda", dtype=torch.bfloat16)
            expected_add = add(conv + bias[:7].view(1, -1, 1, 1, 1), residual)
            value, activation = (gate + bias.view(1, -1, 1, 1, 1)).chunk(2, dim=1)
            expected_gate = silu_multiply(value, activation)
            # When launching both candidate kernels in that stream.
            actual_add = epi.add_bias_residual(conv, bias[:7], residual)
            actual_gate = epi.bias_silu_multiply(gate, bias)
            # Then fresh pointers and stream order yield exact results.
            _exact(actual_add, expected_add)
            _exact(actual_gate, expected_gate)
    stream.synchronize()


def test_warm_cuda_graph_replays_new_values() -> None:
    from flashinfer.diffusion_ops.h3_bridge import add, silu_multiply

    # Given static buffers and both specializations warmed before capture.
    conv, residual = _cl((2, 7, 3, 5, 9)), _cl((2, 7, 3, 5, 9))
    gate, bias = (
        _cl((2, 14, 3, 5, 9)),
        torch.randn(14, device="cuda", dtype=torch.bfloat16),
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            epi.add_bias_residual(conv, bias[:7], residual)
            epi.bias_silu_multiply(gate, bias)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    # When capturing the candidate and replaying after input mutation.
    with torch.cuda.graph(graph, stream=stream):
        actual_add = epi.add_bias_residual(conv, bias[:7], residual)
        actual_gate = epi.bias_silu_multiply(gate, bias)
    conv.add_(0.25)
    gate.mul_(0.5)
    graph.replay()
    # Then replay consumes current buffer values and matches the real API.
    expected_add = add(conv + bias[:7].view(1, -1, 1, 1, 1), residual)
    value, activation = (gate + bias.view(1, -1, 1, 1, 1)).chunk(2, dim=1)
    _exact(actual_add, expected_add)
    _exact(actual_gate, silu_multiply(value, activation))


def test_nonchannelslast_input_is_rejected() -> None:
    # Given ordinary NCDHW storage outside this candidate's explicit contract.
    conv = torch.randn((2, 7, 3, 5, 9), device="cuda", dtype=torch.bfloat16)
    bias = torch.ones(7, device="cuda", dtype=torch.bfloat16)
    # When called, then reject rather than silently interpret NCDHW as NDHWC.
    with pytest.raises(EpilogueError, match="channels_last_3d"):
        epi.add_bias_residual(conv, bias, conv)


@pytest.mark.parametrize("bias_channels", [6, 8])
def test_bias_channel_mismatch_is_rejected(bias_channels: int) -> None:
    # Given valid CL inputs with a channel vector of the wrong length.
    conv = _cl((2, 7, 3, 5, 9))
    bias = torch.ones(bias_channels, device="cuda", dtype=torch.bfloat16)
    # When called, then reject before launching a kernel with invalid bias indices.
    with pytest.raises(EpilogueError, match="bias"):
        epi.add_bias_residual(conv, bias, conv)


@pytest.mark.parametrize("gradient_operand", [0, 1, 2])
def test_gradient_tracking_is_rejected(gradient_operand: int) -> None:
    # Given one operand requiring a gradient in an inference-only operation.
    operands = [
        _cl((2, 7, 3, 5, 9)),
        torch.ones(7, device="cuda", dtype=torch.bfloat16),
        _cl((2, 7, 3, 5, 9)),
    ]
    operands[gradient_operand].requires_grad_(True)
    # When called, then do not silently detach any of the inputs.
    with pytest.raises(EpilogueError, match="inference"):
        epi.add_bias_residual(*operands)


def test_residual_shape_mismatch_is_rejected() -> None:
    # Given a residual with different spatial extent.
    conv, residual = _cl((2, 7, 3, 5, 9)), _cl((2, 7, 3, 5, 8))
    bias = torch.ones(7, device="cuda", dtype=torch.bfloat16)
    # When called, then reject rather than read beyond the residual allocation.
    with pytest.raises(EpilogueError, match="residual"):
        epi.add_bias_residual(conv, bias, residual)


def test_noncontiguous_bias_is_rejected() -> None:
    # Given a sliced channel vector with a non-unit storage stride.
    conv = _cl((2, 14, 3, 5, 9))
    bias = torch.ones(28, device="cuda", dtype=torch.bfloat16)[::2]
    # When called, then retain the explicit contiguous channel-vector contract.
    with pytest.raises(EpilogueError, match="bias"):
        epi.bias_silu_multiply(conv, bias)


def test_float32_activation_is_rejected() -> None:
    # Given a valid layout but an unsupported activation dtype.
    conv = _cl((2, 14, 3, 5, 9)).float()
    bias = torch.ones(14, device="cuda", dtype=torch.bfloat16)
    # When called, then do not reinterpret FP32 storage as BF16.
    with pytest.raises(EpilogueError, match="BF16"):
        epi.bias_silu_multiply(conv, bias)
