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

"""Private backend contracts for generic H3 depthwise autotuning tactics."""

from __future__ import annotations

import importlib.util

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("Generic CuTe depthwise is qualified on SM121")
    if importlib.util.find_spec("cutlass") is None:
        pytest.skip("CUTLASS DSL is unavailable")
    torch.cuda.set_per_process_memory_fraction(
        24 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    return torch.device("cuda", torch.cuda.current_device())


def _operands(
    device: torch.device, shape: tuple[int, ...], taps: int, offset: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, c, t, h, w = shape
    storage = torch.randn(
        n * c * t * h * w + offset, device=device, dtype=torch.bfloat16
    )
    x = storage[offset:].view(n, t, h, w, c).permute(0, 4, 1, 2, 3)
    weight = torch.randn(c * taps + offset, device=device, dtype=torch.bfloat16)[
        offset:
    ].view(c, 1, taps, 1, 1)
    bias = torch.randn(c + offset, device=device, dtype=torch.bfloat16)[offset:]
    return x, weight, bias


@pytest.mark.parametrize(
    "shape,taps,time_tile,threads,vec,offset",
    [
        ((2, 7, 1, 3, 5), 3, 4, 128, 1, 1),
        ((2, 7, 2, 3, 5), 5, 1, 128, 1, 1),
        ((2, 8, 7, 3, 5), 5, 4, 128, 2, 0),
        ((3, 12, 9, 2, 3), 3, 8, 256, 4, 1),
        ((2, 4, 3, 1, 1), 5, 16, 128, 2, 0),
    ],
)
@pytest.mark.parametrize(
    "has_bias,bias_first", [(True, True), (True, False), (False, True), (False, False)]
)
def test_generic_cute_matches_triton(
    device: torch.device,
    shape: tuple[int, ...],
    taps: int,
    time_tile: int,
    threads: int,
    vec: int,
    offset: int,
    has_bias: bool,
    bias_first: bool,
) -> None:
    from flashinfer.experimental.h3_bridge import _depthwise_cute, _depthwise_triton

    # Given batched channels-last inputs, short/non-tile-aligned time and offsets.
    x, w, b = _operands(device, shape, taps, offset)
    bias = b if has_bias else None
    expected = _depthwise_triton.run(x, w, bias, bias_first=bias_first)
    out = torch.empty_like(x, memory_format=torch.channels_last_3d)
    # When a generic tactic writes the caller-provided output.
    result = _depthwise_cute.run(
        x,
        w,
        bias,
        bias_first=bias_first,
        time_tile=time_tile,
        threads=threads,
        vec=vec,
        out=out,
    )
    # Then padding, batch isolation, FP32 FMA order and output identity are exact.
    assert result is out
    assert torch.equal(result, expected)


@pytest.mark.parametrize("frames", [20, 42])
def test_original_high_shapes_remain_exact(device: torch.device, frames: int) -> None:
    from flashinfer.experimental.h3_bridge import _depthwise_cute, _depthwise_triton

    # Given the two previously qualified high-resolution shapes.
    x, w, b = _operands(device, (1, 512, frames, 48, 84), 5)
    expected = _depthwise_triton.run(x, w, b, bias_first=True)
    # When the original time20/threads128/vec2 tactic is used.
    actual = _depthwise_cute.run(x, w, b, time_tile=20, threads=128, vec=2)
    # Then generalizing the kernel preserves every BF16 result.
    assert torch.equal(actual, expected)


def test_generic_tactics_fresh_pointers_and_stream(device: torch.device) -> None:
    from flashinfer.experimental.h3_bridge import _depthwise_cute, _depthwise_triton

    # Given different allocations with identical odd-channel metadata.
    for _ in range(2):
        x, w, b = _operands(device, (2, 7, 5, 2, 3), 3, 1)
        expected = _depthwise_triton.run(x, w, b, bias_first=False)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        # When several tactics launch on the current nondefault stream.
        with torch.cuda.stream(stream):
            outputs = [
                _depthwise_cute.run(
                    x, w, b, bias_first=False, time_tile=tile, threads=128
                )
                for tile in (1, 4, 8, 20, 32)
            ]
        torch.cuda.current_stream(device).wait_stream(stream)
        # Then default vectorization is safe for odd C and pointers/stream are fresh.
        assert all(torch.equal(value, expected) for value in outputs)


@pytest.mark.parametrize("use_cute", [False, True])
def test_generic_warm_cuda_graph(device: torch.device, use_cute: bool) -> None:
    from flashinfer.experimental.h3_bridge import _depthwise_cute, _depthwise_triton

    # Given warmed code and caller-owned output with a nonzero storage offset.
    x, w, _ = _operands(device, (2, 7, 3, 2, 5), 5, 1)
    out, _, _ = _operands(device, tuple(x.shape), 5, 1)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))

    def run() -> torch.Tensor:
        if use_cute:
            return _depthwise_cute.run(
                x, w, None, bias_first=False, time_tile=4, out=out
            )
        return _depthwise_triton.run(
            x, w, None, bias_first=False, block=256, num_warps=4, out=out
        )

    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream(device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = run()
    # When values change and the warm graph replays.
    x.add_(0.125)
    graph.replay()
    # Then both backends write the same fixed buffer and observe current data.
    assert result is out
    assert torch.equal(result, _depthwise_triton.run(x, w, None, bias_first=False))


def test_generic_invalid_vector_width_is_rejected(device: torch.device) -> None:
    from flashinfer.experimental.h3_bridge import _depthwise_cute

    # Given channels not divisible by the requested vector width.
    x, w, b = _operands(device, (2, 7, 2, 3, 5), 5)
    # When a vector tactic is explicitly selected, then reject before any overread.
    with pytest.raises(ValueError, match="vec"):
        _depthwise_cute.run(x, w, b, vec=2)


def test_triton_tactics_reuse_caller_output(device: torch.device) -> None:
    from flashinfer.experimental.h3_bridge import _depthwise_triton

    # Given a fixed output buffer and several valid autotuner configurations.
    x, w, b = _operands(device, (2, 7, 5, 3, 5), 5, 1)
    expected = _depthwise_triton.run(x, w, b)
    out = torch.empty_like(x, memory_format=torch.channels_last_3d)
    # When block/warp choices change, then only launch mapping changes.
    for block, warps in ((256, 2), (512, 4), (1024, 4), (2048, 8)):
        actual = _depthwise_triton.run(x, w, b, block=block, num_warps=warps, out=out)
        assert actual is out
        assert torch.equal(actual, expected)


def test_generic_bias_order_preserves_rounding_boundary(device: torch.device) -> None:
    from flashinfer.experimental.h3_bridge import _depthwise_cute

    # Given BF16 cancellation that erases a small initial FP32 bias.
    x = torch.tensor([2**20, -(2**20), 0], device=device, dtype=torch.bfloat16)
    x = x.view(1, 1, 3, 1, 1)
    weight = torch.ones((1, 1, 3, 1, 1), device=device, dtype=torch.bfloat16)
    bias = torch.tensor([2**-6], device=device, dtype=torch.bfloat16)
    # When the explicitly requested bias order changes.
    first = _depthwise_cute.run(x, weight, bias, bias_first=True, time_tile=4)
    last = _depthwise_cute.run(x, weight, bias, bias_first=False, time_tile=4)
    # Then the final BF16 boundary reflects the original FP32 operation order.
    assert first[0, 0, 1, 0, 0].item() == 0
    assert last[0, 0, 1, 0, 0].item() == 2**-6


def test_cold_capture_error_survives_context_manager(device: torch.device) -> None:
    from flashinfer.experimental.h3_bridge import _depthwise_cute

    # Given a specialization never called eagerly in this process.
    x, w, b = _operands(device, (1, 3, 1, 1, 1), 3)
    graph = torch.cuda.CUDAGraph()
    # When graph capture would compile, then preserve the actionable exception.
    with pytest.raises(RuntimeError, match="warm CuTe"), torch.cuda.graph(graph):
        _depthwise_cute.run(x, w, b, time_tile=4)
