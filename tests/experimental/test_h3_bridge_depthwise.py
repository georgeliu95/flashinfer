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

"""Correctness and dispatch contracts for experimental H3 temporal depthwise."""

from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn.functional as functional


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.cuda.set_per_process_memory_fraction(
        24 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    return torch.device("cuda", torch.cuda.current_device())


def _operands(
    device: torch.device,
    shape: tuple[int, ...],
    taps: int,
    channels_last: bool = True,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, c, t, h, w = shape
    x = torch.randn(n * c * t * h * w + offset, device=device, dtype=torch.bfloat16)[
        offset:
    ]
    x = x.view(n, t, h, w, c).permute(0, 4, 1, 2, 3) if channels_last else x.view(shape)
    weight = torch.randn(c * taps + offset, device=device, dtype=torch.bfloat16)[
        offset:
    ].view(c, 1, taps, 1, 1)
    bias = torch.randn(c + offset, device=device, dtype=torch.bfloat16)[offset:]
    return x, weight, bias


def _reference(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, bias_first: bool
) -> torch.Tensor:
    """Model one FP32 rounding per fused BF16-product plus accumulator."""
    taps = weight.shape[2]
    padded = functional.pad(x.double(), (0, 0, 0, 0, taps // 2, taps // 2))
    b = (
        torch.zeros((1, x.shape[1], 1, 1, 1), device=x.device)
        if bias is None
        else bias.float().view(1, -1, 1, 1, 1)
    )
    acc = (
        b.expand_as(x).clone()
        if bias_first
        else torch.zeros_like(x, dtype=torch.float32)
    )
    for tap in range(taps):
        acc = (
            padded[:, :, tap : tap + x.shape[2]]
            * weight[:, 0, tap, 0, 0].double().view(1, -1, 1, 1, 1)
            + acc.double()
        ).float()
    if not bias_first:
        acc = acc + b
    return acc.to(torch.bfloat16)


@pytest.mark.parametrize("taps,frames", [(3, 1), (3, 4), (5, 2), (5, 7)])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize(
    "bias_first,has_bias", [(True, True), (False, True), (True, False)]
)
def test_triton_matches_ordered_reference(
    device: torch.device,
    taps: int,
    frames: int,
    channels_last: bool,
    bias_first: bool,
    has_bias: bool,
) -> None:
    from flashinfer.experimental.h3_bridge.depthwise import depthwise_temporal

    # Given BF16 inputs including time boundaries and nonzero storage offsets.
    x, w, b = _operands(device, (2, 7, frames, 3, 5), taps, channels_last, 1)
    bias = b if has_bias else None
    # When temporal convolution uses the Triton backend.
    result = depthwise_temporal(x, w, bias, bias_first=bias_first, backend="triton")
    # Then its output matches ascending FP32 FMA and is channels-last.
    assert torch.equal(result, _reference(x, w, bias, bias_first))
    assert result.is_contiguous(memory_format=torch.channels_last_3d)


@pytest.mark.parametrize("frames,offset", [(20, 0), (20, 1), (42, 0)])
def test_cute_fresh_pointers_and_current_stream(
    device: torch.device, frames: int, offset: int
) -> None:
    from flashinfer.experimental.h3_bridge.depthwise import depthwise_temporal

    if (
        torch.cuda.get_device_capability(device) != (12, 1)
        or importlib.util.find_spec("cutlass") is None
    ):
        pytest.skip("CuTe candidate requires SM121 and CUTLASS DSL")
    # Given new input, weight and bias allocations on each invocation.
    for _ in range(2):
        x, w, b = _operands(device, (1, 512, frames, 48, 84), 5, offset=offset)
        expected = depthwise_temporal(x, w, b, backend="triton")
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        # When CuTe launches on the current nondefault stream.
        with torch.cuda.stream(stream):
            result = depthwise_temporal(x, w, b, backend="cute")
        torch.cuda.current_stream(device).wait_stream(stream)
        # Then cached code uses fresh pointers and the requested stream.
        assert torch.equal(result, expected)
        assert result.is_contiguous(memory_format=torch.channels_last_3d)


@pytest.mark.parametrize("backend", ["triton", "cute", "auto"])
def test_warm_cuda_graph_replays_new_values(device: torch.device, backend: str) -> None:
    from flashinfer.experimental.h3_bridge.depthwise import depthwise_temporal

    if backend != "triton" and (
        torch.cuda.get_device_capability(device) != (12, 1)
        or importlib.util.find_spec("cutlass") is None
    ):
        pytest.skip("CuTe graph requires SM121 and CUTLASS DSL")
    # Given a warmed specialization and fixed graph-owned tensor addresses.
    shape = (1, 512, 20, 48, 84) if backend != "triton" else (2, 7, 4, 3, 5)
    x, w, b = _operands(device, shape, 5)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            depthwise_temporal(x, w, b, backend=backend)
    torch.cuda.current_stream(device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = depthwise_temporal(x, w, b, backend=backend)
    # When input values change and the warm CUDA graph replays.
    x.add_(0.125)
    graph.replay()
    # Then captured work observes the new data and preserves the Triton result.
    assert torch.equal(result, depthwise_temporal(x, w, b, backend="triton"))


@pytest.mark.parametrize(
    "reason",
    ["dtype", "shape", "empty", "weight", "bias", "strides", "grad", "cpu", "backend"],
)
def test_invalid_contract_is_rejected(device: torch.device, reason: str) -> None:
    from flashinfer.experimental.h3_bridge.depthwise import depthwise_temporal

    # Given one invalid property of otherwise legal operands.
    x, w, b = _operands(device, (1, 8, 4, 3, 5), 3)
    backend = "triton"
    mutations = {
        "dtype": lambda: (x.float(), w, b),
        "shape": lambda: (x[0], w, b),
        "empty": lambda: (x[:, :, :0], w, b),
        "weight": lambda: (x, w[:, :, :2], b),
        "bias": lambda: (x, w, b.view(2, 4)),
        "strides": lambda: (x[:, :, :, :, ::2], w, b),
        "grad": lambda: (x.requires_grad_(), w, b),
        "cpu": lambda: (x.cpu(), w.cpu(), b.cpu()),
        "backend": lambda: (x, w, b),
    }
    x, w, b = mutations[reason]()
    if reason == "backend":
        backend = "invalid"
    # When the API validates its inference contract, then fail before launch.
    with pytest.raises((ValueError, TypeError), match="."):
        depthwise_temporal(x, w, b, backend=backend)


@pytest.mark.parametrize(
    "channels_last,has_bias,bias_first",
    [(False, True, True), (True, False, True), (True, True, False)],
)
def test_explicit_cute_rejects_unmeasured_contract(
    device: torch.device, channels_last: bool, has_bias: bool, bias_first: bool
) -> None:
    from flashinfer.experimental.h3_bridge.depthwise import depthwise_temporal

    # Given an unmeasured CuTe contract.
    x, w, b = _operands(device, (1, 512, 20, 48, 84), 5, channels_last)
    # When CuTe is explicitly requested, then reject the unsupported contract.
    with pytest.raises(ValueError, match="CuTe"):
        depthwise_temporal(
            x, w, b if has_bias else None, bias_first=bias_first, backend="cute"
        )


def test_auto_falls_back_only_for_missing_dependency(
    device: torch.device, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flashinfer.experimental.h3_bridge import depthwise

    # Given a supported shape but an unavailable optional CuTe dependency.
    x, w, b = _operands(device, (1, 512, 20, 48, 84), 5)
    monkeypatch.setattr(depthwise, "_cute_available", lambda: False)
    # When auto selects its backend, then it still produces the Triton result.
    assert torch.equal(
        depthwise.depthwise_temporal(x, w, b),
        depthwise.depthwise_temporal(x, w, b, backend="triton"),
    )
    if torch.cuda.get_device_capability(device) == (12, 1):
        with pytest.raises(ImportError, match="CuTe"):
            depthwise.depthwise_temporal(x, w, b, backend="cute")


def test_auto_does_not_hide_cute_compile_errors(
    device: torch.device, monkeypatch: pytest.MonkeyPatch
) -> None:
    if importlib.util.find_spec("cutlass") is None:
        pytest.skip("CUTLASS DSL is unavailable")
    from flashinfer.experimental.h3_bridge import _depthwise_cute, depthwise

    if torch.cuda.get_device_capability(device) != (12, 1):
        pytest.skip("CuTe dispatch is SM121 only")
    # Given an eligible shape and a backend compilation failure.
    x, w, b = _operands(device, (1, 512, 20, 48, 84), 5)
    failure = RuntimeError("sentinel compilation failure")

    def fail(*args: torch.Tensor) -> torch.Tensor:
        raise failure

    monkeypatch.setattr(depthwise, "_cute_available", lambda: True)
    monkeypatch.setattr(_depthwise_cute, "run", fail)
    # When auto dispatches CuTe, then the original failure remains visible.
    with pytest.raises(RuntimeError, match="sentinel compilation failure"):
        depthwise.depthwise_temporal(x, w, b)


def test_bias_order_preserves_cancellation_boundary(device: torch.device) -> None:
    from flashinfer.experimental.h3_bridge.depthwise import depthwise_temporal

    # Given cancellation that loses a small initial bias in FP32 accumulation.
    x = torch.tensor([2**20, -(2**20), 0], device=device, dtype=torch.bfloat16).view(
        1, 1, 3, 1, 1
    )
    weight = torch.ones((1, 1, 3, 1, 1), device=device, dtype=torch.bfloat16)
    bias = torch.tensor([2**-6], device=device, dtype=torch.bfloat16)
    # When the requested bias ordering changes.
    first = depthwise_temporal(x, weight, bias, bias_first=True, backend="triton")
    last = depthwise_temporal(x, weight, bias, bias_first=False, backend="triton")
    # Then the BF16 boundary remains observable instead of being reassociated.
    assert first[0, 0, 1, 0, 0].item() == 0
    assert last[0, 0, 1, 0, 0].item() == 2**-6


def test_auto_unmeasured_shape_does_not_import_cute(
    device: torch.device, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flashinfer.experimental.h3_bridge import depthwise

    # Given a legal shape outside the measured CuTe region.
    x, w, b = _operands(device, (1, 8, 3, 2, 2), 5)

    def unexpected_import() -> bool:
        pytest.fail("unmeasured shape attempted a CuTe dependency import")

    monkeypatch.setattr(depthwise, "_cute_available", unexpected_import)
    # When auto selects a backend, then Triton works without probing CuTe.
    assert torch.equal(
        depthwise.depthwise_temporal(x, w, b),
        depthwise.depthwise_temporal(x, w, b, backend="triton"),
    )
    with pytest.raises(ValueError, match="CuTe"):
        depthwise.depthwise_temporal(x, w, b, backend="cute")
