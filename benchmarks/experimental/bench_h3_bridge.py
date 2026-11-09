#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "triton", "nvidia-cutlass-dsl==4.8.1"]
# ///
# ─── How to run ───
# With this FlashInfer checkout installed in the CUDA environment:
# python benchmarks/experimental/bench_h3_bridge.py
# ──────────────────
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

"""Serial CUDA-event benchmark of H3 bridge operators on their real shapes."""

from __future__ import annotations

from collections.abc import Callable
import json
import importlib.util
import random
import statistics
from typing import TypedDict

import torch

from flashinfer.diffusion_ops import h3_bridge


class Measurement(TypedDict):
    operator: str
    shape: tuple[int, ...]
    gpu: str
    torch: str
    milliseconds: dict[str, float]
    taps: int | None
    groups: int | None
    eps: float | None


def measure(calls: dict[str, Callable[[], torch.Tensor]]) -> dict[str, float]:
    """Exclude compile/warmup; shuffle serial calls within ten paired rounds."""
    for call in calls.values():
        for _ in range(3):
            call()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {name: [] for name in calls}
    rng = random.Random(1109)
    for _ in range(10):
        order = list(calls)
        rng.shuffle(order)
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            calls[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
    return {name: statistics.median(times) for name, times in samples.items()}


def report(
    operator: str,
    shape: tuple[int, ...],
    times: dict[str, float],
    *,
    taps: int | None = None,
    groups: int | None = None,
    eps: float | None = None,
) -> None:
    record: Measurement = {
        "operator": operator,
        "shape": shape,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "milliseconds": times,
        "taps": taps,
        "groups": groups,
        "eps": eps,
    }
    print(json.dumps(record), flush=True)


def benchmark_depthwise(shape: tuple[int, ...], taps: int) -> None:
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    x = x.contiguous(memory_format=torch.channels_last_3d)
    w = torch.randn((shape[1], 1, taps, 1, 1), device=x.device, dtype=x.dtype)
    bias = torch.randn(shape[1], device=x.device, dtype=x.dtype)

    def triton() -> torch.Tensor:
        return h3_bridge.depthwise_temporal(x, w, bias, backend="triton")

    def auto() -> torch.Tensor:
        return h3_bridge.depthwise_temporal(x, w, bias)

    torch.testing.assert_close(auto(), triton(), atol=0, rtol=0)
    calls: dict[str, Callable[[], torch.Tensor]] = {"triton": triton, "auto": auto}
    if (
        torch.cuda.get_device_capability() == (12, 1)
        and shape[1] == 512
        and taps == 5
        and importlib.util.find_spec("cutlass") is not None
    ):

        def cute() -> torch.Tensor:
            return h3_bridge.depthwise_temporal(x, w, bias, backend="cute")

        torch.testing.assert_close(cute(), triton(), atol=0, rtol=0)
        calls["cute"] = cute
    report("temporal_depthwise", shape, measure(calls), taps=taps)


def benchmark_norm(shape: tuple[int, ...], groups: int, eps: float) -> None:
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    x = x.contiguous(memory_format=torch.channels_last_3d)
    w = torch.randn(shape[1], device=x.device, dtype=x.dtype)
    bias = torch.randn_like(w)

    def native() -> torch.Tensor:
        return torch.nn.functional.silu(
            torch.nn.functional.group_norm(x, groups, w, bias, eps)
        ).contiguous(memory_format=torch.channels_last_3d)

    def fused() -> torch.Tensor:
        return h3_bridge.group_norm_silu(x, w, bias, groups=groups, eps=eps)

    torch.testing.assert_close(fused(), native(), atol=0.03125, rtol=0.008)
    report(
        "group_norm_silu_to_channels_last",
        shape,
        measure({"native": native, "fused": fused}),
        groups=groups,
        eps=eps,
    )


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(1109)
    with torch.inference_mode():
        for frames in (20, 42):
            benchmark_depthwise((1, 512, frames, 48, 84), 5)
        benchmark_depthwise((1, 752, 17, 24, 42), 3)
        benchmark_norm((1, 512, 42, 48, 84), 32, 1e-5)
        benchmark_norm((1, 752, 17, 24, 42), 16, 1e-6)


if __name__ == "__main__":
    main()
