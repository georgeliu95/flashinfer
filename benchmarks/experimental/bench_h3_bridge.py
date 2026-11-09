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
import os
import random
import time
from pathlib import Path
import statistics
from typing import TypedDict

import torch

from flashinfer import autotune
from flashinfer.diffusion_ops import h3_bridge
from flashinfer.experimental.h3_bridge._tuning import TacticSelection, selected_tactic


class BenchmarkError(RuntimeError):
    """A benchmark failed to exercise the required tuned path."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class Measurement(TypedDict):
    operator: str
    shape: tuple[int, ...]
    gpu: str
    torch: str
    milliseconds: dict[str, float]
    taps: int | None
    groups: int | None
    eps: float | None
    selection: TacticSelection | None
    tuning_seconds: float | None


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
    selection: TacticSelection | None = None,
    tuning_seconds: float | None = None,
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
        "selection": selection,
        "tuning_seconds": tuning_seconds,
    }
    print(json.dumps(record), flush=True)


def benchmark_depthwise(shape: tuple[int, ...], taps: int, cache: str) -> None:
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    x = x.contiguous(memory_format=torch.channels_last_3d)
    w = torch.randn((shape[1], 1, taps, 1, 1), device=x.device, dtype=x.dtype)
    bias = torch.randn(shape[1], device=x.device, dtype=x.dtype)

    def triton() -> torch.Tensor:
        return h3_bridge.depthwise_temporal(x, w, bias, backend="triton")

    def auto() -> torch.Tensor:
        return h3_bridge.depthwise_temporal(x, w, bias)

    torch.testing.assert_close(auto(), triton(), atol=0, rtol=0)
    calls: dict[str, Callable[[], torch.Tensor]] = {
        "triton_default": triton,
        "auto_tuned": auto,
    }
    if (
        torch.cuda.get_device_capability() == (12, 1)
        and importlib.util.find_spec("cutlass") is not None
    ):

        def cute() -> torch.Tensor:
            return h3_bridge.depthwise_temporal(x, w, bias, backend="cute")

        torch.testing.assert_close(cute(), triton(), atol=0, rtol=0)
        calls["cute_default"] = cute
    begin = time.perf_counter()
    with autotune(cache=cache):
        tuned = auto()
    tuning_seconds = time.perf_counter() - begin
    torch.testing.assert_close(tuned, triton(), atol=0, rtol=0)
    selection = selected_tactic(x, w, bias, bias_first=True, backend="auto")
    if not selection["cache_hit"]:
        raise BenchmarkError("benchmark did not produce a tuned DW selection")
    report(
        "temporal_depthwise",
        shape,
        measure(calls),
        taps=taps,
        selection=selection,
        tuning_seconds=tuning_seconds,
    )


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
    cache_root = Path(os.environ.get("FLASHINFER_WORKSPACE_BASE", Path.home()))
    cache = cache_root / ".cache" / "flashinfer" / "h3-bridge-tactics.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for frames in (20, 42):
            for height, width in ((24, 42), (48, 84)):
                benchmark_depthwise((1, 512, frames, height, width), 5, str(cache))
        benchmark_depthwise((1, 752, 17, 24, 42), 3, str(cache))
        benchmark_depthwise((2, 7, 7, 3, 5), 3, str(cache))
        benchmark_norm((1, 512, 42, 48, 84), 32, 1e-5)
        benchmark_norm((1, 752, 17, 24, 42), 16, 1e-6)


if __name__ == "__main__":
    main()
