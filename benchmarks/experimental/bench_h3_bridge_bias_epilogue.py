#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "triton", "typer>=0.12"]
# ///
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

"""Compare complete Conv3D + bias + consumer paths; print one JSON report."""

from __future__ import annotations

import json
import random
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import typer

if TYPE_CHECKING:
    import torch


class BenchmarkError(RuntimeError):
    """The operands or runtime do not satisfy this benchmark's contract."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@dataclass(frozen=True, slots=True)
class Case:
    shape: tuple[int, int, int, int, int]
    weight: tuple[int, int, int, int, int]
    gated: bool = False


CASES: Final = {
    "up_high42": Case((1, 512, 42, 48, 84), (512, 512, 3, 3, 3)),
    "up_high20": Case((1, 512, 20, 48, 84), (512, 512, 3, 3, 3)),
    "up_low42": Case((1, 512, 42, 24, 42), (512, 512, 3, 3, 3)),
    "up_low20": Case((1, 512, 20, 24, 42), (512, 512, 3, 3, 3)),
    "adapter_expand": Case((1, 752, 17, 24, 42), (3008, 752, 1, 1, 1), True),
    "adapter_project": Case((1, 1504, 17, 24, 42), (752, 1504, 1, 1, 1)),
}
SEED: Final = 121
WARMUP: Final = 5


@dataclass(frozen=True, slots=True)
class Operands:
    inputs: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor


@dataclass(frozen=True, slots=True)
class Timing:
    median_ms: float
    min_ms: float
    iqr_ms: float


@dataclass(frozen=True, slots=True)
class Measurement:
    case: str
    shape: tuple[int, ...]
    weight_shape: tuple[int, ...]
    consumer: str
    bitwise: bool
    baseline: Timing
    bias_fused: Timing
    speedup: float


def load_operands(name: str, directory: Path | None) -> Operands:
    """Read saved BF16 operands or generate reproducible channels-last tensors."""
    import torch

    spec = CASES[name]
    torch.manual_seed(SEED + tuple(CASES).index(name))
    if directory is None:
        inputs = torch.empty(
            spec.shape,
            device="cuda",
            dtype=torch.bfloat16,
            memory_format=torch.channels_last_3d,
        ).normal_()
        weight = torch.empty(
            spec.weight,
            device="cuda",
            dtype=torch.bfloat16,
            memory_format=torch.channels_last_3d,
        ).normal_()
        bias = torch.randn(spec.weight[0], device="cuda", dtype=torch.bfloat16)
        return Operands(inputs, weight, bias)
    saved: dict[str, torch.Tensor] = torch.load(
        directory / f"{name}.pt", map_location="cpu", weights_only=True
    )
    tensors = tuple(saved[key] for key in ("input", "weight", "bias"))
    shapes = (spec.shape, spec.weight, (spec.weight[0],))
    for tensor, shape in zip(tensors, shapes):
        if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape:
            raise BenchmarkError(f"{name}: expected BF16 operand with shape {shape}")
    inputs, weight, bias = tensors
    return Operands(
        inputs.to(device="cuda", memory_format=torch.channels_last_3d),
        weight.to(device="cuda", memory_format=torch.channels_last_3d),
        bias.to(device="cuda").contiguous(),
    )


def require_bitwise(left: torch.Tensor, right: torch.Tensor, context: str) -> None:
    """Include signed zero in the BF16 equality check."""
    import torch

    if not torch.equal(left.view(torch.int16), right.view(torch.int16)):
        raise BenchmarkError(f"{context}: outputs are not bitwise equal")


def timed(call: Callable[[], torch.Tensor]) -> float:
    """Keep the result alive until its completion event has synchronized."""
    import torch

    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    result = call()
    end.record()
    end.synchronize()
    del result
    return start.elapsed_time(end)


def summarize(samples: list[float]) -> Timing:
    quartiles = (
        statistics.quantiles(samples, n=4, method="inclusive")
        if len(samples) > 1
        else samples * 3
    )
    return Timing(statistics.median(samples), min(samples), quartiles[2] - quartiles[0])


def benchmark(name: str, directory: Path | None, repeats: int) -> Measurement:
    """Time the same warmed operands in shuffled, paired, serial rounds."""
    import torch
    from torch.nn import functional as F

    from flashinfer.diffusion_ops import h3_bridge

    spec = CASES[name]
    operands = load_operands(name, directory)
    inputs, weight, bias = operands.inputs, operands.weight, operands.bias
    padding = tuple(k // 2 for k in spec.weight[2:])
    raw = F.conv3d(inputs, weight, None, padding=padding)
    biased = F.conv3d(inputs, weight, bias, padding=padding)
    require_bitwise(
        biased, raw + bias.view(1, -1, 1, 1, 1), f"{name} Conv3D bias rounding"
    )
    residual = torch.empty_like(raw, memory_format=torch.channels_last_3d).normal_()
    del raw, biased

    def baseline() -> torch.Tensor:
        value = F.conv3d(inputs, weight, bias, padding=padding)
        if spec.gated:
            value_half, gate = value.chunk(2, dim=1)
            return h3_bridge.silu_multiply(value_half, gate)
        return h3_bridge.add(value, residual)

    def bias_fused() -> torch.Tensor:
        value = F.conv3d(inputs, weight, None, padding=padding)
        if spec.gated:
            return h3_bridge.bias_silu_multiply(value, bias)
        return h3_bridge.add_bias_residual(value, bias, residual)

    require_bitwise(baseline(), bias_fused(), f"{name} full path")
    operations = (baseline, bias_fused)
    for _ in range(WARMUP):
        for operation in operations:
            operation()
    torch.cuda.synchronize()
    samples: tuple[list[float], list[float]] = ([], [])
    rng = random.Random(SEED)
    for _ in range(repeats):
        order = [0, 1]
        rng.shuffle(order)
        for index in order:
            samples[index].append(timed(operations[index]))
    baseline_time, fused_time = (summarize(values) for values in samples)
    return Measurement(
        name,
        spec.shape,
        spec.weight,
        "silu_multiply" if spec.gated else "add",
        True,
        baseline_time,
        fused_time,
        baseline_time.median_ms / fused_time.median_ms,
    )


def main(
    case: str | None = typer.Option(None, help="Case name; default: adapter_expand."),
    all_cases: bool = typer.Option(False, "--all", help="Run all six real H3 shapes."),
    repeats: int = typer.Option(
        20, min=1, help="Paired timed rounds after five warmups."
    ),
    operands_dir: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=False,
        readable=True,
        help="Read CASE.pt input/weight/bias dictionaries; default: random operands.",
    ),
) -> None:
    """Benchmark full Conv3D + consumer paths through the installed FlashInfer API."""
    if all_cases and case is not None:
        raise typer.BadParameter("Use --case or --all, not both.")
    selected = tuple(CASES) if all_cases else (case or "adapter_expand",)
    for name in selected:
        if name not in CASES:
            raise typer.BadParameter(
                f"Unknown case {name!r}; choose from {', '.join(CASES)}."
            )
    import torch

    if not torch.cuda.is_available():
        raise BenchmarkError("A CUDA device is required")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    with torch.inference_mode():
        records = [asdict(benchmark(name, operands_dir, repeats)) for name in selected]
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    report = {
        "hardware": {
            "gpu": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "sm_count": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
        },
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "scope": "full conv + bias + consumer; warm inputs, synthetic residual",
        "dtype": "bfloat16",
        "memory_format": "channels_last_3d",
        "operands": str(operands_dir) if operands_dir is not None else "random",
        "seed": SEED,
        "warmup": WARMUP,
        "repeats": repeats,
        "cudnn_benchmark": False,
        "records": records,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    typer.run(main)
