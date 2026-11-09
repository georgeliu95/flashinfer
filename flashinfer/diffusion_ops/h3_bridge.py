"""
Copyright (c) 2026 by FlashInfer team.

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

from typing import Literal

from torch import Tensor

from ..api_logging import flashinfer_experimental_api

__all__ = [
    "group_norm_silu",
    "modulate_silu",
    "silu_multiply",
    "add",
    "silu",
    "depthwise_temporal",
]


@flashinfer_experimental_api
def group_norm_silu(
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    groups: int,
    eps: float,
    scale: Tensor | None = None,
    shift: Tensor | None = None,
) -> Tensor:
    """Normalize full NCTHW groups, optionally modulate, then apply SiLU.

    All operands are same-device CUDA BF16 tensors with requires_grad=False.
    Input must be contiguous or channels_last_3d; output is channels_last_3d.
    Affine parameters, scale and shift are contiguous channel vectors. Scale
    and shift must be supplied together and are shared across batch elements.
    The FP32 reduction tree can differ from native GroupNorm; BF16 rounding
    boundaries between normalization, modulation and activation are retained.
    """
    from ..experimental.h3_bridge.fused import group_norm_silu as run

    return run(inputs, weight, bias, groups=groups, eps=eps, scale=scale, shift=shift)


@flashinfer_experimental_api
def modulate_silu(inputs: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """Apply BF16-rounded (1 + scale), multiply, shift, then SiLU to NCTHW.

    Scale and shift are contiguous channel vectors shared across the batch.
    All operands must be same-device inference CUDA BF16 tensors. Output uses
    channels_last_3d storage; strided NCTHW input is accepted.
    """
    from ..experimental.h3_bridge.fused import modulate_silu as run

    return run(inputs, scale, shift)


@flashinfer_experimental_api
def silu_multiply(value: Tensor, gate: Tensor) -> Tensor:
    """Multiply value by BF16-rounded SiLU(gate), returning channels-last.

    Operands must have identical NCTHW shapes and be same-device inference
    CUDA BF16 tensors. Strided channel chunks are accepted without copying.
    """
    from ..experimental.h3_bridge.fused import silu_multiply as run

    return run(value, gate)


@flashinfer_experimental_api
def add(left: Tensor, right: Tensor) -> Tensor:
    """Add equal-shaped inference CUDA BF16 NCTHW tensors into channels-last."""
    from ..experimental.h3_bridge.fused import add as run

    return run(left, right)


@flashinfer_experimental_api
def silu(inputs: Tensor) -> Tensor:
    """Apply SiLU to an inference CUDA BF16 NCTHW tensor into channels-last."""
    from ..experimental.h3_bridge.fused import silu as run

    return run(inputs)


@flashinfer_experimental_api
def depthwise_temporal(
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    *,
    bias_first: bool = True,
    backend: Literal["auto", "triton", "cute"] = "auto",
) -> Tensor:
    """Apply same-padded temporal DW Conv3D with BF16 I/O and FP32 FMA.

    Inputs are contiguous or channels-last NCTHW, weights are contiguous
    [C,1,3|5,1,1], and optional bias is a contiguous channel vector. All
    tensors are same-device CUDA BF16 with requires_grad=False. Stride and
    dilation are one; output uses channels_last_3d storage. Bias-first starts
    accumulation at the bias; false adds bias after ascending-tap reduction.

    Auto selects CuTe only for the qualified SM121 high-resolution shapes,
    otherwise Triton. Explicit cute rejects unsupported inputs. Warm up each
    specialization before CUDA graph capture. Calling this experimental API
    is the opt-in; no experimental-auto environment flag is required.
    """
    from ..experimental.h3_bridge.depthwise import depthwise_temporal as run

    return run(inputs, weight, bias, bias_first=bias_first, backend=backend)
