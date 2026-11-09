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

"""Opt-in BF16 bridge fusion with explicit eager-operation rounding boundaries.

GroupNorm uses stable partial mean/M2 reductions over the complete group,
including T/H/W. Its reduction tree differs from native PyTorch; qualify it
separately before enabling. Pointwise fusions preserve each BF16 intermediate.
Importing this module does not import Torch, Triton, or initialize CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, prod
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from torch import Tensor

PARTIAL_BLOCK: Final = 4096


@dataclass(frozen=True, slots=True)
class BridgeFusionError(RuntimeError):
    """An unsupported BF16 inference contract detected before kernel launch."""

    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class GroupNormPlan:
    """Full channel-group and temporal/spatial reduction geometry."""

    batch: int
    channels: int
    groups: int
    spatial: int
    channels_per_group: int
    group_elements: int
    parts: int
    eps: float

    @classmethod
    def from_shape(
        cls, shape: tuple[int, ...], *, groups: int, eps: float
    ) -> GroupNormPlan:
        if (
            len(shape) != 5
            or any(v <= 0 for v in shape)
            or groups <= 0
            or shape[1] % groups
            or not isfinite(eps)
            or eps <= 0
        ):
            raise BridgeFusionError(
                "GroupNorm requires positive NCDHW dimensions, divisible groups and finite eps"
            )
        spatial = prod(shape[2:])
        elements = shape[1] // groups * spatial
        return cls(
            shape[0],
            shape[1],
            groups,
            spatial,
            shape[1] // groups,
            elements,
            (elements + PARTIAL_BLOCK - 1) // PARTIAL_BLOCK,
            eps,
        )


def _check_tensors(*values: Tensor) -> None:
    import torch

    first = values[0]
    if any(
        x.device.type != "cuda"
        or x.device != first.device
        or x.dtype != torch.bfloat16
        or x.requires_grad
        for x in values
    ):
        raise BridgeFusionError(
            "fusion requires same-device inference-only CUDA BF16 tensors"
        )


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
    """Fuse full-domain GroupNorm, optional rounded modulation and SiLU."""
    import torch

    plan = GroupNormPlan.from_shape(tuple(inputs.shape), groups=groups, eps=eps)
    _check_tensors(inputs, weight, bias)
    channels_last = inputs.is_contiguous(memory_format=torch.channels_last_3d)
    if not (channels_last or inputs.is_contiguous()):
        raise BridgeFusionError(
            "GroupNorm input must be contiguous or channels_last_3d"
        )
    if any(x.numel() != plan.channels or not x.is_contiguous() for x in (weight, bias)):
        raise BridgeFusionError(
            "GroupNorm affine parameters must be contiguous channel vectors"
        )
    modulation = scale is not None
    if modulation != (shift is not None):
        raise BridgeFusionError("modulation requires both scale and shift")
    if scale is not None and shift is not None:
        _check_tensors(inputs, scale, shift)
        if any(
            x.numel() != plan.channels or not x.is_contiguous() for x in (scale, shift)
        ):
            raise BridgeFusionError(
                "modulation supports one contiguous channel vector shared by the batch"
            )
    import triton

    with torch.cuda.device(inputs.device):
        partial = torch.empty(
            (plan.batch * groups, plan.parts, 2),
            device=inputs.device,
            dtype=torch.float32,
        )
        stats = torch.empty(
            (plan.batch * groups, 2), device=inputs.device, dtype=torch.float32
        )
        output = torch.empty_like(inputs, memory_format=torch.channels_last_3d)
        from ._norm_kernels import kernels

        moments, statistics, apply = kernels()
        moments[(plan.batch * groups, plan.parts)](
            inputs,
            partial,
            plan.channels,
            plan.spatial,
            plan.channels_per_group,
            groups,
            plan.group_elements,
            plan.parts,
            channels_last,
            PARTIAL_BLOCK,
            num_warps=4,
        )
        statistics[(plan.batch * groups,)](
            partial,
            stats,
            plan.group_elements,
            plan.parts,
            eps,
            PARTIAL_BLOCK,
            triton.next_power_of_2(plan.parts),
            num_warps=4,
        )
        apply[(triton.cdiv(inputs.numel(), 1024),)](
            inputs,
            weight,
            bias,
            stats,
            weight if scale is None else scale,
            bias if shift is None else shift,
            output,
            inputs.numel(),
            plan.channels,
            plan.spatial,
            plan.channels_per_group,
            groups,
            channels_last,
            modulation,
            1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output


def _pointwise(first: Tensor, second: Tensor, third: Tensor, *, mode: int) -> Tensor:
    import torch

    _check_tensors(first, second, third)
    if first.ndim != 5 or any(size <= 0 for size in first.shape):
        raise BridgeFusionError("bridge pointwise inputs must be positive NCDHW")
    if mode == 0:
        if any(
            x.numel() != first.shape[1] or not x.is_contiguous()
            for x in (second, third)
        ):
            raise BridgeFusionError("modulation requires contiguous channel vectors")
    elif first.shape != second.shape:
        raise BridgeFusionError("pointwise operands must have identical shapes")
    import triton

    with torch.cuda.device(first.device):
        output = torch.empty_like(first, memory_format=torch.channels_last_3d)
        from ._pointwise_kernels import kernels

        kernel = kernels()
        kernel[(triton.cdiv(first.numel(), 1024),)](
            first,
            second,
            third,
            output,
            first.numel(),
            first.shape[1],
            prod(first.shape[2:]),
            first.shape[3],
            first.shape[4],
            first.stride(),
            second.stride(),
            mode,
            1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output


def modulate_silu(normalized: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """Keep native GroupNorm but fuse its following rounded modulation and SiLU."""
    return _pointwise(normalized, scale, shift, mode=0)


def silu_multiply(value: Tensor, gate: Tensor) -> Tensor:
    """Fuse SiLU(gate) then value multiplication, including BF16 SiLU rounding."""
    return _pointwise(value, gate, gate, mode=1)


def add(left: Tensor, right: Tensor) -> Tensor:
    """Add BF16 tensors directly into channels_last_3d output."""
    return _pointwise(left, right, right, mode=2)


def silu(inputs: Tensor) -> Tensor:
    """Activate native NCDHW normalization output directly into channels-last."""
    return _pointwise(inputs, inputs, inputs, mode=3)
