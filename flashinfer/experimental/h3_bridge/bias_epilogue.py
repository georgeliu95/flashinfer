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

"""Rounded BF16 convolution-bias consumers with lazy optional dependencies.

Convolution outputs have already rounded to BF16. Bias addition and SiLU keep
separate BF16 rounding boundaries before residual addition or multiplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import prod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from triton.runtime.jit import JITFunction


class EpilogueError(ValueError):
    """A tensor-contract error safe to propagate through context managers."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@dataclass(frozen=True, slots=True)
class EpiloguePlan:
    """Validated positive NCDHW geometry with compact CL output."""

    output_shape: tuple[int, ...]
    channels: int
    elements: int

    @classmethod
    def from_shape(cls, shape: tuple[int, ...], *, gated: bool) -> EpiloguePlan:
        if len(shape) != 5 or any(size <= 0 for size in shape):
            raise EpilogueError("epilogue requires positive NCDHW dimensions")
        if gated and shape[1] % 2:
            raise EpilogueError("gate epilogue requires an even input channel count")
        channels = shape[1] // 2 if gated else shape[1]
        output_shape = (shape[0], channels, *shape[2:])
        return cls(output_shape, channels, prod(output_shape))


@lru_cache(maxsize=1)
def _kernel() -> JITFunction:
    import triton
    import triton.language as tl

    @triton.jit
    def rounded_bias(
        X,
        Bias,
        Residual,
        Y,
        N: tl.constexpr,
        C: tl.constexpr,
        GATED: tl.constexpr,
        B: tl.constexpr,
    ):
        block = tl.program_id(0)
        if N * (2 if GATED else 1) > 2147483647:
            block = block.to(tl.int64)
        i = block * B + tl.arange(0, B)
        valid = i < N
        channel = i % C
        if GATED:
            source = (i // C) * (2 * C) + channel
            value = tl.load(X + source, valid, 0).to(tl.float32)
            gate = tl.load(X + source + C, valid, 0).to(tl.float32)
            value_bias = tl.load(Bias + channel, valid, 0).to(tl.float32)
            gate_bias = tl.load(Bias + channel + C, valid, 0).to(tl.float32)
            value = (value + value_bias).to(tl.bfloat16).to(tl.float32)
            gate = (gate + gate_bias).to(tl.bfloat16).to(tl.float32)
            activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
            result = value * activated
        else:
            value = tl.load(X + i, valid, 0).to(tl.float32)
            bias = tl.load(Bias + channel, valid, 0).to(tl.float32)
            value = (value + bias).to(tl.bfloat16).to(tl.float32)
            residual = tl.load(Residual + i, valid, 0).to(tl.float32)
            result = value + residual
        tl.store(Y + i, result.to(tl.bfloat16), valid)

    return rounded_bias


def _run(operands: tuple[Tensor, Tensor, Tensor], *, gated: bool) -> Tensor:
    import torch

    conv, bias, residual = operands
    plan = EpiloguePlan.from_shape(tuple(conv.shape), gated=gated)
    if any(
        x.device.type != "cuda"
        or x.device != conv.device
        or x.dtype != torch.bfloat16
        or x.requires_grad
        for x in operands
    ):
        raise EpilogueError("epilogue requires same-device inference CUDA BF16 tensors")
    if not conv.is_contiguous(memory_format=torch.channels_last_3d):
        raise EpilogueError("convolution output must be channels_last_3d")
    if bias.ndim != 1 or bias.shape[0] != conv.shape[1] or not bias.is_contiguous():
        raise EpilogueError("bias must be a contiguous vector matching input channels")
    if not gated and (
        residual.shape != conv.shape
        or not residual.is_contiguous(memory_format=torch.channels_last_3d)
    ):
        raise EpilogueError(
            "residual must have matching shape and channels_last_3d layout"
        )
    import triton

    with torch.cuda.device(conv.device):
        output = torch.empty(
            plan.output_shape,
            device=conv.device,
            dtype=torch.bfloat16,
            memory_format=torch.channels_last_3d,
        )
        _kernel()[(triton.cdiv(plan.elements, 1024),)](
            conv,
            bias,
            residual,
            output,
            plan.elements,
            plan.channels,
            gated,
            1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output


def add_bias_residual(conv_no_bias: Tensor, bias: Tensor, residual: Tensor) -> Tensor:
    """Return bf16(bf16(conv_no_bias+bias)+residual) in channels_last_3d."""
    return _run((conv_no_bias, bias, residual), gated=False)


def bias_silu_multiply(conv_no_bias: Tensor, bias: Tensor) -> Tensor:
    """Bias and split CL channels, then apply the maintained rounded gate formula."""
    return _run((conv_no_bias, bias, conv_no_bias), gated=True)
