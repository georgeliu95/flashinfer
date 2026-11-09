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

"""Validated backend dispatch for BF16 H3 temporal depthwise convolution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import Literal

import torch
from typing_extensions import assert_never


@dataclass(frozen=True, slots=True)
class DepthwiseContractError(ValueError):
    """An operand or requested backend violates the inference contract."""

    detail: str

    def __str__(self) -> str:
        return self.detail


class _Backend(str, Enum):
    AUTO = "auto"
    TRITON = "triton"
    CUTE = "cute"


def _validate(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    bias_first: bool,
) -> None:
    """Validate metadata before importing or launching either GPU backend."""
    if inputs.ndim != 5 or any(size <= 0 for size in inputs.shape):
        raise DepthwiseContractError("inputs must have positive NCDHW dimensions")
    channels = inputs.shape[1]
    if (
        weight.ndim != 5
        or weight.shape[0] != channels
        or weight.shape[1] != 1
        or weight.shape[2] not in (3, 5)
        or weight.shape[3:] != (1, 1)
    ):
        raise DepthwiseContractError("weight must have shape [C,1,3|5,1,1]")
    operands = (inputs, weight) if bias is None else (inputs, weight, bias)
    if any(
        tensor.device.type != "cuda"
        or tensor.device != inputs.device
        or tensor.dtype != torch.bfloat16
        or tensor.requires_grad
        for tensor in operands
    ):
        raise DepthwiseContractError(
            "operands must be same-device CUDA BF16 inference tensors "
            "without requires_grad"
        )
    if not (
        inputs.is_contiguous()
        or inputs.is_contiguous(memory_format=torch.channels_last_3d)
    ):
        raise DepthwiseContractError(
            "inputs must be contiguous NCDHW or channels_last_3d"
        )
    if not weight.is_contiguous():
        raise DepthwiseContractError("weight must be contiguous")
    if bias is not None and (bias.shape != (channels,) or not bias.is_contiguous()):
        raise DepthwiseContractError("bias must be a contiguous [C] vector")
    if type(bias_first) is not bool:
        raise DepthwiseContractError("bias_first must be a bool")


def _cute_supported(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    bias_first: bool,
) -> bool:
    """Limit CuTe to the two shapes with measured benefit on GB10 SM121."""
    return (
        tuple(inputs.shape) in ((1, 512, 20, 48, 84), (1, 512, 42, 48, 84))
        and weight.shape[2] == 5
        and bias is not None
        and bias_first
        and inputs.is_contiguous(memory_format=torch.channels_last_3d)
        and torch.cuda.get_device_capability(inputs.device) == (12, 1)
    )


def _cute_available() -> bool:
    """Probe optional imports only; compilation failures are never caught here."""
    try:
        cute = import_module("cutlass.cute")
        import_module("cuda.bindings.driver")
        import_module("tvm_ffi")
    except ImportError:
        return False
    return hasattr(cute.math, "fma")


def depthwise_temporal(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    bias_first: bool = True,
    backend: Literal["auto", "triton", "cute"] = "auto",
) -> torch.Tensor:
    """Apply same-padded temporal depthwise Conv3D with BF16 channels-last output.

    Stride and dilation are one, groups equal C, and taps are 3 or 5. FP32
    accumulation visits taps in ascending order; ``bias_first=False`` adds
    bias after the reduction. This inference-only API rejects gradients.

    ``auto`` uses CuTe only for measured SM121 high-resolution K5 shapes
    with bias-first and channels-last input; otherwise it selects Triton.
    Explicit ``cute`` rejects unsupported contracts or unavailable optional
    dependencies. Warm each specialization before CUDA graph capture.
    """
    _validate(inputs, weight, bias, bias_first)
    try:
        selected = _Backend(backend)
    except ValueError as error:
        raise DepthwiseContractError(
            f"unknown depthwise backend: {backend!r}"
        ) from error
    use_cute = False
    match selected:
        case _Backend.TRITON:
            pass
        case _Backend.AUTO:
            use_cute = (
                _cute_supported(inputs, weight, bias, bias_first) and _cute_available()
            )
        case _Backend.CUTE:
            if not _cute_supported(inputs, weight, bias, bias_first):
                raise DepthwiseContractError(
                    "CuTe requires SM121, [1,512,20|42,48,84] channels-last input, "
                    "K5, bias and bias_first=True"
                )
            if not _cute_available():
                raise ImportError(
                    "CuTe depthwise requires CUTLASS DSL with cute.math.fma, "
                    "CUDA Python and TVM FFI"
                )
            use_cute = True
        case unreachable:
            assert_never(unreachable)
    with torch.cuda.device(inputs.device):
        if use_cute:
            from ._depthwise_cute import run

            assert bias is not None
            return run(inputs, weight, bias)
        from ._depthwise_triton import run

        return run(inputs, weight, bias, bias_first=bias_first)
