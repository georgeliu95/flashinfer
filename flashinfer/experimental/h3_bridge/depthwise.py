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

from typing import Literal

import torch


class DepthwiseContractError(ValueError):
    """An inference contract error whose traceback remains context-manager safe."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


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


def depthwise_temporal(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    bias_first: bool = True,
    backend: Literal["auto", "triton", "cute"] = "auto",
) -> torch.Tensor:
    """Apply temporal DW Conv3D using FlashInfer's per-configuration autotuner.

    Inputs are BF16 contiguous NCDHW or channels-last; K is 3 or 5, with
    stride/dilation one and same padding. Output is channels-last. In an
    ``autotune()`` context, auto compares legal Triton and CuTe tactics using
    the actual shape and strides. Outside that context it reuses the winner
    or falls back to Triton. Explicit backends tune only their own tactics.
    CuTe accepts arbitrary positive NCTHW dimensions in channels-last on
    SM121, with optional bias and either FP32 accumulation order.
    """
    _validate(inputs, weight, bias, bias_first)
    if backend not in ("auto", "triton", "cute"):
        raise DepthwiseContractError(f"unknown depthwise backend: {backend!r}")
    from ._tuning import run

    with torch.cuda.device(inputs.device):
        return run(inputs, weight, bias, bias_first=bias_first, backend=backend)
