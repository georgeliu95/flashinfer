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

"""FP32-FMA temporal depthwise convolution with channels-last BF16 output."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _temporal(
    X,
    Weight,
    Bias,
    Y,
    N: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    CL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BIAS_FIRST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    channel = index % C
    spatial = index // C % (T * H * W)
    time = spatial // (H * W)
    pixel = spatial % (H * W)
    batch = index // (C * T * H * W)
    if HAS_BIAS:
        bias = tl.load(Bias + channel, valid, 0).to(tl.float32)
    else:
        bias = tl.full((BLOCK,), 0, tl.float32)
    if BIAS_FIRST:
        accumulated = bias
    else:
        accumulated = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(K):
        source_time = time + tap - K // 2
        present = valid & (source_time >= 0) & (source_time < T)
        if CL:
            offset = (batch * T * H * W + source_time * H * W + pixel) * C + channel
        else:
            offset = (batch * C + channel) * T * H * W + source_time * H * W + pixel
        value = tl.load(X + offset, present, 0).to(tl.float32)
        weight = tl.load(Weight + channel * K + tap, valid, 0).to(tl.float32)
        accumulated = tl.fma(value, weight, accumulated)
    if not BIAS_FIRST:
        accumulated = accumulated + bias
    tl.store(Y + index, accumulated.to(tl.bfloat16), valid)


def run(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    bias_first: bool,
) -> torch.Tensor:
    """Launch the validated contiguous or channels-last inference contract."""
    _, channels, frames, height, width = inputs.shape
    output = torch.empty_like(inputs, memory_format=torch.channels_last_3d)
    _temporal[(triton.cdiv(inputs.numel(), 1024),)](
        inputs,
        weight,
        weight if bias is None else bias,
        output,
        inputs.numel(),
        channels,
        frames,
        height,
        width,
        weight.shape[2],
        inputs.is_contiguous(memory_format=torch.channels_last_3d),
        bias is not None,
        bias_first,
        1024,
        num_warps=4,
    )
    return output
