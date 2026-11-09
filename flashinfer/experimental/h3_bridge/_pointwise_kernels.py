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

"""Rounded BF16 pointwise kernels; Torch and Triton are imported on demand."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from triton.runtime.jit import JITFunction


@lru_cache(maxsize=1)
def kernels() -> JITFunction:
    import triton
    import triton.language as tl

    @triton.jit
    def pointwise(
        X,
        A,
        Z,
        Y,
        N: tl.constexpr,
        C: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        XS: tl.constexpr,
        AS: tl.constexpr,
        MODE: tl.constexpr,
        B: tl.constexpr,
    ):
        i = tl.program_id(0) * B + tl.arange(0, B)
        valid = i < N
        channel = i % C
        pixel = i // C % S
        batch = i // (C * S)
        time = pixel // (H * W)
        height = pixel // W % H
        width = pixel % W
        xo = (
            batch * XS[0]
            + channel * XS[1]
            + time * XS[2]
            + height * XS[3]
            + width * XS[4]
        )
        value = tl.load(X + xo, valid, 0).to(tl.float32)
        if MODE == 0:
            scale = tl.load(A + channel).to(tl.float32)
            shift = tl.load(Z + channel).to(tl.float32)
            factor = (1.0 + scale).to(tl.bfloat16).to(tl.float32)
            value = (value * factor).to(tl.bfloat16).to(tl.float32)
            value = (value + shift).to(tl.bfloat16).to(tl.float32)
            result = value / (1.0 + tl.exp(-value))
        elif MODE == 3:
            result = value / (1.0 + tl.exp(-value))
        else:
            ao = (
                batch * AS[0]
                + channel * AS[1]
                + time * AS[2]
                + height * AS[3]
                + width * AS[4]
            )
            other = tl.load(A + ao, valid, 0).to(tl.float32)
            if MODE == 1:
                activated = (
                    (other / (1.0 + tl.exp(-other))).to(tl.bfloat16).to(tl.float32)
                )
                result = value * activated
            else:
                result = value + other
        tl.store(Y + i, result.to(tl.bfloat16), valid)

    return pointwise
