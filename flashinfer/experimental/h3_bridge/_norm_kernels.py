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

"""Stable full-group reduction and fused BF16 apply kernels; imported lazily."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from triton.runtime.jit import JITFunction


@lru_cache(maxsize=1)
def kernels() -> tuple[JITFunction, JITFunction, JITFunction]:
    import triton
    import triton.language as tl

    @triton.jit
    def moments(
        X,
        Partial,
        C: tl.constexpr,
        S: tl.constexpr,
        CG: tl.constexpr,
        G: tl.constexpr,
        E: tl.constexpr,
        P: tl.constexpr,
        CL: tl.constexpr,
        B: tl.constexpr,
    ):
        group = tl.program_id(0)
        part = tl.program_id(1)
        j = part * B + tl.arange(0, B)
        valid = j < E
        channel = group % G * CG + j % CG
        spatial = j // CG
        if CL:
            offsets = group // G * C * S + spatial * C + channel
        else:
            offsets = group // G * C * S + channel * S + spatial
        values = tl.load(X + offsets, valid, 0).to(tl.float32)
        count = tl.minimum(B, E - part * B)
        mean = tl.sum(values, 0) / count
        delta = tl.where(valid, values - mean, 0.0)
        m2 = tl.sum(delta * delta, 0)
        tl.store(Partial + (group * P + part) * 2, mean)
        tl.store(Partial + (group * P + part) * 2 + 1, m2)

    @triton.jit
    def statistics(
        Partial,
        Stats,
        E: tl.constexpr,
        P: tl.constexpr,
        EPS: tl.constexpr,
        PB: tl.constexpr,
        B: tl.constexpr,
    ):
        group = tl.program_id(0)
        part = tl.arange(0, B)
        valid = part < P
        count = tl.where(valid, tl.minimum(PB, E - part * PB), 0)
        means = tl.load(Partial + (group * P + part) * 2, valid, 0)
        m2 = tl.load(Partial + (group * P + part) * 2 + 1, valid, 0)
        mean = tl.sum(means * count, 0) / E
        variance = tl.sum(m2 + count * (means - mean) * (means - mean), 0) / E
        tl.store(Stats + group * 2, mean)
        tl.store(Stats + group * 2 + 1, tl.rsqrt(tl.maximum(variance, 0.0) + EPS))

    @triton.jit
    def norm_apply(
        X,
        W,
        Bias,
        Stats,
        Scale,
        Shift,
        Y,
        N: tl.constexpr,
        C: tl.constexpr,
        S: tl.constexpr,
        CG: tl.constexpr,
        G: tl.constexpr,
        CL: tl.constexpr,
        MOD: tl.constexpr,
        B: tl.constexpr,
    ):
        i = tl.program_id(0) * B + tl.arange(0, B)
        valid = i < N
        channel = i % C
        spatial = i // C % S
        batch = i // (C * S)
        offset = i if CL else batch * C * S + channel * S + spatial
        value = tl.load(X + offset, valid, 0).to(tl.float32)
        group = batch * G + channel // CG
        mean = tl.load(Stats + group * 2, valid, 0)
        rstd = tl.load(Stats + group * 2 + 1, valid, 0)
        weight = tl.load(W + channel).to(tl.float32)
        bias = tl.load(Bias + channel).to(tl.float32)
        value = (((value - mean) * rstd) * weight + bias).to(tl.bfloat16).to(tl.float32)
        if MOD:
            scale = tl.load(Scale + channel).to(tl.float32)
            shift = tl.load(Shift + channel).to(tl.float32)
            factor = (1.0 + scale).to(tl.bfloat16).to(tl.float32)
            value = (value * factor).to(tl.bfloat16).to(tl.float32)
            value = (value + shift).to(tl.bfloat16).to(tl.float32)
        result = value / (1.0 + tl.exp(-value))
        tl.store(Y + i, result.to(tl.bfloat16), valid)

    return moments, statistics, norm_apply
