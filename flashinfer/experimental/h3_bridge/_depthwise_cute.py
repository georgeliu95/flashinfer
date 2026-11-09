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

"""Vectorized FP32-FMA temporal window with BF16 input/output."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel


@cute.kernel
def temporal_kernel(
    x: cute.Tensor,
    w: cute.Tensor,
    b: cute.Tensor,
    y: cute.Tensor,
    channels: cutlass.Constexpr,
    frames: cutlass.Constexpr,
    pixels: cutlass.Constexpr,
    taps: cutlass.Constexpr,
    time_tile: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    vec: cutlass.Constexpr,
) -> None:
    """Load contiguous BF16 channel vectors and reuse a local temporal window."""
    tid, _, _ = cute.arch.thread_idx()
    bid, bt, _ = cute.arch.block_idx()
    index = (bid * threads + tid) * vec
    channel = index % channels
    first = bt * time_tile
    if index < channels * pixels:
        weights = cute.make_rmem_tensor((vec, taps), cutlass.Float32)
        values = cute.make_rmem_tensor((vec, time_tile + taps - 1), cutlass.Float32)
        loaded = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
        for tap in cutlass.range_constexpr(taps):
            for v in cutlass.range_constexpr(vec):
                weights[v, tap] = w[(channel + v) * taps + tap].to(cutlass.Float32)
        gb = cute.make_tensor(b.iterator + channel, cute.make_layout((vec,)))
        cute.autovec_copy(gb, loaded)
        bias = loaded.load().to(cutlass.Float32)
        for offset in cutlass.range_constexpr(time_tile + taps - 1):
            t = first + offset - taps // 2
            loaded.fill(0)
            if t >= 0 and t < frames:
                gx = cute.make_tensor(
                    x.iterator + t * pixels * channels + index, cute.make_layout((vec,))
                )
                cute.autovec_copy(gx, loaded)
            values[None, offset].store(loaded.load().to(cutlass.Float32))
        for out in cutlass.range_constexpr(time_tile):
            acc = bias
            for tap in cutlass.range_constexpr(taps):
                acc = cute.math.fma(
                    values[None, out + tap].load(), weights[None, tap].load(), acc
                )
            if first + out < frames:
                loaded.store(acc.to(cutlass.BFloat16))
                gy = cute.make_tensor(
                    y.iterator + (first + out) * pixels * channels + index,
                    cute.make_layout((vec,)),
                )
                cute.autovec_copy(loaded, gy)


@cute.jit
def launch(
    x: cute.Tensor,
    w: cute.Tensor,
    b: cute.Tensor,
    y: cute.Tensor,
    channels: cutlass.Constexpr,
    frames: cutlass.Constexpr,
    pixels: cutlass.Constexpr,
    taps: cutlass.Constexpr,
    time_tile: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    """Launch specialized window and vectorization sizes."""
    temporal_kernel(
        x, w, b, y, channels, frames, pixels, taps, time_tile, threads, vec
    ).launch(
        grid=(
            (channels * pixels + threads * vec - 1) // (threads * vec),
            (frames + time_tile - 1) // time_tile,
            1,
        ),
        block=(threads, 1, 1),
        stream=stream,
    )


@dataclass(frozen=True, slots=True)
class _CaptureError(RuntimeError):
    """A specialization was not warmed before graph capture."""

    detail: str

    def __str__(self) -> str:
        return self.detail


@cache
def _compiled(device_index: int, frames: int, alignment: int) -> Callable[..., None]:
    """Cache code by device, shape and proven alignment; retain no tensors/streams."""
    if torch.cuda.is_current_stream_capturing():
        raise _CaptureError("warm CuTe depthwise before CUDA graph capture")
    elements = 512 * frames * 48 * 84
    operands = tuple(
        make_fake_tensor(cutlass.BFloat16, (size,), (1,), assumed_align=alignment)
        for size in (elements, 512 * 5, 512, elements)
    )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return build_and_load_cute_dsl_kernel(
        "h3_bridge_depthwise",
        f"t{frames}_a{alignment}",
        lambda: cute.compile(
            launch,
            *operands,
            512,
            frames,
            48 * 84,
            5,
            20,
            128,
            2,
            stream,
            options="--enable-tvm-ffi",
        ),
        extra_key_files=(__file__,),
    )


def run(inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Apply the measured temporal window, using fresh pointers and current stream."""
    output = torch.empty_like(inputs, memory_format=torch.channels_last_3d)
    tensors = (
        inputs.permute(0, 2, 3, 4, 1).view(-1),
        weight.view(-1),
        bias,
        output.permute(0, 2, 3, 4, 1).view(-1),
    )
    # Offset BF16 views can have only two-byte alignment; never overstate it.
    alignment = min(16, *(tensor.data_ptr() & -tensor.data_ptr() for tensor in tensors))
    _compiled(inputs.get_device(), inputs.shape[2], alignment)(*tensors)
    return output
