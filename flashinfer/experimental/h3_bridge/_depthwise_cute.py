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

"""Generic channels-last BF16 temporal depthwise with FP32 FMA windows."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from math import prod

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel


class _CaptureError(RuntimeError):
    """A specialization was not warmed before graph capture."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class _TacticError(ValueError):
    """Explicit launch parameters cannot safely cover this tensor."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


@cute.kernel
def temporal_kernel(
    x: cute.Tensor,
    w: cute.Tensor,
    b: cute.Tensor,
    y: cute.Tensor,
    batches: cutlass.Constexpr,
    channels: cutlass.Constexpr,
    frames: cutlass.Constexpr,
    pixels: cutlass.Constexpr,
    taps: cutlass.Constexpr,
    has_bias: cutlass.Constexpr,
    bias_first: cutlass.Constexpr,
    time_tile: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    vec: cutlass.Constexpr,
) -> None:
    """Reuse temporal windows without crossing batch, channel or padding edges."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    spatial_blocks = (channels * pixels + threads * vec - 1) // (threads * vec)
    time_blocks = (frames + time_tile - 1) // time_tile
    spatial_block = bid % spatial_blocks
    time_block = bid // spatial_blocks % time_blocks
    batch = bid // (spatial_blocks * time_blocks)
    if cutlass.const_expr(batches * channels * frames * pixels > 2147483647):
        index = (cutlass.Int64(spatial_block) * threads + tid) * vec
        first = cutlass.Int64(time_block) * time_tile
        batch_base = cutlass.Int64(batch) * frames * pixels * channels
    else:
        index = (spatial_block * threads + tid) * vec
        first = time_block * time_tile
        batch_base = batch * frames * pixels * channels
    channel = index % channels
    if index < channels * pixels:
        weights = cute.make_rmem_tensor((vec, taps), cutlass.Float32)
        values = cute.make_rmem_tensor((vec, time_tile + taps - 1), cutlass.Float32)
        loaded = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
        zero = cute.make_rmem_tensor((vec,), cutlass.Float32)
        zero.fill(0)
        for tap in cutlass.range_constexpr(taps):
            for v in cutlass.range_constexpr(vec):
                weights[v, tap] = w[(channel + v) * taps + tap].to(cutlass.Float32)
        loaded.fill(0)
        if cutlass.const_expr(has_bias):
            gb = cute.make_tensor(b.iterator + channel, cute.make_layout((vec,)))
            cute.autovec_copy(gb, loaded)
        bias = loaded.load().to(cutlass.Float32)
        for offset in cutlass.range_constexpr(time_tile + taps - 1):
            t = first + offset - taps // 2
            loaded.fill(0)
            if t >= 0 and t < frames:
                gx = cute.make_tensor(
                    x.iterator + batch_base + t * pixels * channels + index,
                    cute.make_layout((vec,)),
                )
                cute.autovec_copy(gx, loaded)
            values[None, offset].store(loaded.load().to(cutlass.Float32))
        for out in cutlass.range_constexpr(time_tile):
            if cutlass.const_expr(bias_first):
                acc = bias
            else:
                acc = zero.load()
            for tap in cutlass.range_constexpr(taps):
                acc = cute.math.fma(
                    values[None, out + tap].load(), weights[None, tap].load(), acc
                )
            if cutlass.const_expr(not bias_first):
                acc = acc + bias
            if first + out < frames:
                loaded.store(acc.to(cutlass.BFloat16))
                gy = cute.make_tensor(
                    y.iterator + batch_base + (first + out) * pixels * channels + index,
                    cute.make_layout((vec,)),
                )
                cute.autovec_copy(loaded, gy)


@cute.jit
def launch(
    x: cute.Tensor,
    w: cute.Tensor,
    b: cute.Tensor,
    y: cute.Tensor,
    batches: cutlass.Constexpr,
    channels: cutlass.Constexpr,
    frames: cutlass.Constexpr,
    pixels: cutlass.Constexpr,
    taps: cutlass.Constexpr,
    has_bias: cutlass.Constexpr,
    bias_first: cutlass.Constexpr,
    time_tile: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    """Flatten batch/time/spatial blocks to avoid the grid-z batch-size limit."""
    spatial_blocks = (channels * pixels + threads * vec - 1) // (threads * vec)
    time_blocks = (frames + time_tile - 1) // time_tile
    temporal_kernel(
        x,
        w,
        b,
        y,
        batches,
        channels,
        frames,
        pixels,
        taps,
        has_bias,
        bias_first,
        time_tile,
        threads,
        vec,
    ).launch(
        grid=(batches * time_blocks * spatial_blocks, 1, 1),
        block=(threads, 1, 1),
        stream=stream,
    )


@cache
def _compiled(
    device_index: int,
    shape: tuple[int, ...],
    taps: int,
    has_bias: bool,
    bias_first: bool,
    time_tile: int,
    threads: int,
    vec: int,
    alignment: int,
) -> Callable[..., None]:
    """Cache complete metadata/tactics; retain no tensors or runtime streams."""
    if torch.cuda.is_current_stream_capturing():
        raise _CaptureError("warm CuTe depthwise before CUDA graph capture")
    batches, channels, frames, height, width = shape
    elements = prod(shape)
    bias_elements = channels if has_bias else channels * taps
    operands = tuple(
        make_fake_tensor(cutlass.BFloat16, (size,), (1,), assumed_align=alignment)
        for size in (elements, channels * taps, bias_elements, elements)
    )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    key = (
        f"d{device_index}_n{batches}_c{channels}_t{frames}_h{height}_w{width}"
        f"_k{taps}_bias{int(has_bias)}_bf{int(bias_first)}"
        f"_tt{time_tile}_th{threads}_v{vec}_a{alignment}"
    )
    return build_and_load_cute_dsl_kernel(
        "h3_bridge_depthwise",
        key,
        lambda: cute.compile(
            launch,
            *operands,
            batches,
            channels,
            frames,
            height * width,
            taps,
            has_bias,
            bias_first,
            time_tile,
            threads,
            vec,
            stream,
            options="--enable-tvm-ffi",
        ),
        extra_key_files=(__file__,),
    )


def run(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    bias_first: bool = True,
    time_tile: int = 20,
    threads: int = 128,
    vec: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch a validated CL BF16 contract; optional ``out`` is a matching CL tensor.

    ``vec=None`` chooses two channels when C is even, otherwise one. Explicit
    vector widths must divide C so a thread never crosses channel or row ends.
    """
    channels = inputs.shape[1]
    vector = (2 if channels % 2 == 0 else 1) if vec is None else vec
    if vector not in (1, 2, 4, 8) or channels % vector:
        raise _TacticError("vec must be 1, 2, 4 or 8 and divide the channel count")
    if time_tile <= 0 or threads <= 0 or threads > 1024 or threads % 32:
        raise _TacticError(
            "time_tile must be positive; threads must be a multiple of 32 up to 1024"
        )
    with torch.cuda.device(inputs.device):
        output = (
            torch.empty_like(inputs, memory_format=torch.channels_last_3d)
            if out is None
            else out
        )
        tensors = (
            inputs.permute(0, 2, 3, 4, 1).view(-1),
            weight.view(-1),
            weight.view(-1) if bias is None else bias,
            output.permute(0, 2, 3, 4, 1).view(-1),
        )
        # BF16 storage-offset views may have only two-byte alignment.
        alignment = min(
            16, *(tensor.data_ptr() & -tensor.data_ptr() for tensor in tensors)
        )
        _compiled(
            inputs.get_device(),
            tuple(inputs.shape),
            weight.shape[2],
            bias is not None,
            bias_first,
            time_tile,
            threads,
            vector,
            alignment,
        )(*tensors)
    return output
