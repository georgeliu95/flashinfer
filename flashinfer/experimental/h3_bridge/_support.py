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

"""Depthwise candidate eligibility and stable native-autotuner identities."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from functools import cache
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final, Literal

import torch

Backend = Literal["auto", "triton", "cute"]
Tactic = Literal[-1] | tuple[int, ...]
TRITON_TACTICS: Final = ((256, 4), (512, 4), (1024, 4), (2048, 8))


class BackendUnavailableError(ValueError):
    """An explicitly requested backend cannot execute this contract."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class CuTeDependencyError(ImportError):
    """The caller explicitly requested CuTe without its optional dependencies."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class Policy:
    """Dispatch attributes that must remain distinct in the native cache."""

    backend: Backend
    bias_first: bool
    has_bias: bool
    cute_supported: bool
    cute_available: bool

    @property
    def cute_enabled(self) -> bool:
        return self.cute_supported and self.cute_available


@cache
def _cute_available() -> bool:
    """Probe optional compiler dependencies without loading our kernel module."""
    try:
        import cutlass.cute as cute
        import cuda.bindings.driver  # noqa: F401
        import tvm_ffi  # noqa: F401
    except ImportError:
        return False
    return hasattr(cute.math, "fma")


def _cute_supported(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    bias_first: bool,
) -> bool:
    """The validated contract needs CL storage and the verified SM121 ISA."""
    return inputs.is_contiguous(
        memory_format=torch.channels_last_3d
    ) and torch.cuda.get_device_capability(inputs.device) == (12, 1)


def cute_tactics(inputs: torch.Tensor) -> tuple[tuple[int, ...], ...]:
    """Bound compilation to eight useful temporal/vectorization configurations."""
    frames = inputs.shape[2]
    vec = 2 if inputs.shape[1] % 2 == 0 else 1
    configs = (
        (4, 128, vec),
        (8, 128, vec),
        (20, 128, vec),
        (32, 128, vec),
        (8, 256, vec),
        (20, 256, vec),
        (8, 128, 1),
        (20, 128, 1),
    )
    return tuple(dict.fromkeys((min(t, frames), n, v) for t, n, v in configs))


@cache
def _implementation_identity() -> tuple[str, ...]:
    """Invalidate old tactics when either compiler or implementation changes."""
    digest = sha256()
    for filename in (
        "_support.py",
        "_tuning.py",
        "_depthwise_triton.py",
        "_depthwise_cute.py",
    ):
        digest.update(Path(__file__).with_name(filename).read_bytes())
    versions = [str(torch.__version__)]
    for distribution in (
        "triton",
        "nvidia-cutlass-dsl",
        "cuda-python",
        "apache-tvm-ffi",
    ):
        try:
            versions.append(version(distribution))
        except PackageNotFoundError:
            versions.append("unavailable")
    return (digest.hexdigest(), *versions)


@cache
def _device_identity(index: int) -> tuple[Hashable, ...]:
    properties = torch.cuda.get_device_properties(index)
    return (
        index,
        properties.name,
        properties.major,
        properties.minor,
        properties.multi_processor_count,
    )


def cache_extras(inputs: list[torch.Tensor], policy: Policy) -> tuple[Hashable, ...]:
    """Static profiles preserve these strides/offset alignment classes exactly."""
    device_index = inputs[0].device.index
    assert device_index is not None
    tensor_metadata = tuple(
        (str(t.dtype), tuple(t.stride()), min(16, t.data_ptr() & -t.data_ptr()))
        for t in inputs
    )
    return (
        _implementation_identity(),
        _device_identity(device_index),
        tensor_metadata,
        policy.backend,
        policy.bias_first,
        policy.has_bias,
        policy.cute_supported,
        policy.cute_available,
    )
