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

"""CPU metadata contracts, independent of optional Torch/Triton packages."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable

import pytest
import torch

from flashinfer.diffusion_ops import h3_bridge as api
from flashinfer.experimental.h3_bridge import bias_epilogue as epi


def test_residual_plan_preserves_odd_channels_and_batch() -> None:
    # Given a positive odd-channel, multi-batch CL tensor shape.
    shape = (2, 7, 3, 5, 9)
    # When planning a rounded bias and residual epilogue.
    plan = epi.EpiloguePlan.from_shape(shape, gated=False)
    # Then the output preserves every input element.
    assert plan.output_shape == shape
    assert plan.channels == 7
    assert plan.elements == 1890


def test_gate_plan_splits_channels_without_splitting_spatial_or_batch() -> None:
    # Given contiguous CL value/gate channels and odd output channel count.
    shape = (3, 14, 2, 5, 7)
    # When planning the split-gate epilogue.
    plan = epi.EpiloguePlan.from_shape(shape, gated=True)
    # Then channel halves are removed only from the channel dimension.
    assert plan.output_shape == (3, 7, 2, 5, 7)
    assert plan.channels == 7
    assert plan.elements == 1470


@pytest.mark.parametrize(
    "shape,gated",
    [
        ((1, 2, 3, 4), False),
        ((1, 2, 0, 4, 5), False),
        ((0, 2, 3, 4, 5), False),
        ((1, -2, 3, 4, 5), False),
        ((1, 7, 3, 4, 5), True),
    ],
)
def test_invalid_geometry_fails_before_optional_imports(
    shape: tuple[int, ...], gated: bool
) -> None:
    # Given an invalid NCDHW domain or unmatched value/gate channels.
    # When planning, then reject without requiring Torch or CUDA.
    with pytest.raises(epi.EpilogueError):
        epi.EpiloguePlan.from_shape(shape, gated=gated)


@pytest.mark.parametrize("name", ["add_bias_residual", "bias_silu_multiply"])
def test_public_bias_api_is_an_explicit_experimental_opt_in(name: str) -> None:
    # Given the real public module, when inspecting its exported entry points.
    function = getattr(api, name)
    # Then callers receive the standard experimental marker and export contract.
    assert name in api.__all__
    assert function.is_experimental


def test_public_import_defers_bias_backend_in_a_fresh_process() -> None:
    # Given a fresh interpreter, when importing the public module.
    # Then importing an API does not import its experimental implementation.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import flashinfer.diffusion_ops.h3_bridge as api; "
            "assert callable(api.add_bias_residual); "
            "assert callable(api.bias_silu_multiply); "
            "assert 'flashinfer.experimental.h3_bridge.bias_epilogue' "
            "not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda x, b: api.add_bias_residual(x, b, x), id="residual"),
        pytest.param(api.bias_silu_multiply, id="gate"),
    ],
)
def test_public_cpu_rejection_preserves_typed_error_without_auto_gate(
    call: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flashinfer import autotune

    # Given CPU operands and no experimental-auto-backend opt-in flag.
    monkeypatch.delenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", raising=False)
    x = torch.ones((2, 8, 3, 5, 7), dtype=torch.bfloat16)
    bias = torch.ones(8, dtype=torch.bfloat16)
    # When a surrounding autotune context propagates the public contract error.
    # Then no environment gate or frozen-exception traceback error masks it.
    with pytest.raises(epi.EpilogueError, match="CUDA BF16"), autotune():
        call(x, bias)
