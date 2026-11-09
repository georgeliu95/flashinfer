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

"""Public opt-in API and a representative upscale-to-adapter operator chain."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys

import pytest
import torch


def test_public_module_exists() -> None:
    # Given the installed FlashInfer package.
    # When resolving the public bridge API.
    spec = importlib.util.find_spec("flashinfer.diffusion_ops.h3_bridge")
    # Then callers have a supported import path without importing private kernels.
    assert spec is not None


def test_import_keeps_bridge_backends_lazy() -> None:
    # Given a fresh interpreter, so earlier tests cannot hide eager imports.
    # When importing the public API, then its experimental backends stay unloaded.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import flashinfer.diffusion_ops.h3_bridge; "
            "assert not any(m.startswith('flashinfer.experimental.h3_bridge') for m in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_all_bridge_entry_points_are_explicit_experimental_opt_ins() -> None:
    # Given the core entry point module.
    api = importlib.import_module("flashinfer.diffusion_ops.h3_bridge")
    # When inspecting the advertised functions.
    functions = (
        api.group_norm_silu,
        api.modulate_silu,
        api.silu_multiply,
        api.add,
        api.silu,
        api.depthwise_temporal,
    )
    # Then every entry point carries the repository's experimental marker.
    assert all(getattr(function, "is_experimental", False) for function in functions)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_public_api_preserves_layout_through_bridge_operator_chain() -> None:
    # Given an inference-only BF16 3D feature map and deterministic parameters.
    from flashinfer.diffusion_ops import h3_bridge as api

    torch.manual_seed(1109)
    x = torch.randn((1, 32, 7, 5, 9), device="cuda", dtype=torch.bfloat16)
    x = x.contiguous(memory_format=torch.channels_last_3d)
    weight = torch.randn(32, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn_like(weight)
    dw_weight = torch.randn((32, 1, 5, 1, 1), device="cuda", dtype=torch.bfloat16)
    normalized = torch.nn.functional.silu(
        torch.nn.functional.group_norm(x, 8, weight, bias, 1e-5)
    )
    convolved = torch.nn.functional.conv3d(
        normalized.float(),
        dw_weight.float(),
        bias.float(),
        padding=(2, 0, 0),
        groups=32,
    ).bfloat16()
    expected = convolved * torch.nn.functional.silu(x) + x
    # When executing normalization, temporal convolution, gating and the residual.
    result = api.add(
        api.silu_multiply(
            api.depthwise_temporal(
                api.group_norm_silu(x, weight, bias, groups=8, eps=1e-5),
                dw_weight,
                bias,
            ),
            x,
        ),
        x,
    )
    # Then the public path retains channels-last and the reference operation domain.
    torch.testing.assert_close(result, expected, atol=0.125, rtol=0.03)
    assert result.is_contiguous(memory_format=torch.channels_last_3d)
