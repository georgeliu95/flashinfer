# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native autotuning, cache separation, and capture regression tests."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch
import json

import pytest
import torch

from flashinfer import autotune
from flashinfer.autotuner import AutoTuner
from flashinfer.experimental.h3_bridge import _depthwise_triton, _support, _tuning
from flashinfer.experimental.h3_bridge._support import Backend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def isolated_tuner() -> Iterator[AutoTuner]:
    tuner = AutoTuner.get()
    tuner.clear_cache()
    yield tuner
    tuner.clear_cache()


def require_cute() -> None:
    if torch.cuda.get_device_capability() != (12, 1) or not _support._cute_available():
        pytest.skip("CuTe depthwise requires SM121 and optional DSL dependencies")


def samples(
    shape: tuple[int, int, int, int, int] = (2, 16, 7, 3, 5), taps: int = 3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last_3d
    )
    w = torch.randn((shape[1], 1, taps, 1, 1), device="cuda", dtype=x.dtype)
    b = torch.randn((shape[1],), device="cuda", dtype=x.dtype)
    return x, w, b


def reference(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, bias_first: bool = True
) -> torch.Tensor:
    return _depthwise_triton.run(x, w, b, bias_first=bias_first)


@pytest.mark.parametrize("backend", ["auto", "triton", "cute"])
def test_native_search_and_cache_hit(
    backend: Backend, isolated_tuner: AutoTuner, tmp_path: Path
) -> None:
    if backend == "cute":
        require_cute()
    x, w, b = samples()
    expected = reference(x, w, b)
    has_cute = _support._cute_supported(x, w, b, True) and _support._cute_available()
    with patch.object(
        isolated_tuner,
        "_profile_single_kernel",
        wraps=isolated_tuner._profile_single_kernel,
    ) as measured:
        with autotune(cache=str(tmp_path / "tuning.json")):
            actual = _tuning.run(x, w, b, bias_first=True, backend=backend)
        assert measured.call_count >= (8 if backend == "auto" and has_cute else 4)
    assert torch.equal(actual, expected)
    winner = _tuning.selected_tactic(x, w, b, bias_first=True, backend=backend)
    assert winner["cache_hit"]
    assert winner["backend"] in ({"triton", "cute"} if backend == "auto" else {backend})
    assert winner["tactic"] != -1
    fresh_x, fresh_w, fresh_b = x + 1, w + 1, b - 1
    with (
        patch.object(
            isolated_tuner, "_profile_single_kernel", side_effect=AssertionError
        ),
        autotune(),
    ):
        fresh = _tuning.run(fresh_x, fresh_w, fresh_b, bias_first=True, backend=backend)
    assert torch.equal(fresh, reference(fresh_x, fresh_w, fresh_b))
    print(json.dumps({"backend": backend, "selection": winner}))


def test_cache_separates_contracts() -> None:
    x, w, b = samples()
    with autotune():
        _tuning.run(x, w, b, bias_first=True, backend="triton")
    variants = [
        (x.contiguous(), w, b, True, "triton"),
        (x, w, None, True, "triton"),
        (x, w, b, False, "triton"),
        (x, w, b, True, "auto"),
        (*samples((2, 16, 9, 3, 5)), True, "triton"),
    ]
    offset = torch.empty(x.numel() + 1, device=x.device, dtype=x.dtype)[1:]
    offset = offset.view(2, 7, 3, 5, 16).permute(0, 4, 1, 2, 3)
    offset.copy_(x)
    variants.append((offset, w, b, True, "triton"))
    for a, k, bias, order, backend in variants:
        assert not _tuning.selected_tactic(
            a, k, bias, bias_first=order, backend=backend
        )["cache_hit"]
    assert _tuning.selected_tactic(
        x.clone(), w.clone(), b.clone(), bias_first=True, backend="triton"
    )["cache_hit"]


def test_untuned_fallback_never_profiles(isolated_tuner: AutoTuner) -> None:
    x, w, b = samples()
    with patch.object(
        isolated_tuner, "_profile_single_kernel", side_effect=AssertionError
    ):
        actual = _tuning.run(x, w, b, bias_first=False, backend="auto")
    assert torch.equal(actual, reference(x, w, b, False))
    selection = _tuning.selected_tactic(x, w, b, bias_first=False, backend="auto")
    assert selection == {
        "cache_hit": False,
        "runner": "_TritonRunner",
        "backend": "triton",
        "tactic": -1,
    }


def test_cute_validation_fails_before_profiling(isolated_tuner: AutoTuner) -> None:
    require_cute()
    from flashinfer.experimental.h3_bridge import _depthwise_cute

    x, w, b = samples()

    def wrong(
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        bias_first: bool,
        time_tile: int,
        threads: int,
        vec: int,
        out: torch.Tensor,
    ) -> torch.Tensor:
        return out.zero_()

    with (
        patch.object(_depthwise_cute, "run", side_effect=wrong),
        patch.object(
            isolated_tuner, "_profile_single_kernel", side_effect=AssertionError
        ),
        pytest.raises(RuntimeError, match="bitwise"),
        autotune(),
    ):
        _tuning.run(x, w, b, bias_first=True, backend="cute")


def test_compilation_failure_is_visible() -> None:
    require_cute()
    from flashinfer.experimental.h3_bridge import _depthwise_cute

    x, w, b = samples()
    with (
        patch.object(
            _depthwise_cute, "run", side_effect=RuntimeError("compile sentinel")
        ),
        pytest.raises(RuntimeError, match="compile sentinel"),
        autotune(),
    ):
        _tuning.run(x, w, b, bias_first=True, backend="cute")


@pytest.mark.parametrize(
    "shape,taps",
    [
        ((1, 512, 20, 48, 84), 5),
        ((1, 512, 42, 48, 84), 5),
        ((1, 752, 17, 24, 42), 3),
        ((2, 15, 7, 3, 5), 3),
    ],
)
def test_representative_shapes(
    shape: tuple[int, int, int, int, int], taps: int
) -> None:
    x, w, b = samples(shape, taps)
    with autotune():
        actual = _tuning.run(x, w, b, bias_first=True, backend="auto")
    assert torch.equal(actual, reference(x, w, b))
    print(
        json.dumps(
            {
                "shape": shape,
                "selection": _tuning.selected_tactic(
                    x, w, b, bias_first=True, backend="auto"
                ),
            }
        )
    )


def test_cold_capture_refuses_tuning(isolated_tuner: AutoTuner) -> None:
    x, w, b = samples((1, 18, 11, 2, 3))
    _tuning.run(x, w, b, bias_first=True, backend="triton")
    with (
        patch.object(
            isolated_tuner, "_profile_single_kernel", side_effect=AssertionError
        ),
        pytest.raises(RuntimeError, match="capture"),
        autotune(),
        torch.cuda.graph(torch.cuda.CUDAGraph()),
    ):
        _tuning.run(x, w, b, bias_first=True, backend="triton")


def test_offset_autotuning_preserves_alignment_and_warm_graph(
    isolated_tuner: AutoTuner,
) -> None:
    require_cute()
    x, w, b = samples((1, 18, 7, 3, 5))
    storage = torch.empty(x.numel() + 1, device=x.device, dtype=x.dtype)[1:]
    offset = storage.view(1, 7, 3, 5, 18).permute(0, 4, 1, 2, 3)
    offset.copy_(x)
    assert offset.data_ptr() % 4 == 2
    with autotune():
        actual = _tuning.run(offset, w, b, bias_first=False, backend="cute")
    assert torch.equal(actual, reference(offset, w, b, False))
    assert _tuning.selected_tactic(offset, w, b, bias_first=False, backend="cute")[
        "cache_hit"
    ]
    graph = torch.cuda.CUDAGraph()
    with (
        patch.object(
            isolated_tuner, "_profile_single_kernel", side_effect=AssertionError
        ),
        autotune(),
        torch.cuda.graph(graph),
    ):
        captured = _tuning.run(offset, w, b, bias_first=False, backend="cute")
    offset.add_(1)
    graph.replay()
    assert torch.equal(captured, reference(offset, w, b, False))


@pytest.mark.parametrize(
    "supported,available,error",
    [
        (False, True, ValueError),
        (True, False, ImportError),
    ],
)
def test_explicit_cute_availability_errors(
    supported: bool,
    available: bool,
    error: type[ValueError] | type[ImportError],
) -> None:
    x, w, b = samples()
    with (
        patch.object(_support, "_cute_supported", return_value=supported),
        patch.object(_support, "_cute_available", return_value=available),
        pytest.raises(error),
    ):
        _tuning.run(x, w, b, bias_first=True, backend="cute")
