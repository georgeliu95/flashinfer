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

"""Cross-process native tactic-cache hydration and warm graph replay."""

from collections.abc import Iterator
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from flashinfer import autotune
from flashinfer.autotuner import AutoTuner
from flashinfer.experimental.h3_bridge import _tuning

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def isolated_tuner() -> Iterator[AutoTuner]:
    tuner = AutoTuner.get()
    tuner.clear_cache()
    yield tuner
    tuner.clear_cache()


def test_file_cache_in_fresh_process_with_warm_graph(tmp_path: Path) -> None:
    x = torch.randn((2, 16, 7, 3, 5), device="cuda", dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last_3d
    )
    w = torch.randn((16, 1, 3, 1, 1), device="cuda", dtype=x.dtype)
    b = torch.randn((16,), device="cuda", dtype=x.dtype)
    cache = tmp_path / "tuning.json"
    with autotune(cache=str(cache)):
        _tuning.run(x, w, b, bias_first=True, backend="auto")
    script = r"""
import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch
import torch
from flashinfer import autotune
from flashinfer.autotuner import AutoTuner
from flashinfer.experimental.h3_bridge import _tuning, _depthwise_triton
x = torch.randn((2,16,7,3,5),device="cuda",dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last_3d)
w = torch.randn((16,1,3,1,1),device="cuda",dtype=x.dtype)
b = torch.randn((16,),device="cuda",dtype=x.dtype)
with patch.object(AutoTuner.get(), "_profile_single_kernel", side_effect=AssertionError):
    with autotune(cache=sys.argv[1]):
        _tuning.run(x,w,b,bias_first=True,backend="auto")
        assert _tuning.selected_tactic(x,w,b,bias_first=True,backend="auto")["cache_hit"]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = _tuning.run(x,w,b,bias_first=True,backend="auto")
        x.add_(1)
        graph.replay()
assert torch.equal(y, _depthwise_triton.run(x,w,b,bias_first=True))
print("fresh-process cache hit and graph replay passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(cache)],
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
