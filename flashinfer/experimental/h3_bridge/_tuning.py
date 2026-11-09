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

"""Bounded depthwise search using FlashInfer's native autotuner and file cache."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Final, TypedDict

import torch
from typing_extensions import assert_never

from ...autotuner import AutoTuner, OptimizationProfile, TunableRunner, TuningConfig
from . import _support
from ._support import Backend, Policy, Tactic

_OP: Final = "h3_bridge_temporal_depthwise_conv3d"
# Native cold-L2 batching clones small inputs, losing storage-offset alignment.
# Hot profiling keeps every static input descriptor unchanged.
_CONFIG: Final = TuningConfig(
    use_cuda_graph=True,
    cuda_graph_profile_replays=3,
    profiling_repeat=3,
    use_cold_l2_cache=False,
)


class TacticValidationError(RuntimeError):
    """A candidate disagreed with the established FP32-FMA operation order."""

    def __init__(self, backend: str, tactic: Tactic) -> None:
        self.backend = backend
        self.tactic = tactic
        super().__init__(
            f"{backend} tactic {tactic} failed bitwise Triton reference validation"
        )


class TacticSelection(TypedDict):
    """The native cache's actual winner, or its deterministic default."""

    cache_hit: bool
    runner: str
    backend: str
    tactic: Tactic


class _TritonRunner(TunableRunner):
    """A pointer-free runner; all tensor ownership stays with the current call."""

    backend = "triton"

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.policy))

    def get_cache_key_extras(self, inputs: list[torch.Tensor]) -> tuple[Hashable, ...]:
        return _support.cache_extras(inputs, self.policy)

    def tactics(self, inputs: list[torch.Tensor]) -> tuple[tuple[int, ...], ...]:
        return _support.TRITON_TACTICS

    def validate_tactic(self, inputs: list[torch.Tensor], tactic: Tactic) -> bool:
        """Cheap file-cache validity check; never executes a GPU kernel."""
        return tactic == -1 or tactic in self.tactics(inputs)

    def get_valid_tactics(
        self, inputs: list[torch.Tensor], profile: OptimizationProfile
    ) -> list[tuple[int, ...]]:
        reference = torch.empty_like(inputs[3])
        _TritonRunner(self.policy).forward([*inputs[:3], reference])
        tactics = self.tactics(inputs)
        # Compile and check outside native profiling's exception-skip handler.
        for tactic in tactics:
            self.forward(inputs, tactic)
            if not torch.equal(
                inputs[3].view(torch.int16), reference.view(torch.int16)
            ):
                raise TacticValidationError(self.backend, tactic)
        return list(tactics)

    def forward(self, inputs: list[torch.Tensor], tactic: Tactic = -1) -> torch.Tensor:
        from . import _depthwise_triton

        block, warps = (1024, 4) if tactic == -1 else tactic
        return _depthwise_triton.run(
            inputs[0],
            inputs[1],
            inputs[2] if self.policy.has_bias else None,
            bias_first=self.policy.bias_first,
            block=block,
            num_warps=warps,
            out=inputs[3],
        )


class _CuteRunner(_TritonRunner):
    """CuTe candidates must pass bitwise validation outside the timed region."""

    backend = "cute"

    def tactics(self, inputs: list[torch.Tensor]) -> tuple[tuple[int, ...], ...]:
        return _support.cute_tactics(inputs[0])

    def forward(self, inputs: list[torch.Tensor], tactic: Tactic = -1) -> torch.Tensor:
        from . import _depthwise_cute

        default = (20, 128, 2 if inputs[0].shape[1] % 2 == 0 else 1)
        time_tile, threads, vec = default if tactic == -1 else tactic
        return _depthwise_cute.run(
            inputs[0],
            inputs[1],
            inputs[2] if self.policy.has_bias else None,
            bias_first=self.policy.bias_first,
            time_tile=time_tile,
            threads=threads,
            vec=vec,
            out=inputs[3],
        )


def _runners(policy: Policy) -> list[_TritonRunner]:
    """Only the requested backend participates; auto defaults to Triton."""
    match policy.backend:
        case "triton":
            return [_TritonRunner(policy)]
        case "cute":
            if not policy.cute_supported:
                raise _support.BackendUnavailableError(
                    "CuTe depthwise requires channels_last_3d on SM121"
                )
            if not policy.cute_available:
                raise _support.CuTeDependencyError(
                    "CuTe depthwise requires optional CuTe DSL dependencies"
                )
            return [_CuteRunner(policy)]
        case "auto":
            runners = [_TritonRunner(policy)]
            if policy.cute_enabled:
                runners.append(_CuteRunner(policy))
            return runners
        case unreachable:
            assert_never(unreachable)


def _prepare(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    bias_first: bool,
    backend: Backend,
) -> tuple[list[torch.Tensor], list[_TritonRunner]]:
    output = torch.empty_like(inputs, memory_format=torch.channels_last_3d)
    tensors = [inputs, weight, weight if bias is None else bias, output]
    cute_supported = backend != "triton" and _support._cute_supported(
        inputs, weight, bias, bias_first
    )
    cute_available = cute_supported and _support._cute_available()
    policy = Policy(
        backend, bias_first, bias is not None, cute_supported, cute_available
    )
    return tensors, _runners(policy)


def run(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    bias_first: bool,
    backend: Backend,
) -> torch.Tensor:
    """Tune only in flashinfer.autotune(); otherwise use its cache or default."""
    with torch.cuda.device(inputs.device):
        tensors, runners = _prepare(
            inputs, weight, bias, bias_first=bias_first, backend=backend
        )
        runner, tactic = AutoTuner.get().choose_one(_OP, runners, _CONFIG, tensors)
        return runner(tensors, tactic=tactic)


def selected_tactic(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    bias_first: bool,
    backend: Backend,
) -> TacticSelection:
    """Inspect the native cache without compiling, validating, or profiling."""
    with torch.cuda.device(inputs.device):
        tensors, runners = _prepare(
            inputs, weight, bias, bias_first=bias_first, backend=backend
        )
        hit, index, tactic, _ = AutoTuner.get().search_cache(
            _OP,
            runners,
            tuple(tuple(t.shape) for t in tensors),
            _CONFIG,
            inputs=tensors,
        )
    runner = runners[index]
    return TacticSelection(
        cache_hit=hit,
        runner=type(runner).__name__,
        backend=runner.backend,
        tactic=tactic,
    )
