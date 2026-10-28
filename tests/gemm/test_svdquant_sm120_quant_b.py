"""SM120 QuantB linear: the GEMM consumers quantize X themselves.

Every QuantB launch is compared bit for bit with the fused linear that runs the
same GEMM tile and the same LoRA-down producer: the consumers apply the prefix's
exact NVFP4 recipe, so the operands the MMA sees, and therefore every output bit,
must match.
"""

import functools
from dataclasses import dataclass, replace

import pytest
import torch

from flashinfer import SfLayout, autotune, nvfp4_quantize, svdquant_linear
from flashinfer.gemm import svdquant_sm120_cutlass as backend
from flashinfer.gemm import svdquant_sm120_routes as routes

RANK = 128
# QuantB kernel -> KernelShapeSm120 id of the same swap tile in the tactic table.
_TACTIC_KERNEL_IDS = {0: 19, 1: 26, 2: 16, 3: 18}


def _skip_unless_sm120() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("the QuantB linear is an SM120 rank-128 route")


@functools.cache
def _module():
    return backend.get_nvfp4_svdquant_sm120_module(RANK)


@functools.cache
def _default_row(kernel: int) -> int:
    """The tactic row running a QuantB kernel's tile with scheduler defaults."""
    module = _module()
    for row in range(module.nvfp4_svdquant_gemm_tactic_num()):
        desc = int(module.nvfp4_svdquant_gemm_tactic_row(row))
        if (
            desc & 0xFF == _TACTIC_KERNEL_IDS[kernel]
            and (desc >> 8) & 0xFF == 1
            and (desc >> 16) & 0xFF == 0
            and (desc >> 24) & 0xFF == 1
        ):
            return row
    raise AssertionError(f"no default row for QuantB kernel {kernel}")


@dataclass
class _Case:
    x: torch.Tensor
    pqs: torch.Tensor
    global_sf: torch.Tensor
    wq: torch.Tensor
    w_sf: torch.Tensor
    alpha: torch.Tensor
    l2t: torch.Tensor
    l2t_kmajor: torch.Tensor
    l1: torch.Tensor
    bias: torch.Tensor

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.x.shape[0], self.wq.shape[0], self.x.shape[1]


def _make_case(m: int, n: int, k: int, seed: int = 0) -> _Case:
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(
            *shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )

    x = randn(m, k) / (k**0.25)
    pqs = (1.0 + 0.3 * randn(k)).abs().contiguous()
    smoothed = (x * pqs).to(torch.bfloat16)
    global_sf = ((448.0 * 6.0) / smoothed.float().abs().max()).reshape(1).contiguous()
    w = randn(n, k) / (k**0.25)
    gw = ((448.0 * 6.0) / w.float().abs().max()).reshape(1)
    wq, w_sf = nvfp4_quantize(w, gw, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    alpha = (1.0 / (global_sf * gw)).reshape(1).float()
    l2t = (pqs.unsqueeze(1) * (randn(RANK, k) / (k**0.25)).t()).contiguous()
    l1 = (
        ((randn(n, RANK) / (RANK**0.25)).float() / alpha)
        .to(torch.bfloat16)
        .contiguous()
    )
    return _Case(
        x=x.contiguous(),
        pqs=pqs,
        global_sf=global_sf,
        wq=wq.view(torch.uint8),
        w_sf=w_sf.view(torch.uint8).reshape(-1),
        alpha=alpha,
        l2t=l2t,
        l2t_kmajor=backend._pack_sm120_kmajor_l2t(l2t),
        l1=l1,
        bias=randn(n).contiguous(),
    )


@dataclass
class _Outputs:
    xq: torch.Tensor
    sf: torch.Tensor
    down: torch.Tensor
    out: torch.Tensor
    workspace: torch.Tensor


def _outputs(case: _Case, workspace_bytes: int = 64 << 20) -> _Outputs:
    m, n, k = case.shape
    return _Outputs(
        xq=torch.full((m, k // 2), 0xFF, dtype=torch.uint8, device="cuda"),
        sf=torch.full(
            ((m + 127) // 128 * 128 * k // 16,), 0xFF, dtype=torch.uint8, device="cuda"
        ),
        down=torch.full((m, RANK), float("nan"), dtype=torch.bfloat16, device="cuda"),
        out=torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda"),
        workspace=torch.zeros(workspace_bytes, dtype=torch.uint8, device="cuda"),
    )


def _run_quant_b(case: _Case, outputs: _Outputs, variant, enable_pdl: bool) -> None:
    kernel, head, ctas, family, tiling = variant
    _module().nvfp4_svdquant_linear_quant_b_sm120(
        case.x,
        case.pqs,
        case.global_sf,
        case.l2t_kmajor,
        case.wq,
        case.w_sf,
        case.alpha,
        case.l1,
        case.bias,
        outputs.xq,
        outputs.sf,
        outputs.down,
        outputs.out,
        outputs.workspace,
        kernel,
        head,
        ctas,
        family,
        *tiling,
        enable_pdl,
    )


def _reference_producer(m: int, k: int, variant) -> tuple[int, tuple[int, int, int]]:
    """The fused producer whose LoRA-down CTAs match the QuantB launch's prefix."""
    _, _, _, family, tiling = variant
    for candidate_family, candidate, policy in routes.sm120_producer_variants(
        m, k, RANK
    ):
        if policy != 0 or candidate_family != family:
            continue
        same = candidate[:2] == tiling[:2]
        if family == routes.SM120_FAMILY_SMALL_M_SPLIT:
            same = candidate == tiling
        if same:
            return candidate_family, candidate
    raise AssertionError(f"no fused producer matches {variant} at m={m}, k={k}")


def _run_reference(case: _Case, outputs: _Outputs, variant) -> None:
    m, _, k = case.shape
    family, tiling = _reference_producer(m, k, variant)
    _module().nvfp4_svdquant_linear_sm120(
        case.x,
        case.pqs,
        case.global_sf,
        case.l2t_kmajor,
        case.wq,
        case.w_sf,
        case.alpha,
        case.l1,
        case.bias,
        outputs.xq,
        outputs.sf,
        outputs.down,
        outputs.out,
        outputs.workspace,
        _default_row(variant[0]),
        False,
        family,
        *tiling,
        0,
    )


def _representative_variants(m: int, k: int):
    """Per kernel: pure launches with each down family, and hybrids at both head sizes."""
    picked = []
    for kernel in range(len(routes.SM120_QUANT_B_KERNELS)):
        candidates = [
            v for v in routes.sm120_quant_b_variants(m, k, RANK) if v[0] == kernel
        ]
        pure = [v for v in candidates if v[1] == routes.SM120_QUANT_B_ALL_TILES]
        hybrid = [v for v in candidates if v[1] != routes.SM120_QUANT_B_ALL_TILES]
        for family in (routes.SM120_FAMILY_SMALL_M, routes.SM120_FAMILY_SMALL_M_SPLIT):
            picked += [v for v in pure if v[3] == family][:1]
        picked += hybrid[:1] + hybrid[-1:]
    return list(dict.fromkeys(picked))


_SHAPES = [
    (1, 256, 3072),
    (17, 256, 3072),
    (64, 352, 3072),
    (100, 512, 4096),
    (128, 256, 4096),
    (200, 384, 3072),
    (256, 256, 4096),
    # More output tiles than SMs, so CTAs run several work units; K = 42 chunks
    # leaves a partial last chunk group.
    (128, 12288, 5376),
]


@pytest.mark.parametrize("m,n,k", _SHAPES)
@pytest.mark.parametrize("enable_pdl", [True, False])
def test_quant_b_matches_the_fused_linear_bitwise(
    m: int, n: int, k: int, enable_pdl: bool
):
    _skip_unless_sm120()
    case = _make_case(m, n, k)
    variants = _representative_variants(m, k)
    assert variants
    reference = _outputs(case)
    for variant in variants:
        _run_reference(case, reference, variant)
        got = _outputs(case)
        _run_quant_b(case, got, variant, enable_pdl)
        torch.cuda.synchronize()
        context = f"m={m} n={n} k={k} variant={variant}"
        assert torch.equal(got.down, reference.down), f"{context}: LoRA-down"
        assert torch.equal(
            got.out.view(torch.int16), reference.out.view(torch.int16)
        ), f"{context}: output"
        if variant[1] != routes.SM120_QUANT_B_ALL_TILES:
            # The hybrid prefix writes XQ and the padded scales like the fused one.
            assert torch.equal(got.xq, reference.xq), f"{context}: XQ"
            assert torch.equal(got.sf, reference.sf), f"{context}: padded SF"


@pytest.mark.parametrize("m,n,k", [(17, 256, 3072), (128, 512, 4096)])
def test_quant_b_graph_replay_sees_fresh_inputs_and_scratch(m: int, n: int, k: int):
    """Replayed PDL launches match serialized ones after inputs and scratch change."""
    _skip_unless_sm120()
    case = _make_case(m, n, k)
    module = _module()
    groups = -(-(k // routes.SM120_QUANT_B_CHUNK_K) // 8)
    for variant in _representative_variants(m, k):
        graphed = _outputs(case)
        _run_quant_b(case, graphed, variant, True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run_quant_b(case, graphed, variant, True)
        counters = int(
            module.nvfp4_svdquant_quant_b_workspace_size(m, n, k, variant[0])
        )
        for step in range(3):
            case.x.copy_(torch.randn_like(case.x) / (k**0.25))
            expected = _outputs(case)
            _run_quant_b(case, expected, variant, False)
            graphed.xq.fill_(0xFF)
            graphed.sf.fill_(0xFF)
            graphed.down.fill_(float("nan"))
            graphed.out.fill_(float("nan"))
            # The chunk-group counters must be re-zeroed inside the graph.
            graphed.workspace[counters - 4 * groups : counters].fill_(0xFF)
            graph.replay()
            torch.cuda.synchronize()
            context = f"m={m} n={n} k={k} variant={variant} step={step}"
            assert torch.equal(
                graphed.out.view(torch.int16), expected.out.view(torch.int16)
            ), context
            assert torch.equal(graphed.down, expected.down), context
        graph.reset()


def _forced_choice(monkeypatch, runner_name: str, tactic: int) -> None:
    def choose(tuner, runners, tuning_config, inputs, **kwargs):
        for runner in runners:
            if type(runner).__name__ == runner_name:
                return runner, tactic
        raise AssertionError(f"{runner_name} not offered")

    monkeypatch.setattr(backend, "_choose_sm120_linear_runner", choose)


def _linear(case: _Case) -> torch.Tensor:
    return svdquant_linear(
        case.x,
        case.wq,
        case.w_sf,
        case.alpha,
        case.pqs,
        case.l2t,
        case.l1,
        case.global_sf,
        bias=case.bias,
        backend="cutlass-sm120",
    )


@pytest.mark.parametrize("m,n,k", [(16, 512, 4096), (64, 384, 3072)])
def test_svdquant_linear_serves_the_quant_b_runner(monkeypatch, m: int, n: int, k: int):
    """The public API launches every QuantB tactic exactly like the matching fused one."""
    _skip_unless_sm120()
    case = _make_case(m, n, k)
    variants = routes.sm120_quant_b_variants(m, k, RANK)
    producers = routes.sm120_producer_variants(m, k, RANK)
    for variant in _representative_variants(m, k):
        _forced_choice(monkeypatch, "Sm120QuantBLinearRunner", variants.index(variant))
        got = _linear(case)
        family, tiling = _reference_producer(m, k, variant)
        fused = routes.sm120_pack_tactic(
            _default_row(variant[0]), producers.index((family, tiling, 0))
        )
        _forced_choice(monkeypatch, "Sm120FusedLinearRunner", fused)
        expected = _linear(case)
        assert torch.equal(got.view(torch.int16), expected.view(torch.int16)), variant


def test_autotuned_linear_profiles_the_quant_b_runner(monkeypatch):
    """Tuning measures the QuantB tactics, and whatever wins stays accurate."""
    _skip_unless_sm120()
    m, n, k = 16, 512, 4096
    case = _make_case(m, n, k)
    profiled: list[int] = []
    build = backend._sm120_quant_b_linear_runner

    def counting_runner(enable_pdl, device, rank):
        runner = build(enable_pdl, device, rank)
        forward = runner.forward

        def counted(inputs, tactic=-1, **kwargs):
            profiled.append(tactic)
            return forward(inputs, tactic=tactic, **kwargs)

        runner.forward = counted
        return runner

    monkeypatch.setattr(backend, "_sm120_quant_b_linear_runner", counting_runner)
    backend._cached_sm120_linear_runners.cache_clear()
    try:
        _forced_choice(monkeypatch, "Sm120FusedLinearRunner", -1)
        reference = _linear(case).float()
        monkeypatch.undo()
        monkeypatch.setattr(backend, "_sm120_quant_b_linear_runner", counting_runner)
        with autotune(True):
            tuned = _linear(case).float()
        tuned_again = _linear(case).float()
    finally:
        backend._cached_sm120_linear_runners.cache_clear()
    assert profiled and set(profiled) <= set(
        range(len(routes.sm120_quant_b_variants(m, k, RANK)))
    )
    for got in (tuned, tuned_again):
        error = (got - reference).abs().amax() / reference.abs().amax()
        assert error < 0.05


def test_quant_b_rejects_what_it_cannot_run():
    _skip_unless_sm120()
    m, n, k = 16, 256, 4096
    case = _make_case(m, n, k)
    variant = routes.sm120_quant_b_variants(m, k, RANK)[0]
    # A workspace without room for the GEMM scratch and the chunk-group counters.
    small = _outputs(case, workspace_bytes=16)
    with pytest.raises(Exception, match="workspace"):
        _run_quant_b(case, small, variant, True)
    # A hybrid needs the chunked family-1 prefix.
    hybrid = next(v for v in routes.sm120_quant_b_variants(m, k, RANK) if v[1] >= 0)
    with pytest.raises(Exception, match="chunked family-1 prefix"):
        _run_quant_b(case, _outputs(case), (hybrid[0], hybrid[1], 0, *hybrid[3:]), True)
    with pytest.raises(Exception, match="family 1 or 6"):
        _run_quant_b(case, _outputs(case), (*variant[:3], 0, variant[4]), True)
    with pytest.raises(Exception, match="b_sf is smaller"):
        _run_quant_b(replace(case, w_sf=case.w_sf[:-16]), _outputs(case), variant, True)
    shifted = torch.empty(n * RANK + 8, dtype=torch.bfloat16, device="cuda")
    misaligned_l1 = shifted[1 : 1 + n * RANK].view(n, RANK)
    misaligned_l1.copy_(case.l1)
    with pytest.raises(Exception, match="l1 must be 16-byte aligned"):
        _run_quant_b(replace(case, l1=misaligned_l1), _outputs(case), variant, True)
    # Rejected before anything is enqueued: the prefix never writes LoRA-down.
    untouched = _outputs(case)
    with pytest.raises(Exception, match="bias must have shape"):
        _run_quant_b(replace(case, bias=case.bias[:-32]), untouched, hybrid, True)
    torch.cuda.synchronize()
    assert torch.isnan(untouched.down).all()
