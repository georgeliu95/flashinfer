"""Overlap ceiling on current code. Tune svdquant_linear (cutlass-sm120), take the tuner's
(producer variant, GEMM tactic), then time the same two kernels: serial on one stream vs
concurrent on two streams in several launch orders/priorities (numerically wrong, timing
bound only). Graph replay, 2xL2 write flush before each replay, interleaved rounds."""
import json, statistics, sys
import torch
sys.path.insert(0, "benchmarks")
from svdquant_sm120_benchmark import build_case
import flashinfer
from flashinfer.autotuner import autotune
from flashinfer.gemm import svdquant_sm120_cutlass as sc

picked = {}
orig = sc._choose_sm120_linear_runner
def hook(*a, **kw):
    r, t = orig(*a, **kw); picked["runner"], picked["tactic"] = type(r).__name__, t; return r, t
sc._choose_sm120_linear_runner = hook
rank = 128
for spec in sys.argv[1].split(","):
    m, n, k = (int(v) for v in spec.split("x"))
    c = build_case((m, n, k), seed=0, bias=True, rank=rank)
    l2 = flashinfer.prepare_svdquant_l2t(c.l2t)
    call = lambda: flashinfer.svdquant_linear(c.x, c.wq, c.w_sf, c.alpha, c.pqs, l2, c.l1, c.global_sf, c.bias, enable_pdl=True, backend="cutlass-sm120")
    with autotune(True):
        call()
    call(); torch.cuda.synchronize()
    if picked["runner"] != "Sm120FusedLinearRunner":
        print(json.dumps({spec: {"skip": picked}})); continue
    row, variant = sc._sm120_unpack_tactic(picked["tactic"])
    fam, tiling, pol = sc._sm120_decode_producer_variant(m, k, variant, rank)
    mod = sc.get_nvfp4_svdquant_sm120_module(rank)
    lt = sc._sm120_fused_l2t(l2, m, k, variant, rank)
    xq = torch.empty((m, k // 2), dtype=torch.uint8, device="cuda")
    xsf = torch.empty((sc._swizzled_sf_size(m, k // 16),), dtype=torch.uint8, device="cuda")
    down = torch.empty((m, rank), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    ws = torch.zeros(max(int(mod.nvfp4_svdquant_gemm_workspace_size(m, n, k, row)), 1 << 20), dtype=torch.uint8, device="cuda")
    x, pqs = sc._normalize_nvfp4_quantize_smooth_inputs(c.x, c.pqs)
    prefix = lambda: mod.nvfp4_quantize_smooth_lora_down_dyn_sm120(x, pqs, c.global_sf, lt, xq, xsf, down, fam, *tiling, pol)
    gemm = lambda: mod.nvfp4_svdquant_gemm(xq, c.wq, xsf, c.w_sf, c.alpha, down, c.l1, c.bias, out, ws, row, True)
    lo, hi = torch.cuda.Stream.priority_range() if hasattr(torch.cuda.Stream, "priority_range") else (0, -1)
    sA, sB = torch.cuda.Stream(), torch.cuda.Stream()
    sHi = torch.cuda.Stream(priority=-1); sLo = torch.cuda.Stream(priority=0)
    def two(first, second, s1, s2):
        def f():
            cur = torch.cuda.current_stream(); s1.wait_stream(cur); s2.wait_stream(cur)
            with torch.cuda.stream(s1): first()
            with torch.cuda.stream(s2): second()
            cur.wait_stream(s1); cur.wait_stream(s2)
        return f
    variants = {
        "linear (production)": call,
        "serial": lambda: (prefix(), gemm()),
        "concurrent prefix-first": two(prefix, gemm, sA, sB),
        "concurrent gemm-first": two(gemm, prefix, sA, sB),
        "concurrent prefix high-prio": two(prefix, gemm, sHi, sLo),
        "prefix only": prefix,
        "gemm only": gemm,
    }
    graphs = {}
    for name, f in variants.items():
        f(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): f()
        graphs[name] = g
    flush = torch.empty(2 * torch.cuda.get_device_properties(0).L2_cache_size, dtype=torch.int8, device="cuda")
    res = {nm: [] for nm in graphs}
    for rnd in range(5):
        for nm, g in graphs.items():
            ts = []
            for _ in range(40):
                flush.zero_(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
                a.record(); g.replay(); b.record(); ts.append((a, b))
            torch.cuda.synchronize()
            res[nm].append(statistics.median(x.elapsed_time(y) * 1000 for x, y in ts[5:]))
    out_d = {nm: round(statistics.median(v), 1) for nm, v in res.items()}
    print(json.dumps({spec: {"prefix": [fam, list(tiling), pol], "gemm_tactic": row, "us": out_d}}), flush=True)
