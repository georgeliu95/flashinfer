"""SM-partition probe. The tuned prefix (WS, persistent) and GEMM (persistent) are launched
with their grids capped by SVDQ_PREFIX_SM_CAP / SVDQ_GEMM_SM_CAP (ablation build only).
1) prefix alone vs P SMs; 2) GEMM alone vs G SMs; 3) both concurrently on two streams with
P + G = 110 (no dependency: timing bound only). Graph replay, 2xL2 write flush, interleaved."""
import json, os, statistics, sys
import torch
sys.path.insert(0, "benchmarks")
from svdquant_sm120_benchmark import build_case
import flashinfer
from flashinfer.autotuner import autotune
from flashinfer.gemm import svdquant_sm120_cutlass as sc
assert os.environ.get("SVDQ_SM120_WS_ABLATIONS")
picked = {}
orig = sc._choose_sm120_linear_runner
def hook(*a, **kw):
    r, t = orig(*a, **kw); picked["runner"], picked["tactic"] = type(r).__name__, t; return r, t
sc._choose_sm120_linear_runner = hook
rank, SMS = 128, torch.cuda.get_device_properties(0).multi_processor_count
PS = [int(v) for v in sys.argv[2].split(",")]
for spec in sys.argv[1].split(","):
    m, n, k = (int(v) for v in spec.split("x"))
    c = build_case((m, n, k), seed=0, bias=True, rank=rank)
    l2 = flashinfer.prepare_svdquant_l2t(c.l2t)
    call = lambda: flashinfer.svdquant_linear(c.x, c.wq, c.w_sf, c.alpha, c.pqs, l2, c.l1, c.global_sf, c.bias, enable_pdl=True, backend="cutlass-sm120")
    with autotune(True):
        call()
    call(); torch.cuda.synchronize()
    row, variant = sc._sm120_unpack_tactic(picked["tactic"])
    fam, tiling, pol = sc._sm120_decode_producer_variant(m, k, variant, rank)
    assert fam == 5, ("not the WS family", fam)
    mod = sc.get_nvfp4_svdquant_sm120_module(rank)
    lt = sc._sm120_fused_l2t(l2, m, k, variant, rank)
    xq = torch.empty((m, k // 2), dtype=torch.uint8, device="cuda")
    xsf = torch.empty((sc._swizzled_sf_size(m, k // 16),), dtype=torch.uint8, device="cuda")
    down = torch.empty((m, rank), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    ws = torch.zeros(max(int(mod.nvfp4_svdquant_gemm_workspace_size(m, n, k, row)), 1 << 20), dtype=torch.uint8, device="cuda")
    x, pqs = sc._normalize_nvfp4_quantize_smooth_inputs(c.x, c.pqs)
    def prefix(): mod.nvfp4_quantize_smooth_lora_down_dyn_sm120(x, pqs, c.global_sf, lt, xq, xsf, down, fam, *tiling, pol)
    def gemm(): mod.nvfp4_svdquant_gemm(xq, c.wq, xsf, c.w_sf, c.alpha, down, c.l1, c.bias, out, ws, row, True)
    sA, sB = torch.cuda.Stream(), torch.cuda.Stream()
    def capture(fn, pcap, gcap):
        for key, v in (("SVDQ_PREFIX_SM_CAP", pcap), ("SVDQ_GEMM_SM_CAP", gcap)):
            if v is None: os.environ.pop(key, None)
            else: os.environ[key] = str(v)
        fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): fn()
        os.environ.pop("SVDQ_PREFIX_SM_CAP", None); os.environ.pop("SVDQ_GEMM_SM_CAP", None)
        return g
    def concurrent():
        cur = torch.cuda.current_stream(); sA.wait_stream(cur); sB.wait_stream(cur)
        with torch.cuda.stream(sA): prefix()
        with torch.cuda.stream(sB): gemm()
        cur.wait_stream(sA); cur.wait_stream(sB)
    graphs = {"serial": capture(lambda: (prefix(), gemm()), None, None),
              "prefix@110": capture(prefix, None, None), "gemm@110": capture(gemm, None, None)}
    for P in PS:
        graphs[f"prefix@{P}"] = capture(prefix, P, None)
        graphs[f"gemm@{SMS - P}"] = capture(gemm, None, SMS - P)
        graphs[f"split P={P}"] = capture(concurrent, P, SMS - P)
    flush = torch.empty(2 * torch.cuda.get_device_properties(0).L2_cache_size, dtype=torch.int8, device="cuda")
    res = {nm: [] for nm in graphs}
    for rnd in range(3):
        for nm, g in graphs.items():
            ts = []
            for _ in range(30):
                flush.zero_(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
                a.record(); g.replay(); b.record(); ts.append((a, b))
            torch.cuda.synchronize()
            res[nm].append(statistics.median(u.elapsed_time(v) * 1000 for u, v in ts[5:]))
    print(json.dumps({spec: {"prefix": [fam, list(tiling)], "gemm_tactic": row, "us": {nm: round(statistics.median(v), 1) for nm, v in res.items()}}}), flush=True)
