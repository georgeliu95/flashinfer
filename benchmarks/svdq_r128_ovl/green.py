"""SM-partition probe with green contexts: prefix on P SMs, GEMM on the other SMs, launched
concurrently (no dependency; timing bound only), vs both kernels serial on the full GPU.
Eager launches timed with events on the default (full-GPU) stream; 2xL2 write flush first."""
import json, os, statistics, sys
import torch
import cuda.bindings.driver as drv
sys.path.insert(0, "benchmarks")
from svdquant_sm120_benchmark import build_case
import flashinfer
from flashinfer.autotuner import autotune
from flashinfer.gemm import svdquant_sm120_cutlass as sc

def ck(r):
    err = r[0] if isinstance(r, tuple) else r
    assert err == drv.CUresult.CUDA_SUCCESS, err
    return r[1:] if isinstance(r, tuple) and len(r) > 2 else (r[1] if isinstance(r, tuple) and len(r) == 2 else None)

torch.cuda.init(); torch.zeros(1, device="cuda")
dev = ck(drv.cuDeviceGet(0))
sm_res = ck(drv.cuDeviceGetDevResource(dev, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
TOTAL = sm_res.sm.smCount

def green_streams(P):
    """Two green-context streams: P SMs (rounded by the driver) and the remaining SMs."""
    groups, nb, rem = drv.cuDevSmResourceSplitByCount(1, sm_res, 0, P)[1:]
    assert nb == 1
    out = []
    for res in (groups[0], rem):
        desc = ck(drv.cuDevResourceGenerateDesc([res], 1))
        gctx = ck(drv.cuGreenCtxCreate(desc, dev, drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
        st = ck(drv.cuGreenCtxStreamCreate(gctx, drv.CUstream_flags.CU_STREAM_NON_BLOCKING, 0))
        out.append((res.sm.smCount, gctx, torch.cuda.ExternalStream(int(st))))
    return out

picked = {}
orig = sc._choose_sm120_linear_runner
def hook(*a, **kw):
    r, t = orig(*a, **kw); picked["tactic"] = t; return r, t
sc._choose_sm120_linear_runner = hook
rank = 128
PS = [int(v) for v in sys.argv[2].split(",")]
for spec in sys.argv[1].split(","):
    m, n, k = (int(v) for v in spec.split("x"))
    c = build_case((m, n, k), seed=0, bias=True, rank=rank)
    l2 = flashinfer.prepare_svdquant_l2t(c.l2t)
    with autotune(True):
        flashinfer.svdquant_linear(c.x, c.wq, c.w_sf, c.alpha, c.pqs, l2, c.l1, c.global_sf, c.bias, enable_pdl=True, backend="cutlass-sm120")
    torch.cuda.synchronize()
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
    def prefix(): mod.nvfp4_quantize_smooth_lora_down_dyn_sm120(x, pqs, c.global_sf, lt, xq, xsf, down, fam, *tiling, pol)
    def gemm(): mod.nvfp4_svdquant_gemm(xq, c.wq, xsf, c.w_sf, c.alpha, down, c.l1, c.bias, out, ws, row, True)
    flush = torch.empty(2 * torch.cuda.get_device_properties(0).L2_cache_size, dtype=torch.int8, device="cuda")
    main = torch.cuda.current_stream()

    def timed(body, iters=25):
        ts = []
        for _ in range(iters):
            flush.zero_()
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            a.record(main); body(); b.record(main); ts.append((a, b))
        torch.cuda.synchronize()
        return statistics.median(x.elapsed_time(y) * 1000 for x, y in ts[3:])

    def on(stream, fn):
        stream.wait_stream(main)
        with torch.cuda.stream(stream): fn()
        main.wait_stream(stream)

    def both(sp, sg):
        def body():
            sp.wait_stream(main); sg.wait_stream(main)
            with torch.cuda.stream(sp): prefix()
            with torch.cuda.stream(sg): gemm()
            main.wait_stream(sp); main.wait_stream(sg)
        return body

    splits = {P: green_streams(P) for P in PS}
    cases = {"serial (full GPU)": lambda: (prefix(), gemm()), "prefix (full GPU)": prefix, "gemm (full GPU)": gemm}
    for P, ((np_, _, sp), (ng, _, sg)) in splits.items():
        cases[f"prefix @{np_} SM"] = lambda sp=sp: on(sp, prefix)
        cases[f"gemm @{ng} SM"] = lambda sg=sg: on(sg, gemm)
        cases[f"split {np_}+{ng}"] = both(sp, sg)
    for f in cases.values():
        f()
    torch.cuda.synchronize()
    res = {nm: [] for nm in cases}
    for rnd in range(3):
        for nm, f in cases.items():
            res[nm].append(timed(f))
    print(json.dumps({spec: {"total_sm": TOTAL, "prefix": [fam, list(tiling)], "gemm_tactic": row,
                             "us": {nm: round(statistics.median(v), 1) for nm, v in res.items()}}}), flush=True)
