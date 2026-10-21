"""K sweep at fixed M, N: native CUTLASS (NVFP4 + bias), FI cutlass-sm120 and cute-dsl SVDQ GEMM
(rank 32). Cold-L2 graph replay, interleaved rounds. Reports TFLOPS on the FP4 part and % of the
1069 TFLOPS mma.sync instruction peak."""
import json, statistics, sys
import torch
sys.path.insert(0, "benchmarks")
from svdquant_sm120_benchmark import build_case
from svdquant_sm120_native import native_operands, native_call, _cold_graph_us
import flashinfer
from flashinfer.autotuner import autotune
M, N = (int(v) for v in sys.argv[1].split("x")); Ks = [int(v) for v in sys.argv[2].split(",")]; rank = int(sys.argv[3]) if len(sys.argv) > 3 else 32
PEAK = 1069e12
for K in Ks:
    c = build_case((M, N, K), seed=0, bias=True, rank=rank)
    runs = {}
    for be in ("cutlass-sm120", "cute-dsl"):
        out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        f = lambda be=be, out=out: flashinfer.mm_nvfp4_svdquant(c.xq, c.wq, c.x_sf, c.w_sf, c.alpha, c.down, c.l1, c.bias, out=out, backend=be, enable_pdl=True)
        try:
            with autotune(True):
                f()
            torch.cuda.synchronize(); runs[be] = f
        except Exception as e:
            print(json.dumps({"K": K, "skip": be, "err": str(e)[:120]}), flush=True)
    runs["native-cutlass"], _ = native_call("native-cutlass", native_operands(build_case((M, N, K), seed=0, bias=True, rank=32)), quantize=False)
    res = {nm: [] for nm in runs}
    for _ in range(3):
        for nm, f in runs.items():
            res[nm].append(_cold_graph_us(f))
    fl = 2 * M * N * K
    lora = 2 * M * N * rank / 267e12 * 1e6
    out = {nm: {"us": round(statistics.median(v), 1), "pct_fp4_peak": round(100 * fl / (statistics.median(v) * 1e-6) / PEAK, 1)} for nm, v in res.items()}
    print(json.dumps({"MxN": [M, N], "K": K, "rank": rank, "lora_up_sol_us": round(lora, 1), "fp4_sol_us": round(fl / PEAK * 1e6, 1), "r": out}), flush=True)
