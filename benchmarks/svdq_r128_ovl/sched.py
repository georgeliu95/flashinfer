"""GEMM scheduler A/B: tuned rank-128 FI GEMM vs best native CUTLASS, per shape. Run once with
the default build and once with SVDQ_GEMM_STATIC_PERSISTENT=1 (separate cache dir)."""
import json, os, statistics, sys
import torch
sys.path.insert(0, "benchmarks")
from svdquant_sm120_benchmark import build_case
from svdquant_sm120_native import native_operands, native_call, _cold_graph_us
import flashinfer
from flashinfer.autotuner import AutoTuner, autotune
picked = []
orig = AutoTuner.choose_one
def hook(self, *a, **kw):
    r, t = orig(self, *a, **kw); picked.append(t); return r, t
AutoTuner.choose_one = hook
tag = "static" if os.environ.get("SVDQ_GEMM_STATIC_PERSISTENT") else "clc"
for spec in sys.argv[1].split(","):
    m, n, k = (int(v) for v in spec.split("x"))
    c = build_case((m, n, k), seed=0, bias=True, rank=128)
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    f = lambda: flashinfer.mm_nvfp4_svdquant(c.xq, c.wq, c.x_sf, c.w_sf, c.alpha, c.down, c.l1, c.bias, out=out, backend="cutlass-sm120", enable_pdl=True)
    AutoTuner.get().clear_cache(); picked.clear()
    with autotune(True):
        f()
    torch.cuda.synchronize(); tac = picked[-1]
    ref = out.clone()
    nat, nt = native_call("native-cutlass", native_operands(build_case((m, n, k), seed=0, bias=True, rank=32)), quantize=False)
    fi, nv = [], []
    for _ in range(3):
        fi.append(_cold_graph_us(f)); nv.append(_cold_graph_us(nat))
    print(json.dumps({"build": tag, "shape": spec, "tactic": tac, "fi_us": round(statistics.median(fi), 1),
                      "native_us": round(statistics.median(nv), 1), "ratio": round(statistics.median(nv) / statistics.median(fi), 3),
                      "out_sum": float(ref.float().abs().sum())}), flush=True)
