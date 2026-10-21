"""Small-shape GEMM probe: tuned FI rank-128 and rank-32 GEMM vs best native CUTLASS at one
shape. Prints graph-replay cold-L2 times, then launches each once inside a profiler range."""
import json, statistics, sys
import torch
sys.path.insert(0, "benchmarks")
from svdquant_sm120_benchmark import build_case
from svdquant_sm120_native import native_operands, native_call, _cold_graph_us
import flashinfer
from flashinfer.autotuner import AutoTuner, autotune
m, n, k = (int(v) for v in sys.argv[1].split("x"))
picked = {}
orig = AutoTuner.choose_one
def hook(self, *a, **kw):
    r, t = orig(self, *a, **kw); picked[a[0]] = t; return r, t
AutoTuner.choose_one = hook
runs, info = {}, {}
for rank in (128, 32):
    c = build_case((m, n, k), seed=0, bias=True, rank=rank)
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    f = lambda c=c, out=out: flashinfer.mm_nvfp4_svdquant(c.xq, c.wq, c.x_sf, c.w_sf, c.alpha, c.down, c.l1, c.bias, out=out, backend="cutlass-sm120", enable_pdl=True)
    with autotune(True):
        f()
    torch.cuda.synchronize()
    runs[f"fi r{rank}"] = f
    info[f"fi r{rank} tactic"] = list(picked.values())[-1]; picked.clear()
ops = native_operands(build_case((m, n, k), seed=0, bias=True, rank=32))
for be in ("native-cutlass", "native-cublaslt"):
    fn, t = native_call(be, ops, quantize=False)
    runs[be] = fn; info[f"{be} tactic"] = t
res = {nm: [] for nm in runs}
for _ in range(3):
    for nm, fn in runs.items():
        res[nm].append(_cold_graph_us(fn))
print(json.dumps({"shape": [m, n, k], "us": {nm: round(statistics.median(v), 1) for nm, v in res.items()}, "info": info}), flush=True)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for nm in ("fi r128", "fi r32", "native-cutlass"):
    runs[nm]()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("ORDER fi r128, fi r32, native-cutlass")
