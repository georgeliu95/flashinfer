"""Direction A ceiling probe: serial (same stream) vs two concurrent streams, no dependency.
The concurrent variant is wrong numerically (GEMM reads stale xq/down) but bounds the
time any per-M-tile overlap scheme could reach on the same kernels."""
import sys, statistics, torch
sys.path.insert(0, "benchmarks")
from svdquant_sm120_benchmark import build_case
from flashinfer.gemm import svdquant_sm120_cutlass as sc

spec, k3_row = sys.argv[1], int(sys.argv[2])
fam, t0, t1, t2, pol = (int(v) for v in sys.argv[3].split(","))
rank = 128
m, n, k = (int(v) for v in spec.split("x"))
c = build_case((m, n, k), seed=0, bias=True, rank=rank)
mod = sc.get_nvfp4_svdquant_sm120_module(rank)
l2t = sc._cached_sm120_kmajor_l2t(c.l2t)
xq = torch.empty((m, k // 2), dtype=torch.uint8, device="cuda")
xsf = torch.empty((sc._swizzled_sf_size(m, k // 16),), dtype=torch.uint8, device="cuda")
down = torch.empty((m, rank), dtype=torch.bfloat16, device="cuda")
out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
ws = torch.zeros(64 << 20, dtype=torch.uint8, device="cuda")
x, pqs = sc._normalize_nvfp4_quantize_smooth_inputs(c.x, c.pqs)

def prefix():
    mod.nvfp4_quantize_smooth_lora_down_dyn_sm120(x, pqs, c.global_sf, l2t, xq, xsf, down, fam, t0, t1, t2, pol)
def gemm():
    mod.nvfp4_svdquant_gemm(xq.view(torch.uint8), c.wq, xsf, c.w_sf, c.alpha, down, c.l1, c.bias, out, ws, k3_row, True)

s1 = torch.cuda.Stream(); s2 = torch.cuda.Stream()
def serial():
    prefix(); gemm()
def concurrent():
    cur = torch.cuda.current_stream()
    s1.wait_stream(cur); s2.wait_stream(cur)
    with torch.cuda.stream(s1): prefix()
    with torch.cuda.stream(s2): gemm()
    cur.wait_stream(s1); cur.wait_stream(s2)
def gemm_only(): gemm()
def prefix_only(): prefix()

flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
for name, fn in [("serial", serial), ("concurrent", concurrent), ("prefix", prefix_only), ("gemm", gemm_only)]:
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(30):
        flush.zero_()
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    ts.sort()
    print(f"CEIL {spec} {name:10s} median={statistics.median(ts):8.1f}us p20={ts[len(ts)//5]:8.1f}us min={ts[0]:8.1f}us", flush=True)
