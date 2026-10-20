"""Direction A: tuned cutlass-sm120 linear, print the chosen route, graph replay cold L2."""
import sys, torch, flashinfer
sys.path.insert(0, "benchmarks")
from svdquant_sm120_benchmark import build_case
from flashinfer import svdquant_linear
from flashinfer.gemm import svdquant_sm120_cutlass as sc
from flashinfer.gemm import svdquant_sm120_routes as sr

rank = int(sys.argv[2]) if len(sys.argv) > 2 else 128
for spec in sys.argv[1].split(","):
    m, n, k = (int(v) for v in spec.split("x"))
    c = build_case((m, n, k), seed=0, bias=True, rank=rank)
    args = (c.x, c.wq, c.w_sf, c.alpha, c.pqs, c.l2t, c.l1, c.global_sf, c.bias)
    be = "cutlass-sm120"
    with flashinfer.autotune_v2(mode="tune", persistent_cache=False,
                                measurement_policy=flashinfer.MeasurementPolicy(execution_mode="cuda_graph", cold_l2=True)):
        svdquant_linear(*args, backend=be)
    torch.cuda.synchronize()
    for key, (rid, tac) in sc._SM120_LINEAR_DISPATCH_CACHE.items():
        if key[2:5] == (m, n, k):
            row, var = sr.sm120_unpack_tactic(tac)
            print(f"ROUTE {m}x{n}x{k} runner={rid} tactic={tac} k3_row={row} variant={var} "
                  f"producer={sr.sm120_decode_producer_variant(m, k, var, rank)}", flush=True)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        svdquant_linear(*args, backend=be)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        svdquant_linear(*args, backend=be)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for _ in range(5):
        flush.zero_()
        torch.cuda.nvtx.range_push(f"{be}|{m}x{n}x{k}")
        g.replay()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
