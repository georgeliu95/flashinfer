"""Launch the tuned GEMM with/without SVDQ_GEMM_SM_CAP and the prefix with/without
SVDQ_PREFIX_SM_CAP, eagerly, for nsys grid inspection."""
import os, sys, torch
sys.path.insert(0, "benchmarks")
from svdquant_sm120_benchmark import build_case
from flashinfer.gemm import svdquant_sm120_cutlass as sc
m, n, k = 16384, 3072, 3072; rank = 128
c = build_case((m, n, k), seed=0, bias=True, rank=rank)
mod = sc.get_nvfp4_svdquant_sm120_module(rank)
row = 49
ws = torch.zeros(max(int(mod.nvfp4_svdquant_gemm_workspace_size(m, n, k, row)), 1 << 20), dtype=torch.uint8, device="cuda")
out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
lt = sc._pack_sm120_kmajor_l2t(c.l2t)
xq = torch.empty((m, k // 2), dtype=torch.uint8, device="cuda")
xsf = torch.empty((sc._swizzled_sf_size(m, k // 16),), dtype=torch.uint8, device="cuda")
down = torch.empty((m, rank), dtype=torch.bfloat16, device="cuda")
for cap in (None, "55", "20"):
    for key in ("SVDQ_GEMM_SM_CAP", "SVDQ_PREFIX_SM_CAP"):
        if cap is None: os.environ.pop(key, None)
        else: os.environ[key] = cap
    torch.cuda.nvtx.range_push(f"cap={cap}")
    mod.nvfp4_svdquant_gemm(c.xq, c.wq, c.x_sf, c.w_sf, c.alpha, c.down, c.l1, c.bias, out, ws, row, True)
    mod.nvfp4_quantize_smooth_lora_down_dyn_sm120(c.x, c.pqs, c.global_sf, lt, xq, xsf, down, 5, 160, 64, 205, 0)
    torch.cuda.synchronize(); torch.cuda.nvtx.range_pop()
print("done")
