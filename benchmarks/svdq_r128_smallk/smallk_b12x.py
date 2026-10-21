"""Small-K bound: the CuTeDSL b12x SM120 NVFP4 GEMM (plain, no LoRA/bias) with and without its
per-tile output epilogue (ablate_epilogue=True writes nothing), vs best native CUTLASS NVFP4 GEMM,
across K at fixed M, N. Cold-L2 graph replay, interleaved rounds."""
import json, statistics, sys
import torch
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
sys.path.insert(0, "benchmarks")
from svdquant_sm120_benchmark import build_case
from svdquant_sm120_native import native_operands, native_call, _cold_graph_us
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import Sm120B12xBlockScaledDenseGemmKernel
from flashinfer.cute_dsl.utils import get_max_active_clusters
M, N = (int(v) for v in sys.argv[1].split("x")); Ks = [int(v) for v in sys.argv[2].split(",")]
TILE = tuple(int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "128x128").split("x"))
PEAK = 1069e12
mac = get_max_active_clusters(1)

def compile_b12x(sf_m, sf_n, sf_k, ablate):
    gemm = Sm120B12xBlockScaledDenseGemmKernel(16, TILE, (1, 1), tile_k=128, use_prefetch=False,
                                               enable_pdl=False, swap_ab=False, ablate_epilogue=ablate)
    sm, sk, sn = cute.sym_int(), cute.sym_int(), cute.sym_int()
    a = cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (sm, sk), stride_order=(1, 0), assumed_align=32)
    b = cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (sn, sk), stride_order=(1, 0), assumed_align=32)
    c = cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (sm, sn), stride_order=(1, 0), assumed_align=16)
    asf = make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, 16)
    bsf = make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, 16)
    al = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
    st = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(gemm.wrapper, a, b, c, sf_m, sf_n, sf_k, 1, asf, bsf, al, mac, st, False,
                        options="--opt-level 2 --enable-tvm-ffi")

for K in Ks:
    case = build_case((M, N, K), seed=0, bias=False, rank=32)
    sf_m, sf_n, sf_k = (M + 127) // 128, (N + 127) // 128, (K // 16 + 3) // 4
    out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    runs = {}
    for name, abl in (("b12x", False), ("b12x no-epilogue", True)):
        fn = compile_b12x(sf_m, sf_n, sf_k, abl)
        runs[name] = lambda fn=fn: fn(case.xq, case.wq, out, sf_m, sf_n, sf_k, case.x_sf.data_ptr(), case.w_sf.data_ptr(), case.alpha)
    runs["b12x"](); torch.cuda.synchronize()
    ref = out.clone()
    runs["native-cutlass"], _ = native_call("native-cutlass", native_operands(build_case((M, N, K), seed=0, bias=True, rank=32)), quantize=False)
    res = {nm: [] for nm in runs}
    for _ in range(3):
        for nm, f in runs.items():
            res[nm].append(_cold_graph_us(f))
    fl = 2 * M * N * K
    print(json.dumps({"MxN": [M, N], "K": K, "tile": TILE, "fp4_sol_us": round(fl / PEAK * 1e6, 1),
                      "r": {nm: {"us": round(statistics.median(v), 1), "pct_peak": round(100 * fl / (statistics.median(v) * 1e-6) / PEAK, 1)} for nm, v in res.items()},
                      "b12x_out_abs_sum": float(ref.float().abs().sum())}), flush=True)
