# H3 upscale and adapter operators

These experimental, inference-only operators keep BF16 3D activations in
`torch.channels_last_3d` format. They avoid the layout conversions and grouped
convolution dispatch used by native GroupNorm and temporal depthwise Conv3D on
Spark. The public API and both backends are experimental.

```python
import torch
from flashinfer import autotune
from flashinfer.diffusion_ops import h3_bridge

# Logical tensor order is N,C,T,H,W; storage order is N,T,H,W,C.
x = torch.randn(1, 512, 20, 48, 84, device="cuda", dtype=torch.bfloat16)
x = x.contiguous(memory_format=torch.channels_last_3d)
w = torch.randn(512, 1, 5, 1, 1, device="cuda", dtype=torch.bfloat16)
b = torch.randn(512, device="cuda", dtype=torch.bfloat16)
# Warm up and tune outside CUDA graph capture; persist the selected tactic.
with autotune(cache="h3-dw-tactics.json"):
    y = h3_bridge.depthwise_temporal(x, w, b)

# Subsequent calls reuse the winner without profiling.
y = h3_bridge.depthwise_temporal(x, w, b)
```

Calling any public function opts into the experimental API and warns once.
No environment flag is required. These functions do not enter stable API
routing, and importing their public module does not load the H3 backends.

## Operations and numerical contract

- `group_norm_silu(x, weight, bias, *, groups, eps, scale=None, shift=None)`:
  normalize over **all** `C/groups * T * H * W` elements per sample/group,
  apply learned affine parameters, optionally modulate, then apply SiLU.
  FP32 partial mean/M2 reductions use a different reduction tree from PyTorch;
  results need not be bitwise equal. Scale and shift must both be supplied and
  are channel vectors shared by the batch. This is not a frame-wise norm.
- `modulate_silu(x, scale, shift)`: apply `(1 + scale) * x + shift`, then SiLU,
  preserving BF16 rounding after each original pointwise operation.
- `silu_multiply(value, gate)`: multiply by BF16-rounded `silu(gate)`;
  accepts strided channel chunks produced by the adapter's split projection.
- `add(left, right)` and `silu(x)`: produce channels-last BF16 output.
- `add_bias_residual(conv_no_bias, bias, residual)`: compute
  `bf16(bf16(conv_no_bias + bias) + residual)` in one pass. Convolution and
  residual inputs must have equal positive NCTHW shapes in contiguous
  channels-last storage; bias is a contiguous `[C]` vector.
- `bias_silu_multiply(conv_no_bias, bias)`: add `[2D]` channel bias to a
  channels-last `[N,2D,T,H,W]` projection, rounding to BF16, then split its
  channels into value/gate halves. Round `silu(gate)` to BF16 before multiplying
  by value, returning channels-last `[N,D,T,H,W]`. Odd D and multiple batches
  are supported. Both bias consumers require an already BF16-rounded
  convolution output, preserve storage offsets, and use the current stream;
  they are not replacements for an FP32-accumulator convolution epilogue.
- `depthwise_temporal(x, weight, bias, *, bias_first=True, backend="auto")`:
  temporal 3- or 5-tap cross-correlation, spatial kernel `1x1`, stride/dilation
  one, same zero padding, one filter per channel. FP32 FMA visits ascending
  taps; `bias_first` selects initial-bias versus final-bias accumulation.
  The CuTe specialization preserves the Triton accumulation order.

Every operand must be a CUDA BF16 tensor on the same device with
`requires_grad=False`; no autograd implementation is provided. GroupNorm and
DW accept ordinary contiguous NCTHW or channels-last input and always return
channels-last output. GroupNorm/DW parameters must be contiguous channel
vectors/filters. No model parameters, normalization domain, temporal chunks,
overlap blending, or interpolation rules are changed by these operators.

## DW shape coverage and autotuning

There is no two-shape whitelist. The CuTe kernel supports arbitrary positive
`N,C,T,H,W` in channels-last BF16 storage, `K=3` or `5`, optional bias, and
both bias accumulation orders on SM121. Odd channel counts use scalar vectors;
vectorized tactics are admitted only when their width divides the channel
count. Temporal zero padding stays within each batch item. The
Triton path also accepts ordinary contiguous NCTHW storage.

Use FlashInfer's existing `autotune()` context to search before serving:

- `backend="auto"` compares Triton block/warp configurations and eligible
  CuTe time-tile/thread/vector configurations. Candidates are compiled and
  checked against the Triton FP32-FMA reference before timed profiling.
- `backend="triton"` and `backend="cute"` search only their named backend.
  Explicit CuTe requires SM121, channels-last storage, and its dependencies.
- Outside `autotune()`, the API reads the selected tactic from cache without
  profiling. An unseen `auto` or `triton` configuration uses Triton; explicit
  `cute` uses its generic default configuration. Thus `auto` alone does not
  mean that a timing search occurs during the first ordinary inference call.
- Tuning uses exact static shapes and real tensor layouts. Cache identity
  distinguishes shapes, dtypes, strides, alignment classes, GPU identity,
  bias presence/order, requested backend and implementation/dependency state.
  A winner for one configuration is not reused as a shape bucket for another.
- `autotune(cache="...")` saves selections using the existing FlashInfer
  cache. In a new process, load with `autotune(False, cache="...")`, warm the
  selected specialization, then capture or serve. Compilation artifacts and
  autotune selections are separate caches.

Tuning uses CUDA graphs with repeated, warm-cache inputs. This preserves the
actual tensor strides and pointer alignment throughout profiling. Compilation
and reference checks are outside the measured interval. Tune on representative
workloads and verify application latency separately; this is not a cold-memory
bandwidth measurement. Do not start a new tuning search during an outer CUDA
graph capture; warm cache hits are supported. Inputs and weights are read-only,
outputs are separately allocated, and kernels use the current stream with fresh
pointers on each call.

The general kernels require Triton. The qualified Spark environment uses Torch
2.14.1+cu132, Triton 3.8 and `nvidia-cutlass-dsl` 4.8.1. Missing optional CuTe
dependencies leave auto with Triton candidates. Compilation and correctness
failures during candidate preparation are surfaced. No torch.compile/fullgraph
integration is promised.

## Validation and application integration

From the FlashInfer checkout, run:

```bash
pytest tests/experimental/test_h3_bridge_api.py \
       tests/experimental/test_h3_bridge_depthwise.py \
       tests/experimental/test_h3_bridge_fused.py \
       tests/experimental/test_h3_bridge_depthwise_generic.py \
       tests/experimental/test_h3_bridge_autotune.py \
       tests/experimental/test_h3_bridge_autotune_cache.py \
       tests/experimental/test_h3_bridge_bias_epilogue.py \
       tests/experimental/test_h3_bridge_bias_epilogue_api.py \
       tests/experimental/test_h3_bridge_bias_epilogue_containment.py
python benchmarks/experimental/bench_h3_bridge.py
```

The benchmark reports separate operator times, selected DW tactics and first-call
tuning time, and checks the outputs. It does not estimate a full video-generation
speedup. Application code should replace its corresponding bridge operators
with this API while retaining its model
forward structure, CL Conv3D weights, original temporal chunking and final
output conversion. Keep dense Conv3D on its existing cuDNN path. To use a
bias consumer, compute that convolution with `bias=None`, then pass its original
bias vector to `add_bias_residual` or `bias_silu_multiply`; qualify the full
application output because convolution algorithm selection can depend on the
surrounding integration. Warm both epilogue specializations before graph capture.


The separate bias-consumer benchmark compares complete Conv3D + bias + consumer
paths with the bias-free convolution + fused consumer. With this checkout
installed in a CUDA environment containing Torch, Triton and Typer, run:

```bash
# Default: adapter_expand only, random BF16 channels-last operands.
python benchmarks/experimental/bench_h3_bridge_bias_epilogue.py
python benchmarks/experimental/bench_h3_bridge_bias_epilogue.py --case up_high42 --repeats 40
python benchmarks/experimental/bench_h3_bridge_bias_epilogue.py --all --repeats 40
# Optional captured operands: CASE.pt with BF16 input, weight and bias tensors.
python benchmarks/experimental/bench_h3_bridge_bias_epilogue.py --all --operands-dir /path/to/operands
```

Cases are `up_high42`, `up_high20`, `up_low42`, `up_low20`, `adapter_expand`
and `adapter_project`. Residuals are synthetic even with captured operands.
The script checks convolution-bias rounding and final outputs bitwise, warms
both paths, then shuffles their order within paired CUDA-event rounds. It prints
one JSON report with hardware/software details, shapes, median/min/IQR latency
and the ratio of baseline to fused median. It does not write result files.
These are warm-input operator timings, not an end-to-end bridge speedup; a
runtime whose convolution bias changes the rounding fails the correctness check.

Owner: `georgeliu95`. This feature is maintained on the `h3-spark-unified`
application branch. Before upstream admission, open the experimental tracking
issue and qualify additional inputs and supported software versions. Review
by 2026-12-07 for graduation or continued incubation; graduation requires a
finalized API, trace templates, stable tests, and a nominated upstream release.
