# H3 upscale and adapter operators

These experimental, inference-only operators keep BF16 3D activations in
`torch.channels_last_3d` format. They avoid the layout conversions and grouped
convolution dispatch used by native GroupNorm and temporal depthwise Conv3D on
Spark. The public API and both backends are experimental.

```python
import torch
from flashinfer.diffusion_ops import h3_bridge

# Logical tensor order is N,C,T,H,W; storage order is N,T,H,W,C.
x = torch.randn(1, 512, 20, 48, 84, device="cuda", dtype=torch.bfloat16)
x = x.contiguous(memory_format=torch.channels_last_3d)
w = torch.randn(512, 1, 5, 1, 1, device="cuda", dtype=torch.bfloat16)
b = torch.randn(512, device="cuda", dtype=torch.bfloat16)
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

## DW dispatch and dependencies

`backend="triton"` selects the general implementation. `backend="cute"`
requires the qualified CuTe specialization; unsupported inputs fail explicitly.
`backend="auto"` uses CuTe only for GB10 (SM121), BF16 channels-last inputs
`[1,512,20,48,84]` or `[1,512,42,48,84]`, a contiguous `[512,1,5,1,1]` filter,
and bias-first accumulation with a contiguous 512-channel bias. Other supported
inputs use Triton. An unavailable optional CuTe dependency also leaves auto on
Triton; kernel compilation or execution failures are surfaced.

The general kernels require Triton. The qualified Spark environment uses Torch
2.14.1+cu132, Triton 3.8 and `nvidia-cutlass-dsl` 4.8.1. Install dependencies in
the application environment; this feature does not replace global packages.
Warm up each specialization before CUDA graph capture. Compilation is cached;
input, weight, bias, output pointers and the current stream are refreshed at
execution. No torch.compile/fullgraph integration is promised.

## Validation and application integration

From the FlashInfer checkout, run:

```bash
pytest tests/experimental/test_h3_bridge_api.py \
       tests/experimental/test_h3_bridge_depthwise.py \
       tests/experimental/test_h3_bridge_fused.py
python benchmarks/experimental/bench_h3_bridge.py
```

The benchmark reports separate operator times and checks the outputs. It does
not estimate a full video-generation speedup. Application code should replace
its corresponding bridge operators with this API while retaining its model
forward structure, CL Conv3D weights, original temporal chunking and final
output conversion. Keep dense Conv3D on its existing cuDNN path.

Owner: `georgeliu95`. This feature is maintained on the `h3-spark-unified`
application branch. Before upstream admission, open the experimental tracking
issue and qualify additional inputs and supported software versions. Review
by 2026-12-07 for graduation or continued incubation; graduation requires a
finalized API, trace templates, stable tests, and a nominated upstream release.
