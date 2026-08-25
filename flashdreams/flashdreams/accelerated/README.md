<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# `flashdreams.accelerated`

`flashdreams.accelerated` is a low-level acceleration library used by
FlashDreams to build high-performance modules for streaming video models. It
currently contains two components:

- Quantization Toolkit
- Optimized Multi Head Attention

In the future, `flashdreams.accelerated` should be refactored into
`flashdreams.core` alongside CUDA graph capture, context parallelism,
disaggregated execution, additional low-level optimized kernels such as the
optimized multi-head attention implementation, and a possible autotuning
system. These building blocks will help FlashDreams achieve speed-of-light
performance across different platforms.

## Quantization Toolkit

The toolkit currently has two features:

- **Accelerated quantizer for tensor quantize and dequantize**
- **Quantized linear layer**

### Accelerated Quantizer

`quantize` quantizes a tensor to a specified dtype and scale granularity.

#### Target Dtype

The supported target dtypes/formats are `torch.float8_e4m3fn`,
`torch.float8_e5m2`, and `torch.int8`.

For example, consider the floating-point vector

$$
x = [0.42, 0.37, -0.67].
$$

For INT8, $M_{\mathrm{int8}} = 127$. Using one scale and rounding the scaled
values to the nearest integers gives

$$
s = \frac{0.67}{127} \approx 0.005276, \qquad
\bar{x} = \operatorname{round}(x / s) = [80, 70, -127].
$$

The quantized vector is therefore $[80, 70, -127]$ with scale
$s \approx 0.005276$.

#### Scale Granularity

For an input matrix $X \in \mathbb{R}^{L \times D}$ and a target dtype $t$,
let $M_t$ be the largest finite positive value that $t$ can represent and let
$\epsilon$ be a small positive scale floor.

Per-tensor granularity computes one scale for the complete matrix:

$$
s_{\text{tensor}} =
\max\left(\frac{\max_{i,j}|X_{ij}|}{M_t}, \epsilon\right).
$$

Every element is divided by $s_{\text{tensor}}$ before it is clipped to
$[-M_t, M_t]$ and converted to $t$.

Per-slice granularity computes a scale for every slice along the selected
axis. For a matrix, `axis=0` reduces across rows and produces one scale per
column, while `axis=1` reduces across columns and produces one scale per row:

$$
s_j^{(\text{axis}=0)} =
\max\left(\frac{\max_i|X_{ij}|}{M_t}, \epsilon\right), \qquad
s_i^{(\text{axis}=1)} =
\max\left(\frac{\max_j|X_{ij}|}{M_t}, \epsilon\right).
$$

Thus, `axis=0` divides each $X_{ij}$ by its column scale $s_j$, and `axis=1`
divides it by its row scale $s_i$.

Using the same vector as the first column of a $(3, 2)$ matrix and adding a
second column gives

$$
X =
\begin{bmatrix}
0.42 & 0.12 \\
0.37 & -0.91 \\
-0.67 & 0.55
\end{bmatrix}.
$$

**Per tensor.** One scale covers all six elements:

$$
s_{\mathrm{tensor}} = \frac{0.91}{127} \approx 0.007165, \qquad
\bar{X}_{\mathrm{tensor}} =
\begin{bmatrix}
59 & 17 \\
52 & -127 \\
-94 & 77
\end{bmatrix}.
$$

**Per slice with `axis=0`.** Reducing across the three rows produces one scale
for each of the two columns:

$$
s^{(\mathrm{axis}=0)} =
\begin{bmatrix}0.67 / 127 & 0.91 / 127\end{bmatrix}
\approx
\begin{bmatrix}0.005276 & 0.007165\end{bmatrix}, \qquad
\bar{X}^{(\mathrm{axis}=0)} =
\begin{bmatrix}
80 & 17 \\
70 & -127 \\
-127 & 77
\end{bmatrix}.
$$

**Per slice with `axis=1`.** Reducing across the two columns produces one scale
for each of the three rows:

$$
s^{(\mathrm{axis}=1)} =
\begin{bmatrix}
0.42 / 127 \\
0.91 / 127 \\
0.67 / 127
\end{bmatrix}
\approx
\begin{bmatrix}
0.003307 \\
0.007165 \\
0.005276
\end{bmatrix}, \qquad
\bar{X}^{(\mathrm{axis}=1)} =
\begin{bmatrix}
127 & 36 \\
52 & -127 \\
-127 & 104
\end{bmatrix}.
$$

> **Why retain the scale for FP8 quantization?** Although an FP16 or FP32
> tensor can be cast directly to FP8, a direct cast does not adapt FP8's
> limited representable range to the tensor's magnitude. Dividing by the
> scale before conversion maps the tensor into $[-M_t, M_t]$ and uses more of
> the available FP8 range, preserving more precision. The scale must be kept
> to recover the original magnitude during dequantization or to incorporate
> it into a subsequent operation. This is especially important for quantized
> algorithms such as SageAttention, whose accuracy depends on applying the
> quantization scales correctly.

> **Planned: per-tile quantization.** SageAttention also uses per-tile
> quantization, a powerful scheme that computes scales over individual tensor
> tiles. The Quantization Toolkit does not currently support this granularity;
> it is one of its most important missing features and should be planned for a
> future version.

CUDA tensors use the Triton implementation by default; CPU tensors use the
Torch implementation. The quantizer performs scale computation in detached
FP32 values. The returned scale tensor is FP32 and retains all input
dimensions, with reduced dimensions kept at size one.

`dequantize` casts $\bar X$ to the first scale's dtype and applies every
provided scale in order with normal tensor broadcasting:

$$
X = \operatorname{dequantize}\left(\bar X, s^{(1)}, \ldots, s^{(m)}\right)
\approx \bar X \odot \prod_{r=1}^{m}s^{(r)}.
$$

For example, the INT8 vector and scale from the earlier example are

$$
\bar{x} = [80, 70, -127], \qquad
s = \frac{0.67}{127} \approx 0.005276.
$$

Dequantization multiplies every quantized value by the scale:

$$
\hat{x} = \bar{x}s
= [80s, 70s, -127s]
\approx [0.4220, 0.3693, -0.6700].
$$

The result approximates the original vector $[0.42, 0.37, -0.67]$. The small
difference in the first two values comes from rounding during INT8
quantization.

It then casts the result to the requested output dtype (FP16 by default). With
no scales, it directly casts $\bar X$ to that dtype. This supports, for
example, separately applying activation and weight scales after a GEMM.

For a quantized-GEMM example, let $Q \in \mathbb{R}^{L \times D}$ and
$K \in \mathbb{R}^{S \times D}$ be token-major query and key matrices,
with $L$ query tokens, $S$ key tokens, and feature width $D$. Their
full-precision score matrix is $QK^\mathsf{T} \in \mathbb{R}^{L \times S}$.
With tensorwise quantization, scalar scales $s_Q$ and $s_K$, and quantized
matrices $\bar Q$ and $\bar K$, the scaled product is

$$
QK^\mathsf{T} \approx (s_Qs_K)\,(\bar Q\bar K^\mathsf{T}).
$$

With `Granularity.SLICE` and `axis=-1`, each token instead has a scale:
$s_Q \in \mathbb{R}^{L \times 1}$ and
$s_K \in \mathbb{R}^{S \times 1}$. The corresponding per-token result is

$$
QK^\mathsf{T} \approx
\left(s_Qs_K^\mathsf{T}\right) \odot
\left(\bar Q\bar K^\mathsf{T}\right),
$$

where $s_Qs_K^\mathsf{T} \in \mathbb{R}^{L \times S}$ is the outer product
of the query and key token scales and $\odot$ is elementwise multiplication.
These equations describe composing the tensor quantizer with a GEMM; the
quantizer itself does not perform attention or provide a fused QK kernel.

> **Inner-dimension rule for slice-quantized GEMM.** For
> $C = AB$ with $A \in \mathbb{R}^{M \times K}$ and
> $B \in \mathbb{R}^{K \times N}$, the shared inner dimension $K$ must be the
> quantization axis: use `axis=1` for $A$ and `axis=0` for $B$. This produces
> row scales $s_{A,i}$ and column scales $s_{B,j}$ that remain constant across
> each dot product, allowing
> $C_{ij} \approx s_{A,i}s_{B,j}\sum_k \bar A_{ik}\bar B_{kj}$. If either
> scale varied with $k$, it would have to stay inside the sum; a single scale
> applied after GEMM could not dequantize the accumulator correctly. For
> $QK^\mathsf{T}$, both $Q$ and the untransposed $K$ store their shared feature
> dimension on `axis=1`, so both are quantized with `axis=1`. For token-major
> $Q$ and $K$, this produces one scale per token and is referred to as
> per-token quantization in the
> [SageAttention paper](https://arxiv.org/abs/2410.02367).
> Transposing $K$ then moves
> that dimension to `axis=0` of the GEMM's right operand, satisfying the same
> rule.

#### Worked Slice-Quantized GEMM

This example quantizes $A$ per row and $B$ per column, accumulates their GEMM
in INT32, and dequantizes the result with both scale tensors:

```python
import torch

from flashdreams.accelerated.quantization.quantizer import (
    Granularity,
    dequantize,
    quantize,
)

a = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# a: shape=(3, 2), dtype=torch.float32
b = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
# b: shape=(2, 3), dtype=torch.float32

quantized_a, scale_a = quantize(
    a, torch.int8, Granularity.SLICE, axis=1
)
# quantized_a: shape=(3, 2), dtype=torch.int8
# scale_a: shape=(3, 1), dtype=torch.float32
quantized_b, scale_b = quantize(
    b, torch.int8, Granularity.SLICE, axis=0
)
# quantized_b: shape=(2, 3), dtype=torch.int8
# scale_b: shape=(1, 3), dtype=torch.float32

# Accumulate in INT32 so the INT8 products do not overflow.
quantized_c = torch._int_mm(quantized_a, quantized_b)
# quantized_c: shape=(3, 3), dtype=torch.int32
c = dequantize(
    quantized_c, scale_a, scale_b, dtype=torch.float32
)
# c: shape=(3, 3), dtype=torch.float32
reference_c = a @ b
# reference_c: shape=(3, 3), dtype=torch.float32

assert quantized_a.tolist() == [[64, 127], [95, 127], [106, 127]]
assert quantized_b.tolist() == [[89, 92, 95], [127, 127, 127]]
assert quantized_c.tolist() == [
    [21825, 22017, 22209],
    [24584, 24869, 25154],
    [25563, 25881, 26199],
]
assert scale_a.shape == (3, 1)  # One scale per row of A.
assert scale_b.shape == (1, 3)  # One scale per column of B.

torch.testing.assert_close(c, reference_c, rtol=0, atol=0.2)
print(c)
# tensor([[ 27.0631,  30.0312,  33.0471],
#         [ 60.9684,  67.8428,  74.8585],
#         [ 95.0946, 105.9053, 116.9526]])
```

Passing `scale_a` and `scale_b` separately to `dequantize` applies them with
normal broadcasting, combining each row scale from $A$ with each column scale
from $B$ without materializing their outer product.

### Quantized Linear Layer

`QuantizedNonPersistentLinear` constructs an inference projection from existing
weight $W \in \mathbb{R}^{O \times I}$ and optional bias
$b \in \mathbb{R}^{O}$. It keeps the derived quantized weight, bias, and
FP32 weight scale as nonpersistent buffers, so callers rebuild them from
checkpoint tensors rather than loading them from `state_dict`.

`WeightGranularity.PER_OUT_CHANNEL` quantizes each weight row with
`Granularity.SLICE` and `axis=-1`, giving a weight scale shaped $[O, 1]$.
`WeightGranularity.TENSOR` gives one $[1, 1]$ scale for all weights. The
supported weight granularities are deliberately limited to per-output-channel
and per-tensor quantization by the inner-dimension rule described above. In
$XW^\mathsf{T}$, the input dimension $I$ is the GEMM reduction dimension.
Per-output-channel scales and a tensorwise scale remain constant across $I$
and can therefore be applied after GEMM. A per-input-channel scale would vary
across $I$, so it would have to remain inside the reduction and could not be
applied as a single post-GEMM scale. The layer accepts the same quantized
dtypes as the quantizer.

For activations $X \in \mathbb{R}^{\ldots \times I}$, inference has two
paths. Passing a `Granularity` dynamically quantizes $X$ with `axis=-1`.
Passing prequantized $\bar X$ instead requires the layer's activation dtype
and an FP32 tensorwise scale shaped $[1, \ldots, 1]$ or slice scale shaped
$[\ldots, 1]$. After flattening leading dimensions into GEMM rows, both
paths compute the scaled projection

$$
Y \approx (\bar X\bar W^\mathsf{T}) \odot
\left(S_XS_W^\mathsf{T}\right) + b,
$$

where $S_X$ broadcasts down output rows and $S_W^\mathsf{T}$ broadcasts
across output columns. `int8` uses an integer GEMM followed by application of
the activation and weight scales; FP8 uses scaled GEMM with those same scales.
The result is returned as `out_dtype` (FP16 by default), including after any
internal BF16 FP8 rowwise GEMM result is cast, and the optional bias is applied
in that output dtype.

#### Quantized Forward Example

The layer can quantize a full-precision input during `forward`, or reuse an
input that was quantized beforehand:

```python
import torch

from flashdreams.accelerated.quantization.linear import (
    QuantizedNonPersistentLinear,
    WeightGranularity,
)
from flashdreams.accelerated.quantization.quantizer import Granularity, quantize

weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
layer = QuantizedNonPersistentLinear(
    weight,
    bias=None,
    granularity=WeightGranularity.PER_OUT_CHANNEL,
    dtype=torch.int8,
)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Quantize full-precision x inside the forward call.
dynamic_output = layer(x, Granularity.SLICE, out_dtype=torch.float32)
# dynamic_output: shape=(2, 3), dtype=torch.float32

# Quantize x once, then reuse it for the prequantized forward path.
quantized_x, x_scale = quantize(
    x, layer.dtype, Granularity.SLICE, axis=-1
)
# quantized_x: shape=(2, 2), dtype=torch.int8
# x_scale: shape=(2, 1), dtype=torch.float32
prequantized_output = layer(
    quantized_x, x_scale, out_dtype=torch.float32
)
# prequantized_output: shape=(2, 3), dtype=torch.float32

torch.testing.assert_close(dynamic_output, prequantized_output, rtol=0, atol=0)
```

The prequantized path is useful when several projections consume the same
input. For example, the Q, K, and V projections in self-attention all consume
the same $x$, so its quantized tensor and scale can be shared across those
forward calls.

## Optimized Multi Head Attention

### Generic Multi Head Attention Interface

`AttentionConfig` defines the query width, optional context width, number of
heads, head dimension, Q/K normalization scope and epsilon, and optional
`RoPEConfig`. A missing context width uses the query width; self-attention
requires those widths to be equal. `RoPEConfig` selects interleaved or
split-half feature pairing and whether key rotations occur before or after K/V
cache storage.

`MultiHeadAttention` is an adapter-friendly abstract interface. Concrete
modules expose query, key, value, and output projection accessors plus query
and key normalization accessors, while retaining checkpoint-native attribute
names. `compute_kv(context, rope_freqs)` materializes a static cross-attention
cache. `forward(x, kv_cache, rope_freqs)` accepts query tokens and either a
prepared streaming cache or that static cache.

The caller owns the `BlockKVCache` lifecycle for streaming self-attention:
`before_update(chunk_idx)`, `forward`, then `after_update(chunk_idx)`. The
current query chunk writes its K/V into the rolling cache before it is queried.
Cross-attention instead calls `compute_kv` once for static context and reuses
the finalized cache without rolling-cache update bookkeeping.

### Multi Head Attention Definition

Let query tokens be $X \in \mathbb{R}^{B \times L \times C_Q}$ and
context tokens be $C \in \mathbb{R}^{B \times S \times C_K}$, where
$B$ is the flattened leading batch/group size, $L$ the query length,
$S$ the context length, $H$ the number of heads, and $d$ the head
dimension. With inner width $Hd$, the projections are

$$
Q = \operatorname{reshape}(XW_Q + b_Q), \qquad
K = \operatorname{reshape}(CW_K + b_K), \qquad
V = \operatorname{reshape}(CW_V + b_V),
$$

where each reshaped tensor has shape $[B, L, H, d]$ for $Q$ or
$[B, S, H, d]$ for $K$ and $V$. For head $h$, scaled dot-product
attention is

$$
A_h = \operatorname{softmax}\left(\frac{Q_hK_h^\mathsf{T}}{\sqrt d}\right)V_h,
\qquad
Y = \operatorname{concat}(A_1, \ldots, A_H)W_O + b_O.
$$

Optional Q/K normalization acts on projected Q and K, never V. RoPE rotates
projected Q and K according to its configured pairing; implementations own the
operation order, while the reference below normalizes before applying RoPE.
In self-attention, $C = X$ and the current K/V chunk is written into a
rolling cache before the score calculation. In cross-attention, K/V are
precomputed from static $C$ and reused for each query.

### TorchMultiHeadAttention Reference

`TorchMultiHeadAttention` is the portable PyTorch implementation of that
contract. It projects Q, K, and V, flattens leading dimensions into $B$,
and reshapes to token-major $[B, L, H, d]$ or $[B, S, H, d]$. It applies
the configured Q/K normalization after projection: `QKNormScope.INNER` sees
the concatenated $Hd$ features and `HEAD` sees each $d$-feature head.
Values are not normalized.

`allocate_kv_cache` creates a rolling cache shaped
`[B, sink_size + window_size, H, d]`. For before-cache RoPE, the reference
rotates normalized keys before storing them in either a static cache or the
current rolling chunk. For after-cache RoPE, it stores normalized unrotated
keys and rotates the visible cached keys when querying them. It rotates
normalized queries before attention in both policies, selecting the appropriate
current-query and visible-cache positions from `rope_freqs`.

For self-attention, `forward` writes the projected current K/V after the
caller has prepared the cache, then queries all visible cache entries. For
cross-attention, `compute_kv` creates a finalized static cache and `forward`
only queries it. The reference transposes to head-major layout for PyTorch
`scaled_dot_product_attention` with zero dropout and non-causal attention,
then restores token-major layout, concatenates heads, applies the output
projection, and restores the original leading query shape.

### OptimizedMultiHeadAttention

`OptimizedMultiHeadAttention` is the inference-only base for adapters that
provide their own projection and normalization modules. Its
`OptimizedImplConfig` selects:

- `QKVFusionOption`: independent projections, a fused KV projection, or a
  fully fused QKV projection (the full form requires equal query and context
  widths).
- `SDPABackend.CUDNN`: PyTorch cuDNN SDPA for FP16/BF16, or the cuDNN Frontend
  FP8 path when quantized SDPA is enabled.
- `SDPABackend.FA2`: Triton non-causal FlashAttention2. When `use_tma=True`,
  the optimized module uses the TMA kernel only when the tensors satisfy its
  support check; otherwise it uses the pointer-based kernel.
- `QuantizationOption`: optional FP8 or INT8 Q/K/V projection quantization and
  optional unscaled FP8 e4m3 SDPA.

Optimized attention accepts CUDA FP16/BF16 inputs and requires compute
capability 9.0 or newer. The FlashAttention2 kernels use token-major
`[B, L, H, D]` queries and `[B, S, H, D]` keys and values; head dimensions
must be powers of two from 16 through 256. TMA additionally requires
compatible tensor-descriptor layout and alignment. The native cuDNN FP8 path
requires the `nvidia-cudnn-frontend` package; warm it up once before CUDA graph
capture.

Quantized SDPA directly casts Q, K, and V to FP8 e4m3 and is not an
accuracy-preserving quantization scheme. It can make attention inaccurate, so
validate output quality for the model and workload that use it.

## OmniDreams Integration

OmniDreams adapts `OptimizedMultiHeadAttention` for its transformer blocks and
selects the optimized path independently for self- and cross-attention. Its
installed runner presets are `omnidreams-optimized-gb300` and
`omnidreams-optimized-rtx-pro-6000`; inspect either resolved configuration
without loading a model with:

```bash
uv run flashdreams-run --no-instantiate omnidreams-optimized-gb300
uv run flashdreams-run --no-instantiate omnidreams-optimized-rtx-pro-6000
```

Those presets encode device-specific policies selected by the OmniDreams
attention benchmarks. They are configuration choices, not portable performance
claims for other models, shapes, or hardware.

## Validation and Benchmarks

The accelerated tests cover quantization and projection contracts, reference
agreement for attention implementations, and the two FlashAttention2 kernels.
GPU benchmarks compare attention policies and distinguish prequantized work
from end-to-end quantization where applicable. They are manual tests and do not
publish a single benchmark result in this library; run them on the target CUDA
environment when evaluating a configuration.
