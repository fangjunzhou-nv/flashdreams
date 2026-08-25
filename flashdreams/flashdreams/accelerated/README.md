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

`flashdreams.accelerated` contains CUDA-oriented inference building blocks for
FlashDreams integrations. It has two components:

- Quantization Toolkit
- Optimized Multi Head Attention

## Quantization Toolkit

The toolkit currently has two features:

- **Accelerated quantizer for tensor quantize and dequantize**
- **Quantized linear layer**

### Accelerated Quantizer

`quantize` configures quantization with scale granularity.

For an input $X \in \mathbb{R}^{L \times D}$, `Granularity.TENSOR` uses one
scalar scale shared by the full tensor. `Granularity.SLICE` with `axis=-1`
uses a scale tensor shaped $[L, 1]$, one scale per token row, broadcasting
across the $D$ features.

The supported target dtypes/formats are `torch.float8_e4m3fn`,
`torch.float8_e5m2`, and `torch.int8`. CUDA tensors use the Triton
implementation by default; CPU tensors use the Torch implementation. For a
target format $f$ with largest finite positive value $M_f$, the quantizer
converts $X$ to detached FP32 values and computes
$s = \max(\max |X| / M_f, \epsilon)$ over the selected scope, where $\epsilon$
is a small positive scale floor. It clips $X / s$ to $[-M_f, M_f]$ before
casting. The returned scale tensor is FP32 and retains all input dimensions,
with reduced dimensions kept at size one.

`dequantize` casts $\bar X$ to the first scale's dtype and applies every
provided scale in order with normal tensor broadcasting:

$$
X = \operatorname{dequantize}\left(\bar X, s^{(1)}, \ldots, s^{(m)}\right)
\approx \bar X \odot \prod_{r=1}^{m}s^{(r)}.
$$

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

### Quantized Linear Layer

`QuantizedNonPersistentLinear` constructs an inference projection from existing
weight $W \in \mathbb{R}^{O \times I}$ and optional bias
$b \in \mathbb{R}^{O}$. It keeps the derived quantized weight, bias, and
FP32 weight scale as nonpersistent buffers, so callers rebuild them from
checkpoint tensors rather than loading them from `state_dict`.

`WeightGranularity.PER_OUT_CHANNEL` quantizes each weight row with
`Granularity.SLICE` and `axis=-1`, giving a weight scale shaped $[O, 1]$.
`WeightGranularity.TENSOR` gives one $[1, 1]$ scale for all weights. The
layer accepts the same quantized dtypes as the quantizer. When its activation
dtype is `torch.float8_e5m2`, it stores weights as `torch.float8_e4m3fn` for
the CUDA FP8 GEMM operand pair; otherwise activation and weight formats match.

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
