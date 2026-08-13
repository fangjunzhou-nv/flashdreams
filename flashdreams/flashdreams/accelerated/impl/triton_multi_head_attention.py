# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton implementation of inference-only streaming self-attention.

One GEMM produces Q/K/V, then a Triton kernel fuses Q/K RMSNorm, Q/K RoPE, and
the K/V-cache write. Processed Q attends through autotuned TMA FlashAttention2
before the output projection. An opt-in FP8 policy covers both projection
GEMMs, attention tensors, and cache storage while preserving BF16/FP16 module
inputs and outputs.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashdreams.accelerated.fp8_quantization import fp8_linear, quantize_fp8_weight
from flashdreams.accelerated.impl.triton import (
    flash_attention_2_tma,
    fused_rms_rope_kv_cache_update,
)
from flashdreams.accelerated.multi_head_attention import (
    MultiHeadAttention,
    QKNormScope,
)
from flashdreams.core.attention import BlockKVCache


def _cache_write_slice(kv_cache: BlockKVCache) -> tuple[int, int, int]:
    """Map the current input chunk onto the physical cache write interval.

    Returns:
        The source-token offset, destination-cache offset, and token count for
        the fused cache write.

    The cache normally consumes the complete current chunk. When a rolling
    write would overlap an immutable sink prefix, only the trailing source
    tokens that fit after the sink are copied.
    """
    write_start, write_end = kv_cache._current_write_bounds()
    read_start = 0
    if (
        kv_cache.sink_size > 0
        and not kv_cache._current_chunk_overlaps_sink()
        and write_start < kv_cache.sink_size
    ):
        # Preserve the sink and align the source tail with the write interval.
        # This mirrors BlockKVCache.update without staging processed K/V.
        write_start = kv_cache.sink_size
        write_length = write_end - write_start
        read_start = kv_cache.chunk_size - write_length
    return int(read_start), int(write_start), int(write_end - write_start)


class TritonMultiHeadAttention(MultiHeadAttention[BlockKVCache]):
    """Inference-only streaming self-attention backed by Triton kernels.

    Parameter names and shapes match :class:`TorchMultiHeadAttention`, so a
    reference module's state dict loads directly. Callers own the
    :class:`BlockKVCache` lifecycle and call ``before_update`` before this
    module and ``after_update`` after it for every chunk.
    """

    q_proj: nn.Linear
    """Query projection."""

    k_proj: nn.Linear
    """Key projection."""

    v_proj: nn.Linear
    """Value projection."""

    output_proj: nn.Linear
    """Output projection."""

    q_norm: nn.Module
    """Per-head or inner-width query RMS normalization."""

    k_norm: nn.Module
    """Per-head or inner-width key RMS normalization."""

    use_fp8: bool
    """Whether projection, attention, output, and cache storage use FP8."""

    _fused_qkv_weight: Tensor | None
    """Non-persistent native or packed fused QKV weight."""

    _fused_qkv_bias: Tensor | None
    """Non-persistent fused QKV bias."""

    _fused_qkv_weight_scale: Tensor | None
    """Non-persistent per-row FP8 QKV weight scales."""

    _output_weight_fp8: Tensor | None
    """Non-persistent packed output-projection weight."""

    _output_weight_scale: Tensor | None
    """Non-persistent per-row FP8 output weight scales."""

    def __init__(
        self,
        query_dim: int,
        n_heads: int = 8,
        head_dim: int = 64,
        *,
        qkv_bias: bool = False,
        output_bias: bool = False,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        qk_norm_scope: QKNormScope = QKNormScope.HEAD,
        rope_interleaved: bool = False,
        use_fp8: bool = False,
    ) -> None:
        """Initialize projections and fused attention policies.

        Args:
            query_dim: Feature dimension of input and output tokens.
            n_heads: Number of query, key, and value heads.
            head_dim: Feature dimension of each attention head.
            qkv_bias: Add bias to the query, key, and value projections.
            output_bias: Add bias to the output projection.
            qk_norm: Apply RMS normalization to projected queries and keys.
            qk_norm_eps: Epsilon used by Q/K RMS normalization.
            qk_norm_scope: Feature scope used by Q/K RMS normalization.
            rope_interleaved: Rotate adjacent feature pairs instead of half splits.
            use_fp8: Use row-scaled E4M3 projection/output GEMMs and E4M3
                attention/cache storage.

        Raises:
            ValueError: ``head_dim`` is unsupported by the TMA attention kernel.
        """
        super().__init__(
            query_dim=query_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            qk_norm_scope=qk_norm_scope,
            rope_interleaved=rope_interleaved,
        )
        if not (16 <= head_dim <= 256 and head_dim & (head_dim - 1) == 0):
            raise ValueError(
                "TMA FlashAttention2 requires a power-of-two head_dim in [16, 256]; "
                f"got {head_dim}"
            )
        if use_fp8 and query_dim % 16 != 0:
            raise ValueError("FP8 projections require query_dim to be a multiple of 16")

        self.use_fp8 = use_fp8

        self.q_proj = nn.Linear(self.query_dim, self.inner_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(self.query_dim, self.inner_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.query_dim, self.inner_dim, bias=qkv_bias)
        self.output_proj = nn.Linear(self.inner_dim, self.query_dim, bias=output_bias)
        self.register_buffer("_fused_qkv_weight", None, persistent=False)
        self.register_buffer("_fused_qkv_bias", None, persistent=False)
        self.register_buffer("_fused_qkv_weight_scale", None, persistent=False)
        self.register_buffer("_output_weight_fp8", None, persistent=False)
        self.register_buffer("_output_weight_scale", None, persistent=False)
        self._refresh_derived_weights()
        self.register_load_state_dict_post_hook(self._refresh_derived_weights)

        norm_dim = (
            self.head_dim if qk_norm_scope is QKNormScope.HEAD else self.inner_dim
        )
        self.q_norm = (
            nn.RMSNorm(norm_dim, eps=qk_norm_eps) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            nn.RMSNorm(norm_dim, eps=qk_norm_eps) if qk_norm else nn.Identity()
        )

    @torch.no_grad()
    def _refresh_derived_weights(self, *args: object) -> None:
        """Rebuild fused native or packed FP8 weights from state-dict parameters."""
        del args
        fused_weight = torch.cat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight),
            dim=0,
        ).detach()
        if self.q_proj.bias is None:
            fused_bias = None
        else:
            assert self.k_proj.bias is not None and self.v_proj.bias is not None
            fused_bias = torch.cat(
                (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias),
                dim=0,
            ).detach()

        self._fused_qkv_bias = fused_bias
        if self.use_fp8:
            self._fused_qkv_weight, self._fused_qkv_weight_scale = quantize_fp8_weight(
                fused_weight
            )
            self._output_weight_fp8, self._output_weight_scale = quantize_fp8_weight(
                self.output_proj.weight
            )
        else:
            self._fused_qkv_weight = fused_weight.contiguous()
            self._fused_qkv_weight_scale = None
            self._output_weight_fp8 = None
            self._output_weight_scale = None

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> TritonMultiHeadAttention:
        """Move parameters and rebuild derived weights on their final device/dtype."""
        module = super()._apply(fn, recurse=recurse)
        self._refresh_derived_weights()
        return module

    def initialize_cache(
        self,
        batch_size: int,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> BlockKVCache:
        """Allocate a rolling cache matching the configured precision policy."""
        cache_shape = (
            batch_size,
            sink_size + window_size,
            self.n_heads,
            self.head_dim,
        )
        if self.use_fp8 and dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("FP8 attention requires FP16 or BF16 activations")
        return BlockKVCache(
            k_shape=cache_shape,
            v_shape=cache_shape,
            seq_dim=1,
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            device=device,
            dtype=torch.float8_e4m3fn if self.use_fp8 else dtype,
        )

    def _project_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project Q/K/V with one fused native or row-scaled FP8 GEMM."""
        if self._fused_qkv_weight is None:
            raise RuntimeError("fused QKV weight is not initialized")
        if self.use_fp8:
            if self._fused_qkv_weight_scale is None:
                raise RuntimeError("FP8 QKV weight scales are not initialized")
            qkv = fp8_linear(
                x,
                self._fused_qkv_weight,
                self._fused_qkv_weight_scale,
                self._fused_qkv_bias,
                x.dtype,
            )
        else:
            qkv = F.linear(x, self._fused_qkv_weight, self._fused_qkv_bias)
        qkv = qkv.reshape(
            -1,
            x.shape[-2],
            3,
            self.n_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        return query, key, value

    def _project_output(self, x: Tensor, output_dtype: torch.dtype) -> Tensor:
        """Apply the native or row-scaled FP8 output projection."""
        if not self.use_fp8:
            return self.output_proj(x)
        if self._output_weight_fp8 is None or self._output_weight_scale is None:
            raise RuntimeError("FP8 output weight is not initialized")
        return fp8_linear(
            x,
            self._output_weight_fp8,
            self._output_weight_scale,
            self.output_proj.bias,
            output_dtype,
        )

    def _validate_forward_inputs(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None,
    ) -> None:
        """Validate module inputs before projection or cache mutation."""
        if x.ndim < 2:
            raise ValueError(f"x must have shape [..., L, D]; got {tuple(x.shape)}")
        if x.shape[-1] != self.query_dim:
            raise ValueError(
                f"x feature width must equal query_dim={self.query_dim}; "
                f"got {x.shape[-1]}"
            )
        if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "TritonMultiHeadAttention requires CUDA FP16 or BF16 inputs"
            )
        if torch.cuda.get_device_capability(x.device)[0] < 9:
            raise RuntimeError(
                "TritonMultiHeadAttention TMA kernels require compute capability "
                "9.0 or newer"
            )
        if kv_cache._curr_chunk_idx is None:
            raise RuntimeError("call kv_cache.before_update() before attention")
        if kv_cache.seq_dim != 1 or kv_cache._k.ndim != 4:
            raise ValueError(
                "TritonMultiHeadAttention requires a [B, S, H, D] cache with seq_dim=1"
            )
        if x.shape[-2] != kv_cache.chunk_size:
            raise ValueError(
                f"x sequence length must equal cache chunk_size={kv_cache.chunk_size}; "
                f"got {x.shape[-2]}"
            )

        batch_size = math.prod(x.shape[:-2])
        expected_cache_shape = (batch_size, self.n_heads, self.head_dim)
        cache_shape = (
            kv_cache._k.shape[0],
            kv_cache._k.shape[2],
            kv_cache._k.shape[3],
        )
        if cache_shape != expected_cache_shape:
            raise ValueError(
                "cache batch, head, and feature dimensions must equal "
                f"{expected_cache_shape}; got {cache_shape}"
            )
        if kv_cache._v.shape != kv_cache._k.shape:
            raise ValueError("Triton attention requires identical K/V cache shapes")
        if kv_cache._k.device != x.device or kv_cache._v.device != x.device:
            raise RuntimeError("K/V cache tensors must match the input device")
        expected_cache_dtype = torch.float8_e4m3fn if self.use_fp8 else x.dtype
        if (
            kv_cache._k.dtype != expected_cache_dtype
            or kv_cache._v.dtype != expected_cache_dtype
        ):
            raise RuntimeError(f"K/V cache tensors must use {expected_cache_dtype}")
        if not kv_cache._k.is_contiguous() or not kv_cache._v.is_contiguous():
            raise RuntimeError("K/V cache tensors must be contiguous")

        if rope_freqs is not None:
            expected_rope_shape = (x.shape[-2], 1, 1, self.head_dim)
            if tuple(rope_freqs.shape) != expected_rope_shape:
                raise ValueError(
                    f"rope_freqs must have shape {expected_rope_shape}; "
                    f"got {tuple(rope_freqs.shape)}"
                )
            if rope_freqs.device != x.device:
                raise RuntimeError("rope_freqs and x must be on the same device")

    @torch.no_grad()
    def forward(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None = None,
    ) -> Tensor:
        """Project a token chunk, update its cache, and apply TMA attention.

        Args:
            x: Current query tokens with shape ``[..., L, query_dim]``.
            kv_cache: Streaming cache prepared with ``before_update``.
            rope_freqs: Optional current-chunk angles with shape
                ``[L, 1, 1, head_dim]``.

        Returns:
            Output-projected attention result with the same shape as ``x``.
        """
        self._validate_forward_inputs(x, kv_cache, rope_freqs)
        batch_shape = x.shape[:-2]
        sequence_length = x.shape[-2]
        # Triton uses one physical batch axis. Fold arbitrary leading dimensions
        # into it, then restore the original shape at the module boundary.
        x_flat = x.reshape(-1, sequence_length, self.query_dim)

        # One GEMM produces token-major [B, L, H, D] Q/K/V views.
        query, key, value = self._project_qkv(x_flat)

        cache_read_start, cache_write_start, cache_write_length = _cache_write_slice(
            kv_cache
        )
        if isinstance(self.q_norm, nn.RMSNorm):
            if not isinstance(self.k_norm, nn.RMSNorm):
                raise RuntimeError("Q/K normalization modules must use the same policy")
            if self.q_norm.eps is None:
                raise RuntimeError("Triton Q/K RMSNorm requires an explicit epsilon")
            query_weight: Tensor | None = self.q_norm.weight
            key_weight: Tensor | None = self.k_norm.weight
            norm_eps = self.q_norm.eps
        else:
            if not isinstance(self.q_norm, nn.Identity) or not isinstance(
                self.k_norm, nn.Identity
            ):
                raise RuntimeError("Q/K normalization modules must use the same policy")
            query_weight = None
            key_weight = None
            norm_eps = 0.0
        # Return processed Q while writing processed K and unmodified V directly
        # into the implementation-owned cache storage.
        query = fused_rms_rope_kv_cache_update(
            query,
            key,
            value,
            kv_cache._k,
            kv_cache._v,
            query_weight=query_weight,
            key_weight=key_weight,
            norm_eps=norm_eps,
            norm_scope=self.qk_norm_scope,
            rope_freqs=rope_freqs,
            rope_interleaved=self.rope_interleaved,
            cache_read_start=cache_read_start,
            cache_write_start=cache_write_start,
            cache_write_length=cache_write_length,
        )

        # The visible cache already includes this chunk. Attention is non-causal
        # within a chunk because video-model recipes expose whole chunks at once.
        output = flash_attention_2_tma(
            query,
            kv_cache.cached_k(),
            kv_cache.cached_v(),
        )
        output = output.reshape(-1, sequence_length, self.inner_dim)
        output = self._project_output(output, x.dtype)
        return output.reshape(batch_shape + (sequence_length, self.query_dim))


__all__ = ["TritonMultiHeadAttention"]
