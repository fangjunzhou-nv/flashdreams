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

"""Reference and FlashAttention implementations of streaming self-attention."""

import math

import nvtx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

from flashdreams.core.attention.kvcache import BlockKVCache


class ReferenceSelfAttention(nn.Module):
    """Streaming self-attention implemented with basic PyTorch operations."""

    n_heads: int
    """Number of query, key, and value heads."""

    head_dim: int
    """Feature dimension of each attention head."""

    q_proj: nn.Linear
    """Projection from input features to query heads."""

    k_proj: nn.Linear
    """Projection from input features to key heads."""

    v_proj: nn.Linear
    """Projection from input features to value heads."""

    output_proj: nn.Linear
    """Projection from concatenated attention heads to output features."""

    q_norm: nn.RMSNorm
    """Per-head RMS normalization applied to projected queries."""

    k_norm: nn.RMSNorm
    """Per-head RMS normalization applied to projected keys."""

    def __init__(
        self,
        query_dim: int,
        n_heads: int = 8,
        head_dim: int = 64,
    ) -> None:
        """Initialize the self-attention projections and per-head normalization.

        Args:
            query_dim: Feature dimension of input tokens and projected output.
            n_heads: Number of attention heads.
            head_dim: Feature dimension of each attention head.
        """
        super().__init__()
        inner_dim = n_heads * head_dim

        self.n_heads = n_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Reject context parallelism for this reference implementation."""
        if cp_group is not None:
            raise NotImplementedError(
                "ReferenceSelfAttention does not support context parallelism."
            )

    @staticmethod
    def _apply_rope(x: Tensor, rope_freqs: Tensor) -> Tensor:
        """Apply non-interleaved rotary position embeddings to ``x``.

        Args:
            x: Query or key heads of shape ``[B, L, H, D]``. ``D`` must be even.
            rope_freqs: Rotation angles of shape ``[L, 1, 1, D]``.

        Returns:
            Position-encoded heads with the same shape and dtype as ``x``.
        """
        half_dim = x.shape[-1] // 2
        freqs = rope_freqs[:, 0, 0, :half_dim].unsqueeze(0).unsqueeze(2)
        cos_freqs = torch.cos(freqs).to(dtype=x.dtype)
        sin_freqs = torch.sin(freqs).to(dtype=x.dtype)
        first_half, second_half = x.chunk(2, dim=-1)
        return torch.cat(
            (
                first_half * cos_freqs - second_half * sin_freqs,
                second_half * cos_freqs + first_half * sin_freqs,
            ),
            dim=-1,
        )

    def _apply_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Apply explicit scaled dot-product attention.

        Args:
            query: Current query heads of shape ``[B, H, L, D]``.
            key: Cached key heads of shape ``[B, H, S, D]``.
            value: Cached value heads of shape ``[B, H, S, D]``.

        Returns:
            Attention output of shape ``[B, H, L, D]``.
        """
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_weights, value)

    def initialize_cache(
        self,
        batch_size: int,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BlockKVCache:
        """Initialize the KV cache used by streaming self-attention."""
        total_size = sink_size + window_size
        return BlockKVCache(
            k_shape=(batch_size, total_size, self.n_heads, self.head_dim),
            v_shape=(batch_size, total_size, self.n_heads, self.head_dim),
            seq_dim=-3,
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            device=device,
            dtype=dtype,
        )

    @nvtx.annotate("self_attention.forward")
    def forward(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor,
    ) -> Tensor:
        """Update the KV cache and apply self-attention to ``x``."""
        with nvtx.annotate("self_attention.prepare_shapes"):
            batch_shape = x.shape[:-2]
            batch_size = math.prod(batch_shape)
            sequence_length, hidden_dim = x.shape[-2:]
            inner_dim = self.n_heads * self.head_dim
            assert hidden_dim == inner_dim, "n_heads * head_dim must be equal to D"

        with nvtx.annotate("self_attention.qkv_projection"):
            head_shape = (
                batch_size,
                sequence_length,
                self.n_heads,
                self.head_dim,
            )
            query = self.q_norm(self.q_proj(x).reshape(head_shape))
            key = self.k_norm(self.k_proj(x).reshape(head_shape))
            value = self.v_proj(x).reshape(head_shape)

        with nvtx.annotate("self_attention.rope_and_cache_update"):
            query = self._apply_rope(query, rope_freqs)
            key = self._apply_rope(key, rope_freqs)
            kv_cache.update(key, value)

        with nvtx.annotate("self_attention.attention"):
            query = query.transpose(1, 2)
            key = kv_cache.cached_k().transpose(1, 2)
            value = kv_cache.cached_v().transpose(1, 2)
            output = self._apply_attention(query, key, value)

        with nvtx.annotate("self_attention.output_projection"):
            output = output.transpose(1, 2).reshape(
                batch_shape + (sequence_length, inner_dim)
            )
            return self.output_proj(output)


class FlashAttnSelfAttention(ReferenceSelfAttention):
    """Streaming self-attention backed by PyTorch SDPA FlashAttention."""

    def _apply_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Apply scaled dot-product attention with the FlashAttention backend."""
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
            )


class AcceleratedSelfAttention:
    pass
