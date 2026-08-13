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

"""PyTorch reference implementation of streaming multi-head attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashdreams.accelerated.multi_head_attention import (
    MultiHeadAttention,
    QKNormScope,
)
from flashdreams.core.attention import BlockKVCache


class TorchMultiHeadAttention(MultiHeadAttention[BlockKVCache]):
    """PyTorch reference for streaming self-attention with a block KV cache.

    Shape comments use ``L`` for the current token count, ``S`` for the visible
    cached context, ``H`` for the number of heads, and ``D`` for the head
    dimension. Leading ``...`` dimensions are preserved throughout.
    """

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
    ) -> None:
        """Initialize the reference attention projections and normalization.

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

        Raises:
            TypeError: ``qk_norm_scope`` is not a :class:`QKNormScope`.
            ValueError: A dimension is non-positive.
        """
        super().__init__(
            query_dim=query_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            qk_norm_scope=qk_norm_scope,
            rope_interleaved=rope_interleaved,
        )

        # Q/K/V map token features ``[query_dim]`` to ``[H * D]``; the output
        # projection maps the concatenated heads back to ``[query_dim]``.
        self.q_proj = nn.Linear(self.query_dim, self.inner_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(self.query_dim, self.inner_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.query_dim, self.inner_dim, bias=qkv_bias)
        self.output_proj = nn.Linear(self.inner_dim, self.query_dim, bias=output_bias)

        # RMSNorm parameters span ``[D]`` per head or ``[H * D]`` jointly.
        norm_dim = (
            self.head_dim if qk_norm_scope is QKNormScope.HEAD else self.inner_dim
        )
        self.q_norm: nn.Module = (
            nn.RMSNorm(norm_dim, eps=qk_norm_eps) if qk_norm else nn.Identity()
        )
        self.k_norm: nn.Module = (
            nn.RMSNorm(norm_dim, eps=qk_norm_eps) if qk_norm else nn.Identity()
        )

    def initialize_cache(
        self,
        batch_size: int,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> BlockKVCache:
        """Allocate a native-precision rolling K/V cache.

        Args:
            batch_size: Flattened batch size ``B``.
            chunk_size: Number of current tokens ``L`` written per update.
            window_size: Number of rolling context tokens retained after the sink.
            sink_size: Number of initial context tokens that are never evicted.
            device: Device on which to allocate K/V storage.
            dtype: Data type used by K/V storage.

        Returns:
            Block cache with K/V storage shaped
            ``[B, sink_size + window_size, H, D]``.
        """
        cache_shape = (
            batch_size,
            sink_size + window_size,
            self.n_heads,
            self.head_dim,
        )
        return BlockKVCache(
            k_shape=cache_shape,
            v_shape=cache_shape,
            seq_dim=1,
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None = None,
    ) -> Tensor:
        """Project query tokens, update cached K/V, and apply attention.

        RoPE is applied before the cache update so cached keys retain their
        positional encoding. Passing ``None`` skips RoPE.

        Args:
            x: Query tokens, shape ``[..., L, query_dim]``.
            kv_cache: Streaming block cache prepared for the current update.
            rope_freqs: Optional current-chunk frequencies, shape
                ``[L, 1, 1, D]``.

        Returns:
            Output-projected attention result, shape
            ``[..., L, query_dim]``.
        """
        query, key, value = self._projection(x)
        if rope_freqs is not None:
            query, key = self._rope(query, key, rope_freqs)
        key, value = self._kv_cache_update(key, value, kv_cache)
        output = self._attention(query, key, value)
        return self._output_projection(output)

    def _projection(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project tokens and apply Q/K normalization.

        Args:
            x: Input tokens, shape ``[..., L, query_dim]``.

        Returns:
            Query, key, and value tensors with shape ``[..., L, H, D]``.
        """
        # Project token features: ``[..., L, query_dim] -> [..., L, H * D]``.
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        # Split the projected width into heads:
        # ``[..., L, H * D] -> [..., L, H, D]``.
        head_shape = x.shape[:-1] + (self.n_heads, self.head_dim)
        query = query.reshape(head_shape)
        key = key.reshape(head_shape)
        value = value.reshape(head_shape)

        if self.qk_norm_scope is QKNormScope.INNER:
            # Normalize all heads jointly and restore ``[..., L, H, D]``.
            query = self.q_norm(query.flatten(-2)).reshape(head_shape)
            key = self.k_norm(key.flatten(-2)).reshape(head_shape)
        else:
            # Normalize each ``[D]`` head independently; shape remains
            # ``[..., L, H, D]``.
            query = self.q_norm(query)
            key = self.k_norm(key)
        return query, key, value

    def _apply_rope(self, x: Tensor, rope_freqs: Tensor) -> Tensor:
        """Apply RoPE using native PyTorch tensor operations."""
        if x.shape[-1] % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim; got {x.shape[-1]}")

        # RoPE lookup shape is ``[L, 1, 1, D]`` for an input shaped
        # ``[..., L, H, D]``.
        expected_shape = (x.shape[-3], 1, 1, x.shape[-1])
        if tuple(rope_freqs.shape) != expected_shape:
            raise ValueError(
                f"rope_freqs must have shape {expected_shape}; "
                f"got {tuple(rope_freqs.shape)}"
            )

        # Broadcast positions over leading dimensions and heads:
        # ``[L, 1, 1, D] -> [..., L, 1, D]``.
        freqs = rope_freqs[:, 0, 0, :].reshape(
            (1,) * (x.ndim - 3) + (x.shape[-3], 1, x.shape[-1])
        )

        # Materialize rotation coefficients with shape ``[..., L, 1, D]``.
        cos_freqs = torch.cos(freqs).to(dtype=x.dtype)
        sin_freqs = torch.sin(freqs).to(dtype=x.dtype)
        if self.rope_interleaved:
            # Rotate adjacent feature pairs; shape stays ``[..., L, H, D]``.
            rotated = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)
        else:
            # Rotate matching half-split features; shape stays ``[..., L, H, D]``.
            first, second = x.chunk(2, dim=-1)
            rotated = torch.cat((-second, first), dim=-1)

        # Apply the elementwise complex rotation: ``[..., L, H, D]``.
        return x * cos_freqs + rotated * sin_freqs

    def _rope(
        self,
        query: Tensor,
        key: Tensor,
        rope_freqs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply the configured RoPE layout to query and key tensors."""
        # Rotate current Q and K independently; both remain ``[..., L, H, D]``.
        return self._apply_rope(query, rope_freqs), self._apply_rope(key, rope_freqs)

    def _kv_cache_update(
        self,
        key: Tensor,
        value: Tensor,
        kv_cache: BlockKVCache,
    ) -> tuple[Tensor, Tensor]:
        """Write current K/V and return every cache entry visible to attention."""
        # Write the current chunk, with K/V shaped ``[..., L, H, D]``.
        kv_cache.update(key, value)

        # Read the visible context, expanding the sequence axis to ``S``:
        # ``[..., L, H, D] -> [..., S, H, D]``.
        key = kv_cache.cached_k()
        value = kv_cache.cached_v()
        return key, value

    def _attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Apply non-causal scaled dot-product attention in PyTorch."""
        # Move heads before tokens for SDPA:
        # Q ``[..., L, H, D] -> [..., H, L, D]`` and
        # K/V ``[..., S, H, D] -> [..., H, S, D]``.
        query_heads = query.transpose(-3, -2)
        key_heads = key.transpose(-3, -2)
        value_heads = value.transpose(-3, -2)

        # Attend over the cached context: ``[..., H, L, D]``.
        output = F.scaled_dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            dropout_p=0.0,
            is_causal=False,
        )

        # Restore token-major layout: ``[..., H, L, D] -> [..., L, H, D]``.
        return output.transpose(-3, -2)

    def _output_projection(self, x: Tensor) -> Tensor:
        """Flatten attention heads and project back to ``query_dim``."""
        # Concatenate heads: ``[..., L, H, D] -> [..., L, H * D]``.
        x = x.flatten(-2)

        # Project attention features: ``[..., L, H * D] -> [..., L, query_dim]``.
        return self.output_proj(x)


__all__ = ["TorchMultiHeadAttention"]
