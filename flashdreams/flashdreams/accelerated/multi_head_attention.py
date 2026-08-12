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

"""Extensible multi-head attention interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, TypeVar

from torch import Tensor, nn


KVCacheT = TypeVar("KVCacheT")


class QKNormScope(str, Enum):
    """Feature scope for query and key normalization."""

    HEAD = "head"
    """Normalize each attention head independently."""

    INNER = "inner"
    """Normalize the complete projected inner width."""


class MultiHeadAttention(nn.Module, ABC, Generic[KVCacheT]):
    """Generic multi-head attention interface over an implementation-owned cache.

    The complete attention operation is the extension point so implementations
    may freely fuse projections, normalization, RoPE, cache access, attention,
    and output projection. Streaming self-attention implementations update a
    caller-prepared cache from ``x``. Cross-attention implementations instead
    read projected K/V from a precomputed static or composite cache.
    """

    query_dim: int
    """Feature dimension of input and output tokens."""

    context_dim: int
    """Feature dimension projected into cached keys and values."""

    n_heads: int
    """Number of query, key, and value heads."""

    head_dim: int
    """Feature dimension of each attention head."""

    inner_dim: int
    """Projected feature width, equal to ``n_heads * head_dim``."""

    qk_norm_scope: QKNormScope
    """Feature scope used by query and key normalization."""

    rope_interleaved: bool
    """Whether RoPE rotates adjacent feature pairs instead of half splits."""

    def __init__(
        self,
        query_dim: int,
        n_heads: int = 8,
        head_dim: int = 64,
        *,
        context_dim: int | None = None,
        qk_norm_scope: QKNormScope = QKNormScope.HEAD,
        rope_interleaved: bool = False,
    ) -> None:
        """Initialize shared attention geometry and implementation policies.

        Args:
            query_dim: Feature dimension of input and output tokens.
            n_heads: Number of query, key, and value heads.
            head_dim: Feature dimension of each attention head.
            context_dim: Feature dimension projected into cached keys and values.
                Defaults to ``query_dim`` for self-attention.
            qk_norm_scope: Feature scope used by query and key normalization.
            rope_interleaved: Rotate adjacent feature pairs instead of half splits.

        Raises:
            TypeError: ``qk_norm_scope`` is not a :class:`QKNormScope`.
            ValueError: Any dimension is not positive.
        """
        super().__init__()
        context_dim = query_dim if context_dim is None else context_dim
        if query_dim <= 0:
            raise ValueError(f"query_dim must be positive; got {query_dim}")
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive; got {context_dim}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive; got {n_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive; got {head_dim}")
        if not isinstance(qk_norm_scope, QKNormScope):
            raise TypeError(
                f"qk_norm_scope must be a QKNormScope; got {qk_norm_scope!r}"
            )

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner_dim = n_heads * head_dim
        self.qk_norm_scope = qk_norm_scope
        self.rope_interleaved = rope_interleaved

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        kv_cache: KVCacheT,
        rope_freqs: Tensor | None = None,
    ) -> Tensor:
        """Apply attention using an implementation-owned K/V cache.

        Args:
            x: Query token sequence, shape ``[..., L, query_dim]``.
            kv_cache: Caller-prepared streaming cache or precomputed static or
                composite cross-attention cache.
            rope_freqs: Optional implementation-interpreted positional data.
                Standard RoPE may describe the current query chunk, while
                cache-relative RoPE may describe every visible cache slot.

        Returns:
            Attention result with the same shape as ``x``.
        """


__all__ = ["MultiHeadAttention", "QKNormScope"]
