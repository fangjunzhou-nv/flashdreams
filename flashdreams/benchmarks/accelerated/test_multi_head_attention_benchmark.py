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

"""Matched microbenchmarks for Torch, Cosmos, and Wan multi-head attention.

Cosmos uses the public OmniDreams single-view chunk-2 geometry. Wan uses the
Causal-Forcing Wan 2.1 1.3B chunkwise geometry. Each pair is benchmarked with
identical random weights.

Run the manual GPU benchmarks with::

    uv run --package flashdreams --group test pytest \
        flashdreams/benchmarks/accelerated/test_multi_head_attention_benchmark.py \
        -p no:manual_marker -m manual --benchmark-only -v
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from torch import Tensor

from flashdreams.accelerated.multi_head_attention import QKNormScope
from flashdreams.accelerated.reference import TorchMultiHeadAttention
from flashdreams.core.attention import BlockKVCache, RotaryPositionEmbedding3D
from flashdreams.recipes.cosmos.transformer.impl import modules as cosmos_modules
from flashdreams.recipes.wan.transformer.impl import modules as wan_modules

pytestmark = [
    pytest.mark.manual,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Accelerated multi-head attention benchmarks require CUDA.",
    ),
]

_WARMUP_ROUNDS = 3
"""Warmup calls used to absorb kernel compilation and autotuning."""

_BENCHMARK_ROUNDS = 20
"""Measured calls used for each implementation comparison."""

_ModelFamily = Literal["cosmos", "wan"]
_Implementation = Literal["recipe", "accelerated_torch"]
_Attention = (
    TorchMultiHeadAttention
    | cosmos_modules.MultiHeadAttention
    | wan_modules.MultiHeadAttention
)

_BATCH_SIZE = 1
_DTYPE = torch.bfloat16
_SEED = 42
_SINK_SIZE = 0


@dataclass(frozen=True)
class _AttentionBenchmarkConfig:
    """Multi-head attention geometry for one model family."""

    model_family: _ModelFamily
    """Attention recipe selected by this configuration."""

    query_dim: int
    """Attention input and output width."""

    n_heads: int
    """Number of query, key, and value heads."""

    len_t: int
    """Temporal attention-grid size per chunk."""

    len_h: int
    """Height attention-grid size per chunk."""

    len_w: int
    """Width attention-grid size per chunk."""

    window_chunks: int
    """Number of chunks retained by the local attention window."""

    h_extrapolation_ratio: float
    """RoPE extrapolation ratio along latent height."""

    w_extrapolation_ratio: float
    """RoPE extrapolation ratio along latent width."""

    @property
    def chunk_size(self) -> int:
        """Return the number of tokens in one autoregressive chunk."""
        return self.len_t * self.len_h * self.len_w

    @property
    def window_size(self) -> int:
        """Return the number of tokens in the rolling local window."""
        return self.window_chunks * self.chunk_size

    @property
    def head_dim(self) -> int:
        """Return the feature width of each attention head."""
        return self.query_dim // self.n_heads


_COSMOS_CONFIG = _AttentionBenchmarkConfig(
    model_family="cosmos",
    query_dim=2048,
    n_heads=16,
    len_t=2,
    len_h=44,
    len_w=80,
    window_chunks=3,
    h_extrapolation_ratio=3.0,
    w_extrapolation_ratio=3.0,
)
"""Attention geometry from OmniDreams' public single-view chunk-2 config."""

_WAN_CONFIG = _AttentionBenchmarkConfig(
    model_family="wan",
    query_dim=1536,
    n_heads=12,
    len_t=3,
    len_h=30,
    len_w=52,
    window_chunks=7,
    h_extrapolation_ratio=1.0,
    w_extrapolation_ratio=1.0,
)
"""Attention geometry from Causal-Forcing's chunkwise Wan 2.1 1.3B config."""


def _make_cache(
    config: _AttentionBenchmarkConfig,
    device: torch.device,
) -> BlockKVCache:
    """Build a rolling cache for one benchmark case."""
    return BlockKVCache(
        k_shape=(
            _BATCH_SIZE,
            config.window_size,
            config.n_heads,
            config.head_dim,
        ),
        v_shape=(
            _BATCH_SIZE,
            config.window_size,
            config.n_heads,
            config.head_dim,
        ),
        seq_dim=1,
        chunk_size=config.chunk_size,
        window_size=config.window_size,
        sink_size=_SINK_SIZE,
        device=device,
        dtype=_DTYPE,
    )


def _make_attention(
    config: _AttentionBenchmarkConfig,
    implementation: _Implementation,
) -> _Attention:
    """Build a recipe attention module or its weight-matched Torch reference.

    Args:
        config: Multi-head attention dimensions and family.
        implementation: Recipe module or generic Torch reference to return.

    Returns:
        Configured multi-head attention module with deterministic random weights.
    """
    if config.model_family == "cosmos":
        recipe_attention = cosmos_modules.MultiHeadAttention(
            query_dim=config.query_dim,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
        )
        if implementation == "recipe":
            return recipe_attention

        torch_attention = TorchMultiHeadAttention(
            query_dim=config.query_dim,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
            qkv_bias=False,
            output_bias=False,
            qk_norm_scope=QKNormScope.HEAD,
            rope_interleaved=False,
        )
        torch_attention.load_state_dict(recipe_attention.state_dict())
        return torch_attention

    recipe_attention = wan_modules.MultiHeadAttention(
        query_dim=config.query_dim,
        n_heads=config.n_heads,
        head_dim=config.head_dim,
    )
    if implementation == "recipe":
        return recipe_attention

    torch_attention = TorchMultiHeadAttention(
        query_dim=config.query_dim,
        n_heads=config.n_heads,
        head_dim=config.head_dim,
        qkv_bias=True,
        output_bias=True,
        qk_norm_scope=QKNormScope.INNER,
        rope_interleaved=True,
    )
    torch_attention.q_proj.load_state_dict(recipe_attention.q.state_dict())
    torch_attention.k_proj.load_state_dict(recipe_attention.k.state_dict())
    torch_attention.v_proj.load_state_dict(recipe_attention.v.state_dict())
    torch_attention.output_proj.load_state_dict(recipe_attention.o.state_dict())
    torch_attention.q_norm.load_state_dict(recipe_attention.norm_q.state_dict())
    torch_attention.k_norm.load_state_dict(recipe_attention.norm_k.state_dict())
    return torch_attention


@torch.inference_mode()
def _benchmark_multi_head_attention(
    benchmark: BenchmarkFixture,
    *,
    config: _AttentionBenchmarkConfig,
    implementation: _Implementation,
) -> None:
    """Run a full-window attention benchmark for one implementation.

    Args:
        benchmark: Pytest benchmark fixture used to record synchronized timings.
        config: Multi-head attention dimensions and family.
        implementation: Recipe module or accelerated Torch reference to measure.
    """
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Multi-head attention benchmark requires bfloat16 support.")

    device = torch.device("cuda")
    model_family = config.model_family
    benchmark_chunk_idx = config.window_chunks
    torch.manual_seed(_SEED)
    attention = _make_attention(config, implementation)
    attention.to(device=device, dtype=_DTYPE).eval()

    generator = torch.Generator(device=device).manual_seed(_SEED)
    inputs = [
        torch.randn(
            _BATCH_SIZE,
            config.chunk_size,
            config.query_dim,
            generator=generator,
            device=device,
            dtype=_DTYPE,
        )
        for _ in range(benchmark_chunk_idx + 1)
    ]
    rope = RotaryPositionEmbedding3D(
        head_dim=config.head_dim,
        len_t=config.len_t,
        len_h=config.len_h,
        len_w=config.len_w,
        interleaved=model_family == "wan",
        h_extrapolation_ratio=config.h_extrapolation_ratio,
        w_extrapolation_ratio=config.w_extrapolation_ratio,
        device=device,
    )
    rope_freqs = [
        rope.shift_t(chunk_idx) for chunk_idx in range(benchmark_chunk_idx + 1)
    ]
    cache = _make_cache(config, device)

    # Fill the local window before timing so every measured call uses the same
    # full-context shape and overwrites the final cache slot in place.
    for chunk_idx in range(config.window_chunks):
        cache.before_update(chunk_idx)
        attention(inputs[chunk_idx], cache, rope_freqs[chunk_idx])
        cache.after_update(chunk_idx)
    torch.cuda.synchronize()

    benchmark.group = f"accelerated-multi-head-attention-{model_family}"
    benchmark.extra_info.update(
        {
            "model_family": model_family,
            "implementation": implementation,
            "batch_size": _BATCH_SIZE,
            "attention_grid": [config.len_t, config.len_h, config.len_w],
            "chunk_tokens": config.chunk_size,
            "window_tokens": config.window_size,
            "query_dim": config.query_dim,
            "num_heads": config.n_heads,
            "head_dim": config.head_dim,
            "parameter_count": sum(
                parameter.numel() for parameter in attention.parameters()
            ),
            "checkpoint": "random_init_shared_weights",
            "dtype": str(_DTYPE),
            "qkv_bias": model_family == "wan",
            "output_bias": model_family == "wan",
            "qk_norm_scope": "inner" if model_family == "wan" else "head",
            "attention_backend": (
                "auto_sdpa" if implementation == "accelerated_torch" else "cudnn"
            ),
            "cache_state": "full_window",
            "cache_prefill_chunks": config.window_chunks,
            "benchmark_chunk_idx": benchmark_chunk_idx,
            "rope_interleaved": model_family == "wan",
            "h_extrapolation_ratio": config.h_extrapolation_ratio,
            "w_extrapolation_ratio": config.w_extrapolation_ratio,
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "warmup_rounds": _WARMUP_ROUNDS,
            "benchmark_rounds": _BENCHMARK_ROUNDS,
            "seed": _SEED,
        }
    )

    # Roll the window once outside the timing region, then repeatedly overwrite
    # the same cache slot so every warmup and measured call does identical work.
    cache.before_update(benchmark_chunk_idx)
    torch.cuda.synchronize()

    def synchronized_forward() -> Tensor:
        result = attention(
            inputs[benchmark_chunk_idx],
            cache,
            rope_freqs[benchmark_chunk_idx],
        )
        torch.cuda.synchronize()
        return result

    output = benchmark.pedantic(
        synchronized_forward,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )
    cache.after_update(benchmark_chunk_idx)

    assert output.shape == inputs[benchmark_chunk_idx].shape
    assert torch.isfinite(output).all()


# Explicit benchmark cases


def test_cosmos_recipe_multi_head_attention_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark the Cosmos recipe multi-head attention implementation."""
    _benchmark_multi_head_attention(
        benchmark,
        config=_COSMOS_CONFIG,
        implementation="recipe",
    )


def test_cosmos_accelerated_torch_multi_head_attention_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark accelerated Torch attention configured for Cosmos."""
    _benchmark_multi_head_attention(
        benchmark,
        config=_COSMOS_CONFIG,
        implementation="accelerated_torch",
    )


def test_wan_recipe_multi_head_attention_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark the Wan recipe multi-head attention implementation."""
    _benchmark_multi_head_attention(
        benchmark,
        config=_WAN_CONFIG,
        implementation="recipe",
    )


def test_wan_accelerated_torch_multi_head_attention_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark accelerated Torch attention configured for Wan."""
    _benchmark_multi_head_attention(
        benchmark,
        config=_WAN_CONFIG,
        implementation="accelerated_torch",
    )
