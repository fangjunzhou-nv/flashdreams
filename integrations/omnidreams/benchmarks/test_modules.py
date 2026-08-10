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

"""Microbenchmarks for Omnidreams model modules.

Run the DiT block benchmark with::

    uv run --group test pytest \
        integrations/omnidreams/benchmarks/test_modules.py \
        -p no:manual_marker -m manual --benchmark-only
"""

from __future__ import annotations

import pytest
import torch
from omnidreams.transformer.impl.modules import Block
from omnidreams.transformer.impl.network import CosmosDiTNetworkConfig
from pytest_benchmark.fixture import BenchmarkFixture

from flashdreams.core.attention.rope import RotaryPositionEmbedding3D

pytestmark = pytest.mark.manual

_GPU_REASON = "Omnidreams DiT block benchmark requires CUDA"

# Production single-view, 720p chunk-2 geometry. The VAE reduces 720x1280 to
# 90x160 latents, and the DiT's 2x2 spatial patching produces 45x80 tokens per
# latent frame. The local window holds three two-frame chunks.
_BATCH_SIZE = 1
_NUM_VIEWS = 1
_LATENT_HEIGHT = 90
_LATENT_WIDTH = 160
_CHUNK_SIZE_T = 2
_WINDOW_SIZE_T = 6
_TEXT_TOKENS = 512
_WARMUP_ROUNDS = 3
_BENCHMARK_ROUNDS = 20
_SEED = 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@torch.inference_mode()
def test_dit_block_benchmark(benchmark: BenchmarkFixture) -> None:
    """Benchmark a production-configured DiT block with a full KV window."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Omnidreams DiT block benchmark requires bfloat16 support")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(_SEED)
    config = CosmosDiTNetworkConfig()

    # Keep this constructor in lockstep with CosmosDiTNetwork.__init__.
    block = Block(
        x_dim=config.model_channels,
        context_dim=config.crossattn_emb_channels,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        use_adaln_lora=config.use_adaln_lora,
        adaln_lora_dim=config.adaln_lora_dim,
        enable_cross_view_attn=config.enable_cross_view_attn,
        cp_method=config.cp_method,
    ).to(device=device, dtype=dtype)
    block.eval()

    patch_t = _CHUNK_SIZE_T // config.patch_temporal
    patch_h = _LATENT_HEIGHT // config.patch_spatial
    patch_w = _LATENT_WIDTH // config.patch_spatial
    tokens_per_frame = patch_h * patch_w
    chunk_tokens = patch_t * tokens_per_frame
    window_tokens = _WINDOW_SIZE_T * tokens_per_frame
    head_dim = config.model_channels // config.num_heads

    x = torch.randn(
        (
            _BATCH_SIZE,
            _NUM_VIEWS,
            patch_t,
            tokens_per_frame,
            config.model_channels,
        ),
        device=device,
        dtype=dtype,
    )
    emb = torch.randn((_BATCH_SIZE, config.model_channels), device=device, dtype=dtype)
    adaln_lora = torch.randn(
        (_BATCH_SIZE, 3 * config.model_channels), device=device, dtype=dtype
    )
    context = torch.randn(
        (
            _BATCH_SIZE,
            _NUM_VIEWS,
            _TEXT_TOKENS,
            config.crossattn_emb_channels,
        ),
        device=device,
        dtype=dtype,
    )
    cache = block.initialize_cache(
        chunk_size=chunk_tokens,
        window_size=window_tokens,
        sink_size=0,
        context=context,
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=head_dim,
        len_h=patch_h,
        len_w=patch_w,
        len_t=patch_t,
        h_extrapolation_ratio=3.0,
        w_extrapolation_ratio=3.0,
        device=device,
    )

    def forward(chunk_idx: int, rope_freqs: torch.Tensor) -> torch.Tensor:
        cache.before_update(chunk_idx)
        output = block(
            x=x,
            emb=emb,
            cache=cache,
            rope_freqs=rope_freqs,
            adaln_lora=adaln_lora,
        )
        cache.after_update(chunk_idx)
        return output

    # Fill the rolling cache before timing so every measured call exercises
    # steady-state attention over the full local window. Repeating the final
    # chunk mirrors multiple denoising steps at one autoregressive position.
    steady_chunk_idx = _WINDOW_SIZE_T // _CHUNK_SIZE_T - 1
    rope_freqs = [rope.shift_t(chunk_idx) for chunk_idx in range(steady_chunk_idx + 1)]
    for chunk_idx, chunk_rope_freqs in enumerate(rope_freqs):
        output = forward(chunk_idx, chunk_rope_freqs)
    torch.cuda.synchronize()

    benchmark.group = "omnidreams-dit-block"
    benchmark.extra_info.update(
        {
            "batch_size": _BATCH_SIZE,
            "num_views": _NUM_VIEWS,
            "latent_shape": [_CHUNK_SIZE_T, _LATENT_HEIGHT, _LATENT_WIDTH],
            "chunk_tokens": chunk_tokens,
            "window_tokens": window_tokens,
            "text_tokens": _TEXT_TOKENS,
            "model_channels": config.model_channels,
            "num_heads": config.num_heads,
            "dtype": str(dtype),
            "attention_backend": "cudnn",
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "seed": _SEED,
        }
    )

    def synchronized_forward() -> torch.Tensor:
        result = forward(steady_chunk_idx, rope_freqs[steady_chunk_idx])
        torch.cuda.synchronize()
        return result

    output = benchmark.pedantic(
        synchronized_forward,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
