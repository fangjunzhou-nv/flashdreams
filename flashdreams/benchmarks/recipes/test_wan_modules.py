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

"""Microbenchmarks for Wan self-attention and DiT blocks.

Run the manual GPU benchmarks with::

    uv run --package flashdreams --group test pytest \
        flashdreams/benchmarks/recipes/test_wan_modules.py \
        -p no:manual_marker -m manual --benchmark-only -v
"""

from __future__ import annotations

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from flashdreams.accelerated.multi_head_attention_triton import (
    QKVFusionOption,
    SDPABackend,
)
from flashdreams.core.attention.rope import RotaryPositionEmbedding3D
from flashdreams.recipes.wan.transformer.impl.modules import (
    AttentionBackend,
    Block,
    TritonCrossAttention,
    TritonSelfAttention,
)
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork1pt3BConfig,
)

pytestmark = [
    pytest.mark.manual,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Wan DiT module benchmarks require CUDA.",
    ),
]

# Causal-Forcing's chunkwise 480x832 Wan 2.1 1.3B geometry. The VAE produces
# 3x60x104 latents per AR chunk; 1x2x2 DiT patches produce 3x30x52 tokens.
_PIXEL_HEIGHT = 480
_PIXEL_WIDTH = 832
_LATENT_HEIGHT = 60
_LATENT_WIDTH = 104
_CHUNK_SIZE_T = 3
_ATTENTION_HEIGHT = 30
_ATTENTION_WIDTH = 52
_WINDOW_CHUNKS = 7
_TEXT_TOKENS = 512
_CHUNK_TOKENS = _CHUNK_SIZE_T * _ATTENTION_HEIGHT * _ATTENTION_WIDTH
_WINDOW_TOKENS = _WINDOW_CHUNKS * _CHUNK_TOKENS
_SINK_TOKENS = 0
_WARMUP_ROUNDS = 3
_BENCHMARK_ROUNDS = 20
_SEED = 42


_BENCHMARK_CASES = (
    pytest.param(
        AttentionBackend.WAN,
        SDPABackend.CUDNN,
        False,
        QKVFusionOption.NONE,
        id="wan-torch",
    ),
    *(
        pytest.param(
            AttentionBackend.TRITON,
            sdpa_backend,
            use_fp8,
            qkv_fusion_option,
            id=(
                f"triton-"
                f"{'fa2' if sdpa_backend is SDPABackend.TRITON else 'cudnn'}-"
                f"{'fp8' if use_fp8 else 'bf16'}-"
                f"{qkv_fusion_option.value.replace('_', '-')}"
            ),
        )
        for sdpa_backend in SDPABackend
        for use_fp8 in (False, True)
        for qkv_fusion_option in QKVFusionOption
    ),
)


def _skip_unsupported_device(
    backend: AttentionBackend,
    device: torch.device,
) -> None:
    """Skip Triton attention on devices without tensor-memory acceleration."""
    if backend is AttentionBackend.TRITON and torch.cuda.get_device_capability(
        device
    ) < (9, 0):
        pytest.skip("Triton attention requires compute capability 9.0 or newer.")


def _make_block(
    config: WanDiTNetwork1pt3BConfig,
    backend: AttentionBackend,
) -> Block:
    """Build a backend-selected block with weight-matched random parameters."""

    def make(selected_backend: AttentionBackend) -> Block:
        return Block(
            dim=config.dim,
            ffn_dim=config.ffn_dim,
            num_heads=config.num_heads,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps,
            i2v=config.cross_attn_enable_img,
            apply_rope_before_kvcache=config.apply_rope_before_kvcache,
            cp_method=config.cp_method,
            attention_backend=selected_backend,
            sdpa_backend=config.sdpa_backend,
            cross_attn_sdpa_backend=config.cross_attn_sdpa_backend,
            self_attn_qkv_fusion_option=config.self_attn_qkv_fusion_option,
            cross_attn_qkv_fusion_option=config.cross_attn_qkv_fusion_option,
            use_fp8=config.use_fp8,
        )

    torch.manual_seed(_SEED)
    reference = make(AttentionBackend.WAN)
    if backend is AttentionBackend.WAN:
        return reference

    block = make(backend)
    block.load_state_dict(reference.state_dict(), strict=True)
    return block


def _assert_benchmark_case(
    block: Block,
    backend: AttentionBackend,
    sdpa_backend: SDPABackend,
    use_fp8: bool,
    qkv_fusion_option: QKVFusionOption,
) -> None:
    assert block.attention_backend is backend
    if backend is AttentionBackend.WAN:
        return

    assert isinstance(block.self_attn, TritonSelfAttention)
    assert isinstance(block.cross_attn, TritonCrossAttention)
    for attention in (block.self_attn, block.cross_attn):
        assert attention.sdpa_backend is sdpa_backend
        assert attention.use_fp8 is use_fp8
        assert attention.qkv_fusion_option is qkv_fusion_option


@pytest.mark.parametrize(
    "backend,sdpa_backend,use_fp8,qkv_fusion_option",
    _BENCHMARK_CASES,
)
@torch.inference_mode()
def test_self_attention_benchmark(
    benchmark: BenchmarkFixture,
    backend: AttentionBackend,
    sdpa_backend: SDPABackend,
    use_fp8: bool,
    qkv_fusion_option: QKVFusionOption,
) -> None:
    """Benchmark Wan self-attention against a full production KV window."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Wan self-attention benchmark requires bfloat16 support.")

    device = torch.device("cuda")
    _skip_unsupported_device(backend, device)
    dtype = torch.bfloat16
    config = WanDiTNetwork1pt3BConfig(
        attention_backend=backend,
        sdpa_backend=sdpa_backend,
        cross_attn_sdpa_backend=sdpa_backend,
        self_attn_qkv_fusion_option=qkv_fusion_option,
        cross_attn_qkv_fusion_option=qkv_fusion_option,
        use_fp8=use_fp8,
    )
    block = _make_block(config, backend)
    _assert_benchmark_case(block, backend, sdpa_backend, use_fp8, qkv_fusion_option)
    attention = block.self_attn
    attention.to(device=device, dtype=dtype).eval()
    generator = torch.Generator(device=device).manual_seed(_SEED)

    x = torch.randn(
        (_CHUNK_TOKENS, config.dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    cache = attention.allocate_kv_cache(
        batch_size=1,
        chunk_size=_CHUNK_TOKENS,
        window_size=_WINDOW_TOKENS,
        sink_size=_SINK_TOKENS,
        device=device,
        dtype=dtype,
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=config.dim // config.num_heads,
        len_t=_CHUNK_SIZE_T,
        len_h=_ATTENTION_HEIGHT,
        len_w=_ATTENTION_WIDTH,
        interleaved=True,
        device=device,
    )

    benchmark_chunk_idx = _WINDOW_CHUNKS
    rope_freqs = [
        rope.shift_t(chunk_idx) for chunk_idx in range(benchmark_chunk_idx + 1)
    ]
    for chunk_idx in range(_WINDOW_CHUNKS):
        cache.before_update(chunk_idx)
        output = attention(x, cache, rope_freqs[chunk_idx])
        cache.after_update(chunk_idx)
    torch.cuda.synchronize(device)

    benchmark.group = "wan-dit-self-attention"

    # Roll outside timing, then repeatedly overwrite the same slot to mirror
    # multiple denoising evaluations at one autoregressive position.
    cache.before_update(benchmark_chunk_idx)
    torch.cuda.synchronize(device)

    def synchronized_forward() -> torch.Tensor:
        result = attention(x, cache, rope_freqs[benchmark_chunk_idx])
        torch.cuda.synchronize(device)
        return result

    output = benchmark.pedantic(
        synchronized_forward,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )
    cache.after_update(benchmark_chunk_idx)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "backend,sdpa_backend,use_fp8,qkv_fusion_option",
    _BENCHMARK_CASES,
)
@torch.inference_mode()
def test_dit_block_benchmark(
    benchmark: BenchmarkFixture,
    backend: AttentionBackend,
    sdpa_backend: SDPABackend,
    use_fp8: bool,
    qkv_fusion_option: QKVFusionOption,
) -> None:
    """Benchmark a production-configured Wan DiT block at steady state."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Wan DiT block benchmark requires bfloat16 support.")

    device = torch.device("cuda")
    _skip_unsupported_device(backend, device)
    dtype = torch.bfloat16
    config = WanDiTNetwork1pt3BConfig(
        attention_backend=backend,
        sdpa_backend=sdpa_backend,
        cross_attn_sdpa_backend=sdpa_backend,
        self_attn_qkv_fusion_option=qkv_fusion_option,
        cross_attn_qkv_fusion_option=qkv_fusion_option,
        use_fp8=use_fp8,
    )
    block = _make_block(config, backend).to(device=device, dtype=dtype).eval()
    _assert_benchmark_case(block, backend, sdpa_backend, use_fp8, qkv_fusion_option)
    block.update_parameters_after_loading_checkpoint()
    generator = torch.Generator(device=device).manual_seed(_SEED)

    x = torch.randn(
        (_CHUNK_TOKENS, config.dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    modulation = torch.randn(
        (6, config.dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    context = torch.randn(
        (_TEXT_TOKENS, config.dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    cache = block.initialize_cache(
        chunk_size=_CHUNK_TOKENS,
        window_size=_WINDOW_TOKENS,
        sink_size=_SINK_TOKENS,
        context_text=context,
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=config.dim // config.num_heads,
        len_t=_CHUNK_SIZE_T,
        len_h=_ATTENTION_HEIGHT,
        len_w=_ATTENTION_WIDTH,
        interleaved=True,
        device=device,
    )

    def forward(chunk_idx: int, chunk_rope_freqs: torch.Tensor) -> torch.Tensor:
        cache.before_update(chunk_idx)
        result = block(
            x=x,
            e=modulation,
            cache=cache,
            rope_freqs=chunk_rope_freqs,
        )
        cache.after_update(chunk_idx)
        return result

    benchmark_chunk_idx = _WINDOW_CHUNKS
    rope_freqs = [
        rope.shift_t(chunk_idx) for chunk_idx in range(benchmark_chunk_idx + 1)
    ]
    for chunk_idx in range(_WINDOW_CHUNKS):
        output = forward(chunk_idx, rope_freqs[chunk_idx])
    torch.cuda.synchronize(device)

    benchmark.group = "wan-dit-block"

    cache.before_update(benchmark_chunk_idx)
    torch.cuda.synchronize(device)

    def synchronized_forward() -> torch.Tensor:
        result = block(
            x=x,
            e=modulation,
            cache=cache,
            rope_freqs=rope_freqs[benchmark_chunk_idx],
        )
        torch.cuda.synchronize(device)
        return result

    output = benchmark.pedantic(
        synchronized_forward,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )
    cache.after_update(benchmark_chunk_idx)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
