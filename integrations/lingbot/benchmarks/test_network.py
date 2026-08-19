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

"""Benchmark the complete LingBot camera-control DiT network by backend."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from lingbot.transformer.impl.network import (
    LingbotWorldDiTNetwork,
    LingbotWorldDiTNetwork14BConfig,
)
from pytest_benchmark.fixture import BenchmarkFixture

from flashdreams.core.attention import ContextParallelAttention
from flashdreams.core.attention.rope import RotaryPositionEmbedding3D
from flashdreams.core.distributed import init as init_distributed
from flashdreams.infra.acceleration import (
    CUDAGraphDispatch,
    cuda_graph_capture_ar_index,
)
from flashdreams.infra.compile import compile_module
from flashdreams.recipes.wan.transformer.impl.modules import AttentionBackend
from integrations.lingbot.benchmarks.cases import (
    BENCHMARK_CASES,
    AttentionBenchmarkCase,
    skip_unsupported_device,
)

pytestmark = pytest.mark.manual

_GPU_REASON = "LingBot DiT network benchmark requires CUDA"

# CLI replay geometry and bounded-cache preset. The 352x640 frame becomes a
# 44x80 latent, then 2x2 DiT patching yields 22x40 tokens per
# latent frame. Each AR step contains three latent frames.
_PIXEL_HEIGHT = 352
_PIXEL_WIDTH = 640
_LATENT_HEIGHT = 44
_LATENT_WIDTH = 80
_CHUNK_SIZE_T = 3
_WINDOW_SIZE_T = 15
_SINK_SIZE_T = 3
_TEXT_TOKENS = 512
_DIFFUSION_TIMESTEP = 1000.0
_CUDA_GRAPH_WARMUP_ITERS = 2
_WARMUP_ROUNDS = 3
_BENCHMARK_ROUNDS = 20
_SEED = 0


def _benchmark_device() -> torch.device:
    """Initialize context parallelism and return this rank's GPU."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        init_distributed()
    if dist.is_initialized():
        return torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(0)
    return torch.device("cuda", 0)


def _synchronize_ranks() -> None:
    """Align context-parallel ranks before a benchmark sample."""
    if dist.is_initialized():
        dist.barrier()


@pytest.mark.parametrize(
    "case",
    BENCHMARK_CASES,
    ids=lambda case: case.pytest_id,
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@torch.inference_mode()
def test_dit_network_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark the compiled 14B LingBot DiT at steady state."""
    device = _benchmark_device()
    if not torch.cuda.is_bf16_supported():
        pytest.skip("LingBot DiT network benchmark requires bfloat16 support")

    dtype = torch.bfloat16
    torch.manual_seed(_SEED)
    skip_unsupported_device(case, device)
    if (
        case.self_attention_backend is AttentionBackend.TRITON
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        pytest.skip("Triton attention does not support context parallelism")
    config = LingbotWorldDiTNetwork14BConfig(
        # 16 noise channels + 4 mask channels + 16 first-frame latent
        # channels before the DiT's 1x2x2 patch embedding.
        in_dim=16 + 4 + 16,
        patch_embedding_type="conv3d",
        control_type="cam",
        cp_method="ulysses",
        self_attention_backend=case.self_attention_backend,
        cross_attention_backend=case.cross_attention_backend,
        sdpa_backend=case.sdpa_backend,
        cross_attn_sdpa_backend=case.sdpa_backend,
        self_attn_qkv_fusion_option=case.self_attn_qkv_fusion_option,
        cross_attn_qkv_fusion_option=case.cross_attn_qkv_fusion_option,
        use_fp8=case.use_fp8,
    )

    # Avoid materializing the 14B random initialization as fp32 CPU weights.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device(device):
            network = LingbotWorldDiTNetwork(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    network.eval()
    network.update_parameters_after_loading_checkpoint()

    cp_size = dist.get_world_size() if dist.is_initialized() else 1
    cp_group = dist.group.WORLD if cp_size > 1 else None
    network.set_context_parallel_group(cp_group)
    assert all(
        block.self_attention_backend is case.self_attention_backend
        and block.cross_attention_backend is case.cross_attention_backend
        and block.sdpa_backend is case.sdpa_backend
        and block.cross_attn_sdpa_backend is case.sdpa_backend
        and block.self_attn_qkv_fusion_option is case.self_attn_qkv_fusion_option
        and block.cross_attn_qkv_fusion_option is case.cross_attn_qkv_fusion_option
        and block.use_fp8 is case.use_fp8
        for block in network.blocks
    )
    attention_modules = [
        module
        for module in network.modules()
        if isinstance(module, ContextParallelAttention)
    ]
    cudnn_attention_backends = {attention.backend for attention in attention_modules}
    assert cudnn_attention_backends == (
        {"cudnn"}
        if AttentionBackend.WAN
        in (case.self_attention_backend, case.cross_attention_backend)
        else set()
    )
    cp_enabled_attention_modules = [
        attention
        for attention in attention_modules
        if attention.is_context_parallel_enabled()
    ]
    assert all(
        attention.context_parallel_size() == cp_size
        for attention in cp_enabled_attention_modules
    )
    assert all(
        attention.method == config.cp_method
        for attention in cp_enabled_attention_modules
    )
    assert bool(cp_enabled_attention_modules) == (
        case.self_attention_backend is AttentionBackend.WAN and cp_size > 1
    )

    patch_t = _CHUNK_SIZE_T // config.patch_size[0]
    patch_h = _LATENT_HEIGHT // config.patch_size[1]
    patch_w = _LATENT_WIDTH // config.patch_size[2]
    patch_volume = config.patch_size[0] * config.patch_size[1] * config.patch_size[2]
    tokens_per_frame = patch_h * patch_w
    global_chunk_tokens = patch_t * tokens_per_frame
    global_window_tokens = _WINDOW_SIZE_T * tokens_per_frame
    global_sink_tokens = _SINK_SIZE_T * tokens_per_frame
    assert global_chunk_tokens % cp_size == 0
    assert global_window_tokens % cp_size == 0
    assert global_sink_tokens % cp_size == 0
    chunk_tokens = global_chunk_tokens // cp_size
    window_tokens = global_window_tokens // cp_size
    sink_tokens = global_sink_tokens // cp_size
    head_dim = config.dim // config.num_heads
    generator = torch.Generator(device=device).manual_seed(_SEED)

    x = torch.randn(
        (chunk_tokens, config.in_dim * patch_volume),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    control_channels = 6 if config.control_type == "cam" else 7
    plucker = torch.randn(
        (chunk_tokens, control_channels * 64 * patch_volume),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    timestep = torch.tensor(_DIFFUSION_TIMESTEP, device=device, dtype=dtype)
    text_embeddings = torch.randn(
        (1, _TEXT_TOKENS, config.text_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    cache = network.initialize_cache(
        chunk_size=chunk_tokens,
        window_size=window_tokens,
        sink_size=sink_tokens,
        text_embeddings=text_embeddings,
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=head_dim,
        len_h=patch_h,
        len_w=patch_w,
        len_t=patch_t,
        interleaved=True,
        device=device,
    )
    rope.set_context_parallel_group(cp_group)

    network = compile_module(network)
    capture_ar_index = cuda_graph_capture_ar_index(
        sink_size_t=_SINK_SIZE_T,
        window_size_t=_WINDOW_SIZE_T,
        len_t=_CHUNK_SIZE_T,
    )
    graph_dispatch = CUDAGraphDispatch(
        network,
        enabled=True,
        capture_ar_idx=capture_ar_index,
        warmup_iters=_CUDA_GRAPH_WARMUP_ITERS,
    )

    def forward(chunk_idx: int, rope_freqs: torch.Tensor) -> torch.Tensor:
        return graph_dispatch.select(chunk_idx, uncond=False)(
            plucker=plucker,
            x=x,
            timesteps=timestep,
            cache=cache,
            rope_freqs=rope_freqs,
            current_chunk_idx=chunk_idx,
            eager_mode=False,
        )

    # Fill the KV cache through the production graph threshold. At that final
    # index, emulate the four scheduler evaluations that warm, capture, and
    # replay the production CUDA graph before the benchmark begins.
    benchmark_ar_index = capture_ar_index + 1
    rope_freqs = [
        rope.shift_t(chunk_idx) for chunk_idx in range(benchmark_ar_index + 1)
    ]
    for chunk_idx in range(capture_ar_index + 1):
        cache.before_update(chunk_idx)
        output = forward(chunk_idx, rope_freqs[chunk_idx])
        if chunk_idx == capture_ar_index:
            for _ in range(_CUDA_GRAPH_WARMUP_ITERS + 1):
                output = forward(chunk_idx, rope_freqs[chunk_idx])
        cache.after_update(chunk_idx)
    torch.cuda.synchronize(device)

    benchmark.group = "lingbot-camctrl-dit-network"

    # Scheduler evaluations at one AR position repeatedly overwrite the same
    # cache slot. Cache finalization remains outside the measured callable.
    cache.before_update(benchmark_ar_index)

    def synchronized_forward() -> torch.Tensor:
        result = forward(benchmark_ar_index, rope_freqs[benchmark_ar_index])
        torch.cuda.synchronize(device)
        return result

    output = benchmark.pedantic(
        synchronized_forward,
        setup=_synchronize_ranks,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )
    cache.after_update(benchmark_ar_index)

    expected_output_shape = (chunk_tokens, config.out_dim * patch_volume)
    assert output.shape == expected_output_shape
    assert torch.isfinite(output).all()
