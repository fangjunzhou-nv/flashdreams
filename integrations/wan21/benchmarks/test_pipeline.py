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

"""One-shot full-pipeline benchmarks for the shipped Wan 2.1 T2V runner.

Run the manual GPU benchmarks with::

    uv run --package flashdreams-wan21 --group test pytest \
        integrations/wan21/benchmarks/test_pipeline.py \
        -p no:manual_marker -m manual --benchmark-only -v
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from pytest_benchmark.fixture import BenchmarkFixture
from wan21.config import PIPELINE_WAN21_T2V_1PT3B_480P

from flashdreams.core.attention import ContextParallelAttention
from flashdreams.core.distributed import init as init_distributed
from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.scheduler import (
    FlowMatchUniPCScheduler,
    FlowMatchUniPCSchedulerConfig,
)
from flashdreams.infra.encoder.text.umt5 import UMT5TextEncoderConfig
from flashdreams.infra.pipeline import StreamInferencePipeline
from flashdreams.recipes.wan import (
    Wan21Transformer,
    Wan21TransformerConfig,
    WanDiTNetwork,
    WanInferencePipeline,
    WanInferencePipelineCache,
    WanVAEDecoderConfig,
)
from flashdreams.recipes.wan.transformer.impl.modules import AttentionBackend
from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerCache
from integrations.wan21.benchmarks.cases import (
    BENCHMARK_CASES,
    AttentionBenchmarkCase,
    skip_unsupported_device,
)

pytestmark = pytest.mark.manual

_GPU_REASON = "Wan 2.1 full-pipeline benchmark requires CUDA"

_PIXEL_HEIGHT = 480
_PIXEL_WIDTH = 832
_TEXT_TOKENS = 512
_COMPONENT_PREWARM_ROUNDS = 3
_WARMUP_ROUNDS = 0
_BENCHMARK_ROUNDS = 1
_SEED = 42


def _benchmark_device() -> torch.device:
    """Initialize context parallelism and return this rank's GPU."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        init_distributed()
    if dist.is_initialized():
        return torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(0)
    return torch.device("cuda", 0)


def _synchronize_ranks() -> None:
    """Align context-parallel ranks outside a benchmark sample."""
    if dist.is_initialized():
        dist.barrier()


def _skip_unsupported_case(
    case: AttentionBenchmarkCase,
    device: torch.device,
) -> None:
    """Skip a case where its hardware or context-parallel contract is unmet."""
    skip_unsupported_device(case, device)
    if case.self_attention_backend is not AttentionBackend.TRITON:
        return
    world_size = (
        dist.get_world_size()
        if dist.is_initialized()
        else int(os.environ.get("WORLD_SIZE", "1"))
    )
    if world_size > 1:
        pytest.skip("Triton attention does not support context parallelism")


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize(
    "case",
    BENCHMARK_CASES,
    ids=lambda case: case.pytest_id,
)
def test_full_pipeline_generate_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark the shipped one-shot Wan 2.1 denoise and decode path."""
    _run_full_pipeline_benchmark(benchmark, case=case)


@torch.inference_mode()
def _run_full_pipeline_benchmark(
    benchmark: BenchmarkFixture,
    *,
    case: AttentionBenchmarkCase,
) -> None:
    """Run one backend through the production pipeline."""
    device = _benchmark_device()
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Wan 2.1 full-pipeline benchmark requires bfloat16 support")
    _skip_unsupported_case(case, device)

    torch.manual_seed(_SEED)
    torch.backends.cudnn.benchmark = True

    # UMT5 is a one-shot rollout initializer. Use correctly shaped synthetic
    # embeddings so setup measures the checkpoint-backed recurring pipeline:
    # 50-step CFG diffusion followed by the production Wan VAE decoder.
    pipeline_config = derive_config(
        PIPELINE_WAN21_T2V_1PT3B_480P,
        name=f"wan21-t2v-1.3b-full-pipeline-{case.implementation}-benchmark",
        text_encoder=None,
        enable_sync_and_profile=False,
        diffusion_model={
            "seed": _SEED,
            "transformer": {
                "init_device": str(device),
                "network": {
                    "self_attention_backend": case.self_attention_backend,
                    "cross_attention_backend": case.cross_attention_backend,
                    "sdpa_backend": case.sdpa_backend,
                    "cross_attn_sdpa_backend": case.sdpa_backend,
                    "self_attn_qkv_fusion_option": (case.self_attn_qkv_fusion_option),
                    "cross_attn_qkv_fusion_option": (case.cross_attn_qkv_fusion_option),
                    "use_fp8": case.use_fp8,
                },
            },
        },
    )
    pipeline = pipeline_config.setup().to(device=device)
    assert isinstance(pipeline, WanInferencePipeline)
    pipeline.eval()
    assert pipeline.encoder is None
    assert pipeline.decoder is not None

    source_text_encoder_config = PIPELINE_WAN21_T2V_1PT3B_480P.text_encoder
    assert isinstance(source_text_encoder_config, UMT5TextEncoderConfig)

    diffusion_config = pipeline_config.diffusion_model
    transformer_config = diffusion_config.transformer
    scheduler_config = diffusion_config.scheduler
    decoder_config = pipeline_config.decoder
    assert isinstance(transformer_config, Wan21TransformerConfig)
    assert isinstance(scheduler_config, FlowMatchUniPCSchedulerConfig)
    assert isinstance(decoder_config, WanVAEDecoderConfig)
    assert (
        transformer_config.network.self_attention_backend is case.self_attention_backend
    )
    assert transformer_config.network.cross_attention_backend is (
        case.cross_attention_backend
    )
    assert transformer_config.network.sdpa_backend is case.sdpa_backend
    assert transformer_config.network.cross_attn_sdpa_backend is case.sdpa_backend
    assert (
        transformer_config.network.self_attn_qkv_fusion_option
        is case.self_attn_qkv_fusion_option
    )
    assert (
        transformer_config.network.cross_attn_qkv_fusion_option
        is case.cross_attn_qkv_fusion_option
    )
    assert transformer_config.network.use_fp8 is case.use_fp8
    assert transformer_config.batch_shape == ()
    assert transformer_config.len_t == 21
    assert transformer_config.window_size_t == 21
    assert transformer_config.sink_size_t == 0
    assert transformer_config.guidance_scale == 6.0
    assert scheduler_config.num_inference_steps == 50

    transformer = pipeline.diffusion_model.transformer
    scheduler = pipeline.diffusion_model.scheduler
    assert isinstance(transformer, Wan21Transformer)
    assert isinstance(scheduler, FlowMatchUniPCScheduler)
    network = getattr(transformer.network, "_orig_mod", transformer.network)
    assert isinstance(network, WanDiTNetwork)
    assert network.blocks
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

    context_parallel_attention_modules = [
        module
        for module in pipeline.modules()
        if isinstance(module, ContextParallelAttention)
    ]
    assert {attention.backend for attention in context_parallel_attention_modules} == (
        {"cudnn"}
        if AttentionBackend.WAN
        in (case.self_attention_backend, case.cross_attention_backend)
        else set()
    )
    cp_size = transformer._cp_size
    cp_enabled_attention_modules = [
        attention
        for attention in context_parallel_attention_modules
        if attention.is_context_parallel_enabled()
    ]
    assert all(
        attention.context_parallel_size() == cp_size
        for attention in cp_enabled_attention_modules
    )
    assert all(
        attention.method == transformer_config.network.cp_method
        for attention in cp_enabled_attention_modules
    )
    assert bool(cp_enabled_attention_modules) == (
        case.self_attention_backend is AttentionBackend.WAN and cp_size > 1
    )

    dtype = transformer_config.dtype
    spatial_compression = int(pipeline.decoder.spatial_compression_ratio)
    latent_height = _PIXEL_HEIGHT // spatial_compression
    latent_width = _PIXEL_WIDTH // spatial_compression
    latent_channels = int(transformer_config.network.out_dim)
    text_dim = int(transformer_config.network.text_dim)
    output_frames = pipeline.decoder.get_output_temporal_size(
        0, transformer_config.len_t
    )

    text_embeddings = torch.zeros(
        (1, _TEXT_TOKENS, text_dim),
        device=device,
        dtype=dtype,
    )
    negative_text_embeddings = torch.zeros_like(text_embeddings)

    def initialize_cache() -> WanInferencePipelineCache:
        """Build one production-shaped T2V cache outside sample timing."""
        parent_cache = StreamInferencePipeline.initialize_cache(
            pipeline,
            transformer_context={
                "height": latent_height,
                "width": latent_width,
                "text_embeddings": text_embeddings,
                "negative_text_embeddings": negative_text_embeddings,
                "image_embeddings": None,
            },
        )
        cache = WanInferencePipelineCache(
            transformer_cache=parent_cache.transformer_cache,
            encoder_cache=parent_cache.encoder_cache,
            decoder_cache=parent_cache.decoder_cache,
            image=None,
        )
        assert isinstance(cache.transformer_cache, Wan21TransformerCache)
        assert cache.transformer_cache.network_cache_uncond is not None
        return cache

    # Prime lazy compilation and cuDNN selection without paying for another
    # complete 50-step rollout. Fresh measured caches reset the graph wrappers,
    # so their per-rollout AR0 input staging remains inside the timed call.
    probe_cache = initialize_cache()
    assert isinstance(probe_cache.transformer_cache, Wan21TransformerCache)

    probe_transformer_cache = probe_cache.transformer_cache
    probe_transformer_cache.start(0)
    prewarm_latent = torch.zeros(
        transformer.latent_shape,
        device=device,
        dtype=dtype,
    )
    prewarm_timestep = scheduler.timesteps[0].to(device=device, dtype=dtype)
    prewarm_flow = transformer.predict_flow(
        noisy_latent=prewarm_latent,
        timestep=prewarm_timestep,
        cache=probe_transformer_cache,
    )
    for _ in range(1, _COMPONENT_PREWARM_ROUNDS):
        prewarm_flow = transformer.predict_flow(
            noisy_latent=prewarm_latent,
            timestep=prewarm_timestep,
            cache=probe_transformer_cache,
        )
    probe_transformer_cache.finalize(0)

    assert probe_cache.decoder_cache is not None
    prewarm_decode_input = torch.zeros(
        (
            transformer_config.len_t,
            latent_channels,
            latent_height,
            latent_width,
        ),
        device=device,
        dtype=dtype,
    )
    prewarm_decode_output = pipeline.decoder(
        input=prewarm_decode_input,
        autoregressive_index=0,
        cache=probe_cache.decoder_cache,
    )

    torch.cuda.synchronize(device)
    del prewarm_flow, prewarm_latent, prewarm_timestep
    del prewarm_decode_input
    del prewarm_decode_output
    probe_cache = None

    benchmark.group = "wan21-full-pipeline-generate"

    cache: WanInferencePipelineCache | None = None
    latest_output: torch.Tensor | None = None

    def prepare_sample() -> None:
        nonlocal cache, latest_output
        _synchronize_ranks()
        # Drop the prior cache before allocating the next multi-GiB KV cache,
        # allowing the CUDA allocator to reuse its storage without a peak at 2x.
        transformer._cuda_graph_dispatch.reset()
        latest_output = None
        cache = None
        cache = initialize_cache()
        rng = pipeline.diffusion_model.rng
        assert rng is not None
        rng.manual_seed(_SEED)
        torch.cuda.synchronize(device)
        _synchronize_ranks()

    def setup_generate() -> None:
        prepare_sample()

    def synchronized_generate() -> torch.Tensor:
        nonlocal latest_output
        assert cache is not None
        latest_output = pipeline.generate(autoregressive_index=0, cache=cache)
        torch.cuda.synchronize(device)
        return latest_output

    def teardown_generate() -> None:
        assert cache is not None
        pipeline.finalize(autoregressive_index=0, cache=cache)
        torch.cuda.synchronize(device)

    output = benchmark.pedantic(
        synchronized_generate,
        setup=setup_generate,
        teardown=teardown_generate,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )

    assert output is not None
    assert output.shape == (
        output_frames,
        3,
        _PIXEL_HEIGHT,
        _PIXEL_WIDTH,
    )
    assert torch.isfinite(output).all()
