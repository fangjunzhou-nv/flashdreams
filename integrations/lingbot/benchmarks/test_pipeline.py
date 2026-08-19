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

"""Steady-state full-pipeline benchmarks for LingBot streaming inference.

Run the manual GPU benchmarks with::

    uv run --package flashdreams-lingbot --group test pytest \
        integrations/lingbot/benchmarks/test_pipeline.py \
        -p no:manual_marker -m manual --benchmark-only -v
"""

from __future__ import annotations

import os
from typing import Literal

import pytest
import torch
import torch.distributed as dist
from lingbot.config import (
    PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3,
)
from lingbot.encoder.camctrl import CamCtrlInput, I2VCamCtrlEncoderConfig
from lingbot.pipeline import LingbotWorldInferencePipeline
from lingbot.transformer import (
    LingbotWorldTransformer,
    LingbotWorldTransformerConfig,
)
from lingbot.transformer.impl.network import LingbotWorldDiTNetwork
from pytest_benchmark.fixture import BenchmarkFixture

from flashdreams.core.attention import ContextParallelAttention
from flashdreams.core.distributed import init as init_distributed
from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.scheduler.fm import (
    FlowMatchSchedulerConfig,
)
from flashdreams.infra.pipeline import StreamInferencePipeline
from flashdreams.recipes.taehv import TeahvVAEDecoderConfig
from flashdreams.recipes.wan.autoencoder.vae import WanVAEEncoderConfig
from flashdreams.recipes.wan.pipeline import WanInferencePipelineCache
from flashdreams.recipes.wan.transformer.impl.modules import AttentionBackend
from integrations.lingbot.benchmarks.cases import (
    BENCHMARK_CASES,
    AttentionBenchmarkCase,
    skip_unsupported_device,
)

pytestmark = pytest.mark.manual

_GPU_REASON = "LingBot full-pipeline benchmark requires CUDA"

_PIXEL_HEIGHT = 352
_PIXEL_WIDTH = 640
_TEXT_TOKENS = 512
_WARMUP_ROUNDS = 3
_BENCHMARK_ROUNDS = 20
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
    """Align context-parallel ranks before a benchmark sample."""
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
    """Benchmark steady-state LingBot encode, diffuse, and decode."""
    _run_full_pipeline_benchmark(benchmark, case=case, stage="generate")


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize(
    "case",
    BENCHMARK_CASES,
    ids=lambda case: case.pytest_id,
)
def test_full_pipeline_finalize_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark the LingBot DiT cache-finalization update."""
    _run_full_pipeline_benchmark(benchmark, case=case, stage="finalize")


@torch.inference_mode()
def _run_full_pipeline_benchmark(
    benchmark: BenchmarkFixture,
    *,
    case: AttentionBenchmarkCase,
    stage: Literal["generate", "finalize"],
) -> None:
    """Run one full-pipeline lifecycle-stage benchmark."""
    device = _benchmark_device()
    if not torch.cuda.is_bf16_supported():
        pytest.skip("LingBot full-pipeline benchmark requires bfloat16 support")
    _skip_unsupported_case(case, device)

    torch.manual_seed(_SEED)
    torch.backends.cudnn.benchmark = True

    # UMT5 is a one-shot rollout initializer, so use a correctly shaped
    # precomputed embedding. The recurring pipeline remains production-like:
    # LingBot camera rendering/control, four-step DiT diffusion, and TAEHV
    # decoding. Initializing the 14B DiT directly on this rank's GPU avoids a
    # transient fp32 CPU copy and does not affect timed steady-state stages.
    pipeline_config = derive_config(
        PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3,
        name=f"lingbot-world-v2-full-pipeline-{case.implementation}-benchmark",
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
    assert isinstance(pipeline, LingbotWorldInferencePipeline)
    pipeline.eval()
    assert pipeline.encoder is not None
    assert pipeline.decoder is not None

    context_parallel_attention_modules = [
        module
        for module in pipeline.modules()
        if isinstance(module, ContextParallelAttention)
    ]
    context_parallel_attention_backends = {
        attention.backend for attention in context_parallel_attention_modules
    }
    assert context_parallel_attention_backends == (
        {"cudnn"}
        if AttentionBackend.WAN
        in (case.self_attention_backend, case.cross_attention_backend)
        else set()
    )

    diffusion_config = pipeline_config.diffusion_model
    transformer_config = diffusion_config.transformer
    scheduler_config = diffusion_config.scheduler
    encoder_config = pipeline_config.encoder
    decoder_config = pipeline_config.decoder
    assert isinstance(transformer_config, LingbotWorldTransformerConfig)
    assert isinstance(scheduler_config, FlowMatchSchedulerConfig)
    assert isinstance(encoder_config, I2VCamCtrlEncoderConfig)
    assert isinstance(encoder_config.i2v.encoder, WanVAEEncoderConfig)
    assert isinstance(decoder_config, TeahvVAEDecoderConfig)
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

    transformer = pipeline.diffusion_model.transformer
    assert isinstance(transformer, LingbotWorldTransformer)
    assert transformer.config is transformer_config
    network = getattr(transformer.network, "_orig_mod", transformer.network)
    assert isinstance(network, LingbotWorldDiTNetwork)
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
    text_dim = int(transformer_config.network.text_dim)

    text_embeddings = torch.zeros(
        (1, _TEXT_TOKENS, text_dim),
        device=device,
        dtype=dtype,
    )
    image = torch.zeros(
        (1, 3, _PIXEL_HEIGHT, _PIXEL_WIDTH),
        device=device,
        dtype=dtype,
    )

    # Bypass only Wan's raw-text one-shot initializer. The base cache builder
    # still creates the real recurring encoder, transformer, and decoder
    # caches, and the Wan cache wrapper retains the first frame for I2V.
    parent_cache = StreamInferencePipeline.initialize_cache(
        pipeline,
        transformer_context={
            "height": latent_height,
            "width": latent_width,
            "text_embeddings": text_embeddings,
            "negative_text_embeddings": None,
            "image_embeddings": None,
        },
    )
    cache = WanInferencePipelineCache(
        transformer_cache=parent_cache.transformer_cache,
        encoder_cache=parent_cache.encoder_cache,
        decoder_cache=parent_cache.decoder_cache,
        image=image,
    )
    del text_embeddings

    first_chunk_frames = pipeline.get_num_input_frames(0)
    steady_input_frames = pipeline.get_num_input_frames(1)
    steady_output_frames = pipeline.get_num_output_frames(1)

    def camera_input(num_frames: int) -> CamCtrlInput:
        intrinsics = torch.tensor(
            [416.0, 416.0, _PIXEL_WIDTH / 2, _PIXEL_HEIGHT / 2],
            device=device,
            dtype=torch.float32,
        ).repeat(num_frames, 1)
        poses = torch.eye(4, device=device, dtype=torch.float32).repeat(
            num_frames, 1, 1
        )
        return CamCtrlInput(
            intrinsics=intrinsics,
            poses=poses,
            world_scale=1.0,
        )

    first_camera_input = camera_input(first_chunk_frames)
    steady_camera_input = camera_input(steady_input_frames)

    def run_chunk(
        autoregressive_index: int,
        input: CamCtrlInput,
    ) -> torch.Tensor:
        output = pipeline.generate(
            autoregressive_index=autoregressive_index,
            cache=cache,
            input=input,
        )
        pipeline.finalize(
            autoregressive_index=autoregressive_index,
            cache=cache,
        )
        return output

    # Fill the sink/window cache through the first CUDA-graph index. This also
    # advances the reused Wan I2V encoder past its first five real VAE calls;
    # steady-state timed rounds reuse its cached latent and measure only the
    # LingBot camera-control work on that branch.
    capture_ar_index = transformer._cuda_graph_capture_ar_idx
    cache_prefill_chunks = capture_ar_index + 1
    for autoregressive_index in range(cache_prefill_chunks):
        chunk_input = (
            first_camera_input if autoregressive_index == 0 else steady_camera_input
        )
        output = run_chunk(autoregressive_index, chunk_input)
    torch.cuda.synchronize(device)

    benchmark.group = f"lingbot-full-pipeline-{stage}"

    next_chunk_index = cache_prefill_chunks
    latest_output: torch.Tensor | None = None
    if stage == "generate":

        def setup_generate() -> None:
            _synchronize_ranks()

        def synchronized_generate() -> torch.Tensor:
            nonlocal latest_output
            latest_output = pipeline.generate(
                autoregressive_index=next_chunk_index,
                cache=cache,
                input=steady_camera_input,
            )
            torch.cuda.synchronize(device)
            return latest_output

        def teardown_generate() -> None:
            nonlocal next_chunk_index
            pipeline.finalize(
                autoregressive_index=next_chunk_index,
                cache=cache,
            )
            torch.cuda.synchronize(device)
            next_chunk_index += 1

        output = benchmark.pedantic(
            synchronized_generate,
            setup=setup_generate,
            teardown=teardown_generate,
            iterations=1,
            rounds=_BENCHMARK_ROUNDS,
            warmup_rounds=_WARMUP_ROUNDS,
        )
    else:

        def setup_finalize() -> None:
            nonlocal latest_output
            _synchronize_ranks()
            latest_output = pipeline.generate(
                autoregressive_index=next_chunk_index,
                cache=cache,
                input=steady_camera_input,
            )
            torch.cuda.synchronize(device)
            _synchronize_ranks()

        def synchronized_finalize() -> None:
            pipeline.finalize(
                autoregressive_index=next_chunk_index,
                cache=cache,
            )
            torch.cuda.synchronize(device)

        def teardown_finalize() -> None:
            nonlocal next_chunk_index
            next_chunk_index += 1

        benchmark.pedantic(
            synchronized_finalize,
            setup=setup_finalize,
            teardown=teardown_finalize,
            iterations=1,
            rounds=_BENCHMARK_ROUNDS,
            warmup_rounds=_WARMUP_ROUNDS,
        )
        output = latest_output

    assert output is not None
    assert output.shape == (
        steady_output_frames,
        3,
        _PIXEL_HEIGHT,
        _PIXEL_WIDTH,
    )
    assert torch.isfinite(output).all()
