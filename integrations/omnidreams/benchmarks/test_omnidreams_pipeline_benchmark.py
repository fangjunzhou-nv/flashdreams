# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU benchmarks for the main stages of :class:`OmnidreamsPipeline`.

Run from the FlashDreams workspace root after installing the OmniDreams dev
dependencies and exporting an ``HF_TOKEN`` with model access::

    uv run --no-sync --package flashdreams-omnidreams pytest \
      integrations/omnidreams/benchmarks/test_omnidreams_pipeline_benchmark.py \
      --runxfail --benchmark-only \
      --benchmark-json=/tmp/omnidreams-pipeline-benchmark.json

The module fixture performs one complete two-chunk rollout before measurement,
excluding checkpoint loading, compilation, CUDA graph capture, and autotuning.
Benchmark tables report all durations in milliseconds.
Set ``OMNIDREAMS_PIPELINE_BENCHMARK_ROUNDS`` to override the default five
measured rounds per benchmark.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

import pytest
import torch
from omnidreams.config import SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE
from omnidreams.pipeline import (
    OmnidreamsPipeline,
    OmnidreamsPipelineCache,
    OmnidreamsPipelineConfig,
)

from flashdreams.infra.config import derive_config

pytestmark = pytest.mark.manual

_BASE_CONFIG = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
        name="omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf",
        image_encoder=dict(use_compile=True, use_cuda_graph=True),
        encoder=dict(use_compile=True, use_cuda_graph=True),
        decoder=dict(use_compile=True, use_cuda_graph=True),
    ),
)

_PERF_CONFIG = cast(
    OmnidreamsPipelineConfig,
    derive_config(
        _BASE_CONFIG,
        enable_sync_and_profile=False,
        diffusion_model=dict(
            transformer=dict(
                compile_network=True,
                skip_finalize_kv_cache=True,
                native_dit_acceleration="required",
                native_dit_backend="fp8_kvcache_cudnn",
                native_dit_attention_backend="cudnn",
            ),
            scheduler=dict(
                denoising_timesteps=[1000, 100],
                num_inference_steps=2,
            ),
        ),
    ),
)


@dataclass(frozen=True)
class _PipelineBenchmarkCase:
    name: str
    config: OmnidreamsPipelineConfig
    width: int
    height: int
    device: str = "cuda"


_CASES = (
    _PipelineBenchmarkCase(
        name="base",
        config=_BASE_CONFIG,
        width=1280,
        height=704,
    ),
    _PipelineBenchmarkCase(
        name="perf",
        config=_PERF_CONFIG,
        width=1168,
        height=640,
        device="cuda:0",
    ),
)
_NUM_VIEWS = 1
_PROMPT = "A wide-angle dash-cam view of a suburban road in daylight."
_ROUNDS_ENV = "OMNIDREAMS_PIPELINE_BENCHMARK_ROUNDS"


def _benchmark_rounds() -> int:
    rounds = int(os.environ.get(_ROUNDS_ENV, "5"))
    if rounds < 1:
        raise ValueError(f"{_ROUNDS_ENV} must be at least 1, got {rounds}")
    return rounds


@pytest.fixture(scope="module", autouse=True)
def _report_benchmarks_in_milliseconds(request: pytest.FixtureRequest) -> None:
    request.config.option.benchmark_time_unit = "ms"


@dataclass(frozen=True)
class _PipelineBenchmarkContext:
    case: _PipelineBenchmarkCase
    config: OmnidreamsPipelineConfig
    pipeline: OmnidreamsPipeline
    image: torch.Tensor
    text: list[list[str]]
    hdmaps: tuple[torch.Tensor, torch.Tensor]
    device: torch.device
    width: int
    height: int


def _synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _initialize_cache(
    context: _PipelineBenchmarkContext,
) -> OmnidreamsPipelineCache:
    return context.pipeline.initialize_cache(text=context.text, image=context.image)


def _prepare_cache_for_index(
    context: _PipelineBenchmarkContext,
    autoregressive_index: int,
) -> OmnidreamsPipelineCache:
    cache = _initialize_cache(context)
    for index in range(autoregressive_index):
        context.pipeline.generate(
            index,
            cache=cache,
            hdmap=context.hdmaps[index],
        )
        context.pipeline.finalize(index, cache=cache)
    _synchronize(context.device)
    return cache


def _set_benchmark_metadata(
    benchmark: Any,
    context: _PipelineBenchmarkContext,
    *,
    stage: str,
    autoregressive_index: int | None = None,
) -> None:
    config = context.config
    transformer = config.diffusion_model.transformer
    image_encoder = config.image_encoder
    hdmap_encoder = config.encoder
    decoder = config.decoder
    benchmark.group = context.case.name
    benchmark.extra_info.update(
        {
            "stage": stage,
            "config": config.name,
            "checkpoint": str(getattr(transformer, "checkpoint_path", None)),
            "resolution": f"{context.width}x{context.height}",
            "num_views": _NUM_VIEWS,
            "dtype": str(context.image.dtype),
            "device": torch.cuda.get_device_name(context.device),
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "cudnn_version": torch.backends.cudnn.version(),
            "rounds": _benchmark_rounds(),
            "warmup": "one complete two-chunk rollout",
            "transformer_compile": bool(
                getattr(transformer, "compile_network", False)
            ),
            "transformer_cuda_graph": bool(
                getattr(transformer, "use_cuda_graph", False)
            ),
            "skip_finalize_kv_cache": bool(
                getattr(transformer, "skip_finalize_kv_cache", False)
            ),
            "native_dit_acceleration": str(
                getattr(transformer, "native_dit_acceleration", "disabled")
            ),
            "native_dit_backend": str(
                getattr(transformer, "native_dit_backend", "disabled")
            ),
            "native_dit_attention_backend": str(
                getattr(transformer, "native_dit_attention_backend", "auto")
            ),
            "denoising_timesteps": list(
                getattr(config.diffusion_model.scheduler, "denoising_timesteps", [])
            ),
            "image_encoder_compile": bool(
                getattr(image_encoder, "use_compile", False)
            ),
            "image_encoder_cuda_graph": bool(
                getattr(image_encoder, "use_cuda_graph", False)
            ),
            "hdmap_encoder_compile": bool(
                getattr(hdmap_encoder, "use_compile", False)
            ),
            "hdmap_encoder_cuda_graph": bool(
                getattr(hdmap_encoder, "use_cuda_graph", False)
            ),
            "decoder_compile": bool(getattr(decoder, "use_compile", False)),
            "decoder_cuda_graph": bool(getattr(decoder, "use_cuda_graph", False)),
            "native_vae_acceleration": str(
                getattr(image_encoder, "native_vae_acceleration", "disabled")
            ),
            "native_vae_backend": str(
                getattr(image_encoder, "native_vae_backend", "disabled")
            ),
        }
    )
    if autoregressive_index is not None:
        benchmark.extra_info.update(
            {
                "autoregressive_index": autoregressive_index,
                "num_frames": context.pipeline.get_num_frames(autoregressive_index),
            }
        )


@pytest.fixture(
    scope="module",
    params=[pytest.param(case, id=case.name) for case in _CASES],
)
def pipeline_benchmark_context(
    request: pytest.FixtureRequest,
) -> Iterator[_PipelineBenchmarkContext]:
    case = cast(_PipelineBenchmarkCase, request.param)
    config = case.config
    width, height = case.width, case.height
    device = torch.device(case.device)

    if not torch.cuda.is_available():
        pytest.skip("OmniDreams pipeline benchmarks require CUDA")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    dtype = torch.bfloat16

    pipeline = config.setup().to(device)
    assert isinstance(pipeline, OmnidreamsPipeline)

    image = torch.randn(
        1,
        _NUM_VIEWS,
        1,
        3,
        height,
        width,
        device=device,
        dtype=dtype,
    )
    hdmaps = tuple(
        torch.randn(
            1,
            _NUM_VIEWS,
            pipeline.get_num_frames(index),
            3,
            height,
            width,
            device=device,
            dtype=dtype,
        )
        for index in (0, 1)
    )
    context = _PipelineBenchmarkContext(
        case=case,
        config=config,
        pipeline=pipeline,
        image=image,
        text=[[_PROMPT] * _NUM_VIEWS],
        hdmaps=(hdmaps[0], hdmaps[1]),
        device=device,
        width=width,
        height=height,
    )

    cache = _initialize_cache(context)
    for index, hdmap in enumerate(context.hdmaps):
        pipeline.generate(index, cache=cache, hdmap=hdmap)
        pipeline.finalize(index, cache=cache)
    _synchronize(device)

    yield context

    del cache, context, hdmaps, image, pipeline
    torch.cuda.empty_cache()


def test_initialize_cache(
    benchmark: Any,
    pipeline_benchmark_context: _PipelineBenchmarkContext,
) -> None:
    context = pipeline_benchmark_context
    cache: OmnidreamsPipelineCache | None = None
    _set_benchmark_metadata(benchmark, context, stage="initialize_cache")

    def initialize_cache() -> None:
        nonlocal cache
        cache = _initialize_cache(context)
        _synchronize(context.device)

    def release_cache() -> None:
        nonlocal cache
        cache = None

    benchmark.pedantic(
        initialize_cache,
        teardown=release_cache,
        rounds=_benchmark_rounds(),
        iterations=1,
    )


@pytest.mark.parametrize("autoregressive_index", [0, 1], ids=["initial", "steady"])
def test_generate(
    benchmark: Any,
    pipeline_benchmark_context: _PipelineBenchmarkContext,
    autoregressive_index: int,
) -> None:
    context = pipeline_benchmark_context
    output: torch.Tensor | None = None
    _set_benchmark_metadata(
        benchmark,
        context,
        stage="generate",
        autoregressive_index=autoregressive_index,
    )

    def setup() -> tuple[tuple[OmnidreamsPipelineCache], dict[str, object]]:
        return ((_prepare_cache_for_index(context, autoregressive_index),), {})

    def generate(cache: OmnidreamsPipelineCache) -> None:
        nonlocal output
        output = context.pipeline.generate(
            autoregressive_index,
            cache=cache,
            hdmap=context.hdmaps[autoregressive_index],
        )
        _synchronize(context.device)

    def finalize_and_release(cache: OmnidreamsPipelineCache) -> None:
        nonlocal output
        context.pipeline.finalize(autoregressive_index, cache=cache)
        _synchronize(context.device)
        output = None

    benchmark.pedantic(
        generate,
        setup=setup,
        teardown=finalize_and_release,
        rounds=_benchmark_rounds(),
        iterations=1,
    )


@pytest.mark.parametrize("autoregressive_index", [0, 1], ids=["initial", "steady"])
def test_finalize(
    benchmark: Any,
    pipeline_benchmark_context: _PipelineBenchmarkContext,
    autoregressive_index: int,
) -> None:
    context = pipeline_benchmark_context
    _set_benchmark_metadata(
        benchmark,
        context,
        stage="finalize",
        autoregressive_index=autoregressive_index,
    )

    def setup() -> tuple[tuple[OmnidreamsPipelineCache], dict[str, object]]:
        cache = _prepare_cache_for_index(context, autoregressive_index)
        context.pipeline.generate(
            autoregressive_index,
            cache=cache,
            hdmap=context.hdmaps[autoregressive_index],
        )
        _synchronize(context.device)
        return ((cache,), {})

    def finalize(cache: OmnidreamsPipelineCache) -> None:
        context.pipeline.finalize(autoregressive_index, cache=cache)
        _synchronize(context.device)

    benchmark.pedantic(
        finalize,
        setup=setup,
        rounds=_benchmark_rounds(),
        iterations=1,
    )
