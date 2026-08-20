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

"""Microbenchmark for the LingBot-owned camera-control DiT block.

Run all attention cases with ``uv run --package flashdreams-lingbot
--group test pytest integrations/lingbot/benchmarks/test_modules.py
-p no:manual_marker -m manual --benchmark-only``.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from lingbot.transformer.impl.modules import CamCtrlBlock
from lingbot.transformer.impl.network import LingbotWorldDiTNetwork14BConfig
from pytest_benchmark.fixture import BenchmarkFixture

from flashdreams.accelerated.multi_head_attention_triton import (
    QKVFusionOption,
    SDPABackend,
)
from flashdreams.core.attention.rope import RotaryPositionEmbedding3D
from flashdreams.core.distributed import init as init_distributed
from flashdreams.recipes.wan.transformer.impl.modules import AttentionBackend
from integrations.lingbot.benchmarks.cases import (
    AttentionBenchmarkCase,
    skip_unsupported_device,
)

pytestmark = pytest.mark.manual

_GPU_REASON = "LingBot DiT block benchmark requires CUDA"

# CLI replay geometry: 352x640 pixels become 44x80 Wan
# latents. The DiT consumes three latent frames per chunk; its window15/sink3
# preset retains six chunks in total while bounding cache memory.
_PIXEL_HEIGHT = 352
_PIXEL_WIDTH = 640
_LATENT_HEIGHT = 44
_LATENT_WIDTH = 80
_CHUNK_SIZE_T = 3
_WINDOW_SIZE_T = 15
_SINK_SIZE_T = 3
_TEXT_TOKENS = 512
_WARMUP_ROUNDS = 3
_BENCHMARK_ROUNDS = 20
_SEED = 0

_MODULE_CASE_MATRIX = [
    AttentionBenchmarkCase(
        implementation="wan_torch",
        self_attention_backend=AttentionBackend.WAN,
        cross_attention_backend=AttentionBackend.WAN,
        sdpa_backend=SDPABackend.CUDNN,
        use_fp8=False,
        self_attn_qkv_fusion_option=QKVFusionOption.NONE,
        cross_attn_qkv_fusion_option=QKVFusionOption.NONE,
    ),
    *[
        AttentionBenchmarkCase(
            implementation=(
                f"triton_{sdpa_backend.value}_"
                f"{'fp8' if use_fp8 else 'bf16'}_{qkv_fusion_option.value}"
            ),
            self_attention_backend=AttentionBackend.TRITON,
            cross_attention_backend=AttentionBackend.TRITON,
            sdpa_backend=sdpa_backend,
            use_fp8=use_fp8,
            self_attn_qkv_fusion_option=qkv_fusion_option,
            cross_attn_qkv_fusion_option=(
                qkv_fusion_option
                if qkv_fusion_option is not QKVFusionOption.FULL
                else QKVFusionOption.FUSE_KV
            ),
            minimum_compute_capability=(
                (9, 0) if use_fp8 or sdpa_backend is not SDPABackend.CUDNN else None
            ),
        )
        for sdpa_backend in SDPABackend
        for use_fp8 in (False, True)
        for qkv_fusion_option in QKVFusionOption
    ],
    AttentionBenchmarkCase(
        implementation="triton_cudnn_bf16_full_wan_cross",
        self_attention_backend=AttentionBackend.TRITON,
        cross_attention_backend=AttentionBackend.WAN,
        sdpa_backend=SDPABackend.CUDNN,
        use_fp8=False,
        cross_attn_qkv_fusion_option=QKVFusionOption.NONE,
    ),
]


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


def _make_block(
    config: LingbotWorldDiTNetwork14BConfig,
    case: AttentionBenchmarkCase,
    device: torch.device,
    dtype: torch.dtype,
) -> CamCtrlBlock:
    """Build a backend-selected block with shared random weights."""

    def make(
        self_backend: AttentionBackend, cross_backend: AttentionBackend
    ) -> CamCtrlBlock:
        return CamCtrlBlock(
            dim=config.dim,
            ffn_dim=config.ffn_dim,
            num_heads=config.num_heads,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps,
            cp_method=config.cp_method,
            attention_backend=self_backend,
            self_attention_backend=self_backend,
            cross_attention_backend=cross_backend,
            sdpa_backend=config.sdpa_backend,
            cross_attn_sdpa_backend=config.cross_attn_sdpa_backend,
            self_attn_qkv_fusion_option=config.self_attn_qkv_fusion_option,
            cross_attn_qkv_fusion_option=config.cross_attn_qkv_fusion_option,
            use_fp8=config.use_fp8,
        )

    # Allocate this 14B-sized block directly in BF16 on the target GPU. This
    # changes setup memory only; initialization and checkpoint loading remain
    # outside the measured region.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device(device):
            torch.manual_seed(_SEED)
            reference = make(AttentionBackend.WAN, AttentionBackend.WAN)
            if (
                case.self_attention_backend is AttentionBackend.WAN
                and case.cross_attention_backend is AttentionBackend.WAN
            ):
                return reference
            block = make(case.self_attention_backend, case.cross_attention_backend)
            block.load_state_dict(reference.state_dict(), strict=True)
            return block
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize(
    "case",
    _MODULE_CASE_MATRIX,
    ids=lambda case: case.pytest_id,
)
@torch.inference_mode()
def test_camctrl_dit_block_benchmark(
    benchmark: BenchmarkFixture,
    case: AttentionBenchmarkCase,
) -> None:
    """Benchmark the CLI-resolution LingBot camera-control DiT block."""
    device = _benchmark_device()
    if not torch.cuda.is_bf16_supported():
        pytest.skip("LingBot DiT block benchmark requires bfloat16 support")

    dtype = torch.bfloat16
    config = LingbotWorldDiTNetwork14BConfig(
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
    cp_size = dist.get_world_size() if dist.is_initialized() else 1
    skip_unsupported_device(case, device)
    if case.self_attention_backend is AttentionBackend.TRITON and cp_size > 1:
        pytest.skip("Triton attention does not support context parallelism")
    block = _make_block(config, case, device, dtype)
    block.eval()
    block.update_parameters_after_loading_checkpoint()
    assert block.self_attention_backend is case.self_attention_backend
    assert block.cross_attention_backend is case.cross_attention_backend
    assert block.sdpa_backend is case.sdpa_backend
    assert block.cross_attn_sdpa_backend is case.sdpa_backend
    assert block.self_attn_qkv_fusion_option is case.self_attn_qkv_fusion_option
    assert block.cross_attn_qkv_fusion_option is case.cross_attn_qkv_fusion_option
    assert block.use_fp8 is case.use_fp8
    generator = torch.Generator(device=device).manual_seed(_SEED)

    cp_group = dist.group.WORLD if cp_size > 1 else None
    block.set_context_parallel_group(cp_group)
    self_attention_cp_enabled = block.self_attn.is_context_parallel_enabled()
    cross_attention_cp_enabled = block.cross_attn.is_context_parallel_enabled()
    assert self_attention_cp_enabled == (
        case.self_attention_backend is AttentionBackend.WAN and cp_size > 1
    )
    assert not cross_attention_cp_enabled

    patch_t = _CHUNK_SIZE_T // config.patch_size[0]
    patch_h = _LATENT_HEIGHT // config.patch_size[1]
    patch_w = _LATENT_WIDTH // config.patch_size[2]
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

    x = torch.randn(
        (chunk_tokens, config.dim),
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
    plucker_embedding = torch.randn(
        x.shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    context = torch.randn(
        (1, _TEXT_TOKENS, config.dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    cache = block.initialize_cache(
        chunk_size=chunk_tokens,
        window_size=window_tokens,
        sink_size=sink_tokens,
        context_text=context,
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

    def forward(chunk_idx: int, rope_freqs: torch.Tensor) -> torch.Tensor:
        cache.before_update(chunk_idx)
        result = block(
            x=x,
            e=modulation,
            cache=cache,
            rope_freqs=rope_freqs,
            plucker_embedding=plucker_embedding,
        )
        cache.after_update(chunk_idx)
        return result

    # Fill the fixed sink and rolling window before timing. Prepare the next
    # rolling slot once, then repeatedly overwrite it to mirror denoising at
    # one autoregressive position without timing cache bookkeeping.
    cache_prefill_chunks = (_SINK_SIZE_T + _WINDOW_SIZE_T) // _CHUNK_SIZE_T
    benchmark_ar_index = cache_prefill_chunks
    rope_freqs = [rope.shift_t(idx) for idx in range(benchmark_ar_index + 1)]
    for chunk_idx in range(cache_prefill_chunks):
        output = forward(chunk_idx, rope_freqs[chunk_idx])
    torch.cuda.synchronize(device)

    benchmark.group = "lingbot-camctrl-dit-block"

    cache.before_update(benchmark_ar_index)
    torch.cuda.synchronize(device)

    def synchronized_forward() -> torch.Tensor:
        result = block(
            x=x,
            e=modulation,
            cache=cache,
            rope_freqs=rope_freqs[benchmark_ar_index],
            plucker_embedding=plucker_embedding,
        )
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

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
