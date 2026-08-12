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

"""Cosmos and Wan parity tests for Torch multi-head attention."""

from __future__ import annotations

import pytest
import torch

from flashdreams.accelerated.multi_head_attention import QKNormScope
from flashdreams.accelerated.reference import TorchMultiHeadAttention
from flashdreams.core.attention import BlockKVCache, RotaryPositionEmbedding3D
from flashdreams.recipes.cosmos.transformer.impl import modules as cosmos_modules
from flashdreams.recipes.wan.transformer.impl import modules as wan_modules

pytestmark = pytest.mark.ci_gpu


@pytest.fixture(scope="module")
def cuda_device() -> torch.device:
    """Return the CUDA device used by fused RoPE and recipe attention."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required.")
    return torch.device("cuda")


def _make_cache(
    batch_size: int,
    chunk_size: int,
    n_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> BlockKVCache:
    """Build a two-chunk cache so the third update rolls the window."""
    window_size = 2 * chunk_size
    return BlockKVCache(
        k_shape=(batch_size, window_size, n_heads, head_dim),
        v_shape=(batch_size, window_size, n_heads, head_dim),
        seq_dim=1,
        chunk_size=chunk_size,
        window_size=window_size,
        device=device,
        dtype=dtype,
    )


def _assert_streaming_parity(
    recipe_attention: cosmos_modules.MultiHeadAttention
    | wan_modules.MultiHeadAttention,
    torch_attention: TorchMultiHeadAttention,
    *,
    rope_interleaved: bool,
    device: torch.device,
) -> None:
    """Compare outputs and visible cache contents across streaming phases."""
    # BF16 exercises the production cuDNN attention and fused RoPE paths while
    # keeping this focused parity test small.
    dtype = torch.bfloat16
    recipe_attention.to(device=device, dtype=dtype).eval()
    torch_attention.to(device=device, dtype=dtype).eval()

    # A ``1 x 8 x 8`` patch grid produces one 64-token chunk. The two-chunk
    # cache fills on steps 0 and 1, then rolls its local window on step 2.
    batch_size = 1
    len_t, len_h, len_w = 1, 8, 8
    chunk_size = len_t * len_h * len_w

    # Each implementation mutates its own cache so neither can mask an
    # incorrect update in the other implementation.
    recipe_cache = _make_cache(
        batch_size,
        chunk_size,
        torch_attention.n_heads,
        torch_attention.head_dim,
        device,
        dtype,
    )
    torch_cache = _make_cache(
        batch_size,
        chunk_size,
        torch_attention.n_heads,
        torch_attention.head_dim,
        device,
        dtype,
    )

    # Generate the same model-shaped 3D frequencies used by production. The
    # pairing convention is selected by the recipe-specific test case.
    rope = RotaryPositionEmbedding3D(
        head_dim=torch_attention.head_dim,
        len_t=len_t,
        len_h=len_h,
        len_w=len_w,
        interleaved=rope_interleaved,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(1234)

    with torch.inference_mode():
        for chunk_idx in range(3):
            # Reuse identical tokens and shifted positions for both paths.
            x = torch.randn(
                batch_size,
                chunk_size,
                torch_attention.query_dim,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            rope_freqs = rope.shift_t(chunk_idx)

            # Select the current write region, rolling the window first on the
            # third chunk, before either forward mutates cache storage.
            recipe_cache.before_update(chunk_idx)
            torch_cache.before_update(chunk_idx)

            expected = recipe_attention(x, recipe_cache, rope_freqs)
            actual = torch_attention(x, torch_cache, rope_freqs)

            # Output parity covers Q projection, attention, and output
            # projection. K/V parity isolates normalization, RoPE, and cache
            # updates; the wider K tolerance covers BF16 fused-RoPE rounding.
            torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
            torch.testing.assert_close(
                torch_cache.cached_k(),
                recipe_cache.cached_k(),
                atol=2e-2,
                rtol=2e-2,
            )
            torch.testing.assert_close(
                torch_cache.cached_v(),
                recipe_cache.cached_v(),
                atol=0,
                rtol=0,
            )

            # Commit bookkeeping only after comparing the visible state that
            # was consumed by this chunk's attention call.
            recipe_cache.after_update(chunk_idx)
            torch_cache.after_update(chunk_idx)


def test_torch_multi_head_attention_matches_cosmos(
    cuda_device: torch.device,
) -> None:
    """Compare the Torch reference with Cosmos streaming self-attention.

    Cosmos uses bias-free projections, per-head Q/K RMSNorm, and half-split
    RoPE pairs.
    """
    torch.manual_seed(7)

    # Configure the generic implementation with the exact Cosmos attention
    # geometry and projection policies.
    cosmos_attention = cosmos_modules.MultiHeadAttention(
        query_dim=256,
        n_heads=2,
        head_dim=128,
    )
    torch_attention = TorchMultiHeadAttention(
        query_dim=256,
        n_heads=2,
        head_dim=128,
        qkv_bias=False,
        output_bias=False,
        qk_norm_scope=QKNormScope.HEAD,
        rope_interleaved=False,
    )

    # Cosmos and the Torch reference share parameter names, so a strict state
    # dict load also verifies that their trainable structures align.
    torch_attention.load_state_dict(cosmos_attention.state_dict())

    # Exercise cache filling and rolling with real half-split FlashDreams RoPE.
    _assert_streaming_parity(
        cosmos_attention,
        torch_attention,
        rope_interleaved=False,
        device=cuda_device,
    )


def test_torch_multi_head_attention_matches_wan(
    cuda_device: torch.device,
) -> None:
    """Compare the Torch reference with Wan streaming self-attention.

    Wan uses biased projections, inner-width Q/K RMSNorm, and interleaved RoPE
    pairs.
    """
    torch.manual_seed(8)

    # Configure the generic implementation with Wan's attention geometry and
    # its model-specific normalization and RoPE policies.
    wan_attention = wan_modules.MultiHeadAttention(
        query_dim=256,
        n_heads=2,
        head_dim=128,
    )
    torch_attention = TorchMultiHeadAttention(
        query_dim=256,
        n_heads=2,
        head_dim=128,
        qkv_bias=True,
        output_bias=True,
        qk_norm_scope=QKNormScope.INNER,
        rope_interleaved=True,
    )

    # Wan uses q/k/v/o and norm_q/norm_k names, so copy each equivalent module
    # explicitly before comparing the forward paths.
    torch_attention.q_proj.load_state_dict(wan_attention.q.state_dict())
    torch_attention.k_proj.load_state_dict(wan_attention.k.state_dict())
    torch_attention.v_proj.load_state_dict(wan_attention.v.state_dict())
    torch_attention.output_proj.load_state_dict(wan_attention.o.state_dict())
    torch_attention.q_norm.load_state_dict(wan_attention.norm_q.state_dict())
    torch_attention.k_norm.load_state_dict(wan_attention.norm_k.state_dict())

    # Exercise cache filling and rolling with real interleaved FlashDreams RoPE.
    _assert_streaming_parity(
        wan_attention,
        torch_attention,
        rope_interleaved=True,
        device=cuda_device,
    )
