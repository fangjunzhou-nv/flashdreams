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

"""CPU coverage for Omnidreams DiT attention backend selection."""

import pytest
import torch
from omnidreams.transformer.impl import modules as transformer_modules
from omnidreams.transformer.impl.modules import AttentionBackend, Block
from omnidreams.transformer.impl.network import CosmosDiTNetwork, CosmosDiTNetworkConfig

from flashdreams.accelerated.multi_head_attention_torch import TorchMultiHeadAttention
from flashdreams.accelerated.multi_head_attention_triton import TritonMultiHeadAttention

pytestmark = pytest.mark.ci_cpu


def test_dit_attention_backend_defaults_to_omnidreams() -> None:
    """Keep existing Omnidreams attention as the default."""
    default_block = Block(x_dim=12, context_dim=8, num_heads=1)

    assert default_block.attention_backend is AttentionBackend.OMNIDREAMS
    assert isinstance(default_block.self_attn, transformer_modules.SelfAttention)
    assert isinstance(default_block.cross_attn, transformer_modules.CrossAttention)


def test_network_config_selects_triton_attention() -> None:
    """Select FP8 Triton self-, text cross-, and cross-view attention."""
    config = CosmosDiTNetworkConfig(
        model_channels=16,
        num_blocks=1,
        num_heads=1,
        crossattn_emb_channels=8,
        use_crossattn_projection=False,
        enable_cross_view_attn=True,
        attention_backend=AttentionBackend.TRITON,
    )

    network = CosmosDiTNetwork(config)
    block = network.blocks[0]

    self_attention = block.self_attn
    assert isinstance(self_attention, TritonMultiHeadAttention)
    assert self_attention.use_fp8 is True
    assert self_attention._fused_qkv_weight is not None
    assert self_attention._fused_qkv_weight.dtype == torch.float8_e4m3fn
    assert self_attention._output_weight_fp8 is not None
    assert self_attention._output_weight_fp8.dtype == torch.float8_e4m3fn
    cache = self_attention.initialize_cache(
        batch_size=1,
        chunk_size=2,
        window_size=4,
        sink_size=0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert cache.dtype == torch.float8_e4m3fn
    with pytest.raises(TypeError, match="FP8 attention requires"):
        self_attention.initialize_cache(
            batch_size=1,
            chunk_size=2,
            window_size=4,
            sink_size=0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    assert isinstance(block.cross_attn, transformer_modules._TritonCrossAttention)
    assert isinstance(block.cross_view_attn, transformer_modules._TritonCrossAttention)


def test_triton_backend_preserves_checkpoint_keys() -> None:
    """Load Omnidreams weights into the Triton block strictly."""
    omnidreams_block = Block(x_dim=16, context_dim=8, num_heads=1)
    triton_block = Block(
        x_dim=16,
        context_dim=8,
        num_heads=1,
        attention_backend=AttentionBackend.TRITON,
    )

    triton_block.load_state_dict(omnidreams_block.state_dict(), strict=True)


def test_triton_cross_attention_dispatches_tma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch projected cross-attention tensors through the TMA operator."""
    torch.manual_seed(0)
    triton_block = Block(
        x_dim=16,
        context_dim=8,
        num_heads=1,
        attention_backend=AttentionBackend.TRITON,
    )
    triton_cross_attention = triton_block.cross_attn
    assert isinstance(triton_cross_attention, transformer_modules._TritonCrossAttention)

    calls: list[tuple[torch.Size, torch.Size, torch.Size]] = []

    def record_tma_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        calls.append((query.shape, key.shape, value.shape))
        return TorchMultiHeadAttention._attention(
            triton_cross_attention,
            query,
            key,
            value,
        )

    monkeypatch.setattr(
        transformer_modules,
        "flash_attention_2_tma",
        record_tma_attention,
    )
    x = torch.randn(1, 2, 3, 16)
    context = torch.randn(1, 2, 5, 8)

    triton_output = triton_cross_attention(
        x,
        triton_cross_attention.initialize_cache(context),
    )

    assert calls == [
        (
            torch.Size([2, 3, 1, 16]),
            torch.Size([2, 5, 1, 16]),
            torch.Size([2, 5, 1, 16]),
        )
    ]
    assert triton_output.shape == x.shape
    assert torch.isfinite(triton_output).all()
