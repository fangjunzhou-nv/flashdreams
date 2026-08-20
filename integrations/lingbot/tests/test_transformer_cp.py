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

"""Context-parallel patchify smoke test for the Lingbot World transformer."""

import pytest
import torch
from lingbot.encoder.camctrl import I2VCamCtrlEmbeddings
from lingbot.transformer import (
    LingbotWorldTransformer,
    LingbotWorldTransformerConfig,
)
from lingbot.transformer.impl.network import (
    LingbotWorldDiTNetwork,
    LingbotWorldDiTNetworkConfig,
)

from flashdreams.accelerated.multi_head_attention_triton import (
    QKVFusionOption,
    SDPABackend,
    TritonMultiHeadAttention,
)
from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrl
from flashdreams.recipes.wan.transformer.impl.modules import (
    AttentionBackend,
    CrossAttention,
    SelfAttention,
)

pytestmark = pytest.mark.ci_cpu


def test_lingbot_patchify_marks_i2v_and_plucker_as_patchified() -> None:
    transformer = LingbotWorldTransformer(
        LingbotWorldTransformerConfig(
            network=LingbotWorldDiTNetworkConfig(
                dim=64,
                ffn_dim=128,
                num_heads=4,
                num_layers=1,
                patch_embedding_type="linear",
                control_type="cam",
            ),
            batch_shape=(),
            len_t=2,
            window_size_t=2,
            sink_size_t=0,
            compile_network=False,
        )
    )
    assert transformer.network.attention_backend is AttentionBackend.WAN

    camctrl_embeddings = I2VCamCtrlEmbeddings(
        i2v=I2VCtrl(
            latent=torch.randn(2, 16, 4, 4),
            mask=torch.randn(2, 16, 4, 4),
        ),
        plucker=torch.randn(2, 6 * 64, 4, 4),
    )

    patched = transformer.patchify_and_maybe_split_cp(camctrl_embeddings)
    assert isinstance(patched, I2VCamCtrlEmbeddings)
    assert patched._is_patchified
    assert patched.i2v._is_patchified
    assert patched.i2v.latent.shape == (8, 64)
    assert patched.i2v.mask.shape == (8, 64)
    assert patched.plucker.shape == (8, 1536)

    # Idempotent once marked patchified.
    assert transformer.patchify_and_maybe_split_cp(patched) is patched


@pytest.mark.parametrize(
    ("self_backend", "cross_backend", "sdpa_backend"),
    [
        pytest.param(
            AttentionBackend.WAN,
            AttentionBackend.WAN,
            SDPABackend.CUDNN,
            id="wan",
        ),
        pytest.param(
            AttentionBackend.TRITON,
            AttentionBackend.TRITON,
            SDPABackend.CUDNN,
            id="triton-cudnn",
        ),
        pytest.param(
            AttentionBackend.TRITON,
            AttentionBackend.TRITON,
            SDPABackend.FA2,
            id="triton-fa2",
        ),
        pytest.param(
            AttentionBackend.TRITON,
            AttentionBackend.WAN,
            SDPABackend.CUDNN,
            id="triton-self-wan-cross",
        ),
        pytest.param(
            AttentionBackend.WAN,
            AttentionBackend.TRITON,
            SDPABackend.CUDNN,
            id="wan-self-triton-cross",
        ),
    ],
)
def test_lingbot_network_propagates_attention_backends(
    self_backend: AttentionBackend,
    cross_backend: AttentionBackend,
    sdpa_backend: SDPABackend,
) -> None:
    """Propagate configured attention backends into camera-control blocks."""
    network = LingbotWorldDiTNetwork(
        LingbotWorldDiTNetworkConfig(
            dim=64,
            ffn_dim=128,
            num_heads=4,
            num_layers=1,
            patch_embedding_type="linear",
            control_type="cam",
            self_attention_backend=self_backend,
            cross_attention_backend=cross_backend,
            sdpa_backend=sdpa_backend,
        )
    )

    assert network.attention_backend is self_backend
    assert network.self_attention_backend is self_backend
    assert network.cross_attention_backend is cross_backend
    assert network.sdpa_backend is sdpa_backend
    assert len(network.blocks) == 1
    block = network.blocks[0]
    assert block.attention_backend is self_backend
    assert block.self_attention_backend is self_backend
    assert block.cross_attention_backend is cross_backend
    assert block.sdpa_backend is sdpa_backend
    assert isinstance(
        block.self_attn,
        TritonMultiHeadAttention
        if self_backend is AttentionBackend.TRITON
        else SelfAttention,
    )
    assert isinstance(
        block.cross_attn,
        TritonMultiHeadAttention
        if cross_backend is AttentionBackend.TRITON
        else CrossAttention,
    )
    if self_backend is AttentionBackend.TRITON:
        assert isinstance(block.self_attn, TritonMultiHeadAttention)
        assert block.self_attn.sdpa_backend is sdpa_backend
        cache = block.self_attn.allocate_kv_cache(
            batch_size=1,
            chunk_size=2,
            window_size=4,
            sink_size=0,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        assert cache.dtype is (
            torch.float8_e4m3fn if sdpa_backend is SDPABackend.FA2 else torch.bfloat16
        )


@pytest.mark.parametrize(
    "sdpa_backend", tuple(SDPABackend), ids=lambda backend: backend.value
)
@pytest.mark.parametrize("use_fp8", (False, True), ids=("bf16", "fp8"))
@pytest.mark.parametrize(
    "qkv_fusion_option",
    tuple(QKVFusionOption),
    ids=lambda option: option.value.replace("_", "-"),
)
def test_lingbot_network_propagates_attention_options(
    sdpa_backend: SDPABackend,
    use_fp8: bool,
    qkv_fusion_option: QKVFusionOption,
) -> None:
    """Propagate accelerated attention options into camera-control blocks."""
    cross_attn_qkv_fusion_option = (
        qkv_fusion_option
        if qkv_fusion_option is not QKVFusionOption.FULL
        else QKVFusionOption.FUSE_KV
    )
    network = LingbotWorldDiTNetwork(
        LingbotWorldDiTNetworkConfig(
            dim=64,
            ffn_dim=128,
            num_heads=4,
            num_layers=1,
            patch_embedding_type="linear",
            control_type="cam",
            self_attention_backend=AttentionBackend.TRITON,
            cross_attention_backend=AttentionBackend.TRITON,
            sdpa_backend=sdpa_backend,
            cross_attn_sdpa_backend=sdpa_backend,
            self_attn_qkv_fusion_option=qkv_fusion_option,
            cross_attn_qkv_fusion_option=cross_attn_qkv_fusion_option,
            use_fp8=use_fp8,
        )
    )

    block = network.blocks[0]
    assert block.sdpa_backend is sdpa_backend
    assert block.cross_attn_sdpa_backend is sdpa_backend
    assert block.self_attn_qkv_fusion_option is qkv_fusion_option
    assert block.cross_attn_qkv_fusion_option is cross_attn_qkv_fusion_option
    assert block.use_fp8 is use_fp8
    assert isinstance(block.self_attn, TritonMultiHeadAttention)
    assert isinstance(block.cross_attn, TritonMultiHeadAttention)
    assert block.self_attn.qkv_fusion_option is qkv_fusion_option
    assert block.cross_attn.qkv_fusion_option is cross_attn_qkv_fusion_option
    assert block.self_attn.use_fp8 is block.cross_attn.use_fp8 is use_fp8
