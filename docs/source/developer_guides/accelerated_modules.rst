.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
.. http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

Accelerated modules design plan
===============================

FlashDreams needs reusable acceleration at the level of architectural
components, rather than one-off kernels embedded in individual integrations.
An accelerated module is a composable PyTorch-facing implementation of a
well-defined behavior, such as attention or adaptive normalization. Triton
kernels and other optimized backends are private implementation details behind
that behavior.

This plan is based on an inspection of the ten in-tree integrations, the shared
WAN and Cosmos recipes, and the existing OmniDreams and FlashVSR acceleration
code. It identifies the common module boundaries, their required semantics, and
an adoption and validation sequence.

.. important::

   This document is a roadmap, not a description of a stable public API. The
   proposed package remains experimental, names may change during
   implementation, and the priority order is based on architecture and code
   inspection rather than new GPU measurements. Every optimization still needs
   profiling against the current cuDNN, PyTorch SDPA, ``torch.compile``, and
   CUDA-graph paths.

Goals and boundaries
--------------------

The accelerated-module layer should:

- keep reusable numerical behavior in ``flashdreams.core`` and preserve the
  dependency direction ``core -> infra -> recipes/integrations``;
- separate model semantics from backend selection, so an integration chooses
  normalization, positional encoding, cache, and masking behavior without
  choosing a specific kernel implementation;
- provide a correctness fallback for unsupported devices, shapes, dtypes, or
  distributed configurations;
- preserve checkpoint compatibility or supply an explicit state-dict adapter;
- compose into existing recipe blocks instead of introducing model-specific
  branches in ``core``; and
- treat BF16 as the reference path and FP8 as an opt-in inference policy with
  separate numerical and quality validation.

The public package should be named
``flashdreams.core.experimental.accelerated``. Public module wrappers and cache
state belong at that level, while Triton implementations should remain under a
private ``_kernels`` namespace. This makes it clear that users consume modules,
not kernel launch details.

Architecture survey
-------------------

The ten inspected integrations collapse into two transformer families: eight
are WAN-derived and two are Cosmos-derived. All inspected default DiT
configurations use head dimension 128, which makes an optimized 128-wide path a
high-value common specialization. Other head dimensions still require a generic
fallback.

.. list-table:: In-tree integration architecture
   :header-rows: 1
   :widths: 18 21 37 24

   * - Integration
     - Shared backbone
     - Integration-specific behavior
     - Encoder or decoder
   * - :doc:`Wan 2.1 </models/wan21>`
     - WAN 1.3B or 14B
     - Streaming self-attention, text cross-attention, and an optional
       independent CLIP-image attention branch
     - WAN VAE
   * - ``wan22``
     - WAN TI2V 5B
     - Per-token first-frame timestep conditioning
     - Residual 16x WAN VAE
   * - :doc:`Causal-Forcing </models/causal_forcing>`
     - WAN 1.3B
     - Framewise or three-frame autoregressive chunks
     - WAN VAE
   * - :doc:`Self-Forcing </models/self_forcing>`
     - WAN 1.3B
     - Sink/window caching and optional cache-relative RoPE
     - WAN VAE or TAEHV
   * - :doc:`FastVideo CausalWan 2.2 </models/causal_wan22>`
     - Two WAN 14B stacks
     - High- and low-timestep branch routing
     - WAN VAE
   * - :doc:`LingBot-World </models/lingbot_world>`
     - WAN 14B
     - Per-token camera and Plücker modulation
     - WAN VAE or TAEHV
   * - :doc:`HY-WorldPlay </models/hy_worldplay>`
     - WAN TI2V 5B
     - Action conditioning, a second PRoPE attention branch, and selected-memory
       K/V
     - WAN 2.2 VAE
   * - :doc:`FlashVSR </models/flashvsr>`
     - WAN 1.3B-like
     - 3D-window block-sparse attention and low-resolution feature injection
     - Causal projector, conditioned TAEHV, and AdaIN
   * - :doc:`Cosmos-Predict2.5 </models/cosmos_predict2>`
     - Cosmos 2B, 28 blocks
     - Per-head Q/K normalization and AdaLN-LoRA
     - WAN VAE
   * - :doc:`OmniDreams </models/omnidreams>`
     - Cosmos 2B-derived
     - View-aware execution, cross-view attention, and the current accelerated
       self-attention path
     - WAN VAE/LightVAE or PixelShuffle encoders; WAN VAE or TAEHV/LightTAE
       decoders

Despite their conditioning differences, both families repeat the same broad
transformer-block structure:

.. code-block:: text

   adaptive norm -> self-attention -> gated residual
   norm          -> cross-attention -> residual (optionally gated)
   adaptive norm -> MLP -> gated residual

This repeated structure defines the useful shared boundaries. Whole transformer
blocks are not interchangeable because LingBot injects camera modulation,
HY-WorldPlay adds a second projective-attention branch, FlashVSR injects
low-resolution features, and some variants require FP32 residual arithmetic.

Prioritized module roadmap
--------------------------

.. list-table:: Recommended accelerated modules
   :header-rows: 1
   :widths: 8 25 30 37

   * - Priority
     - Module
     - Initial consumers
     - Recommended boundary
   * - P0
     - ``AcceleratedAttention``
     - All inspected transformers
     - Separate projections, Q/K normalization, positional transforms, cache
       policy, attention backend, and output projection.
   * - P0
     - Pointer-stable K/V cache
     - Streaming WAN/Cosmos variants, static cross-attention, and HY-WorldPlay
       memory
     - Represent static, circular, sink, and selected-memory spans without
       rolling copies or materialized concatenation.
   * - P1
     - Adaptive norm and gated residual
     - Every inspected DiT
     - Fuse normalization, scale/shift modulation, residual gating, and optional
       dtype conversion while preserving each model's accumulation policy.
   * - P1
     - ``AcceleratedLinear`` and accelerated MLP
     - Attention projections, FFNs, conditioning MLPs, and output heads
     - Reuse vendor GEMMs or scaled GEMMs and fuse activation, quantization,
       gate, or residual epilogues.
   * - P1
     - Conditioning projection and cache
     - Cosmos-Predict2.5 and OmniDreams first; fixed-schedule models later
     - Stack AdaLN-LoRA projections and cache values that are invariant for a
       model weight set and scheduler timestep.
   * - P2
     - Streaming video-convolution primitives
     - WAN VAE users and FlashVSR's causal projector
     - Remove temporal-tail concatenation and padding traffic around vendor
       convolutions; fuse channel normalization and activation.
   * - P2
     - TAEHV temporal-memory primitive
     - Self-Forcing, LingBot, OmniDreams, and FlashVSR
     - Consume current and previous-frame channels without materializing the
       shifted concatenation before the first convolution.
   * - P2
     - Block-sparse attention backend
     - FlashVSR initially
     - Move the general Triton backend into core while leaving window and mask
       planning integration-specific.
   * - P3
     - Video layout operations
     - VAE, TAEHV, FlashVSR, and camera encoders
     - Fuse patchify, pixel shuffle, temporal growth, or window partition into
       adjacent projection, convolution, cache-write, or attention work.

Generalized attention
---------------------

Current state
~~~~~~~~~~~~~

The current
``flashdreams.core.experimental.accelerated_kernels.self_attention`` module is a
useful optimized OmniDreams path, but it is not yet a drop-in implementation for
the other families. It assumes that one input produces Q, K, and V; couples the
forward pass to a rolling self-attention cache; uses per-head Q/K RMSNorm; and
rejects context parallelism (CP).

The common attention implementation must represent the following differences
explicitly instead of inferring them from an integration name.

.. list-table:: Attention semantics that vary by family
   :header-rows: 1
   :widths: 21 34 45

   * - Axis
     - WAN family
     - Cosmos family and extensions
   * - Projection bias
     - Biased Q/K/V/output projections
     - Biasless projections in the base Cosmos blocks
   * - Q/K normalization
     - RMSNorm across the full inner width before reshaping into heads
     - RMSNorm independently over each head
   * - RoPE convention
     - Interleaved; some Self-Forcing configurations rotate cached K at read
       time
     - Half-split for the base path; no RoPE for text or cross-view attention
   * - Cache lifecycle
     - Rolling self-attention and precomputed static cross-attention K/V
     - Rolling or one-step self-attention, static text K/V, and uncached
       cross-view attention
   * - Additional branches
     - Independent text and image softmaxes in WAN I2V; HY-WorldPlay's separate
       PRoPE branch
     - OmniDreams cross-view attention

Proposed decomposition
~~~~~~~~~~~~~~~~~~~~~~

``AcceleratedAttention`` should be a convenience wrapper over smaller,
independently usable operations:

#. **Projection layer.** Provide self-attention fused QKV projection, separate
   query and context projection, and output projection. Support different query
   and context dimensions and both biased and biasless checkpoints.
#. **Projected attention operator.** Consume projected
   ``[..., query_length, heads, head_dim]`` Q and
   ``[..., key_length, kv_heads, head_dim]`` K/V with explicit masking,
   causality, scaling, and backend selection. This operator is also the reuse
   point for VAE and custom multi-branch attention.
#. **Q/K post-processing.** Select no normalization, per-head normalization, or
   full-inner-width normalization. Apply half-split or interleaved RoPE, or
   accept tensors transformed externally by specialized integrations.
#. **Cache policy.** Accept no cache, static projected K/V, a rolling cache, or
   a segmented view. Cache mutation must remain separate from the mathematical
   attention operator.
#. **Backend dispatch.** Select PyTorch/cuDNN, dense Triton, context-parallel,
   or block-sparse execution by capability. Unsupported combinations must fall
   back predictably instead of changing semantics.

The wrapper should make query projection, K/V projection, attention over
projected tensors, and output projection independently callable. WAN I2V can
then run text and image attention separately, sum the raw attention results, and
apply one output projection. HY-WorldPlay can keep separate standard-RoPE and
PRoPE output projections. Neither policy needs a model-specific branch in
``core``.

Context parallelism
~~~~~~~~~~~~~~~~~~~

CP support is required before the WAN recipe can adopt the module broadly. The
first implementation should keep the existing context-parallel attention path
as the fallback while still accelerating compatible projection,
normalization, RoPE, and cache operations. A future local Triton attention
backend may participate directly in ring attention if it returns the local
log-sum-exp values required for cross-rank merging.

The module must not own the process-group topology. Integrations continue to
construct and pass their CP group; :doc:`/api/core` documents the current
shared attention surface.

RoPE
~~~~

Standard RoPE should reuse ``flashdreams.core.attention.apply_rope_freqs`` and
its existing Triton path rather than introduce a second standalone accelerated
module. Fused projection/norm/RoPE/cache-write kernels may specialize this
behavior, but their output must remain equivalent to the shared operator.

Pointer-stable K/V cache
------------------------

The current ``BlockKVCache`` physically clones and shifts retained K/V tensors
when the rolling window advances. This is simple, but it moves the full retained
window for every transformer block at steady state. HY-WorldPlay additionally
materializes selected-memory/current-K/V concatenations and changes tensor
pointers, which prevents stable CUDA-graph capture.

The proposed cache layer should expose a common logical K/V view with:

- preallocated, pointer-stable storage;
- a logical write index and valid length;
- sink plus circular-window spans;
- optional read-only static or selected-memory spans;
- BF16 and optional FP8 K/V storage;
- repeated overwrite of the current autoregressive chunk throughout the
  scheduler loop, followed by one explicit finalize operation; and
- attention consumption of one or more physical spans without ``torch.cat`` or
  a cache-wide copy.

The cache lifecycle must preserve the existing distinction between filling,
the final filling step, the first wrap, and steady-state execution. It must also
support Self-Forcing's cache-relative RoPE policy, where unrotated K is stored
and rotated using logical slot positions at read time.

DiT elementwise and linear modules
----------------------------------

Adaptive normalization and gated residual
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shared primitive should cover:

- non-affine or affine LayerNorm followed by scale/shift modulation;
- scalar modulation shaped per sample and per-token modulation shaped per
  sequence element;
- learned modulation tables used by WAN and low-rank modulation generated by
  Cosmos;
- ``residual + gate * branch`` with native-dtype or FP32 accumulation; and
- an optional fused transition from one branch's gated residual directly into
  the next branch's normalization and modulation.

The old OmniDreams native implementation uses these same fusion boundaries,
including direct FP8 output variants. That is useful implementation precedent,
but it is not evidence that every fusion outperforms a compiled PyTorch graph.

Accelerated linear and MLP
~~~~~~~~~~~~~~~~~~~~~~~~~~

The packed FP8 linear helper currently private to accelerated self-attention
should become a reusable inference-only linear module. It should retain BF16
fallbacks and support checkpoint-compatible weight and bias handling.

An accelerated MLP should compose vendor or scaled GEMMs with configurable
epilogues:

- exact GELU for Cosmos or tanh-approximate GELU for WAN;
- optional bias;
- activation followed by FP8 scaling for the next GEMM; and
- gate/residual fusion after the output GEMM.

The initial objective is not to replace general GEMM with Triton. The useful
optimization boundary is eliminating intermediate reads, writes, launches, and
dtype conversions around a high-performance GEMM implementation.

Conditioning projection and caching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cosmos and OmniDreams execute three AdaLN-LoRA modulation paths per block. A
stacked projection can reduce launch overhead and share optimized linear
epilogues. Fixed scheduler timesteps also make the resulting modulation tensors
reusable across autoregressive chunks and conditional/unconditional passes.

Caching must be keyed by the effective model weights, dtype/device, and
scheduler timestep. Per-token WAN 2.2 conditioning is a separate policy: when a
sequence contains only a small number of distinct timestep values, compute the
distinct embeddings and select or broadcast them rather than projecting every
token independently.

Video encoder and decoder modules
---------------------------------

Streaming causal convolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WAN VAE and FlashVSR's causal projector repeatedly concatenate a cached
temporal tail, pad it, call Conv3d, and update the tail. The first reusable
accelerated boundary should be a pointer-stable temporal-halo adapter that
writes current input and cached context into reusable storage, invokes cuDNN
convolution, and updates the next tail without extra materialization.

The contract must parameterize padding mode, kernel, stride, dilation, and cache
lifecycle. A general Triton Conv3d should only replace cuDNN after shape-specific
benchmarks demonstrate an advantage.

Channel normalization and temporal memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WAN VAE residual blocks and FlashVSR's projector repeatedly apply channel-first
RMSNorm followed by SiLU. This is a compact reduction/activation fusion that
should support FP32 reduction and optional affine parameters.

TAEHV ``MemBlock`` shifts the prior frame, concatenates it with the current
frame, and applies three Conv2d layers. A temporal-memory input adapter or fused
first convolution can avoid the materialized concatenation while retaining
cuDNN for the convolution itself.

Layout transformations
~~~~~~~~~~~~~~~~~~~~~~

Common transformations include VAE space-to-depth/depth-to-space, TAEHV
temporal growth, FlashVSR ``PixelShuffle3d`` and window partition/reverse, and
camera-feature unshuffle. These operations are bandwidth-bound when isolated.
Add them to core only when they can be fused into an adjacent projection,
convolution, cache write, or attention kernel. Channel order, temporal padding,
and first-chunk trimming must remain explicit parameters.

Specialized backends
--------------------

FlashVSR already contains a substantial Triton attention backend with dense,
streaming, block-sparse, mixed, and variable-length modes. The backend and its
architecture/dtype dispatch should move into core for shared maintenance. Its
fixed 3D windows, top-k draft-mask planner, batch-size-one assumptions, and
integration operator namespace should remain in FlashVSR until another consumer
establishes a general planning contract.

HY-WorldPlay's PRoPE transform is a worthwhile integration-local optimization:
projection matrices can be computed once per forward or autoregressive chunk
and reused across blocks, followed by a possible fused transform/cache-write
kernel. LingBot's Plücker conditioning is semantically different and should not
be forced into the same camera abstraction.

Non-goals
---------

The first accelerated-module API should not include:

- a monolithic ``AcceleratedDiTBlock`` because branch structure, conditioning,
  and precision policy differ across integrations;
- a monolithic accelerated VAE because WAN VAE, TAEHV, and FlashVSR use
  different convolution and cache topologies;
- a duplicate standalone RoPE implementation;
- custom general GEMM or Conv3d kernels solely to make the stack Triton-only;
- scheduler kernels, which perform little work relative to denoising;
- standalone timestep-embedding or patchify kernels before profiling shows
  them to be material; or
- a single camera-conditioning module for LingBot and HY-WorldPlay.

Implementation and adoption sequence
------------------------------------

Phase 0: establish baselines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Select representative Cosmos/OmniDreams, WAN 1.3B, WAN 14B, WAN 5B,
   FlashVSR, and decoder-only configurations.
#. Record fixed prompts/inputs, seeds, resolution, autoregressive chunk and
   window sizes, checkpoint identifiers, GPU/software versions, dtype, compile
   mode, and CUDA-graph mode.
#. Measure startup and warmup separately from cache filling and steady-state
   execution. Capture median and p90 stage timings with CUDA events.
#. Attribute projection, Q/K processing, attention, cache movement, output
   projection, MLP, DiT elementwise work, encode, and decode before selecting
   kernel work.

Phase 1: decompose existing attention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Move the public surface to
   ``flashdreams.core.experimental.accelerated`` and keep a compatibility
   re-export for ``AcceleratedSelfAttention`` during migration.
#. Extract the projected attention operator and packed FP8 linear helper.
#. Add self-attention, one-source cross-attention, and static projected-K/V
   behavior while preserving the current OmniDreams path.
#. Adopt the module in OmniDreams and Cosmos-Predict2.5 first because their
   per-head normalization and RoPE conventions most closely match the current
   implementation.

Phase 2: WAN semantics and cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Add full-inner-width Q/K normalization, projection biases, interleaved and
   cache-relative RoPE, and independent multi-source composition.
#. Preserve the existing context-parallel backend as the correctness fallback.
#. Introduce pointer-stable circular and segmented K/V views and migrate cache
   update semantics before replacing the mathematical attention backend.
#. Adopt the module in the shared WAN recipe. This automatically reaches most
   WAN-family integrations while specialized branches continue to compose the
   projected operator directly.

Phase 3: repeated block fusions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Add adaptive normalization and gated residual primitives.
#. Add reusable accelerated linear/MLP epilogues and stacked conditioning
   projections.
#. Evaluate fixed-timestep conditioning caches with CFG and CUDA-graph capture.

Phase 4: video and specialized paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Profile WAN VAE and TAEHV independently with fixed latent inputs.
#. Implement temporal-halo, channel-norm/activation, and temporal-memory
   primitives only for measured hot stages.
#. Extract the FlashVSR sparse-attention backend while keeping its planner local.
#. Optimize HY-WorldPlay's PRoPE setup and segmented-memory consumption after
   the common cache interface is stable.

Validation plan
---------------

Correctness and compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Compare against the existing WAN and Cosmos recipe modules at
  ``D/H = 1536/12``, ``2048/16``, ``3072/24``, and ``5120/40``. All have head
  dimension 128.
- Cover single-frame, three-frame, and full-window query lengths, 512-token text
  context, optional image context, unequal query/key lengths, and arbitrary
  leading batch or view dimensions.
- Test biased and biasless projections; no, per-head, and full-width Q/K norm;
  exact and tanh-approximate GELU; and half-split, interleaved, external, and
  cache-relative positional transforms.
- Exercise cold allocation, filling, repeated scheduler overwrite, the final
  filling step, first wrap, steady-state ring behavior, sinks, static context,
  and multiple read-only memory spans.
- Verify independent conditional/unconditional and dual-network caches.
- Preserve state-dict keys where practical and require explicit conversion tests
  for any incompatible checkpoint layout.

Distributed and runtime behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Compare single-GPU output with context-parallel sizes 2 and 4 for ring and
  Ulysses paths.
- Validate eager, ``torch.compile``, and CUDA-graph execution with stable cache
  pointers and fresh per-rollout cache construction.
- Test capability-based fallback on CPU, unsupported GPU architectures,
  unsupported head dimensions, and unsupported dtype/backend combinations.
- Mark every pytest test with exactly one of ``ci_cpu``, ``ci_gpu``, or
  ``manual`` as required by the repository test policy.

Performance and quality
~~~~~~~~~~~~~~~~~~~~~~~

- Benchmark projection, post-processing/cache write, attention, output
  projection, MLP, and decoder primitives separately before reporting an
  end-to-end result.
- Report cold/startup, cache-filling, and steady-state timings separately, with
  warmup excluded from headline median and p90 measurements.
- Compare raw attention against the existing cuDNN/SDPA path; do not assume a
  standalone Triton attention kernel is faster.
- Run same-latent decoder comparisons for decoder changes and deterministic
  tensor parity before autoregressive rollout smoke tests.
- Gate FP8 adoption on both numerical parity tolerances and rollout-quality
  comparisons against the BF16 reference.

Acceptance criteria
-------------------

An accelerated module is ready for shared adoption when:

- its semantic contract has at least two real consumers or a clearly generic
  projected-tensor use case;
- unsupported cases take a documented correctness fallback;
- reference outputs, cache lifecycle, state-dict loading, CP, compile, and
  CUDA-graph tests pass;
- benchmarks show where the gain comes from and include the relevant fallback
  baseline; and
- recipes select behavior through generic options or composition, without an
  integration-specific import or branch in ``core``.

Related documentation
---------------------

- :doc:`/developer_guides/inference_pipeline_overview` explains where
  transformer, cache, and decoder work sits in an autoregressive rollout.
- :doc:`/developer_guides/new_integration` describes how integrations consume
  reusable core and recipe components.
- :doc:`/api/core` documents the current shared attention and cache APIs.
- :doc:`/models/omnidreams` describes the existing native acceleration path.
- :doc:`/models/flashvsr` describes the current sparse-attention integration.
