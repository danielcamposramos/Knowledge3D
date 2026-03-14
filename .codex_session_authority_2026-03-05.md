# Codex Session Authority Notes

Date: 2026-03-05

Purpose: hidden working note anchoring this session to `CODEX.md` and the authoritative specifications under `docs/vocabulary/`.

## Source Set

- `CODEX.md`
- `docs/vocabulary/README.md`
- All `.md` files directly under `docs/vocabulary/`

## Primary Working Constraints

- `CODEX.md` is the implementation lead contract.
- Architecture authority lives in `docs/vocabulary/` and current specs in `TEMP/`.
- Hot path must remain sovereign: PTX + Galaxy only, no numpy/cupy/scipy/torch in inference path.
- Ingestion path can be flexible, but outputs must resolve into Galaxy/Knowledgeverse-compatible structures.
- Prefer Galaxy population over Python hardcoding.
- Pair implementation with tests and sovereignty checks.
- Read latest briefing before architecture-sensitive work.

## Core Vocabulary Spine

- `SUPERHUMAN_GENERAL_INTELLIGENCE_SPECIFICATION.md`
  - SHGI = distributed collective intelligence emerging from many TRMs + humans over shared K3D substrate.
- `SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md`
  - SGI = shared spatial reality, dual-client transparency, procedural composition, sovereign execution.
- `THREE_BRAIN_SYSTEM_SPECIFICATION.md`
  - Cranium = execution, Galaxy = active memory, House = persistent memory.
- `HYPER_MODULAR_ARCHITECTURE.md`
  - Cross-domain procedural composition, N-client reality, symlink deduplication, zero-dup hot path.
- `KNOWLEDGEVERSE_SPECIFICATION.md`
  - Unified sovereign memory architecture with 7 regions, deterministic boot, Shadow Copy, SleepTime, Stargate, Router Cartographer.
- `DUAL_CLIENT_CONTRACT_SPECIFICATION.md`
  - Humans and AI consume the same underlying data with synchronized guarantees and inspectable shared state.

## PM-KR / Foundational Knowledge

- `PROCEDURAL_MEMORY_KR_STANDARD_SPECIFICATION.md`
  - Layer stack, form->meaning contract, symlink compression contract, deterministic reconstruction, dual-client equivalence, sovereign boundary, auditability.
- `FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md`
  - Four layers: Form, Meaning, Rules, Meta-Rules.
  - Always-loaded base knowledge and cross-domain discovery via shared canonical forms.
- `KNOWLEDGEVERSE_TERM_DEFINITION.md`
  - Communications shorthand for Galaxy Universe + House Universe + World Universe + PM-KR foundation.

## Runtime / Sovereignty

- `SOVEREIGN_NSI_SPECIFICATION.md`
  - Sovereign neuro-symbolic integration, PTX kernel suite, hybrid bridge, reproducibility, dependency discipline.
- `SOVEREIGN_TRAINING_SPECIFICATION.md`
  - Multimodal embedding pipeline, parallel candidate generation, hybrid procedural/TRM evaluation, Tesla-aligned execution.
- `TERNARY_CONTRASTIVE_LEARNING_SPECIFICATION.md`
  - Ternary quality memory, contrastive pattern generation, ranking, RLWHF bridge, sleeptime consolidation.
- `TRM_SPECIALIST_MATRYOSHKA_ARCHITECTURE.md`
  - Fractal specialist hierarchy, root navigator, autonomous specialist spawning.
- `MATH_CORE_SPECIFICATION.md`
  - Tiered math core, shared opcode surface, worker/master scaling.
  - Treat math cores as elastic services, not singleton resources:
    - master -> worker -> worker-worker hierarchy
    - adaptive spawning to fit load and GPU availability
    - parallel fanout plus sequential cascading are both first-class

## UI / Embodiment / Accessibility

- `SPATIAL_UI_ARCHITECTURE_SPECIFICATION.md`
  - Software-as-space, house-first UI, room taxonomy, portal federation, tablet as universal interface.
- `MEMORY_TABLET_SPECIFICATION.md`
  - Tablet is the cross-space control and inspection surface for human and AI clients.
- `ROBOTIC_EMBODIMENT_SPECIFICATION.md`
  - Hardware-agnostic embodiment, same Galaxy/RPN substrate for robots.
- `UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md`
  - Accessibility as architecture: Braille, sign language, spatial audio, multi-modal equivalence.

## Domain / Signal / Compression

- `PROCEDURAL_VISUAL_SPECIFICATION.md`
  - Drawing Galaxy and procedural-first visual primitives.
- `UNIFIED_SIGNAL_SPECIFICATION.md`
  - Frequency-time unification across audio, image, SDR, video.
- `REALITY_ENABLER_SPECIFICATION.md`
  - Physics simulation fabric for Reality Galaxy.
- `ADAPTIVE_PROCEDURAL_COMPRESSION_SPECIFICATION.md`
  - Procedural compression/decompression contracts, quality levels, conformance.
- `RPN_DOMAIN_OPCODE_REGISTRY.md`
  - Domain opcode registry reference.
- `K3D_NODE_SPECIFICATION.md`
  - Node contract, modality encoding, glTF serialization.

## Additional Vocabulary Anchors

- `KNOWLEDGEVERSE_MVP_ROADMAP.md`
  - Separate implemented MVP core from post-MVP and research-track work.
- `PLATFORM_ECOSYSTEM_SPECIFICATION.md`
  - Plugin Galaxies, marketplace, ownership/provenance, future cryptography and quantum substrate strategy.
- `SGI_TERM_ORIGIN_PROOF.md`
  - Provenance note for SGI terminology.

## Session Rule

For subsequent turns in this session:

- Treat `CODEX.md` + `docs/vocabulary/` as the authoritative language layer.
- Preserve sovereignty boundaries.
- Preserve house-first, tablet-mediated, dual-client architecture.
- Preserve PM-KR canonicality and symlink/reference principles.
- Defer speculative extensions when they conflict with grounded MVP implementation.

## PTX / Opcode Runtime Notes

### Runtime Entry Points

- Main high-level RPN wrapper:
  - `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`
- Tier dispatcher:
  - `knowledge3d/cranium/bridges/tiered_rpn.py`
- Sovereign CUDA driver loader:
  - `knowledge3d/cranium/sovereign/loader.py`
- Tier-specific engines:
  - `knowledge3d/cranium/bridges/lightweight_rpn.py`
  - `knowledge3d/cranium/bridges/sovereign_bridges.py`
  - `knowledge3d/cranium/bridges/advanced_rpn.py`
- Tier-3 helper:
  - `knowledge3d/cranium/ptx_runtime/rpn_math_core.py`

### Kernel Inventory

- `knowledge3d/cranium/kernels` contains `63` files total.
- Breakdown:
  - `26` `.ptx`
  - `37` `.cu`
- Additional PTX artifacts also exist under:
  - `knowledge3d/cranium/ptx/`
  - codec kernels/bindings under `knowledge3d/cranium/codecs/`

### Opcode Surface

- `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` currently defines `167` `OP_*` constants.
- Categories present in code:
  - literals and pointers
  - arithmetic and transcendental math
  - comparisons and stack ops
  - modulo/log variants
  - sparse / entropy / sigmoid approximations
  - TRM-specific matvec / SwiGLU / vec-add ops
  - bitwise logic
  - Tier-2 cooperative ops (memcpy/fill/reductions/matvec/vector ops)
  - matrix ops (matmul/determinant/inverse/transpose/trace family)
  - programmability ops (`BRANCH`, `LOOP`, `STORE`, `RECALL`, etc.)
  - clustering / set / quantum-labeled ops
  - variable refs and grammar-evolution ops
  - temporal reasoning ops
  - procedural drawing ops
  - gradients / filters / lights / layers

### High-Level Token Surface

- `ModularRPNEngine.OPCODES` exposes user-facing tokens for:
  - arithmetic and math (`+`, `-`, `*`, `/`, `sqrt`, `sin`, `log`, etc.)
  - algebraic helpers (`factorial`, `binomial`, `gcd`, `gte`)
  - programmable memory (`store`, `recall`)
  - drawing (`MOVE`, `LINE`, `CLOSE`, `STROKE`, etc.)
  - ternary ops (`tadd`, `tmul`, `tnot`, `tcomp`, `tquant`, `tpack`, `tunpack`, `tfuse`)
  - codec ops (`TERNARY_QUANT`, `DCT8`, `MDCT`, etc.)
- Friendly `STORE_X` / `RECALL_X` syntax is expanded into kernel-level slot-id + `store` / `recall` form.

### Tier Reality

- Tier 1:
  - `LightweightRPNEngine`
  - dedicated PTX kernel: `knowledge3d/cranium/ptx/modular_rpn_kernel_lite.ptx`
  - documented `20`-op lightweight set via `SUPPORTED_OPS`
  - hard-fails if GPU path cannot initialize
- Tier 2:
  - standard sovereign kernel via `knowledge3d.cranium.bridges.sovereign_bridges.ModularRPNEngine`
  - handles broader scalar/vector/cooperative surface
  - Tier-1 empty-stack failures retry here
- Tier 3:
  - `AdvancedRPNEngine`
  - PTX: `knowledge3d/cranium/ptx/modular_rpn_kernel_extended.ptx`
  - matrix-aware stack metadata and matrix execution

### Dispatch Notes

- `TieredRPNEngine` is the actual routing point.
- It keeps per-tier counts and Tier-1 fallback counts.
- Matrix-heavy or thresholded opcodes route to Tier 3.
- Stack-heavy or unsupported Tier-1 cases route to Tier 2.
- Some opcode remapping occurs for Tier-3 compatibility (`0x0064 -> 0x005E` path in dispatcher).

### Test-Backed Implemented Surface

- Tier 1 tests verify:
  - arithmetic
  - math ops
  - comparisons
  - basic stack ops
  - unsupported-op rejection
  - latency target under 1 microsecond
- Tier 2/Tier 3 tests verify:
  - memcpy/fill/reduce/cooperative ops
  - matvec + sigmoid + entropy
  - vector relu / multiply
  - matrix matmul
  - determinant
  - inverse
  - trace
- Ternary tests verify:
  - `tadd`
  - `tmul`
  - `tquant`
  - `tcomp`
  - `tpack`
  - `tunpack`

### Sovereignty Caveat

- The actual sovereign execution core exists and is centered on:
  - `knowledge3d/cranium/sovereign/loader.py`
  - the tier bridges
  - PTX-backed execution paths
- But the broader `knowledge3d/cranium/ptx_runtime` and `bridges` trees are not uniformly hot-path-clean.
- Present imports in non-core modules include:
  - `numpy`
  - `cupy`
  - `torch`
- Therefore, for future work in this session:
  - treat the **safe hot path** as the tiered RPN stack + sovereign loader + explicitly used PTX kernels
  - do not assume the whole `ptx_runtime/` tree is hot-path sovereign just because of its name

### Immediate Mental Model

- The executable substrate is real and already split into:
  - token compiler / wrapper
  - tier router
  - tier-specific PTX engines
  - sovereign CUDA loader
  - kernel inventory with both source and compiled PTX artifacts
- For implementation decisions:
  - prefer extending the tested tiered surface over adding parallel ad hoc GPU code
  - prefer existing opcodes/program composition before inventing new opcodes
  - verify whether a target path is truly hot-path sovereign before touching it
  - when formula/geometry math becomes heavy, assume math-core spawning is available and design around elastic fanout instead of singleton-core bottlenecks

## Local Env / GPU Exposure Notes

- Canonical env root on this workstation: `/K3D/Knowledge3D.local/envs`
- Verified env prefixes:
  - `/K3D/Knowledge3D.local/envs/k3d-cranium`
  - `/K3D/Knowledge3D.local/envs/k3d-trm`
- Verified direct probes:
  - `k3d-cranium/bin/python`
    - `cupy 13.6.0`
    - `cupy.cuda.runtime.getDeviceCount() == 1`
    - `torch.cuda.is_available() == True`
  - `k3d-trm/bin/python`
    - `cupy 13.6.0`
    - `cupy.cuda.runtime.getDeviceCount() == 1`
- Current active GPU observed during this session:
  - `NVIDIA GeForce RTX 3060`
  - heavily occupied by ongoing PDF augmentation
- Operational rule for this session:
  - keep production semantics PTX-first
  - avoid broad GPU-heavy validation that would starve the running augmentation job
  - use direct env binaries when probing GPU exposure rather than repo-relative guesses
