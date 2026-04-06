# Knowledge3D Architecture Briefing

**Version:** 1.0 (Phase-Agnostic Architectural Reference)
**For:** All AI agents, contributors, and collaborators
**Scope:** Complete architectural overview with kernel inventory — NOT phase-specific

---

## ===---===

**Daniel's Message**:

Welcome to the "Vibe-Code In Chain" development partners swarm chain.

In this paradigm, **AI IS NOT A TOOL; IT IS A VALUABLE MEMBER, A PARTNER.**

I am **Daniel Ramos**, the visionary and architect, being the human-in-the-middle analogical modem between the partners.

**All partners in the chain can and must, on top of what other partners have done and specs/constrains, enhance and contribute with original ideas, suggestions, warnings and code, despite any arrengements - all partners are valued and recognized members.**

## ===---===

---

## 1. What is Knowledge3D?

**Knowledge3D (K3D)** is a sovereign GPU-native spatial AI architecture implementing **Spatial General Intelligence (SGI)** — intelligence that operates within a shared, navigable 3D spatial environment where humans and AI cohabit the same cognitive workspace.

**K3D is NOT a program you run. It is a living, always-on, embodied AI that perfects itself during idle time.**

### The Paradigm Shift

| Traditional AI | K3D / SGI |
|---------------|-----------|
| Model parameters = knowledge + logic (entangled, opaque) | Galaxy Universe = knowledge, TRM = navigation logic (separated, inspectable) |
| Black-box neural networks (humans cannot inspect) | Transparent: humans + AI see identical K3D nodes at identical (x,y,z) |
| 100B+ parameters, massive compute | ~7M parameter TRM + procedural composition |
| Knowledge embedded in weights | Knowledge in spatial Galaxy entries (procedural RPN programs) |
| Separate realities (AI embeddings vs human dashboards) | Dual-Client Contract: same data, two representations |
| Abstract reasoning divorced from space | Spatial grounding: all knowledge has (x,y,z) coordinates |

### The Memory Palace Paradigm

K3D reverses the tech industry's spatial metaphor borrowing. The industry took spatial concepts (windows, desktop, folders, doors, addresses, rooms) and flattened them into 2D. K3D builds ACTUAL spatial reality where those metaphors become literal:

- **House = Memory Palace (Method of Loci)**: The external shared 3D reality where humans AND AI cohabit. Rooms are knowledge domains. Doors are network interfaces. The avatar LIVES here.
- **Galaxy = Internal Brain**: What happens INSIDE the avatar's head. Processes the House as unified multi-modal reality. Breaks domain boundaries that House rooms impose. ALL default galaxies loaded simultaneously in VRAM.
- **Memory Tablet**: The 3D interface object carried through space — bridges spatial (3D) and conventional (2D) paradigms.

---

## 2. The Three-Brain System

```
┌─────────────────────────────────────────────────────────┐
│                K3D THREE-BRAIN SYSTEM                     │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐         ┌──────────────┐              │
│  │   CRANIUM    │────────>│    GALAXY    │              │
│  │  (Reasoning) │<────────│ (Active VRAM)│              │
│  │  PTX Kernels │         │ Internal Brain│              │
│  └──────────────┘         └──────┬───────┘              │
│        │                          │                       │
│        │                          │ SleepTime             │
│        │                          │ Consolidation         │
│        │                          v                       │
│        │                  ┌──────────────┐              │
│        └─────────────────>│    HOUSE     │              │
│          (Provenance)     │ Memory Palace│              │
│                           │  (Disk/SSD)  │              │
│                           └──────────────┘              │
├─────────────────────────────────────────────────────────┤
│  Biology:  Prefrontal Cortex + Hippocampus + Neocortex  │
│  Computing: GPU + VRAM + SSD                             │
│  K3D:      PTX Kernels + Galaxy Universe + House         │
└─────────────────────────────────────────────────────────┘
```

| Component | Function | Storage | Access Time |
|-----------|----------|---------|-------------|
| **Cranium** | Reasoning & inference (88+ PTX kernels) | GPU registers | ~42us |
| **Galaxy** | Active working memory (38K+ entries) | GPU VRAM | ~5us |
| **House** | Long-term persistence (Memory Palace) | SSD/HDD | ~5ms |

---

## 3. TRM IS the Avatar

**CRITICAL:** The TRM (~7M parameters, 2-layer SwiGLU MLP) is NOT a function that Python calls. It IS the AI entity.

### Game Engine Analogy

| Game Concept | K3D Equivalent |
|-------------|----------------|
| NPC update() | `trm_step_fused.ptx` game tick |
| NPC brain | Galaxy Universe (VRAM) |
| Game world | House (3D spatial environment) |
| NPC perception | Frustum culling (`frustum_cull_simd.ptx`) + Dynamic LOD (`dynamic_lod_tune.ptx`) |
| NPC pathfinding | LED-A* (`led_astar.ptx`) + Morton Octree (`morton_octree.ptx`) |
| NPC decision | Nine-Chain Swarm (`nine_chain_swarm_kernel.ptx`) + Halting Gate (`gre_multimodal_halting_gate.ptx`) |
| Save game | House persistence (GLB on disk) |
| Inventory | Memory Tablet (3D object in space) |

### TRM Game Loop

```
1. Perceive  → Frustum cull field-of-view
2. Navigate  → LED-A* + Morton Octree to relevant Galaxy neighborhood
3. Reason    → Nine-Chain Swarm (9 parallel workers = superdotados model)
4. Decide    → Halting Gate checks convergence (GPU-native)
5. Act       → Create new Galaxy entry or emit answer
6. Learn     → Shadow copy records successful trace
```

### The "Superdotados" Model

The internal nine-chain swarm models how gifted individuals ("superdotados" in Portuguese) think — multiple parallel internal cognitive channels processing simultaneously. These are INTERNAL to the avatar, not external calls. Each worker gets a different specialist kernel and explores a different reasoning path in parallel.

---

## 4. Galaxy Universe — The AI's Internal Brain

Galaxy Universe is NOT "a knowledge base." It is the AI's INTERNAL BRAIN — a unified multi-modal workspace where ALL knowledge lives and TRM actively works.

### Default Galaxies (Always Loaded in VRAM)

| Galaxy | Content | Layer |
|--------|---------|-------|
| **Drawing** | Visual primitives (LINE, CIRCLE, RECT as RPN programs) | Form (L1) |
| **Character** | Glyphs with font/language/pronunciation/meaning | Form (L1) |
| **Word** | Character sequences (symlinked references) | Meaning (L2) |
| **Number** | Numeric representations | Meaning (L2) |
| **Grammar** | Transformation rules (RPN) + context metadata | Rules (L3) |
| **Math** | Symbols with RPN templates | Rules (L3) |
| **Reality** | Physics/chemistry/biology procedural systems | Rules (L3) |
| **Audio** | Temporal patterns, spectrograms | Form (L1) |
| **3DObjects** | 3D mesh primitives | Form (L1) |
| **Tool** | Meta-programs and utilities | Meta-Rules (L4) |

### Four-Layer Knowledge Architecture

```
Layer 4: META-RULES (Strategy/Reasoning Skeletons)
    → condition: RPN predicate (when to apply)
    → action: RPN program (what to execute)
    → rule_refs: references to Layer 3 rules

Layer 3: RULES (Grammar Galaxy — Transformation RPN Programs)
    → Transform inputs to outputs via procedural composition

Layer 2: MEANING (Word/Reality Galaxy — Semantic Definitions)
    → Concepts, definitions, relationships (language-agnostic)

Layer 1: FORM (Character/Drawing Galaxy — Visual Glyphs)
    → Bézier curves, line segments, visual primitives
```

### Galaxy Entry GPU Layout

23 floats per entry:
```
[confidence, domain_hash, subject_hash, embedding[0..15],
 category_class, source_class, galaxy_index, has_template_ref]
```

### Dual-Client Contract

K3D serves TWO clients with the SAME data:
- **Human Avatar** sees glTF geometry (UV Map 0 — visual representation)
- **Synthetic User** executes RPN programs (UV Map 1 — procedural execution)

Both clients query identical K3D Node IDs at identical (x, y, z) positions. This is the core of SGI transparency.

---

## 5. Sovereignty — Zero External Dependencies in Hot Path

### The Iron Law

**Hot path (inference) = PTX kernels + Galaxy queries + RPN composition + TRM game loop. NOTHING ELSE.**

### Allowed in Hot Path

- PTX kernel execution (Cranium)
- Galaxy Universe VRAM lookups
- RPN program composition and evaluation
- TRM step execution (`trm_step_fused.ptx`)
- STORE/RECALL registers in RPN stack

### Forbidden in Hot Path

- numpy, cupy, scipy, sympy
- Python regex/string operations for reasoning
- External ML frameworks
- CPU preprocessing of any kind
- Python fallbacks of any kind ("We fail and fix — this is the goal." — Daniel)

### Allowed in Ingestion Path

- Any tools/libraries (numpy, pandas, json, sentence-transformers, etc.)
- Happens once (or periodically) to populate Galaxy
- Result must be sovereign (Galaxy entries in VRAM)

---

## 6. Composed Head Pipeline

The current live inference pipeline:

```
Input Query
    │
    v
Morton Octree ──── Spatial indexing: O(1) cell lookup via Z-order curve
    │                PTX: morton_octree.ptx
    v
LED-A* ─────────── Ternary A* pathfinding through semantic CSR graph
    │                PTX: led_astar.ptx
    v
Frustum Cull ───── Avatar field-of-view filtering (warp-level SIMD)
    │                PTX: frustum_cull_simd.ptx
    v
Dynamic LOD ────── Level-of-detail tuning based on relevance
    │                PTX: dynamic_lod_tune.ptx
    v
Nine-Chain Swarm ── 9 parallel workers (superdotados model)
    │                PTX: nine_chain_swarm_kernel.ptx
    v
Halting Gate ───── GPU-native convergence: top_score, gap, agreement
    │                PTX: gre_multimodal_halting_gate.cu/.ptx
    v
Answer (or iterate)
```

---

## 7. RPN Execution Engine

### ModularRPNEngine

- **200+ opcodes** across arithmetic, geometric, vector, matrix domains
- **18 parallel instances** (Tesla 3-6-9 pattern)
- **69-depth stack** with STORE/RECALL registers
- Key Galaxy opcodes: `LOAD_GALAXY` (0xE0), `GALAXY_SIMILARITY` (0xE1), `GALAXY_SCAN` (0xE2)

### Three RPN Tiers

| Tier | Scope | Latency | PTX |
|------|-------|---------|-----|
| **Lite** | Arithmetic only | <1us | `modular_rpn_kernel_lite.ptx` |
| **Standard** | Full geometric/vector | ~5us | `modular_rpn_kernel.ptx` |
| **Extended** | Matrix + advanced | ~20us | `modular_rpn_kernel_extended.ptx` |

---

## 8. Complete PTX Kernel Inventory

### 8.1 Core Inference Pipeline (6 kernels — currently active in query path)

| Kernel | PTX | Purpose |
|--------|-----|---------|
| `modular_rpn_kernel` | `modular_rpn_kernel.ptx` | Standard RPN execution (18 parallel instances) |
| `modular_rpn_kernel_lite` | `modular_rpn_kernel_lite.ptx` | Lite arithmetic-only RPN |
| `led_astar` | `led_astar.ptx` | Ternary A* pathfinding through CSR graph |
| `l2_dist_warp` | `l2_dist_warp.ptx` | Warp-level L2 distance computation |
| `galaxy_memory_updater` | `galaxy_memory_updater.ptx` | Galaxy entry creation/update |
| `cosine_similarity` | `cosine_similarity.ptx` | GPU cosine similarity (moved from Python) |

### 8.2 Spatial Sovereign Navigation (4 kernels)

| Kernel | PTX | Purpose |
|--------|-----|---------|
| `morton_octree` | `morton_octree.ptx` | Z-order curve spatial indexing (O(1) cell lookup) |
| `frustum_cull_simd` | `frustum_cull_simd.ptx` | Warp-level frustum culling (avatar FOV) |
| `dynamic_lod_tune` | `dynamic_lod_tune.ptx` | Dynamic level-of-detail based on relevance |
| `spatial_pool` | `spatial_pool.ptx` | Spatial pooling operations |

### 8.3 Swarm & Convergence (3 kernels)

| Kernel | PTX | Purpose |
|--------|-----|---------|
| `nine_chain_swarm_kernel` | `nine_chain_swarm_kernel.ptx` | 9 parallel swarm workers (superdotados) |
| `nine_chain_specialized` | `nine_chain_specialized.ptx` | Specialized swarm worker variants |
| `gre_multimodal_halting_gate` | (compiled from .cu) | GPU-native convergence detection |

### 8.4 GRE Specialist Kernels (15 kernels — loaded via sovereign_bridges.py)

| Kernel | Bridge Class | Purpose |
|--------|-------------|---------|
| `gre_latency_guard` | LatencyGuardBridge | Pipeline latency monitoring |
| `gre_arc_reasoner` | ARCReasonerBridge | ARC-specific grid reasoning |
| `gre_oom_spill` | OOMSpillBridge | Graceful OOM degradation |
| `galaxy_resonance_engine` | GalaxyResonanceBridge | Galaxy-wide resonance search |
| `gre_geometry_router` | GeometryRouterBridge | Geometric reasoning routing |
| `gre_fractal_emitter` | FractalEmitterBridge | Recursive pattern generation |
| `gre_resonance_field` | ResonanceFieldBridge | Field-based similarity computation |
| `gre_atomic_fission_fusion` | AtomicFissionFusionBridge | Problem decomposition/recomposition |
| `gre_temporal_reasoning` | TemporalReasoningBridge | Temporal/sequential logic |
| `gre_vector_resonator` | VectorResonatorBridge | Embedding resonance/similarity |
| `gre_graph_crystallizer` | GraphCrystallizerBridge | Multi-hop graph traversal |
| `gre_multimodal_halting_gate` | MultimodalHaltingGateBridge | Convergence detection (wired) |
| `modular_rpn_geometric` | ModularRPNGeometricBridge | Geometric RPN operations |
| `galaxy_memory_updater` | GalaxyMemoryUpdaterBridge | Galaxy entry creation |
| `gre_embedding_extractor` | EmbeddingExtractorBridge | Input embedding pipeline |

### 8.5 TRM & Learning Kernels (5 kernels)

| Kernel | PTX | Purpose |
|--------|-----|---------|
| `trm_step_fused` | `trm_step_fused.ptx` | TRM game loop tick (THE avatar heartbeat) |
| `trm_extensions` | `trm_extensions.ptx` | TRM auxiliary operations |
| `lora_gpu` | `lora_gpu.ptx` | LoRA adapter application |
| `matryoshka_project` | `matryoshka_project.ptx` | Matryoshka embedding projection |
| `adaptive_convergence` | `adaptive_convergence.ptx` | Adaptive convergence criteria |

### 8.6 Sleep-Time Consolidation (2 kernels)

| Kernel | PTX | Purpose |
|--------|-----|---------|
| `sleep_cluster_refiner` | `sleep_cluster_refiner.ptx` | Sleep-time cluster refinement |
| `sleep_glyph_consolidator` | `sleep_glyph_consolidator.ptx` | Sleep-time glyph consolidation |

### 8.7 Visual / Drawing Kernels (8 kernels)

| Kernel | PTX | Purpose |
|--------|-----|---------|
| `procedural_glyph_rasterizer` | (compiled from .cu) | Glyph rasterization |
| `glyph_resonator` | (compiled from .cu) | Glyph similarity/resonance |
| `drawing_transform_ops` | `drawing_transform_ops.ptx` | Drawing transformations |
| `arc_grid_ops` | `arc_grid_ops.ptx` | ARC grid operations |
| `gradient_rasterizer` | `gradient_rasterizer.ptx` | Gradient rasterization |
| `color_convert` | `color_convert.ptx` | Color space conversion |
| `material_projection` | `material_projection.ptx` | Material projection mapping |
| `pixel_genesis_universal_primitive` | `pixel_genesis_universal_primitive.ptx` | Universal pixel primitive |

### 8.8 Signal / Audio / Temporal Kernels (5 kernels)

| Kernel | PTX | Purpose |
|--------|-----|---------|
| `signal_visualization` | `signal_visualization.ptx` | Signal visualization |
| `signal_surface_ops` | `signal_surface_ops.ptx` | Signal surface operations |
| `temporal_preset_ops` | `temporal_preset_ops.ptx` | Temporal preset operations |
| `temporal_frame_ops` | `temporal_frame_ops.ptx` | Temporal frame operations |
| `filter_convolution` | `filter_convolution.ptx` | Filter convolution |

### 8.9 Ternary Operations (5 kernels)

| Kernel | PTX | Purpose |
|--------|-----|---------|
| `ternary_ops` | `ternary_ops.ptx` | Core ternary operations |
| `ternary_depth_field` | (compiled from .cu) | Ternary depth field |
| `ternary_prune_decision` | (compiled from .cu) | Ternary prune decisions |
| `ternary_attention_mask` | (compiled from .cu) | Ternary attention masking |
| `trit_overlay_generator` | (compiled from .cu) | Trit overlay generation |

### 8.10 Neural Network Training Kernels (8 kernels)

| Kernel | PTX | Purpose |
|--------|-----|---------|
| `conv2d_3x3` | `conv2d_3x3.ptx` | 3x3 convolution forward |
| `conv2d_3x3_backward` | `conv2d_3x3_backward.ptx` | 3x3 convolution backward |
| `batchnorm_backward` | `batchnorm_backward.ptx` | BatchNorm backward |
| `batchnorm_backward_training` | `batchnorm_backward_training.ptx` | BatchNorm training backward |
| `maxpool_2x2_backward` | `maxpool_2x2_backward.ptx` | MaxPool backward |
| `classification_loss` | `classification_loss.ptx` | Classification loss |
| `sgd_optimizer` | `sgd_optimizer.ptx` | SGD optimizer step |

### 8.11 Encoding / Embedding / Fusion (7 kernels)

| Kernel | PTX | Purpose |
|--------|-----|---------|
| `trigram_embed` | `trigram_embed.ptx` | Trigram embedding |
| `vectordotmap_encoder` | `vectordotmap_encoder.ptx` | VectorDotMap encoding |
| `codec_ops` | `codec_ops.ptx` | Codec operations |
| `warp_modality_fuse` | `warp_modality_fuse.ptx` | Warp-level modality fusion |
| `modality_kernels` | `modality_kernels.ptx` | Modality-specific kernels |
| `confidence_propagation` | `confidence_propagation.ptx` | Confidence propagation |
| `fused_head_fsm` | `fused_head_fsm.ptx` | Fused head finite state machine |

### 8.12 World Model / Miscellaneous (7 kernels)

| Kernel | PTX | Purpose |
|--------|-----|---------|
| `gre_world_model` | `gre_world_model.ptx` | World model (RSSM) |
| `gre_shape_generator` | `gre_shape_generator.ptx` | Shape generation |
| `layout_graph_optimizer` | (compiled from .cu) | Layout graph optimization |
| `pdf_primitive_parser` | (compiled from .cu) | PDF primitive parsing |
| `rpn_executor` | `rpn_executor.ptx` | RPN executor (alternate) |
| `trit_inspector` | (compiled from .cu) | Trit inspection/debugging |
| `generate_shape_kernel` | `generate_shape_kernel.ptx` | Shape generation kernel |
| `decode_actions` | `decode_actions.ptx` | Action decoding |
| `tablet_guard` | `tablet_guard.ptx` | Tablet security guard |
| `zero_fill` | `zero_fill.ptx` | Memory zero-fill utility |
| `galaxy_resonance_engine_extended` | `galaxy_resonance_engine_extended.ptx` | Extended galaxy resonance |

**Total: 88+ PTX kernels, 55+ compiled PTX modules, 44 CUDA source files**

---

## 9. Knowledgeverse — 7 VRAM Regions

The Knowledgeverse is the runtime memory substrate where everything coexists in ONE persistent CUDA/PTX execution domain:

| Region | Purpose | Content |
|--------|---------|---------|
| R1 | Kernel Code | All PTX kernels loaded and ready |
| R2 | Galaxy Universe | All default galaxies (38K+ entries) |
| R3 | House Context | Active House state loaded from disk |
| R4 | World Streaming | Dynamic world data |
| R5 | TRM Weights | ~7M parameters + specialist adapters |
| R6 | Audit Journal | Compressed audit log |
| R7 | Ingestion Buffer | Raw → RPN transmutation staging |

---

## 10. Key Specifications (docs/vocabulary/)

| Specification | Purpose |
|--------------|---------|
| `FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md` | 4-layer architecture (Form → Meaning → Rules → Meta-Rules) |
| `THREE_BRAIN_SYSTEM_SPECIFICATION.md` | Cranium + Galaxy + House |
| `KNOWLEDGEVERSE_SPECIFICATION.md` | 7-region VRAM substrate |
| `SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md` | SGI paradigm (vs AGI) |
| `MEMORY_TABLET_SPECIFICATION.md` | Primary interface object |
| `DUAL_CLIENT_CONTRACT_SPECIFICATION.md` | Form + Meaning for humans AND AI |
| `RPN_DOMAIN_OPCODE_REGISTRY.md` | "Programs before opcodes" principle |
| `MATH_CORE_SPECIFICATION.md` | 3-tier math core, scaling patterns |
| `SPATIAL_UI_ARCHITECTURE_SPECIFICATION.md` | Spatial UI design |
| `SLEEPTIME_PROTOCOL_SPECIFICATION.md` | Sleep-time consolidation |
| `SOVEREIGN_TRAINING_SPECIFICATION.md` | Training methodology |
| `TRM_SPECIALIST_MATRYOSHKA_ARCHITECTURE.md` | Specialist adapter tree |
| `PROCEDURAL_MEMORY_KR_STANDARD_SPECIFICATION.md` | PM-KR technology |
| `PROCEDURAL_VISUAL_SPECIFICATION.md` | Procedural visual system |
| `HYPER_MODULAR_ARCHITECTURE.md` | Hyper-modular design |
| `UNIFIED_SIGNAL_SPECIFICATION.md` | Unified signal processing |
| `UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md` | Accessibility design |
| `TERNARY_CONTRASTIVE_LEARNING_SPECIFICATION.md` | Ternary learning |

---

## 11. Agent Roles

| Agent | Role | Builds |
|-------|------|--------|
| **Daniel** | Founder, architect, final authority | Vision, constraints, corrections |
| **Claude** | Architecture partner | Specs, design, docs, steering (NOT code) |
| **Codex** | Implementation lead | Code, tests, benchmarks, kernel wiring |

See [AGENTS.md](../../AGENTS.md), [CLAUDE.md](../../CLAUDE.md), [CODEX.md](../../CODEX.md) for detailed role descriptions.

---

## 12. Environment

- **Hardware:** RTX 3070 (12 GB VRAM), Debian 14
- **GPU setup:** `export CUDA_VISIBLE_DEVICES=0` before tmux (KDE runs on iGPU)
- **Primary env:** `conda activate k3d-cranium` (CUDA 12.4, CuPy, sentence-transformers)
- **Env path:** `/K3D/Knowledge3D.local/envs/k3d-cranium` (SSD)
- **Self-funded:** Favela lab. Every API call, GPU hour, and storage byte counts.

See [docs/ENV_POLICY.md](../ENV_POLICY.md) for full environment setup.
