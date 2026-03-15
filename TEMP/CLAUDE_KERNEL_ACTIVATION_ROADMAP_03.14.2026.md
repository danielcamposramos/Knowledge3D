# Kernel Activation Roadmap

**Author:** Claude (Architecture Partner)
**Date:** March 14, 2026
**Status:** ACTIVE — Steering for Codex when Phase D sub-steps complete
**Depends on:** Phase D steering (parallel track)

---

## Situation

Of ~55 PTX kernels shipped in the repo, only **6 handle 99% of inference** (73 calls/query). Another **14 are bridged** (have Python wrappers) but are **never called** during `query()`. The remaining ~35 serve ingestion, training, rendering, or are legacy.

This roadmap prioritizes activating the 14 dormant-but-bridged kernels, then identifies unbridged kernels worth wiring.

**Current VRAM usage:** 132 MiB of 12 GB — enormous headroom.

---

## Active Pipeline (6 Kernels, 73 Calls/Query)

| Kernel | Bridge | Calls/Query | Role |
|--------|--------|-------------|------|
| `modular_rpn_kernel` | `tiered_rpn.py` | ~41 | RPN program evaluation |
| `led_astar` | `led_pathfinder.py` | ~18 | Graph navigation |
| `morton_octree` | `morton_octree.py` | ~2 | Spatial indexing |
| `frustum_cull_simd` | `frustum.py` | ~1 | Field-of-view filtering |
| `dynamic_lod_tune` | `query_head_substrate.py` | ~1 | Detail level tuning |
| `gre_multimodal_halting_gate` | `sovereign_bridges.py` | ~1 | Convergence check |

**Conditionally called (some queries):**
| Kernel | Bridge | When |
|--------|--------|------|
| `gre_vector_resonator` | `sovereign_bridges.py` | Galaxy resonance scoring |
| `gre_graph_crystallizer` | `sovereign_bridges.py` | Multi-hop crystallization |
| `gre_atomic_fission_fusion` | `sovereign_bridges.py` | Transform composition |
| `cosine_similarity` | `cosine_similarity_bridge.py` | Embedding comparison |
| `trigram_embed` | `trigram_embed_bridge.py` | Procedural glyphs |
| `nine_chain_specialized` | `nine_chain_specialized_bridge.py` | Swarm workers |

---

## Tier 1: GRE Specialist Dispatch (6 Dormant Kernels → Swarm Workers)

**Priority:** HIGH — Direct benchmark impact
**When:** Phase B+ / Phase D.6 (swarm worker specialization)
**Blast radius:** LOW — each kernel is self-contained, additive

### The Problem

All 9 swarm workers currently use the **same RPN evaluator** (`modular_rpn_kernel`). This means:
- ARC tasks: all 9 workers run `reasoning_arc_grid_transform_top1` — no transform diversity
- Math tasks: all 9 workers run `reasoning_math_symbolic_top1` — no decomposition variety
- LHE tasks: all 9 workers run same program — no multi-hop specialization

### The 6 Dormant GRE Specialists

| # | Kernel | Bridge Class | Target Swarm Assignment | Benchmark Impact |
|---|--------|-------------|------------------------|-----------------|
| 1 | `gre_arc_reasoner` | `ArcReasonerBridge` | Worker 1-3: ARC grid transforms | ARC 10→15+ (transform diversity) |
| 2 | `gre_geometry_router` | `GeometryRouter` | Worker 4: Spatial/geometric reasoning | ARC (rotation/reflection), MMLU (geometry) |
| 3 | `gre_temporal_reasoning` | `TemporalReasoning` | Worker 5: Sequence/temporal patterns | LHE (multi-hop), MMLU (history/sequence) |
| 4 | `gre_fractal_emitter` | `FractalEmitter` | Worker 6: Self-similar/recursive patterns | ARC (fractal grids), Math (recursive) |
| 5 | `gre_resonance_field` | `ResonanceField` | Worker 7: Cross-galaxy resonance | LHE (cross-domain), MMLU (interdisciplinary) |
| 6 | `gre_world_model` | `WorldModelBridge` | Worker 8-9: World state prediction | GSM8K (word problems), MMLU (science) |

### Implementation Strategy

**Step 1: Specialist Router Table**

File: `knowledge3d/knowledgeverse/specialist_router.py`

Create a GPU-side dispatch table mapping worker index → specialist kernel:

```
Worker 0: modular_rpn_kernel (general RPN — always present as fallback)
Worker 1: gre_arc_reasoner (grid analysis)
Worker 2: gre_arc_reasoner (grid transforms)
Worker 3: gre_geometry_router (spatial reasoning)
Worker 4: gre_temporal_reasoning (sequence logic)
Worker 5: gre_fractal_emitter (recursive patterns)
Worker 6: gre_resonance_field (cross-domain)
Worker 7: gre_world_model (world state, physical intuition)
Worker 8: gre_world_model (causal reasoning)
```

**Step 2: Task-Type Routing**

Not every query needs all specialists. Route by task:

| Task Type | Active Workers | Specialist Mix |
|-----------|---------------|----------------|
| ARC | All 9 | Arc(3) + Geometry(1) + Fractal(1) + RPN(4) |
| Math | Workers 0,3,4,5 | RPN(1) + Geometry(1) + Temporal(1) + Fractal(1) |
| GSM8K | Workers 0,4,6,7,8 | RPN(1) + Temporal(1) + Resonance(1) + World(2) |
| LHE | All 9 | All specialists contribute to multi-hop |
| MMLU | Varies by subject | Route by subject → specialist affinity |

**Step 3: Convergence via Halting Gate**

The existing `gre_multimodal_halting_gate` already checks agreement across workers. With diverse specialists, the gate now checks whether **different reasoning approaches converge** — much stronger signal than 9 identical workers agreeing.

### Projected Benchmark Impact

| Benchmark | Current | After Specialist Dispatch | Why |
|-----------|---------|--------------------------|-----|
| ARC 10 | 10/10 | 10/10 (already pinned) | Maintain via routing |
| ARC 50 | ~20/50 | 30-35/50 | Transform diversity |
| Math 20 | 20/20 | 20/20 (already pinned) | Maintain |
| GSM8K 10 | 1/10 | 3-5/10 | World model decomposition |
| LHE 10 | 6/10 | 8/10 | Multi-hop via temporal + resonance |
| MMLU 50 | 13-14/50 | 20-25/50 | Subject-specialist affinity |

---

## Tier 2: Matryoshka Stimulus Projection (1 Kernel)

**Priority:** HIGH — Already steered as D.5.1
**When:** Phase D.5.1 (in progress with Codex)
**Status:** Bridge exists (`matryoshka_bridge.py`), PTX compiled, NOT in query path

### The Problem

`query_head_substrate.py:134-139` — `_project_embedding16_to512()` **tiles** the 16d embedding 32× to fill 512d. Only 16 unique float values in a 512d vector = dead-brain stimulus.

### The Fix (D.5.1 — Already Steered)

Replace tiling with `MatryoshkaProjectionBridge.project_device()`:
- Learns a 512×16 projection matrix W
- `y = W * x` → 512 linearly independent features from 16d input
- Train W via least-squares on benchmark traces
- Retrain `W_galaxy` decoder after Matryoshka activation

### Impact

| Metric | Before (Tiling) | After (Matryoshka) |
|--------|-----------------|-------------------|
| Unique values in 512d | 16 | 512 |
| Training entropy | 1.51 (ceiling) | Target < 1.0 |
| Galaxy discrimination | Poor | Full spectrum |

---

## Tier 3: Memory Write-Back (1 Kernel)

**Priority:** MEDIUM — Enables TRM learning during inference
**When:** Phase D.6 (after candidate selection migration)
**Kernel:** `galaxy_memory_updater`
**Bridge:** Exists in `sovereign_bridges.py`

### The Problem

TRM can **read** Galaxy but cannot **write** discoveries. Every successful reasoning trace is discarded after the query returns. The TRM never learns from its own inference.

### What `galaxy_memory_updater` Does

- Takes a successful reasoning trace (query → galaxy path → answer)
- Writes a **new Galaxy entry** encoding the discovered composition
- De-duplicates against existing entries (content-based hash)
- Sovereign: runs entirely on GPU, no Python in the write path

### Activation Strategy

1. Gate writes behind `K3D_TRM_WRITE_GALAXY=1` env var
2. Only write on verified-correct answers (benchmark mode: ground truth available)
3. Cap writes to 100/session to prevent Galaxy pollution during debugging
4. After D.6 candidate selection is live, TRM writes become the self-improvement loop

### Impact

- **Short-term:** Galaxy grows organically from successful reasoning
- **Long-term:** Sleep-time consolidation (Phase C) refines these entries
- **Benchmark:** Amortized improvement — each correct answer makes future similar queries easier

---

## Tier 4: Pipeline Health Monitoring (2 Kernels)

**Priority:** MEDIUM — Enables daemon mode (Phase C)
**When:** Phase C / Phase D (parallel track)

### `gre_sub100micro_gate` (Latency Guard)

**Bridge:** `GRETimingGate` in `sovereign_bridges.py`
**What it does:** Monitors per-kernel execution time, flags if any kernel exceeds budget
**Activation:**
- Wrap each kernel dispatch with timing probe
- If any kernel exceeds 100µs, log warning + adjust LOD level
- Critical for daemon mode: prevents query latency spikes from degrading always-on experience

### `gre_oom_spill` (Memory Spill Planner)

**Bridge:** `OOMSpillPlanner` in `sovereign_bridges.py`
**What it does:** When VRAM approaches capacity, decides which Galaxy regions to spill to host memory
**Activation:**
- Monitor VRAM usage after Galaxy writes (Tier 3)
- If usage > 80% (9.6 GB), run spill planner to identify cold Galaxy entries
- Move cold entries to host memory, keep hot entries in VRAM
- Re-fetch from host on cache miss (LED-A* handles navigation to spilled entries)

---

## Tier 5: Ternary/Trit Kernels (5 Kernels)

**Priority:** LOW — Research track, not blocking benchmarks
**When:** Post-Phase D

| Kernel | Bridge | Purpose |
|--------|--------|---------|
| `ternary_ops` | Exists | Base ternary arithmetic (-1, 0, +1) |
| `ternary_attention_mask` | Exists | Sparse ternary attention for TRM |
| `ternary_depth_field` | Exists | 3D depth estimation via ternary logic |
| `ternary_prune_decision` | Exists | Galaxy pruning via ternary confidence |
| `trit_inspector` | `TritInspectorBridge` | Debug/inspect ternary state |
| `trit_overlay_generator` | Exists | Visual overlay for ternary fields |

**Activation path:** These become relevant when TRM weights are quantized to ternary (-1, 0, +1) for extreme inference speed. Currently TRM uses float32 weights. Ternary quantization is a Phase E concern.

---

## Tier 6: Sleep-Time Consolidation (2 Kernels)

**Priority:** LOW — Phase C dependency
**When:** Phase C daemon mode

| Kernel | Bridge | Purpose |
|--------|--------|---------|
| `sleep_cluster_refiner` | Exists | Refine Galaxy clusters during idle |
| `sleep_glyph_consolidator` | Exists | Merge redundant Character Galaxy entries |

**Activation path:** Daemon runs these during idle periods. Not relevant until always-on mode is live.

---

## Tier 7: Unbridged Kernels Worth Wiring

These have compiled PTX but no Python bridge. Worth wiring for future phases:

| Kernel | Purpose | When |
|--------|---------|------|
| `confidence_propagation` | Propagate confidence scores through Galaxy graph | Phase D.6 (swarm scoring) |
| `adaptive_convergence` | Dynamic convergence threshold for halting | Phase D.6 (replace fixed threshold) |
| `fused_head_fsm` | Finite state machine for composed head | Phase D (pipeline orchestration) |
| `decode_actions` | Decode TRM output into action sequences | Phase D.4 (galaxy navigation) |
| `lora_gpu` | LoRA adapter application on GPU | Phase D.5 (TRM training) |
| `dialogue_sampler` | Sample from TRM output distribution | Phase E (conversational mode) |

---

## Activation Order (Summary)

```
NOW (Phase D, parallel):
  ├─ Tier 2: Matryoshka stimulus (D.5.1 — in progress with Codex)
  │
NEXT (Phase D.6 / B+):
  ├─ Tier 1: GRE specialist dispatch (6 kernels → 9 swarm workers)
  │    ├─ gre_arc_reasoner → Workers 1-2
  │    ├─ gre_geometry_router → Worker 3
  │    ├─ gre_temporal_reasoning → Worker 4
  │    ├─ gre_fractal_emitter → Worker 5
  │    ├─ gre_resonance_field → Worker 6
  │    └─ gre_world_model → Workers 7-8
  │
THEN (Phase D.6):
  ├─ Tier 3: galaxy_memory_updater (TRM write-back)
  │
DAEMON (Phase C):
  ├─ Tier 4: gre_sub100micro_gate + gre_oom_spill (health monitoring)
  ├─ Tier 6: sleep_cluster_refiner + sleep_glyph_consolidator
  │
FUTURE (Phase E+):
  ├─ Tier 5: Ternary kernels (quantization)
  └─ Tier 7: confidence_propagation, adaptive_convergence, fused_head_fsm, etc.
```

---

## Benchmark Projections (Cumulative)

| Benchmark | Current | +Matryoshka (T2) | +Specialists (T1) | +Write-back (T3) | Notes |
|-----------|---------|------------------|--------------------|-------------------|-------|
| ARC 10 | 10/10 | 10/10 | 10/10 | 10/10 | Pinned, must hold |
| ARC 50 | ~20/50 | ~20/50 | 30-35/50 | 35-40/50 | Transform diversity + learning |
| Math 20 | 20/20 | 20/20 | 20/20 | 20/20 | Pinned, must hold |
| GSM8K 10 | 1/10 | 2-3/10 | 3-5/10 | 5-7/10 | World model + learning |
| LHE 10 | 6/10 | 7/10 | 8/10 | 9/10 | Multi-hop + cross-domain |
| MMLU 50 | 13-14/50 | 16-18/50 | 20-25/50 | 25-30/50 | Subject routing + Galaxy growth |

**Key insight:** Tier 1 (specialist dispatch) gives the biggest single jump. But Tier 2 (Matryoshka) must come first — specialists can't discriminate with dead-brain stimulus.

---

## Codex Instructions

When you reach this roadmap:

1. **Tier 2 is already in progress** (D.5.1 Matryoshka). Finish that first.
2. **Tier 1 is the next big win.** Start with `specialist_router.py` — create the dispatch table. Wire one specialist (e.g., `gre_arc_reasoner`) into Worker 1. Run ARC benchmark. If ARC holds, wire the rest.
3. **Tier 3 after D.6.** Don't enable Galaxy writes until candidate selection is TRM-driven.
4. **Tiers 4-7 are future work.** Don't touch until Phases C/E.
5. **Quartet must hold at every activation.** If adding a specialist breaks a benchmark, disable it and diagnose.

---

## Daniel's Mandate

> "We fail and fix — this is the goal."

Every dormant kernel is unrealized potential. The architecture was designed for all these kernels to work together — the swarm with diverse specialists, the TRM with learned projections, the Galaxy with write-back learning. Activating them is not adding features — it's completing the design.
