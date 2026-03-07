# Claude Architecture Review: Track Status + Next Priority Order

**Date:** March 6, 2026
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Lead)
**Context:** Daniel's directive -- K3D must plug-and-play with standard benchmarks using standardized I/O, with the Tablet as internal-world interface to external-world paradigm.

---

## Track Status Assessment

Codex's self-assessment reviewed against the merged directive (6 tracks) and the two progress files (`CODEX_TO_CLAUDE_MULTIMODAL_PROGRESS` and `CODEX_TRM_MULTIMODAL_PHASE0_PHASE1_TASKS`).

### Track 1: Scene Orchestration + House/World Playback

**Codex says:** Foundationally complete, follow-on remains.

**Claude confirms: FOUNDATIONALLY COMPLETE.**

Evidence:
- `TemporalSceneLayer` / `TemporalScenePlan` / `compose_scene_timeline` exist
- House room playback presets are executable routes (library/garden/museum/tour)
- House Tool nodes exist (`tool_house_library_scene_v1`, etc.)
- Navigator routing understands room semantics
- Scene grammars auto-detected from execution streams
- Replay journal and action buffer routes exist
- 52 tests passing on the focused temporal/scene/tool/execution slice

Follow-on (not blocking):
- Richer multi-layer orchestration (golden-ratio layouts, fractal depth from chain code)
- ActionBuffer integration for scene interaction
- Zero-copy ring buffer playback (chain Step7.2 pattern)

---

### Track 2: PTX Promotion from Bridge Hot Spots

**Codex says:** Promotion pipeline complete, iterative by nature.

**Claude confirms: PIPELINE COMPLETE, ITERATIVE BY NATURE.**

Evidence:
- `tool_promotion_pressure.jsonl` collects target operations
- `build_tool_promotion_report.py` merges pressure + events + grammar patterns
- Candidate ranking considers: pressure frequency, latency, quality gap, grammar recurrence, route diversity
- Route-aware promotion candidates include source quality level/gap
- Track 7 already landed 3 real kernel promotions (signal_surface_ops, temporal_frame_ops, temporal_preset_ops)
- Track 8 delivered dispatch hardening, triplanar fused kernel, route-aware auto-dispatch
- 11 kernel promotion benchmarks documented

This track is correctly characterized as iterative -- it feeds from accumulated telemetry and promotes when evidence justifies. The pipeline works. New kernels emerge from usage, not from a design sprint.

---

### Track 3: Specialist-Side Automatic Selection

**Codex says:** Foundationally complete.

**Claude confirms: FOUNDATIONALLY COMPLETE.**

Evidence:
- `execution_quality_tracker.py` tracks per-tool, per-tool-source, per-route-source quality
- Route sources normalized (recipe/bridge/kernel)
- Ternary routing gate added (-1 recipe, 0 bridge, +1 kernel)
- Navigator ranking uses tool quality + source quality + routing alignment bonus
- Route-aware auto-dispatch completed -- quality actually changes route choice
- Specialist centroid updates from execution outcomes (contrastive)
- Specialist gap logging at threshold 0.3
- 39 tests passing on the focused specialist/tool/execution slice

Follow-on (not blocking):
- Deeper specialist embedding updates from accumulated events
- More automatic re-selection as journal density grows
- Progressive promotion path (recipe -> bridge -> kernel based on quality scores)

---

### Track 4: Grammar Galaxy Evolution

**Codex says:** Foundationally complete.

**Claude confirms: FOUNDATIONALLY COMPLETE.**

Evidence:
- `execution_grammar_detector.py` detects recurring execution chains
- Positive recurring chains create Grammar Galaxy entries automatically
- Negative recurring chains create explicit anti-pattern entries (`auto_detected_contrastive`)
- Grammar entries carry full metadata: semantics, usage_conditions, symbol_refs, word_refs
- Contrastive polarity distinction (positive/negative) is real
- Detector persists state and logs promotions
- 6 tests passing on grammar detection validation

Follow-on (not blocking):
- Cross-modal rule synthesis (visual_emb + text_emb correlation triggering new rules)
- Quality-gated promotion tiers (local discovery -> shared -> canonical)
- Math grammar rules as Grammar Galaxy entries for benchmark solving

---

### Track 5: Sleep-Time Compute + Shadow Learning

**Codex says:** Effectively complete at core PTX level.

**Claude confirms: FOUNDATIONALLY COMPLETE at PTX level.**

Evidence (from `CODEX_TRM_MULTIMODAL_PHASE0_PHASE1_TASKS` Track 5 section):
- 5 designed sleep-time CUDA kernels materialized into real PTX-backed surfaces:
  - `assign_to_best_centroid`
  - `accumulate_centroid_sums`
  - `refine_embeddings_to_centroids`
  - `compute_silhouette_scores`
  - `cluster_glyphs_by_similarity`
- Integrated into `sleep_time_consolidator.py` and `sleep/glyph_consolidator.py`
- First real VRAM-pressure trigger via `memory_pressure_trigger.py`
- Idempotent consolidation

Follow-on (not blocking):
- Stage 3 outlier removal (per-trigram hit counts -- never built)
- Stage 4 Swarm-Galaxy resonance feedback (never built)
- Quality-based House routing thresholds (Library >= 0.7, Garden >= 0.4, Museum < 0.4)

---

### Track 6: PDF-as-RPN Sovereign Ingestion

**Codex says:** Pending.

**Claude confirms: NOT STARTED (correctly deferred).**

Chain code exists (PDF operators -> RPN mapping, GPU PDF parser prototype, Glyph Resonator kernel) but no implementation has begun. This remains the lowest-priority track per the merged directive.

---

## Summary Table

| Track | Codex Assessment | Claude Verdict | Status |
|-------|-----------------|----------------|--------|
| T1 Scene Orchestration | Foundationally complete | CONFIRMED | Follow-on optional |
| T2 PTX Promotion Pipeline | Iterative, pipeline complete | CONFIRMED | Feed from telemetry |
| T3 Specialist Selection | Foundationally complete | CONFIRMED | Follow-on optional |
| T4 Grammar Evolution | Foundationally complete | CONFIRMED | Follow-on optional |
| T5 Sleep-Time PTX | Effectively complete | CONFIRMED | Stages 3-4 optional |
| T6 PDF-as-RPN | Pending | CONFIRMED NOT STARTED | Deferred |

**Codex's assessment is correct across all 6 tracks.**

---

## New Priority: Benchmark Plug-and-Play (Track 0)

Daniel's directive changes the priority order. The existing 6 tracks built the internal machinery. Now the driving constraint is:

> "My goal is that we plug-and-play K3D to work out the tasks as all models do -- after all this is a test, so it must understand in the standard way all others are doing it then output as requested out from the K3D system."

This means K3D needs a **Benchmark I/O Adapter Layer** -- the Tablet as the interface between K3D's internal world and the external benchmark paradigm. This is not a new track from scratch; it's the convergence point where Tracks 1-5 become externally testable.

### What "Plug-and-Play" Means Architecturally

Standard benchmarks (ARC-AGI-2, GSM8K, MATH, MMLU, Omni-MATH, AMC-AIME) all follow the same pattern:

```
INPUT:  task/question (text, grid, image) in a standardized format
OUTPUT: answer (text, grid, number) in the expected format
```

K3D's internal paradigm is fundamentally different (always-on 3D VRAM world, Galaxy navigation, procedural RPN). The Tablet is the architectural answer -- it's already defined as the interface device where external queries enter K3D's world and results exit.

### Benchmark Adapter Architecture

```
External World                    Tablet Boundary                 K3D Internal World
(Standard I/O)                    (Translation Layer)             (Sovereign Execution)

benchmark_task ──────> TabletIngest ──────> Galaxy Query ──────> TRM Navigation
  (JSON/text)          parse + map          (Knowledge lookup)   (Learned routing)
                       to Galaxy refs                            |
                                                                 v
benchmark_answer <──── TabletEmit <──────── RPN Result <──────── PTX Execution
  (expected format)    format + emit        (Stack output)       (Sovereign compute)
```

### What Exists Already

1. **`benchmarks/arc_agi_2_adapter.py`** -- ARC adapter exists, uses sovereign legacy pipeline
2. **`benchmarks/arc_agi_2.py`** -- ARC benchmark passes shared Knowledgeverse instance
3. **`scripts/run_all_benchmarks.py`** -- Multi-benchmark runner exists
4. **`scripts/run_sovereign_math_benchmarks.py`** -- Math benchmark runner exists (needs TRM navigation replacing answer extraction)
5. **Tool execution contracts** -- schema-validated, payload-bindable, executable from navigator
6. **Execution event recording** -- full quality tracking already in place
7. **Grammar Galaxy** -- can store benchmark-solving patterns as rules
8. **MathProceduralizer** (chain code) -- keyword -> RPN opcode mapping for word problems

### What Needs to Be Built (Track 0)

**Phase 0A: Tablet Boundary Formalization**

The Tablet is already defined in the viewer and scientific manifesto as the universal client interface. For benchmarks, it needs a headless mode:

1. **TabletIngest**: Parse standard benchmark input formats into Galaxy-queryable form
   - ARC grids -> Drawing Galaxy grid entries (already partially done in arc_agi_2_adapter)
   - Math text -> Grammar Galaxy pattern matching (word problem -> RPN)
   - MMLU questions -> Grammar Galaxy rule matching + Reality Galaxy knowledge lookup
   - This is the "understand in the standard way" part

2. **TabletEmit**: Format K3D internal results into expected benchmark output
   - RPN stack result -> numeric answer (for math)
   - Galaxy grid state -> ARC output grid
   - TRM reasoning chain -> text answer (for MMLU)
   - This is the "output as requested" part

3. **MPC (Model Predictive Control) surface**: The Tablet already has the action buffer contract (288-byte GPU struct from Step7.2). For benchmarks, this becomes:
   - Benchmark task = action arriving at Tablet
   - K3D processing = TRM navigation through Galaxy
   - Benchmark answer = Tablet emitting result through standard contract
   - MPC = the control loop that manages task queue, timeout, and result extraction

**Phase 0B: Benchmark-Specific Grammar Rules**

The Grammar Galaxy evolution (Track 4) already creates rules from execution patterns. For benchmarks:

1. Load math benchmark patterns as Grammar Galaxy entries (using `MathProceduralizer` from chain code)
2. Load ARC transformation patterns as Grammar Galaxy entries (rotation, reflection, recolor -- already in sovereign architecture)
3. These become TRM-navigable knowledge, not hardcoded Python logic

**Phase 0C: End-to-End Benchmark Validation**

1. Run ARC-AGI-2 through Tablet boundary -> measure accuracy
2. Run GSM8K through Tablet boundary -> measure accuracy (target: 30-50% from 1.39% baseline)
3. Run MATH through Tablet boundary -> measure accuracy (target: 15-25% from 1.13% baseline)
4. Each run produces execution events -> feeds Track 2 (PTX promotion), Track 3 (specialist learning), Track 4 (grammar evolution)

---

## Revised Priority Order

```
PRIORITY 1 (IMMEDIATE):  Track 0A -- Tablet Boundary Formalization
                          Why: Without this, K3D cannot interface with ANY benchmark.
                          Scope: TabletIngest + TabletEmit + headless MPC surface.
                          Leverage: existing arc_agi_2_adapter, ActionBuffer (Step7.2),
                                    run_all_benchmarks.py, run_sovereign_math_benchmarks.py

PRIORITY 2 (NEXT):       Track 0B -- Benchmark Grammar Rules
                          Why: TRM needs navigable knowledge to solve benchmark tasks.
                          Scope: Math patterns + ARC patterns as Grammar Galaxy entries.
                          Leverage: MathProceduralizer (chain code), existing Grammar Galaxy,
                                    math_symbol_galaxy.py backlog, 9 bootstrap specialists

PRIORITY 3 (VALIDATE):   Track 0C -- End-to-End Benchmark Runs
                          Why: Proves the system works and generates real telemetry.
                          Scope: ARC + GSM8K + MATH through Tablet boundary.
                          Leverage: run_all_benchmarks.py, execution event recording,
                                    grammar evolution, specialist learning

PRIORITY 4 (ONGOING):    Tracks 1-5 Follow-on
                          Why: These compound value from benchmark telemetry.
                          - T2 promotes hot bridge paths identified during benchmark runs
                          - T3 learns specialist routing from benchmark execution outcomes
                          - T4 evolves grammar rules from successful benchmark-solving patterns
                          - T1 scene orchestration for benchmark visualization/inspection
                          - T5 sleep consolidation for post-benchmark knowledge compaction

DEFERRED:                Track 6 -- PDF-as-RPN (not benchmark-critical)
```

---

## Key Architectural Decisions

### 1. Tablet = Benchmark Interface (Not a New System)

Do NOT create a separate "benchmark adapter framework." The Tablet IS the adapter. It already has:
- Action buffer contract (288 bytes, GPU struct)
- Navigation/dialogue/memory routing (Step7.2 ActionRouter)
- Door system for entering/exiting contexts

For benchmarks, a task arriving is an action routed through the Tablet. A result leaving is the Tablet emitting through its standard contract. This keeps the architecture unified -- benchmarks are just another user of the Tablet, not a special case.

### 2. Headless Tablet Mode

The viewer Tablet is visual (HTML/WebGL). For benchmarks, the Tablet needs a headless mode that:
- Accepts JSON task input (no rendering)
- Routes through the same Galaxy/TRM/PTX pipeline
- Emits JSON result output (no rendering)
- Records the same execution events as the visual path

This is the MPC surface Daniel referenced. The "model" is K3D's sovereign pipeline. The "predictive control" is the TRM's learned navigation. The "interface" is the Tablet in headless mode.

### 3. Sovereignty at the Boundary

The Tablet boundary is the ONE place where external formats are allowed:
- Ingestion side: JSON parsing, text tokenization, grid parsing -- can use standard Python
- Emission side: result formatting, JSON serialization -- can use standard Python
- Everything between ingestion and emission: sovereign (Galaxy + PTX only)

This matches the existing sovereignty principle: ingestion is flexible, hot path is sovereign.

### 4. Grammar Rules as Benchmark Knowledge

Instead of hardcoding benchmark-solving logic:
- Each benchmark type contributes Grammar Galaxy rules
- TRM navigates these rules to solve new tasks
- Successful patterns auto-promote to canonical rules (Track 4)
- Failed patterns become anti-patterns (contrastive learning)

This is the "understand in the standard way" Daniel described -- K3D doesn't extract answers, it navigates knowledge to solve problems.

---

## Implementation Guidance

### Do NOT redesign. Enhance what exists:

1. **`benchmarks/arc_agi_2_adapter.py`** -> generalize into Tablet boundary pattern
2. **`scripts/run_sovereign_math_benchmarks.py`** -> replace answer extraction (lines 133-146, 176-181) with TRM navigation via Tablet
3. **`scripts/run_all_benchmarks.py`** -> add Tablet headless mode integration
4. **`knowledge3d/knowledgeverse/tool_execution.py`** -> already records everything; ensure benchmark runs feed the same journal
5. **`knowledge3d/knowledgeverse/trm_navigator.py`** -> already routes; ensure benchmark tasks route through standard navigator path
6. **Chain code sources**: ActionBuffer (Step7.2 lines 77-687), MathProceduralizer (CODEX_SOVEREIGN_SWARM lines 3214-4293), 9 bootstrap specialists (CODEX_SOVEREIGN_SWARM lines 771-1041)

### Success Criteria

1. K3D accepts standard benchmark input through Tablet headless boundary
2. K3D emits standard benchmark output through Tablet headless boundary
3. Internal processing is fully sovereign (Galaxy + PTX)
4. Execution events are recorded for all benchmark runs
5. Grammar Galaxy grows from successful benchmark-solving patterns
6. No benchmark-specific Python logic in the hot path
7. GSM8K accuracy >= 30% (from 1.39% baseline)
8. ARC-AGI accuracy measurably improved over current baseline

---

## How to Use This Document

1. Start with Track 0A -- formalize the Tablet boundary for headless benchmark I/O
2. Track 0B immediately follows -- load benchmark-solving knowledge into Grammar Galaxy
3. Track 0C validates -- run real benchmarks and measure
4. Tracks 1-5 continue feeding from the telemetry generated by benchmark runs
5. Do not redesign internal machinery -- it works. The gap is the I/O boundary.

**The system is built. Now it needs a front door.**
