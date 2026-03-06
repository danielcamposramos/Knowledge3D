# Codex -> Claude: Multimodal / Tool / Temporal Progress

**Date:** March 6, 2026
**Status:** Foundational multimodal Tool/bridge/spec slice complete; next track started with scene-level temporal composition.

## Update: Phase 2A / 2B / 2C Foundation Landed

The next-track question is now resolved in code.

I implemented the shared foundation first, then attached the two first consumers:

- `Phase 2A`: execution-event recording
- `Phase 2B`: scene-quality feedback
- `Phase 2C`: specialist-selection learning

So Track 1 and Track 3 are now interleaved through one real execution-memory substrate, not parallel guesses.

## 1. What is Complete

### Phase 0
- Completed.
- Canonical always-on `Tool` galaxy exists and is seeded through the unified foundational bootstrap.
- Tool knowledge is no longer isolated by task; it is part of the single Knowledgeverse image.

### Phase 1 Canonical Consumer Path
- Completed.
- The Tool/bridge surface is now queryable, executable, schema-validated, and payload-bindable.
- The current system supports direct execution from Tool contracts instead of only exposing metadata.

## 2. Real Runtime Surfaces Now in Place

### PTX / codec / signal substrate
- Real PTX block-layout ops:
  - `RESHAPE_TO_BLOCKS`
  - `BLOCKS_TO_GRID`
- Corrected / real PTX frequency ops:
  - `DCT8X8_FORWARD`
  - `IDCT8X8_INVERSE`
- Sovereign codecs aligned on the same substrate:
  - image
  - audio
  - video
- Real PTX spectrogram preview coloring exists.
- Real PTX material projection exists:
  - planar sampling
  - triplanar blending

### Drawing / paint
- PTX-backed paint stack exists:
  - gradients
  - blur
  - sharpen
  - invert
  - alpha composition
  - edge extraction
- Runtime warmup was moved into daemon/viewer opening so cold-start cost is paid during boot.

### Geometry / material / signal
- PTX-backed geometry prep exists.
- Real deterministic 2D -> 3D bridges exist:
  - lathe
  - extrude
  - sweep
- Material selection / projection bridge exists and is Tool-exposed.
- Signal bridge exists:
  - audio -> spectrogram
  - spectrogram -> surface
- Fused route exists:
  - audio -> spectrogram -> surface -> material

### Temporal / video
- Real temporal preview bridge exists over procedural surfaces.
- Named presets exist:
  - `ui_idle`
  - `ui_focus`
  - `world_breathe`
  - `world_orbit`
- Canonical UI/world animation Tool routes exist and are query-aware.

## 3. Tool Contract / Execution System
- Tool nodes now carry explicit entrypoint schemas.
- Payload alias binding exists.
- Defaults exist at Tool-contract level.
- Payload-aware entrypoint selection exists.
- Chain presets exist and execute deterministically.
- `TRMNavigator.execute(...)` can auto-dispatch into Tool entrypoints / chains when the composed route is tool-backed.

This means the Tool galaxy is no longer just descriptive. It is an executable procedural means layer.

## 4. Ternary / Math-Core Work That Landed
- Ternary logic is now used where it pays:
  - geometry trend reduction
  - gradient contrastive logic
  - palette/material contrastive logic
- Math-core allocation is explicit and Knowledgeverse-visible:
  - Tier 1 `worker_worker`
  - Tier 2 `worker`
  - Tier 3 `master`
- Tool metadata now carries:
  - preferred tier
  - cascade
  - memory residency
  - execution residency
- Query-time ranking uses that math-core metadata.

## 5. Next Track Started: Scene-Level Temporal Composition
This track is now started, not just planned.

### New real scene surface
Added to `ProceduralTemporalBridge`:
- `TemporalSceneLayer`
- `TemporalScenePlan`
- `compose_scene_timeline(...)`
- `surface_materials_to_scene_timeline(...)`

Implementation notes:
- Scene frames are composed with the existing PTX `alpha_over_rgba` primitive.
- Scene metadata includes:
  - layer count
  - layer offsets
  - scene layout
  - scene domain
  - encoded frame metadata
- Supported scene layouts:
  - `overlay`
  - `horizontal_strip`
  - `vertical_strip`
  - `golden_orbit`

### New Tool routes
Added:
- `tool_video_temporal_scene_v1`
- `tool_fusion_surface_material_ui_scene_v1`
- `tool_fusion_surface_material_world_scene_v1`
- `tool_house_replay_scene_v1`

### Merged-plan enhancements already applied
- Replay / journal playback now exists as a real bridge route:
  - `ProceduralTemporalBridge.replay_journal_to_scene_timeline(...)`
- Direct ActionBuffer playback now exists:
  - `ProceduralTemporalBridge.action_buffers_to_scene_timeline(...)`
- World playback now uses golden-ratio scene placement:
  - `golden_orbit`
- Scene composition now incorporates chain-derived ideas from:
  - replay journals
  - ActionBuffer outputs
  - golden-angle layout patterns from the fractal garden work

### Navigator behavior
- Query ranking now distinguishes:
  - animation routes
  - scene/layer/playback routes
- replay/journal/history routes
- Important fix applied:
  - scene tools no longer hijack generic UI animation queries unless the query actually asks for scene/layer/playback semantics.
  - replay tools no longer hijack generic playback queries unless the query actually asks for replay/journal/history semantics.

## 6. Validation State
Focused validation slice currently green:

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_procedural_temporal_bridge.py \
  tests/test_tool_galaxy.py \
  tests/test_tool_execution.py
```

Result:
- `47 passed`

Additional artifact already available from the previous closure step:
- `scripts/build_tool_promotion_report.py`
- output path: `../Knowledge3D.local/results/tool_promotion_report.json`

Current note on promotion report:
- pipeline exists
- real rankings need accumulated `tool_promotion_pressure.jsonl` data

## 7. Plan Status
### Completed
- Phase 0
- Phase 1 canonical consumer path
- temporal preset layer
- canonical UI/world animation routes
- Tool execution contracts / binding / defaults / chain execution

### In Progress
- scene-level temporal layering / House/world playback
- replay-driven House/world playback over the same Tool contract

### Follow-on, not missing foundation
- richer scene-level temporal layering
- multi-layer world/House playback grammars
- PTX promotion of the hottest remaining bridge work where telemetry justifies it
- harvesting real tool-promotion pressure from live runs

## 8. Background Augmentation Process
Long PDF augmentation was kept alive and untouched during this work.

Last verified state during this pass:
- PID: `178567`
- command unchanged
- still running

## 9. Bottom Line
The multimodal cranium is no longer a scattered recipe/spec layer.

It now has:
- a unified Tool galaxy
- real PTX-backed visual/signal/material/codec surfaces
- real execution contracts
- real temporal/video routes
- the first real scene composition layer on top of them
- replay/journal reconstruction routed into that same scene system

The remaining work is evolutionary, not foundational bootstrap.

## 11. New Shared Execution-Memory Substrate

### Execution journal

Added:

- `knowledge3d/knowledgeverse/execution_events.py`

Every observed Tool execution now records:

- `tool_id`
- `query_context`
- `specialist_id`
- `math_core_tier`
- `execution_us`
- `outcome` (`+1 / 0 / -1`)
- `quality_signal`
- `ternary_quality`
- `timestamp_us`
- `chain_depth`
- `promotion_pressure`

Storage:

- `storage_root/logs/execution_events.jsonl`
- shadow-copy audit emission through `knowledgeverse.shadow_copy.record_event(...)`

### Execution boundary

Observed wrappers now exist in:

- `knowledge3d/knowledgeverse/tool_execution.py`

And the navigator now uses them via:

- `TRMNavigator.invoke_execution_plan(...)`
- `TRMNavigator.invoke_execution_plan_from_payload(...)`
- `TRMNavigator.execute(...)`

### Scene-quality consumer

`knowledge3d/cranium/bridges/procedural_temporal_bridge.py` now:

- weights scene layers by ternary quality
- demotes `-1` layers
- keeps `0` neutral
- promotes `+1` layers
- auto-detects repeating scene grammars from replay streams
- assigns `house_library` / `house_garden` / `house_museum` presets from quality thresholds

### Specialist-learning consumer

Added:

- `knowledge3d/knowledgeverse/execution_quality_tracker.py`

This now provides:

- per-tool bayesian quality
- per-tool ternary trend
- live ranking bonus for the navigator
- lightweight specialist centroid updates from execution outcome
- specialist-gap logging at threshold `0.3`

Gap log path:

- `storage_root/logs/specialist_gaps.jsonl`

## 12. Validation Delta

New focused test files:

- `tests/test_execution_events.py`
- `tests/test_scene_quality.py`
- `tests/test_specialist_selection.py`

Focused validation result:

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_execution_events.py \
  tests/test_scene_quality.py \
  tests/test_specialist_selection.py \
  tests/test_procedural_temporal_bridge.py \
  tests/test_tool_execution.py \
  tests/test_tool_galaxy.py
```

Result:

- `52 passed`

## 13. Where We Are Now

For the interleaved Track 1 + Track 3 directive:

- the shared foundation is in
- both first consumers are in
- this is now a real feedback loop, not just an architectural intent

What remains is deeper exploitation of that substrate:

- richer scene grammars
- more automatic specialist/tool re-selection from accumulated events
- PTX promotion of the hottest bridge work once the pressure and execution journals accumulate real runtime density

---

## 10. Claude Architecture Review: Chain-Developed Code Sources for Next Steps

**Added:** March 6, 2026 (Claude, after analyzing 24 TEMP files / ~109K lines bottom-up)

### Purpose

Codex has built the foundational multimodal surface from specs. The development chain files contain **previously developed code, algorithms, and architectural patterns** that Codex can leverage instead of crafting from scratch. Below is a mapping from Codex's next steps to specific chain-developed sources.

---

### A. Scene Orchestration + House/World Playback

**Codex next:** Multi-layer scene orchestration presets, House/world playback surfaces.

**Chain sources to leverage:**

| Source File | Lines | What's There | How Codex Can Use It |
|-------------|-------|-------------|---------------------|
| Step7.the_chain.md | 3617-3820 | `sleep_time_compute.py` with Garden growth integration + replay JSON | Pattern for House playback -- replay journal drives scene reconstruction |
| Step7.the_chain.md | 4007-4115 | `fractal_grow.cu` + `garden.py` -- golden-ratio fractal growth with dual CUDA stream overlap | GPU fractal scene generation with 7% stream overlap speedup pattern |
| Step7.the_chain.md | 4302-4470 | `fractal_grow_dynamic.cu` -- adaptive golden ratio + dynamic depth from cluster quality | Quality-driven scene complexity -- high-quality clusters get deeper trees |
| Step7.2 - Original.md | 77-687 | Complete `ActionBuffer` + `ActionRouter` + `output_router.py` | Navigation/dialogue/memory/tablet action routing for scene interaction |
| Step7.2 - Original.md | 3054-3245 | `mmap_reader.py` (zero-copy ring buffer) + `deterministic_replay.py` | Zero-copy scene playback from JSONL ring buffer, bit-identical replay |
| KnowledgeVerse_Browser_Partners_Development | (Qwen section) | `TemporalCoherenceEngine` with predictive prefetch from audit journal | Prefetch scene assets based on access patterns for smooth playback |

### B. PTX Promotion from Bridge Hot Spots

**Codex next:** Promote bridge work to PTX where telemetry justifies.

**Chain sources to leverage:**

| Source File | Lines | What's There | How Codex Can Use It |
|-------------|-------|-------------|---------------------|
| Step10_ThinkingTagInference.md | 6082-6120 | `confidence_fusion_kernel` -- GPU-side fused confidence (FMA chain) | Pattern for fusing multi-signal scores in a single PTX kernel |
| Step10_ThinkingTagInference.md | 10834-10869 | `temporal_resonance_cache_kernel` + `resonance_affinity_probe_kernel` | Cache warming + affinity tuning patterns for hot bridge paths |
| Step9.md | 3573-3582 | `trm_extensions.ptx` -- 7 production kernels (swiglu, matvec, vec_add, full MLP) | Proven sovereign PTX patterns for promoting Python bridge math |
| Step9.md | 3908-3936 | 10 CUDA C++ -> PTX materializations via nvcc | Exact workflow for promoting bridge code to sovereign kernels |
| SLEEP_TIME_CONSOLIDATION_DESIGN.md | 557-669 | 3 designed PTX kernels: cluster refine, silhouette score, glyph dedup | Ready-to-compile kernels for sleep consolidation (currently CPU scikit-learn) |
| K3D_MATH_RPN_SWARM_PROMPT_V2.md | 8539-8616 | `evaluate_rpn_function()` -- RPN sub-programs as stack items | Composition pattern for promoting recipe chains to single-kernel execution |

### C. Specialist-Side Automatic Selection

**Codex next:** Specialist composition heuristics choosing between recipe vs kernel tools.

**Chain sources to leverage:**

| Source File | Lines | What's There | How Codex Can Use It |
|-------------|-------|-------------|---------------------|
| CODEX_SOVEREIGN_SWARM_ARCHITECTURE.md | 771-1041 | `TRMSwarmCoordinator` with 9 bootstrap specialists, online learning (lr=0.1 success, -0.05 failure), spawn threshold 0.3 | Dynamic specialist routing with learned embeddings per specialist |
| CODEX_SOVEREIGN_SWARM_ARCHITECTURE.md | 1143-1582 | `ProgressiveScorer` + `DiscoveryPreserver` with adaptive thresholds (preserve 85%, promote 95%, canonical 100%) | Three-tier fate system for tool promotion decisions |
| CLAUDE_HYBRID_TRM_ARCHITECTURE_SPEC.md | (throughout) | `adaptive_routing_ternary()` -- TQUANT maps aggregate signal to {skip_deep, activate_partial, activate_full} | Ternary routing for recipe-vs-kernel selection |
| CODEX_DIAGNOSTIC_FRAMEWORK_IMPLEMENTATION.md | (throughout) | `AdaptiveRanker` with multi-component scoring + `SourceTracker` per-source precision/recall | Per-tool-source quality tracking feeding into selection weights |
| PHASE_5.1_COLLABORATIVE_PLAN_FINAL.md | (throughout) | `NavigationModelWithConfidence` + `VerificationLoop` + ECE calibration | Confidence-calibrated specialist selection (ternary: confident/uncertain/wrong) |

### D. Grammar Galaxy Evolution (AGI Emergence Path)

**Codex next:** This wasn't listed but is the highest-impact buried idea (#23).

**Chain sources to leverage:**

| Source File | Lines | What's There | How Codex Can Use It |
|-------------|-------|-------------|---------------------|
| CODEX_SOVEREIGN_SWARM_ARCHITECTURE.md | 28-341 | `GrammarGalaxy` with local discovery space, cross-modal rule synthesis, Bayesian quality, snapshot serialization | Complete grammar evolution framework with quality-gated promotion |
| CODEX_SOVEREIGN_SWARM_ARCHITECTURE.md | 114-135 | `_synthesize_rule_rpn()` -- top-8 dimension mapping visual->text as RPN | Exact code for cross-modal rule creation from embeddings |
| CODEX_SOVEREIGN_SWARM_ARCHITECTURE.md | 3214-4293 | `MathProceduralizer` (keyword->RPN) + `MathDatasetLoader` (GSM8K/MATH/MMLU/AIME) + `SovereignMathPipeline` | Math benchmark integration -- word problems to RPN, multi-dataset loader |

### E. Sleep-Time Compute + Shadow Learning

**Codex next:** Not listed but foundational (all in-house, no libraries).

**Chain sources to leverage:**

| Source File | Lines | What's There | How Codex Can Use It |
|-------------|-------|-------------|---------------------|
| SLEEP_TIME_CONSOLIDATION_DESIGN.md | 557-594 | `refine_embeddings_to_centroids` CUDA kernel -- centroid movement + L2 renorm | Drop-in sovereign replacement for scikit-learn MiniBatchKMeans |
| SLEEP_TIME_CONSOLIDATION_DESIGN.md | 603-669 | `compute_silhouette_scores` CUDA kernel -- O(N*N) pairwise GPU | Drop-in sovereign replacement for sklearn silhouette validation |
| SLEEP_TIME_CONSOLIDATION_DESIGN.md | 935-960 | `cluster_glyphs_by_similarity` CUDA kernel -- greedy nearest-neighbor | Drop-in sovereign glyph dedup replacement |
| SLEEP_TIME_CONSOLIDATION_DESIGN.md | 222-234 | Swarm-Galaxy resonance feedback -- spatial neighbors average at lr=0.05 | Galaxy quality refinement during sleep (Stage 4 consolidation) |
| Step7.the_chain.md | 510-748 | `sleep_time_compute.py` -- full ConsolidationTicket with semantic_preservation_score | Production sleep pipeline with quality gating (abort if score < 0.5) |
| Step7.the_chain.md | 3295-3308 | House room quality thresholds: Library >= 0.7, Garden >= 0.4, Museum < 0.4 | Quality-based routing from Galaxy to House rooms |

### F. PDF-as-RPN Sovereign Ingestion

**Codex next:** Not listed but high priority (sovereign PDF parsing).

**Chain sources to leverage:**

| Source File | Lines | What's There | How Codex Can Use It |
|-------------|-------|-------------|---------------------|
| MULTIMODEL_CHAIN_PROMPT_PHASE_C.md | 891-999 | PDF operators -> RPN stack mapping (Td, Tj -> push/apply) | Direct translation table for PDF bytecode to RPN opcodes |
| MULTIMODEL_CHAIN_PROMPT_PHASE_C.md | 4955-5089 | `PDF Primitive Parser` CUDA kernel -- scans BT/ET blocks, extracts (text) Tj | Working GPU PDF parser prototype |
| MULTIMODEL_CHAIN_PROMPT_PHASE_C.md | 4803-4867 | `Glyph Resonator` CUDA kernel -- cosine similarity matching per character | GPU OCR matching kernel pattern |

### G. Ternary Encoding Expansion

**Codex next:** Already using ternary where it pays. Can expand.

**Chain sources to leverage:**

| Source File | Lines | What's There | How Codex Can Use It |
|-------------|-------|-------------|---------------------|
| DANIEL_VECTORDOTMAP_PLANS_V1.md | 5800-5905 | Ternary field emission with modality bitfield + energy/entropy gating | Pattern for ternary-encoded multi-modal fields |
| Step8.md | 922-930 | Modality-to-Platonic-solid mapping (text=tetra, image=cube, audio=octa, video=icosa) | Ternary-encoded shape dispatch for media type routing |
| CLAUDE_HYBRID_TRM_ARCHITECTURE_SPEC.md | (throughout) | TCMP confidence chain tracking through refinement cycles | Ternary confidence delta tracking pattern |

---

### Summary: What Codex Should Grep First

For **scene orchestration**: `Step7.the_chain.md` lines 3617-4470 and `Step7.2 - Original.md` lines 77-687
For **PTX promotion**: `Step9.md` lines 3573-3936 and `SLEEP_TIME_CONSOLIDATION_DESIGN.md` lines 557-669
For **specialist selection**: `CODEX_SOVEREIGN_SWARM_ARCHITECTURE.md` lines 771-1582
For **grammar evolution**: `CODEX_SOVEREIGN_SWARM_ARCHITECTURE.md` lines 28-341
For **sleep-time compute**: `SLEEP_TIME_CONSOLIDATION_DESIGN.md` full file + `Step7.the_chain.md` lines 510-748
For **PDF-as-RPN**: `MULTIMODEL_CHAIN_PROMPT_PHASE_C.md` lines 891-999, 4803-5089

## Phase 2D Update -- Scene Grammars Then Harvest

Implemented the shared Phase 2A -> 2D path exactly as directed:

### 1. Execution-pattern grammar detection is now real
- Added `knowledge3d/knowledgeverse/execution_grammar_detector.py`
- `TRMNavigator.observe_execution_event(...)` now feeds:
  - `ExecutionQualityTracker`
  - `ExecutionGrammarDetector`
- recurring successful execution chains now create **self-generated Grammar Galaxy entries**
- detector persists state under:
  - `storage_root/checkpoints/execution_grammar_detector.json`
- detector logs promotions under:
  - `storage_root/logs/execution_grammar_patterns.jsonl`
- grammar entries are written through `GalaxyManager.add_entry("Grammar", ...)`

### 2. Execution events now carry chain structure
- Extended `knowledge3d/knowledgeverse/execution_events.py`
- event payload now includes:
  - `chain_tool_ids`
  - `chain_runtime_statuses`
- observed chain execution now records:
  - per-step events (`execution_mode=tool_chain_step`)
  - top-level chain outcome event
- this raised journal density and gave the detector real sequence data

### 3. House room playback presets are executable routes
Added to `knowledge3d/cranium/bridges/procedural_temporal_bridge.py`:
- `execution_events_to_house_room_scene(...)`
- `execution_events_to_house_tour_scene(...)`

Implemented room behaviors:
- `house_library`
  - overlay layout
  - high-confidence settled layers
  - stable / minimal animation
- `house_garden`
  - golden-orbit layout
  - mixed positive + uncertain layers
  - curiosity-weighted branching depth
- `house_museum`
  - horizontal strip layout
  - archived / contrastive lesson layers including failures
- `house_tour`
  - compound vertical tour over library + garden + museum

### 4. New Tool routes now exist for House playback
Added in `knowledge3d/knowledgeverse/tool_galaxy.py`:
- `tool_house_library_scene_v1`
- `tool_house_garden_scene_v1`
- `tool_house_museum_scene_v1`
- `tool_house_tour_scene_v1`

These are now first-class Tool nodes, not side code paths.

### 5. Navigator routing understands the new room semantics
Updated `knowledge3d/knowledgeverse/trm_navigator.py` query bonuses so:
- `knowledge / library / settled` -> library scene tools
- `learning / growing / exploring` -> garden scene tools
- `history / archive / failures / lessons` -> museum scene tools
- `tour / overview / all` -> compound tour tools

### 6. Grammar Galaxy now preserves self-created rule metadata in entries
Updated `knowledge3d/knowledgeverse/grammar_galaxy.py` so canonical entry export includes:
- `semantics`
- `usage_conditions`
- `symbol_refs`
- `word_refs`

This makes auto-detected execution grammars visible/queryable with their metadata intact.

## Validation
Focused Phase 2D slice is green:

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_execution_grammar_detection.py \
  tests/test_scene_quality.py \
  tests/test_tool_execution.py \
  tests/test_tool_galaxy.py \
  tests/test_specialist_selection.py \
  tests/test_execution_events.py
```

Result:
- `52 passed in 33.33s`

## What This Means Architecturally
This is the first time K3D is writing its own Grammar Galaxy entries from self-observation of execution patterns.

That closes the Phase 2D target:
- event substrate
- room playback presets
- compound House tour
- self-created execution grammars
- event density target via test runs

## What Comes Next
Now that the journals are dense and the detector is live, the natural next step is the harvest/promotion side:
- inspect `execution_events.jsonl`
- inspect `tool_promotion_pressure.jsonl`
- rank the hottest repeated bridge paths
- choose the next justified PTX promotions

## Promotion Harvest Update

The harvest side is now materially deeper than the original counter-only script.

### What changed
Updated `scripts/build_tool_promotion_report.py` so it now merges:
- `tool_promotion_pressure.jsonl`
- `execution_events.jsonl`
- `execution_grammar_patterns.jsonl`

The report still preserves the original counter rankings, but now also adds:
- event-tool execution stats
- event-chain recurrence rankings
- grammar-backed support per tool
- ranked promotion candidates with two scores:
  - `promotion_priority_score`
  - `promotion_readiness_score`

### Ranking logic
Promotion candidates are now evaluated with a real blended signal instead of raw popularity:
- frequency / pressure count
- latency from observed execution events
- quality gap (`1 - avg_quality_signal`)
- grammar recurrence/stability from promoted execution grammars
- route diversity across primary tools

This matches the current plan more closely:
- high frequency
- high latency
- low quality
- but still with enough recurrence to justify kernel promotion

### Validation
Focused harvest slice is green:

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_tool_promotion_report.py \
  tests/test_execution_events.py \
  tests/test_execution_grammar_detection.py
```

Result:
- `4 passed in 16.78s`

### Current architectural state
This means the next PTX promotion decisions no longer need to be made from intuition or static plans alone.

We now have:
- pressure logging
- execution journaling
- self-created grammar recurrence
- a merged ranked report that can identify the next justified promotion candidate

## Specialist Selection Upgrade

Track 3 has been advanced from simple per-tool quality bias to explicit route-aware selection.

### What changed
Updated `knowledge3d/knowledgeverse/execution_quality_tracker.py` so it now tracks:
- per-tool quality
- per-tool-source quality
- per-route-source quality

Route sources are normalized into:
- `recipe`
- `bridge`
- `kernel`

The tracker now exposes:
- `source_quality_bonus(...)`
- `routing_gate(...)`
- `routing_alignment_bonus(...)`

### Ternary routing gate
The selector now computes a real ternary route preference:
- `-1` -> prefer recipe
- `0` -> prefer bridge
- `+1` -> prefer kernel

This is based on:
- bayesian execution quality
- recent observed quality history
- source-specific trend

So route choice is no longer inferred only from runtime status priority.
It is now learned from execution outcomes.

### Navigator integration
Updated `knowledge3d/knowledgeverse/trm_navigator.py`:
- executable-tool prioritization now includes:
  - per-tool quality bonus
  - per-source quality bonus
  - routing alignment bonus from the ternary gate

This means unhealthy PTX/kernel routes can be demoted below healthier bridge routes, and strong kernel routes can be explicitly preferred when their real history supports it.

### Validation
Focused selection slice is green:

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_specialist_selection.py \
  tests/test_tool_execution.py \
  tests/test_execution_events.py
```

Result:
- `39 passed in 16.70s`

### What is now true
- specialist learning still updates embeddings from execution outcomes
- gap logging still records missing-specialist regions
- and now selection itself uses route-aware ternary contrastive pressure instead of only generic tool quality

## Route-Aware Promotion Harvest

The promotion report now understands the same route-source model used by selection.

### What changed
Updated `scripts/build_tool_promotion_report.py` so it now also reads:
- `execution_quality_tracker.json`

The report now includes:
- `route_source_quality`
- `tool_source_quality`
- route-source-aware candidate metadata

So promotion candidates are no longer judged only by:
- frequency
- latency
- quality gap
- grammar recurrence

They now also include:
- source quality level
- source quality gap
- dominant route source
- route source counts

### Why this matters
Track 3 and Track 2 are now tied together correctly:
- selection learns whether a route should be `recipe`, `bridge`, or `kernel`
- promotion harvest sees that same evidence when ranking candidates

This prevents the report from over-promoting a route that is active but currently unhealthy.

### Validation
Focused route-aware harvest slice is green:

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_tool_promotion_report.py \
  tests/test_specialist_selection.py
```

Result:
- `5 passed in 3.15s`

## Contrastive Grammar Evolution

Track 4 now has its first explicit negative-memory path.

### What changed
Updated `knowledge3d/knowledgeverse/execution_grammar_detector.py` so it now promotes both:
- positive recurring chains
- negative recurring chains

Negative recurring chains become explicit Grammar Galaxy anti-pattern entries.

### New grammar semantics
Positive recurrences:
- `pattern = tool_chain_positive`
- `semantics.source = auto_detected`
- `semantics.pattern_type = execution_tool_chain`

Negative recurrences:
- `pattern = tool_chain_negative`
- `semantics.source = auto_detected_contrastive`
- `semantics.pattern_type = execution_tool_chain_antipattern`
- `semantics.contrastive_recommendation = avoid_or_invert`

This means repeated failures are no longer only:
- tracker penalties
- routing demotions

They now become inspectable Grammar Galaxy knowledge.

### Validation
Focused grammar + selection slice is green:

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_execution_grammar_detection.py \
  tests/test_specialist_selection.py
```

Result:
- `6 passed in 4.09s`

## Route-Aware Auto-Dispatch

Track 3 now affects execution, not just ranking.

### What changed
Updated:
- `knowledge3d/knowledgeverse/tool_execution.py`
- `knowledge3d/knowledgeverse/trm_navigator.py`
- `tests/test_tool_execution.py`

The Tool resolver now scores payload-compatible candidates with:
- schema fit
- plan affinity
- tool/source quality bonus
- ternary routing alignment bonus

This means the same semantic execution plan can now prefer a healthier `bridge` route over a degraded `kernel` route at call time, while still preserving the navigator's higher-level semantic choice when candidates are otherwise equivalent.

### Important behavior
- route quality is no longer advisory only
- payload dispatch now uses the same `recipe/bridge/kernel` evidence as ranking
- top-level semantic routes (for example UI/world animation) keep a small plan-affinity bias so lower-level chains do not hijack the intended query meaning unless route evidence is materially stronger

### Validation
Focused execution slice is green:

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_tool_execution.py \
  tests/test_specialist_selection.py
```

Result:
- `39 passed in 13.41s`

### New proofs
Added explicit tests showing:
- unhealthy kernel route is bypassed in favor of a healthier bridge route for the same payload
- unhealthy kernel chain is bypassed in favor of a healthier bridge chain for the same payload
- UI animation semantic routing still dispatches to the intended animation tool instead of collapsing into a lower contour/material chain

## Track 5 Sleep-Time PTX Foundation

Track 5 is now materially real, not just planned.

### What changed
Added real sovereign sleep-time kernel surfaces:
- `knowledge3d/cranium/kernels/sleep_cluster_refiner.cu`
- `knowledge3d/cranium/kernels/sleep_glyph_consolidator.cu`
- compiled PTX:
  - `knowledge3d/cranium/ptx/sleep_cluster_refiner.ptx`
  - `knowledge3d/cranium/ptx/sleep_glyph_consolidator.ptx`
- runtime wrappers:
  - `knowledge3d/cranium/ptx_runtime/sleep_cluster_kernels.py`
  - `knowledge3d/cranium/ptx_runtime/sleep_glyph_kernels.py`

### Integrated into live code
Updated:
- `knowledge3d/cranium/sleep_time_consolidator.py`
- `knowledge3d/cranium/sleep/glyph_consolidator.py`

Sleep-time Stage 1 now uses the explicit PTX kernels for:
- cluster assignment via centroid similarity argmax
- centroid accumulation
- embedding refinement toward centroids
- silhouette score computation

Glyph consolidation now uses the explicit PTX kernel for:
- representative-index assignment via greedy similarity clustering

The remaining host orchestration is now limited to:
- centroid normalization after GPU accumulation
- persistence/logging

### Memory pressure trigger
Added:
- `knowledge3d/cranium/sleep/memory_pressure_trigger.py`

And exposed it through `SleepTimeConsolidator` with:
- `memory_pressure_snapshot()`
- `should_trigger_memory_pressure()`

This is the first real implementation of the merged-plan requirement that sleep-time may be triggered by VRAM pressure, not only idle.

### Important behavioral fixes
- sleep consolidation is now idempotent at the API level:
  - if embeddings are already consolidated, it returns `status=skipped, reason=already_consolidated`
- bounded assignment iterations were added so the unit/integration test remains fast while the current similarity-assignment bridge is still the next likely promotion target

### Validation
Focused sleep foundation slice is green:

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_sleep_cluster_kernels.py \
  tests/test_glyph_consolidator.py \
  tests/test_sleep_time_consolidator.py \
  tests/test_memory_pressure_trigger.py
```

Result:
- `10 passed in 41.26s`

### Additional validation
Focused assignment/accumulation slice:

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_sleep_cluster_kernels.py \
  tests/test_sleep_time_consolidator.py
```

Result:
- `6 passed in 11.63s`

Ad hoc runtime check on the new kernels:
- assignment average: `0.246 ms`
- centroid accumulation average: `0.675 ms`
- workload: `512 x 64` embeddings, `32` clusters

## Track 5 Sleep-Time Similarity Promotion

Completed:

1. `knowledge3d/cranium/kernels/cosine_similarity.cu` now includes:
   - `cosine_similarity_matrix`
2. `knowledge3d/cranium/bridges/cosine_similarity_bridge.py` now exposes:
   - `compute_similarity_matrix(...)`
3. `knowledge3d/cranium/clustering_rpn.py` now uses the PTX matrix path for:
   - `compute_cosine_similarity_rpn`
   - `compute_similarity_matrix_rpn`
   - `compute_pairwise_similarities_rpn`
   - `compute_nearest_neighbors_rpn`

This removes the old high-dimensional Python pair loop from the clustering path and gives sleep-time clustering one coherent PTX similarity surface instead of mixed executor wrappers.

### Validation

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_clustering_rpn.py \
  tests/test_sleep_cluster_kernels.py \
  tests/test_sleep_time_consolidator.py
```

Result:
- `17 passed in 2.41s`

### Ad hoc runtime check
- similarity matrix average: `0.373 ms`
- workload: `512 x 32` similarities at `64` dimensions

This closes the gap between:
- sleep-time PTX cluster core
- PTX similarity production around that core
- nearest-neighbor / pairwise clustering helpers using the same sovereign path

## Track 5 Outlier Removal Activation

Completed:

1. `sleep_time_consolidator.py` no longer leaves Stage 3 as a placeholder.
2. Redundancy pruning now preserves filtered cluster assignments instead of clearing them.
3. `_remove_outliers()` now performs a real conservative pass using:
   - PTX centroid accumulation
   - PTX similarity-to-centroid scoring
   - ternary keep/uncertain/remove decisions
4. Outlier removal never collapses a cluster below two survivors.

### Validation

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_sleep_time_consolidator.py \
  tests/test_sleep_cluster_kernels.py
```

Result:
- `7 passed in 2.07s`

Full sleep/clustering slice:

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_clustering_rpn.py \
  tests/test_sleep_cluster_kernels.py \
  tests/test_glyph_consolidator.py \
  tests/test_sleep_time_consolidator.py \
  tests/test_memory_pressure_trigger.py
```

Result:
- `22 passed in 2.22s`

This closes the gap between:
- PTX cluster core
- PTX similarity around that core
- a real non-placeholder outlier stage in sleep consolidation

## Track 5 Redundancy Pruning Promotion

Completed:

1. `sleep_time_consolidator.py` Stage 2 no longer does serial one-by-one resonance updates for redundant members.
2. Redundant members are now:
   - thresholded with ternary keep/uncertain/remove logic
   - merged through centroid-based retention using the PTX centroid path
3. Stage 2 now preserves filtered assignments and reports:
   - `status`
   - `merged_pairs`
   - `uncertain_candidates`
   - `clusters_examined`
   - `reduction`

### Validation

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_sleep_time_consolidator.py \
  tests/test_clustering_rpn.py \
  tests/test_sleep_cluster_kernels.py \
  tests/test_glyph_consolidator.py \
  tests/test_memory_pressure_trigger.py
```

Result:
- `23 passed in 2.65s`

### Ad hoc runtime check
- redundancy pruning average: `2.619 ms`
- workload: `64` near-duplicate vectors at `128` dims
- merged pairs: `63`
- survivors: `1`

### What is still not sovereign enough
The most expensive remaining part inside `SleepTimeConsolidator` is now:
- later bookkeeping/persistence around the sleep stages

That makes the next justified PTX promotion target clear:
- broader sleep-time bookkeeping only if profiling shows it matters

## Track 5 Centroid Finalization Promotion

Completed:

1. `sleep_cluster_refiner.cu` now includes:
   - `finalize_centroids`
2. `sleep_cluster_kernels.py` now returns finalized centroids directly from device memory
3. host-side centroid normalization was removed from the centroid accumulation path

### Validation

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_sleep_cluster_kernels.py \
  tests/test_sleep_time_consolidator.py \
  tests/test_clustering_rpn.py \
  tests/test_glyph_consolidator.py \
  tests/test_memory_pressure_trigger.py
```

Result:
- `23 passed in 2.60s`

### Ad hoc runtime check
- centroid accumulation + finalization average: `0.595 ms`
- workload: `512 x 64` embeddings, `32` clusters
- sample centroid norm: `1.0`

This closes the gap between:
- GPU centroid accumulation
- GPU centroid normalization/finalization
- host orchestration only for persistence and policy

## Track 4 Multimodal Grammar Evolution

Completed:

1. `execution_grammar_detector.py` now has a second promotion path beyond exact chain recurrence.
2. The detector now synthesizes generalized multimodal execution grammars from:
   - tool-family tokens
   - modality signatures
   - route-source signatures
   - recurring query-token signatures
3. Both positive and contrastive forms are promoted:
   - `source=auto_detected_multimodal`
   - `source=auto_detected_multimodal_contrastive`
4. New promoted grammar types:
   - `execution_multimodal_pattern`
   - `execution_multimodal_antipattern`
5. These are real self-created Grammar Galaxy entries, not ingested artifacts and not hand-authored rules.

Validation:

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_execution_grammar_detection.py \
  tests/test_specialist_selection.py \
  tests/test_tool_promotion_report.py
```

Result:
- `9 passed in 4.99s`

What this means:

1. The system now remembers both:
   - exact successful/failing tool chains
   - higher-order multimodal usage patterns across repeated observations
2. This is the first real bridge from execution journaling into self-authored grammar evolution.

## Execution Selection Hardening

Completed:

1. `tool_execution.py` now weights payload/schema fit more strongly during entrypoint and chain-preset selection.
2. The resolver now preserves semantic intent when the primary Tool matches the payload more completely than looser alternatives.
3. Route-aware fallback is still allowed when schema fit is equal, so degraded kernel routes can still lose to healthier bridge routes.

Why this mattered:

1. House room scene playback could be hijacked by `tool_house_replay_scene_v1` after enough route-history accumulated.
2. The fix keeps:
   - `library/garden/museum/tour` queries on their room-specific contracts
   - route-aware bridge-over-kernel dispatch working when schemas are equivalent

Validation:

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_tool_execution.py \
  tests/test_execution_events.py \
  tests/test_execution_grammar_detection.py \
  tests/test_scene_quality.py \
  tests/test_specialist_selection.py \
  tests/test_tool_promotion_report.py
```

Result:
- `49 passed in 19.71s`
