# Codex Grounded Phase 0 / Phase 1 Task Sheet

**Date**: March 6, 2026  
**Scope**: TRM multimodal enhancement, grounded in current sovereign runtime  
**Status**: Phase 0 complete, Phase 1 canonical consumer path complete

## Phase 2 Interleave Status

**New status**: Phase 2A foundation landed, with first consumers in 2B and 2C.

- `Phase 2A` execution-event recording: implemented
- `Phase 2B` scene-quality feedback: first consumer implemented
- `Phase 2C` specialist-selection learning: first consumer implemented

This means Track 1 and Track 3 now share one real execution-memory substrate instead of separate ad-hoc telemetry.

## Purpose

Translate the multimodal architecture spec into implementation work that fits the current K3D surface:

- knowledge first
- tools as knowledge
- recipes before opcodes
- no hot-path disruption

## What Exists Now

Implemented in code:

- `knowledge3d/knowledgeverse/tool_galaxy.py`
  - `ToolNode` schema
  - deterministic seed entries for multimodal fusion recipes
  - deterministic seed entries for PTX-backed procedural codecs
  - one authoritative `default_tool_entries()` builder
  - payload builder for JSONL ingestion
  - additive/idempotent `Tool.jsonl` bootstrap

- `knowledge3d/knowledgeverse/foundational_galaxy_bootstrap.py`
  - single always-on bootstrap entrypoint for operations + Reality + 3DObjects + Tool

- `knowledge3d/knowledgeverse/knowledgeverse.py`
  - Tool galaxy is part of the default always-loaded system image

- `scripts/build_multimodal_tool_nodes.py`
  - writes Tool-galaxy payload JSONL for ingestion

- `knowledge3d/knowledgeverse/galaxy_manager.py`
  - query can now resolve explicitly named on-disk galaxies such as `Tool`
  - specialist filtering now reads `metadata.modalities` and `metadata.tool_kind`
  - query now also reads `metadata.math_core` to boost tier-appropriate tools for scalar / worker / master workloads

- `scripts/fundamental_ingest_payloads.py`
  - symlink compression now extracts text from tool-node descriptions, inputs, outputs, and modalities

- `knowledge3d/knowledgeverse/trm_navigator.py`
  - generation path now consumes Tool entries, surfaces `tool_context`, and logs promotion pressure
  - tool context now synthesizes a `math_core_plan` from tier / spawn / cascade hints

- `knowledge3d/cranium/kernels/codec_ops.cu`
  - real PTX kernels for `RESHAPE_TO_BLOCKS` / `BLOCKS_TO_GRID`
  - corrected DCT8 forward/inverse fidelity

- `knowledge3d/cranium/ternary/ternary_galaxy.py`
  - backward-compatible frame metadata support for codec artifacts

- `knowledge3d/cranium/codecs/sovereign_ternary_audio_codec.py`
  - real procedural MDCT seed
  - persisted logical clip length metadata
  - decode trims padded frames back to original clip length

- `knowledge3d/cranium/codecs/sovereign_ternary_video_codec.py`
  - encode/decode now route block packing and frequency transforms through the sovereign PTX codec path

- `knowledge3d/cranium/codecs/sovereign_ternary_image_codec.py`
  - block layout now routes through sovereign PTX reshape kernels
  - non-8-aligned decode works by reconstructing padded grids then cropping
  - encode/decode now carry real Tier-2 `math_core_plan` metadata like audio/video
  - channel extraction/recombine now use vectorized host orchestration instead of manual nested loops

- `knowledge3d/cranium/bridges/procedural_signal_bridge.py`
  - audio -> spectrogram is now backed by the sovereign PTX audio codec path
  - spectrogram -> surface displacement is now executable as a deterministic bridge
  - signal Tool nodes now expose real bridge entrypoints and verification targets

- `knowledge3d/cranium/bridges/procedural_material_bridge.py`
  - signal-derived spectrogram surfaces now flow directly into material selection/projection
  - fused signal/material plans carry signal, surface, selection, and projection math-core plans together
  - planar/triplanar preview sampling is now PTX-backed through a dedicated projection runtime

- `knowledge3d/knowledgeverse/trm_navigator.py`
  - Tool context now surfaces executable PTX-backed entrypoint chains, not just descriptive metadata
  - fused signal/material tools can now be selected as primary procedural means by the navigator
  - compose/generation outputs now carry an invokable `execution_plan` for the selected tool chain

- `knowledge3d/knowledgeverse/tool_execution.py`
  - Tool execution plans now resolve into importable blueprints and instantiable callables
  - the fused signal/material route is now directly resolvable from navigator output
  - primary tool entrypoints are now invokable through the navigator with explicit argument binding

- `knowledge3d/cranium/bridges/procedural_signal_bridge.py`
  - spectrogram preview coloring is now PTX-backed through signal visualization kernels

- `knowledge3d/cranium/ptx_runtime/drawing_effects.py`
  - one canonical sovereign GPU canvas for gradients, blur, sharpen, invert, edge maps, and alpha composition
  - wraps only real PTX kernels; no fake image-edit surface

- `knowledge3d/cranium/bridges/procedural_drawing_bridge.py`
  - `render_painterly_gpu()` routes procedural drawings through PTX-backed background + filter composition
  - `edge_map_gpu()` exposes GPU Sobel structure extraction on rendered canvases

- `knowledge3d/cranium/kernels/filter_convolution.cu`
  - added real PTX kernels for:
    - RGBA -> luma
    - alpha-over RGBA composition
    - RGBA inversion

- `knowledge3d/cranium/ptx_runtime/math_core_pool.py`
  - pool now exposes runtime snapshots and explicit re-tiering
  - tier roles are surfaced as worker-worker / worker / master

- `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`
  - exposes current math-core descriptor backed by the real pool snapshot

## Seed Tool-Nodes Added

### PTX-Backed Codec Surface

1. `tool_codec_ternary_blocks_v1`
   - ternary quant/dequant + block/grid reshape
   - runtime status: `ptx_rpn_available`

2. `tool_codec_video_dct8_grid_v1`
   - DCT8 visual frequency codec surface
   - runtime status: `ptx_rpn_available`

3. `tool_codec_audio_mdct_v1`
   - batch MDCT audio spectral codec surface
   - runtime status: `ptx_rpn_available`

### Math-Core Runtime Surface

4. `tool_mathcore_tier1_scalar_worker_worker_v1`
   - real Tier-1 worker-worker RPN surface
   - runtime status: `ptx_rpn_available`

5. `tool_mathcore_tier2_vector_worker_v1`
   - real Tier-2 worker surface
   - runtime status: `ptx_rpn_available`

6. `tool_mathcore_tier3_master_v1`
   - real Tier-3 master surface
   - runtime status: `ptx_rpn_available`

7. `tool_mathcore_spawn_cascade_v1`
   - real pool / spawn / reuse / cascade policy surface
   - runtime status: `ptx_runtime_available`

### PTX-Backed Paint Surface

8. `tool_paint_gradient_backdrop_v1`
   - gradient backdrop surface
   - runtime status: `ptx_bridge_available`

9. `tool_paint_filter_stack_v1`
   - blur/sharpen/invert paint filter surface
   - runtime status: `ptx_bridge_available`

10. `tool_paint_composite_edge_v1`
   - alpha compositing + edge extraction
   - runtime status: `ptx_bridge_available`

### Multimodal Fusion Recipes

11. `tool_fusion_contour_to_mesh_v1`
   - 2D contour -> 3D mesh recipe
   - references Drawing + 3DObjects + Reality knowledge

12. `tool_fusion_surface_material_projection_v1`
   - procedural material projection recipe
   - references Drawing + 3DObjects + Reality knowledge

13. `tool_signal_audio_spectrogram_v1`
   - audio -> spectrogram recipe
   - references real Drawing cross-modal signal entries

14. `tool_signal_spectrogram_surface_v1`
   - spectrogram -> surface displacement recipe
   - bridges Audio + Drawing + 3DObjects + Reality

## Phase 0 Tasks

### 0.1 Toolify Existing Surface

Status: implemented for the first unified slice.

Use tool-nodes to express multimodal recipes before adding new opcodes.

Implemented:

- Tool payload bootstraps directly into the always-on `Tool` galaxy
- specialist routing now exposes Tool as auxiliary knowledge across visual, physics, audio, grammar, math, cartographer, and `any`
- TRM generation path now consumes tool-nodes and records their usage
- codec-backed tool-nodes are now anchored to real PTX block/frequency primitives
- image/audio/video sovereign codecs now share the same real block/frequency substrate

### 0.2 Measure Recipe Pressure

Collect which target operations recur most often in tool-nodes:

- `MESH_EXTRUDE`
- `UV_PROJECT`
- `TRIPLANAR_MAP`
- `FFT_FORWARD`
- `AUDIO_TO_SPECTROGRAM`
- `DISPLACEMENT_MAP`

Status: implemented as `storage_root/logs/tool_promotion_pressure.jsonl`.

Only after repetition and profiling should they move to opcode candidacy.

## Phase 1 Tasks

### 1.1 Draw-Extrude-Texture Minimum Viable Fusion

Status: represented and queryable.

Target demo:

`2D contour + procedural material -> textured 3D object`

Required pieces:

- Drawing contour refs
- 3D mesh/grid refs
- Reality-side surface/noise refs
- one recipe chain proving symlink reuse across galaxies

### 1.2 First Query Path

Status: implemented.

TRM or specialist routing should be able to retrieve:

- contour-to-mesh tool for visual/spatial prompts
- spectrogram tool for audio/visual prompts

without needing a new kernel.

Also implemented:

- codec tools are queryable as always-on procedural means
- paint tools are queryable as always-on procedural means
- ternary palette contrastive logic is now executable and exposed as an always-on paint/tool surface
- procedural surface material projection is now a real bridge-backed path, not only a recipe node
- contour -> lathe mesh -> selected material -> projected vertex colors is now one deterministic executable chain
- generated entries carry `tool_context` and `promotion_targets`
- promotion logs distinguish recipe tools from PTX-backed codec tools
- video codec no longer uses manual CPU block packing/unpacking for the transform path
- audio codec no longer stores fake placeholder seeds and now preserves logical clip length
- image codec no longer relies on manual block reshaping and handles padded decode correctly
- drawing now has a canonical PTX-backed paint stack for gradients, filter composition, and edge extraction
- daemon preload now warms drawing, geometry, and material bridges before the scene becomes interactive
- geometry prep now includes PTX-backed contour smoothing metadata
- extrusion is now a second real 2D -> 3D bridge beside lathe, using ternary contour-motion reduction
- sweep is now a third real 2D -> 3D bridge, using ternary width/centerline motion reduction
- math-core planning assumption updated: master -> worker -> worker-worker fanout is elastic and should be treated as spawnable under load
- Tool metadata now stores math-core tier / spawn / cascade intent in the Knowledgeverse itself
- TRM-generated entries and promotion-pressure logs now carry a synthesized `math_core_plan`
- query-time retrieval can now prefer Tier 1 / Tier 2 / Tier 3 tools based on the actual task shape implied by the query
- geometry bridge normal generation is now batched/vectorized and uses a real pool-derived `math_core_plan` to shape batch size

### 1.3 Phase 2A Shared Foundation: Execution Events

Status: implemented.

New runtime surface:

- `knowledge3d/knowledgeverse/execution_events.py`
  - `ExecutionEvent`
  - `ExecutionEventRecorder`
  - quality extraction from real bridge metadata
  - ternary quality quantization
  - event attachment back into result metadata

- `knowledge3d/knowledgeverse/tool_execution.py`
  - observed execution wrappers now record:
    - `tool_id`
    - `query_context`
    - `specialist_id`
    - `math_core_tier`
    - `execution_us`
    - `outcome`
    - `quality_signal`
    - `ternary_quality`
    - `timestamp_us`
    - `chain_depth`
    - `promotion_pressure`

- `knowledge3d/knowledgeverse/trm_navigator.py`
  - execution routes now use observed Tool invocation
  - promotion targets are carried into execution plans
  - navigator consumes the recorded events via `observe_execution_event(...)`

Storage:

- `storage_root/logs/execution_events.jsonl`
- shadow-copy audit emission via `knowledgeverse.shadow_copy.record_event(...)`

### 1.4 Phase 2B First Consumer: Scene Quality Feedback

Status: implemented, first pass.

`knowledge3d/cranium/bridges/procedural_temporal_bridge.py` now:

- applies ternary quality weighting to scene layers
- demotes low-quality layers via opacity reduction and background ordering
- promotes high-quality layers via foreground ordering
- stores:
  - `layer_quality_signals`
  - `layer_ternary_quality`
  - `quality_weighting = ternary_contrastive`
  - `house_room_preset`

Replay scene playback now:

- consumes event `quality_signal` / `ternary_quality` directly when present
- auto-detects repeating scene grammars from replay streams
- maps resulting scenes to:
  - `house_library`
  - `house_garden`
  - `house_museum`

### 1.5 Phase 2C First Consumer: Specialist Learning

Status: implemented, first pass.

New runtime surface:

- `knowledge3d/knowledgeverse/execution_quality_tracker.py`

Capabilities:

- per-tool quality tracking:
  - total executions
  - success / failure / uncertain counts
  - avg execution time
  - bayesian quality
  - ternary trend

- per-specialist online learning:
  - lightweight task embedding
  - centroid update toward success
  - centroid update away from failure
  - exploration count for uncertain outcomes

- specialist gap logging:
  - `storage_root/logs/specialist_gaps.jsonl`
  - threshold-based (`0.3`) gap detection

Navigator ranking now includes live tool quality bonus on top of query relevance.
- lathe / extrude / sweep mesh assembly now uses vectorized topology construction instead of the earlier per-vertex/per-triangle loop nests
- material selection now carries a real Tier-2 selection plan and processes candidate batches using that plan
- planar projection no longer computes all three triplanar sample planes unconditionally; cheap planar cases now follow a Tier-1 projection plan
- material projection metadata now records its execution `math_core_plan`, and contour->textured mesh carries both selection and projection plans forward
- Tool execution now validates payloads against Tool-side entrypoint schemas before invocation
- Tool contracts now support semantic aliases and defaults, so semantic payloads can bind without bridge-specific field names
- execution plans are now resolvable and directly invokable from the navigator, not only inspectable
- the `Tool` galaxy now carries explicit multi-step chain presets for audio->surface and contour->material routes
- chain selection now supports selector fields (for example `geometry_mode=extrude` or `geometry_mode=sweep`) so geometry variants stay encoded in Tool metadata instead of ad hoc Python branching
- `ProceduralMaterialBridge` now exposes direct textured `lathe`, `extrude`, and `sweep` routes as real bridge entrypoints
- `tool_fusion_signal_surface_material_v1` now has explicit executable chain presets, including an auto-target variant that derives a procedural target material from the spectrogram itself
- `ProceduralSignalBridge` now exposes a configured audio->spectrogram entrypoint so chain execution can carry `frame_size` and `threshold` truthfully
- chain enrichment now preserves signal metadata such as `clip_id`, `signal_frame_size`, and `signal_threshold`
- `ProceduralTemporalBridge` now exists as a real bridge over the current signal/material/geometry stack:
  - procedural frame generation
  - PTX temporal deltas
  - PTX temporal coherence
  - optional sovereign ternary video frame encoding
- the Tool galaxy now includes temporal/video surfaces:
  - `tool_video_temporal_preview_v1`
  - `tool_fusion_surface_material_timeline_v1`
  - `tool_fusion_signal_surface_material_timeline_v1`
- contour/material and signal/material routes can now continue into temporal preview chains without leaving the single Tool-execution contract
- executable-tool ranking is now query-aware for temporal/timeline/video language, so generic fusion queries are not hijacked by timeline tools and timeline queries still resolve to the temporal routes
- temporal preview results now carry coherent metadata:
  - frame count
  - frame shape
  - feature grid
  - overall coherence
  - coherence variance
  - encoded frame ids when codec storage is enabled
- temporal/video routing now supports named preset composition without leaving the single Tool contract:
  - `ui_idle`
  - `ui_focus`
  - `world_breathe`
  - `world_orbit`
- `ProceduralTemporalBridge.surface_material_to_timeline_preset(...)` now exists as a real executable route with clean fallback to generic preview when no preset is requested
- the temporal Tool metadata now exposes reusable preset families for UI/world animation instead of only a generic preview surface
- contour/material and signal/material timeline chains can now carry `timeline_preset` semantically through Tool-side schemas and defaults
- canonical UI/world animation routes now exist as first-class Tool surfaces:
  - `tool_fusion_surface_material_ui_animation_v1`
  - `tool_fusion_surface_material_world_animation_v1`
  - `tool_fusion_signal_surface_material_ui_animation_v1`
  - `tool_fusion_signal_surface_material_world_animation_v1`
- navigator ranking now treats UI/HUD and world/ambient animation as explicit semantic intents instead of forcing callers to name raw presets
- the first promotion-pressure aggregation pipeline now exists:
  - `scripts/build_tool_promotion_report.py`
  - default report output: `../Knowledge3D.local/results/tool_promotion_report.json`
- scene-level temporal composition is now started on top of the completed timeline surface:
  - `TemporalSceneLayer`
  - `TemporalScenePlan`
  - `ProceduralTemporalBridge.compose_scene_timeline(...)`
  - `ProceduralTemporalBridge.surface_materials_to_scene_timeline(...)`
- canonical scene Tool routes now exist:
  - `tool_video_temporal_scene_v1`
  - `tool_fusion_surface_material_ui_scene_v1`
  - `tool_fusion_surface_material_world_scene_v1`
- replay/House playback routes now exist on the same surface:
  - `ProceduralTemporalBridge.replay_journal_to_scene_timeline(...)`
  - `ProceduralTemporalBridge.action_buffers_to_scene_timeline(...)`
  - `tool_house_replay_scene_v1`
- world scene playback now has a golden-ratio layout mode:
  - `golden_orbit`
- navigator ranking now distinguishes:
  - animation queries
  - scene/layer/playback queries
  so the new scene tools do not hijack generic animation routes

## Commands

Build payload:

```bash
PYTHONPATH=. python3 scripts/build_multimodal_tool_nodes.py
```

Ingest payload:

```bash
PYTHONPATH=. python3 scripts/fundamental_ingest_payloads.py \
  --payload ../Knowledge3D.local/fundamental_augmentation/tool_nodes_phase0.jsonl \
  --storage-root ../Knowledge3D.local \
  --report ../Knowledge3D.local/results/tool_nodes_phase0_ingestion_report.json
```

Bootstrap directly to galaxy storage:

```bash
PYTHONPATH=. python3 - <<'PY'
from knowledge3d.knowledgeverse.tool_galaxy import bootstrap_tool_galaxy
print(bootstrap_tool_galaxy("../Knowledge3D.local"))
PY
```

## Success Criteria

1. Tool-nodes are queryable by specialist and by galaxy name.
2. Tool-node payload survives ingestion with symlink metadata added.
3. Codec tools expose only real PTX-backed runtime surfaces.
4. No new hot-path kernel dependency was added for multimodal fusion recipes.
5. Opcode requests are treated as promotion targets, not runtime facts.
6. Sovereign image/audio/video codecs run on a shared tested PTX block/frequency substrate.
7. Material projection uses ternary positive/negative pressure to choose compact procedural palettes before projecting them onto geometry.

## Next Implementation Step

1. Add more tool-node families:
   - contour cleaning
   - profile sweep
   - signal windowing/filter chains
2. Start filling and using `tool_promotion_pressure.jsonl` in real runs, then review the ranked opcode promotion report.
3. Add specialist-side composition heuristics that choose between recipe tools and kernel-backed codec tools by task shape.
4. Promote the next geometry/material kernels when repetition justifies them:
   - contour cleanup / smoothing
   - profile sweep beside the now-real lathe + extrude bridges
   - UV or triplanar projection kernels for large batch surfaces
5. Extend the same selector-aware chain model into more fused routes:
   - contour -> sweep/extrude -> material -> temporal layering -> UI/world preset families
   - audio -> spectrogram -> surface -> material -> animation -> UI/world preset families
6. Start moving the repeated signal-surface assembly work toward PTX when telemetry shows the bridge orchestration is hot enough to justify promotion.
7. Extend the now-started scene composition layer into richer procedural video playback surfaces for:
   - House/world timeline playback
   - reusable UI/HUD scene grammars
   - replay journal reconstruction and action-driven playback
   - multi-layer scene orchestration beyond single-surface animation

## Phase 2D Closure

Completed on top of the existing multimodal foundation:

1. `execution_events` now carry chain metadata (`chain_tool_ids`, `chain_runtime_statuses`)
2. observed chain execution records both step-level and top-level events
3. recurring successful execution chains now create Grammar Galaxy entries automatically
4. House room playback routes now exist as Tool nodes:
   - `tool_house_library_scene_v1`
   - `tool_house_garden_scene_v1`
   - `tool_house_museum_scene_v1`
   - `tool_house_tour_scene_v1`
5. navigator query routing now distinguishes:
   - library/settled knowledge
   - garden/learning/exploration
   - museum/history/failures
   - tour/overview/all
6. focused Phase 2D validation slice is green (`52 passed`)

This moves the project out of "scene playback experiments" and into:
- self-observed grammar creation
- executable House room semantics
- dense journals suitable for the next promotion harvest track

## Phase 2E Harvest Foundation

Completed:

1. `scripts/build_tool_promotion_report.py` now merges:
   - `tool_promotion_pressure.jsonl`
   - `execution_events.jsonl`
   - `execution_grammar_patterns.jsonl`
2. report output now includes:
   - `event_tools`
   - `event_chains`
   - `promotion_candidates`
   - `candidate_summary.top_candidate`
3. candidate ranking now considers:
   - pressure frequency
   - observed latency
   - quality gap
   - grammar-backed recurrence
   - route diversity
4. focused harvest validation is green:
   - `tests/test_tool_promotion_report.py`
   - `tests/test_execution_events.py`
   - `tests/test_execution_grammar_detection.py`

This closes the "raw telemetry only" gap and gives the next PTX promotion pass a ranked evidence base.

## Track 3 Routing Gate Upgrade

Completed:

1. `execution_quality_tracker.py` now tracks:
   - per-tool quality
   - per-tool-source quality
   - per-route-source quality
2. route sources are normalized into:
   - `recipe`
   - `bridge`
   - `kernel`
3. ternary routing gate added:
   - `-1` prefer recipe
   - `0` prefer bridge
   - `+1` prefer kernel
4. navigator executable-tool prioritization now uses:
   - tool quality bonus
   - source quality bonus
   - routing alignment bonus
5. focused validation is green:
   - `tests/test_specialist_selection.py`
   - `tests/test_tool_execution.py`
   - `tests/test_execution_events.py`
   - result: `39 passed`

This closes the gap between "quality is recorded" and "quality actually changes route choice."

## Track 2 / Track 3 Connection

Completed:

1. `build_tool_promotion_report.py` now consumes `execution_quality_tracker.json`
2. promotion report now includes:
   - `route_source_quality`
   - `tool_source_quality`
   - route-aware promotion candidate fields
3. candidate ranking now reflects:
   - source quality level
   - source quality gap
   - dominant route source
   - route source counts
4. focused route-aware harvest validation is green:
   - `tests/test_tool_promotion_report.py`
   - `tests/test_specialist_selection.py`
   - result: `5 passed`

This closes the loop between:
- route learning
- route-aware selection
- route-aware PTX promotion harvesting

## Track 4 Contrastive Grammar Upgrade

Completed:

1. `execution_grammar_detector.py` now promotes:
   - positive recurring chains
   - negative recurring chains
2. negative recurring chains become explicit Grammar Galaxy anti-pattern entries
3. new negative grammar metadata includes:
   - `pattern = tool_chain_negative`
   - `source = auto_detected_contrastive`
   - `pattern_type = execution_tool_chain_antipattern`
   - `contrastive_recommendation = avoid_or_invert`
4. focused validation is green:
   - `tests/test_execution_grammar_detection.py`
   - `tests/test_specialist_selection.py`
   - result: `6 passed`

This is the first time the system records repeated failures as inspectable grammar knowledge instead of only scalar penalties.

## Track 3 Route-Aware Auto-Dispatch

Completed:

1. `tool_execution.py` now uses route/source quality during payload-time entrypoint and chain selection
2. `trm_navigator.py` passes execution quality state into Tool dispatch, not only ranking
3. route selection now combines:
   - schema fit
   - plan affinity
   - tool/source quality
   - ternary routing alignment
4. explicit regression guard added so semantic UI/world animation routes are preserved when lower-level chains also bind the payload
5. focused validation is green:
   - `tests/test_tool_execution.py`
   - `tests/test_specialist_selection.py`
   - result: `39 passed`

This closes the last major gap between:
- route-aware ranking
- route-aware execution
- route-aware promotion harvesting

## Track 5 Sleep-Time PTX Foundation

Completed:

1. materialized the three designed sleep-time CUDA kernels into real PTX-backed runtime surfaces:
   - `assign_to_best_centroid`
   - `accumulate_centroid_sums`
   - `refine_embeddings_to_centroids`
   - `compute_silhouette_scores`
   - `cluster_glyphs_by_similarity`
2. integrated those kernels into:
   - `sleep_time_consolidator.py`
   - `sleep/glyph_consolidator.py`
3. added first real VRAM-pressure trigger:
   - `memory_pressure_trigger.py`
   - exposed through `SleepTimeConsolidator`
4. made sleep consolidation idempotent once already consolidated
5. focused validation is green:
   - `tests/test_sleep_cluster_kernels.py`
   - `tests/test_glyph_consolidator.py`
   - `tests/test_sleep_time_consolidator.py`
   - `tests/test_memory_pressure_trigger.py`
   - result: `10 passed`

Additional runtime check:

1. ad hoc kernel benchmark on `512 x 64` embeddings, `32` clusters:
   - assignment average: `0.246 ms`
   - centroid accumulation average: `0.675 ms`

This closes the gap between:
- designed sleep-time GPU kernels
- compiled PTX artifacts
- live consolidator integration
- memory-pressure-based trigger policy

The next promotion target inside Track 5 is now:
- redundancy pruning around the now-PTX cluster core

Track 5 similarity promotion completed:

1. `cosine_similarity.cu` now exports `cosine_similarity_matrix`
2. `CosineSimilarityBridge` now exposes `compute_similarity_matrix(...)`
3. `clustering_rpn.py` now routes:
   - single cosine
   - rectangular similarity matrix
   - pairwise similarities
   - nearest neighbors
   through the same PTX matrix surface

Validation:

1. managed-env focused slice:
   - `tests/test_clustering_rpn.py`
   - `tests/test_sleep_cluster_kernels.py`
   - `tests/test_sleep_time_consolidator.py`
   - result: `17 passed in 2.41s`

Ad hoc runtime check:

1. similarity matrix average on `512 x 32` at `64` dims:
   - `0.373 ms`

Track 5 outlier stage completed:

1. `sleep_time_consolidator.py` now preserves filtered cluster assignments after redundancy pruning
2. `_remove_outliers()` is now real:
   - PTX centroid accumulation
   - PTX similarity-to-centroid scoring
   - ternary keep/uncertain/remove decisions
3. cluster floor guard prevents over-pruning

Validation:

1. focused sleep slice:
   - `tests/test_sleep_time_consolidator.py`
   - `tests/test_sleep_cluster_kernels.py`
   - result: `7 passed in 2.07s`
2. full clustering/sleep slice:
   - `tests/test_clustering_rpn.py`
   - `tests/test_sleep_cluster_kernels.py`
   - `tests/test_glyph_consolidator.py`
   - `tests/test_sleep_time_consolidator.py`
   - `tests/test_memory_pressure_trigger.py`
   - result: `22 passed in 2.22s`

Track 5 redundancy pruning promotion completed:

1. Stage 2 no longer performs serial one-by-one resonance updates for redundant members
2. redundant member selection now uses ternary keep/uncertain/remove logic
3. representative retention now uses centroid-based merge through the PTX centroid path
4. Stage 2 metrics now include:
   - `status`
   - `merged_pairs`
   - `uncertain_candidates`
   - `clusters_examined`
   - `reduction`

Validation:

1. full clustering/sleep slice:
   - `tests/test_clustering_rpn.py`
   - `tests/test_sleep_cluster_kernels.py`
   - `tests/test_glyph_consolidator.py`
   - `tests/test_sleep_time_consolidator.py`
   - `tests/test_memory_pressure_trigger.py`
   - result: `23 passed in 2.65s`

Ad hoc runtime check:

1. redundancy pruning average on `64` near-duplicate vectors at `128` dims:
   - `2.619 ms`
   - merged pairs: `63`
   - survivors: `1`

The next promotion target inside Track 5 is now narrow:
- broader sleep-time bookkeeping only if profiling says it is worth the complexity

Track 5 centroid finalization promotion completed:

1. `sleep_cluster_refiner.cu` now exports `finalize_centroids`
2. `sleep_cluster_kernels.py` now returns finalized centroids directly from device memory
3. host-side centroid normalization has been removed from the accumulation path

Validation:

1. full clustering/sleep slice:
   - `tests/test_clustering_rpn.py`
   - `tests/test_sleep_cluster_kernels.py`
   - `tests/test_glyph_consolidator.py`
   - `tests/test_sleep_time_consolidator.py`
   - `tests/test_memory_pressure_trigger.py`
   - result: `23 passed in 2.60s`

Ad hoc runtime check:

1. centroid accumulation + finalization average on `512 x 64` embeddings, `32` clusters:
   - `0.595 ms`
   - sample centroid norm: `1.0`

Track 4 multimodal grammar evolution completed:

1. `execution_grammar_detector.py` now promotes generalized multimodal execution grammars in addition to exact chain rules.
2. Promotion features now include:
   - tool-family token signatures
   - modality signatures
   - route-source signatures
   - recurring query-token signatures
3. Both positive and contrastive forms are now real Grammar Galaxy entries:
   - `execution_multimodal_pattern`
   - `execution_multimodal_antipattern`

Validation:

1. focused slice:
   - `tests/test_execution_grammar_detection.py`
   - `tests/test_specialist_selection.py`
   - `tests/test_tool_promotion_report.py`
   - result: `9 passed in 4.99s`

Execution selection hardening completed:

1. `tool_execution.py` now weights schema/payload fit more strongly than incidental route history.
2. Primary Tool contracts are preserved when they fit the payload better than looser alternates.
3. Route-aware fallback is still allowed when schema fit is equal, so healthy bridge routes can still replace degraded kernel routes.

Validation:

1. broader execution/journal slice:
   - `tests/test_tool_execution.py`
   - `tests/test_execution_events.py`
   - `tests/test_execution_grammar_detection.py`
   - `tests/test_scene_quality.py`
   - `tests/test_specialist_selection.py`
   - `tests/test_tool_promotion_report.py`
   - result: `49 passed in 19.71s`

Current foundational status:

1. Track 5 sleep-time PTX core: effectively closed.
2. Track 4 grammar evolution: materially in place.
3. The next rational foundational targets are:
   - richer grammar evolution beyond recurrence/signature aggregation
   - using the now stronger promotion harvest on real journals

Promotion harvesting hardening completed:

1. `build_tool_promotion_report.py` now integrates:
   - pressure rows
   - execution events
   - exact chain grammar support
   - multimodal positive grammar support
   - multimodal contrastive grammar pressure
   - route-source and tool-source quality state
2. candidate ranking now distinguishes:
   - recurring healthy bridge demand
   - recurring failing multimodal anti-pattern pressure
3. candidate rows now expose:
   - `multimodal_positive_support`
   - `multimodal_negative_support`
   - `multimodal_positive_patterns`
   - `multimodal_negative_patterns`

Validation:

1. focused harvest slice:
   - `tests/test_tool_promotion_report.py`
   - `tests/test_execution_grammar_detection.py`
   - `tests/test_specialist_selection.py`
   - result: `9 passed in 6.71s`
2. broader execution/journal slice:
   - `tests/test_tool_execution.py`
   - `tests/test_execution_events.py`
   - `tests/test_execution_grammar_detection.py`
   - `tests/test_scene_quality.py`
   - `tests/test_specialist_selection.py`
   - `tests/test_tool_promotion_report.py`
   - result: `49 passed in 18.48s`

CLI sanity:

1. `scripts/build_tool_promotion_report.py` runs cleanly in the managed env.
2. Current live output may still be empty when the default storage root has no populated journal files.

Real journal harvest completed:

1. Generated a persistent execution batch at:
   - `../Knowledge3D.local/runtime_execution_journal_batch`
2. Real counts:
   - `180` execution events
   - `62` grammar rows
   - `48` pressure rows
3. First non-empty real promotion report generated:
   - `../Knowledge3D.local/runtime_execution_journal_batch/results/tool_promotion_report.json`
4. First real top candidate:
   - `SIGNAL_SURFACE_MATERIAL`
   - priority `0.77268`
   - readiness `0.874011`

PTX promotion from real harvest completed:

1. Promoted `spectrogram -> surface` inside the top-ranked `SIGNAL_SURFACE_MATERIAL` route.
2. Added:
   - `knowledge3d/cranium/kernels/signal_surface_ops.cu`
   - `knowledge3d/cranium/ptx_runtime/signal_surface_kernels.py`
3. `procedural_signal_bridge.py` now uses PTX for:
   - heightfield vertex generation
   - heightfield normal generation
4. Focused validation:
   - `tests/test_signal_surface_kernels.py`
   - `tests/test_procedural_signal_bridge.py`
   - `tests/test_procedural_material_bridge.py`
   - result: `12 passed in 2.75s`

Measured gain:

1. previous `spectrogram_to_surface` baseline:
   - `1.326 ms`
2. post-promotion hot average:
   - `0.1865 ms`
3. improvement:
   - about `7.1x`

Current next justified target from real evidence:

1. The route family is still `SIGNAL_SURFACE_MATERIAL`.
2. The promoted surface is no longer the bottleneck.
3. The remaining heavier pressure is downstream in:
   - signal/material orchestration
   - temporal preview/timeline composition

Temporal PTX promotion completed:

1. Added:
   - `knowledge3d/cranium/kernels/temporal_preset_ops.cu`
   - `knowledge3d/cranium/ptx_runtime/temporal_preset_kernels.py`
   - `knowledge3d/cranium/kernels/temporal_frame_ops.cu`
   - `knowledge3d/cranium/ptx_runtime/temporal_frame_kernels.py`
2. `procedural_temporal_bridge.py` now uses PTX for:
   - timeline preset application
   - procedural frame synthesis
3. Focused validation:
   - `tests/test_temporal_frame_kernels.py`
   - `tests/test_temporal_preset_kernels.py`
   - `tests/test_procedural_temporal_bridge.py`
   - `tests/test_scene_quality.py`
   - result: `13 passed in 3.65s`

Measured route gain:

1. `ui_idle`
   - `80.227 ms` -> `75.899 ms`
2. `world_breathe`
   - `129.709 ms` -> `120.361 ms`

Refreshed harvest after temporal promotion:

1. `8` iterations
2. `120` execution events
3. `62` grammar rows
4. `36` pressure rows
5. top candidate remains:
   - `SIGNAL_SURFACE_MATERIAL`
   - priority `0.77215`
   - readiness `0.87454`

Current next justified target from refreshed evidence:

1. stay on the `SIGNAL_SURFACE_MATERIAL` family
2. do not revisit solved sub-surfaces
3. the remaining heavier pressure is in:
   - signal/material orchestration
   - timeline/scene orchestration
   - encode-time bookkeeping


## Latest Promotion Closure: Encode-Time Bookkeeping Removed

Completed:

- `SovereignTernaryAudioCodec` no longer routes encode/decode through generic `ModularRPNEngine.evaluate(...)`.
- `SovereignTernaryVideoCodec` no longer routes encode/decode through generic RPN flatten/reshape.
- `ProceduralMaterialBridge` now caches configured `ProceduralSignalBridge` instances.
- `ProceduralTemporalBridge` now caches configured `SovereignTernaryVideoCodec` instances.
- temporal frame encoding now uses `encode_frame_array(...)` instead of `frame -> TernaryTensor -> Python list -> reshape -> flatten -> TernaryVector`.

Measured effect:

- `signal_to_textured_surface`: `16.087 ms -> 7.700 ms`
- `ui_idle`: `75.899 ms -> 22.947 ms`
- `world_breathe`: `120.361 ms -> 37.488 ms`

Interpretation:

- The dominant remaining cost after earlier PTX promotions was encode-time bookkeeping.
- That bookkeeping is now materially reduced.
- The route family is still the same, but another real bottleneck layer has been removed.


## Scene Playback Alignment

Completed:

- `compose_scene_timeline(...)` now uses the same fast array-based video encode path as temporal preview.
- House/world playback no longer lags behind preview/timeline routes because of older `TernaryTensor` encode bookkeeping.

Measured effect:

- `surface_materials_to_scene_timeline`: `211.399 ms -> 58.189 ms`

Interpretation:

- scene playback is now materially closer to the promoted temporal preview path
- the remaining pressure is increasingly in higher-level orchestration and route selection, not the previously repeated encode bookkeeping


## Shared Ternary Substrate Tightening

Completed:

- `TernaryVector` pack/unpack is now vectorized.
- sovereign audio/video codecs now consume ternary data through `to_numpy()` instead of Python-list roundtrips.

Measured effect:

- `signal_to_textured_surface`: `7.700 ms -> 6.220 ms`
- `world_breathe`: `37.488 ms -> 30.504 ms`

Interpretation:

- This is a true small-detail improvement with system-wide effect.
- The shared ternary substrate is now cheaper across the promoted multimodal routes.

---

## Additional Completed Work: Codec Boundary + Material Signature Promotion

Completed:
- numpy/PTX codec-op boundary for audio/video/block transforms
- immutable-host ternary unpack path for `TernaryVector.to_numpy()`
- cached ternary material signature encoding
- cached material stops / previews / normal hints
- refreshed real promotion harvest after the new route timings

Measured state now:
- `signal_to_textured_surface`: `1.702 ms`
- `ui_idle`: `6.991 ms`
- `world_breathe`: `10.671 ms`
- `scene_timeline`: `18.221 ms` (same band; current bottleneck has moved upward)

Implication:
- the remaining foundational work is no longer the codec boundary for this route family
- the next justified promotions should target scene/layer orchestration or remaining bridge-level composition overhead, not already-solved PTX math surfaces

---

## Additional Completed Work: Scene Layer Encoding Promotion

Completed:
- removed duplicate layer-preview encoding in `surface_materials_to_scene_timeline(...)`
- preserved final scene encoding on the composed route

Measured state now:
- `scene_timeline`: `9.514 ms`
- overall from paused-runtime baseline: `211.399 ms -> 9.514 ms` (`22.22x`)

Implication:
- scene/layer orchestration is materially tighter
- the next justified foundational work is above this route again: surviving scene composition / route composition overhead, not solved codec math

## Added Completion Slice: Dispatch/Journaling Hardening

Completed:
- cache resolved Tool execution plans and entrypoints in `knowledge3d/knowledgeverse/tool_execution.py`
- remove duplicate select/invoke passes in observed payload execution
- debounce `ExecutionQualityTracker` state writes
- debounce `ExecutionGrammarDetector` state writes
- lazy-init `knowledge3d/gpu/rng_pool.py` to eliminate fresh-process temporal import failure
- refresh real execution journal batch and promotion report

Measured result:
- hot observed signal dispatch profiled at ~`60 ms`
- real journal batch top candidate now `TRIPLANAR_MAP`

Remaining route overhead is now primarily:
- execution event append/open cost
- specialist relevance/tokenization cost in learning/journaling

## Added Completion Slice: Event Recorder Batching + Text Embedding Cache

Completed:
- buffer `execution_events.jsonl` appends in `knowledge3d/knowledgeverse/execution_events.py`
- flush buffered step events on top-level route completion or explicit flush
- share one `ExecutionEventRecorder` per storage root in `knowledge3d/knowledgeverse/tool_execution.py`
- cache `_tokenize()` and `_embed_text()` in `knowledge3d/knowledgeverse/execution_quality_tracker.py`
- add regression tests for recorder flush semantics and debounced quality persistence

Measured state:
- signal dispatch hot benchmark now stable in the `~162 ms` band
- world dispatch remains heavier in the `~225 ms` band

Implication:
- remaining pressure has shifted further upward into:
  - shadow-copy / top-level journaling overhead
  - world/scene orchestration
- event append/open cost is no longer the same per-step problem it was before batching

## Added Completion Slice: Chain-Step Audit Reduction

Completed:
- treat `tool_chain_step` events as lightweight observations in `knowledge3d/knowledgeverse/trm_navigator.py`
- keep step events in `execution_events.jsonl` but stop sending them individually into Shadow Copy in `knowledge3d/knowledgeverse/execution_events.py`
- preserve tool/source quality updates for chain steps while skipping:
  - specialist feedback learning
  - gap detection
  - grammar promotion
- add regression coverage in `tests/test_execution_event_recorder.py` and `tests/test_specialist_selection.py`

Measured state:
- dispatch benchmark artifact: `../Knowledge3D.local/results/dispatch_route_promotion_benchmark_20260306_1858.json`
- dispatch benchmark artifact: `../Knowledge3D.local/results/dispatch_route_promotion_benchmark_20260306_1900.json`
- signal dispatch hot: `~162 ms -> 87.852 ms -> 83.251 ms`
- world dispatch hot: `~225 ms -> 86.114 ms -> 93.577 ms`
- hot observed signal dispatch profile now sits around `~13 ms` cumulative for the route itself

Implication:
- dispatch/observation overhead is now materially tighter
- remaining foundational work has shifted above step-level journaling
- next justified plan work is:
  - route composition / payload-binding overhead
  - world/scene orchestration
  - promotion targets emerging from the live `TRIPLANAR_MAP` family pressure

## Added Completion Slice: TRIPLANAR_MAP PTX Promotion

Completed:
- add fused `project_triplanar_rgba_kernel` in `knowledge3d/cranium/kernels/material_projection.cu`
- expose `project_triplanar(...)` in `knowledge3d/cranium/ptx_runtime/material_projection_kernels.py`
- replace the old triplanar bridge path in `knowledge3d/cranium/bridges/procedural_material_bridge.py`
  - old path:
    - three planar sample launches
    - one blend launch
  - new path:
    - one fused PTX launch
- add direct equivalence coverage in `tests/test_material_projection_kernels.py`

Measured state:
- benchmark artifact: `../Knowledge3D.local/results/triplanar_promotion_benchmark_20260306_190724.json`
- `project_material_triplanar`: `0.235 ms`
- `signal_to_textured_surface`: `1.262 ms`
- refreshed promotion report now shows the `TRIPLANAR_MAP` family average around:
  - `3687.403 us`

Implication:
- the promoted triplanar surface itself is no longer the real bottleneck
- the remaining foundational work has moved upward into:
  - material/temporal orchestration
  - world/scene composition
  - route-level composition around the promoted surface

## Added Completion Slice: Temporal Preview/World Cache Promotion

Completed:
- add deterministic preview-plan caching in `knowledge3d/cranium/bridges/procedural_temporal_bridge.py`
- cache key is based on:
  - derived surface seed
  - preset parameters
  - output size
  - encode/codec settings
- explicitly bypass cache when `timeline_id` is provided so encoded frame IDs remain unique and correct
- add regression coverage for:
  - cache reuse without `timeline_id`
  - cache bypass with `timeline_id`

Measured state:
- benchmark artifact: `../Knowledge3D.local/results/temporal_cache_benchmark_20260306_191134.json`
- `ui_idle`: `0.219 ms`
- `world_breathe`: `0.223 ms`
- `scene_timeline`: `5.628 ms`
- dispatch artifact: `../Knowledge3D.local/results/dispatch_route_promotion_benchmark_20260306_1911.json`
- signal dispatch hot: `73.122 ms`
- world dispatch hot: `67.140 ms`

Implication:
- temporal/world preview generation is now in the correct regime
- remaining foundational pressure is above the preview layer:
  - final route composition
  - top-level journaling/observation
  - broader world/scene orchestration

## Added Completion Slice: Buffered Execution Journal Promotion

Completed:
- stop flushing `execution_events.jsonl` on every top-level route in `knowledge3d/knowledgeverse/execution_events.py`
- keep the journal:
  - buffer-size driven
  - interval driven
  - exit-safe via `atexit`
- preserve immediate top-level audit through Shadow Copy
- update execution-event tests to verify the new eventual JSONL contract and explicit flush boundary

Measured state:
- dispatch artifact: `../Knowledge3D.local/results/dispatch_route_promotion_benchmark_20260306_1943.json`
- signal dispatch hot: `66.827 ms`
- world dispatch hot: `81.115 ms`
- refreshed journal batch still yields:
  - `event_rows=106`
  - `grammar_rows=60`
  - `pressure_rows=22`

Implication:
- the JSONL journal is no longer taxing every invocation
- correctness is preserved through:
  - Shadow Copy for immediate audit
  - buffered execution journal for harvest/reporting
- remaining plan work is now concentrated in the last upper-layer orchestration band

## Added Completion Slice: Debounced Specialist Weight Persistence Promotion

Completed:
- debounce automatic `save_weights()` inside `knowledge3d/knowledgeverse/trm_navigator.py`
- preserve explicit `save_weights()` as an immediate persistence boundary
- add process-exit safety via `atexit`
- add regression coverage in `tests/test_trm_matryoshka_specialists.py`

Measured state:
- dispatch artifact: `../Knowledge3D.local/results/dispatch_route_promotion_benchmark_20260306_1949.json`
- signal dispatch hot: `4.713 ms`
- world dispatch hot: `4.052 ms`

Implication:
- the last major upper-layer persistence spike is removed from the live route
- the foundational free-GPU pass is effectively complete
- augmentation can now be resumed on the preserved base state while follow-on work proceeds in parallel

## Added Completion Slice: House Room / Tour Scene Cache Promotion

Completed:
- add deterministic cache for:
  - `execution_events_to_house_room_scene(...)`
  - `execution_events_to_house_tour_scene(...)`
- key over compact ordered execution-event digest plus route parameters
- bypass cache when explicit `scene_id` is provided
- add regression coverage in `tests/test_procedural_temporal_bridge.py`

Measured state:
- benchmark artifact: `../Knowledge3D.local/results/house_scene_cache_benchmark_20260306_2015.json`
- house room scene:
  - cold `19.193 ms`
  - hot average `0.040 ms`
- house tour scene:
  - cold `157.293 ms`
  - hot average `0.026 ms`

Implication:
- repeated House/world playback rebuild is effectively removed from the hot path
- the remaining work is now concentrated in live route mutation and higher-level orchestration, not deterministic scene reconstruction
