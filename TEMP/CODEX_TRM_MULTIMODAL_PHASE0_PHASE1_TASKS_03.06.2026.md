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
   - route-aware promotion harvesting from the now stronger execution journals
