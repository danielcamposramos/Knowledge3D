# Scene Layer Encoding Promotion Benchmark

**Date**: March 6, 2026
**Environment**: `/K3D/Knowledge3D.local/envs/k3d-cranium`
**Scope**: remove duplicate layer-frame encoding inside scene composition so `surface_materials_to_scene_timeline(...)` only encodes final scene frames.

## Change

- `ProceduralTemporalBridge.surface_materials_to_scene_timeline(...)` now builds layer previews with `encode_frames=False`.
- Final scene encoding remains enabled at `compose_scene_timeline(...)` when requested.
- This preserves top-level semantics while eliminating redundant per-layer codec work.

## Measured Results

Compared to the previous material-signature baseline (`material_signature_promotion_benchmark_20260306_1800.json`):

- `surface_materials_to_scene_timeline`
  - previous: `18.221 ms`
  - new: `9.514 ms`
  - improvement: `1.92x`

Reference values from the same run:
- `signal_to_textured_surface`: `1.877 ms`
- `ui_idle`: `7.225 ms`
- `world_breathe`: `11.177 ms`

## Overall Improvement From Original Runtime Baseline

Compared to the paused-runtime baseline (`multimodal_runtime_benchmark_20260306_142408.json`):

- `surface_materials_to_scene_timeline`
  - original: `211.399 ms`
  - current: `9.514 ms`
  - overall improvement: `22.22x`

## Interpretation

The scene profile showed the waste clearly: the system was encoding every layer preview and then encoding the composed scene again. That was not new capability, only duplicate bookkeeping.

Removing layer-level encoding from this path cut the scene route nearly in half again without touching already-solved PTX math surfaces.

This is the correct kind of promotion now: not more kernel proliferation, but orchestration cleanup above the kernel layer.

## Artifacts

- JSON benchmark: `../Knowledge3D.local/results/scene_layer_encoding_promotion_benchmark_20260306_1810.json`
- Updated real report: `../Knowledge3D.local/runtime_execution_journal_batch/results/tool_promotion_report.json`
