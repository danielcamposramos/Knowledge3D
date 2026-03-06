# Scene Encode Promotion Benchmark

**Date**: March 6, 2026
**Environment**: `/K3D/Knowledge3D.local/envs/k3d-cranium`
**Scope**: route scene-level encoding through the same fast video array path already used by temporal preview.

## Change

`ProceduralTemporalBridge.compose_scene_timeline(...)` now uses cached `SovereignTernaryVideoCodec` instances and `encode_frame_array(...)` for scene frame encoding.

This removes the old scene path:

- `frame[..., :3]`
- `TernaryTensor`
- `TernaryVector.to_python()`
- generic encode bookkeeping

## Measured Result

- Baseline `surface_materials_to_scene_timeline`: `211.399 ms`
- New hot average: `58.189 ms`
- Improvement: `3.63x`

## Interpretation

The scene route was still paying the old temporal encode cost even after the preview route was fixed.
Switching the scene encoder to the fast array path removed that mismatch and materially reduced House/world playback overhead.

## Artifact

- JSON benchmark: `../Knowledge3D.local/results/scene_encode_promotion_benchmark_20260306_1722.json`
