# Material Signature Promotion Benchmark

**Date**: March 6, 2026
**Environment**: `/K3D/Knowledge3D.local/envs/k3d-cranium`
**Scope**: cache ternary gradient signatures and move material-selection signature encoding onto the numpy/PTX quantization path.

## Changes

- `TernaryGradientLogic.encode_signature(...)` now:
  - caches signatures by normalized stop tuples + thresholds
  - uses `quantize_numpy(...)` instead of list-based quantization
- `ProceduralMaterialBridge` already caches:
  - material stops
  - material previews
  - normal hints

## Measured Results

Compared to the post-codec-boundary baseline (`codec_boundary_promotion_benchmark_20260306_1749.json`):

- `signal_to_textured_surface`
  - previous: `4.827 ms`
  - new: `1.702 ms`
  - improvement: `2.84x`

- `surface_material_to_timeline_preset.ui_idle`
  - previous: `7.370 ms`
  - new: `6.991 ms`
  - improvement: `1.05x`

- `surface_material_to_timeline_preset.world_breathe`
  - previous: `11.513 ms`
  - new: `10.671 ms`
  - improvement: `1.08x`

- `surface_materials_to_scene_timeline`
  - previous: `17.940 ms`
  - new: `18.221 ms`
  - current sample: same band / no material gain

## Overall Improvement From Original Runtime Baseline

Compared to the paused-runtime baseline (`multimodal_runtime_benchmark_20260306_142408.json`):

- `signal_to_textured_surface`
  - original: `16.087 ms`
  - current: `1.702 ms`
  - overall improvement: `9.45x`

- `surface_material_to_timeline_preset.ui_idle`
  - original: `80.227 ms`
  - current: `6.991 ms`
  - overall improvement: `11.48x`

- `surface_material_to_timeline_preset.world_breathe`
  - original: `129.709 ms`
  - current: `10.671 ms`
  - overall improvement: `12.16x`

## Interpretation

The post-codec profile showed material selection dominating the fused route, specifically ternary signature encoding. Caching the signatures and reusing PTX-backed quantization collapsed that cost.

The scene route did not materially improve in the same sample window, which is useful: the bottleneck has moved up to scene/layer orchestration rather than candidate signature work.

## Artifacts

- JSON benchmark: `../Knowledge3D.local/results/material_signature_promotion_benchmark_20260306_1800.json`
- Updated real report: `../Knowledge3D.local/runtime_execution_journal_batch/results/tool_promotion_report.json`
