# Encode Path Promotion Benchmark

**Date**: March 6, 2026
**Environment**: `/K3D/Knowledge3D.local/envs/k3d-cranium`
**Scope**: remove repeated bridge/codec setup and eliminate generic RPN flatten/reshape from sovereign audio/video encode paths.

## Changes

- `SovereignTernaryAudioCodec` now uses direct PTX codec ops (`BATCH_MDCT`, `TERNARY_QUANT`, `TERNARY_DEQUANT`, `IMDCT`) instead of generic `ModularRPNEngine.evaluate(...)`.
- `SovereignTernaryVideoCodec` now uses direct PTX codec ops (`RESHAPE_TO_BLOCKS`, `DCT8X8_FORWARD`, `TERNARY_QUANT`, inverse path) instead of generic RPN flatten/reshape.
- `ProceduralMaterialBridge.signal_to_textured_surface(...)` now reuses configured `ProceduralSignalBridge` instances instead of rebuilding them on every call.
- `ProceduralTemporalBridge` now reuses `SovereignTernaryVideoCodec` instances and encodes frames through `encode_frame_array(...)`, avoiding `frame -> TernaryTensor -> Python list -> reshape -> flatten -> TernaryVector` churn.

## Measured Results

### Signal / Material Route

- Baseline `signal_to_textured_surface`: `16.087 ms`
- New hot average: `7.700 ms`
- Improvement: `2.09x`

### Temporal Route

Previous post-PTX-frame baseline (`temporal_route_promotion_benchmark_20260306_1648.json`):
- `ui_idle`: `75.899 ms`
- `world_breathe`: `120.361 ms`

New hot averages:
- `ui_idle`: `22.947 ms`
- `world_breathe`: `37.488 ms`

Improvement vs previous promoted baseline:
- `ui_idle`: `3.31x`
- `world_breathe`: `3.21x`

## Interpretation

The remaining cost was not frame synthesis anymore. It was encode-time bookkeeping:

- repeated bridge/codec construction
- generic RPN flatten/reshape in codec paths
- ternary packing churn in the temporal preview route

This promotion removed those costs from the dominant route family without changing semantics.

## Artifacts

- JSON benchmark: `../Knowledge3D.local/results/encode_path_promotion_benchmark_20260306_1710.json`
- Previous runtime baseline: `../Knowledge3D.local/results/multimodal_runtime_benchmark_20260306_142408.json`
- Previous temporal baseline: `../Knowledge3D.local/results/temporal_route_promotion_benchmark_20260306_1648.json`
