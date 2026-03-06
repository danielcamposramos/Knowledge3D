# Ternary Packing Promotion Benchmark

**Date**: March 6, 2026
**Environment**: `/K3D/Knowledge3D.local/envs/k3d-cranium`
**Scope**: reduce shared encode-time overhead by vectorizing `TernaryVector` packing/unpacking and switching codec callers to direct numpy handoff.

## Changes

- `TernaryVector` now uses vectorized normalization and host packing.
- `TernaryVector.to_python()` now unpacks vectorized instead of byte-by-byte Python loops.
- `TernaryVector.to_numpy()` is now available for direct codec handoff.
- sovereign audio/video codecs now use `to_numpy()` instead of bouncing through Python lists first.

## Measured Results

### Signal / Material Route

Previous post-encode-promotion baseline:
- `signal_to_textured_surface`: `7.700 ms`

New hot average:
- `signal_to_textured_surface`: `6.220 ms`

Improvement vs previous promoted baseline:
- `1.24x`

### Temporal Route

Previous post-encode-promotion baseline:
- `ui_idle`: `22.947 ms`
- `world_breathe`: `37.488 ms`

New hot averages:
- `ui_idle`: `23.195 ms`
- `world_breathe`: `30.504 ms`

Interpretation:
- `world_breathe` improved materially again (`1.23x`).
- `ui_idle` stayed in the same range, with some variance from the small sample and GPU context effects.
- The shared ternary substrate is now cheaper across both signal and temporal encode paths.

## Artifact

- JSON benchmark: `../Knowledge3D.local/results/ternary_packing_promotion_benchmark_20260306_1735.json`
