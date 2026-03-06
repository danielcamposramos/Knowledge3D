# Codec Boundary Promotion Benchmark

**Date**: March 6, 2026
**Environment**: `/K3D/Knowledge3D.local/envs/k3d-cranium`
**Scope**: remove Python-list marshaling from sovereign codec ops and stop re-decoding immutable ternary vectors through device downloads on hot routes.

## Changes

- `TernaryCodecOps` now has numpy-returning PTX methods for:
  - `quantize`
  - `dequantize`
  - `batch_mdct`
  - `batch_imdct`
  - `dct8_forward`
  - `dct8_inverse`
  - `reshape_to_blocks`
  - `blocks_to_grid`
- `SovereignTernaryAudioCodec` now uses those numpy/PTX paths directly.
- `SovereignTernaryVideoCodec` now uses those numpy/PTX paths directly.
- `TernaryVector.to_numpy()` now unpacks from the immutable host cache instead of re-downloading device bytes for every read.
- `ProceduralSignalBridge.audio_to_spectrogram(...)` now consumes `residual.to_numpy()` directly.

## Measured Results

Compared to the previous promoted baseline (`encode_path_promotion_benchmark_20260306_1710.json`):

- `signal_to_textured_surface`
  - previous: `7.700 ms`
  - new: `4.827 ms`
  - improvement: `1.59x`

- `surface_material_to_timeline_preset.ui_idle`
  - previous: `22.947 ms`
  - new: `7.370 ms`
  - improvement: `3.11x`

- `surface_material_to_timeline_preset.world_breathe`
  - previous: `37.488 ms`
  - new: `11.513 ms`
  - improvement: `3.26x`

- `surface_materials_to_scene_timeline`
  - previous: `58.189 ms`
  - new: `17.940 ms`
  - improvement: `3.24x`

## Interpretation

The dominant cost was no longer PTX math. It was the codec boundary:

- Python list materialization between PTX calls
- repeated array flatten/list conversion in audio/video codecs
- device-to-host readback when immutable packed host bytes were already available

This promotion removed that bookkeeping from the dominant fused routes without changing semantics.

## Artifacts

- JSON benchmark: `../Knowledge3D.local/results/codec_boundary_promotion_benchmark_20260306_1749.json`
