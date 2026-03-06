# Signal Surface Promotion Benchmark

Date: 2026-03-06
Route: `SIGNAL_SURFACE_MATERIAL`
Promoted surface: `spectrogram_to_surface`

## Result

- Previous baseline: `1.326 ms`
- New hot-path average: `0.1865 ms`
- Improvement: about `7.1x`

## Post-Promotion Measurements

- `spectrogram_to_surface`
  - min: `0.1791 ms`
  - avg: `0.1865 ms`
  - max: `0.2256 ms`
- `signal_to_textured_surface`
  - min: `12.6024 ms`
  - avg: `13.5045 ms`
  - max: `17.0816 ms`

## Interpretation

The PTX promotion hit the correct bottleneck. The signal route no longer spends its time in host-side heightfield vertex and normal generation. The remaining heavier cost in the fused route is now downstream composition and material/temporal orchestration, not surface extraction itself.
