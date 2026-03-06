# Multimodal Runtime Baseline

- Timestamp: `2026-03-06T14:24:10-0300`
- JSON: `../Knowledge3D.local/results/multimodal_runtime_benchmark_20260306_142408.json`

## Warmup

- Drawing warmup total: `104.44626300159143` ms
- Material warmup total: `11.422969000705052` ms

## Benchmarks

- `drawing.execute_simple_line`: min `0.114` ms, avg `0.129` ms, p95 `0.198` ms
- `drawing.render_painterly_gpu`: min `1.125` ms, avg `1.201` ms, p95 `1.322` ms
- `material.select_material`: min `2.422` ms, avg `2.68` ms, p95 `3.143` ms
- `material.contour_to_textured_lathe_mesh`: min `5.922` ms, avg `7.04` ms, p95 `9.222` ms
- `material.contour_to_textured_extrude_mesh`: min `5.62` ms, avg `6.295` ms, p95 `7.066` ms
- `material.contour_to_textured_sweep_mesh`: min `5.754` ms, avg `6.231` ms, p95 `6.936` ms
- `signal.audio_to_spectrogram`: min `1.914` ms, avg `1.968` ms, p95 `2.049` ms
- `signal.spectrogram_to_surface`: min `1.236` ms, avg `1.326` ms, p95 `1.438` ms
- `material.signal_to_textured_surface`: min `15.526` ms, avg `16.087` ms, p95 `16.505` ms
- `temporal.surface_material_to_timeline_preset.ui_idle`: min `76.246` ms, avg `80.227` ms, p95 `90.054` ms
- `temporal.surface_material_to_timeline_preset.world_breathe`: min `125.449` ms, avg `129.709` ms, p95 `138.686` ms
- `temporal.surface_materials_to_scene_timeline`: min `202.325` ms, avg `211.399` ms, p95 `223.337` ms
- `navigator.execute.signal_surface_material`: min `322.345` ms, avg `380.539` ms, p95 `431.493` ms
