# Phase G Batch Pipeline (Deprecated)

This folder preserves the previous Phase G training orchestration scripts that relied on CPU-rendered glyph batches and large host-resident numpy tensors. The approach produced 1,500-epoch jobs with 1,572 fonts per character, which in turn inflated host RAM to ~0.6 GB per worker and triggered the OOM killer whenever more than a few processes ran in parallel. With the move to a sovereign procedural rasterizer, these helpers are archived for reference only.

## Archived Scripts
- `train_all_characters_batch.py`
- `train_atomic_characters_parallel.py`
- `train_atomic_characters_dynamic.py`
- `train_all_atomic_characters.py`

## Legacy Checkpoints
The resulting checkpoints/logs were relocated to:
```
/K3D/Knowledge3D.local/checkpoints/phase_g/old_batch_attempts_2025-11-14
```
leaving `/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars/` empty for the procedural restart. Do not reuse the archived weights with the new pipeline—they still embed the anti-procedural rasterization artifacts.
