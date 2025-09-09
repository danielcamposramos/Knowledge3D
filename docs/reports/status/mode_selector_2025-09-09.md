# Mode Selector (Compose vs Compose‑Generate) — 2025-09-09

We added a learned selector to auto‑choose between `compose` (retrieval+stitching) and `compose_generate` (grounded generative) — a key step toward the "one head" vision.

## Implementation
- Model: logistic regression over lightweight features
  - n_ctx, avg_ctx_len, sum_ctx_len, q_len, media_frac
- Labels (bootstrapped):
  - contexts[] empty → `compose`
  - contexts[] non‑empty → `compose_generate`
- Files:
  - Trainer: `knowledge3d/models/mode_selector.py`
  - Integration: `knowledge3d/skills/spatial_text.py` (new `compose_auto`)

## Train
```
scripts/k3d_env.sh run python -m knowledge3d.models.mode_selector \
  --dataset docs/reports/training/rlwhf_dataset_unified_v3.jsonl \
  --out ../Knowledge3D.local/models/mode_selector.pkl
```
- Info: `{rows: 9220, acc_train: 1.0}` (expected given heuristic labels).

## Use
- Programmatic:
  - `from knowledge3d.skills.spatial_text import compose_auto`
  - `mode, text = compose_auto(question, [(label, text), ...])`
- Fallback: if selector missing, heuristic: contexts→generate, else compose.

## Notes
- This is a first pass. As we log real outcomes, we’ll train the selector on true performance labels (e.g., downstream quality/reward) rather than the presence of contexts only.
