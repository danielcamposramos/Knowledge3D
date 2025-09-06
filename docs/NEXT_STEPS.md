# Next Steps (Codex note)

Focus
1) Viewer LOD swap: read `extras.k3d.lods` and change POSITION by camera distance.
2) Build the 1M LOD GLB with `scripts/pipeline_1m.sh` (already LOD‑enabled).
3) Restart live server and run multilingual `multi_instance` to accumulate fresh logs.
4) Retrain HF intent model (xlm‑roberta‑base), save to `../Knowledge3D.local/models/intent_hf`.
5) (Optional) RL pass with `knowledge3d.rl.unsloth_adapter` to prioritize efficient, safe hops.
6) Knowledge Garden: generate greenhouse GLB and link as a door in the house.

Viewer LOD thresholds (proposal)
- Far (> D2): default POSITION (PCA) — fast overview.
- Mid (D1..D2): `umap_fast` from `extras.k3d.lods`.
- Near (< D1): `umap_high` from `extras.k3d.lods`.

Operational notes
- Local artifacts live under `../Knowledge3D.local/` — safe to move to a faster disk; update mounts accordingly.
- Containers can be stopped/removed anytime; rebuild image via `scripts/docker_build_gpu.sh`.
- Datasets server is optional; viewer can load GLBs directly from disk.

