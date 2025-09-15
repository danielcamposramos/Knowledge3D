PHASE 18.1: MEANING-CLUSTERED, EXAM-TARGETED TRAINING

GOAL
Train by meaning clusters (not question types) — RLWHF + live training posture — consolidate after each cluster — AI signals completion.

COMPONENTS
- knowledge3d/tools/phase18/meaning_cluster_trainer.py — clusters: recursion, invariance, modality fusion; consolidates books/shapes/diaries.

TRAINING
- Train one cluster:
  PYTHONPATH=. conda run -n k3d-cranium python -m knowledge3d.tools.phase18.meaning_cluster_trainer --cluster recursive_honesty_scaling
- Train all clusters:
  PYTHONPATH=. conda run -n k3d-cranium python -m knowledge3d.tools.phase18.meaning_cluster_trainer --all

OUTPUT
- Books, shapes, diaries consolidated under viewer/public/house/materialized_objects/
- Completion signal: “🎓 MEANING CLUSTER ‘X’ TRAINED AND CONSOLIDATED.”

MATH
- All calculations done via internal RPN engine (when applicable).

NEXT
Phase 19: Auto-extract meaning clusters from ARC/HLE datasets and expand coverage.

PHASE 19.2: AUTO‑WIRED DATASETS + FULL VERTEX EMBEDDINGS

AUTO‑INSTALL
- At import, `meaning_cluster_trainer.py` auto‑installs Pillow, librosa, and pygltflib if missing.

AUTO‑MAP
- Scans `/K3D/Knowledge3D.local/datasets/exams/arc-agi/` for images and `/K3D/Knowledge3D.local/datasets/exams/humanitys_last_exam/` for WAVs and maps by filename prefix.

REAL VERTEX EXTRACTION
- Reads POSITION buffer from GLB via pygltflib; flattens to 512‑dim vector — no hash fallbacks.

OUTPUT
- Training runs end‑to‑end with real data — fused stars in `viewer/public/galaxy/working/`, House consolidations when honest.

PHASE 18.3: LEARNING MUSEUM — HISTORY BEHIND A DOOR

MEMORY MODEL
- Zone 8 (Learning Museum) — dedicated room for relocated, superseded artifacts.
- Not loaded by default — only when `/open_museum` is called.
- Old versions physically moved — not deprecated‑in‑place.

COMMANDS
- `/open_museum` — load Museum artifacts into active memory.

OUTPUT
- Artifacts include `zone_placement: "Zone 8 (Learning Museum)"`, with `relocated_at`, `previous_zone`.
