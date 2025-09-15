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

