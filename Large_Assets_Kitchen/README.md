# Large Assets Kitchen

Large (>99 MB) or bulk-generated artifacts live outside the repo under `../Knowledge3D.local/`. Use these recipes to rebuild them when you need to run legacy viewers or regression suites.

## House & Viewer Exports
- Target location: `Knowledge3D.local/old_attempts/legacy_fancy_rag/viewer_public/`
- Recipe: run `python -m knowledge3d.tools.house_memory_builder --house-id <id> --output Knowledge3D.local/...` followed by `python -m knowledge3d.tools.publish_local_artifacts`.
- Reference docs: `docs/HOUSE_MEMORY.md`, `docs/TABLET_APPS.md`

## Benchmark Reports
- Target: `Knowledge3D.local/old_attempts/legacy_fancy_rag/docs_reports/`
- Recipe: rerun the evaluation scripts:
  - Math: `python -m knowledge3d.tools.phase25.math_bench_evaluator --auto`
  - ARC/HLE: `python -m knowledge3d.tools.phase23.arc_hle_tester --all`
  - RLWHF: `python -m knowledge3d.tools.phase25.consistency_trainer --replay latest`
- Save outputs to the `.local` path to keep Git clean.

## AI Compendium CSV
- Target: `Knowledge3D.local/datasets/ai_compendium_80k_vectors.csv`
- Recipe: `python -m knowledge3d.tools.build_ai_compendium --vectors-out Knowledge3D.local/datasets/ai_compendium_80k_vectors.csv --limit 80000`

## ARC-AGI Reasoning Cache
- Target: `Knowledge3D.local/datasets/arc_agi/arc_reasoning_pairs.npz`
- Recipe:
  1. `python scripts/train_trm_on_arc_reasoning.py --epochs 0 --rebuild-cache` (build cache only)
     - Optional: add `--limit-pairs <int>` for quick smoke tests.
  2. The script downloads the ARC-AGI dataset under `Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/`
     and stores metadata next to the cache (`arc_reasoning_pairs.json`).
- Notes: Keep the cache in sync with the latest RPN embeddings; rebuild after significant sleep-time consolidation so reasoning aligns with House/Galaxy updates.

## Galaxy GLBs (viewer/dist)
- Target: `Knowledge3D.local/old_attempts/legacy_fancy_rag/viewer_dist/`
- Recipe: `npm run build` inside `viewer/`, then copy the resulting GLBs to `.local`.
- Make sure `K3D_LOCAL_DIR` points to `.local` before building so runtime loads from the right place.

## Legacy Examples
- Target: `Knowledge3D.local/old_attempts/legacy_fancy_rag/examples/`
- Recipe: regenerate with `python -m k3dgen` per the original README instructions. Update manifests if you add or remove samples.

> Keep this folder up to date whenever you introduce a new large artifact. Describe **where** it should live in `.local` and **how** to rebuild it.
