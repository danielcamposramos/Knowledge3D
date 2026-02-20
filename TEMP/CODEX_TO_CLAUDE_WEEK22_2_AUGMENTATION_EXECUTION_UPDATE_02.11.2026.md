# Codex -> Claude Update (Week 22.2 Augmentation Execution)

Date: 2026-02-11
Status: Executed on main world + validated

## 1) What was implemented

### New scripts
- `scripts/augment_benchmarks_to_galaxy.py`
  - Builds sovereign augmentation payload from ARC/Math/LHE/MMLU datasets.
  - Emits benchmark-derived procedural entries into default galaxies (Grammar/Math/Reality/Drawing/3DObjects/Word).
  - Optional Ollama hints (ingestion-path only), bounded by stride/budget.
- `scripts/run_benchmark_augmentation_ingestion.sh`
  - One-command: generate payload + ingest through one persistent Knowledgeverse.
- `scripts/run_parallel_external_ingestion.sh`
  - Extended with `INCLUDE_BENCHMARK_AUGMENTATION=1` hook to chain benchmark augmentation after lexicon/audio/3D ingestion.

### Important patch
- `augment_benchmarks_to_galaxy.py` path resolution updated to include `/K3D/Knowledge3D.local/datasets/...` candidates.
- Before patch, ARC augmentation processed `0` tasks in real runs.
- After patch, ARC augmentation resolves dataset and emits ARC rows correctly.

## 2) Main-world execution

### A) External ingestion + benchmark augmentation run
Command:
```bash
INCLUDE_BENCHMARK_AUGMENTATION=1 ENABLE_OLLAMA=0 \
MAX_ARC_TASKS=400 MAX_MATH_PROBLEMS=2000 MAX_LHE_QUESTIONS=2500 MAX_MMLU_QUESTIONS=2000 MAX_WORD_ENTRIES=50000 \
bash scripts/run_parallel_external_ingestion.sh ../Knowledge3D.local ../Knowledge3D.local/datasets/external_payloads
```

Artifacts:
- First run dir: `../Knowledge3D.local/datasets/external_payloads/benchmark_aug_20260211_185635`
- Path-fixed run dir: `../Knowledge3D.local/datasets/external_payloads/benchmark_aug_20260211_210632`

### B) Ingestion stats
#### Run `...185635` (pre ARC-path fix)
- Aug payload rows: `35122`
- Added: `35122`, skipped: `0`
- ARC processed: `0` (path miss)
- Galaxy deltas:
  - Word: `9515 -> 35701`
  - Grammar: `869 -> 6120`
  - Math: `1060 -> 3146`
  - Reality: `1914 -> 3513`

#### Run `...210632` (post ARC-path fix)
- Aug payload rows: `36722`
- Added: `1600`, skipped: `35122`
- ARC processed: `400` (fixed)
- New ARC-derived additions were mostly Drawing/3DObjects/Grammar/Math bridges; all previous non-ARC rows deduped.

## 3) Benchmark metrics collected after augmentation

### LHE+MMLU bounded validation (completed)
Output:
- `../Knowledge3D.local/results/week22_2_post_aug_lhe_mmlu20_20260211_202533/week14_benchmark_summary.json`

Enriched metrics:
- LHE (20): `0.15`
- MMLU (20): `0.30`

Integrity flags:
- LHE fallback: `False` (real dataset)
- MMLU fallback: `False` (real dataset)

### ARC bounded checks (completed)
Outputs:
- `../Knowledge3D.local/results/week22_2_post_aug_arc5_20260211_210333/week14_benchmark_summary.json`
- `../Knowledge3D.local/results/week22_2_post_aug_arc5_after_arcingest_20260211_210757/week14_benchmark_summary.json`

Enriched ARC (5-task bounded):
- Top-1 accuracy: `0.00`
- oracle_at_all: `0.00`
- fuzzy_oracle_at_all: `0.00`
- generation_failure_rate: `1.00`

Coverage signal (ARC bounded):
- unique queried galaxies: `['3DObjects', 'Drawing', 'Grammar']`
- avg queried per task: `3.0`
- cross_galaxy_navigation_rate: `1.0`

Interpretation:
- ARC augmentation now ingests correctly (post-fix), but current bounded ARC metric still unchanged.
- Navigation bottleneck remains: Math/Reality still not participating in ARC query coverage.

## 4) Critical runtime/scaling finding

Attempted combined post-augmentation full-ish runs (ARC100/Math100/LHE100/MMLU100 and reduced variants) timed out repeatedly under tool windows.

This indicates a practical scaling bottleneck with enlarged world state.

Likely cause:
- `GalaxyManager.query()` is linear scan over entries with JSON serialization/token matching.
- World expansion (especially Word/Grammar growth) increases query latency substantially.

## 5) Architectural implications

- Single-world + sovereignty path is preserved and operational.
- LHE+MMLU coexistence is real-data compliant.
- Augmentation substrate expansion is now real and reproducible.
- New blocker shifted to query scalability + ARC navigation accessibility (not ingestion).

## 6) Questions / decisions needed from Claude

1. Should we implement an indexed query path for `GalaxyManager.query` (token index and/or per-galaxy inverted index) before further ARC tuning?
2. Should ARC query scope be constrained by default to relevant galaxies (Drawing/Grammar/Math/3D/Reality) to avoid large Word-dominant search overhead?
3. Should Week 22.1b forced-navigation be re-enabled by default after augmentation to force Math/Reality participation now that those bridges exist?
4. Character Galaxy remains `0`; do we hard-gate Word benchmark entries on Character bootstrap next, or continue permissive mode for momentum?

## 7) Suggested immediate next run (bounded + actionable)

```bash
conda run -n k3d-cranium env PYTHONPATH=. python scripts/run_all_benchmarks.py \
  --model-persistence-mode unified \
  --max-arc-tasks 20 --max-math-problems 20 --max-lhe-questions 20 --max-mmlu-questions 20 \
  --arc-enable-full-ptx --arc-enable-contrastive-learning --arc-enable-validity-gates \
  --arc-enable-forced-navigation-curriculum --arc-forced-navigation-ratio 0.4 \
  --arc-forced-navigation-required-galaxies Math,Reality,3DObjects,Drawing,Grammar \
  --track-curriculum-coverage \
  --output-dir ../Knowledge3D.local/results/week22_2_post_aug_forced_nav_20each \
  --storage-root ../Knowledge3D.local
```

Goal: test whether ARC query coverage expands from 3 galaxies to >=5 on enriched substrate.

