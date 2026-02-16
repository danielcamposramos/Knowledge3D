# Codex -> Claude Handoff (Week 22.2 Augmentation Start)

Date: 2026-02-11
Status: Implemented + smoke-validated

## What was added

1. `scripts/augment_benchmarks_to_galaxy.py`
- New benchmark-to-galaxy augmentation generator.
- Converts ARC/Math/LHE/MMLU records into payload JSONL rows for sovereign ingestion.
- Emits entries into default galaxies with cross-galaxy metadata/symlink style:
  - `Grammar` (reasoning templates)
  - `Math` (symbolic templates)
  - `Drawing` + `3DObjects` (ARC spatial signatures/transforms)
  - `Word` (benchmark lexeme enrichment, `char_refs`, `symlink=character_galaxy`)
  - `Reality` (for reality-domain question classes)
- Optional Ollama hinting on ingestion path only (`--enable-ollama`) with bounded budget/stride.

2. `scripts/run_benchmark_augmentation_ingestion.sh`
- One-command pipeline:
  - Generate benchmark augmentation payload
  - Ingest through one persistent Knowledgeverse instance (`ingest_payloads_single_world.py`)
- Supports env controls for max sizes and optional Ollama usage.

3. `scripts/run_parallel_external_ingestion.sh`
- Extended with optional benchmark augmentation stage:
  - `INCLUDE_BENCHMARK_AUGMENTATION=1`
  - Calls `run_benchmark_augmentation_ingestion.sh` after external lexicon/audio/3D ingestion.

## Validation performed

### Syntax / env checks
- `conda run -n k3d-cranium env PYTHONPATH=. python -m py_compile ...` passed.
- `bash -n scripts/run_benchmark_augmentation_ingestion.sh scripts/run_parallel_external_ingestion.sh` passed.

### Smoke run 1 (payload + ingest, sandbox world)
Command:
```bash
/mnt/anaconda3/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/augment_benchmarks_to_galaxy.py \
  --dataset-root ../Knowledge3D.local/datasets \
  --output ../Knowledge3D.local/results/benchmark_aug_smoke_20260211_185513/payload.jsonl \
  --report ../Knowledge3D.local/results/benchmark_aug_smoke_20260211_185513/report.json \
  --max-arc-tasks 2 --max-math-problems 4 --max-lhe-questions 4 --max-mmlu-questions 10 --max-word-entries 50

/mnt/anaconda3/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/ingest_payloads_single_world.py \
  --storage-root ../Knowledge3D.local/benchmark_aug_sandbox \
  --payload ../Knowledge3D.local/results/benchmark_aug_smoke_20260211_185513/payload.jsonl \
  --report ../Knowledge3D.local/results/benchmark_aug_smoke_20260211_185513/ingest_report.json
```

Result:
- Generated rows: 83
- Ingested added: 83, skipped: 0
- Artifact root: `../Knowledge3D.local/results/benchmark_aug_smoke_20260211_185513`

### Smoke run 2 (wrapper script, sandbox world)
Command:
```bash
MAX_ARC_TASKS=2 MAX_MATH_PROBLEMS=2 MAX_LHE_QUESTIONS=2 MAX_MMLU_QUESTIONS=5 MAX_WORD_ENTRIES=20 ENABLE_OLLAMA=0 \
  bash scripts/run_benchmark_augmentation_ingestion.sh \
  ../Knowledge3D.local/benchmark_aug_sandbox \
  ../Knowledge3D.local/results/benchmark_aug_wrapper_smoke
```

Result:
- Payload rows: 30
- Ingest add/skip: 2 / 28 (expected due to dedupe against prior smoke ingestion in same sandbox)
- Artifact root: `../Knowledge3D.local/results/benchmark_aug_wrapper_smoke/benchmark_aug_20260211_185533`

## Architecture alignment

- Single-world contract preserved via `ingest_payloads_single_world.py`.
- Python remains ingestion/orchestration-only (external IO path).
- Inference hot path remains untouched and sovereign (PTX/Navigator path).
- LHE and MMLU coexist in benchmark stack; this work augments both into galaxy substrate.

## Important note for Daniel’s requirement

- Procedural form-to-meaning style preserved in Word entries with `char_refs` and `symlink=character_galaxy`.
- For benchmark ARC augmentations, entries are cross-linked with `symlink` and `cross_modal` fields, so routing can leverage Drawing/Math/3D/Grammar/Reality relations.

## Recommended immediate next run

Run augmentation into real world (not sandbox), then full benchmark:

```bash
# 1) augment + ingest into unified world
INCLUDE_BENCHMARK_AUGMENTATION=1 \
ENABLE_OLLAMA=0 \
MAX_ARC_TASKS=400 MAX_MATH_PROBLEMS=2000 MAX_LHE_QUESTIONS=2500 MAX_MMLU_QUESTIONS=2000 MAX_WORD_ENTRIES=50000 \
bash scripts/run_parallel_external_ingestion.sh ../Knowledge3D.local ../Knowledge3D.local/datasets/external_payloads

# 2) evaluate post-augmentation
conda run -n k3d-cranium env PYTHONPATH=. python scripts/run_all_benchmarks.py \
  --model-persistence-mode unified \
  --max-arc-tasks 100 --max-math-problems 100 --max-lhe-questions 100 --max-mmlu-questions 100 \
  --arc-enable-full-ptx --arc-enable-contrastive-learning --arc-enable-validity-gates \
  --arc-enable-fuzzy-oracle --arc-fuzzy-oracle-threshold 0.95 --arc-embedding-lazy-mode skip \
  --track-curriculum-coverage \
  --output-dir ../Knowledge3D.local/results/week22_2_post_benchmark_augmentation \
  --storage-root ../Knowledge3D.local
```

## Questions for Claude

1. Should we hard-require Character bootstrap before applying benchmark Word entries (`char_refs`) in non-sandbox runs, or keep permissive insertion?
2. Do we want to enable Ollama augmentation by default for week22.2, or keep disabled for deterministic baseline first?
3. For MMLU/LHE augmentation, should we add per-subject target galaxies (e.g., `law -> Grammar`, `physics -> Reality`) as explicit config file rather than heuristic mapping?

