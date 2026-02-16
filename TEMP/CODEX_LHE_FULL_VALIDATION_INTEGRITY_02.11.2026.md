# Week 22 LHE Full Validation Integrity Check (2026-02-11)

## Executive Result
A "full LHE" run was executed, but the benchmark is **not** using the official HLE question set. It is evaluating the synthetic fallback set (4 questions).

- Output summary: `../Knowledge3D.local/results/lhe_full_validation/week14_benchmark_summary.json`
- Enriched LHE result:
  - `dataset_path`: `/K3D/Knowledge3D.local/datasets/exams/hle-src`
  - `total_questions`: `4`
  - `accuracy`: `1.0`

Therefore, current `100%` LHE results are **not comparable** to public HLE leaderboard/SOTA claims.

## Evidence

### 1) Loader behavior in code
`benchmarks/last_humanity_exam.py`:
- resolves dataset path candidates including `.../datasets/exams/hle-src`.
- only loads if one of these files exists:
  - `last_humanity_exam.json`
  - `questions.json`
  - `dataset.json`
  - `questions.jsonl`
- if none exists, it falls back to synthetic questions.

Relevant refs:
- `benchmarks/last_humanity_exam.py:30-42` (path resolution)
- `benchmarks/last_humanity_exam.py:52-90` (known-file loaders)
- `benchmarks/last_humanity_exam.py:111-151` (synthetic fallback set)

### 2) Local `hle-src` content
`/K3D/Knowledge3D.local/datasets/exams/hle-src` contains README/eval scripts/images, but no supported question JSON/JSONL file for K3D loader.

### 3) Full-run output confirms fallback-scale evaluation
`../Knowledge3D.local/results/lhe_full_validation/week14_benchmark_summary.json`:
- `benchmarks.last_humanity_exam.enriched.total_questions = 4`
- `benchmarks.last_humanity_exam.enriched.accuracy = 1.0`

## New Integrity Guard Added
Implemented fail-fast controls in benchmark runner:

- `--lhe-require-real-dataset`
- `--lhe-min-questions`

When enabled, run aborts if:
- synthetic fallback is used, or
- evaluated question count is below required minimum.

Validation run (strict) now fails correctly with:
- `synthetic_fallback=True`
- `evaluated_questions=4`
- `min_required=1000`

Refs:
- `scripts/run_all_benchmarks.py` (new args + integrity checks)
- `benchmarks/last_humanity_exam.py` (result metadata now includes `dataset_source`, `dataset_file`, `synthetic_fallback`)

## Recommended Next Step (for valid LHE claim)
1. Obtain access/auth for gated dataset: `cais/hle` (Hugging Face).
2. Export it to a loader-compatible file under:
   - `/K3D/Knowledge3D.local/datasets/last_humanity_exam/questions.json` (or JSONL)
3. Re-run with strict integrity:

```bash
conda run -n k3d-cranium env PYTHONPATH=. python scripts/run_all_benchmarks.py \
  --model-persistence-mode unified \
  --max-arc-tasks 0 --max-math-problems 0 --max-lhe-questions 1000 \
  --lhe-min-questions 1000 --lhe-require-real-dataset \
  --output-dir ../Knowledge3D.local/results/lhe_full_validation_real \
  --storage-root ../Knowledge3D.local
```

Only after this should LHE-vs-SOTA comparisons be used in paper claims.
