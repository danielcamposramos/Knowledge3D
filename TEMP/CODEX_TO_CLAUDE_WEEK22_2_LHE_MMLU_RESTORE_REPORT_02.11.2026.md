# Codex → Claude: LHE Restore + Input Primer + Validation Report

Date: 2026-02-11
Repo: Knowledge3D

## 1) Implemented Changes

### A. Restored LHE alongside MMLU (coexistence, not replacement)
- `scripts/run_all_benchmarks.py`
  - Added MMLU execution in `dual` mode parity.
  - Added MMLU into `summary["benchmarks"]` and runtime usage `runs`.
  - Added MMLU integrity checks/printing analogous to LHE.
- LHE remains active and wired in same runner.

### B. Added pre-routing input specialist (normalization-only)
- New: `knowledge3d/knowledgeverse/input_primer_specialist.py`
  - `normalize_chat_messages(messages)`
  - `prepare_multiple_choice(question_text, options)`
- Updated: `knowledge3d/knowledgeverse/trm_navigator.py`
  - Bootstraps `InputPrimerSpecialist` as sibling to `ChatSpecialist`.
  - `process_chat(...)` now normalizes input via primer before specialist processing.
  - `answer_multiple_choice(...)` now uses primer pre-processing and maps back to original options.

### C. Fixed chat specialist compatibility for current Knowledgeverse API
- Updated: `knowledge3d/knowledgeverse/chat_specialist.py`
  - Replaced calls to non-existent `Knowledgeverse.query_galaxy(...)`.
  - Added `_query_galaxy(...)` that uses `knowledgeverse.galaxy_manager.query(...)` and normalizes results.

### D. Restored real LHE dataset ingestion from HF and open-ended support
- Updated converter: `scripts/prepare_lhe_dataset_from_hf.py`
  - Now converts open-ended rows from `cais/hle` (not only MCQ).
  - Preserves `answer_type`, `category`, `subject` metadata.
- Updated benchmark: `benchmarks/last_humanity_exam.py`
  - Accepts open-ended questions.
  - Uses `navigator.process_chat(...)` for open-ended answers.
  - Added normalized text matching for open-ended correctness.

### E. MMLU synthetic-fallback flag correctness fix
- Updated: `benchmarks/mmlu.py`
  - `synthetic_fallback` now tracks true fallback path, not `len(questions) <= 10` heuristic.

## 2) Data Preparation Executed

### LHE dataset regeneration (real source)
Command:
```bash
HF_TOKEN=*** conda run -n k3d-cranium env PYTHONPATH=. \
  python scripts/prepare_lhe_dataset_from_hf.py \
  --dataset cais/hle --split test \
  --output-dir ../Knowledge3D.local/datasets/last_humanity_exam
```
Result:
- Converted: 2500
- Dropped: 0
- Output: `../Knowledge3D.local/datasets/last_humanity_exam/questions.json`

## 3) Validation Runs and Metrics

### Run A (explicit all four benchmarks enabled)
Output summary:
- `../Knowledge3D.local/results/week22_2_post_ingestion_validation_lhe_mmlu/week14_benchmark_summary.json`

Key metrics (enriched):
- ARC: `0.06` (6/100)
- Math: `0.00` (0/100)
- LHE: `0.24` (24/100)
- MMLU: `0.27` (27/100)

Integrity:
- LHE: source=`file`, fallback=`false`, evaluated=`100/100`
- MMLU: source=`MMLU`, fallback=`false`, evaluated=`100/100`

Architecture/runtime:
- PTX full/ranking/oracle used rates all `1.0`
- Shared instance confirmed: same instance id for empty/enriched
- Embedding lazy mode: `skip`
- Query coverage still concentrated in ARC on `Drawing/Grammar/3DObjects`

World persistence evidence:
- Galaxy counts (same world, not clean reset):
  - Grammar: `41799 -> 43799` in this run context (+2000)
  - Character/Word/Math/Reality already populated before run

### Run B (large LHE-oriented run)
Output summary:
- `../Knowledge3D.local/results/lhe_validation_200/week14_benchmark_summary.json`

Key metrics (enriched):
- LHE: `0.25` (50/200)
- MMLU: `0.2705` (3794/14042)

Integrity:
- LHE evaluated=200, fallback=false
- MMLU evaluated=14042, fallback=false

## 4) Important Operational Finding

`run_all_benchmarks.py` currently treats `--max-*-questions 0` as effectively unbounded in benchmark constructors (not a hard skip), which can cause accidental full-dataset runs.

Recommendation:
- Add explicit `--skip-arc / --skip-math / --skip-lhe / --skip-mmlu` flags, or
- Normalize `max<=0` to true skip semantics in runner.

## 5) Questions / Suggested Next Steps for Architecture

1. **LHE scoring policy**: current open-ended matching is strict normalized text match/containment; do we want semantic equivalence scoring (with guarded rubric) for integrity runs?
2. **Benchmark skip semantics**: approve explicit skip flags to avoid accidental full runs with `max=0`.
3. **Math 0/100 issue**: decide whether to keep current solver path for MATH, or route through strengthened pre-routing + specialist-specific MCQ/open response adapter.
4. **ARC navigation bottleneck unchanged**: proceed with Week 22.1b forced-navigation curriculum (Math/Reality participation) now that LHE+MMLU integrity is restored?

## 6) Files touched this cycle

- `knowledge3d/knowledgeverse/input_primer_specialist.py` (new)
- `knowledge3d/knowledgeverse/trm_navigator.py`
- `knowledge3d/knowledgeverse/chat_specialist.py`
- `benchmarks/last_humanity_exam.py`
- `benchmarks/mmlu.py`
- `scripts/prepare_lhe_dataset_from_hf.py`
- `scripts/run_all_benchmarks.py`

