# Codex → Claude: Priority 1 & 2 Update (Skip Semantics + Math Investigation)

Date: 2026-02-11

## Scope completed

## 1) Priority 1 complete: `max<=0` now means skip (safe semantics)

Updated `scripts/run_all_benchmarks.py`:
- Added explicit enable checks per benchmark:
  - `run_arc`, `run_math`, `run_lhe`, `run_mmlu` via `_is_enabled_limit(...)`.
- Added skipped-result payloads and skipped runtime metrics:
  - `_arc_skipped_result`, `_math_skipped_result`, `_lhe_skipped_result`, `_mmlu_skipped_result`, `_skip_metrics`.
- Unified mode + dual mode both now skip a benchmark when max<=0.
- ARC stage gate now returns `skipped=true` when ARC is disabled.
- Integrity checks now respect skip mode:
  - `real_dataset_ok` / `question_count_ok` are true when intentionally skipped.
  - `--*-require-real-dataset` no longer fails for explicitly skipped benchmark.
- Console output now prints explicit skip line for LHE/MMLU integrity.

Validation run:
```bash
python scripts/run_all_benchmarks.py \
  --model-persistence-mode unified \
  --max-arc-tasks 1 --max-math-problems 1 --max-lhe-questions 1 --max-mmlu-questions 0 \
  ...
```
Observed:
- `MMLU integrity: skipped (max-mmlu-questions <= 0)`
- No accidental full MMLU run.

## 2) Priority 2 investigation complete: Math `0/100` is not routing failure

Added math diagnostics in `benchmarks/math_competitions.py`:
- `predicted_none_count`
- `predicted_none_rate`
- `predicted_numeric_count`
- `expected_numeric_count`
- `route_specialist_counts`

Focused diagnostic run:
```bash
python scripts/run_all_benchmarks.py \
  --model-persistence-mode unified \
  --max-arc-tasks 0 --max-math-problems 20 --max-lhe-questions 0 --max-mmlu-questions 0 \
  ...
```

Result (`../Knowledge3D.local/results/week22_math_routing_diag/math_competitions_enriched.json`):
- overall accuracy: `0.0` (0/20)
- diagnostics:
  - `predicted_none_count=13`
  - `predicted_none_rate=0.65`
  - `predicted_numeric_count=7`
  - `expected_numeric_count=20`
  - `route_specialist_counts={"math": 20}`

Interpretation:
- Routing is correct (`math` specialist on all 20 tasks, galaxy names `['Math','Grammar']`).
- Failure is solver capability/extraction quality (`_solve_math`) against AMC-style tasks, not router path.

## Recommendation (next implementation)

1. Keep current routing architecture unchanged.
2. Improve math solver path in TRM:
   - broaden expression extraction for competition-style word problems,
   - improve sign/offset handling,
   - add deterministic equation solve patterns before fallback `None`.
3. Keep skip semantics as baseline safety policy for all future benchmark scripts.

## Files changed in this update
- `scripts/run_all_benchmarks.py`
- `benchmarks/math_competitions.py`
