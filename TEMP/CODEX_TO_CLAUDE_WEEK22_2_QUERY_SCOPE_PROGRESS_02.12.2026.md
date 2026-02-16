# Codex -> Claude: Week 22.2 Query Scope Progress (02.12.2026)

## What Was Implemented

### 1) Query Scoping Support in Core Manager
- File: `knowledge3d/knowledgeverse/galaxy_manager.py`
- Changes:
  - Added optional `galaxies` parameter to `GalaxyManager.query(...)`.
  - Added target galaxy resolution helper (`_resolve_target_galaxies`).
  - Added cached serialized haystack (`_entry_text_cache`) to reduce repeated `json.dumps` overhead.
  - Reordered filtering logic so specialist/domain checks happen before expensive token matching where possible.
  - Clears cache on `add_entry` to avoid stale pointers.

### 2) Scoped Queries in Chat Specialist
- File: `knowledge3d/knowledgeverse/chat_specialist.py`
- Changes:
  - `_query_galaxy(...)` now passes `galaxies=[galaxy_name]`.
  - `answer_multiple_choice(...)` accepts `galaxy_scope`.
  - MCQ scoring now includes scoped Galaxy evidence retrieval (still sovereignty-safe, no external calls).

### 3) TRM Bridge for MCQ Scope
- File: `knowledge3d/knowledgeverse/trm_navigator.py`
- Changes:
  - `answer_multiple_choice(...)` now accepts `galaxy_scope` and forwards to `ChatSpecialist`.

### 4) Benchmark-Level Query Scope Inputs
- Files:
  - `benchmarks/arc_agi_2.py`
  - `benchmarks/arc_agi_2_adapter.py`
  - `benchmarks/math_competitions.py`
  - `benchmarks/last_humanity_exam.py`
  - `benchmarks/mmlu.py`
- Changes:
  - Added optional `query_scope_galaxies` in each benchmark class.
  - Added normalization + route scope application in Math/LHE.
  - Added scope propagation into ARC adapter and MMLU MCQ path.

### 5) Runner Wiring + CLI
- File: `scripts/run_all_benchmarks.py`
- Added CLI flags:
  - `--arc-query-scope-galaxies`
  - `--math-query-scope-galaxies`
  - `--lhe-query-scope-galaxies`
  - `--mmlu-query-scope-galaxies`
- Scoped values are passed into benchmark constructors (unified + dual paths).
- Scope values are included in runtime output and summary payload.

## Validation Performed

### Static checks
- `python3 -m py_compile` on modified modules: **PASS**

### Runtime checks (bounded, no PTX dependency)
- Command (example): small run with `--max-arc-tasks 5 --max-math-problems 5 --max-lhe-questions 5 --max-mmlu-questions 5`.
- Storage/output used in `/tmp` sandbox paths to keep run isolated.
- Result: run completes and prints/records all new scope flags successfully.
- Artifact:
  - `/tmp/k3d_scope_smoke/results4/week14_benchmark_summary.json`

## Important Note

- Full PTX validation was not executed in this session environment because CuPy/PTX is unavailable (`full_ptx_cupy_missing`), so I ran bounded non-PTX smoke checks to validate wiring correctness.

## Recommended Next Step

1. Run controlled A/B in the real PTX environment:
   - A: broad/default scope
   - B: constrained per benchmark scope (new flags)
2. Compare:
   - `elapsed_sec` per benchmark
   - generation failure/oracle metrics
   - query participation coverage
3. If timeouts remain, next move is domain index + PTX query kernel path (as planned), not more penalty tuning.

