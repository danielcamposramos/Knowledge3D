# Week 22 Math Debug Audit (Workflow + Root Cause)

Date: 2026-02-12  
Author: Codex (implementer)

## 1) Full workflow review (what actually runs)

### A. Benchmark path (`scripts/run_all_benchmarks.py` -> `benchmarks/math_competitions.py`)
- `MathCompetitionBenchmark._solve_problem()` uses:
  - `TRMNavigator.navigate_and_compose()`
  - `TRMNavigator.execute()` for `program_type="math_expression"`
- **Before this patch**, `TRMNavigator.execute()` called `TRMNavigator._solve_math()` legacy template selection path.
- **After this patch**, `TRMNavigator._solve_math()` delegates to `MathSpecialist.process()` (same specialist composer architecture as daemon).

### B. Daemon path (`knowledge3d/daemon/main.py`)
- `ROUTE` with `specialist=math` already used `MathSpecialist.process()`.
- This path was already sovereign and had GPU telemetry proof from Pass 2.5/3.

## 2) Changes applied in this session

### `knowledge3d/knowledgeverse/trm_navigator.py`
- Added debug flag `K3D_MATH_DEBUG`.
- Added `self._last_math_missing_signal` and `self._last_math_execution_error`.
- Added `get_math_specialist()` lazy resolver (same pattern used for chat/input primer).
- Replaced math solve body to:
  - call `MathSpecialist.process(...)`
  - return result on `status=success`
  - otherwise record explicit missing signal with reason/detail (no fallback).
- Added `get_last_math_missing_signal()`.
- Added structured debug prints for math solve events when `K3D_MATH_DEBUG=1`.

### `benchmarks/math_competitions.py`
- Added per-task trace reset (`navigator.clear_trace()`) to avoid trace bleed between tasks.
- Added failure reason extraction and propagation:
  - `failure_reason`
  - `failure_signal`
- Added benchmark diagnostics aggregation:
  - `failure_reason_counts`
- Kept sovereign path and existing benchmark shape intact.

### `knowledge3d/knowledgeverse/specialists/math_specialist.py`
- Added opt-in debug telemetry (`K3D_MATH_DEBUG=1`) for:
  - solve_start
  - query candidate counts
  - coefficient extraction result
  - RPN composition
  - solve_error / solve_success
- No fallback logic added.

## 3) Validation runs and results

## Run 1 (before delegation fix, with debug)
- Command: `run_all_benchmarks.py` with math=10, others=0, `K3D_MATH_DEBUG=1`
- Outcome:
  - accuracy: 0.0
  - failure reason (dominant): `rpn_execution_failed`
  - concrete errors:
    - `ValueError: Unknown token: A`
    - `ValueError: Unknown token: det(`
  - indicates non-executable template strings were selected in legacy TRM solve path.

## Run 2 (after delegation fix, with debug)
- Command: same as above, output dir `week22_math_debug10_v2_02.12`
- Outcome:
  - accuracy: 0.0
  - diagnostics:
    - `predicted_none_count=10`
    - `predicted_none_rate=1.0`
    - `failure_reason_counts={"coefficient_extraction_failed": 10}`
  - this is a **clean capability signal**:
    - RPN execution path no longer failing on invalid token templates
    - primary blocker is equation parsing/extraction coverage for AMC-style problems.

## Direct sanity check
- Prompt: `If 2x + 3 = 11, what is x?`
- `TRMNavigator.execute()` returned `4.0`
- trace included: `math_solve_success rpn=11 3 - 2 /`

This confirms specialist composition and PTX execution work for supported pattern class.

## 4) TRM state snapshot (from enriched storage)

- `TRMNavigator` loaded successfully.
- specialist tree count (include root): `25`
- root children: `GrammarSpecialist`, `MathSpecialist`, `PhysicsSpecialist`, `VisualSpecialist`
- routing state file exists:
  - `../Knowledge3D.local/galaxies_enriched/checkpoints/trm_routing_state.json`
  - routing signatures: `1195`
- spawner state file exists:
  - `../Knowledge3D.local/galaxies_enriched/checkpoints/trm_specialist_spawner.json`
  - parent buckets: `4`
  - spawn decisions: `12`
- specialist tree checkpoint exists:
  - `../Knowledge3D.local/galaxies_enriched/checkpoints/trm_specialist_tree.json`

Interpretation:
- TRM is not a single dense neural checkpoint in this path; it is a specialist routing/state system with persisted topology and spawn state.
- Current failure is not “TRM absent”, but **math extraction/template coverage**.

## 5) Root cause (current)

After forcing benchmark path onto specialist architecture, the top blocker is:

- `coefficient_extraction_failed` for AMC/AIME natural-language questions.

So the gap is now explicit and aligned with your mandate:
- no hidden fallback,
- no decorative query path,
- clear signal for what must be added.

## 6) Recommended next implementation (small, high-yield)

1. Expand `MathSpecialist` extraction coverage (still sovereign):
- arithmetic-only question extraction (`a + b`, `a*b`, etc.)
- linear forms with verbal wrappers and variable variants
- simple substitution templates.

2. Add minimal template pack in Math/Grammar bootstrap:
- arithmetic add/sub/mul/div
- linear solve variants (`ax+b=c`, `x+a=b`, `a+x=b`)
- one-step substitution.

3. Re-run:
- `max-math-problems 100` with `K3D_MATH_DEBUG=1`
- expect `failure_reason_counts` to shift away from `coefficient_extraction_failed`.

---

This session fixed the key workflow drift (benchmark path now uses specialist composer) and converted a noisy failure (`unknown token`) into an actionable sovereign signal (`coefficient_extraction_failed`).
