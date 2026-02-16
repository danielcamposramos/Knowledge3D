# Codex Pass 3.5-3.8: Runner-Level Sovereignty Enforcement (2026-02-12)

## Scope Completed
- Enforced sovereignty checks directly in benchmark runner scripts (not only sender scripts).
- Added runner-level fail-fast behavior for solved tasks that lack GPU evidence.
- Added explicit sovereignty summaries into runner output payloads and console reports.
- Added debug override flag (`--no-enforce-sovereignty`) for controlled diagnostics.

## Code Changes

### 1) `scripts/run_all_benchmarks.py`
- Added `SovereigntyViolation` exception.
- Added per-task sovereignty extraction helpers:
  - GPU evidence extraction (`gpu_calls_this_command` and ARC PTX row flags).
  - Fallback signal extraction (`fallback_triggered`, `ptx_ranking_error`).
- Added benchmark-phase sovereignty summaries for:
  - `arc_agi_2.empty_mind`, `arc_agi_2.enriched`
  - `math_competitions.empty_mind`, `math_competitions.enriched`
  - `last_humanity_exam.empty_mind`, `last_humanity_exam.enriched`
  - `mmlu.empty_mind`, `mmlu.enriched`
- Added runtime summary block at `runtime_usage.sovereignty` with:
  - solved tasks, GPU-used tasks, no-GPU tasks, missing telemetry count, fallback count, compliance ratio.
- Added fail-fast when enforced and violations exist.
- Added CLI flags:
  - `--enforce-sovereignty` (default)
  - `--no-enforce-sovereignty`

### 2) `scripts/run_all_global_benchmarks.py`
- Added same enforcement model (`SovereigntyViolation`, helper extraction, fail-fast, summary print).
- Added same CLI flags (`--enforce-sovereignty` / `--no-enforce-sovereignty`).
- Added safe skip semantics for integrated suite limits (`max <= 0`), with skipped payloads and metrics.
  - This prevents accidental ARC construction/failure during zero-task smoke runs.

### 3) `tests/test_hot_path_sovereignty.py`
- Added runner contract test:
  - `test_runner_runtime_gpu_enforcement_is_present`
  - Verifies both runner scripts include sovereignty exception type, debug override flag, and sovereignty summary wiring.

## Validation Performed

### Static
- `python3 -m py_compile scripts/run_all_benchmarks.py scripts/run_all_global_benchmarks.py`
  - Pass.

### Tests
- `pytest -q tests/test_hot_path_sovereignty.py`
  - Pass (`6 passed`).

### Runtime Smokes (k3d-cranium)
- `run_all_benchmarks.py` with all max limits = 0
  - Pass.
  - Console includes `Sovereignty summary` (100% compliance on 0 solved tasks).
- `run_all_global_benchmarks.py` with all integrated max limits = 0
  - Pass.
  - Confirms new global skip semantics + sovereignty summary path.

## Notes
- Enforcement is intentionally strict: solved tasks without GPU evidence are violations.
- For benchmarks that do not yet emit per-task GPU telemetry, use `--no-enforce-sovereignty` temporarily only for diagnostics, then wire telemetry to close the gap.
- This completes the requested runner-layer sovereignty gate so regressions cannot hide behind script-level execution.
