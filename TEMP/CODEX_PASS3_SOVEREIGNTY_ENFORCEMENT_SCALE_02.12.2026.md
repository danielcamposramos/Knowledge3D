# Pass 3 - Sovereignty Enforcement at Scale (2026-02-12)

## Objective
Enforce and validate runtime sovereignty at scale:
- sender-level hard checks for GPU usage on solved commands
- daemon lifecycle stability under 100-task load
- CI coverage to prevent GPU-enforcement regression

## Implemented Changes

### 1) Sender-level sovereignty assertions
- File: `benchmarks/daemon_client.py`
  - Added `SovereigntyViolation` exception.
  - Added `assert_gpu_for_solved_command(...)`.
  - Added `send_command(...)` alias for compatibility.

- File: `benchmarks/math_sender.py`
- File: `benchmarks/arc_sender.py`
- File: `benchmarks/lhe_sender.py`
- File: `benchmarks/mmlu_sender.py`
  - Added strict runtime check:
    - when `status=ok` and `task_result.status=success`
    - require `telemetry.gpu_calls_this_command > 0`
    - else raise `SovereigntyViolation`
  - Added `--allow-zero-gpu` flag for explicit debug override.

### 2) Scale stability test (100 tasks)
- File: `tests/test_daemon_stability_100_tasks.py`
  - Starts daemon once (TCP mode), sends 100 math tasks, and asserts:
    - daemon remains alive throughout loop
    - each solved task has `gpu_calls_this_command > 0`
    - `fallback_triggered == false`
    - aggregate `gpu_calls_total >= 100`
  - Graceful shutdown verified.

### 3) Sovereignty CI extension
- File: `tests/test_hot_path_sovereignty.py`
  - Added `test_sender_runtime_gpu_enforcement_is_present`
  - Verifies all sender scripts contain runtime GPU-enforcement hook.

## Validation Results

### Compile checks
`python3 -m py_compile` on modified sender/tests passed.

### Test suite
Command:
```bash
pytest -q tests/test_daemon_stability_100_tasks.py \
          tests/test_hot_path_sovereignty.py \
          tests/test_math_specialist.py \
          tests/test_k3d_daemon.py \
          tests/test_modular_rpn_engine_gpu_counter.py
```
Result:
- `12 passed`

### Live bounded sender smoke
Run:
- start daemon once (TCP, `--allow-nonsovereign-query`)
- run `benchmarks/math_sender.py`
- query daemon `STATUS`
- shutdown

Observed:
- sender summary: `ok=1`, `failed=0`
- daemon stayed alive
- `STATUS` reported cumulative `gpu_calls_total=1`
- `fallback_triggered=false` in telemetry

## Notes
- `PING`/`STATUS`/`SHUTDOWN` naturally report `gpu_calls_this_command=0`; sovereignty check is applied only to solved task commands.
- This pass enforces regression safety: solved-task paths now fail fast if they stop launching PTX kernels.
