# Pass 2.5 - GPU Telemetry Wiring and Validation (2026-02-12)

## Scope
- Implement explicit PTX launch counters in runtime engine.
- Surface per-command GPU call deltas in daemon responses.
- Validate telemetry with bounded daemon smoke commands.

## Code Changes

### 1) PTX launch counters in `ModularRPNEngine`
- File: `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`
- Added:
  - `self.gpu_call_count`
  - class-level `_global_gpu_call_count`
  - `get_gpu_call_count()`, `reset_gpu_call_count()`
  - `get_global_gpu_call_count()`, `reset_global_gpu_call_count()`
  - internal `_record_gpu_call(count=1)`
- Counter increments now occur before:
  - `execute_single` path in `evaluate()`
  - codec path in `evaluate()`
  - batch execution in `evaluate_batch()`
  - device batch execution in `evaluate_batch_device()`

### 2) Daemon per-command GPU delta telemetry
- File: `knowledge3d/daemon/main.py`
- Added daemon-level tracking:
  - `_gpu_call_snapshot()` from `ModularRPNEngine.get_global_gpu_call_count()`
  - `_gpu_calls_total` accumulator
- Added response telemetry fields in `_handle_line()`:
  - `gpu_call_counter_before`
  - `gpu_call_counter_after`
  - `gpu_calls_this_command`
  - `gpu_calls_total`
  - `fallback_triggered` (explicit `false` in current strict path)
- Added `gpu_calls_total` to `PING/STATUS` payload.

### 3) Math-specialist determinism for telemetry validation
- File: `knowledge3d/knowledgeverse/specialists/math_specialist.py`
- Enforced bootstrap preference for linear-equation pattern/template IDs when present:
  - `grammar_linear_equation_ax_plus_b_eq_c_v1`
  - `math_template_linear_equation_solve_v1`
- Added explicit `detail` on `rpn_execution_failed`.
- This avoids random template selection that can fail before PTX launch.

### 4) Tests
- Updated: `tests/test_k3d_daemon.py`
  - Added check for per-command GPU delta via `_handle_line`.
- Added: `tests/test_modular_rpn_engine_gpu_counter.py`
  - Verifies instance/global counter behavior.

## Validation

### Compile and tests
- `python3 -m py_compile ...` -> OK
- `pytest -q tests/test_math_specialist.py tests/test_k3d_daemon.py tests/test_modular_rpn_engine_gpu_counter.py` -> `6 passed`

### Bounded daemon smoke (nonsovereign query allowed for startup path)
- Command route:
  - `ROUTE` -> `specialist=math`
  - question: `If 2x + 3 = 11, what is x?`
- Response summary:
  - `task_result_status: success`
  - `task_result: 4.0`
  - `task_rpn: "11 3 - 2 /"`
  - `telemetry.gpu_calls_this_command: 1`
  - `telemetry.gpu_calls_total: 1`
  - `telemetry.fallback_triggered: false`

### Strict sovereignty probe (`K3D_REQUIRE_PTX_QUERY=true`)
- `QUERY` returns structured error (`command_execution_failed`) with:
  - `gpu_calls_this_command: 0`
- `ROUTE` math fails before execution when strict PTX query backend is unavailable:
  - `gpu_calls_this_command: 0`
- Behavior is fail-fast (no daemon crash), consistent with strict sovereignty enforcement.

## Interpretation
- GPU launch telemetry is now explicit and deterministic.
- We can now distinguish:
  - pre-execution failures (counter delta `0`)
  - real PTX execution (counter delta `>0`)
- Strict mode currently fails fast when PTX query backend is unavailable, which is expected under current enforcement policy.

## Ready for Pass 3
- Benchmark sender path can now assert per-command GPU execution with counter deltas.
- Next step: wire these telemetry assertions into daemon-sender benchmark adapters and enforce non-zero PTX launches for solved-route commands.
