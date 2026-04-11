# CODEX Embodied Game Loop Phase 1.1 Completion Report — 2026-04-10

## Scope

This report covers the Phase 1.1 hotfix slice only. Per Claude's workflow, Phase 2 has not started.

Authoritative spec:
- `TEMP/CLAUDE_EMBODIED_GAME_LOOP_PHASE1_1_HOTFIX_AND_PHASE2_SPEC_2026-04-10.md`

## Landed Hotfixes

### 1. GPUEvent bridge ABI alignment

Updated:
- `knowledge3d/cranium/bridges/trm_step_fused_bridge.py`

Changes:
- Removed `_pack_ = 1` from `_GPUEventStruct`
- Kept `ctypes.sizeof(_GPUEventStruct) == 16`
- Added a bridge-side assert that the allocated ring base pointer is 16-byte aligned

Result:
- Bridge-side `GPUEvent` layout now matches the CUDA-side 16-byte event wire format without packed-field pointer arithmetic risk

### 2. `__threadfence()` after `HANDLING_QUERY` pop

Updated:
- `knowledge3d/cranium/ptx/trm_step_fused.cu`

Changes:
- Added `__threadfence()` immediately after `trm_state_pop(...)` and the mirrored `sleep_state` write in the `TRM_STATE_HANDLING_QUERY` branch

Result:
- State visibility after query completion is now explicitly fenced before the kernel exits

### 3. GPU-side host batch enqueue kernel

Updated:
- `knowledge3d/cranium/cuda/gpu_event_queue.cu`
- `knowledge3d/cranium/bridges/trm_step_fused_bridge.py`

Changes:
- Added `gpu_event_queue_enqueue_host_batch(...)`
- Removed the old Python pattern that wrote ring slots directly and bumped `head_ptr` from host code
- `enqueue_event()` now routes through `enqueue_events()`, which:
  - copies a host batch to a temporary device buffer
  - launches the new GPU enqueue kernel
  - reads back `push_results`

Result:
- Host event injection now enters the queue through the same atomic GPU path as future GPU-side producers

### 4. Bridge single-producer guard

Updated:
- `knowledge3d/cranium/bridges/trm_step_fused_bridge.py`

Changes:
- Added `self._gpu_producers_active = False`
- Added `set_gpu_producers_active(...)`
- `enqueue_events()` / `enqueue_event()` now assert if host injection is attempted while GPU producers are marked active

Result:
- Phase 2 cannot silently mix host-side queue injection with GPU-side producers without tripping the bridge contract

### 5. Lifecycle consistency fix discovered during Phase 1.1 verification

Updated:
- `knowledge3d/cranium/cuda/trm_game_loop.cuh`

Changes:
- On IO-driven transition into `HANDLING_QUERY`, `TRM_DEFERRED_QUERY_POP` is now armed in the state machine

Result:
- The standalone lifecycle/state-machine path now correctly returns `HANDLING_QUERY -> previous_state` on the next tick when run without the full fused reasoning kernel
- This was necessary for the existing scripted CUDA lifecycle test to pass under direct execution

### 6. Real-CUDA host batch tests

Added:
- `tests/test_gpu_event_queue_enqueue_host_batch.py`

Coverage:
- event ABI size/alignment contract
- 256-event host batch enqueue lands exactly once
- bridge single-producer guard trips correctly

### 7. Pytest GPU probe repair

Updated:
- `tests/conftest.py`

Changes:
- When `K3D_PYTEST_PROBE_CUDA=1`, pytest can now probe CUDA via the sovereign loader if CuPy is not installed

Result:
- `@pytest.mark.gpu` tests no longer auto-skip solely because the test harness lacks CuPy

### 8. HANDLING_QUERY parity fix

Updated:
- `knowledge3d/cranium/cuda/trm_recursive_core.cuh`
- `knowledge3d/cranium/bridges/trm_step_fused_bridge.py`
- `knowledge3d/cranium/sovereign/trm_launcher.py`
- `knowledge3d/cranium/ptx/trm_step_fused.cu`

Changes:
- Split the query fast-lane cleanly from the autonomous reasoning path
- Removed the remaining `HANDLING_QUERY` awareness side effect so the fast-lane is query-math only
- Restored parity-sensitive kernels to the legacy/default nvcc math contract instead of mixing `--use_fast_math` on one path and strict IEEE flags on another
- Added legacy PTX rebuild enforcement for `trm_extensions.ptx` through `TRMLauncher`
- Made `TRMLauncher.cleanup()` idempotent to stop invalid-argument frees during repeated teardown
- Matched the fused SwiGLU scalar implementation to the legacy PTX formulation:
  - `sig = 1 / (1 + exp(-x)); return x * sig;`
  - not the alternate `x / (1 + exp(-x))` form

Result:
- `HANDLING_QUERY` now skips the reasoning gate and matches the `trm_recursive_fused` oracle exactly
- The legacy PTX backend and fused backend also match again on the existing parity suite

## Verification

### Passed

Python syntax:
- `python3 -m py_compile tests/conftest.py knowledge3d/cranium/bridges/trm_step_fused_bridge.py tests/test_gpu_event_queue_enqueue_host_batch.py tests/test_trm_embodied_tick_phase1.py`

Direct CUDA execution:
- Imported and executed:
  - `tests/test_gpu_event_queue_enqueue_host_batch.py`
  - `tests/test_trm_embodied_tick_phase1.py`
- Result: all direct CUDA checks passed

Targeted pytest:
- `python3 -m pytest -q tests/test_trm_weight_persistence.py::test_phase_d_trm_shadow_probe_returns_expected_diagnostics`
- Result: passed

Additional direct runtime verification:
- ring alignment assert passed
- 256 host-batch events enqueued and drained exactly once
- bridge single-producer guard passed

Real-CUDA parity suite:
- `K3D_PYTEST_PROBE_CUDA=1 CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q tests/test_trm_fused_parity.py`
- Result: `3 passed`

Real-CUDA targeted Phase 1.1 batch:
- `K3D_PYTEST_PROBE_CUDA=1 CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q tests/test_trm_fused_parity.py tests/test_gpu_event_queue_enqueue_host_batch.py tests/test_trm_embodied_tick_phase1.py tests/test_trm_weight_persistence.py::test_phase_d_trm_shadow_probe_returns_expected_diagnostics`
- Result: `10 passed`

### Current blocker

The parity blocker is fixed, but the benchmark pin is not green on the authoritative GPU env.

GPU env used:
- `bash scripts/k3d_env.sh run -e k3d-cranium ...`
- Python: `/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python`
- CuPy: `13.6.0`

Math pin rerun:
- `bash scripts/k3d_env.sh run -e k3d-cranium python -m pytest -q tests/test_gpu_math_query.py::test_math_first_twenty_problems_stay_green_on_gpu_path`
- Result: **FAIL**
- Observed: `summary["correct"] == 1`, expected `20`

ARC pin rerun:
- Executed `ARCAGI2Benchmark(... max_tasks=10, query_scope_galaxies='Drawing,Grammar,3DObjects,Math,Reality').run_benchmark(use_enriched=True)` under `k3d-cranium`
- Dataset path resolved to: `/K3D/Knowledge3D.local/datasets/exams/arc-src/data/evaluation`
- Result: **FAIL**
- Observed: `0 / 10`, accuracy `0.0`

Interpretation:
- Phase 1.1 hotfixes are landed
- Phase 1.1 parity is landed
- Phase 2 should still remain blocked because the benchmark gate required by the spec is currently red in the live GPU env
- The next investigation is no longer embodied tick parity; it is benchmark-truth drift versus the `CODEX.md` claim of `ARC 10/10` and `Math 20/20`

## Files Changed In Phase 1.1

- `knowledge3d/cranium/bridges/trm_step_fused_bridge.py`
- `knowledge3d/cranium/cuda/gpu_event_queue.cu`
- `knowledge3d/cranium/cuda/trm_game_loop.cuh`
- `knowledge3d/cranium/cuda/trm_recursive_core.cuh`
- `knowledge3d/cranium/ptx/trm_step_fused.cu`
- `knowledge3d/cranium/sovereign/trm_launcher.py`
- `knowledge3d/cranium/kernels/ptx_compiler.py`
- `tests/conftest.py`
- `tests/test_gpu_event_queue_enqueue_host_batch.py`

## Recommendation

Do not start Phase 2 yet.

Next action should be a focused benchmark-truth investigation for:
- `tests/test_gpu_math_query.py::test_math_first_twenty_problems_stay_green_on_gpu_path`
- ARC curated `10/10` rerun path in the live `k3d-cranium` env

The embodied tick parity repair is complete. The remaining Phase 1.1 blocker is benchmark regression or benchmark-baseline drift.
