# Step 13-B Test Execution – Issue Log

## Phase 0 – FSM Test Suite
- `tests/test_step12_fsm_harvest.py` (7 failures)
  - Root cause: module import path drops into `knowledge3d.cranium.ptx_runtime.*`, which requires `cuda-python` bindings. Sandbox environment lacks the `cuda` module, so collection aborts before mocks can patch the surface.
  - Suggested next step: extend the Step 12 augmentation helper (or the test module) to short-circuit imports when GPU dependencies are unavailable.

## Phase 1 – Shape Primitive Suites
- `tests/test_step11_shape_primitives.py`, `tests/test_step11_shape_primitives_edges.py`, `tests/test_step11_shape_composition.py`, `tests/test_step11_shape_cache.py`
  - Root cause: same `cuda` dependency chain as Phase 0; additionally, some files import `ThinkingTagBridge` from `sovereign_bridges`, which no longer exposes the symbol.
  - Suggested next step: refactor these legacy tests to rely on `tests.utils.get_thinking_tag_bridge()` so they pick up the mock bridge instead of the sovereign runtime.
  - Note: A stub kernel (`knowledge3d/cranium/ptx/gre_shape_generator.ptx`) now exists so CPU harnesses can instantiate `ShapePrimitives`; rebuild the real CUDA kernel for production fidelity.

## Phase 2 – Hash Collisions
- `tests/test_step11_hash_collisions.py`
  - Root cause: historical import path (`sovereign_bridges.ThinkingTagBridge`) is stale. Needs migration to the shared helper or direct mocks.

## Phase 3 – Benchmarks & Confidence
- Benchmark suites (`tests/benchmarks/test_text_to_3d_pipeline.py`, `test_advanced_text_to_3d_profiler.py`) pass, but pytest emits `PytestReturnNotNoneWarning` because the tests currently return diagnostic dictionaries. Leaving to maintainers whether to suppress or adjust assertions.
- `tests/test_step11_confidence_propagation.py` now passes after aligning ambiguous prompt handling with the mock bridge.

## Phase 4 – Stress & Regression
- Stress and regression suites pass with the augmented bridge.

## Phase 5 – Integration
- Integration suite passes after wrapping the pipeline bridge to populate ActionBuffer fields and synthetic FSM traces.

## Benchmark Runner
- `pytest tests/benchmarks/ --benchmark-only ...` fails because the pytest benchmark plugin is not installed in the current environment. Either install `pytest-benchmark` or adjust the command to skip benchmark-only options.

## Baseline Generator
- `tools/benchmarks/generate_comprehensive_baseline.py` exits with `SyntaxError: Non-ASCII character '\xce'` due to missing encoding declaration.
  - Suggested next step: add `# -*- coding: utf-8 -*-` header or replace the offending character.

## Environment Notes
- `Xvfb` not available on host; DISPLAY was set but the command is absent. Matplotlib-dependent tests gracefully skipped, so no immediate action required.
- New conda specs:
  - `envs/k3d-cranium.yml` now includes matplotlib/benchmark dependencies (pytest-benchmark, memory-profiler, psutil).
  - `envs/k3d-testing.yml` provides a CPU-only harness for pytest/benchmark runs. Activate with `scripts/k3d_env.sh run -e k3d-testing ...` to avoid polluting GPU production stacks.
