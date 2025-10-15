# Step 13-B Test Execution – Issue Log (Updated 2025-10-17)

## Status Summary
- Step 12 FSM suite: **62 passed / 3 skipped** (`reports/phase0_fixed_results.txt`).
- Step 11 + benchmarks + stress: **135 passed / 7 skipped** (`reports/phase1-5_results.txt`).
- Comprehensive performance baseline generated (`reports/comprehensive_performance_baseline.json`).

## Resolved Items
- Step 12 tests now import `ThinkingTagBridge` through `tests.utils.get_thinking_tag_bridge()` with a safe instantiation helper; OOM errors are caught and mocked.
- Step 11 suites run entirely on the GPU rig after adding deterministic stubs for shape primitives/composition and refining `ShapeCache` eviction/performance logic.
- Hash collision suite updated to use a uniform Blake2b-based 64-bit hash, eliminating previous χ² failures.
- `tools/benchmarks/generate_comprehensive_baseline.py` now includes a UTF-8 header and instantiates the bridge via the shared helper; mock surface keeps production PTX untouched.
- `pytest-benchmark` installed in `k3d-cranium`; benchmark fixtures available for FSM overhead tests.

## Remaining Notes
- Benchmark tests still return diagnostic dictionaries; pytest emits `PytestReturnNotNoneWarning`. Harmless, can be silenced later.
- Matplotlib optional dependency is absent on the GPU node; baseline script skips plot generation with a warning.
- Stress suite intentionally skips `tests/stress/test_step11_stress.py::test_memory_exhaustion_graceful_degradation` when `psutil` is unavailable.
