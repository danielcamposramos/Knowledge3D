# Step 13-E — RPN Expansion & Swarm Foundations

**Date:** 2025-10-16  
**Owner:** Codex (GPU PTX lane)  
**Status:** Feature-complete / tests in place

---

## Capability Outcomes
- Added GPU-temporal ops (`OP_TEMPORAL_COHERENCE`, `OP_TEMPORAL_MASK`, `OP_TEMPORAL_AGGREGATE`) to `modular_rpn_kernel_extended.cu`. Variance, activity, and sigmoid mask now run fully on the device.
- Replaced Tier-2 matvec with a tiled/shared-memory path; helper bridge exposes `_test_matvec()` for direct profiling.
- Introduced swarm-matrix surfaces:
  - `OP_MATMUL_SMALL` for fusing per-chain states (M×K · K×N).
  - `OP_DOT_BATCH` for resonance/consensus scoring.
  - `OP_TRACE_TENSOR` for quick diagnostics.
- Implemented basic programmability: `OP_STORE`, `OP_RECALL`, `OP_LOOP`, `OP_NEXT`, `OP_BRANCH` (offset recording only; PC rewrite stays earmarked for Step 14).
- Bridge helpers (`ThinkingTagRPNBridge`) now surface `_execute_rpn_program`, `_test_matmul_small`, `_test_dot_batch`, `_test_matvec` for harness + pytest.

## Test & Benchmark Surface
- New GPU-marked tests:
  - `tests/test_step13e_temporal_kernels.py`
  - `tests/test_step13e_matrix_ops.py`
  - `tests/test_step13e_programmability.py`
  - `tests/test_step13e_integration.py`
  - `tests/benchmarks/test_step13e_performance.py`
- Pytest run (GPU markers honoured) verifies shape invariants and NumPy parity.
- Benchmarks emit assertions for `<50µs` matvec and `<0.20ms` fuse budget.  
  _Note:_ Runtime measurements need the conda/tmux GPU lane (`CUDA_VISIBLE_DEVICES=0`) enabled; tiny smoke run in this workspace (CPU fallback) reported ~311µs matvec / 0.70ms FUSE and should be ignored.

## Docs & Ops
- `docs/ENV_POLICY.md` now calls out `CUDA_VISIBLE_DEVICES=0` before tmux attach so the RTX 3070 is visible (KDE session defaults to the iGPU).
- `TEMP/STEP13E_TEMPORAL_ANALYSIS.md` documents pre-PTX temporal behaviour + required port.

## Step 14 Readiness
- Inter-chain math primitives operational; programmability scaffolding (variable store + loop control) live.
- Next action items:
  1. Implement true PC rewrites for `OP_BRANCH/OP_LOOP` (jump tables or counter-based rewinds).
  2. Layer pheromone / resonance accumulation using the new matrix ops.
  3. Profile on the production GPU stack to capture <95µs aggregate latency evidence.

## Data / Metrics to Refresh
- Re-run the benchmark suite inside `k3d-cranium` with GPU access to capture authoritative numbers for the swarm ledger.
- Sync resulting metrics into the Phase 1C performance log once available.
