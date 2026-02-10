# Week 21.8 PTX Profiling Findings (Priority 0)

## Scope
Added opt-in PTX micro-profiling to `ARCPTXOps` and ran a profiled benchmark pass.

- Code: `knowledge3d/cranium/ptx/arc_ops.py`
- Toggle: `K3D_PTX_PROFILE=1`
- Run: `../Knowledge3D.local/results/week21_8_ptx_profile5/run.log`

## What was validated
PTX kernels are actually executing in live ARC path.

Observed profile labels:
- `weighted_score_kernel`
- `argmax_kernel`
- `extract_pattern_features_kernel`
- `discovery_score_kernel`
- `compare_grids_kernel`
- `check_oracle_fuzzy_ptx.total`
- `discover_patterns_ptx.total`
- `rank_candidates_ternary.total`

## Timing summary
From `run.log`:

- `weighted_score_kernel`: calls=12, mean=2.205ms (warmup-heavy)
  - excluding first JIT-heavy call: mean=0.046ms (11 calls)
- `argmax_kernel`: calls=12, mean=0.031ms
- `extract_pattern_features_kernel`: calls=10, mean=0.076ms
- `discovery_score_kernel`: calls=10, mean=0.031ms
- `compare_grids_kernel`: calls=10, mean=0.069ms
- `discover_patterns_ptx.total`: calls=10, mean=1.335ms
- `check_oracle_fuzzy_ptx.total`: calls=10, mean=0.413ms
- `rank_candidates_ternary.total`: calls=12, mean=4.134ms
  - excluding first JIT-heavy call: mean=0.770ms (11 calls)

## Interpretation
1. PTX execution is real (not fake flags).
2. Kernel launches are very small (`n` mostly 1-2 for ranking, ~10 for discovery), so duty cycle is tiny.
3. This explains low average GPU utilization despite PTX path being active.
4. In `penalty` mode, validity kernels are largely bypassed by design (score penalties preferred over hard filter), reducing GPU workload further.

## Immediate recommendation
To raise GPU utilization and throughput:
1. Batch candidates/patterns before ranking/discovery so kernels operate on larger `n`.
2. Fuse/port penalty component calculations to device-side batched kernels.
3. Keep PTX profiling enabled in short diagnostics (`K3D_PTX_PROFILE=1`) for regression checks.

