# Codex Week 21.4 Full PTX Progress (02.09.2026)

## Scope completed
- Wired **full PTX mode** through ARC stack:
  - `benchmarks/arc_agi_2_adapter.py`
  - `benchmarks/arc_agi_2.py`
  - `scripts/run_all_benchmarks.py`
  - `scripts/run_all_global_benchmarks.py`
- Extended PTX ops in `knowledge3d/cranium/ptx/arc_ops.py` and normalized fuzzy key naming (`oracle_fuzzy_0_80`, etc.).
- Added tests to validate full PTX routing paths in `tests/test_arc_agi_2_adapter.py`.

## New controls
- `--arc-enable-full-ptx`
- `--arc-ptx-validity-strictness {strict,medium,relaxed}`

## Runtime wiring delivered
- Discovery can run through `ARC_PTX_OPS.discover_patterns_ptx(...)`.
- Validity gates can run through `ARC_PTX_OPS.apply_validity_gates_relaxed_ptx(...)`.
- Oracle checks can run through `ARC_PTX_OPS.check_oracle_fuzzy_ptx(...)`.
- ARC result/diagnostics now emit:
  - `ptx_full_enabled`, `ptx_full_available`, `ptx_full_used`
  - `ptx_oracle_used`, `ptx_validity_mode`, `ptx_validity_strictness`
- Aggregated ARC diagnostics now include:
  - `ptx_full_enabled_rate`, `ptx_full_used_rate`, `ptx_oracle_used_rate`
  - `oracle_failure_mode_counts`

## Validation (code)
- Compile check:
  - `python -m py_compile` on modified files: ✅
- Tests:
  - `pytest -q tests/test_arc_agi_2_adapter.py tests/test_benchmarks.py`: ✅ 15 passed

## Validation (runtime)
### Full-PTX adapter probe (1 task)
- `ptx_full_enabled`: true
- `ptx_full_used`: true
- `ptx_validity_mode`: `ptx_validity`
- `ptx_oracle_used`: true

### ARC full-PTX strictness sweep (20 tasks)
Artifact folder:
- `../Knowledge3D.local/results/week21_4_full_ptx_arc_only_sweep/`
- Main summary:
  - `../Knowledge3D.local/results/week21_4_full_ptx_arc_only_sweep/arc_full_ptx_sweep_summary.json`

Results:
- `strict`
  - accuracy: **0.25**
  - oracle_at_all: **0.0**
  - fuzzy_oracle_at_all: **0.05**
  - validity_reject_rate_mean: **0.7333**
  - ptx_full_used_rate: **1.0**
- `medium`
  - accuracy: **0.25**
  - oracle_at_all: **0.0**
  - fuzzy_oracle_at_all: **0.05**
  - validity_reject_rate_mean: **0.3417**
  - ptx_full_used_rate: **1.0**
- `relaxed`
  - accuracy: **0.25**
  - oracle_at_all: **0.0**
  - fuzzy_oracle_at_all: **0.05**
  - validity_reject_rate_mean: **0.1167**
  - ptx_full_used_rate: **1.0**

## Interpretation
- **Execution sovereignty milestone achieved** for ARC operations path:
  - Full PTX hooks are active and used end-to-end in benchmark runtime.
- **Current bottleneck remains candidate correctness/matching**, not PTX activation:
  - oracle remains 0.0 even with full PTX and relaxed gating.
  - fuzzy and failure-mode diagnostics confirm near-miss signal exists but exact-match remains blocked.

## Recommended next pass
1. Tighten train-family inference quality (family label quality still weak under relaxed gates).
2. Improve generated candidate fidelity before ranking (especially autonomous + contrastive transforms).
3. Add PTX-assisted train-pair consistency scoring into candidate generation, not only post-generation gating.
4. Keep `medium` as default strictness for now (balanced reject rate).
