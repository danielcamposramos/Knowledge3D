# Codex Workflow Review + Strict PTX Enforcement (Feb 12, 2026)

## Scope
- Per Daniel directive, no benchmark execution before full workflow review.
- Reviewed end-to-end path:
  - `scripts/run_all_benchmarks.py`
  - `benchmarks/arc_agi_2_adapter.py`
  - `benchmarks/math_competitions.py`
  - `knowledge3d/knowledgeverse/trm_navigator.py`
  - `knowledge3d/knowledgeverse/specialist_router.py`
  - `knowledge3d/knowledgeverse/galaxy_manager.py`
  - `knowledge3d/cranium/ptx/arc_ops.py`

## Findings (Root Drift Points)
1. Query layer still had a CPU O(n) scanning path in `GalaxyManager` when `K3D_REQUIRE_PTX_QUERY=false`.
2. `ARCPTXOps` still contained internal CPU fallback behavior:
   - GPU argsort fallback to NumPy.
   - CPU passthrough path in validity gates.
   - CPU fallback in fuzzy oracle when PTX unavailable.
   - Ungated fallback when filtering produced zero survivors.
3. Runner scripts enforced ARC PTX mode, but did not force `K3D_REQUIRE_PTX_QUERY=true` in full PTX runs.

## Changes Applied
### 1) GPU Query Path Added to GalaxyManager
- File: `knowledge3d/knowledgeverse/galaxy_manager.py`
- Added `_query_ptx_implementation(...)` and routed sovereign query mode to it.
- Added hashed-vector query caching:
  - `self._entry_vector_cache`
  - `self._specialist_vector_cache`
- Query scoring now uses GPU matvec (`matrix_gpu.dot(query_gpu)`).
- CPU string scan path remains available only under explicit non-sovereign mode (`K3D_REQUIRE_PTX_QUERY=false`).

### 2) ARC PTX Ops Hardened (No CPU Fallbacks)
- File: `knowledge3d/cranium/ptx/arc_ops.py`
- Enforced hard failure on unavailable PTX:
  - `raise RuntimeError("arc_ptx_unavailable")`
- Removed argsort fallback behavior:
  - now raises `RuntimeError("ptx_argsort_failed")` instead of dropping to NumPy sort.
- Validity gate no longer silently falls back to ungated candidates when zero survive.
- Oracle path no longer falls back to CPU when PTX unavailable.

### 3) Full PTX Run Forces PTX Query Requirement
- Files:
  - `scripts/run_all_benchmarks.py`
  - `scripts/run_all_global_benchmarks.py`
- In `--arc-enable-full-ptx` mode, runner now sets:
  - `os.environ["K3D_REQUIRE_PTX_QUERY"] = "true"`

### 4) Sovereignty Test Coverage Expanded
- File: `tests/test_hot_path_sovereignty.py`
- Added assertions for:
  - strict ARC PTX fallback blocking behavior
  - runner enforcement of `K3D_REQUIRE_PTX_QUERY=true` in full PTX mode

## Verification (Static Only)
- `python3 -m py_compile` passed on modified files.
- `pytest -q tests/test_hot_path_sovereignty.py` passed (`4 passed`).
- No benchmark workload was run in this pass.

## Current State
- Math solve path in `TRMNavigator._solve_math` is Galaxy-first and no regex/eval fallback path is present.
- Query/ranking/oracle sovereignty enforcement is now strict under full PTX runs.
- If PTX query/kernel path is unavailable, system now fails fast instead of silently drifting to CPU fallback.

## Next Step (Recommended)
- Execute a bounded PTX smoke run only after explicit go-ahead:
  - validate that failures are now explicit sovereignty failures (if any),
  - then fill missing PTX/query/template components rather than reintroducing Python fallbacks.
