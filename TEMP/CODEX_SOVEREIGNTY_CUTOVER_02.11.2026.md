# Codex Sovereignty Cutover (Phase 1A + 1B)

Date: 2026-02-12

## Objective
Apply strict sovereignty cutover: remove CPU/legacy fallback behavior from hot path and enforce Galaxy/PTX-only execution (or explicit fail-fast).

## Changes Applied

### 1) ARC adapter: PTX-only mandatory, legacy removed
File: `benchmarks/arc_agi_2_adapter.py`

- Enforced PTX-only initialization:
  - raises if full PTX unavailable
  - raises if PTX ranking unavailable
- Removed legacy pipeline execution path from `solve_task(...)`.
- Removed fallback solver usage from `solve_task(...)` (now explicit RuntimeError if fallback provided).
- Removed `_fallback_or_raise(...)` fallback method.
- Removed CPU ranking fallback from `_score_and_sort_candidates(...)`.
- Removed CPU validity fallback from `_apply_validity_gates(...)`.
- Removed CPU oracle fallback from `_compute_oracle_metrics(...)`.

Evidence strings present:
- `PTX-only ARC path is mandatory...`
- `PTX ranking is mandatory for ARC hot path...`
- `Fallback solver is forbidden in PTX-only ARC mode...`
- `PTX ranking required but unavailable; CPU ranking fallback is disabled.`
- `PTX validity gates required but unavailable; CPU validity fallback is disabled.`
- `PTX oracle required but unavailable; CPU oracle fallback is disabled.`

### 2) Query path: CPU linear scan disabled by default (fail-fast)
File: `knowledge3d/knowledgeverse/galaxy_manager.py`

- Added strict sovereignty gate:
  - `self.require_ptx_query = _env_true("K3D_REQUIRE_PTX_QUERY", "true")`
- In query implementation:
  - raises `NotImplementedError` when strict mode is on
  - message explicitly states CPU O(n) scan is disabled and PTX query kernel is required.

### 3) TRM query fallback loop removed
File: `knowledge3d/knowledgeverse/trm_navigator.py`

- Deleted local Python scoring loop in `query(...)`.
- `TRMNavigator.query(...)` now always delegates to `GalaxyManager.query(...)` with scoped galaxies.

### 4) Regex-based routing cue removed
File: `knowledge3d/knowledgeverse/specialist_router.py`

- Removed `re.search`/`re.split` based domain inference.
- Replaced with character/token scanning helpers:
  - `_tokenize_query(...)`
  - `_has_math_symbol_cue(...)`

### 5) LHE eval removed
File: `benchmarks/last_humanity_exam.py`

- Removed `eval(...)` path in enriched math reasoning.
- `_enriched_reasoning(...)` now uses TRM navigation + execution for math domain:
  - `navigate_and_compose(...)` then `navigator.execute(...)`.

### 6) Sovereignty CI test added
File: `tests/test_hot_path_sovereignty.py`

- Added assertions that hot-path files do not contain:
  - `re.search(`
  - `re.match(`
  - `ast.parse(`
  - `eval(`
- Added assertion that `GalaxyManager` defaults to `K3D_REQUIRE_PTX_QUERY=true` and fail-fast message exists.

## Static Verification

### Forbidden-pattern grep
Command:
`rg -n "re\\.search\\(|re\\.match\\(|ast\\.parse\\(|eval\\(" knowledge3d/knowledgeverse/trm_navigator.py benchmarks/arc_agi_2_adapter.py benchmarks/last_humanity_exam.py knowledge3d/knowledgeverse/specialist_router.py || true`

Output: *(empty)*

### Legacy ARC markers grep
Command:
`rg -n "legacy_sovereign_pipeline|SovereignAIPipeline|_fallback_or_raise" benchmarks/arc_agi_2_adapter.py || true`

Output: *(empty)*

### Syntax check
Command:
`/home/daniel/miniforge/bin/conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium python -m py_compile benchmarks/arc_agi_2_adapter.py benchmarks/last_humanity_exam.py knowledge3d/knowledgeverse/galaxy_manager.py knowledge3d/knowledgeverse/specialist_router.py knowledge3d/knowledgeverse/trm_navigator.py tests/test_hot_path_sovereignty.py`

Result: success.

## Important Runtime Note
I did **not** run benchmark workloads in this pass. This cutover intentionally makes CPU query fallback unavailable by default (`K3D_REQUIRE_PTX_QUERY=true`), so runtime now requires a PTX query kernel implementation (next step) or explicit non-sovereign override for diagnostics.

## Next Required Foundation Work
1. Implement PTX query kernel path for `GalaxyManager.query(...)`.
2. Populate minimal Math/Grammar templates for sovereign math composition diagnostics.
3. Run bounded math diagnostic after PTX query path exists.

