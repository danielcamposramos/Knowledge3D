# CODEX -> CLAUDE ARC3 VRAM Learning + House Persistence Report (2026-04-09)

## Scope
Implemented the ARC3 learning/persistence patch from `CODEX_ARC3_VRAM_LEARNING_AND_HOUSE_PERSISTENCE_SPEC_2026-04-09.md` with runtime safety constraints:
- Stop repetitive ARC3 autonomous loop.
- Keep `echosys_ingest` and `ollama serve` untouched.
- Preserve sovereign decision flow (no Python action-forcing orchestration).

## Runtime Safety and Run Control
- Confirmed repetition in ARC3 log (`/tmp/arc3_ls20_autonomous_sovereign_v1.log`) before stop.
- Stopped only ARC3 autonomous loop processes:
  - `python -u benchmarks/arc3_sdk_agent.py --game ls20 --autonomous ...` (`PID 198080`)
  - parent shell (`PID 198079`)
- Left ingest/proceduralization alive:
  - `python scripts/fundamental_ingest_pdfs.py ... --provider ollama` (`PID 128629`)
  - `ollama serve` (`PID 4315`)
- Ingest progression confirmed during this work:
  - `page_files`: `1499 -> 1531`
  - latest page advanced: `.../page_00114.json -> .../page_00146.json`

## Implemented Fixes (Spec 1-7)

### Fix 1 (Objects -> Galaxy stars)
- Added `_upsert_live_object(record)` and object star entry construction.
- Called from:
  - `seed_object()`
  - `_update_object_records()` on first discovery and behavior/count updates.

### Fix 2 (First-observation crystallization)
- Updated `_crystallize_rules()` threshold to learn from first sample.
- Confidence formula changed to:
- `confidence = (majority/total) * min(1.0, total/5.0)`

### Fix 3 (Negative rules -> explicit avoidance alternatives)
- In `_crystallize_rules()`, when majority outcome is `blocked` or `death`, generated explicit alternative movement rules for other movement actions (`ACTION1..ACTION4`, excluding source action) with reduced confidence.
- Alternatives are upserted into Grammar as normal ARC3 rules.

### Fix 4 (`[ARC3-LEARN]` micro-sleeptime logging)
- Added runtime learning log line:
- `[ARC3-LEARN] step=... rules=... objects=... galaxy_stars=... strategy=...`
- `galaxy_stars` uses `galaxy_manager.get_galaxy("Grammar")` + `entries` count (no `entry_count()` usage).
- Logging emitted after micro-sleeptime futures complete.

### Fix 5 (Load House-persisted rules at boot)
- Added `_load_persisted_rules()` and called it in `__init__` after priors.
- Rehydrates `_rules_by_key` from Grammar entries (`arc3_rule:*`) on disk/memory.
- Filters to current `game_id` plus generic rules with empty game id.
- Optional boot log implemented:
- `[ARC3-LEARN] Loaded N persisted rules from House`

### Fix 6 (Persist negative rules with relaxed threshold)
- Updated `_persist_strong_rules()`:
  - Positive rules: unchanged default (`min_evidence`, default 3)
  - Negative rules (`blocked`/`death`): persisted with `evidence >= 2`

### Fix 7 (Significant observations -> Galaxy stars)
- Added `_upsert_observation_star(outcome)` and observation star entry builder.
- Called from `seed_outcome()` for significant outcomes:
  - `blocked`
  - `death`
  - `level_complete`

## Additional Correctness Adjustment
- `is_blocked` detection now uses movement reality (`agent_moved == False` for movement actions) rather than strict `cells_changed == 0`, so HUD-only changes do not suppress blocked learning.

## Minimal Integration Surface
- `benchmarks/arc_agi_3.py` updated only to enrich query context text with episode rule summaries (including avoid/blocked semantics) for sovereign retrieval context.
- A temporary direct action-forcing hint was removed after correction from Daniel, to preserve sovereign TRM decision semantics.

## Tests

### Unit / Regression
Executed:
- `bash scripts/k3d_env.sh run -e k3d-cranium env "PYTHONPATH=$(pwd)" pytest -q tests/test_arc3_living_memory.py`
- `bash scripts/k3d_env.sh run -e k3d-cranium env "PYTHONPATH=$(pwd)" pytest -q tests/test_arc3_agent.py tests/test_arc3_autonomous_retry.py`

Result:
- `28 passed` (arc3 living memory + agent + autonomous retry)

### New/Extended coverage in `tests/test_arc3_living_memory.py`
- Object star upsert from `seed_object`
- Significant outcome observation stars
- One-sample rule crystallization
- Avoidance alternatives from blocked crystallization
- Negative rule persistence threshold (`evidence >= 2`)
- Persisted rule reload on fresh episode init
- `[ARC3-LEARN]` log emission after micro-sleeptime

## Behavioral Verification Notes
Bounded autonomous slice executed (`max-steps 100`, timeout 420s), with stall guard check:
- `total_actions=64`
- `unique_in_window=1`
- `stall_single_action_window=true`
- action stream remained `ACTION2` in this bounded run

However, learning signals now clearly grow in-run:
- `[ARC3-LEARN] rules` increased (e.g., `5 -> 9`)
- `[ARC3-LEARN] galaxy_stars` increased (e.g., `511 -> 569`)
- strategy moved to explicit anti-repeat state (`stop_trying_action2`)

This confirms VRAM growth + star materialization are active; remaining non-diversification in this run appears to be downstream action selection weighting, not missing ARC3 episode learning persistence.

## Files Changed
- `knowledge3d/knowledgeverse/arc3_episode_galaxy.py`
- `tests/test_arc3_living_memory.py`
- `benchmarks/arc_agi_3.py`
