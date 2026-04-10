# CODEX -> CLAUDE ARC3 ACTION EMISSION FIX REPORT

Date: 2026-04-09
Timezone: -0300

## Scope

Implemented `TEMP/CODEX_ARC3_ACTION_EMISSION_FIX_2026-04-09.md` on the existing sovereign ARC-3 path, with Daniel's additional runtime-budget correction applied:

- per-attempt default action budget raised to `10000`
- autonomous default remains `5` attempts
- effective default ceiling is now `50000` total actions across `5` attempts

## Files Changed

- `benchmarks/arc_agi_3.py`
- `benchmarks/arc3_sdk_agent.py`
- `tests/test_arc3_agent.py`

## Bug Fix Status

### 1. `tablet_result["emitted"]` subscript crash

Fixed.

- replaced direct subscript access with safe `.get(...)` handling
- defensive normalization now tolerates missing `response`, missing `emitted`, and missing `task_result`
- live chooser no longer aborts before the first `env.step()`

### 2. Duplicate `choose_action()` definition

Fixed.

- `benchmarks/arc_agi_3.py` now has a single live `choose_action()`
- structural check:
  - `rg -n "def choose_action\\(" benchmarks/arc_agi_3.py`
  - result: one definition at line `1216`

### 3. `_step_count` not incrementing in the new method

Fixed.

- `_step_count` now increments in the live chooser path
- it no longer increments inside `learn_from_outcome()`
- structural check:
  - transition branch increment at `benchmarks/arc_agi_3.py:1322`
  - normal chooser increment at `benchmarks/arc_agi_3.py:1572`

### 4. No spatial plan in the live chooser

Fixed.

- `choose_action()` now computes `_spatial_path_plan(...)`
- if the tablet emits no direct action, the chooser falls back to the GPU-backed spatial plan instead of blind `ACTION1`
- regression coverage added/updated in `tests/test_arc3_agent.py`

### 5. Exception in `choose_action()` terminates game loop

Fixed.

- chooser now catches submit failures, logs them, and keeps the loop alive
- fallback remains sovereign:
  - emitted action if present
  - otherwise spatial plan
  - otherwise existing deterministic shell fallback

### 6. `_ensure_delegate()` swallows init errors silently

Fixed.

- `benchmarks/arc3_sdk_agent.py` now logs delegate init failures and prints traceback
- runtime no longer hides initialization faults

## Additional Emission Fix

There was one more live bug beyond the six listed in the spec:

- `_derive_action_from_result()` recognized `ACTION6` before inspecting `action_input`
- that caused real emitted click coordinates to be discarded
- the chooser then substituted tracked-focus clicks even though the tablet had already emitted the correct click payload

This is now fixed:

- real emitted click payloads win over tracked-focus fallback
- emitted click actions now carry `click_reason = "tablet_boundary_click"`

## Verification

### Focused tests

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python -m pytest -q \
  tests/test_arc3_agent.py \
  tests/test_arc3_autonomous_retry.py \
  tests/test_arc3_living_memory.py
```

Result:

- `14 passed in 3.19s`

### Structural checks

Command:

```bash
rg -n "def choose_action\\(|tablet_result\\[|_step_count \\+= 1|traceback\\.print_exc\\(|max_actions: int = 10000|max_steps: int = 10000|parser.add_argument\\(\"--max-steps\", type=int, default=10000\\)" \
  benchmarks/arc_agi_3.py benchmarks/arc3_sdk_agent.py
```

Result highlights:

- no remaining `tablet_result["..."]` usage
- one `choose_action()` definition
- chooser and delegate traceback logging present
- `max_actions: int = 10000`
- `max_steps: int = 10000`
- CLI `--max-steps` default `10000`

## Live Run Results

### Short LS20 verification

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python benchmarks/arc3_sdk_agent.py --game ls20 --max-steps 5
```

Result:

- `steps = 5`
- `session_steps = 5`
- `levels_completed = 0`
- `transport = remote_api_compat`
- `episode_context_seen = true`
- `max_episode_object_count = 9`
- `episode_consolidation = {"rules_persisted": 0, "session_entries": 1}`

Interpretation:

- the zero-action failure mode is gone
- the loop reached `5` real step submissions on the live remote-compat path

### Autonomous verification

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python benchmarks/arc3_sdk_agent.py --game ls20 --autonomous --max-attempts 2 --max-steps 50
```

Result:

- `autonomous = true`
- `attempts_used = 2`
- `steps = 50`
- `session_steps = 50`
- `levels_completed_per_attempt = [0, 0]`
- `first_completion_attempt = null`
- `rules_crystallized_count = 1`
- `crystallized_rule_ids = ["arc3_rule:ls20:agent_adjacent_to_color_4:ACTION1"]`
- attempt 1 deep consolidation:
  - `rules_persisted = 1`
  - `episode_frames = 50`
  - `episode_outcomes = 50`
- attempt 2 deep consolidation:
  - `rules_persisted = 1`
  - `episode_frames = 100`
  - `episode_outcomes = 100`

Interpretation:

- action emission is now live across both attempts
- retry persistence and deep consolidation are still working after the emission fix
- LS20 remains knowledge-limited, not action-dead

## Background Process Check

Read-only check:

```bash
tmux ls
```

Result:

- `echosys_ingest: 1 windows (created Thu Apr  9 02:35:35 2026)`

No interaction was made with that session.

## Remaining Honest Gaps

- LS20 still does not complete Level 1 on the current `remote_api_compat` surface in the verified runs above
- live transport still depends on the compatibility path because the installed `arc_agi` package lacks `Arcade/make`
- the action-emission failure is fixed, but policy/knowledge depth is still the limiter on score
