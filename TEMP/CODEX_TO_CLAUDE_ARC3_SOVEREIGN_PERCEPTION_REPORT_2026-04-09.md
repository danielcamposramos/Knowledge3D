# CODEX TO CLAUDE: ARC-3 Sovereign Perception Report

**Date:** 2026-04-09
**Spec:** `TEMP/CODEX_ARC3_SOVEREIGN_GAME_PERCEPTION_SPEC_2026-04-09.md`

## Implemented

- `benchmarks/arc_agi_3.py`
  - `_walkable_cells()` now uses background-colored floor cells plus the avatar cell, instead of the older permissive color set
  - `_spatial_path_plan()` now emits `ACTION5` when LED-A* places the avatar at an untested target
  - `choose_action()` now records `avatar_centroid`, computes `stuck_signal` / `centroid_drift`, clears blocked state on target change, and passes the new perception signals through WINE/task context
  - `choose_action()` now queries Episode rules with `has_adjacent_untested`
  - `learn_from_outcome()` now turns successful `ACTION5` frame changes into Episode rules and marks nearby objects as interaction-tested
- `knowledge3d/knowledgeverse/arc3_episode_galaxy.py`
  - seeded prior rule: `agent_adjacent_to_untested_object -> ACTION5`
  - added `seed_rule(...)`
  - extended `query_rule_for_state(...)` to match both color-adjacency rules and object-adjacency rules
- `knowledge3d/tablet/wine/game2d_wine.py`
  - added `task_context_extras` merge path so stuckness/perception signals reach the routed task payload
- tests updated:
  - `tests/test_arc3_agent.py`
  - `tests/test_arc3_living_memory.py`

## Verified

- focused ARC-3 gate:
  - `22 passed in 3.87s`

### Fix 1: Spatial Plan Emits `ACTION5` at Target

- unit proof:
  - `tests/test_arc3_agent.py::test_spatial_plan_emits_interact_when_already_at_target`
  - result: pass
- 10-step live run:
  - command:
    - `python -u benchmarks/arc3_sdk_agent.py --game ls20 --max-steps 10`
  - summary:
    - `game_id = ls20-9607627b`
    - `steps = 10`
    - `session_steps = 10`
    - `levels_completed = 0`
    - action distribution:
      - `ACTION2 = 4`
      - `ACTION3 = 3`
      - `ACTION1 = 2`
  - honest result:
    - the code path now supports `ACTION5 at target`, but this 10-step live slice did not reach a target state that triggered it

### Fix 7: Walkable Cells Exclude Walls

- implementation now uses:
  - `background = _background_value(gameplay)`
  - walkable cells = `background-colored floor + avatar cell`
- live probe on the ARC-3 reset frame:
  - `resolved_game_id = ls20-9607627b`
  - `background = 4`
  - `walkable_count = 2322`
  - `non_background_walkable_colors = [0]`
- grounded read:
  - walls/objects were excluded in that sampled frame; only the avatar color remained as a non-background walkable exception

### 1 Attempt, Requested 500 Steps

- command:
  - `python -u benchmarks/arc3_sdk_agent.py --game ls20 --max-steps 500`
- actual run summary:
  - `game_id = ls20-9607627b`
  - `steps = 129`
  - `session_steps = 129`
  - `levels_completed = 0`
  - `score = 0.0`
  - `max_episode_rule_count = 6`
  - `max_episode_object_count = 9`
  - `episode_consolidation.rules_persisted = 6`
  - scorecard:
    - `https://three.arcprize.org/scorecards/c7c73a35-3636-4586-8825-cc1e47813612`
- action distribution:
  - `ACTION2 = 84`
  - `ACTION1 = 41`
  - `ACTION3 = 3`
  - `ACTION5 = 0`
- spatial plan targets:
  - `[]`
- visited/object tracking:
  - `visited_cells_count = 1`
  - `known_object_count = 4`
- honest result:
  - the avatar still did **not** reach and interact with the white cross in this bounded live attempt
  - no level was completed
  - `ACTION5` is still absent from the live distribution
  - the code-level perception fixes are real, but the live policy is still not surfacing them strongly enough into the actual emitted control stream

## Restarted Long Autonomous Run

- previous `arc3_ls20_autonomous` session was replaced so it uses the new perception patch
- live tmux session:
  - `arc3_ls20_autonomous`
- live PID:
  - `1363304`
- command:
  - `python -u benchmarks/arc3_sdk_agent.py --game ls20 --autonomous --max-attempts 5 --max-steps 10000`
- log:
  - `/tmp/arc3_ls20_autonomous_5x10000_v3.log`
- read-only tmux check:
  - `echosys_ingest: 1 windows`
  - `arc3_ls20_autonomous: 1 windows`

## Bottom Line

- Fix 1 is implemented and unit-proven
- Fix 7 is implemented and live-probed
- the bounded live LS20 attempt is still honest failure:
  - no `ACTION5`
  - no white-cross interaction
  - no level completion
- the remaining blocker is no longer transport or zero-action failure; it is still the sovereign game-perception/policy quality required to turn the new signals into the correct live action sequence
