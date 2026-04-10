# CODEX -> CLAUDE ARC3 LEARNING LOOP REPORT

Date: 2026-04-09
Timezone: -0300

## Summary

I implemented `TEMP/CODEX_ARC3_LEARNING_LOOP_SPEC_2026-04-09.md` on top of the corrected full-id ARC-3 transport.

The key changes are now live:

- one-time LS20 gameplay color diagnostics
- hardcoded color targeting replaced by frame-derived visible objects
- episode rules can now be queried before exploration fallback
- exploration fallback is no longer the old LEFT/RIGHT bounce; it is a visited-cell heading policy
- avatar movement now feeds a persistent visited-cell set and object map

The bounded live verification did not solve LS20 Level 1 yet, but it did materially change behavior in the right direction:

- previous autonomous run: almost pure LEFT/RIGHT bounce
- new bounded run: varied movement across `UP`, `DOWN`, `RIGHT`, and some `LEFT`
- spatial plan target is now non-empty
- visited map is growing

## Files Changed

- `benchmarks/arc_agi_3.py`
- `knowledge3d/knowledgeverse/arc3_episode_galaxy.py`
- `benchmarks/arc3_sdk_agent.py`
- `tests/test_arc3_agent.py`
- `tests/test_arc3_living_memory.py`

## 1. Fix 5: Actual LS20 gameplay colors

One-time diagnostic now logs real gameplay colors from the first live gameplay frame.

Observed live LS20 colors in the gameplay area:

- color `9`: `5` pixels, centroid `(11.6, 36.4)`, `upper_center`
- color `5`: `43` pixels, centroid `(12.0, 36.0)`, `upper_center`
- color `9`: `1` pixel, centroid `(13.0, 35.0)`, `upper_center`
- color `5`: `200` pixels, centroid `(24.5, 1.5)`, `center_left`
- color `0`: `3` pixels, centroid `(31.7, 21.3)`, `lower_center`
- color `1`: `1` pixel, centroid `(32.0, 20.0)`, `lower_center`
- color `1`: `1` pixel, centroid `(33.0, 21.0)`, `lower_center`
- color `3`: `892` pixels, centroid `(34.6, 35.2)`, `lower_center`
- color `12`: `10` pixels, centroid `(45.5, 31.0)`, `bottom_center`
- color `9`: `15` pixels, centroid `(48.0, 31.0)`, `bottom_center`

Ground truth:

- old switch colors `{11, 15}` were wrong for this LS20 frame
- old door colors `{5, 9}` partially overlapped the visible upper-center structure
- recharge-relevant `12` is visibly present in the lower gameplay region

## 2. Fix 1: Target detection

I did not keep the old hardcoded `_switch_components()` / `_door_components()` behavior as the live source of truth.

What changed:

- visible objects are now discovered from the current gameplay frame
- each object is stored with:
  - `color`
  - `centroid`
  - `size`
  - `position_label`
  - `semantic_hint`
  - `interaction_tested`
  - `points`
- `choose_action()` now merges those objects into the persistent local object map and seeds them into the Episode Galaxy
- `_select_mechanic_target()` and `_spatial_path_plan()` now operate over those visible objects instead of relying on fixed color sets

Live result:

- `spatial_plan_targets = ["switch"]`

So the planner is no longer targetless. It is now finding a concrete target class in the frame.

## 3. Fix 3: Exploration fallback

Implemented:

- replaced the old bounce-prone exploration fallback with visited-cell heading exploration
- added:
  - `_visited_cells`
  - `_explore_heading`
  - `_record_visited_cell()`

This is still lightweight shell policy, not a new reasoning engine.

Behavioral result:

- previous autonomous run after 645 actions:
  - `LEFT` and `RIGHT` dominated almost everything
- new 100-step bounded run:
  - `ACTION4`: `41`
  - `ACTION1`: `31`
  - `ACTION2`: `24`
  - `ACTION3`: `3`

So the fallback is no longer the old left-right ricochet.

## 4. Fix 2: Episode rule consultation

Implemented:

- added `ARC3EpisodeGalaxy.query_rule_for_state(...)`
- `choose_action()` now queries episode rules before dropping to exploration fallback when:
  - no direct emitted action exists
  - there is no spatial-plan fallback already available
- rule query ignores crystallized rules whose predicted outcome is `blocked` or `death`

This closes the previously broken loop:

- outcomes crystallize into rules
- later decisions can actually read those rules

Regression added:

- `tests/test_episode_rule_query_prefers_non_blocking_rule`
- `tests/test_arc3_agent_uses_episode_rule_before_exploration_fallback`

## 5. Fix 4: Object map from avatar movement

Implemented:

- avatar centroid is recorded into `_visited_cells`
- `learn_from_outcome()` now detects static objects from the current frame and merges them into the persistent object map
- those objects are also seeded into the Episode Galaxy through `seed_object(...)`

Bounded live run result:

- `visited_cells_count = 18`
- `known_object_count = 13`

So the agent is now preserving a non-trivial local map instead of rediscovering the same surface every frame.

## Live Test: 1 attempt, 100 steps

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium \
  python -u benchmarks/arc3_sdk_agent.py --game ls20 --max-steps 100
```

Structured result:

- `game_id = ls20-9607627b`
- `steps = 100`
- `levels_completed = 0`
- `score = 0.0`
- `max_episode_rule_count = 7`
- `max_episode_object_count = 9`
- `episode_consolidation.rules_persisted = 7`
- scorecard:
  - `https://three.arcprize.org/scorecards/cfe95835-a9ef-4ab0-aeae-828b83d1a493`

Action distribution:

- `ACTION4`: `41`
- `ACTION1`: `31`
- `ACTION2`: `24`
- `ACTION3`: `3`

This is clearly more varied than the earlier ~50/50 `LEFT`/`RIGHT` oscillation.

Spatial-plan target:

- non-empty
- `["switch"]`

New areas reached:

- yes, in the bounded sense that the shell now recorded `18` visited cells and `13` known objects
- that is not yet enough to clear LS20 Level 1, but it is real state expansion rather than the previous 2-direction bounce

## Assessment

This pass fixed the three architectural breaks described in the spec:

- targeting is no longer hardcoded to stale colors
- episode rules are no longer write-only
- exploration is no longer the original LEFT/RIGHT bounce

What remains wrong is the quality of target semantics and route choice, not the transport or the existence of the learning loop itself.

The most important live fact is:

- the planner now has a non-empty target (`switch`)
- but the chosen behavior still does not convert into a winning LS20 policy

So the next pressure is on semantic interpretation quality and target prioritization, not on action delivery or the existence of the loop.

## Verification

Focused tests:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python -m pytest -q \
  tests/test_arc3_agent.py \
  tests/test_arc3_autonomous_retry.py \
  tests/test_arc3_living_memory.py
```

Result:

- `19 passed in 5.80s`

## Follow-on Autonomous Run

After the bounded 100-step verification, I restarted the improved long learner in tmux so the new policy can accumulate experience:

```bash
tmux new-session -d -s arc3_ls20_autonomous bash
tmux send-keys -t arc3_ls20_autonomous \
  "cd '.../Knowledge3D' && bash scripts/k3d_env.sh run -e k3d-cranium \
   python -u benchmarks/arc3_sdk_agent.py --game ls20 --autonomous \
   --max-attempts 5 --max-steps 10000 > /tmp/arc3_ls20_autonomous_5x10000_v2.log 2>&1" C-m
```

Live status at handoff time:

- tmux session:
  - `arc3_ls20_autonomous`
- worker PID:
  - `1342944`
- log:
  - `/tmp/arc3_ls20_autonomous_5x10000_v2.log`
- current state:
  - boot complete
  - TRM launcher initialized
  - run active under the improved learning-loop policy

## Background ingest

Read-only check:

```bash
tmux ls
```

Result:

- `echosys_ingest: 1 windows`

Untouched.
