# CODEX → CLAUDE: ARC-3 Sovereign Game Loop Report

## Status

Implemented the ARC-3 sovereign game-loop cut. The live chooser no longer performs Python target selection, spatial-path fallback, episode-dict fallback, exploration fallback, click fallback, or default-action fallback.

## What Changed

### 1. Live episode rules are route-capable Grammar entries

- File: `knowledge3d/knowledgeverse/arc3_episode_galaxy.py`
- `seed_rule()`, `_seed_game_mechanics_priors()`, and `_crystallize_rules()` now upsert live `Grammar` entries immediately through `galaxy_manager.upsert_entry(...)`
- live rule entries now carry:
  - `route_family = GAME_2D`
  - `selection_role = validator`
  - `answer_eligible = true`
  - `route_policy = {"branch_topk": 0}`
  - `metadata.action_index`
  - `metadata.action_name`
  - `meta_refs` with `answer_kind:action`

This makes the episode rules visible to the sovereign route path instead of only living in `_rules_by_key`.

### 2. Action materialization now understands action stars

- File: `knowledge3d/knowledgeverse/knowledgeverse.py`
- `materialize_runtime_result()` now also reads `meta_refs` for:
  - `answer_kind:action`
  - `action_index:N`
  - `action_name:ACTIONN`

This means a winning game-rule star can materialize a control answer even when the action is expressed through star metadata rather than answer text.

### 3. Perception query is now perception-only

- File: `benchmarks/arc_agi_3.py`
- `_frame_to_query_text()` no longer injects Python guidance like:
  - `primary action ...`
  - `action move up/right/...`
  - target-priority strings from `_select_mechanic_target()`
- it now emits perception signals only:
  - avatar row/col
  - background color
  - visible object rows/cols/colors/sizes
  - adjacent cell colors
  - `object adjacent to avatar ...`
  - budget/lives/frame-state signals
  - available actions

### 4. Python decision chain removed

- File: `benchmarks/arc_agi_3.py`
- `choose_action()` is now frame → envelope → tablet submit → direct action extraction
- removed from the live path:
  - `_select_mechanic_target()`
  - `_spatial_path_plan()`
  - `_exploration_fallback()`
  - `_episode_galaxy.query_rule_for_state()` fallback use
  - default-to-first-action behavior
  - click-coordinate fallback
  - transition neutral-action bridge
- if no direct action is materialized, it now fails honestly with:
  - `arc3_sovereign_action_not_materialized`

### 5. Step-0 probe removed

- File: `benchmarks/arc3_sdk_agent.py`
- `run_level()` no longer injects a hardcoded probe action before the sovereign loop
- `decide_action()` no longer defaults to `ACTION1`

## Verification

### Focused tests

- `bash scripts/k3d_env.sh run -e k3d-cranium python -m pytest -q tests/test_arc3_agent.py tests/test_arc3_living_memory.py tests/test_arc3_autonomous_retry.py`
- result:
  - `21 passed in 4.05s`

### Dead fallback grep

- no remaining references in `benchmarks/arc_agi_3.py` to:
  - `_select_mechanic_target`
  - `_exploration_fallback`
  - `_spatial_path_plan`
  - `_walkable_cells`
  - `_tracked_click_target`
  - `_salient_click_centers`
  - `_movement_action_indices`
  - `_exploration_order`
  - `_navigation_state_key`

### Bounded live LS20 run

- command:
  - `bash scripts/k3d_env.sh run -e k3d-cranium python benchmarks/arc3_sdk_agent.py --game ls20 --max-steps 10`
- result:
  - `steps = 10`
  - `session_steps = 10`
  - `levels_completed = 0`
  - `transport = remote_api_compat`
  - `episode_context_seen = true`
  - `max_episode_rule_count = 2`
  - `max_episode_object_count = 9`
  - `episode_consolidation = {"rules_persisted": 2, "session_entries": 1}`
  - action distribution:
    - `ACTION2 = 10`
  - scorecard:
    - `https://three.arcprize.org/scorecards/9d3c05ef-4f10-416f-82d5-ee38f8eab149`

### Important interpretation

- the shell did not inject a probe, spatial plan, rule-dict result, click fallback, or default action
- the 10 live actions were all server-acknowledged `ACTION2` emissions coming through the sovereign route result path
- so the game loop is now failing or succeeding for the right reason: sovereign policy quality, not Python orchestration

## Live Rule Visibility Note

- a fresh standalone `Knowledgeverse()` boot after the bounded run showed:
  - `arc3_rule_entries = 0`
- this does **not** contradict the in-session live insertion
- it means the live session-memory rule visibility is currently process-local, while the fresh boot measurement only sees what is already durably present on disk at startup
- the stronger evidence for the current phase is:
  - unit-proven live `upsert_entry(...)` on rule seeding/crystallization
  - direct game action materialization on the bounded live run

## Background Processes

- relaunched long ARC-3 learner on the patched sovereign build:
  - tmux session:
    - `arc3_ls20_autonomous`
  - log:
    - `/tmp/arc3_ls20_autonomous_sovereign_v1.log`
  - command:
    - `bash scripts/k3d_env.sh run -e k3d-cranium python -u benchmarks/arc3_sdk_agent.py --game ls20 --autonomous --max-attempts 5 --max-steps 10000`
- read-only tmux check:
  - `arc3_ls20_autonomous: 1 windows`
  - `echosys_ingest: 1 windows`

## Remaining Gap

- the chooser is now sovereign, but the policy still collapses to repeated `ACTION2` on the bounded LS20 run
- the next real problem is now inside rule quality / route selection / perception-to-rule matching, not Python fallback code
