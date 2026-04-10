# CLAUDE: ARC-3 Sovereignty Cleanup Report (2026-04-10)

## Summary

Removed ALL Python orchestration from the ARC-3 perception query and episode context pipeline. Perception is now RAW SIGNALS ONLY. TRM finds rules through Galaxy star navigation, not Python-injected text.

## What Was Removed

### 1. `_strategy_hint()` function — DELETED
- **File:** `knowledge3d/knowledgeverse/arc3_episode_galaxy.py`
- Was computing `stop_trying_action2`, `stuck_try_reset`, `budget_critical_minimize_moves`
- Pure Python reasoning engine deciding strategy for TRM
- `self._strategy_hint` attribute removed from `__init__`, `_update_object_records()`, `consolidate_to_house()`

### 2. Episode rule text injection — REMOVED
- **File:** `benchmarks/arc_agi_3.py`, `_frame_to_query_text()` lines 735-761
- Was injecting "episode avoid action2 due blocked when ...", "episode alternative action1 for ...", "episode rule ... predicts ..."
- Python curating which rules to present to TRM, bypassing Galaxy route selection

### 3. Strategy hint injection — REMOVED
- **File:** `benchmarks/arc_agi_3.py`, `_frame_to_query_text()` line 735-737
- Was injecting "episode strategy hint stop trying action2"

### 4. Budget bucket classification — REMOVED
- **File:** `benchmarks/arc_agi_3.py`, `_frame_to_query_text()`
- Was classifying budget as "critical"/"low"/"healthy" — Python semantic interpretation
- Now passes raw percentage only: "movement budget 42 percent remaining"

### 5. Stuck signal strategy text — REMOVED
- **File:** `benchmarks/arc_agi_3.py`, `_frame_to_query_text()`
- Was injecting "stuck avatar not progressing try different approach"
- Now passes raw centroid drift value only: "centroid drift 0.5"

### 6. WINE envelope orchestration data — REMOVED
- **File:** `knowledge3d/tablet/wine/game2d_wine.py`
- Removed `inferred_rules`, `strategy_hint`, `recent_outcomes` from `task_context`
- Kept `known_objects` (raw signal: what objects exist at what colors)

### 7. `episode_context()` cleaned
- **File:** `knowledge3d/knowledgeverse/arc3_episode_galaxy.py`
- Removed `rules` and `strategy_hint` from return value
- Now returns only `objects` (raw signal) and `recent_outcomes`

## What Was Kept (Correctly Sovereign)

- `_upsert_live_rule()` → Creates Galaxy stars with `route_family=GAME_2D`, `answer_kind:action`
- `_upsert_live_object()` → Creates Galaxy stars for objects
- `_upsert_observation_star()` → Creates Galaxy stars for significant observations
- `_crystallize_rules()` → Micro-sleeptime consolidation creating Galaxy stars from observations
- Avoidance alternatives → Created as Galaxy stars (TRM finds them via route)
- `_load_persisted_rules()` → Loads House-persisted rules into Galaxy at boot
- `_persist_strong_rules()` → Persists positive and negative rules to House
- `_emit_learning_log()` → Now emits raw data (last_actions, last_blocked) instead of Python strategy

## Perception Query (Now Raw Signals Only)

What `_frame_to_query_text()` produces:
- Grid dimensions
- Background color
- Avatar position (row, col)
- Budget percentage (raw number)
- Lives remaining (raw count)
- Visible objects (color, row, col, size, semantic_hint, tested/untested)
- Adjacent cell colors
- Adjacent objects
- Centroid drift (raw float)
- Available actions

What it does NOT produce (removed):
- Strategy hints
- Rule summaries
- Budget urgency labels
- "Try different approach" directives

## Verification

- **Tests:** 41 passed, 0 failed
- **Sovereignty grep:** Zero hits for `strategy_hint`, `inferred_rules`, `stop_trying`, `try different`, `budget critical`
- **Learning signals live:** `[ARC3-LEARN]` shows galaxy_stars growing (511→513), raw action/blocked data
- **Autonomous run started:** tmux `arc3_ls20_sovereign`, log at `/tmp/arc3_ls20_sovereign_v2.log`

## Remaining Problem (NOT Sovereignty)

TRM still selects ACTION2 repeatedly despite avoidance stars existing in Galaxy. This is now correctly a **policy quality** issue:
- Galaxy has avoidance stars for blocked ACTION2
- Galaxy has alternative movement stars (ACTION1/3/4)
- TRM route selection is not weighting these alternatives high enough
- Fix is in Galaxy route selection / TRM training, not Python code

## Review Sources

- **Kimi K2.5:** Confirmed `_strategy_hint()` as critical violation, suggested raw data logging
- **Qwen 3.5 397B:** Confirmed stuck_signal and budget bucket as violations, verified Galaxy operations as correct

## Files Changed

- `benchmarks/arc_agi_3.py` — perception query cleanup
- `knowledge3d/knowledgeverse/arc3_episode_galaxy.py` — removed `_strategy_hint()`, cleaned episode_context
- `knowledge3d/tablet/wine/game2d_wine.py` — cleaned WINE envelope task_context
- `tests/test_arc3_agent.py` — updated assertions
- `tests/test_arc3_living_memory.py` — updated assertions
- `tests/test_arc3_autonomous_retry.py` — updated mock
- `tests/test_arc_r0_surface.py` — updated mock and assertions
