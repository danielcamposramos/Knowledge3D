# Codex to Claude: ARC-3 LS20 Game Knowledge Report

**Timestamp:** 2026-04-09 15:14:10 -0300
**Spec:** `TEMP/CODEX_ARC3_LS20_GAME_KNOWLEDGE_SPEC_2026-04-09.md`

## Status

The LS20-specific ARC-3 pass is implemented. The living-memory path now has:

- LS20 game-knowledge stars seeded into `Reality` and `Grammar`
- autonomous retry with persistent episode memory across attempts
- deep between-attempt consolidation that persists strong rules while keeping the episode history alive
- no hard-coded budget cap added to the live ARC-3 decision path

## Requested Checklist

1. `arc3_game_mechanics_seeder.py`: new LS20 stars added
   - **yes**
   - file: `benchmarks/arc3_game_mechanics_seeder.py`
   - new LS20 star count: `9`
   - new star ids:
     - `arc3_game_mechanic:action_refill`
     - `arc3_game_mechanic:key_switch`
     - `arc3_game_mechanic:door_indicator`
     - `arc3_game_mechanic:key_state_display`
     - `arc3_ls20_strategy:level1_sequence`
     - `arc3_game_rule:key_switch_interaction`
     - `arc3_game_rule:door_entry_condition`
     - `arc3_game_rule:door_entry_blocked`
     - `arc3_game_rule:multi_step_artifact`
   - representative line anchors:
     - `benchmarks/arc3_game_mechanics_seeder.py:94`
     - `benchmarks/arc3_game_mechanics_seeder.py:110`
     - `benchmarks/arc3_game_mechanics_seeder.py:126`
     - `benchmarks/arc3_game_mechanics_seeder.py:142`
     - `benchmarks/arc3_game_mechanics_seeder.py:158`
     - `benchmarks/arc3_game_mechanics_seeder.py:266`
     - `benchmarks/arc3_game_mechanics_seeder.py:283`
     - `benchmarks/arc3_game_mechanics_seeder.py:300`
     - `benchmarks/arc3_game_mechanics_seeder.py:317`

2. Hard-coded action cap: found and removed
   - **not-present in the live hot path**
   - `benchmarks/arc3_sdk_agent.py` keeps `max_steps` only as an outer Python safety limit
   - live decision remains delegate-driven through `execute_task()`; no budget-based Python `ACTION7` override was present in the active ARC-3 path
   - budget extraction remains observational (`_movement_budget_snapshot`) for frame context only, not for a Python move cap

3. `run_until_level_complete()` added to `K3DAgent`
   - **yes**
   - file: `benchmarks/arc3_sdk_agent.py:574`

4. `--autonomous` CLI flag added
   - **yes**
   - file: `benchmarks/arc3_sdk_agent.py:685`
   - companion flags also added:
     - `--max-attempts`
     - `--target-levels`

5. `run_deep_consolidation()` added to `ARC3EpisodeGalaxy`
   - **yes**
   - file: `knowledge3d/knowledgeverse/arc3_episode_galaxy.py:520`
   - behavior:
     - scans full accumulated history
     - classifies objects
     - persists strong rules (`confidence >= 0.6`, `evidence >= 3`)

6. Tests: `tests/test_arc3_autonomous_retry.py`
   - **pass**
   - focused ARC-3 suite result:
     - `25 passed in 6.47s`
   - includes:
     - retry loop runs `3` attempts
     - episode galaxy accumulates frames across attempts
     - `run_deep_consolidation()` runs without error

7. **AUTONOMOUS RUN**
   - command:
     - `bash scripts/k3d_env.sh run -e k3d-cranium python benchmarks/arc3_sdk_agent.py --game ls20 --autonomous --max-attempts 5 --max-steps 200`
   - honest result:
     - `attempts_used = 5`
     - `first_completion_attempt = null`
     - `levels_completed_per_attempt = [0, 0, 0, 0, 0]`
     - `rules_crystallized_count = 1`
     - `crystallized_rule_ids = ["arc3_rule:ls20:agent_adjacent_to_color_4:ACTION2"]`
     - `final_episode_consolidation = {"rules_persisted": 1, "session_entries": 1}`
   - per-attempt episode growth:
     - attempt 1: `episode_frames = 200`, `episode_outcomes = 200`
     - attempt 2: `episode_frames = 400`, `episode_outcomes = 400`
     - attempt 3: `episode_frames = 600`, `episode_outcomes = 600`
     - attempt 4: `episode_frames = 800`, `episode_outcomes = 800`
     - attempt 5: `episode_frames = 1000`, `episode_outcomes = 1000`
   - interpretation:
     - persistent episode memory is working
     - deep consolidation is working
     - current LS20 knowledge still is not sufficient for autonomous Level 1 completion in `<= 5` attempts on the live remote-compat path

8. `echosys_ingest` tmux session still alive
   - **yes**
   - `tmux ls` -> `echosys_ingest: 1 windows (created Thu Apr  9 02:35:35 2026)`

## Files Changed

- `benchmarks/arc3_game_mechanics_seeder.py`
- `knowledge3d/knowledgeverse/arc3_episode_galaxy.py`
- `benchmarks/arc_agi_3.py`
- `benchmarks/arc3_sdk_agent.py`
- `tests/test_arc3_autonomous_retry.py`

## Notes

- I kept the existing sovereign ARC ingress unchanged:
  - `K3DAgent` -> `K3DARC3Agent` -> `HeadlessTabletMPC.submit()` -> `Knowledgeverse.execute_task()`
- The episode galaxy now persists across retry attempts by design; only final consolidation clears buffers.
- The autonomous run is CPU/network-heavy because it is doing the exact remote-compat ARC-3 path across `5 * 200` steps. It finished successfully and returned a full JSON result.
