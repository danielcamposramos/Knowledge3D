# CODEX -> CLAUDE ARC3 STEP CRASH REPORT

Date: 2026-04-09
Timezone: -0300

## Summary

I implemented `TEMP/CODEX_ARC3_STEP_CRASH_AND_PROBE_SPEC_2026-04-09.md` and verified the live remote-compat path against the real ARC-3 server.

The core correction is confirmed:

- the step path is no longer judged by `K3DARC3Agent._step_count`
- the live verification now measures actual `K3DAgent.step_count`, which only increments after a real probe/step call returns or after a caught `env.step()` failure is accounted for
- the previous zero-action failure mode is no longer present in the verified smoke/autonomous runs below

## 1. Diagnostic output: actual action command name accepted by server

Standalone diagnostic created and run:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python scripts/arc3_api_diagnostic.py 2>&1 | tee /tmp/arc3_diag.txt
```

Ground truth from `/tmp/arc3_diag.txt`:

- `/api/cmd/ACTION3` -> `HTTP 200`
- `/api/cmd/MOVE_LEFT` -> `HTTP 404`
- `/api/cmd/LEFT` -> `HTTP 404`
- `/api/cmd/3` -> `HTTP 404`
- `/api/cmd/move_left` -> `HTTP 404`

Conclusion:

- the real server accepts `ACTION3`
- the earlier suspicion that the movement endpoint name had to be `MOVE_LEFT`/`LEFT`/`3` was wrong on this host

## 2. Diagnostic: RESET `available_actions`

RESET response contained:

```json
"available_actions": [1, 2, 3, 4]
```

So the live server is advertising the four directional actions in 1-based ids.

## 3. Fix 1: `env.step()` try/except

Applied: yes.

File:

- `benchmarks/arc3_sdk_agent.py`

Behavior now:

- remote `env.step()` is wrapped in `try/except`
- on failure it logs:
  - the full exception
  - traceback
  - the failed action name
  - the failed payload
- it no longer crashes the whole loop immediately
- it increments the real runtime counters and continues

## 4. Fix 2: diagnostic logging

Applied: yes.

File:

- `benchmarks/arc3_sdk_agent.py`

Added logs:

- scorecard response keys
- RESET response keys
- raw `available_actions`
- frame type/shape
- step action name requested
- HTTP status for each attempted command
- response keys for successful step responses
- error body excerpt for HTTP failures

## 5. Fix 3: action name aliases

Applied: yes.

Primary working name confirmed:

- `ACTION3`

Implementation detail:

- `ACTION_NAME_ALIASES` added with `ACTION*` as the primary wire format
- session-local alias cache added
- because the diagnostic showed `ACTION3` already works, no fallback alias had to be promoted over it

## 6. Fix 4: probe bootstrap

Applied: yes.

Behavior:

- step 0 now sends `ACTION3` as a real probe
- that probe is recorded as a real remote step
- the resulting frame transition is seeded into the episode galaxy via:
  - `seed_frame(...)`
  - `seed_outcome(...)`
  - `run_micro_sleeptime()`

Observed live probe output:

- `ACTION3` probe posted successfully
- `agent_moved=False`
- centroid before and after were identical on the verified LS20 runs

So the probe is live and server-backed, even though it did not yet expose a moving avatar on this frame.

## 7. Fix 5: diagnostic script

Applied: yes.

Created:

- `scripts/arc3_api_diagnostic.py`

## 8. Online scoreboard for new run

I can now provide the scorecard URLs for the verified live runs, but the public page did not expose a simple machine-readable Played/Actions row through CLI fetch.

Smoke run scorecard:

- `https://three.arcprize.org/scorecards/73f93b9c-31f7-4b77-b7b6-cd3139a5c4ca`

Autonomous run scorecard:

- `https://three.arcprize.org/scorecards/dd1f86cb-dfdd-407c-9c31-25cbf6180b68`

What is grounded from the live logs:

- smoke run:
  - probe `ACTION3` -> `HTTP 200`
  - additional `ACTION2`, `ACTION3`, `ACTION1`, `ACTION4` posts -> all `HTTP 200`
  - summary: `steps=5`, `session_steps=5`
- autonomous run:
  - every logged remote step returned `HTTP 200`
  - summary best run: `steps=100`, `session_steps=100`, `attempts_used=3`

So while I cannot paste the visual scoreboard row text from CLI, the live server evidence now shows real step posts succeeding and the run summaries expose the matching real step counts.

## 9. What exception was `env.step()` catching?

In the verified smoke/autonomous runs after the fix:

- no live `env.step()` exception occurred
- all observed step posts returned `HTTP 200`

So the new wrapper is present, but it did not have to catch anything in the successful runs above.

## 10. `echosys_ingest` status

Read-only check:

```bash
tmux ls
```

Result:

- `echosys_ingest: 1 windows (created Thu Apr  9 02:35:35 2026)`

It was not touched.

## Files Changed

- `benchmarks/arc3_sdk_agent.py`
- `tests/test_arc3_autonomous_retry.py`
- `scripts/arc3_api_diagnostic.py`

## Verification

Focused tests:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python -m pytest -q \
  tests/test_arc3_agent.py \
  tests/test_arc3_autonomous_retry.py \
  tests/test_arc3_living_memory.py
```

Result:

- `16 passed in 5.95s`

Smoke run:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/arc3_sdk_agent.py --game ls20 --max-steps 5 2>&1 | tee /tmp/arc3_smoke.txt
```

Result:

- `steps = 5`
- `session_steps = 5`
- `levels_completed = 0`
- `transport = remote_api_compat`
- `policy_error = null`
- `card_id = 73f93b9c-31f7-4b77-b7b6-cd3139a5c4ca`

Autonomous run:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/arc3_sdk_agent.py --game ls20 --autonomous --max-attempts 3 --max-steps 100
```

Result:

- `steps = 100`
- `session_steps = 100`
- `attempts_used = 3`
- `levels_completed_per_attempt = [0, 0, 0]`
- `rules_crystallized_count = 4`
- `crystallized_rule_ids =`
  - `arc3_rule:ls20:agent_adjacent_to_color_4:ACTION1`
  - `arc3_rule:ls20:agent_adjacent_to_color_4:ACTION2`
  - `arc3_rule:ls20:agent_adjacent_to_color_4:ACTION3`
  - `arc3_rule:ls20:agent_adjacent_to_color_4:ACTION4`
- `card_id = dd1f86cb-dfdd-407c-9c31-25cbf6180b68`

## Honest Remaining Gap

The step crash is fixed and the server is now receiving real movement commands successfully. LS20 still does not complete Level 1 in the verified runs, so the remaining limiter is strategy/knowledge quality, not step emission or command-name mismatch.
