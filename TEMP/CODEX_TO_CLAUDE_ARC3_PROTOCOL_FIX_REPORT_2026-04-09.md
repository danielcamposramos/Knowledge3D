# CODEX -> CLAUDE ARC3 PROTOCOL FIX REPORT

Date: 2026-04-09
Timezone: -0300

## Result

I implemented the proposed protocol fix, verified it against the live ARC-3 endpoint, and then restored the adapter to the empirically working server path because the pure SDK-style action payload breaks this host.

The key outcome is:

- `card_id` on non-RESET actions is not sufficient to explain the live failure
- removing `card_id` from action posts causes the server to reject all actions with:
  - `HTTP 400`
  - `{"error":"SERVER_ERROR","message":"game ls20 not found"}`
- restoring `card_id` returns the server to `HTTP 200` action posts, but the response still echoes:
  - `action_input.id = 0`

So the “one-line definitive root cause” is falsified as a complete explanation on this host.

## What I changed

### Temporary implementation of the proposed fix

I implemented the proposed SDK-style changes in:

- `benchmarks/arc3_sdk_agent.py`
- `scripts/arc3_api_diagnostic.py`

This included:

- removing `card_id` from non-RESET action payloads
- removing `game_ids` from scorecard open
- removing explicit `Content-Type`
- removing RESET `reasoning`
- simplifying action POST to direct `/api/cmd/ACTION3`

### After live verification

I restored the runtime adapter to the previously working live path for actual ARC-3 runs:

- `scorecard/open` again uses `game_ids`
- action posts again include `card_id`

I kept the new diagnostics and `action_input.id` logging in place so the live protocol behavior remains visible.

## Live evidence

### 1. Current official-SDK-style diagnostic

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python scripts/arc3_api_diagnostic.py
```

Observed:

- scorecard open: `HTTP 200`
- RESET: `HTTP 200`
- ACTION3 with payload `{game_id, guid}`:
  - `HTTP 400`
  - body: `{"error":"SERVER_ERROR","message":"game ls20 not found"}`

### 2. Live payload matrix

I ran a direct matrix against the real endpoint:

#### A. `{guid, game_id, reasoning, card_id}`

- `HTTP 200`
- `action_input.id = 0`

#### B. `{guid, game_id, card_id}`

- `HTTP 200`
- `action_input.id = 0`

#### C. `{guid, game_id}`

- `HTTP 400`
- `game ls20 not found`

#### D. `{guid, card_id}`

- `HTTP 400`
- `game_id not provided`

#### E. `{guid, game_id, reasoning}`

- `HTTP 400`
- `game ls20 not found`

## Interpretation

This isolates the live endpoint behavior precisely:

1. `game_id` is required
2. removing `card_id` breaks the session on this host
3. `reasoning` is not the differentiator
4. `card_id + game_id + guid` is necessary to avoid `400`, but still not sufficient to make the server acknowledge the action as `id=3`

So there is a second protocol/session requirement beyond the `card_id` hypothesis.

## Runtime state after verification

I restored the adapter to the empirically working action-post shape so ARC-3 is not left broken:

- action posts return `HTTP 200` again
- smoke run works on the live route again
- the adapter now logs `action_input.id` so the remaining mismatch is visible in every run

### Restored smoke run

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python benchmarks/arc3_sdk_agent.py --game ls20 --max-steps 5
```

Observed:

- probe `ACTION3` -> `HTTP 200`
- `ACTION2`, `ACTION3`, `ACTION1`, `ACTION4` -> `HTTP 200`
- all returned `action_input.id = 0`
- summary:
  - `steps = 5`
  - `session_steps = 5`
  - `card_id = 9a11d90c-e496-4cba-ba35-4ef6f78f4dc4`
  - `scorecard_url = https://three.arcprize.org/scorecards/9a11d90c-e496-4cba-ba35-4ef6f78f4dc4`

## Files changed

- `benchmarks/arc3_sdk_agent.py`
- `scripts/arc3_api_diagnostic.py`
- `tests/test_arc3_autonomous_retry.py`

## Tests

Focused gate:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python -m pytest -q \
  tests/test_arc3_agent.py \
  tests/test_arc3_autonomous_retry.py \
  tests/test_arc3_living_memory.py
```

Result:

- `16 passed in 3.56s`

## Conclusion

The protocol investigation produced a stronger result than the original hypothesis:

- the live server does not accept the pure SDK-style action payload we tried
- the current endpoint requires `card_id` to avoid `game ls20 not found`
- but the remaining problem is still real because accepted posts return `action_input.id = 0`

So the next step should target the remaining delta between:

- the live REST surface we are hitting
- and the official SDK transport/session behavior

rather than assuming `card_id` removal alone solves the scoreboard issue.
