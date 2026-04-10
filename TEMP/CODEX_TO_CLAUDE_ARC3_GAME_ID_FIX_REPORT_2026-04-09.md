# CODEX -> CLAUDE ARC3 GAME ID FIX REPORT

Date: 2026-04-09
Timezone: -0300

## Summary

I implemented `TEMP/CODEX_ARC3_GAME_ID_FIX_AND_SDK_ENV_2026-04-09.md` on the current sovereign ARC-3 path.

The definitive root cause is now fixed in the live adapter:

- short `ls20` is resolved at startup through `GET /api/games`
- the resolved runtime ID is `ls20-9607627b`
- non-RESET actions now use the SDK-style payload:
  - `{"game_id": "<full_id>", "guid": "<guid>"}`
- `card_id` is still used for `RESET` and `scorecard/close`, but no longer sent on action posts

This supersedes my earlier protocol-fix report, which was correct about the short-name failure but wrong about the full cause because the runtime was still using the short `ls20` ID.

## Files Changed

- `benchmarks/arc3_sdk_agent.py`
- `scripts/arc3_api_diagnostic.py`
- `tests/test_arc3_autonomous_retry.py`

## Implementation

### 1. Full game-id resolution

Added `_RemoteArcCompatEnv._resolve_full_game_id()`:

- calls `GET /api/games`
- exact-match checks first
- then prefix-match resolves `ls20` -> `ls20-9607627b`
- stores:
  - `requested_game_id`
  - resolved `game_id`
  - `game_tags`
  - `game_title`

### 2. Scorecard and RESET protocol

Aligned to the working SDK protocol:

- scorecard open:
  - `{"tags": ["k3d-sovereign-r0"]}`
- RESET:
  - `{"card_id": "<id>", "game_id": "ls20-9607627b"}`

### 3. Action protocol

Aligned action posts to the working SDK shape:

- step payload:
  - `{"game_id": "ls20-9607627b", "guid": "<guid>"}`
- no `card_id` on `ACTION1-7`
- no alias loop

### 4. Runtime summary consistency

`K3DAgent` now adopts the resolved remote `game_id`, so live summaries report the true server-side ID instead of the short input alias.

## Verification

### Local regression

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python -m pytest -q \
  tests/test_arc3_agent.py \
  tests/test_arc3_autonomous_retry.py \
  tests/test_arc3_living_memory.py
```

Result:

- `17 passed in 5.75s`

Added focused proof:

- short `ls20` resolves to `ls20-9607627b`
- scorecard open sends tags only
- RESET uses full `game_id`
- action post omits `card_id`

### Live standalone diagnostic

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python scripts/arc3_api_diagnostic.py
```

Ground truth:

- `GET /api/games` -> `HTTP 200`
- resolved game:
  - `game_id = ls20-9607627b`
  - `title = LS20`
  - `tags = ["keyboard"]`
- scorecard open -> `HTTP 200`
- RESET with full ID -> `HTTP 200`
- `ACTION3` with payload `{game_id: "ls20-9607627b", guid: "..."}`
  - `HTTP 200`
  - `action_input.id = 3`

Diagnostic card:

- `8d22d41f-d03d-4148-a00f-cba947086f5b`

### Live ARC-3 smoke

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/arc3_sdk_agent.py --game ls20 --max-steps 5
```

Ground truth from the run:

- runtime log:
  - `Resolved game_id: 'ls20' -> 'ls20-9607627b'`
- probe:
  - `ACTION3` payload keys = `['game_id', 'guid']`
  - response `action_input.id = 3`
- later actions:
  - `ACTION2` -> `action_input.id = 2`
  - `ACTION3` -> `action_input.id = 3`
  - `ACTION1` -> `action_input.id = 1`
  - `ACTION3` -> `action_input.id = 3`

Smoke summary:

- `game_id = ls20-9607627b`
- `steps = 5`
- `session_steps = 5`
- `levels_completed = 0`
- `transport = remote_api_compat`
- `policy_error = null`

Smoke scorecard:

- `https://three.arcprize.org/scorecards/a38fd2cd-d24d-4ee2-a0ad-6650ff3b3d52`

## Result

The earlier `action_input.id = 0` failure mode is gone on the corrected full-ID path.

The ARC-3 server is now acknowledging the actual actions we send:

- `ACTION3 -> 3`
- `ACTION2 -> 2`
- `ACTION1 -> 1`

So the transport layer is no longer blocked on silent action failure. Remaining ARC-3 limits are now strategy/knowledge quality, not game-id or action-wire-format mismatch.

## Long Autonomous Run Update

Per Daniel's follow-up, I also removed the remaining short autonomous ceiling in the live helper:

- `K3DAgent.run_until_level_complete(..., steps_per_attempt=10000)`

That now matches the CLI/runtime contract already in place:

- `--max-steps 10000`
- `--max-attempts 5`
- effective ceiling: `50000` actions across one autonomous session

I then launched the long LS20 learning run in a detached tmux session instead of leaving it attached to the terminal:

```bash
tmux new-session -d -s arc3_ls20_autonomous bash
tmux send-keys -t arc3_ls20_autonomous \
  "cd '.../Knowledge3D' && bash scripts/k3d_env.sh run -e k3d-cranium \
   python -u benchmarks/arc3_sdk_agent.py --game ls20 --autonomous \
   --max-attempts 5 --max-steps 10000 > /tmp/arc3_ls20_autonomous_5x10000.log 2>&1" C-m
```

Live status at `2026-04-09 19:45:36 -0300`:

- tmux session:
  - `arc3_ls20_autonomous`
- worker PID:
  - `1321493`
- command:
  - `python -u benchmarks/arc3_sdk_agent.py --game ls20 --autonomous --max-attempts 5 --max-steps 10000`
- log:
  - `/tmp/arc3_ls20_autonomous_5x10000.log`
- observed progress snapshot:
  - `24` acknowledged remote steps already logged
  - log tail still shows real server acknowledgement on the corrected path:
    - `ACTION3 -> action_input.id = 3`
    - `ACTION4 -> action_input.id = 4`

This run is intentionally still active. It was launched to keep learning in the background rather than to stop after a short smoke.

## arc3-sdk env note

I did not switch the runtime to the Python 3.12 `arc3-sdk` env in this fix. The current implementation used Option B from the spec:

- keep the sovereign engine in `k3d-cranium`
- port the proven SDK wire protocol into the existing I/O adapter

That was sufficient to verify the fix live.

## Background ingest

Read-only check:

```bash
tmux ls
```

Result:

- `arc3_ls20_autonomous: 1 windows`
- `echosys_ingest: 1 windows (created Thu Apr  9 02:35:35 2026)`

Untouched.
