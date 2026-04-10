# Codex Direction: ARC-3 REST Protocol Fix — 0 Actions Root Cause

**Date:** 2026-04-09
**Authority:** CLAUDE.md (sovereignty), KNOWLEDGEVERSE_SPECIFICATION.md
**Priority:** CRITICAL — this is THE root cause of 0 actions on the scoreboard
**Evidence:** Downloaded and read `arc-agi-3` v0.0.1 SDK source (`_agent.py`, `_structs.py`, `_swarm.py`)

---

## Root Cause (Definitive)

We downloaded the official `arc-agi-3` SDK (v0.0.1, Python >=3.12) and read its source.
The SDK's `Agent.do_action_request()` method at `_agent.py:158-178` shows the EXACT
protocol the server expects. Our `_RemoteArcCompatEnv` violates it in one critical way.

### The Protocol Rule

The SDK's `do_action_request()` does this:

```python
data = action.action_data.model_dump()   # {"game_id": ""}
if action == GameAction.RESET:
    data["card_id"] = self.card_id       # ONLY for RESET
if self.guid:
    data["guid"] = self.guid             # ONLY after RESET (guid="" initially)
if action.reasoning:
    data["reasoning"] = action.reasoning
if self.game_id:
    data["game_id"] = self.game_id       # Always
```

This means:
- **RESET payload:** `{"game_id": "ls20", "card_id": "<card_id>"}`
  (no guid because self.guid is "" at start)
- **ACTION3 payload:** `{"game_id": "ls20", "guid": "<guid>"}`
  (no card_id — EVER for non-RESET actions)

### What Our Code Does Wrong

**File:** `benchmarks/arc3_sdk_agent.py`, `_RemoteArcCompatEnv.step()`, lines 262-268

```python
payload: dict[str, Any] = {
    "guid": self.guid,
    "game_id": self.game_id,
    "reasoning": {"agent": "k3d-sovereign-r0", "source": "remote_api_compat"},
}
if self.card_id:
    payload["card_id"] = self.card_id    # ← THIS LINE BREAKS EVERYTHING
```

We send `card_id` with EVERY action. The server sees `card_id` in a non-RESET request
and treats it as a session re-initialization or no-op. That's why the response always
returns `action_input.id=0` (the default for GameAction.RESET) — the server doesn't
register our action at all.

---

## Fix 1 (CRITICAL): Remove card_id from step() payload

**File:** `benchmarks/arc3_sdk_agent.py`
**Method:** `_RemoteArcCompatEnv.step()`
**Lines:** 262-268

**Change FROM:**
```python
payload: dict[str, Any] = {
    "guid": self.guid,
    "game_id": self.game_id,
    "reasoning": {"agent": "k3d-sovereign-r0", "source": "remote_api_compat"},
}
if self.card_id:
    payload["card_id"] = self.card_id
```

**Change TO:**
```python
payload: dict[str, Any] = {
    "game_id": self.game_id,
}
if self.guid:
    payload["guid"] = self.guid
```

Key changes:
1. **Remove `card_id`** — the SDK NEVER sends card_id with non-RESET actions
2. **Conditional guid** — the SDK only sends guid if it's truthy (`if self.guid:`)
3. **Remove `reasoning` from base payload** — add it only if the K3DARC3Agent provides one

If the caller (K3DAgent.run_level) wants to attach reasoning, pass it via the `data` param:
```python
# In run_level, when calling env.step():
step_data = {}
if last_decision.get("reasoning"):
    step_data["reasoning"] = last_decision["reasoning"]
obs, reward, done, info = env.step(action, data=step_data)
```

---

## Fix 2 (HIGH): Remove game_ids from scorecard open

**File:** `benchmarks/arc3_sdk_agent.py`
**Method:** `_RemoteArcCompatEnv.reset()`
**Line:** 228

**Change FROM:**
```python
scorecard = session.post(
    f"{self.api_url}/api/scorecard/open",
    json={"game_ids": [self.game_id], "tags": ["k3d-sovereign-r0"]},
    timeout=30,
).json()
```

**Change TO:**
```python
scorecard = session.post(
    f"{self.api_url}/api/scorecard/open",
    json={"tags": ["k3d-sovereign-r0"]},
    timeout=30,
).json()
```

The SDK's `Swarm.open_scorecard()` sends ONLY `{"tags": [...]}`. No `game_ids`.

---

## Fix 3 (MEDIUM): Remove explicit Content-Type header

**File:** `benchmarks/arc3_sdk_agent.py`
**Method:** `_RemoteArcCompatEnv._ensure_session()`
**Lines:** 198-201

**Change FROM:**
```python
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
}
```

**Change TO:**
```python
headers = {
    "Accept": "application/json",
}
```

The SDK does NOT set Content-Type explicitly. The `requests` library sets it automatically
when using `json=` parameter. Having it in the session headers AND in the automatic
header could cause issues with some servers.

---

## Fix 4 (LOW): Match SDK RESET payload exactly

**File:** `benchmarks/arc3_sdk_agent.py`
**Method:** `_RemoteArcCompatEnv.reset()`
**Lines:** 233-237

**Change FROM:**
```python
payload = session.post(
    f"{self.api_url}/api/cmd/RESET",
    json={"card_id": self.card_id, "game_id": self.game_id, "reasoning": "K3D ARC R0 init"},
    timeout=30,
).json()
```

**Change TO:**
```python
payload = session.post(
    f"{self.api_url}/api/cmd/RESET",
    json={"card_id": self.card_id, "game_id": self.game_id},
    timeout=30,
).json()
```

The SDK RESET does NOT include reasoning. Keep it clean — match the SDK exactly.

---

## Fix 5 (LOW): Simplify step() alias resolution

The current step() tries multiple URL candidates for each action (ACTION3, MOVE_LEFT, LEFT).
The SDK always uses `/api/cmd/{action.name}` — i.e., `/api/cmd/ACTION3`. No aliases.

Remove the alias resolution loop. Post to `/api/cmd/{action_name}` directly:

```python
def step(self, action: Any, data: dict[str, Any] | None = None):
    session = self._ensure_session()
    action_name = _action_name(action).strip().upper()
    
    # Build payload matching SDK protocol exactly
    payload: dict[str, Any] = {"game_id": self.game_id}
    if self.guid:
        payload["guid"] = self.guid
    if data:
        payload.update({k: v for k, v in dict(data).items() if v is not None})
    
    print(f"[ARC3] step: POST /api/cmd/{action_name} payload_keys={list(payload.keys())}")
    
    http_response = session.post(
        f"{self.api_url}/api/cmd/{action_name}",
        json=payload,
        timeout=30,
    )
    
    if http_response.status_code >= 400:
        body = http_response.text[:500]
        print(f"[ARC3] step HTTP {http_response.status_code}: {body[:200]}")
        raise RuntimeError(
            f"remote_api_step_failed action={action_name!r} "
            f"status={http_response.status_code} body={body[:200]!r}"
        )
    
    response = http_response.json()
    
    # Log action_input.id to verify server recognized the action
    action_input = response.get("action_input", {})
    action_id = action_input.get("id", -1) if isinstance(action_input, dict) else -1
    print(f"[ARC3] step response: action_input.id={action_id} state={response.get('state')!r}")
    
    # ... rest of response processing unchanged ...
```

The key diagnostic: after this fix, `action_input.id` should be 3 (not 0) when we send
ACTION3. If it's still 0, there's another protocol issue to investigate.

---

## Fix 6 (HIGH): Match SDK close_scorecard payload exactly

**File:** `benchmarks/arc3_sdk_agent.py`
**Method:** `_RemoteArcCompatEnv.close()`

The SDK's close_scorecard sends: `{"card_id": "<card_id>"}` — this matches our code.
No change needed here. ✓

---

## SDK Reference: Full Protocol Summary

From `arc-agi-3` v0.0.1 source:

| Endpoint | Payload | When |
|----------|---------|------|
| `POST /api/scorecard/open` | `{"tags": [...]}` | Once per session |
| `POST /api/cmd/RESET` | `{"game_id": "ls20", "card_id": "<id>"}` | Start each game |
| `POST /api/cmd/ACTION1-5` | `{"game_id": "ls20", "guid": "<guid>"}` | Each move (simple) |
| `POST /api/cmd/ACTION6` | `{"game_id": "ls20", "guid": "<guid>", "x": N, "y": N}` | Click (complex) |
| `POST /api/cmd/ACTION7` | `{"game_id": "ls20", "guid": "<guid>"}` | Undo/Reset level |
| `POST /api/scorecard/close` | `{"card_id": "<id>"}` | End session |
| `GET /api/scorecard/{card_id}/{game_id}` | — | Check score |
| `GET /api/games` | — | List available games |

Headers (set on Session):
```
X-API-Key: <from ARC_API_KEY env>
Accept: application/json
```
Content-Type: NOT set explicitly (requests adds it via json= param).

Response for each action: `FrameData` with:
- `game_id`, `frame` (64×64×3 grid), `state` (NOT_PLAYED/NOT_FINISHED/WIN/GAME_OVER)
- `score`, `action_input` (echoes back action with id), `guid`, `available_actions`

**Key validation:** `action_input.id` in the response should match the action sent.
If we send ACTION3, response should have `action_input.id = 3`. If it returns 0, the
server didn't register our action.

---

## SDK Reference: GameAction Enum Values

From `_structs.py`:
```
RESET   = 0 (SimpleAction: game_id only)
ACTION1 = 1 (SimpleAction: Move Up)
ACTION2 = 2 (SimpleAction: Move Down)
ACTION3 = 3 (SimpleAction: Move Left)
ACTION4 = 4 (SimpleAction: Move Right)
ACTION5 = 5 (SimpleAction: Interact/Perform)
ACTION6 = 6 (ComplexAction: Click at x,y)
ACTION7 = 7 (SimpleAction: Undo/Reset Level)
```

---

## SDK Reference: Agent.main() Loop

From `_agent.py:92-108`:
```python
def main(self) -> None:
    self.timer = time.time()
    while (
        not self.is_done(self.frames, self.frames[-1])
        and self.action_counter <= self.MAX_ACTIONS
    ):
        action = self.choose_action(self.frames, self.frames[-1])
        if frame := self.take_action(action):
            self.append_frame(frame)
        self.action_counter += 1
    self.cleanup()
```

The first frame is `FrameData(score=0)` with state=NOT_PLAYED. The agent's
`choose_action()` should return RESET for this state. After RESET, the state
becomes NOT_FINISHED and the agent starts choosing movement actions.

---

## Verification Steps

After applying fixes 1-5, run:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/arc3_sdk_agent.py --game ls20 --max-steps 5
```

**Expected output (CRITICAL CHECK):**
1. `action_input.id=0` for RESET (correct — RESET is id 0)
2. `action_input.id=3` for ACTION3 (THIS is what proves the fix works)
3. Online scoreboard shows `Actions: 5` (or close — RESET might count differently)

If `action_input.id` is still 0 for non-RESET actions after removing `card_id`,
the issue is in cookies or session management. Report this immediately.

```bash
# Also run the autonomous loop to test retry
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/arc3_sdk_agent.py --game ls20 --autonomous --max-attempts 3 --max-steps 100
```

---

## Future: Python 3.12 Environment for Official SDK

The `arc-agi-3` package requires Python >=3.12. System has Python 3.12.9 and 3.13.12.
k3d-cranium has Python 3.10.18 (too old).

For Phase 2, create a Python 3.12 conda env that can import `arc-agi-3` directly and
subclass `Agent`. This would give us:
- Automatic protocol compliance (no need to reverse-engineer HTTP)
- FrameData/GameAction Pydantic models with validation
- Recorder/Playback for debugging sessions
- Swarm orchestration for multi-game runs

But the immediate fix (remove `card_id` from step()) is more urgent and doesn't need
a new env. Do that first.

---

## Report Back

Write `TEMP/CODEX_TO_CLAUDE_ARC3_PROTOCOL_FIX_REPORT_2026-04-09.md` with:

1. Fix 1 applied: `card_id` removed from step() payload (yes/no)
2. Fix 2 applied: `game_ids` removed from scorecard open (yes/no)
3. Fix 3 applied: Content-Type removed from session headers (yes/no)
4. Fix 4 applied: reasoning removed from RESET (yes/no)
5. Fix 5 applied: alias loop simplified (yes/no)
6. **CRITICAL: action_input.id values in 5-step test run** — list each step's action_input.id
7. Online scoreboard after test: Played=?, Actions=?, Levels=?
8. If action_input.id is STILL 0 for all actions, paste the full HTTP request/response log
9. `echosys_ingest` tmux session still alive (tmux ls)
