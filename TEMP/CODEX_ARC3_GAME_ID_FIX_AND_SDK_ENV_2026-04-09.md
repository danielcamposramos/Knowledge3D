# Codex Direction: ARC-3 Game ID Fix + Python 3.12 SDK Environment

**Date:** 2026-04-09
**Authority:** CLAUDE.md (sovereignty), THREE_BRAIN_SYSTEM_SPECIFICATION.md §I/O adapters,
MEMORY_TABLET_SPECIFICATION.md §2 (architectural position)
**Priority:** CRITICAL — THIS IS THE DEFINITIVE ROOT CAUSE OF 0 ACTIONS
**Proven:** Claude ran live tests against three.arcprize.org with the official SDK.
All 5 actions registered. Frames changed. Avatar moved. Scorecard shows actions.

---

## Root Cause (DEFINITIVE — Proven Live)

**The game ID is `ls20-9607627b`, NOT `ls20`.**

Every game on the ARC-3 server has a full ID with hash suffix:
```
GET /api/games → [
  {"game_id": "ls20-9607627b", "title": "LS20", "tags": ["keyboard"]},
  {"game_id": "wa30-ee6fef47", "title": "WA30", "tags": ["keyboard"]},
  {"game_id": "bp35-0a0ad940", "title": "BP35", "tags": ["keyboard_click"]},
  ... (25 games total)
]
```

The RESET endpoint fuzzy-matches short names (`ls20` → `ls20-9607627b`), which is why
RESET always worked. But all ACTION endpoints (`/api/cmd/ACTION1-7`) require the EXACT
full game_id. With the short name, the server can't find the game session.

**Proof (live test, 2026-04-09 22:19 UTC):**

```
RESET with game_id="ls20-9607627b": HTTP 200, guid=bda003a0
ACTION3 with {game_id: "ls20-9607627b", guid: "bda003a0"}: HTTP 200
  → action_input.id=3 ← SERVER RECOGNIZED ACTION3!
  → frame hash changed: 06ff5221f0f4 → 310b7842aaee
  → 7 pixels changed ← AVATAR MOVED LEFT!
```

Five actions (LEFT LEFT LEFT DOWN INTERACT): all registered. Scorecard: `total_actions: 7`.

**card_id is NOT needed for non-RESET actions** when the full game_id is used. The
official SDK protocol (game_id + guid only) works perfectly.

**Scorecard URL (proof):**
https://three.arcprize.org/scorecards/6213c85b-4f46-4c16-8e97-0618d962b20a

---

## Architecture Grounding

Per THREE_BRAIN_SYSTEM_SPECIFICATION.md:
> Input (any format) → I/O adapter normalizes → kv.execute_task(query=...) →
> Result → I/O adapter formats for external consumer

The ARC-3 HTTP client is an **I/O adapter** (Python's role per CLAUDE.md). It wraps the
game server's REST API into the Memory Tablet's TabletEnvelope format. The fix is purely
at this I/O boundary — no sovereignty implications. TRM reasoning stays on GPU.

Per MEMORY_TABLET_SPECIFICATION.md §2, the Tablet sits at the top of the stack as the
**interface** through which the AI avatar perceives external worlds. The ARC-3 game is
an external world rendered as 64×64×3 frames through the Tablet's full-screen mode.

---

## Fix 1 (CRITICAL): Resolve full game_id from /api/games at startup

**File:** `benchmarks/arc3_sdk_agent.py`
**Class:** `_RemoteArcCompatEnv.__init__()`

At initialization, call `GET /api/games` to resolve the short game name to the full ID:

```python
def _resolve_full_game_id(self) -> str:
    """Resolve short game name (e.g. 'ls20') to full ID (e.g. 'ls20-9607627b')."""
    session = self._ensure_session()
    try:
        r = session.get(f"{self.api_url}/api/games", timeout=10)
        if r.status_code == 200:
            games = r.json()
            # Exact match first
            for g in games:
                if g["game_id"] == self.game_id:
                    return str(g["game_id"])
            # Prefix match (ls20 → ls20-9607627b)
            for g in games:
                if str(g["game_id"]).startswith(self.game_id):
                    print(f"[ARC3] Resolved game_id: {self.game_id!r} → {g['game_id']!r}")
                    return str(g["game_id"])
    except Exception as exc:
        print(f"[ARC3] Failed to resolve game_id: {exc}")
    return self.game_id  # Fall back to what was given
```

Call this in `__init__()` or at the start of `reset()`:
```python
self.game_id = self._resolve_full_game_id()
```

---

## Fix 2 (CRITICAL): Remove card_id from step() payload

With the correct game_id, card_id is NOT needed for actions. Match the SDK protocol:

**File:** `benchmarks/arc3_sdk_agent.py`
**Method:** `_RemoteArcCompatEnv.step()`

```python
def step(self, action, data=None):
    session = self._ensure_session()
    action_name = _action_name(action).strip().upper()

    payload = {"game_id": self.game_id}  # Full ID (e.g. ls20-9607627b)
    if self.guid:
        payload["guid"] = self.guid
    if data:
        payload.update({k: v for k, v in dict(data).items() if v is not None})

    r = session.post(
        f"{self.api_url}/api/cmd/{action_name}",
        json=payload,
        timeout=30,
    )

    if r.status_code >= 400:
        raise RuntimeError(f"step failed: HTTP {r.status_code} {r.text[:200]}")

    response = r.json()
    ai = response.get("action_input", {})
    print(f"[ARC3] {action_name}: action_input.id={ai.get('id', '?')} state={response.get('state')}")

    # ... rest of response processing unchanged ...
```

---

## Fix 3 (HIGH): Match SDK scorecard and RESET protocol

**Scorecard open** — SDK sends `{"tags": [...]}` only, no `game_ids`:
```python
scorecard = session.post(
    f"{self.api_url}/api/scorecard/open",
    json={"tags": ["k3d-sovereign-r0"]},
    timeout=30,
).json()
```

**RESET** — Use full game_id, no reasoning:
```python
payload = session.post(
    f"{self.api_url}/api/cmd/RESET",
    json={"card_id": self.card_id, "game_id": self.game_id},
    timeout=30,
).json()
```

**Session headers** — Match SDK exactly:
```python
headers = {
    "Accept": "application/json",
    # NO Content-Type — requests adds it via json=
}
if self._api_key:
    headers["X-API-Key"] = self._api_key
```

---

## Fix 4 (HIGH): New Python 3.12 arc3-sdk environment

A Python 3.12 venv has been created at `/K3D/Knowledge3D.local/envs/arc3-sdk/` with
the official `arc-agi-3` package. This env should be used for the WINE adapter layer.

**Path:** `/K3D/Knowledge3D.local/envs/arc3-sdk/`
**Python:** 3.12.9
**Packages:** arc-agi-3 v0.0.1, requests, pydantic, numpy, pillow

The K3D sovereign engine (TRM, Galaxy, kernels) stays in `k3d-cranium` (Python 3.10,
CUDA 12.4, CuPy). The arc3-sdk env is for the I/O adapter ONLY.

**Two options for wiring:**
- **Option A (recommended):** Run the I/O adapter as a subprocess from k3d-cranium,
  communicating via stdin/stdout JSON. Simple, clean boundary.
- **Option B:** Port the HTTP protocol fix to k3d-cranium's `_RemoteArcCompatEnv` (no
  SDK import needed — just fix the game_id and payload). This doesn't need 3.12.

**I recommend Option B for now** — the fix is just:
1. Call `/api/games` to get full game_id
2. Use full game_id in all requests
3. Don't send card_id with non-RESET actions

This requires ZERO new dependencies. The arc3-sdk env is available for future use
(Pydantic models, recording, swarm orchestration) but isn't needed for the immediate fix.

---

## Fix 5 (MEDIUM): Remove alias resolution loop

The SDK posts directly to `/api/cmd/{ACTION_NAME}`. No aliases needed. Remove the
candidate loop in `step()` and post to one URL.

---

## SDK Protocol Reference (Proven Working)

| Step | Endpoint | Payload | Notes |
|------|----------|---------|-------|
| 1 | `GET /api/games` | — | Get full game IDs |
| 2 | `POST /api/scorecard/open` | `{"tags": [...]}` | Get card_id |
| 3 | `POST /api/cmd/RESET` | `{"card_id": "<id>", "game_id": "ls20-9607627b"}` | Get guid + frame |
| 4 | `POST /api/cmd/ACTION3` | `{"game_id": "ls20-9607627b", "guid": "<guid>"}` | Move left! |
| 5 | `POST /api/cmd/ACTION5` | `{"game_id": "ls20-9607627b", "guid": "<guid>"}` | Interact! |
| 6 | `POST /api/scorecard/close` | `{"card_id": "<id>"}` | End session |

Headers: `X-API-Key` + `Accept: application/json` (on Session, auto-managed cookies).

**Verification:** `action_input.id` in response MUST match the action sent:
- ACTION3 → `action_input.id = 3`
- ACTION5 → `action_input.id = 5`
- If `action_input.id = 0`, the action was NOT processed.

---

## All Available Games (25 total)

From `GET /api/games` (2026-04-09):
```
ka59-9f096b4a (KA59)      r11l-aa269680 (R11L, click)
g50t-5849a774 (G50T)      m0r0-dadda488 (M0R0)
tr87-cd924810 (TR87)      s5i5-a48e4b1d (S5I5)
sk48-41055498 (SK48)      su15-4c352900 (SU15)
sp80-0ee2d095 (SP80)      ls20-9607627b (LS20, keyboard)
ar25-e3c63847 (AR25)      sb26-7fbdac44 (SB26)
vc33-9851e02b (VC33)      tn36-ab4f63cc (TN36, click)
cd82-fb555c5d (CD82)      wa30-ee6fef47 (WA30, keyboard)
tu93-2b534c15 (TU93)      lp85-305b61c3 (LP85)
sc25-f9b21a2f (SC25)      dc22-4c9bff3e (DC22)
lf52-271a04aa (LF52)      re86-4e57566e (RE86)
ft09-0d8bbf25 (FT09)      bp35-0a0ad940 (BP35, keyboard_click)
cn04-65d47d14 (CN04)
```

Tags: `keyboard` (directional), `click` (x,y coordinates), `keyboard_click` (both).

---

## Verification Commands

**Option B (k3d-cranium, after fixing game_id):**
```bash
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/arc3_sdk_agent.py --game ls20-9607627b --max-steps 5
```

**Expected:**
- `action_input.id` = 3, 3, 3, 2, 5 (not 0)
- Scorecard shows `total_actions: 5` or more
- Frame hashes change between steps

**Using arc3-sdk env directly (for validation):**
```bash
ARC_API_KEY=$(cat /K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt) \
  /K3D/Knowledge3D.local/envs/arc3-sdk/bin/python /tmp/arc3_sdk_working_test.py
```

**Scorecard (already proven):**
https://three.arcprize.org/scorecards/6213c85b-4f46-4c16-8e97-0618d962b20a

---

## SUPERSEDES Previous Protocol Specs

This spec supersedes:
- `CODEX_ARC3_PROTOCOL_FIX_2026-04-09.md` (card_id hypothesis — incomplete)
- `CODEX_ARC3_STEP_CRASH_AND_PROBE_SPEC_2026-04-09.md` (probe with wrong game_id)

The following specs remain valid and should be applied AFTER this fix:
- `CODEX_ARC3_ACTION_EMISSION_FIX_2026-04-09.md` (bugs 1-6 in choose_action)
- `CODEX_ARC3_LIVING_MEMORY_SPEC_2026-04-09.md` (episode Galaxy)
- `CODEX_ARC3_LS20_GAME_KNOWLEDGE_SPEC_2026-04-09.md` (game mechanics stars)

---

## Report Back

Write `TEMP/CODEX_TO_CLAUDE_ARC3_GAME_ID_FIX_REPORT_2026-04-09.md` with:

1. Fix 1: full game_id resolution from /api/games (yes/no)
2. Fix 2: card_id removed from step() (yes/no)
3. Fix 3: scorecard/RESET protocol matched (yes/no)
4. **action_input.id values for 5-step test** — must be non-zero
5. **Online scorecard total_actions** — must be > 0
6. **Scorecard URL** — paste it
7. Frame hashes — do they change between steps?
8. Which env was used (k3d-cranium or arc3-sdk)?
9. `echosys_ingest` tmux session still alive (tmux ls)
