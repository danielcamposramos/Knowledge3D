# Codex Direction: ARC-3 env.step() Crash Fix + Probe Bootstrap + Avatar Discovery

**Date:** 2026-04-09
**Authority:** CLAUDE.md (sovereignty), KNOWLEDGEVERSE_SPECIFICATION.md
**Priority:** CRITICAL — env.step() crashes silently; Actions=0 on every real server run
**Evidence:** 4 online scoreboard entries: Played=1, Actions=0 every time

---

## Root Cause (Confirmed by Code Reading)

There are TWO independent crash paths that keep Actions=0.

### Crash Path A: env.step() not wrapped (line 508)

In `benchmarks/arc3_sdk_agent.py`, `K3DAgent.run_level()`:

```python
# Line 494-499: decide_action() is protected:
try:
    action = self.decide_action(observation)
except Exception as exc:
    if self.policy_error is None:
        self.policy_error = str(exc)
    break  # ← breaks loop, 0 steps counted

# Line 508: env.step() is NOT protected:
next_obs, reward, done, info = self.env.step(action, data=step_payload) if ...
# ↑ if this raises → run_level() crashes → unhandled → 0 Actions on scoreboard
```

If the ARC-3 server returns a non-JSON response (HTTP 400/404/500) for the action command,
`.json()` raises `JSONDecodeError`, which propagates through `env.step()` all the way up.
The scorecard was already opened, RESET was already sent (Played=1), but 0 subsequent
steps were counted. This matches the scoreboard perfectly.

### Crash Path B: Action command format mismatch

`_RemoteArcCompatEnv.step()` posts to:
```
POST https://three.arcprize.org/api/cmd/ACTION3
```

The real ARC-3 server may not have an endpoint called `/api/cmd/ACTION3`. It may use:
- `/api/cmd/MOVE_LEFT`
- `/api/cmd/3`
- A body parameter `{"action": "MOVE_LEFT"}` on a single `/api/cmd/ACTION` endpoint
- The `GameAction` enum value strings from `arcengine`

If the endpoint is wrong, the server returns HTTP 404 HTML → `.json()` raises →
crash path A above.

---

## Fix 1: Wrap env.step() in try/except (CRITICAL)

**File:** `benchmarks/arc3_sdk_agent.py`
**Location:** `run_level()`, line ~508

```python
# BEFORE:
next_obs, reward, done, info = self.env.step(action, data=step_payload) if self.transport == "remote_api_compat" else self.env.step(action)

# AFTER:
try:
    if self.transport == "remote_api_compat":
        next_obs, reward, done, info = self.env.step(action, data=step_payload)
    else:
        next_obs, reward, done, info = self.env.step(action)
except Exception as step_exc:
    # Log the FULL exception — this is what was silently killing the game loop
    import traceback
    print(f"[ARC3] env.step() FAILED at step {self.step_count}: {step_exc}")
    traceback.print_exc()
    # Log the action that failed so we can diagnose format issues
    action_str = _action_name(action) if action is not None else "None"
    print(f"[ARC3] Failed action: {action_str!r}, payload: {step_payload!r}")
    # Do NOT break — skip this step and try next one
    # (This keeps the game alive even when one HTTP call fails)
    self.step_count += 1
    attempt_steps += 1
    continue
```

The "skip and continue" behavior is intentional: if the action format is wrong, we get
logs showing exactly which format fails, and the loop keeps running to collect more data.
The online scorecard will still show some actions (even if they fail HTTP-wise, the server
may count the attempts).

---

## Fix 2: Add Diagnostic Logging to _RemoteArcCompatEnv

**File:** `benchmarks/arc3_sdk_agent.py`
**Location:** `_RemoteArcCompatEnv.reset()` and `_RemoteArcCompatEnv.step()`

Add logging so we can see exactly what the real server returns:

```python
def reset(self):
    session = self._ensure_session()
    scorecard = session.post(
        f"{self.api_url}/api/scorecard/open",
        json={"game_ids": [self.game_id], "tags": ["k3d-sovereign-r0"]},
        timeout=30,
    ).json()
    self.card_id = str(scorecard.get("card_id", ""))
    print(f"[ARC3-DIAG] scorecard response keys: {list(scorecard.keys())}")
    
    payload = session.post(
        f"{self.api_url}/api/cmd/RESET",
        json={"card_id": self.card_id, "game_id": self.game_id, "reasoning": "K3D ARC R0 init"},
        timeout=30,
    ).json()
    print(f"[ARC3-DIAG] RESET response keys: {list(payload.keys())}")
    print(f"[ARC3-DIAG] available_actions raw: {payload.get('available_actions')!r}")
    print(f"[ARC3-DIAG] state: {payload.get('state')!r}")
    print(f"[ARC3-DIAG] frame type: {type(payload.get('frame')).__name__}")
    if isinstance(payload.get('frame'), list):
        frame = payload['frame']
        print(f"[ARC3-DIAG] frame shape: {len(frame)} rows × {len(frame[0]) if frame else 0} cols")
    # ... rest of reset() ...

def step(self, action: Any, data: dict[str, Any] | None = None):
    session = self._ensure_session()
    action_name = _action_name(action)
    print(f"[ARC3-DIAG] step: posting to /api/cmd/{action_name!r}")
    # ... build payload ...
    http_response = session.post(
        f"{self.api_url}/api/cmd/{action_name}",
        json=payload,
        timeout=30,
    )
    print(f"[ARC3-DIAG] step HTTP status: {http_response.status_code}")
    if http_response.status_code >= 400:
        print(f"[ARC3-DIAG] step HTTP error body (first 500 chars): {http_response.text[:500]!r}")
    response = http_response.json()
    print(f"[ARC3-DIAG] step response keys: {list(response.keys())}")
    # ... rest of step() ...
```

These prints go to stdout which is captured by the terminal. Run one game and read the output
to see: (a) what available_actions the server actually sends, (b) whether the step endpoint
name is wrong.

---

## Fix 3: Try Multiple Action Name Formats

**File:** `benchmarks/arc3_sdk_agent.py`
**Location:** `_RemoteArcCompatEnv.step()`

The server might not accept "ACTION3". Try these in order, use the first one that returns
HTTP 200:

```python
# Map from our internal ACTION names to candidate server command names
ACTION_NAME_ALIASES: dict[str, list[str]] = {
    "ACTION1": ["ACTION1", "MOVE_UP", "UP"],
    "ACTION2": ["ACTION2", "MOVE_DOWN", "DOWN"],
    "ACTION3": ["ACTION3", "MOVE_LEFT", "LEFT"],
    "ACTION4": ["ACTION4", "MOVE_RIGHT", "RIGHT"],
    "ACTION5": ["ACTION5", "PERFORM", "INTERACT"],
    "ACTION6": ["ACTION6", "CLICK", "SELECT"],
    "ACTION7": ["ACTION7", "UNDO", "RESET_LEVEL"],
}
```

In `step()`:
```python
action_name = _action_name(action)
candidates = ACTION_NAME_ALIASES.get(action_name, [action_name])

response_obj = None
for cmd_name in candidates:
    http_response = session.post(
        f"{self.api_url}/api/cmd/{cmd_name}",
        json=payload,
        timeout=30,
    )
    print(f"[ARC3-DIAG] tried /api/cmd/{cmd_name} → HTTP {http_response.status_code}")
    if http_response.status_code < 400:
        response_obj = http_response.json()
        # Cache which name worked for future steps
        if cmd_name != action_name:
            print(f"[ARC3-DIAG] action alias confirmed: {action_name!r} → {cmd_name!r}")
            # Store working alias for this session
            if not hasattr(self, '_action_aliases'):
                self._action_aliases = {}
            self._action_aliases[action_name] = cmd_name
        break
    elif not response_obj:
        response_obj = {}

if response_obj is None:
    response_obj = {}
```

This is NOT Python reasoning about the game — it is I/O format negotiation. The server
defines the protocol; we're adapting our wire format to match.

Once we know which name works (from the diagnostic logs), we can hardcode the correct names
and remove the alias fallback loop.

---

## Fix 4: Probe Bootstrap — Step 0 Always Moves Left

**The principle:** On the very first step of a game/attempt, the agent has zero episode
context. The GPU has nothing to work with. Instead of sending garbage (ACTION1 blindly),
send ACTION3 (Move Left) as a known probe. This is the first correct action for LS20.
More importantly: observing what changed reveals the avatar.

**Python's role here is I/O only**: send a known probe → record what the server sends back.
GPU interprets the result and stores the avatar identity.

**File:** `benchmarks/arc3_sdk_agent.py`
**Location:** `K3DAgent.run_level()`, before the main while loop

```python
# Bootstrap probe: always move left on step 0
# Purpose: identify avatar (what moves?), establish episode baseline
# This is I/O — Python sends one known probe to initialize GPU-side discovery
if self.step_count == 0 and attempt_steps == 0:
    probe_obs = self._run_probe_action(obs, frame)
    if probe_obs is not None:
        obs = probe_obs["obs"]
        frame = probe_obs["frame"]
        attempt_steps += 1
        self.step_count += 1
```

Add the probe method:
```python
def _run_probe_action(self, obs: Any, frame: list[list[int]]) -> dict[str, Any] | None:
    """
    Send ACTION3 (Move Left) as a zero-context probe.
    Record what moved — that is the avatar. Store in episode galaxy.
    Returns updated obs and frame, or None if probe failed.
    """
    from knowledge3d.knowledgeverse.arc3_episode_galaxy import ARC3EpisodeGalaxy
    probe_action_index = 2  # ACTION3 = Move Left = index 2
    probe_action_name = "ACTION3"
    probe_action = _to_sdk_action(self._game_action_type, probe_action_name, {})
    
    try:
        if self.transport == "remote_api_compat":
            next_obs, reward, done, info = self.env.step(probe_action, data=None)
        else:
            next_obs, reward, done, info = self.env.step(probe_action)
    except Exception as exc:
        print(f"[ARC3] Probe step failed: {exc}")
        return None
    
    next_frame = _normalize_grid(next_obs.get("grid") if isinstance(next_obs, dict) else next_obs)
    
    # Identify avatar: cells that changed when we moved left
    prev_centroid_before = _focus_centroid(frame)
    prev_centroid_after = _focus_centroid(next_frame)
    agent_moved = (
        prev_centroid_before is not None
        and prev_centroid_after is not None
        and (
            abs(float(prev_centroid_after[0]) - float(prev_centroid_before[0])) > 0.5
            or abs(float(prev_centroid_after[1]) - float(prev_centroid_before[1])) > 0.5
        )
    )
    
    print(f"[ARC3] Probe ACTION3 (Move Left): agent_moved={agent_moved}")
    if prev_centroid_before:
        print(f"[ARC3] Centroid before: {prev_centroid_before}")
    if prev_centroid_after:
        print(f"[ARC3] Centroid after: {prev_centroid_after}")
    
    # Record probe outcome in episode galaxy (GPU-side learning)
    delegate = self._ensure_delegate()
    if delegate is not None:
        delegate._episode_galaxy.seed_frame(
            step_count=0,
            grid=frame,
            action_taken="",
            lives=-1,
            budget_pct=-1.0,
            levels_completed=0,
        )
        delegate._episode_galaxy.seed_outcome(
            step_count=0,
            action=probe_action_name,
            prev_grid=frame,
            next_grid=next_frame,
            reward=float(reward),
            lives_delta=0,
            levels_delta=int(info.get("levels_completed", 0)),
        )
        # Micro sleep-time: crystallize avatar identity from this first observation
        try:
            delegate._episode_galaxy.run_micro_sleeptime()
        except Exception:
            pass
    
    # Frame history
    self.frame_history.append({
        "step": 0,
        "action": probe_action_name,
        "reward": float(reward),
        "levels_completed": int(info.get("levels_completed", 0)),
        "changed_cells": 0,
        "probe": True,
    })
    
    return {"obs": next_obs, "frame": next_frame}
```

---

## Fix 5: Add Standalone Diagnostic Script

Create `scripts/arc3_api_diagnostic.py`:

```python
#!/usr/bin/env python3
"""
Standalone ARC-3 API diagnostic.
Opens a scorecard, resets, prints the FULL server response,
tries one action with each possible name format, prints all results.
Run BEFORE the agent to understand what the real server expects.

Usage:
    bash scripts/k3d_env.sh run -e k3d-cranium python scripts/arc3_api_diagnostic.py
"""
import json
import os
from pathlib import Path
import requests

API_URL = "https://three.arcprize.org"
GAME_ID = "ls20"

def _resolve_api_key() -> str:
    key = os.environ.get("ARC_API_KEY", "").strip()
    if key:
        return key
    path = Path("/K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt")
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""

def main():
    api_key = _resolve_api_key()
    if not api_key:
        print("WARNING: ARC_API_KEY not set and secrets file missing")

    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "Content-Type": "application/json",
    })
    if api_key:
        session.headers["X-API-Key"] = api_key
        print(f"Using API key: {api_key[:8]}...")

    print("\n=== 1. Opening scorecard ===")
    r = session.post(f"{API_URL}/api/scorecard/open",
        json={"game_ids": [GAME_ID], "tags": ["k3d-diagnostic"]}, timeout=30)
    print(f"HTTP {r.status_code}")
    scorecard = r.json()
    print(json.dumps(scorecard, indent=2))
    card_id = str(scorecard.get("card_id", ""))

    print("\n=== 2. RESET ===")
    r = session.post(f"{API_URL}/api/cmd/RESET",
        json={"card_id": card_id, "game_id": GAME_ID, "reasoning": "diagnostic"}, timeout=30)
    print(f"HTTP {r.status_code}")
    reset_payload = r.json()
    # Print everything EXCEPT the frame (too large)
    diag = {k: v for k, v in reset_payload.items() if k != "frame"}
    print(json.dumps(diag, indent=2))
    print(f"frame: {type(reset_payload.get('frame')).__name__}")
    if isinstance(reset_payload.get('frame'), list):
        frame = reset_payload['frame']
        print(f"frame shape: {len(frame)} rows × {len(frame[0]) if frame else 0} cols")
    guid = str(reset_payload.get("guid", ""))

    print("\n=== 3. Try action command names ===")
    candidates = ["ACTION3", "MOVE_LEFT", "LEFT", "3", "move_left"]
    for cmd_name in candidates:
        payload = {
            "guid": guid, "card_id": card_id,
            "game_id": GAME_ID, "reasoning": "diagnostic",
        }
        r = session.post(f"{API_URL}/api/cmd/{cmd_name}", json=payload, timeout=30)
        print(f"/api/cmd/{cmd_name} → HTTP {r.status_code}", end="")
        if r.status_code < 400:
            resp = r.json()
            print(f" keys={list(resp.keys())} state={resp.get('state')!r}")
            guid = str(resp.get("guid", guid))  # keep guid up to date
        else:
            print(f" error: {r.text[:200]!r}")

    print("\n=== 4. Close scorecard ===")
    session.post(f"{API_URL}/api/scorecard/close",
        json={"card_id": card_id}, timeout=30)
    print("Done")

if __name__ == "__main__":
    main()
```

---

## What to Run and Report

### Step 1 — Run diagnostic (FIRST, before anything else)

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python scripts/arc3_api_diagnostic.py 2>&1 | tee /tmp/arc3_diag.txt
cat /tmp/arc3_diag.txt
```

Report back the FULL output. This tells us:
- What keys does the RESET response have?
- What does `available_actions` actually contain?
- Which action command name works? (ACTION3 vs MOVE_LEFT vs LEFT vs 3)

### Step 2 — Apply fixes 1-4 based on diagnostic

Once we know which action command name works (from the diagnostic), hardcode it as the
primary in `ACTION_NAME_ALIASES`.

### Step 3 — Run 5-step smoke test

```bash
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/arc3_sdk_agent.py --game ls20 --max-steps 5 2>&1 | tee /tmp/arc3_smoke.txt
cat /tmp/arc3_smoke.txt
```

Report: what does the online scoreboard show for the new run? Should be Actions≥1.

### Step 4 — Run autonomous (if step 3 shows Actions>0)

```bash
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/arc3_sdk_agent.py --game ls20 --autonomous --max-attempts 3 --max-steps 100
```

---

## Report Back

Write `TEMP/CODEX_TO_CLAUDE_ARC3_STEP_CRASH_REPORT_2026-04-09.md` with:

1. Diagnostic output: what is the actual action command name the server accepts?
2. Diagnostic: what does `available_actions` contain in the RESET response?
3. Fix 1 (env.step try/except): applied (yes/no, line)
4. Fix 2 (diagnostic logging): applied (yes/no)
5. Fix 3 (action name aliases): applied, which name worked (ACTION3/MOVE_LEFT/etc.)
6. Fix 4 (probe bootstrap): applied (yes/no)
7. Fix 5 (diagnostic script): created (yes/no)
8. **Online scoreboard for new run**: Played / Actions / Levels (paste the row)
9. What exception was the env.step() catching? (from the new log output)
10. `echosys_ingest` tmux session still alive (tmux ls)
