# Codex Direction: ARC-3 Action Emission Fix — 0 Actions Bug

**Date:** 2026-04-09
**Authority:** CLAUDE.md (sovereignty), KNOWLEDGEVERSE_SPECIFICATION.md
**Priority:** CRITICAL — agent is connecting to ARC-3 server but sending 0 actions
**Evidence:** Online scoreboard shows: Played=1, Actions=0, Levels=0/7

---

## Root Cause

The online scoreboard proves the agent opened the game (scorecard/reset succeeded) but sent
ZERO actions. The game loop in `run_level()` calls `decide_action()`, which raises an
Exception, which is caught, which breaks the loop before any `env.step()` is called.

The `decide_action()` chain is:
```
K3DAgent.decide_action()
  → K3DARC3Agent.choose_action()        # line 1515
      → self.tablet_boundary.submit()   # can return unexpected structure
      → tablet_result["emitted"]        # CRASH: KeyError if "emitted" not in result
  → Exception propagates to decide_action()
  → RuntimeError raised to run_level()
  → Caught by except → break → 0 env.step() calls
```

There are THREE bugs introduced when the living-memory `choose_action()` was added.

---

## Bug 1 (CRITICAL): Unsafe "emitted" key access

**File:** `benchmarks/arc_agi_3.py`
**Location:** Line ~1562 (second `choose_action()` method, around `tablet_result["emitted"]`)

```python
# BROKEN — raises KeyError if tablet_result is None or has no "emitted" key
tablet_result = self.tablet_boundary.submit(envelope, use_enriched=True)
emitted = dict(tablet_result["emitted"])
```

**Fix:**
```python
raw_tablet_result = self.tablet_boundary.submit(envelope, use_enriched=True)
tablet_result = dict(raw_tablet_result or {})
emitted = dict(tablet_result.get("emitted") or {})
task_result = dict(emitted.get("task_result") or {})
```

Also fix any other subscript access on `tablet_result` below this line that uses `["key"]`
instead of `.get("key")`. Every dict access on an external result must use `.get()`.

---

## Bug 2 (CRITICAL): Duplicate choose_action() — dead code shadow

**File:** `benchmarks/arc_agi_3.py`

There are now TWO `choose_action()` methods defined in `K3DARC3Agent`:
- **First** (line ~1209): old method, uses `self.kv.execute_task()` directly, has spatial
  plan fallback via LED-A*
- **Second** (line ~1515): new method added for living-memory, uses `tablet_boundary.submit()`

Python silently uses the SECOND one (it shadows the first). The first is unreachable dead
code — 300 lines that will never run.

**Fix:** Remove the FIRST `choose_action()` method (lines ~1209 to ~1513). Keep only
the second. The spatial plan logic from the first is valuable — wire it into the second
as described in Bug 3 below.

---

## Bug 3 (HIGH): _step_count never increments in new choose_action()

**File:** `benchmarks/arc_agi_3.py`, new `choose_action()` method (line ~1515)

The old method (removed by Fix 2) incremented `self._step_count` implicitly. The new
method does not. The WINE envelope is built with `step_count=int(self._step_count)` but
`_step_count` is always 0.

**Fix:** Add at the end of the new `choose_action()` method, before `return record`:
```python
self._step_count += 1
```

---

## Bug 4 (HIGH): No spatial plan in new choose_action()

The old `choose_action()` (now removed by Fix 2) computed a spatial path plan via
`self._spatial_path_plan()` using LED-A*, and used it as a GPU-computed fallback when
`execute_task()` returned no action. The new one has no equivalent.

When `emitted` contains no `action_name` or `action_index`, the new method defaults to
`ACTION_NAMES[0]` = ACTION1 ("Move Up") blindly on every step. This means the avatar
will just bang its head against the top wall forever.

**Fix:** After parsing `emitted` in the new `choose_action()`, add the spatial plan:

```python
# Spatial plan: GPU path to nearest mechanic target (switch → door → recharge)
spatial_plan = None
budget_snapshot = _movement_budget_snapshot(normalized_frame)
avatar_centroid = _avatar_centroid(normalized_frame) or _focus_centroid(normalized_frame)
valid_action_indices = _available_action_indices(available_actions)
if avatar_centroid is not None and frame_state != "transition":
    spatial_plan = self._spatial_path_plan(
        normalized_frame,
        avatar_centroid=avatar_centroid,
        budget_snapshot=budget_snapshot,
        valid_action_indices=valid_action_indices,
    )

# Parse action from tablet result
action_name = str(emitted.get("action_name") or "").strip().upper()
action_index = emitted.get("action_index")
if not action_name and isinstance(action_index, int) and 0 <= int(action_index) < len(ACTION_NAMES):
    action_name = ACTION_NAMES[int(action_index)]

# If tablet returned no action, use spatial plan (GPU path computed above)
if action_name not in ACTION_NAMES and spatial_plan is not None:
    action_index = int(spatial_plan["action_index"])
    action_name = ACTION_NAMES[action_index]
elif action_name not in ACTION_NAMES:
    action_name = ACTION_NAMES[0]  # ACTION1 absolute last resort
action_index = ACTION_NAMES.index(action_name)
```

The spatial plan calls `self._spatial_path_plan()` which uses LED-A* on the GPU
(via `self.kv.get_led_pathfinder()`). This is sovereign. The spatial plan IS the GPU
deciding the path — Python is only reading the result.

---

## Bug 5 (HIGH): choose_action() can raise and break the game loop

Even after fixing Bug 1, if `tablet_boundary.submit()` raises (network error, CUDA OOM,
import error, etc.), the exception propagates to `K3DAgent.decide_action()`, which
raises RuntimeError, caught by `run_level()` → break → 0 actions.

The issue: ANY unhandled exception in `choose_action()` causes the entire game to end
with 0 remaining actions. This is too fragile for a live competition.

**Fix:** Wrap the tablet call in `choose_action()` with a specific exception boundary:

```python
try:
    raw_tablet_result = self.tablet_boundary.submit(envelope, use_enriched=True)
    tablet_result = dict(raw_tablet_result or {})
    emitted = dict(tablet_result.get("emitted") or {})
    task_result = dict(emitted.get("task_result") or {})
except Exception as exc:
    # Tablet path failed. Log it. Use spatial plan if available, else ACTION1.
    # This is NOT Python reasoning — it is I/O keepalive when GPU path fails.
    import traceback
    print(f"[ARC3] tablet_boundary.submit() failed at step {self._step_count}: {exc}")
    traceback.print_exc()
    tablet_result = {}
    emitted = {}
    task_result = {}
```

The spatial plan (Bug 4 fix) then provides an action based on LED-A* path to the switch
or door. If that also fails, ACTION1 is sent. Crucially: the GAME LOOP CONTINUES. The
agent collects data. Sleep-time crystallizes rules. The autonomous retry loop can learn.

This is the I/O boundary responsibility: keep the channel open when GPU fails.

---

## Bug 6 (MEDIUM): _ensure_delegate() swallows init errors silently

**File:** `benchmarks/arc3_sdk_agent.py`

When `K3DARC3Agent.__init__()` fails, the error is stored in `self.policy_error` but
never printed. The game ends with 0 actions and no visible cause.

**Fix:** Add logging in `_ensure_delegate()`:
```python
except Exception as exc:
    message = str(exc)
    import traceback
    print(f"[ARC3] K3DARC3Agent init failed: {message}")
    traceback.print_exc()
    if self._allow_remote_compat and "sovereign_build_feed_missing" in message:
        self.policy_warning = "sovereign_build_feed_missing: proceeding with spatial primitives only"
        self.policy_error = None
    else:
        self.policy_error = message
    self._delegate = None
```

Also log in `decide_action()` when delegate is None:
```python
if delegate is None:
    reason = self.policy_error or self.policy_warning or "arc3_sovereign_delegate_unavailable"
    print(f"[ARC3] decide_action: delegate unavailable — {reason}")
    raise RuntimeError(str(reason))
```

---

## Consolidation: What the Fixed choose_action() Should Do

The new `choose_action()` (kept, with fixes applied) should have this structure:

1. **Parse frame** — `_normalize_grid`, `_frame_state`, `_movement_budget_snapshot`, etc.
2. **Build spatial plan** — `_spatial_path_plan()` via LED-A* (GPU, sovereign)
3. **Build WINE envelope** — `arc3_game_envelope()` with episode_context
4. **Try tablet path** — `tablet_boundary.submit()` in try/except with logging
5. **Parse action from result** — use `.get()` everywhere, fall through to spatial plan
6. **Apply loop-detection overrides** — `_blocked_actions_by_state`, `_blocked_repeat_count`
7. **Apply budget-critical reset** — `_should_force_reset()` when budget is critical
8. **Increment step_count** — `self._step_count += 1`
9. **Return record dict** — with `"action"`, `"action_index"`, all metadata

This matches the OLD `choose_action()` structure (line 1209) but routed through
`tablet_boundary.submit()` instead of `kv.execute_task()` directly.

The loop detection, blocked action avoidance, and forced reset logic from the OLD method
should be preserved — they are spatial computations (which cells changed → action was
blocked), not Python reasoning about the game's semantic meaning.

---

## Verification Steps

After the fix, run:

```bash
# 1. Verify 0 actions bug is gone
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/arc3_sdk_agent.py --game ls20 --max-steps 5
```

Report: `steps=5` AND `actions_sent=5` (not 0). The online scoreboard should show
`Actions: 5` (or close to it — some may be RESET which the server may count differently).

```bash
# 2. Run autonomous loop (short — just to see it doesn't crash)
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/arc3_sdk_agent.py --game ls20 --autonomous --max-attempts 2 --max-steps 50
```

Report: what actions were sent, did spatial plan fire (look for `spatial_plan_target` in logs).

```bash
# 3. Run tests
bash scripts/k3d_env.sh run -e k3d-cranium \
  python -m pytest -q tests/test_arc3_agent.py tests/test_arc3_autonomous_retry.py tests/test_arc3_living_memory.py
```

---

## Report Back

Write `TEMP/CODEX_TO_CLAUDE_ARC3_ACTION_FIX_REPORT_2026-04-09.md` with:

1. Bug 1 fixed: `tablet_result["emitted"]` → `.get("emitted")` (yes/no, line)
2. Bug 2 fixed: duplicate `choose_action()` removed (yes/no, lines removed)
3. Bug 3 fixed: `self._step_count += 1` in new `choose_action()` (yes/no)
4. Bug 4 fixed: spatial plan wired into new `choose_action()` (yes/no)
5. Bug 5 fixed: try/except around `tablet_boundary.submit()` with logging (yes/no)
6. Bug 6 fixed: `_ensure_delegate()` logs init failures (yes/no)
7. Tests: pass/fail count
8. **5-step smoke run result** — was step_count=5 AND did the online scoreboard update to show actions > 0?
9. **What exception was being raised** before the fix (from logs, now visible)
10. `echosys_ingest` tmux session still alive (tmux ls)
