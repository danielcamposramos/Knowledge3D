# Codex — Phase E.44: Level 2 Pathology Deep Fix

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL — this is the complete diagnosis of why level 2 never completes

---

## Log Analysis: The Exact Cycle

From `arc3_live_20260330_203706.jsonl` (98 actions):

```
CYCLE 1 (actions 1-28):
  [001-013] Level 1: Left×3 Up×4 Right×3 Up×3 → levels=1 ✓
            click_reason=transitional_live_script:ls20:level_0
  [014]     Level 2: Move Right (fresh=True) → target=recharge
  [015-027] Level 2: Move Left ×13 → target=switch (drifting into wall)
  [028]     RESET → levels=0 (ENTIRE game reset)

CYCLE 2 (actions 29-56): IDENTICAL to cycle 1
CYCLE 3 (actions 57-84): IDENTICAL to cycle 1
CYCLE 4 (actions 85-98): Same start, stopped by user
```

Every cycle: script solves L1 → drift left in L2 → reset → replay L1.

---

## Root Cause Analysis: FOUR Compounding Failures

### Failure 1: Level 1 Is a Hardcoded Script, Not Reasoning

```python
# arc_agi_3.py line 18-21
LIVE_TRANSITIONAL_ACTION_SCRIPTS = {
    ("ls20-9607627b", 0): [2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0],
}
```

Level 1 is solved by replaying a recorded action sequence. The GPU is
NEVER consulted. Confidence = 0.000 for ALL 13 actions. This means:
- The agent has ZERO learned spatial navigation ability
- No reasoning patterns are being built for level 1
- Sleep-time consolidation has nothing meaningful to strengthen
- Level 2 is the FIRST time the agent attempts real reasoning

**Fix**: Remove `LIVE_TRANSITIONAL_ACTION_SCRIPTS` entirely. The agent
must solve level 1 through actual GPU reasoning. If it can't solve level 1
without a script, it certainly can't solve level 2.

### Failure 2: RESET in Level 2+ Resets the ENTIRE Game

When the agent sends RESET at action 28:
- `levels_completed` goes from 1 → 0
- The game returns to level 1's initial state
- The agent must re-solve level 1 before attempting level 2 again

This is catastrophic for multi-level play. The agent burns ~13 actions
re-solving level 1 after every reset, and then has the EXACT SAME
level 2 experience (because nothing was learned between attempts).

**Fix**: The agent must NEVER send RESET after level 1 is completed.
Once `levels_completed >= 1`, RESET is forbidden. The agent must either:
- Navigate the level correctly (the goal)
- Accept budget depletion and lose a life (level restarts, not game)
- Try different directions (exploration)

If `_should_force_reset()` triggers in level 2+, it should be OVERRIDDEN.
The cost of a full-game reset (re-solve all previous levels) far exceeds
the cost of losing one life (restart current level only).

### Failure 3: No Spatial Reasoning for Level 2

After level 1 completes, the GPU path returns:
```
program_type=transitional_io_decode
program_id=reasoning_arc_grid_transform_top1
result_action=None
```

`_derive_action_from_result` falls through all checks:
- No `action_name` → skip
- No `answer_index` → skip
- No `x,y` → skip
- `output_grid` → attempts delta comparison, fails or computes wrong delta
- **Default**: return `0` (Move Up)

But the ACTUAL action chosen (Move Left ×13) comes from
`_select_mechanic_target()` which finds a "switch" (color 11) component
to the LEFT of the avatar. The agent mindlessly walks toward the nearest
switch target until it hits a wall.

**The agent is NOT reasoning.** It's following a simple heuristic:
1. Find avatar centroid (colors 0+1)
2. Find nearest game mechanic target (switch > recharge > pattern > door)
3. Move toward it

This is pure reactive navigation, not the composed head pipeline.

**Fix**: The `_frame_to_query_text()` function converts the frame into
tokens that go to `kv.execute_task()`. The query must encode enough
spatial information for the GPU pipeline to produce a meaningful action.
Currently, it generates position tokens like "object right of goal move
left" — but the goal is wrong (it's chasing a switch, not the level
objective).

The deeper fix: the frame perception must identify:
1. The avatar position (orange+blue block)
2. The WALKABLE path (light grey squares on dark background)
3. The TARGET (door/target room with lock shape)
4. Obstacles (walls, gaps in the path)
5. Intermediate objectives (white cross for shape change, recharge blocks)

This is a spatial graph problem — exactly what Morton Octree + LED-A*
are designed for.

### Failure 4: Budget Detection Returns Nothing

`budget=?/?` on ALL 98 actions. `_movement_budget_snapshot()` looks for
colors {3, 11} in the bottom 2-3 rows but finds fewer than 8 cells.
This means the color mapping is wrong — the actual game frame uses
different color indices for the yellow movement bar.

Without budget detection:
- No `force_reset` is triggered by budget logic
- The RESET at attempt=14 comes from the GPU result itself (the
  Knowledgeverse returns `action_name=RESET` based on query context)
- Strategic reset is firing based on vibes, not data

**Fix**: Inspect actual frame data from the log at the status bar
rows. Extract the real color values used for the movement bar and
life indicators. Update the color constants in:
- `_movement_budget_snapshot()`: `track_colors = {3, 11}` → actual values
- `_lives_remaining()`: `value == 8` → actual red value
- `_select_mechanic_target()`: color sets for switch/recharge/pattern/door

---

## The Real Path Forward

The fundamental issue isn't parameter tuning — it's that the agent has
NO spatial reasoning capability. Level 1 was solved by recording the
answer. Level 2 is the first real test and it fails completely.

### What Should Happen (Architectural Vision)

1. **Frame → Spatial Graph**: The 64×64 grid should be converted into
   a walkable graph (light grey = walkable, dark = wall). This IS what
   Morton Octree + LED-A* are for — spatial indexing and pathfinding.

2. **Identify Objects**: Color clustering should identify:
   - Avatar (orange+blue composite, ~6 cells)
   - Target room (bordered dark area with colored shape inside)
   - White cross (5-cell plus sign, color 15)
   - Recharge blocks (hollow yellow squares, 8 cells)
   - Reference box (bottom-left, not walkable)
   - Status bar (bottom, not walkable)

3. **Plan Path**: LED-A* from avatar to first objective (e.g., white
   cross) to second objective (target room), avoiding walls, routing
   through recharge blocks.

4. **Execute Path**: Convert path waypoints to ACTION1-4 sequence.

5. **Adapt**: If path fails (wall collision = frame unchanged), update
   walkable graph and replan.

### Immediate Fixes (E.44)

These won't give full spatial reasoning but will stop the pathological
cycle:

1. **Remove the hardcoded script** — agent must reason from frame
2. **Forbid RESET after level 1** — prevent catastrophic game reset
3. **Fix color constants** — enable budget/lives detection
4. **Add wall-collision detection** — if frame unchanged after move,
   that direction is blocked → try a different direction
5. **Add exploration diversity** — if the agent repeats the same action
   3+ times with no frame change, force a different action

### Medium-Term (E.45+)

1. **Walkable graph extraction** — convert frame to spatial graph
2. **LED-A* pathfinding on the walkable graph** — actual pathfinding
3. **Object identification** — semantic labeling of frame components
4. **Multi-objective planning** — route through recharge → cross → door

---

## Implementation Steps for Codex

### Step 1: Remove Hardcoded Script

Delete `LIVE_TRANSITIONAL_ACTION_SCRIPTS` dictionary (lines 18-21)
and the `_next_transitional_script_action()` method. Remove all
references to transitional scripts from `choose_action()`.

Level 1 must be solved through the same path that level 2 uses.
If level 1 breaks, that reveals the real gap — fix the reasoning,
don't paper over it with recordings.

### Step 2: Forbid RESET After Level 1

In `choose_action()`, after computing `force_reset` and before
creating the gpu_task:

```python
# NEVER reset after level 1 — it resets the ENTIRE game
if levels_completed >= 1:
    force_reset = False
# Also: if GPU returns RESET after level 1, override it
```

And in `_derive_action_from_result`, or after the result:

```python
if action_choice == RESET_ACTION_NAME and levels_completed >= 1:
    # RESET would undo all progress — forbidden
    # Instead: try a different direction
    action_choice = _exploration_fallback(frame, self.action_history)
```

### Step 3: Fix Color Constants from Real Frame Data

Extract a few frames from the log at different game states.
Check the actual color values in:
- Bottom 3 rows (status bar area)
- The movement bar region
- The life indicator region

Update `_movement_budget_snapshot()`, `_lives_remaining()`, and
`_select_mechanic_target()` with correct color mappings.

To inspect frame data:
```python
import json
with open('/K3D/Knowledge3D.local/logs/arc3_live_20260330_203706.jsonl') as f:
    row = json.loads(f.readline())
    frame = row['frame']
    # Check bottom rows for status bar
    for i in range(len(frame)-5, len(frame)):
        print(f"Row {i}: {sorted(set(frame[i]))}")
```

### Step 4: Wall-Collision Detection

If the frame is UNCHANGED after an action (the agent hit a wall):

```python
if self._last_frame is not None and normalized_frame == self._last_frame:
    # Frame didn't change — action was blocked (wall collision)
    # Record this direction as blocked
    # Try a different direction next time
```

Track blocked directions per position. When all 4 are blocked, the
agent is trapped (shouldn't happen on valid levels).

### Step 5: Exploration Diversity

If the agent repeats the same action N times:

```python
recent = [a['action_index'] for a in self.action_history[-3:]]
if len(recent) == 3 and len(set(recent)) == 1:
    # Same action 3× in a row — force exploration
    # Try perpendicular directions
```

This breaks the "Move Left ×13" pathology while real spatial
reasoning is being developed.

---

## Success Criteria

- [ ] No hardcoded action scripts (level 1 solved by reasoning)
- [ ] No RESET after level 1 (game progress preserved)
- [ ] Budget/lives detection working (actual values, not ?/?)
- [ ] Wall-collision detected (same frame = blocked)
- [ ] No 3+ identical consecutive actions without frame change
- [ ] Level 1 solved by GPU reasoning (may take more than 13 actions)
- [ ] Level 2 shows meaningful exploration (not just left-drift)
- [ ] At least 2 levels completed in one session (the goal)

---

## Architectural Note

The transitional script was a CRUTCH — it proved the API protocol works
but it masked the real gap: the agent cannot reason spatially about novel
frames. E.44 removes the crutch and forces the agent to develop real
capability. The score may temporarily drop (level 1 might take more
attempts) but this is the only path to multi-level play.

Daniel: "The TRM should be able to use knowledge to adapt by its own on
moving forward." — scripts are the opposite of adaptation.
