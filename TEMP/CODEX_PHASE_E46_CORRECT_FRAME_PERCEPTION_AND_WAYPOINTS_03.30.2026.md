# Codex — Phase E.46: Correct Frame Perception and Waypoint Strategy

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL — E.45 bridge works, but perception is wrong

---

## What E.45 Achieved

The spatial bridge is LIVE and REAL:
- `program_type=spatial_frame_pathfinder` on 58/60 actions
- Non-zero confidence (0.316-0.424)
- Budget detection working (42 units, tracks depletion)
- Blocked-action detection working

But level 1 still doesn't solve because of TWO compounding errors:
1. Wrong color mapping → wrong object identification
2. Wrong target selection → pathfinds to wrong destination

---

## The Real Color Map (From Actual Frame Analysis)

I extracted the ACTUAL frame from `arc3_live_20260330_225843.jsonl` action 1
and rendered the complete board. Here's the ground truth:

```
COLOR MAP (verified from frame data):
  Color 0  = Blue (avatar component)     — 3 cells
  Color 1  = Orange (avatar component)   — 2 cells
  Color 3  = WALKABLE FLOOR (grey path)  — 894 cells ← THIS IS THE FLOOR
  Color 4  = BACKGROUND (dark/wall)      — 2609 cells
  Color 5  = REFERENCE BOX BORDER/UI     — 439 cells (rows 55-63, UI elements)
  Color 8  = RED LIFE INDICATORS         — 12 cells (rows 61-62, cols 56-63)
  Color 9  = DOOR / TARGET STRUCTURE     — 45 cells (multiple clusters)
  Color 11 = YELLOW MOVEMENT BAR         — 82 cells (rows 61-62, cols 14-54)
  Color 12 = RECHARGE BLOCK              — 10 cells (rows 45-46, cols 34-38)
```

### CRITICAL CORRECTIONS for arc_agi_3.py:

| What | Current assumption | Actual value |
|------|-------------------|-------------|
| Walkable floor | Unknown/mixed | **Color 3** (894 cells) |
| Background/wall | Unknown | **Color 4** (2609 cells) |
| Avatar | Colors {0,1} | Colors {0,1} ✓ correct |
| Door | Color 9 | Color 9 ✓ correct |
| Recharge | Color 12 | Color 12 ✓ correct |
| Switch/cross | Colors {11,15} | **NO COLOR 15 EXISTS. Color 11 = movement bar (UI!)** |
| Movement bar | Colors {3,11} | Color 11 only (rows 61-62) ✓ partially right |
| Lives | Color 8 | Color 8 ✓ correct |
| Reference box | Color 5+9 | Colors 5+9 (rows 55-63, cols 1-10) |
| UI area | Bottom 3 rows | **Rows 55-63** (9 rows, not 3) |

### The Switch/Cross Color Error

`_select_mechanic_target()` defines:
```python
("switch", {11, 15}, 5, 16, 0),  # ← WRONG: color 11 is the movement bar!
```

Color 11 = 82 cells of YELLOW MOVEMENT BAR in the status area (rows 61-62).
The agent identifies the movement bar as a "switch target" and tries to
pathfind toward it. This is why `target=switch` appeared in earlier runs.

There is NO white cross (color 15) in the Level 1 frame. The white cross
may appear in later levels or use a different color.

---

## The Actual Level 1 Layout

```
LEVEL 1 BOARD (from real frame data):

         cols: 13          32    40         55
              ┊            ┊     ┊          ┊
    row 8:    ·············#########·········    TARGET ROOM border
    row 9-15: ·············#+++++++#·········    TARGET ROOM interior (color 5+9)
              ·············#++DDD++#·········    Door shape inside (color 9)
              ·············#++++D++#·········
              ·············#++D+D++#·········
              ·············#+++++++#·········
    row 16:   ·············#########·········    TARGET ROOM border
    row 17-24:···············#####···········    CORRIDOR (5 wide, color 3)
                             ┊   ┊
    row 25-29:·#########################################·    MAIN BRIDGE (full width)
    row 30-39:·###############·····####################·
              ·#######A#######·····####################·    LEFT SECTION: avatar here
              ·######aAA######·····####################·    Avatar at ~(32, 20)
              ·#######a#######·····####################·
              ·###############·····####################·    GAP (cols 30-34, color 4)
    row 40-44:······#####··········####################·    LEGS
    row 45-46:······###############RRRRR###############·    RECHARGE (color 12)
    row 47-49:······###############DDDDD###############·    BOTTOM DOOR (color 9)
```

### Level 1 Solution Path

The avatar is at approximately (32, 20) in the LEFT section.
The TARGET is the room at top (rows 8-16, cols 32-40).

**Correct path:**
1. Avatar → move RIGHT to reach the main bridge (row 25-29)
2. On the bridge → move RIGHT to cross the gap (past col 34)
3. At the corridor entrance → move UP through the corridor (cols 33-37)
4. Enter the target room at top

**The agent is going to the WRONG door:**
The bottom structure (rows 47-49) also contains color 9 (door).
`_select_mechanic_target` picks the NEAREST door cluster. The bottom
door at (47-49, 34-38) is closer to the avatar than the top room door
at (11-13, 35-37). So the agent pathfinds DOWN instead of UP.

But the bottom door is NOT the objective — it's the reference/lock
structure (corresponding to the reference box in the status bar). The
ACTUAL objective is the target room at the TOP.

---

## The Two Problems to Fix

### Problem 1: Color Classification

Update `_select_mechanic_target()` and walkable mask:

```python
# CORRECT color sets:
WALKABLE_COLORS = {3}          # Grey floor
BACKGROUND_COLORS = {4}       # Dark wall
AVATAR_COLORS = {0, 1}        # Blue + orange
UI_AREA_MIN_ROW = 55          # Rows 55-63 are UI, not game board

# Target identification:
# Door/target room: color 9 clusters, but need to distinguish
#   TOP cluster = actual target room
#   BOTTOM cluster = reference structure (NOT target)
# Recharge: color 12 (correct)
# Switch/cross: NOT present in level 1 (remove {11,15} assumption)
```

### Problem 2: Target Selection Strategy

The agent must NOT just pathfind to the nearest door. It must identify
the OBJECTIVE:

**For LS20 specifically:**
- The target room is ENCLOSED (surrounded by color 3 border + color 5/9 interior)
- The reference structure at bottom is also enclosed but larger and lower
- The target room is ABOVE the main bridge; the reference is BELOW

**Universal heuristic (not LS20-specific):**
- The target room typically has a BORDER (walkable color forming a rectangle)
  with a SMALL interior containing the door pattern
- The reference box is in the STATUS BAR area (rows 55+) — exclude it
- Among remaining door clusters, prefer the one that:
  1. Is ABOVE the avatar (game typically progresses upward)
  2. Is ENCLOSED in a bordered room
  3. Is REACHABLE via the walkable graph

**Even simpler: spatial ordering**
Among all color-9 clusters NOT in the UI area (row < 55):
- The one with the SMALLEST average row (highest on screen) is likely
  the target room (games typically place objectives "up" or "far")
- The one closest to the avatar is likely the mechanism/reference

---

## What Codex Must Implement

### Step 1: Fix Walkable Mask

```python
# In the walkable board construction:
# Walkable = color 3 ONLY (not 5, not 9, not 0/1)
# Exclude UI area (rows >= 55)
walkable = set()
for r in range(min(55, len(frame))):  # exclude status bar
    for c in range(len(frame[r])):
        if frame[r][c] == 3:
            walkable.add((r, c))
        # Avatar cells are also walkable (the avatar IS on the floor)
        if frame[r][c] in {0, 1}:
            walkable.add((r, c))
```

### Step 2: Fix Target Selection

```python
# Find all door clusters (color 9) NOT in UI area
door_clusters = find_clusters(frame, color=9, max_row=55)

# Among door clusters, select the TOPMOST one as objective
# (smallest average row = highest on screen)
target_cluster = min(door_clusters, key=lambda c: c.avg_row)
target_centroid = target_cluster.centroid

# The bottommost door cluster is the reference/mechanism
# (ignore it for pathfinding)
```

### Step 3: Remove Wrong Switch Target

Delete or correct the switch definition:
```python
# WRONG: ("switch", {11, 15}, 5, 16, 0)
# Color 11 is the movement bar, not a switch!
# Color 15 doesn't exist in the frame!
# Remove switch from target_specs entirely for now
```

### Step 4: Ensure Avatar Cells Are Walkable

The avatar (colors 0,1) sits ON the walkable floor. When building the
walkable mask, these cells must be included — otherwise the pathfinder
can't start from the avatar's position.

### Step 5: Exclude UI Area from Game Board

Rows 55-63 contain the status bar (reference box, movement bar, lives).
These should NEVER be part of the walkable graph or target search. The
existing `_movement_budget_snapshot()` and `_lives_remaining()` correctly
read from these rows, but the walkable mask and target search must
exclude them.

---

## Verification: What Level 1 Solution Should Look Like

With correct perception, the pathfinder should produce:

```
Avatar at (32, 20)
Target room door at (11, 36) — topmost door cluster
Walkable path exists: (32,20) → right to bridge → right across gap →
                      up through corridor → into target room

Estimated path length: ~35-40 steps
Budget: 42 units → sufficient

Action sequence: Right×12 (to col 32+), Up×20 (through corridor to room)
≈ 32-35 actions to reach target room
```

This is achievable in one budget (42 units). Level 1 should solve
without recharge blocks or resets.

---

## Level 2 Preview (For After Level 1 Works)

Level 2 has the inverted-E layout (from Gemini's description):
- Three legs descending from a top bridge
- Yellow recharge blocks on two legs
- White cross on the right leg
- Target room on the left leg

For level 2, the waypoint sequence matters:
1. Navigate to recharge block (refill budget)
2. Navigate to white cross (change character shape)
3. Navigate to target room (if shape matches, door opens)

But level 2 is a LATER concern. Fix level 1 first — it requires
ZERO waypoint sequencing, just correct target identification.

---

## Success Criteria

- [ ] Walkable mask uses color 3 (verified: 894 cells)
- [ ] UI area (rows 55+) excluded from game board
- [ ] Avatar cells (colors 0,1) included in walkable set
- [ ] Switch target ({11,15}) removed — color 11 is movement bar
- [ ] Door target selection picks TOPMOST cluster, not nearest
- [ ] Pathfinder finds path from avatar (32,20) to target room (11,36)
- [ ] Level 1 solved in ~35-40 actions (within budget of 42)
- [ ] No Left-Right oscillation (path goes consistently toward target)
- [ ] Level 2 re-perception fires after level 1 completes
- [ ] At minimum: level 1 completed. Ideal: level 2 attempted meaningfully
