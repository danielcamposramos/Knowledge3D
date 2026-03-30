# Codex — Phase E.36e: ARC3 LS20 — Complete Game Understanding

**Date:** 2026-03-30
**From:** Claude (Architecture Partner, with Daniel's visual descriptions)
**To:** Codex (Implementation)
**Priority:** IMMEDIATE — supersedes E.36b/c/d with full game knowledge
**Game:** `ls20-9607627b` (keyboard type, 7 levels)

---

## Full Game Structure (From Gemini Visual Analysis)

### The 64×64 Frame Contains:

```
┌──────────────────────────────────────────────┐
│                                              │
│         ┌─────────┐                          │
│         │ DOOR    │  ← Top Room (grey border)│
│         │ (blue   │     3×3 interior          │
│         │  shape) │     Blue bracket shape    │
│         └────┬────┘     = TARGET PATTERN      │
│              │                                │
│         ┌────┴────┐  ← Connecting Corridor    │
│         │ (3 wide)│     (grey walkable)       │
│         └────┬────┘                           │
│    ┌─────────┴──────────────────────┐        │
│    │                                │        │
│    │   ╋ WHITE    Main grey body    │        │
│    │   CROSS      (walkable area)   │        │
│    │   (switch)                     │        │
│    │              ┌──────┐          │        │
│    │              │CUTOUT│          │        │
│    │              │(dark)│          │        │
│    ├──────────────┘      └──┬───────┤        │
│    │ floor path    ██  CHARACTER ────│        │
│    └───────────────██──(orange+blue)─┘        │
│                                              │
│  ┌─────────┐  ┌─────────────────────────┐    │
│  │ KEY BOX │  │ PROGRESS BAR            │    │
│  │ (blue L)│  │ ████████████░░░█░█░█    │    │
│  │ = CURRENT│  │ yellow      gap red×3  │    │
│  │  PATTERN│  └─────────────────────────┘    │
│  └─────────┘                                 │
└──────────────────────────────────────────────┘
```

### Game Elements:

| Element | Visual | Purpose |
|---------|--------|---------|
| **Character** | 2×3 block: orange top row, blue bottom 2 rows | Player — moves with ACTION1-4 |
| **White Cross** | 5 squares in + pattern (upper-left of main body) | SWITCH — walk over to change key pattern |
| **Door** (top room) | Blue bracket shape in bordered 3×3 room | TARGET — must match key to pass through |
| **Key Box** (bottom-left) | Blue L-shape in bordered box | Shows CURRENT key pattern |
| **Grey area** | Light grey grid squares | WALKABLE terrain |
| **Dark area** | Near-black / dark grey | WALLS / not walkable |
| **Progress bar** | Yellow = progress, red = errors/lives | Status indicator |
| **Grid** | Faint dark grey overlay | Game uses discrete grid cells |

### Puzzle Logic:

1. **Character** starts at bottom of main body (near the colored block)
2. **Key pattern** (bottom-left box) shows current key shape
3. **Door pattern** (top room) shows required shape
4. Walking character over the **white cross** CHANGES the key pattern
5. When key matches door → door UNLOCKS
6. Move character through the unlocked door → **LEVEL COMPLETE**
7. Click to proceed to next level (7 levels total)

---

## Level 1 Solution (From Daniel)

Starting position: character is at bottom-center-right of the main body.

```
Step 1: Three LEFT  (ACTION3 × 3) — move character left
Step 2: Three UP    (ACTION1 × 3) — move character up to white cross
→ Character passes over white cross → key pattern changes → door unlocks

Step 3: One UP      (ACTION1 × 1) — move up toward corridor
Step 4: Three RIGHT (ACTION4 × 3) — move right along corridor
Step 5: Three UP    (ACTION1 × 3) — move up through corridor into door room
→ Character enters the door → LEVEL 1 COMPLETE
```

Total: 3 + 3 + 1 + 3 + 3 = **13 actions** (baseline is 21, so there's
either more nuance or the path is slightly longer).

---

## Protocol (Complete)

### 1. Open Scorecard + RESET

```
POST /api/scorecard/open
  {"game_ids": ["ls20-9607627b"], "tags": ["k3d-sovereign"]}
→ card_id

POST /api/cmd/RESET
  {"card_id": "...", "game_id": "ls20-9607627b"}
→ frame (START screen), state="NOT_FINISHED", available_actions, guid
```

### 2. Click START Button

The first frame shows the ARC PRIZE logo + "START" button (large white
rectangle centered on screen).

```
POST /api/cmd/ACTION6
  {"game_id": "...", "guid": "...", "x": 32, "y": 32}
→ frame (game board), available_actions should NOW include [1,2,3,4,5,6]
```

If (32,32) doesn't hit the button, find the white pixel cluster center
in the start screen frame.

**CRITICAL CHECK:** Verify `available_actions` now includes ACTION1-4.
If it does, the start screen is dismissed. If still only [6], the click
missed the button — try other coordinates.

### 3. Movement Phase

Send directional actions. Each action moves the character one grid cell:

```
POST /api/cmd/ACTION3  (Left)
  {"game_id": "...", "guid": "..."}  ← guid from PREVIOUS response
→ frame, guid, state, available_actions, levels_completed

POST /api/cmd/ACTION1  (Up)
  {"game_id": "...", "guid": "..."}
→ ...
```

### 4. Level Transition

When `levels_completed` increments, the game shows a transition screen.
`available_actions` may change to `[6]` (click to proceed).

```
POST /api/cmd/ACTION6
  {"game_id": "...", "guid": "...", "x": 32, "y": 32}
→ Next level frame, available_actions back to [1,2,3,4,5,6]
```

### 5. Win Condition

When all 7 levels complete: `state` = "WIN".

---

## Agent Strategy for Levels 2-7

The agent doesn't know each level's layout in advance. It must:

1. **Parse the 64×64 frame** to identify:
   - Character position (orange+blue block)
   - White cross position (5-square + pattern)
   - Door position (top room with blue shape)
   - Walkable area (light grey cells)
   - Walls (dark cells)

2. **Pathfind** from character to white cross (LED-A* on the grey grid)

3. **Pathfind** from white cross to door room

4. **Execute** the path as ACTION1-4 sequence

This is EXACTLY what the composed head pipeline does:
- **Frustum cull**: identify relevant elements in field of view
- **Morton octree**: spatial index of game elements
- **LED-A***: pathfind on the walkable graph
- **Nine-chain swarm**: parallel path evaluation

The local ARC3 20/20 proves the spatial reasoning works. The delta is:
- Parse 64×64 visual frames (element detection by color signature)
- Map pixel positions to grid coordinates
- Handle multi-step puzzle logic (key → cross → door)

---

## Frame Parsing Guide

### Color Signatures (From Gemini Description)

| Color | Meaning |
|-------|---------|
| Near-black / dark grey | Background / walls / not walkable |
| Light grey | Walkable floor |
| Medium grey | Room borders |
| White (bright) | The cross/switch (5 squares in + shape) |
| Orange | Character top row |
| Light blue | Character bottom rows + key shapes + door shape |
| Yellow | Progress bar fill |
| Red | Progress bar error markers |

### Detecting Character

Find the 2×3 block where:
- Top 2 cells = orange (specific color index, check palette)
- Bottom 4 cells = light blue
- Surrounded by grey (walkable) or dark (wall)

### Detecting White Cross

Find the + pattern:
```
  W
W W W
  W
```
Where W = white (brightest color, likely color index 15)

### Detecting Door Room

Find the bordered rectangle at top of map with blue shape inside.
The border is medium grey, interior is dark, blue shape is the target.

### Detecting Key Box

Bottom-left corner: bordered rectangle with blue shape inside.
Comparing key box shape to door shape tells you if the door is locked.

---

## Important: The Grid Is Coarse

The 64×64 pixel frame has a "graph-paper-like" grid overlay. The actual
game grid is MUCH coarser — probably 16×16 or 20×20 logical cells within
the 64×64 pixels. Each logical cell is several pixels wide.

To find the grid cell size: look at the character block (2×3 cells). Measure
its pixel dimensions. If it's 6×9 pixels, each grid cell is 3 pixels wide.

---

## Success Criteria

- [ ] LS20 START screen dismissed (ACTION6 click → movement available)
- [ ] `available_actions` includes ACTION1-4 after start
- [ ] Character moves with ACTION1-4 (frame changes detected)
- [ ] Level 1 completed (`levels_completed` increments from 0 to 1)
- [ ] At least 3 levels completed
- [ ] Agent generalizes to level 2+ (spatial parsing + pathfinding)
- [ ] Full log saved with per-action frames
