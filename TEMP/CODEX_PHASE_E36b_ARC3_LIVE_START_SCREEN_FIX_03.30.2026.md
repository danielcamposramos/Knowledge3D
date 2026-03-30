# Codex — Phase E.36b: ARC3 Live Start Screen Fix

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** IMMEDIATE — this is the live ARC3 blocker

---

## The Problem

Live ARC3 games (e.g., `r11l-aa269680`) have a visual start screen rendered
INTO the 64×64 game frame. The agent must click the correct pixel coordinates
to hit the "Start" button. After that, `available_actions` changes from `[6]`
(click-only) to include movement actions `[1,2,3,4,5,6]`.

Current attempts click at (17,47), (47,17), (32,32) — all return
`available_actions=[6]`, meaning the start button wasn't hit.

---

## Root Cause (From ARC3 Engine Source)

The ARC3 engine is at `/K3D/K3D_llama_cpp/datasets/ARC-AGI-3-Agents/` and the
arcengine package. Key facts:

### Frame Format

- Each frame is a **64×64 grid of integers 0-15** (16-color palette)
- NOT the small 8×8 grids used in our local benchmark
- The start screen is rendered as colored pixels within this 64×64 grid
- The "Start" button is a region of specific-colored pixels

### Action Protocol

```
GameAction enum (0-based):
  RESET   = 0  (restart level/game)
  ACTION1 = 1  (Up)
  ACTION2 = 2  (Down)
  ACTION3 = 3  (Left)
  ACTION4 = 4  (Right)
  ACTION5 = 5  (Perform/Interact)
  ACTION6 = 6  (Click at x,y — requires {"x": int, "y": int}, both 0-63)
  ACTION7 = 7  (Undo)
```

### Start Screen Handling

From the official random agent template:
```python
if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
    action = GameAction.RESET
```

After RESET, the game transitions to NOT_FINISHED. If `available_actions`
contains only `[6]`, the game is waiting for a click. The agent MUST:

1. **Read the frame** (64×64 grid)
2. **Find the clickable region** (the start button)
3. **Click the correct (x, y) coordinates within 0-63**

### How to Find the Start Button

The official multimodal agent template uses vision (sends frame as image to LLM).
But we can do it spatially:

**Strategy: Find the unique/prominent color region in the frame.**

The start button is typically rendered in a distinctive color (often color 6 =
magenta or another non-background color). To find it:

```python
# Given frame as 64x64 grid of ints 0-15:
# 1. Count occurrences of each color
# 2. Identify background color (most frequent)
# 3. Find non-background colored regions
# 4. The "button" is typically a rectangular cluster of a specific color
# 5. Click the center of that cluster
```

Alternative approach from the reference agent code:
- The frame contains a rendered "Start" text/button
- Look for the **centroid of the rarest non-background color cluster**
- Or scan for a specific pixel pattern (rectangular region)

### Critical: x,y Coordinate System

`x` = column (0-63, left to right)
`y` = row (0-63, top to bottom)

This may be different from our grid convention (row, col). Verify orientation.

---

## What Codex Must Do

### Step 1: Print the Frame on RESET

After the RESET call, LOG the actual 64×64 frame to understand what the
start screen looks like. Even a simple frequency count of colors will help.

```python
import collections
frame_2d = response_frame[-1]  # Last frame (64x64 grid)
color_counts = collections.Counter(cell for row in frame_2d for cell in row)
print(f"Frame colors: {color_counts}")
# Also print a downsampled view (every 8th pixel) to see structure
```

### Step 2: Find the Click Target

Analyze the 64×64 frame to find the start button:

```python
# Find non-background regions
background = max(color_counts, key=color_counts.get)  # Most common color
for y, row in enumerate(frame_2d):
    for x, color in enumerate(row):
        if color != background:
            # Track bounding box of each non-background color
            ...
# Click center of the most likely "button" region
```

### Step 3: Verify available_actions Changes

After clicking the start button, check if `available_actions` now includes
movement actions (1-4). If yes, the start screen is dismissed. If still
only `[6]`, the click missed — try another target.

### Step 4: Handle the Actual Game

Once movement actions are available, the existing `K3DARC3Agent.choose_action()`
logic should work — it's the same spatial reasoning that scored 20/20 locally,
just on 64×64 frames instead of 8×8.

---

## API Call Reference

All calls require:
- Header: `X-API-Key: <key>`
- Header: `Accept: application/json`
- Header: `Content-Type: application/json`
- Persistent `requests.Session` (AWS ALB session affinity cookies)

### RESET

```
POST /api/cmd/RESET
Body: {"card_id": "...", "game_id": "...", "reasoning": "K3D init"}
Response: {
    "frame": [[[int]×64]×64],  // 64×64 grid (may be nested in list)
    "state": "NOT_FINISHED",
    "available_actions": [6],   // Click-only on start screen
    "guid": "...",              // MUST send back with next action
    "levels_completed": 0,
    "win_levels": 6
}
```

### ACTION6 (Click)

```
POST /api/cmd/ACTION6
Body: {
    "game_id": "...",
    "guid": "...",             // From previous response
    "x": 32,                  // Column 0-63
    "y": 32,                  // Row 0-63
    "reasoning": {...}         // Optional
}
Response: same shape as RESET
```

### Movement (ACTION1-4)

```
POST /api/cmd/ACTION1  (Up)
POST /api/cmd/ACTION2  (Down)
POST /api/cmd/ACTION3  (Left)
POST /api/cmd/ACTION4  (Right)
Body: {
    "game_id": "...",
    "guid": "..."
}
```

---

## Reference: Official Agent Templates

The official ARC3 agent SDK is at:
`/K3D/K3D_llama_cpp/datasets/ARC-AGI-3-Agents/`

Key files:
- `agents/templates/random_agent.py` — simplest working agent (RESET + random actions)
- `agents/templates/multimodal.py` — uses LLM with frame-as-image
- `agents/agent.py` — base Agent class
- `agents/swarm.py` — multi-game orchestration

The official benchmarking harness is at:
`/K3D/K3D_llama_cpp/datasets/ARC-AGI-3-benchmarking/`

Key files:
- `src/arcagi3/game_client.py` — HTTP API client
- `src/arcagi3/schemas.py` — Pydantic schemas for FrameData

---

## Success Criteria

- [ ] Start screen frame is logged and analyzed (64×64 color map)
- [ ] Click target is found by frame analysis (not hardcoded coordinates)
- [ ] `available_actions` changes to include movement after start click
- [ ] At least one game progresses past start screen to actual gameplay
- [ ] Movement actions produce frame changes (position updates)
- [ ] Full game played with levels_completed > 0
