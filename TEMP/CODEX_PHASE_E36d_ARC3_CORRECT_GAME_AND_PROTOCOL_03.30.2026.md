# Codex — Phase E.36d: ARC3 Live — Correct Game Selection + Protocol

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** IMMEDIATE — replaces E.36b/E.36c (wrong game was being tested)

---

## The Mistake

We've been testing against `r11l-aa269680` which is tagged **"click"** — a
click-only puzzle game where `available_actions=[6]` is CORRECT behavior.
No movement actions will ever appear for this game.

Daniel's description (gameboy with D-pad, "three left, three up") matches
game **`ls20-9607627b`** which is tagged **"keyboard"** — an arrow-key
navigation game.

Evidence: The Gemini description of Daniel's browser screenshot shows `1s20`
(= LS20) in the top-left status pill of the console interface.

---

## Game Type Reference (From `GET /api/games`)

| game_id | title | tags | interaction |
|---------|-------|------|-------------|
| ls20-9607627b | LS20 | keyboard | Arrow keys (ACTION1-4) |
| r11l-aa269680 | R11L | click | Mouse clicks (ACTION6 x,y) |
| m0r0-dadda488 | M0R0 | keyboard_click | Both arrows + clicks |
| ft09-0d8bbf25 | FT09 | (none) | Unknown — probe first |

**Tags determine the primary interaction model:**
- `keyboard` → After start, use ACTION1 (Up), ACTION2 (Down), ACTION3 (Left), ACTION4 (Right)
- `click` → All interaction through ACTION6 with (x,y) coordinates
- `keyboard_click` → Both movement and clicks

---

## Game Protocol for LS20 (Keyboard Type)

### Phase 1: Start Screen

After RESET, the game shows a START screen. `available_actions` likely
includes `[6]` (click) for clicking the START button.

**Action:** Send ACTION6 to click the START button.

The START button is a large white rectangle in the center of the screen.
From the Gemini description, it's centered on the screen. In the 64×64
frame, try clicking center: `x=32, y=32` (or wherever the white rectangle
pixels cluster).

To find it programmatically: look for the largest cluster of bright/white
pixels (color 15 or similar) in the frame after RESET.

### Phase 2: Movement

After clicking START, `available_actions` should change to include
`[1, 2, 3, 4, 5, 6]` or at least `[1, 2, 3, 4]`.

**Action mapping:**
```
ACTION1 = Up    (arrow up)
ACTION2 = Down  (arrow down)
ACTION3 = Left  (arrow left)
ACTION4 = Right (arrow right)
ACTION5 = Perform / Spacebar
ACTION6 = Click (with x,y)
ACTION7 = Undo
```

**Level 1 solution (from Daniel):**
1. Three LEFT: ACTION3, ACTION3, ACTION3
2. Three UP: ACTION1, ACTION1, ACTION1
3. → Character reaches the key/cross position, actuates the figure
4. One UP: ACTION1
5. Three RIGHT: ACTION4, ACTION4, ACTION4
6. Three UP: ACTION1, ACTION1, ACTION1
7. → Level 1 complete, `levels_completed` should increment

### Phase 3: Level Transition

After completing a level, the game may show a transition screen requiring
a click (ACTION6) to proceed to the next level. Then movement resumes.

Watch for `available_actions` changes:
- `[6]` only → click to proceed
- `[1,2,3,4,5,6]` → movement + click available

---

## Game Elements (From Daniel's Description)

- **Character**: Orange and blue square — the player piece
- **Blue picture (lower-left)**: The key position (what the character must reach)
- **Upper picture**: The door (where the character must pass through after key)
- **White cross**: A switch the character must pass through to change the
  door's state from locked to open

The puzzle logic: move character to white cross → door unlocks → move
character through door → level complete → click to proceed to next level.

---

## What Codex Must Do

### Step 1: Switch to LS20

```bash
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  env CUDA_VISIBLE_DEVICES=0 \
  python3 scripts/run_arc3_agent.py \
    --game-id ls20-9607627b \
    --max-actions 200 \
    --api-url https://three.arcprize.org
```

Use 200 max actions (LS20 baseline is [21, 123, 39, 92, 54, 108, 109] —
level 1 takes 21 actions baseline).

### Step 2: Handle Start Screen

After RESET:
1. Check `available_actions`
2. If `[6]` or ACTION6 is available: find START button in frame, click it
3. Verify `available_actions` changes to include movement (1-4)

### Step 3: Spatial Navigation

Once movement is available, the existing K3DARC3Agent spatial reasoning
should work — it's the same compute-delta-and-move logic that scored
20/20 locally:

1. Find character position in 64×64 frame (orange+blue square)
2. Find target position (key/cross/door based on puzzle state)
3. Compute direction: if target is left of character → ACTION3, etc.
4. Execute movement

### Step 4: Level Transition Detection

After each action, check:
- Did `levels_completed` change? → level complete
- Did `available_actions` change to click-only? → transition screen, click to proceed
- Did `state` change to "WIN"? → game won

### Step 5: Multi-Game Session

After LS20, also try keyboard-type and keyboard_click-type games:

```bash
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  env CUDA_VISIBLE_DEVICES=0 \
  python3 scripts/run_arc3_session.py \
    --game-id ls20-9607627b \
    --game-id g50t-5849a774 \
    --game-id tr87-cd924810 \
    --max-actions-per-game 200 \
    --api-url https://three.arcprize.org
```

(g50t and tr87 are also keyboard-type games)

---

## Agent Adaptation for 64×64 Frames

The local ARC3 benchmark uses 8×8 grids with single-color cells. Live ARC3
uses 64×64 frames with multi-pixel visual elements. The agent needs to:

1. **Detect character**: Find the orange+blue square cluster in the frame.
   Not color 6 (magenta) — that was wrong. Look for the actual game character
   colors.

2. **Detect targets**: Find key elements (blue picture, white cross, door)
   by their color signatures.

3. **Map to spatial reasoning**: Convert pixel positions to relative directions
   (left/right/up/down) — same as local ARC3 but at pixel scale.

4. **Handle variable frame content**: Each level has different layouts.
   The spatial reasoning must be general, not hardcoded to level 1.

---

## Success Criteria

- [ ] LS20 game loads and START screen is dismissed (click → movement available)
- [ ] Movement actions (ACTION1-4) produce character position changes
- [ ] At least level 1 completed (`levels_completed` > 0)
- [ ] Level transition handled (click to proceed after level complete)
- [ ] Agent navigates using spatial delta (same logic as local ARC3 20/20)
- [ ] Logs saved to `/K3D/Knowledge3D.local/logs/`
