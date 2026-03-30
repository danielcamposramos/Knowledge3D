# Codex — Phase E.36c: ARC3 Click Navigation — Understanding the Game Mechanic

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** IMMEDIATE — fixes the live ARC3 click-game blocker

---

## What Daniel Told Us

> "In the browser human version I can see a gameboy like visual interface with
> the game inside it and a 'Start' button with the instruction 'Press Start to
> Play' under it — this unlocks the real game where the first movements should
> be three left, three up"
>
> "the three left, three up is only part of the puzzle, this will move the
> character to a position where it actuates the figure that unlocks the next
> level, then one up, three right and three up (the first puzzle)"

---

## Frame Analysis (From Action #1 Log)

The 64×64 frame contains these visual elements:

```
Element 1: Top diamond (color 15, center ~x=39, y=21)
    - Outlined diamond shape at rows 18-24, cols 36-42
    - Could be: a target figure, an obstacle, or decoration

Element 2: Upper-left diamond (color 0 outline + color 15 center, at x=7, y=36)
    - Small diamond at rows 34-38, cols 5-9
    - This is likely the "figure that unlocks the next level"

Element 3: CHARACTER (color 6=magenta center, color 15 outline, at x=17, y=47)
    - Diamond at rows 45-49, cols 15-19
    - Color 6 is UNIQUE in the frame (1 pixel) — this IS the player

Element 4: Bottom-right target (color 3 cluster, center ~x=27, y=59)
    - Diamond at rows 57-61, cols 25-29
    - This might be the end position for level 1

Trail: color 1 diagonal line from (~9,38) to (~25,57)
    - Connects upper-left diamond to the area near bottom-right target
    - May show a path, or be a visual element
```

---

## The Click-Movement Mechanic

### What the Agent Is Doing WRONG

The agent clicks ON the character (color 6 at 17,47). Each click moves the
character DOWN-RIGHT along the trail toward the bottom target:

```
Click 1: (17,47) → character stays at (17,47) [initial/start click]
Click 2: (17,47) → character moves to (22,53) [moved toward bottom-right!]
Click 3: (22,53) → character stays at (22,53)
Click 4: (22,53) → character moves to (24,56)
Click 5: (24,56) → character stays at (24,56)
Click 6: (24,56) → character moves to (25,57) [approaching bottom target]
```

The character is gravitating toward the bottom-right target because the agent
clicks on it (or near it), and the game interprets this as "move toward click."

### What Daniel Says Is CORRECT

"Three left, three up" — the character at (17,47) needs to move:
- LEFT 3 times: toward smaller x values (toward x≈7)
- UP 3 times: toward smaller y values (toward y≈36)

This would bring the character from (17,47) to approximately (7,36) — which
is EXACTLY where the upper-left diamond (Element 2) is!

Then "one up, three right and three up" continues the puzzle path.

### The Click Mechanic (Hypothesis)

For click-only games, clicking at (x,y) means: **"move the character TOWARD
this position."** The game moves the character one step in the direction of
the click relative to the character's current position.

So to move LEFT: click at a position to the LEFT of the character.
To move UP: click at a position ABOVE (lower y) the character.

The exact mechanic could be:
a) Click anywhere left of character → move left one step
b) Click on a specific grid cell to the left → move there
c) Click relative position determines direction and distance

Given that "three left" is specified as discrete steps, it's likely (a) or (b):
clicking left of the character moves it one grid cell left.

---

## What Codex Must Do

### Step 1: Test the Direction-Click Hypothesis

Instead of clicking ON the character (color 6), click LEFT of the character:

```python
# Character is at (cx, cy) — find color 6 pixel
# To move LEFT: click at (cx - STEP, cy) where STEP = grid cell size
# To move UP: click at (cx, cy - STEP)

# From the diamond sizes (~5 pixels across), try STEP = 5 or STEP = 1
```

Try the sequence Daniel described for R11L level 1:
1. Click START (on color 6 or wherever the start action is)
2. Click LEFT of character, 3 times
3. Click ABOVE character, 3 times
4. → Character should now be at the upper-left figure
5. Click UP once
6. Click RIGHT of character, 3 times
7. Click UP of character, 3 times

### Step 2: Observe Frame Changes

After each click, log:
- Where color 6 (character) moved to
- Whether `available_actions` changed
- Whether `levels_completed` changed
- The full color frequency count (to detect game state changes)

### Step 3: Determine Grid Cell Size

The game's internal grid might not be 1:1 with the 64×64 pixels. Compare:
- How far color 6 moves per click (in pixels)
- Whether movement snaps to specific coordinates

If the character moves in increments (e.g., always 5 pixels), that's the
grid cell size.

### Step 4: Generalize the Navigation

Once the click mechanic is understood, the agent needs spatial reasoning:
- Find character position (color 6)
- Find target/goal position (identify which element to reach)
- Compute direction: delta_x, delta_y from character to target
- Click in the correct direction relative to character
- Repeat until character reaches target

This maps to the SAME spatial reasoning as the local ARC3 (20/20):
compute delta from current to goal, move in the larger delta direction first.

---

## Reference: Official Agent Templates

Check `/K3D/K3D_llama_cpp/datasets/ARC-AGI-3-Agents/agents/templates/` for
how the official random agent handles click games. The random agent likely
clicks at random positions within the frame — observe which random clicks
cause movement and in which direction.

Also check:
- `arcengine/base_game.py` — how the engine processes ACTION6 clicks
- Any R11L-specific game definition files in the engine package

---

## Architectural Note

This is EXACTLY the kind of spatial reasoning the composed head pipeline is
built for:
- Perceive: identify character (color 6) and targets (color clusters) in frame
- Navigate: compute spatial path from character to target (LED-A* / Morton)
- Decide: which direction to click
- Act: submit click with correct (x,y)

The 20/20 local ARC3 proves the spatial reasoning works. The gap is only
in the TRANSLATION from 64×64 pixel frames to the spatial reasoning — which
is the E.35 WINE layer's job.

---

## Success Criteria

- [ ] Character (color 6) moves LEFT when agent clicks left of it
- [ ] Character moves UP when agent clicks above it
- [ ] Level 1 of R11L solved: three left, three up → actuate figure
- [ ] `levels_completed` increments from 0 to 1
- [ ] Navigation generalized: spatial delta → click direction
