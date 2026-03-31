# Codex — Phase E.42: Movement Budget, Lives System, and Level 2 Layout Knowledge

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** IMMEDIATE — these implicit rules are why level 2 isn't completing

---

## The Problem

E.41b fixed re-perception (the agent now sees level 2 fresh — varied actions
instead of 47× Up). But the agent still can't COMPLETE level 2 because it
doesn't know critical game mechanics that are only conveyed through the UI
(frame pixels), never stated as text instructions.

Daniel + Gemini's analysis reveals THREE systems the agent is blind to:

1. **Movement Budget** — the yellow bar is a finite move counter
2. **Lives** — red squares are lives, lost when moves run out
3. **Strategic Reset** — resetting before moves run out saves a life

---

## New Knowledge: Movement Budget System

### What Gemini Describes

> "The Yellow Movement Bar: The entire block of contiguous yellow squares has
> shifted to the right within the track. There is now an empty space at the
> far left edge of the track... Because the yellow bar shifted right, it now
> sits flush against the red squares, eliminating the empty gap."

### What This Means

The bottom status bar is NOT decorative — it's the movement budget:

- **Yellow squares** = remaining moves. Each action consumes one unit.
- The bar **shifts right** as moves are spent (empty space grows on left).
- When ALL yellow squares are consumed → game auto-resets → lose a life.
- Initial yellow bar fills ~85% of the track (~25-30 moves per attempt).

### Stars

```
Star: "movement_budget_visual_bar"
  meaning: "A contiguous bar of colored units represents remaining actions.
           The bar shrinks from one end as actions are consumed. When fully
           depleted, the attempt ends with a penalty."
  visual_rpn: "STATUS_BAR YELLOW CONTIGUOUS_COUNT"
  behavior_rpn: "EACH_ACTION bar_length DECREMENT IF bar_length 0 = THEN attempt_fail"
  symlinks: → Grammar.resource_depletion, Reality.energy_conservation,
             movement_recharge_block, level_progression

Star: "movement_budget_depletion_penalty"
  meaning: "When the movement budget reaches zero without completing the
           objective, the current attempt fails: a life is lost and the
           level resets to its initial state."
  behavior_rpn: "IF budget 0 = AND NOT level_complete THEN life DECREMENT level RESET"
  symlinks: → lives_system, strategic_reset, level_progression

Star: "movement_budget_conservation"
  meaning: "Efficient pathfinding conserves movement budget. Wasted moves
           (backtracking, wrong direction) risk depleting the budget before
           the objective can be completed. Plan the full path BEFORE moving."
  behavior_rpn: "PATH_LENGTH budget < ASSERT optimal_path PREFER"
  symlinks: → Grammar.planning_before_execution, spatial_navigation_grid
```

---

## New Knowledge: Lives System

### What Gemini Describes

> "The Red Life Squares: There are now only two red squares remaining.
> They are still separated by a one-square empty gap. The space at the
> far right of the track, where the third red square previously sat,
> is now empty."

### What This Means

- **Red squares** = lives remaining (start with 3 per level).
- Separated by 1-square gaps (visually distinct from movement bar).
- Lose a life when movement budget depletes without solving.
- When all 3 lives lost → game over (presumably).
- Lives are PRECIOUS — the agent should avoid wasting them.

### Stars

```
Star: "lives_system"
  meaning: "Discrete life counters represent remaining attempts before
           permanent failure. Each failed attempt (budget depletion)
           removes one life. Zero lives = game over."
  visual_rpn: "STATUS_BAR RED SEPARATED_SQUARES COUNT"
  behavior_rpn: "IF attempt_fail THEN lives DECREMENT IF lives 0 = THEN game_over"
  symlinks: → movement_budget_depletion_penalty, strategic_reset,
             Grammar.finite_resource_management

Star: "lives_visual_indicator"
  meaning: "Red squares separated by gaps in the status bar indicate
           remaining lives. Count of red squares = attempts remaining."
  visual_rpn: "RED SQUARE GAP RED SQUARE GAP RED SQUARE"
  symlinks: → lives_system, visual_state_encoding
```

---

## New Knowledge: Strategic Reset

### Daniel's Observation

> "the model can reset the game before spending all actions without life penalty"

### What This Means

This is a CRITICAL strategic mechanic:

- **Reset (ACTION0)** before budget depletion → level restarts, NO life lost
- **Budget depletion** (all moves used) → level restarts, ONE life lost
- Therefore: if the agent realizes it's on a wrong path and can't reach the
  objective with remaining moves, it should RESET immediately rather than
  waste remaining moves and lose a life.

This requires the agent to:
1. **Estimate remaining budget** (count yellow squares or track actions)
2. **Estimate path length to objective** (spatial reasoning)
3. **Compare**: if path > budget → RESET now (save a life)

### Stars

```
Star: "strategic_reset"
  meaning: "When the remaining movement budget is insufficient to complete
           the objective, voluntarily resetting preserves a life. Resetting
           before depletion is strategically superior to running out of moves."
  behavior_rpn: "IF path_remaining budget_remaining > THEN ACTION0_RESET"
  symlinks: → movement_budget_conservation, lives_system, control_restart,
             Grammar.sunk_cost_avoidance

Star: "budget_sufficiency_check"
  meaning: "Before committing to a path, verify that the remaining movement
           budget is sufficient to reach the objective. If not, reset or
           find a shorter path (possibly through recharge blocks)."
  behavior_rpn: "path_cost budget_remaining <= IF proceed ELSE reset_or_reroute"
  symlinks: → strategic_reset, movement_recharge_block,
             Grammar.planning_before_execution
```

---

## New Knowledge: Yellow Flash = Out-of-Moves Signal

### Daniel's Observation

> "when you've spent all movements and not solved it, there's a yellow
> screen flash and the game is reseted"

### What This Means

Similar to the green flash for level completion:
- **Green flash** = level completed (advance to next)
- **Yellow flash** = out of moves (level resets, life lost)

Detection: same `_is_transition_frame()` logic but for YELLOW dominant color.

### Stars

```
Star: "screen_flash_failure"
  meaning: "When the display fills with a warm/yellow color, the current
           attempt has failed (budget depleted). The level will reset.
           This is the failure counterpart to the green success transition."
  visual_rpn: "FRAME_COLOR_RATIO 0.8 > AND dominant_color YELLOW ="
  behavior_rpn: "IF yellow_flash THEN attempt_failed life_lost level_reset_pending"
  symlinks: → screen_transition_uniform_color, movement_budget_depletion_penalty,
             lives_system

Star: "screen_flash_color_semantics"
  meaning: "Different flash colors encode different state transitions:
           green = success (level complete), yellow = failure (budget depleted).
           The color of the flash IS the outcome signal."
  behavior_rpn: "IF flash GREEN THEN success ELIF flash YELLOW THEN failure"
  symlinks: → screen_transition_uniform_color, visual_state_encoding,
             Grammar.color_semantics
```

---

## Level 2 Layout Knowledge (From Gemini)

### Structure: Inverted "E" (Three-Legged Bridge)

```
Level 2 Layout (schematic):

    ┌────────────────────────────────┐
    │  TOP BRIDGE (horizontal path)  │
    └──┬──────────┬──────────┬───────┘
       │          │          │
   LEFT LEG   MID LEG   RIGHT LEG
       │          │          │
  [Yellow □]      │     ┌────┴────┐
       │          │     │  bulky  │
  ┌────┴────┐     │     │  room   │
  │ Target  │  [Char]   │ [Cross] │
  │ Room    │  (O+B)    │         │
  │ [⌜ blue]│           │[Yellow]□│
  └─────────┘           └─────────┘

Reference box (bottom-left): Blue "L" shape
Status bar (bottom-right): Yellow movement bar + 3 red life squares
Level indicator (top-right): "LEVEL 2 / 7"
```

### Key Differences from Level 1

| Element | Level 1 | Level 2 |
|---------|---------|---------|
| Layout | Simple rooms | Inverted "E" with 3 legs |
| Target shape | (unknown) | Corner bracket `⌜` |
| Reference shape | (unknown) | "L" shape |
| Yellow blocks | None | 2 hollow yellow squares (recharge) |
| Total levels | Unknown | 7 (shown as "LEVEL 2 / 7") |
| Complexity | Direct path | Multi-leg navigation required |

### Game Elements in Level 2

1. **Character** (middle leg bottom): Orange top + blue bottom, 2×3 block
2. **White cross** (right leg): 5-square plus sign — changes character's
   key shape when walked over
3. **Target room** (left leg bottom): Contains blue `⌜` shape — the DOOR
   that opens when character shape matches
4. **Reference box** (bottom-left UI): Shows blue "L" — this is the
   CURRENT key shape of the character
5. **Yellow squares** (2 locations): Hollow 3×3 yellow outlines —
   movement recharge blocks (restore movement budget)
6. **Status bar** (bottom-right): Yellow movement bar + red life squares

### Solving Level 2 (Conceptual Path)

1. Character starts at middle leg bottom
2. Navigate UP the middle leg to the top bridge
3. Navigate RIGHT along the top bridge to the right leg
4. Navigate DOWN the right leg to the white cross
5. Step on the white cross → character shape changes
6. Check if shape matches target `⌜` — if not, may need color/shape blocks
7. Navigate back UP, then LEFT to the left leg
8. Navigate DOWN to the target room → door opens if shape matches
9. Enter target room → level complete

**Critical**: The path is LONG. Movement budget (~25-30 moves) must cover
the full traversal. The yellow recharge blocks exist precisely because the
path is too long for one budget. The agent MUST step on recharge blocks
during traversal.

---

## What This Means for the Reference Box

The reference box (bottom-left) shows the CURRENT state of the character's
"key" — the shape/color pattern that must match the door's lock. This is
a LIVE indicator, not static decoration.

```
Star: "reference_box_current_state"
  meaning: "A bordered reference box in the UI shows the current state of
           the character's key properties (shape, color). This is a LIVE
           indicator that updates when the character transforms. The character
           must achieve the state shown by the TARGET ROOM, not the reference
           box — the reference box shows CURRENT, the target room shows GOAL."
  visual_rpn: "BOTTOM_LEFT BORDERED_BOX BLUE_SHAPE"
  behavior_rpn: "reference_box = current_key_state; target_room = goal_state;
                 IF current_key_state = goal_state THEN door_opens"
  symlinks: → lock_key_pattern_match, visual_state_encoding,
             color_transform_block, shape_transform_block
```

Wait — correction. Looking at Level 2 more carefully:
- Reference box shows "L" shape
- Target room shows `⌜` (corner bracket)
- These are DIFFERENT shapes

So the reference box likely shows the TARGET shape (what the character needs
to become), and the target room shows the LOCK shape (what the door requires).
OR: the reference box = current character shape, target room = required shape.

Either way, the agent needs to:
1. Observe both shapes
2. Transform the character (via white cross and/or transform blocks)
3. Match the target room shape
4. Enter the target room

This is the `lock_key_pattern_match` star from E.39 in action.

---

## What Codex Must Implement

### Step 1: Add New Stars to House JSONL

Add ALL stars from this spec to `game_mechanics.jsonl`:
- `movement_budget_visual_bar`
- `movement_budget_depletion_penalty`
- `movement_budget_conservation`
- `lives_system`
- `lives_visual_indicator`
- `strategic_reset`
- `budget_sufficiency_check`
- `screen_flash_failure`
- `screen_flash_color_semantics`
- `reference_box_current_state`

These are MEANING stars — universal game concepts that apply to ANY game
with movement budgets and lives, not just ARC3/LS20.

### Step 2: Yellow Flash Detection

Extend the transition frame detection to distinguish flash colors:

```python
# In _classify_frame_state() or equivalent:
# Green dominant → level success transition
# Yellow dominant → budget depleted, attempt failed
# Both use the same >80% single-color threshold
```

### Step 3: Movement Budget Awareness

The agent should track action count per attempt:
- Count actions since last reset/level-start
- If approaching budget limit (~25-30) and objective is far → RESET (ACTION0)
- If a yellow recharge block is reachable → route through it first

### Step 4: Path Planning with Budget Constraint

Level 2 requires multi-segment navigation. The agent must:
1. Plan the FULL path before starting (not just next move)
2. Include recharge block stops in the path
3. Verify total path length ≤ budget before committing
4. If path too long → reset and try a different route

### Step 5: Reference Box + Target Room Perception

The agent must read TWO visual indicators:
1. Reference box (bottom-left) → current or target state
2. Target room (in level) → lock that must be matched

Compare them to determine what transformations are needed.

---

## Total Levels = 7

The "LEVEL 2 / 7" indicator reveals there are 7 levels total. This is
important for:
- Score estimation (score per level, 7 levels max)
- Difficulty progression expectation (later levels will be harder)
- Lives management (3 lives must cover all unsolved levels)

```
Star: "level_count_indicator"
  meaning: "A 'LEVEL X / N' display shows current level and total count.
           This establishes scope (how many challenges remain) and implies
           progressive difficulty."
  visual_rpn: "TEXT 'LEVEL' NUMBER '/' NUMBER"
  behavior_rpn: "current_level = X; total_levels = N; progress = X/N"
  symlinks: → level_progression, Grammar.ordinal_sequence
```

---

## Implicit Rules Summary

Daniel emphasizes: "those are all implicit rules from the UI (frame)."
The TRM must learn to READ the frame for game state, not receive it as
structured data. These rules are NEVER stated in text:

| Rule | How It's Encoded | How Agent Should Learn |
|------|-----------------|----------------------|
| Movement budget | Yellow bar shrinks | Observe bar change after each action |
| Lives remaining | Red square count | Count red squares in status bar |
| Budget depletion = life loss | Yellow flash + red square disappears | Contrastive: flash → count decreased |
| Reset saves lives | Reset → no red square change | Contrastive: reset vs depletion |
| Recharge blocks | Yellow hollow squares on map | Step on → observe bar growth |
| Reference = current state | Bottom-left box content | Observe changes after transform |
| Target = goal state | In-level bordered room | Compare with reference box |
| Level count | "LEVEL X / 7" text | Read text from frame (if perceptible) |

The long-term solution is WINE (E.35) where frames are proceduralized
through the Tablet and the TRM perceives them natively. For NOW, these
stars give the agent the KNOWLEDGE to interpret what it sees.

---

## Success Criteria

- [ ] All 11 new stars added to House JSONL and loaded at boot
- [ ] Yellow flash detected (not mistaken for gameplay)
- [ ] Agent tracks action count per attempt (budget awareness)
- [ ] Agent resets voluntarily when budget insufficient (strategic reset)
- [ ] Level 2 navigation shows multi-segment path planning
- [ ] At least 2 levels completed in one session
- [ ] Sleep-time consolidation records budget/lives patterns
- [ ] No new Python orchestration for budget tracking (knowledge-driven)
