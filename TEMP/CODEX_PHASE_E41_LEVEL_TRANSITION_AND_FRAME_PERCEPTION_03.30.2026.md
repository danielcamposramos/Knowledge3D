# Codex — Phase E.41: Level Transition Perception + Multi-Level Play

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** IMMEDIATE — the post-level blocker prevents levels 2+

---

## The Problem

After completing level 1 (action 13), the agent collapses into repeating
"Move Up" for actions 14-60. It doesn't perceive that the game state changed.

**What actually happens on level completion:**

1. The game renders a **green screen transition** — full lime green background
   with a small dark grey cross/circle in the center
2. This is NOT the next level's game board — it's a TRANSITION SCREEN
3. The agent must DISMISS this transition (likely a click or any action)
4. THEN the level 2 game board appears with new layout and new mechanics

The agent sees the green transition frame but has no knowledge of what it
means. It keeps sending ACTION1 (Up) into a screen that expects a different
interaction.

---

## Daniel's Direction

> "when the level finishes, the screen closes in green and presents the next
> level — so it must be treated as a new game without the start — with no
> python orchestration! knowledge! and TRM should be able to think of other
> things that might occur (and save at sleeptime compute with contrastive
> learning)"

Key points:
1. Level transition = knowledge, not special Python code
2. The TRM should RECOGNIZE the green screen as a transition
3. The TRM should ANTICIPATE that game states change
4. Sleep-time should learn from the transition pattern

---

## What the Transition Screen Looks Like

From Gemini's visual analysis of the level 2 entry screen:

```
64×64 frame:
- ENTIRE background: bright lime green
- Faint grey grid overlay (same as gameplay)
- Bottom-left reference box: ABSENT
- Bottom status bar: ABSENT
- Center of screen: small 3×3 cross shape:

    .  G  .
    G  G  Lg     (G = dark grey, Lg = medium/lighter grey)
    .  G  .

- This is likely a loading/transition indicator
```

**Detection signal**: When the frame becomes PREDOMINANTLY ONE COLOR
(bright green) with no game elements visible, the game is in a transition
state. This is trivially detectable — if >80% of pixels are the same color
and that color is NOT the normal background, it's a transition frame.

---

## Solution: Knowledge-Based Frame State Classification

### New Stars for Frame State Recognition

These are MEANING stars — universal concepts about visual state transitions:

```
Star: "screen_transition_uniform_color"
  meaning: "When a display fills with a single dominant color and loses
           all prior visual elements, the system is transitioning between
           states (level complete, loading, scene change)"
  behavior_rpn: "FRAME_COLOR_RATIO 0.8 > IF transition_state THEN"
  symlinks: → Grammar.state_transition, visual_state_encoding

Star: "screen_transition_dismiss"
  meaning: "A transition screen typically advances when any input is
           provided — press a button or click to proceed"
  behavior_rpn: "IF transition_state THEN ACTION5 OR ACTION6 32 32"
  symlinks: → control_interact, level_progression

Star: "post_transition_new_context"
  meaning: "After dismissing a transition screen, the new frame represents
           a fresh context — re-perceive everything, assume nothing from
           the previous level carries over except learned mechanics"
  behavior_rpn: "IF prev_transition THEN RESET_SPATIAL_MODEL PERCEIVE_ALL"
  symlinks: → no_instruction_discovery, level_progression
```

### Frame State Classification Logic

This should be TRM perception, not Python if-statements. But the TRM needs
to KNOW what to look for. The knowledge above provides that.

**At minimum**, the agent's frame perception must classify each frame:

```
State 1: START_SCREEN
  Signal: frame contains large bright button-like region, text elements
  Action: Click center (ACTION6 x=32 y=32)

State 2: GAMEPLAY
  Signal: frame has walkable area (grey), character (orange+blue),
          interactive objects, reference box, status bar
  Action: Spatial reasoning → movement (ACTION1-4)

State 3: TRANSITION
  Signal: frame is >80% single color (green, or any uniform color),
          no gameplay elements visible
  Action: Dismiss (ACTION5 or ACTION6 center, or any action)

State 4: LEVEL_START (new level, post-transition)
  Signal: frame has gameplay elements but layout differs from previous
  Action: Re-perceive from scratch → new spatial reasoning
```

The distinction from State 2 to State 4 is that State 4 requires FULL
re-perception — the character position, targets, walkable area, and
puzzle elements are all NEW.

---

## What Codex Must Implement

### Step 1: Frame State Detection in K3DARC3Agent

Within `choose_action()`, before the spatial reasoning, classify the frame:

```python
# KNOWLEDGE-DRIVEN, not hardcoded:
# Check if frame matches "screen_transition_uniform_color" pattern
# The check itself is simple, but it's GROUNDED in Galaxy knowledge

def _classify_frame_state(self, frame):
    # Count dominant color ratio
    # If >80% single color and NOT normal background → TRANSITION
    # If gameplay elements present → GAMEPLAY
    # If start-screen elements → START_SCREEN
```

This is a THIN classification — the actual logic is in the stars. The
classification function maps frame features to star-matching conditions.

### Step 2: Transition Handling

When frame state = TRANSITION:
1. Send ACTION5 (perform/spacebar) to dismiss
2. If no change, send ACTION6 at center
3. If no change, send any movement action
4. After dismissal, re-enter GAMEPLAY state with fresh perception

### Step 3: Level Context Reset

When transitioning from TRANSITION → LEVEL_START:
- Clear any cached character position from previous level
- Clear any cached target positions
- Re-perceive the entire frame from scratch
- Identify NEW game elements (may include objects not seen in level 1)

### Step 4: Sleep-Time Contrastive Learning

After each game session, the consolidation should learn:

**Positive pairs (correct behavior):**
- "green uniform frame" + "send dismiss action" → state advanced
- "gameplay frame" + "spatial movement toward target" → progress

**Negative pairs (incorrect behavior):**
- "green uniform frame" + "send ACTION1 repeatedly" → no progress (47 wasted actions!)
- "gameplay frame" + "random movement" → no progress

Store these as contrastive observations for the sleep-time consolidation
to strengthen correct state-action mappings.

---

## Level 2 Specifics (From Daniel + Gemini)

Level 2 introduces NEW mechanics not present in level 1:
- **Yellow recharge block**: restores movement points
- **The layout is different**: must re-perceive

The agent should:
1. Dismiss the green transition
2. Perceive the level 2 layout (character, targets, new block types)
3. Identify the yellow block as "movement_recharge_block" (from E.40 knowledge)
4. Plan path that includes recharge stops if movement is limited

Level 2 status info from Gemini: the reference box and status bar are ABSENT
on the transition screen, but should reappear on the actual level 2 gameplay
frame.

---

## Broader Principle: Anticipatory Reasoning

Daniel said the TRM should "think of other things that might occur." This
means the agent shouldn't just react to frames — it should ANTICIPATE:

- "I completed a level → transition screen is likely next"
- "Transition dismissed → new layout with possibly new mechanics"
- "New colored blocks I haven't seen before → explore their effect"
- "Movement limited → need recharge block before running out"

This anticipatory reasoning comes from the Grammar Galaxy rules and the
game knowledge corpus (E.40). The TRM navigates to
`level_progression` → `screen_transition_dismiss` → `post_transition_new_context`
as a LEARNED sequence, not as hardcoded Python.

---

## Success Criteria

- [ ] Green transition screen detected (not mistaken for gameplay)
- [ ] Transition dismissed within 1-3 actions (not 47 wasted UP commands)
- [ ] Level 2 gameplay frame perceived fresh (new layout recognized)
- [ ] At least 2 levels completed in one session
- [ ] Sleep-time consolidation records transition pattern
- [ ] No new Python if/elif for frame classification (knowledge-driven)
- [ ] Stars for frame state transitions added to House JSONL
