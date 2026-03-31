# Codex — Phase E.41b: Level Transition = Re-Perceive (Replaces E.41)

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** IMMEDIATE — this is the last blocker before multi-level play

---

## Correction: No Separate Transition Screen in the API

E.41 assumed the green screen was a gated transition requiring a dismiss
action. **That's wrong.** Codex's API investigation proves:

- `levels_completed` increments from 0→1 at action 13
- `state` stays `NOT_FINISHED`
- `available_actions` stays `[1, 2, 3, 4]` — movement is still available
- No guid change, no full_reset, no separate click required

**The green screen Daniel sees in the browser is a visual animation within
the normal frame stream. The API does not pause. The agent can keep sending
movement actions right through it.**

The green frames are just 1-2 frames in the stream where the game renders
a transition animation. The next frame after that IS the level 2 layout.

---

## The Actual Problem

After level 1 completes (action 13), the agent sends "Move Up" 47 times
straight. This happens because:

1. The agent's spatial reasoning computed "move up" as the correct action
   for level 1's endgame (approaching the door at the top)
2. After `levels_completed` incremented, the FRAME CHANGED to a new layout
3. But the agent didn't notice — it kept using stale reasoning from level 1
4. "Move Up" was the last good action, so it repeats indefinitely

---

## The Fix: Detect `levels_completed` Change → Reset Perception

### What Must Happen

When the agent detects that `levels_completed` has increased since the
last action:

1. **Acknowledge**: "Level complete — new level started"
2. **Reset spatial state**: Clear cached character position, target positions,
   walkable map, and any level-specific reasoning
3. **Re-perceive**: Analyze the current frame FROM SCRATCH as a brand new
   level (new layout, possibly new mechanics)
4. **Continue playing**: Use the fresh perception to compute the next action

This is the `post_transition_new_context` star from E.41 — but triggered
by the `levels_completed` delta, not by green pixel detection.

### Where to Implement

In `K3DARC3Agent.learn_from_outcome()` and `choose_action()`:

```python
# In learn_from_outcome():
if levels_completed > self._last_levels_completed:
    # Level completed! Signal fresh perception needed.
    self._needs_reperceive = True
    self._last_levels_completed = levels_completed

# In choose_action():
if self._needs_reperceive:
    # Don't reuse ANY cached spatial state from previous level.
    # Treat this frame as if it's the first frame of a new game.
    self._needs_reperceive = False
    # ... fresh perception of frame ...
```

### Handle Transition Frames

The frame immediately after level completion might be the green transition
animation (predominantly one color, no gameplay elements). The agent should:

1. Detect if current frame is "empty" (>80% single color = transition anim)
2. If so: send a neutral action (ACTION5/perform or any direction) to advance
3. On the NEXT frame: if it has gameplay elements, perceive it fresh

This is at most 1-2 wasted actions instead of 47.

---

## What the Agent Should Do for Level 2

After re-perceiving the level 2 frame:

1. **Find character**: Look for the orange+blue block in the new layout
2. **Find targets**: Identify key elements (cross/switch, door, new blocks)
3. **Identify new elements**: Yellow blocks (recharge), colored blocks, etc.
   Match these to E.40 game knowledge stars
4. **Plan path**: Character → switch → door (same pattern as level 1,
   but different layout and possibly new mechanics)
5. **Execute**: Send ACTION1-4 sequence

### New Mechanics in Level 2 (From Daniel)

- **Yellow recharge block**: "movement_recharge_block" star — step on it
  to restore movement budget
- The reference box and status bar may differ from level 1
- Layout is completely different — must be perceived fresh

---

## Broader Frame Perception Improvement

The agent currently uses `_frame_to_query_text()` which converts the frame
to text tokens with position/guidance information. This works for simple
grids but doesn't handle:

- Frame state changes (level transition)
- New element detection (blocks not seen before)
- Layout re-perception (character in a completely different position)

The fix for NOW is the `levels_completed` delta trigger. The fix LONG-TERM
is the WINE approach (E.35) where frames are proceduralized through the
Tablet and the TRM perceives them natively.

---

## Green Frame Detection (Bonus, Not Critical)

If you want to detect the transition animation frames specifically:

```python
from collections import Counter

def _is_transition_frame(frame):
    flat = [cell for row in frame for cell in row]
    counts = Counter(flat)
    dominant_color, dominant_count = counts.most_common(1)[0]
    ratio = dominant_count / len(flat)
    # If >80% single color, it's likely a transition animation
    return ratio > 0.80 and dominant_color != 0  # 0 = normal black background
```

This is a THIN detection (5 lines) grounded in the
`screen_transition_uniform_color` star. It's not critical for the fix —
the `levels_completed` delta is the primary trigger.

---

## Success Criteria

- [ ] Agent detects `levels_completed` increment → resets spatial state
- [ ] Transition frames handled gracefully (at most 1-2 extra actions)
- [ ] Level 2 frame perceived fresh (character + targets identified)
- [ ] At least 2 levels completed in one session (`levels_completed` ≥ 2)
- [ ] Action log shows meaningful movement in level 2 (not 47× Up)
- [ ] Sleep-time consolidation records level transition pattern
