# Codex — Phase E.43: Remove Action Cap — Game Drives Termination

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** IMMEDIATE — this is why level 2 hasn't been completed

---

## The Problem

The agent has an artificial `max_actions=80` cap that cuts the session
short. But the GAME has no hard action limit — yellow recharge blocks
restore the movement budget, so the player can keep going indefinitely
as long as they hit recharge blocks and don't lose all 3 lives.

The real termination signals from the API are:
- `state == "WIN"` → game completed (all 7 levels done)
- `state == "GAME_OVER"` → all lives lost
- `state` leaves `ACTIVE_STATES` for any other reason

The `max_actions=80` cap is an artificial safety valve that prevents the
agent from completing multi-level games. With level 1 taking ~13 actions,
level 2 taking ~20-30 actions (longer path + recharge stops), and 7 total
levels, the agent could easily need 200+ actions to complete the game.

---

## What Daniel Said

> "We need to take out the actions cap — that can be reset by the yellow
> items found along the game"

The movement budget is the game's own resource management. The yellow
recharge blocks are placed along the path PRECISELY so the player can
continue. Our artificial Python cap is overriding the game's own design.

---

## The Fix

### 1. Remove `max_actions` from the Game Loop Condition

**In `run_arc3_agent.py` line 172:**

```python
# BEFORE (artificial cap):
while state in ACTIVE_STATES and action_count < max_actions:

# AFTER (game drives termination):
while state in ACTIVE_STATES:
```

**In `run_arc3_session.py` line 115:**

```python
# BEFORE:
while state in ACTIVE_STATES and action_count < int(agent.max_actions):

# AFTER:
while state in ACTIVE_STATES:
```

### 2. Keep `max_actions` as a Safety Ceiling Only

Don't remove the parameter entirely — keep it as an extreme safety valve
to prevent infinite loops in case of API bugs. But set it MUCH higher:

```python
# In K3DARC3Agent.__init__:
# Default to 500 (7 levels × ~30 actions × ~2 resets each = ~420 max reasonable)
self.max_actions = int(max_actions) if max_actions else 500

# In the game loops:
while state in ACTIVE_STATES and action_count < self.max_actions:
```

The default should be 500, not 80. And `--max-actions` CLI arg should
default to 500.

### 3. Termination Should Be Game-Driven

The ONLY reasons the loop should exit:

| Condition | Meaning | Correct? |
|-----------|---------|----------|
| `state == "WIN"` | All levels completed | YES — success |
| `state == "GAME_OVER"` | All lives lost | YES — failure |
| `state` not in `ACTIVE_STATES` | API says done | YES — follow API |
| `action_count >= 500` | Safety ceiling | YES — but shouldn't trigger |
| `action_count >= 80` | Artificial cap | NO — remove this |

### 4. Strategic Reset Already Handles Budget

E.42 already implemented strategic reset: when the agent detects its
movement budget is insufficient, it sends RESET (ACTION0). This preserves
lives. The game's own feedback loop (budget + lives + recharge) is the
correct throttle, not our Python counter.

---

## Files to Change

1. **`scripts/run_arc3_agent.py`**:
   - Line 287: `default=80` → `default=500`
   - Line 172: Keep the condition but with higher ceiling

2. **`scripts/run_arc3_session.py`**:
   - Line 225: `max_actions_per_game: int = 80` → `500`
   - Line 115: Keep condition with higher ceiling

3. **`benchmarks/arc_agi_3.py`**:
   - Line 741: `max_actions: int = 80` → `max_actions: int = 500`

4. **Tests**: Update any test that hardcodes `max_actions=80` or expects
   exact action count matching. Test assertions should verify game-driven
   termination, not cap-driven termination.

---

## Why 500 and Not Unlimited

- 7 levels × ~30 actions/level × ~2 attempts/level = ~420 actions max
- 500 gives ~20% headroom above theoretical max
- Prevents infinite loops if API returns `IN_PROGRESS` forever
- The game itself will terminate via lives system well before 500

---

## Success Criteria

- [ ] Default max_actions raised from 80 to 500
- [ ] Game loop terminates on `state` change, not action count
- [ ] Agent can play through multiple levels without being cut off
- [ ] Strategic reset (E.42) continues to work as the budget throttle
- [ ] Safety ceiling at 500 prevents infinite loops
- [ ] Tests updated to reflect new defaults
- [ ] Live rerun shows more than 80 actions when needed
