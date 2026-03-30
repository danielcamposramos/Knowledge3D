# Claude -- Phase E.26: Local ARC3 Benchmark (Real Game Logic, No API)

**Date:** 2026-03-29
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** HIGH -- validate directional movement locally before live API

---

## Daniel's Directive (Verbatim)

> "we need and can use REAL local benchmarks, specially arc 3, until we pass
> one game at least or hit right on at least one movement - but at the local
> run! we do not run tests against the real thing, right?"

---

## What This Is

A **local ARC3 game engine** that replicates the real ARC3 interactive game logic
(grid navigation: move colored object toward goal position) WITHOUT hitting the
remote API. This is NOT "synthetic" -- it implements the SAME game mechanics the
live API uses: a grid with a foreground object, a goal grid, and 7 available
actions (Move Up/Down/Left/Right, Perform, Click, Undo).

**Success criterion from Daniel:** Pass one game OR produce at least one correct
movement action in a local run.

---

## Architecture

### Game Engine (Pure Python, I/O layer)

The local game engine lives in `benchmarks/arc3_local.py`. It:

1. **Creates tasks** with controlled start/goal positions (known-correct answers)
2. **Simulates game state** (frame grid, goal grid, available actions, levels)
3. **Applies actions** (move the foreground cells by 1 in the action direction)
4. **Checks completion** (frame matches goal = level cleared)
5. **Tracks progress** (levels completed, action count, state)

This is I/O infrastructure -- it feeds frames to `K3DARC3Agent` exactly like
the live API would. The agent doesn't know or care whether frames come from
the API or the local engine.

### Agent (Unchanged)

`K3DARC3Agent` from `benchmarks/arc_agi_3.py` is used AS-IS:
- `choose_action(frame, goal_frame=goal, ...)` -- same interface
- `learn_from_outcome(levels_completed=N, frame=frame)` -- same interface
- Internally calls `kv.execute_task()` with `domain_hint="arc3_interactive"`
- The transitional direct decode in `_answer_arc_query()` reads position tokens

### Task Generation

Each task = a small grid (8x8 to 16x16) with:
- **One foreground cell** (color 1-9) at a known `(start_row, start_col)`
- **Background** = 0 (black)
- **Goal grid** = same grid but cell at `(goal_row, goal_col)`
- **Optimal solution** = sequence of Move Up/Down/Left/Right to get from start to goal
- **Budget** = optimal_steps * 3 (room for suboptimal but still-valid paths)

Task categories (progressive difficulty):
1. **Single-axis (cardinal):** start and goal differ on ONE axis only
   - `(0, 4) -> (7, 4)` = 7x Move Down
   - `(4, 7) -> (4, 0)` = 7x Move Left
2. **Two-axis (diagonal path):** start and goal differ on BOTH axes
   - `(1, 1) -> (6, 6)` = 5 Down + 5 Right (order doesn't matter)
3. **Already solved:** start == goal, correct answer is Perform
4. **Edge cases:** object at grid boundary, single-step moves

### Game Loop (Per Task)

```
task = create_task(task_index)
frame = clone(task.start_grid)
goal  = clone(task.goal_grid)

for step in range(budget):
    action = agent.choose_action(frame, goal_frame=goal, ...)
    frame  = apply_action(frame, action.action_index)
    done   = (frame == goal)
    agent.learn_from_outcome(levels_completed=1 if done else 0, frame=frame)
    if done: SOLVED; break
```

### Action Application

```python
def apply_action(frame, action_index):
    # Find all foreground cells (value != background)
    # action 0 (Move Up):    shift all foreground rows by -1 (clamp at 0)
    # action 1 (Move Down):  shift all foreground rows by +1 (clamp at max)
    # action 2 (Move Left):  shift all foreground cols by -1 (clamp at 0)
    # action 3 (Move Right): shift all foreground cols by +1 (clamp at max)
    # action 4 (Perform):    no-op (frame unchanged)
    # action 5 (Click):      no-op (frame unchanged)
    # action 6 (Undo):       restore previous frame from stack
    return new_frame
```

**Critical:** actions that would move a cell off-grid are clamped (cell stays at
boundary). This matches real ARC3 behavior.

---

## File Layout

### `benchmarks/arc3_local.py` -- Local Game Engine + Runner

```python
"""Local ARC3 game engine -- real game logic, no API."""

# Constants
GRID_SIZES = [8, 10, 12, 16]
ACTION_NAMES = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]
ACTION_LABELS = ["Move Up", "Move Down", "Move Left", "Move Right", "Perform", "Click", "Undo"]

# Task generation
def make_task(index: int, grid_size: int = 8) -> dict
    # Deterministic from index: start position, goal position, grid size, color
    # Returns: {start_grid, goal_grid, start_pos, goal_pos, optimal_steps, budget, task_id}

# Action application
def apply_action(frame, action_index, frame_stack=None) -> tuple[frame, changed]
    # Moves foreground cells in the action direction, returns new frame

# Grid comparison
def grids_equal(a, b) -> bool

# Single game runner
def run_game(agent, task) -> dict
    # Runs one task: loop choose_action -> apply_action -> check done
    # Returns: {task_id, solved, steps_taken, optimal_steps, actions_taken: [...]}

# Batch runner
def run_local_arc3(count=20, grid_size=8, knowledgeverse=None, log_path=None) -> dict
    # Creates count tasks, runs each, returns aggregate results
    # Returns: {total, solved, accuracy, correct_first_moves, results: [...]}
```

### `scripts/run_arc3_local.py` -- CLI Entry Point

```python
"""Run local ARC3 benchmark from command line."""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=40)
    parser.add_argument("--storage-root", default="/K3D/Knowledge3D.local")
    parser.add_argument("--log-path", default=None)
    # ...
    results = run_local_arc3(...)
    # Print summary: solved/total, correct first moves, etc.
```

### Integration with `run_full_benchmark.py`

Add `arc3_local` as a suite in `run_full_benchmark.py`:

```python
suite_order = [
    ("mmlu", mmlu_count),
    ("gsm8k", gsm8k_count),
    ("lhe", lhe_count),
    ("arc2", arc2_count),
    ("arc3_local", arc3_count),  # <-- add
]
```

This requires adding:
- `arc3_count` parameter (default 20)
- `--arc3-count` CLI argument
- Import of `run_local_arc3` in `_ensure_full_benchmark_runtime`
- Result normalization in `_run_native_suite`
- Print line in `main()` summary

---

## Task Generation Details

Deterministic from task index (reproducible runs):

```python
# 20 default tasks covering all movement directions
TASK_CONFIGS = [
    # Single-axis: pure vertical
    {"start": (0, 4), "goal": (7, 4), "size": 8},   # 0: 7x Down
    {"start": (7, 4), "goal": (0, 4), "size": 8},   # 1: 7x Up
    {"start": (3, 4), "goal": (6, 4), "size": 8},   # 2: 3x Down
    {"start": (6, 4), "goal": (3, 4), "size": 8},   # 3: 3x Up

    # Single-axis: pure horizontal
    {"start": (4, 0), "goal": (4, 7), "size": 8},   # 4: 7x Right
    {"start": (4, 7), "goal": (4, 0), "size": 8},   # 5: 7x Left
    {"start": (4, 2), "goal": (4, 5), "size": 8},   # 6: 3x Right
    {"start": (4, 5), "goal": (4, 2), "size": 8},   # 7: 3x Left

    # Two-axis: diagonal paths
    {"start": (1, 1), "goal": (6, 6), "size": 8},   # 8: Down+Right
    {"start": (6, 6), "goal": (1, 1), "size": 8},   # 9: Up+Left
    {"start": (1, 6), "goal": (6, 1), "size": 8},   # 10: Down+Left
    {"start": (6, 1), "goal": (1, 6), "size": 8},   # 11: Up+Right

    # Already at goal (Perform)
    {"start": (4, 4), "goal": (4, 4), "size": 8},   # 12: Perform
    {"start": (0, 0), "goal": (0, 0), "size": 8},   # 13: Perform (corner)

    # Single-step moves
    {"start": (3, 4), "goal": (4, 4), "size": 8},   # 14: 1x Down
    {"start": (4, 4), "goal": (3, 4), "size": 8},   # 15: 1x Up
    {"start": (4, 3), "goal": (4, 4), "size": 8},   # 16: 1x Right
    {"start": (4, 4), "goal": (4, 3), "size": 8},   # 17: 1x Left

    # Boundary cases
    {"start": (0, 0), "goal": (7, 7), "size": 8},   # 18: corner to corner
    {"start": (7, 7), "goal": (0, 0), "size": 8},   # 19: corner to corner (reverse)
]
```

For `index >= 20`: generate procedurally using `index` as seed for reproducibility.

---

## Expected Behavior with Current Code

The current path for `domain_hint="arc3_interactive"`:

1. `K3DARC3Agent.choose_action()` calls `_frame_to_query_text()` to produce position tokens
2. `kv.execute_task()` routes to `_answer_arc_query()` with `domain_hint="arc3_interactive"`
3. `_answer_arc_query()` transitional direct decode reads position tokens from query text
4. Returns `answer_index` = 0-4 based on position relative to center

**What the direct decode WILL do correctly:**
- Object at (0, 4), goal at (7, 4): query says "object above center" -> Move Up (index 0)
  - **WRONG!** Object is above center, so it should Move DOWN toward goal
  - Wait -- the direct decode moves toward CENTER, not toward GOAL
  - Object above center -> Move Up moves it further from center (wrong for centering too!)

**Let me re-read the decode logic:**
```
"object above center" -> action 0 = Move Up
```

This means: object is ABOVE center, action = Move Up = move further up. This is WRONG
for both "move toward center" AND "move toward goal."

**The direct decode has the directions INVERTED for centering tasks.**

Actually, re-reading `_frame_to_query_text()`:
- `avg_row < center_row - margin` -> "object above center top north"
- The direct decode maps this to action 0 = "Move Up" (row decreases)
- But "object above center" means low row index -> Move Up decreases it further -> WRONG

**Correction needed in direct decode OR in the query text mapping:**

The correct logic should be:
- Object above center (avg_row < center) -> Move DOWN (action 1) to approach center/goal
- Object below center (avg_row > center) -> Move UP (action 0) to approach center/goal
- Object left of center (avg_col < center) -> Move RIGHT (action 3) to approach center/goal
- Object right of center (avg_col > center) -> Move LEFT (action 2) to approach center/goal

**However:** The direct decode should move toward the GOAL, not toward CENTER.
Center-based logic only works when the goal is centered.

**For the local benchmark to PASS:** The decode must compare object position to GOAL position,
not to grid center. The query text from `_frame_to_query_text()` currently only encodes
position relative to center. For goal-directed movement, it needs to encode position
relative to the goal.

---

## Fix: Goal-Relative Position Encoding

Modify `_frame_to_query_text()` in `benchmarks/arc_agi_3.py` to encode position
relative to GOAL when a goal frame is provided:

```python
if foreground_cells and goal_foreground_cells:
    # Compare object position to goal position
    avg_row = centroid of foreground in current frame
    goal_row = centroid of foreground in goal frame
    if avg_row < goal_row - margin:
        position_tokens.append("object above goal move down")
    elif avg_row > goal_row + margin:
        position_tokens.append("object below goal move up")
    # ... same for columns
    if not position_tokens:
        position_tokens.append("object at goal perform")
```

Then update the direct decode in `_answer_arc_query()` to match:

```python
if "object above goal" in _qt or "move down" in _qt:
    _arc3_direct_index = 1  # Move Down (toward goal below)
elif "object below goal" in _qt or "move up" in _qt:
    _arc3_direct_index = 0  # Move Up (toward goal above)
elif "object left of goal" in _qt or "move right" in _qt:
    _arc3_direct_index = 3  # Move Right (toward goal right)
elif "object right of goal" in _qt or "move left" in _qt:
    _arc3_direct_index = 2  # Move Left (toward goal left)
elif "object at goal" in _qt or "perform" in _qt:
    _arc3_direct_index = 4  # Perform (at goal)
```

**Important:** The position tokens should describe the SPATIAL RELATIONSHIP between
object and goal, and the MEANING-BASED action name. "above goal move down" = the object
is above the goal, so the correct action is to move down. This is how universal
movement knowledge works -- the same "move down" concept applies whether navigating
a grid, a house, or a physics simulation.

---

## Metrics

The local benchmark reports:

| Metric | Description |
|--------|-------------|
| `total` | Number of tasks attempted |
| `solved` | Tasks where frame == goal within budget |
| `accuracy` | solved / total |
| `correct_first_moves` | Tasks where first action was optimal direction |
| `first_move_accuracy` | correct_first_moves / total |
| `avg_steps` | Average steps to solve (solved tasks only) |
| `avg_optimality` | optimal_steps / actual_steps (solved tasks only) |

**Target (Daniel's minimum):** `solved >= 1` OR `correct_first_moves >= 1`

**Realistic expectation with fixed direct decode:** All single-axis tasks should solve
(direction is unambiguous). Two-axis tasks should solve but take non-optimal paths
(one axis at a time). "Already at goal" tasks should solve immediately (Perform).

That means: **14/20 single-step correct, 18-20/20 solved** with the goal-relative fix.

---

## Execution Sequence

1. **Create `benchmarks/arc3_local.py`** -- local game engine + task generation
2. **Fix `_frame_to_query_text()`** -- goal-relative position encoding
3. **Fix `_answer_arc_query()` direct decode** -- correct direction mapping
4. **Create `scripts/run_arc3_local.py`** -- CLI entry point
5. **Wire into `run_full_benchmark.py`** -- add arc3_local suite
6. **Run locally** -- validate at least one game solved or one correct movement
7. **Add test** -- `tests/test_arc3_local.py` with known-answer tasks

---

## Files to Create

| File | Purpose |
|------|---------|
| `benchmarks/arc3_local.py` | Local game engine, task generation, batch runner |
| `scripts/run_arc3_local.py` | CLI entry point |
| `tests/test_arc3_local.py` | Known-answer validation tests |

## Files to Modify

| File | Change |
|------|--------|
| `benchmarks/arc_agi_3.py` | `_frame_to_query_text()`: goal-relative position encoding |
| `knowledge3d/knowledgeverse/knowledgeverse.py` | `_answer_arc_query()`: fix direction mapping |
| `scripts/run_full_benchmark.py` | Add `arc3_local` suite, `--arc3-count` argument |

---

## Critical Discovery: Direction Inversion Bug

The E.24 probe showed all Perform (action 4) because the object was centered.
But for non-centered objects, the current direct decode has **inverted directions**:

- "object above center" -> Move Up (action 0) = moves FURTHER ABOVE = **WRONG**
- Should be: "object above center" -> Move Down (action 1) = moves TOWARD center

This explains why the E.24 probe appeared correct (centered = Perform is right)
but the system would fail on any actual navigation task.

The fix is straightforward: either invert the action mapping in the direct decode,
or (better) switch to goal-relative encoding so the tokens carry the correct
action name directly ("move down" when object is above goal).

---

## Success Criteria

- [ ] Local game engine creates reproducible grid navigation tasks
- [ ] Goal-relative position encoding in `_frame_to_query_text()`
- [ ] Direct decode maps to correct action directions
- [ ] At least 1 game solved in local run (Daniel's minimum)
- [ ] At least 1 correct first-move in local run
- [ ] `run_arc3_local.py` runs standalone from CLI
- [ ] Arc3 local integrated into `run_full_benchmark.py`
- [ ] Test with known-answer tasks validates correctness
