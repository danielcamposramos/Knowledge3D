# Codex — Phase E.16: Fix `gpu_arc_no_output_grid` for Live ARC3

**Date:** 2026-03-29
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Prerequisite:** E.15 DONE. ARC3 routes through kv.execute_task(). Live: gpu_arc_no_output_grid on every step.

---

## Root Cause (Traced to Exact Line)

The Knowledgeverse `_answer_arc_query()` path (line 14120) calls `_arc_exact_task_navigation_candidates()` and `_select_composed_head_candidate()`.

Both require `match.get("arc_primitive_plan")` or `match.get("arc_transform_chain")` to be non-empty (lines 4056, 4083). When both are empty or None, the code hits:

```python
if not isinstance(output_grid, list):  # line 4102
    return {"status": "error", "error": "gpu_arc_no_output_grid", ...}
```

**Why are they empty for live ARC3?**

`_arc_exact_task_navigation_candidates()` (line 7448) requires `task_id` to be set:
```python
task_id = str(payload.get("task_id", "")).strip()
if not task_id:
    return []
```

The current `arc_agi_3.py` packs the task without a `task_id`:
```python
gpu_task = {
    "type": "ARC_TASK",
    "query": "navigate arc3 interactive frame toward goal",
    "input_grid": normalized_frame,
    ...
    # "task_id" MISSING
}
```

Even with `task_id` set, `_arc_exact_task_navigation_candidates()` only matches entries where:
```python
str(entry.get("galaxy", "")) == "Drawing"
str(entry.get("category", "")).lower() == "arc_benchmark_curriculum"
str(entry.get("arc_task_id", "")) == task_id
```

Live ARC3 game IDs (`tn36-ab4f63cc`, `dc22-4c9bff3e`) are not in the Drawing Galaxy ARC curriculum — they are LIVE INTERACTIVE GAMES, not pre-loaded ARC-2 training tasks.

**The composed head path (`_select_composed_head_candidate`) DOES find a match** (it navigates the Galaxy using the query embedding), but the matched entry also has no `arc_primitive_plan` or `arc_transform_chain` because Galaxy entries for visual reasoning are patterns and transforms, not interactive game state solutions.

**Summary:** The ARC path expects to find a pre-stored grid transform in the Galaxy that exactly matches the task. For live ARC3 interactive games, no such stored transform exists. The GPU path succeeds in REASONING but has no mechanism to EMIT an action when the match carries no grid transform.

---

## The Fix: Two Layers

### Layer 1: Add `task_id` + `game_id` to the task dict

This costs nothing and ensures `_arc_exact_task_navigation_candidates()` doesn't short-circuit. It also gives the Knowledgeverse tracing a meaningful identifier.

In `arc_agi_3.py`, add to the `gpu_task` dict:
```python
gpu_task = {
    "type": "ARC_TASK",
    "task_id": f"arc3_live_{len(self.action_history)}",  # per-step identity
    "query": _frame_to_query_text(normalized_frame, normalized_goal),  # semantic tokens
    "input_grid": normalized_frame,
    "expected_output": normalized_goal if normalized_goal != [[]] else [],
    "training_examples": list(task_context.get("train") or []),
    "action_options": list(ACTION_NAMES),
    "options": list(ACTION_NAMES),
}
```

### Layer 2: `_derive_action_from_result` must handle `output_grid=None` gracefully

This is already partially done — the fallback at lines 47-65 reads `output_grid` from `result`, computes cell delta, and falls back to `ACTION1` (Move Up) when nothing works. But `ACTION1` is wrong when the goal is to the right.

The fix: when `output_grid` is None AND `answer_index` is None, use the `input_grid` and `expected_output` the task already contains to derive direction directly — WITHOUT going back through the GPU. This is pure I/O geometry:

```python
def _derive_action_from_frames(
    frame: list[list[int]],
    goal_frame: list[list[int]] | None,
) -> int:
    """Geometric action derivation from frame delta. Pure I/O — no GPU."""
    if not goal_frame or goal_frame == [[]]:
        return 0  # no goal → Move Up as neutral default

    current = _find_primary_cell(frame)
    target = _find_primary_cell(goal_frame)
    if current is None or target is None:
        return 0

    dr = target[0] - current[0]  # positive = need to go down
    dc = target[1] - current[1]  # positive = need to go right

    if abs(dr) >= abs(dc):
        return 0 if dr < 0 else 1  # ACTION1=up, ACTION2=down
    else:
        return 2 if dc < 0 else 3  # ACTION3=left, ACTION4=right
```

Update `_derive_action_from_result` to call this as the final fallback:

```python
def _derive_action_from_result(
    frame: list[list[int]],
    result: dict[str, Any],
    *,
    goal_frame: list[list[int]] | None = None,  # NEW parameter
) -> tuple[int, dict[str, int]]:
    # Priority 1: explicit answer_index from Knowledgeverse
    raw_answer_index = result.get("answer_index")
    if isinstance(raw_answer_index, (int, float)):
        action_index = max(0, min(int(raw_answer_index), len(ACTION_NAMES) - 1))
        return action_index, {}

    # Priority 2: x/y click coordinates
    if isinstance(result.get("x"), (int, float)) and isinstance(result.get("y"), (int, float)):
        return 5, {"x": int(result["x"]), "y": int(result["y"])}

    # Priority 3: output_grid delta
    predicted = _normalize_grid(result.get("output_grid"))
    if predicted != [[]]:
        current_cell = _find_primary_cell(frame)
        predicted_cell = _find_primary_cell(predicted)
        if current_cell is not None and predicted_cell is not None:
            delta_row = predicted_cell[0] - current_cell[0]
            delta_col = predicted_cell[1] - current_cell[1]
            if abs(delta_row) > abs(delta_col):
                return (0 if delta_row < 0 else 1), {}
            if abs(delta_col) > 0:
                return (2 if delta_col < 0 else 3), {}
            if predicted != frame:
                return 4, {}

    # Priority 4: geometric fallback from goal frame (sovereign I/O, no GPU)
    return _derive_action_from_frames(frame, goal_frame), {}
```

Update the call site in `choose_action`:
```python
action_index, payload = _derive_action_from_result(
    normalized_frame,
    dict(result or {}),
    goal_frame=normalized_goal,  # pass goal for geometric fallback
)
```

### Layer 3: Semantic query text from frame state

Add `_frame_to_query_text()` to `arc_agi_3.py`:

```python
def _frame_to_query_text(
    frame: list[list[int]],
    goal_frame: list[list[int]] | None,
) -> str:
    """Describe the frame state in semantic tokens commensurable with Galaxy embeddings."""
    current = _find_primary_cell(frame)
    if current is None:
        return "navigate arc3 interactive frame toward goal visual spatial"

    row, col, _ = current
    height = max(1, len(frame))
    width = max(1, len(frame[0]) if frame else 1)

    direction_parts = []
    if goal_frame and goal_frame != [[]]:
        target = _find_primary_cell(goal_frame)
        if target is not None:
            dr = target[0] - row
            dc = target[1] - col
            if dr < 0: direction_parts.append("up north")
            elif dr > 0: direction_parts.append("down south")
            if dc < 0: direction_parts.append("left west")
            elif dc > 0: direction_parts.append("right east")

    direction = " ".join(direction_parts) if direction_parts else "center balanced"
    return (
        f"move colored cell {direction} navigate spatial grid "
        f"translate arc3 visual transformation toward goal"
    ).strip()
```

---

## Why the ARC3 Stall in `run_full_benchmark`

The 21-minute stall in `arc3_synthetic` is a separate issue. `run_arc3_synthetic()` in `run_full_benchmark.py` creates a `K3DARC3Agent(knowledgeverse=kv)` and runs 10 synthetic ARC3 tasks through `kv.execute_task()`. The stall is likely the Knowledgeverse initialization on the SECOND `Knowledgeverse()` creation inside `run_arc3_synthetic` — or the Galaxy loading triggering a cold-path rebuild.

**Check:** Does `run_arc3_synthetic` create a new `Knowledgeverse()` internally, or does it receive the existing `kv`?

```bash
grep -n "Knowledgeverse\|K3DARC3Agent" scripts/run_full_benchmark.py | head -20
```

If it creates a fresh `Knowledgeverse()`, that's loading 130k stars twice. Fix: pass `kv` from `run_full_benchmark` into `run_arc3_synthetic()` and from there into `K3DARC3Agent(knowledgeverse=kv)`.

If it already passes `kv`, the stall is inside `kv.execute_task()` for ARC_TASK on a synthetic grid. Add a `timeout` or check if `_select_composed_head_candidate` is looping. A quick diagnostic: add `print(f"arc3 step {i}")` around the `choose_action` call in `run_arc3_synthetic` to see if it's stuck on step 0 or progressing slowly.

---

## Files to Modify

| File | Change |
|------|--------|
| `benchmarks/arc_agi_3.py` | Add `task_id` to gpu_task. Add `_frame_to_query_text()`. Add `_derive_action_from_frames()`. Update `_derive_action_from_result()` signature + add Priority 4 fallback. Pass `goal_frame` to derive call. |
| `scripts/run_full_benchmark.py` | Verify `kv` is passed to `run_arc3_synthetic`. If not, add it. |

## Files NOT to Touch

| File | Why |
|------|-----|
| `knowledge3d/knowledgeverse/knowledgeverse.py` | The GPU path is correct. It returns `gpu_arc_no_output_grid` honestly when no transform exists — that's not a bug, it's the correct sovereign behavior. The adapter must handle this. |
| All other benchmark files | Already working. |

---

## Execution Sequence

1. Add `_frame_to_query_text()` and `_derive_action_from_frames()` to `arc_agi_3.py`
2. Add `task_id` to the gpu_task dict in `choose_action()`
3. Update `_derive_action_from_result()` to accept `goal_frame` and use geometric fallback
4. Verify `run_full_benchmark.py` passes `kv` to `run_arc3_synthetic` (fix if not)
5. `python3 -m py_compile benchmarks/arc_agi_3.py` — must pass
6. Run live ARC3 probe (10 actions): `python scripts/run_arc3_agent.py --game-id <ID> --max-actions 10`
   - Expected: actions vary based on goal direction (not all Move Up)
   - Still `gpu_execution=True`, still `output_grid=None` initially — that's OK
   - `_derive_action_from_frames` fires as Priority 4 and gives geometric direction
7. Run full benchmark — check if arc3_synthetic no longer stalls

---

## Expected Behavior After Fix

| Step | Before | After |
|------|--------|-------|
| `output_grid` | None | Still None initially (no stored transform for live games) |
| `gpu_execution` | True | True |
| `error` | gpu_arc_no_output_grid | gpu_arc_no_output_grid (honest — no transform stored) |
| `action_index` | 0 (Move Up default) | geometric direction to goal |
| Live action distribution | All Move Up | Varied (up/down/left/right based on goal position) |
| `confidence` | 0.000 | Still low initially; builds as Galaxy learns |

The `gpu_arc_no_output_grid` error is HONEST. The GPU path is correct — it found no stored ARC transform for a live game ID. That's the right answer from the Knowledgeverse perspective. The I/O adapter handles this gracefully with the geometric fallback. Over many live games, the Knowledgeverse sleep-time will begin associating successful action traces with the visual patterns it does recognize — and eventually the GPU path will start returning meaningful transforms.

This is the pattern: I/O adapter handles what the GPU can't yet answer. The GPU learns. Over time the geometric fallback fires less.
