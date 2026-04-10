# Codex Direction: ARC R1 — Transformation Inference + ARC-3 Policy Fix

**Date:** 2026-04-08
**Depends on:** R0 infrastructure ✅ (arc2_local_runner.py, arc3_sdk_agent.py verified green)
**Goal:** Get a real non-zero ARC-2 score by inferring and applying transformation rules from
         training pairs. Fix the ARC-3 sovereign feed policy gate so ls20 can run.
**Do not touch:** protected encyclopedia ingest (PID 101379)

---

## Context — Why R0 scored 0.00%

The R0 pipeline is correctly wired end-to-end. The 0% is honest and expected:

`_choose_rule` finds the nearest training pair, then `k3d_rule_apply(rule["output_root"], [], [])`
returns the stored *training* output root verbatim. `decode_grid_from_star` reconstructs that
training output grid. But ARC test expected outputs are almost never identical to a training
output — the task requires applying the same *transformation* the training pairs illustrate
to a *new* input.

R1 adds transformation inference: detect the rule from training pairs, apply it to the test input.

---

## Context — Why ARC-3 ls20 scored 0

`K3DAgent.run_level()` encountered `policy_error: sovereign_build_feed_missing`, which blocked
it from making meaningful moves for all 60 steps. The sovereign build feed check was written to
gate Galaxy population — but ARC-3 spatial navigation only needs spatial primitives (up/down/
left/right, walls, goals) that already exist in the House from H15+ work. The encyclopedia feed
is irrelevant to the spatial game. The gate must not block when `allow_remote_compat=True`.

---

## Deliverable 1 — `benchmarks/arc_transform_inferrer.py`

New file. Ingestion-path analysis only — no hot-path computation, Python grid operations are
correct here.

```python
"""Infer the transformation rule shared by ARC training pairs, then apply it."""
from __future__ import annotations
from typing import Any

Grid = list[list[int]]

# Transformation type constants
TRANSFORM_IDENTITY = "identity"
TRANSFORM_FLIP_H = "flip_h"
TRANSFORM_FLIP_V = "flip_v"
TRANSFORM_ROT90 = "rot90"
TRANSFORM_ROT180 = "rot180"
TRANSFORM_ROT270 = "rot270"
TRANSFORM_COLOR_PERM = "color_perm"
TRANSFORM_TILE_2X = "tile_2x"
TRANSFORM_TILE_3X = "tile_3x"
TRANSFORM_SCALE_2X = "scale_2x"
TRANSFORM_SCALE_3X = "scale_3x"
TRANSFORM_UNKNOWN = "nearest_training_pair"
```

### `detect_transform(input_grid, output_grid) -> dict`

Returns `{"type": TRANSFORM_*, ...extra}` or `{"type": TRANSFORM_UNKNOWN}`.

Try each candidate in order (cheapest first):

1. **Identity**: `output == input` → `{"type": "identity"}`

2. **flip_h** (mirror left-right): `output[r][c] == input[r][W-1-c]` for all r,c
   → `{"type": "flip_h"}`

3. **flip_v** (mirror top-bottom): `output[r][c] == input[H-1-r][c]` for all r,c
   → `{"type": "flip_v"}`

4. **rot90** (clockwise 90°): `output[c][H-1-r] == input[r][c]` — output shape is (W, H)
   → `{"type": "rot90"}`

5. **rot180**: `output[H-1-r][W-1-c] == input[r][c]` — output shape is (H, W)
   → `{"type": "rot180"}`

6. **rot270** (CCW 90° = CW 270°): `output[W-1-c][r] == input[r][c]` — output shape is (W, H)
   → `{"type": "rot270"}`

7. **color_perm**: Same shape, find bijective mapping `{src_color → dst_color}` from
   corresponding cells. If consistent across all cells and bijective:
   → `{"type": "color_perm", "mapping": {0: 3, 1: 5, ...}}`

8. **tile_2x**: Output is input tiled 2× horizontally AND 2× vertically (output shape = 2H×2W).
   → `{"type": "tile_2x"}`

9. **tile_3x**: Same, tiled 3×.
   → `{"type": "tile_3x"}`

10. **scale_2x**: Each input cell becomes a 2×2 block of the same color.
    → `{"type": "scale_2x"}`

11. **scale_3x**: Each input cell becomes a 3×3 block.
    → `{"type": "scale_3x"}`

12. **Unknown**: None of the above.
    → `{"type": "nearest_training_pair"}`

### `infer_task_transform(training_pairs: list[dict]) -> dict`

Given a list of `{"input": grid, "output": grid}` training pairs:

1. Detect the transform for each pair individually.
2. If ALL pairs agree on the same transform type (and compatible extra params):
   → Return that shared transform.
3. If majority (>= 2/3) agree:
   → Return the majority transform, add `{"confidence": "majority"}`.
4. Otherwise:
   → Return `{"type": "nearest_training_pair"}`.

### `apply_transform(grid: Grid, transform: dict) -> Grid`

Apply the detected transformation to a new grid:

- `identity`: return a copy of the grid unchanged
- `flip_h`: reverse each row
- `flip_v`: reverse the list of rows
- `rot90`: standard 90° CW rotation — zip(*grid[::-1])
- `rot180`: apply flip_h then flip_v
- `rot270`: standard 90° CCW rotation — zip(*grid)[::-1]
- `color_perm`: apply `mapping.get(cell, cell)` to each cell
- `tile_2x`: concatenate rows horizontally × 2, then vertically × 2
- `tile_3x`: same × 3
- `scale_2x`: expand each cell to 2×2 block
- `scale_3x`: expand each cell to 3×3 block
- `nearest_training_pair`: raise `ValueError("use nearest_training_pair fallback path")`

### `transform_type_to_rpn(transform: dict) -> str`

Return a compact RPN description for logging and Grammar tagging:

```
"identity"       → "GRID IDENTITY_TRANSFORM"
"flip_h"         → "GRID FLIP_H"
"flip_v"         → "GRID FLIP_V"
"rot90"          → "GRID ROT90"
"rot180"         → "GRID ROT180"
"rot270"         → "GRID ROT270"
"color_perm"     → "GRID {mapping_json} COLOR_PERM"
"tile_2x"        → "GRID 2 2 TILE"
"tile_3x"        → "GRID 3 3 TILE"
"scale_2x"       → "GRID 2 SCALE_UNIFORM"
"scale_3x"       → "GRID 3 SCALE_UNIFORM"
"nearest_tp"     → "GRID NEAREST_TRAINING_PAIR"
```

---

## Deliverable 2 — Wire inferrer into `arc2_local_runner.py`

### `run_one_task` changes

Add one pre-loop step after `compiled_rules` is built:

```python
# Infer shared transformation from training pairs
from benchmarks.arc_transform_inferrer import infer_task_transform, apply_transform

training_pairs = [
    {"input": rule["input_grid"], "output": rule["output_grid"]}
    for rule in compiled_rules
]
task_transform = infer_task_transform(training_pairs)
```

Then in the per-test-sample loop, after `rule, match_type = _choose_rule(...)`:

```python
predicted: list[list[int]] | None = None

if task_transform["type"] != "nearest_training_pair":
    # Apply inferred transform to the test input itself
    try:
        predicted = apply_transform(test_grid, task_transform)
        match_type = task_transform["type"]   # overwrite match_type for logging
    except Exception:
        predicted = None  # fall through to CAS/fallback path

if predicted is None and rule is not None:
    # Existing CAS + fallback path (unchanged)
    predicted_root = bridge.launch_k3d_rule_apply(int(rule["output_root"]), [], [])
    predicted = decode_grid_from_star(bridge, predicted_root)
    if predicted is None:
        predicted = rpn_to_grid(str(rule["star"].behavior_rpn or ""))
    if predicted is None:
        predicted = [list(row) for row in rule["output_grid"]]
```

Add `"task_transform": task_transform["type"]` to each row dict and to `task_summary`.

### Summary output additions

`match_type` distribution now includes transform type names, not just `exact_poly_match` /
`nearest_training_pair`. This is more informative for the paper.

---

## Deliverable 3 — Fix ARC-3 sovereign feed policy gate

**File:** `benchmarks/arc3_sdk_agent.py`

**Problem:** `policy_error: "sovereign_build_feed_missing"` blocks all 60 steps.
The check exists to ensure the Galaxy has been populated. But ARC-3 spatial navigation
only needs spatial primitives already loaded in the House — not the encyclopedia feed.

**Fix:** Relax the policy gate when `allow_remote_compat=True`.

In `K3DAgent.__init__` or wherever the policy check runs, change:

```python
# BEFORE (blocks entirely)
if not _sovereign_feed_present():
    self._policy_error = "sovereign_build_feed_missing:run scripts/rebuild_sovereign_artifact.py ..."
    # ... which causes run_level() to return immediately

# AFTER (warn only when allow_remote_compat=True)
if not _sovereign_feed_present():
    if self._allow_remote_compat:
        # Log warning but do NOT set blocking policy_error
        self._policy_warning = "sovereign_build_feed_missing: proceeding with spatial primitives only"
    else:
        self._policy_error = "sovereign_build_feed_missing:run scripts/rebuild_sovereign_artifact.py ..."
```

**In `run_level()`:** Only early-return on `self._policy_error` (not `_policy_warning`).

**In the returned JSON:** Include `policy_warning` field (not `policy_error`) so the report
is honest about the gap without blocking execution.

**Expected result after fix:** ls20 levels attempted via HTTP compat transport, with honest
reporting of what spatial knowledge is and isn't available.

---

## Deliverable 4 — Fix submission output filename

In `CODEX_ARC_R0_RUN_2026-04-08.md` and any future run commands:

The ARC submission format IS JSON (not CSV). The formatter correctly writes JSON.
The `.csv` extension in the R0 run command was wrong.

- Change `--submission-output /tmp/arc2_r0_submission.csv` → `--submission-output /tmp/arc2_r1_submission.json`
- No change needed to `arc_submission_formatter.py` — it is correct.

---

## Tests

Add to `tests/test_arc_transform_inferrer.py`:

```python
def test_flip_h():
    input_grid = [[1, 2], [3, 4]]
    output_grid = [[2, 1], [4, 3]]
    t = detect_transform(input_grid, output_grid)
    assert t["type"] == "flip_h"
    assert apply_transform(input_grid, t) == output_grid

def test_flip_v():
    input_grid = [[1, 2], [3, 4]]
    output_grid = [[3, 4], [1, 2]]
    t = detect_transform(input_grid, output_grid)
    assert t["type"] == "flip_v"
    assert apply_transform(input_grid, t) == output_grid

def test_rot90():
    input_grid = [[1, 2, 3], [4, 5, 6]]
    # CW 90°: col 2→row 0, etc.
    output_grid = [[4, 1], [5, 2], [6, 3]]
    t = detect_transform(input_grid, output_grid)
    assert t["type"] == "rot90"
    assert apply_transform(input_grid, t) == output_grid

def test_color_perm():
    input_grid = [[1, 2], [3, 4]]
    output_grid = [[5, 6], [7, 8]]
    t = detect_transform(input_grid, output_grid)
    assert t["type"] == "color_perm"
    assert t["mapping"] == {1: 5, 2: 6, 3: 7, 4: 8}
    assert apply_transform(input_grid, t) == output_grid

def test_scale_2x():
    input_grid = [[1, 2], [3, 4]]
    output_grid = [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
    t = detect_transform(input_grid, output_grid)
    assert t["type"] == "scale_2x"
    assert apply_transform(input_grid, t) == output_grid

def test_tile_2x():
    input_grid = [[1, 2], [3, 4]]
    output_grid = [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]
    t = detect_transform(input_grid, output_grid)
    assert t["type"] == "tile_2x"
    assert apply_transform(input_grid, t) == output_grid

def test_infer_consistent_pairs():
    pairs = [
        {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
        {"input": [[5, 6], [7, 8]], "output": [[6, 5], [8, 7]]},
    ]
    t = infer_task_transform(pairs)
    assert t["type"] == "flip_h"

def test_infer_mixed_falls_back():
    pairs = [
        {"input": [[1, 2]], "output": [[2, 1]]},   # flip_h
        {"input": [[3, 4]], "output": [[3, 4]]},   # identity
    ]
    t = infer_task_transform(pairs)
    assert t["type"] == "nearest_training_pair"
```

Minimum: all 8 tests pass. Run via:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium \
  env PYTHONPATH=$(pwd) pytest -q tests/test_arc_transform_inferrer.py
```

---

## R1 Evaluation Run

After all deliverables pass tests:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/arc2_local_runner.py \
  --max-tasks 20 \
  --submission-output /tmp/arc2_r1_submission.json \
  --summary-output /tmp/arc2_r1_summary.json
```

Then ARC-3 (after policy gate fix):

```bash
bash scripts/k3d_env.sh run -e trmc_core python - <<'PY'
from benchmarks.arc3_sdk_agent import K3DAgent
import json
agent = K3DAgent("ls20", max_steps=60, allow_remote_compat=True)
try:
    result = agent.run_level()
finally:
    agent.close()
print(json.dumps(result, indent=2))
PY
```

---

## Report back

Write `TEMP/CODEX_TO_CLAUDE_ARC_R1_RUN_REPORT_2026-04-08.md` with:

1. ARC-2 R1 score: `tasks=20, correct=K, total_inputs=20, score=X.XX%`
2. `match_type` distribution across 20 tasks (now includes transform names)
3. `task_transform` distribution (flip_h=N, color_perm=N, nearest_training_pair=N, etc.)
4. ARC-3 ls20 result after policy fix: `steps, levels_completed, policy_warning`
5. All tests passing: test count and command used

---

## What NOT to do

- Do NOT rebuild the sovereign artifact or disturb PID 101379
- Do NOT add transformation logic to the hot path (CAS/SAS kernels)
- Do NOT claim a score higher than what the runner actually produces
- Do NOT change the submission formatter — the JSON format is correct for ARC
- Do NOT add transformation types not in the list above (R2 concern: pattern completion,
  object detection, object movement, symmetry completion)
