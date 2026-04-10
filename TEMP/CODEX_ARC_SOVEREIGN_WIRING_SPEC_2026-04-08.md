# Codex Direction: ARC Sovereign Wiring — Route Through the Living AI

**Date:** 2026-04-08
**Supersedes:** CODEX_ARC_R1_SPEC_2026-04-08.md (wrong architecture — see below)
**Authority:** CLAUDE.md § "TRM IS the Avatar", MEMORY_TABLET_SPECIFICATION.md,
               KNOWLEDGEVERSE_SPECIFICATION.md, THREE_BRAIN_SYSTEM_SPECIFICATION.md

---

## What Is Wrong and Why

`arc2_local_runner.py` and `arc_transform_inferrer.py` are architecturally wrong. They bypass
the entire living AI and use Python as the reasoning engine:

```
WRONG (current):
ARC JSON → Python grid parse → k3d_canonicalize → k3d_pattern_match → Python returns answer
```

Python is ORCHESTRATING intelligence. That is a sovereignty violation. EVERY benchmark already
routes through the composed head pipeline via `Knowledgeverse.execute_task()`. ARC is not
special — it must use the SAME path. The WINE layer for this already exists.

The `arc_transform_inferrer.py` is Python pattern-matching disguised as AI. Delete it entirely.

---

## The Correct Architecture

```
[ARC JSON file]                         ← Python reads this (boot/I-O only)
        ↓
[arc2_game_envelope()]                  ← WINE layer: knowledge3d/tablet/wine/game2d_wine.py
        ↓                                  Already exists. Translates ARC JSON to TabletEnvelope.
[Knowledgeverse.execute_task()]         ← The ONE sovereign entry point ALL benchmarks use
        ↓
[knowledgeverse_gpu_query()]
        ↓
[Morton Octree → LED-A* → Frustum Cull → LOD → Nine-Chain Swarm → Halting Gate]
        ↓                                  Existing composed head pipeline on GPU
[Swarm Workers — dispatched by TRM]
  ├── Worker 0: gre_arc_reasoner          ← ARC-specific kernel (LOADED, not wired)
  ├── Worker 1: gre_geometry_router       ← Geometric transform routing (LOADED, not wired)
  ├── Worker 2: gre_vector_resonator      ← Embedding similarity → nearest Grammar star
  ├── Worker 3: gre_fractal_emitter       ← Recursive/fractal patterns
  └── Workers 4-8: general Galaxy nav     ← LED-A* across Drawing+Grammar+Math galaxies
        ↓
[CAS/SAS math cores]                    ← OP_CANONICALIZE, OP_SEMANTIC_RESOLVE,
        ↓                                  k3d_pattern_match, k3d_rule_apply
        ↓                                  Used BY swarm workers, NOT called from Python
[Grammar Galaxy]                        ← Training pairs seeded here by arc_task_galaxy_seeder.py
        ↓                                  Stars have meaning_rpn (input) + behavior_rpn (output)
        ↓                                  TRM navigates to find them via LED-A*
[Halting Gate]                          ← gre_multimodal_halting_gate.cu (already wired)
        ↓
[Task result: predicted grid]
        ↓
[Python extracts grid, scores vs expected]  ← Python (boot/I-O only, not reasoning)
```

**ALL galaxies must be loaded.** ARC grids reference Drawing Galaxy (color primitives),
Grammar Galaxy (transformation rules), Math Galaxy (coordinate math). Symlinks between
galaxies break if any galaxy is absent. No partial loads.

---

## Step 1 — Delete the Wrong Files

Delete these files. They represent the wrong architecture:

- `benchmarks/arc_transform_inferrer.py` — Python pattern-matching, not AI reasoning
- Remove `arc_transform_inferrer` import and all calls from `arc2_local_runner.py`
- Remove `task_transform` field from result rows (it came from the wrong inferrer)

Do NOT remove `arc_task_galaxy_seeder.py` — it correctly seeds Grammar Galaxy stars.
Do NOT remove `arc_submission_formatter.py` — it formats the output artifact (I-O only).

---

## Step 2 — Rewrite `arc2_local_runner.py` as a Thin Bootstrap

The new runner has ONE job: feed tasks to the living AI via the correct entry point.

```python
"""ARC-2 evaluation runner — thin bootstrap over the sovereign execute_task path."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.tablet.wine.game2d_wine import arc2_game_envelope
from benchmarks.arc_task_galaxy_seeder import seed_task, _normalize_grid
from benchmarks.arc_submission_formatter import (
    format_arc_submission, validate_arc_submission, write_arc_submission,
)

# Python only: read files, call execute_task, extract result, score.
# ZERO reasoning logic in Python. ZERO grid manipulation in Python.
```

### `_boot_knowledgeverse() -> Knowledgeverse`

Boot the sovereign system with ALL default galaxies loaded.

```python
def _boot_knowledgeverse() -> Knowledgeverse:
    kv = Knowledgeverse()
    kv.boot()             # loads ALL default galaxies (Drawing, Character, Word,
                          # Grammar, Math, Reality, Audio, 3DObjects, Tool)
    kv.load_all_galaxies()  # ensure no partial load — symlinks require everything
    return kv
```

If `Knowledgeverse` does not have a `load_all_galaxies()` method, add it (or use whatever
method currently loads all default galaxies). The requirement is: ALL default galaxies present.

### `_seed_task_into_galaxy(kv, task_json, task_id)`

Seed training pairs into Grammar Galaxy so TRM can navigate to them:

```python
def _seed_task_into_galaxy(kv, task_json: dict, task_id: str) -> None:
    stars = seed_task(task_json, galaxy_manager=kv.galaxy_manager, task_id=task_id)
    # Stars are now in Grammar Galaxy. TRM will find them via LED-A* during reasoning.
```

`seed_task()` already works. It just wasn't being passed `galaxy_manager` before.

### `_run_one_task(kv, task_id, task_json) -> tuple[int, int, list[dict]]`

```python
def _run_one_task(kv, task_id: str, task_json: dict) -> tuple[int, int, list[dict]]:
    _seed_task_into_galaxy(kv, task_json, task_id)
    test_rows = list(task_json.get("test") or [])
    rows = []
    correct = 0
    for idx, sample in enumerate(test_rows):
        input_grid = _normalize_grid(sample.get("input"))
        expected = _normalize_grid(sample.get("output"))

        # WINE layer: translate ARC input to TabletEnvelope
        envelope = arc2_game_envelope(
            task_id=f"{task_id}:{idx}",
            training_examples=list(task_json.get("train") or []),
            input_grid=input_grid,
            expected_output=expected,
        )

        # Execute through the composed head pipeline — same path as EVERY benchmark
        result = kv.execute_task(envelope.task, route=envelope.route())

        # Extract predicted grid from result (I-O only — no reasoning here)
        predicted = _extract_grid(result)
        is_correct = (predicted == expected) if expected else False
        correct += int(is_correct)
        rows.append({
            "task_id": task_id,
            "sample_index": idx,
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
            "match_type": str(result.get("match_type", "gpu_swarm")),
        })
    return correct, len(test_rows), rows
```

### `_extract_grid(result) -> list[list[int]] | None`

Extract the predicted grid from the execute_task result dict.
The result comes from the halting gate winner — it will be in one of these fields:
- `result.get("predicted_grid")`
- `result.get("output_grid")`
- `result.get("answer")` (if serialized as grid)

Check all three. If none is a valid 2D int list, return `None`.
Do NOT do any grid transformation in Python — if the grid is wrong, that means
the swarm workers need better Galaxy knowledge, not Python post-processing.

---

## Step 3 — Wire `gre_arc_reasoner` and `gre_geometry_router` for GAME_2D

**File:** `knowledge3d/knowledgeverse/knowledgeverse.py` or wherever the swarm worker
dispatch table lives (grep for `SURFACE_KIND_GAME_2D` or `game_2d` or swarm dispatch).

The nine-chain swarm workers are dispatched per task type. For `SURFACE_KIND_GAME_2D`:

```
Worker 0 → gre_arc_reasoner        (ARC-specific: navigate Grammar Galaxy for transform rules)
Worker 1 → gre_geometry_router     (geometric transform selection: flip, rotate, tile, scale)
Worker 2 → gre_vector_resonator    (embedding similarity: find nearest Grammar star by grid shape/palette)
Worker 3 → gre_fractal_emitter     (recursive/fractal patterns in ARC)
Worker 4 → gre_graph_crystallizer  (multi-hop: compose multiple Grammar rules)
Workers 5-8 → general Galaxy nav   (broad Grammar + Drawing + Math galaxy search)
```

The `gre_arc_reasoner` kernel receives the CAS canonical root of the input grid
(already computed by the CAS/SAS layer during task ingestion) and navigates Grammar
Galaxy to find stars whose `meaning_rpn` matches. It then returns the `behavior_rpn`
of the matched star as the predicted transformation.

CAS/SAS opcodes available to swarm workers:
- `OP_CANONICALIZE (0x238)` — canonical form of input grid STAR node
- `OP_SEMANTIC_RESOLVE (0x23A)` — resolve grid symbols against symbol table
- `OP_RULE_SELECT (0x23B)` — select matching Grammar rule
- `OP_CONTEXTUAL_REWRITE (0x23C)` — apply rule to produce output grid STAR
- `OP_SEMANTIC_EQUIV (0x23D)` — verify equivalence of predicted vs expected

This is the correct use of CAS/SAS: as math cores inside swarm workers, not as
Python orchestration.

---

## Step 4 — Ensure All Galaxies Load at Boot

**The symlink constraint:** Grammar Galaxy ARC rules reference Drawing Galaxy (color codes
0-9 map to Drawing primitives), Math Galaxy (grid coordinate arithmetic), and Character
Galaxy (labels/descriptions). If Drawing or Math are absent, the symlinks in Grammar stars
break and LED-A* cannot traverse them.

In `Knowledgeverse.boot()` or `Knowledgeverse.load_all_galaxies()`:

```python
REQUIRED_GALAXIES = [
    "Drawing", "Character", "Word", "Grammar", "Math",
    "Reality", "Audio", "3DObjects", "Tool",
]
for galaxy_name in REQUIRED_GALAXIES:
    if not self.galaxy_manager.is_loaded(galaxy_name):
        self.galaxy_manager.load_galaxy(galaxy_name)
```

Fail loudly if any required galaxy is missing — do NOT silently skip or partial-load.
A missing galaxy means broken symlinks means wrong answers, which is worse than an error.

---

## Step 5 — ARC-3 Sovereign Wiring

ARC-3 uses the SAME path but with `arc3_game_envelope()` for each game frame:

```python
# Per game step — not per full task
envelope = arc3_game_envelope(
    task_id=f"ls20:step_{step_idx}",
    frame=current_frame,
    goal_frame=goal_frame,
    available_actions=action_space,
)
result = kv.execute_task(envelope.task, route=envelope.route())
action = result.get("game_action") or result.get("recommended_action")
```

The TRM perceives the game frame as a Tablet update, reasons via its internal swarm,
and emits an action. Python submits the action to the HTTP API. That is ALL Python does.

The ARC-3 API key (`/K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt`) is used
ONLY for HTTP transport to `three.arcprize.org`. It is NOT used for ARC-2 or ARC-1.

---

## What to Keep Unchanged

- `arc_task_galaxy_seeder.py` — correct architecture, keep as-is (pass `galaxy_manager`)
- `arc_submission_formatter.py` — I-O only, keep as-is
- `arc3_sdk_agent.py` — the `_RemoteArcCompatEnv` HTTP transport is correct; wire it to
  use `kv.execute_task()` for decision-making instead of its own Python logic
- The nine-chain swarm, halting gate, Morton+LED-A*+Frustum+LOD — do NOT touch

---

## Tests

```bash
# Verify WINE layer routes ARC correctly
bash scripts/k3d_env.sh run -e k3d-cranium \
  env PYTHONPATH=$(pwd) pytest -q tests/test_arc_wine_routing.py

# Run ARC-2 evaluation through sovereign path (20 tasks)
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/arc2_local_runner.py --max-tasks 20 \
  --summary-output /tmp/arc2_sovereign_summary.json
```

`test_arc_wine_routing.py` should verify:
1. `arc2_game_envelope()` returns a `TabletEnvelope` with `surface_kind == "GAME_2D"`
2. `execute_task()` is called (not `k3d_canonicalize` directly)
3. All 9 default galaxies are present after `_boot_knowledgeverse()`
4. `gre_arc_reasoner` is dispatched for `SURFACE_KIND_GAME_2D` tasks
5. `arc_transform_inferrer` is NOT imported anywhere in the evaluation path

---

## Expected R1→Sovereign Score

The score after this wiring may still be low (the Grammar Galaxy only has training pairs
from `seed_task()` — no broader world knowledge yet). That is honest and correct.
The score will improve as:
- Grammar Galaxy accumulates more transformation rules via sleep-time consolidation
- `gre_arc_reasoner` learns to navigate the Galaxy more accurately
- The encyclopedias ingest completes (procedural knowledge feeds Grammar rules)

Do NOT add Python pattern-matching to compensate for low scores. Low score = knowledge gap,
not architecture gap. Fix knowledge, not architecture.

---

## Report Back

Write `TEMP/CODEX_TO_CLAUDE_ARC_SOVEREIGN_WIRING_REPORT_2026-04-08.md` with:

1. Confirmation that `arc_transform_inferrer.py` is deleted
2. Confirmation that `execute_task()` is now the ARC-2 evaluation entry point
3. Which swarm workers are dispatched for `SURFACE_KIND_GAME_2D`
4. Which galaxies are confirmed loaded at boot
5. ARC-2 score: `tasks=20, correct=K, total_inputs=20, score=X.XX%`
   (honest result from sovereign path — expected: still low, but now architecturally correct)
6. All tests passing: command and count

---

## What NOT to Do

- Do NOT add Python grid manipulation to compensate for wrong answers
- Do NOT call `k3d_canonicalize` directly from Python in the evaluation path
- Do NOT call `k3d_pattern_match` directly from Python in the evaluation path
- Do NOT reimport or recreate `arc_transform_inferrer` under a different name
- Do NOT add Python fallbacks if `execute_task()` returns None — log it and count as wrong
- Do NOT partial-load galaxies — ALL default galaxies or none
- Do NOT use the ARC-3 API key for ARC-2 (offline evaluation only)
- Do NOT make `arc2_local_runner.py` longer than ~150 lines — if it grows, something is
  wrong: Python is doing reasoning it shouldn't
