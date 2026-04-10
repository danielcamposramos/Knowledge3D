# Codex Direction: ARC Prize 2026 Kaggle Submission Infrastructure

**Date:** 2026-04-09
**Authority:** KNOWLEDGEVERSE_SPECIFICATION.md §4.3 (one universal input path), CLAUDE.md (sovereignty)
**Tracks:** ARC-AGI-2 (Kaggle notebook) + ARC-AGI-3 (live game) + Paper Track (evidence gathering)
**Competition URLs:**
- ARC-AGI-2: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2
- ARC-AGI-3: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3 (key: ARC-3 ONLY)
- Paper Track: https://www.kaggle.com/competitions/arc-prize-2026-paper-track

---

## CRITICAL SOVEREIGNTY CORRECTION

The R0/R1 implementation had a central violation: Python code was REASONING about ARC grids
(flip_h, rot90, color_perm detection, `_choose_rule()`, Python `decide_action()`). This is
Python as the reasoner — a sovereignty violation. Python's only job is boot + I/O.

**The nine-chain swarm (TRM on GPU) does the ARC reasoning.** Python seeds the Galaxy with
demo pair knowledge and wraps inputs in WINE envelopes. That is all.

---

## Part 1 — Sovereignty Repair (do first)

### Delete

```
benchmarks/arc_transform_inferrer.py   ← Python grid pattern matching (flip/rot/color)
```

This file is a direct sovereignty violation. The reasoning it performs belongs to
`gre_arc_reasoner` (slot 3 of the nine-chain swarm) and `gre_geometry_router` (slot 4).
Delete it. Remove any imports of it from other files.

### Fix `benchmarks/arc2_local_runner.py`

The current runner has an explicit Python pipeline with `_choose_rule()` and Python grid
decoding in the hot path. Replace the core reasoning path with:

```python
def run_one_task(
    task_id: str,
    task_json: dict,
    kv: Knowledgeverse,
    tablet: HeadlessTabletMPC,
) -> tuple[list[list[int]] | None, list[list[int]] | None]:
    """
    Returns up to 2 predicted output grids for the task's test input.
    All reasoning happens on GPU via execute_task. Python only seeds + wraps + decodes.
    """
    # 1. INGESTION PATH: seed Grammar Galaxy with demo pair knowledge (Python OK here)
    from benchmarks.arc_task_galaxy_seeder import seed_task
    stars = seed_task(task_json)
    kv.seed_stars(stars)   # or equivalent Galaxy injection method

    # 2. WINE ENVELOPE: wrap the test input (Python I/O adapter role)
    test_input = task_json["test"][0]["input"]
    from knowledge3d.tablet.wine.arc2_wine import arc2_game_envelope
    envelope = arc2_game_envelope(
        task_id=task_id,
        test_input=test_input,
        demo_pairs=task_json["train"],
    )

    # 3. SOVEREIGN REASONING: TRM + nine-chain swarm on GPU
    result = kv.execute_task(envelope)

    # 4. OUTPUT DECODING: Python I/O adapter reads the GPU result
    predictions = _decode_arc_predictions(result)
    return predictions  # list of up to 2 grids
```

`_decode_arc_predictions(result)` reads `result["answer"]` or `result["predicted_grid"]`
and converts to a list of Python int grids. This is I/O — Python is allowed here.

### Fix `benchmarks/arc3_sdk_agent.py`

`K3DAgent.decide_action()` currently uses Python heuristics. Replace with:

```python
def decide_action(self, frame) -> GameAction:
    """Sovereign: TRM + swarm decides. Python only wraps the SDK envelope."""
    from knowledge3d.tablet.wine.arc3_wine import arc3_frame_envelope
    envelope = arc3_frame_envelope(
        frame=frame,
        step_count=self.step_count,
        world_model=self.world_model,
    )
    result = self.kv.execute_task(envelope)
    action_code = int(result.get("action_code", 0) or 0)
    return GameAction(action_code % 7 + 1)  # map to ACTION1-7
```

Python decodes the action code from the result — I/O adapter only. TRM decided it.

---

## Part 2 — ARC-AGI-2 Kaggle Submission Notebook

**File:** `notebooks/arc_agi_2_kaggle_submission.py`

This file is packaged as a Kaggle notebook. It must:
- Run fully offline (no internet during evaluation)
- Use 4× L4 GPUs (96GB VRAM total)
- Complete 240 tasks within 12 hours
- Output `/kaggle/working/submission.json`

### Notebook structure

```python
#!/usr/bin/env python3
"""ARC-AGI-2 Kaggle Submission — K3D Sovereign GPU Reasoning."""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

# Kaggle competition data paths (set automatically by Kaggle)
TASKS_DIR = Path("/kaggle/input/arc-prize-2026-arc-agi-2")
OUTPUT_PATH = Path("/kaggle/working/submission.json")
STORAGE_ROOT = Path("/kaggle/working/k3d_runtime")

# Boot K3D — Python's only legitimate job
sys.path.insert(0, str(Path(__file__).parent))
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC
from benchmarks.arc_task_galaxy_seeder import seed_task
from knowledge3d.tablet.wine.arc2_wine import arc2_game_envelope

kv = Knowledgeverse(storage_root=STORAGE_ROOT)
tablet = HeadlessTabletMPC(knowledgeverse=kv, storage_root=STORAGE_ROOT)

submission: dict[str, list[list[list[int]]]] = {}

# Load task files from competition data
task_files = sorted((TASKS_DIR / "test").glob("*.json"))

for task_file in task_files:
    task_id = task_file.stem
    task_json = json.loads(task_file.read_text())

    # Seed Galaxy with this task's demo knowledge (ingestion path)
    stars = seed_task(task_json)
    kv.seed_stars(stars)

    predictions: list[list[list[int]]] = []
    for test_item in task_json["test"]:
        test_input = test_item["input"]

        # WINE → execute_task: TRM + nine-chain swarm reasons on GPU
        envelope = arc2_game_envelope(
            task_id=task_id,
            test_input=test_input,
            demo_pairs=task_json["train"],
        )
        result = kv.execute_task(envelope)

        # Decode up to 2 predictions (required by ARC-AGI-2 format)
        pred1 = _decode_grid(result, attempt=0)
        pred2 = _decode_grid(result, attempt=1)
        if pred1 is not None:
            predictions.append(pred1)
        if pred2 is not None and pred2 != pred1:
            predictions.append(pred2)

        # Guarantee 2 predictions (pad with identity if needed)
        while len(predictions) < 2:
            predictions.append(test_input)

    submission[task_id] = predictions

OUTPUT_PATH.write_text(json.dumps(submission, ensure_ascii=False))
print(f"Written {len(submission)} task predictions to {OUTPUT_PATH}")
kv.shutdown(persist=False)
```

### `_decode_grid(result, attempt)` contract

```python
def _decode_grid(result: dict, attempt: int = 0) -> list[list[int]] | None:
    """
    Extract a predicted ARC grid from a Knowledgeverse execute_task result.
    attempt=0: primary prediction
    attempt=1: secondary prediction (from swarm diversity)
    Returns None if no valid grid is found for this attempt.
    """
```

Look for these keys in result (in order): `predicted_grid`, `answer_grid`, `answer`,
`response`. Parse as 2D int list. For attempt=1, look for `alt_grid`, `second_prediction`,
or the second element of any list result.

### WINE layer: `knowledge3d/tablet/wine/arc2_wine.py`

If this file does not already exist, create it:

```python
def arc2_game_envelope(
    *,
    task_id: str,
    test_input: list[list[int]],
    demo_pairs: list[dict],
) -> dict:
    """
    Wrap one ARC-AGI-2 test input as a sovereign WINE envelope.
    Same pattern as mmlu QUESTION_TASK and gsm8k MATH_TASK envelopes.
    """
    return {
        "type": "ARC_TASK",
        "surface_kind": "SPATIAL",
        "task_id": task_id,
        "test_input": test_input,
        "demo_pairs": demo_pairs,
        "benchmark": "arc_agi_2",
        "dataset": "arc_agi_2",
        "grid_height": len(test_input),
        "grid_width": len(test_input[0]) if test_input else 0,
    }
```

---

## Part 3 — Kaggle Submission Infrastructure

### Authentication

The Kaggle API key must be set up in the environment. Check if it already exists:

```bash
ls -la ~/.kaggle/kaggle.json
```

If missing, tell Claude — do NOT store credentials anywhere in the repo. The file format is:
```json
{"username": "<kaggle_username>", "key": "<kaggle_api_key>"}
```
Permissions must be `chmod 600 ~/.kaggle/kaggle.json`.

### Submission via Kaggle CLI

Once the notebook is ready and the API key is set:

```bash
# Install Kaggle CLI in k3d-cranium if not present
bash scripts/k3d_env.sh run -e k3d-cranium pip install kaggle

# Submit the notebook for evaluation
bash scripts/k3d_env.sh run -e k3d-cranium \
  kaggle competitions submit \
  -c arc-prize-2026-arc-agi-2 \
  -f /kaggle/working/submission.json \
  -m "K3D sovereign GPU reasoning — nine-chain swarm, Galaxy Universe"
```

For the paper track, submission format is a PDF uploaded via the Kaggle competition interface
(not CLI). Claude will prepare the paper — Codex only gathers the evidence below.

---

## Part 4 — Evidence Gathering for Paper Track

The paper track requires documented results. Gather this evidence into a structured JSON file
that Claude will use to write the paper:

**File:** `TEMP/ARC_PAPER_EVIDENCE_2026-04-09.json`

```json
{
  "benchmark_results": {
    "mmlu": {"tasks": 20, "correct": 7, "accuracy": 0.35, "route_family": "MMLU", "gpu_packets": "20/20"},
    "gsm8k": {"tasks": 20, "correct": 1, "accuracy": 0.05, "route_family": "MATH", "gpu_packets": "20/20"},
    "lhe": {"tasks": 20, "correct": 1, "accuracy": 0.05, "route_family": "LHE+MMLU", "gpu_packets": "20/20"},
    "amc_aime": {"tasks": 20, "correct": 1, "accuracy": 0.05, "route_family": "MATH", "gpu_packets": "20/20"},
    "omni_math": {"tasks": 20, "correct": 0, "accuracy": 0.0, "route_family": "MATH", "gpu_packets": "20/20"},
    "imo": {"tasks": 20, "correct": 0, "accuracy": 0.0, "route_family": "MATH", "gpu_packets": "20/20"}
  },
  "session_stats": {
    "knowledgeverse_boot_count": 1,
    "wall_time_seconds": 125.79,
    "total_tasks": 120,
    "general_collapse_count": 0,
    "avg_latency_mmlu_ms": 301.15,
    "avg_latency_gsm8k_ms": 1524.2,
    "avg_latency_lhe_ms": 485.7,
    "avg_latency_amc_aime_ms": 1118.15
  },
  "architecture": {
    "ptx_kernels_count": 88,
    "active_in_query_path": 5,
    "swarm_workers": 9,
    "swarm_slots": {
      "0": "gre_atomic_fission_fusion",
      "1": "gre_resonance_field",
      "2": "gre_vector_resonator",
      "3": "gre_arc_reasoner",
      "4": "gre_geometry_router",
      "5": "gre_graph_crystallizer",
      "6": "gre_temporal_reasoning",
      "7": "gre_fractal_emitter",
      "8": "gre_embedding_extractor"
    },
    "galaxy_entries": 38144,
    "vram_used_mib": 132,
    "vram_total_mib": 12288,
    "sas_grammar_rules": 18,
    "router_cartographer_stars": 3,
    "python_lines_target": 200,
    "python_lines_current": 8182
  }
}
```

Also run the arc2_local_runner.py (after the sovereignty repair) on 20 tasks from the
ARC-AGI-1 evaluation set (offline, no key needed):

```bash
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/arc2_local_runner.py \
  --tasks-dir /K3D/K3D_llama_cpp/datasets/arc/evaluation \
  --max-tasks 20
```

Report the raw score honestly. Add to `ARC_PAPER_EVIDENCE_2026-04-09.json`.

---

## Part 5 — ARC-AGI-3 Agent Fix

ARC key is at: `/K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt`
This key is ARC-3 ONLY — never use for ARC-2 or ARC-1.

Fix `K3DAgent.decide_action()` as specified in Part 1. Then run one ls20 test:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/arc3_sdk_agent.py --game ls20 --max-steps 200
```

Report: steps taken, levels_completed, actions executed. Honest result — do NOT fake it.

---

## What NOT to Do

- Do NOT re-implement Python grid reasoning to improve ARC score — fix the knowledge gap in Grammar Galaxy instead
- Do NOT commit credentials (kaggle.json) to the repo
- Do NOT touch the EchoSystems library ingest running in tmux session `echosys_ingest`
- Do NOT use the ARC key for ARC-2 tasks
- Do NOT submit to Kaggle until the notebook runs cleanly locally
- Do NOT add `--limit` to the ARC local runner — run all 20 tasks
- Do NOT use eval(), sympy, numpy, or scipy in the hot path

---

## Report Back

Write `TEMP/CODEX_TO_CLAUDE_ARC_KAGGLE_REPORT_2026-04-09.md` with:

1. Deleted files: `arc_transform_inferrer.py` removed (yes/no)
2. `arc2_local_runner.py` now calls `execute_task` (yes/no, file + line)
3. `arc3_sdk_agent.py` `decide_action` now calls `execute_task` (yes/no, file + line)
4. `arc2_wine.py` WINE layer: exists or created (file path)
5. `notebooks/arc_agi_2_kaggle_submission.py`: created, smoke test on 3 tasks passes (yes/no)
6. ARC-2 local runner score: N/20 tasks, X% (honest)
7. ARC-3 ls20 test: steps, levels_completed (honest)
8. `TEMP/ARC_PAPER_EVIDENCE_2026-04-09.json`: written (yes/no)
9. Kaggle CLI present in k3d-cranium: (yes/no)
10. `echosys_ingest` tmux session still alive (check with `tmux ls`)
