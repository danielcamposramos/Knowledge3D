# CODEX PHASE E.10: Honest ARC3 Evaluation + Multi-Benchmark Wiring

**Date:** 2026-03-28
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL — Daniel cannot assess progress because scoring is fake

---

## Context

Daniel's exact words:

> "ask him to actually evaluate ARC3 correct or not? I can not understand the meaning to know if we're advancing"
> "where are the other benchmarks?"
> "Let's keep iterating in all of them, this is a single AI with multiple capabilities. Cap to small rounds as we aim for ARC3, but we don't want to skip what we already have as well"

**Problem 1 — ARC3 scoring is fake:**
`run_full_benchmark.py` line 43: `"correct": len(actions)` — counts EVERY action as correct. The "20/20" metric is meaningless. Daniel cannot tell if we're advancing.

**Problem 2 — Missing benchmarks:**
GSM8K, LHE, ARC-AGI-2 all have existing benchmark classes but are NOT wired into the GPU dispatch runner (`run_full_benchmark.py`). Only synthetic + MMLU + ARC3-synthetic are running.

**Problem 3 — Single AI, multiple capabilities:**
This is one living mind, not separate benchmark modes. All suites must run together in a single pass.

---

## Deliverable 1: Honest ARC3 Synthetic Evaluation

### What's Wrong

`run_arc3_synthetic()` in `scripts/run_full_benchmark.py`:
```python
# THIS IS FAKE — every action is counted as "correct"
return {
    "correct": len(actions),   # ← THIS LINE IS THE BUG
    "accuracy": 1.0 if actions else 0.0,
}
```

### What "Correct" Means for ARC3 Synthetic

Since we're running synthetic frames (not the live API), we need a **goal-frame comparison**. Each synthetic task should have a known goal state. The agent's actions transform the frame. Evaluate whether the final frame matches the goal.

### Implementation

In `run_arc3_synthetic()`:

1. **Generate synthetic task WITH a goal.** Each task has:
   - An initial frame (grid with a colored cell)
   - A goal frame (grid with the cell moved to a target position)
   - A correct action sequence (e.g., "move the colored cell from (3,2) to (3,4)" = 2× Move Right)

2. **Simulate action effects on the frame.** After each `choose_action()`:
   - Apply the action to the frame (move the colored cell according to ACTION1-4 displacement)
   - Call `learn_from_outcome()` with the updated frame
   - Compare current frame to goal frame

3. **Score honestly:**
   - `correct = 1` ONLY if final frame matches goal frame (grid equality)
   - `correct = 0` if the agent failed to reach the goal within its action budget
   - Track `steps_to_goal` for tasks that succeed (fewer steps = better)

4. **Minimal synthetic goal generator** (10 tasks):
   - Place a colored cell at position (r1, c1) on an 8×8 grid
   - Goal: move it to position (r2, c2)
   - Correct action count = |r2-r1| + |c2-c1| (Manhattan distance)
   - Budget: correct_count × 3 (allow 3× the optimal steps)
   - ACTION1=up(-1,0), ACTION2=down(+1,0), ACTION3=left(0,-1), ACTION4=right(0,+1)

### Success Criteria
- ARC3 synthetic "correct" count reflects ACTUAL goal completion
- A score of 0/10 is HONEST and EXPECTED at this stage — that's fine
- Action distribution still logged (diversity matters)
- `steps_to_goal` logged for successful tasks

---

## Deliverable 2: Wire GSM8K into GPU Dispatch

### Existing Infrastructure
- `benchmarks/gsm8k.py` → `GSM8KBenchmark` class (wraps `UnifiedMathBenchmark`)
- Task type: `GSM8K_TASK = 2` (already in `TASK_TYPE_IDS` and kernel switch case 2u)
- Kernel path for case 2u: `atomic_fission_fusion_device` → `temporal_reason_device`

### Implementation

Add `run_gsm8k_gpu()` to `scripts/run_full_benchmark.py`:

1. Instantiate `GSM8KBenchmark(max_questions=count)` to load questions
2. For each question, create a GPU task:
   ```python
   task = {
       "type": "GSM8K_TASK",
       "query_embedding": embed(question["question_text"]),
       "option_embeddings": [],  # open-ended, no options
       "subject": "gsm8k_math",
       "domain_hint": "word_problem",
   }
   ```
3. Pack into `VRAMTaskBuffer`, launch via `GPUTaskDispatch` with brain + galaxy
4. Read result. Compare `answer_text_hash` or `answer_index` against `correct_answer`
5. For GSM8K numeric answers: extract numeric value from kernel output, compare to expected

**Scoring:** GSM8K answers are numeric. The kernel's `answer_text_hash` field carries the FNV-1a hash of the predicted answer. Compare predicted numeric value to expected. Exact match or within ε=1e-6 for floats.

**Note:** GSM8K currently uses `kv._embed_query_gpu()` for embedding. Use the same embedding path as MMLU uses in `run_gpu_benchmark.py`. If Knowledgeverse is needed for embedding, instantiate it once and share.

### Cap: 10 questions per run

---

## Deliverable 3: Wire LHE into GPU Dispatch

### Existing Infrastructure
- `benchmarks/last_humanity_exam.py` → `LastHumanityExamBenchmark` class
- Task type: `LHE_TASK = 3` (already in `TASK_TYPE_IDS` and kernel switch case 3u)
- Kernel path for case 3u: `graph_crystallize_device`
- Questions are multiple-choice with domain tags

### Implementation

Add `run_lhe_gpu()` to `scripts/run_full_benchmark.py`:

1. Instantiate `LastHumanityExamBenchmark(max_questions=count)` to load questions
2. For each question with options, create a GPU task:
   ```python
   task = {
       "type": "LHE_TASK",
       "query_embedding": embed(question["question_text"]),
       "option_embeddings": [embed(opt) for opt in question["options"][:4]],
       "subject": question["domain"],
       "domain_hint": question["domain"],
   }
   ```
3. Pack into `VRAMTaskBuffer`, launch via `GPUTaskDispatch` with brain + galaxy
4. Read `answer_index` from result, compare to correct option index

### Scoring
- Multiple choice: `answer_index == correct_index`
- Open-ended: compare `answer_text_hash` to FNV-1a of correct answer

### Cap: 10 questions per run

---

## Deliverable 4: Wire ARC-AGI-2 into GPU Dispatch

### Existing Infrastructure
- `benchmarks/arc_agi_2.py` → `ARCAGI2Benchmark` class
- Task type: `ARC_TASK = 0` (already in `TASK_TYPE_IDS` and kernel switch case 0u)
- Kernel path for case 0u: `arc_reason_device` → `geometry_route_device` → `fractal_emit_device`
- ARC-2 tasks have input/output grid pairs; evaluation = grid match

### Implementation

Add `run_arc2_gpu()` to `scripts/run_full_benchmark.py`:

1. Instantiate `ARCAGI2Benchmark(max_tasks=count)` to load tasks
2. For each task, flatten the input grid into a 32-dim embedding (same FNV-1a hash approach as ARC3 frame encoder)
3. Create GPU task:
   ```python
   task = {
       "type": "ARC_TASK",
       "query_embedding": embed_grid(input_grid),
       "option_embeddings": [],  # ARC is generative, no options
       "subject": "arc_agi_2",
       "domain_hint": "visual_reasoning",
   }
   ```
4. Launch via `GPUTaskDispatch` with brain + galaxy
5. Scoring: ARC-2 requires grid generation, which the kernel doesn't produce yet. For now, log that the sovereign pipeline WAS invoked (GPU execution = true) and mark correctness as TBD. **Be honest about this** — don't fake the score.

### Cap: 10 tasks per run

---

## Deliverable 5: Unified Runner

### Update `run_full_benchmark()`

The function signature becomes:
```python
def run_full_benchmark(
    *,
    synthetic_count: int = 10,
    mmlu_count: int = 50,
    arc3_count: int = 10,
    gsm8k_count: int = 10,
    lhe_count: int = 10,
    arc2_count: int = 10,
    storage_root: str | Path,
    log_root: str | Path = LOG_ROOT,
) -> dict[str, object]:
```

### Run order (all share one brain + one galaxy):
1. Synthetic (10) — sanity check
2. MMLU (50) — knowledge breadth
3. GSM8K (10) — word problems
4. LHE (10) — multi-domain reasoning
5. ARC-AGI-2 (10) — visual reasoning
6. ARC3 Synthetic (10) — interactive game (HONEST scoring)

### Shared Resources
- One `PersistentBrainState` across all suites (single mind)
- One `GalaxyVRAMTable` loaded with foundational stars across all suites
- Sleep-time consolidation runs between suites (the brain learns from each benchmark)

This is critical: **the brain state carries forward from suite to suite**. What the AI learns in MMLU strengthens its reasoning for GSM8K. This is Daniel's "single AI with multiple capabilities."

### Summary Output
```
Phase E Full Benchmark — 20260328_HHMMSS
  Synthetic:  10/10  (100.0%)
  MMLU:       15/50  (30.0%)
  GSM8K:      ?/10   (?%)
  LHE:        ?/10   (?%)
  ARC-AGI-2:  ?/10   (?%)  [grid generation TBD]
  ARC3:       ?/10   (?%)  [HONEST goal-frame scoring]
  Elapsed:    ??s
```

---

## DO NOT

- **DO NOT** count actions as correct. `"correct": len(actions)` is the bug we're fixing.
- **DO NOT** add Python fallbacks for any benchmark. If the kernel can't answer, that's a 0 — log it honestly.
- **DO NOT** instantiate separate brain/galaxy per suite. ONE brain, ONE galaxy. Single mind.
- **DO NOT** skip sleep-time consolidation between suites. The brain must consolidate.
- **DO NOT** inflate scores. 0/10 is a valid honest result. We need to SEE the real numbers to know where to improve.
- **DO NOT** add new Python orchestration logic in the hot path. Embedding + packing + launching + reading = the only Python allowed.
- **DO NOT** change the kernel dispatch switch (cases 0u-8u). Those are already correct.

---

## Files to Modify / Create

| File | Change |
|------|--------|
| `scripts/run_full_benchmark.py` | Add `run_gsm8k_gpu()`, `run_lhe_gpu()`, `run_arc2_gpu()`. Fix `run_arc3_synthetic()`. Update `run_full_benchmark()` signature and body. Share brain + galaxy across suites. Add submission.json generation after ARC-2 suite. |
| `scripts/run_gpu_benchmark.py` | Add GSM8K and LHE suite support to `run_gpu_benchmark()` (optional, if cleaner) |
| `benchmarks/arc_submission_formatter.py` | CREATE — formats K3D ARC-2 results into Kaggle `submission.json` (Pass@2 format) |
| `scripts/run_arc2_submission.py` | CREATE — standalone full-eval ARC-2 submission runner (all 240+ tasks, outputs submission.json) |

## Files to Read (do not modify)

| File | Why |
|------|-----|
| `benchmarks/gsm8k.py` | Question loading interface |
| `benchmarks/last_humanity_exam.py` | Question loading interface |
| `benchmarks/arc_agi_2.py` | Task loading interface |
| `benchmarks/arc_agi_3.py` | K3DARC3Agent — already correct, use as-is |
| `knowledge3d/knowledgeverse/vram_task_buffer.py` | TASK_TYPE_IDS, slot layout |
| `knowledge3d/knowledgeverse/gpu_task_dispatch.py` | Launch interface with brain_ptr, galaxy_ptr |
| `knowledge3d/knowledgeverse/persistent_brain.py` | PersistentBrainState class |
| `knowledge3d/knowledgeverse/galaxy_vram_table.py` | GalaxyVRAMTable, build_arc3_galaxy_table |
| `knowledge3d/knowledgeverse/foundational_galaxy_builder.py` | build_foundational_galaxy_table (93 stars) |
| `knowledge3d/cranium/cuda/gpu_task_dispatch.cu` | Kernel switch cases — don't modify, just understand |
| `TEMP/CODEX_PROMPT_ARC_PRIZE_PREPARATION_03.21.2026.md` | ARC Prize submission format spec (Phase 2 section) |
| `benchmarks/arc_agi_2_adapter.py` | ArcAgi2Adapter.solve_task() — sovereign solver interface |

## Tests

Add `tests/test_e10_multi_bench.py`:
1. Test that ARC3 synthetic with a trivial 1-step goal scores honestly (not 100%)
2. Test that GSM8K, LHE, ARC2 task packing produces valid VRAM buffers
3. Test that shared brain state is non-zero after running multiple suites
4. Test that summary includes all 6 suites

---

## Sovereignty Compliance

All benchmark evaluation follows the same pattern:
1. Python loads question text → embeds to 32-float vector (ingestion path, OK)
2. Python packs into VRAMTaskBuffer (I/O, OK)
3. GPU kernel does ALL reasoning (sovereign)
4. Python reads result and compares to ground truth (I/O, OK)

No Python reasoning in the hot path. The kernel's specialist switch (cases 0u-8u) handles all task types. The brain persists across launches. The galaxy provides knowledge. This is the sovereign game loop.

---

## Expected Honest Results

Be prepared for low scores. This is GOOD — it means we're measuring reality:
- Synthetic: 10/10 (these are designed to pass)
- MMLU: ~15/50 (Galaxy neighborhood coverage still sparse)
- GSM8K: ~0-3/10 (word problem decomposition is new)
- LHE: ~0-3/10 (multi-hop reasoning is new)
- ARC-AGI-2: 0/10 (grid generation not yet in kernel)
- ARC3: ~0-2/10 (goal-frame scoring is honest now)

Daniel said it: **we need to see where we are to know where to go.** Fake 100% helps nobody.

---

## Deliverable 6: ARC-AGI-2 Online Submission Formatter

### Context

ARC Prize competition (arcprize.org / Kaggle) requires a `submission.json` in a specific format. This was specced in `TEMP/CODEX_PROMPT_ARC_PRIZE_PREPARATION_03.21.2026.md` Phase 2 but never implemented. Daniel wants to start submitting results online.

### Submission Format (Kaggle Standard)

```json
{
    "00576224": [
        {"attempt_1": [[0, 1], [2, 3]], "attempt_2": [[0, 1], [2, 4]]}
    ],
    "009d5c81": [
        {"attempt_1": [[5, 5, 5], [5, 0, 5]], "attempt_2": [[5, 5, 5], [5, 1, 5]]}
    ]
}
```

**Rules:**
- Each key = task_id (the JSON filename stem, e.g., `"00576224"`)
- Each value = list with one dict per test case (most tasks have exactly one test)
- Each dict has `attempt_1` and `attempt_2` — both are 2D integer grids
- Scoring: **Pass@2** — task is correct if EITHER attempt exactly matches the expected output
- Grid values: integers 0-9 (10 colors), max size 30×30

### Implementation

Create `benchmarks/arc_submission_formatter.py`:

```python
"""Format K3D ARC-AGI-2 results into Kaggle submission.json."""

import json
from pathlib import Path
from typing import Any


def format_submission(
    results: list[dict[str, Any]],
    output_path: str | Path = "submission.json",
) -> Path:
    submission: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        task_id = str(result["task_id"])
        attempt_1 = result.get("predicted") or result.get("predicted_output")
        attempt_2 = result.get("alternate_prediction") or attempt_1
        if attempt_1 is None:
            attempt_1 = [[0]]
        if attempt_2 is None:
            attempt_2 = attempt_1
        submission[task_id] = [{"attempt_1": attempt_1, "attempt_2": attempt_2}]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(submission, separators=(",", ":")), encoding="utf-8")
    return path
```

### Two-Attempt Strategy

For the GPU dispatch path, the kernel's nine-chain swarm naturally produces multiple candidate reasoning traces. Use these for the two attempts:

1. **attempt_1**: The primary prediction (best confidence from swarm halting gate)
2. **attempt_2**: Second-best swarm worker output, or a variation with different thinking budget

For now (E.10), both attempts can be the same prediction — what matters is that the submission file is correctly formatted and can be uploaded. The two-attempt diversification is an optimization for later.

### Integration with `run_full_benchmark.py`

After the ARC-AGI-2 suite runs in `run_full_benchmark()`:

1. Collect all ARC-2 results with their `task_id` and `predicted` grids
2. Call `format_submission(arc2_results, log_dir / "submission.json")`
3. Log the submission path in the summary

### Runner Script

Create `scripts/run_arc2_submission.py`:

1. Load ALL evaluation tasks from dataset path (not capped — full eval set)
2. Run each task through GPU dispatch with brain + galaxy (same sovereign path)
3. Format into `submission.json`
4. Print summary: total tasks, tasks with non-null predictions, output path
5. Log to `/K3D/Knowledge3D.local/logs/arc2_submission_TIMESTAMP/`

**Dataset paths** (check in order, use first that exists):
- `/K3D/Knowledge3D.local/datasets/exams/arc-src/data/evaluation/` (240+ tasks)
- `/K3D/K3D_llama_cpp/datasets/ARC-AGI-master/data/evaluation`
- `/K3D/Knowledge3D.local/datasets/arc_agi_2/evaluation`

### Honest Expectations

The kernel currently produces `answer_index` (which option was best) but does NOT produce `output_grid` (a generated 2D grid). This means:

- **For now:** The submission formatter will produce files, but predictions will be empty/default grids
- **This is honest:** We're building the pipeline end-to-end so when grid generation comes online, submission is automatic
- **What the kernel needs (future):** A grid-generation specialist that outputs predicted cell values in the output buffer

### DO NOT
- **DO NOT** fake grid predictions. If the kernel can't generate an output grid, submit `[[0]]` as placeholder
- **DO NOT** use Python to "solve" ARC tasks. The kernel is the solver
- **DO NOT** skip tasks. Every eval task gets a submission entry (even if the prediction is empty)

### Files

| File | Action |
|------|--------|
| `benchmarks/arc_submission_formatter.py` | CREATE — submission.json formatter |
| `scripts/run_arc2_submission.py` | CREATE — standalone full-eval submission runner |
| `scripts/run_full_benchmark.py` | MODIFY — add submission.json generation after ARC-2 suite |

### Tests

Add to `tests/test_e10_multi_bench.py`:
1. Test `format_submission()` produces valid JSON with correct structure
2. Test that every task_id has exactly one entry with `attempt_1` and `attempt_2`
3. Test that attempts are 2D integer lists (even if placeholder)
4. Test round-trip: load submission.json, verify all task_ids present
