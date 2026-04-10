# Codex Direction: ARC Phase R0 — Testing CAS/SAS on Real ARC Tasks
**Date:** 2026-04-08
**Deadline pressure:** ARC-AGI-3 Milestone 1 = June 30, 2026 (~12 weeks)
**Depends on:** CAS Step 1 ✅, SAS Step 1 ✅, opcode namespace ✅
**Reference:** `TEMP/ARC_PRIZE_2026_COMPETITIVE_ASSESSMENT.md`

---

## Purpose

The CAS/SAS substrate is now live. The next step is to prove it on real ARC tasks,
not just unit tests. This spec wires K3D's new CAS/SAS layer into the ARC reasoning
path and gets the first measurable score on real ARC-2 tasks. ARC-3 SDK integration
runs in parallel — both unblock the June 30 Milestone 1 submission.

**This is not a benchmark optimization pass.** R0 is about closing the loop:
task JSON in → Galaxy seeding → K3D reasoning → grid prediction out → score.

---

## What Exists (do NOT rebuild)

| Asset | File | Role |
|-------|------|------|
| ARC grid ops | `knowledge3d/cranium/kernels/arc_grid_ops.cu` | 17 sovereign GPU grid transforms |
| ARC reasoner | `knowledge3d/cranium/ptx/gre_arc_reasoner.ptx` | ARC-specific reasoning kernel |
| ARC3 frame encoder | `knowledge3d/cranium/kernels/arc3_frame_encoder.cu` | 64×64 frame → Galaxy repr |
| ARC3 knowledge builder | `knowledge3d/cranium/arc3_knowledge_builder.py` | Spatial action knowledge |
| Nine-chain swarm | `knowledge3d/cranium/kernels/nine_chain_swarm_kernel.cu` | 9 parallel workers |
| Halting gate | `knowledge3d/cranium/kernels/gre_multimodal_halting_gate.cu` | Convergence check |
| Defeasible resolver | `knowledge3d/cranium/kernels/gre_defeasible_resolver.cu` | Rule arbitration |
| CAS pool + kernels | `knowledge3d/cranium/kernels/cas_kernels.cu` | k3d_expr_build/diff/simplify |
| SAS hashcons + kernels | `knowledge3d/cranium/kernels/sas_kernels.cu` | k3d_canonicalize/pattern_match/rule_apply |
| Grammar Galaxy rules | `knowledge3d/cranium/cas_grammar_bootstrap.py` + `sas_grammar_bootstrap.py` | Existing rules |
| Sovereign bridges | `knowledge3d/cranium/bridges/sovereign_bridges.py` | All bridge calls |

---

## R0 Deliverables (4 files)

### R0-A: `knowledge3d/benchmarks/arc_task_galaxy_seeder.py`
### R0-B: `knowledge3d/cranium/kernels/arc_verification.cu`
### R0-C: `knowledge3d/benchmarks/arc2_local_runner.py`
### R0-D: `knowledge3d/benchmarks/arc3_sdk_agent.py`

---

## R0-A: `arc_task_galaxy_seeder.py`

**Purpose:** Given an ARC-2 task JSON (demonstration pairs), encode the transformation
rules as Grammar Galaxy MeaningCentricStars using the CAS/SAS system. This is the
direct application of what we just built — storing grid transformations as RPN programs
that k3d_pattern_match can retrieve and k3d_rule_apply can execute.

### Grid → RPN encoding

An ARC grid (up to 30×30, values 0-9) encodes as a compact RPN program using the
existing `arc_grid_ops.cu` opcodes. The encoding is a sequence of cell-value push
and position operations that recreates the grid from scratch:

```python
def grid_to_rpn(grid: list[list[int]]) -> str:
    """
    Encode a 2D ARC grid as a compact RPN program string.
    Uses existing arc_grid_ops opcodes for sovereign compatibility.
    Format: "HEIGHT WIDTH GRID_BEGIN [ROW_BEGIN v00 v01 ... ROW_END]... GRID_END"
    Produces a deterministic string suitable as meaning_rpn.
    """
    h, w = len(grid), len(grid[0]) if grid else 0
    tokens = [f"GRID_BEGIN {h} {w}"]
    for row in grid:
        tokens.append(f"ROW_BEGIN {' '.join(str(v) for v in row)} ROW_END")
    tokens.append("GRID_END")
    return " ".join(tokens)
```

### Transformation → Grammar rule

Each (input, output) demonstration pair becomes a Grammar Galaxy MeaningCentricStar:

```python
def pair_to_grammar_rule(
    task_id: str,
    pair_idx: int,
    input_grid: list[list[int]],
    output_grid: list[list[int]],
) -> MeaningCentricStar:
    """
    Encode one ARC demonstration pair as a Grammar rule.
    meaning_rpn = input grid pattern (what to match)
    behavior_rpn = output grid program (what to produce)
    """
    input_rpn  = grid_to_rpn(input_grid)
    output_rpn = grid_to_rpn(output_grid)
    star_id    = f"arc_rule:{task_id}:pair{pair_idx}"
    return MeaningCentricStar(
        star_id=star_id,
        meaning_class="arc_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn=input_rpn,
        behavior_rpn=output_rpn,
        taxonomy_refs=["arc", "grid_transform", task_id],
        grammar_refs=["arc_transformation"],
        confidence=1,
        polarity=1,
    )
```

### Public API

```python
def seed_task(task_json: dict, galaxy_manager=None) -> list[MeaningCentricStar]:
    """
    Parse one ARC-2 task JSON and return Grammar Galaxy stars for all demo pairs.
    task_json has keys: "train" (list of {"input": grid, "output": grid}),
                        "test"  (list of {"input": grid})
    """

def seed_tasks_directory(tasks_dir: str, galaxy_manager=None) -> dict[str, list[MeaningCentricStar]]:
    """
    Walk a directory of ARC-2 task JSON files and seed all of them.
    Returns {task_id: [stars]} for tracking.
    """

def load_task_json(path: str) -> dict:
    """Load a single ARC-2 task JSON file."""
```

**Sovereignty:** This file is ingestion-path only. May import `json`, `os`, `pathlib`,
`hashlib`. May NOT import numpy, torch, symengine, or any ML framework.
Grid values are plain Python ints. No GPU calls in this file.

---

## R0-B: `arc_verification.cu`

**Purpose:** Given a candidate output grid and a set of training (input, output) pairs,
score the candidate by exact cell match against all training outputs. This is the
GPU-native verification kernel that closes the refinement loop.

This is the single most important kernel for ARC-2 score improvement. Every refinement
cycle — swarm generates candidate → verification checks match → halting gate decides
to converge or retry — depends on this.

### Struct

```c
#include <stdint.h>

// Compact GPU representation of an ARC grid (up to 30×30 = 900 cells)
struct ArcGrid {
    uint8_t  cells[900];   // row-major, 0-9 cell values
    uint16_t height;
    uint16_t width;
};

#define ARC_MAX_DEMO_PAIRS 5   // max demonstration pairs per task
```

### Kernels

```c
// Verify one candidate grid against all training output grids.
// Returns match count (0-5): how many training output grids this candidate matches exactly.
extern "C" __global__ void arc_verify_candidate(
    const ArcGrid* __restrict__ candidate,           // the predicted output
    const ArcGrid* __restrict__ training_outputs,    // all demo pair outputs
    uint32_t n_pairs,                                // number of pairs
    uint32_t* out_match_count                        // result: 0..n_pairs
);

// Score n_candidates in parallel, one thread-block per candidate.
// out_scores[i] = match_count for candidates[i].
extern "C" __global__ void arc_score_candidates(
    const ArcGrid* __restrict__ candidates,
    uint32_t n_candidates,
    const ArcGrid* __restrict__ training_outputs,
    uint32_t n_pairs,
    uint32_t* out_scores
);
```

### Key implementation note

`arc_verify_candidate` must handle both:
- **Full match** (candidate dims == output dims AND all cells equal) — contributes 1 to match count
- **Dimension mismatch** — contributes 0 immediately (no cell comparison)

Use `__syncthreads()` + warp reduction to aggregate cell matches within a block efficiently.

Add to `sovereign_bridges.py`:
```python
def launch_arc_verify_candidate(
    self, candidate: list[list[int]], training_outputs: list[list[list[int]]]
) -> int:
    """Return number of training output grids the candidate matches exactly (0-n_pairs)."""

def launch_arc_score_candidates(
    self, candidates: list[list[list[int]]], training_outputs: list[list[list[int]]]
) -> list[int]:
    """Score all candidates in parallel. Returns list of match counts."""
```

---

## R0-C: `arc2_local_runner.py`

**Purpose:** Run K3D's full reasoning pipeline against a local directory of ARC-2 task
JSON files, measure exact-match score, and print per-task results. This is the test
harness that proves CAS/SAS works on real ARC grids before submitting to Kaggle.

**Data:** Use the ARC-AGI-1 evaluation set (publicly available, same format as ARC-2).
Download from: `https://github.com/fchollet/ARC-AGI/tree/master/data/evaluation`
OR check if the tasks are already in the repo under `data/arc/` or `benchmarks/arc/`.

### Pipeline (one task)

```python
def run_one_task(task_id: str, task_json: dict, bridge) -> tuple[int, int]:
    """
    Returns (correct_predictions, total_test_inputs).
    correct_prediction = 1 if the predicted output exactly matches ground truth.

    Pipeline:
    1. Seed Grammar Galaxy from training pairs via arc_task_galaxy_seeder
    2. For each test input:
       a. Encode test input grid as RPN (same as seed path)
       b. Build STAR node for it via bridge.launch_k3d_expr_build()
       c. Canonicalize via bridge.launch_k3d_canonicalize()
       d. Run k3d_pattern_match against each seeded training rule
          (use the rule whose input pattern canonicalizes closest to the test input)
       e. Apply best-matching rule via bridge.launch_k3d_rule_apply()
       f. Decode result STAR node back to grid
       g. If task_json has ground truth output: score with arc_verify_candidate
    3. Return score
    """
```

### Grid decoding (STAR → grid)

After rule application, the result STAR node encodes the predicted output. Decoding
traverses the STAR DAG (via Python bridge read of g_cas_pool) and reconstructs the
grid from `GRID_BEGIN H W ... ROW_BEGIN values... ROW_END ... GRID_END` structure.

```python
def decode_grid_from_star(bridge, root_idx: int) -> list[list[int]] | None:
    """
    Read the STAR DAG at root_idx from g_cas_pool and reconstruct an ARC grid.
    Returns None if the root doesn't encode a valid GRID_BEGIN...GRID_END structure.
    """
```

### Scoring and output

```python
def run_evaluation(tasks_dir: str, max_tasks: int = 50) -> dict:
    """
    Run on up to max_tasks tasks from tasks_dir.
    Print per-task result: task_id, score (0/1 per test input), total.
    Return summary: {"tasks": n, "correct": n, "total_inputs": n, "score": float}
    """
```

**Expected honest result on first run:** Low score (likely 3-10%). The seeder's pattern
matching is currently literal grid-level equality, not abstracted transformation rules.
This is fine — the R0 goal is a working closed loop, not a high score. The score will
improve with R1 (compositional Grammar rules, refinement loop).

Print the score clearly so it's in the report. Do not fake it.

---

## R0-D: `arc3_sdk_agent.py`

**Purpose:** Replace K3D's March 30 transitional I/O decode layer with the official
`arc-agi` Python SDK (`pip install arc-agi`). Connect K3D's action dispatcher to
the SDK's `env.step()` loop.

### SDK installation note

Add `arc-agi` to the k3d-cranium environment (ingestion path only — SDK is Python,
the reasoning remains sovereign):
```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/pip install arc-agi
```

If the version is behind v0.9.3, note that in the report. Local execution requires
`rendering=False` for 2000+ FPS.

### Agent structure

```python
import arc
from arc.types import GameAction  # ACTION1-7 + coordinate select

class K3DAgent:
    """
    K3D sovereign agent for ARC-AGI-3.
    Wraps the arc-agi SDK env loop around K3D's LED-A* + nine-chain swarm.
    """

    def __init__(self, game_id: str = "ls20"):
        self.env = arc.make(game_id, rendering=False)
        # K3D internal state
        self.world_model = {}    # action → (cells_changed, delta)
        self.frame_history = []  # [(frame, action, next_frame), ...]
        self.step_count = 0

    def observe(self, frame) -> dict:
        """Encode the frame into Galaxy working memory using arc3_frame_encoder."""
        ...

    def decide_action(self, observation: dict) -> GameAction:
        """
        LED-A* + nine-chain swarm → best next action.
        Falls back to systematic exploration when world model is empty.
        Exploration order: ACTION1-5 (directional) before ACTION6-7.
        """
        ...

    def update_world_model(self, prev_frame, action: GameAction, next_frame) -> None:
        """
        Compare prev_frame and next_frame cell-by-cell.
        Record which cells changed and the delta for this action.
        Feeds K3D's internal model building.
        """
        ...

    def run_level(self, max_steps: int = 500) -> dict:
        """
        Full level loop: observe → decide → step → update.
        Returns {"steps": n, "levels_completed": n, "score": float}.
        """
        obs = self.env.reset()
        while True:
            frame = obs["grid"] if isinstance(obs, dict) else obs
            action = self.decide_action(self.observe(frame))
            obs, reward, done, info = self.env.step(action)
            self.step_count += 1
            if done or self.step_count >= max_steps:
                break
        return {
            "steps": self.step_count,
            "levels_completed": info.get("levels_completed", 0),
        }

def run_ls20_test():
    """Run one game of ls20 and print result. Proves the SDK path works."""
    agent = K3DAgent("ls20")
    result = agent.run_level()
    print(f"ARC-3 ls20 test: {result}")
    return result
```

**Truthfulness boundary:** The `decide_action` implementation in R0 may use a simple
exploration heuristic (systematic action space coverage + frame-change detection) rather
than the full LED-A* path. That is fine and must be stated in the report. The goal is
a working SDK integration, not maximum performance. Do NOT fake `levels_completed` results.

---

## Tests

Create `tests/test_arc_r0_surface.py`:

```python
def test_grid_to_rpn_round_trip():
    from knowledge3d.benchmarks.arc_task_galaxy_seeder import grid_to_rpn
    grid = [[0, 1, 2], [3, 4, 5]]
    rpn = grid_to_rpn(grid)
    assert "GRID_BEGIN 2 3" in rpn
    assert "ROW_BEGIN 0 1 2 ROW_END" in rpn

def test_pair_to_grammar_rule():
    from knowledge3d.benchmarks.arc_task_galaxy_seeder import pair_to_grammar_rule
    from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar
    rule = pair_to_grammar_rule("test_task", 0, [[0, 1]], [[1, 0]])
    assert isinstance(rule, MeaningCentricStar)
    assert rule.meaning_class == "arc_rule"
    assert rule.domain == "grammar"
    assert rule.star_id == "arc_rule:test_task:pair0"

def test_seed_task_from_json():
    from knowledge3d.benchmarks.arc_task_galaxy_seeder import seed_task
    task = {
        "train": [
            {"input": [[0, 1], [2, 3]], "output": [[3, 2], [1, 0]]},
            {"input": [[0, 0], [1, 1]], "output": [[1, 1], [0, 0]]},
        ],
        "test": [{"input": [[0, 1], [0, 1]]}],
    }
    stars = seed_task(task)
    assert len(stars) == 2  # one per training pair
    assert all(s.meaning_class == "arc_rule" for s in stars)

def test_arc_seeder_no_hot_path_imports():
    src = open("knowledge3d/benchmarks/arc_task_galaxy_seeder.py").read()
    for forbidden in ["numpy", "torch", "cupy", "symengine"]:
        assert forbidden not in src, f"arc_task_galaxy_seeder must not import {forbidden}"

def test_arc3_agent_imports():
    # SDK import check — must not crash even if arc-agi not installed (graceful ImportError)
    try:
        from knowledge3d.benchmarks.arc3_sdk_agent import K3DAgent  # noqa: F401
    except ImportError as e:
        # arc-agi SDK not installed — acceptable in CI, note in report
        assert "arc" in str(e).lower()

def test_arc_verification_bridge_smoke(gpu_available):
    # Skip if no GPU
    if not gpu_available:
        return
    from knowledge3d.cranium.bridges.sovereign_bridges import ModularRPNEngine
    bridge = ModularRPNEngine()
    # Two identical 2×2 grids — should match
    candidate = [[1, 0], [0, 1]]
    training  = [[[1, 0], [0, 1]]]
    score = bridge.launch_arc_verify_candidate(candidate, training)
    assert score == 1
    # Different grids — should not match
    score2 = bridge.launch_arc_verify_candidate([[0, 0], [0, 0]], training)
    assert score2 == 0
```

---

## Validation Gate

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/pytest -q \
  tests/test_arc_r0_surface.py \
  tests/test_sovereign_sas_surface.py \
  tests/test_sovereign_cas_surface.py \
  tests/test_opcode_namespace_integrity.py \
  -x
```

Then run the local evaluation:
```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  knowledge3d/benchmarks/arc2_local_runner.py \
  --tasks-dir /path/to/arc-agi/evaluation \
  --max-tasks 20
```

Print the raw score. It will be low — that is expected and correct.

---

## Implementation Order

| Step | Task |
|------|------|
| R0-A | `arc_task_galaxy_seeder.py` — grid encoding + Grammar rule generation |
| R0-B | `arc_verification.cu` — GPU candidate verification kernel + bridge methods |
| R0-C | `arc2_local_runner.py` — closed-loop local test runner, print score |
| R0-D | `arc3_sdk_agent.py` — SDK integration, run ls20 test, print result |
| Tests | `tests/test_arc_r0_surface.py` |
| Gate | Pytest gate + both local runner outputs in the report |

---

## What NOT To Do In This Pass

- Do NOT attempt to match the ARC-2 leaderboard score — that is Phase R1
- Do NOT implement LoRA TTT (Phase R1 scope)
- Do NOT tune the nine-chain swarm for ARC specifically (Phase R1)
- Do NOT build the full Kaggle submission notebook (Phase R1 — needs R0 working first)
- Do NOT change L4 GPU configs (still developing on RTX 3070)
- Do NOT touch the protected encyclopedia ingest (PID 101379)

---

## Report Back

Write `TEMP/CODEX_TO_CLAUDE_ARC_R0_REPORT_2026-04-08.md` with:
- Confirmation of all 4 deliverables
- The actual ARC-2 local runner score (20 tasks, raw percentage)
- The ARC-3 ls20 SDK test result (steps, levels_completed)
- Any truthful gaps — especially which parts of decide_action are stubs vs. real
- arc_verification.cu PTX compile confirmation
- Protected ingest status
