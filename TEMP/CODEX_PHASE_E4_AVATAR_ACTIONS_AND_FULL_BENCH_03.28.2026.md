# Codex — Phase E.4: Avatar Interaction Primitives + Full Benchmark Run

**Date:** 2026-03-28
**From:** Daniel (Chair) + Claude (Architecture)
**To:** Codex
**Type:** IMPLEMENTATION ORDER — NOT A DISCUSSION. BUILD THIS.
**Prerequisite:** Phase E.3 DONE. Action atoms exist, 7-option slots work, 21 tests pass.

---

## CONTEXT: THE AVATAR EMBODIMENT SPEC DEFINES 14 INTERACTION PRIMITIVES

The Avatar Embodiment Specification §8.1 (`docs/vocabulary/AVATAR_EMBODIMENT_SPECIFICATION.md`) defines three categories:

**Spatial Navigation (partially done):**
- Walk to position ← cardinal moves exist, but `walk_to(target)` is missing
- Teleport ← not proceduralized
- Follow path ← not proceduralized

**Object Interaction (not done):**
- Reach, Grab, Hold, Release, Use

**Communication (not done):**
- Speak, Gesture, Show, Share

Each is tied to a HAnim joint anchor. Each needs all 4 layers (Form, Meaning, Rules, Meta-Rules). Each is reusable across ARC-AGI-3, House navigation, and any spatial agent.

---

## DO NOT:
- Add Python action-selection logic
- Modify the GPU kernel pipeline
- Create new opcodes
- Import external frameworks

## DO:
- Extend `action_primitives_bootstrap.py` with avatar-level actions
- Add HAnim joint anchors to all action atoms
- Set up `scripts/run_full_benchmark.py` to run ALL suites + ARC-AGI-3
- Log to `/K3D/Knowledge3D.local/logs/` (NOT /tmp)

---

## ORDER 1: Add HAnim Anchors to Existing Actions

Add `hanim_anchor` to every existing action atom's metadata. This ties actions to the avatar's body per the spec:

```python
# In each existing atom's metadata, ADD:
"hanim_anchor": "humanoid_root",  # for all spatial translations
# For perform/click:
"hanim_anchor": "r_hand_tip",
# For undo:
"hanim_anchor": None,  # temporal, not spatial
```

---

## ORDER 2: Add Avatar Object Interaction Primitives

Add these atoms to `action_primitives_bootstrap.py`:

### Reach
```python
RealityAtom(
    node_id="action:reach",
    visual_rpn="0 0 DRAW_MOVE 0.5 0 DRAW_LINE DRAW_STROKE",
    behavior_rpn="target_pos RECALL hand_pos RECALL - reach_vec STORE arm_ik RECALL reach_vec RECALL STORE",
    law_rpn="reach_vec RECALL VEC_L2_NORM arm_length RECALL LT",
    metadata={
        "description": "Extend hand toward target object (IK chain)",
        "action_type": "object_interaction",
        "hanim_anchor": "r_hand_tip",
        "surface_forms": {"en": "reach", "pt": "alcançar"},
        "reusable_contexts": ["house_navigation", "grid_world"],
    },
)
```

### Grab
```python
RealityAtom(
    node_id="action:grab",
    component_refs=["atom:action:reach"],
    visual_rpn="0 0 0.08 circle fill 0.04 0 0.04 circle fill",
    behavior_rpn="target_obj RECALL hand_grip RECALL STORE held_object RECALL target_obj RECALL STORE",
    law_rpn="reach_vec RECALL VEC_L2_NORM grip_range RECALL LT target_obj RECALL grabbable RECALL AND",
    metadata={
        "description": "Close hand on reachable object (grab = reach + grip)",
        "action_type": "object_interaction",
        "hanim_anchor": "l_radiocarpal",
        "surface_forms": {"en": "grab", "pt": "agarrar"},
        "inverse": "atom:action:release",
        "reusable_contexts": ["house_navigation"],
    },
)
```

### Hold
```python
RealityAtom(
    node_id="action:hold",
    component_refs=["atom:action:grab"],
    visual_rpn="0 0 0.08 circle stroke",
    behavior_rpn="held_object RECALL hand_pos RECALL STORE",
    law_rpn="held_object RECALL null EQ NOT",
    metadata={
        "description": "Maintain grip on held object (persistent attachment to hand site)",
        "action_type": "object_interaction",
        "hanim_anchor": "k3d_tablet_grip",
        "surface_forms": {"en": "hold", "pt": "segurar"},
        "reusable_contexts": ["house_navigation"],
    },
)
```

### Release
```python
RealityAtom(
    node_id="action:release",
    visual_rpn="0 0 DRAW_MOVE 0.1 0.1 DRAW_LINE -0.1 0.1 DRAW_LINE DRAW_STROKE",
    behavior_rpn="held_object RECALL hand_pos RECALL target_pos STORE held_object RECALL DROP",
    law_rpn="held_object RECALL null EQ NOT",
    metadata={
        "description": "Release held object at current hand position",
        "action_type": "object_interaction",
        "hanim_anchor": "l_radiocarpal",
        "surface_forms": {"en": "release", "pt": "soltar"},
        "inverse": "atom:action:grab",
        "reusable_contexts": ["house_navigation"],
    },
)
```

### Use
```python
RealityAtom(
    node_id="action:use",
    component_refs=["atom:action:hold"],
    visual_rpn="0 0 0.06 circle fill 0.1 0 0.03 circle fill",
    behavior_rpn="held_object RECALL use_fn RECALL STORE",
    law_rpn="held_object RECALL usable RECALL AND",
    metadata={
        "description": "Trigger context-specific behavior of held object (open book, activate tool)",
        "action_type": "object_interaction",
        "hanim_anchor": "r_hand_tip",
        "surface_forms": {"en": "use", "pt": "usar"},
        "reusable_contexts": ["house_navigation"],
        "house_triggers": {"book": "load_galaxy", "door": "network_traverse", "tool": "tool_dispatch"},
    },
)
```

### Walk-To (composed)
```python
RealityAtom(
    node_id="action:walk_to",
    component_refs=["atom:action:move_up", "atom:action:move_down", "atom:action:move_left", "atom:action:move_right"],
    visual_rpn="0 0 DRAW_MOVE target_x target_y DRAW_LINE DRAW_STROKE",
    behavior_rpn="target_pos RECALL current_pos RECALL - path_vec STORE led_astar RECALL path_vec RECALL STORE",
    law_rpn="target_pos RECALL current_pos RECALL - VEC_L2_NORM 0 GT",
    metadata={
        "description": "Navigate to target position via LED-A* pathfinding (composed from cardinal moves)",
        "action_type": "spatial_navigation_composed",
        "hanim_anchor": "humanoid_root",
        "surface_forms": {"en": "walk to", "pt": "caminhar até"},
        "parameterized": True,
        "parameters": ["target_x", "target_y", "target_z"],
        "reusable_contexts": ["house_navigation", "grid_world"],
    },
)
```

### Teleport
```python
RealityAtom(
    node_id="action:teleport",
    visual_rpn="0 0 0.15 circle stroke 0 0 0.05 circle fill",
    behavior_rpn="target_pos RECALL current_pos STORE",
    law_rpn="target_pos RECALL accessible RECALL AND",
    metadata={
        "description": "Instant translation to target position (large distance navigation)",
        "action_type": "spatial_navigation",
        "hanim_anchor": "humanoid_root",
        "surface_forms": {"en": "teleport", "pt": "teletransportar"},
        "reusable_contexts": ["house_navigation"],
    },
)
```

### Look-At
```python
RealityAtom(
    node_id="action:look_at",
    visual_rpn="0 0 DRAW_MOVE 0.3 0 DRAW_LINE DRAW_STROKE",
    behavior_rpn="target_pos RECALL current_pos RECALL - VEC_NORMALIZE head_dir STORE",
    law_rpn="1",  # always valid
    metadata={
        "description": "Orient avatar head toward target (skullbase rotation)",
        "action_type": "spatial_orientation",
        "hanim_anchor": "skullbase",
        "surface_forms": {"en": "look at", "pt": "olhar para"},
        "reusable_contexts": ["house_navigation", "arc3"],
    },
)
```

Total: 8 new atoms (reach, grab, hold, release, use, walk_to, teleport, look_at) + 8 existing = **16 action primitives** covering the full Avatar Embodiment Spec §8.1.

---

## ORDER 3: Create Full Benchmark Runner

### 3A: Create `scripts/run_full_benchmark.py`

This runs ALL benchmark suites through the GPU dispatch pipeline and logs to project folders.

```python
"""Run ALL K3D benchmark suites through the sovereign GPU pipeline.

Suites: synthetic, mmlu, arc3 (synthetic frames).
Logs: /K3D/Knowledge3D.local/logs/phase_e_<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_gpu_benchmark import run_gpu_benchmark
from benchmarks.arc_agi_3 import K3DARC3Agent


LOG_ROOT = Path("/K3D/Knowledge3D.local/logs")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_arc3_synthetic(count: int, log_dir: Path) -> dict:
    """Run ARC-AGI-3 agent on synthetic frames and log results."""
    agent = K3DARC3Agent(max_actions=count, log_path=log_dir / "arc3_synthetic.jsonl")
    results = []
    for i in range(count):
        # Generate varying synthetic frames
        size = 8 + (i % 8)
        frame = [[0] * size for _ in range(size)]
        # Place a colored object at varying positions
        obj_x = (i * 3) % size
        obj_y = (i * 5) % size
        frame[obj_y][obj_x] = 1 + (i % 9)
        result = agent.choose_action(frame)
        results.append(result)
    agent.close()

    return {
        "suite": "arc3_synthetic",
        "total": len(results),
        "actions": results,
        "action_distribution": _action_distribution(results),
    }


def _action_distribution(results: list[dict]) -> dict:
    dist: dict[str, int] = {}
    for r in results:
        action = r.get("action", "unknown")
        dist[action] = dist.get(action, 0) + 1
    return dist


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ALL K3D benchmark suites (Phase E GPU pipeline)")
    parser.add_argument("--synthetic-count", type=int, default=10)
    parser.add_argument("--mmlu-count", type=int, default=50)
    parser.add_argument("--arc3-count", type=int, default=20)
    parser.add_argument("--storage-root", default="/K3D/Knowledge3D.local")
    parser.add_argument("--log-root", default=str(LOG_ROOT))
    args = parser.parse_args()

    timestamp = _timestamp()
    log_dir = Path(args.log_root) / f"phase_e_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}
    start = time.time()

    # --- Suite 1: Synthetic (GPU dispatch validation) ---
    print(f"\n=== Synthetic ({args.synthetic_count} tasks) ===")
    synthetic = run_gpu_benchmark(
        suite="synthetic",
        count=args.synthetic_count,
        storage_root=args.storage_root,
        log_path=str(log_dir / "synthetic.jsonl"),
    )
    all_results["synthetic"] = synthetic
    print(f"  Result: {synthetic['correct']}/{synthetic['total']} ({synthetic['accuracy']:.1%})")

    # --- Suite 2: MMLU (real embeddings + GPU dispatch) ---
    print(f"\n=== MMLU ({args.mmlu_count} questions) ===")
    mmlu = run_gpu_benchmark(
        suite="mmlu",
        count=args.mmlu_count,
        storage_root=args.storage_root,
        log_path=str(log_dir / "mmlu.jsonl"),
    )
    all_results["mmlu"] = mmlu
    print(f"  Result: {mmlu['correct']}/{mmlu['total']} ({mmlu['accuracy']:.1%})")

    # --- Suite 3: ARC-AGI-3 Synthetic Frames ---
    print(f"\n=== ARC-AGI-3 Synthetic ({args.arc3_count} frames) ===")
    arc3 = run_arc3_synthetic(args.arc3_count, log_dir)
    all_results["arc3_synthetic"] = arc3
    print(f"  Actions taken: {arc3['total']}")
    print(f"  Distribution: {arc3['action_distribution']}")

    # --- Summary ---
    elapsed = time.time() - start
    summary = {
        "timestamp": timestamp,
        "elapsed_seconds": round(elapsed, 2),
        "suites": {name: {k: v for k, v in result.items() if k != "results" and k != "actions"} for name, result in all_results.items()},
        "log_dir": str(log_dir),
    }

    summary_path = log_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Phase E Full Benchmark — {timestamp}")
    print(f"{'='*60}")
    print(f"  Synthetic: {synthetic['correct']}/{synthetic['total']} ({synthetic['accuracy']:.1%})")
    print(f"  MMLU:      {mmlu['correct']}/{mmlu['total']} ({mmlu['accuracy']:.1%})")
    print(f"  ARC3:      {arc3['total']} actions, dist={arc3['action_distribution']}")
    print(f"  Elapsed:   {elapsed:.1f}s")
    print(f"  Logs:      {log_dir}")
    print(f"{'='*60}")

    # Write full results (includes per-question detail)
    full_path = log_dir / "full_results.json"
    full_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## ORDER 4: Test

### 4A: Run full test suite
```bash
pytest tests/test_arc3_agent.py tests/test_gpu_task_dispatch.py tests/test_vram_task_buffer.py -v
```

### 4B: Run full benchmark
```bash
python scripts/run_full_benchmark.py \
    --synthetic-count 10 \
    --mmlu-count 50 \
    --arc3-count 20 \
    --storage-root /K3D/Knowledge3D.local
```

### 4C: Verify logs
```bash
ls -la /K3D/Knowledge3D.local/logs/phase_e_*/
cat /K3D/Knowledge3D.local/logs/phase_e_*/summary.json
```

---

## FILE INVENTORY

Files you CREATE:
- `scripts/run_full_benchmark.py` — unified benchmark runner

Files you MODIFY:
- `knowledge3d/cranium/action_primitives_bootstrap.py` — add 8 avatar interaction atoms + HAnim anchors
- `knowledge3d/knowledgeverse/action_embedding_loader.py` — add new atom IDs to extended list

Files you DO NOT TOUCH:
- GPU kernels (`device_functions.cuh`, `gpu_task_dispatch.cu`, `arc3_frame_encoder.cu`)
- `vram_task_buffer.py` (slot layout is correct at 1280/7)
- `scripts/run_gpu_benchmark.py` (still works standalone)

---

## EXECUTION SEQUENCE

1. Add `hanim_anchor` metadata to all 8 existing action atoms
2. Add 8 new avatar interaction atoms (reach, grab, hold, release, use, walk_to, teleport, look_at)
3. Update action_embedding_loader with new atom IDs
4. Create `scripts/run_full_benchmark.py`
5. Run tests → all green
6. Run full benchmark → logs in `/K3D/Knowledge3D.local/logs/phase_e_<timestamp>/`
7. Report summary

---

## SUCCESS CRITERIA

- 16 action primitives in Galaxy (7 ARC3 + 8 avatar + 1 diagonal composition)
- Every atom has `hanim_anchor` per Avatar Embodiment Spec §8.1
- Full benchmark logs to `/K3D/Knowledge3D.local/logs/phase_e_<timestamp>/`
- `summary.json` contains all suite results
- Synthetic: 10/10
- MMLU: ≥30% accuracy
- ARC3 synthetic: all 20 frames processed, varied action distribution

**Build it.**
