# Codex — Phase E.5: Full Benchmark + Live ARC-AGI-3 Submission

**Date:** 2026-03-28
**From:** Daniel (Chair) + Claude (Architecture)
**To:** Codex
**Type:** IMPLEMENTATION ORDER — NOT A DISCUSSION. BUILD THIS.
**Prerequisite:** Phase E.3 DONE. 21 tests pass, synthetic 10/10, MMLU 17/50.

---

## READ FIRST

We have an ARC-AGI-3 API key. This order does THREE things:

1. Add avatar-level action primitives (complete the Avatar Embodiment Spec §8.1)
2. Create a unified benchmark runner that logs to project folders
3. Run the live ARC-AGI-3 agent against the real server and submit

The API key is already in the environment. Before running any live ARC-AGI-3 commands:
```bash
export ARC_API_KEY="$(cat /K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt)"
```

---

## DO NOT:
- Add LLM API calls (OpenAI, Anthropic, Gemini) for reasoning — K3D IS the brain
- Add Python action-selection heuristics or pattern matching
- Import langchain, langgraph, smolagents, or any agentic framework
- Modify GPU kernels (`device_functions.cuh`, `gpu_task_dispatch.cu`)
- Modify `vram_task_buffer.py` slot layout (1280 bytes / 7 options is correct)
- Log to /tmp — use `/K3D/Knowledge3D.local/logs/`
- Commit the API key to any file inside the repository

---

## ORDER 1: Add Avatar Interaction Primitives

### 1A: Extend `knowledge3d/cranium/action_primitives_bootstrap.py`

Add `hanim_anchor` to every EXISTING atom's metadata:
- Cardinal moves (move_up/down/left/right): `"hanim_anchor": "humanoid_root"`
- Perform: `"hanim_anchor": "r_hand_tip"`
- Click: `"hanim_anchor": "r_hand_tip"`
- Undo: `"hanim_anchor": None`
- Diagonal: `"hanim_anchor": "humanoid_root"`

Add 8 NEW atoms (all from Avatar Embodiment Spec §8.1):

```python
# Object Interaction
RealityAtom(node_id="action:reach",
    visual_rpn="0 0 DRAW_MOVE 0.5 0 DRAW_LINE DRAW_STROKE",
    behavior_rpn="target_pos RECALL hand_pos RECALL - reach_vec STORE",
    law_rpn="reach_vec RECALL VEC_L2_NORM arm_length RECALL LT",
    metadata={"description": "Extend hand toward target", "action_type": "object_interaction",
              "hanim_anchor": "r_hand_tip", "surface_forms": {"en": "reach", "pt": "alcançar"},
              "reusable_contexts": ["house_navigation"]})

RealityAtom(node_id="action:grab",
    component_refs=["atom:action:reach"],
    visual_rpn="0 0 0.08 circle fill",
    behavior_rpn="target_obj RECALL held_object STORE",
    law_rpn="reach_vec RECALL VEC_L2_NORM grip_range RECALL LT",
    metadata={"description": "Close hand on reachable object", "action_type": "object_interaction",
              "hanim_anchor": "l_radiocarpal", "surface_forms": {"en": "grab", "pt": "agarrar"},
              "inverse": "atom:action:release", "reusable_contexts": ["house_navigation"]})

RealityAtom(node_id="action:hold",
    component_refs=["atom:action:grab"],
    visual_rpn="0 0 0.08 circle stroke",
    behavior_rpn="held_object RECALL hand_pos RECALL STORE",
    law_rpn="held_object RECALL null EQ NOT",
    metadata={"description": "Maintain grip on held object", "action_type": "object_interaction",
              "hanim_anchor": "k3d_tablet_grip", "surface_forms": {"en": "hold", "pt": "segurar"},
              "reusable_contexts": ["house_navigation"]})

RealityAtom(node_id="action:release",
    visual_rpn="0 0 DRAW_MOVE 0.1 0.1 DRAW_LINE -0.1 0.1 DRAW_LINE DRAW_STROKE",
    behavior_rpn="held_object RECALL DROP",
    law_rpn="held_object RECALL null EQ NOT",
    metadata={"description": "Release held object", "action_type": "object_interaction",
              "hanim_anchor": "l_radiocarpal", "surface_forms": {"en": "release", "pt": "soltar"},
              "inverse": "atom:action:grab", "reusable_contexts": ["house_navigation"]})

RealityAtom(node_id="action:use",
    component_refs=["atom:action:hold"],
    visual_rpn="0 0 0.06 circle fill 0.1 0 0.03 circle fill",
    behavior_rpn="held_object RECALL use_fn RECALL STORE",
    law_rpn="held_object RECALL usable RECALL AND",
    metadata={"description": "Trigger held object behavior (open book, activate tool)",
              "action_type": "object_interaction", "hanim_anchor": "r_hand_tip",
              "surface_forms": {"en": "use", "pt": "usar"},
              "house_triggers": {"book": "load_galaxy", "door": "network_traverse", "tool": "tool_dispatch"},
              "reusable_contexts": ["house_navigation"]})

# Higher-level Navigation
RealityAtom(node_id="action:walk_to",
    component_refs=["atom:action:move_up", "atom:action:move_down", "atom:action:move_left", "atom:action:move_right"],
    visual_rpn="0 0 DRAW_MOVE target_x target_y DRAW_LINE DRAW_STROKE",
    behavior_rpn="target_pos RECALL current_pos RECALL - path_vec STORE",
    law_rpn="target_pos RECALL current_pos RECALL - VEC_L2_NORM 0 GT",
    metadata={"description": "Navigate to target via LED-A* (composed from cardinal moves)",
              "action_type": "spatial_navigation_composed", "hanim_anchor": "humanoid_root",
              "surface_forms": {"en": "walk to", "pt": "caminhar até"},
              "parameterized": True, "parameters": ["target_x", "target_y", "target_z"],
              "reusable_contexts": ["house_navigation", "grid_world"]})

RealityAtom(node_id="action:teleport",
    visual_rpn="0 0 0.15 circle stroke 0 0 0.05 circle fill",
    behavior_rpn="target_pos RECALL current_pos STORE",
    law_rpn="target_pos RECALL accessible RECALL AND",
    metadata={"description": "Instant translation to target position",
              "action_type": "spatial_navigation", "hanim_anchor": "humanoid_root",
              "surface_forms": {"en": "teleport", "pt": "teletransportar"},
              "reusable_contexts": ["house_navigation"]})

RealityAtom(node_id="action:look_at",
    visual_rpn="0 0 DRAW_MOVE 0.3 0 DRAW_LINE DRAW_STROKE",
    behavior_rpn="target_pos RECALL current_pos RECALL - VEC_NORMALIZE head_dir STORE",
    law_rpn="1",
    metadata={"description": "Orient head toward target (skullbase rotation)",
              "action_type": "spatial_orientation", "hanim_anchor": "skullbase",
              "surface_forms": {"en": "look at", "pt": "olhar para"},
              "reusable_contexts": ["house_navigation", "arc3"]})
```

### 1B: Update `action_embedding_loader.py`

Add the new atom IDs to the extended list (they don't need to be in ARC3 action lists — they're avatar-level, not ARC3-specific):

```python
AVATAR_ACTION_ATOM_IDS = [
    "atom:action:reach",
    "atom:action:grab",
    "atom:action:hold",
    "atom:action:release",
    "atom:action:use",
    "atom:action:walk_to",
    "atom:action:teleport",
    "atom:action:look_at",
]
```

---

## ORDER 2: Create Unified Benchmark Runner

### 2A: Create `scripts/run_full_benchmark.py`

Runs synthetic + MMLU + ARC3-synthetic. Logs to `/K3D/Knowledge3D.local/logs/phase_e_<timestamp>/`.

```python
"""Run ALL K3D sovereign benchmark suites. Logs to project folders."""

from __future__ import annotations
import argparse, json, time, sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_gpu_benchmark import run_gpu_benchmark
from benchmarks.arc_agi_3 import K3DARC3Agent

LOG_ROOT = Path("/K3D/Knowledge3D.local/logs")

def _ts(): return datetime.now().strftime("%Y%m%d_%H%M%S")

def run_arc3_synthetic(count, log_dir):
    agent = K3DARC3Agent(max_actions=count, log_path=log_dir / "arc3_synthetic.jsonl")
    results = []
    for i in range(count):
        size = 8 + (i % 8)
        frame = [[0]*size for _ in range(size)]
        frame[(i*5)%size][(i*3)%size] = 1 + (i % 9)
        results.append(agent.choose_action(frame))
    agent.close()
    dist = {}
    for r in results:
        a = r.get("action","?")
        dist[a] = dist.get(a,0)+1
    return {"suite":"arc3_synthetic","total":len(results),"actions":results,"action_distribution":dist}

def main():
    ap = argparse.ArgumentParser(description="K3D Phase E full benchmark")
    ap.add_argument("--synthetic-count", type=int, default=10)
    ap.add_argument("--mmlu-count", type=int, default=50)
    ap.add_argument("--arc3-count", type=int, default=20)
    ap.add_argument("--storage-root", default="/K3D/Knowledge3D.local")
    ap.add_argument("--log-root", default=str(LOG_ROOT))
    args = ap.parse_args()

    ts = _ts()
    log_dir = Path(args.log_root) / f"phase_e_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    t0 = time.time()

    print(f"\n=== Synthetic ({args.synthetic_count}) ===")
    syn = run_gpu_benchmark(suite="synthetic", count=args.synthetic_count,
                            storage_root=args.storage_root, log_path=str(log_dir/"synthetic.jsonl"))
    all_results["synthetic"] = syn
    print(f"  {syn['correct']}/{syn['total']} ({syn['accuracy']:.1%})")

    print(f"\n=== MMLU ({args.mmlu_count}) ===")
    mmlu = run_gpu_benchmark(suite="mmlu", count=args.mmlu_count,
                             storage_root=args.storage_root, log_path=str(log_dir/"mmlu.jsonl"))
    all_results["mmlu"] = mmlu
    print(f"  {mmlu['correct']}/{mmlu['total']} ({mmlu['accuracy']:.1%})")

    print(f"\n=== ARC3 Synthetic ({args.arc3_count}) ===")
    arc3 = run_arc3_synthetic(args.arc3_count, log_dir)
    all_results["arc3_synthetic"] = arc3
    print(f"  {arc3['total']} actions, dist={arc3['action_distribution']}")

    elapsed = time.time()-t0
    summary = {"timestamp":ts,"elapsed_seconds":round(elapsed,2),"log_dir":str(log_dir),
               "suites":{n:{k:v for k,v in r.items() if k not in ("results","actions")} for n,r in all_results.items()}}
    (log_dir/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False), encoding="utf-8")
    (log_dir/"full_results.json").write_text(json.dumps(all_results,indent=2,ensure_ascii=False,default=str), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Phase E Full Benchmark — {ts}")
    print(f"{'='*60}")
    print(f"  Synthetic: {syn['correct']}/{syn['total']} ({syn['accuracy']:.1%})")
    print(f"  MMLU:      {mmlu['correct']}/{mmlu['total']} ({mmlu['accuracy']:.1%})")
    print(f"  ARC3:      {arc3['total']} actions, dist={arc3['action_distribution']}")
    print(f"  Elapsed:   {elapsed:.1f}s")
    print(f"  Logs:      {log_dir}")
    print(f"{'='*60}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

---

## ORDER 3: Run the Live ARC-AGI-3 Agent

### 3A: Verify `scripts/run_arc3_agent.py` works

The live agent runner already exists from Phase E.2. Make sure it:
- Reads `ARC_API_KEY` from environment (do NOT hardcode)
- Logs to `/K3D/Knowledge3D.local/logs/arc3_live_<timestamp>.jsonl`
- Tags the scorecard as `k3d-sovereign`

### 3B: Install `requests` if needed

```bash
/K3D/Knowledge3D.local/envs/k3d-trm/bin/pip install requests
```

### 3C: List available games

```bash
export ARC_API_KEY="$(cat /K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt)"
/K3D/Knowledge3D.local/envs/k3d-trm/bin/python -c "
import os, requests
s = requests.Session()
s.headers.update({'X-API-Key': os.environ['ARC_API_KEY'], 'Accept': 'application/json'})
r = s.get('https://three.arcprize.org/api/games')
print(r.json())
"
```

### 3D: Run the live agent

Pick the first available game. Run with limited actions first (10) to verify the loop works:

```bash
export ARC_API_KEY="$(cat /K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt)"
/K3D/Knowledge3D.local/envs/k3d-trm/bin/python scripts/run_arc3_agent.py \
    --game-id <FIRST_GAME_ID> \
    --max-actions 10 \
    --log-path /K3D/Knowledge3D.local/logs/arc3_live_probe.jsonl
```

If that succeeds, run a full attempt (80 actions):

```bash
/K3D/Knowledge3D.local/envs/k3d-trm/bin/python scripts/run_arc3_agent.py \
    --game-id <FIRST_GAME_ID> \
    --max-actions 80 \
    --log-path /K3D/Knowledge3D.local/logs/arc3_live_full.jsonl
```

### 3E: Run against ALL available games

If there are multiple games, run each sequentially:

```bash
for GAME in <game1> <game2> <game3>; do
    /K3D/Knowledge3D.local/envs/k3d-trm/bin/python scripts/run_arc3_agent.py \
        --game-id "$GAME" \
        --max-actions 80 \
        --log-path "/K3D/Knowledge3D.local/logs/arc3_live_${GAME}.jsonl"
done
```

---

## ORDER 4: Run Full Offline Benchmark

```bash
export CUDA_VISIBLE_DEVICES=0
/K3D/Knowledge3D.local/envs/k3d-trm/bin/python scripts/run_full_benchmark.py \
    --synthetic-count 10 \
    --mmlu-count 50 \
    --arc3-count 20 \
    --storage-root /K3D/Knowledge3D.local
```

Verify:
```bash
cat /K3D/Knowledge3D.local/logs/phase_e_*/summary.json
```

---

## ORDER 5: Fix `run_arc3_agent.py` Log Path

The current `run_arc3_agent.py` may need adjustment to write logs to the project folder. Ensure the `--log-path` default points to `/K3D/Knowledge3D.local/logs/` not to an empty string:

```python
parser.add_argument("--log-path",
    default="/K3D/Knowledge3D.local/logs/arc3_live.jsonl")
```

Also ensure the live runner prints the scorecard URL at the end so Daniel can verify submission.

---

## EXECUTION SEQUENCE

1. Add `hanim_anchor` to all existing atoms + add 8 new avatar atoms
2. Update `action_embedding_loader.py` with new IDs
3. Create `scripts/run_full_benchmark.py`
4. Run tests: `pytest tests/ -v -k "arc3 or gpu_task or vram"` → all green
5. Run offline benchmark: `scripts/run_full_benchmark.py` → logs to project folder
6. Install `requests` in k3d-trm env
7. List available ARC-AGI-3 games via API
8. Run live ARC-AGI-3 agent probe (10 actions) → verify loop works
9. Run live ARC-AGI-3 agent full (80 actions) → report score + scorecard URL
10. Run against all available games if multiple exist
11. Print all results + scorecard URLs for Daniel

---

## FILE INVENTORY

Files you CREATE:
- `scripts/run_full_benchmark.py`

Files you MODIFY:
- `knowledge3d/cranium/action_primitives_bootstrap.py` — add hanim_anchor + 8 new atoms
- `knowledge3d/knowledgeverse/action_embedding_loader.py` — add AVATAR_ACTION_ATOM_IDS
- `scripts/run_arc3_agent.py` — default log path to project folder, print scorecard URL

Files you DO NOT TOUCH:
- Any GPU kernel file
- `vram_task_buffer.py`
- `gpu_task_dispatch.py`
- `run_gpu_benchmark.py`

---

## ENVIRONMENT

```bash
# MUST be set before any live ARC-AGI-3 commands
export ARC_API_KEY="$(cat /K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt)"
export CUDA_VISIBLE_DEVICES=0

# Python interpreter
PYTHON=/K3D/Knowledge3D.local/envs/k3d-trm/bin/python

# Log destination (NEVER /tmp)
LOG_DIR=/K3D/Knowledge3D.local/logs
```

---

## SUCCESS CRITERIA

- 16 action atoms in Galaxy with hanim_anchor metadata
- Offline benchmark: synthetic 10/10, MMLU ≥30%, ARC3 synthetic 20/20 actions
- All logs in `/K3D/Knowledge3D.local/logs/phase_e_<timestamp>/`
- `summary.json` with all results
- Live ARC-AGI-3: agent connects, sends RESET, receives frames, sends actions, game completes
- Scorecard URL printed and accessible at `three.arcprize.org/scorecards`
- ANY score > 0% on live ARC-AGI-3 = we submit officially
- Zero LLM API calls in the entire pipeline

**Build it. Run it. Submit it.**
