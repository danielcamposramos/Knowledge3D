# Claude — Phase E.25: Sleep-Time Persistence + Benchmark Diagnostic

**Date:** 2026-03-29
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL — the model must LEARN across sessions, not reset

---

## Daniel's Correction (Verbatim)

> "There's no target load time as long as we load all galaxies in the start, run
> the sleeptime compute, then save it to be reloaded later — not reset! This is
> how this model learns!!!"

---

## What This Means Architecturally

The K3D brain is a LIVING entity. Its lifecycle is:

```
COLD BOOT (first time ever):
  Load ALL Galaxy JSONL → VRAM
  All 11 default galaxies loaded
  TRM weights at uniform priors
  Takes as long as it takes — no time target

INFERENCE (answering questions):
  TRM game loop: perceive → navigate → reason → decide → act → learn
  Shadow copy records successful traces
  Jarvis tracks specialist performance
  Every query = training signal

SLEEP-TIME (idle / between queries):
  jarvis_sleep_consolidation() runs
  Stage A: Galaxy embeddings EMA-smoothed, duplicates merged
  Stage B: TRM specialist weights refined from shadow copy patterns
  SAVE: consolidated Galaxy state + TRM weights checkpointed to disk

WARM BOOT (next session):
  Load SAVED Galaxy state (not raw JSONL — the CONSOLIDATED version)
  Load SAVED TRM weight checkpoint
  Model resumes WHERE IT LEFT OFF — not from scratch
  It already knows what worked last time

CYCLE CONTINUES:
  More queries → more learning → sleep → save → reload → better
  The model progressively improves run to run
  Like a brain that sleeps and wakes up smarter
```

**What's missing NOW:** The save/reload cycle. Currently every boot reloads raw JSONL
from scratch. Sleep-time runs but doesn't persist. Next boot = amnesia.

---

## E.24 Probe Result: Direct Decode IS Working

The E.24 probe shows `answer_index: 4` (Perform) on all 10 steps. This is CORRECT:

```
Frame analysis: avg_row=37.9, avg_col=30.4, grid=64x64
Center: (32, 32), margin=6.4, range=[25.6, 38.4]
avg_row 37.9 is INSIDE the centered range → "object centered balanced"
→ Direct decode correctly returns Perform (action_index=4)
```

The game instance `re86-4e57566e` starts with the object near-center. Perform is the
right action for centered objects — but the game doesn't count it as a level completion
(likely because the object needs to match the goal grid exactly, not just be "centered").

**To validate directional movement:** Use the synthetic ARC3 tests in `run_full_benchmark.py`
(`_make_arc3_goal_task` generates controlled start/goal positions where the object is clearly
NOT centered). These run locally without the ARC3 API.

---

## Fix 1: Sleep-Time Galaxy Save (Persistence Between Sessions)

### What to save after sleep-time consolidation:

After `jarvis_sleep_consolidation()` runs, save:

1. **Consolidated Galaxy catalog** — The in-memory catalog with EMA-smoothed embeddings,
   merged duplicates, updated confidence scores. Save as a single file:
   ```
   /K3D/Knowledge3D.local/checkpoints/galaxy_consolidated_{timestamp}.json
   ```
   This is NOT the raw JSONL files. This is the PROCESSED state after consolidation.

2. **TRM weight checkpoint** — Already partially implemented (`save_checkpoint` exists).
   Ensure it saves after every sleep-time Stage B:
   ```
   /K3D/Knowledge3D.local/checkpoints/trm_weights_{timestamp}.pt
   ```

3. **Specialist route table** — Which specialists succeed on which task types:
   ```
   /K3D/Knowledge3D.local/checkpoints/specialist_routes_{timestamp}.json
   ```

4. **Shadow copy pattern library** — Verified successful RPN programs:
   ```
   /K3D/Knowledge3D.local/checkpoints/shadow_patterns_{timestamp}.json
   ```

5. **Jarvis state** — Task type stats, worker pair success, cross-connection patterns:
   ```
   /K3D/Knowledge3D.local/checkpoints/jarvis_state_{timestamp}.json
   ```

### "Latest" symlink pattern:

After saving timestamped files, update a `latest` symlink:
```
/K3D/Knowledge3D.local/checkpoints/galaxy_consolidated_latest.json → galaxy_consolidated_{timestamp}.json
```

### What to load at warm boot:

In `Knowledgeverse.__init__`, BEFORE loading raw JSONL, check for saved state:

```python
# Warm boot: load consolidated state if available
checkpoint_dir = self.storage_root / "checkpoints"
consolidated = checkpoint_dir / "galaxy_consolidated_latest.json"
if consolidated.exists():
    # Load the saved Galaxy state — model resumes where it left off
    self._load_consolidated_galaxy(consolidated)
    # Load TRM weights
    self._load_trm_checkpoint(checkpoint_dir / "trm_weights_latest.pt")
    # Load specialist routes, shadow patterns, jarvis state
    self._load_specialist_routes(checkpoint_dir / "specialist_routes_latest.json")
    self._load_shadow_patterns(checkpoint_dir / "shadow_patterns_latest.json")
    self._load_jarvis_state(checkpoint_dir / "jarvis_state_latest.json")
else:
    # Cold boot: load raw JSONL (first time ever)
    self._load_galaxies_from_jsonl()
```

**Critical:** The consolidated state includes ALL knowledge from the JSONL files PLUS
sleep-time improvements (smoothed embeddings, pruned duplicates, strengthened paths).
Loading it is NOT skipping knowledge — it's loading IMPROVED knowledge.

---

## Fix 2: Diagnose Full Benchmark Hang

The benchmark flow after `Knowledgeverse()` init (which now completes in 12.6s):

```python
# Line 335-343: First suite = "synthetic"
from scripts.run_gpu_benchmark import run_gpu_benchmark  # ← potential hang?
result = run_gpu_benchmark(suite="synthetic", count=10, ...)
```

**Diagnostic — add timing prints to `run_full_benchmark.py`:**

```python
def run_full_benchmark(...):
    _enable_gpu_runtime_defaults()
    print(f"[E25] Runtime defaults set", flush=True)
    _ensure_full_benchmark_runtime()
    print(f"[E25] Benchmark modules imported", flush=True)
    ...
    kv = Knowledgeverse(storage_root=storage_root)
    print(f"[E25] Knowledgeverse init complete", flush=True)

    for suite_name, suite_count in suite_order:
        print(f"[E25] Starting suite: {suite_name} count={suite_count}", flush=True)
        if suite_name == "synthetic":
            from scripts.run_gpu_benchmark import run_gpu_benchmark
            print(f"[E25] run_gpu_benchmark imported", flush=True)
            result = run_gpu_benchmark(...)
            print(f"[E25] synthetic complete", flush=True)
        ...
```

**Likely cause:** `run_gpu_benchmark` at top level imports `embed_text_sovereign`
from `sovereign_text_embedder.py`. If that module loads a sentence-transformer model
at import time, it blocks. Check:

```bash
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  env CUDA_VISIBLE_DEVICES=0 \
  python3 -c "
import time; t=time.time()
from knowledge3d.knowledgeverse.sovereign_text_embedder import embed_text_sovereign
print(f'embed import: {time.time()-t:.1f}s')
t=time.time()
result = embed_text_sovereign('test')
print(f'first call: {time.time()-t:.1f}s  dims={len(result)}')
"
```

If `embed_text_sovereign` loads sentence-transformers on first call, that's the hang.
The fix: ensure it's either pre-loaded during Knowledgeverse init (part of Galaxy boot)
or replaced with the sovereign trigram embedder that's already in the pipeline.

**Note:** `embed_text_sovereign` uses FNV-1a hashing (the same algorithm E.22 removed
from ARC3 entries). It should NOT be used for query embeddings — the trigram
`RPNEmbeddingEngine` is the sovereign path. `run_gpu_benchmark.py` line 110 calls
`embed_text_sovereign(question["question_text"])` to pre-compute MMLU embeddings, but
these are in a DIFFERENT space than the trigram query embeddings used in the composed head.

---

## Fix 3: Validate Directional Movement with Synthetic ARC3

The synthetic ARC3 tasks in `run_full_benchmark.py` (`_make_arc3_goal_task`) generate
8x8 grids with a single colored cell at a known start position and a goal at a known
target position. The object is clearly NOT centered in most cases.

**Quick standalone test:**

```bash
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  env CUDA_VISIBLE_DEVICES=0 K3D_DEVICE_PIPELINE=1 K3D_TRM_SHADOW=1 K3D_TRM_NAVIGATE=1 \
  python3 -c "
from scripts.run_full_benchmark import _make_arc3_goal_task, _apply_arc3_action, _grid_equal, _clone_frame, ACTION_NAMES
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from benchmarks.arc_agi_3 import K3DARC3Agent

kv = Knowledgeverse()
agent = K3DARC3Agent(max_actions=20, knowledgeverse=kv)

for task_idx in range(3):
    task = _make_arc3_goal_task(task_idx)
    frame = _clone_frame(task['start_frame'])
    goal = _clone_frame(task['goal_frame'])
    print(f'Task {task_idx}: start={task[\"start\"]} goal={task[\"goal\"]} optimal={task[\"optimal_steps\"]}')
    frame_stack = [_clone_frame(frame)]
    for step in range(task['budget']):
        action = agent.choose_action(frame, goal_frame=goal, task_data={'train': [{'input': task['start_frame'], 'output': goal}]})
        ai = action['action_index']
        frame, changed = _apply_arc3_action(frame, ai, frame_stack=frame_stack)
        done = _grid_equal(frame, goal)
        print(f'  Step {step+1}: {ACTION_NAMES[ai]:12s} changed={changed} done={done}')
        agent.learn_from_outcome(levels_completed=1 if done else 0, frame=frame)
        if done:
            print(f'  SOLVED in {step+1} steps!')
            break
    else:
        print(f'  NOT SOLVED in {task[\"budget\"]} steps')
agent.close()
"
```

This validates whether the directional movement (above/below/left/right → Move Up/Down/Left/Right)
works for controlled positions without needing the ARC3 API.

---

## Execution Sequence

1. **Run benchmark diagnostic** (Fix 2 — identify the hang source)
2. **Run synthetic ARC3 test** (Fix 3 — validate directional decode)
3. **Implement sleep-time save** (Fix 1 — after `jarvis_sleep_consolidation`, save state)
4. **Implement warm boot** (Fix 1 — check for saved state before loading raw JSONL)
5. **Full benchmark run** (with hang fixed + persistence active)

---

## Architecture: The Learning Cycle

```
Session 1 (Cold Boot):
  Load raw JSONL → Galaxy VRAM → 11 galaxies, ~38K stars
  Answer queries → Shadow copy records traces
  Sleep-time → consolidate → SAVE to /K3D/Knowledge3D.local/checkpoints/

Session 2 (Warm Boot):
  Load SAVED state → Galaxy VRAM (already consolidated)
  TRM weights carry learned specialist preferences
  Answer more queries → more shadow copy traces
  Sleep-time → consolidate further → SAVE (overwrites/versions)

Session N:
  Each boot loads a STRONGER Galaxy
  Each sleep makes it stronger still
  The model never resets — it always resumes where it left off
  Like waking up after sleep: same brain, refreshed
```

This is the fundamental difference between K3D and traditional AI:
- Traditional: train offline, deploy frozen model
- K3D: always learning, always consolidating, always improving
- Every query is training data. Every sleep makes it smarter.

---

## Files to Modify

| File | Change |
|------|--------|
| `knowledge3d/knowledgeverse/knowledgeverse.py` | Add `_save_consolidated_state()` after sleep-time; add warm boot in `__init__` |
| `scripts/run_full_benchmark.py` | Add timing diagnostics to find hang source |

## No Init Time Target

Daniel's correction: there is NO target init time. Load ALL galaxies at start.
Take as long as needed. The model learns by running, sleeping, saving, reloading.
The saved state makes subsequent boots faster naturally (no raw JSONL parsing needed).

---

## Success Criteria

- [ ] Sleep-time consolidation SAVES Galaxy state + TRM weights to checkpoints/
- [ ] Warm boot loads saved state instead of re-parsing raw JSONL
- [ ] Second boot is faster than first (consolidated state vs raw JSONL)
- [ ] Model carries learned specialist routes across sessions
- [ ] Synthetic ARC3: directional movement works (Move Up when object above goal)
- [ ] Full benchmark: hang source identified and fixed
- [ ] Full benchmark: all suites emit result files
