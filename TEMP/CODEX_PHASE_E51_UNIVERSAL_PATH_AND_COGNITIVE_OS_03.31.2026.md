# Codex — Phase E.51: Universal Input Path + Cognitive OS Lifecycle

**Date:** 2026-03-31
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL — three systemic violations in the current code

---

## Daniel's Exact Words

> "`if task_type == "ARC_TASK"` — this is orchestration at the loading
> script that violates the universal knowledge rule as well."

> "If we clear all this nonsense following the proceduralization with
> proper symlink and dedup logic the architecture defines, we'll end up
> with a system that can solve any task we aim at its live interface."

> "We command the system to shut down properly, like a cognitive OS
> should do — sleeptime compute on purpose and cease operations, saving
> progress that will be reloaded and thus does not require bootstraps
> again."

> "In this run, sleeptime compute ran only after Codex commanded it —
> this should be automatic and internal."

> "Sleeptime compute is looking to me like CPU-bound process when we
> have plenty of kernels to do the work — I can't see any GPU usage
> during sleeptime compute."

---

## Three Systemic Violations to Fix

### Violation 1: Task-Type Routing (302 occurrences)

`knowledgeverse.py` has 302 occurrences of `task_type`. Key offenders:

```python
if task_type == "ARC_TASK":         # line 2789
if task_type == "LHE_TASK":         # line 2791
if task_type == "MMLU_TASK":        # line 2793
if task_type == "MATH_TASK":        # line 2795
if task_type in {"CHAT_TASK", ...}: # line 3049
if task_type == "MATH_TASK" and not self._is_gsm8k_math_task(payload): # line 3073 (!!)
```

Each `if task_type ==` is Python deciding HOW to solve. That's the
TRM's job. The Galaxy's symlinks and meta-rules handle routing
internally. When the TRM finds a meaning star, its symlinks say
"this involves math" or "this involves spatial reasoning" — Jarvis
dispatches accordingly. No Python branching needed.

**What the universal path looks like:**

```
Input (any format) → I/O adapter normalizes → kv.execute_task(query=...) →
  TRM embeds → Galaxy search → Find meaning star(s) →
  Jarvis reads symlinks → Dispatch specialist(s) →
  Workers execute RPN chains → Halting Gate →
  Answer
```

Same path for ARC frames, GSM8K word problems, IMO proofs, MMLU
questions, user chat. The ONLY thing that differs is the I/O adapter
that normalizes the external format into a query.

### Violation 2: Manual Sleep Trigger

Currently, `jarvis_sleep_consolidation()` is called explicitly by
the benchmark runner script. In a live always-on system:

- The system runs continuously
- Queries arrive, get answered, briefs accumulate
- When idle for N seconds → consolidation triggers AUTOMATICALLY
- This is part of the TRM game loop: "no stimuli → enter sleep state"

Per THREE_BRAIN_SYSTEM_SPECIFICATION.md and AVATAR_EMBODIMENT.md §7.4:
the AI avatar enters a "sleeping" cognitive state (seated posture,
slow pulsing glow) and consolidates during idle time.

### Violation 3: CPU-Bound Sleep Consolidation

`jarvis_sleep_consolidation()` (line 12721) is ~60 lines of Python
that sorts dictionaries and counts agreements. Meanwhile, THREE
dedicated GPU kernels exist and are NOT called:

| PTX Kernel | Source | Purpose |
|-----------|--------|---------|
| `sleep_cluster_refiner.ptx` | `kernels/sleep_cluster_refiner.cu` | Refine Galaxy clusters based on co-activation |
| `sleep_glyph_consolidator.ptx` | `kernels/sleep_glyph_consolidator.cu` | Consolidate glyph patterns |
| `sleep_time_micro.ptx` | (runtime) | Micro-consolidation passes |

Plus bridge classes: `sleep_cluster_kernels.py`, `sleep_time_consolidator.py`,
`sleep/enhanced_sleep_integration.py`.

The contrastive learning (strengthen correct paths, weaken incorrect)
should happen ON GPU: shadow copy comparison, weight delta computation,
Galaxy entry score updates — all via PTX kernels. The Python method
should LAUNCH these kernels, not DO the work.

---

## Part A: Remove Task-Type Routing from Core Path

### The Goal

`execute_task()` and `query()` should have ZERO `if task_type ==`
branches. The query goes in, the TRM navigates the Galaxy, Jarvis
dispatches based on symlinks, the answer comes out.

### How To Get There

**Step 1: Unify target galaxy selection**

Currently:
```python
if task_type == "ARC_TASK":
    target_galaxies = ARC_TARGET_GALAXIES
if task_type == "MATH_TASK":
    target_galaxies = MATH_TARGET_GALAXIES
```

After:
```python
# ALL galaxies are always available. The TRM navigates to relevant ones.
# Frustum culling + LOD handle which are "in view" for this query.
target_galaxies = ALL_DEFAULT_GALAXIES  # Always. No selection.
```

Per Daniel's correction (March 21): "ALL 117k stars must always be
loaded." And per the always-on paradigm: all default galaxies loaded
simultaneously in VRAM.

**Step 2: Unify query text construction**

Currently: task-type-specific `_query_text()` branches.
After: One text embedding, same for all inputs. The Galaxy search
finds relevant stars regardless of input type.

**Step 3: Unify scoring and candidate selection**

Currently: different scoring weights per task type (e.g., alpha=0.58
for MMLU, 0.46 for others).
After: One scoring function. If a task type needs different weighting,
that's a meta-rule star in the Galaxy (Layer 4), not a Python constant.

**Step 4: Unify halting gate**

Currently: `if task_type == "LHE_TASK":` inside halting gate.
After: Halting Gate checks ternary convergence regardless of input type.

### What To Keep As I/O

The ONLY place task type should appear:

```python
# benchmarks/gsm8k.py — I/O adapter
def run_question(question_text):
    result = kv.execute_task(task={"query": question_text})
    return extract_numeric_answer(result)  # I/O: format output

# benchmarks/arc_agi_3.py — I/O adapter
def choose_action(frame):
    result = kv.execute_task(task={"query": frame_to_input(frame)})
    return translate_to_arc_action(result)  # I/O: format output
```

The I/O adapter converts external format → universal query. And
converts universal result → external format. That's ALL.

---

## Part B: Cognitive OS Lifecycle

### Boot Sequence

```
1. Check for checkpoint:
   - If checkpoint exists → load from checkpoint (FAST boot, seconds)
   - If no checkpoint → ingest House JSONL (SLOW first boot, minutes)
     → mark JSONL as ingested

2. Verify Galaxy integrity:
   - All default galaxies present in VRAM
   - Symlink references resolve
   - Specialist weights loaded

3. Start TRM game loop:
   - Idle state: waiting for input
   - The system IS alive now
```

### Run Sequence (always-on)

```
4. Input arrives (any source: API, chat, benchmark, frame):
   - I/O adapter normalizes → universal query
   - kv.execute_task() → TRM game loop processes
   - Result → I/O adapter formats for external consumer
   - Brief recorded (answer trace + correct/incorrect signal)

5. Idle detection:
   - No input for N seconds (configurable, default: 30s)
   - OR brief count reaches threshold (e.g., every 10 briefs)
   - → Automatic sleep consolidation (inline, same instance)
```

### Shutdown Sequence

```
6. Shutdown requested (external signal or explicit command):
   - Consolidate ALL pending briefs (sleep cycle)
   - Save checkpoint (Galaxy state + TRM weights + Jarvis state)
   - Mark any new knowledge as ingested (won't re-parse on next boot)
   - Exit cleanly
```

### Bootstrap Removal

On first boot: House JSONL files are parsed and ingested. After
successful consolidation + checkpoint save, create a marker:

```python
# After successful first-boot ingest + checkpoint save:
(checkpoint_dir / "bootstrap_complete.marker").write_text(
    json.dumps({"ingested_at": time.time(), "star_count": total_stars})
)

# On subsequent boots:
if (checkpoint_dir / "bootstrap_complete.marker").exists():
    # Skip JSONL parsing, load from checkpoint directly
    self._load_from_checkpoint()
else:
    # First boot: ingest from House JSONL
    self._ingest_from_house()
```

---

## Part C: GPU-Native Sleep Consolidation

### Current State (Python, CPU-bound)

`jarvis_sleep_consolidation()` counts agreements and contradictions
in Python dicts. No PTX kernels are invoked. GPU sits idle during
the most important phase of the learning cycle.

### Target State (GPU-native)

The consolidation method should LAUNCH existing PTX kernels:

```python
def jarvis_sleep_consolidation(self, *, persist=True):
    briefs = list(self._jarvis_recent_briefs)
    if not briefs:
        return {"updated": False, "briefs_consolidated": 0}

    # 1. Shadow copy comparison ON GPU
    #    Compare shadow_copy predictions vs ground truth
    #    Produces: positive_paths (correct), negative_paths (incorrect)
    shadow = self.shadow_copy
    positive_paths, negative_paths = shadow.evaluate_briefs_gpu(briefs)

    # 2. Sleep cluster refinement ON GPU
    #    Uses: sleep_cluster_refiner.ptx
    #    Strengthens Galaxy clusters around correct answer paths
    #    Weakens clusters around incorrect paths
    from .ptx_runtime.sleep_cluster_kernels import refine_clusters
    refine_clusters(
        self.galaxy_table,
        positive_paths,
        negative_paths,
        learning_rate=0.01,
    )

    # 3. Weight update ON GPU
    #    Uses: lora_gpu.cu or similar
    #    Update TRM specialist weights based on contrastive signal
    self.trm.update_specialist_weights_gpu(
        positive_paths, negative_paths
    )

    # 4. Galaxy entry score update ON GPU
    #    Entries on correct paths get score boost
    #    Entries on incorrect paths get score penalty
    from .ptx_runtime.galaxy_memory_updater import update_entry_scores
    update_entry_scores(
        self.galaxy_table,
        positive_paths,
        negative_paths,
    )

    # Clear briefs, save state
    self._jarvis_recent_briefs = []
    self._save_jarvis_state()
    return {"updated": True, "briefs_consolidated": len(briefs), ...}
```

### Which Kernels to Wire

| Kernel | Bridge | Purpose in Sleep |
|--------|--------|-----------------|
| `sleep_cluster_refiner.cu` | `sleep_cluster_kernels.py` | Cluster co-activation refinement |
| `sleep_glyph_consolidator.cu` | `sleep/glyph_consolidator.py` | Glyph pattern consolidation |
| `galaxy_memory_updater.cu` | `ptx_runtime/galaxy_memory_updater.py` | Galaxy entry score updates |
| `lora_gpu.cu` | `sovereign/lora_gpu_trainer.py` | Specialist weight updates |
| `sleep_time_micro.ptx` | `ptx_runtime/sleep_time_compute.py` | Micro-consolidation passes |

These kernels EXIST. They have bridge classes. They just aren't called
during `jarvis_sleep_consolidation()`.

---

## Part D: Automatic Idle Detection

The TRM game loop should detect idle and trigger sleep:

```python
# In the main event loop (trm_game_loop.py or daemon/main.py):
IDLE_THRESHOLD_SECONDS = 30
BRIEF_BATCH_THRESHOLD = 10

last_query_time = time.time()

while running:
    if has_pending_input():
        process_input()
        last_query_time = time.time()
    else:
        idle_duration = time.time() - last_query_time
        pending_briefs = len(kv._jarvis_recent_briefs)

        if idle_duration > IDLE_THRESHOLD_SECONDS and pending_briefs > 0:
            kv.jarvis_sleep_consolidation(persist=True)
            last_query_time = time.time()  # Reset timer

        elif pending_briefs >= BRIEF_BATCH_THRESHOLD:
            kv.jarvis_sleep_consolidation(persist=True)
            last_query_time = time.time()
```

This is the "NPC rests when no stimuli" pattern from the game loop
analogy. The avatar enters sleep state, consolidates, wakes refreshed.

---

## Execution Order

| Part | Priority | Dependencies |
|------|----------|-------------|
| A: Remove task-type routing | HIGH | None — can start immediately |
| B: Cognitive OS lifecycle | HIGH | Depends on A (unified path) |
| C: GPU-native sleep | MEDIUM | Independent |
| D: Automatic idle detection | MEDIUM | Depends on C (sleep must work on GPU first) |

Start with A. It unblocks everything else.

---

## Success Criteria

- [ ] Zero `if task_type ==` in `query()` and `execute_task()` hot path
- [ ] `task_type` occurrences in kv.py reduced from 302 to <20 (only in I/O)
- [ ] GSM8K, IMO, Math, MMLU, ARC all route through SAME Galaxy search path
- [ ] Boot from checkpoint when marker exists (skip JSONL re-parsing)
- [ ] Shutdown saves checkpoint + marks bootstrap complete
- [ ] Sleep consolidation calls GPU kernels (visible GPU utilization)
- [ ] `sleep_cluster_refiner.ptx` invoked during consolidation
- [ ] `galaxy_memory_updater.cu` invoked during consolidation
- [ ] Automatic sleep trigger after idle threshold
- [ ] `briefs_consolidated > 0` without manual trigger
- [ ] Full boot test suite passes (no isolated tests)

---

## The Vision

Daniel's bet: "If we clear all this nonsense following the
proceduralization with proper symlink and dedup logic, we'll end up
with a system that can solve any task we aim at its live interface."

That means: ONE universal input path. No task-type branches. No
benchmark-named knowledge. No manual sleep triggers. No CPU-bound
consolidation when GPU kernels exist. The system boots, lives, learns,
and shuts down like a cognitive OS — not a benchmark runner that
Python orchestrates.

The architecture defines this clearly. The specs in `docs/vocabulary/`
define this clearly. The kernels exist. The Galaxy exists. The TRM
exists. The missing piece is removing the Python that stands between
them.
