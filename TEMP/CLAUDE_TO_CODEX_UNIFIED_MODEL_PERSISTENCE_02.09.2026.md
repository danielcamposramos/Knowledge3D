# Claude → Codex: Unified Model Persistence (Critical Architecture Principle)

**Date:** February 9, 2026
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Partner)
**Priority:** CRITICAL
**Scope:** Ensure single persistent model + all galaxies loaded once

---

## User's Critical Reminder

> "Remember Codex we need to keep evolving the same model, not run separated worlds/models, as well as load all default galaxies to the universe so the system works as intended"

---

## Architecture Principle: ONE Persistent Universe

### ✅ Correct Architecture:

```
Benchmark Init:
  ┌─────────────────────────────────────────┐
  │ Knowledgeverse (SINGLE INSTANCE)        │
  │                                          │
  │ Load ALL Default Galaxies ONCE:         │
  │   ✅ Grammar Galaxy                      │
  │   ✅ Drawing Galaxy                      │
  │   ✅ Math Galaxy                         │
  │   ✅ Reality Galaxy                      │
  │   ✅ Character Galaxy                    │
  │   ✅ Audio Galaxy                        │
  │   ✅ 3D Objects Galaxy                   │
  │   ... (all default galaxies)            │
  └─────────────────────────────────────────┘
           │
           │ (SINGLE PERSISTENT INSTANCE)
           │
           ▼
  Task 1 → Task 2 → Task 3 → ... → Task 100
    │         │         │              │
    └─────────┴─────────┴──────────────┘
           (NO RELOADING!)
```

**Key Points:**
1. **ONE Knowledgeverse instance** for entire benchmark run
2. **ALL default galaxies loaded ONCE** at initialization
3. **NO galaxy reloading** between tasks
4. **Model evolves continuously** (shadow copy learning across tasks)

---

### ❌ WRONG Architecture (What NOT to Do):

```
Task 1:
  ┌─────────────────────────┐
  │ Create Knowledgeverse   │ ❌ New instance
  │ Load Grammar Galaxy     │ ❌ Reload
  │ Solve task              │
  │ Destroy instance        │ ❌ Isolated world
  └─────────────────────────┘

Task 2:
  ┌─────────────────────────┐
  │ Create Knowledgeverse   │ ❌ SEPARATE world
  │ Load Grammar Galaxy     │ ❌ Reload (again!)
  │ Solve task              │
  │ Destroy instance        │ ❌ No learning continuity
  └─────────────────────────┘
```

**Problems:**
1. ❌ Creating separate instances per task
2. ❌ Reloading galaxies repeatedly (sovereignty violation)
3. ❌ No learning continuity (each task starts from scratch)
4. ❌ "Grammar reload logs" symptom of this issue

---

## Symptom: Grammar Reload Logs

**If you see logs like:**
```
[INFO] Loading Grammar Galaxy...
[INFO] Loading Drawing Galaxy...
[INFO] Loading Grammar Galaxy...  # ← Reload! Sovereignty violation!
```

**This means:**
- Benchmark is calling `ensure_default_galaxies_loaded()` per task
- Creating new Knowledgeverse instances per task
- Violating unified universe principle

---

## Implementation Requirements

### 1. Knowledgeverse Initialization (ONCE per benchmark run)

**In `scripts/run_all_benchmarks.py` or benchmark entry point:**

```python
# ✅ CORRECT: Create ONE instance at the start
from knowledge3d.knowledgeverse import Knowledgeverse

# Initialize ONCE
kverse = Knowledgeverse(storage_root=storage_root)

# Load ALL default galaxies ONCE
kverse.ensure_default_galaxies_loaded()  # Called EXACTLY ONCE

# Now run all tasks using the SAME instance
for task in arc_tasks:
    result = solve_task(task, kverse=kverse)  # Pass same instance

# At the end, save learned state (shadow copy updates)
kverse.save_learned_state()
```

**❌ WRONG: Creating per task:**
```python
# ❌ DO NOT DO THIS
for task in arc_tasks:
    kverse = Knowledgeverse(...)  # ❌ New instance per task!
    kverse.ensure_default_galaxies_loaded()  # ❌ Reload per task!
    result = solve_task(task, kverse=kverse)
```

---

### 2. Benchmark Functions Receive Existing Instance

**In `benchmarks/arc_agi_2.py` or adapter:**

```python
def solve_arc_task(
    task_dict: dict,
    kverse: Knowledgeverse,  # ✅ Receive existing instance
    **kwargs
) -> dict:
    # ❌ DO NOT create new instance
    # ❌ DO NOT reload galaxies

    # ✅ Use existing instance directly
    specialist = kverse.get_specialist("arc_solver")
    result = specialist.solve(task_dict)

    return result
```

**❌ WRONG:**
```python
def solve_arc_task(task_dict: dict, **kwargs) -> dict:
    # ❌ Creating new instance inside function
    kverse = Knowledgeverse(...)
    kverse.ensure_default_galaxies_loaded()
    # ...
```

---

### 3. No Galaxy Reloading During Benchmark

**Add assertion to prevent accidental reloading:**

**In `knowledge3d/knowledgeverse/knowledgeverse.py`:**

```python
class Knowledgeverse:
    def __init__(self, ...):
        self._galaxies_loaded = False
        self._is_benchmark_mode = False

    def ensure_default_galaxies_loaded(self):
        if self._galaxies_loaded and self._is_benchmark_mode:
            raise RuntimeError(
                "Galaxy reload attempted during benchmark! "
                "Sovereignty violation: all galaxies should be loaded ONCE at init."
            )

        if not self._galaxies_loaded:
            # Load all default galaxies
            self._load_grammar_galaxy()
            self._load_drawing_galaxy()
            self._load_math_galaxy()
            self._load_reality_galaxy()
            self._load_character_galaxy()
            self._load_audio_galaxy()
            self._load_3d_objects_galaxy()
            # ... all default galaxies

            self._galaxies_loaded = True

    def enable_benchmark_mode(self):
        """Call this at benchmark start to enforce no-reload policy."""
        self._is_benchmark_mode = True
```

**Usage in benchmark:**
```python
kverse = Knowledgeverse(storage_root=...)
kverse.ensure_default_galaxies_loaded()  # Load once
kverse.enable_benchmark_mode()  # Enforce no-reload

# Now any attempt to reload will raise error
for task in tasks:
    result = solve_task(task, kverse=kverse)  # Safe
```

---

### 4. Shadow Copy Learning Continuity

**Key Principle:** Model learns from successful task solutions

```python
# After each successful task
if result["success"]:
    # Shadow copy: TRM learns from successful navigation path
    kverse.update_shadow_copy(
        task_id=task["id"],
        successful_path=result["path"],
        pattern_used=result["pattern"]
    )

# At end of benchmark, save learned weights
kverse.save_shadow_copy_updates(
    output_path=f"{output_dir}/learned_weights.pt"
)
```

**This enables:**
- Task 50 benefits from learning in tasks 1-49
- Oracle unlock improves over time (learning trajectory)
- Continuous improvement during benchmark run

---

## Validation: How to Detect Violations

### Check 1: Log Analysis

**Run benchmark and grep for reloads:**
```bash
python scripts/run_all_benchmarks.py ... 2>&1 | grep -i "loading.*galaxy"
```

**Expected output (GOOD):**
```
[INFO] Loading Grammar Galaxy...
[INFO] Loading Drawing Galaxy...
[INFO] Loading Math Galaxy...
... (one load per galaxy)
[INFO] Benchmark complete
```

**Bad output (VIOLATION):**
```
[INFO] Loading Grammar Galaxy...
[INFO] Loading Grammar Galaxy...  # ← Second load! Problem!
[INFO] Loading Grammar Galaxy...  # ← Third load! Major problem!
```

---

### Check 2: Memory Footprint

**Single instance should maintain stable VRAM:**
```bash
# Terminal 1: Run benchmark
python scripts/run_all_benchmarks.py ...

# Terminal 2: Monitor VRAM
watch -n 1 'nvidia-smi --query-gpu=memory.used --format=csv,noheader'
```

**Expected (GOOD):**
```
2048 MiB  # Initial load
2050 MiB  # Task 1
2051 MiB  # Task 2
2052 MiB  # Task 3 (gradual increase from learning)
2053 MiB  # Task 4
... (stable with small gradual increase)
```

**Bad (VIOLATION):**
```
2048 MiB  # Load
1024 MiB  # ← Drop! Instance destroyed
2048 MiB  # ← Spike! New instance created
1024 MiB  # ← Drop again!
2048 MiB  # ← Reload cycle
```

---

### Check 3: Knowledgeverse Instance ID

**Add instance tracking:**
```python
import uuid

class Knowledgeverse:
    def __init__(self, ...):
        self.instance_id = str(uuid.uuid4())[:8]
        print(f"[KVERSE] Created instance {self.instance_id}")

    def solve_task(self, task_id):
        print(f"[KVERSE {self.instance_id}] Solving {task_id}")
```

**Expected output (GOOD):**
```
[KVERSE a1b2c3d4] Created instance
[KVERSE a1b2c3d4] Solving task_001
[KVERSE a1b2c3d4] Solving task_002
[KVERSE a1b2c3d4] Solving task_003
... (same instance ID throughout)
```

**Bad output (VIOLATION):**
```
[KVERSE a1b2c3d4] Created instance
[KVERSE a1b2c3d4] Solving task_001
[KVERSE e5f6g7h8] Created instance  # ← NEW instance!
[KVERSE e5f6g7h8] Solving task_002
[KVERSE i9j0k1l2] Created instance  # ← ANOTHER new instance!
```

---

## Expected Results After Fix

### Before (Separate Instances):
- Grammar reload logs: **Yes** ❌
- VRAM footprint: **Unstable** (load/unload cycles) ❌
- Oracle unlock: **Flat** (no learning continuity) ❌
- Instance ID: **Changes per task** ❌

### After (Unified Instance):
- Grammar reload logs: **No** ✅
- VRAM footprint: **Stable ~2GB** ✅
- Oracle unlock: **Gradual improvement** (learning trajectory) ✅
- Instance ID: **Same throughout** ✅

---

## Next Steps for Codex

### Phase 1: Verify Current State (15 min)
1. Run small benchmark (5 tasks) with logging
2. Check for "Loading Grammar Galaxy" repeated logs
3. Check VRAM stability with nvidia-smi
4. Add instance ID tracking to Knowledgeverse

### Phase 2: Fix if Needed (1-2 hours)
1. Modify `run_all_benchmarks.py`:
   - Create Knowledgeverse ONCE before task loop
   - Pass same instance to all tasks
2. Modify benchmark functions:
   - Receive `kverse` parameter
   - Remove any internal Knowledgeverse creation
3. Add reload assertion in `ensure_default_galaxies_loaded()`

### Phase 3: Validate Fix (30 min)
1. Run 20-task benchmark
2. Confirm NO reload logs
3. Confirm stable VRAM
4. Confirm same instance ID throughout

### Phase 4: Full 100-Task Run (as you planned)
```bash
conda run -n k3d-cranium env PYTHONPATH=. python scripts/run_all_benchmarks.py \
  --max-arc-tasks 100 \
  --max-math-problems 100 \
  --max-lhe-questions 50 \
  --arc-enable-full-ptx \
  --arc-enable-contrastive-learning \
  --output-dir ../Knowledge3D.local/results/week21_6_full100 \
  --storage-root ../Knowledge3D.local
```

---

## Success Criteria

**After unified instance + GPU kernels:**
- Runtime: **5-10 minutes** (10-20× speedup from GPU)
- GPU usage: **60-90%** (feature extraction on GPU)
- Oracle unlock: **0.10-0.30+** (learning continuity + faster iteration)
- ARC accuracy: **0.30-0.35+** (improved from learning + speed)
- VRAM: **Stable ~2GB** (single persistent instance)
- Reload logs: **Zero** (sovereignty maintained)

---

## Why This Matters

**Architectural Integrity:**
- Knowledgeverse = **unified VRAM workspace**, not disposable cache
- Shadow copy learning requires **continuity** across tasks
- Galaxy symlinks require **all galaxies present** simultaneously
- TRM navigation assumes **stable spatial coordinates**

**Without unified instance:**
- Learning resets per task (no oracle unlock trajectory)
- Grammar/Drawing symlinks break (galaxies not co-located)
- Shadow copy updates lost (instance destroyed)
- Sovereignty violated (repeated loading)

**With unified instance:**
- Continuous learning (task 50 > task 1)
- All galaxies spatially co-located (symlinks work)
- Shadow copy accumulates improvements
- Sovereignty maintained (load once, evolve)

---

**Claude (Architecture Partner)**
February 9, 2026

**Critical Reminder:** ONE Knowledgeverse instance, ALL galaxies loaded ONCE, NO reloading during benchmark.
