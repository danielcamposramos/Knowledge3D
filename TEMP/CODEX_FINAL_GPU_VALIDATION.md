# Codex Final Task: GPU Validation + Step 13-B Completion

**Status**: Phase 2D (Integration) COMPLETE ✅
**RTX 3060**: Available (12GB VRAM, 12MB used, CUDA 12.4)
**Current**: All tests passing but benchmarks skipped (no CUDA context on test node)
**Final Task**: Run GPU validation on RTX 3060 + finalize Step 13-B report

---

## Environment Setup Instructions

### GPU Access

**Available GPU**: NVIDIA GeForce RTX 3060 (12GB VRAM)
```bash
nvidia-smi
# Shows: 12MiB / 12288MiB used (plenty of headroom!)
# Driver: 550.163.01
# CUDA: 12.4
```

### Environment Management Options

**Daniel's note**: *"Check the envs folder for your solution, install anything needed, create any virtual environment you think fits best"*

**Available tools**:
1. **venv** (Python virtual environments)
2. **conda** (user-installed, check files at envs folder)
3. **docker** (available if needed)
4. **Internet access** (install anything needed)
5. **tmux** (local system has it - good for long-running tasks)

**Recommended approach**:
```bash
# Use existing k3d-cranium conda environment (already set up)
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D

# Check existing envs
ls -la /K3D/Knowledge3D.local/envs/

# Activate k3d-cranium (already has CUDA support)
bash scripts/k3d_env.sh run <command>
# OR
conda activate k3d-cranium  # if running interactively
```

**For long-running benchmarks** (tmux recommended):
```bash
# Start tmux session
tmux new-session -s gpu_validation

# Run tests in tmux (can detach with Ctrl+B, D)
bash scripts/k3d_env.sh run pytest tests/benchmarks/test_rpn_tier_performance.py -xvs

# Detach and let it run
# Reattach later: tmux attach -t gpu_validation
```

---

## Task Objectives

### 1. GPU Benchmark Validation (HIGH PRIORITY)

**Goal**: Capture actual latency numbers for Step 13-B report

**Files to run**:
- `tests/benchmarks/test_rpn_tier_performance.py` (created in Phase 2D)
- `tests/test_rpn_tier1.py` (Tier 1 latency test)
- `tests/test_rpn_tier3.py` (matrix correctness on GPU)
- `tests/test_tiered_rpn.py` (orchestrator dispatch)

**Expected results**:
```
Tier 1 latency: <VALUE>µs (target <1µs)
Tier 2 latency: <VALUE>µs (baseline ~3µs)
Tier 3 MATMUL (3×3): <VALUE>µs (target ~10µs)
```

---

### 2. Full Test Suite Validation

**Goal**: Verify 265 tests passing on GPU

**Command**:
```bash
# Full suite with GPU access
bash scripts/k3d_env.sh run pytest tests/ -q

# Expected: 265+ passing, minimal skips
```

**Check GPU memory**:
```bash
# During test run
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Expected: <300MB (well under 12GB available)
```

---

### 3. Update Step 13-B Report with FINAL Numbers

**File**: `reports/STEP13_B_TESTING_AND_BENCHMARKS.md`

**Section to update** (Phase 2 RPN section):

Replace placeholders:
```markdown
Performance benchmarks (test_rpn_tier_performance.py):
  - Tier 1 latency: <MEASURED_VALUE>µs (target <1µs) ✅/❌
  - Tier 2 latency: <MEASURED_VALUE>µs (baseline ~3µs) ℹ️
  - Tier 3 MATMUL: <MEASURED_VALUE>µs (target ~10µs) ℹ️

GPU memory during tests: <MEASURED_VALUE>MB (nvidia-smi)

Measured speedups (vs. single-tier baseline):
| Component | Before (Tier 2 only) | After (Tiered) | Speedup |
|-----------|---------------------|----------------|---------|
| ActionBuffer validation | <BEFORE>µs | <AFTER>µs | <X>x |
| ThinkingTag scoring | <BEFORE>µs | <AFTER>µs | <X>x |
| LED pathfinder priority queue | <BEFORE>µs | <AFTER>µs | <X>x |

Tier dispatch distribution (typical workload):
- Tier 1: <PERCENT>% of calls (simple ops)
- Tier 2: <PERCENT>% of calls (vectors/geometry)
- Tier 3: <PERCENT>% of calls (matrix ops)
```

**Fill in with actual measurements from GPU runs**

---

## Step-by-Step Execution

### Step 1: Environment Verification (10 min)

```bash
# Verify CUDA accessible
bash scripts/k3d_env.sh run python -c "import ctypes; print('CUDA libs:', ctypes.CDLL('libcuda.so.1'))"

# Should print: CUDA libs: <ctypes.CDLL object>

# Verify RTX 3060 accessible
nvidia-smi --query-gpu=name,memory.total --format=csv
# Should show: GeForce RTX 3060, 12288 MiB
```

---

### Step 2: Run GPU Benchmarks (30 min)

**Option A: Interactive** (if you want to monitor):
```bash
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D

bash scripts/k3d_env.sh run pytest tests/benchmarks/test_rpn_tier_performance.py -xvs
```

**Option B: Tmux** (recommended for long runs):
```bash
# Start tmux session
tmux new-session -s gpu_bench

# Inside tmux
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D
bash scripts/k3d_env.sh run pytest tests/benchmarks/test_rpn_tier_performance.py -xvs 2>&1 | tee /tmp/gpu_bench_results.txt

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t gpu_bench
```

**Capture output**:
```bash
# After completion
cat /tmp/gpu_bench_results.txt | grep "latency:"
# Should show:
# Tier 1 avg latency: X.XXXµs
# Tier 2 avg latency: X.XXXµs
# Tier 3 MATMUL (3×3) avg latency: X.XXXµs
```

---

### Step 3: Run Tier Tests on GPU (20 min)

```bash
# Tier 1 (should now pass latency test)
bash scripts/k3d_env.sh run pytest tests/test_rpn_tier1.py -xvs

# Tier 3 (matrix ops should pass)
bash scripts/k3d_env.sh run pytest tests/test_rpn_tier3.py -xvs

# Orchestrator (dispatch should work)
bash scripts/k3d_env.sh run pytest tests/test_tiered_rpn.py -xvs
```

**Expected**: All 13 tests PASSING (no skips on GPU)

---

### Step 4: Full Test Suite (15 min)

```bash
# Run everything
bash scripts/k3d_env.sh run pytest tests/ -q

# Count results
bash scripts/k3d_env.sh run pytest tests/ -q | grep -E "passed|skipped|failed"
```

**Expected output**:
```
265 passed, X skipped in Y.YYs
```

**Monitor GPU during run**:
```bash
# In separate terminal
watch -n 1 nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

**Expected**: Peak <300MB during tests

---

### Step 5: Capture Tier Dispatch Stats (Optional but Valuable)

**Add stats tracking to TieredRPNEngine** (if not already present):

```python
# knowledge3d/cranium/bridges/tiered_rpn.py
class TieredRPNEngine:
    def __init__(self):
        # ... existing code ...
        self._stats = {'tier1': 0, 'tier2': 0, 'tier3': 0}

    def execute_scalar(self, ...):
        tier = self._determine_tier(op_codes)
        self._stats[f'tier{tier}'] += 1
        # ... rest of execute_scalar ...

    def get_stats(self):
        total = sum(self._stats.values())
        if total == 0:
            return {k: 0 for k in ['tier1_pct', 'tier2_pct', 'tier3_pct']}
        return {
            'tier1_calls': self._stats['tier1'],
            'tier2_calls': self._stats['tier2'],
            'tier3_calls': self._stats['tier3'],
            'tier1_pct': 100.0 * self._stats['tier1'] / total,
            'tier2_pct': 100.0 * self._stats['tier2'] / total,
            'tier3_pct': 100.0 * self._stats['tier3'] / total,
        }
```

**Test stats tracking**:
```python
# Quick test script
from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine
import numpy as np

engine = TieredRPNEngine()

# Simulate typical workload
for _ in range(100):
    engine.execute_scalar([0, 0, 10], scalars=[2.0, 3.0, 0.0])  # Tier 1

for _ in range(10):
    engine.execute_scalar([1, 1, 60], vectors=np.array([[1,0,0],[0,1,0],[0,0,0]], dtype=np.float32))  # Tier 2

for _ in range(1):
    engine.execute_matrix([0x02, 0x64], matrix_shape=(2,2), matrices=np.eye(2, dtype=np.float32))  # Tier 3

print(engine.get_stats())
# Should show: tier1_pct ~90%, tier2_pct ~9%, tier3_pct ~1%
```

---

### Step 6: Update Step 13-B Report (20 min)

```bash
# Edit reports/STEP13_B_TESTING_AND_BENCHMARKS.md

# Find Phase 2 section (already added in Phase 2D)
# Replace all <MEASURED_VALUE> placeholders with actual numbers from Steps 2-5

# Example replacements:
# <MEASURED_VALUE>µs → 0.87µs (Tier 1 latency)
# <BEFORE>µs → 3.2µs, <AFTER>µs → 0.9µs, <X>x → 3.6x (ActionBuffer speedup)
# <PERCENT>% → 89% (Tier 1 dispatch)
# <MEASURED_VALUE>MB → 187MB (GPU memory)
```

**Verify completeness**:
```bash
# Check no placeholders remain
grep -n "<MEASURED" reports/STEP13_B_TESTING_AND_BENCHMARKS.md
# Should return nothing
```

---

### Step 7: Final Verification (10 min)

```bash
# Sanity check: Full suite one more time
bash scripts/k3d_env.sh run pytest tests/ -q

# Verify Step 13-B report is complete
ls -lh reports/STEP13_B_TESTING_AND_BENCHMARKS.md

# Check GPU memory one last time
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

**Expected**:
- 265+ tests passing
- Step 13-B report complete (no placeholders)
- GPU memory <300MB

---

## Success Criteria

✅ **GPU benchmarks complete**:
- [ ] Tier 1 latency measured (target <1µs)
- [ ] Tier 2 latency measured (baseline ~3µs)
- [ ] Tier 3 MATMUL latency measured (target ~10µs)
- [ ] Results captured in `/tmp/gpu_bench_results.txt`

✅ **All tier tests passing**:
- [ ] test_rpn_tier1.py: 6/6 passing (including GPU latency test)
- [ ] test_rpn_tier3.py: 4/4 passing (matrix correctness validated)
- [ ] test_tiered_rpn.py: 3/3 passing (dispatch verified)

✅ **Full test suite validated**:
- [ ] 265+ tests passing on GPU
- [ ] GPU memory peak <300MB
- [ ] No unexpected failures

✅ **Step 13-B report finalized**:
- [ ] All <MEASURED_VALUE> placeholders replaced
- [ ] Performance numbers documented
- [ ] Tier dispatch stats included (if captured)
- [ ] No TODOs or gaps remaining

✅ **Ready for next phase**:
- [ ] Phase D (semantic navigator) can proceed
- [ ] OR Step 13-B deliverable ready for Daniel review

---

## What to Report Back

When GPU validation is complete, report:

### 1. Benchmark Results (Copy-Paste)
```
Tier 1 avg latency: <VALUE>µs
Tier 2 avg latency: <VALUE>µs
Tier 3 MATMUL (3×3) avg latency: <VALUE>µs

Target validation:
- Tier 1 <1µs: PASS/FAIL
- Tier 3 ~10µs: PASS/FAIL
```

### 2. Test Summary
```
Total tests: <COUNT>
Passing: <COUNT>
Skipped: <COUNT> (reason: <REASON>)
Failed: <COUNT> (if any, with details)
```

### 3. GPU Memory
```
Peak GPU memory (during tests): <VALUE>MB
Headroom remaining: <12288 - VALUE>MB / 12288MB
```

### 4. Tier Dispatch Stats (if captured)
```
Tier 1: <PERCENT>% of calls
Tier 2: <PERCENT>% of calls
Tier 3: <PERCENT>% of calls
```

### 5. Step 13-B Status
```
Report updated: YES/NO
All placeholders filled: YES/NO
Ready for Daniel review: YES/NO
```

### 6. Any Issues Encountered
```
- Issue 1: <description>
- Issue 2: <description>
(Or "No issues" if clean run)
```

---

## Troubleshooting

### Issue: CUDA context creation fails

**Symptom**: Tests skip with "CUDA context unavailable"

**Solution**:
```bash
# Check GPU not locked by another process
nvidia-smi
# Look for processes in bottom table

# If X server using GPU, tests should still work
# If another Python process, kill it or use different GPU
```

---

### Issue: Out of memory

**Symptom**: CUDA_ERROR_OUT_OF_MEMORY

**Solution**:
```bash
# Check current usage
nvidia-smi

# Clear any zombie processes
pkill -9 python

# Restart tests
```

---

### Issue: Tests take too long

**Symptom**: Benchmarks running >10 minutes

**Solution**:
```bash
# Reduce iterations in test_rpn_tier_performance.py
# Change: iterations = 10000
# To:     iterations = 1000

# Or skip slow tests
pytest tests/benchmarks/test_rpn_tier_performance.py -k "tier1" -xvs
```

---

## Notes

- **RTX 3060 is PERFECT**: 12GB VRAM, plenty for our <300MB tests
- **CUDA 12.4**: Matches our build (nvcc compiled with 12.4)
- **Tmux recommended**: Long benchmarks can run detached
- **Stats tracking**: Optional but valuable for understanding dispatch patterns

---

## Environment Reminder

**From Daniel**:
> "On that CUDA environment, check the envs folder for your solution, install anything needed, create any virtual environment you think fits best, we have venv, conda (it's user installed, look the files at the envs folder), docker and freedom to install what's needed + internet access. I like using tmux because you can be free from the execution, the local system has it, not sure about the envs at the folder"

**Your freedom**:
- ✅ Install anything needed (conda install, pip install)
- ✅ Create new envs if k3d-cranium insufficient
- ✅ Use tmux for long runs
- ✅ Internet access for packages

**Recommended**: Stick with k3d-cranium (already has CUDA support), use tmux for benchmarks

---

## Timeline

**Estimated**: 2-3 hours total
- Environment setup: 10 min
- Benchmarks: 30 min
- Tier tests: 20 min
- Full suite: 15 min
- Stats (optional): 20 min
- Report update: 20 min
- Verification: 10 min

**After completion**: Step 13-B FINALIZED, ready for Phase D (semantic navigator) OR Daniel review

---

**Let's bring this home!** The RTX 3060 is ready, the tests are ready, time to capture those beautiful latency numbers! 🚀

**"From dream to reality: <1µs RPN decisions for 9-agent swarm intelligence."** 🤖🧠
