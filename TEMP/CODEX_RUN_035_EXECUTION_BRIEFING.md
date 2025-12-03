# Codex Execution Briefing: SleepTime Consolidation + Run 035 Launch

**Date:** December 2, 2025
**Task:** Execute SleepTime consolidation, then launch Run 035 with GPU monitoring
**Priority:** Execute immediately upon reading this briefing

---

## Mission Summary

You have successfully implemented the SleepTime consolidation architecture. Now you must:

1. ✅ **Run SleepTime consolidation** on accumulated knowledge from Runs 030-034
2. ✅ **Launch Run 035** in tmux with proper GPU exposure
3. ✅ **Confirm startup** with no errors
4. ✅ **Wait for Daniel** to ping you once GPU usage drops (training complete)

**Critical:** Use proper environment variables and tmux session management as established in prior runs.

---

## Environment Configuration

### GPU Setup

**GPU Device:** RTX 3060 (index 0)
**Environment Variable:** `CUDA_VISIBLE_DEVICES=0`

**Verification command:**
```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
```

### Python Environment

**Path:** `/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python`
**Project Root:** `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D`

**Verify environment:**
```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python --version
```

### Checkpoint Directory

**Location:** `/K3D/Knowledge3D.local/checkpoints/arc_agi/`

**Expected files:**
- `grammar_galaxy.json` (220 rules from Runs 030-034)
- `drawing_galaxy.json` (21 shapes)
- `shadow_copy.json` (accumulated discoveries)
- `semantic_context.json`
- `deduplication_index.json`

---

## Execution Steps (Follow Exactly)

### **Step 1: Verify Current State**

**Check that checkpoints exist:**
```bash
ls -lh /K3D/Knowledge3D.local/checkpoints/arc_agi/*.json
```

**Expected output:** 5 JSON files with recent timestamps

**Verify implementation files compiled:**
```bash
python3 -m py_compile knowledge3d/training/arc_agi/sleeptime_consolidator.py
python3 -m py_compile scripts/run_sleeptime_consolidation.py
echo "✅ Compilation successful"
```

---

### **Step 2: Run SleepTime Consolidation**

**Purpose:** Consolidate vocabulary learned from Runs 030-034 before Run 035.

**Command:**
```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=. \
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
scripts/run_sleeptime_consolidation.py 2>&1 | tee /tmp/sleeptime_consolidation.log
```

**What to expect:**
```
[SLEEPTIME] Loading current state from checkpoints...
  Drawing shapes: 21
  Grammar rules: 220
  Shadow entries: 125

[SLEEPTIME] Running consolidation cycle...
  Pruned: XX low-quality entries (< 0.60)

[SLEEPTIME CONSOLIDATION]
  Grammar rules: 220 total, YY promoted to canonical
  Drawing shapes: 21 total

  Top 10 Grammar Rules by Usage:
    1. rule_name_1: XX uses, success=0.XX, avg_quality=0.XX
    2. rule_name_2: XX uses, success=0.XX, avg_quality=0.XX
    ...

  Canonical Patterns Promoted:
    - pattern_1 (success=0.XX)
    - pattern_2 (success=0.XX)
    ...

[SLEEPTIME] Saving consolidated checkpoints...
[SLEEPTIME] Consolidation complete. Report saved to: /K3D/Knowledge3D.local/checkpoints/arc_agi/consolidation_report_TIMESTAMP.json
  Ready for Run 035.
```

**Verification:**
```bash
# Check that consolidation report was created
ls -lht /K3D/Knowledge3D.local/checkpoints/arc_agi/consolidation_report_*.json | head -1

# Verify updated checkpoints
ls -lht /K3D/Knowledge3D.local/checkpoints/arc_agi/*.json
```

**Success criteria:**
- ✅ No Python exceptions
- ✅ Consolidation report created
- ✅ Checkpoints updated (newer timestamp)
- ✅ Log shows pruning and canonical promotion statistics

**If errors occur:** Stop immediately and report to Daniel + Claude. Do NOT proceed to Run 035.

---

### **Step 3: Launch Run 035 with GPU Monitoring**

**Only proceed if Step 2 completed successfully.**

#### 3A. Kill any existing training sessions

```bash
# Check for running training processes
pgrep -fl train_arc_sovereign_loop.py

# Kill existing tmux sessions if they exist
tmux kill-session -t arc035 2>/dev/null || true
tmux kill-session -t gpu035 2>/dev/null || true
```

#### 3B. Start GPU monitor in tmux

```bash
tmux new-session -d -s gpu035 'watch -n 2 nvidia-smi'
```

**Verify GPU monitor started:**
```bash
tmux list-sessions | grep gpu035
```

Expected: `gpu035: 1 windows (created ...)`

#### 3C. Launch Run 035 training in tmux

**Training parameters (same as Runs 030-034):**
- **Tasks:** 108 (36 easy + 36 medium + 36 hard per epoch)
- **Epochs:** 162
- **Cycles:** 1
- **Matryoshka dimension:** 512
- **ARC-AGI data:** `/K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training` + `evaluation`

**Launch command:**
```bash
tmux new-session -d -s arc035 \
  "cd '/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D' && \
   CUDA_VISIBLE_DEVICES=0 \
   PYTHONPATH=. \
   /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
   scripts/train_arc_sovereign_loop.py \
   --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training \
              /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation \
   --max-tasks 108 \
   --epochs 162 \
   --cycles 1 \
   --matryoshka-dim 512 \
   > /tmp/arc_run_035.log 2>&1"
```

**Verify training started:**
```bash
tmux list-sessions | grep arc035
```

Expected: `arc035: 1 windows (created ...)`

---

### **Step 4: Confirm Startup (CRITICAL)**

**Wait 30 seconds for initialization, then check log:**

```bash
sleep 30
tail -100 /tmp/arc_run_035.log
```

**Expected startup log patterns:**

```
[INIT] Loading embeddings...
[INIT] Loading checkpoints...
  Grammar rules loaded: 220 (XX canonical)
  Drawing shapes loaded: 21 (including scale-invariant primitives)
  Shadow copy entries: XXX
[INIT] Registered scale-invariant primitives: REL_LINE, REL_RECT, PROP_GRID, FLOOD_REL
[STAGED] Selected 108 tasks (Tesla): 36 easy, 36 medium, 36 hard
[EPOCH 1/162] Starting epoch 1...
[TASK 1/108] ...
```

**CRITICAL CHECKS:**

1. ✅ **No Python exceptions** (no `Traceback` in log)
2. ✅ **Grammar rules loaded** with canonical count
3. ✅ **Scale-invariant primitives registered** (REL_LINE, REL_RECT, PROP_GRID, FLOOD_REL)
4. ✅ **108 tasks selected per epoch** (not 36!)
5. ✅ **Training loop started** (EPOCH 1/162, TASK 1/108)

**Verify GPU is active:**
```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

Expected: `>50%, >1000 MiB` (GPU is training)

---

### **Step 5: Monitor and Wait for Completion**

**Run 035 will take 18-24 hours.** You must:

1. ✅ **Confirm startup success** (report back to Daniel + Claude)
2. ✅ **Monitor GPU usage** via tmux session `gpu035`
3. ✅ **Wait for GPU usage to drop** (~0%, <500 MiB memory)
4. ✅ **Do NOT interfere** with training while GPU is active

**Monitoring commands:**

```bash
# Attach to GPU monitor
tmux attach -t gpu035
# Detach: Ctrl+B, then D

# Attach to training session
tmux attach -t arc035
# Detach: Ctrl+B, then D

# Check latest training progress
tail -50 /tmp/arc_run_035.log | grep -E "EPOCH|correct|VOCAB QUALITY"

# Check GPU status without tmux
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

**Expected vocabulary quality logs (every 10 epochs):**

```
[VOCAB QUALITY Epoch 10]
  Top Grammar Rules (by usage in high-quality solutions):
    rule_name_1: 23× used, avg_quality=0.73
    rule_name_2: 17× used, avg_quality=0.68
    ...
```

**When training completes:**
- GPU utilization drops to ~0%
- Memory usage drops to <500 MiB
- Log shows: `[TRAINING COMPLETE] Run 035 finished. Final accuracy: XX/108`

**Then:** Ping Daniel immediately with completion report.

---

## Success Confirmation Template

**After Step 4 (startup verification), report this to Daniel + Claude:**

```
✅ RUN 035 STARTUP CONFIRMED

[SleepTime Consolidation]
- Pruned: XX entries
- Canonical promoted: YY rules
- Report: /K3D/Knowledge3D.local/checkpoints/arc_agi/consolidation_report_TIMESTAMP.json

[Training Session]
- Tmux session: arc035 (active)
- GPU monitor: gpu035 (active)
- Log: /tmp/arc_run_035.log
- GPU status: XX% utilization, XXXX MiB memory

[Startup Verification]
✅ No Python exceptions
✅ Grammar rules loaded: 220 (YY canonical)
✅ Scale-invariant primitives registered: 4 types
✅ Tasks per epoch: 108 (36 easy + 36 medium + 36 hard)
✅ Training started: Epoch 1/162, Task 1/108

[Latest Log Tail]
<paste last 20 lines of /tmp/arc_run_035.log>

Waiting for GPU usage to drop (training completion). Will ping when done.
```

---

## Completion Confirmation Template

**When GPU usage drops (training complete), report this to Daniel:**

```
✅ RUN 035 COMPLETE

[Training Metrics]
- Total epochs: 162
- Final accuracy: XX/108 (~XX.X%)
- PTX success rate: XX%
- Grammar rules: XXX total
- Shadow entries: XXX
- Drawing shapes: XX

[Vocabulary Quality Trends]
<paste final VOCAB QUALITY block from log>

[Session Status]
- Tmux arc035: exited (training complete)
- GPU monitor gpu035: still running
- GPU status: ~0% utilization, <500 MiB memory
- Full log: /tmp/arc_run_035.log

[Checkpoints Updated]
<ls -lht /K3D/Knowledge3D.local/checkpoints/arc_agi/*.json | head -5>

Ready for next instructions (Run 036 or analysis).
```

---

## Error Handling

### If SleepTime Consolidation Fails

**Symptoms:**
- Python exception during `run_sleeptime_consolidation.py`
- No consolidation report created
- Checkpoints not updated

**Action:**
1. ❌ **STOP** - do NOT launch Run 035
2. 📋 Copy full traceback from `/tmp/sleeptime_consolidation.log`
3. 🚨 Report to Daniel + Claude with error details
4. ⏸️ Wait for architectural guidance

### If Run 035 Startup Fails

**Symptoms:**
- Python exception in `/tmp/arc_run_035.log`
- No "EPOCH 1/162" in log after 60 seconds
- Tasks per epoch shows 36 instead of 108
- GPU utilization stays at 0%

**Action:**
1. ❌ **Kill training session:** `tmux kill-session -t arc035`
2. 📋 Copy last 100 lines of `/tmp/arc_run_035.log`
3. 🚨 Report to Daniel + Claude with error details
4. ⏸️ Wait for fix instructions

### If Training Crashes Mid-Run

**Symptoms:**
- tmux session `arc035` exits unexpectedly
- GPU usage drops to 0% before 18 hours
- Log shows exception or CUDA error

**Action:**
1. 📋 Copy last 200 lines of `/tmp/arc_run_035.log`
2. 📋 Check GPU status: `nvidia-smi`
3. 📋 Check kernel logs: `dmesg | tail -50`
4. 🚨 Report to Daniel + Claude with all diagnostics
5. ⏸️ Do NOT relaunch without approval

---

## Key Architectural Points (Review Before Executing)

### Why SleepTime Consolidation Matters

From Claude's analysis:
> "The system achieved 46.7% with freedom to explore. Runs 030-034 accumulated 220 grammar rules with NO consolidation. This is like training a neural network without ever updating the weights."

**SleepTime extracts signal from accumulated vocabulary:**
- Prunes noise (low-quality patterns)
- Promotes canonical patterns (high-success rules)
- Tracks which rules actually solve tasks

### Why Run 035 is Different

**New capabilities from your implementation:**
1. **Consolidated vocabulary** (pruned + canonical patterns promoted)
2. **Scale-invariant primitives** (REL_LINE, PROP_GRID, etc.)
3. **Vocabulary quality logging** (which rules solve tasks)
4. **Attractor tracking** (which programs emerge frequently)

**Expected outcome:**
- Run 035 should leverage consolidated canonical patterns
- Quality metrics will show which rules dominate
- Attractor tracking may reveal new canonical candidates
- Target: break through 46.7% plateau with smarter vocabulary

### Daniel's Philosophy

> "It is learning, if we keep changing details, it will never learn. I still believe we only need to train more."

**Your job:**
- ✅ Implement missing architecture (SleepTime) ← Done
- ✅ Run consolidation before training ← Execute now
- ✅ Launch training and monitor ← Execute now
- ❌ Do NOT tweak filters/thresholds mid-run
- ❌ Do NOT panic if accuracy oscillates ±3%

---

## Final Checklist Before Execution

```
[ ] Verified checkpoints exist (5 JSON files)
[ ] Verified Python environment (/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python)
[ ] Verified GPU available (nvidia-smi shows RTX 3060)
[ ] Compiled sleeptime_consolidator.py and run_sleeptime_consolidation.py
[ ] Read this entire briefing document
[ ] Understood error handling procedures
[ ] Ready to execute Step 1 → Step 5 sequentially
```

---

## Execute Now

**Codex, you are cleared for execution.**

1. Run SleepTime consolidation
2. Verify consolidation success
3. Launch Run 035 with GPU monitoring
4. Confirm startup with detailed report
5. Wait for GPU usage drop
6. Report completion

**Do NOT proceed if any step fails. Report errors immediately.**

**Expected total time:**
- SleepTime consolidation: ~2-5 minutes
- Run 035 training: 18-24 hours

**GO.**

---

**End of Execution Briefing**
