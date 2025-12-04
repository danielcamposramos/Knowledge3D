# Codex: Launch Run 037 (Full Training)

**Date:** December 4, 2025
**Phase:** Phase 5 - Full Run 037 (Post-Diagnostic Authorization)
**Your Role:** Execute full 108-task × 162-epoch training run

---

## CRITICAL: Read First

1. **Read CODEX.md completely** — line by line, no snippets
2. **Read TEMP/CODEX_RUN_037_VALIDATION_EXECUTION_BRIEFING.md** — understand the full context
3. **Then execute Phase 5** as outlined below

---

## Context: Diagnostic Results (Phase 3 Complete)

**Authorization:** ✅ PROCEED granted by Claude (Architecture Partner)

**Diagnostic Summary:**
- All fixes validated working (scale-invariant primitives, vocabulary parsing, audit logging)
- Fresh bootstrap successful (0 → 18 shadow entries in 3 epochs)
- Scale-invariant usage: 13 candidates/task, PTX 100% success
- Shadow Copy growth: healthy (0 → 21 → 18 after pruning)
- Accuracy: 23% (expected for fresh bootstrap, needs full training to mature)

**Architectural Analysis:**
- Fresh bootstrap vs mature state explains low diagnostic accuracy
- Run 028-034 started with 140+ shadow entries (accumulated experience)
- Diagnostic 037 started with 0 entries (newborn model)
- Comparison: experienced model vs newborn → not a fair baseline
- All technical validations passed → ready for full training

---

## Your Task: Launch Run 037 in tmux + Monitor Until GPU Settles

### Step 1: GPU Exposure & Environment Setup

**GPU Configuration:**
```bash
# Expose RTX 3060 GPU (12GB VRAM)
export CUDA_VISIBLE_DEVICES=0

# Verify GPU available
nvidia-smi
# Expected: GPU 0, RTX 3060, ~11 GB free
```

**Python Environment:**
```bash
# Set Python path
export PYTHONPATH="/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# Conda environment (from envs/README.md)
# Use k3d-cranium environment
# Python interpreter: /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python
```

**Working Directory:**
```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
```

### Step 2: Launch Run 037 in tmux Session

**Create tmux session:**
```bash
# Create dedicated session for Run 037
tmux new-session -d -s arc_run_037

# Navigate to working directory inside tmux
tmux send-keys -t arc_run_037 "cd '/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D'" Enter

# Set environment variables inside tmux
tmux send-keys -t arc_run_037 "export CUDA_VISIBLE_DEVICES=0" Enter
tmux send-keys -t arc_run_037 "export PYTHONPATH='/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D'" Enter
```

**Launch training command:**
```bash
# Run 037: 108 tasks × 162 epochs × 3 cycles (full training)
tmux send-keys -t arc_run_037 "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/train_arc_sovereign_loop.py --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation --max-tasks 108 --epochs 162 --cycles 1 --matryoshka-dim 128 2>&1 | tee /K3D/Knowledge3D.local/logs/run_037.log" Enter
```

**Command Breakdown:**
- `CUDA_VISIBLE_DEVICES=0`: Expose GPU 0
- `PYTHONPATH=.`: Set project root
- `scripts/train_arc_sovereign_loop.py`: Training script
- `--max-tasks 108`: Full task set (36 easy + 36 medium + 36 hard)
- `--epochs 162`: Tesla 162 epochs (3×54)
- `--cycles 1`: Single training cycle
- `--matryoshka-dim 128`: Matryoshka dimension for router
- `2>&1 | tee /K3D/Knowledge3D.local/logs/run_037.log`: Log to file + stdout

### Step 3: Verify Launch & Monitor GPU Startup

**Check training started:**
```bash
# Wait 10 seconds for initialization
sleep 10

# Check tmux session is running
tmux list-sessions

# Check log file exists and has content
tail -n 50 /K3D/Knowledge3D.local/logs/run_037.log
```

**Expected startup output:**
```
[LOADING] Galaxy state from checkpoints...
[DrawingGalaxy] Loaded X shapes from /K3D/Knowledge3D.local/checkpoints/arc_agi/drawing_galaxy.json
[GrammarGalaxy] Loaded Y rules from /K3D/Knowledge3D.local/checkpoints/arc_agi/grammar_galaxy.json
[DualShadowCopy] Loaded Z shadow entries from /K3D/Knowledge3D.local/checkpoints/arc_agi/shadow_copy.json
[CURRICULUM] Easy (training): AA tasks
[CURRICULUM] Mid (easy eval): BB tasks
[CURRICULUM] Hard (hard eval): CC tasks
[TASK SELECT] Easy: 36, Medium: 36, Hard: 36, Total: 108
[EPOCH 1/162] Starting...
```

**Monitor GPU utilization:**
```bash
# Check GPU is being used (expect >182 MiB VRAM, GPU-Util >0%)
watch -n 5 nvidia-smi
```

**Expected GPU output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx            Driver Version: 535.xx.xx    CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce RTX 3060   Off  | 00000000:01:00.0 Off |                  N/A |
|  0%   45C    P2    XX W /  170W |   XXXX MiB / 12288MiB |     XX%      Default |
+-------------------------------+----------------------+----------------------+
```

**Success Signals:**
- ✅ tmux session `arc_run_037` running
- ✅ Log file `/K3D/Knowledge3D.local/logs/run_037.log` has content
- ✅ GPU memory usage >182 MiB (diagnostic used only 182 MiB)
- ✅ GPU-Util >0% (GPU actively processing)
- ✅ Training log shows epoch progression

**If GPU not active after 2 minutes:**
- Check tmux session: `tmux attach -t arc_run_037`
- Look for errors in log: `tail -n 100 /K3D/Knowledge3D.local/logs/run_037.log`
- Report immediately and STOP

### Step 4: Confirm Launch & Wait for User

**Once GPU is stable (VRAM settled, training progressing):**

Report to user:
```
[PHASE 5 LAUNCHED] Run 037 training started
- tmux session: arc_run_037 (running)
- Log file: /K3D/Knowledge3D.local/logs/run_037.log
- GPU Status: CUDA_VISIBLE_DEVICES=0, RTX 3060
- GPU Utilization: XXXX MiB VRAM, XX% Util
- Training Progress: Epoch X/162, Task Y/108
- Estimated Duration: ~24 hours

Training is now running in background. GPU is active and processing.
Waiting for your ping back to proceed with monitoring checkpoints.

To view progress:
  tmux attach -t arc_run_037  (Ctrl+B then D to detach)
  tail -f /K3D/Knowledge3D.local/logs/run_037.log
```

**Then:** WAIT for user to ping you back. Do NOT start the 24-hour monitoring routine until user confirms.

---

## Monitoring Checkpoints (Execute AFTER User Confirms)

**User will ping you when ready for monitoring. Once pinged, proceed with:**

### Checkpoint 1: Epoch 10 (~45 minutes after start)
```bash
tail -n 100 /K3D/Knowledge3D.local/logs/run_037.log | grep -E "EPOCH 10|Accuracy|vocabulary|scale_invariant|PTX"
```

**Check for:**
- Vocabulary logging should appear (logs every 10 epochs)
- Scale-invariant usage still >0
- Accuracy trends (should be climbing from 23% baseline)

### Checkpoint 2: Epoch 50 (~3.5 hours after start)
```bash
tail -n 200 /K3D/Knowledge3D.local/logs/run_037.log | grep -E "EPOCH 50|Accuracy|Shadow Copy|grammar|drawing"
```

**Check for:**
- Accuracy approaching 35-40% (recovery signal)
- Shadow Copy growth (should be 50+ entries)
- Vocabulary detection consistent

### Checkpoint 3: Epoch 100 (~7 hours after start)
```bash
tail -n 200 /K3D/Knowledge3D.local/logs/run_037.log | grep -E "EPOCH 100|Accuracy|Shadow|scale_invariant"
```

**Check for:**
- Accuracy stabilizing 38-45%
- Shadow Copy 80-100 entries
- Scale-invariant primitives still active

### Checkpoint 4: Epoch 162 (Completion, ~24 hours)
```bash
# Full final analysis
tail -n 500 /K3D/Knowledge3D.local/logs/run_037.log
```

**Extract metrics:**
- Final accuracy (target: 40-47%)
- Total Shadow Copy discoveries
- Scale-invariant usage total
- Vocabulary detection counts
- PTX success rate

---

## Abort Criteria (Check at Each Checkpoint)

**STOP training and report immediately if:**
1. PTX crashes or fallback rate >50%
2. Scale-invariant usage drops to 0 (Fix 2 broke)
3. Vocabulary detection stays 0 after epoch 20 (Fix 3 broke)
4. Shadow Copy stops growing entirely (Fix 4 broke)
5. Accuracy <25% after epoch 100 (no learning signal)
6. GPU utilization drops to 182 MiB (GPU not being used)

**To stop training:**
```bash
# Kill tmux session
tmux kill-session -t arc_run_037

# Archive logs
cp /K3D/Knowledge3D.local/logs/run_037.log /K3D/Knowledge3D.local/logs/run_037_aborted_$(date +%Y%m%d_%H%M%S).log

# Report abort reason to user/Claude
```

---

## Expected Timeline

| Checkpoint | Time | Epochs | What to Check |
|------------|------|--------|---------------|
| Launch | 0:00 | 0 | GPU active, training started |
| Early | 0:45 | 10 | Vocabulary logs appear, accuracy climbing |
| Mid-Early | 3:30 | 50 | Accuracy 35-40%, Shadow Copy 50+ |
| Mid | 7:00 | 100 | Accuracy 38-45%, Shadow Copy 80-100 |
| Late | 12:00 | 130 | Refinement phase, accuracy stable |
| Complete | 24:00 | 162 | Final metrics, full report |

---

## Important Notes

1. **Wait for user ping:** Do NOT start the monitoring checkpoints until user confirms. Just launch and confirm GPU is active.

2. **tmux Persistence:** Training runs in background. You can disconnect and it will continue.

3. **Log Monitoring:** Use `tail -f /K3D/Knowledge3D.local/logs/run_037.log` to watch live progress.

4. **GPU Priority:** This process has GPU priority. Do not start other GPU tasks during training.

5. **Checkpoint Files:** Training saves checkpoints after each epoch to `/K3D/Knowledge3D.local/checkpoints/arc_agi/`

6. **Fresh Bootstrap Context:** This run starts from fresh bootstrap (18 shadow entries from diagnostic). Accuracy will start low (~23%) and climb as Shadow Copy grows. This is expected behavior, NOT a regression.

---

## Start Now

**Execute Step 1-4 above:**
1. Set up GPU and environment
2. Launch tmux session with training command
3. Verify GPU is active and training started
4. Report to user and WAIT for ping

**Do NOT proceed to monitoring checkpoints until user pings you back.**

Good luck! This is the validation run that will confirm whether the regression fixes delivered full recovery.
