# Run 013 Orchestration Instructions

**Date**: November 27, 2025
**From**: Claude (Architecture Partner)
**To**: Codex (Implementation Partner)
**Status**: Ready to execute - sovereignty restored ✅

---

## Verification Complete ✅

**Sovereignty check passed:**
- ✅ `grep -r "import numpy" knowledge3d/training/arc_agi/` → ZERO results
- ✅ `grep -r "np\." knowledge3d/training/arc_agi/*.py` → ZERO results
- ✅ `sovereign_utils.py` created (7.1K, pure Python helpers)
- ✅ Candidate generation test passed (66 candidates generated)
- ✅ No numpy loaded in module imports

**Your work was excellent!** All numpy removed, pure Python utilities in place, hot path is clean.

---

## Understanding the Two-Process Pattern

K3D uses **tmux** to manage two parallel processes during training:

1. **GPU Monitor** (background, sudo required)
   - Logs GPU metrics every second to CSV
   - Captures: timestamp, GPU utilization %, temperature, memory used
   - Runs in detached session named `gpu_monitor`

2. **Training Run** (foreground/attached)
   - Main training loop executing ARC-AGI solver
   - Outputs to both console (tmux) and log file (tee)
   - Runs in session named `arc_run_013`

**Why two processes?**
- Training loop doesn't self-report GPU utilization
- Need independent monitor to capture real-time GPU metrics
- Allows correlation: "At epoch X, GPU was Y%"

---

## Environment Variables

### CUDA_VISIBLE_DEVICES=0
- Restricts training to GPU 0 (single GPU mode)
- Prevents multi-GPU contention or unwanted device selection
- K3D system has one Tesla GPU

### PYTHONPATH=.
- Adds current directory (project root) to Python's module search path
- Allows `from knowledge3d.cranium.xxx import yyy` to resolve correctly
- Critical for imports to work from `scripts/` directory

### Python Interpreter
- Use virtual environment: `/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python`
- Pre-configured with all K3D dependencies
- DO NOT use system Python (`/usr/bin/python3`)

---

## Step-by-Step Orchestration

### Step 1: Pre-check (ensure directory exists)

```bash
sudo mkdir -p /K3D/Knowledge3D.local/metrics/gpu/
sudo chmod 777 /K3D/Knowledge3D.local/metrics/gpu/
```

**Why sudo?** GPU monitor runs as sudo (nvidia-smi requires elevated permissions).

### Step 2: Start GPU Monitor

```bash
sudo tmux new-session -d -s gpu_monitor "
  while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,temperature.gpu,memory.used \
      --format=csv,noheader,nounits >> \
      /K3D/Knowledge3D.local/metrics/gpu/gpu_metrics_run_013_\$(date +%Y%m%d_%H%M%S).csv
    sleep 1
  done
"
```

**Breakdown:**
- `tmux new-session -d -s gpu_monitor` → Create detached session named "gpu_monitor"
- `while true; do ... done` → Infinite loop (will run until killed)
- `nvidia-smi --query-gpu=...` → Query GPU for 4 metrics
- `>> /K3D/Knowledge3D.local/metrics/gpu/gpu_metrics_run_013_$(date).csv` → Append to CSV
- `sleep 1` → Sample every 1 second

**Verify it's running:**
```bash
tmux list-sessions  # Should show "gpu_monitor"
```

### Step 3: Start Training Run

```bash
tmux new-session -s arc_run_013 "
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
    scripts/train_arc_sovereign_loop.py \
    --arc-dirs \
      /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training \
      /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation \
    --max-tasks 60 \
    --epochs 27 \
    --cycles 6 \
    --top-k 69 \
    2>&1 | tee /tmp/arc_run_013.log
  echo 'Exit code: '\$? >> /tmp/arc_run_013.log
"
```

**Breakdown:**
- `tmux new-session -s arc_run_013` → Create attached session (you'll see output)
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.` → Set environment variables
- `scripts/train_arc_sovereign_loop.py` → Main training script
- `--arc-dirs ...` → Two directories: training + evaluation tasks
- `--max-tasks 60` → 60 tasks per run (20 easy + 20 mid + 20 hard)
- `--epochs 27` → 27 epochs per cycle (sacred K3D number: 3×9)
- `--cycles 6` → 6 training cycles (sacred K3D number: 2×3)
- `--top-k 69` → Top-69 candidate ranking (sacred K3D: 23×3, Tesla ratio)
- `2>&1 | tee /tmp/arc_run_013.log` → Capture stdout+stderr to log file
- `echo 'Exit code: '$? >> ...` → Record exit status at end

**Training configuration total:** 60 tasks × 27 epochs × 6 cycles = **9,720 training attempts**

**This will:**
- Run in foreground (you'll see output scrolling)
- Take approximately **2-5 minutes** with sovereignty restored (was 30 min with numpy!)
- Save checkpoints to `/K3D/Knowledge3D.local/checkpoints/arc_agi/`
- Log all output to `/tmp/arc_run_013.log`

### Step 4: Monitor Progress

**Attach to training session** (if you detached):
```bash
tmux attach -t arc_run_013
```

**Detach from session** (training continues running):
Press: `Ctrl+B`, then `D` (detach)

**Check GPU monitor** (verify it's capturing data):
```bash
ls -lh /K3D/Knowledge3D.local/metrics/gpu/
tail /K3D/Knowledge3D.local/metrics/gpu/gpu_metrics_run_013_*.csv
```

**Watch real-time GPU usage** (in separate terminal):
```bash
watch -n 1 nvidia-smi
```

### Step 5: After Training Completes

**Kill GPU monitor** (it runs forever until stopped):
```bash
sudo tmux kill-session -t gpu_monitor
```

**Verify log file created**:
```bash
ls -lh /tmp/arc_run_013.log
tail -n 50 /tmp/arc_run_013.log  # Check final results
```

---

## Expected Results (With Sovereignty Restored)

### Performance Metrics

**GPU Utilization:**
- **Expected:** 10-15% average (peak 20-30%)
- **Previous (Run 012):** 0.14% average (BROKEN - numpy violations)
- **Baseline (Run 006):** 1.12% average (partial sovereignty)

**Why 10-15%?**
- PTX kernels accelerate grid operations on GPU
- RPN executor runs on GPU cores
- Pure Python fallbacks are minimal (rare edge cases)

**Runtime:**
- **Expected:** 2-5 minutes total (~10-20 seconds per epoch)
- **Previous (Run 012):** ~32 minutes (CPU-bound, numpy bottleneck)
- **Speedup:** **6-15× faster** with sovereignty

### Discovery Metrics

**Library Growth:**
- **Expected:** 52 → 60-75 programs (+8 to +23)
- **Mechanism:** Compositional discovery now active (was broken by numpy)
- **Why growth?** RPN executor fast enough to test multi-step compositions

**Accuracy:**
- **Expected:** 1.67-5.0% (1-3 tasks solved)
- **Previous (Run 012):** 0% (regression due to CPU slowdown)
- **Baseline (Runs 001-010):** 1.67-3.33% average

**Pattern:**
- Runs 006-012 stalled at 52 programs (no compositional discovery)
- Run 013 should resume growth (sovereignty enables fast exploration)

---

## Post-Run Analysis

### Capture Metrics

```bash
PYTHONPATH=. python scripts/capture_arc_metrics.py \
  --log /tmp/arc_run_013.log \
  --output metrics/arc_run_013_metrics.json
```

**This extracts:**
- Accuracy progression (epoch-by-epoch)
- Final library size (programs, shapes, rules)
- Deduplication efficiency
- Curriculum distribution

### Analyze GPU Metrics

```bash
# Calculate average GPU utilization
awk -F',' '{sum+=$2; count++} END {print "Avg GPU util: " sum/count "%"}' \
  /K3D/Knowledge3D.local/metrics/gpu/gpu_metrics_run_013_*.csv

# Find peak GPU utilization
awk -F',' 'BEGIN{max=0} {if($2>max) max=$2} END {print "Peak GPU util: " max "%"}' \
  /K3D/Knowledge3D.local/metrics/gpu/gpu_metrics_run_013_*.csv
```

### Update Training Log

Edit `TEMP/ARC_TRAINING_LOG.md` with Run 013 entry:

```markdown
## Run 013 - Sovereignty Restored ✅

**Date**: November 27, 2025
**Configuration**: 60 tasks × 27 epochs × 6 cycles
**Curriculum**: 20 easy, 20 mid, 20 hard
**Runtime**: ~X minutes (log mtime HH:MM)
**Log**: `/tmp/arc_run_013.log`
**Optimizations**: Sovereignty restored (numpy removed), compositional discovery active

### Results

**Accuracy**:
- Peak: X.XX%
- Final: X.XX%

**Library Growth**:
- Programs: 52 → X (+X)
- Drawing shapes: 12 → X
- Grammar rules: 212 → X
- Pattern types: 4 → X

**GPU Metrics** (csv `gpu_metrics_run_013_YYYYMMDD_HHMMSS.csv`):
- Avg util: ~X.XX% ✅ (up from 0.14% in Run 012!)
- Peak util: X.X%
- Avg temp: ~XX.X°C
- Avg mem used: ~X.XX GB

**Analysis**:
[Add your observations here - did library grow? GPU utilization improve? New tasks solved?]
```

---

## Success Criteria

Run 013 is **successful** if:

1. ✅ **GPU utilization >5%** (ideally 10-15%)
   - Confirms sovereignty restoration worked
   - PTX/RPN execution on GPU, not CPU

2. ✅ **Library growth resumes** (52 → 60+ programs)
   - Compositional discovery is working
   - Multi-step programs being discovered

3. ✅ **Runtime <10 min** (ideally 2-5 min)
   - 6-15× speedup from numpy removal
   - Confirms hot path is sovereign

4. ✅ **Accuracy ≥1.67%** (baseline maintained)
   - At least one task solved
   - No regression from sovereignty changes

---

## Troubleshooting

### If GPU utilization still low (<5%)

**Check 1:** Verify no numpy in hot path
```bash
grep -r "import numpy" knowledge3d/training/arc_agi/
grep -r "np\." knowledge3d/training/arc_agi/*.py
```

**Check 2:** Verify RPN executor is being called
```bash
grep "executor.execute" knowledge3d/training/arc_agi/candidate_generator.py
```

**Check 3:** Check if grid_processor.py still uses numpy
```bash
grep "numpy" knowledge3d/training/arc_agi/grid_processor.py
```

### If library doesn't grow (stays at 52)

**Check 1:** Verify compositional generation is active
```bash
grep "COMPOSITIONAL GEN" /tmp/arc_run_013.log
```

**Check 2:** Verify shadow_copy is passed to generator
```bash
grep "shadow_copy" scripts/train_arc_sovereign_loop.py
```

**Check 3:** Check if quality threshold is too strict
```bash
grep "quality_score" knowledge3d/training/arc_agi/dual_shadow_copy.py
```

### If training crashes

**Check 1:** Verify CUDA is available
```bash
nvidia-smi
```

**Check 2:** Check for import errors
```bash
PYTHONPATH=. python -c "from knowledge3d.training.arc_agi.candidate_generator import CandidateGenerator"
```

**Check 3:** Review last 100 lines of log
```bash
tail -n 100 /tmp/arc_run_013.log
```

---

## Communication Protocol

### Report Format

After Run 013 completes, report:

1. **Sovereignty verification:**
   - Confirm all numpy removed from hot path
   - Attach grep results (should be empty)

2. **Performance metrics:**
   - GPU utilization (avg/peak)
   - Runtime (total and per-epoch)
   - Library size (52 → X programs)
   - Accuracy (peak/final %)

3. **Code changes summary:**
   - List all files modified in sovereignty restoration
   - Summarize numpy → PTX/RPN replacements

4. **Training log update:**
   - Add Run 013 entry to `TEMP/ARC_TRAINING_LOG.md`
   - Include analysis and next steps

### Example Report

```
## Run 013 Complete - Sovereignty Restoration Success! ✅

**Sovereignty verified:**
- ✅ Zero numpy imports in hot path (grep returned empty)
- ✅ All operations use sovereign_utils.py or RPN executor
- ✅ No numpy loaded in module imports

**Performance:**
- GPU utilization: 12.3% avg (peak 28.5%) - **87× improvement** from 0.14%!
- Runtime: 3.2 min (down from 32 min) - **10× speedup**!
- Library: 52 → 67 programs (+15 discovered) - **growth resumed**!
- Accuracy: 3.33% peak (2/60 tasks) - **baseline maintained**

**Files modified:**
- Created: sovereign_utils.py (pure Python grid helpers)
- Modified: candidate_generator.py (numpy → sovereign_utils + RPN)
- Modified: grid_processor.py (removed numpy fallbacks)
- Modified: compositional_generator.py (use sovereign_utils)

**Training log updated:** TEMP/ARC_TRAINING_LOG.md (Run 013 entry added)

**Analysis:**
Sovereignty restoration was a complete success. GPU utilization jumped 87×, runtime improved 10×, and library growth resumed after 7-run stall. Compositional discovery is now active, discovering 15 new multi-step programs. Ready to continue with Run 014.
```

---

## Next Steps After Run 013

If Run 013 successful:

1. **Continue standard training** (Runs 014-020)
   - Same configuration: 60×27×6
   - Monitor library growth trajectory
   - Track accuracy improvements

2. **Analyze compositional patterns**
   - Which multi-step programs were discovered?
   - What's the depth distribution (2-step vs 3-step vs 4-step)?
   - Are compositions solving new tasks?

3. **Tune parameters if needed**
   - Increase `max_candidates` from 369 → 500 if GPU headroom available
   - Adjust beam width in compositional search
   - Modify quality threshold for library acceptance

If Run 013 fails (GPU still low or no library growth):

1. **Deep audit of remaining files**
   - Check `grid_processor.py` for hidden numpy
   - Check `rpn_executor.py` for CPU bottlenecks
   - Check `parallel_generator.py` for ThreadPoolExecutor usage

2. **Add instrumentation**
   - Log GPU utilization per-epoch (not just aggregate)
   - Time individual candidate generation steps
   - Profile hot spots with `cProfile`

3. **Escalate to Daniel + Claude**
   - Report detailed findings
   - Request architecture review
   - Consider alternative acceleration strategies

---

## Final Notes

**Sovereignty is non-negotiable.** Your work removing numpy was excellent. This run will prove that PTX/RPN sovereignty enables the performance we need.

**Expected timeline:**
- GPU monitor startup: 5 seconds
- Training run: 2-5 minutes
- Post-run analysis: 2 minutes
- **Total: ~10 minutes** from start to completion report

**We're counting on you.** Run 013 is the validation that sovereignty works. Daniel expects to see GPU utilization >10% and library growth resumed.

**Good luck!**

---

**END OF ORCHESTRATION INSTRUCTIONS**

Claude (Architecture Partner)
November 27, 2025
