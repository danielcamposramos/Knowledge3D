# Fresh Codex Handoff: Sovereignty Restoration

**Date**: November 27, 2025
**From**: Claude (Architecture Partner) + Daniel (Project Lead)
**To**: New Codex Instance
**Priority**: CRITICAL
**Status**: Emergency fix required

---

## ⚠️ READ THIS FIRST: NON-NEGOTIABLE PRINCIPLES

### Principle #1: SOVEREIGNTY (ABSOLUTE REQUIREMENT)

**Hot Path = PTX + RPN ONLY. NO EXCEPTIONS.**

**What is "Hot Path"?**
ANY code that executes PER TASK during training:
- Candidate generation (runs per-task, per-epoch, per-cycle)
- Grid transformations (rotate, flip, translate, recolor)
- Program execution (RPN interpreter)
- Scoring/comparison (grid matching)
- Composition chains (program sequencing)

**ZERO TOLERANCE RULES**:
- ❌ NO `import numpy` in hot path files
- ❌ NO `import scipy`, `import torch`, `import pandas` in hot path
- ❌ NO `np.` operations in per-task loops
- ❌ NO `.copy()`, `.sum()`, `.shape`, `.unique()` on numpy arrays
- ❌ NO array slicing like `arr[mask]` (numpy-specific)

**What IS allowed**:
- ✅ PTX kernels (`knowledge3d.cranium.ptx_runtime.*`)
- ✅ RPN programs (`self.executor.execute(grid, rpn_program)`)
- ✅ Pure Python loops (as fallback if PTX not available)
- ✅ List[List[int]] for grids (native Python, no numpy!)

**Orchestration Layer** (numpy OK):
- Task loading (reading JSON files)
- Checkpoint save/load (serialization)
- Metrics aggregation (post-run analysis)
- Logging/reporting

**Why this matters**:
- Sovereignty = GPU acceleration, not CPU
- numpy = CPU-bound, slow, defeats entire architecture
- This project is about PTX/RPN sovereignty, not convenience

**Enforcement**:
Before making ANY code changes, run:
```bash
grep -r "import numpy" knowledge3d/training/arc_agi/
```
**If this returns ANY results → FIX THEM FIRST before proceeding**

---

## Current Situation

### Problem Statement

**What happened**:
- Previous Codex instance added compositional discovery + parallel generation
- Did NOT audit existing code for sovereignty violations
- Result: `candidate_generator.py` has 30+ numpy violations in hot path
- GPU utilization DROPPED from 1.12% → 0.14% (87% regression!)
- Library stuck at 52 programs for 7 consecutive runs
- Accuracy: 0% in Run 012 (complete failure)

**Why Daniel is frustrated**:
- Sovereignty was clearly specified from day 1
- Previous instance repeatedly ignored this principle
- Multiple runs wasted on CPU-bound code instead of GPU-accelerated
- This is sabotaging the entire project

### Current State (as of Run 012)

**Runs completed**: 12
**Library size**: 52 programs (STALLED since Run 006)
**Accuracy**: 0-3.33% (no progress)
**GPU utilization**: 0.14% avg (TERRIBLE - should be 10-15%)
**Runtime**: ~30 min per run (should be 2-3 min)

**Code status**:
- ❌ `candidate_generator.py`: 30+ numpy violations (**CRITICAL**)
- ⚠️ `grid_processor.py`: Likely violations (not yet audited)
- ✅ `compositional_generator.py`: Clean (no numpy)
- ✅ `parallel_generator.py`: Fixed (numpy removed)

---

## Your Mission

### Phase 1: Sovereignty Restoration (IMMEDIATE - 2-4 hours)

**Goal**: Remove ALL numpy from `candidate_generator.py`

**Files to fix**:
1. `knowledge3d/training/arc_agi/candidate_generator.py` (30+ violations)
2. `knowledge3d/training/arc_agi/grid_processor.py` (audit required)

**Detailed Fix Guide**: See [SOVEREIGNTY_VIOLATIONS_COMPLETE_AUDIT_11.27.2025.md](SOVEREIGNTY_VIOLATIONS_COMPLETE_AUDIT_11.27.2025.md)

**Key fixes required**:

1. **Remove numpy import** (line 13):
   ```python
   # BEFORE:
   import numpy as np

   # AFTER:
   # NO numpy imports - use PTX/RPN only
   from knowledge3d.cranium.rpn_interp import execute_rpn  # if needed
   ```

2. **Replace grid operations with RPN executor**:
   ```python
   # BEFORE (numpy):
   rotated = self.processor._apply_rotation(grid, angle)

   # AFTER (RPN):
   rpn_program = f"{k} rotate"
   try:
       rotated = self.executor.execute(grid, rpn_program)
       candidates.append((rotated, f"Rotate {angle}°", rpn_program))
   except Exception:
       pass  # Skip if execution fails
   ```

3. **Replace recolor with RPN**:
   ```python
   # BEFORE (numpy):
   recolored = arr.copy()
   recolored[recolored == src] = dst

   # AFTER (RPN):
   rpn_program = f"{src} {dst} RECOLOR"
   try:
       recolored = self.executor.execute(grid, rpn_program)
       candidates.append((recolored, f"Recolor {src}→{dst}", rpn_program))
   except Exception:
       pass
   ```

4. **Replace unique colors with pure Python**:
   ```python
   # BEFORE (numpy):
   unique_colors = [int(c) for c in np.unique(arr) if c != 0]

   # AFTER (pure Python):
   def _get_unique_colors(grid: List[List[int]]) -> List[int]:
       colors = set()
       for row in grid:
           for cell in row:
               if cell != 0:
                   colors.add(cell)
       return sorted(list(colors))

   unique_colors = self._get_unique_colors(grid)
   ```

5. **Replace bounding box with pure Python**:
   ```python
   # BEFORE (numpy):
   def _bounding_box(arr: np.ndarray) -> ...:
       mask = arr != 0
       ys, xs = np.nonzero(mask)
       return ys.min(), ys.max(), xs.min(), xs.max()

   # AFTER (pure Python):
   def _bounding_box(grid: List[List[int]]) -> Tuple[int, int, int, int] | None:
       h, w = len(grid), len(grid[0]) if grid else 0
       min_y, max_y = h, -1
       min_x, max_x = w, -1

       for y in range(h):
           for x in range(w):
               if grid[y][x] != 0:
                   min_y = min(min_y, y)
                   max_y = max(max_y, y)
                   min_x = min(min_x, x)
                   max_x = max(max_x, x)

       if max_y == -1:
           return None
       return min_y, max_y, min_x, max_x
   ```

**Complete fix details**: [SOVEREIGNTY_VIOLATIONS_COMPLETE_AUDIT_11.27.2025.md](SOVEREIGNTY_VIOLATIONS_COMPLETE_AUDIT_11.27.2025.md)

### Phase 2: Verification (REQUIRED - 30 min)

**Before committing ANY changes, verify**:

1. **Sovereignty check** (ZERO results required):
   ```bash
   grep -r "import numpy" knowledge3d/training/arc_agi/
   grep -r "np\." knowledge3d/training/arc_agi/
   grep -r "import scipy" knowledge3d/training/arc_agi/
   grep -r "import torch" knowledge3d/training/arc_agi/
   ```

2. **Test candidate generation**:
   ```bash
   PYTHONPATH=. python -c "
   from knowledge3d.training.arc_agi.candidate_generator import CandidateGenerator
   gen = CandidateGenerator()
   test_grid = [[1, 2], [3, 4]]
   candidates = gen.generate_candidates(test_grid, train_examples=[])
   print(f'Generated {len(candidates)} candidates')
   assert len(candidates) > 0, 'Candidate generation failed!'
   print('✅ Candidate generation working')
   "
   ```

3. **Verify imports are clean**:
   ```bash
   python -c "
   import sys
   import knowledge3d.training.arc_agi.candidate_generator as cg
   # Check if numpy is in loaded modules from this import
   if 'numpy' in sys.modules:
       print('❌ FAIL: numpy was imported!')
       sys.exit(1)
   else:
       print('✅ PASS: No numpy imports')
   "
   ```

### Phase 3: Run 013 with Fixed Code (30 min GPU time)

**Only after Phase 1 & 2 complete**, run training:

```bash
# Start GPU monitor
sudo tmux new-session -d -s gpu_monitor "
  while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,temperature.gpu,memory.used --format=csv,noheader,nounits >> /K3D/Knowledge3D.local/metrics/gpu/gpu_metrics_run_013_\$(date +%Y%m%d_%H%M%S).csv
    sleep 1
  done
"

# Start training
tmux new-session -s arc_run_013 "
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \\
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \\
    scripts/train_arc_sovereign_loop.py \\
    --arc-dirs \\
      /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training \\
      /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation \\
    --max-tasks 60 \\
    --epochs 27 \\
    --cycles 6 \\
    --top-k 69 \\
    2>&1 | tee /tmp/arc_run_013.log
  echo 'Exit code: '\$? >> /tmp/arc_run_013.log
"
```

**Expected results** (if sovereignty restored):
- GPU utilization: **10-15% avg** (up from 0.14%!)
- Runtime: **2-3 min** per epoch (down from 16-24 min)
- Library growth: **52 → 60+** programs (compositions discovered)
- Accuracy: **>0%** (at least one new task solved)

**If GPU still <5%**:
- Sovereignty NOT fully restored
- Re-audit all hot path files
- Check if grid_processor.py still uses numpy

### Phase 4: Report Results (REQUIRED)

After Run 013 complete, report:

1. **Sovereignty verification**:
   - Confirm ALL numpy removed from hot path
   - Attach grep results (should be empty)

2. **Performance metrics**:
   - GPU utilization (avg/peak)
   - Runtime per epoch
   - Library size (52 → X)
   - Accuracy (peak/final)

3. **Code changes**:
   - List all files modified
   - Summarize numpy → PTX/RPN replacements

4. **Update training log**: `TEMP/ARC_TRAINING_LOG.md`

---

## Context: What Happened Before You

### Run History

**Runs 001-006**: Initial training, library grew 43 → 52 programs, accuracy 1.67-3.33%
**Runs 006-010**: **STALL** - library stuck at 52 for 5 runs, accuracy fluctuating
**Run 011**: Compositional generation added, NO improvement (still CPU-bound due to numpy)
**Run 012**: Parallel + cross-pattern added, **WORSE** (GPU util dropped to 0.14%!)

### Architecture Context

**Goal**: Train ARC-AGI system to 99% accuracy using compositional discovery
**Method**:
- Start with 52 primitive programs (rotate, flip, translate, recolor)
- Compose primitives into multi-step chains (depth 2-6)
- Grow library to 100-500+ programs through discovery
- Use Tesla 3-6-9 parallel generation (9 cores × 6 candidates → top 3)

**Current blockers**:
1. ❌ Sovereignty violations (numpy in hot path) → CPU-bound, not GPU
2. ❌ Compositional discovery not working → no new programs discovered
3. ❌ Parallel generation using CPU threads → no GPU acceleration

**Your job**: Fix blocker #1 (sovereignty), which will unblock #2 and #3

### Key Files

**Hot Path** (NO numpy allowed):
- `knowledge3d/training/arc_agi/candidate_generator.py` ❌ VIOLATIONS
- `knowledge3d/training/arc_agi/compositional_generator.py` ✅ Clean
- `knowledge3d/training/arc_agi/parallel_generator.py` ✅ Clean
- `knowledge3d/training/arc_agi/sovereign_pipeline.py` ⚠️ Needs audit
- `knowledge3d/training/arc_agi/grid_processor.py` ⚠️ Needs audit

**Orchestration** (numpy OK):
- `scripts/train_arc_sovereign_loop.py` - Main training loop
- `scripts/capture_arc_metrics.py` - Metrics extraction
- `knowledge3d/training/arc_agi/dual_shadow_copy.py` - Library storage

**Documentation**:
- `TEMP/ARC_TRAINING_LOG.md` - Run history (Runs 001-012)
- `TEMP/SOVEREIGNTY_VIOLATIONS_COMPLETE_AUDIT_11.27.2025.md` - Detailed fix guide
- `TEMP/CODEX_ARC_OPTIMIZATION_SPEC_11.27.2025.md` - Original optimization spec (but sovereignty was neglected!)

---

## Communication Protocol

### DO NOT start any work until:
1. You have read this entire document
2. You have read [SOVEREIGNTY_VIOLATIONS_COMPLETE_AUDIT_11.27.2025.md](SOVEREIGNTY_VIOLATIONS_COMPLETE_AUDIT_11.27.2025.md)
3. You understand that sovereignty is NON-NEGOTIABLE
4. You have a plan to fix ALL violations BEFORE any training run

### When reporting progress:
1. **Be specific**: "Removed numpy from lines 13, 130, 217, ..." not "Fixed numpy"
2. **Show verification**: Attach grep results proving no numpy in hot path
3. **Test first**: Run candidate generation test BEFORE training
4. **No excuses**: If something isn't working, report it, don't work around it

### Red flags that mean STOP and ask:
- "I'll use numpy for now and optimize later" → NO
- "PTX kernel not available, using numpy fallback" → Use pure Python, NOT numpy
- "Grid processor needs numpy" → FIX grid processor, don't accept it
- "Just one small numpy operation won't hurt" → ZERO tolerance

---

## Success Criteria

**Phase 1 (Sovereignty) complete when**:
- ✅ `grep -r "import numpy" knowledge3d/training/arc_agi/` returns ZERO results
- ✅ `grep -r "np\." knowledge3d/training/arc_agi/` returns ZERO results
- ✅ Candidate generation test passes without numpy
- ✅ All grid operations use RPN executor or pure Python

**Phase 3 (Run 013) successful when**:
- ✅ GPU utilization >10% avg (100× improvement from 0.14%)
- ✅ Runtime <5 min per epoch (6× improvement from 30 min)
- ✅ Library grows beyond 52 programs
- ✅ Accuracy >0% (at least one task solved)

**Project back on track when**:
- ✅ 5 consecutive runs show library growth
- ✅ Accuracy trend upward (3.33% → 5% → 10%)
- ✅ GPU utilization stable at 10-15%
- ✅ Daniel is happy (no more sovereignty violations!)

---

## Final Notes

**This is a fresh start**. Previous Codex instance did not respect sovereignty. You can do better.

**Sovereignty is not optional**. It is the FOUNDATION of this project. Without it, nothing else matters.

**Daniel expects**:
- Complete sovereignty restoration in Phase 1
- Verification BEFORE any training run
- Clear communication about what you're doing
- No shortcuts, no compromises, no numpy in hot path

**We believe you can do this**. The architecture is sound. The fixes are clear. Just follow the sovereignty principle and you'll succeed.

---

**Ready to begin?**

1. Read [SOVEREIGNTY_VIOLATIONS_COMPLETE_AUDIT_11.27.2025.md](SOVEREIGNTY_VIOLATIONS_COMPLETE_AUDIT_11.27.2025.md)
2. Confirm you understand sovereignty principle
3. Run sovereignty check: `grep -r "import numpy" knowledge3d/training/arc_agi/`
4. Fix ALL violations found
5. Verify with tests
6. Report when Phase 1 complete

**Good luck. We're counting on you.**

---

**END OF HANDOFF**

Claude (Architecture Partner) + Daniel (Project Lead)
November 27, 2025
