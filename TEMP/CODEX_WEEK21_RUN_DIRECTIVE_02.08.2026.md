# CODEX: Week 21 Full Training Run — Execute Now

**Date:** February 8, 2026
**Status:** 🚀 LAUNCH DIRECTIVE
**Priority:** Execute full progressive curriculum training

---

## 🎯 Mission: Week 21 Progressive Curriculum Training

### Execute This Command

```bash
python3 scripts/train_deterministic_foundation.py \
  --iterations 10 \
  --tasks-per-category 100 \
  --storage-root ../Knowledge3D.local/foundation_curriculum_world \
  --output-dir ../Knowledge3D.local/results/foundation_training_week21
```

**Parameters:**
- `--iterations 10` — Run 10 training iterations
- `--tasks-per-category 100` — 500 total tasks (100 geometric, 100 arithmetic, etc.)
- `--storage-root` — Persistent world (TRM weights, Galaxy, Shadow Copy evolve)
- `--output-dir` — Results directory for this run

**Expected runtime:** 2-4 hours

---

## 📊 Monitor During Execution

**Watch for:**

1. **Stage A saturation (iterations 0-2):**
   ```
   Iteration 0: Stage A, overall=1.00
   Iteration 1: Stage A, overall=1.00
   Iteration 2: Stage A, overall=1.00
   >>> Auto-advance to Stage B
   ```

2. **Stage B progression (iterations 3-10):**
   ```
   Iteration 3: Stage B, overall=0.60-0.75 (alias inference hard!)
   Iteration 5: Stage B, overall=0.75-0.80 (TRM learning)
   Iteration 8: Stage B, overall=0.80-0.85 (approaching gate)
   ```

3. **Possible Stage C advancement (if Stage B completes):**
   ```
   If 3 consecutive iterations ≥0.85:
   >>> Auto-advance to Stage C
   ```

4. **Shadow Copy consolidation:**
   - Events consolidated per iteration
   - TRM weight deltas (learning signal)

---

## 📝 Report Back When Complete

**Provide:**

1. **Final summary:**
   - Total iterations run
   - Final stage reached (A/B/C/D)
   - Final overall accuracy
   - Stage progression timeline (which iterations advanced?)

2. **Iteration-by-iteration progression:**
   - Stage + accuracy for each iteration
   - Example: `[A:1.00, A:1.00, A:1.00, B:0.68, B:0.74, B:0.78, B:0.82, B:0.84, B:0.86, B:0.88]`

3. **Training history artifact:**
   - Path: `../Knowledge3D.local/results/foundation_training_week21/training_history.json`
   - Confirm file created and size

4. **Shadow Copy learning:**
   - Total events consolidated across all iterations
   - TRM weight progression (initial → final)

5. **Per-category breakdown (final iteration):**
   - Geometric: X%
   - Arithmetic: X%
   - Pattern: X%
   - Compositional: X%
   - RPN: X%

---

## 🔬 After Training: ARC-AGI Transfer Test

**Immediately after Week 21 completes, run:**

```bash
python3 benchmarks/arc_agi_2.py \
  --num-tasks 100 \
  --enriched \
  --storage-root ../Knowledge3D.local/foundation_curriculum_world
```

**Critical:** Use SAME `storage-root` (persistent world with trained TRM)

**Expected results:**
- Baseline (before curriculum): 28%
- After Stage B curriculum: **35-40%** (+7-12% improvement!)

**Report:**
- ARC-AGI accuracy (after curriculum training)
- Comparison to baseline (28%)
- Delta (improvement from curriculum)
- Sample correct/incorrect tasks

---

## 🎯 Success Criteria

**Week 21 training success:**
- ✅ Stage A completes (3 iterations at 1.00)
- ✅ Stage B progression visible (60% → 80%+)
- ✅ Shadow Copy learning active (events consolidated, weights updated)
- ✅ Persistent world intact (Galaxy + TRM weights + Shadow Copy)

**ARC-AGI transfer success:**
- ✅ Accuracy improves from 28% baseline
- ✅ Expected: 35-40% (+7-12%)
- ✅ If achieved: Stage B curriculum validated! 🎉

---

## 🚀 Execute Now

**Launch the Week 21 run and report back when complete!**

**We're watching TRM learn to walk before running the marathon.** 👶➡️🧒➡️🏃

---

**Directive issued by:** Claude (Architecture Partner)
**For:** Codex (Implementation Partner)
**Date:** February 8, 2026
**Status:** 🚀 EXECUTE NOW
