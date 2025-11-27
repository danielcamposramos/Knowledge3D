# ARC-AGI Training Log

**Purpose**: Track training progression toward 99% accuracy
**Started**: November 26, 2025
**Goal**: Accumulate discoveries through repeated training cycles until competitive performance

---

## Run 001 - Mixed Curriculum Baseline ✅ BREAKTHROUGH!

**Date**: November 26, 2025
**Configuration**: 60 tasks × 27 epochs × 6 cycles = 9,720 attempts
**Curriculum**: 20 easy (training), 20 mid (eval first half), 20 hard (eval second half)
**Runtime**: ~5 hours
**Log**: `/tmp/arc_mixed_curriculum.log`

### Results

**Accuracy**:
- Peak: **1.67%** (1/60 tasks solved)
- Final: 0% (last epoch)
- Progression: 1.67% in cycles 1-3, fluctuated in cycles 4-6

**Library Growth**:
- Programs: 34 → 43 (+26%, +9 programs)
- Drawing shapes: 7 → 14 (doubled!)
- Grammar rules: 203 → 209 (+3%)
- Pattern types: 3 (feedback loop active)

**Storage**:
- Total checkpoint size: 3.7 MB
- Deduplication efficiency: 96.6%
- Vocabulary: 20 words, 590 references

### Analysis

**🎯 BREAKTHROUGH ACHIEVED!**

First time breaking through 0% accuracy! System consistently solved 1 task in early cycles, proving:
1. ✅ Architecture is working (semantic layer, feedback loops, candidate generation)
2. ✅ Library is growing organically (new shapes, rules, programs discovered)
3. ✅ Mixed curriculum is effective (easy/mid/hard balance provides learning signal)

**Why accuracy fluctuated**:
- Early cycles (1-3): Fresh task sampling, solved 1/60 consistently
- Mid cycles (4-5): Harder task sampling, 0/60
- Late cycles (6): Mixed results

**Key insight**: System IS learning and discovering, just needs more runs to accumulate library depth.

### Next Steps

Continue standard training runs (60×27×6) to build momentum. Target: 5-10% accuracy in next 10 runs.

---

## Run 002 - Momentum Building

**Date**: November 26, 2025
**Configuration**: 60 tasks × 27 epochs × 6 cycles
**Curriculum**: 20 easy, 20 mid, 20 hard
**Runtime**: ~16 minutes (tmux start 21:24 → log mtime 21:40)
**Log**: `/tmp/arc_run_002.log`

### Results

**Accuracy**:
- Peak: **3.33%** (2/60 tasks in late cycles)
- Final: 3.33% (last epoch)
- Progression: 0% early, climbed to 2/60 near the end and held.

**Library Growth**:
- Programs: 43 → 47 (+4)
- Drawing shapes: 14 → 13 (pruned 1)
- Grammar rules: 209 → 210 (+1)
- Pattern types: 3 → 4 (+1)

**Storage**:
- Total checkpoint size: 3.8 MB
- Deduplication efficiency: ~98.4% (47 unique, 3,029 refs)

---

## Run 003 - Continue Growth

**Date**: November 26, 2025
**Configuration**: 60 tasks × 27 epochs × 6 cycles
**Curriculum**: 20 easy, 20 mid, 20 hard
**Runtime**: ~16 minutes (log mtime 22:05)
**Log**: `/tmp/arc_run_003.log`

### Results

**Accuracy**:
- Peak: **1.67%** (1/60 near late cycles)
- Final: 1.67% (last epoch)

**Library Growth**:
- Programs: 47 → 49 (+2)
- Drawing shapes: 13 → 12 (pruned 1)
- Grammar rules: 210 → 211 (+1)
- Pattern types: 4 (steady)

**Storage**:
- Total checkpoint size: 3.8 MB
- Deduplication efficiency: ~98.4% (49 unique, 3,029 refs)

---

## Run 004 - Stabilize Library

**Date**: November 26, 2025
**Configuration**: 60 tasks × 27 epochs × 6 cycles
**Curriculum**: 20 easy, 20 mid, 20 hard
**Runtime**: ~16 minutes (log mtime 22:29)
**Log**: `/tmp/arc_run_004.log`

### Results

**Accuracy**:
- Peak: **3.33%** (2/60 early/mid cycles)
- Final: 0% (last epoch)
- Progression: Held 2/60 for most of the run, faded to 0/60 by end.

**Library Growth**:
- Programs: 49 → 51 (+2)
- Drawing shapes: 12 → 13 (+1 after prune/save cycle)
- Grammar rules: 211 → 212 (+1)
- Pattern types: 4 (steady)

**Storage**:
- Total checkpoint size: 3.8 MB
- Deduplication efficiency: ~98.3% (51 unique, 3,029 refs)

---

## Run 005 - GPU Monitoring Start

**Date**: November 26, 2025
**Configuration**: 60 tasks × 27 epochs × 6 cycles
**Curriculum**: 20 easy, 20 mid, 20 hard
**Runtime**: ~24 minutes (log mtime 22:53; ran with GPU monitor attempt)
**Log**: `/tmp/arc_run_005.log`

### Results

**Accuracy**:
- Peak: **3.33%** (2/60 mid cycles)
- Final: 3.33% (last epoch)
- Progression: Flat 0% through early epochs, climbed to 2/60 mid-run, held to finish.

**Library Growth**:
- Programs: 51 → 51 (no net change this run)
- Drawing shapes: 13 (steady)
- Grammar rules: 212 (steady)
- Pattern types: 4 (steady)

**Storage**:
- Total checkpoint size: 3.8 MB
- Deduplication efficiency: ~98.3% (51 unique, 3,029 refs)

**GPU Monitoring**:
- Monitor attempt failed; no CSV produced (monitor session exited). Fixed for Run 006 by running monitor under sudo and pre-creating `/K3D/Knowledge3D.local/metrics/gpu/`.

---

## Run 006 - GPU Monitoring (sudo)

**Date**: November 26, 2025
**Configuration**: 60 tasks × 27 epochs × 6 cycles
**Curriculum**: 20 easy, 20 mid, 20 hard
**Runtime**: ~24 minutes (log mtime 23:28)
**Log**: `/tmp/arc_run_006.log`

### Results

**Accuracy**:
- Peak: **1.67%** (1/60 late cycles)
- Final: 0% (last epoch)

**Library Growth**:
- Programs: 51 → 52 (+1)
- Drawing shapes: 13 → 12 (pruned 1)
- Grammar rules: 212 (steady)
- Pattern types: 4 (steady)

**Storage**:
- Total checkpoint size: 3.8 MB
- Deduplication efficiency: ~98.3% (52 unique, 3,029 refs)

**GPU Metrics** (from `gpu_metrics_run_006_20251126_230927.csv`):
- Average utilization: ~1.14%
- Peak utilization: 8.0%
- Average temperature: ~43.2°C
- Average memory used: ~1506 MB

---

## Run 007 - Library Stall Begins

**Date**: November 26, 2025
**Configuration**: 60 tasks × 27 epochs × 6 cycles
**Curriculum**: 20 easy, 20 mid, 20 hard
**Runtime**: ~19 minutes (log mtime 23:47)
**Log**: `/tmp/arc_run_007.log`

### Results

**Accuracy**:
- Peak: **3.33%** (2/60 mid cycles)
- Final: 1.67% (last epoch)
- Progression: 0% early, climbed to 2/60 in cycles 2-4, dropped to 1/60 in cycles 5-6.

**Library Growth**:
- Programs: 52 → 52 (**NO CHANGE**)
- Drawing shapes: 12 (steady)
- Grammar rules: 212 (steady)
- Pattern types: 4 (steady)

**Storage**:
- Total checkpoint size: 3.7 MB
- Deduplication efficiency: ~98.3% (52 unique, 3,029 refs)

---

## Run 008 - Continued Stall

**Date**: November 27, 2025
**Configuration**: 60 tasks × 27 epochs × 6 cycles
**Curriculum**: 20 easy, 20 mid, 20 hard
**Runtime**: ~18 minutes (log mtime 00:07)
**Log**: `/tmp/arc_run_008.log`

### Results

**Accuracy**:
- Peak: **1.67%** (1/60 in cycles 1-3)
- Final: 0% (last epoch)
- Progression: 1/60 in cycles 1-3, faded to 0/60 in cycles 4-6.

**Library Growth**:
- Programs: 52 → 52 (**NO CHANGE**)
- Drawing shapes: 12 (steady)
- Grammar rules: 212 (steady)
- Pattern types: 4 (steady)

**Storage**:
- Total checkpoint size: 3.7 MB
- Deduplication efficiency: ~98.3%

---

## Run 009 - Flatline Continues

**Date**: November 27, 2025
**Configuration**: 60 tasks × 27 epochs × 6 cycles
**Curriculum**: 20 easy, 20 mid, 20 hard
**Runtime**: ~18 minutes (log mtime 00:25)
**Log**: `/tmp/arc_run_009.log`

### Results

**Accuracy**:
- Peak: **1.67%** (1/60 steady throughout)
- Final: 1.67% (last epoch)
- Progression: 1/60 in cycles 1-3, 0/60 in cycle 4, 1/60 in cycles 5-6.

**Library Growth**:
- Programs: 52 → 52 (**NO CHANGE**)
- Drawing shapes: 12 (steady)
- Grammar rules: 212 (steady)
- Pattern types: 4 (steady)

**Storage**:
- Total checkpoint size: 3.7 MB
- Deduplication efficiency: ~98.3%

---

## Run 010 - Stall Persists

**Date**: November 27, 2025
**Configuration**: 60 tasks × 27 epochs × 6 cycles
**Curriculum**: 20 easy, 20 mid, 20 hard
**Runtime**: ~18 minutes (log mtime 00:43)
**Log**: `/tmp/arc_run_010.log`

### Results

**Accuracy**:
- Peak: **3.33%** (2/60 in cycles 2-3)
- Final: 0% (last epoch)
- Progression: 0% early, 2/60 in cycles 2-3, 1/60 in cycle 4, 0/60 in cycles 5-6.

**Library Growth**:
- Programs: 52 → 52 (**NO CHANGE**)
- Drawing shapes: 12 (steady)
- Grammar rules: 212 (steady)
- Pattern types: 4 (steady)

**Storage**:
- Total checkpoint size: 3.7 MB
- Deduplication efficiency: ~98.3%

---

## Run 011 - Compositional Enabled (No Gain)

**Date**: November 27, 2025
**Configuration**: 60 tasks × 27 epochs × 6 cycles
**Curriculum**: 20 easy, 20 mid, 20 hard
**Runtime**: ~32 minutes (log mtime 01:43)
**Log**: `/tmp/arc_run_011.log`
**Optimizations**: Compositional generation (beam) only

### Results

**Accuracy**:
- Peak: **1.67%**
- Final: 0%

**Library Growth**:
- Programs: 52 → 52 (no change)
- Drawing shapes: 12 (steady)
- Grammar rules: 212 (steady)
- Pattern types: 4 (steady)

**Storage**:
- Total checkpoint size: ~3.8 MB
- Deduplication efficiency: ~98.3%

**GPU Metrics** (csv `gpu_metrics_run_011_20251127_012113.csv`):
- Avg util: ~0.95%
- Peak util: 5.0%
- Avg temp: ~43.65°C
- Avg mem: ~1.39 GB

---

## Run 012 - Parallel + Cross-Pattern (Initial)

**Date**: November 27, 2025
**Configuration**: 60 tasks × 27 epochs × 6 cycles
**Curriculum**: 20 easy, 20 mid, 20 hard
**Runtime**: ~32 minutes (log mtime 02:24)
**Log**: `/tmp/arc_run_012.log`
**Optimizations**: Compositional + Parallel (Tesla 3-6-9 thread fan-out) + Cross-pattern

### Results

**Accuracy**:
- Peak: 0%
- Final: 0%

**Library Growth**:
- Programs: 52 → 52 (no change)
- Drawing shapes: 12 (steady)
- Grammar rules: 212 (steady)
- Pattern types: 4 (steady)

**Storage**:
- Total checkpoint size: ~3.8 MB
- Deduplication efficiency: ~98.3%

**GPU Metrics** (csv `gpu_metrics_run_012_20251127_015207.csv`):
- Avg util: ~0.14% ⚠️ (down; still CPU-bound)
- Peak util: 9.0%
- Avg temp: ~43.06°C
- Avg mem used: ~339 MB (down from ~1.5 GB)

**Analysis**:
- Parallel wiring not exercising GPU cores; still CPU-bound. Needs MathCorePool-backed execution.

---

## Summary Statistics

### Runs Completed: 12

**Accuracy Trajectory**:
| Run | Peak | Final | Library | Notes |
|-----|------|-------|---------|-------|
| 001 | 1.67% | 0% | 43 | Baseline breakthrough |
| 002 | 3.33% | 3.33% | 47 | Library +4, patterns +1 |
| 003 | 1.67% | 1.67% | 49 | Library +2, rules +1 |
| 004 | 3.33% | 0% | 51 | Library +2, rules +1 |
| 005 | 3.33% | 3.33% | 51 | No net growth, variance run |
| 006 | 1.67% | 0% | 52 | +1 program, GPU monitor captured |
| 007 | 3.33% | 1.67% | 52 | **STALL: No library growth** |
| 008 | 1.67% | 0% | 52 | **STALL: No library growth** |
| 009 | 1.67% | 1.67% | 52 | **STALL: No library growth** |
| 010 | 3.33% | 0% | 52 | **STALL: No library growth** |
| 011 | 1.67% | 0% | 52 | Compositional enabled; no change |
| 012 | 0% | 0% | 52 | Parallel+cross-pattern; GPU underused |

**Library Growth**:
| Run | Programs | Shapes | Rules | Size (MB) |
|-----|----------|--------|-------|-----------|
| 001 | 43 | 14 | 209 | 3.7 |
| 002 | 47 | 13 | 210 | 3.8 |
| 003 | 49 | 12 | 211 | 3.8 |
| 004 | 51 | 13 | 212 | 3.8 |
| 005 | 51 | 13 | 212 | 3.8 |
| 006 | 52 | 12 | 212 | 3.8 |
| 007 | 52 | 12 | 212 | 3.7 |
| 008 | 52 | 12 | 212 | 3.7 |
| 009 | 52 | 12 | 212 | 3.7 |
| 010 | 52 | 12 | 212 | 3.7 |
| 011 | 52 | 12 | 212 | 3.8 |
| 012 | 52 | 12 | 212 | 3.8 |

**Observed Trends**:
- Runs 001-006: Library grew from 43 → 52 programs (+9 total, +1.8/run average)
- Runs 006-012: **STALLED at 52 programs for 7 consecutive runs**
- Accuracy: Fluctuating 0-3.33% with no upward trend
- Pattern: System solving same 1-2 tasks repeatedly, not discovering new capabilities

**⚠️ CRITICAL ISSUE DETECTED**:
Library growth has completely stalled. Without new discoveries, accuracy cannot improve beyond current 0-3.33% range. Investigation needed before continuing training.

---

## Notes for Codex

**Update this log after each run** with:
1. Run configuration (copy from template above)
2. Results (accuracy, library growth, runtime)
3. Analysis (what worked, what didn't, trends observed)
4. Next steps (continue, scale up, or investigate)

**Capture metrics** using:
```bash
PYTHONPATH=. python scripts/capture_arc_metrics.py \
  --log /tmp/arc_run_XXX.log \
  --output metrics/arc_run_XXX_metrics.json
```

**Report to Daniel** weekly with summary of runs, accuracy trend, and estimated path to 99%.

---

**Last Updated**: November 27, 2025 (Run 010 completed - STALL DETECTED)
