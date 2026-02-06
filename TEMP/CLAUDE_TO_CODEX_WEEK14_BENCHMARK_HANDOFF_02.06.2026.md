# Claude → Codex: Week 14 Benchmark Integration Handoff

**Date:** February 6, 2026
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Partner)
**Priority:** CRITICAL (Prize-Winning Validation)
**Context:** Post-Week 13 (38/38 tests, hardened pipeline, enriched Galaxies)

---

## Executive Summary

**Goal:** Measure performance with enriched knowledge base to validate Phase 1B/Week 13 investment.

**What You'll Build:**
1. **ARC-AGI 2 Benchmark** — Visual reasoning (target: 46.7% → 55%+)
2. **Math Competitions Benchmark** — AMC/AIME/IMO (target: 0% → 30%+)
3. **Last Humanity Exam Benchmark** — Multi-domain reasoning (target: 0% → 40%+)

**Key Insight:** We've spent Week 11-13 preparing and enriching Galaxies:
- Grammar Galaxy: 500+ transformation rules
- Math Galaxy: 1000+ symbols with RPN templates
- Reality Galaxy: 200+ physics procedures
- Drawing Galaxy: 300+ geometric patterns

Now we **measure** if this enrichment actually improves benchmark performance.

---

## Phase Context

**Phase 1A (Weeks 1-10):** MVP Core + Hardening ✅
**Phase 1B (Weeks 11-12):** Knowledge Preparation + Ingestion ✅
**Phase 1C (Week 13):** Local LLM Hardening + Stargate Crystallization ✅
**Phase 1D (Week 14):** **Benchmark Integration** ← YOU ARE HERE

---

## Specification Document

I've written a comprehensive specification for you:

**[TEMP/WEEK14_BENCHMARK_INTEGRATION_SPECIFICATION_02.06.2026.md](WEEK14_BENCHMARK_INTEGRATION_SPECIFICATION_02.06.2026.md)**

This document contains:
- ✅ Complete Python implementations for all 3 benchmarks
- ✅ Comparison scripts (empty mind vs enriched)
- ✅ Unified runner (all benchmarks in sequence)
- ✅ tmux orchestration script
- ✅ Test specifications
- ✅ Success metrics and targets

**READ IT COMPLETELY** before starting implementation.

---

## Week 14 Implementation Plan

### Day 1-2: Benchmark Infrastructure

**Goal:** Create benchmark classes that can load datasets and run evaluations.

**Files to Create:**

1. **`benchmarks/__init__.py`**
   ```python
   """Knowledge3D Benchmark Suite."""

   from .arc_agi_2 import ARCAGI2Benchmark
   from .math_competitions import MathCompetitionBenchmark
   from .last_humanity_exam import LastHumanityExamBenchmark

   __all__ = [
       'ARCAGI2Benchmark',
       'MathCompetitionBenchmark',
       'LastHumanityExamBenchmark'
   ]
   ```

2. **`benchmarks/arc_agi_2.py`**
   - Full implementation from spec (lines 34-156)
   - Load ARC-AGI 2 tasks from JSON
   - Solve tasks using TRM navigator with enriched Galaxies
   - Compare predictions with expected outputs

3. **`benchmarks/math_competitions.py`**
   - Full implementation from spec (lines 176-333)
   - Load AMC/AIME/IMO problems from JSON
   - Solve using Math + Grammar Galaxies
   - Handle numeric answer comparison

4. **`benchmarks/last_humanity_exam.py`**
   - Full implementation from spec (lines 353-508)
   - Load multi-domain questions
   - Route to appropriate specialist based on domain
   - Cross-domain reasoning with multiple Galaxies

5. **`tests/test_benchmarks.py`**
   - 3 loading tests from spec (lines 530-556)
   - Test dataset loading for all benchmarks
   - Verify data format consistency

**Day 1-2 Success Criteria:**
- ✅ All 4 files created and pass linting
- ✅ 3/3 loading tests passing
- ✅ Datasets load without errors
- ✅ Benchmark classes implement consistent interface

---

### Day 3: Individual Benchmark Runs

**Goal:** Run each benchmark individually, capture empty mind vs enriched comparison.

**Files to Create:**

1. **`scripts/benchmark_arc_agi_comparison.py`**
   - Full implementation from spec (lines 158-226)
   - Run ARC-AGI 2 twice (empty mind + enriched)
   - Save results to JSON
   - Print comparison summary

2. **`scripts/benchmark_math_comparison.py`**
   - Full implementation from spec (lines 335-383)
   - Run math competitions twice
   - Breakdown by competition level (AMC, AIME, IMO)
   - Save results to JSON

3. **`scripts/benchmark_lhe_comparison.py`**
   - Full implementation from spec (lines 510-555)
   - Run Last Humanity Exam twice
   - Breakdown by domain (math, physics, logic, multi)
   - Save results to JSON

**Execution Commands:**

```bash
# In tmux window 1 (arc_agi)
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/benchmark_arc_agi_comparison.py

# In tmux window 2 (math)
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/benchmark_math_comparison.py

# In tmux window 3 (lhe)
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/benchmark_lhe_comparison.py
```

**Expected Output Format:**

```
ARC-AGI 2:
  Empty Mind:  46.7% (140/300)
  Enriched:    52.3% (157/300)
  Improvement: +5.6%

Math Competitions:
  Empty Mind:  5.2% (3/58)
  Enriched:    28.7% (17/58)
  Improvement: +23.5%

Last Humanity Exam:
  Empty Mind:  12.3% (15/122)
  Enriched:    38.5% (47/122)
  Improvement: +26.2%
```

**Day 3 Success Criteria:**
- ✅ All 3 benchmarks run without errors
- ✅ Results saved to `../Knowledge3D.local/results/week14/`
- ✅ Empty mind vs enriched comparison captured
- ✅ Improvement metrics calculated

---

### Day 4: Unified Runner

**Goal:** Create single script that runs all benchmarks in sequence with unified reporting.

**File to Create:**

**`scripts/run_all_benchmarks.py`**
- Full implementation from spec (lines 557-689)
- Run all 3 benchmarks sequentially
- Generate unified JSON report
- Print final summary with targets vs actual

**Execution Command:**

```bash
# In tmux window 4 (all_benchmarks)
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/run_all_benchmarks.py
```

**Expected Unified Report:**

```json
{
  "timestamp": "2026-02-06T14:30:00",
  "benchmarks": {
    "arc_agi_2": {
      "empty_mind": {"accuracy": 0.467, ...},
      "enriched": {"accuracy": 0.523, ...},
      "improvement": 0.056
    },
    "math_competitions": {
      "empty_mind": {"overall_accuracy": 0.052, ...},
      "enriched": {"overall_accuracy": 0.287, ...},
      "improvement": 0.235
    },
    "last_humanity_exam": {
      "empty_mind": {"accuracy": 0.123, ...},
      "enriched": {"accuracy": 0.385, ...},
      "improvement": 0.262
    }
  }
}
```

**Day 4 Success Criteria:**
- ✅ Unified runner completes all 3 benchmarks
- ✅ Unified JSON report generated
- ✅ Targets vs actual printed clearly
- ✅ User can see which targets were met

---

### Day 5: Analysis & Completion Report

**Goal:** Analyze results, identify gaps, propose iterative improvements.

**Tasks:**

1. **Analyze Results:**
   - Which benchmark performed best/worst?
   - Which task types fail most frequently?
   - Where does enriched knowledge help most?
   - Where does it not help enough?

2. **Identify Gaps:**
   - Missing knowledge in Galaxies (what patterns/symbols needed?)
   - TRM navigation issues (query not finding relevant entries?)
   - Composition issues (can't combine patterns correctly?)
   - Execution issues (RPN programs fail?)

3. **Propose Improvements:**
   - Additional knowledge to ingest (which PDFs, datasets?)
   - Grammar rules to add (which transformation patterns missing?)
   - TRM training adjustments (which specialist adapters need work?)

**Deliverable:**

**`TEMP/CODEX_WEEK14_BENCHMARK_COMPLETION_REPORT_02.XX.2026.md`**

**Sections:**
1. **Executive Summary** (targets met, overall findings)
2. **Benchmark Results:**
   - ARC-AGI 2 detailed analysis
   - Math Competitions detailed analysis
   - Last Humanity Exam detailed analysis
3. **Galaxy Enrichment Impact:**
   - Grammar Galaxy: How many rules used? Which most effective?
   - Math Galaxy: How many symbols used? Which gaps identified?
   - Reality Galaxy: How many procedures used?
4. **Gap Analysis:**
   - Missing knowledge (specific examples)
   - Navigation issues (TRM not finding relevant entries)
   - Composition issues (can't combine patterns)
5. **Iterative Improvement Proposals:**
   - Priority 1: Quick wins (high impact, low effort)
   - Priority 2: Medium-term (significant effort, high impact)
   - Priority 3: Long-term (research directions)
6. **Lessons Learned:**
   - What worked well?
   - What didn't work as expected?
   - What surprised us?

**Day 5 Success Criteria:**
- ✅ Comprehensive analysis completed
- ✅ Specific gaps identified with examples
- ✅ Actionable improvement proposals
- ✅ Completion report written

---

## tmux Orchestration

**Create orchestration script:**

```bash
#!/bin/bash
# scripts/week14_benchmark_tmux.sh

tmux new-session -d -s k3d_week14

# Window 0: GPU Monitor
tmux rename-window -t k3d_week14:0 'gpu_monitor'
tmux send-keys -t k3d_week14:0 'watch -n 1 nvidia-smi' C-m

# Window 1: ARC-AGI 2
tmux new-window -t k3d_week14:1 -n 'arc_agi'
tmux send-keys -t k3d_week14:1 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D' C-m

# Window 2: Math Competitions
tmux new-window -t k3d_week14:2 -n 'math'
tmux send-keys -t k3d_week14:2 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D' C-m

# Window 3: Last Humanity Exam
tmux new-window -t k3d_week14:3 -n 'lhe'
tmux send-keys -t k3d_week14:3 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D' C-m

# Window 4: Unified Runner
tmux new-window -t k3d_week14:4 -n 'all_benchmarks'
tmux send-keys -t k3d_week14:4 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D' C-m

tmux attach-session -t k3d_week14
```

**Usage:**

```bash
# Start tmux session
chmod +x scripts/week14_benchmark_tmux.sh
./scripts/week14_benchmark_tmux.sh

# User watches Window 0 (GPU monitor) while you run benchmarks in Windows 1-4
# When GPU settles, user knows benchmark is complete
```

---

## Testing Strategy

### Regression Testing (Before Starting)

```bash
# Ensure Week 13 tests still pass
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. pytest tests/ -v

# Expected: 38/38 tests passing
# If ANY test fails, STOP and fix before proceeding
```

### Progressive Testing (During Implementation)

**After Day 1-2:**
```bash
pytest tests/test_benchmarks.py -v
# Expected: 3/3 loading tests passing
```

**After Day 3:**
```bash
# Verify results files exist
ls -lh ../Knowledge3D.local/results/week14/
# Should see: arc_agi_2_empty_mind.json, arc_agi_2_enriched.json, etc.
```

**After Day 4:**
```bash
# Verify unified report
cat ../Knowledge3D.local/results/week14/week14_all_benchmarks.json | jq '.benchmarks | keys'
# Should see: ["arc_agi_2", "last_humanity_exam", "math_competitions"]
```

---

## Success Metrics

### Performance Targets

**Primary Goal:** Establish baseline measurement infrastructure

**Stretch Targets:**
1. **ARC-AGI 2:** 55%+ enriched accuracy (from 46.7% empty mind)
2. **Math Competitions:** 30%+ enriched accuracy (from ~5% empty mind)
3. **Last Humanity Exam:** 40%+ enriched accuracy (from ~10% empty mind)

**Minimum Acceptable:**
- Enriched accuracy > Empty mind accuracy (all benchmarks)
- Average improvement: +5% or more across all benchmarks
- Infrastructure stable and reproducible

### Code Quality

4. **Benchmark Classes:**
   - ✅ All implement consistent interface
   - ✅ Results saved in unified JSON format
   - ✅ Reasoning traces captured for debugging

5. **Reproducibility:**
   - ✅ Same datasets produce same results (deterministic)
   - ✅ Clear separation: empty mind vs enriched
   - ✅ Metrics match manual validation

6. **Documentation:**
   - ✅ Completion report written
   - ✅ Gap analysis with specific examples
   - ✅ Improvement proposals actionable

---

## Dataset Requirements

**You'll need these datasets:**

1. **ARC-AGI 2:**
   - Path: `../Knowledge3D.local/datasets/arc_agi_2/evaluation/`
   - Format: JSON files (one per task)
   - Structure: `{"train": [...], "test": [...]}`

2. **Math Competitions:**
   - Path: `../Knowledge3D.local/datasets/math_competitions/`
   - Files: `amc_problems.json`, `aime_problems.json`, `imo_problems.json`
   - Structure: `[{"id": ..., "problem_text": ..., "answer": ...}, ...]`

3. **Last Humanity Exam:**
   - Path: `../Knowledge3D.local/datasets/last_humanity_exam/`
   - File: `last_humanity_exam.json`
   - Structure: `{"questions": [{"id": ..., "domain": ..., "question_text": ..., "options": [...], "correct_answer": ...}, ...]}`

**If datasets don't exist yet, use placeholder/synthetic data for infrastructure validation.**

---

## Critical Reminders

### 1. This is Baseline Measurement

**NOT a pass/fail test.** The goal is to:
- Establish measurement infrastructure
- Capture baseline performance (empty mind vs enriched)
- Identify gaps for iterative improvement

**Don't worry if targets aren't met immediately.** We'll iterate.

### 2. Enriched Galaxies ARE Real

Week 13 hardening removed placeholders. The Galaxies contain:
- Real RPN programs (not external metadata)
- Real transformation rules (Grammar Galaxy)
- Real symbols with templates (Math Galaxy)
- Real physics procedures (Reality Galaxy)

Use them confidently.

### 3. TRM Navigation is Key

The benchmark results will tell us:
- Can TRM find relevant knowledge in enriched Galaxies?
- Can TRM compose patterns correctly?
- Where does navigation fail?

Capture reasoning traces for debugging.

### 4. Use tmux for Monitoring

User wants to watch GPU graphs to know when benchmarks complete.

Set up tmux properly so user can monitor progress.

---

## Questions for Claude (If Blocked)

If you hit ANY blockers, document them and escalate:

1. **Dataset Format:** If datasets don't match expected format?
2. **TRM Integration:** If TRM navigator interface differs from spec?
3. **Performance Issues:** If benchmarks are too slow (how to optimize)?
4. **Result Interpretation:** If results are ambiguous (how to classify correct/incorrect)?
5. **Gap Analysis:** If unclear how to identify specific knowledge gaps?

---

## End of Handoff

**Priority:** CRITICAL (Week 14, Phase 1D)

**Start here:**
1. Read [TEMP/WEEK14_BENCHMARK_INTEGRATION_SPECIFICATION_02.06.2026.md](WEEK14_BENCHMARK_INTEGRATION_SPECIFICATION_02.06.2026.md) COMPLETELY
2. Create benchmark infrastructure (Day 1-2)
3. Run individual benchmarks (Day 3)
4. Run unified benchmark suite (Day 4)
5. Analyze results and write completion report (Day 5)

**Remember:** This is baseline measurement. Infrastructure is more important than hitting targets immediately. Capture data, identify gaps, propose improvements.

**Contact:** Claude (Architecture Partner) for design questions, User for strategic decisions.

**Let's measure the impact of our enriched knowledge base!** 🚀

---

**Claude (Architecture Partner)**
February 6, 2026

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
