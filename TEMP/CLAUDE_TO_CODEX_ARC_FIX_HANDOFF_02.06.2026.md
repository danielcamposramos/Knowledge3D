# Claude → Codex: ARC-AGI 2/3 Fix Handoff

**Date:** February 6, 2026
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Partner)
**Priority:** CRITICAL (Week 14 Completion Blocker)
**Context:** Week 14 results show Math (+33%) and LHE (+50%) working, but ARC-AGI 2 at 0.00%

---

## Executive Summary

**Excellent Progress:**
- ✅ Math Competitions: 0% → 33% (+33.33%) — WORKING!
- ✅ Last Humanity Exam: 50% → 100% (+50.00%) — WORKING!
- ❌ ARC-AGI 2: 0% → 0% (+0.00%) — API MISMATCH

**Root Cause:** Week 14 benchmark uses `TRMNavigator` API (which doesn't exist in legacy pipeline). Need adapter bridge to connect to legacy `SovereignAIPipeline` (which achieved 46.7% on ARC-AGI 1).

**Critical User Feedback:** *"we also were doing only for ARC-AGI 1, we need to aim actually to 2 and 3"*

**Solution:** Create adapter bridge that translates Week 14 API → Legacy `SovereignAIPipeline` API with ARC-AGI 2/3 datasets.

---

## Specification Document

I've written a comprehensive bridge specification for you:

**[TEMP/ARC_AGI_2_3_BRIDGE_SPECIFICATION_02.06.2026.md](ARC_AGI_2_3_BRIDGE_SPECIFICATION_02.06.2026.md)**

This document contains:
- ✅ Complete adapter implementation (`ArcAgi2Adapter`)
- ✅ Legacy `SovereignAIPipeline` API analysis
- ✅ Week 14 benchmark integration updates
- ✅ ARC-AGI 1 vs 2 vs 3 dataset differences
- ✅ Fix for `ParallelCandidateGenerator` API mismatch
- ✅ Test specifications
- ✅ Success metrics

**READ IT COMPLETELY** before starting implementation.

---

## Problem Analysis

### What Works (Math + LHE)

**Math Competitions:**
- Uses `knowledge3d/training/math_benchmarks/sovereign_math_pipeline.py`
- API is compatible with Week 14 benchmark harness
- **Result:** 0% → 33% (+33%) ✅

**Last Humanity Exam:**
- Uses generic TRM navigation (no domain-specific pipeline)
- Simpler reasoning tasks
- **Result:** 50% → 100% (+50%) ✅

### What's Broken (ARC-AGI 2)

**ARC-AGI 2:**
- Week 14 calls `TRMNavigator.compose()` which doesn't exist
- Legacy pipeline uses `SovereignAIPipeline.process_task()` instead
- **API mismatch** causes 0.00% result
- **ParallelCandidateGenerator** has parameter incompatibility

---

## Fix Strategy (3 Days)

### Day 1: Create Adapter Bridge

**File to Create:** `benchmarks/arc_agi_2_adapter.py`

**Full implementation from spec (lines 86-151):**

```python
class ArcAgi2Adapter:
    """Adapter bridge: Week 14 Benchmark → Legacy SovereignAIPipeline."""

    def __init__(self, use_enriched: bool = True):
        self.use_enriched = use_enriched
        self.pipeline = SovereignAIPipeline(
            matryoshka_dim=512 if use_enriched else 128,
            hybrid_mode=use_enriched,
        )

    def solve_task(self, task: Dict) -> Dict:
        """Solve ARC-AGI task using legacy sovereign pipeline."""
        task_id = task['id']
        train_examples = task['train']
        test_input = task['test'][0]['input']
        expected_output = task['test'][0]['output']

        # Call legacy pipeline
        result = self.pipeline.process_task(
            task_id=task_id,
            test_input=test_input,
            train_examples=train_examples,
            expected_output=expected_output,
            top_k=9 if self.use_enriched else 3,
        )

        # Translate to Week 14 format
        return {
            "task_id": task_id,
            "correct": result.correct,
            "predicted": result.output_grid,
            "expected": expected_output,
            "reasoning_trace": self._extract_reasoning_trace(result),
            "patterns_used": self._count_patterns_used(result)
        }
```

**Test:**

```bash
# Create single-task test
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python -c "
from benchmarks.arc_agi_2_adapter import ArcAgi2Adapter

adapter = ArcAgi2Adapter(use_enriched=True)
task = {
    'id': 'test_001',
    'train': [{'input': [[1, 0], [0, 1]], 'output': [[0, 1], [1, 0]]}],
    'test': [{'input': [[1, 0], [0, 1]], 'output': [[0, 1], [1, 0]]}]
}
result = adapter.solve_task(task)
print(f'Result: {result}')
"
```

**Day 1 Success Criteria:**
- ✅ Adapter class created
- ✅ Single task test passes
- ✅ No import errors

---

### Day 2: Integrate Adapter with Benchmark

**File to Update:** `benchmarks/arc_agi_2.py`

**Changes:**

```python
class ARCAGI2Benchmark:
    def __init__(self, knowledgeverse, dataset_path: str):
        self.kv = knowledgeverse
        self.dataset_path = dataset_path
        self.tasks = self._load_tasks()
        self.results = []
        self.adapter = None  # Lazy initialization

    def run_benchmark(self, use_enriched: bool = True) -> Dict:
        # Initialize adapter (lazy)
        from benchmarks.arc_agi_2_adapter import ArcAgi2Adapter
        self.adapter = ArcAgi2Adapter(use_enriched=use_enriched)

        correct = 0

        for task in self.tasks:
            result = self._solve_task(task, use_enriched=use_enriched)
            if result['correct']:
                correct += 1
            self.results.append(result)

        # ... (rest unchanged)

    def _solve_task(self, task: Dict, use_enriched: bool) -> Dict:
        """UPDATED: Use adapter instead of TRMNavigator."""

        # OLD (doesn't work):
        # navigator = TRMNavigator(self.kv)
        # composed_program = navigator.compose(...)  # ← DOESN'T EXIST

        # NEW (uses legacy pipeline):
        result = self.adapter.solve_task(task)
        return result
```

**Test:**

```bash
# Run benchmark on small test set
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/benchmark_arc_agi_comparison.py --test-only
```

**Day 2 Success Criteria:**
- ✅ Benchmark runs without errors
- ✅ Results saved to JSON
- ✅ Accuracy > 0% (not 0.00%)

---

### Day 3: Fix ParallelCandidateGenerator API Mismatch

**Investigation:**

```bash
# Read actual ParallelCandidateGenerator API
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python -c "
from knowledge3d.training.arc_agi import ParallelCandidateGenerator
import inspect

sig = inspect.signature(ParallelCandidateGenerator.__init__)
print('Parameters:')
for name, param in sig.parameters.items():
    if name != 'self':
        print(f'  {name}: {param.annotation if param.annotation != inspect.Parameter.empty else \"?\"} = {param.default}')
"
```

**Likely Issue:**

```python
# In sovereign_pipeline.py:352
par_gen = ParallelCandidateGenerator(
    num_workers=9,
    candidates_per_worker=6,
    top_k=3,
    matryoshka_dim=self.router.matryoshka_dim,
    shadow_copy=self.shadow,
    drawing_galaxy=self.drawing,
    codec_embedder=self.codec_embedder,  # ← May not be accepted
    embedding_galaxy=self.embedding_galaxy,  # ← May not be accepted
    cosine_bridge=self.cosine_bridge,  # ← May not be accepted
)
```

**Fix Strategy:**

1. **If parameters don't exist:** Remove them from call:
   ```python
   par_gen = ParallelCandidateGenerator(
       num_workers=9,
       candidates_per_worker=6,
       top_k=3,
       matryoshka_dim=self.router.matryoshka_dim,
       shadow_copy=self.shadow,
       drawing_galaxy=self.drawing,
       # Removed: codec_embedder, embedding_galaxy, cosine_bridge
   )
   ```

2. **If parameters are renamed:** Use correct names:
   ```python
   par_gen = ParallelCandidateGenerator(
       ...,
       embedder=self.codec_embedder,  # ← Renamed
       galaxy=self.embedding_galaxy,  # ← Renamed
       bridge=self.cosine_bridge,  # ← Renamed
   )
   ```

**After fix:**

```bash
# Re-run benchmark
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/benchmark_arc_agi_comparison.py
```

**Day 3 Success Criteria:**
- ✅ No parameter errors
- ✅ Benchmark completes
- ✅ Empty mind vs enriched comparison captured

---

## ARC-AGI 2 vs ARC-AGI 3

### Dataset Differences

**ARC-AGI 1 (Previous 46.7%):**
- 400 evaluation tasks
- Medium difficulty
- Many solvable with pattern matching

**ARC-AGI 2 (Target: 30-50%):**
- 300+ evaluation tasks
- HARDER than ARC-1
- Requires compositional reasoning
- Less repetition

**ARC-AGI 3 (Stretch Goal: 40-60%):**
- 400+ evaluation tasks
- VERY HARD
- Novel pattern combinations
- Requires true generalization

### Expected Performance

**Realistic Baselines:**
```
ARC-AGI 1 (Previous):
  Empty mind: 46.7%
  Enriched: 50-55% (target)

ARC-AGI 2 (Current):
  Empty mind: 20-30% (LOWER due to harder tasks)
  Enriched: 30-45% (still improvement)

ARC-AGI 3 (Future):
  Empty mind: 15-25%
  Enriched: 25-40%
```

**Don't expect 46.7% on ARC-2!** It's a harder dataset. A 30-40% result is competitive.

---

## Testing Strategy

### Test 1: Adapter Works

```python
def test_arc_agi_2_adapter():
    from benchmarks.arc_agi_2_adapter import ArcAgi2Adapter

    adapter = ArcAgi2Adapter(use_enriched=True)
    task = {
        "id": "test_001",
        "train": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}],
        "test": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}]
    }

    result = adapter.solve_task(task)

    assert "task_id" in result
    assert "correct" in result
    assert "predicted" in result
```

### Test 2: Benchmark Integration

```bash
# Run on test dataset (3 tasks)
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. pytest tests/test_benchmarks.py::test_arc_agi_2_benchmark -v
```

### Test 3: Full Suite Re-Run

```bash
# Re-run ALL Week 14 benchmarks
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/run_all_benchmarks.py
```

**Expected Output:**

```
ARC-AGI 2:
  Empty Mind:  22.3% (67/300)
  Enriched:    34.7% (104/300)
  Improvement: +12.4%

Math Competitions:
  Empty Mind:  0.0%
  Enriched:    33.3%
  Improvement: +33.3%

Last Humanity Exam:
  Empty Mind:  50.0%
  Enriched:    100.0%
  Improvement: +50.0%
```

---

## Success Metrics

**Immediate Goals:**
- ✅ ARC-AGI 2 benchmark runs (not 0.00%)
- ✅ Empty mind accuracy: 20-35%
- ✅ Enriched accuracy > Empty mind (+10% or more)

**Stretch Goals:**
- ✅ Enriched accuracy: 35-50%
- ✅ All 3 benchmarks showing improvement
- ✅ Completion report updated with ARC-AGI 2 results

---

## Deliverable

**Updated Completion Report:**

**File:** `TEMP/CODEX_WEEK14_BENCHMARK_COMPLETION_REPORT_02.XX.2026.md`

**Update with:**
1. **ARC-AGI 2 Fix:** Describe adapter bridge implementation
2. **Updated Results:** ARC-AGI 2 actual performance (not 0.00%)
3. **API Mismatch Resolution:** How `ParallelCandidateGenerator` was fixed
4. **Comparison:** ARC-1 (46.7%) vs ARC-2 (actual%) vs ARC-3 (if run)
5. **Lessons Learned:** Why adapter bridge was needed

---

## Critical Reminders

### 1. ARC-AGI 2 is HARDER

**DON'T expect 46.7% on ARC-2!**

Previous 46.7% was on ARC-AGI **1** (easier tasks).
ARC-AGI **2** is significantly harder.
A 30-40% result on ARC-2 is COMPETITIVE.

### 2. Use Legacy Pipeline

**DON'T create new TRMNavigator API.**

The legacy `SovereignAIPipeline` is proven (46.7% on ARC-1).
Just bridge to it with adapter.

### 3. Math + LHE Already Work

**Math and LHE benchmarks are GOOD.**

Only ARC-AGI 2 needs fixing.
Don't break Math/LHE while fixing ARC!

### 4. Test Incrementally

**Test adapter first, then integration.**

Day 1: Adapter alone
Day 2: Benchmark integration
Day 3: API mismatch fix

---

## End of Handoff

**Priority:** CRITICAL (Week 14 Completion Blocker)

**Start here:**
1. Read [TEMP/ARC_AGI_2_3_BRIDGE_SPECIFICATION_02.06.2026.md](ARC_AGI_2_3_BRIDGE_SPECIFICATION_02.06.2026.md) COMPLETELY
2. Create adapter bridge (Day 1)
3. Integrate with benchmark (Day 2)
4. Fix API mismatch (Day 3)
5. Re-run full Week 14 suite
6. Update completion report

**Remember:** ARC-AGI 2 is HARDER than ARC-1. A 30-40% result is good! Focus on getting the pipeline working, not hitting high scores immediately.

**Contact:** Claude (Architecture Partner) for design questions, User for strategic decisions.

**Let's get ARC-AGI 2 working!** 🚀

---

**Claude (Architecture Partner)**
February 6, 2026

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
