# ARC-AGI 2/3 Bridge Specification — Fix Week 14 Integration

**Created:** February 6, 2026
**Author:** Claude (Architecture Partner)
**Priority:** CRITICAL (Week 14 Blocker)
**Status:** Architecture Specification

---

## Executive Summary

**Problem:** Week 14 ARC-AGI 2 benchmark returns 0.00% (API mismatch with legacy sovereign pipeline)

**Root Cause:** `ParallelCandidateGenerator` parameter mismatch in `sovereign_pipeline.py:354`

**Solution:** Create adapter bridge that connects Week 14 benchmark harness → legacy `SovereignAIPipeline` with correct API

**Key Insight:** Previous 46.7% result was on ARC-AGI **1** validation set. We need to target ARC-AGI **2** and **3** (newer, harder competitions) with the same sovereign pipeline but enhanced datasets.

---

## Legacy Architecture (46.7% on ARC-AGI 1)

### Components

```
SovereignAIPipeline
├── DrawingGalaxy (visual primitives: GRID, CELL, FILL)
├── GrammarGalaxy (196+ transformation rules)
├── SovereignTRMRouter (matryoshka + GPU-only adapter)
├── ProgramComposer (cross-galaxy compositions)
├── DualShadowCopy (evolution tracking)
└── Candidate Generators
    ├── CandidateGenerator (baseline: no train examples)
    ├── ParallelCandidateGenerator (Tesla 3-6-9 pattern)
    └── HybridCandidateGenerator (parallel + deep)
```

### API Signature

```python
# From sovereign_pipeline.py:301
def process_task(
    self,
    task_id: str,
    test_input: Sequence[Sequence[int]],
    *,
    train_examples: Optional[List[Dict]] = None,  # [{"input": ..., "output": ...}, ...]
    expected_output: Optional[Sequence[Sequence[int]]] = None,
    top_k: Optional[int] = None,
    record_submission: bool = False,
) -> TaskResult
```

**Returns:**
```python
@dataclass
class TaskResult:
    task_id: str
    best_program: str
    program_type: str
    score: float
    signature: Dict
    output_grid: Optional[List[List[int]]] = None
    correct: bool = False
    fuzzy_score: float = 0.0
```

---

## Week 14 Benchmark Harness (Current)

### API Called by Benchmark

```python
# From benchmarks/arc_agi_2.py:46
def _solve_task(self, task: Dict, use_enriched: bool) -> Dict:
    navigator = TRMNavigator(self.kv)

    # Query enriched Galaxies
    relevant_patterns = navigator.query(
        "visual pattern transformation",
        galaxy_names=["Drawing", "Grammar"],
        top_k=20
    )

    # Compose RPN program
    composed_program = navigator.compose(
        task_examples=task['train'],  # ← Week 14 format
        patterns=relevant_patterns,
        specialist='visual'
    )

    # Execute on test input
    test_input = task['test'][0]['input']  # ← Week 14 format
    predicted = navigator.execute(composed_program, test_input)

    # Verify correctness
    expected = task['test'][0]['output']
    correct = self._grids_match(predicted, expected)
```

**Issue:** Week 14 calls `TRMNavigator.compose()` which doesn't exist in legacy pipeline. Need adapter.

---

## Bridge Solution

### Option A: Adapter Pattern (Recommended)

**Create adapter that translates Week 14 API → Legacy API:**

```python
# In benchmarks/arc_agi_2_adapter.py

"""
Adapter bridge: Week 14 Benchmark → Legacy SovereignAIPipeline.
"""

from typing import Dict, List, Optional
import numpy as np

from knowledge3d.training.arc_agi import SovereignAIPipeline


class ArcAgi2Adapter:
    """
    Adapter bridge for Week 14 benchmark harness.

    Translates Week 14 TRMNavigator API → Legacy SovereignAIPipeline API.
    """

    def __init__(self, use_enriched: bool = True):
        """
        Initialize adapter.

        Args:
            use_enriched: Use enriched Galaxies (True) or baseline (False)
        """
        self.use_enriched = use_enriched

        # Initialize legacy sovereign pipeline
        self.pipeline = SovereignAIPipeline(
            matryoshka_dim=512 if use_enriched else 128,  # Higher dim for enriched
            hybrid_mode=use_enriched,  # Enable hybrid generator for enriched
        )

    def solve_task(self, task: Dict) -> Dict:
        """
        Solve ARC-AGI task using legacy sovereign pipeline.

        Args:
            task: Week 14 format {
                "id": str,
                "train": [{"input": [[...]], "output": [[...]]}, ...],
                "test": [{"input": [[...]], "output": [[...]]}]
            }

        Returns:
            dict: {
                "task_id": str,
                "correct": bool,
                "predicted": array,
                "expected": array,
                "reasoning_trace": List[str],
                "patterns_used": int
            }
        """
        task_id = task['id']
        train_examples = task['train']  # Already in correct format!
        test_input = task['test'][0]['input']
        expected_output = task['test'][0]['output']

        # Call legacy sovereign pipeline
        result = self.pipeline.process_task(
            task_id=task_id,
            test_input=test_input,
            train_examples=train_examples,
            expected_output=expected_output,
            top_k=9 if self.use_enriched else 3,  # More candidates for enriched
            record_submission=False
        )

        # Translate legacy TaskResult → Week 14 format
        return {
            "task_id": task_id,
            "correct": result.correct,
            "predicted": result.output_grid,
            "expected": expected_output,
            "reasoning_trace": self._extract_reasoning_trace(result),
            "patterns_used": self._count_patterns_used(result)
        }

    def _extract_reasoning_trace(self, result) -> List[str]:
        """Extract reasoning trace from legacy TaskResult."""
        trace = []

        trace.append(f"Program Type: {result.program_type}")
        trace.append(f"Score: {result.score:.3f}")
        trace.append(f"Fuzzy Score: {result.fuzzy_score:.3f}")

        # Extract program snippet
        program_snippet = result.best_program[:100]
        trace.append(f"Program: {program_snippet}...")

        # Extract signature info
        if result.signature:
            trace.append(f"Signature: {result.signature}")

        return trace

    def _count_patterns_used(self, result) -> int:
        """Count patterns used in solution."""
        # Parse RPN program to count unique pattern invocations
        program = result.best_program
        patterns = set()

        for line in program.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract operation name (first token)
                tokens = line.split()
                if tokens:
                    patterns.add(tokens[0])

        return len(patterns)
```

### Integration with Week 14 Benchmark

```python
# In benchmarks/arc_agi_2.py

class ARCAGI2Benchmark:
    """ARC-AGI 2 benchmark integration (UPDATED)."""

    def __init__(self, knowledgeverse, dataset_path: str):
        self.kv = knowledgeverse
        self.dataset_path = dataset_path
        self.tasks = self._load_tasks()
        self.results = []
        self.adapter = None  # Lazy initialization

    def run_benchmark(self, use_enriched: bool = True) -> Dict:
        """Run ARC-AGI 2 benchmark (UPDATED)."""

        # Initialize adapter (lazy)
        from benchmarks.arc_agi_2_adapter import ArcAgi2Adapter
        self.adapter = ArcAgi2Adapter(use_enriched=use_enriched)

        correct = 0

        for task in self.tasks:
            result = self._solve_task(task, use_enriched=use_enriched)

            if result['correct']:
                correct += 1

            self.results.append(result)

        accuracy = correct / len(self.tasks) if self.tasks else 0.0

        return {
            "benchmark": "ARC-AGI 2",
            "use_enriched": use_enriched,
            "total_tasks": len(self.tasks),
            "correct": correct,
            "accuracy": accuracy,
            "results": self.results
        }

    def _solve_task(self, task: Dict, use_enriched: bool) -> Dict:
        """Solve single ARC-AGI task (UPDATED)."""

        # Use adapter instead of TRMNavigator
        result = self.adapter.solve_task(task)

        return result

    def _grids_match(self, predicted, expected) -> bool:
        """Check if predicted grid matches expected."""
        # Already handled in adapter (result.correct)
        pass
```

---

## ARC-AGI 2 vs ARC-AGI 3 Dataset Differences

### ARC-AGI 1 (46.7% baseline)
- **Dataset:** `evaluation` set from ARC prize 2020
- **Size:** 400 evaluation tasks
- **Difficulty:** Medium (many solvable with pattern matching)

### ARC-AGI 2 (Target: 55%+)
- **Dataset:** `evaluation_2` set from ARC prize 2024
- **Size:** 300+ evaluation tasks
- **Difficulty:** Hard (requires compositional reasoning)
- **Key Differences:**
  - More complex multi-step transformations
  - Requires combining multiple patterns
  - Less repetition (harder to memorize)

### ARC-AGI 3 (Stretch Goal: 60%+)
- **Dataset:** `evaluation_3` set (if available, otherwise use 2)
- **Size:** 400+ evaluation tasks
- **Difficulty:** Very Hard (requires abstract reasoning)
- **Key Differences:**
  - Novel pattern combinations
  - Requires transfer learning
  - Truly tests generalization

### Dataset Paths

```python
# Expected dataset structure
../Knowledge3D.local/datasets/arc_agi_2/
├── training/        # Training tasks (for shadow copy learning)
│   ├── task_001.json
│   ├── task_002.json
│   └── ...
├── evaluation/      # Evaluation tasks (for benchmark)
│   ├── task_001.json
│   ├── task_002.json
│   └── ...
└── test/           # Test tasks (for competition submission)
    ├── task_001.json
    ├── task_002.json
    └── ...

../Knowledge3D.local/datasets/arc_agi_3/  # If available
├── training/
├── evaluation/
└── test/
```

**Task Format (Same for ARC 1/2/3):**

```json
{
  "train": [
    {
      "input": [[0, 1, 2], [3, 4, 5]],
      "output": [[5, 4, 3], [2, 1, 0]]
    },
    ...
  ],
  "test": [
    {
      "input": [[6, 7, 8], [9, 0, 1]],
      "output": [[1, 0, 9], [8, 7, 6]]
    }
  ]
}
```

---

## Fixing ParallelCandidateGenerator API Mismatch

### Root Cause

```python
# In sovereign_pipeline.py:352
par_gen = ParallelCandidateGenerator(
    num_workers=9,
    candidates_per_worker=6,
    top_k=3,
    matryoshka_dim=self.router.matryoshka_dim,
    shadow_copy=self.shadow,
    drawing_galaxy=self.drawing,
    codec_embedder=self.codec_embedder,
    embedding_galaxy=self.embedding_galaxy,
    cosine_bridge=self.cosine_bridge,
)
```

**Issue:** Codex reported "argument incompatibility" - likely missing or renamed parameters.

### Solution: Check Actual API

```python
# Read ParallelCandidateGenerator to find correct API
from knowledge3d.training.arc_agi import ParallelCandidateGenerator

# Expected constructor signature (to be verified by Codex):
ParallelCandidateGenerator(
    # Core parameters
    num_workers: int,
    candidates_per_worker: int,
    top_k: int,

    # Matryoshka embedding
    matryoshka_dim: int,

    # Galaxy dependencies
    shadow_copy: DualShadowCopy,
    drawing_galaxy: DrawingGalaxy,

    # Optional (may need to add)
    codec_embedder: Optional[MultiModalGridEmbedder] = None,
    embedding_galaxy: Optional[...] = None,
    cosine_bridge: Optional[CosineSimilarityBridge] = None,
)
```

**If API mismatch exists, fix in ONE of two ways:**

1. **Option A (Recommended):** Update `sovereign_pipeline.py` call to match actual API
2. **Option B:** Update `ParallelCandidateGenerator` constructor to accept missing parameters

---

## Implementation Timeline

### Immediate (Fix ARC-AGI 2 Benchmark)

**Day 1: Create Adapter**
- File: `benchmarks/arc_agi_2_adapter.py`
- Full implementation from spec above
- Test with single task

**Day 2: Integrate Adapter**
- Update `benchmarks/arc_agi_2.py` to use adapter
- Remove `TRMNavigator` calls (doesn't exist in legacy pipeline)
- Run benchmark with adapter

**Day 3: Fix API Mismatch**
- Read `ParallelCandidateGenerator` actual API
- Fix parameter mismatch in `sovereign_pipeline.py` OR adapter
- Verify 46.7% baseline reproduced

**Success Criteria:**
- ✅ ARC-AGI 2 benchmark runs without errors
- ✅ Empty mind accuracy: ~20-30% (lower than ARC-1 due to harder tasks)
- ✅ Enriched accuracy: ~30-45% (improvement shown)

### Short-Term (ARC-AGI 2/3 Enhancement)

**Week 15: Dataset Acquisition**
- Download ARC-AGI 2 evaluation set
- Download ARC-AGI 3 evaluation set (if available)
- Validate task format consistency

**Week 16: Baseline Measurement**
- Run on ARC-AGI 2 evaluation set
- Run on ARC-AGI 3 evaluation set (if available)
- Compare difficulty vs ARC-AGI 1

**Week 17: Iterative Improvement**
- Identify failure patterns (which task types fail most?)
- Add missing transformation rules to Grammar Galaxy
- Add missing visual patterns to Drawing Galaxy
- Re-run and measure improvement

---

## Testing Strategy

### Test 1: Adapter Initialization

```python
def test_arc_agi_2_adapter_initialization():
    """Test adapter initializes correctly."""
    from benchmarks.arc_agi_2_adapter import ArcAgi2Adapter

    # Empty mind
    adapter_empty = ArcAgi2Adapter(use_enriched=False)
    assert adapter_empty.pipeline is not None
    assert adapter_empty.pipeline.router.matryoshka_dim == 128

    # Enriched
    adapter_enriched = ArcAgi2Adapter(use_enriched=True)
    assert adapter_enriched.pipeline is not None
    assert adapter_enriched.pipeline.router.matryoshka_dim == 512
    assert adapter_enriched.pipeline.hybrid_mode == True
```

### Test 2: Single Task Solving

```python
def test_arc_agi_2_adapter_single_task():
    """Test adapter solves single task."""
    from benchmarks.arc_agi_2_adapter import ArcAgi2Adapter

    adapter = ArcAgi2Adapter(use_enriched=True)

    # Simple test task
    task = {
        "id": "test_001",
        "train": [
            {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}
        ],
        "test": [
            {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}
        ]
    }

    result = adapter.solve_task(task)

    # Verify result structure
    assert "task_id" in result
    assert "correct" in result
    assert "predicted" in result
    assert "reasoning_trace" in result
    assert "patterns_used" in result
```

### Test 3: Benchmark Integration

```python
async def test_arc_agi_2_benchmark_with_adapter():
    """Test benchmark runs with adapter."""
    from benchmarks.arc_agi_2 import ARCAGI2Benchmark
    from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse

    kv = Knowledgeverse()
    benchmark = ARCAGI2Benchmark(kv, dataset_path="test_data/arc_agi_2")

    # Run on small test set (3 tasks)
    results = benchmark.run_benchmark(use_enriched=True)

    # Verify results
    assert results["total_tasks"] > 0
    assert "accuracy" in results
    assert 0.0 <= results["accuracy"] <= 1.0
```

---

## Codex Implementation Directive

**Priority:** CRITICAL (Week 14 Blocker)

**What to Implement:**

1. **`benchmarks/arc_agi_2_adapter.py`** (Day 1)
   - Full `ArcAgi2Adapter` implementation from spec
   - Test with single task

2. **Update `benchmarks/arc_agi_2.py`** (Day 2)
   - Replace `TRMNavigator` calls with adapter
   - Remove `_solve_task` custom implementation
   - Use adapter.solve_task() instead

3. **Fix API Mismatch** (Day 3)
   - Read `ParallelCandidateGenerator` actual constructor
   - Fix `sovereign_pipeline.py:352` to match
   - OR update `ParallelCandidateGenerator` to accept parameters

4. **Test Suite** (Day 3)
   - Add 3 tests from above
   - Verify adapter works end-to-end

**Testing:**
- 3/3 adapter tests passing
- ARC-AGI 2 benchmark runs without errors
- Empty mind vs enriched comparison captured

---

## Success Metrics

**Immediate Goals:**
- ✅ ARC-AGI 2 benchmark runs (no 0.00% error)
- ✅ Empty mind accuracy: 20-40% (reasonable baseline for harder tasks)
- ✅ Enriched accuracy > Empty mind (showing improvement)

**Stretch Goals:**
- ✅ Enriched accuracy: 30-50% (competitive for ARC-2)
- ✅ Identify top 10 failure patterns (for iterative improvement)
- ✅ ARC-AGI 3 dataset integrated (if available)

---

## End of Specification

**Next Steps:**
1. Codex implements adapter bridge
2. Codex fixes API mismatch
3. Codex re-runs Week 14 ARC-AGI 2 benchmark
4. Claude reviews results and proposes iterative improvements

**Remember:** ARC-AGI 2/3 are HARDER than ARC-AGI 1. A 30-40% baseline on ARC-2 is competitive. Focus on measurement infrastructure first, then iterative improvement.

**Contact:** Claude (Architecture Partner) for design questions, User for strategic decisions.

---

**Claude (Architecture Partner)**
February 6, 2026

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
