# CODEX: Deterministic Foundation Curriculum — Phase 0 Training

**Date:** February 8, 2026
**Author:** Claude (Architecture Partner)
**For:** Codex (Implementation Partner)
**Status:** 🔴 CRITICAL STRATEGIC PIVOT
**Priority:** Implement BEFORE returning to ARC-AGI diagnostics

---

## 🎯 Strategic Context: Why This Matters

### The Problem We Discovered

**Your diagnostic findings (ARC-100 run):**
- **oracle_at_all = 0.0** → Correct answer not even generated
- **autonomous_generation precision = 23.17%** → Low quality patterns
- **legacy_pipeline precision = 50%** → Better, but still only half
- **ranking_change_rate = 0.39** → Ranking works (reorders candidates)

**Critical insight:** We're NOT blocked by ranking. We're blocked by **generation quality**.

**User's strategic observation:**
> "We might need to train the model on other tasks that are more deterministic before venturing into 'brute force learning' (what we are doing now)... if we do it like this is easier to teach the model how to think and organize thought (train the TRM part), and then we can go to the BFL"

### The Solution: Curriculum Learning

**Current approach (failing):**
- Jump straight to ARC-AGI (complex visual reasoning)
- TRM hasn't learned HOW to navigate Galaxy effectively
- TRM hasn't learned HOW to compose RPN programs correctly
- Result: oracle_at_all = 0.0 (can't generate correct answers)

**New approach (curriculum learning):**
1. **Phase 0:** Train on deterministic tasks (this file)
   - Simple transformations, arithmetic, pattern completion
   - TRM learns navigation, composition, organization of thought
   - Expected: 80-95% accuracy (these are deterministic!)

2. **Phase 1:** Train on compositional tasks
   - Multi-step operations, conditional logic
   - Expected: 60-80% accuracy

3. **Phase 2:** Return to ARC-AGI (brute force learning)
   - With TRM foundation, generation quality improves
   - Expected: 28% → 40-50% (better navigation = better candidates)

**Bottom line:** Teach TRM to walk before running. Train foundation, THEN tackle hard problems.

---

## 📊 Deterministic Foundation Curriculum

### Task Categories (500 tasks total)

All tasks are:
- ✅ **Deterministic:** Single correct answer
- ✅ **Verifiable:** Exact match validation
- ✅ **Procedural:** RPN programs in Galaxy
- ✅ **Sovereign:** PTX + Galaxy only
- ✅ **Progressive:** Simple → Complex

### Category 1: Geometric Transformations (100 tasks)

**Purpose:** Teach TRM to navigate Drawing + Grammar galaxies for basic visual operations.

**Operations:**
- Rotate 90°, 180°, 270°, 360° (identity)
- Mirror horizontal, vertical
- Translate by offset (dx, dy)
- Scale (2× grid size)

**Task format:**
```python
{
    "category": "geometric_transforms",
    "task_id": "geo_001",
    "input": np.array([[1, 0], [0, 1]]),  # 2×2 grid
    "operation": "ROTATE_90",
    "expected": np.array([[0, 1], [1, 0]]),  # Rotated 90° clockwise
    "rpn_program": "GRID ROTATE_90",  # RPN program in Grammar Galaxy
}
```

**Expected baseline (empty Galaxy):** 10-15% (random guessing)
**Expected after training:** 90-95% (deterministic operations)

**What TRM learns:**
- Navigate Grammar Galaxy to find "ROTATE_90" rule
- Apply procedural transformation to grid
- Validate output matches expected

---

### Category 2: Grid Arithmetic (100 tasks)

**Purpose:** Teach TRM to navigate Math Galaxy for counting, summing, filtering operations.

**Operations:**
- Count cells by color (count red, count blue, count all)
- Sum positions (sum of x coords, sum of y coords)
- Find extrema (max value, min value, max position)
- Filter operations (cells where value > threshold)

**Task format:**
```python
{
    "category": "grid_arithmetic",
    "task_id": "arith_001",
    "input": np.array([[1, 2, 1], [1, 1, 2], [2, 1, 1]]),  # 3×3 grid
    "operation": "COUNT_VALUE",
    "value": 1,  # Count cells with value 1
    "expected": 6,  # Six cells have value 1
    "rpn_program": "GRID 1 EQUAL COUNT",
}
```

**Expected baseline:** 15-20% (guessing)
**Expected after training:** 85-90% (arithmetic is deterministic)

**What TRM learns:**
- Navigate Math Galaxy for COUNT, SUM, MAX operations
- Apply arithmetic to spatial data (grids)
- Return scalar results (not just grid transformations)

---

### Category 3: Pattern Completion (100 tasks)

**Purpose:** Teach TRM to infer missing rules from examples, then apply.

**Task types:**
- Sequence completion (A B _ D → C)
- Symmetry completion (half grid → full grid via mirror)
- Periodic extension (tile pattern → extend by 2×)
- Missing tile (3×3 grid with one missing → infer rule + fill)

**Task format:**
```python
{
    "category": "pattern_completion",
    "task_id": "pattern_001",
    "input": {
        "sequence": [
            np.array([[1, 0]]),  # A
            np.array([[0, 1]]),  # B
            None,                # Missing (what TRM should generate)
            np.array([[1, 0]]),  # D (same as A)
        ]
    },
    "rule": "alternating",  # Alternates between two patterns
    "expected": np.array([[0, 1]]),  # C = B (alternating pattern)
    "rpn_program": "SEQUENCE PATTERN_INFER APPLY",
}
```

**Expected baseline:** 20-25% (some patterns are guessable)
**Expected after training:** 75-85% (pattern inference is learnable)

**What TRM learns:**
- Infer procedural rules from example pairs
- Apply inferred rule to generate missing element
- Validate consistency (does generated pattern fit rule?)

---

### Category 4: Compositional Operations (100 tasks)

**Purpose:** Teach TRM to chain multiple operations into RPN programs.

**Operation chains:**
- ROTATE_90 → MIRROR_H (rotate then mirror)
- COUNT_RED → FILTER_>_5 (count, then filter)
- TRANSLATE_1_0 → ROTATE_180 (translate, then rotate)
- IF_VALUE_1 → MIRROR_V ELSE ROTATE_90 (conditional composition)

**Task format:**
```python
{
    "category": "compositional",
    "task_id": "comp_001",
    "input": np.array([[1, 0], [0, 1]]),
    "operations": ["ROTATE_90", "MIRROR_H"],
    "expected": np.array([[1, 0], [0, 1]]),  # Result of both operations
    "rpn_program": "GRID ROTATE_90 MIRROR_H",
}
```

**Expected baseline:** 10-15% (compositional reasoning is hard)
**Expected after training:** 70-80% (TRM learns to chain operations)

**What TRM learns:**
- Compose multiple Galaxy queries into single RPN program
- Chain operations left→right (RPN execution order)
- Validate intermediate results (each step correct)

---

### Category 5: Symbolic RPN Evaluation (100 tasks)

**Purpose:** Teach TRM to execute RPN programs directly (procedural evaluation).

**RPN operations:**
- Arithmetic: "2 3 ADD" → 5, "10 3 DIV" → 3
- Stack operations: "5 DUP" → [5, 5], "3 4 SWAP" → [4, 3]
- Grid operations: "GRID 90 ROTATE" → rotated grid
- Conditional: "VALUE 5 GT IF MIRROR_H ELSE ROTATE_90" → conditional transform

**Task format:**
```python
{
    "category": "symbolic_rpn",
    "task_id": "rpn_001",
    "rpn_program": "2 3 ADD",
    "expected": 5,
}
```

**Expected baseline:** 25-30% (simple RPN is learnable even without training)
**Expected after training:** 80-90% (RPN execution is deterministic)

**What TRM learns:**
- Execute RPN programs from Grammar Galaxy
- Understand stack-based evaluation
- Validate program correctness (syntax + semantics)

---

## 🏗️ Implementation Architecture

### File Structure

```
benchmarks/
  deterministic_foundation.py     # Main benchmark suite (NEW)
  tasks/
    geometric_tasks.py            # Generate 100 geometric tasks
    arithmetic_tasks.py           # Generate 100 arithmetic tasks
    pattern_tasks.py              # Generate 100 pattern completion tasks
    compositional_tasks.py        # Generate 100 compositional tasks
    rpn_tasks.py                  # Generate 100 RPN evaluation tasks

scripts/
  train_deterministic_foundation.py  # Training protocol (10 iterations)

tests/
  test_deterministic_foundation.py   # Validate task generation
```

### Core Implementation

```python
# benchmarks/deterministic_foundation.py

import numpy as np
from pathlib import Path
from typing import Any
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse

class DeterministicFoundationBenchmark:
    """
    Deterministic task suite for TRM foundation training.

    Teaches TRM:
    - Galaxy navigation (Drawing, Grammar, Math)
    - RPN composition (chain operations)
    - Thought organization (pattern matching → validation)
    """

    def __init__(self):
        self.tasks = self._generate_all_tasks()

    def _generate_all_tasks(self) -> dict:
        """Generate 500 deterministic tasks across 5 categories."""
        from benchmarks.tasks.geometric_tasks import generate_geometric_tasks
        from benchmarks.tasks.arithmetic_tasks import generate_arithmetic_tasks
        from benchmarks.tasks.pattern_tasks import generate_pattern_tasks
        from benchmarks.tasks.compositional_tasks import generate_compositional_tasks
        from benchmarks.tasks.rpn_tasks import generate_rpn_tasks

        return {
            "geometric_transforms": generate_geometric_tasks(100),
            "grid_arithmetic": generate_arithmetic_tasks(100),
            "pattern_completion": generate_pattern_tasks(100),
            "compositional": generate_compositional_tasks(100),
            "symbolic_rpn": generate_rpn_tasks(100),
        }

    def run_benchmark(self, kv: Knowledgeverse) -> dict:
        """
        Run all 500 deterministic tasks.

        Returns:
            {
                "geometric_transforms": {"accuracy": 0.92, "correct": 92, "total": 100},
                "grid_arithmetic": {...},
                ...
                "overall": {"accuracy": 0.83, "correct": 415, "total": 500},
            }
        """
        results = {}

        for category, task_list in self.tasks.items():
            category_correct = 0
            category_total = len(task_list)

            for task in task_list:
                result = self._solve_task(task, kv)

                if self._validate_result(result, task["expected"]):
                    category_correct += 1

            results[category] = {
                "accuracy": category_correct / category_total,
                "correct": category_correct,
                "total": category_total,
            }

        # Overall metrics
        total_correct = sum(r["correct"] for r in results.values())
        total_tasks = sum(r["total"] for r in results.values())

        results["overall"] = {
            "accuracy": total_correct / total_tasks,
            "correct": total_correct,
            "total": total_tasks,
        }

        return results

    def _solve_task(self, task: dict, kv: Knowledgeverse) -> Any:
        """
        Solve single deterministic task using TRM navigation.

        Flow:
        1. TRM queries Galaxy for operation (e.g., "ROTATE_90")
        2. TRM retrieves RPN program from Grammar Galaxy
        3. Cranium executes RPN program (sovereign PTX)
        4. Return result
        """
        category = task["category"]

        if category == "geometric_transforms":
            return self._solve_geometric(task, kv)
        elif category == "grid_arithmetic":
            return self._solve_arithmetic(task, kv)
        elif category == "pattern_completion":
            return self._solve_pattern(task, kv)
        elif category == "compositional":
            return self._solve_compositional(task, kv)
        elif category == "symbolic_rpn":
            return self._solve_rpn(task, kv)
        else:
            raise ValueError(f"Unknown category: {category}")

    def _solve_geometric(self, task: dict, kv: Knowledgeverse) -> np.ndarray:
        """
        Solve geometric transformation task.

        Example:
            task = {"input": grid, "operation": "ROTATE_90", ...}

        TRM flow:
        1. Query Grammar Galaxy: "rotate transformation"
        2. Retrieve: {"rpn_program": "GRID ROTATE_90"}
        3. Execute in Cranium PTX
        4. Return: transformed grid
        """
        operation = task["operation"]
        input_grid = task["input"]

        # TRM navigates Grammar Galaxy to find operation
        grammar_results = kv.galaxy_manager.query(
            query=f"{operation.lower()} transformation",
            specialist="grammar",
            top_k=1
        )

        if not grammar_results:
            # Fallback: TRM hasn't learned this yet, return input (wrong)
            return input_grid

        # Retrieve RPN program
        rpn_program = grammar_results[0].get("rpn_program", "")

        # Execute in Cranium (sovereign!)
        result = kv.cranium.execute_rpn(rpn_program, {"GRID": input_grid})

        return result

    def _solve_arithmetic(self, task: dict, kv: Knowledgeverse) -> int | float:
        """
        Solve grid arithmetic task.

        Example:
            task = {"input": grid, "operation": "COUNT_VALUE", "value": 1, ...}

        TRM flow:
        1. Query Math Galaxy: "count operation"
        2. Retrieve: {"rpn_program": "GRID VALUE EQUAL COUNT"}
        3. Execute in Cranium
        4. Return: scalar result
        """
        operation = task["operation"]
        input_grid = task["input"]
        value = task.get("value")

        # TRM navigates Math Galaxy
        math_results = kv.galaxy_manager.query(
            query=f"{operation.lower()} operation",
            specialist="math",
            top_k=1
        )

        if not math_results:
            return 0  # Wrong, but deterministic

        rpn_program = math_results[0].get("rpn_program", "")

        # Execute with parameters
        result = kv.cranium.execute_rpn(
            rpn_program,
            {"GRID": input_grid, "VALUE": value}
        )

        return result

    def _solve_pattern(self, task: dict, kv: Knowledgeverse) -> np.ndarray:
        """
        Solve pattern completion task.

        Example:
            task = {
                "input": {"sequence": [A, B, None, D]},
                "rule": "alternating",
                ...
            }

        TRM flow:
        1. Analyze input sequence (find pattern)
        2. Query Grammar Galaxy: "alternating pattern rule"
        3. Apply rule to generate missing element
        4. Return: completed pattern
        """
        sequence = task["input"]["sequence"]
        rule = task["rule"]

        # TRM infers pattern from non-None elements
        pattern_results = kv.galaxy_manager.query(
            query=f"{rule} pattern",
            specialist="grammar",
            top_k=1
        )

        if not pattern_results:
            # Fallback: return first element (often wrong)
            return sequence[0] if sequence else np.array([[0]])

        rpn_program = pattern_results[0].get("rpn_program", "")

        # Execute pattern completion
        result = kv.cranium.execute_rpn(
            rpn_program,
            {"SEQUENCE": [s for s in sequence if s is not None]}
        )

        return result

    def _solve_compositional(self, task: dict, kv: Knowledgeverse) -> np.ndarray:
        """
        Solve compositional task (chain operations).

        Example:
            task = {
                "input": grid,
                "operations": ["ROTATE_90", "MIRROR_H"],
                ...
            }

        TRM flow:
        1. Query Grammar Galaxy for each operation
        2. Compose into single RPN program: "GRID ROTATE_90 MIRROR_H"
        3. Execute composed program in Cranium
        4. Return: final result
        """
        input_grid = task["input"]
        operations = task["operations"]

        # TRM composes operations into RPN program
        rpn_parts = ["GRID"]

        for op in operations:
            op_results = kv.galaxy_manager.query(
                query=f"{op.lower()} transformation",
                specialist="grammar",
                top_k=1
            )

            if op_results:
                # Extract operation name from RPN program
                op_rpn = op_results[0].get("rpn_program", "")
                # Assuming format: "GRID OP_NAME" → extract "OP_NAME"
                op_name = op_rpn.split()[-1] if op_rpn else op
                rpn_parts.append(op_name)

        # Compose full RPN program
        rpn_program = " ".join(rpn_parts)

        # Execute composed program
        result = kv.cranium.execute_rpn(rpn_program, {"GRID": input_grid})

        return result

    def _solve_rpn(self, task: dict, kv: Knowledgeverse) -> Any:
        """
        Solve symbolic RPN evaluation task.

        Example:
            task = {"rpn_program": "2 3 ADD", "expected": 5}

        TRM flow:
        1. Parse RPN program
        2. Execute in Cranium (stack-based evaluation)
        3. Return: result
        """
        rpn_program = task["rpn_program"]

        # Direct RPN execution (no Galaxy query needed)
        result = kv.cranium.execute_rpn(rpn_program, {})

        return result

    def _validate_result(self, result: Any, expected: Any) -> bool:
        """Validate result matches expected (exact match for deterministic tasks)."""
        if isinstance(result, np.ndarray) and isinstance(expected, np.ndarray):
            return np.array_equal(result, expected)
        else:
            return result == expected
```

---

## 🔄 Training Protocol

### 10-Iteration Training Loop

```python
# scripts/train_deterministic_foundation.py

from pathlib import Path
import json
from datetime import datetime
from benchmarks.deterministic_foundation import DeterministicFoundationBenchmark
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.sleeptime import consolidate_iteration_events

def train_deterministic_foundation(
    num_iterations: int = 10,
    output_dir: Path = Path("foundation_training_results")
):
    """
    Train TRM on deterministic tasks for foundation learning.

    Protocol:
    1. Run 500 deterministic tasks
    2. Record successes in Shadow Copy
    3. Consolidate Shadow Copy → TRM weight updates
    4. Repeat for 10 iterations

    Expected progression:
    - Iteration 0: 20-30% (baseline, untrained TRM)
    - Iteration 3: 50-60% (TRM learning navigation)
    - Iteration 6: 70-80% (TRM learning composition)
    - Iteration 10: 80-95% (TRM foundation complete)
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    kv = Knowledgeverse()
    benchmark = DeterministicFoundationBenchmark()

    training_history = []

    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")

        # Run benchmark (500 tasks)
        results = benchmark.run_benchmark(kv)

        # Display results
        print(f"\nResults:")
        for category, stats in results.items():
            if category == "overall":
                print(f"\n{'='*60}")
                print(f"OVERALL: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
                print(f"{'='*60}")
            else:
                print(f"  {category:25s}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

        # Consolidate Shadow Copy events → TRM weight updates
        consolidation_result = consolidate_iteration_events(iteration, kv)

        print(f"\nLearning:")
        print(f"  Events consolidated: {consolidation_result['events_consolidated']}")
        print(f"  Weight deltas: {consolidation_result['weight_deltas']}")

        # Record iteration
        training_history.append({
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "consolidation": consolidation_result,
        })

        # Save checkpoint
        checkpoint_path = output_dir / f"iteration_{iteration:03d}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(training_history[-1], f, indent=2)

    # Save full training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "num_iterations": num_iterations,
            "training_history": training_history,
        }, f, indent=2)

    # Final analysis
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")

    initial_accuracy = training_history[0]["results"]["overall"]["accuracy"]
    final_accuracy = training_history[-1]["results"]["overall"]["accuracy"]
    improvement = final_accuracy - initial_accuracy

    print(f"\nProgression:")
    print(f"  Iteration 0:  {initial_accuracy:.1%}")
    print(f"  Iteration 10: {final_accuracy:.1%}")
    print(f"  Improvement:  {improvement:+.1%}")

    if final_accuracy >= 0.80:
        print(f"\n✅ Foundation training SUCCESS! (≥80% accuracy)")
        print(f"   TRM ready for compositional tasks (Phase 1)")
    else:
        print(f"\n⚠️  Foundation training incomplete (<80% accuracy)")
        print(f"   Recommend continuing training or adjusting tasks")

    return training_history

if __name__ == "__main__":
    train_deterministic_foundation(num_iterations=10)
```

---

## 📈 Expected Training Progression

### Iteration-by-Iteration Targets

| Iteration | Overall | Geometric | Arithmetic | Pattern | Compositional | RPN |
|-----------|---------|-----------|------------|---------|---------------|-----|
| **0** (baseline) | 20-30% | 15% | 20% | 25% | 15% | 30% |
| **1** | 30-40% | 25% | 30% | 30% | 20% | 40% |
| **2** | 40-50% | 40% | 45% | 40% | 25% | 50% |
| **3** | 50-60% | 60% | 55% | 50% | 35% | 60% |
| **5** | 60-70% | 75% | 70% | 60% | 45% | 70% |
| **7** | 70-80% | 85% | 80% | 70% | 60% | 80% |
| **10** | **80-95%** | **90%+** | **85%+** | **80%+** | **75%+** | **85%+** |

### Learning Milestones

**Iteration 0-3: Basic Navigation**
- TRM learns to query Grammar Galaxy for operations
- TRM learns to retrieve RPN programs
- Simple operations (ROTATE, MIRROR, COUNT) working

**Iteration 4-6: Composition**
- TRM learns to chain operations
- Compositional tasks improve (15% → 45%)
- Pattern inference begins working

**Iteration 7-10: Foundation Complete**
- All categories >75% accuracy
- TRM can navigate, compose, validate
- Ready for Phase 1 (compositional reasoning) and Phase 2 (ARC-AGI)

---

## 🎯 Success Criteria

### Foundation Training Success

**Must achieve ALL of:**
- ✅ Overall accuracy ≥ 80% (400/500 tasks correct)
- ✅ Geometric transforms ≥ 90% (simple operations deterministic)
- ✅ Grid arithmetic ≥ 85% (counting/summing deterministic)
- ✅ Pattern completion ≥ 75% (inference learnable)
- ✅ Compositional ≥ 70% (TRM can chain operations)
- ✅ Symbolic RPN ≥ 85% (RPN execution deterministic)

**If achieved:**
- ✅ Proceed to Phase 1 (compositional reasoning tasks)
- ✅ Then proceed to Phase 2 (return to ARC-AGI)
- ✅ Expected ARC-AGI improvement: 28% → 40-50% (better TRM foundation)

**If NOT achieved:**
- ⚠️ Continue training (10 more iterations)
- ⚠️ Analyze failure modes (which categories stuck?)
- ⚠️ Adjust task difficulty (maybe too hard for initial training?)

---

## 🔗 Integration with Existing Codebase

### Knowledgeverse Integration

**Galaxy population required:**

```python
# knowledge3d/knowledgeverse/foundational_operations_bootstrap.py - NEW FILE

def populate_foundational_operations(galaxy_manager):
    """
    Populate Grammar + Math galaxies with foundational operations.

    These are the operations TRM needs to solve deterministic tasks.
    """

    # Geometric transformations → Grammar Galaxy
    geometric_ops = [
        {
            "id": "rotate_90_cw",
            "rpn_program": "GRID ROTATE_90",
            "metadata": {"operation": "rotate", "angle": 90, "direction": "clockwise"},
        },
        {
            "id": "mirror_horizontal",
            "rpn_program": "GRID MIRROR_H",
            "metadata": {"operation": "mirror", "axis": "horizontal"},
        },
        {
            "id": "mirror_vertical",
            "rpn_program": "GRID MIRROR_V",
            "metadata": {"operation": "mirror", "axis": "vertical"},
        },
        # ... 20 geometric operations
    ]

    for op in geometric_ops:
        galaxy_manager.add_entry(galaxy="Grammar", entry=op)

    # Arithmetic operations → Math Galaxy
    arithmetic_ops = [
        {
            "id": "count_value",
            "rpn_program": "GRID VALUE EQUAL COUNT",
            "metadata": {"operation": "count", "type": "value_match"},
        },
        {
            "id": "sum_positions",
            "rpn_program": "GRID POSITIONS SUM",
            "metadata": {"operation": "sum", "type": "spatial"},
        },
        # ... 20 arithmetic operations
    ]

    for op in arithmetic_ops:
        galaxy_manager.add_entry(galaxy="Math", entry=op)

    # Pattern rules → Grammar Galaxy
    pattern_rules = [
        {
            "id": "alternating_pattern",
            "rpn_program": "SEQUENCE ALTERNATE",
            "metadata": {"rule": "alternating", "period": 2},
        },
        {
            "id": "mirror_symmetry",
            "rpn_program": "HALF_GRID MIRROR_COMPLETE",
            "metadata": {"rule": "symmetry", "axis": "vertical"},
        },
        # ... 15 pattern rules
    ]

    for rule in pattern_rules:
        galaxy_manager.add_entry(galaxy="Grammar", entry=rule)
```

### Cranium PTX Kernels

**Required PTX operations:**

```python
# knowledge3d/procedural/foundational_kernels.py - NEW FILE

"""
PTX kernels for deterministic foundation tasks.

All operations must be sovereign (PTX + Galaxy only).
"""

# Geometric transformations
ROTATE_90_PTX = """
__global__ void rotate_90_cw(float* input, float* output, int rows, int cols) {
    // PTX implementation of 90° clockwise rotation
    // ...
}
"""

MIRROR_H_PTX = """
__global__ void mirror_horizontal(float* input, float* output, int rows, int cols) {
    // PTX implementation of horizontal mirror
    // ...
}
"""

# Arithmetic operations
COUNT_VALUE_PTX = """
__global__ void count_value(float* grid, float value, int* count, int size) {
    // PTX implementation of value counting
    // ...
}
"""

# Pattern operations (more complex, may need multiple kernels)
# ...
```

---

## 📋 Task Generation Specifications

### Category 1: Geometric Tasks (100 tasks)

```python
# benchmarks/tasks/geometric_tasks.py - NEW FILE

import numpy as np
from typing import List, Dict

def generate_geometric_tasks(num_tasks: int = 100) -> List[Dict]:
    """
    Generate 100 geometric transformation tasks.

    Distribution:
    - 25 tasks: ROTATE (90°, 180°, 270°, 360°)
    - 25 tasks: MIRROR (horizontal, vertical)
    - 25 tasks: TRANSLATE (various offsets)
    - 25 tasks: SCALE (2×, 3×)
    """
    tasks = []

    # Grid sizes: 2×2, 3×3, 4×4, 5×5
    grid_sizes = [2, 3, 4, 5]

    # Rotation tasks (25)
    for i in range(25):
        size = grid_sizes[i % len(grid_sizes)]
        angle = [90, 180, 270, 360][i % 4]

        grid = np.random.randint(0, 3, (size, size))
        rotated = rotate_grid(grid, angle)

        tasks.append({
            "category": "geometric_transforms",
            "task_id": f"geo_rotate_{i:03d}",
            "input": grid,
            "operation": f"ROTATE_{angle}",
            "expected": rotated,
            "rpn_program": f"GRID ROTATE_{angle}",
        })

    # Mirror tasks (25)
    for i in range(25):
        size = grid_sizes[i % len(grid_sizes)]
        axis = "H" if i % 2 == 0 else "V"

        grid = np.random.randint(0, 3, (size, size))
        mirrored = mirror_grid(grid, axis)

        tasks.append({
            "category": "geometric_transforms",
            "task_id": f"geo_mirror_{i:03d}",
            "input": grid,
            "operation": f"MIRROR_{axis}",
            "expected": mirrored,
            "rpn_program": f"GRID MIRROR_{axis}",
        })

    # Translation tasks (25)
    for i in range(25):
        size = grid_sizes[i % len(grid_sizes)]
        dx, dy = i % 3, (i // 3) % 3  # Offsets 0-2

        grid = np.random.randint(0, 3, (size, size))
        translated = translate_grid(grid, dx, dy)

        tasks.append({
            "category": "geometric_transforms",
            "task_id": f"geo_translate_{i:03d}",
            "input": grid,
            "operation": f"TRANSLATE_{dx}_{dy}",
            "expected": translated,
            "rpn_program": f"GRID {dx} {dy} TRANSLATE",
        })

    # Scale tasks (25)
    for i in range(25):
        size = 2  # Small grids for scaling
        scale_factor = 2 if i % 2 == 0 else 3

        grid = np.random.randint(0, 3, (size, size))
        scaled = scale_grid(grid, scale_factor)

        tasks.append({
            "category": "geometric_transforms",
            "task_id": f"geo_scale_{i:03d}",
            "input": grid,
            "operation": f"SCALE_{scale_factor}",
            "expected": scaled,
            "rpn_program": f"GRID {scale_factor} SCALE",
        })

    return tasks

def rotate_grid(grid: np.ndarray, angle: int) -> np.ndarray:
    """Rotate grid by angle (90, 180, 270, 360)."""
    k = angle // 90
    return np.rot90(grid, k=-k)  # Clockwise rotation

def mirror_grid(grid: np.ndarray, axis: str) -> np.ndarray:
    """Mirror grid horizontally or vertically."""
    if axis == "H":
        return np.fliplr(grid)
    elif axis == "V":
        return np.flipud(grid)
    else:
        raise ValueError(f"Unknown axis: {axis}")

def translate_grid(grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate grid by (dx, dy) with zero padding."""
    rows, cols = grid.shape
    result = np.zeros_like(grid)

    # Compute valid ranges
    src_y_start = max(0, -dy)
    src_y_end = min(rows, rows - dy)
    src_x_start = max(0, -dx)
    src_x_end = min(cols, cols - dx)

    dst_y_start = max(0, dy)
    dst_x_start = max(0, dx)

    result[
        dst_y_start:dst_y_start + (src_y_end - src_y_start),
        dst_x_start:dst_x_start + (src_x_end - src_x_start)
    ] = grid[src_y_start:src_y_end, src_x_start:src_x_end]

    return result

def scale_grid(grid: np.ndarray, factor: int) -> np.ndarray:
    """Scale grid by factor (nearest neighbor)."""
    return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)
```

### Category 2: Arithmetic Tasks (100 tasks)

```python
# benchmarks/tasks/arithmetic_tasks.py - NEW FILE

def generate_arithmetic_tasks(num_tasks: int = 100) -> List[Dict]:
    """
    Generate 100 grid arithmetic tasks.

    Distribution:
    - 30 tasks: COUNT_VALUE
    - 25 tasks: SUM_POSITIONS
    - 20 tasks: MAX/MIN operations
    - 25 tasks: FILTER operations
    """
    tasks = []

    # COUNT_VALUE tasks (30)
    for i in range(30):
        size = [2, 3, 4, 5][i % 4]
        value = i % 3  # Count 0s, 1s, or 2s

        grid = np.random.randint(0, 3, (size, size))
        count = np.sum(grid == value)

        tasks.append({
            "category": "grid_arithmetic",
            "task_id": f"arith_count_{i:03d}",
            "input": grid,
            "operation": "COUNT_VALUE",
            "value": value,
            "expected": int(count),
            "rpn_program": f"GRID {value} EQUAL COUNT",
        })

    # SUM_POSITIONS tasks (25)
    for i in range(25):
        size = [2, 3, 4][i % 3]

        grid = np.random.randint(0, 3, (size, size))
        # Sum of all x-coordinates where grid value is non-zero
        positions = np.argwhere(grid > 0)
        sum_x = int(np.sum(positions[:, 1])) if len(positions) > 0 else 0

        tasks.append({
            "category": "grid_arithmetic",
            "task_id": f"arith_sum_pos_{i:03d}",
            "input": grid,
            "operation": "SUM_X_POSITIONS",
            "expected": sum_x,
            "rpn_program": "GRID NONZERO_POSITIONS X_COORDS SUM",
        })

    # MAX/MIN tasks (20)
    for i in range(20):
        size = [2, 3, 4][i % 3]

        grid = np.random.randint(0, 10, (size, size))

        if i % 2 == 0:
            result = int(np.max(grid))
            operation = "MAX"
        else:
            result = int(np.min(grid))
            operation = "MIN"

        tasks.append({
            "category": "grid_arithmetic",
            "task_id": f"arith_{operation.lower()}_{i:03d}",
            "input": grid,
            "operation": operation,
            "expected": result,
            "rpn_program": f"GRID {operation}",
        })

    # FILTER tasks (25)
    for i in range(25):
        size = [2, 3, 4][i % 3]
        threshold = i % 5

        grid = np.random.randint(0, 10, (size, size))
        count = np.sum(grid > threshold)

        tasks.append({
            "category": "grid_arithmetic",
            "task_id": f"arith_filter_{i:03d}",
            "input": grid,
            "operation": "FILTER_GT",
            "threshold": threshold,
            "expected": int(count),
            "rpn_program": f"GRID {threshold} GT COUNT",
        })

    return tasks
```

### Categories 3-5: Similar Detailed Specifications

(Pattern, Compositional, RPN task generators follow same structure - detailed in separate files)

---

## 🚀 Deployment Plan

### Week 20, Day 1-2: Task Generation

**Codex implements:**
1. ✅ `benchmarks/deterministic_foundation.py` (main benchmark)
2. ✅ `benchmarks/tasks/geometric_tasks.py` (100 geometric tasks)
3. ✅ `benchmarks/tasks/arithmetic_tasks.py` (100 arithmetic tasks)
4. ✅ `benchmarks/tasks/pattern_tasks.py` (100 pattern tasks)
5. ✅ `benchmarks/tasks/compositional_tasks.py` (100 compositional tasks)
6. ✅ `benchmarks/tasks/rpn_tasks.py` (100 RPN tasks)
7. ✅ `tests/test_deterministic_foundation.py` (validate task generation)

**Validation:**
- Run test suite: `pytest tests/test_deterministic_foundation.py`
- Verify 500 tasks generated correctly
- Check RPN programs are valid

### Week 20, Day 3: Galaxy Population

**Codex implements:**
1. ✅ `knowledge3d/knowledgeverse/foundational_operations_bootstrap.py`
   - Populate Grammar Galaxy (geometric + pattern operations)
   - Populate Math Galaxy (arithmetic operations)
2. ✅ Run bootstrap: `python -m knowledge3d.knowledgeverse.foundational_operations_bootstrap`
3. ✅ Verify: Galaxy Universe has ~60 foundational operations

### Week 20, Day 4: Baseline Run

**Codex runs:**
1. ✅ Train iteration 0 (baseline, untrained TRM)
2. ✅ Expected: 20-30% overall accuracy
3. ✅ Save results: `foundation_training_results/iteration_000.json`

**Analysis:**
- Which categories work best at baseline? (RPN likely highest)
- Which categories are hardest? (Compositional likely lowest)
- Does TRM find ANY operations in Galaxy? (sanity check)

### Week 20, Day 5-6: Full Training (10 iterations)

**Codex runs:**
1. ✅ `python scripts/train_deterministic_foundation.py`
2. ✅ Monitor progression (expect 20% → 80% over 10 iterations)
3. ✅ Save full history: `foundation_training_results/training_history.json`

**Expected timeline:**
- 10 iterations × 500 tasks/iteration = 5,000 task evaluations
- Estimated: 2-4 hours runtime

### Week 20, Day 7: Analysis + Next Steps

**Codex analyzes:**
1. ✅ Did we achieve ≥80% accuracy? (success criteria)
2. ✅ Which categories improved most? (learning effectiveness)
3. ✅ TRM weight progression (how did weights change?)

**If successful (≥80%):**
- ✅ Proceed to Phase 1 (compositional reasoning tasks)
- ✅ Design Phase 1 tasks (Week 21)

**If unsuccessful (<80%):**
- ⚠️ Analyze failure modes (which tasks stuck?)
- ⚠️ Continue training (10 more iterations)
- ⚠️ Adjust task difficulty or Galaxy population

### Week 21: Return to ARC-AGI

**After foundation training:**
1. ✅ Re-run ARC-AGI benchmark
2. ✅ Expected: 28% → 40-50% (better TRM navigation)
3. ✅ Re-run diagnostics (oracle@k, counterfactual, ablation)
4. ✅ Now diagnostics will be meaningful (generation quality improved)

---

## 📊 Reporting & Telemetry

### Per-Iteration Report

```json
{
  "iteration": 5,
  "timestamp": "2026-02-08T14:30:00Z",
  "results": {
    "geometric_transforms": {"accuracy": 0.75, "correct": 75, "total": 100},
    "grid_arithmetic": {"accuracy": 0.70, "correct": 70, "total": 100},
    "pattern_completion": {"accuracy": 0.60, "correct": 60, "total": 100},
    "compositional": {"accuracy": 0.45, "correct": 45, "total": 100},
    "symbolic_rpn": {"accuracy": 0.70, "correct": 70, "total": 100},
    "overall": {"accuracy": 0.64, "correct": 320, "total": 500}
  },
  "consolidation": {
    "events_read": 500,
    "events_consolidated": 320,
    "weight_deltas": {
      "grammar_confidence": 0.032,
      "navigation_efficiency": 0.028
    }
  }
}
```

### Final Summary

```json
{
  "num_iterations": 10,
  "initial_accuracy": 0.24,
  "final_accuracy": 0.83,
  "improvement": 0.59,
  "success": true,
  "per_category_final": {
    "geometric_transforms": 0.92,
    "grid_arithmetic": 0.86,
    "pattern_completion": 0.78,
    "compositional": 0.72,
    "symbolic_rpn": 0.87
  },
  "ready_for_phase_1": true
}
```

---

## 🎯 Bottom Line for Codex

### What You're Building

**A deterministic task curriculum that trains TRM to:**
- Navigate Galaxy Universe effectively
- Compose RPN programs correctly
- Organize thought (pattern matching → validation)

**Why this matters:**
- Current ARC-AGI: oracle_at_all = 0.0 (can't generate correct answers)
- Root cause: TRM hasn't learned HOW to think/navigate
- Solution: Train on easier deterministic tasks FIRST
- Expected outcome: ARC-AGI 28% → 40-50% after foundation

### Success Criteria

**Phase 0 complete when:**
- ✅ Overall accuracy ≥ 80% (400/500 tasks)
- ✅ All categories ≥ 70% (balanced learning)
- ✅ TRM weights show consistent improvement
- ✅ Shadow Copy consolidation working (events → weights)

**Then proceed to:**
- Phase 1: Compositional reasoning (multi-step, conditional)
- Phase 2: ARC-AGI (with better TRM foundation)

### Immediate Next Steps

1. **Day 1-2:** Implement task generators (5 files, 500 tasks total)
2. **Day 3:** Populate Galaxy with foundational operations
3. **Day 4:** Run baseline (iteration 0, expect 20-30%)
4. **Day 5-6:** Full training (10 iterations, expect 20% → 80%)
5. **Day 7:** Analysis + proceed to Phase 1 or continue training

**Estimated timeline:** 1 week (Week 20)

**Let's build the foundation that makes brute force learning actually work!** 🚀

---

**Document prepared by:** Claude (Architecture Partner)
**Date:** February 8, 2026
**For:** Codex (Implementation Partner)
**Status:** Ready for implementation
**Priority:** CRITICAL — Foundation before BFL
