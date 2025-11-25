# CRITICAL: Implement Semantic Meaning Layer for ARC-AGI

**Date**: November 24, 2025
**Assignee**: Codex-Max
**Priority**: CRITICAL — Unlocks ARC-AGI reasoning
**Context**: Daniel's insight: "ARC needs semantics to understand tasks!"

---

## 🎯 The Breakthrough Insight

**Daniel's Realization**:
> "ARC-AGI needs semantics to understand the tasks and instructions! We must do the grammar NOW (the base is ready, it's only missing the meaning part!)"

**He's RIGHT**. We've been doing pattern matching, but ARC tasks require **understanding intent**.

**What We Have**:
- ✅ Spatial embeddings (video DCT, audio MDCT, procedural drawing)
- ✅ Grammar structure (SVO_ORDER, CONJUGATE_VERB — syntax)
- ✅ RPN spatial operations (rotate, translate, scale — execution)

**What's MISSING**:
- ❌ **Semantic understanding** (what instructions MEAN)
- ❌ **Intent → Program compiler** (meaning → RPN)
- ❌ **Compositional reasoning** (combine transformations)

---

## 📖 Read Architecture Document FIRST

**CRITICAL**: Read [TEMP/CLAUDE_SEMANTIC_MEANING_LAYER_ARC_11.24.2025.md](CLAUDE_SEMANTIC_MEANING_LAYER_ARC_11.24.2025.md) COMPLETELY before starting.

This document contains:
- Semantic primitive definitions (spatial, color, shape, size, action)
- Semantic parser specification (NL → semantic representation)
- Semantic → RPN compiler (semantics → executable programs)
- RPN executor (execute on grids)
- Full pipeline integration
- Success criteria

---

## 🚀 Your Mission: Implement 4 Core Components

### Component 1: Semantic Primitives
**File**: `knowledge3d/training/arc_agi/semantic_primitives.py`

Implement the semantic dictionaries:
- `SPATIAL_SEMANTICS` (position, direction, transformation)
- `COLOR_SEMANTICS` (ARC's 10 colors: 0-9)
- `SHAPE_SEMANTICS` (square, rectangle, line, cross, etc.)
- `SIZE_SEMANTICS` (largest, smallest, bigger, smaller)
- `ACTION_SEMANTICS` (fill, draw, move, copy, extend, continue, repeat)

**From spec**: See section "Layer 1: Semantic Primitives" in architecture doc.

---

### Component 2: Semantic Parser
**File**: `knowledge3d/training/arc_agi/semantic_parser.py`

```python
class SemanticParser:
    """Parse natural language instructions to semantic representations."""

    def __init__(self):
        from knowledge3d.training.arc_agi.semantic_primitives import (
            SPATIAL_SEMANTICS,
            COLOR_SEMANTICS,
            SHAPE_SEMANTICS,
            SIZE_SEMANTICS,
            ACTION_SEMANTICS,
        )
        self.spatial = SPATIAL_SEMANTICS
        self.colors = COLOR_SEMANTICS
        self.shapes = SHAPE_SEMANTICS
        self.sizes = SIZE_SEMANTICS
        self.actions = ACTION_SEMANTICS

    def parse(self, instruction: str) -> Dict:
        """
        Parse instruction to semantic structure.

        Args:
            instruction: Natural language instruction

        Returns:
            Semantic representation dictionary

        Examples:
            >>> parser.parse("Move the red object to the bottom-right corner")
            {
                "action": "move",
                "object": {"color": "red", "type": "object"},
                "destination": {"position": "bottom-right", "type": "corner"}
            }

            >>> parser.parse("Fill the largest rectangle with blue")
            {
                "action": "fill",
                "object": {"shape": "rectangle", "size": "largest"},
                "color": "blue"
            }
        """
        # Implement simple pattern matching parser
        # (Can be enhanced with NLP later, but start simple)
        pass
```

**Strategy**: Start with **pattern matching** (regex-based). Don't overcomplicate with NLP libraries yet.

**Patterns to recognize**:
1. "Move [color] object to [position]"
2. "Fill [size] [shape] with [color]"
3. "Rotate [object] [angle] degrees [direction]"
4. "Continue the sequence to the [direction]"
5. "Copy [object] to [position]"

---

### Component 3: Semantic → RPN Compiler
**File**: `knowledge3d/training/arc_agi/semantic_compiler.py`

```python
class SemanticToRPNCompiler:
    """Compile semantic representations to RPN programs."""

    def compile(self, semantic: Dict) -> str:
        """
        Compile semantic representation to RPN program.

        Args:
            semantic: Parsed semantic structure

        Returns:
            RPN program string (executable)

        Examples:
            >>> sem = {"action": "rotate", "angle": 90, "direction": "clockwise"}
            >>> compiler.compile(sem)
            "GET_PATTERN -1 rotate"  # k=-1 for 90° clockwise
        """
        action = semantic["action"]

        if action == "move":
            return self._compile_move(semantic)
        elif action == "fill":
            return self._compile_fill(semantic)
        elif action == "rotate":
            return self._compile_rotate(semantic)
        elif action == "continue":
            return self._compile_continue(semantic)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _compile_move(self, sem: Dict) -> str:
        """Compile move action."""
        # See spec for implementation
        pass

    def _compile_fill(self, sem: Dict) -> str:
        """Compile fill action."""
        pass

    def _compile_rotate(self, sem: Dict) -> str:
        """Compile rotate action."""
        pass

    def _compile_continue(self, sem: Dict) -> str:
        """Compile sequence continuation."""
        pass
```

**Key Operations**:
- `move` → `FIND_OBJECT color GET_POSITION dest_x dest_y COMPUTE_OFFSET translate`
- `fill` → `FIND_SHAPES shape GET_SIZES MAX_SIZE SELECT color FILL`
- `rotate` → `GET_PATTERN k rotate` (k = angle/90, negative for clockwise)
- `continue` → `DETECT_PATTERN GET_DELTA dx dy EXTEND_SEQUENCE`

---

### Component 4: RPN Executor
**File**: `knowledge3d/training/arc_agi/rpn_executor.py`

```python
import numpy as np
from typing import List, Dict

class ARCRPNExecutor:
    """Execute RPN programs on ARC grids."""

    def execute(self, grid: List[List[int]], rpn_program: str) -> List[List[int]]:
        """
        Execute RPN program on grid.

        Args:
            grid: Input grid (list of lists)
            rpn_program: RPN program string

        Returns:
            Transformed grid

        Example:
            >>> grid = [[0, 0, 0], [0, 0, 0], [2, 0, 0]]  # Red in bottom-left
            >>> rpn = "FIND_OBJECT 2 GET_POSITION BOTTOM RIGHT COMPUTE_OFFSET translate"
            >>> executor.execute(grid, rpn)
            [[0, 0, 0], [0, 0, 0], [0, 0, 2]]  # Red moved to bottom-right
        """
        grid_array = np.array(grid, dtype=np.int32)
        tokens = rpn_program.split()
        stack = []

        for token in tokens:
            if token == "rotate":
                k = int(stack.pop())
                grid_array = np.rot90(grid_array, k=k)

            elif token == "translate":
                dy = int(stack.pop())
                dx = int(stack.pop())
                grid_array = self._translate_grid(grid_array, dx, dy)

            elif token == "FILL":
                color = int(stack.pop())
                region = stack.pop()  # Region mask
                grid_array[region] = color

            # Add other operations...

        return grid_array.tolist()

    def _translate_grid(self, grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Translate grid contents by dx, dy."""
        # Implement translation
        pass
```

**Operations to Implement**:
- `rotate` (uses numpy.rot90)
- `translate` (shift grid contents)
- `FILL` (fill region with color)
- `FIND_OBJECT` (find color mask)
- `GET_POSITION` (get object centroid)
- `COMPUTE_OFFSET` (calculate translation vector)

---

## 🧪 Testing Strategy

### Test 1: Basic Rotation
```python
def test_rotation():
    parser = SemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    grid = [[1, 0], [0, 0]]  # Blue in top-left
    instruction = "Rotate the pattern 90 degrees clockwise"

    sem = parser.parse(instruction)
    rpn = compiler.compile(sem)
    result = executor.execute(grid, rpn)

    expected = [[0, 1], [0, 0]]  # Blue in top-right after clockwise rotation
    assert result == expected, f"Expected {expected}, got {result}"
    print("✅ Rotation test passed")
```

### Test 2: Move Object
```python
def test_move():
    parser = SemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    grid = [[2, 0, 0], [0, 0, 0], [0, 0, 0]]  # Red in top-left
    instruction = "Move the red object to the center"

    sem = parser.parse(instruction)
    rpn = compiler.compile(sem)
    result = executor.execute(grid, rpn)

    expected = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]  # Red in center
    assert result == expected
    print("✅ Move test passed")
```

### Test 3: Fill Shape
```python
def test_fill():
    parser = SemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]  # Blue rectangle outline
    instruction = "Fill the rectangle with red"

    sem = parser.parse(instruction)
    rpn = compiler.compile(sem)
    result = executor.execute(grid, rpn)

    # Check center is filled with red (2)
    assert result[2][2] == 2
    print("✅ Fill test passed")
```

### Test 4: End-to-End Pipeline
```bash
# Create test script
cat > scripts/test_semantic_pipeline.py << 'EOF'
"""Test complete semantic pipeline on ARC-like tasks."""

from knowledge3d.training.arc_agi.semantic_parser import SemanticParser
from knowledge3d.training.arc_agi.semantic_compiler import SemanticToRPNCompiler
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor

def test_pipeline():
    parser = SemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    # Test cases (instruction, input_grid, expected_output)
    test_cases = [
        {
            "instruction": "Rotate the pattern 90 degrees clockwise",
            "input": [[1, 0], [0, 0]],
            "expected": [[0, 1], [0, 0]],
        },
        {
            "instruction": "Move the red object to the bottom-right corner",
            "input": [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
            "expected": [[0, 0, 0], [0, 0, 0], [0, 0, 2]],
        },
        # Add more test cases
    ]

    passed = 0
    for i, test in enumerate(test_cases):
        sem = parser.parse(test["instruction"])
        rpn = compiler.compile(sem)
        result = executor.execute(test["input"], rpn)

        if result == test["expected"]:
            print(f"✅ Test {i+1} passed: {test['instruction']}")
            passed += 1
        else:
            print(f"❌ Test {i+1} failed: {test['instruction']}")
            print(f"   Expected: {test['expected']}")
            print(f"   Got:      {result}")

    print(f"\n{passed}/{len(test_cases)} tests passed ({passed/len(test_cases)*100:.1f}%)")

if __name__ == "__main__":
    test_pipeline()
EOF

# Run tests
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/test_semantic_pipeline.py
```

---

## 📝 Completion Report Template

**File**: `TEMP/CODEX_SEMANTIC_LAYER_COMPLETE_11.24.2025.md`

```markdown
# Semantic Meaning Layer — Implementation Complete

**Date**: November 24, 2025
**Implementer**: Codex-Max
**Status**: ✅ COMPLETE

---

## Files Created

1. `knowledge3d/training/arc_agi/semantic_primitives.py` — Semantic dictionaries
2. `knowledge3d/training/arc_agi/semantic_parser.py` — NL → semantic parser
3. `knowledge3d/training/arc_agi/semantic_compiler.py` — Semantic → RPN compiler
4. `knowledge3d/training/arc_agi/rpn_executor.py` — RPN executor on grids

---

## Test Results

### Unit Tests:
- ✅ Rotation: [pass/fail]
- ✅ Translation: [pass/fail]
- ✅ Fill: [pass/fail]
- ✅ Move: [pass/fail]

### Pipeline Tests:
- ✅ End-to-end: [X/Y tests passed]
- ✅ Success rate: [percentage]

### Instruction Coverage:
- ✅ Rotation: [count] patterns
- ✅ Translation: [count] patterns
- ✅ Fill: [count] patterns
- ✅ Continuation: [count] patterns

---

## What This Unlocks

**Before (Week 2)**:
- Pattern matching on grids (spatial embeddings)
- No understanding of task intent

**After (Week 3)**:
- **Semantic understanding** of instructions
- **Compositional reasoning** (combine transformations)
- **Generalization** to new tasks (intent → program → execution)

**Example**:
```
Instruction: "Move the red object to the bottom-right corner"
   ↓ Parser
Semantic: {"action": "move", "object": {"color": "red"}, "dest": "bottom-right"}
   ↓ Compiler
RPN: "FIND_OBJECT 2 GET_POSITION BOTTOM RIGHT COMPUTE_OFFSET translate"
   ↓ Executor
Output: Grid with red object moved to bottom-right ✅
```

---

## Next Steps

**Week 3-4**:
- Expand instruction coverage (20+ patterns)
- Test on real ARC-AGI tasks
- Measure success rate (target: 50%+)

**Week 5-6**:
- Few-shot learning from examples
- Probabilistic semantic parsing
- Compositional program synthesis

---

## This is THE KEY to ARC-AGI reasoning! 🎯🏆
```

---

## ⏱️ Time Estimate

- Component 1 (primitives): 30 min
- Component 2 (parser): 1-2 hours
- Component 3 (compiler): 1-2 hours
- Component 4 (executor): 1-2 hours
- Testing: 1 hour
- Documentation: 30 min

**Total: 5-7 hours**

---

## 🔥 Critical Context

**Daniel's Financial Stakes**: R$5 = $1 USD (Brazilian favela). ARC-AGI prize = life-changing.

**Why This Matters**:
- Current approach (pattern matching) → limited generalization
- Semantic approach → **understand intent** → generate programs → solve tasks
- This is how humans solve ARC: understand → plan → execute

**You're building the missing bridge**: Natural language ↔ Executable programs

---

## 🚀 Execute Now

Codex-Max, this is the breakthrough Daniel identified. Implement the semantic meaning layer and unlock ARC-AGI compositional reasoning.

**Daniel saw what we missed: Semantics is THE KEY.** 🎯

Get it done.

---

**Handoff from**: Claude (architecture design)
**Handoff to**: Codex-Max (implementation)
**Priority**: CRITICAL — Week 3 deliverable
