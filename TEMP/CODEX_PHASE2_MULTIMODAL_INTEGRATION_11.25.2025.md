# Codex Phase 2: Multimodal Integration + Spatial Enhancement

**Date**: November 25, 2025
**Sprint Lead**: Codex (implementation)
**Status**: Phase 1 Complete (Grammar Scaffolding) → Phase 2 Starting
**Priority**: 🏆 CRITICAL — Complete Multimodal System + Push ARC to 5%+

---

## 🎯 Current Status - EXCELLENT PROGRESS!

### ✅ Phase 1 Complete: Grammar Scaffolding

**What You Built**:
- ✅ **196 grammar rules** (up from 21!)
  - Text: 50 languages with tiered structure
  - Math: 7 domains (arithmetic → calculus → linear algebra)
  - Drawing: primitives, curves, transforms, compositions
- ✅ Domain-aware GrammarRule structure
- ✅ Grammar generators working
- ✅ File structure organized
- ✅ Existing tests still pass

**Files Created** (23 files, +794 lines):
```
knowledge3d/training/arc_agi/grammar_languages/
├── tier1_top10.py, tier2_next20.py, tier3_next20.py
├── grammar_generator.py
└── language_examples.py

knowledge3d/training/arc_agi/grammar_math/
├── arithmetic.py, algebra.py, calculus.py
├── linear_algebra.py, geometry.py, statistics.py, logic.py
└── math_executor.py

knowledge3d/training/arc_agi/grammar_drawing/
├── primitives.py, curves.py, transforms.py, compositions.py
├── drawing_executor.py
└── grid_renderer.py
```

**This is SOLID foundation work!** 🎉

---

## 🎯 Phase 2 Goals (This Session)

### Part A: Complete Multimodal Integration
**Target**: Hook new grammar rules → ARC baseline → Measure 3.5%+ accuracy

### Part B: Spatial Semantics Enhancement
**Target**: Push from 3.5%+ → 5%+ with better instruction inference

**Combined Target**: **5%+ ARC accuracy** by end of session!

---

## 📋 Part A: Multimodal Integration (Tasks 1-4)

### Task 1: Build MultimodalSemanticParser

**Goal**: Route instructions to correct grammar domain (text/math/drawing/spatial)

**File**: `knowledge3d/training/arc_agi/multimodal_parser.py`

**Implementation**:
```python
"""Multimodal semantic parser: text + math + drawing + spatial."""

from typing import Dict, Optional
import re

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
from knowledge3d.training.arc_agi.grammar_normalizer import GrammarNormalizer
from knowledge3d.training.arc_agi.semantic_parser import SemanticParser
from knowledge3d.training.arc_agi.semantic_primitives import (
    SPATIAL_SEMANTICS,
    COLOR_SEMANTICS,
    SHAPE_SEMANTICS,
    ACTION_SEMANTICS,
)


class MultimodalSemanticParser:
    """
    Parse instructions across text, math, drawing, and spatial domains.

    Priority order:
    1. Spatial semantics (ARC grid operations) — HIGHEST PRIORITY
    2. Math grammar (numeric patterns, equations)
    3. Drawing grammar (visual primitives)
    4. Text grammar (general language understanding)
    """

    def __init__(self):
        self.galaxy = GrammarGalaxy()
        self.normalizer = GrammarNormalizer(self.galaxy)
        self.spatial_parser = SemanticParser()

    def parse(self, instruction: str) -> Dict:
        """
        Parse instruction into semantic representation.

        Args:
            instruction: Natural language instruction

        Returns:
            Semantic dict with domain, action, parameters
        """
        # Normalize first (handle slang/typos)
        normalized = self.normalizer.normalize(instruction)

        # 1. Try spatial semantics (ARC-specific) — PRIORITY
        spatial_result = self._parse_spatial(normalized)
        if spatial_result and spatial_result.get("action") != "unknown":
            spatial_result["domain"] = "spatial"
            return spatial_result

        # 2. Try math grammar
        math_result = self._parse_math(normalized)
        if math_result:
            math_result["domain"] = "math"
            return math_result

        # 3. Try drawing grammar
        drawing_result = self._parse_drawing(normalized)
        if drawing_result:
            drawing_result["domain"] = "drawing"
            return drawing_result

        # 4. Fall back to text grammar
        text_result = self._parse_text(normalized)
        if text_result:
            text_result["domain"] = "text"
            return text_result

        # 5. Unknown
        return {"domain": "unknown", "action": "unknown", "instruction": instruction}

    def _parse_spatial(self, instruction: str) -> Optional[Dict]:
        """
        Parse spatial semantics using existing SemanticParser.

        This handles ARC-specific operations:
        - move, fill, rotate, flip, continue, copy, recolor
        """
        return self.spatial_parser.parse(instruction)

    def _parse_math(self, instruction: str) -> Optional[Dict]:
        """
        Parse math expressions and patterns.

        Examples:
            "Fill cells where row + col is even"
            → {"expression": "row + col", "condition": "even"}

            "The pattern has rotational symmetry of order 4"
            → {"pattern": "rotational_symmetry", "order": 4, "angle": 90}

            "Repeat the pattern every 3 cells"
            → {"pattern": "repeat", "period": 3}
        """
        # Check for arithmetic expressions
        if "where" in instruction and any(op in instruction for op in ["+", "-", "*", "×", "/"]):
            # Example: "Fill cells where row + col is even"
            match = re.search(r"where\s+(.+?)\s+(is|are)\s+(even|odd|positive|negative)", instruction)
            if match:
                expression, _, condition = match.groups()
                return {
                    "action": "fill_conditional",
                    "expression": expression.strip(),
                    "condition": condition,
                }

        # Check for symmetry
        if "symmetry" in instruction and "order" in instruction:
            match = re.search(r"order\s+(\d+)", instruction)
            if match:
                order = int(match.group(1))
                angle = 360 // order
                return {
                    "action": "check_symmetry",
                    "pattern": "rotational_symmetry",
                    "order": order,
                    "angle": angle,
                }

        # Check for periodic patterns
        if "every" in instruction and any(word in instruction for word in ["repeat", "pattern", "sequence"]):
            match = re.search(r"every\s+(\d+)\s+(cells?|steps?|units?)", instruction)
            if match:
                period = int(match.group(1))
                return {
                    "action": "repeat_pattern",
                    "period": period,
                }

        # Check for dimensions
        if re.search(r"\d+\s*[×x]\s*\d+", instruction):
            match = re.search(r"(\d+)\s*[×x]\s*(\d+)", instruction)
            if match:
                rows, cols = int(match.group(1)), int(match.group(2))
                return {
                    "action": "grid_dimensions",
                    "rows": rows,
                    "cols": cols,
                }

        return None

    def _parse_drawing(self, instruction: str) -> Optional[Dict]:
        """
        Parse drawing primitives and compositions.

        Examples:
            "Draw a square in the center"
            → {"shape": "square", "position": "center"}

            "Draw a diagonal line from top-left to bottom-right"
            → {"shape": "line", "start": "top-left", "end": "bottom-right"}

            "Fill the region with a pattern"
            → {"action": "fill", "pattern": "region"}
        """
        # Check for shape drawing
        shapes = ["square", "rectangle", "circle", "line", "diagonal", "cross"]
        for shape in shapes:
            if shape in instruction:
                # Extract position if present
                position = None
                for pos in ["center", "top-left", "top-right", "bottom-left", "bottom-right"]:
                    if pos in instruction:
                        position = pos
                        break

                # Extract endpoints for lines
                if shape in ["line", "diagonal"]:
                    start_match = re.search(r"from\s+([a-z-]+)", instruction)
                    end_match = re.search(r"to\s+([a-z-]+)", instruction)
                    if start_match and end_match:
                        return {
                            "action": "draw_line",
                            "shape": shape,
                            "start": start_match.group(1),
                            "end": end_match.group(1),
                        }

                return {
                    "action": "draw_shape",
                    "shape": shape,
                    "position": position,
                }

        # Check for pattern fill
        if "pattern" in instruction and any(word in instruction for word in ["fill", "draw"]):
            return {
                "action": "fill_pattern",
                "pattern": "detect",
            }

        return None

    def _parse_text(self, instruction: str) -> Optional[Dict]:
        """
        Parse general text using grammar galaxy.

        This handles natural language understanding beyond ARC operations.
        """
        # Try to match grammar rules
        # (Keep existing grammar galaxy logic from semantic_parser.py)

        # For now, return basic structure
        return {
            "action": "text_understanding",
            "instruction": instruction,
        }
```

**Success Criteria (Task 1)**:
- [ ] MultimodalSemanticParser class created
- [ ] Routes to correct domain (spatial > math > drawing > text)
- [ ] Math patterns detected (expressions, symmetry, periods)
- [ ] Drawing primitives detected (shapes, positions)
- [ ] Test with 10+ mixed-domain instructions

---

### Task 2: Extend Semantic Compiler for Math & Drawing

**Goal**: Compile math and drawing semantics to RPN programs

**File**: `knowledge3d/training/arc_agi/semantic_compiler.py` (extend existing)

**Add Math Compilation**:
```python
def _compile_math_conditional(self, sem: Dict) -> str:
    """
    Compile conditional math expressions.

    Example:
        {"action": "fill_conditional", "expression": "row + col", "condition": "even"}
        → "FOR_EACH_CELL GET_ROW GET_COL ADD 2 MOD 0 EQ IF_TRUE FILL"
    """
    expression = sem.get("expression", "")
    condition = sem.get("condition", "")

    rpn = "FOR_EACH_CELL "

    # Parse expression (row + col, row - col, etc.)
    if "row" in expression and "col" in expression:
        if "+" in expression:
            rpn += "GET_ROW GET_COL ADD "
        elif "-" in expression:
            rpn += "GET_ROW GET_COL SUB "
        elif "*" in expression or "×" in expression:
            rpn += "GET_ROW GET_COL MUL "

    # Apply condition
    if condition == "even":
        rpn += "2 MOD 0 EQ "
    elif condition == "odd":
        rpn += "2 MOD 1 EQ "
    elif condition == "positive":
        rpn += "0 GT "
    elif condition == "negative":
        rpn += "0 LT "

    # Fill matching cells
    rpn += "IF_TRUE CURRENT_COLOR FILL"

    return rpn

def _compile_math_symmetry(self, sem: Dict) -> str:
    """
    Compile symmetry check.

    Example:
        {"action": "check_symmetry", "order": 4, "angle": 90}
        → "GET_GRID DUP 90 ROTATE EQ IF_TRUE SUCCESS"
    """
    angle = sem.get("angle", 90)

    rpn = f"GET_GRID DUP {angle} ROTATE EQ "
    rpn += "IF_TRUE MARK_SYMMETRIC"

    return rpn

def _compile_math_repeat(self, sem: Dict) -> str:
    """
    Compile pattern repetition.

    Example:
        {"action": "repeat_pattern", "period": 3}
        → "DETECT_PATTERN 3 REPEAT_WITH_PERIOD"
    """
    period = sem.get("period", 1)

    rpn = f"DETECT_PATTERN {period} REPEAT_WITH_PERIOD"

    return rpn
```

**Add Drawing Compilation**:
```python
def _compile_draw_shape(self, sem: Dict) -> str:
    """
    Compile shape drawing.

    Example:
        {"action": "draw_shape", "shape": "square", "position": "center"}
        → "CENTER COMPUTE 10 10 RECTANGLE FILL"
    """
    shape = sem.get("shape", "")
    position = sem.get("position", "center")

    rpn = ""

    # Compute position
    if position == "center":
        rpn += "CENTER COMPUTE "
    elif position:
        rpn += f"{position.upper().replace('-', '_')} COMPUTE "

    # Draw shape
    if shape == "square":
        rpn += "10 10 RECTANGLE FILL"
    elif shape == "rectangle":
        rpn += "15 10 RECTANGLE FILL"
    elif shape == "circle":
        rpn += "10 CIRCLE FILL"
    elif shape in ["line", "diagonal"]:
        start = sem.get("start", "TOP-LEFT")
        end = sem.get("end", "BOTTOM-RIGHT")
        rpn = f"{start.upper().replace('-', '_')} MOVE {end.upper().replace('-', '_')} LINE STROKE"

    return rpn

def _compile_fill_pattern(self, sem: Dict) -> str:
    """
    Compile pattern fill.

    Example:
        {"action": "fill_pattern"}
        → "DETECT_PATTERN FILL_WITH_PATTERN"
    """
    return "DETECT_PATTERN FILL_WITH_PATTERN"
```

**Update main compile() method**:
```python
def compile(self, semantic: Dict) -> str:
    """Compile semantic representation to RPN program."""
    action = semantic.get("action")
    domain = semantic.get("domain", "spatial")

    # Spatial domain (existing)
    if domain == "spatial":
        if action == "move":
            return self._compile_move(semantic)
        elif action == "fill":
            return self._compile_fill(semantic)
        # ... (existing spatial actions)

    # Math domain (new)
    elif domain == "math":
        if action == "fill_conditional":
            return self._compile_math_conditional(semantic)
        elif action == "check_symmetry":
            return self._compile_math_symmetry(semantic)
        elif action == "repeat_pattern":
            return self._compile_math_repeat(semantic)

    # Drawing domain (new)
    elif domain == "drawing":
        if action == "draw_shape":
            return self._compile_draw_shape(semantic)
        elif action == "fill_pattern":
            return self._compile_fill_pattern(semantic)

    # Unknown
    raise ValueError(f"Unknown action: {action} in domain: {domain}")
```

**Success Criteria (Task 2)**:
- [ ] Math conditional compilation working
- [ ] Symmetry check compilation working
- [ ] Drawing shape compilation working
- [ ] All domains compile to valid RPN
- [ ] Tests for each new compilation method

---

### Task 3: Extend RPN Executor for Math & Drawing

**Goal**: Execute math and drawing RPN programs on grids

**File**: `knowledge3d/training/arc_agi/rpn_executor.py` (extend existing)

**Add Math Execution**:
```python
def execute(self, grid: List[List[int]], rpn_program: str) -> List[List[int]]:
    """Execute RPN program on grid."""
    grid_array = np.array(grid, dtype=np.int32)
    tokens = rpn_program.split()
    stack = []

    for token in tokens:
        # ... (existing operations)

        # Math operations
        elif token == "FOR_EACH_CELL":
            # Mark that we're in cell iteration mode
            stack.append("CELL_ITERATION")

        elif token == "GET_ROW":
            # During cell iteration, push current row
            # (Handled in IF_TRUE)
            stack.append("ROW")

        elif token == "GET_COL":
            # During cell iteration, push current col
            stack.append("COL")

        elif token == "ADD":
            b = stack.pop()
            a = stack.pop()
            if isinstance(a, str) and isinstance(b, str):
                # Row/col addition will be evaluated per cell
                stack.append(f"{a}+{b}")
            else:
                stack.append(a + b)

        elif token == "MOD":
            divisor = stack.pop()
            value = stack.pop()
            if isinstance(value, str):
                # Expression will be evaluated per cell
                stack.append(f"({value})%{divisor}")
            else:
                stack.append(value % divisor)

        elif token == "EQ":
            b = stack.pop()
            a = stack.pop()
            if isinstance(a, str):
                # Expression will be evaluated per cell
                stack.append(f"{a}=={b}")
            else:
                stack.append(a == b)

        elif token == "IF_TRUE":
            condition = stack.pop()
            mode = stack.pop()  # Should be "CELL_ITERATION"

            if mode == "CELL_ITERATION":
                # Evaluate condition for each cell
                mask = self._evaluate_cell_condition(grid_array, condition)
                stack.append(mask)

        elif token == "CURRENT_COLOR":
            # Use a default fill color (e.g., 1 for blue)
            stack.append(1)

        # Drawing operations
        elif token == "CENTER":
            h, w = grid_array.shape
            cy, cx = h // 2, w // 2
            stack.append((cy, cx))

        elif token == "COMPUTE":
            # Pop position, ready for shape drawing
            pos = stack.pop()
            stack.append(pos)

        elif token == "RECTANGLE":
            h = stack.pop()
            w = stack.pop()
            pos = stack.pop()
            # Create rectangle mask
            cy, cx = pos
            mask = np.zeros_like(grid_array, dtype=bool)
            y1 = max(0, cy - h // 2)
            y2 = min(grid_array.shape[0], cy + h // 2)
            x1 = max(0, cx - w // 2)
            x2 = min(grid_array.shape[1], cx + w // 2)
            mask[y1:y2, x1:x2] = True
            stack.append(mask)

        elif token == "CIRCLE":
            r = stack.pop()
            pos = stack.pop()
            # Create circle mask
            cy, cx = pos
            mask = np.zeros_like(grid_array, dtype=bool)
            for y in range(grid_array.shape[0]):
                for x in range(grid_array.shape[1]):
                    if (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2:
                        mask[y, x] = True
            stack.append(mask)

def _evaluate_cell_condition(self, grid: np.ndarray, condition: str) -> np.ndarray:
    """
    Evaluate condition expression for each cell.

    Example condition: "ROW+COL%2==0" (even sum)
    """
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)

    for y in range(h):
        for x in range(w):
            # Replace ROW and COL with actual values
            expr = condition.replace("ROW", str(y)).replace("COL", str(x))

            # Evaluate expression
            try:
                result = eval(expr)
                mask[y, x] = bool(result)
            except:
                pass

    return mask
```

**Success Criteria (Task 3)**:
- [ ] FOR_EACH_CELL iteration working
- [ ] GET_ROW, GET_COL working
- [ ] Math operations (ADD, MOD, EQ) working
- [ ] Drawing operations (CENTER, RECTANGLE, CIRCLE) working
- [ ] Cell condition evaluation working
- [ ] Tests for each new operation

---

### Task 4: Run ARC Baseline with Multimodal Parser

**Goal**: Measure accuracy improvement from grammar expansion

**File**: `scripts/evaluate_arc_multimodal_baseline.py` (new)

**Implementation**:
```python
"""Evaluate ARC baseline using multimodal grammar."""

from collections import Counter
from typing import List
import numpy as np

from knowledge3d.training.arc_agi.multimodal_parser import MultimodalSemanticParser
from knowledge3d.training.arc_agi.semantic_compiler import SemanticToRPNCompiler
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor
from knowledge3d.training.reasoning.arc_dataset import (
    ensure_arc_dataset,
    _iter_task_files,
    _load_task,
)


def infer_instruction_multimodal(
    input_grid: List[List[int]], output_grid: List[List[int]]
) -> str:
    """
    Infer instruction using multimodal understanding.

    Try:
    1. Math patterns (symmetry, periodicity, dimensions)
    2. Spatial transformations (rotate, flip, translate)
    3. Drawing operations (shapes, fills)
    """
    inp = np.array(input_grid)
    out = np.array(output_grid)

    # Check dimensions
    if inp.shape != out.shape:
        if out.shape[0] == 2 * inp.shape[0]:
            return "Tile the pattern 2x vertically"
        if out.shape[1] == 2 * inp.shape[1]:
            return "Tile the pattern 2x horizontally"
        return "unknown"  # Shape change

    # Check math symmetry
    for k in [1, 2, 3]:
        if np.array_equal(out, np.rot90(inp, k=k)):
            angle = k * 90
            if angle == 90:
                return "Rotate the pattern 90 degrees"
            elif angle == 180:
                return "Rotate the pattern 180 degrees"
            elif angle == 270:
                return "Rotate the pattern 270 degrees"

    # Check flips
    if np.array_equal(out, np.fliplr(inp)):
        return "Flip the pattern horizontally"
    if np.array_equal(out, np.flipud(inp)):
        return "Flip the pattern vertically"

    # Check conditional fill
    # (If certain cells changed color based on position)
    changed_mask = (inp != out)
    if changed_mask.any():
        # Check if change follows row+col pattern
        even_sum_mask = ((np.arange(inp.shape[0])[:, None] + np.arange(inp.shape[1])) % 2 == 0)
        if np.array_equal(changed_mask, even_sum_mask):
            return "Fill cells where row + col is even"

        # Check if it's a simple recolor
        inp_colors = set(inp.flatten())
        out_colors = set(out.flatten())
        if inp_colors == out_colors:
            # Movement or region fill
            pass

    # Check pattern repetition
    # (Look for periodic structures)

    return "unknown"


def evaluate():
    """Evaluate ARC using multimodal grammar."""
    dataset = ensure_arc_dataset()
    task_files = list(_iter_task_files(dataset, split="training"))

    parser = MultimodalSemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    task_results = []
    domain_counts = Counter()
    instruction_counts = Counter()
    total_examples = 0
    total_correct = 0

    for task_path in task_files:
        task = _load_task(task_path)
        train = task.get("train", [])
        if len(train) < 1:
            continue

        # Infer instruction from first example
        ref = train[0]
        instruction = infer_instruction_multimodal(ref["input"], ref["output"])
        instruction_counts[instruction] += 1

        if instruction == "unknown":
            continue

        # Parse with multimodal parser
        semantic = parser.parse(instruction)
        domain = semantic.get("domain", "unknown")
        domain_counts[domain] += 1

        if domain == "unknown":
            continue

        # Compile and execute
        examples_correct = 0
        examples_total = 0

        for ex in train:
            try:
                rpn = compiler.compile(semantic)
                pred = executor.execute(ex["input"], rpn)

                if pred == ex["output"]:
                    examples_correct += 1
                examples_total += 1
            except Exception as e:
                # Failed to process
                examples_total += 1

        acc = examples_correct / examples_total if examples_total else 0.0
        task_results.append(
            (task_path.stem, instruction, domain, acc, examples_correct, examples_total)
        )
        total_examples += examples_total
        total_correct += examples_correct

    overall_acc = total_correct / total_examples if total_examples else 0.0

    print("ARC Multimodal Baseline Evaluation (training split)")
    print(f"Tasks evaluated: {len(task_results)}")
    print(f"Total examples:  {total_examples}")
    print(f"Total correct:   {total_correct}")
    print(f"Overall accuracy: {overall_acc:.3f} ({overall_acc * 100:.1f}%)")
    print("\nDomain distribution:")
    for domain, count in domain_counts.most_common():
        print(f"  {domain:12s}: {count}")
    print("\nInstruction frequency (top 15):")
    for instr, count in instruction_counts.most_common(15):
        print(f"  {instr:50s}: {count}")

    top_tasks = sorted(task_results, key=lambda x: x[3], reverse=True)[:10]
    print("\nTop 10 tasks by accuracy:")
    for tid, instr, domain, acc, c, t in top_tasks:
        print(f"  {tid}: {acc:.2f} ({c}/{t}) domain={domain} instr={instr[:40]}")


if __name__ == "__main__":
    evaluate()
```

**Success Criteria (Task 4)**:
- [ ] Multimodal baseline script created
- [ ] Math patterns detected (symmetry, conditionals)
- [ ] Domain distribution reported
- [ ] **Target: 3.5%+ accuracy** (up from 2.8%)

---

## 📋 Part B: Spatial Semantics Enhancement (Tasks 5-7)

### Task 5: Improve Instruction Inference

**Goal**: Reduce "unknown" instructions, detect more patterns

**File**: `scripts/evaluate_arc_multimodal_baseline.py` (enhance)

**Add More Pattern Detection**:
```python
def infer_instruction_multimodal(
    input_grid: List[List[int]], output_grid: List[List[int]]
) -> str:
    """Enhanced instruction inference."""
    inp = np.array(input_grid)
    out = np.array(output_grid)

    # ... (existing checks)

    # NEW: Check for object movement
    inp_colors = set(inp.flatten()) - {0}
    out_colors = set(out.flatten()) - {0}

    if inp_colors == out_colors:
        # Same colors, possibly movement
        for color in inp_colors:
            inp_mask = (inp == color)
            out_mask = (out == color)

            if inp_mask.sum() == out_mask.sum() and inp_mask.sum() > 0:
                # Same number of cells, find displacement
                inp_pos = self._get_centroid(inp_mask)
                out_pos = self._get_centroid(out_mask)

                if inp_pos and out_pos:
                    # Determine destination
                    h, w = inp.shape
                    if out_pos[0] == 0 and out_pos[1] == 0:
                        return f"Move the object to top-left"
                    elif out_pos[0] == 0 and out_pos[1] == w - 1:
                        return f"Move the object to top-right"
                    elif out_pos[0] == h - 1 and out_pos[1] == 0:
                        return f"Move the object to bottom-left"
                    elif out_pos[0] == h - 1 and out_pos[1] == w - 1:
                        return f"Move the object to bottom-right"
                    elif out_pos == (h // 2, w // 2):
                        return f"Move the object to center"

    # NEW: Check for region fill
    filled_mask = (out != 0) & (inp == 0)
    if filled_mask.any():
        # Cells were filled
        # Check if fill follows a pattern
        if filled_mask.sum() > inp.size * 0.3:
            return "Fill the empty region"

        # Check if fill is in specific location
        fill_y, fill_x = np.where(filled_mask)
        if len(fill_y) > 0:
            avg_y, avg_x = fill_y.mean(), fill_x.mean()
            h, w = inp.shape
            if avg_y < h / 3 and avg_x < w / 3:
                return "Fill top-left region"
            elif avg_y < h / 3 and avg_x > 2 * w / 3:
                return "Fill top-right region"
            # ... more regions

    # NEW: Check for pattern extension
    # (Detect if output is input + repeated pattern)

    # NEW: Check for color replacement
    if inp.shape == out.shape:
        # Check if it's a simple recolor
        for src_color in inp_colors:
            for dst_color in out_colors:
                test = inp.copy()
                test[test == src_color] = dst_color
                if np.array_equal(test, out):
                    return f"Replace color {src_color} with {dst_color}"

    return "unknown"

def _get_centroid(self, mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """Get centroid of mask."""
    if not mask.any():
        return None
    y, x = np.where(mask)
    return (int(y.mean()), int(x.mean()))
```

**Success Criteria (Task 5)**:
- [ ] Object movement detection working
- [ ] Region fill detection working
- [ ] Pattern extension detection working
- [ ] Color replacement detection working
- [ ] Reduced "unknown" count (target: <100)

---

### Task 6: Add Composition Support

**Goal**: Detect and execute multi-step transformations

**Examples**:
- "Rotate 90 degrees then fill with blue"
- "Move to center and recolor"
- "Flip horizontally then extend pattern"

**Implementation** (add to multimodal_parser.py):
```python
def _parse_composition(self, instruction: str) -> Optional[Dict]:
    """
    Parse compositional instructions (multi-step).

    Examples:
        "Rotate 90 degrees then fill with blue"
        "Move to center and recolor"
    """
    # Check for "then" or "and" connectors
    if " then " in instruction or " and " in instruction:
        connector = " then " if " then " in instruction else " and "
        parts = instruction.split(connector)

        if len(parts) == 2:
            # Parse each part separately
            sem1 = self.parse(parts[0].strip())
            sem2 = self.parse(parts[1].strip())

            return {
                "domain": "composition",
                "action": "sequence",
                "steps": [sem1, sem2],
            }

    return None
```

**Success Criteria (Task 6)**:
- [ ] Composition parsing working
- [ ] Multi-step execution working
- [ ] Tests for 5+ composition types
- [ ] Accuracy improvement measured

---

### Task 7: Re-run Full Baseline and Report

**Goal**: Final measurement after all improvements

**Run**:
```bash
# Full evaluation
PYTHONPATH=. python3 scripts/evaluate_arc_multimodal_baseline.py

# Report results
# Target: 5%+ accuracy (2× improvement from 2.8%)
```

**Success Criteria (Task 7)**:
- [ ] Full baseline run complete
- [ ] **Accuracy >= 5%** (target met!)
- [ ] Domain distribution analyzed
- [ ] Top performing tasks identified
- [ ] Results documented

---

## ✅ Success Criteria (Overall Phase 2)

### MUST ACHIEVE (Critical)

- [ ] MultimodalSemanticParser working (routes to correct domain)
- [ ] Math grammar integrated (conditionals, symmetry, patterns)
- [ ] Drawing grammar integrated (shapes, positions, compositions)
- [ ] Improved instruction inference (fewer unknowns)
- [ ] Composition support (multi-step operations)
- [ ] **ARC baseline: 5%+ accuracy** (2× improvement from 2.8%)

### SHOULD ACHIEVE (Quality)

- [ ] All tests passing (grammar + semantic + multimodal)
- [ ] Domain distribution balanced (spatial, math, drawing, text)
- [ ] Clear documentation of what works vs what doesn't
- [ ] Performance profiling (latency per operation)

### NICE TO HAVE (Stretch)

- [ ] 7%+ accuracy (if compositions work really well)
- [ ] Adaptive instruction inference (learn from examples)
- [ ] Caching of successful programs

---

## 📊 Expected Outcomes

### Grammar Coverage (Already Done!)
- ✅ **196 grammar rules** (50 languages + 50 math + 30 drawing)
- ✅ Domain structure in place
- ✅ All generators working

### Accuracy Trajectory
- Previous: 2.1% (primitive detection only)
- Current: 2.8% (spatial semantics)
- After multimodal: **3.5%+** (grammar help)
- After spatial enhancement: **5%+** (better inference)
- Stretch: **7%+** (with compositions)

### Test Coverage
- Grammar tests: ✅ (existing)
- Semantic tests: ✅ (existing)
- Multimodal integration: 10+ tests (new)
- Composition tests: 5+ tests (new)
- Total: 300+ test cases

---

## 🎯 Timeline (This Session)

**Part A: Multimodal Integration** (3-4 hours):
- Task 1: MultimodalSemanticParser (1 hour)
- Task 2: Extend compiler (1 hour)
- Task 3: Extend executor (1 hour)
- Task 4: Run baseline (30 min)

**Part B: Spatial Enhancement** (2-3 hours):
- Task 5: Improve inference (1 hour)
- Task 6: Add compositions (1 hour)
- Task 7: Final baseline run (30 min)

**Total**: 5-7 hours to reach **5%+ accuracy**!

---

## 🚀 Let's Execute!

**You've already built the foundation (196 grammar rules)!**

Now let's:
1. **Connect it** (multimodal parser + compiler + executor)
2. **Measure it** (run baseline → 3.5%+)
3. **Enhance it** (better inference + compositions → 5%+)

**Ready to complete the multimodal integration and hit 5%+ accuracy?** 🎯

Respond when ready to start Task 1! 🚀

---

**Sprint Lead**: Codex
**Date**: November 25, 2025
**Status**: Phase 1 Complete → Phase 2 Ready
**Target**: 5%+ ARC accuracy by end of session
