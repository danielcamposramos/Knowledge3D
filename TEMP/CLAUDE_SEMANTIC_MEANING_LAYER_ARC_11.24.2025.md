# Semantic Meaning Layer for ARC-AGI — Architecture Specification

**Date**: November 24, 2025
**Architect**: Claude
**Priority**: CRITICAL — Enables ARC-AGI task understanding
**Status**: Ready for implementation

---

## 🎯 The Problem Daniel Identified

**Current ARC Approach** (Week 2):
- ✅ Spatial embeddings (DCT, MDCT, procedural drawing)
- ✅ Pattern matching (cosine similarity)
- ❌ **NO understanding of what the task asks us to DO**

**Example ARC Task**:
```
Input: [3×3 grid with red square in top-left]
Output: [3×3 grid with red square in bottom-right]
Instruction: "Move the red object to the bottom-right corner"
```

**What We Can Do Now**:
- ✅ Embed both grids (procedural/video/audio)
- ✅ Detect they're different
- ❌ **Can't understand "move to bottom-right"**
- ❌ **Can't generate the transformation**

**What We Need**:
**Semantic understanding** → Parse instruction → Generate RPN program → Execute on grid

---

## Architecture: Semantic Meaning Layer

### Layer 1: Semantic Primitives (Concepts)

**Spatial Concepts**:
```python
SPATIAL_SEMANTICS = {
    # Position
    "top": {"type": "position", "y": 0, "anchor": "top"},
    "bottom": {"type": "position", "y": "max", "anchor": "bottom"},
    "left": {"type": "position", "x": 0, "anchor": "left"},
    "right": {"type": "position", "x": "max", "anchor": "right"},
    "center": {"type": "position", "x": "mid", "y": "mid"},
    "corner": {"type": "position", "compound": True},

    # Direction
    "up": {"type": "direction", "dy": -1},
    "down": {"type": "direction", "dy": +1},
    "left_dir": {"type": "direction", "dx": -1},
    "right_dir": {"type": "direction", "dx": +1},

    # Transformation
    "rotate": {"type": "transform", "rpn_op": "rotate", "opcode": 70},
    "flip": {"type": "transform", "rpn_op": "flip", "opcode": None},
    "mirror": {"type": "transform", "rpn_op": "flip", "opcode": None},
    "scale": {"type": "transform", "rpn_op": "scale", "opcode": 71},
    "translate": {"type": "transform", "rpn_op": "translate", "opcode": 72},
    "move": {"type": "transform", "rpn_op": "translate", "opcode": 72},

    # Angle (for rotation)
    "90_degrees": {"type": "angle", "degrees": 90, "k": 1},
    "180_degrees": {"type": "angle", "degrees": 180, "k": 2},
    "clockwise": {"type": "direction", "sign": -1},
    "counterclockwise": {"type": "direction", "sign": +1},
}

COLOR_SEMANTICS = {
    "black": {"type": "color", "value": 0},
    "blue": {"type": "color", "value": 1},
    "red": {"type": "color", "value": 2},
    "green": {"type": "color", "value": 3},
    "yellow": {"type": "color", "value": 4},
    "grey": {"type": "color", "value": 5},
    "pink": {"type": "color", "value": 6},
    "orange": {"type": "color", "value": 7},
    "cyan": {"type": "color", "value": 8},
    "brown": {"type": "color", "value": 9},
}

SHAPE_SEMANTICS = {
    "square": {"type": "shape", "pattern": "filled_rectangle"},
    "rectangle": {"type": "shape", "pattern": "filled_rectangle"},
    "line": {"type": "shape", "pattern": "line"},
    "cross": {"type": "shape", "pattern": "cross"},
    "diagonal": {"type": "shape", "pattern": "diagonal"},
    "border": {"type": "shape", "pattern": "border"},
    "fill": {"type": "shape", "pattern": "fill_region"},
}

SIZE_SEMANTICS = {
    "largest": {"type": "size", "comparator": "max"},
    "smallest": {"type": "size", "comparator": "min"},
    "bigger": {"type": "size", "comparator": "greater"},
    "smaller": {"type": "size", "comparator": "less"},
}

ACTION_SEMANTICS = {
    "fill": {"type": "action", "rpn_op": "FILL", "opcode": 0x6B},
    "draw": {"type": "action", "rpn_op": "LINE", "opcode": 0x65},
    "move": {"type": "action", "rpn_op": "translate", "opcode": 72},
    "copy": {"type": "action", "rpn_op": "duplicate"},
    "extend": {"type": "action", "rpn_op": "extend_pattern"},
    "continue": {"type": "action", "rpn_op": "continue_sequence"},
    "repeat": {"type": "action", "rpn_op": "repeat_pattern"},
}
```

### Layer 2: Semantic Parser (NL → Semantic Representation)

**Input**: Natural language instruction (from ARC task description or human)
**Output**: Semantic representation (structured concepts)

**Examples**:

**Example 1**: "Move the red object to the bottom-right corner"
```python
semantic_parse = {
    "action": "move",
    "object": {
        "color": "red",
        "type": "object"
    },
    "destination": {
        "position": "bottom-right",
        "type": "corner"
    }
}
```

**Example 2**: "Fill the largest rectangle with blue"
```python
semantic_parse = {
    "action": "fill",
    "object": {
        "shape": "rectangle",
        "size": "largest"
    },
    "color": "blue"
}
```

**Example 3**: "Rotate the pattern 90 degrees clockwise"
```python
semantic_parse = {
    "action": "rotate",
    "object": "pattern",
    "angle": 90,
    "direction": "clockwise"
}
```

**Example 4**: "Continue the sequence to the right"
```python
semantic_parse = {
    "action": "continue",
    "object": "sequence",
    "direction": "right"
}
```

### Layer 3: Semantic → RPN Compiler

**Input**: Semantic representation
**Output**: RPN program (executable on grids)

**Compilation Rules**:

```python
class SemanticToRPNCompiler:
    """Compile semantic representations to RPN programs."""

    def compile(self, semantic: Dict) -> str:
        """
        Compile semantic representation to RPN program.

        Args:
            semantic: Parsed semantic structure

        Returns:
            RPN program string
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
        """
        Compile move action to RPN.

        Example:
            "Move red object to bottom-right"
            → "FIND_OBJECT red GET_POSITION BOTTOM RIGHT COMPUTE_OFFSET translate"
        """
        # Find object
        color = COLOR_SEMANTICS[sem["object"]["color"]]["value"]
        dest = sem["destination"]["position"]

        # Compile to RPN
        rpn = f"FIND_OBJECT {color} GET_POSITION "  # Get current position

        # Parse destination
        if dest == "bottom-right":
            rpn += "BOTTOM RIGHT COMPUTE_OFFSET "
        elif dest == "center":
            rpn += "CENTER COMPUTE_OFFSET "

        # Execute translation
        rpn += "translate"  # OpCode 72

        return rpn

    def _compile_fill(self, sem: Dict) -> str:
        """
        Compile fill action to RPN.

        Example:
            "Fill largest rectangle with blue"
            → "FIND_SHAPES rectangle GET_SIZES MAX_SIZE SELECT blue FILL"
        """
        shape = sem["object"]["shape"]
        size = sem["object"]["size"]
        color = COLOR_SEMANTICS[sem["color"]]["value"]

        rpn = f"FIND_SHAPES {shape} "

        if size == "largest":
            rpn += "GET_SIZES MAX_SIZE SELECT "
        elif size == "smallest":
            rpn += "GET_SIZES MIN_SIZE SELECT "

        rpn += f"{color} FILL"  # OpCode 0x6B

        return rpn

    def _compile_rotate(self, sem: Dict) -> str:
        """
        Compile rotate action to RPN.

        Example:
            "Rotate pattern 90 degrees clockwise"
            → "GET_PATTERN 1 rotate"  # k=1 for 90°, negative for clockwise
        """
        angle = sem["angle"]
        direction = sem.get("direction", "counterclockwise")

        # Convert to k parameter for np.rot90
        k = angle // 90
        if direction == "clockwise":
            k = -k

        rpn = f"GET_PATTERN {k} rotate"  # OpCode 70

        return rpn

    def _compile_continue(self, sem: Dict) -> str:
        """
        Compile sequence continuation to RPN.

        Example:
            "Continue the sequence to the right"
            → "DETECT_PATTERN GET_DELTA +1 0 EXTEND_SEQUENCE"
        """
        direction = sem["direction"]

        dx, dy = 0, 0
        if direction == "right":
            dx = 1
        elif direction == "left":
            dx = -1
        elif direction == "down":
            dy = 1
        elif direction == "up":
            dy = -1

        rpn = f"DETECT_PATTERN GET_DELTA {dx} {dy} EXTEND_SEQUENCE"

        return rpn
```

### Layer 4: RPN Executor (Execute on Grids)

**Input**: RPN program + Grid
**Output**: Transformed grid

```python
class ARCRPNExecutor:
    """Execute RPN programs on ARC grids."""

    def __init__(self):
        from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
        self.rpn = ModularRPNEngine()

    def execute(self, grid: List[List[int]], rpn_program: str) -> List[List[int]]:
        """
        Execute RPN program on grid.

        Args:
            grid: Input grid (list of lists)
            rpn_program: RPN program string

        Returns:
            Transformed grid
        """
        # Convert grid to numpy for operations
        import numpy as np
        grid_array = np.array(grid, dtype=np.int32)

        # Parse and execute RPN program
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

            elif token == "FIND_OBJECT":
                color = int(stack.pop())
                mask = (grid_array == color)
                stack.append(mask)

            elif token.isdigit() or (token.startswith('-') and token[1:].isdigit()):
                stack.append(int(token))

            else:
                # Handle other operations
                pass

        return grid_array.tolist()

    def _translate_grid(self, grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Translate grid contents by dx, dy."""
        result = np.zeros_like(grid)
        h, w = grid.shape

        for y in range(h):
            for x in range(w):
                new_y = y + dy
                new_x = x + dx
                if 0 <= new_y < h and 0 <= new_x < w:
                    result[new_y, new_x] = grid[y, x]

        return result
```

---

## Integration with ARC-AGI Pipeline

### Full Pipeline:

```
ARC Task → Semantic Parser → Semantic Repr → RPN Compiler → RPN Program → Executor → Output Grid
    ↓              ↓                ↓               ↓              ↓            ↓           ↓
"Move red    {"action":       "FIND_OBJECT 2   Execute on    Transformed
to bottom-    "move",         GET_POSITION     input grid     grid
right"        "object":       BOTTOM RIGHT                    (output)
              {"color":       COMPUTE_OFFSET
               "red"},        translate"
              "dest":
              "bottom-right"}
```

### Example End-to-End:

```python
from knowledge3d.training.arc_agi.semantic_parser import SemanticParser
from knowledge3d.training.arc_agi.semantic_compiler import SemanticToRPNCompiler
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor

# Initialize components
parser = SemanticParser()
compiler = SemanticToRPNCompiler()
executor = ARCRPNExecutor()

# ARC task
instruction = "Move the red object to the bottom-right corner"
input_grid = [
    [0, 0, 0],
    [0, 0, 0],
    [2, 0, 0],  # Red (2) in bottom-left
]

# Step 1: Parse instruction to semantics
semantic = parser.parse(instruction)
# → {"action": "move", "object": {"color": "red"}, "destination": "bottom-right"}

# Step 2: Compile semantics to RPN
rpn_program = compiler.compile(semantic)
# → "FIND_OBJECT 2 GET_POSITION BOTTOM RIGHT COMPUTE_OFFSET translate"

# Step 3: Execute RPN on grid
output_grid = executor.execute(input_grid, rpn_program)
# → [[0, 0, 0],
#    [0, 0, 0],
#    [0, 0, 2]]  # Red moved to bottom-right ✅

print("Success! Output matches expected transformation.")
```

---

## Implementation Files

### 1. `knowledge3d/training/arc_agi/semantic_primitives.py`
```python
"""Semantic primitive definitions for ARC-AGI tasks."""

SPATIAL_SEMANTICS = {...}  # From above
COLOR_SEMANTICS = {...}
SHAPE_SEMANTICS = {...}
SIZE_SEMANTICS = {...}
ACTION_SEMANTICS = {...}
```

### 2. `knowledge3d/training/arc_agi/semantic_parser.py`
```python
"""Parse natural language instructions to semantic representations."""

class SemanticParser:
    def parse(self, instruction: str) -> Dict:
        """Parse instruction to semantic structure."""
        pass
```

### 3. `knowledge3d/training/arc_agi/semantic_compiler.py`
```python
"""Compile semantic representations to RPN programs."""

class SemanticToRPNCompiler:
    def compile(self, semantic: Dict) -> str:
        """Compile semantic to RPN program."""
        pass
```

### 4. `knowledge3d/training/arc_agi/rpn_executor.py`
```python
"""Execute RPN programs on ARC grids."""

class ARCRPNExecutor:
    def execute(self, grid: List[List[int]], rpn: str) -> List[List[int]]:
        """Execute RPN program on grid."""
        pass
```

---

## Success Criteria

**MUST ACHIEVE**:
- ✅ Semantic primitives defined (spatial, color, shape, size, action)
- ✅ Semantic parser working (5+ instruction types)
- ✅ Semantic → RPN compiler working (5+ transformations)
- ✅ RPN executor working on ARC grids

**SHOULD ACHIEVE**:
- ✅ 20+ instruction patterns covered
- ✅ Compositional semantics (combine multiple actions)
- ✅ Integration with ARC-AGI embedders (Week 2)

**NICE TO HAVE**:
- ⚠️ Learning from examples (few-shot semantic parsing)
- ⚠️ Probabilistic semantic parsing (handle ambiguity)
- ⚠️ Semantic similarity (cluster similar instructions)

---

## Timeline

**Week 3 (Now)**:
- Day 1-2: Implement semantic primitives + parser
- Day 3-4: Implement semantic → RPN compiler
- Day 5-6: Implement RPN executor + integration tests
- Day 7: Test on 50 ARC tasks, measure success rate

**Target**: 50%+ success rate on ARC training tasks with explicit instructions.

---

## Why This Unlocks ARC-AGI

**Current Limitation**:
- Pattern matching can't explain **why** a transformation happens
- Can't generalize to new instructions

**With Semantics**:
- Understand **intent**: "move", "fill", "rotate", "continue"
- Compose transformations: "rotate then fill"
- Generalize: "move red object" → works for any color/position

**The Key**:
> **Semantics bridge natural language understanding to procedural execution.**

This is what humans do when solving ARC tasks:
1. Read instruction → understand intent (semantics)
2. Plan transformation → generate mental program (RPN)
3. Execute transformation → apply to grid (PTX execution)

**We're building the same pipeline, end-to-end.**

---

**Status**: Ready for implementation NOW. This is the missing piece for ARC-AGI reasoning! 🎯🏆

---

**Handoff to**: Codex (implementation)
**Priority**: CRITICAL — Week 3 deliverable
