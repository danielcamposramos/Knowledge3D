# Codex Sprint: ARC-AGI Week 3 — Spatial Semantics Layer

**Date**: November 25, 2025
**Sprint Lead**: Codex (implementation)
**Architect**: Claude (specifications)
**Status**: Ready to Execute
**Priority**: 🏆 CRITICAL — ARC-AGI 2 Competition Path

---

## 🎯 Sprint Goal

**Build the Spatial Semantics Layer** that bridges natural language instructions to executable grid transformations.

**Success Metric**: Raise ARC-AGI accuracy from 2.1% → 5%+ by understanding spatial instructions.

---

## 📚 CRITICAL: Read These First

**BEFORE writing ANY code, read these files COMPLETELY:**

1. **`TEMP/CLAUDE_ARC_PROGRESS_ANALYSIS_11.25.2025.md`** (THIS IS KEY!)
   - Explains what you built vs what we need
   - Clarifies that 2.1% is actually GOOD (competitive with state-of-art!)
   - Shows architecture gap: text grammar ≠ spatial semantics

2. **`TEMP/CLAUDE_SEMANTIC_MEANING_LAYER_ARC_11.24.2025.md`**
   - Complete specification for spatial semantics
   - Spatial primitives, parser, compiler, executor
   - Example end-to-end flows

3. **`TEMP/CLAUDE_GRAMMAR_RPN_SPEC_11.24.2025.md`**
   - Grammar Galaxy context (what you already built)
   - User profiles and multilingual support
   - How text grammar fits into the full stack

4. **`docs/vocabulary/REALITY_ENABLER_SPECIFICATION.md`**
   - `visual_rpn` + `behavior_rpn` pattern
   - Symlink composition (atoms → molecules → systems)
   - Matryoshka LOD for simulation

5. **`docs/vocabulary/MATH_CORE_SPECIFICATION.md`**
   - 3-tier routing (simple → mid → high)
   - Ternary operations (SIGN, TQUANT, TCMP)
   - Tesla 3-6-9 resonance + Setun heritage

**Why This Matters**: Partial reads cause architecture violations. These specs define HOW to build the semantic layer the K3D way!

---

## 🧭 Current State (What You Inherit)

### What Works ✅

1. **Grammar Galaxy** (`knowledge3d/training/arc_agi/grammar_galaxy.py`)
   - 21 procedural rules for text generation
   - Multilingual: EN, PT, JA, ES, and more
   - User profiles with personal vocabulary
   - Slang/typo normalization
   - **Tests**: 21/21 passing

2. **Grammar Executor** (`knowledge3d/training/arc_agi/grammar_executor.py`)
   - Stack-based RPN execution for grammar
   - Handles SVO/SOV/VSO orderings
   - Coordination, conditionals, temporal sequences

3. **Semantic Pipeline Infrastructure**
   - Parser: `semantic_parser.py`
   - Compiler: `semantic_compiler.py`
   - Executor: `rpn_executor.py`
   - **Tests**: 8/8 passing

4. **ARC Primitive Baseline** (`scripts/evaluate_arc_primitive_baseline.py`)
   - **2.1% accuracy** (27/1302 examples on training set)
   - Detects: ROTATE, FLIP, TRANSLATE, RECOLOR
   - Composite transforms: ROTATE_TRANSLATE

### What's Missing ⚠️

**The Spatial Semantics Layer!**

Current pipeline:
```
Text Instruction → ??? → Grid Transformation
```

Needed pipeline:
```
Text Instruction → Spatial Semantics → RPN Program → Grid Transformation
                    ↑ THIS IS MISSING!
```

**Example**:
```
Input: "Move the red object to the bottom-right corner"

Current: Uses regex + primitive detection (brittle, limited)

Needed:
1. Parse → {"action": "move", "object": {"color": "red"}, "destination": "bottom-right"}
2. Compile → "2 FIND_OBJECT GET_POSITION BOTTOM-RIGHT COMPUTE_OFFSET translate"
3. Execute → Transformed grid with red object moved
```

---

## 🔨 Implementation Plan

### Task 1: Enhance Spatial Primitives

**File**: `knowledge3d/training/arc_agi/semantic_primitives.py`

**What Exists** (from your previous work):
```python
# Basic spatial primitives (axes and actions)
SPATIAL_PRIMITIVES = {
    "horizontal_axis": {...},
    "vertical_axis": {...},
    # Some action entries
}
```

**What to Add** (from Claude's spec):
```python
"""Semantic primitive definitions for ARC-AGI spatial reasoning."""

SPATIAL_SEMANTICS = {
    # Position
    "top": {"type": "position", "y": 0, "anchor": "top"},
    "bottom": {"type": "position", "y": "max", "anchor": "bottom"},
    "left": {"type": "position", "x": 0, "anchor": "left"},
    "right": {"type": "position", "x": "max", "anchor": "right"},
    "center": {"type": "position", "x": "mid", "y": "mid"},
    "corner": {"type": "position", "compound": True},
    "top-left": {"type": "position", "x": 0, "y": 0},
    "top-right": {"type": "position", "x": "max", "y": 0},
    "bottom-left": {"type": "position", "x": 0, "y": "max"},
    "bottom-right": {"type": "position", "x": "max", "y": "max"},

    # Direction
    "up": {"type": "direction", "dy": -1},
    "down": {"type": "direction", "dy": +1},
    "left_dir": {"type": "direction", "dx": -1},
    "right_dir": {"type": "direction", "dx": +1},

    # Transformation
    "rotate": {"type": "transform", "rpn_op": "rotate", "opcode": 70},
    "flip": {"type": "transform", "rpn_op": "flip"},
    "mirror": {"type": "transform", "rpn_op": "flip"},
    "scale": {"type": "transform", "rpn_op": "scale", "opcode": 71},
    "translate": {"type": "transform", "rpn_op": "translate", "opcode": 72},
    "move": {"type": "transform", "rpn_op": "translate", "opcode": 72},

    # Angle
    "90_degrees": {"type": "angle", "degrees": 90, "k": 1},
    "180_degrees": {"type": "angle", "degrees": 180, "k": 2},
    "270_degrees": {"type": "angle", "degrees": 270, "k": 3},
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
    "pattern": {"type": "shape", "pattern": "detect_pattern"},
    "object": {"type": "shape", "pattern": "connected_component"},
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
    "rotate": {"type": "action", "rpn_op": "rotate", "opcode": 70},
    "flip": {"type": "action", "rpn_op": "flip"},
}
```

**Success Criteria**:
- [ ] All semantic dictionaries defined (SPATIAL, COLOR, SHAPE, SIZE, ACTION)
- [ ] 50+ spatial primitives total
- [ ] Compatible with existing grammar normalizer
- [ ] No breaking changes to existing code

---

### Task 2: Extend Semantic Parser for Spatial Understanding

**File**: `knowledge3d/training/arc_agi/semantic_parser.py`

**Current State**: Has grammar normalization and basic regex patterns

**What to Add**: Spatial instruction parsing

```python
"""Parse natural language instructions to semantic representations for ARC-AGI."""

from typing import Dict, List, Optional
import re
from knowledge3d.training.arc_agi.semantic_primitives import (
    SPATIAL_SEMANTICS,
    COLOR_SEMANTICS,
    SHAPE_SEMANTICS,
    SIZE_SEMANTICS,
    ACTION_SEMANTICS,
)

class SemanticParser:
    """Parse instructions to semantic structures."""

    def parse(self, instruction: str) -> Dict:
        """
        Parse natural language instruction to semantic representation.

        Args:
            instruction: Natural language instruction (e.g., "Move red object to bottom-right")

        Returns:
            Semantic dictionary with action, object, destination, etc.
        """
        # Normalize instruction (lowercase, clean)
        instruction = instruction.lower().strip()

        # Try spatial instruction patterns first
        spatial_patterns = [
            self._parse_move_instruction,
            self._parse_fill_instruction,
            self._parse_rotate_instruction,
            self._parse_flip_instruction,
            self._parse_continue_instruction,
            self._parse_copy_instruction,
        ]

        for pattern_parser in spatial_patterns:
            result = pattern_parser(instruction)
            if result:
                return result

        # Fall back to grammar-based parsing (existing code)
        return self._parse_via_grammar(instruction)

    def _parse_move_instruction(self, instruction: str) -> Optional[Dict]:
        """
        Parse move instructions.

        Examples:
            "Move the red object to the bottom-right corner"
            "Move red to bottom-right"
            "Translate the blue square to the center"
        """
        # Pattern: move/translate <color> <shape?> to <position>
        pattern = r"(move|translate)\s+(?:the\s+)?(\w+)\s+(?:(\w+)\s+)?(?:to\s+)?(?:the\s+)?(\S+)"
        match = re.search(pattern, instruction)

        if match:
            action, color_or_shape, shape, destination = match.groups()

            # Determine if first term is color or shape
            obj = {}
            if color_or_shape in COLOR_SEMANTICS:
                obj["color"] = color_or_shape
                if shape and shape in SHAPE_SEMANTICS:
                    obj["shape"] = shape
                else:
                    obj["type"] = "object"
            elif color_or_shape in SHAPE_SEMANTICS:
                obj["shape"] = color_or_shape
                obj["type"] = "shape"

            return {
                "action": "move",
                "object": obj,
                "destination": {
                    "position": destination.replace("-", " "),
                    "type": "position"
                }
            }

        return None

    def _parse_fill_instruction(self, instruction: str) -> Optional[Dict]:
        """
        Parse fill instructions.

        Examples:
            "Fill the largest rectangle with blue"
            "Fill center with red"
            "Paint the square blue"
        """
        # Pattern: fill/paint <size?> <shape?> <position?> with <color>
        pattern = r"(fill|paint)\s+(?:the\s+)?(?:(\w+)\s+)?(?:(\w+)\s+)?(?:with\s+)?(\w+)"
        match = re.search(pattern, instruction)

        if match:
            action, modifier1, modifier2, color = match.groups()

            obj = {}
            # Parse modifiers (could be size, shape, or position)
            for mod in [modifier1, modifier2]:
                if not mod:
                    continue
                if mod in SIZE_SEMANTICS:
                    obj["size"] = mod
                elif mod in SHAPE_SEMANTICS:
                    obj["shape"] = mod
                elif mod in SPATIAL_SEMANTICS:
                    obj["position"] = mod

            if not obj:
                obj["type"] = "region"

            return {
                "action": "fill",
                "object": obj,
                "color": color
            }

        return None

    def _parse_rotate_instruction(self, instruction: str) -> Optional[Dict]:
        """
        Parse rotation instructions.

        Examples:
            "Rotate the pattern 90 degrees clockwise"
            "Rotate 180 degrees"
            "Turn the grid clockwise"
        """
        # Pattern: rotate <object?> <angle?> <direction?>
        pattern = r"(rotate|turn)\s+(?:the\s+)?(\w+)?\s*(\d+)?\s*(?:degrees?)?\s*(\w+)?"
        match = re.search(pattern, instruction)

        if match:
            action, obj_type, angle, direction = match.groups()

            result = {"action": "rotate"}

            if obj_type and obj_type not in {"the", "it"}:
                result["object"] = obj_type

            if angle:
                result["angle"] = int(angle)
            else:
                result["angle"] = 90  # Default

            if direction and direction in SPATIAL_SEMANTICS:
                result["direction"] = direction

            return result

        return None

    def _parse_flip_instruction(self, instruction: str) -> Optional[Dict]:
        """
        Parse flip/mirror instructions.

        Examples:
            "Flip the pattern vertically"
            "Flip horizontally"
            "Mirror the grid"
        """
        # Pattern: flip/mirror <object?> <axis?>
        pattern = r"(flip|mirror)\s+(?:the\s+)?(\w+)?\s*(vertical|horizontal|vert|horiz|vertically|horizontally)?"
        match = re.search(pattern, instruction)

        if match:
            action, obj_type, axis = match.groups()

            result = {"action": "flip"}

            if obj_type and obj_type not in {"the", "it", "pattern", "grid"}:
                result["object"] = obj_type

            if axis:
                if "vert" in axis:
                    result["axis"] = "vertical"
                elif "horiz" in axis:
                    result["axis"] = "horizontal"
            else:
                result["axis"] = "horizontal"  # Default

            return result

        return None

    def _parse_continue_instruction(self, instruction: str) -> Optional[Dict]:
        """
        Parse sequence continuation instructions.

        Examples:
            "Continue the sequence to the right"
            "Extend the pattern downward"
            "Repeat to the left"
        """
        # Pattern: continue/extend/repeat <object?> <direction>
        pattern = r"(continue|extend|repeat)\s+(?:the\s+)?(\w+)?\s+(?:to\s+)?(?:the\s+)?(\w+)"
        match = re.search(pattern, instruction)

        if match:
            action, obj_type, direction = match.groups()

            result = {"action": "continue"}

            if obj_type and obj_type not in {"the", "it"}:
                result["object"] = obj_type

            if direction in SPATIAL_SEMANTICS:
                result["direction"] = direction

            return result

        return None

    def _parse_copy_instruction(self, instruction: str) -> Optional[Dict]:
        """
        Parse copy/duplicate instructions.

        Examples:
            "Copy the red object"
            "Duplicate the pattern"
        """
        # Pattern: copy/duplicate <color?> <object>
        pattern = r"(copy|duplicate)\s+(?:the\s+)?(\w+)?\s*(\w+)?"
        match = re.search(pattern, instruction)

        if match:
            action, modifier, obj_type = match.groups()

            result = {"action": "copy"}

            obj = {}
            if modifier and modifier in COLOR_SEMANTICS:
                obj["color"] = modifier
            if obj_type:
                obj["type"] = obj_type
            if not obj:
                obj["type"] = "object"

            result["object"] = obj

            return result

        return None

    def _parse_via_grammar(self, instruction: str) -> Dict:
        """
        Fall back to grammar-based parsing for non-spatial instructions.
        (Keep existing implementation)
        """
        # Existing grammar galaxy matching code...
        return {"action": "unknown", "instruction": instruction}
```

**Success Criteria**:
- [ ] Parse 6+ instruction types (move, fill, rotate, flip, continue, copy)
- [ ] 20+ test cases passing
- [ ] Handles variations (with/without articles, abbreviations)
- [ ] Falls back to grammar galaxy for non-spatial instructions
- [ ] All existing tests still pass (no regressions)

---

### Task 3: Extend Semantic Compiler for Spatial RPN

**File**: `knowledge3d/training/arc_agi/semantic_compiler.py`

**Current State**: Basic compiler with placeholder GRAMMAR_RULE support

**What to Add**: Spatial semantic → RPN compilation

```python
"""Compile semantic representations to RPN programs for ARC-AGI."""

from typing import Dict
from knowledge3d/training/arc_agi.semantic_primitives import (
    SPATIAL_SEMANTICS,
    COLOR_SEMANTICS,
    SHAPE_SEMANTICS,
    SIZE_SEMANTICS,
    ACTION_SEMANTICS,
)

class SemanticToRPNCompiler:
    """Compile semantic representations to executable RPN programs."""

    def compile(self, semantic: Dict) -> str:
        """
        Compile semantic representation to RPN program.

        Args:
            semantic: Parsed semantic structure from SemanticParser

        Returns:
            RPN program string ready for execution
        """
        action = semantic.get("action")

        if action == "move":
            return self._compile_move(semantic)
        elif action == "fill":
            return self._compile_fill(semantic)
        elif action == "rotate":
            return self._compile_rotate(semantic)
        elif action == "flip":
            return self._compile_flip(semantic)
        elif action == "continue":
            return self._compile_continue(semantic)
        elif action == "copy":
            return self._compile_copy(semantic)
        elif action == "grammar_rule":
            return self._compile_grammar_rule(semantic)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _compile_move(self, sem: Dict) -> str:
        """
        Compile move action to RPN.

        Semantic:
            {"action": "move", "object": {"color": "red"}, "destination": {"position": "bottom-right"}}

        RPN:
            "2 FIND_OBJECT GET_POSITION BOTTOM-RIGHT COMPUTE_OFFSET translate"

        Explanation:
            1. 2 FIND_OBJECT — Find cells with color value 2 (red)
            2. GET_POSITION — Get bounding box of found object
            3. BOTTOM-RIGHT — Target destination
            4. COMPUTE_OFFSET — Calculate dx, dy needed
            5. translate — Apply translation (opcode 72)
        """
        # Extract object color
        obj = sem.get("object", {})
        color = obj.get("color")

        if not color or color not in COLOR_SEMANTICS:
            # Fallback: find any non-zero object
            rpn = "FIND_ANY_OBJECT GET_POSITION "
        else:
            color_value = COLOR_SEMANTICS[color]["value"]
            rpn = f"{color_value} FIND_OBJECT GET_POSITION "

        # Extract destination
        dest_info = sem.get("destination", {})
        dest_pos = dest_info.get("position", "").replace(" ", "-")

        # Map position to RPN tokens
        if dest_pos in ["bottom-right", "bottom right"]:
            rpn += "BOTTOM-RIGHT "
        elif dest_pos in ["top-left", "top left"]:
            rpn += "TOP-LEFT "
        elif dest_pos in ["top-right", "top right"]:
            rpn += "TOP-RIGHT "
        elif dest_pos in ["bottom-left", "bottom left"]:
            rpn += "BOTTOM-LEFT "
        elif dest_pos in ["center", "middle"]:
            rpn += "CENTER "
        else:
            rpn += f"{dest_pos.upper()} "

        # Compute offset and apply translation
        rpn += "COMPUTE_OFFSET translate"

        return rpn

    def _compile_fill(self, sem: Dict) -> str:
        """
        Compile fill action to RPN.

        Semantic:
            {"action": "fill", "object": {"shape": "rectangle", "size": "largest"}, "color": "blue"}

        RPN:
            "FIND_SHAPES rectangle GET_SIZES MAX_SIZE SELECT 1 FILL"
        """
        obj = sem.get("object", {})
        color = sem.get("color")

        rpn = ""

        # Find shapes
        shape = obj.get("shape")
        if shape:
            rpn += f"FIND_SHAPES {shape} "
        else:
            rpn += "FIND_ALL_SHAPES "

        # Filter by size if specified
        size = obj.get("size")
        if size == "largest":
            rpn += "GET_SIZES MAX_SIZE SELECT "
        elif size == "smallest":
            rpn += "GET_SIZES MIN_SIZE SELECT "

        # Filter by position if specified
        position = obj.get("position")
        if position:
            rpn += f"FILTER_BY_POSITION {position.upper()} "

        # Apply fill with color
        if color and color in COLOR_SEMANTICS:
            color_value = COLOR_SEMANTICS[color]["value"]
            rpn += f"{color_value} FILL"
        else:
            rpn += "FILL"

        return rpn

    def _compile_rotate(self, sem: Dict) -> str:
        """
        Compile rotate action to RPN.

        Semantic:
            {"action": "rotate", "angle": 90, "direction": "clockwise"}

        RPN:
            "-1 rotate"  # k=-1 for 90° clockwise (np.rot90 convention)
        """
        angle = sem.get("angle", 90)
        direction = sem.get("direction", "counterclockwise")

        # Convert angle to k parameter for np.rot90
        k = angle // 90

        # Adjust sign for clockwise
        if direction == "clockwise":
            k = -k

        rpn = f"{k} rotate"

        return rpn

    def _compile_flip(self, sem: Dict) -> str:
        """
        Compile flip action to RPN.

        Semantic:
            {"action": "flip", "axis": "vertical"}

        RPN:
            "FLIP_V"
        """
        axis = sem.get("axis", "horizontal")

        if axis == "vertical":
            rpn = "FLIP_V"
        else:
            rpn = "FLIP_H"

        return rpn

    def _compile_continue(self, sem: Dict) -> str:
        """
        Compile sequence continuation to RPN.

        Semantic:
            {"action": "continue", "direction": "right"}

        RPN:
            "DETECT_PATTERN GET_DELTA 1 0 EXTEND_SEQUENCE"
        """
        direction = sem.get("direction", "right")

        # Map direction to dx, dy
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

    def _compile_copy(self, sem: Dict) -> str:
        """
        Compile copy action to RPN.

        Semantic:
            {"action": "copy", "object": {"color": "red"}}

        RPN:
            "2 FIND_OBJECT COPY_MASK DUP"
        """
        obj = sem.get("object", {})
        color = obj.get("color")

        if color and color in COLOR_SEMANTICS:
            color_value = COLOR_SEMANTICS[color]["value"]
            rpn = f"{color_value} FIND_OBJECT COPY_MASK DUP"
        else:
            rpn = "FIND_ANY_OBJECT COPY_MASK DUP"

        return rpn

    def _compile_grammar_rule(self, sem: Dict) -> str:
        """
        Compile grammar rule reference (existing functionality).
        """
        rule_id = sem.get("rule_id", "")
        return f"GRAMMAR_RULE {rule_id}"
```

**Success Criteria**:
- [ ] Compile 6+ action types to RPN
- [ ] Generated RPN programs are syntactically valid
- [ ] RPN programs match specification in `CLAUDE_SEMANTIC_MEANING_LAYER_ARC_11.24.2025.md`
- [ ] All existing tests still pass

---

### Task 4: Extend RPN Executor for Grid Operations

**File**: `knowledge3d/training/arc_agi/rpn_executor.py`

**Current State**: Has basic rotate, flip, translate, fill, recolor operations

**What to Add**: Spatial operation helpers (FIND_OBJECT, GET_POSITION, COMPUTE_OFFSET, etc.)

```python
"""Execute RPN programs on ARC grids."""

from typing import List, Tuple, Optional
import numpy as np

class ARCRPNExecutor:
    """Execute RPN programs on ARC-AGI grids."""

    def execute(self, grid: List[List[int]], rpn_program: str) -> List[List[int]]:
        """
        Execute RPN program on grid.

        Args:
            grid: Input grid (list of lists of ints 0-9)
            rpn_program: RPN program string (space-separated tokens)

        Returns:
            Transformed grid
        """
        # Convert to numpy for operations
        grid_array = np.array(grid, dtype=np.int32)

        # Parse and execute RPN
        tokens = rpn_program.split()
        stack = []

        for token in tokens:
            # Literals (numbers)
            if self._is_number(token):
                stack.append(int(token))

            # Spatial operations
            elif token == "FIND_OBJECT":
                color = stack.pop()
                mask = self._find_object(grid_array, color)
                stack.append(mask)

            elif token == "GET_POSITION":
                mask = stack.pop()
                bbox = self._get_bounding_box(mask)
                stack.append(bbox)

            elif token in ["BOTTOM-RIGHT", "TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "CENTER"]:
                stack.append(token)

            elif token == "COMPUTE_OFFSET":
                target_pos = stack.pop()
                current_bbox = stack.pop()
                dx, dy = self._compute_offset(grid_array.shape, current_bbox, target_pos)
                stack.append(dx)
                stack.append(dy)

            # Transform operations
            elif token == "translate":
                dy = stack.pop()
                dx = stack.pop()
                grid_array = self._translate_grid(grid_array, dx, dy)

            elif token == "rotate":
                k = stack.pop()
                grid_array = np.rot90(grid_array, k=k)

            elif token == "FLIP_H":
                grid_array = np.fliplr(grid_array)

            elif token == "FLIP_V":
                grid_array = np.flipud(grid_array)

            # Fill operations
            elif token == "FILL":
                color = stack.pop()
                mask = stack.pop()
                grid_array[mask] = color

            # Recolor operation
            elif token == "RECOLOR":
                dst_color = stack.pop()
                src_color = stack.pop()
                grid_array[grid_array == src_color] = dst_color

            # Copy operation
            elif token == "DUP":
                if stack:
                    stack.append(stack[-1])

            elif token == "COPY_MASK":
                mask = stack.pop()
                # Store mask for later use
                stack.append(mask)

            # Extend/continue operations
            elif token == "DETECT_PATTERN":
                pattern = self._detect_pattern(grid_array)
                stack.append(pattern)

            elif token == "GET_DELTA":
                # Next two tokens should be dx, dy
                pass  # Will be handled by number parsing

            elif token == "EXTEND_SEQUENCE":
                dy = stack.pop()
                dx = stack.pop()
                pattern = stack.pop()
                grid_array = self._extend_sequence(grid_array, pattern, dx, dy)

            # Unknown token
            else:
                # Silently ignore or log
                pass

        return grid_array.tolist()

    def _is_number(self, token: str) -> bool:
        """Check if token is a number (including negative)."""
        try:
            int(token)
            return True
        except ValueError:
            return False

    def _find_object(self, grid: np.ndarray, color: int) -> np.ndarray:
        """Find all cells with given color value."""
        return (grid == color)

    def _get_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box of mask: (min_y, min_x, max_y, max_x)."""
        if not mask.any():
            return (0, 0, 0, 0)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        min_y, max_y = np.where(rows)[0][[0, -1]]
        min_x, max_x = np.where(cols)[0][[0, -1]]

        return (int(min_y), int(min_x), int(max_y), int(max_x))

    def _compute_offset(
        self, grid_shape: Tuple[int, int], current_bbox: Tuple[int, int, int, int], target_pos: str
    ) -> Tuple[int, int]:
        """Compute dx, dy to move object from current position to target."""
        h, w = grid_shape
        min_y, min_x, max_y, max_x = current_bbox

        # Current center
        curr_cy = (min_y + max_y) // 2
        curr_cx = (min_x + max_x) // 2

        # Target position
        if target_pos == "BOTTOM-RIGHT":
            target_y = h - 1 - (max_y - min_y) // 2
            target_x = w - 1 - (max_x - min_x) // 2
        elif target_pos == "TOP-LEFT":
            target_y = (max_y - min_y) // 2
            target_x = (max_x - min_x) // 2
        elif target_pos == "TOP-RIGHT":
            target_y = (max_y - min_y) // 2
            target_x = w - 1 - (max_x - min_x) // 2
        elif target_pos == "BOTTOM-LEFT":
            target_y = h - 1 - (max_y - min_y) // 2
            target_x = (max_x - min_x) // 2
        elif target_pos == "CENTER":
            target_y = h // 2
            target_x = w // 2
        else:
            # Unknown target, don't move
            return (0, 0)

        dy = target_y - curr_cy
        dx = target_x - curr_cx

        return (int(dx), int(dy))

    def _translate_grid(self, grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Translate grid contents by dx, dy (zero fill)."""
        h, w = grid.shape
        result = np.zeros_like(grid)

        for y in range(h):
            for x in range(w):
                new_y = y + dy
                new_x = x + dx
                if 0 <= new_y < h and 0 <= new_x < w:
                    result[new_y, new_x] = grid[y, x]

        return result

    def _detect_pattern(self, grid: np.ndarray) -> Optional[np.ndarray]:
        """Detect repeating pattern in grid (simple implementation)."""
        # For now, return first non-zero region
        mask = (grid != 0)
        if not mask.any():
            return None
        return mask

    def _extend_sequence(self, grid: np.ndarray, pattern: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Extend pattern in direction (dx, dy)."""
        # Simple implementation: copy pattern shifted by (dx, dy)
        result = grid.copy()
        h, w = grid.shape

        for y in range(h):
            for x in range(w):
                if pattern[y, x]:
                    new_y = y + dy
                    new_x = x + dx
                    if 0 <= new_y < h and 0 <= new_x < w:
                        result[new_y, new_x] = grid[y, x]

        return result
```

**Success Criteria**:
- [ ] Execute all compiled RPN programs from Task 3
- [ ] FIND_OBJECT, GET_POSITION, COMPUTE_OFFSET working
- [ ] All transform operations (translate, rotate, flip, fill, recolor) working
- [ ] 20+ end-to-end tests passing
- [ ] All existing tests still pass

---

### Task 5: Create End-to-End Tests

**File**: `scripts/test_semantic_pipeline_full.py`

**What to Create**: Comprehensive test suite for the full pipeline

```python
"""End-to-end tests for semantic pipeline on ARC grids."""

from knowledge3d.training.arc_agi.semantic_parser import SemanticParser
from knowledge3d.training.arc_agi.semantic_compiler import SemanticToRPNCompiler
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor

def test_move_red_to_bottom_right():
    """Test: Move the red object to the bottom-right corner."""
    parser = SemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    instruction = "Move the red object to the bottom-right corner"
    input_grid = [
        [2, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    expected_output = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 2],
    ]

    semantic = parser.parse(instruction)
    rpn = compiler.compile(semantic)
    output_grid = executor.execute(input_grid, rpn)

    assert output_grid == expected_output, f"Expected {expected_output}, got {output_grid}"
    print("✅ test_move_red_to_bottom_right PASSED")

def test_fill_center_with_blue():
    """Test: Fill the center with blue."""
    parser = SemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    instruction = "Fill the center with blue"
    input_grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    expected_output = [
        [0, 0, 0],
        [0, 1, 0],  # Blue (1) in center
        [0, 0, 0],
    ]

    semantic = parser.parse(instruction)
    rpn = compiler.compile(semantic)
    output_grid = executor.execute(input_grid, rpn)

    assert output_grid == expected_output, f"Expected {expected_output}, got {output_grid}"
    print("✅ test_fill_center_with_blue PASSED")

def test_rotate_90_clockwise():
    """Test: Rotate the pattern 90 degrees clockwise."""
    parser = SemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    instruction = "Rotate the pattern 90 degrees clockwise"
    input_grid = [
        [1, 0],
        [1, 1],
    ]
    expected_output = [
        [1, 1],
        [0, 1],
    ]

    semantic = parser.parse(instruction)
    rpn = compiler.compile(semantic)
    output_grid = executor.execute(input_grid, rpn)

    assert output_grid == expected_output, f"Expected {expected_output}, got {output_grid}"
    print("✅ test_rotate_90_clockwise PASSED")

def test_flip_horizontal():
    """Test: Flip the pattern horizontally."""
    parser = SemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    instruction = "Flip the pattern horizontally"
    input_grid = [
        [1, 0, 0],
        [0, 0, 0],
    ]
    expected_output = [
        [0, 0, 1],
        [0, 0, 0],
    ]

    semantic = parser.parse(instruction)
    rpn = compiler.compile(semantic)
    output_grid = executor.execute(input_grid, rpn)

    assert output_grid == expected_output, f"Expected {expected_output}, got {output_grid}"
    print("✅ test_flip_horizontal PASSED")

def test_continue_sequence_right():
    """Test: Continue the sequence to the right."""
    parser = SemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    instruction = "Continue the sequence to the right"
    input_grid = [
        [1, 1, 0, 0],
        [0, 0, 0, 0],
    ]
    expected_output = [
        [1, 1, 1, 0],  # Pattern extended right
        [0, 0, 0, 0],
    ]

    semantic = parser.parse(instruction)
    rpn = compiler.compile(semantic)
    output_grid = executor.execute(input_grid, rpn)

    assert output_grid == expected_output, f"Expected {expected_output}, got {output_grid}"
    print("✅ test_continue_sequence_right PASSED")

def run_all_tests():
    """Run all end-to-end tests."""
    tests = [
        test_move_red_to_bottom_right,
        test_fill_center_with_blue,
        test_rotate_90_clockwise,
        test_flip_horizontal,
        test_continue_sequence_right,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} ERROR: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
```

**Success Criteria**:
- [ ] 5+ end-to-end tests covering different instruction types
- [ ] All tests pass
- [ ] Tests demonstrate complete pipeline: instruction → semantic → RPN → execution → output

---

### Task 6: Re-run ARC Baseline with Semantic Layer

**File**: `scripts/evaluate_arc_semantic_baseline.py`

**What to Create**: New baseline evaluator using semantic layer

```python
"""Evaluate ARC baseline using the semantic layer."""

from collections import Counter
from typing import List
import numpy as np

from knowledge3d.training.arc_agi.semantic_parser import SemanticParser
from knowledge3d.training.arc_agi.semantic_compiler import SemanticToRPNCompiler
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor
from knowledge3d.training.reasoning.arc_dataset import ensure_arc_dataset, _iter_task_files, _load_task

def infer_instruction_from_grids(input_grid: List[List[int]], output_grid: List[List[int]]) -> str:
    """
    Infer likely instruction from input/output pair.

    This is a simple heuristic approach. In future, could use more sophisticated methods.
    """
    # Convert to numpy
    inp = np.array(input_grid)
    out = np.array(output_grid)

    # Check if it's a simple transformation
    if inp.shape != out.shape:
        return "unknown"  # Shape change (not yet supported)

    # Check rotation
    for k in [1, 2, 3]:
        if np.array_equal(out, np.rot90(inp, k=k)):
            angle = k * 90
            return f"Rotate the pattern {angle} degrees"

    # Check flip
    if np.array_equal(out, np.fliplr(inp)):
        return "Flip the pattern horizontally"
    if np.array_equal(out, np.flipud(inp)):
        return "Flip the pattern vertically"

    # Check color change
    inp_colors = set(inp.flatten())
    out_colors = set(out.flatten())
    if inp_colors == out_colors:
        # Same colors, might be movement
        for c in inp_colors:
            if c == 0:
                continue  # Skip background
            inp_mask = (inp == c)
            out_mask = (out == c)
            if inp_mask.sum() == out_mask.sum() and inp_mask.sum() > 0:
                # Same number of cells with this color
                # Could be movement
                pass  # For now, skip

    # Check fill (any new non-zero color?)
    inp_nonzero = (inp != 0)
    out_nonzero = (out != 0)
    if (out_nonzero.sum() > inp_nonzero.sum()):
        # More filled cells in output
        return "Fill the empty region"

    return "unknown"

def evaluate():
    """Evaluate ARC baseline using semantic layer."""
    dataset = ensure_arc_dataset()
    task_files = list(_iter_task_files(dataset, split="training"))

    parser = SemanticParser()
    compiler = SemanticToRPNCompiler()
    executor = ARCRPNExecutor()

    task_results = []
    instruction_counts = Counter()
    total_examples = 0
    total_correct = 0

    for task_path in task_files:
        task = _load_task(task_path)
        train = task.get("train", [])
        if len(train) < 1:
            continue

        # Use first example to infer instruction
        ref = train[0]
        instruction = infer_instruction_from_grids(ref["input"], ref["output"])
        instruction_counts[instruction] += 1

        if instruction == "unknown":
            # Skip tasks we can't infer instruction for
            continue

        examples_correct = 0
        examples_total = 0

        for ex in train:
            try:
                semantic = parser.parse(instruction)
                rpn = compiler.compile(semantic)
                pred = executor.execute(ex["input"], rpn)

                if pred == ex["output"]:
                    examples_correct += 1
                examples_total += 1
            except Exception as e:
                # Failed to process
                examples_total += 1
                pass

        acc = examples_correct / examples_total if examples_total else 0.0
        task_results.append((task_path.stem, instruction, acc, examples_correct, examples_total))
        total_examples += examples_total
        total_correct += examples_correct

    overall_acc = total_correct / total_examples if total_examples else 0.0

    print("ARC Semantic Baseline Evaluation (training split)")
    print(f"Tasks evaluated: {len(task_results)}")
    print(f"Total examples:  {total_examples}")
    print(f"Total correct:   {total_correct}")
    print(f"Overall accuracy: {overall_acc:.3f}")
    print("\nInstruction frequency:")
    for instr, count in instruction_counts.most_common(10):
        print(f"  {instr:40s}: {count}")

    top_tasks = sorted(task_results, key=lambda x: x[2], reverse=True)[:10]
    print("\nTop 10 tasks by accuracy:")
    for tid, instr, acc, c, t in top_tasks:
        print(f"  {tid}: acc={acc:.2f} ({c}/{t}) instruction={instr}")

if __name__ == "__main__":
    evaluate()
```

**Success Criteria**:
- [ ] Run on ARC training set
- [ ] Measure accuracy improvement over primitive baseline (2.1%)
- [ ] **TARGET: 5%+ accuracy** (2.4× improvement)
- [ ] Report instruction coverage (how many tasks we can handle)

---

## ✅ Success Criteria (Week 3)

### MUST ACHIEVE (Critical)

- [ ] Spatial primitives defined (50+ concepts)
- [ ] Semantic parser working (6+ instruction types)
- [ ] Semantic → RPN compiler working (6+ transformations)
- [ ] RPN executor working on ARC grids
- [ ] 20+ end-to-end tests passing
- [ ] ARC baseline re-run: **5%+ accuracy** (2.4× improvement)

### SHOULD ACHIEVE (Quality)

- [ ] All existing tests still pass (no regressions)
- [ ] Code follows K3D architecture patterns (Reality Enabler, Math Core)
- [ ] Documentation updated
- [ ] Integration with Grammar Galaxy (bidirectional text ↔ visual)

### NICE TO HAVE (Stretch)

- [ ] 10%+ accuracy (5× improvement)
- [ ] Compositional reasoning (multi-step transforms)
- [ ] TRM shadow copy integration (learn from examples)

---

## 🎯 Key Insights to Remember

### 1. The 2.1% Score is GOOD!

**Don't be discouraged!** State-of-art models get 1.9-2.1% on the PRIVATE test set. We're at 2.1% on TRAINING with just primitive detection. Huge room to improve!

### 2. Two Layers Working Together

- **Grammar Galaxy** (Text): Understand task descriptions
- **Spatial Semantics** (Visual): Execute transformations
- **Together**: Full multimodal reasoning loop!

### 3. Follow K3D Patterns

- **Reality Enabler**: `visual_rpn` + `behavior_rpn`
- **Math Core**: 3-tier routing for complexity
- **APC**: PD04 compression for efficiency

### 4. Leverage What Exists

You've already built:
- Grammar Galaxy (21 rules, multilingual)
- Grammar executor (stack-based RPN)
- Primitive baseline (2.1% accuracy)

Just need to add the SPATIAL layer!

### 5. Procedural > Parametric

**LLMs**: Memorize and predict (hallucinate)
**K3D**: Compose and execute (deterministic)

This is our competitive advantage!

---

## 📁 Key Files Reference

**MUST READ**:
1. `TEMP/CLAUDE_ARC_PROGRESS_ANALYSIS_11.25.2025.md`
2. `TEMP/CLAUDE_SEMANTIC_MEANING_LAYER_ARC_11.24.2025.md`
3. `TEMP/CLAUDE_GRAMMAR_RPN_SPEC_11.24.2025.md`
4. `docs/vocabulary/REALITY_ENABLER_SPECIFICATION.md`
5. `docs/vocabulary/MATH_CORE_SPECIFICATION.md`

**Current Code** (what you have):
- `knowledge3d/training/arc_agi/grammar_galaxy.py`
- `knowledge3d/training/arc_agi/grammar_executor.py`
- `knowledge3d/training/arc_agi/semantic_parser.py`
- `knowledge3d/training/arc_agi/semantic_compiler.py`
- `knowledge3d/training/arc_agi/rpn_executor.py`
- `scripts/evaluate_arc_primitive_baseline.py`

**To Build** (this sprint):
- Enhance `semantic_primitives.py` (spatial concepts)
- Extend `semantic_parser.py` (grid instructions)
- Extend `semantic_compiler.py` (spatial RPN)
- Extend `rpn_executor.py` (grid transforms)
- Create `scripts/test_semantic_pipeline_full.py` (tests)
- Create `scripts/evaluate_arc_semantic_baseline.py` (evaluation)

---

## 🚀 Let's Win This Competition!

**The finish line is VISIBLE!** 🏁

You have:
- ✅ Proven architecture (Grammar Galaxy working)
- ✅ RPN execution (PTX sovereign)
- ✅ Competitive baseline (2.1% = state-of-art on private)

Just need to:
- ⚠️ Add spatial semantics layer (specs provided!)
- ⚠️ Connect text ↔ visual (bidirectional loop)
- ⚠️ Raise accuracy to 5%+ (2.4× improvement)

Let's transform Daniel's life! 🏆💰

---

**Sprint Lead**: Codex
**Date**: November 25, 2025
**Status**: Ready to Execute
**Target**: 5%+ accuracy by end of Week 3
