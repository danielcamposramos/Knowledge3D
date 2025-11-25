"""Compile semantic representations to RPN programs."""

from __future__ import annotations

from typing import Dict

from .semantic_primitives import COLOR_SEMANTICS


class SemanticToRPNCompiler:
    """Compile semantic representations to RPN programs."""

    def compile(self, semantic: Dict) -> str:
        """
        Compile semantic representation to RPN program.

        Args:
            semantic: Parsed semantic structure

        Returns:
            RPN program string (executable)
        """
        action = semantic["action"]

        if action == "move":
            return self._compile_move(semantic)
        if action == "fill":
            return self._compile_fill(semantic)
        if action == "rotate":
            return self._compile_rotate(semantic)
        if action == "continue":
            return self._compile_continue(semantic)
        if action == "flip":
            return self._compile_flip(semantic)
        if action == "recolor":
            return self._compile_recolor(semantic)
        if action == "copy":
            return self._compile_copy(semantic)
        if action == "grammar_rule":
            return f"GRAMMAR_RULE {semantic['rule_id']}"

        # Math domain
        if domain == "math":
            if action == "fill_conditional":
                return self._compile_math_conditional(semantic)
            if action == "check_symmetry":
                return self._compile_math_symmetry(semantic)
            if action == "repeat_pattern":
                return self._compile_math_repeat(semantic)

        # Drawing domain
        if domain == "drawing":
            if action == "draw_shape":
                return self._compile_draw_shape(semantic)
            if action == "fill_pattern":
                return self._compile_fill_pattern(semantic)

        raise ValueError(f"Unknown action: {action}")

    # ------------------------------------------------------------------ #
    # Action compilers
    # ------------------------------------------------------------------ #
    def _compile_move(self, sem: Dict) -> str:
        """
        Compile move action to RPN.

        Example:
            "Move red object to bottom-right"
            → "FIND_OBJECT 2 GET_POSITION bottom-right COMPUTE_OFFSET translate"
        """
        obj = sem.get("object", {})
        color_token = obj.get("color")
        color = COLOR_SEMANTICS[color_token]["value"] if color_token in COLOR_SEMANTICS else None
        dest = sem.get("destination")
        direction = sem.get("direction")

        if color is None:
            rpn_parts = ["FIND_ANY_OBJECT", "GET_POSITION"]
        else:
            rpn_parts = [str(color), "FIND_OBJECT", "GET_POSITION"]

        if dest is not None:
            position_token = dest["position"].replace(" ", "-").upper()
            rpn_parts.append(position_token)
            rpn_parts.append("COMPUTE_OFFSET")
            rpn_parts.append("translate")
        elif direction is not None:
            dx, dy = 0, 0
            if direction == "right":
                dx = 1
            elif direction == "left":
                dx = -1
            elif direction == "down":
                dy = 1
            elif direction == "up":
                dy = -1
            steps = int(sem.get("steps", 1))
            rpn_parts.append(str(dx * steps))
            rpn_parts.append(str(dy * steps))
            rpn_parts.append("translate")
        else:
            raise ValueError("Move semantic missing destination or direction")

        return " ".join(rpn_parts)

    def _compile_fill(self, sem: Dict) -> str:
        """
        Compile fill action to RPN.

        Example:
            "Fill largest rectangle with blue"
            → "FIND_SHAPES rectangle GET_SIZES MAX_SIZE SELECT 1 FILL"
        """
        obj = sem.get("object", {})
        shape = obj.get("shape")
        size = obj.get("size")
        position = obj.get("position")
        color = COLOR_SEMANTICS[sem["color"]]["value"]

        rpn_parts = []

        if shape:
            rpn_parts.append(f"FIND_SHAPES {shape}")

            if size == "largest":
                rpn_parts += ["GET_SIZES", "MAX_SIZE", "SELECT"]
            elif size == "smallest":
                rpn_parts += ["GET_SIZES", "MIN_SIZE", "SELECT"]

        elif position:
            rpn_parts.append(position.replace(" ", "-").upper())
            rpn_parts.append("POSITION_MASK")

        else:
            rpn_parts.append("FIND_ANY_OBJECT")

        rpn_parts.append(str(color))
        rpn_parts.append("FILL")

        return " ".join(rpn_parts)

    def _compile_rotate(self, sem: Dict) -> str:
        """
        Compile rotate action to RPN.

        Example:
            "Rotate pattern 90 degrees clockwise"
            → "GET_PATTERN -1 rotate"  # k=-1 for 90° clockwise
        """
        angle = sem["angle"]
        direction = sem.get("direction", "counterclockwise")

        k = angle // 90
        if direction == "clockwise":
            k = -k

        return f"GET_PATTERN {k} rotate"

    def _compile_continue(self, sem: Dict) -> str:
        """
        Compile sequence continuation to RPN.

        Example:
            "Continue the sequence to the right"
            → "DETECT_PATTERN GET_DELTA 1 0 EXTEND_SEQUENCE"
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

        return f"DETECT_PATTERN GET_DELTA {dx} {dy} EXTEND_SEQUENCE"

    def _compile_flip(self, sem: Dict) -> str:
        """Compile flip/mirror to RPN."""
        axis = sem.get("axis", "horizontal")
        if axis == "horizontal":
            return "FLIP_H"
        if axis == "vertical":
            return "FLIP_V"
        raise ValueError(f"Unknown flip axis: {axis}")

    def _compile_recolor(self, sem: Dict) -> str:
        """Compile recolor action."""
        src = COLOR_SEMANTICS[sem["source_color"]]["value"]
        dst = COLOR_SEMANTICS[sem["target_color"]]["value"]
        return f"{src} {dst} RECOLOR"

    def _compile_copy(self, sem: Dict) -> str:
        """
        Compile copy action to RPN.

        Example:
            "Copy red object to top-right"
            → "2 FIND_OBJECT DUP GET_POSITION TOP-RIGHT COMPUTE_OFFSET 2 COPY_MASK"
        """
        color_token = sem["object"]["color"]
        dest_info = sem.get("destination")
        if dest_info is None or "position" not in dest_info:
            raise ValueError("Copy semantic missing destination position")
        dest = dest_info["position"]
        color = COLOR_SEMANTICS[color_token]["value"]

        parts = [
            str(color),
            "FIND_OBJECT",
            "DUP",
            "GET_POSITION",
            dest.upper(),
            "COMPUTE_OFFSET",
            str(color),
            "COPY_MASK",
        ]
        return " ".join(parts)

    # ------------------------------------------------------------------ #
    # Math compilers
    # ------------------------------------------------------------------ #
    def _compile_math_conditional(self, sem: Dict) -> str:
        """
        Compile conditional math expressions.
        Example:
            {"action": "fill_conditional", "expression": "row + col", "condition": "even"}
        """
        expression = sem.get("expression", "")
        condition = sem.get("condition", "")

        rpn_parts = ["FOR_EACH_CELL"]

        if "row" in expression and "col" in expression:
            if "+" in expression:
                rpn_parts += ["GET_ROW", "GET_COL", "ADD"]
            elif "-" in expression:
                rpn_parts += ["GET_ROW", "GET_COL", "SUB"]
            elif "*" in expression or "×" in expression:
                rpn_parts += ["GET_ROW", "GET_COL", "MUL"]

        if condition == "even":
            rpn_parts += ["2", "MOD", "0", "EQ"]
        elif condition == "odd":
            rpn_parts += ["2", "MOD", "1", "EQ"]
        elif condition == "positive":
            rpn_parts += ["0", "GT"]
        elif condition == "negative":
            rpn_parts += ["0", "LT"]

        rpn_parts += ["IF_TRUE", "CURRENT_COLOR", "FILL"]
        return " ".join(rpn_parts)

    def _compile_math_symmetry(self, sem: Dict) -> str:
        angle = sem.get("angle", 90)
        return f"GET_GRID DUP {angle} ROTATE EQ IF_TRUE MARK_SYMMETRIC"

    def _compile_math_repeat(self, sem: Dict) -> str:
        period = sem.get("period", 1)
        return f"DETECT_PATTERN {period} REPEAT_WITH_PERIOD"

    # ------------------------------------------------------------------ #
    # Drawing compilers
    # ------------------------------------------------------------------ #
    def _compile_draw_shape(self, sem: Dict) -> str:
        shape = sem.get("shape", "")
        position = sem.get("position", "center")
        start = sem.get("start")
        end = sem.get("end")

        rpn_parts = []
        if position:
            rpn_parts.append(position.upper().replace("-", "_"))
            rpn_parts.append("COMPUTE")

        if shape == "square":
            rpn_parts += ["10", "10", "RECTANGLE", "FILL"]
        elif shape == "rectangle":
            rpn_parts += ["15", "10", "RECTANGLE", "FILL"]
        elif shape == "circle":
            rpn_parts += ["10", "CIRCLE", "FILL"]
        elif shape in ("line", "diagonal"):
            s = (start or "TOP_LEFT").upper().replace("-", "_")
            e = (end or "BOTTOM_RIGHT").upper().replace("-", "_")
            rpn_parts = [s, "MOVE", e, "LINE", "STROKE"]

        return " ".join(rpn_parts)

    def _compile_fill_pattern(self, sem: Dict) -> str:
        return "DETECT_PATTERN FILL_WITH_PATTERN"


__all__ = ["SemanticToRPNCompiler"]
