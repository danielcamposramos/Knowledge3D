"""Execute RPN programs on ARC grids."""

from __future__ import annotations

from typing import List

import numpy as np


class ARCRPNExecutor:
    """Execute RPN programs on ARC grids (numpy-based, CPU path)."""

    def execute(self, grid: List[List[int]], rpn_program: str) -> List[List[int]]:
        """
        Execute RPN program on grid.

        Args:
            grid: Input grid (list of lists)
            rpn_program: RPN program string

        Returns:
            Transformed grid (list of lists)
        """
        grid_array = np.array(grid, dtype=np.int32)
        tokens = rpn_program.split()
        stack = []
        idx = 0

        while idx < len(tokens):
            token = tokens[idx]
            if token == "rotate":
                k = int(stack.pop())
                grid_array = np.rot90(grid_array, k=k)

            elif token == "translate":
                dy = int(stack.pop())
                dx = int(stack.pop())
                grid_array = self._translate_grid(grid_array, dx, dy)

            elif token == "FOR_EACH_CELL":
                stack.append("CELL_ITERATION")

            elif token == "GET_ROW":
                stack.append("ROW")

            elif token == "GET_COL":
                stack.append("COL")

            elif token in ("ADD", "SUB", "MUL"):
                b = stack.pop()
                a = stack.pop()
                if token == "ADD":
                    stack.append(self._combine_math(a, b, "+"))
                elif token == "SUB":
                    stack.append(self._combine_math(a, b, "-"))
                elif token == "MUL":
                    stack.append(self._combine_math(a, b, "*"))

            elif token == "MOD":
                divisor = stack.pop()
                value = stack.pop()
                stack.append(self._combine_math(value, divisor, "%"))

            elif token in ("EQ", "GT", "LT"):
                b = stack.pop()
                a = stack.pop()
                op = {"EQ": "==", "GT": ">", "LT": "<"}[token]
                stack.append(self._combine_math(a, b, op))

            elif token == "IF_TRUE":
                condition = stack.pop()
                mode = stack.pop() if stack else None
                if mode == "CELL_ITERATION":
                    mask = self._evaluate_cell_condition(grid_array, condition)
                    stack.append(mask)
                else:
                    stack.append(condition)

            elif token == "CURRENT_COLOR":
                stack.append(1)

            elif token == "FILL":
                color = int(stack.pop())
                region = stack.pop()
                grid_array[region] = color

            elif token == "FIND_OBJECT":
                color = int(stack.pop())
                mask = grid_array == color
                stack.append(mask)

            elif token == "FIND_ANY_OBJECT":
                mask = grid_array != 0
                stack.append(mask)

            elif token == "DUP":
                if not stack:
                    raise RuntimeError("DUP on empty stack")
                stack.append(stack[-1])

            elif token == "RECOLOR":
                to_color = int(stack.pop())
                from_color = int(stack.pop())
                grid_array[grid_array == from_color] = to_color

            elif token == "COPY_MASK":
                color = int(stack.pop())
                dy = int(stack.pop())
                dx = int(stack.pop())
                mask = stack.pop()
                translated = self._translate_mask(mask, dx, dy)
                grid_array[translated] = color

            elif token == "FIND_SHAPES":
                # Simple heuristic: bounding-box mask of all non-zero cells.
                shape = tokens[idx + 1] if idx + 1 < len(tokens) else "shape"
                mask = self._find_shape_mask(grid_array, shape)
                stack.append(mask)
                idx += 1  # consumed shape token

            elif token == "GET_SIZES":
                # Push a list with a single size (area) for current mask.
                mask = stack.pop()
                size = int(mask.sum())
                stack.append([size])
                stack.append(mask)

            elif token in ("MAX_SIZE", "MIN_SIZE"):
                mask = stack.pop()
                # With single mask, selection is identity.
                stack.append(mask)

            elif token == "SELECT":
                # No-op for single mask selection.
                pass

            elif token == "GET_POSITION":
                mask = stack.pop()
                pos = self._get_centroid(mask)
                stack.append(pos)

            elif token == "POSITION_MASK":
                dest = stack.pop()
                mask = self._position_mask(grid_array.shape, dest)
                stack.append(mask)

            elif token == "CENTER":
                h, w = grid_array.shape
                stack.append((h // 2, w // 2))

            elif token == "COMPUTE":
                # Passthrough; position already on stack.
                pass

            elif token == "RECTANGLE":
                height = int(stack.pop())
                width = int(stack.pop())
                pos = stack.pop()
                mask = self._rectangle_mask(grid_array.shape, pos, width, height)
                stack.append(mask)

            elif token == "CIRCLE":
                radius = int(stack.pop())
                pos = stack.pop()
                mask = self._circle_mask(grid_array.shape, pos, radius)
                stack.append(mask)

            elif token in (
                "BOTTOM",
                "TOP",
                "LEFT",
                "RIGHT",
                "CENTER",
                "BOTTOM-RIGHT",
                "BOTTOM-LEFT",
                "TOP-RIGHT",
                "TOP-LEFT",
            ):
                stack.append(token.lower())

            elif token == "COMPUTE_OFFSET":
                dest = stack.pop()
                current = stack.pop()
                dx, dy = self._compute_offset(grid_array.shape, current, dest)
                stack.append(dx)
                stack.append(dy)

            elif token == "GET_PATTERN":
                # No-op placeholder: stack already has k; pass through.
                pass

            elif token == "DETECT_PATTERN":
                # Placeholder: push current grid as pattern reference.
                stack.append(grid_array.copy())

            elif token == "GET_DELTA":
                # Expect dx, dy literals next.
                dx = int(tokens[idx + 1]) if idx + 1 < len(tokens) else 0
                dy = int(tokens[idx + 2]) if idx + 2 < len(tokens) else 0
                stack.append(dx)
                stack.append(dy)
                idx += 2  # consumed dx, dy

            elif token == "EXTEND_SEQUENCE":
                dy = int(stack.pop())
                dx = int(stack.pop())
                pattern = stack.pop()
                translated = self._translate_grid(pattern, dx, dy)
                grid_array = np.where(translated != 0, translated, grid_array)

            elif token == "FLIP_H":
                grid_array = np.fliplr(grid_array)
            elif token == "FLIP_V":
                grid_array = np.flipud(grid_array)

            elif token.lstrip("-").isdigit():
                stack.append(int(token))

            elif token == "GRAMMAR_RULE":
                # Placeholder: grammar rules are handled upstream; no-op here.
                pass

            idx += 1

        return grid_array.tolist()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _translate_grid(self, grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Translate grid contents by dx, dy with zero fill."""
        result = np.zeros_like(grid)
        h, w = grid.shape
        for y in range(h):
            for x in range(w):
                ny = y + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    result[ny, nx] = grid[y, x]
        return result

    def _get_centroid(self, mask: np.ndarray) -> tuple[int, int]:
        """Compute centroid (y, x) of a boolean mask; default to center if empty."""
        if mask.sum() == 0:
            h, w = mask.shape
            return h // 2, w // 2
        ys, xs = np.nonzero(mask)
        return int(np.mean(ys)), int(np.mean(xs))

    def _compute_offset(self, shape: tuple[int, int], current: tuple[int, int], dest: str) -> tuple[int, int]:
        """Compute translation offset from current position to destination keyword."""
        h, w = shape
        cy, cx = current
        dest = dest.lower()

        if dest in ("center", "centre", "middle"):
            ty, tx = h // 2, w // 2
        elif dest == "bottom-right":
            ty, tx = h - 1, w - 1
        elif dest == "bottom-left":
            ty, tx = h - 1, 0
        elif dest == "top-right":
            ty, tx = 0, w - 1
        elif dest == "top-left":
            ty, tx = 0, 0
        elif dest == "bottom":
            ty, tx = h - 1, cx
        elif dest == "top":
            ty, tx = 0, cx
        elif dest == "right":
            ty, tx = cy, w - 1
        elif dest == "left":
            ty, tx = cy, 0
        else:
            # Default to no move if unknown
            ty, tx = cy, cx

        dy = ty - cy
        dx = tx - cx
        return dx, dy

    def _find_shape_mask(self, grid: np.ndarray, shape: str) -> np.ndarray:
        """
        Simple shape detector placeholder.

        For now, returns a mask covering non-zero cells (or bounding box interior
        for rectangles/borders).
        """
        mask = grid != 0
        if not mask.any():
            return mask

        ys, xs = np.nonzero(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        if shape in ("rectangle", "square", "border"):
            box = np.zeros_like(mask)
            box[y0 : y1 + 1, x0 : x1 + 1] = True
            return box
        return mask

    def _translate_mask(self, mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Translate a boolean mask by dx, dy."""
        result = np.zeros_like(mask, dtype=bool)
        h, w = mask.shape
        for y in range(h):
            for x in range(w):
                if not mask[y, x]:
                    continue
                ny = y + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    result[ny, nx] = True
        return result

    def _position_mask(self, shape: tuple[int, int], dest: str) -> np.ndarray:
        """Create a boolean mask for a single target position keyword."""
        h, w = shape
        mask = np.zeros((h, w), dtype=bool)
        dest = dest.lower()

        if dest in ("center", "centre", "middle"):
            ty, tx = h // 2, w // 2
        elif dest == "bottom-right":
            ty, tx = h - 1, w - 1
        elif dest == "bottom-left":
            ty, tx = h - 1, 0
        elif dest == "top-right":
            ty, tx = 0, w - 1
        elif dest == "top-left":
            ty, tx = 0, 0
        elif dest == "top":
            ty, tx = 0, w // 2
        elif dest == "bottom":
            ty, tx = h - 1, w // 2
        elif dest == "left":
            ty, tx = h // 2, 0
        elif dest == "right":
            ty, tx = h // 2, w - 1
        else:
            return mask

        mask[ty, tx] = True
        return mask

    def _rectangle_mask(self, shape: tuple[int, int], pos: tuple[int, int], width: int, height: int) -> np.ndarray:
        h, w = shape
        cy, cx = pos
        mask = np.zeros((h, w), dtype=bool)
        y1 = max(0, cy - height // 2)
        y2 = min(h, cy + height // 2)
        x1 = max(0, cx - width // 2)
        x2 = min(w, cx + width // 2)
        mask[y1:y2, x1:x2] = True
        return mask

    def _circle_mask(self, shape: tuple[int, int], pos: tuple[int, int], radius: int) -> np.ndarray:
        h, w = shape
        cy, cx = pos
        mask = np.zeros((h, w), dtype=bool)
        for y in range(h):
            for x in range(w):
                if (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2:
                    mask[y, x] = True
        return mask

    def _combine_math(self, a, b, op: str):
        """Combine two operands, keeping symbolic strings for per-cell eval."""
        if isinstance(a, str) or isinstance(b, str):
            return f"({a}){op}({b})"
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "%":
            return a % b
        if op == "==":
            return a == b
        if op == ">":
            return a > b
        if op == "<":
            return a < b
        return 0

    def _evaluate_cell_condition(self, grid: np.ndarray, condition: str) -> np.ndarray:
        """
        Evaluate condition expression for each cell.
        Example: "(ROW)+(COL)%2==(0)"
        """
        h, w = grid.shape
        mask = np.zeros((h, w), dtype=bool)
        for y in range(h):
            for x in range(w):
                expr = condition.replace("ROW", str(y)).replace("COL", str(x))
                try:
                    result = eval(expr)
                    mask[y, x] = bool(result)
                except Exception:
                    pass
        return mask


__all__ = ["ARCRPNExecutor"]
