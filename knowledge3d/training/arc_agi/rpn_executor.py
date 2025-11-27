"""Execute ARC RPN programs via the sovereign PTX drawing bridge."""

from __future__ import annotations

from typing import List, Sequence

from knowledge3d.cranium.bridges.drawing_bridge import DrawingBridge
from knowledge3d.cranium.ptx_runtime.math_core_pool import MathCorePool, get_global_math_core_pool
from knowledge3d.cranium.sovereign import loader


class ARCRPNExecutor:
    """
    PTX-backed RPN executor for ARC grids.

    Uses DrawingBridge to convert ARC grids to PTX drawing surfaces and execute
    drawing opcodes directly on GPU. No CPU fallbacks are allowed.
    """

    def __init__(self, *, pool: MathCorePool | None = None, instance_id: int | None = None) -> None:
        self.pool = pool or get_global_math_core_pool()
        self.instance_id = instance_id
        self._owns_instance = False
        self.ptx_success_count = 0
        self.ptx_fallback_count = 0  # Maintained for instrumentation interface
        # Operation codes mirrored in arc_grid_ops.cu
        self.OP_ROT90 = 0
        self.OP_ROT180 = 1
        self.OP_ROT270 = 2
        self.OP_FLIP_H = 3
        self.OP_FLIP_V = 4
        self.OP_TRANSLATE = 5
        self.OP_RECOLOR = 6

        if self.instance_id is None:
            # Allocate a dedicated core (Tier-1 default) for this executor.
            try:
                self.instance_id = self.pool.spawn_core(tier=1, reuse=True)
                self._owns_instance = True
            except Exception:
                # If allocation fails, allow engine to handle instance selection internally.
                self.instance_id = None

        # Grid ↔ surface converter; raises loudly until implemented.
        self.bridge = DrawingBridge()

    def execute(self, grid: Sequence[Sequence[int]], rpn_program: str) -> List[List[int]]:
        """
        Execute an ARC RPN program on a grid.

        Args:
            grid: List[List[int]] ARC grid.
            rpn_program: RPN string (e.g., "1 rotate", "FLIP_H", "3 5 RECOLOR").

        Returns:
            Transformed grid as List[List[int]].
        """
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        op, p1, p2, dst_w, dst_h = self._parse_program(rpn_program, width, height)

        # No-op shortcut (rotate 0)
        if op is None:
            return [[int(c) for c in row] for row in grid]

        src_surface, src_w, src_h = self.bridge.grid_to_surface(grid)
        dst_surface = None
        try:
            dst_surface = self.bridge.execute_on_surface(
                src_surface,
                src_w=src_w,
                src_h=src_h,
                dst_w=dst_w,
                dst_h=dst_h,
                op=op,
                p1=p1,
                p2=p2,
            )
            out = self.bridge.surface_to_grid(dst_surface, dst_w, dst_h)
            dst_surface = None  # freed in surface_to_grid
            self.ptx_success_count += 1
            return out
        finally:
            # Free source/destination surfaces if still allocated
            try:
                loader.gpu_free(src_surface)
            except Exception:
                pass
            if dst_surface is not None:
                try:
                    loader.gpu_free(dst_surface)
                except Exception:
                    pass

    def get_stats(self) -> str:
        """Return PTX vs fallback execution counts."""
        total = self.ptx_success_count + self.ptx_fallback_count
        if total == 0:
            return "PTX: 0/0 (0.0%)"
        rate = 100.0 * self.ptx_success_count / float(total)
        return f"PTX: {self.ptx_success_count}/{total} ({rate:.1f}%)"

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _parse_program(
        self, rpn_program: str, width: int, height: int
    ) -> tuple[int | None, int, int, int, int]:
        """
        Parse ARC RPN program into grid op code and params.

        Returns:
            (op_code or None for no-op, p1, p2, dst_w, dst_h)
        """
        tokens = rpn_program.strip().split()
        lower = [t.lower() for t in tokens]

        if "rotate" in lower:
            idx = lower.index("rotate")
            k = int(tokens[idx - 1]) if idx > 0 else 0
            k = k % 4
            if k == 0:
                return None, 0, 0, width, height
            if k == 1:
                return self.OP_ROT90, 0, 0, height, width
            if k == 2:
                return self.OP_ROT180, 0, 0, width, height
            return self.OP_ROT270, 0, 0, height, width

        if "flip_h" in lower or "FLIP_H" in tokens:
            return self.OP_FLIP_H, 0, 0, width, height
        if "flip_v" in lower or "FLIP_V" in tokens:
            return self.OP_FLIP_V, 0, 0, width, height

        if "TRANSLATE" in tokens or "translate" in lower:
            try:
                # Expect dx dy TRANSLATE
                if "translate" in lower:
                    idx = lower.index("translate")
                else:
                    idx = tokens.index("TRANSLATE")
                dx = int(float(tokens[idx - 2]))
                dy = int(float(tokens[idx - 1]))
            except Exception as exc:
                raise ValueError(f"Invalid TRANSLATE program: {rpn_program}") from exc
            return self.OP_TRANSLATE, dx, dy, width, height

        if "RECOLOR" in tokens or "recolor" in lower:
            try:
                idx = lower.index("recolor")
                src = int(float(tokens[idx - 2]))
                dst = int(float(tokens[idx - 1]))
            except Exception as exc:
                raise ValueError(f"Invalid RECOLOR program: {rpn_program}") from exc
            return self.OP_RECOLOR, src, dst, width, height

        raise ValueError(f"Unsupported ARC RPN program: {rpn_program}")

    def __del__(self) -> None:
        try:
            if self._owns_instance and self.instance_id is not None and self.pool is not None:
                self.pool.release_core(self.instance_id, pool=True)
        except Exception:
            pass


__all__ = ["ARCRPNExecutor"]
