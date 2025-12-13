"""Execute ARC RPN programs via the sovereign PTX drawing bridge."""

from __future__ import annotations

from typing import List, Sequence

import cupy as cp

from knowledge3d.cranium.bridges.drawing_bridge import DrawingBridge
from knowledge3d.cranium.ptx_runtime.math_core_pool import MathCorePool, get_global_math_core_pool
from knowledge3d.cranium.sovereign import loader

try:
    from knowledge3d.cranium.ptx_runtime.drawing_transform_kernels import (
        flip_h,
        flip_v,
        overlay,
        recolor,
        rot90_ccw,
        rot90_cw,
        scale_2x,
        tile_2x2,
        transpose,
        crop_gpu,
        extract_bbox_gpu,
        find_bbox_gpu,
    )

    _HAS_DRAWING_KERNELS = True
except Exception:
    _HAS_DRAWING_KERNELS = False


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
        # Fast path: direct drawing transformation tokens using CuPy kernels.
        if _HAS_DRAWING_KERNELS:
            direct = self._execute_transformation(rpn_program, grid)
            if direct is not None:
                self.ptx_success_count += 1
                return direct

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

    def _execute_transformation(self, rpn_program: str, grid: Sequence[Sequence[int]]) -> List[List[int]] | None:
        """
        Execute drawing transformation tokens using GPU kernels.

        Supported tokens (uppercase or lowercase):
          ROT90_CW / ROT90, ROT90_CCW, ROT180, FLIP_H, FLIP_V, FLIP_DIAG / TRANSPOSE,
          SCALE_2X, TILE_2X2, RECOLOR old new
        """
        tokens = rpn_program.strip().split()
        if not tokens:
            return None
        op = tokens[-1].upper()
        try:
            grid_cp = cp.asarray(grid, dtype=cp.int32)
        except Exception:
            return None

        # Iteration wrappers: REPEAT_N OP or UNTIL_STABLE OP
        if tokens[0].upper().startswith("REPEAT_") and len(tokens) >= 2:
            try:
                n = int(tokens[0].split("_")[1])
            except Exception:
                n = 1
            op_tok = tokens[-1]
            out_grid = grid_cp
            for _ in range(max(1, n)):
                transformed = self._execute_transformation(" ".join([op_tok]), out_grid.tolist())
                out_grid = cp.asarray(transformed, dtype=cp.int32)
            return out_grid.tolist()

        if tokens[0].upper() == "UNTIL_STABLE" and len(tokens) >= 2:
            op_tok = tokens[-1]
            out_grid = grid_cp
            for _ in range(10):
                prev = out_grid.copy()
                transformed = self._execute_transformation(" ".join([op_tok]), out_grid.tolist())
                out_grid = cp.asarray(transformed, dtype=cp.int32)
                if cp.all(prev == out_grid):
                    break
            return out_grid.tolist()

        out: cp.ndarray | None = None
        if op in {"ROT90_CW", "ROT90"}:
            out = rot90_cw(grid_cp)
        elif op == "ROT90_CCW":
            out = rot90_ccw(grid_cp)
        elif op == "ROT180":
            out = rot90_cw(rot90_cw(grid_cp))
        elif op == "FLIP_H":
            out = flip_h(grid_cp)
        elif op == "FLIP_V":
            out = flip_v(grid_cp)
        elif op in {"TRANSPOSE", "FLIP_DIAG"}:
            out = transpose(grid_cp)
        elif op == "SCALE_2X":
            out = scale_2x(grid_cp)
        elif op == "TILE_2X2":
            out = tile_2x2(grid_cp)
        elif op == "CROP" and len(tokens) >= 5:
            try:
                y = int(float(tokens[-4])); x = int(float(tokens[-3])); h = int(float(tokens[-2])); w = int(float(tokens[-1]))
                out = crop_gpu(grid_cp, y, x, h, w)
            except Exception:
                out = None
        elif op == "EXTRACT_BBOX":
            color = 0
            if len(tokens) >= 2:
                try:
                    color = int(float(tokens[-2]))
                except Exception:
                    color = 0
            out = extract_bbox_gpu(grid_cp, color)
        elif op == "FIND_BBOX":
            color = 0
            if len(tokens) >= 2:
                try:
                    color = int(float(tokens[-2]))
                except Exception:
                    color = 0
            bbox = find_bbox_gpu(grid_cp, color)
            return [[bbox[0], bbox[1], bbox[2], bbox[3]]]
        elif op == "RECOLOR" and len(tokens) >= 3:
            try:
                old_color = int(float(tokens[-3]))
                new_color = int(float(tokens[-2]))
            except Exception:
                old_color = new_color = None  # type: ignore[assignment]
            if old_color is not None and new_color is not None:
                out = recolor(grid_cp, old_color, new_color)
        # Overlay requires two grids; unsupported in single-grid executor.

        if out is None:
            return None
        return out.tolist()

    def __del__(self) -> None:
        try:
            if self._owns_instance and self.instance_id is not None and self.pool is not None:
                self.pool.release_core(self.instance_id, pool=True)
        except Exception:
            pass


__all__ = ["ARCRPNExecutor"]
