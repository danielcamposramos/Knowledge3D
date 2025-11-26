"""
Semantic signature extraction for ARC grids.

Captures lightweight structural, color, and pattern cues to tag contexts without
duplicating character/word layers.
"""

from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np


class SemanticSignature:
    """Extract coarse semantic signatures from integer grids."""

    @staticmethod
    def extract(grid: np.ndarray, multi_scale: bool = True) -> Dict:
        grid = np.asarray(grid)
        sig: Dict = {
            "structural": SemanticSignature._extract_structural(grid),
            "color": SemanticSignature._extract_color(grid),
            "pattern": SemanticSignature._extract_pattern(grid),
        }

        if multi_scale:
            sig["scale_2x2"] = SemanticSignature._downsample_signature(grid, 2)
            sig["scale_4x4"] = SemanticSignature._downsample_signature(grid, 4)

        sig["topology"] = SemanticSignature._compute_topology(grid)
        sig["signature_hash"] = SemanticSignature._compute_signature_hash(grid)

        # Flatten a few commonly used fields for easier matching/usage
        sig["dimensions"] = sig["structural"].get("dimensions")
        sig["sparsity"] = sig["structural"].get("sparsity")
        sig["sparsity_label"] = sig["structural"].get("sparsity_label")
        sig["symmetry_vertical"] = sig["structural"].get("symmetric_vertical")
        sig["symmetry_horizontal"] = sig["structural"].get("symmetric_horizontal")
        sig["symmetry_diagonal"] = sig["structural"].get("symmetric_diagonal")
        sig["num_colors"] = sig["color"].get("num_colors")
        sig["color_distribution"] = sig["color"].get("color_distribution")
        sig["has_border"] = sig["pattern"].get("has_border")
        sig["has_repetition"] = sig["pattern"].get("has_repetition")
        sig["connected_components"] = sig["pattern"].get("num_components")

        return sig

    @staticmethod
    def _extract_structural(grid: np.ndarray) -> Dict:
        h, w = grid.shape
        is_sym_v = bool(np.array_equal(grid, np.flip(grid, axis=0)))
        is_sym_h = bool(np.array_equal(grid, np.flip(grid, axis=1)))
        is_sym_d = bool(np.array_equal(grid, grid.T))
        sparsity = 1.0 - (float(np.count_nonzero(grid)) / float(grid.size or 1))
        if sparsity > 0.7:
            sparsity_label = "sparse"
        elif sparsity < 0.3:
            sparsity_label = "dense"
        else:
            sparsity_label = "medium"
        return {
            "dimensions": f"{h}x{w}",
            "aspect_ratio": "square" if h == w else "rectangular",
            "symmetric_vertical": is_sym_v,
            "symmetric_horizontal": is_sym_h,
            "symmetric_diagonal": is_sym_d,
            "sparsity": round(sparsity, 2),
            "sparsity_label": sparsity_label,
        }

    @staticmethod
    def _extract_color(grid: np.ndarray) -> Dict:
        uniques = np.unique(grid)
        color_counts = {int(c): int(np.sum(grid == c)) for c in uniques}
        background = int(max(color_counts, key=color_counts.get))
        foreground = [int(c) for c in uniques if int(c) != background]
        return {
            "num_colors": len(uniques),
            "colors": [int(c) for c in uniques],
            "background": background,
            "foreground": foreground,
            "color_distribution": color_counts,
        }

    @staticmethod
    def _extract_pattern(grid: np.ndarray) -> Dict:
        return {
            "num_components": SemanticSignature._count_components(grid),
            "has_border": SemanticSignature._has_border(grid),
            "has_repetition": SemanticSignature._check_repetition(grid),
        }

    @staticmethod
    def _count_components(grid: np.ndarray) -> int:
        """Simple 4-neighbor flood fill to count foreground components."""
        h, w = grid.shape
        background = np.argmax(np.bincount(grid.flatten()))
        visited = np.zeros_like(grid, dtype=bool)
        comps = 0

        def neighbors(r: int, c: int):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    yield nr, nc

        for r in range(h):
            for c in range(w):
                if visited[r, c] or grid[r, c] == background:
                    continue
                comps += 1
                stack = [(r, c)]
                visited[r, c] = True
                while stack:
                    cr, cc = stack.pop()
                    for nr, nc in neighbors(cr, cc):
                        if not visited[nr, nc] and grid[nr, nc] != background:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
        return comps

    @staticmethod
    def _has_border(grid: np.ndarray) -> bool:
        if grid.shape[0] < 3 or grid.shape[1] < 3:
            return False
        top, bottom = grid[0, :], grid[-1, :]
        left, right = grid[:, 0], grid[:, -1]
        interior = grid[1:-1, 1:-1]
        edge_color = top[0]
        edge_uniform = (
            np.all(top == edge_color)
            and np.all(bottom == edge_color)
            and np.all(left == edge_color)
            and np.all(right == edge_color)
        )
        interior_different = not np.all(interior == edge_color)
        return bool(edge_uniform and interior_different)

    @staticmethod
    def _check_repetition(grid: np.ndarray) -> bool:
        h, w = grid.shape
        if h >= 4 and w >= 4:
            block = grid[:2, :2]
            for r in (0, 2):
                for c in (0, 2):
                    if r == 0 and c == 0:
                        continue
                    if np.array_equal(block, grid[r : r + 2, c : c + 2]):
                        return True
        return False

    @staticmethod
    def _downsample_signature(grid: np.ndarray, factor: int) -> Dict:
        """Extract coarse signature from a downsampled view."""
        h, w = grid.shape
        if h < factor or w < factor:
            return {}
        ds = grid[::factor, ::factor]
        uniques, counts = np.unique(ds, return_counts=True)
        return {
            "num_colors": len(uniques),
            "sparsity": round(1.0 - (float(np.count_nonzero(ds)) / float(ds.size or 1)), 2),
            "color_distribution": {int(c): float(cnt / ds.size) for c, cnt in zip(uniques, counts)},
        }

    @staticmethod
    def _compute_topology(grid: np.ndarray) -> Dict:
        """Compute simple topological descriptors."""
        beta0 = SemanticSignature._count_components(grid)
        holes = SemanticSignature._count_holes(grid)
        return {
            "beta_0": beta0,
            "beta_1": holes,
            "euler_characteristic": beta0 - holes,
        }

    @staticmethod
    def _count_holes(grid: np.ndarray) -> int:
        """Count holes by components of background minus exterior."""
        h, w = grid.shape
        background = np.argmax(np.bincount(grid.flatten()))
        visited = np.zeros_like(grid, dtype=bool)
        comps = 0

        def neighbors(r: int, c: int):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    yield nr, nc

        touches_border: List[bool] = []
        for r in range(h):
            for c in range(w):
                if visited[r, c] or grid[r, c] != background:
                    continue
                comps += 1
                stack = [(r, c)]
                visited[r, c] = True
                border = False
                while stack:
                    cr, cc = stack.pop()
                    if cr in (0, h - 1) or cc in (0, w - 1):
                        border = True
                    for nr, nc in neighbors(cr, cc):
                        if not visited[nr, nc] and grid[nr, nc] == background:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                touches_border.append(border)
        # Holes are background components that do not touch border; subtract exterior
        interior = sum(1 for b in touches_border if not b)
        return interior

    @staticmethod
    def _compute_signature_hash(grid: np.ndarray) -> str:
        h, w = grid.shape
        unique_colors = len(np.unique(grid))
        sparsity = 1.0 - (float(np.count_nonzero(grid)) / float(grid.size or 1))
        signature_str = f"{h}x{w}_{unique_colors}c_{sparsity:.1f}s"
        return hashlib.md5(signature_str.encode()).hexdigest()[:8]

    @staticmethod
    def compute_transformation_type(input_sig: Dict, output_sig: Dict) -> str:
        if input_sig["structural"]["dimensions"] == output_sig["structural"]["dimensions"]:
            if input_sig["structural"]["symmetric_vertical"] != output_sig["structural"]["symmetric_vertical"]:
                return "rotation_or_reflection"
            if input_sig["color"]["num_colors"] == output_sig["color"]["num_colors"] and input_sig["structural"]["sparsity"] == output_sig["structural"]["sparsity"]:
                return "recoloring"
            if input_sig["structural"]["sparsity_label"] == "sparse" and output_sig["structural"]["sparsity_label"] in ("medium", "dense"):
                return "pattern_completion"
        return "complex_transformation"


__all__ = ["SemanticSignature"]
