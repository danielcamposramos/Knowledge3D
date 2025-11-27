"""
Semantic signature extraction for ARC grids.

SOVEREIGN ARCHITECTURE:
- Signatures are RPN PROGRAMS (not Python code!)
- Executed via RPN Math Core + PTX kernels
- Lightweight metadata output only (ints/floats)

Dual Client Reality: Signature extraction programs are procedural and readable.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

from .sovereign_utils import (
    count_nonzero_grid,
    grid_shape,
    most_common_value,
    unique_counts,
    zeros_like_grid,
)

class SemanticSignature:
    """
    Extract coarse semantic signatures from integer grids.

    SOVEREIGN: Uses RPN operations where possible, minimal Python fallback.
    Future: All operations should become RPN programs in Grammar Galaxy.
    """

    # SOVEREIGN RPN PROGRAMS (future: execute via RPNMathCore)
    # These should become RPN programs in Grammar Galaxy
    RPN_SIGNATURES = {
        "count_nonzero": "FLATTEN 0 NE REDUCE_SUM",  # Count non-background cells
        "count_colors": "FLATTEN SET_UNIQUE LEN",     # Unique color count
        "check_sym_v": "DUP FLIP_V EQ",               # Vertical symmetry check
        "check_sym_h": "DUP FLIP_H EQ",               # Horizontal symmetry check
    }

    @staticmethod
    def extract(grid: Sequence[Sequence[int]], multi_scale: bool = False) -> Dict:
        """
        Extract semantic signature from grid.

        SOVEREIGN: No numpy! Minimal Python logic.
        TODO: Replace with RPN program execution via RPNMathCore.
        """
        # Convert to list of lists if needed (lightweight)
        grid_list = [list(row) for row in grid]

        sig: Dict = {
            "structural": SemanticSignature._extract_structural(grid_list),
            "color": SemanticSignature._extract_color(grid_list),
            "pattern": SemanticSignature._extract_pattern(grid_list),
        }

        # Skip multi-scale for now (not needed for matching, saves compute)
        # if multi_scale:
        #     sig["scale_2x2"] = SemanticSignature._downsample_signature(grid_list, 2)
        #     sig["scale_4x4"] = SemanticSignature._downsample_signature(grid_list, 4)

        # Skip topology for now (expensive, not critical for matching)
        # sig["topology"] = SemanticSignature._compute_topology(grid_list)

        sig["signature_hash"] = SemanticSignature._compute_signature_hash(grid_list)

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
    def _extract_structural(grid: List[List[int]]) -> Dict:
        """Extract structural features (SOVEREIGN: pure Python)."""
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0

        # Vertical symmetry: grid == flipped vertically
        is_sym_v = all(grid[i] == grid[h - 1 - i] for i in range(h // 2))

        # Horizontal symmetry: each row reversed
        is_sym_h = all(row[j] == row[w - 1 - j] for row in grid for j in range(w // 2))

        # Diagonal symmetry: grid == transposed
        is_sym_d = all(grid[i][j] == grid[j][i] for i in range(min(h, w)) for j in range(min(h, w))) if h == w else False

        # Sparsity: fraction of zero cells
        total_cells = h * w if h and w else 1
        nonzero_count = sum(1 for row in grid for val in row if val != 0)
        sparsity = 1.0 - (float(nonzero_count) / float(total_cells))

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
    def _extract_color(grid: List[List[int]]) -> Dict:
        """Extract color features (SOVEREIGN: pure Python)."""
        # Flatten grid and count colors
        all_values = [val for row in grid for val in row]
        color_counts: Dict[int, int] = {}
        for val in all_values:
            color_counts[val] = color_counts.get(val, 0) + 1

        uniques = list(color_counts.keys())
        background = max(color_counts, key=color_counts.get) if color_counts else 0
        foreground = [c for c in uniques if c != background]

        return {
            "num_colors": len(uniques),
            "colors": uniques,
            "background": background,
            "foreground": foreground,
            "color_distribution": color_counts,
        }

    @staticmethod
    def _extract_pattern(grid: List[List[int]]) -> Dict:
        """Extract pattern features (SOVEREIGN: minimal Python)."""
        # TODO: Replace with RPN programs for component counting
        return {
            "num_components": SemanticSignature._count_components(grid),
            "has_border": SemanticSignature._has_border(grid),
            "has_repetition": SemanticSignature._check_repetition(grid),
        }

    @staticmethod
    def _count_components(grid: List[List[int]]) -> int:
        """
        Simple 4-neighbor flood fill to count foreground components.

        SOVEREIGN: Minimal Python. TODO: Replace with RPN graph traversal program.
        """
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0

        # Find background (most common color)
        all_vals = [val for row in grid for val in row]
        if not all_vals:
            return 0
        color_counts = {}
        for v in all_vals:
            color_counts[v] = color_counts.get(v, 0) + 1
        background = max(color_counts, key=color_counts.get)

        visited = [[False] * w for _ in range(h)]
        comps = 0

        def neighbors(r: int, c: int):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    yield nr, nc

        for r in range(h):
            for c in range(w):
                if visited[r][c] or grid[r][c] == background:
                    continue
                comps += 1
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    for nr, nc in neighbors(cr, cc):
                        if not visited[nr][nc] and grid[nr][nc] != background:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
        return comps

    @staticmethod
    def _has_border(grid: List[List[int]]) -> bool:
        """Check for uniform border (SOVEREIGN: minimal Python)."""
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        if h < 3 or w < 3:
            return False

        edge_color = grid[0][0]

        # Check top and bottom rows
        top_uniform = all(grid[0][c] == edge_color for c in range(w))
        bottom_uniform = all(grid[h - 1][c] == edge_color for c in range(w))

        # Check left and right columns
        left_uniform = all(grid[r][0] == edge_color for r in range(h))
        right_uniform = all(grid[r][w - 1] == edge_color for r in range(h))

        edge_uniform = top_uniform and bottom_uniform and left_uniform and right_uniform

        # Check interior is different
        interior_different = any(
            grid[r][c] != edge_color for r in range(1, h - 1) for c in range(1, w - 1)
        )

        return edge_uniform and interior_different

    @staticmethod
    def _check_repetition(grid: List[List[int]]) -> bool:
        """Check for 2x2 block repetition (SOVEREIGN: minimal Python)."""
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        if h >= 4 and w >= 4:
            # Get reference 2x2 block from top-left
            block = [grid[0][:2], grid[1][:2]]

            # Check other 2x2 positions
            for r in (0, 2):
                for c in (0, 2):
                    if r == 0 and c == 0:
                        continue
                    # Compare 2x2 blocks
                    other_block = [grid[r][c : c + 2], grid[r + 1][c : c + 2]]
                    if block == other_block:
                        return True
        return False

    @staticmethod
    def _downsample_signature(grid: List[List[int]], factor: int) -> Dict:
        """Extract coarse signature from a downsampled view."""
        h, w = grid_shape(grid)
        if h < factor or w < factor:
            return {}
        ds = [
            [int(grid[r][c]) for c in range(0, w, factor)]
            for r in range(0, h, factor)
        ]
        flat_ds = [cell for row in ds for cell in row]
        uniques, counts = unique_counts(flat_ds)
        size = len(flat_ds) or 1
        nonzero = count_nonzero_grid(ds)
        return {
            "num_colors": len(uniques),
            "sparsity": round(1.0 - (float(nonzero) / float(size)), 2),
            "color_distribution": {int(c): float(cnt / size) for c, cnt in zip(uniques, counts)},
        }

    @staticmethod
    def _compute_topology(grid: List[List[int]]) -> Dict:
        """Compute simple topological descriptors."""
        beta0 = SemanticSignature._count_components(grid)
        holes = SemanticSignature._count_holes(grid)
        return {
            "beta_0": beta0,
            "beta_1": holes,
            "euler_characteristic": beta0 - holes,
        }

    @staticmethod
    def _count_holes(grid: List[List[int]]) -> int:
        """Count holes by components of background minus exterior."""
        h, w = grid_shape(grid)
        background = most_common_value([int(v) for row in grid for v in row])
        visited = zeros_like_grid(grid, False)
        comps = 0

        def neighbors(r: int, c: int):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    yield nr, nc

        touches_border: List[bool] = []
        for r in range(h):
            for c in range(w):
                if visited[r][c] or grid[r][c] != background:
                    continue
                comps += 1
                stack = [(r, c)]
                visited[r][c] = True
                border = False
                while stack:
                    cr, cc = stack.pop()
                    if cr in (0, h - 1) or cc in (0, w - 1):
                        border = True
                    for nr, nc in neighbors(cr, cc):
                        if not visited[nr][nc] and grid[nr][nc] == background:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                touches_border.append(border)
        # Holes are background components that do not touch border; subtract exterior
        interior = sum(1 for b in touches_border if not b)
        return interior

    @staticmethod
    def _compute_signature_hash(grid: List[List[int]]) -> str:
        """Compute signature hash (SOVEREIGN: minimal Python)."""
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0

        # Count unique colors
        all_vals = [val for row in grid for val in row]
        unique_colors = len(set(all_vals)) if all_vals else 0

        # Compute sparsity
        total = h * w if h and w else 1
        nonzero = sum(1 for v in all_vals if v != 0)
        sparsity = 1.0 - (float(nonzero) / float(total))

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
