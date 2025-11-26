"""
ARC-AGI Grid Processor: Grid → Procedural RPN Program → Galaxy Embedding

Key Insight:
    Grids are drawings. We apply the same procedural pattern used for character
    glyphs and procedural video: describe structure as an RPN program and keep
    the heavy math in sovereign PTX kernels.

Architecture:
    1. Grid cells → Visual primitives (rectangles with colors)
    2. Visual primitives → RPN drawing program
    3. RPN program → Execute (procedural rasterisation or GPU path)
    4. Visual result → Fractal / spatial features
    5. Features → Matryoshka Galaxy embedding

Notes:
    - This module lives on the ingestion / training side, not the core physics
      hot path. NumPy usage here is acceptable and mirrors other ingestion
      helpers (e.g. ARC text cache, audio/video codecs).
    - The visual embedding step is pluggable so tests can use a lightweight
      CPU stub, while production can swap in GPU FractalEmitter or even the
      ternary video codec for richer features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
from knowledge3d.cranium.specialists.procedural_drawing_specialist import (
    ProceduralDrawingSpecialist,
)
from knowledge3d.cranium.adaptive_swarm import AdaptiveSwarmTRM, SwarmConfig
from .embedders.multimodal_grid_embedder import MultiModalGridEmbedder
from .embedders.video_grid_embedder import VideoGridEmbedder
from .embedders.audio_grid_embedder import AudioGridEmbedder


@dataclass
class _DefaultVisualEmbedder:
    """
    Lightweight visual embedder used by default in ARCGridProcessor.

    This is intentionally simple and CPU-only so we can exercise the full
    Grid → RPN → features → embedding pipeline without requiring GPU kernels.

    Production runs can supply a richer embedder (e.g. an adapter that uses
    FractalEmitter or the ternary video codec) via the `visual_embedder`
    parameter on ARCGridProcessor.
    """

    matryoshka_dim: int

    def emit_fractal_features(self, raster: np.ndarray) -> np.ndarray:
        """
        Derive a deterministic feature vector from a raster.

        Strategy:
            - Flatten the raster, normalise to [0, 1].
            - Take a truncated or padded copy to matryoshka_dim.
        """
        flat = np.asarray(raster, dtype=np.float32).ravel()
        if flat.size == 0:
            return np.zeros(self.matryoshka_dim, dtype=np.float32)

        max_val = float(np.max(np.abs(flat)))
        if max_val > 0.0:
            flat = flat / max_val

        if flat.size >= self.matryoshka_dim:
            return flat[: self.matryoshka_dim].astype(np.float32, copy=False)

        out = np.zeros(self.matryoshka_dim, dtype=np.float32)
        out[: flat.size] = flat
        return out


class ARCGridProcessor:
    """
    Convert ARC-AGI grids to procedural RPN programs and Galaxy embeddings.

    Applies the same procedural pattern used for character glyphs:
        Character → Visual RPN → Embedding
        Grid → Visual RPN → Embedding
    """

    def __init__(
        self,
        matryoshka_dim: int = 512,
        *,
        visual_embedder: Any | None = None,
        embedder_type: str = "procedural",
    ):
        """
        Initialize grid processor.

        Args:
            matryoshka_dim: Embedding dimension (128-2048 adaptive).
            visual_embedder:
                Optional object providing `emit_fractal_features(raster) -> np.ndarray`
                used for the procedural RPN→raster path.
            embedder_type:
                "procedural" (default) uses the RPN drawing + visual embedder path.
                "video" / "audio" / "multimodal" route through ternary codecs via
                VideoGridEmbedder / AudioGridEmbedder / MultiModalGridEmbedder.
        """
        self.matryoshka_dim = matryoshka_dim
        self.embedder_type = embedder_type
        self.rpn_engine = ModularRPNEngine()

        # Reuse procedural drawing infrastructure
        # (Character glyphs and grids are BOTH procedural drawings.)
        cfg = SwarmConfig()
        cfg.base_dims = matryoshka_dim
        cfg.min_dims = 64
        cfg.enable_auto_expansion = False
        self.swarm = AdaptiveSwarmTRM(config=cfg)
        self.drawing_specialist = ProceduralDrawingSpecialist(
            swarm=self.swarm,
            matryoshka_dim=matryoshka_dim,
        )

        # Visual embedder (pluggable for CPU vs GPU backends) used when
        # embedder_type == "procedural" or when a custom embedder is provided.
        self.visual_embedder = (
            visual_embedder
            if visual_embedder is not None
            else _DefaultVisualEmbedder(matryoshka_dim=matryoshka_dim)
        )

        # Optional ternary codec embedders. These are only constructed when
        # requested so that tests can continue to use the lightweight path.
        self.codec_embedder: Any | None = None
        if visual_embedder is None:
            if embedder_type == "video":
                self.codec_embedder = VideoGridEmbedder()
            elif embedder_type == "audio":
                self.codec_embedder = AudioGridEmbedder()
            elif embedder_type == "multimodal":
                self.codec_embedder = MultiModalGridEmbedder(
                    matryoshka_dim=matryoshka_dim
                )

        # ARC-AGI color palette (10 colors: 0-9)
        self.arc_colors: Dict[int, Tuple[int, int, int]] = {
            0: (0, 0, 0),  # Black (background)
            1: (0, 116, 217),  # Blue
            2: (255, 65, 54),  # Red
            3: (46, 204, 64),  # Green
            4: (255, 220, 0),  # Yellow
            5: (170, 170, 170),  # Gray
            6: (240, 18, 190),  # Magenta
            7: (255, 133, 27),  # Orange
            8: (127, 219, 255),  # Sky Blue
            9: (135, 12, 37),  # Maroon
        }

    # --------------------------------------------------------------------- #
    # Grid → RPN program
    # --------------------------------------------------------------------- #
    def grid_to_rpn_program(self, grid: Sequence[Sequence[int]]) -> str:
        """
        Convert grid to procedural RPN drawing program.

        Args:
            grid: 2D array of color indices (0-9).

        Returns:
            RPN program string that reconstructs the grid.
        """
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        rpn_commands: List[str] = []

        # Generate RPN commands for each non-background cell.
        for y in range(height):
            for x in range(width):
                color = int(grid[y][x])
                if color == 0:
                    continue

                # Draw filled rectangle for this cell:
                # RPN: x y MOVE x+1 y LINE x+1 y+1 LINE x y+1 LINE CLOSE
                #      SET_FILL_COLOR color FILL
                rpn_commands.extend(
                    [
                        f"{x}",
                        f"{y}",
                        "MOVE",
                        f"{x + 1}",
                        f"{y}",
                        "LINE",
                        f"{x + 1}",
                        f"{y + 1}",
                        "LINE",
                        f"{x}",
                        f"{y + 1}",
                        "LINE",
                        "CLOSE",
                        "SET_FILL_COLOR",
                        f"{color}",
                        "FILL",
                    ]
                )

        return " ".join(rpn_commands)

    # --------------------------------------------------------------------- #
    # Grid → spatial embedding
    # --------------------------------------------------------------------- #
    def grid_to_spatial_embedding(
        self,
        grid: Sequence[Sequence[int]],
        routing: int = 0,
    ) -> np.ndarray:
        """
        Convert grid to spatial embedding for Galaxy Universe.

        Args:
            grid: 2D array of color indices.

        Returns:
            1D embedding (matryoshka_dim dimensions).

        Architecture (procedural path):
            1. Grid → RPN program
            2. RPN program → Execute (raster)
            3. Raster → visual / fractal features
            4. Features → Matryoshka projection

        Architecture (codec path):
            - Video / audio / multimodal ternary codecs produce features which
              are projected to Matryoshka dimension directly.
        """
        # Codec path: delegate to ternary codec embedders if configured.
        if self.codec_embedder is not None:
            if self.embedder_type == "video":
                raw = self.codec_embedder.grid_to_video_embedding(grid)
            elif self.embedder_type == "audio":
                raw = self.codec_embedder.grid_to_audio_embedding(
                    grid, target_dim=self.matryoshka_dim
                )
            else:
                # Multimodal fusion with ternary routing.
                raw = self.codec_embedder.grid_to_multimodal_embedding(
                    grid, routing=routing
                )
            raw = np.asarray(raw, dtype=np.float32).ravel()
            if raw.size != self.matryoshka_dim:
                raw = self._project_to_matryoshka(raw)
            return raw.astype(np.float32, copy=False)

        # Procedural RPN path (default).
        rpn_program = self.grid_to_rpn_program(grid)
        visual_features = self._execute_visual_rpn(rpn_program)
        fractal_embedding = self.visual_embedder.emit_fractal_features(visual_features)

        fractal_embedding = np.asarray(fractal_embedding, dtype=np.float32).ravel()
        if fractal_embedding.size != self.matryoshka_dim:
            fractal_embedding = self._project_to_matryoshka(fractal_embedding)

        return fractal_embedding.astype(np.float32, copy=False)

    def _execute_visual_rpn(self, rpn_program: str) -> np.ndarray:
        """
        Execute RPN program to generate a simple raster representation.

        This is a CPU placeholder. In a future phase we will:
            - Route the RPN program through ProceduralDrawingBridge to PTX.
            - Potentially reuse the ternary video codec path by treating
              the grid as a low-resolution frame.

        Returns:
            2D raster (float32) suitable for feature extraction.
        """
        tokens = rpn_program.split()
        coords: List[float] = [float(t) for t in tokens if self._is_number(t)]

        if not coords:
            return np.zeros((32, 32), dtype=np.float32)

        max_coord = int(max(coords)) + 1
        size = max(4, max_coord * 2)
        raster = np.zeros((size, size), dtype=np.float32)

        # Simple occupancy raster: mark cell centers that appear in RPN.
        # Full implementation will rely on GPU RPN executor.
        for i in range(0, len(coords), 2):
            x = int(coords[i])
            if i + 1 >= len(coords):
                break
            y = int(coords[i + 1])
            if 0 <= x < size and 0 <= y < size:
                raster[y, x] = 1.0

        return raster

    @staticmethod
    def _is_number(token: str) -> bool:
        """Check if token is a number."""
        try:
            float(token)
            return True
        except ValueError:
            return False

    def _project_to_matryoshka(self, embedding: np.ndarray) -> np.ndarray:
        """Project embedding to target Matryoshka dimension."""
        current_dim = int(embedding.size)

        if current_dim == self.matryoshka_dim:
            return embedding.astype(np.float32, copy=False)
        if current_dim > self.matryoshka_dim:
            # Truncate (Matryoshka nested property)
            return embedding[: self.matryoshka_dim].astype(np.float32, copy=False)

        # Pad with zeros
        padded = np.zeros(self.matryoshka_dim, dtype=np.float32)
        padded[:current_dim] = embedding
        return padded

    # --------------------------------------------------------------------- #
    # Primitive detection (before/after grids)
    # --------------------------------------------------------------------- #
    def detect_spatial_primitive(
        self,
        grid_before: Sequence[Sequence[int]],
        grid_after: Sequence[Sequence[int]],
    ) -> Dict[str, Any]:
        """
        Detect spatial transformation primitive from before/after grids.

        Args:
            grid_before: Input grid.
            grid_after: Output grid.

        Returns:
            {
                "primitive": str,     # e.g. "ROTATE_90", "FLIP_H", "TRANSLATE"
                "parameters": dict,   # transformation parameters
                "rpn_program": str,   # RPN snippet to apply transformation
                "confidence": float,  # pattern match confidence
            }
        """
        # Embed both grids (reserved for future TRM-based scoring).
        _ = self.grid_to_spatial_embedding(grid_before)
        _ = self.grid_to_spatial_embedding(grid_after)

        primitives: List[Dict[str, Any]] = []

        # Test rotations
        for angle in (0, 90, 180, 270):
            rotated = self._apply_rotation(grid_before, angle) if angle != 0 else grid_before
            if self._grids_match(rotated, grid_after):
                primitives.append(
                    {
                        "primitive": f"ROTATE_{angle}",
                        "parameters": {"angle": angle},
                        "rpn_program": f"{angle} ROTATE",
                        "confidence": 1.0,
                    }
                )

        # Test horizontal flip
        flipped_h = self._apply_flip_horizontal(grid_before)
        if self._grids_match(flipped_h, grid_after):
            primitives.append(
                {
                    "primitive": "FLIP_H",
                    "parameters": {},
                    "rpn_program": "-1 1 SCALE",
                    "confidence": 1.0,
                }
            )

        # Test vertical flip
        flipped_v = self._apply_flip_vertical(grid_before)
        if self._grids_match(flipped_v, grid_after):
            primitives.append(
                {
                    "primitive": "FLIP_V",
                    "parameters": {},
                    "rpn_program": "1 -1 SCALE",
                    "confidence": 1.0,
                }
            )

        # Test pure translation
        translation = self._detect_translation(grid_before, grid_after)
        if translation is not None:
            dx, dy = translation
            primitives.append(
                {
                    "primitive": "TRANSLATE",
                    "parameters": {"dx": dx, "dy": dy},
                    "rpn_program": f"{dx} {dy} TRANSLATE",
                    "confidence": 1.0,
                }
            )

        # Test rotation + translation composition
        for angle in (90, 180, 270):
            rotated = self._apply_rotation(grid_before, angle)
            translation = self._detect_translation(rotated, grid_after)
            if translation is not None:
                dx, dy = translation
                primitives.append(
                    {
                        "primitive": "ROTATE_TRANSLATE",
                        "parameters": {"angle": angle, "dx": dx, "dy": dy},
                        "rpn_program": f"{angle} ROTATE {dx} {dy} TRANSLATE",
                        "confidence": 1.0,
                    }
                )

        # Test simple color remap (single-source to single-target color)
        color_map = self._detect_color_map(grid_before, grid_after)
        if color_map is not None:
            src, dst = color_map
            primitives.append(
                {
                    "primitive": "RECOLOR",
                    "parameters": {"src": src, "dst": dst},
                    "rpn_program": f"{src} {dst} RECOLOR",
                    "confidence": 1.0,
                }
            )

        # Compose rotate/flip/translate with recolor.
        transforms = []
        transforms.extend([("ROTATE", angle, self._apply_rotation(grid_before, angle)) for angle in (90, 180, 270)])
        transforms.append(("FLIP_H", None, self._apply_flip_horizontal(grid_before)))
        transforms.append(("FLIP_V", None, self._apply_flip_vertical(grid_before)))
        trans = self._detect_translation(grid_before, grid_after)
        if trans is not None:
            dx, dy = trans
            transforms.append(("TRANSLATE", (dx, dy), self._apply_translation(grid_before, dx, dy)))
        # Rotate then translate combos
        for angle in (90, 180, 270):
            rotated = self._apply_rotation(grid_before, angle)
            trans_rt = self._detect_translation(rotated, grid_after)
            if trans_rt is not None:
                dx, dy = trans_rt
                transforms.append(
                    ("ROTATE_TRANSLATE", (angle, dx, dy), self._apply_translation(rotated, dx, dy))
                )

        for name, param, tgrid in transforms:
            cmap = self._detect_color_map(tgrid, grid_after)
            if cmap is not None:
                src, dst = cmap
                rpn = []
                if name == "ROTATE":
                    rpn.append(f"{param} ROTATE")
                elif name == "FLIP_H":
                    rpn.append("FLIP_H")
                elif name == "FLIP_V":
                    rpn.append("FLIP_V")
                elif name == "TRANSLATE":
                    dx, dy = param
                    rpn.append(f"{dx} {dy} TRANSLATE")
                elif name == "ROTATE_TRANSLATE":
                    angle, dx, dy = param
                    rpn.append(f"{angle} ROTATE {dx} {dy} TRANSLATE")
                rpn.append(f"{src} {dst} RECOLOR")
                primitives.append(
                    {
                        "primitive": f"{name}_RECOLOR",
                        "parameters": {"transform": name, "param": param, "src": src, "dst": dst},
                        "rpn_program": " ".join(rpn),
                        "confidence": 1.0,
                    }
                )

        # Final heuristic: prefer recolor, then translate/rotate-translate, else first match.
        return self._prioritize(primitives)

    @staticmethod
    def _apply_rotation(
        grid: Sequence[Sequence[int]],
        angle: int,
    ) -> List[List[int]]:
        """Apply rotation to grid."""
        np_grid = np.asarray(grid, dtype=int)
        if angle == 90:
            rotated = np.rot90(np_grid, k=-1)  # Clockwise
        elif angle == 180:
            rotated = np.rot90(np_grid, k=2)
        elif angle == 270:
            rotated = np.rot90(np_grid, k=1)  # Counter-clockwise
        else:
            rotated = np_grid
        return rotated.tolist()

    @staticmethod
    def _apply_flip_horizontal(grid: Sequence[Sequence[int]]) -> List[List[int]]:
        """Apply horizontal flip to grid."""
        return np.fliplr(np.asarray(grid, dtype=int)).tolist()

    @staticmethod
    def _apply_flip_vertical(grid: Sequence[Sequence[int]]) -> List[List[int]]:
        """Apply vertical flip to grid."""
        return np.flipud(np.asarray(grid, dtype=int)).tolist()

    @staticmethod
    def _detect_translation(
        grid_before: Sequence[Sequence[int]],
        grid_after: Sequence[Sequence[int]],
    ) -> Tuple[int, int] | None:
        """
        Detect translation offset between grids.

        Returns:
            (dx, dy) if translation detected, None otherwise.
        """
        before = np.asarray(grid_before, dtype=int)
        after = np.asarray(grid_after, dtype=int)

        if before.shape != after.shape:
            return None

        h, w = before.shape

        # Brute-force search over plausible integer shifts within bounds.
        for dy in range(-h + 1, h):
            for dx in range(-w + 1, w):
                shifted = np.zeros_like(before)
                src_y_start = max(0, -dy)
                src_y_end = min(h, h - dy)
                src_x_start = max(0, -dx)
                src_x_end = min(w, w - dx)

                dst_y_start = max(0, dy)
                dst_y_end = dst_y_start + (src_y_end - src_y_start)
                dst_x_start = max(0, dx)
                dst_x_end = dst_x_start + (src_x_end - src_x_start)

                if src_y_end <= src_y_start or src_x_end <= src_x_start:
                    continue

                shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = before[
                    src_y_start:src_y_end, src_x_start:src_x_end
                ]

                if np.array_equal(shifted, after):
                    return dx, dy

        return None

    @staticmethod
    def _apply_translation(grid: Sequence[Sequence[int]], dx: int, dy: int) -> List[List[int]]:
        """Apply translation to grid with zero fill."""
        arr = np.asarray(grid, dtype=int)
        h, w = arr.shape
        out = np.zeros_like(arr)
        for y in range(h):
            for x in range(w):
                ny = y + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    out[ny, nx] = arr[y, x]
        return out.tolist()

    @staticmethod
    def _grids_match(
        grid1: Sequence[Sequence[int]],
        grid2: Sequence[Sequence[int]],
    ) -> bool:
        """Check if two grids are identical."""
        return np.array_equal(np.asarray(grid1, dtype=int), np.asarray(grid2, dtype=int))

    @staticmethod
    def _detect_color_map(
        grid_before: Sequence[Sequence[int]],
        grid_after: Sequence[Sequence[int]],
    ) -> Tuple[int, int] | None:
        """Detect a simple one-to-one color remap if only one color changed."""
        before = np.asarray(grid_before, dtype=int)
        after = np.asarray(grid_after, dtype=int)
        if before.shape != after.shape:
            return None
        diff_mask = before != after
        if not diff_mask.any():
            return None
        src_colors = before[diff_mask]
        dst_colors = after[diff_mask]
        if len(np.unique(src_colors)) == 1 and len(np.unique(dst_colors)) == 1:
            return int(src_colors[0]), int(dst_colors[0])
        return None

    def _prioritize(self, primitives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple prioritization: favor recolor, then translate/rotate-translate, else first."""
        if not primitives:
            return {
                "primitive": "UNKNOWN",
                "parameters": {},
                "rpn_program": "",
                "confidence": 0.0,
            }
        recolors = [p for p in primitives if "RECOLOR" in p["primitive"]]
        if recolors:
            return recolors[0]
        translates = [
            p
            for p in primitives
            if p["primitive"].startswith("TRANSLATE") or p["primitive"].startswith("ROTATE_TRANSLATE")
        ]
        if translates:
            return translates[0]
        return primitives[0]


def example_arc_grid_processing() -> Tuple[ARCGridProcessor, Dict[str, Any]]:
    """Example of how to process ARC-AGI grids."""
    processor = ARCGridProcessor(matryoshka_dim=512)

    grid_input = [
        [0, 1, 0],
        [1, 2, 1],
        [0, 1, 0],
    ]

    grid_output = [
        [0, 0, 1],
        [1, 2, 1],
        [1, 0, 0],
    ]

    rpn_program = processor.grid_to_rpn_program(grid_input)
    print(f"RPN Program: {rpn_program}")

    embedding = processor.grid_to_spatial_embedding(grid_input)
    print(f"Spatial Embedding: {embedding.shape}")

    transformation = processor.detect_spatial_primitive(grid_input, grid_output)
    print(f"Detected Transformation: {transformation}")

    return processor, transformation


if __name__ == "__main__":
    _, _ = example_arc_grid_processing()
    print("✅ Grid processor working!")
