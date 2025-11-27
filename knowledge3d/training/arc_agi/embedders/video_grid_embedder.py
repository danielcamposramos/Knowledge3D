from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from knowledge3d.cranium.codecs import SovereignTernaryVideoCodec
from knowledge3d.cranium.ternary import TernaryVector, TernaryTensor
from knowledge3d.training.arc_agi.sovereign_utils import flatten, pad_or_truncate

from knowledge3d.training.arc_agi.sovereign_utils import flatten, pad_or_truncate, zeros2d


class VideoGridEmbedder:
    """
    Treat ARC grids as tiny video frames and extract DCT-based features
    using the ternary video codec.

    This lives strictly on the ingestion/training side. It relies on the
    sovereign `TernaryVideoCodec` PTX path but is written so tests can inject
    a lightweight fake codec without touching GPU.
    """

    def __init__(
        self,
        width: int = 32,
        height: int = 32,
        codec: Any | None = None,
    ):
        """
        Args:
            width: Target frame width (must be multiple of 8).
            height: Target frame height (must be multiple of 8).
            codec: Optional codec instance with an `encode(frame)` method.
                If omitted, we lazily import and construct `TernaryVideoCodec`.
        """
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("width and height must be multiples of 8")

        self.width = int(width)
        self.height = int(height)

        if codec is None:
            # Lazy import to avoid loading GPU bindings in environments that
            # only exercise CPU tests.
            from knowledge3d.cranium.codecs.ternary_video_codec import (
                TernaryVideoCodec,
            )

            self.codec = TernaryVideoCodec(width=self.width, height=self.height)
        else:
            self.codec = codec

        # ARC color palette (0-9) mapped to RGB.
        self.palette: Dict[int, tuple[int, int, int]] = {
            0: (0, 0, 0),
            1: (0, 116, 217),
            2: (255, 65, 54),
            3: (46, 204, 64),
            4: (255, 220, 0),
            5: (170, 170, 170),
            6: (240, 18, 190),
            7: (255, 133, 27),
            8: (127, 219, 255),
            9: (135, 12, 37),
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def grid_to_video_embedding(
        self,
        grid: Sequence[Sequence[int]],
    ) -> List[float]:
        """
        Convert grid to video-style embedding using ternary DCT features.

        Returns:
            1D float32 feature vector (length ~510 before Matryoshka).
        """
        padded = self._pad_to_frame_size(grid)
        rgb_flat: List[int] = []
        for row in padded:
            for color in row:
                rgb_flat.extend([int(color)] * 3)  # replicate across RGB
        # Map to ternary palette for tensor
        ternary_rgb = [0 if v == 0 else (1 if v > 5 else -1) for v in rgb_flat]
        tensor = TernaryTensor((self.height, self.width, 3), TernaryVector(ternary_rgb))

        # Allow injected legacy codec for tests; otherwise use sovereign path.
        if hasattr(self, "codec") and self.codec is not None and not isinstance(self.codec, SovereignTernaryVideoCodec):
            import numpy as np  # local, test-only

            rgb_frame = np.array(self._grid_to_rgb(padded), dtype=np.uint8)
            encoded = self.codec.encode(rgb_frame)  # type: ignore[arg-type]
            quantized_raw = encoded.get("quantized", [])
            if hasattr(quantized_raw, "ravel"):
                quantized_flat = [int(v) for v in quantized_raw.ravel().tolist()]  # type: ignore[arg-type]
            else:
                quantized_flat = [int(v) for v in flatten([quantized_raw])]
            stats = self._compute_ternary_stats(quantized_flat)
            tail_len = 500
            tail = pad_or_truncate([float(v) for v in quantized_flat[:tail_len]], tail_len, 0.0)
            features: List[float] = []
            features.extend([float(v) for v in flatten([encoded.get("seed", [])])])
            features.extend(
                [
                    float(stats["sparsity"]),
                    float(stats["pos_ratio"]),
                    float(stats["neg_ratio"]),
                    float(stats["entropy"]),
                ]
            )
            features.extend(tail)
            decoded_vals = features
        else:
            codec = SovereignTernaryVideoCodec(width=self.width, height=self.height)
            codec.encode("frame_embed", tensor)
            decoded = codec.decode("frame_embed")
            decoded_vals = decoded.values.to_python()

        # Use flattened ternary values; project to fixed embedding length (510 as before).
        return pad_or_truncate([float(v) for v in decoded_vals], 510, 0.0)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _pad_to_frame_size(self, grid: Sequence[Sequence[int]]) -> Sequence[Sequence[int]]:
        """Pad (or crop) grid to codec frame size."""
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        arr = zeros2d(self.height, self.width, 0)
        if h == 0 or w == 0:
            return arr

        h_use = min(h, self.height)
        w_use = min(w, self.width)
        for y in range(h_use):
            for x in range(w_use):
                arr[y][x] = int(grid[y][x])
        return arr  # type: ignore[return-value]

    def _grid_to_rgb(self, grid: Sequence[Sequence[int]]) -> Sequence[Sequence[Sequence[int]]]:
        """Map grid colors 0-9 to RGB frame."""
        rgb = [[[0, 0, 0] for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                color_idx = grid[y][x] if y < len(grid) and x < len(grid[y]) else 0
                rgb_value = self.palette.get(int(color_idx), (0, 0, 0))
                rgb[y][x] = [int(rgb_value[0]), int(rgb_value[1]), int(rgb_value[2])]
        return rgb

    @staticmethod
    def _compute_ternary_stats(quantized: Sequence[int]) -> Dict[str, float]:
        """Compute basic statistics of ternary quantisation."""
        q = [int(v) for v in quantized]
        total = len(q)
        if total == 0:
            return {
                "sparsity": 1.0,
                "pos_ratio": 0.0,
                "neg_ratio": 0.0,
                "entropy": 0.0,
            }

        zeros = sum(1 for v in q if v == 0)
        pos = sum(1 for v in q if v == 1)
        neg = sum(1 for v in q if v == -1)

        p_zero = zeros / total
        p_pos = pos / total
        p_neg = neg / total

        entropy = 0.0
        for p in (p_zero, p_pos, p_neg):
            if p > 0.0:
                entropy -= p * float(math.log2(p))

        return {
            "sparsity": float(p_zero),
            "pos_ratio": float(p_pos),
            "neg_ratio": float(p_neg),
            "entropy": float(entropy),
        }
