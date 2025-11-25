from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


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
    ) -> np.ndarray:
        """
        Convert grid to video-style embedding using ternary DCT features.

        Returns:
            1D float32 feature vector (length ~510 before Matryoshka).
        """
        padded = self._pad_to_frame_size(grid)
        rgb_frame = self._grid_to_rgb(padded)

        encoded = self.codec.encode(rgb_frame)
        seed = np.asarray(encoded["seed"], dtype=np.float32).ravel()
        quantized = np.asarray(encoded["quantized"], dtype=np.int8)

        stats = self._compute_ternary_stats(quantized)
        quantized_flat = quantized.ravel()

        # Truncate quantized tail for compact Matryoshka-friendly vector.
        tail_len = 500
        if quantized_flat.size < tail_len:
            tail = np.zeros(tail_len, dtype=np.float32)
            tail[: quantized_flat.size] = quantized_flat.astype(np.float32)
        else:
            tail = quantized_flat[:tail_len].astype(np.float32)

        features = np.concatenate(
            [
                seed.astype(np.float32),  # typically 6-D
                np.array(
                    [
                        stats["sparsity"],
                        stats["pos_ratio"],
                        stats["neg_ratio"],
                        stats["entropy"],
                    ],
                    dtype=np.float32,
                ),
                tail,
            ]
        )
        return features.astype(np.float32, copy=False)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _pad_to_frame_size(self, grid: Sequence[Sequence[int]]) -> np.ndarray:
        """Pad (or crop) grid to codec frame size."""
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        arr = np.zeros((self.height, self.width), dtype=np.uint8)
        if h == 0 or w == 0:
            return arr

        h_use = min(h, self.height)
        w_use = min(w, self.width)
        arr[:h_use, :w_use] = np.asarray(grid, dtype=np.uint8)[:h_use, :w_use]
        return arr

    def _grid_to_rgb(self, grid: np.ndarray) -> np.ndarray:
        """Map grid colors 0-9 to RGB frame."""
        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for color_idx, rgb_value in self.palette.items():
            mask = grid == color_idx
            if not np.any(mask):
                continue
            rgb[mask] = np.asarray(rgb_value, dtype=np.uint8)
        return rgb

    @staticmethod
    def _compute_ternary_stats(quantized: np.ndarray) -> Dict[str, float]:
        """Compute basic statistics of ternary quantisation."""
        q = np.asarray(quantized, dtype=np.int8)
        total = int(q.size)
        if total == 0:
            return {
                "sparsity": 1.0,
                "pos_ratio": 0.0,
                "neg_ratio": 0.0,
                "entropy": 0.0,
            }

        zeros = int(np.sum(q == 0))
        pos = int(np.sum(q == 1))
        neg = int(np.sum(q == -1))

        p_zero = zeros / total
        p_pos = pos / total
        p_neg = neg / total

        entropy = 0.0
        for p in (p_zero, p_pos, p_neg):
            if p > 0.0:
                entropy -= p * float(np.log2(p))

        return {
            "sparsity": float(p_zero),
            "pos_ratio": float(p_pos),
            "neg_ratio": float(p_neg),
            "entropy": float(entropy),
        }

