from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from knowledge3d.cranium.codecs import SovereignTernaryVideoCodec
from knowledge3d.cranium.ternary import TernaryVector, TernaryTensor
from knowledge3d.training.arc_agi.sovereign_utils import pad_or_truncate, zeros2d


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
            self.codec = SovereignTernaryVideoCodec(width=self.width, height=self.height)
        else:
            if not isinstance(codec, SovereignTernaryVideoCodec):
                raise RuntimeError("CPU/legacy video codecs are not permitted in sovereign path")
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

        if not isinstance(self.codec, SovereignTernaryVideoCodec):
            raise RuntimeError("Non-sovereign video codec detected; CPU fallbacks are forbidden")

        codec = self.codec
        codec.encode("frame_embed", tensor)
        decoded = codec.decode("frame_embed")
        decoded_vals = decoded.values.to_python()

        # Use flattened ternary values; project to fixed embedding length (510 as before).
        return pad_or_truncate([float(v) for v in decoded_vals], 510, 0.0)

    def grid_to_video_embedding_batch(
        self,
        grids: Sequence[Sequence[Sequence[int]]],
    ) -> List[List[float]]:
        """
        Batch video embeddings; uses sovereign codec path when available.
        """
        if not grids:
            return []

        if not isinstance(getattr(self, "codec", None), SovereignTernaryVideoCodec):
            return [self.grid_to_video_embedding(g) for g in grids]

        blocks_all: list[float] = []
        grids_count = len(grids)
        # Each grid contributes fixed number of blocks.
        blocks_per_grid = (self.width * self.height // 64) * 3
        values_per_grid = blocks_per_grid * 64

        for grid in grids:
            padded = self._pad_to_frame_size(grid)
            rgb_flat: List[int] = []
            for row in padded:
                for color in row:
                    rgb_flat.extend([int(color)] * 3)
            ternary_rgb = [0 if v == 0 else (1 if v > 5 else -1) for v in rgb_flat]
            tensor = TernaryTensor((self.height, self.width, 3), TernaryVector(ternary_rgb))
            # Use codec helpers to build contiguous blocks for all channels.
            for channel in range(3):
                chan_vals = self.codec._extract_channel(tensor.values.to_python(), channel)
                chan_blocks = self.codec._blocks_from_channel(chan_vals, self.width, self.height)
                blocks_all.extend(chan_blocks)

        rpn_program = f"DCT8X8_FORWARD {self.codec.ops.threshold} TERNARY_QUANT"
        quantized_all = self.codec.rpn.evaluate(rpn_program, data=blocks_all, return_vector=True)

        embeddings: List[List[float]] = []
        for idx in range(grids_count):
            start = idx * values_per_grid
            end = start + values_per_grid
            slice_vals = quantized_all[start:end]
            embeddings.append(pad_or_truncate([float(v) for v in slice_vals], 510, 0.0))

        return embeddings

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
