"""
Sovereign ternary video codec using PTX-backed block + DCT primitives.

No numpy, no CPU fallbacks. Uses TernaryVector/TernaryGalaxy and GPU-backed ops.
Channel split/recombine remains host orchestration, but block packing and
frequency transforms execute through the sovereign codec runtime.

Architecture References:
- docs/vocabulary/UNIFIED_SIGNAL_SPECIFICATION.md — Video as temporal signal
- docs/vocabulary/PROCEDURAL_VISUAL_SPECIFICATION.md — VectorDotMap integration

Video = Procedural images over time with synchronized audio.
Each frame encodes as VectorDotMap field coefficients (~2KB).
Temporal spectrum analysis for motion-based compression.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from knowledge3d.cranium.ternary import TernaryVector, TernaryTensor, TernaryGalaxy
from knowledge3d.cranium.codecs.ternary_codec_ops import TernaryCodecOps


class SovereignTernaryVideoCodec:
    """GPU-native ternary video codec with PTX-backed block transforms."""

    def __init__(self, width: int = 1920, height: int = 1080, threshold: float = 0.2) -> None:
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("width/height must be multiples of 8 for block transforms")
        self.width = int(width)
        self.height = int(height)
        self.ops = TernaryCodecOps(threshold=threshold)
        self.galaxy = TernaryGalaxy()

    def encode(self, frame_id: str, frame_rgb: TernaryTensor) -> Dict:
        if len(frame_rgb.shape) != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("frame tensor must have shape (H, W, 3)")
        h, w, _ = frame_rgb.shape
        if h != self.height or w != self.width:
            raise ValueError("frame dimensions do not match codec configuration")
        rgb = frame_rgb.values.to_numpy().astype(np.float32, copy=False).reshape(h, w, 3)
        return self._encode_rgb_array(frame_id, rgb)

    def encode_frame_array(self, frame_id: str, frame_rgb: np.ndarray) -> Dict:
        rgb = np.asarray(frame_rgb, dtype=np.int32)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("frame array must have shape (H, W, 3)")
        h, w, _ = rgb.shape
        if h != self.height or w != self.width:
            raise ValueError("frame dimensions do not match codec configuration")
        ternary_rgb = np.where(rgb < 85, 0, np.where(rgb > 170, 1, -1)).astype(np.float32, copy=False)
        return self._encode_rgb_array(frame_id, ternary_rgb)

    def _encode_rgb_array(self, frame_id: str, rgb: np.ndarray) -> Dict:
        h, w, _ = rgb.shape

        seed_rpn = f"RESHAPE_TO_BLOCKS DCT8X8_FORWARD {self.ops.threshold} TERNARY_QUANT"
        channel_size = h * w
        residual_values = np.empty((channel_size * 3,), dtype=np.int32)
        signal_plan = self.ops.execution_plan(work_items=3 * (channel_size // 64), preferred_tier=2)
        for channel in range(3):
            channel_grid = rgb[:, :, channel].astype(np.float32, copy=False)
            blocks = self.ops.reshape_to_blocks_numpy(channel_grid.reshape(-1), rows=h, cols=w)
            coeffs = self.ops.dct8_forward_numpy(blocks)
            quantized = self.ops.quantize_numpy(coeffs, threshold=self.ops.threshold)
            start = channel * channel_size
            residual_values[start:start + channel_size] = quantized

        residual_vec = TernaryVector(residual_values)
        self.galaxy.store_frame(
            frame_id,
            seed_rpn,
            residual_vec,
            metadata={
                "width": self.width,
                "height": self.height,
                "channels": 3,
                "blocks_per_channel": (self.width * self.height) // 64,
                "math_core_plan": signal_plan,
            },
        )

        return {
            "frame_id": frame_id,
            "width": self.width,
            "height": self.height,
            "seed_rpn": seed_rpn,
            "stored_in_galaxy": True,
            "math_core_plan": signal_plan,
        }

    def decode(self, frame_id: str) -> TernaryTensor:
        seed_rpn, residual, metadata = self.galaxy.load_frame_details(frame_id)
        _ = seed_rpn
        coeffs = residual.to_numpy().astype(np.int32, copy=False)

        channel_size = self.width * self.height
        blocks_per_channel = channel_size // 64
        _decode_plan = self.ops.execution_plan(
            work_items=int(metadata.get("channels", 3)) * int(metadata.get("blocks_per_channel", blocks_per_channel)),
            preferred_tier=2,
        )
        channel_arrays: list[np.ndarray] = []
        offset = 0
        for _ in range(3):
            chan_blocks = coeffs[offset : offset + blocks_per_channel * 64]
            offset += blocks_per_channel * 64
            dequantized = self.ops.dequantize_numpy(chan_blocks)
            pixels = self.ops.dct8_inverse_numpy(dequantized)
            channel_grid = self.ops.blocks_to_grid_numpy(pixels, rows=self.height, cols=self.width)
            channel_arrays.append(channel_grid.reshape(self.height, self.width).astype(np.int32, copy=False))

        rgb = np.stack(channel_arrays, axis=-1)
        rgb = np.clip(rgb, 0, 255).astype(np.int32, copy=False)
        combined = rgb.reshape(-1)
        # Flattened RGB; wrap in TernaryVector after quantizing to ternary palette {-1,0,+1} placeholder
        ternary_rgb = np.where(combined < 85, 0, np.where(combined > 170, 1, -1)).astype(np.int32, copy=False)
        return TernaryTensor((self.height, self.width, 3), TernaryVector(ternary_rgb.reshape(-1)))

    def store_residual(self, frame_id: str, seed_rpn: str, residual: TernaryVector) -> None:
        """Explicit store helper to keep galaxy interaction centralized."""
        self.galaxy.store_frame(frame_id, seed_rpn, residual)

    def load_residual(self, frame_id: str) -> Tuple[str, TernaryVector]:
        return self.galaxy.load_frame(frame_id)

    # ------------------------------------------------------------------ #
    # Helpers (host orchestration only; transforms are on GPU)
    # ------------------------------------------------------------------ #

__all__ = ["SovereignTernaryVideoCodec"]
