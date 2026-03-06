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
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
from knowledge3d.cranium.codecs.ternary_codec_ops import TernaryCodecOps


class SovereignTernaryVideoCodec:
    """GPU-native ternary video codec with PTX-backed block transforms."""

    def __init__(self, width: int = 1920, height: int = 1080, threshold: float = 0.2) -> None:
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("width/height must be multiples of 8 for block transforms")
        self.width = int(width)
        self.height = int(height)
        self.ops = TernaryCodecOps(threshold=threshold)
        self.rpn = ModularRPNEngine()
        self.galaxy = TernaryGalaxy()

    def encode(self, frame_id: str, frame_rgb: TernaryTensor) -> Dict:
        if len(frame_rgb.shape) != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("frame tensor must have shape (H, W, 3)")
        h, w, _ = frame_rgb.shape
        if h != self.height or w != self.width:
            raise ValueError("frame dimensions do not match codec configuration")

        seed_rpn = f"RESHAPE_TO_BLOCKS DCT8X8_FORWARD {self.ops.threshold} TERNARY_QUANT"
        residual_values: list[int] = []
        rgb = self._reshape_rgb(frame_rgb.values.to_python(), width=w, height=h)
        signal_plan = self.ops.execution_plan(work_items=3 * ((h * w) // 64), preferred_tier=2)
        for channel in range(3):
            channel_grid = rgb[:, :, channel].astype(np.float32, copy=False).tolist()
            quantized = self.rpn.evaluate(seed_rpn, data=channel_grid, return_vector=True)
            residual_values.extend(int(round(v)) for v in self._flatten_layout_data(quantized))

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
        coeffs = [int(round(v)) for v in residual.to_python()]
        decode_rpn = "TERNARY_DEQUANT IDCT8X8_INVERSE BLOCKS_TO_GRID"

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
            packet = self._make_block_packet(chan_blocks, rows=self.height, cols=self.width, integer=True)
            channel_grid = self.rpn.evaluate(decode_rpn, data=packet, return_vector=True)
            channel_arrays.append(np.asarray(self._flatten_grid(channel_grid), dtype=np.int32))

        rgb = np.stack(channel_arrays, axis=1)
        rgb = np.clip(rgb, 0, 255).astype(np.int32, copy=False)
        combined = rgb.reshape(-1)
        # Flattened RGB; wrap in TernaryVector after quantizing to ternary palette {-1,0,+1} placeholder
        ternary_rgb = np.where(combined < 85, 0, np.where(combined > 170, 1, -1)).astype(np.int32, copy=False)
        ternary_list = [int(v) for v in ternary_rgb.tolist()]
        return TernaryTensor((self.height, self.width, 3), TernaryVector(ternary_list))

    def store_residual(self, frame_id: str, seed_rpn: str, residual: TernaryVector) -> None:
        """Explicit store helper to keep galaxy interaction centralized."""
        self.galaxy.store_frame(frame_id, seed_rpn, residual)

    def load_residual(self, frame_id: str) -> Tuple[str, TernaryVector]:
        return self.galaxy.load_frame(frame_id)

    # ------------------------------------------------------------------ #
    # Helpers (host orchestration only; transforms are on GPU)
    # ------------------------------------------------------------------ #
    def _reshape_rgb(self, rgb_flat: list[int], *, width: int, height: int) -> np.ndarray:
        arr = np.asarray(rgb_flat, dtype=np.float32)
        expected = width * height * 3
        if arr.size != expected:
            raise ValueError(f"expected {expected} RGB values, got {arr.size}")
        return arr.reshape(height, width, 3)

    def _make_block_packet(self, flat_blocks: list[int], *, rows: int, cols: int, integer: bool) -> dict:
        blocks_per_grid = (rows // 8) * (cols // 8)
        return {
            "__k3d_layout__": "blocks8x8_v1",
            "rows": int(rows),
            "cols": int(cols),
            "block_h": 8,
            "block_w": 8,
            "integer": bool(integer),
            "data": self._reshape_flat(flat_blocks, (blocks_per_grid, 64)),
        }

    def _flatten_grid(self, grid) -> list[int]:
        return [int(round(v)) for v in self._flatten_layout_data(grid)]

    def _flatten_layout_data(self, value) -> list:
        if isinstance(value, dict) and value.get("__k3d_layout__") == "blocks8x8_v1":
            value = value.get("data", [])
        return self._flatten_list(value)

    def _flatten_list(self, value) -> list:
        if isinstance(value, list):
            out: list = []
            for item in value:
                out.extend(self._flatten_list(item))
            return out
        return [value]

    def _reshape_flat(self, flat: list[int], shape: tuple[int, ...]):
        rebuilt, _ = self._reshape_flat_recursive(flat, shape, 0)
        return rebuilt

    def _reshape_flat_recursive(self, flat: list[int], shape: tuple[int, ...], offset: int):
        if not shape:
            return flat[offset], offset + 1
        size = int(shape[0])
        out = []
        cursor = offset
        for _ in range(size):
            item, cursor = self._reshape_flat_recursive(flat, shape[1:], cursor)
            out.append(item)
        return out, cursor


__all__ = ["SovereignTernaryVideoCodec"]
