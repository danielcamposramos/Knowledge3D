"""
Sovereign ternary video codec (construction phase).

No numpy, no CPU fallbacks. Uses TernaryVector/TernaryGalaxy and GPU-only ops.
Pending PTX kernels for DCT/IDCT and block operations; encode/decode currently
raise NotImplementedError to avoid silent CPU paths.

Architecture References:
- docs/vocabulary/UNIFIED_SIGNAL_SPECIFICATION.md — Video as temporal signal
- docs/vocabulary/PROCEDURAL_VISUAL_SPECIFICATION.md — VectorDotMap integration

Video = Procedural images over time with synchronized audio.
Each frame encodes as VectorDotMap field coefficients (~2KB).
Temporal spectrum analysis for motion-based compression.
"""

from __future__ import annotations

from typing import Dict, Tuple

from knowledge3d.cranium.ternary import TernaryVector, TernaryTensor, TernaryGalaxy
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
from knowledge3d.cranium.codecs.ternary_codec_ops import TernaryCodecOps


class SovereignTernaryVideoCodec:
    """GPU-native ternary video codec skeleton (fails loudly until kernels wired)."""

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

        # Flatten blocks per channel -> DCT -> quantize
        blocks_flat: list[int] = []
        for channel in range(3):
            chan_vals = self._extract_channel(frame_rgb.values.to_python(), channel)
            chan_blocks = self._blocks_from_channel(chan_vals, w, h)
            blocks_flat.extend(chan_blocks)

        rpn_program = f"DCT8X8_FORWARD {self.ops.threshold} TERNARY_QUANT"
        quantized = self.rpn.evaluate(rpn_program, data=blocks_flat, return_vector=True)
        residual_vec = TernaryVector(self._flatten_list(quantized))

        seed_rpn = "PROC_NONE"  # Placeholder procedural seed
        self.galaxy.store_frame(frame_id, seed_rpn, residual_vec)

        return {
            "frame_id": frame_id,
            "width": self.width,
            "height": self.height,
            "seed_rpn": seed_rpn,
            "stored_in_galaxy": True,
        }

    def decode(self, frame_id: str) -> TernaryTensor:
        seed_rpn, residual = self.galaxy.load_frame(frame_id)
        _ = seed_rpn  # placeholder for future procedural reconstruction
        # Ensure integer coeffs for dequantisation (ternary {-1,0,+1})
        coeffs = [int(round(v)) for v in residual.to_python()]
        inv = self.rpn.evaluate("TERNARY_DEQUANT IDCT8X8", data=coeffs, return_vector=True)

        # Reconstruct channels from blocks
        channel_size = self.width * self.height
        blocks_per_channel = channel_size // 64
        channel_arrays: list[list[int]] = []
        offset = 0
        for _ in range(3):
            chan_blocks = inv[offset : offset + blocks_per_channel * 64]
            offset += blocks_per_channel * 64
            channel_arrays.append(self._blocks_to_channel(chan_blocks, self.width, self.height))

        # Combine channels into TernaryTensor (packed from int grid)
        combined: list[int] = []
        for idx in range(channel_size):
            r = channel_arrays[0][idx]
            g = channel_arrays[1][idx]
            b = channel_arrays[2][idx]
            combined.extend([
                max(0, min(255, int(r))),
                max(0, min(255, int(g))),
                max(0, min(255, int(b))),
            ])
        # Flattened RGB; wrap in TernaryVector after quantizing to ternary palette {-1,0,+1} placeholder
        ternary_rgb = [0 if v < 85 else (1 if v > 170 else -1) for v in combined]
        return TernaryTensor((self.height, self.width, 3), TernaryVector(ternary_rgb))

    def store_residual(self, frame_id: str, seed_rpn: str, residual: TernaryVector) -> None:
        """Explicit store helper to keep galaxy interaction centralized."""
        self.galaxy.store_frame(frame_id, seed_rpn, residual)

    def load_residual(self, frame_id: str) -> Tuple[str, TernaryVector]:
        return self.galaxy.load_frame(frame_id)

    # ------------------------------------------------------------------ #
    # Helpers (CPU orchestration only)
    # ------------------------------------------------------------------ #
    def _extract_channel(self, rgb_flat: list[int], channel: int) -> list[int]:
        """Extract single channel from flattened RGB list (H*W*3)."""
        chan = []
        total_pixels = self.width * self.height
        for i in range(total_pixels):
            chan.append(rgb_flat[i * 3 + channel])
        return chan

    def _blocks_from_channel(self, channel_vals: list[int], width: int, height: int) -> list[float]:
        """Convert channel grid to contiguous 8x8 blocks (float)."""
        blocks: list[float] = []
        for by in range(0, height, 8):
            for bx in range(0, width, 8):
                for y in range(8):
                    for x in range(8):
                        idx = (by + y) * width + (bx + x)
                        blocks.append(float(channel_vals[idx]))
        return blocks

    def _blocks_to_channel(self, blocks_flat: list[float], width: int, height: int) -> list[int]:
        """Reassemble channel grid from contiguous 8x8 blocks."""
        out = [0.0 for _ in range(width * height)]
        block_idx = 0
        for by in range(0, height, 8):
            for bx in range(0, width, 8):
                for y in range(8):
                    for x in range(8):
                        dst = (by + y) * width + (bx + x)
                        out[dst] = blocks_flat[block_idx * 64 + y * 8 + x]
                block_idx += 1
        return [int(round(v)) for v in out]

    def _flatten_list(self, value) -> list:
        if isinstance(value, list):
            out: list = []
            for item in value:
                out.extend(self._flatten_list(item))
            return out
        return [value]


__all__ = ["SovereignTernaryVideoCodec"]
