"""
Ternary video codec leveraging procedural texture generation and ternary DCT residuals.

GPU-only: uses sovereign PTX DCT8x8 binding with no CPU fallback. If the GPU path
fails to initialise, the codec raises instead of silently dropping to NumPy.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from .procedural_video import ProceduralVideoGenerator
from .ternary_quantization import dequantize_ternary, quantize_ternary
from .ptx_bindings import TernaryDCT8x8Kernel


class TernaryVideoCodec:
    """
    Ternary video codec that pairs procedural seeds with ternary-quantised DCT residuals.
    All transforms run on GPU (ternary_dct8x8_binding). No CPU fallback path.
    """

    def __init__(self, width: int = 1920, height: int = 1080):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("width and height must be multiples of 8 for 8x8 DCT blocks")
        self.width = int(width)
        self.height = int(height)
        self.generator = ProceduralVideoGenerator(width=width, height=height)
        self.dct8 = TernaryDCT8x8Kernel()

    def _reshape_blocks(self, img: np.ndarray) -> np.ndarray:
        """Reshape (H,W) into (num_blocks,8,8) contiguous blocks."""
        h_blocks = self.height // 8
        w_blocks = self.width // 8
        reshaped = img.reshape(h_blocks, 8, w_blocks, 8).swapaxes(1, 2).reshape(h_blocks * w_blocks, 8, 8)
        return np.ascontiguousarray(reshaped, dtype=np.float32)

    def _blocks_to_image(self, blocks: np.ndarray) -> np.ndarray:
        """Inverse of _reshape_blocks for a single channel."""
        h_blocks = self.height // 8
        w_blocks = self.width // 8
        blocks = blocks.reshape(h_blocks, w_blocks, 8, 8).swapaxes(1, 2)
        return blocks.reshape(self.height, self.width)

    def encode(self, frame: np.ndarray, seed: Optional[np.ndarray] = None) -> Dict:
        """
        Encode a single RGB frame using GPU 8x8 DCT and ternary quantisation.
        """
        img = np.asarray(frame, dtype=np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("frame must have shape (H, W, 3)")
        if img.shape[0] != self.height or img.shape[1] != self.width:
            raise ValueError(f"frame must match codec resolution ({self.height}, {self.width})")

        if seed is None:
            mean_channels = img.mean(axis=(0, 1))
            std_channels = img.std(axis=(0, 1)) + 1e-6
            seed = np.concatenate([mean_channels, std_channels]).astype(np.float32)
        else:
            seed = np.asarray(seed, dtype=np.float32)

        procedural = self.generator.generate_frame(seed)
        residual = img - procedural.astype(np.float32)

        coeffs_channels = []
        for c in range(3):
            blocks = self._reshape_blocks(residual[:, :, c])
            dct_blocks = self.dct8.forward(blocks)
            coeffs_channels.append(self._blocks_to_image(dct_blocks))
        coeffs = np.stack(coeffs_channels, axis=-1)

        quantized, meta = quantize_ternary(coeffs, adaptive=False, threshold=0.2)
        return {
            "seed": seed,
            "quantized": quantized,
            "metadata": meta,
            "width": self.width,
            "height": self.height,
        }

    def decode(self, encoded: Dict) -> np.ndarray:
        """
        Decode a single RGB frame from the encoded representation using GPU inverse DCT.
        """
        for key in ("seed", "quantized", "metadata"):
            if key not in encoded:
                raise ValueError(f"encoded missing '{key}'")

        seed = np.asarray(encoded["seed"], dtype=np.float32)
        quantized = np.asarray(encoded["quantized"], dtype=np.int8)
        meta = encoded.get("metadata")

        coeffs = dequantize_ternary(quantized, metadata=meta)
        if coeffs.shape != (self.height, self.width, 3):
            raise ValueError("quantized shape does not match expected frame dimensions")

        residual_channels = []
        for c in range(3):
            blocks = self._reshape_blocks(coeffs[:, :, c])
            time_blocks = self.dct8.inverse(blocks)
            residual_channels.append(self._blocks_to_image(time_blocks))
        residual = np.stack(residual_channels, axis=-1)

        procedural = self.generator.generate_frame(seed).astype(np.float32)
        reconstructed = np.clip(procedural + residual, 0.0, 255.0)
        return reconstructed.astype(np.uint8)

    def compute_compression_ratio(self, original_size: int, encoded: Dict) -> float:
        """
        Estimate compression ratio given encoded payload.
        """
        quantized = np.asarray(encoded.get("quantized"))
        seed = np.asarray(encoded.get("seed"))
        compressed_size = quantized.size + seed.size * 4
        if compressed_size == 0:
            return float("inf")
        return float(original_size) / float(compressed_size)
