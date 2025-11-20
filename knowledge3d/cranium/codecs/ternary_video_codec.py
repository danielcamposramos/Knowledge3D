"""
Ternary video codec leveraging procedural texture generation and ternary DCT residuals.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .procedural_video import ProceduralVideoGenerator
from .ternary_quantization import dequantize_ternary, quantize_ternary


def _dct_1d(x: np.ndarray) -> np.ndarray:
    """DCT-II implementation (orthonormal)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    k = np.arange(n)
    factor = math.sqrt(2.0 / n)
    mat = np.cos(np.pi / n * (np.arange(n)[:, None] + 0.5) * k[None, :])
    mat[0, :] *= 1.0 / math.sqrt(2.0)
    return (factor * (mat @ x)).astype(np.float32)


def _idct_1d(X: np.ndarray) -> np.ndarray:
    """Inverse DCT-II (equals DCT-III with matching scaling)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.size
    k = np.arange(n)
    factor = math.sqrt(2.0 / n)
    mat = np.cos(np.pi / n * (k[:, None] + 0.5) * np.arange(n)[None, :])
    mat[:, 0] *= 1.0 / math.sqrt(2.0)
    return (factor * (mat.T @ X)).astype(np.float32)


def _dct2(block: np.ndarray) -> np.ndarray:
    """2D DCT-II via separable 1D transforms."""
    return np.apply_along_axis(_dct_1d, 0, np.apply_along_axis(_dct_1d, 1, block))


def _idct2(block: np.ndarray) -> np.ndarray:
    """2D inverse DCT-II."""
    return np.apply_along_axis(_idct_1d, 0, np.apply_along_axis(_idct_1d, 1, block))


class TernaryVideoCodec:
    """
    Minimal ternary video codec that pairs procedural seeds with ternary-quantised DCT residuals.
    """

    def __init__(self, width: int = 1920, height: int = 1080):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self.width = int(width)
        self.height = int(height)
        self.generator = ProceduralVideoGenerator(width=width, height=height)

    def encode(self, frame: np.ndarray, seed: Optional[np.ndarray] = None) -> Dict:
        """
        Encode a single RGB frame.

        Args:
            frame: Input RGB frame uint8/float in shape (H, W, 3).
            seed: Optional procedural seed; if None, derived from frame statistics.
        """
        img = np.asarray(frame, dtype=np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("frame must have shape (H, W, 3)")
        if img.shape[0] != self.height or img.shape[1] != self.width:
            raise ValueError(f"frame must match codec resolution ({self.height}, {self.width})")

        if seed is None:
            # Simple deterministic seed from mean/std to allow procedural baseline.
            mean_channels = img.mean(axis=(0, 1))
            std_channels = img.std(axis=(0, 1)) + 1e-6
            seed = np.concatenate([mean_channels, std_channels])

        procedural = self.generator.generate_frame(seed)
        proc_f = procedural.astype(np.float32)
        residual = img - proc_f

        # Apply 2D DCT per channel.
        coeffs = np.stack([_dct2(residual[:, :, c]) for c in range(3)], axis=-1)
        quantized, meta = quantize_ternary(coeffs, adaptive=True, threshold=0.2)

        return {
            "seed": np.asarray(seed, dtype=np.float32),
            "quantized": quantized,
            "metadata": meta,
            "width": self.width,
            "height": self.height,
        }

    def decode(self, encoded: Dict) -> np.ndarray:
        """
        Decode a single RGB frame from the encoded representation.
        """
        for key in ("seed", "quantized", "metadata"):
            if key not in encoded:
                raise ValueError(f"encoded missing '{key}'")

        seed = encoded["seed"]
        quantized = np.asarray(encoded["quantized"], dtype=np.int8)
        meta = encoded.get("metadata")

        coeffs = dequantize_ternary(quantized, metadata=meta)
        if coeffs.shape != (self.height, self.width, 3):
            raise ValueError("quantized shape does not match expected frame dimensions")

        residual = np.stack([_idct2(coeffs[:, :, c]) for c in range(3)], axis=-1)
        procedural = self.generator.generate_frame(seed).astype(np.float32)
        reconstructed = np.clip(procedural.astype(np.float32) + residual, 0.0, 255.0)
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
