"""
ConvolutionalCompressor: DeepSeek-inspired 16× convolutional compressor.

Maps DeepSeek-OCR's 16× compression stage to K3D's sovereign stack.
Reduces vision tokens via strided convolutions.

Phase E: Simple block pooling (max pooling)
Phase F: Full PTX strided convolution kernels
"""

from __future__ import annotations

import numpy as np


class ConvolutionalCompressor:
    """
    DeepSeek-inspired 16× convolutional compressor.

    DeepSeek approach:
    - Strided convolutions for spatial token reduction
    - 16× total reduction (4× from local encoder + 4× from compressor)
    - Preserves semantic structure while reducing tokens

    K3D Phase E implementation:
    - Stub: Block-based max pooling
    - Phase F: PTX strided convolution kernels
    """

    def __init__(self, compression_ratio: int = 16):
        """
        Initialize convolutional compressor.

        Args:
            compression_ratio: Target compression ratio (default: 16)
        """
        self.compression_ratio = compression_ratio
        self.stride = int(np.sqrt(compression_ratio // 4))  # Already 4× from local encoder

    def compress(self, features: np.ndarray) -> np.ndarray:
        """
        Compress local features via strided convolution.

        Args:
            features: Local features (H/4, W/4, 256) from LocalPerceptionEncoder

        Returns:
            Compressed features (H/16, W/16, 256) float32
        """
        if features.size == 0:
            return np.zeros((1, 1, features.shape[-1] if features.ndim == 3 else 256), dtype=np.float32)

        h, w = features.shape[:2]
        channels = features.shape[2] if features.ndim == 3 else 256

        # DeepSeek approach: Strided conv with downsampling
        # Phase E: Use block pooling (Phase F: PTX strided conv)
        target_h = max(1, h // self.stride)
        target_w = max(1, w // self.stride)

        try:
            from skimage.measure import block_reduce  # type: ignore
            compressed = block_reduce(
                features,
                block_size=(self.stride, self.stride, 1),
                func=np.max
            )
        except ImportError:
            # Fallback: manual max pooling
            compressed = self._max_pool_manual(features, target_h, target_w)

        return compressed.astype(np.float32)

    def _max_pool_manual(self, features: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Manual max pooling fallback (no skimage dependency)."""
        h, w, c = features.shape
        block_h = h // target_h
        block_w = w // target_w

        pooled = np.zeros((target_h, target_w, c), dtype=np.float32)

        for i in range(target_h):
            for j in range(target_w):
                block = features[
                    i*block_h:(i+1)*block_h,
                    j*block_w:(j+1)*block_w,
                    :
                ]
                pooled[i, j, :] = block.max(axis=(0, 1))

        return pooled

    def get_compression_ratio(self, input_shape: tuple, output_shape: tuple) -> float:
        """
        Calculate actual compression ratio achieved.

        Args:
            input_shape: Input feature shape (H, W, C)
            output_shape: Output feature shape (H', W', C)

        Returns:
            Compression ratio (input_tokens / output_tokens)
        """
        input_tokens = input_shape[0] * input_shape[1]
        output_tokens = output_shape[0] * output_shape[1]

        if output_tokens == 0:
            return 1.0

        return float(input_tokens) / float(output_tokens)
