"""
LocalPerceptionEncoder: SAM-base inspired local text perception.

Maps DeepSeek-OCR's SAM-base (80M params) to K3D's sovereign stack.
Uses window attention for fine-grained character/word recognition.

Phase E: Simple CPU stubs (resize + pooling)
Phase F: Full PTX implementation with window attention kernels
"""

from __future__ import annotations

import numpy as np


class LocalPerceptionEncoder:
    """
    SAM-base inspired local text perception.

    DeepSeek approach:
    - Window attention with local receptive fields
    - Fine-grained feature extraction (character/word level)
    - Output: Local features at 1/4 resolution

    K3D Phase E implementation:
    - Stub: Simple resize + pooling
    - Phase F: PTX window attention kernels
    """

    def __init__(self, window_size: int = 16):
        """
        Initialize local perception encoder.

        Args:
            window_size: Window size for local attention (default: 16)
        """
        self.window_size = window_size
        self.feature_dim = 256  # Match DeepSeek feature dimension

    def encode_local_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract local text features using window attention.

        Args:
            image: Input image (H, W, 3) RGB uint8

        Returns:
            Local features (H/4, W/4, 256) float32
        """
        try:
            from skimage.transform import resize  # type: ignore
        except ImportError:
            # Fallback: simple numpy resize
            return self._resize_numpy(image)

        # Convert to grayscale for Phase E
        if image.ndim == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            gray = gray.astype(np.float32)
        else:
            gray = image.astype(np.float32)

        # Normalize to [0, 1]
        if gray.max() > 1.0:
            gray = gray / 255.0

        # DeepSeek-style: Extract features at 1/4 resolution
        h, w = gray.shape[:2]
        target_h = max(1, h // 4)
        target_w = max(1, w // 4)

        # Phase E: Simple resize (Phase F will use PTX window attention)
        features_2d = resize(
            gray,
            (target_h, target_w),
            order=1,  # Bilinear
            anti_aliasing=True,
            preserve_range=False
        ).astype(np.float32)

        # Expand to feature_dim channels (Phase E: simple tiling)
        # Phase F: PTX kernel will compute actual window attention features
        features = np.tile(features_2d[:, :, np.newaxis], (1, 1, self.feature_dim))

        return features

    def _resize_numpy(self, image: np.ndarray) -> np.ndarray:
        """Fallback resize using numpy (no skimage dependency)."""
        if image.ndim == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image.astype(np.float32)

        if gray.max() > 1.0:
            gray = gray / 255.0

        h, w = gray.shape[:2]
        target_h = max(1, h // 4)
        target_w = max(1, w // 4)

        # Simple block averaging
        block_h = h // target_h
        block_w = w // target_w

        resized = np.zeros((target_h, target_w), dtype=np.float32)
        for i in range(target_h):
            for j in range(target_w):
                block = gray[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                resized[i, j] = block.mean()

        # Expand to feature_dim
        features = np.tile(resized[:, :, np.newaxis], (1, 1, self.feature_dim))
        return features
