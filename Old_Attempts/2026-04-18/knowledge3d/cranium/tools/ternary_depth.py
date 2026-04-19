"""
GPU-native ternary depth field computation.

Uses TernaryDepthField bridge to produce packed 2-bit trits indicating
attract / neutral / repel relative to a query embedding.

All computation is GPU-side; only the packed trits are returned to host.
"""

from __future__ import annotations

import numpy as np

from knowledge3d.cranium.bridges.sovereign_bridges import TernaryDepthField


class TernaryDepthComputer:
    """High-level wrapper for ternary depth fields."""

    def __init__(self, bridge: TernaryDepthField | None = None) -> None:
        self.bridge = bridge or TernaryDepthField()

    def compute(
        self,
        embeddings: np.ndarray,
        query: np.ndarray,
        attract_thresh: float = 0.35,
        repel_thresh: float = -0.05,
    ) -> np.ndarray:
        """
        Compute packed ternary depth trits for all nodes.

        Returns:
            np.ndarray uint32 packed (2 bits per node): 00=-1 (repel/far), 01=0 (neutral), 10=+1 (attract/near)
        """
        return self.bridge.compute(
            embeddings=embeddings,
            query=query,
            attract_thresh=attract_thresh,
            repel_thresh=repel_thresh,
        )
