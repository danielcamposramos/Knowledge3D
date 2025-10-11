"""Galaxy Memory Updater using sovereign PTX architecture.

This module provides exponential moving average (EMA) blending of galaxy embeddings
using our sovereign GalaxyMemoryUpdater bridge (pure ctypes + hand-authored PTX).

Python is used ONLY for:
- Entry point (API convenience)
- I/O (loading/saving, path handling)

All computation happens on GPU via galaxy_memory_updater.ptx.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

# Import sovereign bridge
from knowledge3d.cranium.bridges.sovereign_bridges import GalaxyMemoryUpdater as SovereignGalaxyMemory


class GalaxyMemoryUpdater:
    """High-level galaxy embedding blender using sovereign PTX architecture.

    This is a thin Python wrapper around the sovereign GalaxyMemoryUpdater bridge.
    All GPU computation happens via hand-authored PTX kernels.

    Example:
        updater = GalaxyMemoryUpdater()
        old = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        teacher = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        new = updater.blend(old, teacher, blend_factor=0.3)
        # new ≈ [1.9, 2.9, 3.9] = old * 0.7 + teacher * 0.3
    """

    def __init__(self, galaxy_dir: Path | None = None, ptx_path: str | None = None):
        """Initialize galaxy memory updater with sovereign PTX backend.

        Args:
            galaxy_dir: Optional directory for galaxy data (for future use)
            ptx_path: Optional PTX path (ignored, uses sovereign loader)
        """
        self.galaxy_dir = Path(galaxy_dir) if galaxy_dir else None
        self._sovereign_updater = SovereignGalaxyMemory()

    def blend(self, old: np.ndarray, teacher: np.ndarray, blend_factor: float) -> np.ndarray:
        """Blend old and teacher embeddings using GPU-accelerated EMA.

        Args:
            old: Old embedding (any shape, will be flattened)
            teacher: Teacher embedding (same shape as old)
            blend_factor: Blend factor in [0, 1]
                         0.0 = keep old completely
                         1.0 = use teacher completely
                         0.3 = 70% old + 30% teacher (common default)

        Returns:
            Blended embedding (same shape as input)

        Formula:
            new = old * (1 - blend_factor) + teacher * blend_factor

        Example:
            >>> updater = GalaxyMemoryUpdater()
            >>> old = np.array([1.0, 2.0, 3.0])
            >>> teacher = np.array([4.0, 5.0, 6.0])
            >>> result = updater.blend(old, teacher, 0.3)
            >>> print(result)  # [1.9, 2.9, 3.9]
        """
        # Delegate to sovereign bridge (handles all GPU work)
        return self._sovereign_updater.blend(old, teacher, blend_factor)

    def blend_sequence(
        self,
        base: np.ndarray,
        teachers: List[np.ndarray],
        blend_factor: float = 0.3
    ) -> np.ndarray:
        """Blend base embedding with sequence of teacher embeddings.

        Applies EMA blending iteratively with each teacher in sequence.

        Args:
            base: Base embedding to start from
            teachers: List of teacher embeddings to blend in sequence
            blend_factor: Blend factor for each step (default 0.3)

        Returns:
            Final blended embedding after all teachers applied

        Example:
            >>> updater = GalaxyMemoryUpdater()
            >>> base = np.array([1.0, 1.0, 1.0])
            >>> teachers = [
            ...     np.array([2.0, 2.0, 2.0]),
            ...     np.array([3.0, 3.0, 3.0]),
            ... ]
            >>> result = updater.blend_sequence(base, teachers, 0.5)
            # Step 1: [1, 1, 1] + 0.5 * ([2, 2, 2] - [1, 1, 1]) = [1.5, 1.5, 1.5]
            # Step 2: [1.5, 1.5, 1.5] + 0.5 * ([3, 3, 3] - [1.5, 1.5, 1.5]) = [2.25, 2.25, 2.25]
        """
        # Delegate to sovereign bridge
        return self._sovereign_updater.blend_sequence(base, teachers, blend_factor)


__all__ = ["GalaxyMemoryUpdater"]
