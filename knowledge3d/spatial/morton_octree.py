"""Legacy import path for the sovereign Morton octree implementation."""

from knowledge3d.cranium.spatial_sovereign.morton_octree import (
    MortonOctreeSovereign,
)


class MortonOctree(MortonOctreeSovereign):
    """Backward compatible alias for existing imports."""


__all__ = ["MortonOctree", "MortonOctreeSovereign"]
