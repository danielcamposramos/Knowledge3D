"""Legacy import that now forwards to the sovereign implementation."""

from knowledge3d.cranium.spatial_sovereign.led_pathfinder import LEDPathfinderSovereign


class LEDPathfinder(LEDPathfinderSovereign):
    """Compatibility alias."""


__all__ = ["LEDPathfinder", "LEDPathfinderSovereign"]
