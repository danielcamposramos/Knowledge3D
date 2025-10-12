"""Thin wrapper for ResonanceField maintaining Kimi's zero-copy strategy.

This module provides direct access to the operational ResonanceField from
sovereign_bridges without forcing unnecessary memory copies.
"""
from knowledge3d.cranium.bridges.sovereign_bridges import ResonanceField

__all__ = ["ResonanceField"]
