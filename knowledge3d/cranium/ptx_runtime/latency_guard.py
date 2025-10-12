"""Thin wrapper for LatencyGuard maintaining Kimi's zero-copy strategy.

This module provides direct access to the operational LatencyGuard from
sovereign_bridges without forcing unnecessary memory copies.
"""
from knowledge3d.cranium.bridges.sovereign_bridges import LatencyGuard

__all__ = ["LatencyGuard"]
