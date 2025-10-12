"""Thin wrapper for TemporalReasoning maintaining Kimi's zero-copy strategy.

This module provides direct access to the operational TemporalReasoning from
sovereign_bridges without forcing unnecessary memory copies.
"""
from knowledge3d.cranium.bridges.sovereign_bridges import TemporalReasoning

__all__ = ["TemporalReasoning"]
