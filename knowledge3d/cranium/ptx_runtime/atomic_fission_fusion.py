"""Thin wrapper for AtomicFissionFusion maintaining Kimi's zero-copy strategy.

This module provides direct access to the operational AtomicFissionFusion from
sovereign_bridges without forcing unnecessary memory copies.
"""
from knowledge3d.cranium.bridges.sovereign_bridges import AtomicFissionFusion

__all__ = ["AtomicFissionFusion"]
