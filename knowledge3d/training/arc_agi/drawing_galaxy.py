"""Compatibility wrapper for arc_agi.drawing_galaxy."""

from __future__ import annotations

import importlib


_module = importlib.import_module("drawing_galaxy")

DrawingGalaxy = _module.DrawingGalaxy

__all__ = ["DrawingGalaxy"]
