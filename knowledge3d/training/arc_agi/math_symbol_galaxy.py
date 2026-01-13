"""Compatibility wrapper for arc_agi.math_symbol_galaxy."""

from __future__ import annotations

import importlib


_module = importlib.import_module("math_symbol_galaxy")

MATH_GALAXY = _module.MATH_GALAXY

__all__ = ["MATH_GALAXY"]
