from __future__ import annotations

"""
GPU helper utilities for Knowledge3D.

This package hosts light-weight wrappers shared across PTX-aware modules.
The helpers intentionally avoid heavy dependencies so they can be imported
from both training and inference contexts.
"""

from .rng_pool import RNGPool, global_rng_pool  # noqa: F401
