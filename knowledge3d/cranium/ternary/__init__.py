"""
Ternary primitives (vectors, tensors, and galaxy storage).

These classes provide GPU-resident ternary representations {-1, 0, +1} with
packed 2-bit storage and content-addressed deduplication hooks.
"""

from .ternary_vector import TernaryVector, TernaryTensor
from .ternary_galaxy import TernaryGalaxy

__all__ = ["TernaryVector", "TernaryTensor", "TernaryGalaxy"]
