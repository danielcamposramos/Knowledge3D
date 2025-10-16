"""
Shared configuration knobs for the sovereign RPN executors.

These defaults favour wide parallel execution (single block of 256 threads)
while keeping behaviour consistent across Tier‑1, Tier‑2, and Tier‑3 engines.
"""
from __future__ import annotations

# Universal defaults
RPN_BLOCK_DIM = 256
RPN_GRID_DIM = 1
USE_SHARED_MEMORY_STACK = True
USE_COOPERATIVE_OPS = True

# Per-tier overrides (currently identical but kept for clarity/extensibility)
TIER1_BLOCK_DIM = 256
TIER2_BLOCK_DIM = 256
TIER3_BLOCK_DIM = 256

__all__ = [
    "RPN_BLOCK_DIM",
    "RPN_GRID_DIM",
    "USE_SHARED_MEMORY_STACK",
    "USE_COOPERATIVE_OPS",
    "TIER1_BLOCK_DIM",
    "TIER2_BLOCK_DIM",
    "TIER3_BLOCK_DIM",
]
