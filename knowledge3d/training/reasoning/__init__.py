"""
Reasoning-focused training utilities (ARC-AGI, logic puzzles, etc.).

These helpers live under `knowledge3d.training.reasoning` so we can orchestrate
offline training runs without wiring them into the runtime PTX path.
"""

from .arc_dataset import (
    ARC_DATASET_URL,
    ARCReasoningCache,
    ensure_arc_dataset,
    prepare_arc_reasoning_cache,
    load_arc_reasoning_cache,
)

__all__ = [
    "ARC_DATASET_URL",
    "ARCReasoningCache",
    "ensure_arc_dataset",
    "prepare_arc_reasoning_cache",
    "load_arc_reasoning_cache",
]
