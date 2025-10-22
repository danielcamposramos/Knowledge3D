"""
Utility helpers for TRM (Tiny Recursive Model) workflows.

These helpers convert between the 128‑dimension RPN embedding space and the
512‑dimension TRM working space while keeping data in NumPy float32 format.
"""

from __future__ import annotations

import math
import numpy as np

# Canonical dimensions for the current TRM pipeline.
RPN_EMBED_DIM = 128
TRM_STATE_DIM = 512


def expand_embedding_to_trm(
    embedding: np.ndarray,
    target_dim: int = TRM_STATE_DIM,
) -> np.ndarray:
    """
    Project an embedding vector (typically 128‑dim RPN) to the TRM working space.

    The TRM kernels expect 512‑dim vectors. We expand the input by tiling and
    trimming so downstream code can assume a fixed shape. Inputs larger than the
    target dimension are truncated deterministically.
    """
    vector = np.asarray(embedding, dtype=np.float32).ravel()
    if vector.size == target_dim:
        return vector.astype(np.float32, copy=False)
    if vector.size > target_dim:
        return vector[:target_dim].astype(np.float32, copy=False)

    repeats = int(math.ceil(target_dim / max(1, vector.size)))
    expanded = np.tile(vector, repeats)[:target_dim]
    return expanded.astype(np.float32, copy=False)


def compress_trm_to_embedding(
    trm_vector: np.ndarray,
    source_dim: int = RPN_EMBED_DIM,
) -> np.ndarray:
    """
    Reduce a TRM vector back to the base embedding dimensionality.

    We slice the prefix matching the original embedding size and renormalise.
    """
    vector = np.asarray(trm_vector, dtype=np.float32).ravel()
    if vector.size < source_dim:
        raise ValueError(
            f"Cannot compress vector of length {vector.size} to {source_dim}"
        )
    reduced = vector[:source_dim]
    norm = float(np.linalg.norm(reduced))
    if norm == 0.0:
        return np.zeros(source_dim, dtype=np.float32)
    return (reduced / norm).astype(np.float32)


__all__ = [
    "expand_embedding_to_trm",
    "compress_trm_to_embedding",
    "RPN_EMBED_DIM",
    "TRM_STATE_DIM",
]
