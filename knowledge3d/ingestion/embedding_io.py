"""Ingestion-side embedding I/O.

Bridges the Tablet live-loop commands that need to read .npy embedding
files or load a torch ThinkingTagEmbedder checkpoint. Lives under
`knowledge3d/ingestion/` — NOT on the sovereign hot path — so numpy and
torch are allowed here. Hot-path callers import functions from this
module; the imports stay inside the functions so `from
knowledge3d.ingestion.embedding_io import load_npy_row` does not drag
numpy/torch into module-import time.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence


def load_npy_rows(path: str | Path) -> List[List[float]]:
    """Load every row of a .npy matrix as a list-of-lists of floats.

    The returned structure is pure Python (no numpy). Callers on the
    sovereign hot path can consume these rows without importing numpy.
    """
    import numpy as _np  # ingestion-side only

    arr = _np.load(str(path))
    if arr.ndim == 1:
        return [[float(x) for x in arr.tolist()]]
    if arr.ndim == 2:
        return [[float(x) for x in row.tolist()] for row in arr]
    return [[float(x) for x in arr.reshape(-1).tolist()]]


def load_npy_row(path: str | Path, row_index: int) -> Optional[List[float]]:
    """Load a single row from a .npy matrix as a list of floats.

    Returns None if the row_index is out of range. Uses numpy's memory
    mapping to avoid pulling the full matrix into RAM for large stacks.
    """
    import numpy as _np  # ingestion-side only

    mat = _np.load(str(path), mmap_mode="r")
    if mat.ndim == 1:
        if row_index != 0:
            return None
        return [float(x) for x in mat.tolist()]
    if mat.ndim == 2:
        if row_index < 0 or row_index >= mat.shape[0]:
            return None
        return [float(x) for x in mat[row_index].reshape(-1).tolist()]
    flat = mat.reshape(-1)
    if row_index != 0:
        return None
    return [float(x) for x in flat.tolist()]


def load_first_npy(path: str | Path) -> List[float]:
    """Load the first row of a .npy (or the vector itself, if 1D).

    Used by Tablet handlers that accept `file:/path/to/foo.npy` args. The
    returned list is sovereign-safe (pure Python floats).
    """
    rows = load_npy_rows(path)
    return rows[0] if rows else []


def predict_thinking_tags_from_pth(
    checkpoint_path: str | Path,
    embedding: Sequence[float],
    tag_names: Sequence[str],
) -> List[str]:
    """Load a ThinkingTagEmbedder checkpoint and predict tags for an embedding.

    Isolated here so the /think command path does not import torch.
    Returns [] if the checkpoint file is missing or torch is unavailable.
    """
    p = Path(str(checkpoint_path))
    if not p.exists():
        return []
    try:
        import torch as _t  # ingestion-side only
        from knowledge3d.cranium.ptx_runtime import ThinkingTagEmbedder  # type: ignore
    except Exception:
        return []
    try:
        model = ThinkingTagEmbedder()
        model.load_state_dict(_t.load(str(p), map_location="cpu"))
        model.eval()
        return list(model.predict_thinking_tags(list(embedding), list(tag_names)))
    except Exception:
        return []
