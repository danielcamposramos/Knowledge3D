"""Simple storage for embeddings in .k3d format."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.neighbors import NearestNeighbors

# Default location for the storage file. Tests and callers may overwrite this
# module-level variable to point elsewhere.
DATA_PATH: Path = Path(__file__).with_suffix(".k3d")


def _load_data() -> List[Dict]:
    """Load records from the .k3d file."""
    if DATA_PATH.exists():
        return json.loads(DATA_PATH.read_text())
    return []


def _save_data(data: List[Dict]) -> None:
    """Write records to the .k3d file."""
    DATA_PATH.write_text(json.dumps(data, indent=2))


def store_embedding(id: str, vector: List[float], metadata: Dict) -> None:
    """Store an embedding with associated metadata.

    Parameters
    ----------
    id:
        Identifier for the vector.
    vector:
        High-dimensional embedding vector. Must have at least three values.
    metadata:
        Additional metadata to store with the vector.
    """
    if len(vector) < 3:
        raise ValueError("vector must contain at least three dimensions")

    data = _load_data()
    # Remove existing record if the id already exists to avoid duplicates.
    data = [r for r in data if r.get("id") != id]
    record = {
        "id": str(id),
        "vector": vector[:3],  # 3D projection placeholder
        "embedding": vector,
        "metadata": metadata,
        "neighbors": [],
    }
    data.append(record)
    _save_data(data)


def retrieve_neighbors(id: str, k: int) -> List[str]:
    """Retrieve the identifiers of the k nearest neighbors of `id`.

    Parameters
    ----------
    id:
        Identifier of the query vector.
    k:
        Number of nearest neighbors to return.
    """
    data = _load_data()
    ids = [r["id"] for r in data]
    if id not in ids:
        raise KeyError(f"id '{id}' not found")
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if len(ids) <= k:
        raise ValueError("not enough vectors to find neighbors")

    embeddings = np.array([r["embedding"] for r in data], dtype=float)
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(embeddings)

    index = ids.index(id)
    neighbors_idx = nn.kneighbors([embeddings[index]], return_distance=False)[0][1:]
    return [ids[i] for i in neighbors_idx]
