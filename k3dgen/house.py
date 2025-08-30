from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.neighbors import NearestNeighbors


class House:
    """A K3D House, representing a collection of embeddings stored at a given URI."""

    def __init__(self, uri: str):
        """Initialize a House instance.

        Parameters
        ----------
        uri:
            The URI of the .k3d file (e.g., 'file:///path/to/data.k3d').
            For now, only file URIs are supported.
        """
        if not uri.startswith("file://"):
            raise ValueError("Only file:// URIs are currently supported.")
        self.uri = uri
        self.path = Path(uri.removeprefix("file://"))
        self._data: List[Dict[str, Any]] | None = None

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load records from the .k3d file."""
        if self._data is not None:
            return self._data
        if self.path.exists():
            self._data = json.loads(self.path.read_text())
        else:
            self._data = []
        return self._data

    def _save_data(self) -> None:
        """Write records to the .k3d file."""
        if self._data is None:
            return
        self.path.write_text(json.dumps(self._data, indent=2))

    def store_embedding(self, id: str, vector: List[float], metadata: Dict) -> None:
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

        data = self._load_data()
        # Remove existing record if the id already exists to avoid duplicates.
        data = [r for r in data if r.get("id") != id]
        record = {
            "id": str(id),
            "vector": vector[:3],  # 3D projection (first 3 dims)
            "embedding": vector,
            "metadata": metadata,
            "neighbors": [],
        }
        data.append(record)
        self._data = data
        self._save_data()

    def retrieve_neighbors(self, id: str, k: int) -> List[str]:
        """Retrieve the identifiers of the k nearest neighbors of `id`.

        Parameters
        ----------
        id:
            Identifier of the query vector.
        k:
            Number of nearest neighbors to return.
        """
        data = self._load_data()
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

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Return all records from the house."""
        return self._load_data()
