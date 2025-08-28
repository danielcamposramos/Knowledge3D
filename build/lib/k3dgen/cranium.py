from __future__ import annotations

from typing import List

import numpy as np
from sklearn.decomposition import PCA

from k3dgen.house import House


class Cranium:
    """The AI's internal cognitive space, responsible for processing knowledge."""

    def __init__(self, house: House):
        """Initialize the Cranium with a reference to its House.

        Parameters
        ----------
        house:
            The House instance containing the knowledge to be processed.
        """
        self.house = house

    def project_to_3d(self) -> List[List[float]]:
        """Project the high-dimensional embeddings from the House into 3D space.

        Uses PCA for dimensionality reduction.

        Returns
        -------
        A list of 3D coordinates, corresponding to the records in the House.
        """
        records = self.house.get_all_records()
        if not records:
            return []

        embeddings = np.array([r["embedding"] for r in records])

        # Ensure we have enough data for 3D PCA
        if embeddings.shape[1] < 3:
            # If source dimension is less than 3, pad with zeros
            padded_embeddings = np.zeros((embeddings.shape[0], 3))
            padded_embeddings[:, :embeddings.shape[1]] = embeddings
            return padded_embeddings.tolist()

        n_samples = embeddings.shape[0]
        n_components = min(n_samples, 3)

        # PCA can't run with fewer samples than components
        if n_components < 3:
            # Not enough data to create a 3D projection, return as is with padding
            padded_embeddings = np.zeros((embeddings.shape[0], 3))
            padded_embeddings[:, :n_components] = embeddings[:, :n_components]
            return padded_embeddings.tolist()

        pca = PCA(n_components=3)
        projected_vectors = pca.fit_transform(embeddings)
        return projected_vectors.tolist()

    def update_house_vectors(self) -> None:
        """Project embeddings to 3D and update the vectors in the House."""
        projected_vectors = self.project_to_3d()
        records = self.house.get_all_records()

        for record, vector_3d in zip(records, projected_vectors):
            record["vector"] = vector_3d

        # This is a bit inefficient as it rewrites the whole file.
        # For this stage of the project, it's an acceptable trade-off.
        self.house._data = records
        self.house._save_data()
