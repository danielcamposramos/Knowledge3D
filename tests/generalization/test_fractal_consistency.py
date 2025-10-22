from __future__ import annotations

import itertools

import numpy as np


class TestFractalConsistency:
    DOMAINS = {
        "machine_learning": [
            "neural network",
            "gradient descent",
            "backpropagation",
            "overfitting",
        ],
        "physics": [
            "velocity",
            "acceleration",
            "momentum",
            "energy",
        ],
        "biology": [
            "cell",
            "mitosis",
            "DNA",
            "protein",
        ],
    }

    def test_cluster_separation(self, rpn_engine):
        """Post-consolidation clusters should separate semantically."""
        centers = {}
        for domain, terms in self.DOMAINS.items():
            embeddings = [rpn_engine.embed_word(term) for term in terms]
            centers[domain] = np.mean(embeddings, axis=0)

        inter_distances = []
        for d1, d2 in itertools.combinations(centers.keys(), 2):
            inter_distances.append(np.linalg.norm(centers[d1] - centers[d2]))

        intra_distances = []
        for domain, terms in self.DOMAINS.items():
            center = centers[domain]
            for term in terms:
                emb = rpn_engine.embed_word(term)
                intra_distances.append(np.linalg.norm(emb - center))

        avg_inter = float(np.mean(inter_distances))
        avg_intra = float(np.mean(intra_distances)) or 1.0
        separation_ratio = avg_inter / avg_intra

        assert separation_ratio > 0.8, (
            f"Poor clustering separation ratio: {separation_ratio:.2f}"
        )
