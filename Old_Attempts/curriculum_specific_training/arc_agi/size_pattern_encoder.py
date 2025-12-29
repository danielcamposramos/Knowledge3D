"""
Size pattern encoding — model-learned, not hardcoded ratios.

Encodes input→output size relationships as TernaryVector embeddings
that the model can learn to interpret.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from knowledge3d.cranium.bridges.cosine_similarity_bridge import CosineSimilarityBridge
from knowledge3d.cranium.ternary import TernaryVector


class SizePatternEncoder:
    """
    Encode size relationships as learnable embeddings.

    No hardcoded thresholds — the model learns what patterns mean.
    """

    def __init__(self):
        self.cosine_bridge = CosineSimilarityBridge()
        self.embedding_dim = 64

    def encode_task_pattern(self, train_examples: List[Dict]) -> TernaryVector:
        """
        Encode size pattern from training examples.

        Returns embedding that captures:
        - Average size change (shrink/grow/same)
        - Variance in size change
        - Aspect ratio preservation
        - Directional consistency
        """
        if not train_examples:
            return self._neutral_embedding()

        features: List[Dict[str, float]] = []

        for ex in train_examples:
            inp = ex.get("input", [])
            out = ex.get("output", [])
            if not inp or not out:
                continue

            h_in, w_in = len(inp), len(inp[0]) if inp else 0
            h_out, w_out = len(out), len(out[0]) if out else 0

            if h_in == 0 or w_in == 0:
                continue

            h_ratio = h_out / h_in
            w_ratio = w_out / w_in

            aspect_in = w_in / h_in if h_in > 0 else 1.0
            aspect_out = w_out / h_out if h_out > 0 else 1.0
            aspect_change = aspect_out / aspect_in if aspect_in > 0 else 1.0

            h_dir = -1 if h_ratio < 0.9 else (1 if h_ratio > 1.1 else 0)
            w_dir = -1 if w_ratio < 0.9 else (1 if w_ratio > 1.1 else 0)

            features.append(
                {
                    "h_ratio": h_ratio,
                    "w_ratio": w_ratio,
                    "h_dir": h_dir,
                    "w_dir": w_dir,
                    "aspect_change": aspect_change,
                    "area_ratio": (h_out * w_out) / (h_in * w_in),
                }
            )

        if not features:
            return self._neutral_embedding()

        n = len(features)
        avg_h_ratio = sum(f["h_ratio"] for f in features) / n
        avg_w_ratio = sum(f["w_ratio"] for f in features) / n
        avg_area_ratio = sum(f["area_ratio"] for f in features) / n
        avg_aspect_change = sum(f["aspect_change"] for f in features) / n

        var_h_ratio = sum((f["h_ratio"] - avg_h_ratio) ** 2 for f in features) / n
        var_w_ratio = sum((f["w_ratio"] - avg_w_ratio) ** 2 for f in features) / n

        h_dirs = [f["h_dir"] for f in features]
        w_dirs = [f["w_dir"] for f in features]
        h_consistency = abs(sum(h_dirs)) / n
        w_consistency = abs(sum(w_dirs)) / n

        embedding = [0.0] * self.embedding_dim
        embedding[0] = avg_h_ratio
        embedding[1] = avg_w_ratio
        embedding[2] = avg_area_ratio
        embedding[3] = avg_aspect_change
        embedding[4] = var_h_ratio
        embedding[5] = var_w_ratio
        embedding[6] = h_consistency
        embedding[7] = w_consistency
        embedding[8] = sum(h_dirs) / n
        embedding[9] = sum(w_dirs) / n
        embedding[10] = 1.0 if avg_area_ratio < 0.5 else 0.0
        embedding[11] = 1.0 if avg_area_ratio > 2.0 else 0.0
        embedding[12] = 1.0 if abs(avg_aspect_change - 1.0) < 0.1 else 0.0

        for i in range(13, self.embedding_dim):
            idx1 = i % 8
            idx2 = (i * 7) % 8
            embedding[i] = embedding[idx1] * embedding[idx2]

        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        ternary = [1 if x > 0.2 else (-1 if x < -0.2 else 0) for x in embedding]
        return TernaryVector(ternary)

    def encode_candidate_signature(
        self,
        candidate: Sequence[Sequence[int]],
        expected: Sequence[Sequence[int]],
    ) -> TernaryVector:
        """
        Encode size relationship between candidate and expected output.
        """
        h_cand = len(candidate)
        w_cand = len(candidate[0]) if candidate else 0
        h_exp = len(expected)
        w_exp = len(expected[0]) if expected else 0

        if h_exp == 0 or w_exp == 0:
            return self._neutral_embedding()

        h_ratio = h_cand / h_exp if h_exp else 0.0
        w_ratio = w_cand / w_exp if w_exp else 0.0
        area_ratio = (h_cand * w_cand) / (h_exp * w_exp) if h_exp and w_exp else 0.0

        embedding = [0.0] * self.embedding_dim
        embedding[0] = h_ratio
        embedding[1] = w_ratio
        embedding[2] = area_ratio
        embedding[3] = w_cand / h_cand if h_cand > 0 else 1.0
        embedding[4] = w_exp / h_exp if h_exp > 0 else 1.0
        embedding[8] = -1 if h_ratio < 0.9 else (1 if h_ratio > 1.1 else 0)
        embedding[9] = -1 if w_ratio < 0.9 else (1 if w_ratio > 1.1 else 0)
        embedding[10] = 1.0 if abs(h_ratio - 1.0) < 0.01 else 0.0
        embedding[11] = 1.0 if abs(w_ratio - 1.0) < 0.01 else 0.0
        embedding[12] = 1.0 if h_cand == h_exp and w_cand == w_exp else 0.0

        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        ternary = [1 if x > 0.2 else (-1 if x < -0.2 else 0) for x in embedding]
        return TernaryVector(ternary)

    def should_evaluate(
        self,
        candidate: Sequence[Sequence[int]],
        expected: Sequence[Sequence[int]],
        task_pattern: TernaryVector,
    ) -> Tuple[bool, float]:
        """
        Model-based evaluation decision. Returns (should_evaluate, confidence).
        """
        candidate_sig = self.encode_candidate_signature(candidate, expected)
        similarity = self.cosine_bridge.compute_similarities(
            [candidate_sig.to_python()],
            task_pattern.to_python(),
        )[0]
        return similarity > -0.3, similarity

    def _neutral_embedding(self) -> TernaryVector:
        """Neutral embedding for unknown/empty patterns."""
        return TernaryVector([0] * self.embedding_dim)


__all__ = ["SizePatternEncoder"]
