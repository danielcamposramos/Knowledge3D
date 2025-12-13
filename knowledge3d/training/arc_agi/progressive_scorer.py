"""
Progressive scoring — discoveries improve over iterations.

No fixed thresholds — model learns what "good enough to keep" means.
Near-misses are preserved and refined, not discarded.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from knowledge3d.cranium.bridges.cosine_similarity_bridge import CosineSimilarityBridge
from knowledge3d.cranium.ternary import TernaryGalaxy, TernaryVector


class ProgressiveScorer:
    """
    Scores discoveries with preservation-first philosophy.
    """

    def __init__(self):
        self.cosine_bridge = CosineSimilarityBridge()
        self.galaxy = TernaryGalaxy()

        self._preserve_threshold = 0.85
        self._promote_threshold = 0.95
        self._canonical_threshold = 1.0
        self._threshold_lr = 0.01
        self._score_history: List[float] = []
        self._max_history = 1000

    def score_discovery(
        self,
        candidate_output: List[List[int]],
        expected_output: List[List[int]],
        context: str,
    ) -> Tuple[float, str]:
        """
        Score a discovery and determine its fate.
        """
        exact_score = self._compute_exact_score(candidate_output, expected_output)
        fuzzy_score = self._compute_fuzzy_score(candidate_output, expected_output)
        combined = 0.7 * exact_score + 0.3 * fuzzy_score
        self._record_score(combined)

        if combined >= self._canonical_threshold:
            fate = "canonical"
        elif combined >= self._promote_threshold:
            fate = "promote"
        elif combined >= self._preserve_threshold:
            fate = "preserve"
        else:
            fate = "discard"
        return combined, fate

    def _compute_exact_score(self, candidate: List[List[int]], expected: List[List[int]]) -> float:
        """Compute pixel-exact match ratio."""
        if not candidate or not expected:
            return 0.0

        h_cand, w_cand = len(candidate), len(candidate[0]) if candidate else 0
        h_exp, w_exp = len(expected), len(expected[0]) if expected else 0

        if h_cand != h_exp or w_cand != w_exp:
            size_match = min(h_cand, h_exp) * min(w_cand, w_exp)
            size_total = max(h_cand, h_exp) * max(w_cand, w_exp)
            return 0.5 * (size_match / size_total) if size_total > 0 else 0.0

        matches = 0
        total = h_exp * w_exp
        for y in range(h_exp):
            for x in range(w_exp):
                if candidate[y][x] == expected[y][x]:
                    matches += 1
        return matches / total if total > 0 else 0.0

    def _compute_fuzzy_score(self, candidate: List[List[int]], expected: List[List[int]]) -> float:
        """Compute fuzzy/semantic similarity."""
        if not candidate or not expected:
            return 0.0

        cand_features = self._extract_features(candidate)
        exp_features = self._extract_features(expected)
        similarity = self.cosine_bridge.compute_similarities([cand_features], exp_features)[0]
        return (similarity + 1.0) / 2.0

    def _extract_features(self, grid: List[List[int]]) -> List[float]:
        """Extract feature vector from grid."""
        if not grid:
            return [0.0] * 64

        h, w = len(grid), len(grid[0]) if grid else 0
        features = [0.0] * 64

        features[0] = h / 30.0
        features[1] = w / 30.0
        features[2] = (h * w) / 900.0
        features[3] = w / h if h > 0 else 1.0

        color_counts = [0] * 10
        for row in grid:
            for pixel in row:
                if 0 <= pixel < 10:
                    color_counts[pixel] += 1

        total_pixels = h * w
        for i, count in enumerate(color_counts):
            features[4 + i] = count / total_pixels if total_pixels > 0 else 0.0

        mid_h, mid_w = h // 2, w // 2
        quadrants = [0, 0, 0, 0]
        for y, row in enumerate(grid):
            for x, pixel in enumerate(row):
                if pixel != 0:
                    q = (1 if y >= mid_h else 0) + (2 if x >= mid_w else 0)
                    quadrants[q] += 1

        for i, q in enumerate(quadrants):
            features[14 + i] = q / total_pixels if total_pixels > 0 else 0.0

        edges = 0
        for y in range(h):
            for x in range(w):
                if x > 0 and grid[y][x] != grid[y][x - 1]:
                    edges += 1
                if y > 0 and grid[y][x] != grid[y - 1][x]:
                    edges += 1
        features[18] = edges / (2 * total_pixels) if total_pixels > 0 else 0.0

        norm = sum(x * x for x in features) ** 0.5
        if norm > 0:
            features = [x / norm for x in features]

        return features

    def _record_score(self, score: float) -> None:
        """Record score for adaptive threshold learning."""
        self._score_history.append(score)
        if len(self._score_history) > self._max_history:
            self._score_history.pop(0)

    def adapt_thresholds(self) -> None:
        """
        Adapt thresholds based on score distribution.
        """
        if len(self._score_history) < 100:
            return

        sorted_scores = sorted(self._score_history)
        n = len(sorted_scores)
        target_preserve_idx = int(n * 0.40)
        target_preserve = sorted_scores[target_preserve_idx]

        self._preserve_threshold += self._threshold_lr * (target_preserve - self._preserve_threshold)
        self._preserve_threshold = max(0.70, min(0.90, self._preserve_threshold))
        self._promote_threshold = min(0.99, self._preserve_threshold + 0.10)

    def get_thresholds(self) -> Dict[str, float]:
        """Get current thresholds."""
        return {
            "preserve": self._preserve_threshold,
            "promote": self._promote_threshold,
            "canonical": self._canonical_threshold,
        }


class DiscoveryPreserver:
    """
    Preserves near-miss discoveries for progressive refinement.
    """

    def __init__(self):
        self.scorer = ProgressiveScorer()
        self.galaxy = TernaryGalaxy()
        self._preserved: Dict[str, Dict] = {}

    def evaluate_and_preserve(
        self,
        discovery_id: str,
        rpn_program: str,
        candidate_output: List[List[int]],
        expected_output: List[List[int]],
        context: str,
    ) -> Tuple[float, str, bool]:
        """
        Evaluate discovery and preserve if worthy.
        """
        score, fate = self.scorer.score_discovery(candidate_output, expected_output, context)

        was_preserved = False
        if fate != "discard":
            self._preserved[discovery_id] = {
                "rpn_program": rpn_program,
                "score": score,
                "fate": fate,
                "context": context,
                "attempts": 1,
                "best_score": score,
                "improvement_history": [score],
            }
            embedding = self._program_to_embedding(rpn_program)
            self.galaxy.store_frame(f"discovery_{discovery_id}", f"{fate}:{score:.3f}", embedding)
            was_preserved = True

            if fate == "preserve":
                print(f"[PRESERVE] {discovery_id}: {score:.2%} — near-miss, kept for refinement")
            elif fate == "promote":
                print(f"[PROMOTE] {discovery_id}: {score:.2%} — high confidence")
            elif fate == "canonical":
                print(f"[CANONICAL] {discovery_id}: {score:.2%} — perfect match!")

        return score, fate, was_preserved

    def attempt_refinement(
        self,
        discovery_id: str,
        new_candidate: List[List[int]],
        expected: List[List[int]],
    ) -> Tuple[float, bool]:
        """
        Attempt to improve a preserved discovery.
        """
        if discovery_id not in self._preserved:
            return 0.0, False

        record = self._preserved[discovery_id]
        new_score, _ = self.scorer.score_discovery(new_candidate, expected, record["context"])

        record["attempts"] += 1
        record["improvement_history"].append(new_score)

        improved = new_score > record["best_score"]
        if improved:
            old_best = record["best_score"]
            record["best_score"] = new_score
            record["score"] = new_score

            if new_score >= 1.0:
                record["fate"] = "canonical"
            elif new_score >= 0.95:
                record["fate"] = "promote"

            print(f"[IMPROVED] {discovery_id}: {old_best:.2%} → {new_score:.2%}")

        return new_score, improved

    def get_refinement_candidates(self, k: int = 10) -> List[Tuple[str, Dict]]:
        """
        Get top-k discoveries most likely to benefit from refinement.
        """
        candidates = []
        for disc_id, record in self._preserved.items():
            if record["fate"] == "canonical":
                continue

            if record["fate"] == "preserve":
                gap = 0.95 - record["score"]
                priority = 1.0 - gap
            else:
                gap = 1.0 - record["score"]
                priority = 1.0 - gap

            if record["attempts"] < 5:
                priority *= 1.5

            history = record["improvement_history"]
            if len(history) >= 2 and history[-1] > history[-2]:
                priority *= 1.2

            candidates.append((disc_id, record, priority))

        candidates.sort(key=lambda x: -x[2])
        return [(disc_id, record) for disc_id, record, _ in candidates[:k]]

    def _program_to_embedding(self, rpn_program: str) -> TernaryVector:
        """Convert RPN program to embedding."""
        tokens = rpn_program.split()
        embedding = [0.0] * 128
        for i, token in enumerate(tokens):
            idx = hash(token) % 128
            embedding[idx] += 1.0 / (i + 1)

        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        ternary = [1 if x > 0.1 else (-1 if x < -0.1 else 0) for x in embedding]
        return TernaryVector(ternary)

    def get_preservation_stats(self) -> Dict:
        """Get statistics about preserved discoveries."""
        stats = {"total": len(self._preserved), "by_fate": {"preserve": 0, "promote": 0, "canonical": 0}, "avg_score": 0.0, "avg_attempts": 0.0}

        if not self._preserved:
            return stats

        total_score = 0.0
        total_attempts = 0
        for record in self._preserved.values():
            stats["by_fate"][record["fate"]] += 1
            total_score += record["score"]
            total_attempts += record["attempts"]

        n = len(self._preserved)
        stats["avg_score"] = total_score / n
        stats["avg_attempts"] = total_attempts / n
        return stats


__all__ = ["ProgressiveScorer", "DiscoveryPreserver"]
