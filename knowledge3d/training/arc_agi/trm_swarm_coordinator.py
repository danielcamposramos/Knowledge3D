"""
TRM Swarm Coordinator — router-as-model for dynamic worker spawning.

The router learns:
- Which specialists to invoke for a task
- How many workers to spawn
- How to aggregate results

Weights = logic (routing decisions)
Galaxy = knowledge (specialist embeddings, task patterns)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, Future
from typing import Any, Callable, Dict, List, Optional, Tuple

from knowledge3d.cranium.bridges.cosine_similarity_bridge import CosineSimilarityBridge
from knowledge3d.cranium.ternary import TernaryVector
from knowledge3d.training.arc_agi.specialist_registry import SpecialistRegistry
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy, get_grammar_galaxy


class TRMSwarmCoordinator:
    """
    Master coordinator that routes tasks to specialists dynamically.
    """

    def __init__(
        self,
        max_workers: int = 9,
        spawn_threshold: float = 0.3,
        min_workers: int = 1,
    ):
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.spawn_threshold = spawn_threshold

        self.specialist_registry = SpecialistRegistry()
        self.cosine_bridge = CosineSimilarityBridge()

        self._aggregation_weights: Dict[str, float] = {
            spec_id: 1.0 for spec_id in self.specialist_registry.list_specialists()
        }

    def embed_task(self, task: Dict) -> TernaryVector:
        """
        Compute task embedding from examples and hints.
        """
        embedding = [0.0] * 128
        train_examples = task.get("train", [])

        for ex in train_examples:
            inp = ex.get("input", [])
            out = ex.get("output", [])
            if not inp or not out:
                continue

            h_ratio = len(out) / max(1, len(inp))
            w_ratio = (len(out[0]) if out else 1) / max(1, len(inp[0]) if inp else 1)

            embedding[0] += h_ratio
            embedding[1] += w_ratio

            inp_colors = set()
            out_colors = set()
            for row in inp:
                inp_colors.update(row)
            for row in out:
                out_colors.update(row)

            embedding[2] += len(inp_colors)
            embedding[3] += len(out_colors)
            embedding[4] += len(out_colors - inp_colors)
            embedding[5] += len(inp_colors - out_colors)

        n = max(1, len(train_examples))
        embedding = [x / n for x in embedding]

        hints = task.get("semantic_hints", [])
        for hint in hints:
            for i, char in enumerate(str(hint)):
                idx = (ord(char) + i * 13) % 128
                embedding[idx] += 0.1

        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        ternary = [1 if x > 0.1 else (-1 if x < -0.1 else 0) for x in embedding]
        return TernaryVector(ternary)

    def route_task(self, task_embedding: TernaryVector) -> List[Tuple[str, float]]:
        """
        Route task to specialists based on learned similarity.
        """
        scores: List[Tuple[str, float]] = []
        task_emb = task_embedding.to_python()

        for spec_id in self.specialist_registry.list_specialists():
            spec_emb = self.specialist_registry.get_specialist_embedding(spec_id)
            if spec_emb is None:
                continue

            sim = self.cosine_bridge.compute_similarities([task_emb], spec_emb.to_python())[0]
            weight = self._aggregation_weights.get(spec_id, 1.0)
            scores.append((spec_id, sim * weight))

        scores.sort(key=lambda x: -x[1])
        selected = [(s, c) for s, c in scores if c >= self.spawn_threshold]

        if len(selected) < self.min_workers and scores:
            selected = scores[: self.min_workers]

        return selected[: self.max_workers]

    def get_optimal_parallelism(self, task_embedding: TernaryVector, work_items: int) -> int:
        """
        Determine optimal worker count based on routing and workload.
        """
        routing = self.route_task(task_embedding)
        n_specialists = len(routing)
        optimal = min(n_specialists, work_items, self.max_workers)
        return max(self.min_workers, optimal)

    def partition_work_dynamic(self, items: List[Any], task_embedding: TernaryVector) -> List[List[Any]]:
        """
        Partition work based on task complexity using round-robin.
        """
        n_workers = self.get_optimal_parallelism(task_embedding, len(items))

        if n_workers <= 0 or not items:
            return [items] if items else []

        return [items[i::n_workers] for i in range(n_workers)]

    def spawn_workers(
        self,
        task: Dict,
        worker_fn: Callable,
        grammar_snapshot: bytes,
    ) -> List[Tuple[Optional[str], Future]]:
        """
        Spawn workers dynamically based on routing decision.
        """
        task_embedding = task.get("task_embedding") or self.embed_task(task)
        routing = self.route_task(task_embedding)

        n_workers = len(routing)
        if n_workers == 0:
            n_workers = self.min_workers
            routing = [(None, 0.0)] * n_workers

        futures: List[Tuple[Optional[str], Future]] = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for i, (spec_id, confidence) in enumerate(routing):
                future = executor.submit(
                    worker_fn,
                    task,
                    spec_id,
                    confidence,
                    grammar_snapshot,
                    i,
                    n_workers,
                )
                futures.append((spec_id, future))

        return futures

    def aggregate_results(
        self,
        results: List[Tuple[str, Any, float]],
        task_embedding: TernaryVector,
    ) -> Tuple[Any, float]:
        """
        Aggregate specialist outputs via learned weighting.
        """
        if not results:
            return None, 0.0

        weighted_results = []
        for spec_id, output, confidence in results:
            weight = self._aggregation_weights.get(spec_id, 1.0)
            weighted_results.append((output, confidence * weight, spec_id))

        weighted_results.sort(key=lambda x: -x[1])
        best_output, best_score, best_spec = weighted_results[0]
        return best_output, best_score

    def update_from_feedback(self, task_embedding: TernaryVector, specialist_results: List[Tuple[str, bool]]) -> None:
        """
        Update routing weights based on task outcome.
        """
        for spec_id, success in specialist_results:
            if spec_id is None:
                continue

            self.specialist_registry.update_specialist(spec_id, task_embedding, success)

            current = self._aggregation_weights.get(spec_id, 1.0)
            delta = 0.1 if success else -0.05
            self._aggregation_weights[spec_id] = max(0.1, min(2.0, current + delta))


__all__ = ["TRMSwarmCoordinator"]
