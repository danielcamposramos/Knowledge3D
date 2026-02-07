"""ARC-AGI benchmark integration for Knowledgeverse Week 14."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.arc_agi_2_adapter import ArcAgi2Adapter
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.trm_navigator import TRMNavigator


class ARCAGI2Benchmark:
    """Run ARC-style visual reasoning tasks with empty/enriched modes."""

    def __init__(
        self,
        knowledgeverse: Knowledgeverse | None = None,
        dataset_path: str | Path | None = None,
        max_tasks: int | None = None,
        dataset_version: str = "arc_agi_2",
        strict_legacy: bool = False,
    ):
        self.kv = knowledgeverse or Knowledgeverse()
        self.dataset_version = dataset_version
        self.strict_legacy = strict_legacy
        self.dataset_path = self._resolve_dataset_path(dataset_path, dataset_version)
        self.max_tasks = max_tasks
        self.tasks = self._load_tasks()
        self.results: list[dict[str, Any]] = []
        self.adapter: ArcAgi2Adapter | None = None

    def _resolve_dataset_path(self, dataset_path: str | Path | None, dataset_version: str) -> Path:
        if dataset_path is not None:
            return Path(dataset_path)

        version = dataset_version.lower().strip()
        if version in {"arc_agi_3", "arc3", "3"}:
            candidates = [
                Path("/K3D/Knowledge3D.local/datasets/arc_agi_3/evaluation"),
                Path("/K3D/Knowledge3D.local/datasets/exams/arc-agi-3/evaluation"),
                Path("../Knowledge3D.local/datasets/arc_agi_3/evaluation"),
                Path("../Knowledge3D.local/datasets/exams/arc-agi-3/evaluation"),
            ]
        else:
            candidates = [
                Path("/K3D/Knowledge3D.local/datasets/exams/arc-src/data/evaluation"),
                Path("/K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation"),
                Path("/K3D/Knowledge3D.local/datasets/arc_agi_2/evaluation"),
                Path("../Knowledge3D.local/datasets/exams/arc-src/data/evaluation"),
                Path("../Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation"),
                Path("../Knowledge3D.local/datasets/arc_agi_2/evaluation"),
            ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return Path("")

    def _load_tasks(self) -> list[dict[str, Any]]:
        if self.dataset_path and self.dataset_path.exists():
            files = sorted(self.dataset_path.glob("*.json"))
            if self.max_tasks is not None:
                files = files[: max(0, int(self.max_tasks))]
            tasks: list[dict[str, Any]] = []
            for file_path in files:
                try:
                    payload = json.loads(file_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                train = payload.get("train")
                test = payload.get("test")
                if not isinstance(train, list) or not isinstance(test, list) or not test:
                    continue
                tasks.append(
                    {
                        "id": file_path.stem,
                        "train": train,
                        "test": test,
                    }
                )
            if tasks:
                return tasks

        # Fallback synthetic set keeps benchmark infrastructure runnable.
        return [
            {
                "id": "synthetic_flip_h",
                "train": [
                    {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
                    {"input": [[5, 6], [7, 8]], "output": [[6, 5], [8, 7]]},
                ],
                "test": [{"input": [[9, 0], [1, 2]], "output": [[0, 9], [2, 1]]}],
            },
            {
                "id": "synthetic_color_map",
                "train": [
                    {"input": [[1, 1], [2, 2]], "output": [[3, 3], [4, 4]]},
                    {"input": [[2, 1], [1, 2]], "output": [[4, 3], [3, 4]]},
                ],
                "test": [{"input": [[1, 2], [2, 1]], "output": [[3, 4], [4, 3]]}],
            },
        ]

    def run_benchmark(self, use_enriched: bool = True) -> dict[str, Any]:
        self.results = []
        correct = 0
        self.adapter = ArcAgi2Adapter(
            use_enriched=use_enriched,
            strict_legacy=self.strict_legacy,
            knowledgeverse=self.kv,
        )
        for task in self.tasks:
            result = self._solve_task(task=task, use_enriched=use_enriched)
            self.results.append(result)
            if result["correct"]:
                correct += 1
        total = len(self.tasks)
        accuracy = (correct / total) if total else 0.0
        return {
            "benchmark": "ARC-AGI 2/3",
            "dataset_path": str(self.dataset_path) if self.dataset_path else "synthetic",
            "dataset_version": self.dataset_version,
            "use_enriched": use_enriched,
            "total_tasks": total,
            "correct": correct,
            "accuracy": accuracy,
            "results": self.results,
        }

    def _solve_task(
        self,
        *,
        task: dict[str, Any],
        use_enriched: bool,
    ) -> dict[str, Any]:
        if use_enriched:
            self._seed_visual_knowledge(task)
        assert self.adapter is not None
        result = self.adapter.solve_task(task, fallback_solver=self._solve_task_fallback)
        correct = bool(result["correct"])
        route = result.get("route", {})
        route_galaxies = route.get("galaxy_names") or ["Drawing"]
        event_type = "arc_task_success" if correct else "arc_task_failure"
        self.kv.log_event(
            event_type,
            {
                "specialist": route.get("specialist", "visual"),
                "task_id": task["id"],
                "confidence": 0.9 if correct else 0.4,
                "galaxy": route_galaxies[0],
                "verification": "legacy_pipeline",
            },
        )
        return result

    def _solve_task_fallback(self, task: dict[str, Any], use_enriched: bool) -> dict[str, Any]:
        """Fallback solver keeps benchmark operable when legacy path cannot execute."""
        navigator = TRMNavigator(knowledgeverse=self.kv)
        route = navigator.route(
            query="visual pattern transformation",
            specialist="auto",
            domain_hint="visual",
        )
        patterns = navigator.query(
            query="visual pattern transformation",
            galaxy_names=route["galaxy_names"],
            top_k=20 if use_enriched else 5,
            specialist=route["specialist"],
            domain_hint=route["domain"],
        )
        composed = navigator.compose(
            task_examples=task["train"],
            patterns=patterns,
            specialist=route["specialist"],
            use_enriched=use_enriched,
        )
        test_sample = task["test"][0]
        predicted = navigator.execute(composed, test_sample["input"])
        expected = test_sample.get("output")
        return {
            "task_id": task["id"],
            "correct": self._grids_match(predicted, expected),
            "exact_match": self._grids_match(predicted, expected),
            "predicted": predicted,
            "expected": expected,
            "transform": composed.get("transform"),
            "patterns_used": len(patterns),
            "reasoning_trace": navigator.get_reasoning_trace(),
            "route": route,
            "score": 1.0 if self._grids_match(predicted, expected) else 0.0,
            "fuzzy_score": 1.0 if self._grids_match(predicted, expected) else 0.0,
        }

    def _seed_visual_knowledge(self, task: dict[str, Any]) -> None:
        self.kv.galaxy_manager.add_entry(
            "Drawing",
            {
                "domain": "visual",
                "task_id": task["id"],
                "train_examples": len(task.get("train", [])),
                "kind": "arc_pattern",
            },
        )
        self.kv.galaxy_manager.add_entry(
            "Grammar",
            {
                "domain": "visual",
                "task_id": task["id"],
                "kind": "transform_rule",
            },
        )

    def _grids_match(self, predicted: list[list[int]], expected: list[list[int]]) -> bool:
        if len(predicted) != len(expected):
            return False
        for pred_row, exp_row in zip(predicted, expected):
            if list(pred_row) != list(exp_row):
                return False
        return True

    def save_results(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": "ARC-AGI 2",
            "total_tasks": len(self.results),
            "correct": sum(1 for row in self.results if row.get("correct")),
            "accuracy": (
                sum(1 for row in self.results if row.get("correct")) / len(self.results)
                if self.results
                else 0.0
            ),
            "results": self.results,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
