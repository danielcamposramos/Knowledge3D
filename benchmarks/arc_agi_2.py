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
        enable_contrastive_learning: bool = False,
        enable_validity_gates: bool = False,
        enable_fuzzy_oracle: bool = False,
        fuzzy_oracle_threshold: float = 0.95,
        enable_ptx_ranking: bool = False,
        runtime_seed_knowledge: bool = False,
    ):
        self.kv = knowledgeverse or Knowledgeverse()
        self.dataset_version = dataset_version
        self.strict_legacy = strict_legacy
        self.enable_contrastive_learning = bool(enable_contrastive_learning)
        self.enable_validity_gates = bool(enable_validity_gates)
        self.enable_fuzzy_oracle = bool(enable_fuzzy_oracle)
        self.fuzzy_oracle_threshold = float(fuzzy_oracle_threshold)
        self.enable_ptx_ranking = bool(enable_ptx_ranking)
        self.runtime_seed_knowledge = bool(runtime_seed_knowledge)
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
        generated_patterns_total = 0
        tasks_with_generated_patterns = 0
        self.adapter = ArcAgi2Adapter(
            use_enriched=use_enriched,
            strict_legacy=self.strict_legacy,
            knowledgeverse=self.kv,
            enable_contrastive_learning=self.enable_contrastive_learning,
            enable_validity_gates=self.enable_validity_gates,
            enable_fuzzy_oracle=self.enable_fuzzy_oracle,
            fuzzy_oracle_threshold=self.fuzzy_oracle_threshold,
            enable_ptx_ranking=self.enable_ptx_ranking,
        )
        for task in self.tasks:
            result = self._solve_task(task=task, use_enriched=use_enriched)
            self.results.append(result)
            if result["correct"]:
                correct += 1
            generated_count = int(result.get("generated_pattern_count", len(result.get("generated_patterns", []))))
            generated_patterns_total += generated_count
            if generated_count > 0:
                tasks_with_generated_patterns += 1
        total = len(self.tasks)
        accuracy = (correct / total) if total else 0.0
        source_accuracy = self._compute_pattern_source_accuracy(self.results)
        oracle_diagnostics = self._compute_oracle_diagnostics(self.results)
        return {
            "benchmark": "ARC-AGI 2/3",
            "dataset_path": str(self.dataset_path) if self.dataset_path else "synthetic",
            "dataset_version": self.dataset_version,
            "use_enriched": use_enriched,
            "total_tasks": total,
            "correct": correct,
            "accuracy": accuracy,
            "generated_pattern_total": generated_patterns_total,
            "tasks_with_generated_patterns": tasks_with_generated_patterns,
            "pattern_source_accuracy": source_accuracy,
            "oracle_diagnostics": oracle_diagnostics,
            "results": self.results,
        }

    def _solve_task(
        self,
        *,
        task: dict[str, Any],
        use_enriched: bool,
    ) -> dict[str, Any]:
        if use_enriched and self.runtime_seed_knowledge:
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

    def _compute_pattern_source_accuracy(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        stats: dict[str, dict[str, float]] = {}
        for row in rows:
            source = str(row.get("pattern_source", "unknown"))
            bucket = stats.setdefault(source, {"correct": 0.0, "total": 0.0, "accuracy": 0.0})
            bucket["total"] += 1.0
            if bool(row.get("correct", False)):
                bucket["correct"] += 1.0
        for bucket in stats.values():
            total = bucket["total"]
            bucket["accuracy"] = (bucket["correct"] / total) if total else 0.0
        return stats

    def _compute_oracle_diagnostics(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(rows)
        if total == 0:
            return {
                "top_1_accuracy": 0.0,
                "oracle_at_3": 0.0,
                "oracle_at_10": 0.0,
                "oracle_at_all": 0.0,
                "avg_correct_rank": None,
                "generation_failure_rate": 0.0,
                "ranking_failure_rate": 0.0,
                "ranking_change_rate": 0.0,
            }

        top_1 = sum(1 for row in rows if bool(row.get("correct", False)))
        oracle_3 = sum(1 for row in rows if bool(row.get("oracle_at_3", False)))
        oracle_10 = sum(1 for row in rows if bool(row.get("oracle_at_10", False)))
        oracle_all = sum(1 for row in rows if bool(row.get("oracle_at_all", False)))
        generation_failures = sum(1 for row in rows if not bool(row.get("oracle_at_all", False)))
        ranking_failures = sum(
            1 for row in rows if bool(row.get("oracle_at_all", False)) and not bool(row.get("correct", False))
        )
        ranking_changes = sum(1 for row in rows if bool(row.get("ranking_changed_top1", False)))
        ptx_enabled = sum(1 for row in rows if bool(row.get("ptx_ranking_enabled", False)))
        ptx_used = sum(1 for row in rows if bool(row.get("ptx_ranking_used", False)))
        ptx_errors = sum(1 for row in rows if bool(row.get("ptx_ranking_error")))
        fuzzy_oracle_all = sum(1 for row in rows if bool(row.get("fuzzy_oracle_at_all", False)))
        fuzzy_oracle_080 = sum(1 for row in rows if bool(row.get("oracle_fuzzy_0_80", False)))
        fuzzy_oracle_085 = sum(1 for row in rows if bool(row.get("oracle_fuzzy_0_85", False)))
        fuzzy_oracle_090 = sum(1 for row in rows if bool(row.get("oracle_fuzzy_0_90", False)))
        fuzzy_oracle_095 = sum(1 for row in rows if bool(row.get("oracle_fuzzy_0_95", False)))
        fuzzy_scores = [float(row.get("fuzzy_best_score", 0.0)) for row in rows]
        gate_reject_rates = [float(row.get("validity_reject_rate", 0.0)) for row in rows]
        family_rejects = [int(row.get("validity_family_rejects", 0)) for row in rows]

        ranks = [int(row["correct_rank"]) for row in rows if row.get("correct_rank") is not None]
        avg_rank = (sum(ranks) / len(ranks)) if ranks else None
        return {
            "top_1_accuracy": top_1 / total,
            "oracle_at_3": oracle_3 / total,
            "oracle_at_10": oracle_10 / total,
            "oracle_at_all": oracle_all / total,
            "avg_correct_rank": avg_rank,
            "generation_failure_rate": generation_failures / total,
            "ranking_failure_rate": ranking_failures / total,
            "ranking_change_rate": ranking_changes / total,
            "fuzzy_oracle_at_all": fuzzy_oracle_all / total,
            "oracle_fuzzy_0_80": fuzzy_oracle_080 / total,
            "oracle_fuzzy_0_85": fuzzy_oracle_085 / total,
            "oracle_fuzzy_0_90": fuzzy_oracle_090 / total,
            "oracle_fuzzy_0_95": fuzzy_oracle_095 / total,
            "fuzzy_best_score_mean": (sum(fuzzy_scores) / total) if fuzzy_scores else 0.0,
            "validity_reject_rate_mean": (sum(gate_reject_rates) / total) if gate_reject_rates else 0.0,
            "family_rejects_mean": (sum(family_rejects) / total) if family_rejects else 0.0,
            "ptx_ranking_enabled_rate": ptx_enabled / total,
            "ptx_ranking_used_rate": ptx_used / total,
            "ptx_ranking_error_rate": ptx_errors / total,
        }

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
