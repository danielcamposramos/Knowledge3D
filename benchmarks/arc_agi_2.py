"""ARC-AGI benchmark integration for Knowledgeverse Week 14."""

from __future__ import annotations

import json
from collections import Counter
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
        enable_full_ptx: bool = False,
        ptx_validity_strictness: str = "medium",
        constraint_mode: str = "reject",
        enable_figure_ground_reversal: bool = False,
        enable_object_aware_generation: bool = False,
        enable_forced_navigation_curriculum: bool = False,
        forced_navigation_ratio: float = 0.0,
        forced_navigation_required_galaxies: str | list[str] | None = None,
        enable_rescue_lane: bool = False,
        rescue_lane_size: int = 16,
        oracle_search_lane_size: int = 32,
        enable_oracle_rejected_rescue: bool = False,
        oracle_rejected_rescue_size: int = 16,
        oracle_rejected_rescue_fuzzy_threshold: float = 0.90,
        enable_dual_track_oracle: bool = False,
        family_penalty_weight: float = 1.0,
        shape_penalty_weight: float = 1.0,
        palette_penalty_weight: float = 1.0,
        object_penalty_weight: float = 1.0,
        query_scope_galaxies: str | list[str] | None = None,
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
        self.enable_full_ptx = bool(enable_full_ptx)
        self.ptx_validity_strictness = str(ptx_validity_strictness or "medium")
        self.constraint_mode = str(constraint_mode or "reject").strip().lower()
        if self.constraint_mode not in {"reject", "penalty"}:
            self.constraint_mode = "reject"
        self.enable_figure_ground_reversal = bool(enable_figure_ground_reversal)
        self.enable_object_aware_generation = bool(enable_object_aware_generation)
        self.enable_forced_navigation_curriculum = bool(enable_forced_navigation_curriculum)
        self.forced_navigation_ratio = float(forced_navigation_ratio)
        self.forced_navigation_required_galaxies = forced_navigation_required_galaxies
        self.enable_rescue_lane = bool(enable_rescue_lane)
        self.rescue_lane_size = int(rescue_lane_size)
        self.oracle_search_lane_size = int(oracle_search_lane_size)
        self.enable_oracle_rejected_rescue = bool(enable_oracle_rejected_rescue)
        self.oracle_rejected_rescue_size = int(oracle_rejected_rescue_size)
        self.oracle_rejected_rescue_fuzzy_threshold = float(oracle_rejected_rescue_fuzzy_threshold)
        self.enable_dual_track_oracle = bool(enable_dual_track_oracle)
        self.family_penalty_weight = float(family_penalty_weight)
        self.shape_penalty_weight = float(shape_penalty_weight)
        self.palette_penalty_weight = float(palette_penalty_weight)
        self.object_penalty_weight = float(object_penalty_weight)
        self.query_scope_galaxies = self._normalize_query_scope(query_scope_galaxies)
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
            strict_legacy=(self.strict_legacy or self.enable_full_ptx),
            knowledgeverse=self.kv,
            enable_contrastive_learning=self.enable_contrastive_learning,
            enable_validity_gates=self.enable_validity_gates,
            enable_fuzzy_oracle=self.enable_fuzzy_oracle,
            fuzzy_oracle_threshold=self.fuzzy_oracle_threshold,
            enable_ptx_ranking=self.enable_ptx_ranking,
            enable_full_ptx=self.enable_full_ptx,
            ptx_validity_strictness=self.ptx_validity_strictness,
            constraint_mode=self.constraint_mode,
            enable_figure_ground_reversal=self.enable_figure_ground_reversal,
            enable_object_aware_generation=self.enable_object_aware_generation,
            enable_forced_navigation_curriculum=self.enable_forced_navigation_curriculum,
            forced_navigation_ratio=self.forced_navigation_ratio,
            forced_navigation_required_galaxies=self.forced_navigation_required_galaxies,
            enable_rescue_lane=self.enable_rescue_lane,
            rescue_lane_size=self.rescue_lane_size,
            oracle_search_lane_size=self.oracle_search_lane_size,
            enable_oracle_rejected_rescue=self.enable_oracle_rejected_rescue,
            oracle_rejected_rescue_size=self.oracle_rejected_rescue_size,
            oracle_rejected_rescue_fuzzy_threshold=self.oracle_rejected_rescue_fuzzy_threshold,
            enable_dual_track_oracle=self.enable_dual_track_oracle,
            family_penalty_weight=self.family_penalty_weight,
            shape_penalty_weight=self.shape_penalty_weight,
            palette_penalty_weight=self.palette_penalty_weight,
            object_penalty_weight=self.object_penalty_weight,
            query_scope_galaxies=self.query_scope_galaxies,
        )
        for task in self.tasks:
            result = self._solve_task(task=task, use_enriched=use_enriched)
            if self.enable_full_ptx and str(result.get("solver")) != "arc_ptx_ops":
                raise RuntimeError(
                    "PTX ARC solver contract violated: expected solver='arc_ptx_ops' "
                    f"but got '{result.get('solver')}'."
                )
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
                "verification": str(result.get("solver", "arc_solver")),
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
            galaxy_names=self.query_scope_galaxies,
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

    def _normalize_query_scope(self, value: str | list[str] | None) -> list[str] | None:
        if isinstance(value, list):
            raw = [str(item).strip() for item in value]
        elif isinstance(value, str):
            raw = [segment.strip() for segment in value.split(",")]
        else:
            raw = []
        normalized: list[str] = []
        seen: set[str] = set()
        for item in raw:
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(item)
        return normalized or None

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
        ptx_full_enabled = sum(1 for row in rows if bool(row.get("ptx_full_enabled", False)))
        ptx_full_used = sum(1 for row in rows if bool(row.get("ptx_full_used", False)))
        ptx_oracle_used = sum(1 for row in rows if bool(row.get("ptx_oracle_used", False)))
        fuzzy_oracle_all = sum(1 for row in rows if bool(row.get("fuzzy_oracle_at_all", False)))
        fuzzy_oracle_080 = sum(1 for row in rows if bool(row.get("oracle_fuzzy_0_80", False)))
        fuzzy_oracle_085 = sum(1 for row in rows if bool(row.get("oracle_fuzzy_0_85", False)))
        fuzzy_oracle_090 = sum(1 for row in rows if bool(row.get("oracle_fuzzy_0_90", False)))
        fuzzy_oracle_095 = sum(1 for row in rows if bool(row.get("oracle_fuzzy_0_95", False)))
        fuzzy_scores = [float(row.get("fuzzy_best_score", 0.0)) for row in rows]
        gate_reject_rates = [float(row.get("validity_reject_rate", 0.0)) for row in rows]
        family_rejects = [int(row.get("validity_family_rejects", 0)) for row in rows]
        shape_rejects = [int(row.get("validity_shape_rejects", 0)) for row in rows]
        palette_rejects = [int(row.get("validity_palette_rejects", 0)) for row in rows]
        object_rejects = [int(row.get("validity_object_rejects", 0)) for row in rows]
        generation_family_rejects = [int(row.get("generation_filter_family_rejects", 0)) for row in rows]
        generation_shape_rejects = [int(row.get("generation_filter_shape_rejects", 0)) for row in rows]
        generation_palette_rejects = [int(row.get("generation_filter_palette_rejects", 0)) for row in rows]
        generation_object_rejects = [int(row.get("generation_filter_object_rejects", 0)) for row in rows]
        rescue_candidate_counts = [int(row.get("oracle_rejected_rescue_candidate_count", 0)) for row in rows]
        rescue_exact_hits = sum(1 for row in rows if bool(row.get("oracle_rejected_rescue_exact", False)))
        rescue_fuzzy_hits = sum(1 for row in rows if bool(row.get("oracle_rejected_rescue_fuzzy", False)))
        rescue_reason_counts: Counter[str] = Counter()
        for row in rows:
            for key, value in (row.get("oracle_rejected_rescue_reason_counts", {}) or {}).items():
                try:
                    rescue_reason_counts[str(key)] += int(value)
                except Exception:
                    continue
        rejected_was_better_count = sum(1 for row in rows if bool(row.get("rejected_was_better", False)))
        fuzzy_delta_positive = [
            float(row.get("fuzzy_delta", 0.0))
            for row in rows
            if bool(row.get("rejected_was_better", False))
        ]
        family_scores = [
            float((row.get("ranking_top_components", {}) or {}).get("family_score", 0.0))
            for row in rows
        ]
        shape_scores = [
            float((row.get("ranking_top_components", {}) or {}).get("shape_score", 0.0))
            for row in rows
        ]
        palette_scores = [
            float((row.get("ranking_top_components", {}) or {}).get("palette_score", 0.0))
            for row in rows
        ]
        object_scores = [
            float((row.get("ranking_top_components", {}) or {}).get("object_score", 0.0))
            for row in rows
        ]
        generation_accept_rates = [float(row.get("generation_filter_accept_rate", 0.0)) for row in rows]
        generation_reject_rates = [float(row.get("generation_filter_reject_rate", 0.0)) for row in rows]
        generation_totals = [int(row.get("generation_filter_generated_total", 0)) for row in rows]
        query_counts = [int(row.get("queried_galaxy_count", 0)) for row in rows]
        cross_galaxy_navigation = sum(
            1
            for row in rows
            if int(row.get("queried_galaxy_count", 0)) >= 3
            or int(row.get("cross_galaxy_composition_count", 0)) > 0
        )
        queried_galaxy_task_counts: Counter[str] = Counter()
        source_galaxy_task_counts: Counter[str] = Counter()
        target_galaxy_task_counts: Counter[str] = Counter()
        object_count_distribution: Counter[str] = Counter()
        object_count_distribution_accepted: Counter[str] = Counter()
        object_count_distribution_rejected: Counter[str] = Counter()
        for row in rows:
            queried = row.get("queried_galaxies", [])
            if isinstance(queried, list):
                for name in {str(item) for item in queried if str(item).strip()}:
                    queried_galaxy_task_counts[name] += 1
            for key, value in (row.get("source_galaxy_counts", {}) or {}).items():
                try:
                    source_galaxy_task_counts[str(key)] += int(value)
                except Exception:
                    continue
            for key, value in (row.get("target_galaxy_counts", {}) or {}).items():
                try:
                    target_galaxy_task_counts[str(key)] += int(value)
                except Exception:
                    continue
            for key, value in (row.get("generation_object_count_distribution", {}) or {}).items():
                try:
                    object_count_distribution[str(key)] += int(value)
                except Exception:
                    continue
            for key, value in (row.get("generation_object_count_distribution_accepted", {}) or {}).items():
                try:
                    object_count_distribution_accepted[str(key)] += int(value)
                except Exception:
                    continue
            for key, value in (row.get("generation_object_count_distribution_rejected", {}) or {}).items():
                try:
                    object_count_distribution_rejected[str(key)] += int(value)
                except Exception:
                    continue
        rescue_enabled = sum(1 for row in rows if bool(row.get("rescue_lane_enabled", False)))
        selected_exact = sum(1 for row in rows if bool(row.get("selected_exact_match", False)))
        selected_tracks = Counter(str(row.get("selected_oracle_track", "rank_top1")) for row in rows)
        failure_modes = {"family": 0, "shape": 0, "palette": 0, "object_count": 0, "generation_gap": 0, "near_miss": 0}
        failure_mode_query_counts: dict[str, list[int]] = {
            "family": [],
            "shape": [],
            "palette": [],
            "object_count": [],
            "generation_gap": [],
            "near_miss": [],
        }
        failure_mode_galaxy_counts: dict[str, Counter[str]] = {
            key: Counter() for key in failure_mode_query_counts
        }
        composition_depth_distribution: Counter[str] = Counter()
        for row in rows:
            details = row.get("oracle_failure_modes", {})
            if not isinstance(details, dict):
                continue
            query_list = row.get("queried_galaxies", [])
            if not isinstance(query_list, list):
                query_list = []
            query_set = {str(item) for item in query_list if str(item).strip()}
            query_count = len(query_set)
            top_components = row.get("ranking_top_components", {})
            if isinstance(top_components, dict):
                try:
                    depth = max(1, int(top_components.get("composition_depth", 1)))
                    composition_depth_distribution[str(depth)] += 1
                except Exception:
                    composition_depth_distribution["1"] += 1
            if bool(details.get("family_mismatch", False)):
                failure_modes["family"] += 1
                failure_mode_query_counts["family"].append(query_count)
                failure_mode_galaxy_counts["family"].update(query_set)
            if bool(details.get("shape_mismatch", False)):
                failure_modes["shape"] += 1
                failure_mode_query_counts["shape"].append(query_count)
                failure_mode_galaxy_counts["shape"].update(query_set)
            if bool(details.get("palette_mismatch", False)):
                failure_modes["palette"] += 1
                failure_mode_query_counts["palette"].append(query_count)
                failure_mode_galaxy_counts["palette"].update(query_set)
            if bool(details.get("object_count_mismatch", False)):
                failure_modes["object_count"] += 1
                failure_mode_query_counts["object_count"].append(query_count)
                failure_mode_galaxy_counts["object_count"].update(query_set)
            root = str(details.get("root_cause", "")).lower()
            if root == "generation_gap":
                failure_modes["generation_gap"] += 1
                failure_mode_query_counts["generation_gap"].append(query_count)
                failure_mode_galaxy_counts["generation_gap"].update(query_set)
            if root == "near_miss_generation":
                failure_modes["near_miss"] += 1
                failure_mode_query_counts["near_miss"].append(query_count)
                failure_mode_galaxy_counts["near_miss"].update(query_set)

        failure_mode_navigation: dict[str, Any] = {}
        for mode, counts in failure_mode_query_counts.items():
            mode_counter = failure_mode_galaxy_counts[mode]
            failure_mode_navigation[mode] = {
                "tasks": int(failure_modes.get(mode, 0)),
                "avg_queried_galaxies": (sum(counts) / len(counts)) if counts else 0.0,
                "queried_galaxy_task_counts": dict(mode_counter),
                "unique_queried_galaxies": sorted(mode_counter.keys()),
            }

        composition_depth_total = sum(int(v) for v in composition_depth_distribution.values())
        composition_depth_avg = 0.0
        if composition_depth_total > 0:
            weighted = 0
            for depth_key, count in composition_depth_distribution.items():
                try:
                    weighted += int(depth_key) * int(count)
                except Exception:
                    continue
            composition_depth_avg = weighted / composition_depth_total

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
            "rejected_was_better_count": int(rejected_was_better_count),
            "rejected_was_better_rate": (rejected_was_better_count / total),
            "rejected_fuzzy_delta_mean": (
                (sum(fuzzy_delta_positive) / len(fuzzy_delta_positive))
                if fuzzy_delta_positive
                else 0.0
            ),
            "validity_reject_rate_mean": (sum(gate_reject_rates) / total) if gate_reject_rates else 0.0,
            "family_rejects_mean": (sum(family_rejects) / total) if family_rejects else 0.0,
            "shape_rejects_mean": (sum(shape_rejects) / total) if shape_rejects else 0.0,
            "palette_rejects_mean": (sum(palette_rejects) / total) if palette_rejects else 0.0,
            "object_rejects_mean": (sum(object_rejects) / total) if object_rejects else 0.0,
            "generation_family_rejects_mean": (
                (sum(generation_family_rejects) / total) if generation_family_rejects else 0.0
            ),
            "generation_shape_rejects_mean": (
                (sum(generation_shape_rejects) / total) if generation_shape_rejects else 0.0
            ),
            "generation_palette_rejects_mean": (
                (sum(generation_palette_rejects) / total) if generation_palette_rejects else 0.0
            ),
            "generation_object_rejects_mean": (
                (sum(generation_object_rejects) / total) if generation_object_rejects else 0.0
            ),
            "ranking_family_score_mean": (sum(family_scores) / total) if family_scores else 0.0,
            "ranking_shape_score_mean": (sum(shape_scores) / total) if shape_scores else 0.0,
            "ranking_palette_score_mean": (sum(palette_scores) / total) if palette_scores else 0.0,
            "ranking_object_score_mean": (sum(object_scores) / total) if object_scores else 0.0,
            "generation_filter_accept_rate_mean": (
                (sum(generation_accept_rates) / total) if generation_accept_rates else 0.0
            ),
            "generation_filter_reject_rate_mean": (
                (sum(generation_reject_rates) / total) if generation_reject_rates else 0.0
            ),
            "generation_filter_generated_total": int(sum(generation_totals)),
            "generation_object_count_distribution": dict(object_count_distribution),
            "generation_object_count_distribution_accepted": dict(object_count_distribution_accepted),
            "generation_object_count_distribution_rejected": dict(object_count_distribution_rejected),
            "query_based_coverage": {
                "avg_queried_galaxies_per_task": (sum(query_counts) / total) if query_counts else 0.0,
                "unique_queried_galaxies": sorted(queried_galaxy_task_counts.keys()),
                "queried_galaxy_task_counts": dict(queried_galaxy_task_counts),
                "source_galaxy_query_counts": dict(source_galaxy_task_counts),
                "target_galaxy_query_counts": dict(target_galaxy_task_counts),
                "cross_galaxy_navigation_rate": (cross_galaxy_navigation / total),
            },
            "failure_mode_query_coverage": failure_mode_navigation,
            "composition_depth": {
                "avg": composition_depth_avg,
                "distribution": dict(composition_depth_distribution),
            },
            "rescue_lane_enabled_rate": rescue_enabled / total,
            "oracle_rejected_rescue_candidate_count_mean": (
                (sum(rescue_candidate_counts) / total) if rescue_candidate_counts else 0.0
            ),
            "oracle_rejected_rescue_exact_rate": (rescue_exact_hits / total),
            "oracle_rejected_rescue_fuzzy_rate": (rescue_fuzzy_hits / total),
            "oracle_rejected_rescue_reason_counts": dict(rescue_reason_counts),
            "selected_exact_rate": selected_exact / total,
            "selected_oracle_track_counts": dict(selected_tracks),
            "ptx_ranking_enabled_rate": ptx_enabled / total,
            "ptx_ranking_used_rate": ptx_used / total,
            "ptx_ranking_error_rate": ptx_errors / total,
            "ptx_full_enabled_rate": ptx_full_enabled / total,
            "ptx_full_used_rate": ptx_full_used / total,
            "ptx_oracle_used_rate": ptx_oracle_used / total,
            "oracle_failure_mode_counts": failure_modes,
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
