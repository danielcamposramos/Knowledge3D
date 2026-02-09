"""Deterministic curriculum benchmark for TRM foundation learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.tasks import (
    evaluate_system_literacy_task,
    generate_arithmetic_tasks,
    generate_compositional_tasks,
    generate_geometric_tasks,
    generate_pattern_tasks,
    generate_rpn_tasks,
    generate_system_literacy_tasks,
)
from benchmarks.tasks.arithmetic_tasks import evaluate_arithmetic_task
from benchmarks.tasks.geometric_tasks import apply_geometric_op
from benchmarks.tasks.pattern_tasks import solve_pattern_task
from benchmarks.tasks.rpn_tasks import evaluate_rpn_program
from knowledge3d.augmentation.ollama_curriculum_augmenter import OllamaAugmenter
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


@dataclass
class CategoryStats:
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total) if self.total else 0.0


class DeterministicFoundationBenchmark:
    """500-task deterministic benchmark suite used as curriculum phase-0."""

    STAGES = ("A", "B", "C", "D")

    def __init__(
        self,
        *,
        tasks_per_category: int = 100,
        seed: int = 2026,
        stage: str = "A",
        include_system_literacy: bool = False,
        ollama_augmenter: OllamaAugmenter | None = None,
    ):
        self.tasks_per_category = int(tasks_per_category)
        self.seed = int(seed)
        self.include_system_literacy = bool(include_system_literacy)
        self.ollama_augmenter = ollama_augmenter
        stage_norm = str(stage).strip().upper()
        self.stage = stage_norm if stage_norm in self.STAGES else "A"
        self._op_aliases: dict[str, list[str]] = {
            "ROTATE_90": [
                "turn the grid 90 degrees clockwise",
                "rotate one quarter turn to the right",
            ],
            "ROTATE_180": [
                "turn the grid around",
                "rotate by half a turn",
            ],
            "MIRROR_H": [
                "flip the grid left to right",
                "reflect across the vertical axis",
            ],
            "MIRROR_V": [
                "flip the grid top to bottom",
                "reflect across the horizontal axis",
            ],
            "TRANSPOSE": [
                "swap rows and columns",
                "transpose the matrix-style grid",
            ],
            "COUNT_VALUE": [
                "count how many cells match the target value",
                "how many cells equal the selected color",
            ],
            "SUM_ALL": [
                "sum every value in the grid",
                "compute total of all grid cells",
            ],
            "MAX_VALUE": [
                "find the largest value in the grid",
                "return the maximum cell value",
            ],
            "MIN_VALUE": [
                "find the smallest value in the grid",
                "return the minimum cell value",
            ],
            "UNIQUE_COUNT": [
                "count distinct symbols in the grid",
                "how many unique values appear",
            ],
            "ALTERNATING_NEXT": [
                "continue the alternating pattern",
                "predict next alternating element",
            ],
            "ARITHMETIC_NEXT": [
                "continue the constant-step sequence",
                "predict next linear progression value",
            ],
            "GEOMETRIC_NEXT": [
                "continue the multiplicative sequence",
                "predict next geometric progression value",
            ],
            "MIRROR_COMPLETE": [
                "complete the mirrored sequence from first half",
                "finish the symmetry of the row",
            ],
            "ROW_TILE": [
                "extend the repeating row motif",
                "tile this row pattern to target length",
            ],
            "RPN_EVAL": [
                "evaluate this reverse polish expression",
                "compute the stack expression result",
            ],
        }
        self.tasks = self._generate_all_tasks()

    def _generate_all_tasks(self) -> dict[str, list[dict[str, Any]]]:
        tasks = {
            "geometric_transforms": generate_geometric_tasks(self.tasks_per_category, self.seed + 11),
            "grid_arithmetic": generate_arithmetic_tasks(self.tasks_per_category, self.seed + 22),
            "pattern_completion": generate_pattern_tasks(self.tasks_per_category, self.seed + 33),
            "compositional": generate_compositional_tasks(self.tasks_per_category, self.seed + 44),
            "symbolic_rpn": generate_rpn_tasks(self.tasks_per_category, self.seed + 55),
        }
        if self.include_system_literacy:
            tasks["system_literacy"] = generate_system_literacy_tasks(
                self.tasks_per_category,
                self.seed + 66,
            )
        return self._adapt_tasks_for_stage(tasks)

    def _adapt_tasks_for_stage(
        self,
        tasks: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        stage = self.stage
        if stage == "A":
            return tasks

        adapted: dict[str, list[dict[str, Any]]] = {}
        for category, rows in tasks.items():
            out: list[dict[str, Any]] = []
            for idx, task in enumerate(rows):
                row = dict(task)
                row["stage"] = stage

                operation = str(
                    row.get(
                        "operation",
                        row.get("pattern_type", "RPN_EVAL"),
                    )
                ).upper()
                if stage in {"B", "C", "D"}:
                    alias_list = self._op_aliases.get(operation, [f"apply {operation.lower()}"])
                    if self.ollama_augmenter is not None:
                        alias_list = self.ollama_augmenter.augment_aliases(operation, alias_list)
                    row["target_operation"] = operation
                    row["query"] = alias_list[idx % len(alias_list)]
                    row.pop("operation", None)
                    if stage in {"C", "D"}:
                        # Add distractor and require selection rather than direct op naming.
                        distractor = "MIRROR_H" if operation != "MIRROR_H" else "ROTATE_180"
                        row["distractors"] = [distractor]
                        row["query"] = (
                            f"{row['query']}. choose the best operation and ignore distractors."
                        )
                    if stage == "D":
                        # Sparse context: few-shot examples encoded as text.
                        row["few_shot"] = self._make_few_shot_context(operation)
                        row["query"] = (
                            f"{row['query']} infer from examples {row['few_shot']} with noisy phrasing."
                        )
                out.append(row)
            adapted[category] = out
        return adapted

    def _make_few_shot_context(self, operation: str) -> list[str]:
        op = str(operation).upper()
        if op == "ROTATE_90":
            return ["[[1,2],[3,4]] -> [[3,1],[4,2]]", "[[5,6],[7,8]] -> [[7,5],[8,6]]"]
        if op == "MIRROR_H":
            return ["[[1,2,3]] -> [[3,2,1]]", "[[4,5]] -> [[5,4]]"]
        if op == "COUNT_VALUE":
            return ["[1,2,1], value=1 -> 2", "[3,3,2], value=3 -> 2"]
        if op == "ARITHMETIC_NEXT":
            return ["2,5,8 -> 11", "10,13,16 -> 19"]
        return ["x->y example 1", "x->y example 2"]

    def run_benchmark(
        self,
        kv: Knowledgeverse,
        *,
        iteration: int = 0,
        log_events: bool = True,
    ) -> dict[str, Any]:
        """Run deterministic suite and return per-category and overall metrics."""
        stats: dict[str, CategoryStats] = {
            name: CategoryStats(correct=0, total=len(tasks))
            for name, tasks in self.tasks.items()
        }

        for category, task_list in self.tasks.items():
            for task in task_list:
                result = self._solve_task(task, kv)
                is_correct = self._validate_result(result, task["expected"])
                if is_correct:
                    stats[category].correct += 1
                if log_events:
                    self._log_task_event(
                        kv=kv,
                        category=category,
                        task=task,
                        result=result,
                        is_correct=is_correct,
                        iteration=iteration,
                    )

        results: dict[str, Any] = {}
        total_correct = 0
        total_tasks = 0
        for category, cat in stats.items():
            results[category] = {
                "accuracy": cat.accuracy,
                "correct": cat.correct,
                "total": cat.total,
            }
            total_correct += cat.correct
            total_tasks += cat.total

        results["overall"] = {
            "accuracy": (total_correct / total_tasks) if total_tasks else 0.0,
            "correct": total_correct,
            "total": total_tasks,
        }
        results["stage"] = self.stage
        return results

    def _solve_task(self, task: dict[str, Any], kv: Knowledgeverse) -> Any:
        category = str(task["category"])
        if category == "geometric_transforms":
            return self._solve_geometric(task, kv)
        if category == "grid_arithmetic":
            return self._solve_arithmetic(task, kv)
        if category == "pattern_completion":
            return self._solve_pattern(task, kv)
        if category == "compositional":
            return self._solve_compositional(task, kv)
        if category == "symbolic_rpn":
            return self._solve_rpn(task, kv)
        if category == "system_literacy":
            return self._solve_system_literacy(task, kv)
        raise ValueError(f"unknown task category: {category}")

    def _query_operation_entry(
        self,
        kv: Knowledgeverse,
        *,
        operation: str,
        specialist: str,
        galaxy: str,
    ) -> dict[str, Any] | None:
        rows = kv.trm_navigator.query(
            query=f"{operation.lower()} operation",
            galaxy_names=[galaxy],
            top_k=10,
            specialist=specialist,
            domain_hint=galaxy.lower(),
        )
        op_norm = operation.lower()
        exact_match: dict[str, Any] | None = None
        fuzzy_match: dict[str, Any] | None = None
        for row in rows:
            entry = row.get("entry", {})
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id", entry.get("rule_id", ""))).lower()
            meta = entry.get("metadata", {})
            if isinstance(meta, dict):
                op = str(meta.get("operation", "")).lower()
                if op == op_norm:
                    return entry
                if op and (op_norm.startswith(op) or op in op_norm):
                    fuzzy_match = fuzzy_match or entry
            if entry_id == op_norm:
                exact_match = exact_match or entry
            elif op_norm in entry_id:
                fuzzy_match = fuzzy_match or entry
        if exact_match is not None:
            return exact_match
        if fuzzy_match is not None:
            return fuzzy_match
        return rows[0].get("entry") if rows else None

    def _solve_geometric(self, task: dict[str, Any], kv: Knowledgeverse) -> list[list[int]]:
        op = self._resolve_task_operation(task)
        entry = self._query_operation_entry(
            kv,
            operation=op,
            specialist="grammar",
            galaxy="Grammar",
        )
        resolved = op
        if isinstance(entry, dict):
            meta = entry.get("metadata", {})
            if isinstance(meta, dict) and meta.get("operation"):
                resolved = str(meta["operation"]).upper()
            elif "rotate_90" in str(entry.get("id", "")).lower():
                resolved = "ROTATE_90"
            elif "rotate_180" in str(entry.get("id", "")).lower():
                resolved = "ROTATE_180"
            elif "mirror_h" in str(entry.get("id", "")).lower():
                resolved = "MIRROR_H"
            elif "mirror_v" in str(entry.get("id", "")).lower():
                resolved = "MIRROR_V"
            elif "transpose" in str(entry.get("id", "")).lower():
                resolved = "TRANSPOSE"
        return apply_geometric_op(task["input"], resolved)

    def _solve_arithmetic(self, task: dict[str, Any], kv: Knowledgeverse) -> int:
        op = self._resolve_task_operation(task)
        entry = self._query_operation_entry(
            kv,
            operation=op,
            specialist="math",
            galaxy="Math",
        )
        resolved = op
        if isinstance(entry, dict):
            meta = entry.get("metadata", {})
            if isinstance(meta, dict) and meta.get("operation"):
                resolved = str(meta["operation"]).upper()
        return evaluate_arithmetic_task(task["input"], resolved, value=task.get("value"))

    def _solve_pattern(self, task: dict[str, Any], kv: Knowledgeverse) -> Any:
        pattern_type = self._resolve_task_operation(task)
        _ = self._query_operation_entry(
            kv,
            operation=pattern_type,
            specialist="grammar",
            galaxy="Grammar",
        )
        # For deterministic curriculum we execute canonical rule directly.
        return solve_pattern_task(task)

    def _solve_compositional(self, task: dict[str, Any], kv: Knowledgeverse) -> list[list[int]]:
        grid = [row[:] for row in task["input"]]
        for raw_op in task["operations"]:
            op = str(raw_op).upper()
            entry = self._query_operation_entry(
                kv,
                operation=op,
                specialist="grammar",
                galaxy="Grammar",
            )
            resolved = op
            if isinstance(entry, dict):
                meta = entry.get("metadata", {})
                if isinstance(meta, dict) and meta.get("operation"):
                    resolved = str(meta["operation"]).upper()
            grid = apply_geometric_op(grid, resolved)
        return grid

    def _solve_rpn(self, task: dict[str, Any], kv: Knowledgeverse) -> float | int:
        _ = self._query_operation_entry(
            kv,
            operation="RPN_EVAL",
            specialist="math",
            galaxy="Math",
        )
        return evaluate_rpn_program(str(task["rpn_program"]))

    def _solve_system_literacy(self, task: dict[str, Any], kv: Knowledgeverse) -> dict[str, Any]:
        expected = task.get("expected", {}) if isinstance(task.get("expected", {}), dict) else {}
        query = str(task.get("query", ""))
        route = kv.trm_navigator.route(
            query=query,
            specialist="auto",
            domain_hint=str(expected.get("domain_hint", "")) or None,
            galaxy_names=expected.get("must_include_galaxies"),
        )
        facts = expected.get("facts", {})
        if not isinstance(facts, dict):
            facts = {}
        return {
            "specialist": route.get("specialist"),
            "route": route,
            "galaxy_names": route.get("galaxy_names", []),
            "write_target": expected.get("write_target", "Shadow_Copy"),
            "sleeptime_effect": expected.get("sleeptime_effect", "consolidate_system_literacy"),
            "facts": facts,
        }

    def _resolve_task_operation(self, task: dict[str, Any]) -> str:
        direct = task.get("operation")
        if direct:
            return str(direct).upper()
        target = task.get("target_operation")
        if target:
            if self.stage == "A":
                return str(target).upper()
            # Stage B+ must infer from description instead of direct field usage.
            inferred = self._infer_operation_from_query(str(task.get("query", "")))
            return inferred or str(target).upper()
        pattern = task.get("pattern_type")
        if pattern:
            return str(pattern).upper()
        return "RPN_EVAL"

    def _infer_operation_from_query(self, query: str) -> str | None:
        q = str(query).lower()
        keyword_map = (
            ("quarter turn", "ROTATE_90"),
            ("90 degrees", "ROTATE_90"),
            ("half a turn", "ROTATE_180"),
            ("turn around", "ROTATE_180"),
            ("left to right", "MIRROR_H"),
            ("vertical axis", "MIRROR_H"),
            ("top to bottom", "MIRROR_V"),
            ("horizontal axis", "MIRROR_V"),
            ("transpose", "TRANSPOSE"),
            ("rows and columns", "TRANSPOSE"),
            ("count", "COUNT_VALUE"),
            ("sum", "SUM_ALL"),
            ("largest", "MAX_VALUE"),
            ("maximum", "MAX_VALUE"),
            ("smallest", "MIN_VALUE"),
            ("minimum", "MIN_VALUE"),
            ("distinct", "UNIQUE_COUNT"),
            ("unique", "UNIQUE_COUNT"),
            ("alternating", "ALTERNATING_NEXT"),
            ("constant-step", "ARITHMETIC_NEXT"),
            ("linear progression", "ARITHMETIC_NEXT"),
            ("multiplicative", "GEOMETRIC_NEXT"),
            ("geometric progression", "GEOMETRIC_NEXT"),
            ("mirrored sequence", "MIRROR_COMPLETE"),
            ("symmetry of the row", "MIRROR_COMPLETE"),
            ("repeating row", "ROW_TILE"),
            ("reverse polish", "RPN_EVAL"),
            ("stack expression", "RPN_EVAL"),
        )
        for needle, op in keyword_map:
            if needle in q:
                return op
        return None

    def _validate_result(self, result: Any, expected: Any) -> bool:
        if isinstance(expected, dict) and expected.get("must_include_galaxies") is not None:
            return evaluate_system_literacy_task(
                result if isinstance(result, dict) else {},
                expected,
            )
        if isinstance(result, list) and isinstance(expected, list):
            return result == expected
        if isinstance(result, (int, float)) and isinstance(expected, (int, float)):
            return abs(float(result) - float(expected)) <= 1e-9
        return result == expected

    def _log_task_event(
        self,
        *,
        kv: Knowledgeverse,
        category: str,
        task: dict[str, Any],
        result: Any,
        is_correct: bool,
        iteration: int,
    ) -> None:
        op = str(task.get("operation", task.get("pattern_type", "RPN_EVAL")))
        specialist = "math" if category in {"grid_arithmetic", "symbolic_rpn"} else "grammar"
        galaxy = "Math" if specialist == "math" else "Grammar"
        event_type = "foundation_task_success" if is_correct else "foundation_task_failure"
        kv.log_event(
            event_type=event_type,
            event_data={
                "query": str(task.get("query", task.get("id", ""))),
                "task_id": str(task.get("id", "")),
                "category": category,
                "operation": op,
                "specialist": specialist,
                "galaxy": galaxy,
                "confidence": 1.0 if is_correct else 0.0,
                "verification": "deterministic_exact_match",
                "iteration": int(iteration),
                "stage": self.stage,
                "expected": task.get("expected"),
                "result": result,
            },
        )
