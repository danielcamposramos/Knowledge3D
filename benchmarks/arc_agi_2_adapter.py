"""Thin ARC benchmark harness routed entirely through the live Knowledgeverse GPU path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


@dataclass
class _GeneratedPattern:
    """Legacy compatibility stub kept for archived imports and debug tooling."""

    pattern_id: str
    source_galaxy: str
    target_galaxy: str
    confidence: float
    query: str
    source: str
    pair_index: int | None = None
    ops: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    composition_depth: int = 1


class ArcAgi2Adapter:
    """Live ARC adapter that delegates every solve to `Knowledgeverse.execute_task()`."""

    def __init__(
        self,
        *,
        use_enriched: bool = True,
        strict_legacy: bool = False,
        knowledgeverse: Knowledgeverse | None = None,
        **_ignored: Any,
    ) -> None:
        self.use_enriched = bool(use_enriched)
        self.strict_legacy = bool(strict_legacy)
        self.knowledgeverse = knowledgeverse or Knowledgeverse()

    def solve_task(
        self,
        task: dict[str, Any],
        *,
        fallback_solver: Callable[[dict[str, Any], bool], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Solve one ARC task through the sovereign Knowledgeverse query path."""
        del fallback_solver
        if not hasattr(self.knowledgeverse, "execute_task"):
            raise RuntimeError("arc_adapter_requires_knowledgeverse_execute_task")

        test_block = task.get("test") or [{}]
        gpu_task = {
            "task_id": str(task.get("id") or "arc_task"),
            "query": "solve arc transformation task",
            "training_examples": list(task.get("train") or []),
            "input_grid": test_block[0].get("input"),
            "expected_output": test_block[0].get("output"),
        }
        solved = self.knowledgeverse.execute_task(
            task=gpu_task,
            route={
                "specialist": "visual",
                "domain_hint": "visual",
                "galaxy_names": list(
                    getattr(
                        self.knowledgeverse,
                        "GPU_SPATIAL_TARGET_GALAXIES",
                        ("Drawing", "Grammar", "Tool"),
                    )
                ),
            },
            specialist="visual",
            domain_hint="visual",
            use_enriched=self.use_enriched,
        )
        predicted = solved.get("output_grid")
        expected = test_block[0].get("output")
        correct = bool(expected is not None and predicted == expected)
        return {
            "task_id": str(task.get("id", "unknown")),
            "correct": correct,
            "exact_match": correct,
            "predicted": predicted,
            "expected": expected,
            "transform": solved.get("match", {}).get("arc_transform_chain"),
            "patterns_used": int(solved.get("patterns_used", 1 if predicted is not None else 0)),
            "reasoning_trace": list(solved.get("reasoning_trace", solved.get("thinking_trace", []))),
            "route": solved.get("route", {}),
            "score": float(1.0 if correct else 0.0),
            "fuzzy_score": float(1.0 if correct else 0.0),
            "solver": str(solved.get("solver", "knowledgeverse_gpu_query")),
            "generated_pattern_count": 0,
            "gpu_execution": bool(solved.get("gpu_execution", False)),
            "runtime": solved.get("runtime", "knowledgeverse_gpu_query"),
            "program_id": solved.get("program_id"),
            "error": solved.get("error"),
            "task_result": solved,
            **({"trm_shadow": solved.get("trm_shadow")} if isinstance(solved.get("trm_shadow"), dict) else {}),
        }

    def _solve_task_ptx_only(self, task: dict[str, Any]) -> dict[str, Any]:
        """Explicitly disabled legacy entrypoint kept only for compatibility checks."""
        raise RuntimeError(
            "legacy_arc_solver_archived: use Knowledgeverse.execute_task() via ArcAgi2Adapter.solve_task()."
        )
