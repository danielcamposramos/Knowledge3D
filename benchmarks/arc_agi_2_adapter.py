"""Thin ARC benchmark harness routed entirely through the live Knowledgeverse GPU path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.tablet.wine.game2d_wine import arc2_game_envelope


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
    """Live ARC adapter that delegates every solve through the tablet/WINE boundary."""

    def __init__(
        self,
        *,
        use_enriched: bool = True,
        strict_legacy: bool = False,
        knowledgeverse: Knowledgeverse | None = None,
        tablet_boundary: HeadlessTabletMPC | None = None,
        **_ignored: Any,
    ) -> None:
        self.use_enriched = bool(use_enriched)
        self.strict_legacy = bool(strict_legacy)
        self.knowledgeverse = knowledgeverse or Knowledgeverse()
        self.tablet_boundary = tablet_boundary or self._build_tablet_boundary()

    def _build_tablet_boundary(self) -> HeadlessTabletMPC:
        if isinstance(self.knowledgeverse, Knowledgeverse):
            return HeadlessTabletMPC(knowledgeverse=self.knowledgeverse)

        def _handler(payload: dict[str, Any]) -> dict[str, Any]:
            task = dict(payload.get("task") or {})
            route = {
                "specialist": str(payload.get("specialist") or "visual"),
                "domain_hint": str(payload.get("domain_hint") or task.get("domain_hint") or "game_2d"),
                "galaxy_names": list(payload.get("galaxies") or task.get("galaxies") or []),
                "route_policy": str(payload.get("route_policy") or task.get("route_policy") or ""),
            }
            solved = self.knowledgeverse.execute_task(
                task=task,
                route=route,
                specialist=str(route["specialist"]),
                domain_hint=str(route["domain_hint"]),
                use_enriched=bool(payload.get("use_enriched", True)),
            )
            return {"status": "ok", "route": route, "task_result": dict(solved or {})}

        return HeadlessTabletMPC(command_handler=_handler)

    def solve_task(
        self,
        task: dict[str, Any],
        *,
        fallback_solver: Callable[[dict[str, Any], bool], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Solve one ARC task through the sovereign Knowledgeverse query path."""
        del fallback_solver

        test_block = task.get("test") or [{}]
        envelope = arc2_game_envelope(
            task_id=str(task.get("id") or "arc_task"),
            training_examples=list(task.get("train") or []),
            input_grid=test_block[0].get("input"),
            expected_output=test_block[0].get("output"),
        )
        tablet_result = self.tablet_boundary.submit(envelope, use_enriched=self.use_enriched)
        emitted = dict(tablet_result["emitted"])
        task_result = dict(emitted.get("task_result") or {})
        predicted = emitted.get("output_grid")
        expected = test_block[0].get("output")
        correct = bool(expected is not None and predicted == expected)
        return {
            "task_id": str(task.get("id", "unknown")),
            "correct": correct,
            "exact_match": correct,
            "predicted": predicted,
            "expected": expected,
            "transform": task_result.get("match", {}).get("arc_transform_chain"),
            "patterns_used": int(task_result.get("patterns_used", 1 if predicted is not None else 0)),
            "reasoning_trace": list(task_result.get("reasoning_trace", task_result.get("thinking_trace", []))),
            "route": emitted.get("route", {}),
            "score": float(task_result.get("score", 1.0 if correct else 0.0)),
            "fuzzy_score": float(task_result.get("fuzzy_score", 1.0 if correct else 0.0)),
            "solver": str(task_result.get("solver", "tablet_boundary")),
            "generated_pattern_count": int(task_result.get("generated_pattern_count", 0)),
            "gpu_execution": bool(task_result.get("gpu_execution", False)),
            "runtime": task_result.get("runtime", "knowledgeverse_gpu_query"),
            "program_id": task_result.get("program_id"),
            "error": emitted.get("failure_code"),
            "task_result": task_result,
            "tablet_contract": tablet_result.get("tablet_contract"),
            **({"trm_shadow": task_result.get("trm_shadow")} if isinstance(task_result.get("trm_shadow"), dict) else {}),
        }

    def _solve_task_ptx_only(self, task: dict[str, Any]) -> dict[str, Any]:
        """Explicitly disabled legacy entrypoint kept only for compatibility checks."""
        raise RuntimeError(
            "legacy_arc_solver_archived: use Knowledgeverse.execute_task() via ArcAgi2Adapter.solve_task()."
        )
