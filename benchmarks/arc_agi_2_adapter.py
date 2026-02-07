"""Adapter bridge from Week 14 ARC benchmark to legacy SovereignAIPipeline."""

from __future__ import annotations

from typing import Any, Callable

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


class ArcAgi2Adapter:
    """Translate Week 14 benchmark tasks into legacy ARC sovereign pipeline calls."""

    def __init__(
        self,
        *,
        use_enriched: bool = True,
        strict_legacy: bool = False,
        knowledgeverse: Knowledgeverse | None = None,
    ):
        self.use_enriched = use_enriched
        self.strict_legacy = strict_legacy
        self.knowledgeverse = knowledgeverse
        self.pipeline = None
        self._init_error: str | None = None

        try:
            from knowledge3d.training.arc_agi import SovereignAIPipeline

            self.pipeline = SovereignAIPipeline(
                matryoshka_dim=512 if use_enriched else 128,
                hybrid_mode=use_enriched,
                knowledgeverse=self.knowledgeverse,
            )
        except Exception as exc:  # pragma: no cover - environment dependent.
            self._init_error = str(exc)
            if strict_legacy:
                raise

    def solve_task(
        self,
        task: dict[str, Any],
        *,
        fallback_solver: Callable[[dict[str, Any], bool], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Solve one ARC task.

        If the legacy pipeline is unavailable or fails and `strict_legacy=False`,
        an optional fallback solver can be used to keep the benchmark runnable.
        """
        if self.pipeline is None:
            return self._fallback_or_raise(task, "pipeline_unavailable", fallback_solver)

        task_id = str(task.get("id", "unknown"))
        test_block = task.get("test") or [{}]
        test_input = test_block[0].get("input")
        expected_output = test_block[0].get("output")
        train_examples = task.get("train") or []

        try:
            result = self.pipeline.process_task(
                task_id=task_id,
                test_input=test_input,
                train_examples=train_examples,
                expected_output=expected_output,
                top_k=9 if self.use_enriched else 3,
                record_submission=False,
            )
            predicted = result.output_grid
            exact_match = self._grids_match(predicted, expected_output)
            return {
                "task_id": task_id,
                "correct": bool(result.correct),
                "exact_match": exact_match,
                "predicted": predicted,
                "expected": expected_output,
                "reasoning_trace": self._extract_reasoning_trace(result),
                "patterns_used": self._count_patterns_used(result.best_program),
                "solver": "legacy_sovereign_pipeline",
                "score": float(result.score),
                "fuzzy_score": float(getattr(result, "fuzzy_score", 0.0)),
            }
        except Exception as exc:
            return self._fallback_or_raise(task, str(exc), fallback_solver)

    def _fallback_or_raise(
        self,
        task: dict[str, Any],
        reason: str,
        fallback_solver: Callable[[dict[str, Any], bool], dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if self.strict_legacy or fallback_solver is None:
            raise RuntimeError(f"Legacy ARC pipeline unavailable: {reason}")
        fallback_result = fallback_solver(task, self.use_enriched)
        fallback_result["fallback_reason"] = reason
        fallback_result["solver"] = "trm_navigator_fallback"
        return fallback_result

    def _extract_reasoning_trace(self, result: Any) -> list[str]:
        lines = [
            f"program_type={result.program_type}",
            f"score={float(result.score):.4f}",
            f"fuzzy_score={float(getattr(result, 'fuzzy_score', 0.0)):.4f}",
        ]
        program = (result.best_program or "").strip()
        if program:
            snippet = program.splitlines()[:4]
            lines.extend(f"program::{line}" for line in snippet)
        signature = getattr(result, "signature", None)
        if signature:
            lines.append(f"signature={signature}")
        return lines

    def _count_patterns_used(self, program: str) -> int:
        patterns: set[str] = set()
        for raw_line in (program or "").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split()[0]
            patterns.add(token)
        return len(patterns)

    def _grids_match(
        self,
        predicted: list[list[int]] | None,
        expected: list[list[int]] | None,
    ) -> bool:
        if not isinstance(predicted, list) or not isinstance(expected, list):
            return False
        if len(predicted) != len(expected):
            return False
        for pred_row, exp_row in zip(predicted, expected):
            if list(pred_row) != list(exp_row):
                return False
        return True
