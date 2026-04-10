"""GSM8K benchmark integration routed through the unified sovereign math path."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Callable

from benchmarks.math_competitions import UnifiedMathBenchmark
from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


class GSM8KBenchmark:
    """Thin GSM8K wrapper over the shared UnifiedMathBenchmark solver/comparator path."""

    def __init__(
        self,
        knowledgeverse: Knowledgeverse | None = None,
        dataset_path: str | Path | None = None,
        max_questions: int | None = None,
        query_scope_galaxies: str | list[str] | None = None,
        runtime_seed_knowledge: bool = False,
        tablet_boundary: HeadlessTabletMPC | None = None,
    ) -> None:
        self._bench = UnifiedMathBenchmark(
            knowledgeverse=knowledgeverse,
            gsm8k_dataset_path=dataset_path,
            max_math_questions=max_questions,
            source_filter=["gsm8k"],
            query_scope_galaxies=query_scope_galaxies,
            runtime_seed_knowledge=runtime_seed_knowledge,
            tablet_boundary=tablet_boundary,
        )
        self.kv = self._bench.kv
        self.dataset_path = self._bench.gsm8k_dataset_path
        self.max_questions = max_questions
        self.query_scope_galaxies = self._bench.query_scope_galaxies
        self.runtime_seed_knowledge = self._bench.runtime_seed_knowledge
        self.used_synthetic_fallback = self._bench.used_synthetic_math_fallback
        self.questions = [
            {
                "id": row["id"],
                "question_text": row["problem_text"],
                "correct_answer": row["answer"],
            }
            for row in self._bench.problems
        ]
        self.results: list[dict[str, Any]] = []
        self.last_summary: dict[str, Any] = {}

        self.dataset_source = "GSM8K"
        self.dataset_file = str(self.dataset_path) if self.dataset_path.exists() else "not_found"
        self.synthetic_fallback = self.used_synthetic_fallback

    def run_benchmark(
        self,
        use_enriched: bool = True,
        *,
        progress_cb: Callable[[dict[str, Any]], None] | None = None,
        progress_every: int | None = None,
        row_cb: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
        start_index: int = 0,
        initial_correct: int = 0,
        initial_source_completed: dict[str, int] | None = None,
        initial_source_correct: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        summary = self._bench.run_benchmark(
            use_enriched=use_enriched,
            progress_cb=progress_cb,
            progress_every=progress_every,
            row_cb=row_cb,
            start_index=start_index,
            initial_correct=initial_correct,
            initial_source_completed=initial_source_completed,
            initial_source_correct=initial_source_correct,
        )
        self.questions = [
            {
                "id": row["id"],
                "question_text": row["problem_text"],
                "correct_answer": row["answer"],
            }
            for row in self._bench.problems
        ]
        self.results = [
            {
                "question_id": row["problem_id"],
                "question_text": source["question_text"],
                "predicted_answer": row.get("predicted_answer"),
                "answer_format": self._classify_answer_format(row.get("predicted_answer")),
                "correct_answer": source["correct_answer"],
                "correct": bool(row.get("correct", False)),
                "elapsed_s": float(row.get("elapsed_s", 0.0) or 0.0),
                "elapsed_ms": round(float(row.get("elapsed_s", 0.0) or 0.0) * 1000.0, 3),
                "reasoning_trace": list(row.get("reasoning_trace", [])),
                "route": row.get("route"),
                "solver": str(row.get("solver", "")),
                "runtime": str(row.get("runtime", "")),
                "gpu_execution": bool(row.get("gpu_execution", False)),
                "program_id": str(row.get("program_id", "")),
                "task_result": row.get("task_result", {}),
            }
            for source, row in zip(self.questions, self._bench.results)
        ]
        output = {
            "benchmark": "GSM8K",
            "total_questions": len(self.questions),
            "correct": sum(1 for row in self.results if row.get("correct")),
            "accuracy": summary["overall_accuracy"],
            "use_enriched": use_enriched,
            "dataset_path": self.dataset_file,
            "dataset_source": self.dataset_source,
            "synthetic_fallback": self.synthetic_fallback,
            "timing_summary": self._timing_summary(),
            "answer_format_counts": self._answer_format_counts(),
            "results": self.results,
        }
        self.last_summary = dict(output)
        return output

    def _timing_summary(self) -> dict[str, float | int]:
        timings = [
            float(row.get("elapsed_s", 0.0) or 0.0)
            for row in self.results
            if float(row.get("elapsed_s", 0.0) or 0.0) >= 0.0
        ]
        if not timings:
            return {
                "count": 0,
                "avg_task_s": 0.0,
                "avg_task_ms": 0.0,
                "median_task_s": 0.0,
                "median_task_ms": 0.0,
                "min_task_s": 0.0,
                "max_task_s": 0.0,
            }
        avg_s = sum(timings) / len(timings)
        med_s = float(median(timings))
        return {
            "count": len(timings),
            "avg_task_s": round(avg_s, 6),
            "avg_task_ms": round(avg_s * 1000.0, 3),
            "median_task_s": round(med_s, 6),
            "median_task_ms": round(med_s * 1000.0, 3),
            "min_task_s": round(min(timings), 6),
            "max_task_s": round(max(timings), 6),
        }

    def _answer_format_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter(
            str(row.get("answer_format") or "unknown")
            for row in self.results
        )
        return {key: int(count) for key, count in sorted(counts.items())}

    @staticmethod
    def _classify_answer_format(value: Any) -> str:
        if value is None:
            return "empty"
        text = str(value).strip()
        if not text:
            return "empty"
        try:
            float(text)
            return "numeric"
        except Exception:
            return "text"

    def save_results(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.last_summary) if self.last_summary else {
            "benchmark": "GSM8K",
            "total_questions": len(self.results),
            "correct": sum(1 for row in self.results if row.get("correct")),
            "accuracy": (
                sum(1 for row in self.results if row.get("correct")) / len(self.results)
                if self.results
                else 0.0
            ),
            "timing_summary": self._timing_summary(),
            "answer_format_counts": self._answer_format_counts(),
            "results": self.results,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sovereign GSM8K benchmark through the tablet boundary.")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--query-scope-galaxies", type=str, default=None)
    parser.add_argument("--summary-output", type=str, default=None)
    parser.add_argument("--use-enriched", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    benchmark = GSM8KBenchmark(
        dataset_path=args.dataset_path,
        max_questions=args.max_tasks,
        query_scope_galaxies=args.query_scope_galaxies,
    )
    summary = benchmark.run_benchmark(use_enriched=bool(args.use_enriched))
    if args.summary_output:
        benchmark.save_results(args.summary_output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
