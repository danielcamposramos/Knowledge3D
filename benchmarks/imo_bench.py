"""IMO Bench integration routed through the unified sovereign math path."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

from benchmarks.math_competitions import UnifiedMathBenchmark
from benchmarks.sampling import stratified_sample
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


class IMOBenchmark:
    """Thin IMO wrapper over the shared UnifiedMathBenchmark math solver path."""

    def __init__(
        self,
        knowledgeverse: Knowledgeverse | None = None,
        dataset_path: str | Path | None = None,
        max_questions: int | None = None,
        query_scope_galaxies: str | list[str] | None = None,
        runtime_seed_knowledge: bool = False,
    ) -> None:
        self.kv = knowledgeverse or Knowledgeverse()
        self.dataset_path = self._resolve_dataset_path(dataset_path)
        self.max_questions = max_questions
        self.query_scope_galaxies = query_scope_galaxies
        self.runtime_seed_knowledge = bool(runtime_seed_knowledge)
        self.used_synthetic_fallback = False
        self.questions = self._load_questions()
        self.results: list[dict[str, Any]] = []
        self.dataset_source = "IMO-AnswerBench"
        self.dataset_file = str(self.dataset_path) if self.dataset_path.exists() else "not_found"
        self.synthetic_fallback = self.used_synthetic_fallback

        self._bench = UnifiedMathBenchmark(
            knowledgeverse=self.kv,
            dataset_mode="synthetic",
            max_problems=0,
            max_gsm8k_questions=0,
            source_filter=["math"],
            query_scope_galaxies=query_scope_galaxies,
            runtime_seed_knowledge=runtime_seed_knowledge,
        )
        self._bench.problems = [
            {
                "id": row["id"],
                "suite": "imo",
                "source_key": "imo",
                "competition": "IMO",
                "problem_text": row["question_text"],
                "answer": row["correct_answer"],
            }
            for row in self.questions
        ]

    def _resolve_dataset_path(self, dataset_path: str | Path | None) -> Path:
        if dataset_path is not None:
            return Path(dataset_path)
        candidates = [
            Path("/K3D/Knowledge3D.local/datasets/imo_bench/answerbench_v2.csv"),
            Path("/K3D/Knowledge3D.local/datasets/imo_bench/answerbench.csv"),
            Path("../Knowledge3D.local/datasets/imo_bench/answerbench_v2.csv"),
            Path("../Knowledge3D.local/datasets/imo_bench/answerbench.csv"),
            Path("../Knowledge3D.local/datasets/math_competitions/imo_problems.json"),
            Path("/K3D/Knowledge3D.local/datasets/math_competitions/imo_problems.json"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return Path("")

    def _load_questions(self) -> list[dict[str, Any]]:
        if self.dataset_path and self.dataset_path.exists():
            suffix = self.dataset_path.suffix.lower()
            if suffix == ".csv":
                questions = self._load_from_csv(self.dataset_path)
            elif suffix == ".json":
                questions = self._load_from_json(self.dataset_path)
            else:
                questions = []
            if questions:
                return stratified_sample(questions, self.max_questions)
        self.used_synthetic_fallback = True
        return stratified_sample(self._synthetic_questions(), self.max_questions)

    def _load_from_csv(self, path: Path) -> list[dict[str, Any]]:
        questions: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                problem = str(row.get("Problem") or "").strip()
                answer = str(row.get("Short Answer") or "").strip()
                if not problem or not answer:
                    continue
                questions.append(
                    {
                        "id": str(row.get("Problem ID") or f"imo_{idx}"),
                        "question_text": problem,
                        "correct_answer": answer,
                        "category": str(row.get("Category") or "").strip(),
                        "subcategory": str(row.get("Subcategory") or "").strip(),
                        "source": str(row.get("Source") or "").strip(),
                    }
                )
        return questions

    def _load_from_json(self, path: Path) -> list[dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        records = payload.get("problems", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            return []
        questions: list[dict[str, Any]] = []
        for idx, row in enumerate(records):
            if not isinstance(row, dict):
                continue
            problem = str(row.get("problem_text") or row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if not problem or not answer:
                continue
            questions.append(
                {
                    "id": str(row.get("id") or f"imo_{idx}"),
                    "question_text": problem,
                    "correct_answer": answer,
                    "category": str(row.get("category") or "IMO").strip(),
                    "subcategory": str(row.get("subcategory") or "").strip(),
                    "source": str(row.get("source") or "").strip(),
                }
            )
        return questions

    def _synthetic_questions(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "imo_synthetic_1",
                "question_text": "Evaluate derivative of cos(x) at x=0.",
                "correct_answer": "0",
                "category": "synthetic",
                "subcategory": "calculus",
                "source": "synthetic",
            },
            {
                "id": "imo_synthetic_2",
                "question_text": "Evaluate derivative of x^4 at x=1.",
                "correct_answer": "4",
                "category": "synthetic",
                "subcategory": "calculus",
                "source": "synthetic",
            },
        ]

    def run_benchmark(
        self,
        use_enriched: bool = True,
        *,
        progress_cb: Callable[[dict[str, Any]], None] | None = None,
        progress_every: int | None = None,
        row_cb: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
        start_index: int = 0,
        initial_correct: int = 0,
    ) -> dict[str, Any]:
        self.results = []
        total = len(self._bench.problems)
        resume_index = max(0, min(int(start_index), total))
        correct = max(0, int(initial_correct))
        step = max(1, int(progress_every or 25))

        for index, problem in enumerate(self._bench.problems[resume_index:], start=resume_index + 1):
            result = self._bench._solve_problem(problem=problem, use_enriched=use_enriched)
            self.results.append(
                {
                    "question_id": result["problem_id"],
                    "question_text": problem["problem_text"],
                    "predicted_answer": result.get("predicted_answer"),
                    "correct_answer": problem["answer"],
                    "correct": bool(result.get("correct", False)),
                    "reasoning_trace": list(result.get("reasoning_trace", [])),
                    "route": result.get("route"),
                    "solver": str(result.get("solver", "")),
                    "runtime": str(result.get("runtime", "")),
                    "gpu_execution": bool(result.get("gpu_execution", False)),
                    "program_id": str(result.get("program_id", "")),
                    "task_result": result.get("task_result", {}),
                    "category": next(
                        (row.get("category", "") for row in self.questions if row["id"] == problem["id"]),
                        "",
                    ),
                    "subcategory": next(
                        (row.get("subcategory", "") for row in self.questions if row["id"] == problem["id"]),
                        "",
                    ),
                }
            )
            if result.get("correct"):
                correct += 1
            if row_cb is not None:
                row_cb(
                    {
                        "id": problem["id"],
                        "question_text": problem["problem_text"],
                        "correct_answer": problem["answer"],
                    },
                    self.results[-1],
                )
            if progress_cb and (index % step == 0 or index == total):
                progress_cb(
                    {
                        "completed": index,
                        "total": total,
                        "correct": correct,
                        "benchmark": "imo",
                        "category": self.results[-1].get("category"),
                    }
                )
        accuracy = (correct / total) if total else 0.0
        return {
            "benchmark": "IMO Bench",
            "total_questions": total,
            "correct": correct,
            "accuracy": accuracy,
            "use_enriched": use_enriched,
            "dataset_path": self.dataset_file,
            "dataset_source": self.dataset_source,
            "synthetic_fallback": self.synthetic_fallback,
            "results": self.results,
        }

    def save_results(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": "IMO Bench",
            "total": len(self.results),
            "correct": sum(1 for row in self.results if row.get("correct")),
            "accuracy": (
                sum(1 for row in self.results if row.get("correct")) / len(self.results)
                if self.results
                else 0.0
            ),
            "results": self.results,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
