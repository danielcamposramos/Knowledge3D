"""GSM8K benchmark integration routed through the sovereign Knowledgeverse math path."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


class GSM8KBenchmark:
    """Run GSM8K grade-school math questions through `Knowledgeverse.execute_task()`."""

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
        self.query_scope_galaxies = self._normalize_query_scope(query_scope_galaxies)
        self.runtime_seed_knowledge = bool(runtime_seed_knowledge)
        self.used_synthetic_fallback = False
        self.questions = self._load_questions()
        self.results: list[dict[str, Any]] = []

        self.dataset_source = "GSM8K"
        self.dataset_file = str(self.dataset_path) if self.dataset_path.exists() else "not_found"
        self.synthetic_fallback = self.used_synthetic_fallback

    def _resolve_dataset_path(self, dataset_path: str | Path | None) -> Path:
        if dataset_path is not None:
            return Path(dataset_path)
        candidates = [
            Path("/K3D/K3D_llama_cpp/datasets/GSM8K/grade_school_math/data/test.jsonl"),
            Path("/K3D/K3D_llama_cpp/datasets/grade_school_math/data/test.jsonl"),
            Path("../K3D_llama_cpp/datasets/GSM8K/grade_school_math/data/test.jsonl"),
            Path("/K3D/Knowledge3D.local/datasets/GSM8K/grade_school_math/data/test.jsonl"),
            Path("../Knowledge3D.local/datasets/GSM8K/grade_school_math/data/test.jsonl"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return Path("")

    def _load_questions(self) -> list[dict[str, Any]]:
        if not self.dataset_path or not self.dataset_path.exists():
            self.used_synthetic_fallback = True
            return self._synthetic_questions()
        questions: list[dict[str, Any]] = []
        with self.dataset_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if self.max_questions is not None and len(questions) >= int(self.max_questions):
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                question = str(payload.get("question") or "").strip()
                answer = self._extract_answer(payload.get("answer"))
                if not question or answer is None:
                    continue
                questions.append(
                    {
                        "id": f"gsm8k_{idx}",
                        "question_text": question,
                        "correct_answer": answer,
                    }
                )
        if questions:
            return questions
        self.used_synthetic_fallback = True
        return self._synthetic_questions()

    def _synthetic_questions(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "gsm8k_synthetic_0",
                "question_text": (
                    "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and "
                    "bakes muffins for her friends every day with four. She sells the remainder at the "
                    "farmers' market daily for $2 per fresh duck egg. How much in dollars does she make "
                    "every day at the farmers' market?"
                ),
                "correct_answer": "18",
            }
        ]

    def run_benchmark(
        self,
        use_enriched: bool = True,
        *,
        progress_cb: Callable[[dict[str, Any]], None] | None = None,
        progress_every: int | None = None,
    ) -> dict[str, Any]:
        self.results = []
        correct = 0
        total = len(self.questions)
        step = max(1, int(progress_every or 25))
        start = time.monotonic()
        for index, question in enumerate(self.questions, start=1):
            result = self._solve_question(question=question, use_enriched=use_enriched)
            self.results.append(result)
            if result["correct"]:
                correct += 1
            if progress_cb and (index % step == 0 or index == total):
                progress_cb(
                    {
                        "completed": index,
                        "total": total,
                        "correct": correct,
                        "elapsed_s": time.monotonic() - start,
                        "benchmark": "gsm8k",
                    }
                )
        return {
            "benchmark": "GSM8K",
            "total_questions": total,
            "correct": correct,
            "accuracy": (correct / total) if total else 0.0,
            "use_enriched": use_enriched,
            "dataset_path": self.dataset_file,
            "dataset_source": self.dataset_source,
            "synthetic_fallback": self.synthetic_fallback,
            "results": self.results,
        }

    def _solve_question(self, *, question: dict[str, Any], use_enriched: bool) -> dict[str, Any]:
        if use_enriched and self.runtime_seed_knowledge:
            self._seed_math_knowledge(question)
        route = self._apply_query_scope(
            {
                "specialist": "math",
                "domain_hint": "math",
                "galaxy_names": list(Knowledgeverse.GPU_MATH_TARGET_GALAXIES),
            }
        )
        task_result = self.kv.execute_task(
            task={
                "type": "MATH_TASK",
                "task_id": question["id"],
                "query": question["question_text"],
                "question": question["question_text"],
                "competition": "GSM8K",
                "expected_answer": question["correct_answer"],
            },
            route=route,
            specialist="math",
            domain_hint="math",
            use_enriched=use_enriched,
        )
        predicted = task_result.get("predicted_answer", task_result.get("result"))
        correct = self._answers_match(predicted, question["correct_answer"])
        return {
            "question_id": question["id"],
            "question_text": question["question_text"],
            "predicted_answer": predicted,
            "correct_answer": question["correct_answer"],
            "correct": correct,
            "reasoning_trace": list(task_result.get("reasoning_trace", [])),
            "route": task_result.get("route", route),
            "solver": str(task_result.get("solver", "")),
            "runtime": str(task_result.get("runtime", "")),
            "gpu_execution": bool(task_result.get("gpu_execution", False)),
            "program_id": str(task_result.get("program_id", "")),
            "task_result": task_result,
        }

    @staticmethod
    def _extract_answer(raw_answer: Any) -> str | None:
        text = str(raw_answer or "").strip()
        if not text:
            return None
        match = re.search(r"####\s*([^\n]+)", text)
        if match:
            return match.group(1).strip().replace(",", "")
        tail = text.splitlines()[-1].strip()
        return tail.replace(",", "") or None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            try:
                cleaned = str(value).strip().replace(",", "").replace("\\", "")
                if cleaned.endswith("%"):
                    cleaned = cleaned[:-1]
                if cleaned.startswith("$"):
                    cleaned = cleaned[1:]
                return float(cleaned)
            except Exception:
                return None

    def _answers_match(self, predicted: Any, expected: Any) -> bool:
        pred = self._to_float(predicted)
        exp = self._to_float(expected)
        if pred is not None and exp is not None:
            return abs(pred - exp) <= 1e-3
        return str(predicted).strip().lower() == str(expected).strip().lower()

    def _seed_math_knowledge(self, question: dict[str, Any]) -> None:
        self.kv.galaxy_manager.add_entry(
            "Math",
            {
                "domain": "math",
                "task_id": question["id"],
                "kind": "gsm8k_word_problem_anchor",
            },
        )

    def _normalize_query_scope(self, value: str | list[str] | None) -> list[str] | None:
        if isinstance(value, list):
            raw = [str(item).strip() for item in value]
        elif isinstance(value, str):
            raw = [segment.strip() for segment in value.split(",")]
        else:
            raw = []
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out or None

    def _apply_query_scope(self, route: dict[str, Any]) -> dict[str, Any]:
        if not self.query_scope_galaxies:
            return route
        route_names = [str(name) for name in route.get("galaxy_names") or [] if str(name).strip()]
        if not route_names:
            route["galaxy_names"] = list(self.query_scope_galaxies)
            return route
        scope_keys = {name.lower() for name in self.query_scope_galaxies}
        filtered = [name for name in route_names if name.lower() in scope_keys]
        route["galaxy_names"] = filtered if filtered else list(self.query_scope_galaxies)
        return route

    def save_results(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": "GSM8K",
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
