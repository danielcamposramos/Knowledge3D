"""MMLU benchmark integration routed through the sovereign Knowledgeverse query path."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

from benchmarks.sampling import stratified_sample
from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.tablet.wine.question_wine import mmlu_question_envelope


class MMLUBenchmark:
    """Run MMLU multiple-choice questions through `Knowledgeverse.execute_task()`."""

    def __init__(
        self,
        knowledgeverse: Knowledgeverse | None = None,
        dataset_path: str | Path | None = None,
        max_questions: int | None = None,
        query_scope_galaxies: str | list[str] | None = None,
        subjects: str | list[str] = "all",
        split: str = "test",
        runtime_seed_knowledge: bool = False,
        tablet_boundary: HeadlessTabletMPC | None = None,
    ) -> None:
        self.kv = knowledgeverse or Knowledgeverse()
        self.dataset_path = self._resolve_dataset_path(dataset_path)
        self.max_questions = max_questions
        self.query_scope_galaxies = self._normalize_query_scope(query_scope_galaxies)
        self.subjects = self._parse_subjects(subjects)
        self.split = str(split).strip().lower()
        self.runtime_seed_knowledge = bool(runtime_seed_knowledge)
        self.tablet_boundary = tablet_boundary
        self.used_synthetic_fallback = False
        self.questions = self._load_questions()
        self.results: list[dict[str, Any]] = []

        self.dataset_source = "MMLU"
        self.dataset_file = str(self.dataset_path) if self.dataset_path.exists() else "not_found"
        self.synthetic_fallback = self.used_synthetic_fallback

    def _resolve_dataset_path(self, dataset_path: str | Path | None) -> Path:
        if dataset_path is not None:
            return Path(dataset_path)
        candidates = [
            Path("/K3D/K3D_llama_cpp/datasets/MMLU/data"),
            Path("../K3D_llama_cpp/datasets/MMLU/data"),
            Path("/K3D/Knowledge3D.local/datasets/MMLU/data"),
            Path("../Knowledge3D.local/datasets/MMLU/data"),
            Path("data/MMLU"),
        ]
        for candidate in candidates:
            if candidate.exists() and (candidate / "test").exists():
                return candidate
        return Path("")

    def _parse_subjects(self, subjects: str | list[str]) -> list[str]:
        if isinstance(subjects, list):
            return [str(item).strip().lower() for item in subjects if str(item).strip()]
        lowered = str(subjects).strip().lower()
        if lowered == "all":
            return []
        return [item.strip() for item in lowered.split(",") if item.strip()]

    def _load_questions(self) -> list[dict[str, Any]]:
        if not self.dataset_path or not self.dataset_path.exists():
            self.used_synthetic_fallback = True
            return self._synthetic_questions()

        split_dir = self.dataset_path / self.split
        if not split_dir.exists():
            self.used_synthetic_fallback = True
            return self._synthetic_questions()

        questions: list[dict[str, Any]] = []
        for csv_file in sorted(split_dir.glob("*_test.csv")):
            subject = csv_file.stem.replace("_test", "").strip().lower()
            if self.subjects and subject not in self.subjects:
                continue
            try:
                with csv_file.open("r", encoding="utf-8") as handle:
                    reader = csv.reader(handle)
                    for idx, row in enumerate(reader):
                        if len(row) < 6:
                            continue
                        question_text = row[0].strip()
                        if not question_text:
                            continue
                        options = [row[1].strip(), row[2].strip(), row[3].strip(), row[4].strip()]
                        correct_letter = row[5].strip().upper()
                        if correct_letter not in {"A", "B", "C", "D"}:
                            continue
                        correct_answer = options[ord(correct_letter) - ord("A")]
                        questions.append(
                            {
                                "id": f"mmlu_{subject}_{idx}",
                                "subject": subject,
                                "domain": self._subject_to_domain(subject),
                                "question_text": question_text,
                                "options": options,
                                "correct_answer": correct_answer,
                                "correct_letter": correct_letter,
                            }
                        )
            except Exception:
                continue

        if questions:
            return stratified_sample(questions, self.max_questions)
        self.used_synthetic_fallback = True
        return stratified_sample(self._synthetic_questions(), self.max_questions)

    def _subject_to_domain(self, subject: str) -> str:
        stem_subjects = {
            "abstract_algebra",
            "college_mathematics",
            "elementary_mathematics",
            "high_school_mathematics",
            "college_physics",
            "high_school_physics",
            "astronomy",
            "college_chemistry",
            "high_school_chemistry",
            "college_biology",
            "high_school_biology",
            "college_computer_science",
            "computer_security",
            "machine_learning",
            "electrical_engineering",
        }
        humanities_subjects = {
            "formal_logic",
            "philosophy",
            "moral_scenarios",
            "moral_disputes",
            "prehistory",
            "world_religions",
            "jurisprudence",
            "professional_law",
        }
        social_subjects = {
            "econometrics",
            "high_school_microeconomics",
            "high_school_macroeconomics",
            "professional_accounting",
            "business_ethics",
            "marketing",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_us_history",
            "high_school_world_history",
            "sociology",
        }
        if subject in stem_subjects:
            return "stem"
        if subject in humanities_subjects:
            return "humanities"
        if subject in social_subjects:
            return "social_sciences"
        return "other"

    def _synthetic_questions(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "mmlu_synthetic_math_1",
                "subject": "elementary_mathematics",
                "domain": "stem",
                "question_text": "What is 7 * (3 + 2)?",
                "options": ["35", "30", "42", "28"],
                "correct_answer": "35",
                "correct_letter": "A",
            },
            {
                "id": "mmlu_synthetic_logic_1",
                "subject": "formal_logic",
                "domain": "humanities",
                "question_text": "If all A are B and all B are C, which statement is true?",
                "options": ["Some A are not C", "All A are C", "No B are C", "All C are A"],
                "correct_answer": "All A are C",
                "correct_letter": "B",
            },
            {
                "id": "mmlu_synthetic_physics_1",
                "subject": "college_physics",
                "domain": "stem",
                "question_text": "An object at rest remains at rest unless acted on by which quantity?",
                "options": ["Force", "Mass", "Time", "Temperature"],
                "correct_answer": "Force",
                "correct_letter": "A",
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
        total = len(self.questions)
        resume_index = max(0, min(int(start_index), total))
        correct = max(0, int(initial_correct))
        total = len(self.questions)
        step = max(1, int(progress_every or 100))
        start = time.monotonic()
        for index, question in enumerate(self.questions[resume_index:], start=resume_index + 1):
            row_start = time.monotonic()
            result = self._solve_question(question=question, use_enriched=use_enriched)
            result["elapsed_s"] = round(max(0.0, time.monotonic() - row_start), 3)
            self.results.append(result)
            if result["correct"]:
                correct += 1
            if row_cb is not None:
                row_cb(question, result)
            if progress_cb and (index % step == 0 or index == total):
                progress_cb(
                    {
                        "completed": index,
                        "total": total,
                        "correct": correct,
                        "elapsed_s": time.monotonic() - start,
                        "benchmark": "mmlu",
                        "subject": question.get("subject"),
                    }
                )

        return {
            "benchmark": "MMLU",
            "total_questions": total,
            "correct": correct,
            "accuracy": (correct / total) if total else 0.0,
            "use_enriched": use_enriched,
            "dataset_path": self.dataset_file,
            "dataset_source": self.dataset_source,
            "synthetic_fallback": self.synthetic_fallback,
            "subjects_tested": len({q["subject"] for q in self.questions}),
            "domain_breakdown": self._compute_domain_stats(),
            "results": self.results,
        }

    def _solve_question(self, *, question: dict[str, Any], use_enriched: bool) -> dict[str, Any]:
        if self.tablet_boundary is not None:
            return self._solve_question_via_tablet(question=question, use_enriched=use_enriched)
        if use_enriched and self.runtime_seed_knowledge:
            self._seed_subject_knowledge(question)
        route = self._apply_query_scope(
            {
                "specialist": "chat",
                "domain_hint": question["subject"],
                "galaxy_names": list(Knowledgeverse.GPU_QUESTION_TARGET_GALAXIES),
            }
        )
        task_result = self.kv.execute_task(
            task={
                "task_id": question["id"],
                "query": question["question_text"],
                "prompt": question["question_text"],
                "messages": [{"role": "user", "content": question["question_text"]}],
                "options": list(question["options"]),
                "subject": question["subject"],
                "expected_answer": question["correct_answer"],
            },
            route=route,
            specialist="chat",
            domain_hint=question["subject"],
            use_enriched=use_enriched,
        )
        predicted = self._normalize_option_prediction(
            raw_answer=task_result.get("response", task_result.get("answer", task_result.get("result"))),
            options=question["options"],
        )
        correct = predicted == question["correct_answer"]
        return {
            "id": question["id"],
            "subject": question["subject"],
            "domain": question["domain"],
            "question_text": question["question_text"],
            "options": list(question["options"]),
            "correct_answer": question["correct_answer"],
            "predicted_answer": predicted,
            "correct": correct,
            "reasoning_trace": list(task_result.get("reasoning_trace", [])),
            "route": task_result.get("route", route),
            "solver": str(task_result.get("solver", "")),
            "runtime": str(task_result.get("runtime", "")),
            "gpu_execution": bool(task_result.get("gpu_execution", False)),
            "program_id": str(task_result.get("program_id", "")),
            "task_result": task_result,
        }

    def _solve_question_via_tablet(
        self,
        *,
        question: dict[str, Any],
        use_enriched: bool,
    ) -> dict[str, Any]:
        envelope = mmlu_question_envelope(
            task_id=str(question["id"]),
            question=str(question["question_text"]),
            options=list(question["options"]),
            subject=str(question["subject"]),
            expected_answer=str(question["correct_answer"]),
        )
        tablet_result = self.tablet_boundary.submit(envelope, use_enriched=use_enriched)
        emitted = dict(tablet_result["emitted"])
        route = emitted.get("route", {})
        predicted = str(emitted.get("answer_choice") or emitted.get("answer_text") or "").strip()
        correct = bool(emitted.get("correct", False))
        return {
            "id": question["id"],
            "subject": question["subject"],
            "domain": question["domain"],
            "question_text": question["question_text"],
            "options": list(question["options"]),
            "correct_answer": question["correct_answer"],
            "predicted_answer": predicted,
            "correct": correct,
            "reasoning_trace": emitted.get("task_result", {}).get("reasoning_trace", []),
            "route": route,
            "solver": str(emitted.get("task_result", {}).get("solver", "tablet_boundary")),
            "runtime": str(emitted.get("task_result", {}).get("runtime", "")),
            "gpu_execution": bool(emitted.get("task_result", {}).get("gpu_execution", False)),
            "program_id": str(emitted.get("task_result", {}).get("program_id", "")),
            "task_result": emitted.get("task_result", {}),
            "tablet_contract": tablet_result["tablet_contract"],
        }

    def _compute_domain_stats(self) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for result in self.results:
            domain = str(result["domain"])
            bucket = stats.setdefault(domain, {"correct": 0, "total": 0})
            bucket["total"] += 1
            if result["correct"]:
                bucket["correct"] += 1
        for bucket in stats.values():
            total = int(bucket["total"])
            bucket["accuracy"] = (int(bucket["correct"]) / total) if total else 0.0
        return stats

    def _seed_subject_knowledge(self, question: dict[str, Any]) -> None:
        self.kv.galaxy_manager.add_entry(
            "Grammar",
            {
                "domain": "reasoning",
                "subject": question["subject"],
                "kind": "mmlu_subject_anchor",
            },
        )

    @staticmethod
    def _normalize_option_prediction(raw_answer: Any, options: list[str]) -> str:
        text = str(raw_answer or "").strip()
        if not text:
            return ""
        lowered = f" {text.lower()} "
        for option in options:
            candidate = str(option).strip()
            if not candidate:
                continue
            if f" {candidate.lower()} " in lowered or lowered.strip() == candidate.lower():
                return candidate
        if text in options:
            return text
        return text

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
            "benchmark": "MMLU",
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
