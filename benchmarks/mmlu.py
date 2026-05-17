"""Multi-subject multiple-choice question benchmark (MMLU corpus, sovereign tablet path)."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Callable

from benchmarks.sampling import stratified_sample
from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.tablet.wine.question_wine import (
    build_question_session_tape,
    mmlu_question_envelope,
)


class MMLUBenchmark:
    """Run MMLU multiple-choice questions through the sovereign tablet boundary."""

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
        self._ensure_tablet_boundary()
        self.used_synthetic_fallback = False
        self.questions = self._load_questions()
        self.results: list[dict[str, Any]] = []
        self.last_summary: dict[str, Any] = {}

        self.dataset_source = "MMLU"
        self.dataset_file = str(self.dataset_path) if self.dataset_path.exists() else "not_found"
        self.synthetic_fallback = self.used_synthetic_fallback

    def _ensure_tablet_boundary(self) -> HeadlessTabletMPC:
        if self.tablet_boundary is None:
            self.tablet_boundary = HeadlessTabletMPC(
                knowledgeverse=self.kv,
                storage_root=getattr(self.kv, "storage_root", None),
            )
        return self.tablet_boundary

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
            raise FileNotFoundError("mmlu_dataset_missing: MMLU runtime requires the real dataset at the canonical roots")

        split_dir = self.dataset_path / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"mmlu_split_missing: expected split '{self.split}' under {self.dataset_path}")

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
        raise FileNotFoundError(f"mmlu_dataset_empty: no valid MMLU questions found under {split_dir}")

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
        boundary = self._ensure_tablet_boundary()
        tape = build_question_session_tape(
            session_id=f"question_{int(time.time() * 1000)}",
            suite_name="question",
            rows=self.questions[resume_index:],
            use_enriched=use_enriched,
        )
        tablet_rows = list(boundary.run_tape_session(tape)["results"])
        for offset, question in enumerate(self.questions[resume_index:]):
            index = resume_index + offset + 1
            result = self._question_result_from_tablet(question=question, tablet_result=tablet_rows[offset])
            task_result = dict(result.get("task_result") or {})
            result["elapsed_s"] = round(
                max(0.0, float(task_result.get("trm_latency_us", 0.0) or 0.0) / 1_000_000.0),
                3,
            )
            result["elapsed_ms"] = round(float(result["elapsed_s"]) * 1000.0, 3)
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
                        "surface_kind": "QUESTION",
                        "subject": question.get("subject"),
                    }
                )

        summary = {
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
            "timing_summary": self._timing_summary(),
            "answer_format_counts": self._answer_format_counts(),
            "results": self.results,
        }
        self.last_summary = dict(summary)
        return summary

    def _solve_question(self, *, question: dict[str, Any], use_enriched: bool) -> dict[str, Any]:
        boundary = self._ensure_tablet_boundary()
        envelope = mmlu_question_envelope(
            task_id=str(question["id"]),
            question=str(question["question_text"]),
            options=list(question["options"]),
            subject=str(question["subject"]),
            expected_answer=str(question["correct_answer"]),
        )
        tablet_result = boundary.submit(envelope, use_enriched=use_enriched)
        return self._question_result_from_tablet(question=question, tablet_result=tablet_result)

    def _question_result_from_tablet(
        self,
        *,
        question: dict[str, Any],
        tablet_result: dict[str, Any],
    ) -> dict[str, Any]:
        emitted = dict(tablet_result["emitted"])
        route = emitted.get("route", {})
        predicted = str(emitted.get("answer_choice") or emitted.get("answer_text") or "").strip()
        correct = bool(emitted.get("correct", False))
        raw_answer = str(
            emitted.get("answer_choice")
            or emitted.get("answer_text")
            or emitted.get("task_result", {}).get("response")
            or emitted.get("task_result", {}).get("answer")
            or ""
        ).strip()
        return {
            "id": question["id"],
            "subject": question["subject"],
            "domain": question["domain"],
            "question_text": question["question_text"],
            "options": list(question["options"]),
            "correct_answer": question["correct_answer"],
            "raw_answer": raw_answer,
            "predicted_answer": predicted,
            "predicted_matches_option_text": bool(predicted and predicted in question["options"]),
            "answer_format": self._classify_answer_format(
                raw_answer=raw_answer,
                predicted_answer=predicted,
                options=question["options"],
            ),
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
        return {
            "count": len(timings),
            "avg_task_s": round(avg_s, 6),
            "avg_task_ms": round(avg_s * 1000.0, 3),
            "median_task_s": round(float(median(timings)), 6),
            "median_task_ms": round(float(median(timings)) * 1000.0, 3),
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
    def _classify_answer_format(
        *,
        raw_answer: Any,
        predicted_answer: Any,
        options: list[str],
    ) -> str:
        raw_text = str(raw_answer or "").strip()
        predicted_text = str(predicted_answer or "").strip()
        option_set = {str(option).strip() for option in options if str(option).strip()}
        if not raw_text:
            return "empty"
        if raw_text.upper() in {"A", "B", "C", "D"} and raw_text not in option_set:
            return "single_letter_code"
        if raw_text in option_set:
            return "option_text_exact"
        if predicted_text and predicted_text in option_set:
            return "normalized_to_option_text"
        return "free_text"

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

    def save_results(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.last_summary) if self.last_summary else {
            "benchmark": "MMLU",
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
    parser = argparse.ArgumentParser(description="Run the sovereign MMLU benchmark through the tablet boundary.")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--query-scope-galaxies", type=str, default=None)
    parser.add_argument("--subjects", type=str, default="all")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--summary-output", type=str, default=None)
    parser.add_argument("--use-enriched", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    benchmark = MMLUBenchmark(
        dataset_path=args.dataset_path,
        max_questions=args.max_tasks,
        query_scope_galaxies=args.query_scope_galaxies,
        subjects=args.subjects,
        split=args.split,
    )
    summary = benchmark.run_benchmark(use_enriched=bool(args.use_enriched))
    if args.summary_output:
        benchmark.save_results(args.summary_output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
