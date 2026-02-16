"""MMLU (Massive Multitask Language Understanding) benchmark integration for K3D."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.trm_navigator import TRMNavigator


class MMLUBenchmark:
    """
    MMLU benchmark: 14,000+ multiple-choice questions across 57 subjects.

    Replaces synthetic LHE with established SOTA-comparable benchmark.
    Dataset: https://github.com/hendrycks/test (MMLU paper)
    """

    def __init__(
        self,
        knowledgeverse: Knowledgeverse | None = None,
        dataset_path: str | Path | None = None,
        max_questions: int | None = None,
        query_scope_galaxies: str | list[str] | None = None,
        subjects: str | list[str] = "all",
        split: str = "test",
        runtime_seed_knowledge: bool = False,
    ):
        self.kv = knowledgeverse or Knowledgeverse()
        self.dataset_path = self._resolve_dataset_path(dataset_path)
        self.max_questions = max_questions
        self.query_scope_galaxies = self._normalize_query_scope(query_scope_galaxies)
        self.subjects = self._parse_subjects(subjects)
        self.split = str(split).strip().lower()
        self.runtime_seed_knowledge = bool(runtime_seed_knowledge)
        self.used_synthetic_fallback = False
        self.questions = self._load_questions()
        self.results: list[dict[str, Any]] = []

        # Metadata for integrity validation
        self.dataset_source = "MMLU"
        self.dataset_file = str(self.dataset_path) if self.dataset_path.exists() else "not_found"
        self.synthetic_fallback = self.used_synthetic_fallback

    def _resolve_dataset_path(self, dataset_path: str | Path | None) -> Path:
        """Resolve MMLU dataset path from multiple candidate locations."""
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

        return Path("")  # Will trigger fallback

    def _parse_subjects(self, subjects: str | list[str]) -> list[str]:
        """Parse subject filter (all, or specific subject names)."""
        if isinstance(subjects, list):
            return subjects

        subjects_str = str(subjects).strip().lower()
        if subjects_str == "all":
            return []  # Empty means all subjects

        # Parse comma-separated subjects
        return [s.strip() for s in subjects_str.split(",") if s.strip()]

    def _load_questions(self) -> list[dict[str, Any]]:
        """Load MMLU questions from CSV files."""
        if not self.dataset_path or not self.dataset_path.exists():
            self.used_synthetic_fallback = True
            return self._synthetic_questions()

        split_dir = self.dataset_path / self.split
        if not split_dir.exists():
            self.used_synthetic_fallback = True
            return self._synthetic_questions()

        questions: list[dict[str, Any]] = []
        csv_files = sorted(split_dir.glob("*_test.csv"))

        for csv_file in csv_files:
            # Extract subject name (e.g., "abstract_algebra" from "abstract_algebra_test.csv")
            subject = csv_file.stem.replace("_test", "")

            # Filter by subjects if specified
            if self.subjects and subject not in self.subjects:
                continue

            # Load questions from CSV
            try:
                with csv_file.open("r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for idx, row in enumerate(reader):
                        if len(row) < 6:
                            continue

                        question_text = row[0].strip()
                        options = [row[1].strip(), row[2].strip(), row[3].strip(), row[4].strip()]
                        correct_letter = row[5].strip().upper()

                        # Convert letter to option text
                        letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
                        if correct_letter not in letter_to_idx:
                            continue

                        correct_answer = options[letter_to_idx[correct_letter]]

                        questions.append({
                            "id": f"mmlu_{subject}_{idx}",
                            "subject": subject,
                            "domain": self._subject_to_domain(subject),
                            "question_text": question_text,
                            "options": options,
                            "correct_answer": correct_answer,
                            "correct_letter": correct_letter,
                        })

                        # Early exit if max reached
                        if self.max_questions and len(questions) >= self.max_questions:
                            return questions

            except Exception as e:
                # Skip malformed files
                continue

        if questions:
            return questions

        # Fallback if no questions loaded
        self.used_synthetic_fallback = True
        return self._synthetic_questions()

    def _subject_to_domain(self, subject: str) -> str:
        """Map MMLU subject to K3D domain for telemetry."""
        # Categorize 57 subjects into broad domains
        stem_subjects = {
            "abstract_algebra", "college_mathematics", "elementary_mathematics",
            "high_school_mathematics", "college_physics", "high_school_physics",
            "astronomy", "college_chemistry", "high_school_chemistry",
            "college_biology", "high_school_biology", "college_computer_science",
            "computer_security", "machine_learning", "electrical_engineering"
        }

        humanities_subjects = {
            "formal_logic", "philosophy", "moral_scenarios", "moral_disputes",
            "prehistory", "world_religions", "jurisprudence", "professional_law"
        }

        social_subjects = {
            "econometrics", "high_school_microeconomics", "high_school_macroeconomics",
            "professional_accounting", "business_ethics", "marketing",
            "high_school_geography", "high_school_government_and_politics",
            "high_school_us_history", "high_school_world_history", "sociology"
        }

        if subject in stem_subjects:
            return "stem"
        elif subject in humanities_subjects:
            return "humanities"
        elif subject in social_subjects:
            return "social_sciences"
        else:
            return "other"

    def _synthetic_questions(self) -> list[dict[str, Any]]:
        """
        Synthetic fallback questions (for development only).

        CRITICAL: This should NEVER be used for paper claims!
        Only for keeping benchmark infrastructure runnable.
        """
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
                "options": [
                    "Some A are not C",
                    "All A are C",
                    "No B are C",
                    "All C are A",
                ],
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
            {
                "id": "mmlu_synthetic_cs_1",
                "subject": "computer_security",
                "domain": "stem",
                "question_text": "Which of the following is NOT a symmetric encryption algorithm?",
                "options": ["AES", "DES", "RSA", "Blowfish"],
                "correct_answer": "RSA",
                "correct_letter": "C",
            },
        ]

    def run_benchmark(self, use_enriched: bool = True) -> dict[str, Any]:
        """Run MMLU benchmark with TRM Navigator."""
        self.results = []
        correct = 0
        navigator = TRMNavigator(knowledgeverse=self.kv)

        for question in self.questions:
            # Use TRM to answer multiple-choice question
            # (Similar to current LHE benchmark logic)
            predicted_answer = navigator.answer_multiple_choice(
                question_text=question["question_text"],
                options=question["options"],
                use_enriched=use_enriched,
                galaxy_scope=self.query_scope_galaxies,
            )

            is_correct = predicted_answer == question["correct_answer"]
            if is_correct:
                correct += 1

            self.results.append({
                "id": question["id"],
                "subject": question["subject"],
                "domain": question["domain"],
                "question_text": question["question_text"],
                "options": question["options"],
                "correct_answer": question["correct_answer"],
                "predicted_answer": predicted_answer,
                "correct": is_correct,
            })

        total = len(self.questions)
        accuracy = correct / total if total > 0 else 0.0

        # Domain-level accuracy breakdown
        domain_stats = self._compute_domain_stats()

        return {
            "benchmark": "MMLU",
            "total_questions": total,
            "correct": correct,
            "accuracy": accuracy,
            "use_enriched": use_enriched,
            "dataset_path": self.dataset_file,
            "dataset_source": self.dataset_source,
            "synthetic_fallback": self.synthetic_fallback,
            "subjects_tested": len(set(q["subject"] for q in self.questions)),
            "domain_breakdown": domain_stats,
            "results": self.results,
        }

    def _compute_domain_stats(self) -> dict[str, dict[str, Any]]:
        """Compute per-domain accuracy statistics."""
        domain_stats: dict[str, dict[str, Any]] = {}

        for result in self.results:
            domain = result["domain"]
            if domain not in domain_stats:
                domain_stats[domain] = {"correct": 0, "total": 0}

            domain_stats[domain]["total"] += 1
            if result["correct"]:
                domain_stats[domain]["correct"] += 1

        # Compute accuracy for each domain
        for domain, stats in domain_stats.items():
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

        return domain_stats

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
