"""
Convert text-based math problems into procedural RPN form.

This is an ingestion-layer utility (not hot path).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

MATH_DATASETS = {
    "gsm8k": Path("/K3D/K3D_llama_cpp/datasets/GSM8K/grade_school_math/data"),
    "math": Path("/K3D/K3D_llama_cpp/datasets/math"),
    "mmlu": Path("/K3D/K3D_llama_cpp/datasets/MMLU"),
    "omni_math": Path("/K3D/K3D_llama_cpp/datasets/Omni-MATH"),
    "amc_aime": Path("/K3D/K3D_llama_cpp/datasets/AMC-AIME"),
}


class MathProceduralizer:
    """
    Transform math problems into procedural RPN representations.
    """

    def __init__(self):
        self._operation_patterns = {
            r"\b(add|plus|sum|more|together)\b": "ADD",
            r"\b(subtract|minus|less|fewer|difference)\b": "SUB",
            r"\b(multiply|times|product|of)\b": "MUL",
            r"\b(divide|split|share|per|ratio)\b": "DIV",
            r"\b(square|squared)\b": "SQR",
            r"\b(root|sqrt)\b": "SQRT",
            r"\b(power|exponent)\b": "POW",
            r"\b(percent|%)\b": "PCT",
            r"\b(total|altogether|result)\b": "EQ",
        }
        self._number_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|\b\d+/\d+\b")

    def proceduralize_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a math problem to procedural form.
        """
        text = problem.get("question", problem.get("problem", ""))
        solution = problem.get("answer", problem.get("solution", ""))

        quantities = self._extract_quantities(text)
        operations = self._extract_operations(text)

        problem_rpn = self._build_problem_rpn(quantities, operations)
        solution_rpn = self._parse_solution(solution) if solution else []
        answer = self._extract_answer(solution)

        return {
            "problem_id": problem.get("id", problem.get("problem_id", str(hash(text)))),
            "source": problem.get("source", "unknown"),
            "difficulty": problem.get("level", problem.get("difficulty", 5)),
            "problem_rpn": problem_rpn,
            "solution_rpn": solution_rpn,
            "answer": answer,
            "metadata": {
                "original_text": text,
                "original_solution": solution,
                "quantities_found": quantities,
                "operations_implied": operations,
            },
        }

    def _extract_quantities(self, text: str) -> List[float]:
        matches = self._number_pattern.findall(text)
        quantities: List[float] = []
        for m in matches:
            try:
                if "/" in m:
                    num, den = m.split("/")
                    quantities.append(float(num) / float(den))
                else:
                    quantities.append(float(m))
            except ValueError:
                continue
        return quantities

    def _extract_operations(self, text: str) -> List[str]:
        text_lower = text.lower()
        operations: List[str] = []
        for pattern, opcode in self._operation_patterns.items():
            if re.search(pattern, text_lower):
                operations.append(opcode)
        return operations

    def _build_problem_rpn(self, quantities: List[float], operations: List[str]) -> List[str]:
        rpn: List[str] = []
        for q in quantities:
            rpn.append(f"PUSH {q}")
        for op in operations:
            rpn.append(f"HINT {op}")
        rpn.append("SOLVE")
        return rpn

    def _parse_solution(self, solution: str) -> List[str]:
        rpn: List[str] = []
        steps = re.findall(r"=\s*([-+]?\d*\.?\d+)", solution)
        for i, step in enumerate(steps):
            rpn.append(f"STEP_{i}: {step}")

        answer_match = re.search(r"(?:answer|final|result)[:\s]*\\?\$?([-+]?\d*\.?\d+)", solution, re.IGNORECASE)
        if answer_match:
            rpn.append(f"ANSWER: {answer_match.group(1)}")
        return rpn

    def _extract_answer(self, solution: str) -> Any:
        if not solution:
            return None

        gsm8k_match = re.search(r"####\s*([-+]?\d*\.?\d+)", solution)
        if gsm8k_match:
            try:
                return float(gsm8k_match.group(1))
            except ValueError:
                pass

        boxed_match = re.search(r"\\boxed\{([^}]+)\}", solution)
        if boxed_match:
            try:
                return float(boxed_match.group(1))
            except ValueError:
                return boxed_match.group(1)

        numbers = self._number_pattern.findall(solution)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                return numbers[-1]
        return None


class MathDatasetLoader:
    """
    Unified loader for math benchmark datasets.
    """

    def __init__(self, datasets: Optional[List[str]] = None, difficulty_filter: Optional[range] = None, shuffle: bool = True):
        self._datasets = datasets or list(MATH_DATASETS.keys())
        self._difficulty_filter = difficulty_filter
        self._shuffle = shuffle
        self._proceduralizer = MathProceduralizer()
        self._problems: List[Dict[str, Any]] = []
        self._load_datasets()

    def _load_datasets(self) -> None:
        for ds_name in self._datasets:
            path = MATH_DATASETS.get(ds_name)
            if path and path.exists():
                self._load_dataset(ds_name, path)

    def _load_dataset(self, name: str, path: Path) -> None:
        if name == "gsm8k":
            self._load_gsm8k(path)
        elif name == "math":
            self._load_math(path)
        elif name == "mmlu":
            self._load_mmlu(path)
        elif name == "omni_math":
            self._load_omni_math(path)
        elif name == "amc_aime":
            self._load_amc_aime(path)

    def _load_gsm8k(self, path: Path) -> None:
        for split in ["train", "test"]:
            split_file = path / f"{split}.jsonl"
            if split_file.exists():
                with open(split_file, "r") as f:
                    for line in f:
                        problem = json.loads(line)
                        problem["source"] = "gsm8k"
                        problem["split"] = split
                        proc = self._proceduralizer.proceduralize_problem(problem)
                        self._add_if_passes_filter(proc)

    def _load_math(self, path: Path) -> None:
        """Load MATH dataset (competition problems) from JSONL or Level folders."""
        # JSONL format (preferred; HuggingFace download)
        data_dir = path / "data"
        for split_file in ["train.jsonl", "test.jsonl"]:
            jsonl_path = data_dir / split_file
            if jsonl_path.exists():
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        problem = json.loads(line)
                        problem["source"] = "math"
                        level_str = problem.get("level", "Level 3")
                        try:
                            level = int(str(level_str).replace("Level", "").strip())
                        except Exception:
                            level = 3
                        problem["level"] = level * 2
                        proc = self._proceduralizer.proceduralize_problem(problem)
                        self._add_if_passes_filter(proc)
                return  # Loaded JSONL; done

        # Legacy Level*/ directory structure
        for difficulty_dir in path.iterdir():
            if difficulty_dir.is_dir() and difficulty_dir.name.startswith("Level"):
                try:
                    level = int(difficulty_dir.name.replace("Level", "").strip())
                except ValueError:
                    level = 5
                for problem_file in difficulty_dir.glob("*.json"):
                    with open(problem_file, "r") as f:
                        problem = json.load(f)
                    problem["source"] = "math"
                    problem["level"] = level * 2
                    proc = self._proceduralizer.proceduralize_problem(problem)
                    self._add_if_passes_filter(proc)

    def _load_mmlu(self, path: Path) -> None:
        math_subjects = [
            "abstract_algebra",
            "college_mathematics",
            "elementary_mathematics",
            "high_school_mathematics",
            "high_school_statistics",
        ]
        for subject in math_subjects:
            subject_file = path / "data" / "test" / f"{subject}_test.csv"
            if subject_file.exists():
                import csv

                with open(subject_file, "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 5:
                            problem = {
                                "question": row[0],
                                "choices": row[1:5],
                                "answer": row[5] if len(row) > 5 else None,
                                "source": "mmlu",
                                "subject": subject,
                            }
                            proc = self._proceduralizer.proceduralize_problem(problem)
                            self._add_if_passes_filter(proc)

    def _load_omni_math(self, path: Path) -> None:
        jsonl_file = path / "Omni-Math.jsonl"
        if jsonl_file.exists():
            with open(jsonl_file, "r") as f:
                for line in f:
                    problem = json.loads(line)
                    problem["source"] = "omni_math"
                    problem["difficulty"] = problem.get("difficulty", 5)
                    proc = self._proceduralizer.proceduralize_problem(problem)
                    self._add_if_passes_filter(proc)

    def _load_amc_aime(self, path: Path) -> None:
        """Load AMC/AIME competition problems from JSONL files or ZIP."""
        data_dir = path / "data"

        # Try JSONL files first (AIME 2024, AI-MO datasets)
        loaded = 0
        for jsonl_file in data_dir.glob("*.jsonl") if data_dir.exists() else []:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    problem = json.loads(line)
                    # Normalize field names (different datasets use different keys)
                    answer = problem.get("answer", problem.get("Answer", ""))
                    solution = problem.get("solution", problem.get("Solution", ""))
                    normalized = {
                        "question": problem.get("problem", problem.get("Problem", "")),
                        "answer": str(answer) if answer else "",
                        "solution": str(solution) if solution else "",
                        "source": "amc_aime",
                        "difficulty": 10,  # Competition level
                    }
                    proc = self._proceduralizer.proceduralize_problem(normalized)
                    self._add_if_passes_filter(proc)
                    loaded += 1

        if loaded > 0:
            return  # Loaded from JSONL

        # Fallback: ZIP file (legacy)
        zip_file = path / "AMC.zip"
        if zip_file.exists():
            import tempfile
            import zipfile

            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(zip_file, "r") as z:
                    z.extractall(tmpdir)
                for json_file in Path(tmpdir).rglob("*.json"):
                    try:
                        with open(json_file, "r") as f:
                            problem = json.load(f)
                        problem["source"] = "amc_aime"
                        proc = self._proceduralizer.proceduralize_problem(problem)
                        self._add_if_passes_filter(proc)
                    except json.JSONDecodeError:
                        continue

    def _add_if_passes_filter(self, problem: Dict[str, Any]) -> None:
        if self._difficulty_filter is None:
            self._problems.append(problem)
        elif problem.get("difficulty", 5) in self._difficulty_filter:
            self._problems.append(problem)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        problems = self._problems.copy()
        if self._shuffle:
            import random

            random.shuffle(problems)
        yield from problems

    def __len__(self) -> int:
        return len(self._problems)

    def get_stats(self) -> Dict[str, Any]:
        from collections import Counter

        sources = Counter(p["source"] for p in self._problems)
        difficulties = Counter(p.get("difficulty", 5) for p in self._problems)
        return {
            "total_problems": len(self._problems),
            "by_source": dict(sources),
            "by_difficulty": dict(difficulties),
            "datasets_loaded": self._datasets,
        }
