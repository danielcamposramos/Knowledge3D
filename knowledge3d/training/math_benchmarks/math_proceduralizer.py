"""
Convert text-based math problems into procedural RPN form.

This is an ingestion-layer utility (not hot path).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import re

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
        source = problem.get("source", "unknown")

        quantities = self._extract_quantities(text)
        operations = self._extract_operations(text)

        problem_rpn = self._build_problem_rpn(quantities, operations)

        # Executable RPN extracted from solution when present (source-specific)
        solution_rpn_str = ""
        if source == "gsm8k" or ("<<" in str(solution) and ">>" in str(solution)):
            solution_rpn_str = self._parse_math_solution(str(solution))
        elif source in ("math", "omni_math") or "\\boxed" in str(solution):
            solution_rpn_str = self._parse_math_solution(str(solution))

        solution_rpn = self._parse_solution(solution) if solution else []
        answer = self._extract_answer(solution)

        return {
            "problem_id": problem.get("id", problem.get("problem_id", str(hash(text)))),
            "source": problem.get("source", "unknown"),
            "difficulty": problem.get("level", problem.get("difficulty", 5)),
            "problem_rpn": problem_rpn,
            "solution_rpn": solution_rpn_str,
            "answer": answer,
            "metadata": {
                "original_text": text,
                "original_solution": solution,
                "solution_rpn": solution_rpn_str,
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

    def _parse_math_solution(self, solution: str) -> str:
        """
        Enhanced GSM8K parser with better multi-step coverage.
        """
        calc_pattern = r"<<([^=]+)=([^>]+)>>"
        calculations = re.findall(calc_pattern, solution)

        rpn_steps: List[str] = []
        for expr, _ in calculations:
            expr = expr.strip()
            if "%" in expr:
                expr = self._handle_percentage(expr)
            rpn = self._infix_to_rpn(expr)
            if rpn:
                rpn_steps.append(rpn)

        # Inline calculations without markers (fallback)
        if not rpn_steps:
            inline_pattern = r"(\d+(?:\.\d+)?)\s*([+\-*/×÷])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)"
            inline_calcs = re.findall(inline_pattern, solution)
            for a, op, b, _ in inline_calcs:
                op_map = {"+": "+", "-": "-", "*": "*", "/": "/", "×": "*", "÷": "/"}
                rpn_steps.append(f"{a} {b} {op_map.get(op, '+')}")

        return " ".join(rpn_steps)

    def _handle_percentage(self, expr: str) -> str:
        """Convert percentage expressions."""
        pct_of = re.match(r"(\d+(?:\.\d+)?)\s*%\s*(?:of)?\s*(\d+(?:\.\d+)?)", expr)
        if pct_of:
            pct, base = pct_of.groups()
            return f"{base} {pct} * 100 /"
        return expr

    def _parse_math_solution(self, solution: str) -> str:
        """
        Parse MATH/Omni-MATH LaTeX solutions into executable RPN.
        """
        math_pattern = r"\$+([^$]+)\$+"
        math_exprs = re.findall(math_pattern, solution)

        rpn_parts: List[str] = []
        for expr in math_exprs:
            rpn = self._latex_to_rpn(expr)
            if rpn:
                rpn_parts.append(rpn)

        return " ".join(rpn_parts) if rpn_parts else ""

    def _latex_to_rpn(self, latex: str) -> str:
        """
        Convert LaTeX math expression to RPN.

        Handles +,-,*,/, fractions, powers, roots, boxed answers.
        """
        latex = latex.strip()

        boxed = re.search(r"\\\\boxed\\{([^}]+)\\}", latex)
        if boxed:
            latex = boxed.group(1)

        while "\\\\binom" in latex:
            latex = re.sub(r"\\\\binom\\{([^}]+)\\}\\{([^}]+)\\}", r"((\\1)! (\\2)! (\\1-\\2)! / )", latex, count=1)

        while "\\\\frac" in latex:
            latex = re.sub(r"\\\\frac\\{([^}]+)\\}\\{([^}]+)\\}", r"((\\1)/(\\2))", latex, count=1)

        latex = re.sub(r"\\sqrt\[([^\]]+)\]\{([^}]+)\}", r"((\\2) 1 \\1 / ^)", latex)
        latex = re.sub(r"\\sqrt\{([^}]+)\}", r"((\\1) sqrt)", latex)
        latex = re.sub(r"(\\d+)\\^\\{([^}]+)\\}", r"(\\1 \\2 ^)", latex)
        latex = re.sub(r"(\\d+)\\^(\\d+)", r"(\\1 \\2 ^)", latex)

        latex = latex.replace("!", " ! ")
        latex = latex.replace("\\\\sin", "sin").replace("\\\\cos", "cos").replace("\\\\tan", "tan")
        latex = latex.replace("\\\\log", "log").replace("\\\\ln", "ln")
        latex = latex.replace("\\\\mod", "mod").replace("\\\\bmod", "mod")

        latex = re.sub(r"\\\\[a-zA-Z]+", "", latex)
        return self._infix_to_rpn(latex)

    def _infix_to_rpn(self, expr: str) -> str:
        """
        Convert infix expression to RPN with proper precedence.

        Handles:
            - Single ops: "48/2" -> "48 2 /"
            - Chains: "100-50-30-15" -> "100 50 - 30 - 15 -"
            - Mixed: "12/60*50" -> "12 60 / 50 *"
            - Parentheses: "(2+3)*4" -> "2 3 + 4 *"
        """
        expr = expr.replace("x", "*").replace("×", "*").replace("÷", "/")

        token_pattern = r"(\d+\.?\d*|sin|cos|tan|log|ln|sqrt|abs|mod|[+\-*/()^!])"
        tokens = re.findall(token_pattern, expr)
        if not tokens:
            return ""

        output: List[str] = []
        op_stack: List[str] = []
        precedence = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3, "!": 4}
        right_assoc = {"^"}
        functions = {"sin", "cos", "tan", "log", "ln", "sqrt", "abs", "mod"}

        num_pattern = re.compile(r"\d+\.?\d*")

        for token in tokens:
            if num_pattern.match(token):
                output.append(token)
            elif token in functions:
                op_stack.append(token)
            elif token in precedence:
                while (
                    op_stack
                    and op_stack[-1] != "("
                    and (op_stack[-1] in precedence or op_stack[-1] in functions)
                    and (
                        (op_stack[-1] in precedence and precedence[op_stack[-1]] > precedence[token])
                        or (
                            op_stack[-1] in precedence
                            and precedence[op_stack[-1]] == precedence[token]
                            and token not in right_assoc
                        )
                    )
                ):
                    output.append(op_stack.pop())
                op_stack.append(token)
            elif token == "(":
                op_stack.append(token)
            elif token == ")":
                while op_stack and op_stack[-1] != "(":
                    output.append(op_stack.pop())
                if op_stack and op_stack[-1] == "(":
                    op_stack.pop()
                if op_stack and op_stack[-1] in functions:
                    output.append(op_stack.pop())

        while op_stack:
            output.append(op_stack.pop())

        return " ".join(output)


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
