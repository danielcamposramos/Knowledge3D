"""Math competitions benchmark integration for Knowledgeverse Week 14."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable

from benchmarks.sampling import stratified_sample
from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC, TabletIngest
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.tablet.wine.math_wine import build_math_session_tape, math_dataset_envelope


SAFE_MATH_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "inf": math.inf,
}

LATEX_SPACING_RE = re.compile(r"\\[,;! ]")
TEXT_WRAPPER_RE = re.compile(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}")
COMMAND_BRACE_RE = re.compile(r"\\(?:boxed|fbox)\{([^{}]*)\}")
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _extract_group(text: str, start: int, *, open_ch: str = "{", close_ch: str = "}") -> tuple[str | None, int]:
    if start >= len(text) or text[start] != open_ch:
        return None, start
    depth = 0
    chars: list[str] = []
    for index in range(start, len(text)):
        ch = text[index]
        if ch == open_ch:
            depth += 1
            if depth == 1:
                continue
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return "".join(chars), index + 1
        chars.append(ch)
    return None, start


def _normalize_latex_wrappers(text: str) -> str:
    current = str(text or "").strip()
    if not current:
        return ""
    current = current.replace("$", "")
    current = current.replace("\\left", "").replace("\\right", "")
    current = current.replace("\\{", "{").replace("\\}", "}")
    current = LATEX_SPACING_RE.sub("", current)
    previous = None
    while previous != current:
        previous = current
        current = TEXT_WRAPPER_RE.sub(lambda match: match.group(1), current)
        current = COMMAND_BRACE_RE.sub(lambda match: match.group(1), current)
    return current.strip()


def normalize_latex_answer(text: Any) -> str:
    raw = _normalize_latex_wrappers(str(text or ""))
    if not raw:
        return ""
    out: list[str] = []
    i = 0
    while i < len(raw):
        if raw.startswith(("\\frac", "\\dfrac", "\\tfrac"), i):
            if raw.startswith("\\dfrac", i):
                i += len("\\dfrac")
            elif raw.startswith("\\tfrac", i):
                i += len("\\tfrac")
            else:
                i += len("\\frac")
            numerator, next_pos = _extract_group(raw, i)
            if numerator is None:
                out.append(raw[i:])
                break
            denominator, final_pos = _extract_group(raw, next_pos)
            if denominator is None:
                out.append(raw[i:])
                break
            out.append(f"(({normalize_latex_answer(numerator)})/({normalize_latex_answer(denominator)}))")
            i = final_pos
            continue
        if raw.startswith("\\sqrt", i):
            i += len("\\sqrt")
            root_expr = None
            if i < len(raw) and raw[i] == "[":
                root_expr, i = _extract_group(raw, i, open_ch="[", close_ch="]")
            body, final_pos = _extract_group(raw, i)
            if body is None:
                out.append("\\sqrt")
                continue
            normalized_body = normalize_latex_answer(body)
            if root_expr is None:
                out.append(f"sqrt({normalized_body})")
            else:
                out.append(f"(({normalized_body})**(1/({normalize_latex_answer(root_expr)})))")
            i = final_pos
            continue
        if raw.startswith("\\pi", i):
            out.append("pi")
            i += len("\\pi")
            continue
        if raw.startswith("\\cdot", i):
            out.append("*")
            i += len("\\cdot")
            continue
        if raw.startswith("\\times", i):
            out.append("*")
            i += len("\\times")
            continue
        if raw.startswith("\\div", i):
            out.append("/")
            i += len("\\div")
            continue
        if raw.startswith("\\%", i):
            out.append("%")
            i += len("\\%")
            continue
        if raw[i] == "\\":
            j = i + 1
            while j < len(raw) and raw[j].isalpha():
                j += 1
            command = raw[i + 1 : j]
            if command:
                out.append(command)
                i = j
                continue
        out.append(raw[i])
        i += 1
    normalized = "".join(out)
    normalized = normalized.replace("^", "**")
    normalized = normalized.replace("{", "(").replace("}", ")")
    normalized = re.sub(r"(?<=\d)(?=(?:pi|sqrt)\b|\()", "*", normalized)
    normalized = re.sub(r"(?<=\))(?=(?:\d|pi|sqrt|\())", "*", normalized)
    normalized = re.sub(r"(?<=pi)(?=(?:\d|sqrt|\())", "*", normalized)
    normalized = normalized.replace(",", "")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _safe_eval_math(expr: str) -> float | None:
    text = str(expr or "").strip()
    if not text:
        return None
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Num,
                ast.Constant,
                ast.Name,
                ast.Load,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.Pow,
                ast.USub,
                ast.UAdd,
            ),
        ):
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in SAFE_MATH_NAMES:
            continue
        return None
    try:
        value = eval(compile(tree, "<answer>", "eval"), {"__builtins__": {}}, SAFE_MATH_NAMES)
    except Exception:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_math_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        pass
    text = normalize_latex_answer(value)
    if not text:
        return None
    if text.endswith("%"):
        pct = _safe_eval_math(text[:-1])
        return (pct / 100.0) if pct is not None else None
    if text.startswith("$"):
        text = text[1:]
    direct = _safe_eval_math(text)
    if direct is not None:
        return direct
    cleaned = text.replace("\\", "")
    try:
        return float(cleaned)
    except Exception:
        return None


def normalize_text_answer(value: Any) -> str:
    text = normalize_latex_answer(value)
    if not text:
        return ""
    text = text.replace("\\", "")
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    return text


def math_answers_match(predicted: Any, expected: Any, *, tolerance: float = 1e-3) -> bool:
    pred = _coerce_math_number(predicted)
    exp = _coerce_math_number(expected)
    if pred is not None and exp is not None:
        return abs(pred - exp) <= tolerance
    return normalize_text_answer(predicted) == normalize_text_answer(expected)


class UnifiedMathBenchmark:
    """Single sovereign math benchmark surface for MATH-style and GSM8K-style prompts."""

    def __init__(
        self,
        knowledgeverse: Knowledgeverse | None = None,
        dataset_path: str | Path | None = None,
        gsm8k_dataset_path: str | Path | None = None,
        dataset_mode: str | None = None,
        max_problems: int | None = None,
        max_math_questions: int | None = None,
        source_filter: str | list[str] | None = None,
        query_scope_galaxies: str | list[str] | None = None,
        runtime_seed_knowledge: bool = False,
        tablet_boundary: HeadlessTabletMPC | None = None,
    ):
        self.kv = knowledgeverse or Knowledgeverse()
        resolved_mode = str(dataset_mode or "").strip().lower()
        if not resolved_mode:
            resolved_mode = "present" if dataset_path is not None else "synthetic"
        if resolved_mode not in {"synthetic", "present"}:
            resolved_mode = "synthetic"
        self.dataset_mode = resolved_mode
        self.dataset_path = (
            self._resolve_dataset_path(dataset_path)
            if self.dataset_mode == "present"
            else Path("")
        )
        self.gsm8k_dataset_path = self._resolve_math_dataset_path(gsm8k_dataset_path)
        self.max_problems = max_problems
        self.max_math_questions = max_math_questions
        self.source_filter = self._normalize_source_filter(source_filter)
        self.query_scope_galaxies = self._normalize_query_scope(query_scope_galaxies)
        self.runtime_seed_knowledge = bool(runtime_seed_knowledge)
        self.tablet_boundary = tablet_boundary
        self.dataset_sources: list[str] = []
        self.used_synthetic_math_fallback = False
        self.used_synthetic_word_math_fallback = False
        self.problems = self._load_problems()
        self.results: list[dict[str, Any]] = []

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
            Path("/K3D/K3D_llama_cpp/datasets"),
            Path("/K3D/K3D_llama_cpp/datasets/GSM8K"),
            Path("/K3D/K3D_llama_cpp/datasets/math"),
            Path("/K3D/Knowledge3D.local/datasets/math_competitions"),
            Path("../Knowledge3D.local/datasets/math_competitions"),
            Path("data"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return Path("")

    def _resolve_math_dataset_path(self, dataset_path: str | Path | None) -> Path:
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

    def _normalize_source_filter(self, value: str | list[str] | None) -> set[str]:
        if isinstance(value, list):
            raw = [str(item).strip().lower() for item in value]
        elif isinstance(value, str):
            raw = [segment.strip().lower() for segment in value.split(",")]
        else:
            raw = []
        normalized = {item for item in raw if item}
        return normalized or {"math", "gsm8k"}

    def _has_present_dataset(self, root: Path) -> bool:
        if not root:
            return False
        candidates = [
            root / "data" / "train.jsonl",
            root / "math" / "data" / "train.jsonl",
            root / "data_train.jsonl",
            root / "math" / "data_train.jsonl",
        ]
        return any(candidate.exists() for candidate in candidates)

    def _load_problems(self) -> list[dict[str, Any]]:
        max_problems = None if self.max_problems is None else int(self.max_problems)
        max_math_questions = None if self.max_math_questions is None else int(self.max_math_questions)
        if (
            max_problems is not None
            and max_math_questions is not None
            and max_problems <= 0
            and max_math_questions <= 0
        ):
            return []
        problems: list[dict[str, Any]] = []
        if "math" in self.source_filter:
            problems.extend(self._load_competition_math_problems())
        if "gsm8k" in self.source_filter:
            problems.extend(self._load_word_math_problems())
        if "amc_aime" in self.source_filter:
            problems.extend(self._load_amc_aime_dataset())
        if "omni_math" in self.source_filter:
            problems.extend(self._load_omni_math_dataset())
        return problems

    def _load_amc_aime_dataset(self) -> list[dict[str, Any]]:
        root = self.dataset_path if self.dataset_path and self.dataset_path.exists() else self._resolve_dataset_path(None)
        candidates = [
            root / "AMC-AIME" / "data",
            root / "AMC-AIME",
            Path("/K3D/K3D_llama_cpp/datasets/AMC-AIME/data"),
            Path("../K3D_llama_cpp/datasets/AMC-AIME/data"),
        ]
        out: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for candidate_root in candidates:
            if not candidate_root.exists():
                continue
            jsonl_files = (
                sorted(candidate_root.glob("*.jsonl"))
                if candidate_root.is_dir()
                else [candidate_root] if candidate_root.suffix.lower() == ".jsonl" else []
            )
            for jsonl_file in jsonl_files:
                try:
                    with jsonl_file.open("r", encoding="utf-8") as handle:
                        for idx, line in enumerate(handle):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                payload = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if not isinstance(payload, dict):
                                continue
                            text = str(payload.get("problem") or payload.get("Problem") or payload.get("question") or "").strip()
                            answer = payload.get("answer")
                            if answer in (None, ""):
                                answer = self._extract_symbolic_math_answer(payload.get("solution"))
                            if not text or answer in (None, ""):
                                continue
                            record_id = str(payload.get("id") or payload.get("problem_id") or f"amc_aime_{jsonl_file.stem}_{idx}")
                            if record_id in seen_ids:
                                continue
                            seen_ids.add(record_id)
                            competition = str(payload.get("competition") or payload.get("source") or "").strip().upper()
                            if competition not in {"AMC", "AIME"}:
                                competition = "AIME" if "aime" in jsonl_file.stem.lower() else "AMC"
                            out.append(
                                {
                                    "id": record_id,
                                    "suite": "math",
                                    "source_key": "amc_aime",
                                    "competition": competition,
                                    "problem_text": text,
                                    "answer": answer,
                                }
                            )
                except Exception:
                    continue
            if out:
                self.dataset_sources.append(str(candidate_root))
                return stratified_sample(out, self.max_problems)
        return []

    def _load_omni_math_dataset(self) -> list[dict[str, Any]]:
        root = self.dataset_path if self.dataset_path and self.dataset_path.exists() else self._resolve_dataset_path(None)
        candidates = [
            root / "Omni-Math.jsonl",
            root / "Omni-MATH" / "Omni-Math.jsonl",
            root / "Omni-MATH" / "Omni-MATH.jsonl",
            Path("/K3D/K3D_llama_cpp/datasets/Omni-Math.jsonl"),
            Path("/K3D/K3D_llama_cpp/datasets/Omni-MATH/Omni-Math.jsonl"),
            Path("../K3D_llama_cpp/datasets/Omni-MATH/Omni-Math.jsonl"),
        ]
        out: list[dict[str, Any]] = []
        for path in candidates:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for idx, line in enumerate(handle):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(payload, dict):
                            continue
                        text = str(payload.get("problem") or payload.get("question") or "").strip()
                        answer = payload.get("answer")
                        if answer in (None, ""):
                            answer = self._extract_symbolic_math_answer(payload.get("solution"))
                        if not text or answer in (None, ""):
                            continue
                        out.append(
                            {
                                "id": str(payload.get("id") or f"omni_math_{idx}"),
                                "suite": "math",
                                "source_key": "omni_math",
                                "competition": str(payload.get("source") or "Omni-MATH"),
                                "problem_text": text,
                                "answer": answer,
                            }
                        )
            except Exception:
                continue
            if out:
                self.dataset_sources.append(str(path))
                return stratified_sample(out, self.max_problems)
        return []

    def _load_competition_math_problems(self) -> list[dict[str, Any]]:
        if self.dataset_mode == "synthetic":
            present_root = self._resolve_dataset_path(None)
            if self._has_present_dataset(present_root):
                self.dataset_mode = "present"
                self.dataset_path = present_root
        if self.dataset_mode == "synthetic":
            raise FileNotFoundError(
                "math_dataset_missing: unified math runtime requires the real dataset roots, not synthetic fallback"
            )
        if self.dataset_mode == "present" and self.dataset_path and self.dataset_path.exists():
            staged = self._load_from_present_datasets(self.dataset_path, limit=None)
            if staged:
                return stratified_sample(staged, self.max_problems)
        fallback = self._load_from_calculus_microbench()
        if fallback:
            return stratified_sample(fallback, self.max_problems)
        raise FileNotFoundError(
            f"math_dataset_empty: no valid competition math records found under {self.dataset_path or self._resolve_dataset_path(None)}"
        )

    def _load_word_math_problems(self) -> list[dict[str, Any]]:
        limit = self.max_math_questions
        if self.gsm8k_dataset_path and self.gsm8k_dataset_path.exists():
            out: list[dict[str, Any]] = []
            with self.gsm8k_dataset_path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    question = str(payload.get("question") or "").strip()
                    answer = self._extract_word_math_answer(payload.get("answer"))
                    if not question or answer is None:
                        continue
                    out.append(
                        {
                            "id": f"gsm8k_{idx}",
                            "suite": "gsm8k",
                            "source_key": "gsm8k",
                            "competition": "GSM8K",
                            "problem_text": question,
                            "answer": answer,
                        }
                    )
            if out:
                self.dataset_sources.append(str(self.gsm8k_dataset_path))
                return stratified_sample(out, limit)
        raise FileNotFoundError(
            f"gsm8k_dataset_missing: no valid GSM8K records found at {self.gsm8k_dataset_path or 'canonical roots'}"
        )

    def _load_from_present_datasets(self, root: Path, limit: int | None = None) -> list[dict[str, Any]]:
        batches: list[list[dict[str, Any]]] = []
        for loader in (
            self._load_from_competition_files,
            self._load_from_math_dataset,
        ):
            batch = loader(root, limit=limit)
            if batch:
                batches.append(batch)
        if limit is None:
            return [row for batch in batches for row in batch]
        out: list[dict[str, Any]] = []
        offset = 0
        while len(out) < int(limit):
            progressed = False
            for batch in batches:
                if offset < len(batch):
                    out.append(batch[offset])
                    progressed = True
                    if len(out) >= int(limit):
                        break
            if not progressed:
                break
            offset += 1
        return out

    def _load_from_competition_files(self, root: Path, limit: int | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        sources = [
            ("AMC", root / "amc_problems.json"),
            ("AIME", root / "aime_problems.json"),
            ("IMO", root / "imo_problems.json"),
        ]
        for competition, source in sources:
            if not source.exists():
                continue
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                records = payload.get("problems", [])
            else:
                records = payload
            if not isinstance(records, list):
                continue
            for idx, record in enumerate(records):
                if not isinstance(record, dict):
                    continue
                text = str(record.get("problem_text") or record.get("question") or "").strip()
                answer = record.get("answer")
                if not text or answer is None:
                    continue
                out.append(
                    {
                        "id": str(record.get("id") or f"{competition.lower()}_{idx}"),
                        "suite": "math",
                        "source_key": "math",
                        "competition": competition,
                        "problem_text": text,
                        "answer": answer,
                    }
                )
        if out:
            self.dataset_sources.append("competition_json")
        return out

    def _load_from_math_dataset(self, root: Path, limit: int | None = None) -> list[dict[str, Any]]:
        candidates = [
            root / "data" / "train.jsonl",
            root / "math" / "data" / "train.jsonl",
            root / "data_train.jsonl",
            root / "math" / "data_train.jsonl",
        ]
        for path in candidates:
            if not path.exists():
                continue
            out: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = str(payload.get("problem") or "").strip()
                    answer = self._extract_symbolic_math_answer(payload.get("solution"))
                    if not text or answer is None:
                        continue
                    math_type = str(payload.get("type") or "MATH").strip() or "MATH"
                    out.append(
                        {
                            "id": f"math_{idx}",
                            "suite": "math",
                            "source_key": "math",
                            "competition": f"MATH:{math_type}",
                            "problem_text": text,
                            "answer": answer,
                        }
                    )
            if out:
                self.dataset_sources.append(str(path))
                return out
        return []

    def _load_from_calculus_microbench(self) -> list[dict[str, Any]]:
        paths = [
            Path("data/calculus_microbench.jsonl"),
            Path("../Knowledge3D.local/datasets/calculus_microbench.jsonl"),
        ]
        records: list[dict[str, Any]] = []
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = str(payload.get("problem", "")).strip()
                    answer = payload.get("answer")
                    if not text or answer is None:
                        continue
                    records.append(
                        {
                            "id": str(payload.get("id") or f"calculus_{idx}"),
                            "suite": "math",
                            "source_key": "math",
                            "competition": "AMC",
                            "problem_text": text,
                            "answer": answer,
                        }
                    )
        return records

    def _with_source_defaults(
        self,
        rows: list[dict[str, Any]],
        *,
        suite: str,
        source_key: str,
        competition: str | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            enriched = dict(row)
            enriched.setdefault("suite", suite)
            enriched.setdefault("source_key", source_key)
            if competition is not None:
                enriched.setdefault("competition", competition)
            out.append(enriched)
        return out

    def _phase_cd_gap_problems(self) -> list[dict[str, Any]]:
        return self._with_source_defaults([
            {
                "id": "amc_poly_1",
                "competition": "AMC",
                "problem_text": "Find derivative of x^2 + 4x at x=3",
                "answer": "10",
            },
            {
                "id": "amc_quotient_1",
                "competition": "AMC",
                "problem_text": "Find f'(1) where f(x) = (3x-4)/(2x+3)",
                "answer": "0.68",
            },
            {
                "id": "aime_basic_1",
                "competition": "AIME",
                "problem_text": "Compute 12 * (3 + 2)",
                "answer": "60",
            },
            {
                "id": "imo_basic_1",
                "competition": "IMO",
                "problem_text": "Evaluate derivative of cos(x) at x=0",
                "answer": "0",
            },
            {
                "id": "amc_poly_2",
                "competition": "AMC",
                "problem_text": "Find derivative of 3x^2 + 2x at x=1",
                "answer": "8",
            },
            {
                "id": "aime_chain_1",
                "competition": "AIME",
                "problem_text": "Evaluate f'(1) where f(x) = (x+2)^3",
                "answer": "27",
            },
            {
                "id": "amc_linear_1",
                "competition": "AMC",
                "problem_text": "Find derivative of 7x - 4 at x=3",
                "answer": "7",
            },
            {
                "id": "imo_power_1",
                "competition": "IMO",
                "problem_text": "Evaluate derivative of x^4 at x=1",
                "answer": "4",
            },
        ], suite="math", source_key="math")

    def _synthetic_problems(self) -> list[dict[str, Any]]:
        # Honest Phase B+ guard set: only prompt families that the current composed-head
        # runtime can solve on the executable template path. Legacy derivative/calculus
        # prompts are retained separately in _phase_cd_gap_problems() for future phases.
        return self._with_source_defaults([
            {
                "id": "guard_linear_1",
                "competition": "AMC",
                "problem_text": "Solve linear equation 2x + 3 = 11.",
                "answer": "4",
                "tier": "tier2_algebra_template",
            },
            {
                "id": "guard_linear_2",
                "competition": "AMC",
                "problem_text": "Solve linear equation 3x + 5 = 20.",
                "answer": "5",
                "tier": "tier2_algebra_template",
            },
            {
                "id": "guard_linear_3",
                "competition": "AMC",
                "problem_text": "Solve linear equation 4x + 7 = 31.",
                "answer": "6",
                "tier": "tier2_algebra_template",
            },
            {
                "id": "guard_linear_4",
                "competition": "AMC",
                "problem_text": "Solve linear equation 6x + 2 = 26.",
                "answer": "4",
                "tier": "tier2_algebra_template",
            },
            {
                "id": "guard_linear_5",
                "competition": "AMC",
                "problem_text": "Solve linear equation 8x + 1 = 41.",
                "answer": "5",
                "tier": "tier2_algebra_template",
            },
            {
                "id": "guard_factorial_1",
                "competition": "AIME",
                "problem_text": "What is 4 factorial?",
                "answer": "24",
                "tier": "tier1_combinatorics_template",
            },
            {
                "id": "guard_factorial_2",
                "competition": "AIME",
                "problem_text": "What is 5 factorial?",
                "answer": "120",
                "tier": "tier1_combinatorics_template",
            },
            {
                "id": "guard_factorial_3",
                "competition": "AIME",
                "problem_text": "What is 6 factorial?",
                "answer": "720",
                "tier": "tier1_combinatorics_template",
            },
            {
                "id": "guard_factorial_4",
                "competition": "AIME",
                "problem_text": "What is 7 factorial?",
                "answer": "5040",
                "tier": "tier1_combinatorics_template",
            },
            {
                "id": "guard_factorial_5",
                "competition": "AIME",
                "problem_text": "What is 10 factorial?",
                "answer": "3628800",
                "tier": "tier1_combinatorics_template",
            },
            {
                "id": "guard_binomial_1",
                "competition": "AIME",
                "problem_text": "What is 8 choose 2?",
                "answer": "28",
                "tier": "tier1_combinatorics_template",
            },
            {
                "id": "guard_binomial_2",
                "competition": "AIME",
                "problem_text": "What is 10 choose 3?",
                "answer": "120",
                "tier": "tier1_combinatorics_template",
            },
            {
                "id": "guard_binomial_3",
                "competition": "AIME",
                "problem_text": "What is 12 choose 4?",
                "answer": "495",
                "tier": "tier1_combinatorics_template",
            },
            {
                "id": "guard_binomial_4",
                "competition": "AIME",
                "problem_text": "What is 7 choose 1?",
                "answer": "7",
                "tier": "tier1_combinatorics_template",
            },
            {
                "id": "guard_binomial_5",
                "competition": "AIME",
                "problem_text": "What is 9 choose 0?",
                "answer": "1",
                "tier": "tier1_combinatorics_template",
            },
            {
                "id": "guard_series_1",
                "competition": "AMC",
                "problem_text": "What is the sum of first 10 positive integers?",
                "answer": "55",
                "tier": "tier1_series_template",
            },
            {
                "id": "guard_series_2",
                "competition": "AMC",
                "problem_text": "What is the sum of an arithmetic series with first term 3, last term 21, and 7 terms?",
                "answer": "84",
                "tier": "tier1_series_template",
            },
            {
                "id": "guard_series_3",
                "competition": "AMC",
                "problem_text": "What is the sum of a geometric series with first term 2, common ratio 3, and 4 terms?",
                "answer": "80",
                "tier": "tier1_series_template",
            },
            {
                "id": "guard_series_4",
                "competition": "AMC",
                "problem_text": "What is the sum of a geometric series with first term 5, common ratio 2, and 5 terms?",
                "answer": "155",
                "tier": "tier1_series_template",
            },
            {
                "id": "guard_series_5",
                "competition": "AMC",
                "problem_text": "What is the sum of a geometric series with first term 3, common ratio 4, and 3 terms?",
                "answer": "63",
                "tier": "tier1_series_template",
            },
        ], suite="math", source_key="math")

    def _synthetic_guard_problems(self) -> list[dict[str, Any]]:
        return list(self._synthetic_problems())

    def _synthetic_word_math_questions(self) -> list[dict[str, Any]]:
        return self._with_source_defaults(
            [
                {
                    "id": "gsm8k_synthetic_0",
                    "problem_text": (
                        "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and "
                        "bakes muffins for her friends every day with four. She sells the remainder at the "
                        "farmers' market daily for $2 per fresh duck egg. How much in dollars does she make "
                        "every day at the farmers' market?"
                    ),
                    "answer": "18",
                }
            ],
            suite="gsm8k",
            source_key="gsm8k",
            competition="GSM8K",
        )

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
        self.results = []
        by_competition: dict[str, dict[str, Any]] = {}
        by_source: dict[str, dict[str, Any]] = {}
        total = len(self.problems)
        resume_index = max(0, min(int(start_index), total))
        correct = max(0, int(initial_correct))
        source_completed = {
            str(key): int(value)
            for key, value in dict(initial_source_completed or {}).items()
            if str(key).strip()
        }
        source_correct = {
            str(key): int(value)
            for key, value in dict(initial_source_correct or {}).items()
            if str(key).strip()
        }
        step = max(1, int(progress_every or 25))
        start = time.monotonic()
        tablet_rows: list[dict[str, Any]] = []
        if self.tablet_boundary is not None:
            tape = build_math_session_tape(
                session_id=f"math_{int(time.time() * 1000)}",
                suite_name="unified_math",
                rows=self.problems[resume_index:],
                use_enriched=use_enriched,
            )
            tablet_rows = list(self.tablet_boundary.run_tape_session(tape)["results"])
        for offset, problem in enumerate(self.problems[resume_index:]):
            index = resume_index + offset + 1
            if self.tablet_boundary is not None:
                result = self._math_result_from_tablet(problem=problem, tablet_result=tablet_rows[offset])
                task_result = dict(result.get("task_result") or {})
                result["elapsed_s"] = round(
                    max(0.0, float(task_result.get("trm_latency_us", 0.0) or 0.0) / 1_000_000.0),
                    3,
                )
            else:
                row_start = time.monotonic()
                result = self._solve_problem(problem=problem, use_enriched=use_enriched)
                result["elapsed_s"] = round(max(0.0, time.monotonic() - row_start), 3)
            self.results.append(result)
            if result["correct"]:
                correct += 1
            if row_cb is not None:
                row_cb(problem, result)
            comp = result["competition"]
            if comp not in by_competition:
                by_competition[comp] = {"total": 0, "correct": 0, "results": []}
            by_competition[comp]["total"] += 1
            if result["correct"]:
                by_competition[comp]["correct"] += 1
            by_competition[comp]["results"].append(result)
            source_key = str(problem.get("source_key") or result.get("suite") or "math")
            if source_key not in by_source:
                by_source[source_key] = {"total": 0, "correct": 0, "results": []}
            by_source[source_key]["total"] += 1
            if result["correct"]:
                by_source[source_key]["correct"] += 1
            by_source[source_key]["results"].append(result)
            source_completed[source_key] = int(source_completed.get(source_key, 0)) + 1
            if result["correct"]:
                source_correct[source_key] = int(source_correct.get(source_key, 0)) + 1
            if progress_cb and (index % step == 0 or index == total):
                progress_cb(
                    {
                        "completed": index,
                        "total": total,
                        "correct": correct,
                        "elapsed_s": time.monotonic() - start,
                        "benchmark": "unified_math",
                        "current_source": source_key,
                        "source_completed": dict(source_completed),
                        "source_correct": dict(source_correct),
                    }
                )
        for comp_data in by_competition.values():
            comp_total = comp_data["total"]
            comp_data["accuracy"] = (comp_data["correct"] / comp_total) if comp_total else 0.0
        for source_data in by_source.values():
            source_total = source_data["total"]
            source_data["accuracy"] = (source_data["correct"] / source_total) if source_total else 0.0
        total_correct = correct
        total_count = len(self.results)
        pred_none_count = sum(1 for row in self.results if row.get("predicted_answer") is None)
        pred_numeric_count = sum(1 for row in self.results if self._to_float(row.get("predicted_answer")) is not None)
        exp_numeric_count = sum(1 for row in self.results if self._to_float(row.get("expected_answer")) is not None)
        route_specialists: dict[str, int] = {}
        failure_reason_counts: dict[str, int] = {}
        for row in self.results:
            specialist = str(row.get("route", {}).get("specialist", "unknown"))
            route_specialists[specialist] = int(route_specialists.get(specialist, 0)) + 1
            if row.get("predicted_answer") is None:
                reason = str(row.get("failure_reason", "") or "unknown")
                failure_reason_counts[reason] = int(failure_reason_counts.get(reason, 0)) + 1
        return {
            "benchmark": "Unified Math",
            "dataset_mode": self.dataset_mode,
            "dataset_path": str(self.dataset_path) if self.dataset_path else "synthetic",
            "gsm8k_dataset_path": str(self.gsm8k_dataset_path) if self.gsm8k_dataset_path else "synthetic",
            "dataset_sources": list(self.dataset_sources),
            "use_enriched": use_enriched,
            "source_filter": sorted(self.source_filter),
            "results_by_source": by_source,
            "results_by_competition": by_competition,
            "overall_accuracy": (total_correct / total_count) if total_count else 0.0,
            "total": total_count,
            "correct": total_correct,
            "results": self.results,
            "diagnostics": {
                "predicted_none_count": int(pred_none_count),
                "predicted_none_rate": (pred_none_count / total_count) if total_count else 0.0,
                "predicted_numeric_count": int(pred_numeric_count),
                "expected_numeric_count": int(exp_numeric_count),
                "route_specialist_counts": route_specialists,
                "failure_reason_counts": failure_reason_counts,
            },
        }

    def _solve_problem(
        self,
        *,
        problem: dict[str, Any],
        use_enriched: bool,
    ) -> dict[str, Any]:
        self._ensure_tablet_boundary()
        return self._solve_problem_via_tablet(problem=problem, use_enriched=use_enriched)

    def _solve_problem_via_tablet(
        self,
        *,
        problem: dict[str, Any],
        use_enriched: bool,
    ) -> dict[str, Any]:
        envelope = math_dataset_envelope(
            task_id=str(problem["id"]),
            question=str(problem["problem_text"]),
            expected_answer=problem.get("answer"),
        )
        tablet_result = self.tablet_boundary.submit(envelope, use_enriched=use_enriched)
        return self._math_result_from_tablet(problem=problem, tablet_result=tablet_result)

    def _math_result_from_tablet(
        self,
        *,
        problem: dict[str, Any],
        tablet_result: dict[str, Any],
    ) -> dict[str, Any]:
        emitted = dict(tablet_result["emitted"])
        route = emitted.get("route", {})
        predicted = emitted.get("numeric_answer")
        if predicted is None:
            predicted = emitted.get("answer_text")
        expected = problem["answer"]
        correct = bool(emitted.get("correct", False))
        self.kv.log_event(
            "math_problem_solved" if correct else "math_problem_failed",
            {
                "specialist": route.get("specialist", "math"),
                "confidence": 0.9 if correct else 0.35,
                "competition": problem["competition"],
            },
        )
        return {
            "problem_id": problem["id"],
            "suite": str(problem.get("suite") or "math"),
            "source_key": str(problem.get("source_key") or "math"),
            "competition": problem["competition"],
            "correct": int(correct),
            "predicted_answer": predicted,
            "expected_answer": expected,
            "failure_reason": "" if predicted is not None else "tablet_boundary_no_result",
            "failure_signal": None,
            "symbols_used": 0,
            "reasoning_trace": emitted.get("task_result", {}).get("reasoning_trace", []),
            "route": route,
            "meta_specialist": route.get("specialist"),
            "method": "tablet_boundary",
            "solver": str(emitted.get("task_result", {}).get("solver", "tablet_boundary")),
            "runtime": str(emitted.get("task_result", {}).get("runtime", "")),
            "gpu_execution": bool(emitted.get("task_result", {}).get("gpu_execution", False)),
            "program_id": str(emitted.get("task_result", {}).get("program_id", "")),
            "task_result": emitted.get("task_result", {}),
            "generated_id": None,
            "tablet_contract": tablet_result["tablet_contract"],
        }

    def _extract_failure_reason(self, reasoning_trace: list[str]) -> str:
        for item in reversed(reasoning_trace):
            if not isinstance(item, str):
                continue
            marker = "math_solve_missing reason="
            if marker in item:
                return item.split(marker, 1)[1].strip() or "unknown"
        return "unknown"

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

    def _seed_math_knowledge(self, problem: dict[str, Any]) -> None:
        source_key = str(problem.get("source_key") or problem.get("suite") or "math")
        self.kv.galaxy_manager.add_entry(
            "Math",
            {
                "domain": "math",
                "competition": problem["competition"],
                "problem_id": problem["id"],
                "kind": "gsm8k_word_problem_anchor" if source_key == "gsm8k" else "symbolic_pattern",
            },
        )
        if source_key != "gsm8k":
            self.kv.galaxy_manager.add_entry(
                "Grammar",
                {
                    "domain": "math",
                    "problem_id": problem["id"],
                    "kind": "derivation_rule",
                },
            )

    def _answers_match(self, predicted: Any, expected: Any) -> bool:
        return math_answers_match(predicted, expected)

    def _to_float(self, value: Any) -> float | None:
        return _coerce_math_number(value)

    def _normalize_text_answer(self, value: Any) -> str:
        return normalize_text_answer(value)

    def _extract_word_math_answer(self, raw_answer: Any) -> str | None:
        text = str(raw_answer or "").strip()
        if not text:
            return None
        match = re.search(r"####\s*([^\n]+)", text)
        if match:
            return match.group(1).strip().replace(",", "")
        return text.splitlines()[-1].strip() or None

    def _extract_symbolic_math_answer(self, solution: Any) -> str | None:
        text = str(solution or "").strip()
        if not text:
            return None
        boxed = self._extract_last_boxed(text)
        if boxed:
            return boxed.strip()
        match = re.search(r"####\s*([^\n]+)", text)
        if match:
            return match.group(1).strip()
        tail = text.splitlines()[-1].strip()
        tail = tail.rstrip(".")
        eq_match = re.search(r"=\s*([^=]+)$", tail)
        if eq_match:
            return eq_match.group(1).strip()
        return tail or None

    def _extract_last_boxed(self, text: str) -> str | None:
        markers = (r"\boxed{", r"\fbox{")
        last_pos = -1
        marker_used = ""
        for marker in markers:
            pos = text.rfind(marker)
            if pos > last_pos:
                last_pos = pos
                marker_used = marker
        if last_pos < 0:
            return None
        start = last_pos + len(marker_used)
        depth = 1
        chars: list[str] = []
        for ch in text[start:]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(chars)
            chars.append(ch)
        return None

    def _should_attempt_autonomous_generation(self, text: str) -> bool:
        lowered = text.lower()
        triggers = (
            "differential",
            "rate of change",
            "decay",
            "growth",
            "velocity",
            "acceleration",
            "projectile",
            "pendulum",
            "field",
            "thermo",
        )
        return any(token in lowered for token in triggers)

    def save_results(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": "Unified Math",
            "total": len(self.results),
            "correct": sum(int(row.get("correct", 0)) for row in self.results),
            "accuracy": (
                sum(int(row.get("correct", 0)) for row in self.results) / len(self.results)
                if self.results
                else 0.0
            ),
            "results": self.results,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class MathCompetitionBenchmark(UnifiedMathBenchmark):
    """Backward-compatible math-only wrapper over the unified sovereign math benchmark."""

    def __init__(
        self,
        knowledgeverse: Knowledgeverse | None = None,
        dataset_path: str | Path | None = None,
        dataset_mode: str | None = None,
        max_problems: int | None = None,
        query_scope_galaxies: str | list[str] | None = None,
        runtime_seed_knowledge: bool = False,
        tablet_boundary: HeadlessTabletMPC | None = None,
    ) -> None:
        super().__init__(
            knowledgeverse=knowledgeverse,
            dataset_path=dataset_path,
            dataset_mode=dataset_mode,
            max_problems=max_problems,
            max_math_questions=0,
            source_filter=["math"],
            query_scope_galaxies=query_scope_galaxies,
            runtime_seed_knowledge=runtime_seed_knowledge,
            tablet_boundary=tablet_boundary,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sovereign math benchmark through the tablet boundary.")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--dataset-mode", type=str, default=None)
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--query-scope-galaxies", type=str, default=None)
    parser.add_argument("--summary-output", type=str, default=None)
    parser.add_argument("--use-enriched", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    benchmark = MathCompetitionBenchmark(
        dataset_path=args.dataset_path,
        dataset_mode=args.dataset_mode,
        max_problems=args.max_tasks,
        query_scope_galaxies=args.query_scope_galaxies,
    )
    summary = benchmark.run_benchmark(use_enriched=bool(args.use_enriched))
    if args.summary_output:
        benchmark.save_results(args.summary_output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
