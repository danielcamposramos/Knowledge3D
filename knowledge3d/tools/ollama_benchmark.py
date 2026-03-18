"""Ollama-specific benchmark runner with model routing and H17-backed RAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Callable

from knowledge3d.ingestion.ollama_manager import OllamaModelManager
from knowledge3d.ingestion.universal_knowledge import iter_domains, iter_elements, iter_physical_constants


OllamaQueryFn = Callable[[dict[str, Any]], dict[str, Any]]

SUITE_MODEL_MAP: dict[str, str] = {
    "mmlu": "qwen3:8b",
    "gsm8k": "qwen2.5:32b",
    "math": "qwen2.5:32b",
    "lhe": "qwen2.5:32b",
    "arc": "qwen2.5:32b",
}

MODEL_OPTIONS: dict[str, dict[str, Any]] = {
    "qwen3:8b": {
        "temperature": 0.1,
        "num_predict": 4096,   # thinking + content share the budget
    },
    "qwen2.5:32b": {
        "temperature": 0.3,
        "num_predict": 2048,
    },
}

SYSTEM_PROMPTS: dict[str, str] = {
    "mmlu": (
        "You answer multiple-choice benchmark questions. "
        "Use reference facts if provided. "
        "Write ONLY the final letter A, B, C, or D on the last line."
    ),
    "gsm8k": (
        "You solve grade-school math word problems carefully. "
        "Use reference facts if provided. "
        "Solve step by step. Last line: ONLY the number."
    ),
    "math": (
        "You solve mathematics competition problems carefully. "
        "Use reference facts if provided. "
        "Show concise reasoning. Last line: ONLY the final answer."
    ),
    "lhe": (
        "You solve difficult reasoning questions. "
        "Use reference facts if provided. "
        "Reason carefully and keep the final answer explicit on the last line."
    ),
    "arc": (
        "You solve ARC-style grid transformation tasks. "
        "Use reference facts if provided. "
        "Output ONLY the JSON grid."
    ),
}


def get_model_for_suite(suite: str) -> str:
    return SUITE_MODEL_MAP.get(str(suite or "").strip().lower(), "qwen2.5:32b")


def _contains_keyword(text: str, keyword: str) -> bool:
    pattern = rf"(?<!\w){re.escape(keyword.lower())}(?!\w)"
    return re.search(pattern, text.lower()) is not None


def _normalize_search_text(subject: str, question: str) -> str:
    combined = f"{subject} {question}".lower()
    return combined.replace("_", " ").replace("-", " ")


def _mentions(subject: str, question: str, keywords: set[str]) -> bool:
    combined = _normalize_search_text(subject, question)
    return any(_contains_keyword(combined, keyword) for keyword in keywords)


def _mentions_chemistry(subject: str, question: str) -> bool:
    return _mentions(
        subject,
        question,
        {
            "chemistry",
            "element",
            "periodic",
            "atom",
            "molecule",
            "compound",
            "electron",
            "proton",
            "neutron",
            "ion",
            "oxide",
            "acid",
            "base",
            "metal",
            "halogen",
            "noble gas",
        },
    )


def _mentions_physics(subject: str, question: str) -> bool:
    return _mentions(
        subject,
        question,
        {
            "physics",
            "force",
            "energy",
            "velocity",
            "acceleration",
            "gravity",
            "speed of light",
            "planck",
            "boltzmann",
            "newton",
            "electric",
            "magnetic",
            "thermo",
            "momentum",
            "wavelength",
            "frequency",
        },
    )


def _mentions_math(subject: str, question: str) -> bool:
    return _mentions(
        subject,
        question,
        {
            "math",
            "mathematics",
            "algebra",
            "calculus",
            "geometry",
            "trigonometry",
            "probability",
            "statistics",
            "equation",
            "integral",
            "derivative",
            "matrix",
            "vector",
            "unit",
            "convert",
            "distance",
            "speed",
            "length",
            "mass",
            "temperature",
            "pressure",
        },
    )


def _chemistry_context(question: str) -> str:
    q_lower = str(question or "").lower()
    matches: list[str] = []
    for element in iter_elements():
        if element.name_en.lower() in q_lower or element.symbol.lower() in q_lower:
            matches.append(
                f"{element.name_en} ({element.symbol}): atomic_number={element.atomic_number}, "
                f"atomic_mass={element.atomic_mass}, category={element.category}, period={element.period}, group={element.group}"
            )
        if len(matches) >= 8:
            break
    if not matches:
        first_elements = list(iter_elements())[:8]
        matches = [
            f"{element.name_en} ({element.symbol}): atomic_number={element.atomic_number}, atomic_mass={element.atomic_mass}"
            for element in first_elements
        ]
    return "Periodic table facts:\n" + "\n".join(f"- {line}" for line in matches)


def _physics_context() -> str:
    facts = [
        f"{constant.name} ({constant.symbol}) = {constant.value} {constant.unit}"
        for constant in iter_physical_constants()
    ]
    return "Physical constants:\n" + "\n".join(f"- {line}" for line in facts)


def _math_context(question: str) -> str:
    q_lower = str(question or "").lower()
    lines: list[str] = []
    for domain in iter_domains():
        domain_hit = domain.key.replace("_", " ") in q_lower or domain.key in q_lower
        unit_hit = any(
            name.replace("_", " ") in q_lower
            or definition.symbol.lower() in q_lower
            for name, definition in domain.units.items()
        )
        if domain_hit or unit_hit:
            sample_units = list(domain.units.items())[:6]
            unit_text = ", ".join(
                f"{name}({definition.symbol})" for name, definition in sample_units
            )
            lines.append(f"{domain.key}: SI base={domain.si_base}; units={unit_text}")
    if not lines:
        for domain in list(iter_domains())[:5]:
            sample_units = list(domain.units.items())[:4]
            unit_text = ", ".join(
                f"{name}({definition.symbol})" for name, definition in sample_units
            )
            lines.append(f"{domain.key}: SI base={domain.si_base}; units={unit_text}")
    return "Measurement and conversion references:\n" + "\n".join(f"- {line}" for line in lines)


def build_rag_context(row: dict[str, Any], suite: str) -> str:
    payload = dict(row.get("payload") or {})
    subject = str(payload.get("subject") or row.get("subject") or "").strip().lower()
    question = str(row.get("question") or payload.get("question_text") or "").strip()
    parts: list[str] = []
    if _mentions_chemistry(subject, question):
        parts.append(_chemistry_context(question))
    if _mentions_physics(subject, question):
        parts.append(_physics_context())
    if _mentions_math(subject, question) or str(suite).strip().lower() in {"gsm8k", "math"}:
        parts.append(_math_context(question))
    if not parts:
        return ""
    return "=== REFERENCE FACTS (use these to ground your answer) ===\n" + "\n\n".join(parts)


_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", str(text or "")).strip()


def _extract_mmlu_letter(text: str) -> str:
    raw = _strip_thinking(text)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        match = re.fullmatch(r"(?:answer\s*[:\-]?\s*)?([A-D])", line.upper())
        if match:
            return match.group(1)
    match = re.search(r"\b([A-D])\b", raw.upper())
    return match.group(1) if match else raw.strip()


def _extract_last_number(text: str) -> str:
    raw = _strip_thinking(text)
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", raw)
    return numbers[-1].replace(",", "") if numbers else raw.strip()


def _extract_json_grid(text: str) -> str:
    raw = _strip_thinking(text)
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    candidates = fenced + [raw]
    for candidate in candidates:
        start = candidate.find("[")
        if start < 0:
            continue
        snippet = candidate[start:].strip()
        try:
            return json.dumps(json.loads(snippet))
        except json.JSONDecodeError:
            continue
    return raw.strip()


def extract_answer(text: str, suite: str) -> str:
    canonical = str(suite or "").strip().lower()
    if canonical == "mmlu":
        return _extract_mmlu_letter(text)
    if canonical in {"gsm8k", "math", "lhe"}:
        return _extract_last_number(text)
    if canonical == "arc":
        return _extract_json_grid(text)
    return _strip_thinking(text)


def _build_user_message(row: dict[str, Any], suite: str, rag_context: str) -> str:
    payload = dict(row.get("payload") or {})
    question = str(row.get("question") or "").strip()
    lines: list[str] = []
    if rag_context:
        lines.append(rag_context)
        lines.append("")
    if suite == "mmlu":
        lines.append("Question:")
        lines.append(question)
        options = payload.get("options")
        if isinstance(options, list) and options:
            lines.append("")
            for index, option in enumerate(options):
                lines.append(f"{chr(ord('A') + index)}. {option}")
    else:
        lines.append(question)
    return "\n".join(lines).strip()


def create_ollama_query_fn(
    *,
    ollama: OllamaModelManager | None = None,
    timeout: float = 120.0,
    medium_model: str = "qwen3:8b",
    large_model: str = "qwen2.5:32b",
) -> OllamaQueryFn:
    manager = ollama or OllamaModelManager(default_timeout=timeout)
    model_map = dict(SUITE_MODEL_MAP)
    model_map["mmlu"] = medium_model
    for suite in ("gsm8k", "math", "lhe", "arc"):
        model_map[suite] = large_model

    def _query(row: dict[str, Any]) -> dict[str, Any]:
        suite = str(row.get("suite") or "").strip().lower()
        model = model_map.get(suite, large_model)
        rag_context = build_rag_context(row, suite)
        user_message = _build_user_message(row, suite, rag_context)
        options = dict(MODEL_OPTIONS.get(get_model_for_suite(suite), {}))
        if model == medium_model and medium_model != get_model_for_suite(suite):
            options = dict(MODEL_OPTIONS.get("qwen3:8b", {}))
        elif model == large_model and large_model != get_model_for_suite(suite):
            options = dict(MODEL_OPTIONS.get("qwen2.5:32b", {}))
        temperature = float(options.pop("temperature", 0.3))
        result = manager.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS.get(suite, "Answer the question accurately.")},
                {"role": "user", "content": user_message},
            ],
            timeout=timeout,
            temperature=temperature,
            options=options,
        )
        cleaned = _strip_thinking(result.output)
        answer = extract_answer(cleaned, suite)
        return {
            "answer": answer,
            "provider": "ollama",
            "model": model,
            "suite": suite,
            "used_rag": bool(rag_context),
            "raw_response": cleaned,
            "source": "provider",
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    return _query


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="arc, gsm8k, math, lhe, or mmlu")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("../Knowledge3D.local/logs/health_log.jsonl"),
    )
    parser.add_argument("--medium-model", default="qwen3:8b")
    parser.add_argument("--large-model", default="qwen2.5:32b")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-query timeout in seconds")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from knowledge3d.tools.benchmark_health_check import run_health_check

    args = _parse_args(argv)
    query_fn = create_ollama_query_fn(
        medium_model=args.medium_model,
        large_model=args.large_model,
        timeout=args.timeout,
    )
    summary = run_health_check(args.suite, args.count, args.log, query_fn=query_fn)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
