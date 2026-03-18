"""Provider-backed benchmark query bridge for ingestion-path health checks."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from knowledge3d.tools.augmentation_providers import AugmentationResult, create_provider


BenchmarkQueryFn = Callable[[dict[str, Any]], dict[str, Any]]


def build_benchmark_prompt(row: dict[str, Any], suite: str) -> str:
    """Build a suite-aware prompt for generic provider-backed evaluation."""
    question = str(row.get("question") or "").strip()
    payload = dict(row.get("payload") or {})
    canonical = str(suite or row.get("suite") or "").strip().lower()
    lines = [question]
    if canonical == "mmlu":
        options = payload.get("options")
        if isinstance(options, list) and options:
            lines.append("")
            for index, option in enumerate(options):
                lines.append(f"{chr(ord('A') + index)}. {option}")
            lines.append("")
            lines.append("Reply with ONLY the final letter A, B, C, or D.")
    elif canonical == "gsm8k":
        lines.append("")
        lines.append("Solve step by step. Final line: ONLY the number.")
    elif canonical == "arc":
        lines.append("")
        lines.append("Reply with ONLY the JSON grid.")
    else:
        lines.append("")
        lines.append("Answer concisely.")
    return "\n".join(lines).strip()


def extract_answer(result: AugmentationResult, suite: str) -> str:
    """Extract a benchmark answer from a generic augmentation payload."""
    canonical = str(suite or "").strip().lower()
    raw = str(result.raw_response or result.summary or "").strip()
    if canonical == "mmlu":
        match = re.search(r"\b([A-D])\b", raw.upper())
        return match.group(1) if match else raw
    if canonical in {"gsm8k", "math"}:
        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", raw)
        return numbers[-1].replace(",", "") if numbers else raw
    if canonical == "arc":
        start = raw.find("[")
        if start >= 0:
            candidate = raw[start:]
            try:
                parsed = json.loads(candidate)
                return json.dumps(parsed)
            except json.JSONDecodeError:
                return candidate
    return raw


def create_provider_query_fn(provider_name: str = "auto", **provider_kwargs: Any) -> BenchmarkQueryFn:
    """Wrap the augmentation provider interface as a benchmark query function."""
    provider = create_provider(provider_name, **provider_kwargs)

    def _query(row: dict[str, Any]) -> dict[str, Any]:
        suite = str(row.get("suite") or "").strip().lower()
        prompt = build_benchmark_prompt(row, suite)
        context = {
            "name": row.get("id") or row.get("question_id") or "benchmark_question",
            "path": f"benchmark://{suite}/{row.get('id') or 'question'}",
            "domain_hint": str(suite or "General").title(),
        }
        result = provider.augment(prompt, context)
        answer = extract_answer(result, suite)
        return {
            "answer": answer,
            "provider": result.provider,
            "suite": suite,
            "source": "provider",
            "raw_response": result.raw_response,
        }

    return _query


__all__ = [
    "BenchmarkQueryFn",
    "build_benchmark_prompt",
    "create_provider_query_fn",
    "extract_answer",
]
