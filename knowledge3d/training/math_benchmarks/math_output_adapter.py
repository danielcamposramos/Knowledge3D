"""
Convert sovereign RPN results to benchmark-specific outputs.
"""

from __future__ import annotations

import json
from typing import Any, Dict
import math


class MathOutputAdapter:
    """
    Formats answers for different math benchmarks.
    """

    def __init__(self):
        self._results: Dict[str, Dict[str, Any]] = {}

    def record_result(self, problem_id: str, rpn_stack: Any, source: str) -> None:
        raw_answer = rpn_stack[-1] if rpn_stack else None
        formatted = self._format_for_benchmark(raw_answer, source)
        self._results[problem_id] = {
            "raw_answer": raw_answer,
            "formatted_answer": formatted,
            "source": source,
        }

    def _format_for_benchmark(self, answer: Any, source: str) -> str:
        if answer is None:
            return ""
        if isinstance(answer, float) and (math.isinf(answer) or math.isnan(answer)):
            return ""
        if source == "gsm8k":
            return str(self._to_number(answer))
        if source == "math":
            return f"\\boxed{{{answer}}}"
        if source == "mmlu":
            return self._to_letter_choice(answer)
        if source == "omni_math":
            return str(answer)
        if source == "amc_aime":
            num = self._to_number(answer)
            if num is not None:
                return str(int(num) % 1000)
            return str(answer)
        return str(answer)

    def _to_number(self, value: Any):
        if isinstance(value, (int, float)):
            f = float(value)
            if math.isinf(f) or math.isnan(f):
                return None
            return f
        if isinstance(value, str):
            try:
                f = float(value.replace(",", ""))
                if math.isinf(f) or math.isnan(f):
                    return None
                return f
            except ValueError:
                return None
        return None

    def _to_letter_choice(self, value: Any) -> str:
        if isinstance(value, str) and value.upper() in "ABCD":
            return value.upper()
        if isinstance(value, int) and 0 <= value <= 3:
            return "ABCD"[value]
        return "A"

    def to_submission_format(self, benchmark: str) -> str:
        results = {pid: data["formatted_answer"] for pid, data in self._results.items() if data["source"] == benchmark or benchmark == "all"}
        return json.dumps(results, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        from collections import Counter

        sources = Counter(d["source"] for d in self._results.values())
        return {"total_recorded": len(self._results), "by_source": dict(sources)}


__all__ = ["MathOutputAdapter"]
