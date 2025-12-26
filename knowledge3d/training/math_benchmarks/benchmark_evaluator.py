"""
Evaluate sovereign math reasoning against benchmarks.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _safe_float(s: Any) -> Optional[float]:
    """Safely convert string to float, handling malformed numbers like '2.7.' or '2...'."""
    import math
    try:
        text = str(s).replace(",", "").strip()
        # Remove trailing dots
        while text.endswith("."):
            text = text[:-1]
        # Remove multiple consecutive dots
        while ".." in text:
            text = text.replace("..", ".")
        # Remove leading dots (except "0.X" patterns)
        if text.startswith("."):
            text = "0" + text
        if not text:
            return None
        val = float(text)
        if math.isinf(val) or math.isnan(val):
            return None
        return val
    except (ValueError, TypeError, OverflowError):
        return None


class MathBenchmarkEvaluator:
    """
    Evaluate predictions with benchmark-specific rules.
    """

    def __init__(self, tolerance: float = 1e-6):
        self._tolerance = tolerance
        self._results: List[Dict[str, Any]] = []

    def evaluate(self, problem_id: str, predicted: Any, ground_truth: Any, source: str) -> Dict[str, Any]:
        correct = False
        match_type = "none"

        if source == "gsm8k":
            correct, match_type = self._evaluate_gsm8k(predicted, ground_truth)
        elif source == "math":
            correct, match_type = self._evaluate_math(predicted, ground_truth)
        elif source == "mmlu":
            correct, match_type = self._evaluate_mmlu(predicted, ground_truth)
        elif source == "omni_math":
            correct, match_type = self._evaluate_omni(predicted, ground_truth)
        elif source == "amc_aime":
            correct, match_type = self._evaluate_amc(predicted, ground_truth)

        result = {
            "problem_id": problem_id,
            "correct": correct,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "source": source,
            "match_type": match_type,
        }
        self._results.append(result)
        return result

    def _evaluate_gsm8k(self, predicted: Any, truth: Any):
        pred_num = _safe_float(predicted)
        # GSM8K answers are solution text with "#### number" at the end
        truth_str = str(truth)
        truth_num = None
        # Try to extract #### answer
        hash_match = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", truth_str)
        if hash_match:
            truth_num = _safe_float(hash_match.group(1))
        else:
            # Fallback: try last number in text
            numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", truth_str)
            if numbers:
                truth_num = _safe_float(numbers[-1])
            else:
                truth_num = _safe_float(truth_str)
        if pred_num is None or truth_num is None:
            return str(predicted) == str(truth), "string"
        if abs(pred_num - truth_num) < self._tolerance:
            return True, "exact"
        if truth_num != 0:
            rel_diff = abs(pred_num - truth_num) / abs(truth_num)
            if rel_diff < 1e-4:
                return True, "numerical"
        return False, "none"

    def _evaluate_math(self, predicted: Any, truth: Any):
        pred_str = str(predicted)
        truth_str = str(truth)
        pred_str = self._strip_box(pred_str)
        truth_str = self._strip_box(truth_str)
        pred_norm = self._normalize_latex(pred_str)
        truth_norm = self._normalize_latex(truth_str)
        if pred_norm == truth_norm:
            return True, "exact"
        pred_num = _safe_float(pred_norm)
        truth_num = _safe_float(truth_norm)
        if pred_num is not None and truth_num is not None:
            if abs(pred_num - truth_num) < self._tolerance:
                return True, "numerical"
        return False, "none"

    def _strip_box(self, s: str) -> str:
        boxed = re.search(r"\\boxed\{([^}]+)\}", s)
        return boxed.group(1) if boxed else s

    def _normalize_latex(self, latex: str) -> str:
        s = re.sub(r"\\(left|right|big|Big)", "", latex)
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
        return s.lower()

    def _evaluate_mmlu(self, predicted: Any, truth: Any):
        pred_letter = str(predicted).strip().upper()[:1]
        truth_letter = str(truth).strip().upper()[:1]
        if pred_letter == truth_letter:
            return True, "exact"
        return False, "none"

    def _evaluate_omni(self, predicted: Any, truth: Any):
        pred_num = _safe_float(predicted)
        truth_num = _safe_float(truth)
        if pred_num is not None and truth_num is not None:
            if abs(pred_num - truth_num) < self._tolerance:
                return True, "numerical"
        if str(predicted).strip() == str(truth).strip():
            return True, "exact"
        return False, "none"

    def _evaluate_amc(self, predicted: Any, truth: Any):
        pred_f = _safe_float(predicted)
        truth_f = _safe_float(truth)
        if pred_f is None or truth_f is None:
            return False, "none"
        try:
            pred_int = int(pred_f)
            truth_int = int(truth_f)
            if pred_int % 1000 == truth_int % 1000:
                return True, "exact"
            return False, "none"
        except (ValueError, TypeError, OverflowError):
            return False, "none"

    def get_metrics(self) -> Dict[str, Any]:
        from collections import defaultdict

        by_source = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in self._results:
            src = r["source"]
            by_source[src]["total"] += 1
            if r["correct"]:
                by_source[src]["correct"] += 1

        metrics = {
            "overall": {
                "correct": sum(1 for r in self._results if r["correct"]),
                "total": len(self._results),
                "accuracy": (sum(1 for r in self._results if r["correct"]) / len(self._results)) if self._results else 0.0,
            },
            "by_source": {},
        }
        for source, data in by_source.items():
            acc = data["correct"] / data["total"] if data["total"] > 0 else 0.0
            metrics["by_source"][source] = {"correct": data["correct"], "total": data["total"], "accuracy": acc}
        return metrics


__all__ = ["MathBenchmarkEvaluator"]
