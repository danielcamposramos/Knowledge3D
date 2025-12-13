"""
Evaluate sovereign math reasoning against benchmarks.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


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
        try:
            pred_num = float(str(predicted).replace(",", ""))
            truth_num = float(str(truth).replace(",", ""))
            if abs(pred_num - truth_num) < self._tolerance:
                return True, "exact"
            if truth_num != 0:
                rel_diff = abs(pred_num - truth_num) / abs(truth_num)
                if rel_diff < 1e-4:
                    return True, "numerical"
            return False, "none"
        except (ValueError, TypeError):
            return str(predicted) == str(truth), "string"

    def _evaluate_math(self, predicted: Any, truth: Any):
        pred_str = str(predicted)
        truth_str = str(truth)
        pred_str = self._strip_box(pred_str)
        truth_str = self._strip_box(truth_str)
        pred_norm = self._normalize_latex(pred_str)
        truth_norm = self._normalize_latex(truth_str)
        if pred_norm == truth_norm:
            return True, "exact"
        try:
            pred_num = float(pred_norm)
            truth_num = float(truth_norm)
            if abs(pred_num - truth_num) < self._tolerance:
                return True, "numerical"
        except ValueError:
            pass
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
        try:
            pred_num = float(str(predicted))
            truth_num = float(str(truth))
            if abs(pred_num - truth_num) < self._tolerance:
                return True, "numerical"
        except ValueError:
            pass
        if str(predicted).strip() == str(truth).strip():
            return True, "exact"
        return False, "none"

    def _evaluate_amc(self, predicted: Any, truth: Any):
        try:
            pred_int = int(float(str(predicted)))
            truth_int = int(float(str(truth)))
            if pred_int % 1000 == truth_int % 1000:
                return True, "exact"
            return False, "none"
        except (ValueError, TypeError):
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
