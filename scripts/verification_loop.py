#!/usr/bin/env python3
"""
Verification loop for Phase 5.1 confidence calibration.
"""

from __future__ import annotations

import contextlib
import io
from typing import List

from knowledge3d.training.math_benchmarks.navigation_model import CONTROL_TOKENS
from knowledge3d.training.math_benchmarks.recursive_solver import RecursiveSolver


class VerificationLoop:
    """Generate correctness labels via symbolic verification."""

    def __init__(self, *, verbose: bool = False, quiet: bool = False):
        self._solver = RecursiveSolver(verbose=verbose)
        self._quiet = bool(quiet)

    def verify_rule_sequence(self, problem_text: str, predicted_rules: List[str]) -> List[int]:
        filtered = [r for r in predicted_rules if r not in CONTROL_TOKENS]
        if not filtered:
            return []

        if self._quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                result = self._solver.solve(problem_text)
                trace = self._solver.get_last_trace() if result is not None else {}
        else:
            result = self._solver.solve(problem_text)
            trace = self._solver.get_last_trace() if result is not None else {}
        if result is None:
            return [0] * len(filtered)
        step_sequence = trace.get("step_sequence") or []
        actual_rules = [step.get("rule") for step in step_sequence if step.get("rule")]

        correctness: List[int] = []
        for idx, rule in enumerate(filtered):
            if idx >= len(actual_rules):
                correctness.append(0)
                continue
            if rule == actual_rules[idx]:
                correctness.append(1)
            else:
                correctness.append(0)
                correctness.extend([0] * (len(filtered) - len(correctness)))
                break
        return correctness


__all__ = ["VerificationLoop"]
