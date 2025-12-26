"""
TRM Math Navigator - infrastructure for navigating Galaxy Universe during math solving.

This module deliberately avoids hardcoding "solvers". Instead, it provides a
pluggable routing + execution loop where:
- knowledge lives in galaxies (MathSymbolGalaxy + GrammarRule banks)
- a TRM-like engine ranks candidates and decides what to execute
- execution is performed by the sovereign PTX-backed RPN engine
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RuleMatch:
    rule: Any
    match: re.Match[str]
    score: float


class HeuristicTRMMathEngine:
    """
    Minimal TRM-like interface for ranking + validation.

    This is intentionally lightweight and deterministic. When a real TRM engine
    is provided, it can replace these methods while keeping the same interface.
    """

    def rank_rules(self, matches: Sequence[RuleMatch], problem_text: str) -> List[RuleMatch]:
        _ = problem_text
        return sorted(matches, key=lambda m: (-m.score, getattr(m.rule, "rule_id", "")))

    def validate_result(self, result: Any, problem_text: str) -> float:
        _ = problem_text
        if result is None:
            return 0.0
        try:
            val = float(result)
            if val != val:  # NaN
                return 0.0
            return 0.9
        except Exception:
            return 0.6

    def enhance_adapter(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def embed(self, text: str) -> str:
        return text

    def compose_from_symbols(self, symbols: Sequence[Any], problem_text: str) -> str:
        _ = problem_text
        # Best-effort fallback: attempt direct symbol name execution if it is an opcode.
        for sym in symbols:
            tmpl = getattr(sym, "rpn_template", "") or ""
            if tmpl.strip():
                return tmpl
        return ""


class TRMMathNavigator:
    """
    Navigate Galaxy rules/symbols to solve math problems.

    The navigator is a thin orchestration layer: it queries a rule bank,
    asks the TRM engine to rank candidates, composes an RPN program, executes
    on the sovereign RPN engine, and reports back metadata.
    """

    def __init__(
        self,
        *,
        rule_bank: Sequence[Any],
        math_galaxy: Any,
        rpn_engine: Any,
        trm_engine: Optional[Any] = None,
        shadow_copy: Optional[Any] = None,
        galaxy_reader: Optional[Any] = None,
        record_on_confidence: bool = True,
    ) -> None:
        self.rule_bank = list(rule_bank)
        self.math_galaxy = math_galaxy
        self.engine = rpn_engine
        self.trm = trm_engine or HeuristicTRMMathEngine()
        self.shadow = shadow_copy
        self.galaxy_reader = galaxy_reader
        self.record_on_confidence = bool(record_on_confidence)

    def query_matches(self, problem_text: str) -> List[RuleMatch]:
        matches: List[RuleMatch] = []
        for rule in self.rule_bank:
            pattern = getattr(rule, "pattern", None)
            if not isinstance(pattern, str) or not pattern:
                continue
            try:
                m = re.search(pattern, problem_text, re.IGNORECASE | re.DOTALL)
            except re.error:
                continue
            if not m:
                continue
            score = self._score_match(rule, m, problem_text)
            matches.append(RuleMatch(rule=rule, match=m, score=score))
        return matches

    def solve(self, problem_text: str) -> Tuple[Any, Dict[str, Any]]:
        # Preferred path: read via Galaxy (Word → Grammar word_sequence rules),
        # with simple course correction (try a few composition templates).
        if self.galaxy_reader is not None:
            try:
                result, meta = self.galaxy_reader.solve(
                    problem_text=problem_text,
                    rpn_engine=self.engine,
                    max_attempts=3,
                )
                if result is not None:
                    rpn_program = str(meta.get("rpn_program") or "")
                    confidence = self.trm.validate_result(result, problem_text)
                    return (
                        result,
                        {
                            "rule_used": "galaxy_read",
                            "rpn_program": rpn_program,
                            "confidence": confidence,
                            "read_trace": meta.get("read_trace", {}),
                            "read_understanding": meta.get("read_understanding", {}),
                            "read_composition": meta.get("read_composition", {}),
                            "attempts": meta.get("attempts", []),
                            "template_used": meta.get("template_used", ""),
                            "subgoals": meta.get("subgoals", []),
                            "exploration": meta.get("exploration", {}),
                            "test_time": meta.get("test_time", {}),
                        },
                    )
            except Exception:
                pass

        matched = self.query_matches(problem_text)
        ranked = self.trm.rank_rules(matched, problem_text) if matched else []
        best = ranked[0] if ranked else None

        if best is None:
            return self._semantic_fallback(problem_text)

        rpn_program = self._compose_rpn(best.rule, best.match)
        result = None
        error = None
        if rpn_program:
            try:
                result = self.engine.evaluate(rpn_program)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                result = None

        confidence = self.trm.validate_result(result, problem_text)
        if self.record_on_confidence and confidence > 0.8:
            self._record_success(
                rule=best.rule,
                rpn_program=rpn_program,
                result=result,
                problem_text=problem_text,
                confidence=confidence,
            )

        return (
            result,
            {
                "rule_used": getattr(best.rule, "rule_id", None),
                "rpn_program": rpn_program,
                "confidence": confidence,
                "error": error,
            },
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _record_success(
        self,
        *,
        rule: Any,
        rpn_program: str,
        result: Any,
        problem_text: str,
        confidence: float,
    ) -> None:
        """
        Record successful solve to DualShadowCopy for TRM learning.

        This is deliberately post-inference (after PTX execution returns) and is
        allowed to do book-keeping / persistence separate from the hot path.
        """
        if self.shadow is None:
            return

        rule_id = getattr(rule, "rule_id", "unknown")
        domain = getattr(rule, "domain", "math_general")

        task_signature = {
            "problem_text": (problem_text or "")[:200],
            "rule_id": rule_id,
            "result": str(result),
            "problem_type": domain,
        }

        try:
            self.shadow.record(
                task_signature=task_signature,
                program=rpn_program,
                program_type="math",
                score=float(confidence),
                task_id=f"math_{hash(problem_text) % 10000}",
            )
        except Exception:
            return

        try:
            if rule_id:
                self.shadow.update_pattern_confidence(rule_id, float(confidence))
        except Exception:
            pass

    def _compose_rpn(self, rule: Any, match: re.Match[str]) -> str:
        rpn_program = getattr(rule, "rpn_program", "")
        if callable(rpn_program):
            try:
                return str(rpn_program(match))
            except Exception:
                return ""
        if not isinstance(rpn_program, str):
            return ""
        out = rpn_program
        for idx, group in enumerate(match.groups()):
            out = out.replace(f"{{g{idx}}}", str(group))
            out = out.replace(f"{{{idx}}}", str(group))
        return out

    def _score_match(self, rule: Any, match: re.Match[str], problem_text: str) -> float:
        _ = problem_text
        rule_id = getattr(rule, "rule_id", "")
        domain = getattr(rule, "domain", "")
        score = 0.0
        score += 0.01 * float(len(getattr(rule, "pattern", "") or ""))
        score += 0.1 * float(len(match.groups()))
        if "gsm" in rule_id:
            score += 0.3
        if domain.startswith("math_"):
            score += 0.2
        if match.group(0):
            score += 0.05 * float(min(40, len(match.group(0))))
        return score

    def _semantic_fallback(self, problem_text: str) -> Tuple[Any, Dict[str, Any]]:
        try:
            query = self.trm.embed(problem_text)
        except Exception:
            query = problem_text

        try:
            symbols = self.math_galaxy.query_semantic(str(query), k=5)
        except Exception:
            symbols = []

        try:
            rpn_program = self.trm.compose_from_symbols(symbols, problem_text)
        except Exception:
            rpn_program = ""

        result = None
        error = None
        if rpn_program:
            try:
                result = self.engine.evaluate(rpn_program)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

        confidence = self.trm.validate_result(result, problem_text)
        return (
            result,
            {
                "rule_used": None,
                "rpn_program": rpn_program,
                "confidence": confidence,
                "error": error,
            },
        )


__all__ = ["TRMMathNavigator", "HeuristicTRMMathEngine", "RuleMatch"]
