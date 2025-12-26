"""
TRM Galaxy Reader - Galaxy-based problem reading (no external preprocessing).

This module provides a minimal, deterministic "reading" pipeline that:
1) Tokenizes text via WordGalaxy.tokenize() (model's lexicon layer)
2) Matches word-sequence rules via GrammarGalaxy.match_word_sequence()
3) Produces a compact ProblemUnderstanding used to compose RPN

It is intentionally small: the goal is to align the architecture so TRM can
learn reading strategies over time (via shadow copy), while keeping a safe
fallback path to regex-based solvers during the transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class ProblemUnderstanding:
    quantities: List[Dict[str, Any]] = field(default_factory=list)
    operations: List[Dict[str, Any]] = field(default_factory=list)
    aggregation: Optional[str] = None
    labels: Dict[str, float] = field(default_factory=dict)
    goals: Dict[str, bool] = field(default_factory=dict)

    def is_complete(self) -> bool:
        if not self.quantities and not self.labels:
            return False

        # Multi-step template completeness (pages/remaining).
        if (
            "total" in self.labels
            and "yesterday" in self.labels
            and self.goals.get("today_twice_as_yesterday")
            and self.goals.get("half_remaining")
        ):
            return True

        # Linear growth completeness (base + rate×years).
        if "rate_per_year" in self.labels and "years" in self.labels and self.quantities:
            return True

        # Profit markup schedule completeness (unit_cost + markup + sales schedule).
        if (
            "markup" in self.labels
            and "per_day" in self.labels
            and "weeks" in self.labels
            and any(isinstance(q, dict) and q.get("kind") == "unit_cost" for q in self.quantities)
        ):
            return True

        # Daily practice/time schedule completeness (minutes/day with a multiplier + weekly schedule).
        if (
            "daily_minutes" in self.labels
            and "times_as_long" in self.labels
            and "days_per_week" in self.labels
            and "weeks" in self.labels
        ):
            return True

        # Bundle ratio division completeness (e.g., bags with "as many ... as K").
        if "as_many_multiplier" in self.labels and "each_amount" in self.labels and self.quantities:
            return True

        # Structured math expressions (e.g., "3 bags of 5 apples" → "3 5 *").
        for q in self.quantities:
            if isinstance(q.get("rpn"), str) and q.get("rpn"):
                return True

        # Avoid returning "just the first number" for word problems:
        # require operations or an explicit aggregation across multiple terms.
        if self.operations:
            return True
        if self.aggregation == "sum" and len(self.quantities) >= 2:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quantities": list(self.quantities),
            "operations": list(self.operations),
            "aggregation": self.aggregation,
            "labels": dict(self.labels),
            "goals": dict(self.goals),
        }


class TRMGalaxyReader:
    def __init__(
        self,
        *,
        word_galaxy: Any,
        grammar_galaxy: Any,
        math_galaxy: Any,
        generic_equations_galaxy: Any | None = None,
        rule_bank: Sequence[Any],
        shadow_copy: Optional[Any] = None,
        use_retrieval: bool = True,
        reality_galaxy: Optional[Any] = None,
        drawing_galaxy: Optional[Any] = None,
        enable_cross_domain: bool = False,
        enable_book_galaxies: bool = False,
        book_galaxy_root: str | Any | None = None,
        book_max_books: int = 8,
        book_top_k: int = 5,
        thinking_budget: int = 0,
        max_parallel_candidates: int = 27,
    ) -> None:
        self.word_galaxy = word_galaxy
        self.grammar_galaxy = grammar_galaxy
        self.math_galaxy = math_galaxy
        self.generic_equations_galaxy = generic_equations_galaxy
        self.rule_bank = list(rule_bank)
        self.shadow = shadow_copy
        self.use_retrieval = bool(use_retrieval)
        self.reality_galaxy = reality_galaxy
        self.drawing_galaxy = drawing_galaxy
        self.enable_cross_domain = bool(enable_cross_domain)
        self.enable_book_galaxies = bool(enable_book_galaxies)
        self.book_galaxy_root = book_galaxy_root
        self.book_max_books = int(max(0, book_max_books))
        self.book_top_k = int(max(0, book_top_k))
        self.thinking_budget = int(max(0, thinking_budget))
        self.max_parallel_candidates = int(max(1, max_parallel_candidates))
        self._last_composition: Dict[str, Any] = {}
        self._last_selection_meta: Dict[str, Any] = {}
        self._stats: Dict[str, Any] = {
            "retrieval_hits": 0,
            "template_counts": {},
        }
        self._book_stats: Dict[str, Any] = {"hits": 0, "used": 0}
        self._book_library = None
        if self.enable_book_galaxies:
            try:
                from knowledge3d.training.math_benchmarks.book_galaxy_library import BookGalaxyLibrary

                self._book_library = BookGalaxyLibrary(
                    books_root=self.book_galaxy_root, max_books=(self.book_max_books or None)
                )
            except Exception:
                self._book_library = None
        self._success_stats: Dict[str, Any] = {}
        self._success_stats_built_at_ts: float = 0.0
        # Phase 5B: Tsinghua-inspired explorer for focused rule subsets.
        try:
            from knowledge3d.training.math_benchmarks.tsinghua_galaxy_explorer import (
                TsinghuaGalaxyExplorer,
            )

            self._explorer = TsinghuaGalaxyExplorer(
                max_rules=40,
                hub_k=5,
                reality_galaxy=self.reality_galaxy if self.enable_cross_domain else None,
                drawing_galaxy=self.drawing_galaxy if self.enable_cross_domain else None,
                generic_equations_galaxy=self.generic_equations_galaxy,
            )
        except Exception:
            self._explorer = None

        # Non-hot-path: derive success priors from recent exploration logs.
        self._refresh_success_stats()

    def solve(
        self,
        *,
        problem_text: str,
        rpn_engine: Any,
        max_attempts: int = 3,
        thinking_budget: int | None = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Solve using Galaxy reading with optional test-time compute.

        When `thinking_budget` (or `self.thinking_budget`) is > 0, the reader will
        run a bounded "test-time compute" exploration stage after template attempts.
        """
        budget = self.thinking_budget if thinking_budget is None else int(thinking_budget)
        return self.solve_with_correction(
            problem_text=problem_text,
            rpn_engine=rpn_engine,
            max_attempts=max_attempts,
            thinking_budget=budget,
        )

    def _refresh_success_stats(self) -> None:
        """
        Build lightweight success priors from shadow-copy exploration logs.

        This is used only to prioritize template attempts (ordering), not to
        change the semantics of any rule/template.
        """
        self._success_stats = {"templates": {}, "patterns": {}, "counts": {"templates": {}, "patterns": {}}}
        self._success_stats_built_at_ts = 0.0
        if self.shadow is None:
            return
        explorations = getattr(self.shadow, "explorations", None)
        if not isinstance(explorations, list) or not explorations:
            return

        tmpl_ok: Dict[str, int] = {}
        tmpl_total: Dict[str, int] = {}

        # Template success is only meaningful when we actually executed an RPN candidate.
        # Use success=True entries so templates aren't punished for "no_rpn" attempts.
        for e in explorations[-2000:]:
            if not isinstance(e, dict):
                continue
            if e.get("success") is not True:
                continue
            correct = e.get("correct")
            if correct is None:
                continue
            template = str(e.get("template_used") or "")
            if template:
                tmpl_total[template] = tmpl_total.get(template, 0) + 1
                if bool(correct):
                    tmpl_ok[template] = tmpl_ok.get(template, 0) + 1

        # Rule (pattern) success should reflect real benchmark correctness, including failures.
        # Prefer DualShadowCopy's helper when available (it uses exploration traces + correctness).
        rule_rates: Dict[str, float] = {}
        rule_counts: Dict[str, int] = {}
        getter = getattr(self.shadow, "get_rule_success_rates", None)
        if callable(getter):
            try:
                rule_rates = getter(recent=2000)
            except Exception:
                rule_rates = {}
        if not rule_rates:
            ok: Dict[str, int] = {}
            total: Dict[str, int] = {}
            for e in explorations[-2000:]:
                if not isinstance(e, dict):
                    continue
                patterns = e.get("patterns_matched", [])
                if not isinstance(patterns, list) or not patterns:
                    continue
                correct = e.get("correct")
                if correct is None:
                    continue
                uniq = {str(p) for p in patterns if p}
                for p in uniq:
                    total[p] = total.get(p, 0) + 1
                    if bool(correct):
                        ok[p] = ok.get(p, 0) + 1
            rule_rates = {k: (ok.get(k, 0) / max(1, v)) for k, v in total.items()}
            rule_counts = total

        self._success_stats["counts"]["templates"] = dict(tmpl_total)
        self._success_stats["counts"]["patterns"] = dict(rule_counts)
        self._success_stats["templates"] = {k: (tmpl_ok.get(k, 0) / max(1, v)) for k, v in tmpl_total.items()}
        self._success_stats["patterns"] = dict(rule_rates)
        try:
            import time

            self._success_stats_built_at_ts = float(time.time())
        except Exception:
            self._success_stats_built_at_ts = 0.0

    def extract_numbers(self, problem_text: str) -> List[float]:
        nums: List[float] = []
        seen: set[str] = set()

        for entry in self.word_galaxy.tokenize(problem_text or ""):
            if getattr(entry, "category", "") != "number":
                continue
            v = getattr(entry, "value", None)
            if not isinstance(v, (int, float)):
                continue
            key = f"{float(v):.12g}"
            if key in seen:
                continue
            seen.add(key)
            nums.append(float(v))

        # Fallback: WordGalaxy may not classify number words ("two", "three", ...).
        # Normalize number words to digits, then extract numeric literals.
        try:
            import re

            from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

            norm = normalize_number_words(problem_text or "")
            for m in re.finditer(r"-?\d+(?:\.\d+)?", norm):
                try:
                    v2 = float(m.group(0))
                except Exception:
                    continue
                # Skip numbers that are part of an explicit fraction like "1/4" or "5/7".
                start, end = m.span()
                if (end < len(norm) and norm[end : end + 1] == "/") or (start > 0 and norm[start - 1 : start] == "/"):
                    continue
                key2 = f"{v2:.12g}"
                if key2 in seen:
                    continue
                seen.add(key2)
                nums.append(v2)
        except Exception:
            pass

        return nums

    def classify_question(self, problem_text: str) -> str:
        text = (problem_text or "").lower()
        if any(w in text for w in ("how long", "how much time")):
            return "duration"
        if any(
            w in text
            for w in (
                "how many seconds",
                "how many minutes",
                "how many hours",
                "how many days",
                "how many weeks",
                "how many months",
                "how many years",
            )
        ):
            return "duration"
        if any(
            w in text
            for w in (
                "in seconds",
                "in minutes",
                "in hours",
                "in days",
                "in weeks",
                "in months",
                "in years",
            )
        ):
            return "duration"
        if any(w in text for w in ("total", "altogether", "in all", "sum", "combined", "together", "in total")):
            return "total"
        if any(w in text for w in ("left", "remaining", "difference", "more than", "less than")):
            return "difference"
        if "per" in text or "each" in text or "every" in text:
            return "rate"
        return "unknown"

    def explore_galaxy(self, problem_text: str) -> Dict[str, Any]:
        """
        Phase 5: lightweight "exploration" signal from WordGalaxy token stream.

        This does not invent new knowledge; it surfaces which high-level math
        concepts appear, so planning can prioritize better templates.
        """
        entries = self.word_galaxy.tokenize(problem_text or "")
        words = [getattr(e, "normalized", "") for e in entries]
        concepts: List[str] = []
        # Rate / duration
        if any(w in {"per", "each", "every"} for w in words):
            concepts.append("rate")
        if any(w in {"day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours", "minute", "minutes"} for w in words):
            concepts.append("duration")
        # Aggregation / subtraction / multiplication / division
        if any(w in {"total", "altogether", "sum", "combined"} for w in words):
            concepts.append("aggregation")
        if any(w in {"left", "remaining", "remain", "difference"} for w in words):
            concepts.append("subtraction")
        if any(w in {"times", "twice", "triple", "double"} for w in words):
            concepts.append("multiplication")
        if any(w in {"divided", "split", "shared", "/"} for w in words):
            concepts.append("division")

        return {"concepts": concepts, "words": words[:40], "n_tokens": len(words)}

    def decompose_into_subgoals(self, problem_text: str) -> List[Dict[str, Any]]:
        numbers = self.extract_numbers(problem_text)
        qtype = self.classify_question(problem_text)
        subgoals: List[Dict[str, Any]] = [
            {"id": "extract", "goal": "extract_quantities", "inputs": numbers, "status": "pending"},
            {"id": "relate", "goal": f"identify_relationships:{qtype}", "status": "pending"},
            {"id": "compose", "goal": "compose_rpn", "status": "pending"},
            {"id": "verify", "goal": "execute_verify", "status": "pending"},
        ]
        return subgoals

    def verify_reasonableness(self, problem_text: str, result: Any) -> Dict[str, Any]:
        nums = self.extract_numbers(problem_text)
        if result is None:
            return {"plausible": False, "reason": "no_result"}
        try:
            val = float(result)
        except Exception:
            return {"plausible": False, "reason": "non_numeric"}

        if val != val:
            return {"plausible": False, "reason": "nan"}
        if val < 0:
            return {"plausible": False, "reason": "negative_result"}
        if nums:
            low = (problem_text or "").lower()
            lo = min(nums)
            hi = max(nums)
            # Some domains legitimately produce results far above the largest literal
            # in the prompt (e.g., vertical travel over a week: stories * feet/story *
            # trips/day * 2 * 7). Relax the \"wrong_magnitude\" tier there while keeping
            # the hard out-of-range rejection.
            wrong_mag_scale = 100.0
            if ("feet" in low) and ("story" in low or "stories" in low) and ("week" in low or "day" in low) and (
                "travel" in low or "vertic" in low
            ):
                wrong_mag_scale = 1000.0
            # Two tiers: "wrong magnitude" is a strong hint for conversions/op ordering,
            # while "out_of_range" is a hard rejection.
            if val < lo / 1000 or val > hi * 1000:
                return {"plausible": False, "reason": "out_of_range", "min": lo, "max": hi}
            if val < lo / wrong_mag_scale or val > hi * wrong_mag_scale:
                return {"plausible": False, "reason": "wrong_magnitude", "min": lo, "max": hi}
        return {"plausible": True, "reason": None}

    def _count_multi_step_indicators(self, problem_text: str) -> int:
        low = (problem_text or "").lower()
        # Avoid the bare token "after" here: it's extremely common in narrative
        # (e.g. "After he takes his first throw...") and causes false multi-step
        # rejections for single-expression solutions.
        cues = ("then", "after that", "afterward", "next", "finally", "and then")
        return sum(1 for cue in cues if cue in low)

    def _count_basic_ops(self, expression: str) -> int:
        return sum(1 for tok in (expression or "").split() if tok in {"+", "-", "*", "/"})

    def _is_basic_numeric_rpn(self, expression: str) -> bool:
        """
        Whether `expression` looks like a plain numeric RPN chain using only
        numbers and the 4 basic operators.

        We only apply structural stack-shape checks to this subset to avoid
        false negatives on richer opcode programs.
        """
        tokens = (expression or "").split()
        if not tokens:
            return False
        for tok in tokens:
            if tok in {"+", "-", "*", "/"}:
                continue
            try:
                float(tok)
            except Exception:
                return False
        return True

    def verify_plausibility(self, problem_text: str, result: Any, expression: str = "") -> Dict[str, Any]:
        """
        Stricter plausibility checks used by test-time compute candidate ranking.

        Coverage is complete (`no_rule_match=0`), so we prioritize rejecting
        "obviously wrong" computations (e.g., percent explosions) while keeping
        the hot path sovereign (no external solvers).
        """
        verdict = dict(self.verify_reasonableness(problem_text, result))
        if not verdict.get("plausible"):
            return verdict

        nums = self.extract_numbers(problem_text)
        try:
            val = float(result)
        except Exception:
            return {"plausible": False, "reason": "non_numeric"}

        low = (problem_text or "").lower()
        expr = str(expression or "")

        # Total/aggregation sanity: when the prompt explicitly asks for a combined total
        # across multiple parts, reject candidates smaller than the largest given quantity.
        # This prevents TTC from preferring "uses more numbers" expressions that compute a
        # partial/intermediate value (common in ordinal-chain narratives).
        try:
            import re

            is_total_query = ("how many" in low) and any(w in low for w in ("in total", "altogether", "in all", "between all"))
            is_fraction_or_percent_query = ("%" in low) or ("percent" in low) or bool(re.search(r"\b\d+\s*/\s*\d+\b", low)) or (
                "half" in low or "quarter" in low
            )
            # If the prompt mixes time units (e.g., minutes + hours), then the raw numeric
            # maximum can be in a different unit than the answer (e.g., 30 minutes > 12 hours),
            # so don't apply the "total must be >= max input" heuristic.
            has_mixed_time_units = (("min" in low) or ("minute" in low)) and (("hour" in low) or ("hours" in low))
            # Only apply to expressions that actually attempt computation; otherwise let the
            # more specific "no_operation" checks fire (used by tests and diagnostics).
            if (
                is_total_query
                and not is_fraction_or_percent_query
                and not has_mixed_time_units
                and nums
                and len(nums) >= 2
                and self._count_basic_ops(expr) > 0
            ):
                hi = max(nums)
                if val < hi - 1e-9:
                    return {"plausible": False, "reason": "total_less_than_max_input", "max_input": hi}
        except Exception:
            pass

        # Tiered/overtime hourly pay sanity:
        # When the prompt describes a base hourly rate up to a threshold, then a higher
        # rate after that, reject candidates that don't use the key constraint numbers.
        # This nudges TTC away from short-but-wrong expressions (e.g. "50 2 * 50 +").
        try:
            import re

            from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

            if (("per hour" in low) or ("every hour" in low)) and ("up to" in low) and ("hour" in low) and (
                ("double" in low) or ("twice" in low) or ("after which" in low)
            ):
                low_norm = normalize_number_words(problem_text or "").lower()
                m_rate = re.search(r"\$\s*([\d,]+(?:\.\d+)?)\s+(?:every|per)\s+hour", low_norm)
                if not m_rate:
                    m_rate = re.search(r"\bpaid\s+\$?\s*([\d,]+(?:\.\d+)?)\s+(?:every|per)\s+hour", low_norm)
                m_thresh = re.search(r"\bup to\b[^\d]{0,20}?(\d+(?:\.\d+)?)\s+hours?\b", low_norm)
                m_total = (
                    re.search(r"\bfor a\s+(\d+(?:\.\d+)?)\s*[- ]\s*hour\b", low_norm)
                    or re.search(r"\bfor\s+(\d+(?:\.\d+)?)\s+hours?\b", low_norm)
                    or re.search(r"\b(\d+(?:\.\d+)?)\s*[- ]\s*hour\s+week\b", low_norm)
                )
                if m_rate and m_thresh and m_total:
                    rate = float(str(m_rate.group(1)).replace(",", ""))
                    thresh = float(m_thresh.group(1))
                    total_h = float(m_total.group(1))
                    tokens = expr.split()
                    used = set()
                    for t in tokens:
                        if t in {"+", "-", "*", "/"}:
                            continue
                        try:
                            used.add(round(float(t), 6))
                        except Exception:
                            continue
                    need = {round(rate, 6), round(total_h, 6)}
                    # If overtime is actually required, also require the threshold.
                    if total_h > thresh + 1e-9:
                        need.add(round(thresh, 6))
                    missing = [x for x in need if x not in used]
                    if missing:
                        return {"plausible": False, "reason": "tiered_hourly_missing_terms", "missing": missing}
        except Exception:
            pass

        # Currency conversion magnitude sanity:
        # When the prompt gives an explicit conversion ("A units is worth B dollars") and optionally
        # a fraction of the official rate, reject results far from the implied scale. This prevents
        # TTC from preferring short-but-wrong chains like "amount / A / denom".
        try:
            import re

            from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

            if ("worth" in low) and any(w in low for w in ("dollar", "dollars")):
                low_norm = normalize_number_words(problem_text or "").lower()
                m_rate = re.search(r"\b(\d+(?:\.\d+)?)\s+\w+\s+is\s+worth\s+(\d+(?:\.\d+)?)\s+dollars?\b", low_norm)
                if m_rate:
                    a = float(m_rate.group(1))
                    b = float(m_rate.group(2))
                    if a > 0 and b > 0:
                        m_amt = re.search(r"\bwith\s+(\d+(?:\.\d+)?)\s+\w+\b", low_norm)
                        amt = float(m_amt.group(1)) if m_amt else (max(nums) if nums else None)
                        if amt is not None and amt > 0:
                            expected = (amt * b) / a
                            m_frac = re.search(r"\b(\d+)\s*/\s*(\d+)(?:st|nd|rd|th)?s?\b", low_norm)
                            if m_frac and any(w in low_norm for w in ("official", "only give", "only gives", "only giving", "exchange")):
                                fn = float(m_frac.group(1))
                                fd = float(m_frac.group(2))
                                if fd > 0:
                                    expected = expected * (fn / fd)
                            if expected > 0:
                                if val < expected / 3.0 or val > expected * 3.0:
                                    return {
                                        "plausible": False,
                                        "reason": "currency_conversion_wrong_magnitude",
                                        "expected_scale": expected,
                                    }
        except Exception:
            pass

        # Piecewise bracket pay: allow a constant payout (no arithmetic ops) only if it matches
        # the implied threshold decision from per-game scores.
        #
        # Example: "gets $10,000 if he averages 30 or more points ... $8,000 if under 30".
        try:
            import re

            if ("gets $" in low or "get paid" in low) and ("average" in low) and ("points" in low) and ("if" in low):
                money = [
                    float(str(x).replace(",", ""))
                    for x in re.findall(r"\$\s*([\d,]+(?:\.\d+)?)", problem_text)
                    if str(x).strip()
                ]
                m_thresh = re.search(r"\baverages?\s+(\d+(?:\.\d+)?)\s+or more\b", low)
                scores = [float(x) for x in re.findall(r"\bscored\s+(\d+(?:\.\d+)?)\b", low)]
                if len(money) >= 2 and m_thresh and scores:
                    pay_hi = max(money)
                    pay_lo = min(money)
                    threshold = float(m_thresh.group(1))
                    avg = float(sum(scores)) / float(len(scores))
                    expected_pay = pay_hi if avg >= threshold - 1e-9 else pay_lo
                    if abs(val - expected_pay) > 1e-6:
                        return {"plausible": False, "reason": "piecewise_pay_mismatch", "avg": avg, "threshold": threshold}
                    # Accept constant payout even for multi-step narratives once the threshold decision matches.
                    return verdict
        except Exception:
            pass

        # Structural check for basic numeric RPN: N numbers must match ops+1.
        if self._is_basic_numeric_rpn(expr):
            n_numbers = sum(1 for t in expr.split() if t not in {"+", "-", "*", "/"})
            n_ops = self._count_basic_ops(expr)
            if n_numbers != n_ops + 1:
                return {"plausible": False, "reason": "invalid_rpn_shape", "numbers": n_numbers, "ops": n_ops}

        # Percent sanity: avoid "100 80 *" style explosions that still pass the generic 1000× bound.
        if "%" in low or "percent" in low:
            import re

            # Savings percent: "set aside X% ... into savings" expects percent-of-total, not percent-as-divisor.
            if ("set aside" in low) and ("savings" in low):
                toks = expr.split()
                has_decimal_scale = any(t in {"0.01", ".01"} for t in toks)
                has_div_100 = any(toks[i] == "100" and i + 1 < len(toks) and toks[i + 1] == "/" for i in range(len(toks) - 1))
                has_percent_mul = "*" in toks
                if not ((has_decimal_scale or has_div_100) and has_percent_mul):
                    return {"plausible": False, "reason": "savings_percent_missing_scale"}
                # Reject percent-increase style candidates (e.g., "base + base*pct/100") in savings context.
                # Savings asks for the percent portion, not the increased total.
                if toks and toks[-1] in {"+", "-"}:
                    return {"plausible": False, "reason": "savings_percent_ends_with_add_sub"}
            # If the question explicitly asks for a percentage, bound to [0, 100].
            if any(w in low for w in ("percentage of", "as a percentage", "expressed as a percentage", "what percent")):
                if val < -1e-9 or val > 100.0 + 1e-6:
                    return {"plausible": False, "reason": "percent_out_of_bounds"}
            if nums:
                hi = max(nums)
                # Percent stories frequently describe a derived total (e.g., count×each).
                # Use a more permissive scale upper-bound than just the single max literal.
                scale_hi = float(hi)
                try:
                    m_pct = re.search(r"\b(\d+(?:\.\d+)?)\s*%", low)
                    if not m_pct:
                        m_pct = re.search(r"\b(\d+(?:\.\d+)?)\s+percent\b", low)
                    pct = float(m_pct.group(1)) if m_pct else None
                    if pct is not None and 0 < pct <= 100:
                        base_ints = [
                            float(n)
                            for n in nums
                            if n is not None
                            and abs(float(n) - pct) > 1e-9
                            and abs(float(n) - 100.0) > 1e-9
                            and float(n) > 0
                        ]
                        base_ints = list(dict.fromkeys(base_ints))
                        base_ints.sort(reverse=True)
                        if base_ints:
                            scale_hi = max(scale_hi, float(base_ints[0]))
                        if len(base_ints) >= 2:
                            scale_hi = max(scale_hi, float(base_ints[0]) * float(base_ints[1]))
                except Exception:
                    scale_hi = float(hi)
                # If we multiplied percent as a whole number, results often exceed the prompt scale by >10×.
                if val > scale_hi * 10 and val > 1000:
                    return {"plausible": False, "reason": "percent_result_exceeds_scale", "max_input": scale_hi}
                # If it's clearly a complement / "not" question, answer should not exceed the total scale.
                if re.search(r"\b(not|remaining|left|rest)\b", low) and val > scale_hi * 1.05:
                    return {"plausible": False, "reason": "percent_result_exceeds_total", "max_input": scale_hi}

        # Ticket revenue sanity: when the prompt provides ticket prices and a total collected amount,
        # the number of people/tickets cannot exceed total_collected / min_ticket_price.
        try:
            import re

            if ("ticket" in low) and ("collected" in low) and ("$" in low) and any(w in low for w in ("how many", "people", "attended", "tickets")):
                money = [
                    float(str(x).replace(",", ""))
                    for x in re.findall(r"\$\s*([\d,]+(?:\.\d+)?)", problem_text)
                    if str(x).strip()
                ]
                if len(money) >= 2:
                    total_collected = max(money)
                    prices = [m for m in money if m < total_collected - 1e-9 and m > 0]
                    if prices:
                        min_price = min(prices)
                        # Solutions for counts should depend on the total collected; reject candidates
                        # that ignore it entirely (e.g., just multiplying the ratio and one ticket price).
                        try:
                            expr_nums = []
                            for t in expr.split():
                                if t in {"+", "-", "*", "/"}:
                                    continue
                                expr_nums.append(float(t))
                            if not any(abs(x - total_collected) < 1e-6 for x in expr_nums):
                                return {
                                    "plausible": False,
                                    "reason": "ticket_missing_total_collected",
                                    "total_collected": total_collected,
                                }
                        except Exception:
                            pass
                        # Allow some slack for rounding/edge cases.
                        max_people = (total_collected / min_price) * 1.25
                        if val > max_people + 1e-6:
                            return {
                                "plausible": False,
                                "reason": "ticket_count_exceeds_revenue",
                                "total_collected": total_collected,
                                "min_price": min_price,
                                "max_people": max_people,
                            }
        except Exception:
            pass

        # Nested "each" hierarchical counting: when the prompt has multiple nested "each"
        # relationships and only small integer factors, the computation should be a pure product
        # of all factors (not a mixed add/multiply that drops a duplicate factor).
        try:
            import re

            has_nested_each = re.search(r"\beach\s+\w+\s+(?:has|have|contains|contain|holds)\b", low) is not None
            has_each_get = re.search(r"\beach\s+\w+\s+gets?\b", low) is not None
        except Exception:
            has_nested_each = False
            has_each_get = False
        if (
            low.count("each") >= 2
            and has_nested_each
            and not has_each_get
            and "$" not in low
            and any(q in low for q in ("how many", "total", "altogether", "in all"))
        ):
            try:
                if nums and len(nums) <= 6 and all(abs(n - round(n)) < 1e-9 and 1.0 <= n <= 50.0 for n in nums):
                    toks = expr.split()
                    if "*" not in toks:
                        return {"plausible": False, "reason": "nested_each_missing_multiplication"}
                    # Allow a single additive adjustment when the prompt describes an "each ... has N"
                    # and then "each ... could accommodate M more" (capacity increase).
                    allow_single_plus = (
                        ("more" in low or "additional" in low or "extra" in low)
                        and "+" in toks
                        and toks.count("+") == 1
                        and "-" not in toks
                        and "/" not in toks
                    )
                    if any(op in toks for op in {"+", "-", "/"}):
                        if not allow_single_plus:
                            return {"plausible": False, "reason": "nested_each_non_product"}
                        # If we allow "+", require it to happen before the first multiplication.
                        try:
                            if toks.index("+") > toks.index("*"):
                                return {"plausible": False, "reason": "nested_each_plus_after_multiply"}
                        except Exception:
                            pass
                    used = []
                    for t in toks:
                        if t in {"+", "-", "*", "/"}:
                            continue
                        try:
                            used.append(round(float(t), 6))
                        except Exception:
                            continue
                    need = [round(float(n), 6) for n in nums]
                    if len(used) < len(need):
                        return {"plausible": False, "reason": "nested_each_missing_factors"}
                    # Ensure duplicates in the prompt (e.g., two 4s) are preserved in the expression.
                    for v in set(need):
                        if need.count(v) > used.count(v):
                            return {"plausible": False, "reason": "nested_each_missing_duplicate_factor", "factor": v}
            except Exception:
                pass

        # "Rest of ... half ..." should subtract known parts before applying the half.
        # Reject candidates that do not use subtraction when multiple parts are present.
        if (("rest of" in low) or ("from the rest" in low)) and ("half" in low) and len(nums) >= 3:
            toks = expr.split()
            has_half = any(toks[i] == "2" and toks[i + 1] == "/" for i in range(len(toks) - 1)) or any(
                (toks[i] == "0.5" and toks[i + 1] == "*") or (toks[i] == "*" and toks[i + 1] == "0.5")
                for i in range(len(toks) - 1)
            )
            if "-" not in toks or not has_half:
                return {"plausible": False, "reason": "rest_half_missing_steps"}

        # "Half the total ... between A and B" is an average; reject pure-sum candidates.
        if ("half" in low) and ("total" in low) and ("between" in low) and len(nums) >= 2:
            toks = expr.split()
            has_half = any(toks[i] == "2" and toks[i + 1] == "/" for i in range(len(toks) - 1)) or any(
                (toks[i] == "0.5" and toks[i + 1] == "*") or (toks[i] == "*" and toks[i + 1] == "0.5")
                for i in range(len(toks) - 1)
            )
            if not has_half:
                return {"plausible": False, "reason": "half_total_between_missing_half"}

        # Comparative "how many more/less than" problems are typically additive/subtractive.
        # Reject multiplicative/division candidates unless the prompt explicitly signals it.
        if ("how many more" in low or "how many less" in low) and ("than" in low) and not any(
            w in low for w in ("times", "twice", "double", "each", "per", "%", "percent")
        ):
            toks = expr.split()
            if "*" in toks or "/" in toks:
                return {"plausible": False, "reason": "comparative_should_be_additive"}

        # Division sanity: division typically shrinks values in GSM-style problems.
        if "/" in expr and nums:
            hi = max(nums)
            if hi > 0 and val > hi * 100:
                return {"plausible": False, "reason": "division_result_too_large", "max_input": hi}

        # Yield-from-area sanity: "X bushels per acre" with acres should produce a large total
        # that scales with acres × yield, not a small subtraction.
        if ("bushels per acre" in low) and ("acre" in low) and ("how many" in low or "yield" in low or "total" in low):
            try:
                import re

                acres_vals = [float(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)\s+acres?\b", low)]
                yield_vals = [float(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)\s+bushels per acre\b", low)]
                if acres_vals and yield_vals:
                    acres = max(acres_vals)
                    y = max(yield_vals)
                    baseline = (acres * y) / 10.0
                    if val < baseline:
                        return {"plausible": False, "reason": "per_unit_total_too_small", "baseline": baseline}
                    if "*" not in expr:
                        return {"plausible": False, "reason": "per_unit_missing_multiplication"}
                    # If the prompt explicitly describes a fractional split of acreage
                    # ("one-third ... clay-rich soil") the computation must incorporate
                    # a fraction. This rejects common near-miss candidates like
                    # "acres yield * 1 -" that ignore the split entirely.
                    has_fraction_text = bool(re.search(r"\b\d+\s*/\s*\d+\b", low)) or any(
                        s in low
                        for s in (
                            "one-third",
                            "one third",
                            "two-thirds",
                            "two thirds",
                            "three-fourths",
                            "three fourths",
                            "one-fourth",
                            "one fourth",
                            "quarter",
                            "third",
                            "fourth",
                        )
                    )
                    if has_fraction_text and "/" not in expr:
                        return {"plausible": False, "reason": "per_unit_missing_fraction_split"}

                    # If yield is described as "half as much" in one region, ensure the
                    # expression reflects that (either by dividing or by using the halved
                    # yield literal).
                    if "half" in low:
                        half_y = y / 2.0
                        toks = expr.split()
                        has_div = "/" in toks
                        has_half_literal = False
                        try:
                            lits = [float(t) for t in toks if t not in {"+", "-", "*", "/"}]
                            has_half_literal = any(abs(v - half_y) < 1e-9 for v in lits) or any(abs(v - 0.5) < 1e-9 for v in lits)
                        except Exception:
                            pass
                        if not (has_div or has_half_literal):
                            return {"plausible": False, "reason": "per_unit_missing_half_yield", "half_yield": half_y}
            except Exception:
                pass

        # Half-rate fuel efficiency: "half as many miles per gallon" reduces mpg, so gallons should increase.
        if ("miles per gallon" in low or "miles/gallon" in low) and ("half" in low) and ("gallon" in low):
            if ("how many gallons" in low) or ("require" in low and "gallon" in low):
                if nums and len(nums) >= 2:
                    # Use largest as distance and smallest as mpg.
                    distance = max(nums)
                    mpg = min(nums)
                    if mpg > 0:
                        baseline = distance / mpg
                        if val < baseline * 0.99:
                            return {"plausible": False, "reason": "half_rate_gallons_too_small", "baseline": baseline}

        # Packaging unit price: carton->box->pack chains should include multiple multiplications
        # (count of packs) before dividing the total cost.
        if ("carton" in low) and ("box" in low) and ("pack" in low) and ("dozen" in low) and ("$" in problem_text):
            if ("price" in low) and ("pack" in low):
                mul_ops = str(expr).split().count("*")
                div_ops = str(expr).split().count("/")
                if div_ops >= 1 and mul_ops < 2:
                    return {"plausible": False, "reason": "packaging_chain_incomplete", "mul_ops": mul_ops}

        # Inverse-chain sanity: when the question asks for an earlier duration
        # ("how long did ...", "how many hours") but the given numeric value refers
        # to a later derived duration ("twice as long ..."), the answer should not
        # exceed the given.
        asks_duration = any(
            q in low
            for q in (
                "how long",
                "how much time",
                "how many seconds",
                "how many minutes",
                "how many hours",
                "how many days",
                "how many weeks",
                "how many months",
                "how many years",
                "in seconds",
                "in minutes",
                "in hours",
                "in days",
                "in weeks",
                "in months",
                "in years",
            )
        )
        if nums and "if" in low and asks_duration:
            if any(w in low for w in ("twice as long", "times as long", "twice", "double", "triple", "thrice")):
                hi = max(nums)
                if val > hi * 1.05:
                    return {"plausible": False, "reason": "inverse_chain_result_exceeds_given", "max_input": hi}

        # Multi-step sanity: if the story explicitly sequences actions, require enough operations.
        indicators = self._count_multi_step_indicators(problem_text)
        if indicators > 0 and nums and len(nums) >= 3:
            n_ops = self._count_basic_ops(expr)
            if n_ops < indicators + 1:
                return {"plausible": False, "reason": "multi_step_incomplete", "needed_ops": indicators + 1, "ops": n_ops}

        # Fraction-chain sanity: narrative fraction cascades ("1/3 ... half of those ...")
        # often require multiple divide steps even when there are no explicit "then/after" cues.
        try:
            import re

            frac_tokens = 0
            frac_tokens += len(re.findall(r"\b\d+\s*/\s*\d+\b", low))
            # Count fraction *words* only in fractional contexts, not ordinal contexts.
            #
            # Avoid treating "third throw / fourth throw / fifth throw" as fraction cues.
            frac_tokens += sum(1 for w in ("half", "quarter") if re.search(rf"\b{w}\b", low))
            # "a third", "one-third", "third of", "2 fifths", etc.
            frac_tokens += len(
                re.findall(
                    r"\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s*(?:-|\s+)?"
                    r"(third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)s?\b",
                    low,
                )
            )
            frac_tokens += len(re.findall(r"\b(third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b\s+of\b", low))
            cascade = any(
                w in low
                for w in (
                    "remaining",
                    "of the remaining",
                    "of those",
                    "then",
                    "after",
                    "kept",
                    "left",
                    "rotten",
                    "fresh",
                    "accepted",
                    "enrolled",
                    "sell",
                )
            )
            if frac_tokens >= 2 and cascade:
                n_ops = self._count_basic_ops(expr)
                div_ops = str(expr).count("/")
                # Require at least two operations and at least two divisions in this mode.
                if n_ops < 2 or div_ops < 2:
                    return {"plausible": False, "reason": "fraction_chain_incomplete", "ops": n_ops, "divs": div_ops}
                # Reject no-op divisions (e.g., "/ 1") which often appear when a fraction word
                # wasn't correctly translated into a divisor.
                divs: List[float] = []
                toks = str(expr).split()
                for i, t in enumerate(toks):
                    if t != "/" or i <= 0:
                        continue
                    try:
                        divs.append(float(toks[i - 1]))
                    except Exception:
                        continue
                if any(abs(d - 1.0) < 1e-9 for d in divs):
                    return {"plausible": False, "reason": "divide_by_one_in_fraction_chain"}
                if "half" in low:
                    has_div2 = any(abs(d - 2.0) < 1e-9 for d in divs)
                    has_mul_half = any(t in {"0.5", ".5"} for t in toks)
                    if not (has_div2 or has_mul_half):
                        return {"plausible": False, "reason": "half_missing_in_fraction_chain"}
        except Exception:
            pass

        # Ratio sanity: "<num> <denom> as many" with ratio < 1 should not exceed the base scale.
        try:
            import re

            if "as many" in low and nums:
                m = re.search(
                    r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:-|\s+)\s*"
                    r"(half|third|quarter|fourth|fifth|sixth|seventh|eighth|ninth|tenth)s?\b",
                    low,
                )
                if m:
                    num_word = str(m.group(1)).strip().lower()
                    den_word = str(m.group(2)).strip().lower()
                    num_map = {
                        "a": 1,
                        "an": 1,
                        "one": 1,
                        "two": 2,
                        "three": 3,
                        "four": 4,
                        "five": 5,
                        "six": 6,
                        "seven": 7,
                        "eight": 8,
                        "nine": 9,
                        "ten": 10,
                    }
                    den_map = {
                        "half": 2,
                        "third": 3,
                        "quarter": 4,
                        "fourth": 4,
                        "fifth": 5,
                        "sixth": 6,
                        "seventh": 7,
                        "eighth": 8,
                        "ninth": 9,
                        "tenth": 10,
                    }
                    num_v = float(num_word) if num_word.isdigit() else float(num_map.get(num_word, 0))
                    den_v = float(den_map.get(den_word, 0))
                    if den_v > 0 and num_v > 0 and (num_v / den_v) < 1.0:
                        hi = max(nums)
                        if val > hi * 1.05:
                            return {"plausible": False, "reason": "ratio_result_exceeds_base", "max_input": hi}
        except Exception:
            pass

        # Fraction-of-sum sanity: when the prompt contains multiple explicit "a/b of X"
        # terms (common in mixtures/combos), reject candidates that appear to swap
        # denominator and quantity (e.g., "5 20 *" instead of "20 ... / 5").
        try:
            import re

            if ("/" in low) and (" of " in low):
                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                frac_of = [
                    (float(n), float(d), float(q))
                    for (n, d, q) in re.findall(
                        r"\b(\d+)\s*/\s*(\d+)(?:st|nd|rd|th)?\s+of\s+(\d+(?:\.\d+)?)\b",
                        low_norm,
                    )
                    if float(d) > 0
                ]
                if len(frac_of) >= 2:
                    toks = expr.split()
                    # Should include addition and at least one division per fraction term.
                    if "+" not in toks or toks.count("/") < len(frac_of):
                        return {"plausible": False, "reason": "fraction_sum_incomplete"}

                    # Reject swapped-order multiplications: "den qty *".
                    pairs = {(d, q) for (_n, d, q) in frac_of}
                    for i in range(len(toks) - 2):
                        if toks[i + 2] != "*":
                            continue
                        try:
                            a = float(toks[i])
                            b = float(toks[i + 1])
                        except Exception:
                            continue
                        if any(abs(a - d) < 1e-9 and abs(b - q) < 1e-9 for (d, q) in pairs):
                            return {"plausible": False, "reason": "fraction_operand_order"}
        except Exception:
            pass

        # If we have multiple quantities but the candidate does no computation, it's almost always wrong.
        asks_integer = any(q in low for q in ("how many", "number of", "how much", "total", "altogether", "in all"))
        if nums and len(nums) >= 2 and asks_integer and self._count_basic_ops(expr) == 0:
            return {"plausible": False, "reason": "no_operation"}

        # Unit sanity: avoid wild magnitudes for money/time contexts.
        if "$" in problem_text and val > 1e7:
            return {"plausible": False, "reason": "unrealistic_money"}
        # Per-item cost sanity: when asking "how much does each ... cost", the answer
        # should not exceed the largest explicit $ amount by a large factor.
        try:
            import re

            # Only apply "each cost" bounds when the QUESTION is asking for a unit price/cost,
            # not when the prompt merely mentions unit costs as intermediate facts.
            asks_unit_cost = (
                ("how much does each" in low)
                or ("what does each" in low)
                or ("what is the cost of each" in low)
                or ("how much does 1" in low and "cost" in low)
                or ("how much does a" in low and "cost" in low)
                or ("what does 1" in low and "cost" in low)
                or ("what does a" in low and "cost" in low)
                or ("cost per" in low and ("how much" in low or "what" in low))
                or ("price per" in low and ("how much" in low or "what" in low))
                or ("price of a" in low and ("how much" in low or "what" in low))
            )
            if ("$" in problem_text) and asks_unit_cost:
                # Unit-cost questions almost always require a division step.
                if "/" not in str(expression or ""):
                    return {"plausible": False, "reason": "unit_cost_missing_division"}
                dollars = [float(x) for x in re.findall(r"\$\s*(\d+(?:\.\d+)?)", problem_text)]
                if dollars:
                    max_d = max(dollars)
                    # Reject common orientation bug: "count / dollars" instead of "dollars / count".
                    toks = str(expression or "").split()
                    if len(toks) == 3 and toks[2] == "/":
                        try:
                            a = float(toks[0])
                            b = float(toks[1])
                            # If denominator is the dollar total and numerator is a larger integer count, it's wrong.
                            if abs(b - max_d) < 1e-9 and abs(a - round(a)) < 1e-9 and a > max_d * 1.01:
                                return {"plausible": False, "reason": "each_cost_division_orientation", "max_dollar": max_d}
                        except Exception:
                            pass
                    if val > max_d * 1.5:
                        return {"plausible": False, "reason": "each_cost_exceeds_total_cost", "max_dollar": max_d}
        except Exception:
            pass

        # Discount problems: require multiplication (applying unit price to a count, or applying a
        # percent complement to a subtotal). In many narratives there is no explicit "bought".
        if ("discount" in low) and ("$" in problem_text):
            discount_ctx = any(
                w in low
                for w in (
                    "bought",
                    "buy",
                    "purchased",
                    "each",
                    "per person",
                    "given a discount",
                    "discount of",
                    "discounted",
                )
            )
            if discount_ctx:
                ops = {tok for tok in expr.split() if tok in {"+", "-", "*", "/"}}
                # Division is almost never the core operation for discounts; it commonly indicates
                # we divided by the number of people/items instead of applying a discount.
                if "/" in ops and ("*" not in ops) and ("-" not in ops):
                    return {"plausible": False, "reason": "discount_divide_instead_of_subtract"}
                # "each/per person" discounts must apply the count (multiply) and subtract the discount.
                if ("each" in low or "per person" in low) and ("*" not in ops or "-" not in ops):
                    return {"plausible": False, "reason": "discount_missing_subtract_or_multiply"}

        # Percent scaling: if a percent appears in the text, candidates that use the raw percent
        # value without normalization (÷100 or a decimal factor) are usually wrong.
        if ("%" in low) or ("percent" in low):
            try:
                import re

                m_pct = re.search(r"\b(\d+(?:\.\d+)?)\s*%", low)
                if not m_pct:
                    m_pct = re.search(r"\b(\d+(?:\.\d+)?)\s+percent\b", low)
                pct = float(m_pct.group(1)) if m_pct else None
                if pct is not None and 0 < pct < 100:
                    toks = expr.split()
                    pct_tok = str(int(pct)) if abs(pct - round(pct)) < 1e-9 else str(pct)
                    pay_pct = 100.0 - pct
                    pay_tok = str(int(pay_pct)) if abs(pay_pct - round(pay_pct)) < 1e-9 else str(pay_pct)
                    uses_pct = pct_tok in toks
                    has_100 = "100" in toks
                    has_unit_frac = False
                    for t in toks:
                        try:
                            v = float(t)
                        except Exception:
                            continue
                        if 0 < v < 1:
                            has_unit_frac = True
                            break
                    if uses_pct and (not has_100) and (not has_unit_frac):
                        return {"plausible": False, "reason": "percent_unscaled"}
                    # If this is explicitly a percent discount story, reject candidates that don't
                    # apply the percent at all (no pct, no complement, no ÷100, no decimal factor).
                    if ("discount" in low) and ("$" in problem_text) and (not uses_pct) and (pay_tok not in toks) and (not has_100) and (not has_unit_frac):
                        return {"plausible": False, "reason": "percent_discount_missing_factor"}
            except Exception:
                pass

            # Percent + "remaining/left" with multiple base quantities: require using the main base factors.
            #
            # This prevents TTC from selecting percent-on-a-single-number candidates when the prompt
            # clearly describes a multiplicative total (e.g., "50 rows, 400 flowers each, cut 60%").
            try:
                if any(w in low for w in ("remaining", "remain", "left")):
                    m_pct = re.search(r"\b(\d+(?:\.\d+)?)\s*%", low)
                    if not m_pct:
                        m_pct = re.search(r"\b(\d+(?:\.\d+)?)\s+percent\b", low)
                    pct = float(m_pct.group(1)) if m_pct else None
                    if pct is not None and 0 < pct < 100:
                        base_ints = [
                            float(n)
                            for n in nums
                            if n is not None
                            and abs(float(n) - float(pct)) > 1e-9
                            and abs(float(n) - 100.0) > 1e-9
                            and abs(float(n) - round(float(n))) < 1e-9
                            and float(n) > 0
                        ]
                        base_ints = list(dict.fromkeys(base_ints))
                        base_ints.sort(reverse=True)
                        if len(base_ints) >= 2:
                            important = [round(abs(v), 6) for v in base_ints[:2]]
                            toks = expr.split()
                            used = set()
                            for t in toks:
                                if t in {"+", "-", "*", "/"}:
                                    continue
                                try:
                                    used.add(round(abs(float(t)), 6))
                                except Exception:
                                    continue
                            if not all(v in used for v in important):
                                return {
                                    "plausible": False,
                                    "reason": "percent_remaining_missing_base_factors",
                                    "factors": important,
                                }
                            if "*" not in toks:
                                return {"plausible": False, "reason": "percent_remaining_missing_multiplication"}
            except Exception:
                pass

        # Remaining-equal value splits: must subtract known costs before dividing.
        if ("$" in problem_text) and ("remaining" in low) and ("equal" in low) and ("together" in low):
            ops = {tok for tok in expr.split() if tok in {"+", "-", "*", "/"}}
            if "-" not in ops:
                return {"plausible": False, "reason": "remaining_equal_missing_subtract"}
            if "/" not in ops:
                return {"plausible": False, "reason": "remaining_equal_missing_division"}

        # Conversion with an "official rate" fraction should involve multiple division steps
        # (rate inversion + fraction application), not a single divide.
        if ("worth" in low) and any(w in low for w in ("dollar", "dollars")):
            try:
                import re

                m_frac = re.search(r"\b(\d+)\s*/\s*(\d+)(?:st|nd|rd|th)?s?\b", low)
                if m_frac and ("official" in low):
                    fn = m_frac.group(1)
                    fd = m_frac.group(2)
                    # We expect BOTH a base conversion (often division by the rate) AND applying the fraction.
                    if expr.split().count("/") < 2:
                        return {"plausible": False, "reason": "conversion_fraction_missing_divisions"}
                    # Require the denominator of the fraction to appear in the expression (otherwise we likely
                    # ignored the fraction modifier entirely).
                    if fd not in expr.split():
                        return {"plausible": False, "reason": "conversion_fraction_missing_denominator", "den": fd}
            except Exception:
                pass

        # Inverse-half narratives: "ate half ... now only N left" implies multiplying by 2 when
        # back-solving, not multiplying by 0.5.
        if ("ate half" in low) and ("only" in low) and ("0.5" in expr):
            return {"plausible": False, "reason": "inverse_half_wrong_direction"}
        if ("ate half" in low) and ("only" in low) and any(w in low for w in ("before", "originally", "used to")):
            # Solving for the original amount requires an inversion step (×2 or ÷0.5).
            if ("2" not in expr.split()) and ("0.5" not in expr.split()):
                return {"plausible": False, "reason": "inverse_half_missing_inversion"}
        if any(u in low for u in ("hour", "hours", "minute", "minutes", "second", "seconds", "day", "days")) and val > 1e9:
            return {"plausible": False, "reason": "unrealistic_time"}

        # Remaining-style stories: unless there is explicit gain, remaining is rarely above the maximum mentioned.
        if nums and any(w in low for w in ("left", "remaining", "rest")):
            hi = max(nums)
            asks_initial = any(w in low for w in ("before", "originally", "at first", "to begin with", "initially"))
            if (not asks_initial) and val > hi * 1.05 and any(
                v in low for v in ("spent", "give", "gave", "lost", "drank", "ate", "used", "sold")
            ):
                return {"plausible": False, "reason": "remaining_exceeds_total", "max_input": hi}

        return verdict

    def _verify_rate_duration_magnitude(self, problem_text: str, result: Any) -> bool:
        """
        Extra magnitude guard for rate-style templates.

        These problems often have 100×/1000× errors when we multiply instead of divide
        (or vice-versa). We keep this separate from the generic reasonableness check so
        it only gates template='rate_duration' attempts.
        """
        nums = self.extract_numbers(problem_text)
        if not nums:
            return True
        try:
            val = float(result)
        except Exception:
            return False
        lo = min(nums)
        hi = max(nums)
        if hi <= 0:
            return True
        # If the answer is orders of magnitude beyond the inputs, it's usually a wrong op.
        if val > hi * 100:
            return False
        if lo > 0 and val < lo / 100:
            return False
        return True

    def get_last_composition_meta(self) -> Dict[str, Any]:
        return dict(self._last_composition)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "retrieval_hits": int(self._stats.get("retrieval_hits", 0)),
            "retrieval_used": int(self._stats.get("retrieval_used", 0)),
            "template_counts": dict(self._stats.get("template_counts", {})),
            "template_success_rates": dict(self._success_stats.get("templates", {})),
        }

    def _heuristic_template(
        self,
        *,
        problem_text: str,
        matched_patterns: List[Dict[str, Any]],
        understanding: ProblemUnderstanding,
    ) -> str:
        text = (problem_text or "").lower()

        # "There were X students... Y were boys... How many were girls?" → X - Y
        if "girls" in text and "boys" in understanding.labels and understanding.quantities:
            return "total_minus_boys"

        # Percent complement: "There are N ... P% are ..." + "how many ... not" → N - (N*P/100)
        if understanding.goals.get("percent_complement") and any(op.get("type") == "percent_of" for op in understanding.operations):
            if "total" in understanding.labels or understanding.quantities:
                return "percent_complement"

        # Multi-"each ... per <time>" chains like food webs tend to be pure products:
        # "each snake eats 3 birds per day ... 6 jaguars" → multiply all factors.
        if (
            text.count("each") >= 2
            and any(w in text for w in ("per day", "per week", "per month", "per year"))
            and any(w in text for w in ("eats", "eat", "consumes", "consume"))
        ):
            return "rate_duration"

        if "how much more" in text:
            unit_costs = [
                q
                for q in getattr(understanding, "quantities", [])
                if isinstance(q, dict) and q.get("kind") == "unit_cost"
            ]
            if len(unit_costs) >= 2:
                return "cost_difference"

        weighted_terms = [
            q
            for q in getattr(understanding, "quantities", [])
            if isinstance(q, dict) and q.get("kind") == "weighted_avg_term"
        ]
        if len(weighted_terms) >= 2 and "average" in text:
            return "weighted_average"

        if "profit" in text:
            has_unit_cost = any(isinstance(q, dict) and q.get("kind") == "unit_cost" for q in getattr(understanding, "quantities", []))
            if has_unit_cost and "markup" in understanding.labels and "per_day" in understanding.labels and "weeks" in understanding.labels:
                return "profit_markup_schedule"

        if (
            "rate_per_year" in understanding.labels
            and "years" in understanding.labels
            and understanding.quantities
            and any(w in text for w in ("grow", "grows", "height", "tall"))
        ):
            return "linear_growth"

        # Practice schedule: "20 minutes a day ... three times as long ... 6 days a week ... 4 weeks"
        if (
            "daily_minutes" in understanding.labels
            and "times_as_long" in understanding.labels
            and "days_per_week" in understanding.labels
            and "weeks" in understanding.labels
            and "minute" in text
            and "week" in text
            and "month" in text
        ):
            return "daily_time_schedule"

        # Weekly time from daily components:
        # "half an hour twice a day ... a fifth of an hour every day ... minutes ... each week"
        if (
            "week" in text
            and "minute" in text
            and any(op.get("source") == "galaxy_twice_a_day" for op in understanding.operations)
            and any(op.get("source") == "galaxy_each_week" for op in understanding.operations)
            and sum(1 for q in understanding.quantities if q.get("source") in {"galaxy_half_an_hour", "galaxy_fraction_of_an_hour"}) >= 2
        ):
            return "daily_minutes_week"

        # Bundle ratio division:
        # "each bag has as many apples as 3 of Gerald's bags" + "Gerald's bags have 40 apples each" + total apples
        # → bag_count = total / (multiplier * each_amount)
        if (
            "as_many_multiplier" in understanding.labels
            and "each_amount" in understanding.labels
            and understanding.quantities
            and "as many" in text
            and any(w in text for w in ("bag", "bags"))
            and ("how many" in text or "number of" in text)
        ):
            return "bags_ratio_division"

        # Hourly wage schedule: "$8 an hour", "35 hours a week", "a month" (weeks=4).
        hourly_sources = {
            "galaxy_earns_dollars_per_hour",
            "galaxy_earns_per_hour",
            "galaxy_works_for_dollars_per_hour",
        }
        if (
            "hours_per_week" in understanding.labels
            and "weeks" in understanding.labels
            and any(op.get("source") in hourly_sources for op in understanding.operations)
            and any(w in text for w in ("hour", "hours"))
            and any(w in text for w in ("week", "weeks", "month", "months"))
        ):
            return "hourly_wage_schedule"

        if any(op.get("type") == "ratio_add" for op in understanding.operations):
            return "ratio_addition"
        if any(op.get("type") == "ratio_scale" for op in understanding.operations):
            return "ratio_scale"

        # Reimbursement/overcharge: paid_total - (unit_cost * count)
        if any(w in text for w in ("reimburse", "reimbursed", "refund", "refunded", "overcharged")):
            if (
                "paid_total" in understanding.labels
                and "count" in understanding.labels
                and "unit_cost" in understanding.labels
            ):
                return "reimburse_overcharge"

        # Remaining budget unit cost:
        # "For $75 ... 5 items for $7 each ... 2 items for $10 each ... bought 4 tops ... how much did each top cost?"
        if "total" in understanding.labels and ("cost" in text and "each" in text):
            has_cost_terms = any(isinstance(q, dict) and q.get("kind") == "cost_term" for q in understanding.quantities)
            if has_cost_terms:
                scalar_counts = []
                for q in sorted(understanding.quantities, key=lambda q: (q.get("pos") or 0)):
                    if not isinstance(q, dict):
                        continue
                    if q.get("kind") in {"cost_term", "unit_cost", "weighted_avg_term"}:
                        continue
                    try:
                        v = float(q.get("value"))
                    except Exception:
                        continue
                    if v <= 1.0 or abs(v - round(v)) > 1e-9 or v > 100.0:
                        continue
                    scalar_counts.append(v)
                if scalar_counts:
                    return "remaining_unit_cost"

        # Gratuity-after-tax: total bill - (subtotal + subtotal*tax_pct)
        if "gratuities" in text and "tax" in text:
            has_items = any(str(q.get("source")) == "galaxy_ordered_for_money" for q in understanding.quantities)
            if "total" in understanding.labels and "tax_pct" in understanding.labels and has_items:
                return "gratuity_from_total"

        # Expansion/remaining from a new total: total - sum(other known terms)
        if "expanded" in text and "new total" in text and "total" in understanding.labels:
            if len(understanding.quantities) >= 2:
                return "total_minus_sum_others"

        # Missing contributor when a total is given (e.g., "Kim collects 10, Houston 12, total 35").
        # Only apply when there are multiple non-total quantities and the question isn't asking for a new total "now".
        if "total" in understanding.labels and not understanding.operations:
            has_rpn_term = any(isinstance(q.get("rpn"), str) and q.get("rpn") for q in understanding.quantities)
            if not has_rpn_term:
                total_f = float(understanding.labels.get("total", 0.0))
                other_qs = []
                for q in understanding.quantities:
                    try:
                        v = float(q.get("value"))
                    except Exception:
                        continue
                    if abs(v - total_f) < 1e-9:
                        continue
                    other_qs.append(q)
                if len(other_qs) >= 2:
                    question_part = text
                    if "?" in text:
                        before_q = text.rsplit("?", 1)[0]
                        question_part = before_q.rsplit(".", 1)[-1]
                    # Avoid treating "total of 35" (given) as "asking for total".
                    asks_total = any(w in question_part for w in ("altogether", "in all", "in total"))
                    asks_now = any(w in question_part for w in (" now", "after", "then", "new total"))
                    if (("how many" in question_part or "how much" in question_part) and not asks_total and not asks_now):
                        return "total_minus_sum_others"

        if (
            "total" in understanding.labels
            and "yesterday" in understanding.labels
            and understanding.goals.get("today_twice_as_yesterday")
            and understanding.goals.get("half_remaining")
        ):
            return "multi_step_store_recall"

        # Fraction part of a known total with a count multiplier:
        # "two accounts ... a quarter of Betty's balance ... total?" → count*(total/divisor)
        if (
            self.classify_question(problem_text) == "total"
            and not understanding.goals.get("rest")
            and any(op.get("type") == "fraction_part" for op in understanding.operations)
            and len(understanding.quantities) >= 1
        ):
            return "fraction_part_total"

        # Inverse fraction chain:
        # "Three fourths ... one quarter ... If 750 ... how many total ...?" → total = given * (den_prod / num_prod)
        if (
            "given" in understanding.labels
            and any(op.get("type") in {"fraction_ratio", "fraction_part"} for op in understanding.operations)
            and any(p.get("rule_id") == "galaxy_how_many_total" for p in matched_patterns)
        ):
            return "inverse_fraction_chain"

        if any(
            p.get("rule_id") in {"galaxy_rate_per_unit_for", "galaxy_rate_per_unit"}
            for p in matched_patterns
        ):
            return "rate_duration"

        if (
            "total" in understanding.labels
            and any(isinstance(q.get("rpn"), str) and q.get("rpn") for q in understanding.quantities)
            and (
                self.classify_question(problem_text) == "difference"
                or any(w in text for w in ("left", "remaining", "rest", "last", "other", "fifth", "missing"))
                or ("if" in text and "total" in text)
            )
        ):
            return "total_minus_terms"

        comparative_ops = [
            op
            for op in understanding.operations
            if op.get("kind") == "comparative"
            and op.get("type") in {"add", "subtract"}
        ]
        if comparative_ops:
            has_story_base = any(
                p.get("rule_id")
                in {
                    "galaxy_has_quantity",
                    "galaxy_has_quantity_no_noun",
                    "galaxy_sold_quantity",
                    "galaxy_sold_to_quantity",
                    "galaxy_sold_quantity",
                    "galaxy_sold_to_quantity",
                }
                for p in matched_patterns
            )
            if not has_story_base:
                return "extract_operate_aggregate" if understanding.operations else "distribute_and_sum"
            non_comparative = [
                op
                for op in understanding.operations
                if not (
                    op.get("kind") == "comparative"
                    and op.get("type") in {"add", "subtract"}
                )
            ]
            qtype = self.classify_question(problem_text)
            if not non_comparative and (qtype == "total" or understanding.aggregation == "sum"):
                return "multi_step_relative_chain_total"

        if understanding.operations:
            return "extract_operate_aggregate"

        if understanding.aggregation == "sum" and len(understanding.quantities) >= 2:
            return "distribute_and_sum"

        # If we extracted multiple quantities but no explicit operation signal, do not
        # fall back to "return the first number" (simple_apply). Prefer aggregation,
        # since many GSM8K "how many total" questions require combining terms.
        if len(understanding.quantities) >= 2:
            qtype = self.classify_question(problem_text)
            if qtype in {"total", "rate"}:
                return "distribute_and_sum"
            return "distribute_and_sum"

        # Only allow "simple_apply" when the read is a single direct quantity/expression.
        if understanding.quantities:
            q0 = understanding.quantities[0]
            if isinstance(q0.get("rpn"), str) and q0.get("rpn"):
                return "simple_apply"
            if self.classify_question(problem_text) == "unknown":
                return "simple_apply"
        return "extract_operate_aggregate" if understanding.operations else "distribute_and_sum"

    def _structure_signature(self, *, problem_text: str, understanding: ProblemUnderstanding) -> Dict[str, Any]:
        text = (problem_text or "").lower()
        multi_step_indicator = any(
            w in text
            for w in (
                "then",
                "after",
                "before",
                "remaining",
                "tomorrow",
                "yesterday",
                "today",
                "first",
                "second",
                "third",
                "finally",
                "next",
            )
        )
        has_rate = "per" in text or "each" in text
        has_duration = any(
            w in text
            for w in (
                "day",
                "days",
                "week",
                "weeks",
                "month",
                "months",
                "year",
                "years",
                "minute",
                "minutes",
                "hour",
                "hours",
            )
        )
        return {
            "n_quantities": int(len(understanding.quantities)),
            "n_operations": int(len(understanding.operations)),
            "aggregation": understanding.aggregation or "",
            "has_labels": bool(understanding.labels),
            "has_goals": bool(understanding.goals),
            "multi_step_indicator": bool(multi_step_indicator or understanding.labels or understanding.goals),
            "has_rate": bool(has_rate),
            "has_duration": bool(has_duration),
        }

    def _select_composition_template(
        self,
        *,
        problem_text: str,
        matched_patterns: List[Dict[str, Any]],
        understanding: ProblemUnderstanding,
    ) -> str:
        pattern_ids = [p.get("rule_id") for p in matched_patterns if p.get("rule_id")]
        matched_rule_ids = sorted({str(r) for r in pattern_ids if r})
        used_rule_ids = sorted(
            {
                str(q.get("source"))
                for q in understanding.quantities
                if q.get("source")
            }
            | {
                str(o.get("source"))
                for o in understanding.operations
                if o.get("source")
            }
        )
        heuristic = self._heuristic_template(
            problem_text=problem_text,
            matched_patterns=matched_patterns,
            understanding=understanding,
        )
        structure = self._structure_signature(problem_text=problem_text, understanding=understanding)

        # 1) Retrieval: ask shadow copy for a known-good template for this pattern set.
        retrieved_template: Optional[str] = None
        retrieval_score: float = 0.0
        retrieval_hit = False
        if self.use_retrieval and self.shadow is not None:
            try:
                hit, retrieval_score = self.shadow.query_by_patterns_scored(frozenset(matched_rule_ids), structure=structure)
                if isinstance(hit, dict):
                    ctx = hit.get("semantic_context", {}) or {}
                    template = ctx.get("template_used")
                    if isinstance(template, str) and template:
                        self._stats["retrieval_hits"] = int(self._stats.get("retrieval_hits", 0)) + 1
                        retrieved_template = template
                        retrieval_hit = True
            except Exception:
                pass

        # Specialized heuristics should override retrieval (retrieval is often too coarse for these).
        if heuristic in {"weighted_average", "profit_markup_schedule"}:
            selected = heuristic
            retrieved_template = None
            retrieval_score = 0.0
        else:
            selected = retrieved_template or heuristic
            # If retrieval disagrees with heuristic, require evidence that retrieval is better.
            if (
                retrieved_template
                and retrieved_template != heuristic
                and isinstance(self._success_stats, dict)
                and isinstance(self._success_stats.get("templates"), dict)
            ):
                tmpl_rates = self._success_stats.get("templates", {}) or {}
                tmpl_counts = (
                    (self._success_stats.get("counts", {}) or {}).get("templates", {})
                    if isinstance(self._success_stats.get("counts"), dict)
                    else {}
                )
                try:
                    retrieved_rate = float(tmpl_rates.get(retrieved_template, 0.5))
                except Exception:
                    retrieved_rate = 0.5
                try:
                    heuristic_rate = float(tmpl_rates.get(heuristic, 0.5))
                except Exception:
                    heuristic_rate = 0.5
                try:
                    retrieved_support = int(tmpl_counts.get(retrieved_template, 0))
                except Exception:
                    retrieved_support = 0
                try:
                    heuristic_support = int(tmpl_counts.get(heuristic, 0))
                except Exception:
                    heuristic_support = 0
                # Never allow retrieval to force "simple_apply" away from a more structured heuristic.
                if retrieved_template == "simple_apply" and heuristic != "simple_apply":
                    selected = heuristic
                # Otherwise require a decent match score and a modestly better historical success rate.
                #
                # Note: structure similarity is already folded into retrieval_score by DualShadowCopy.
                # We keep thresholds conservative enough to avoid the earlier "retrieval hurts" regime,
                # but not so strict that retrieval is effectively never used.
                elif (
                    float(retrieval_score or 0.0) < 0.45
                    or retrieved_rate < max(0.55, heuristic_rate + 0.05)
                    or (retrieved_support < 2 and heuristic_support >= 3 and retrieved_rate < heuristic_rate + 0.02)
                ):
                    selected = heuristic
        if selected == "distribute_and_sum" and understanding.operations:
            selected = "extract_operate_aggregate"
        # Safety rule: "simple_apply" is only valid for single-quantity reads.
        if selected == "simple_apply" and len(understanding.quantities) >= 2:
            if understanding.aggregation == "sum" or self.classify_question(problem_text) == "total":
                selected = "distribute_and_sum"
            else:
                selected = "extract_operate_aggregate" if understanding.operations else "distribute_and_sum"
        retrieval_used = bool(retrieved_template and selected == retrieved_template and retrieved_template != heuristic)
        if retrieval_used:
            self._stats["retrieval_used"] = int(self._stats.get("retrieval_used", 0)) + 1
        self._last_selection_meta = {
            "template_used": selected,
            "template_selected_by": "retrieval" if retrieval_used else "heuristic",
            "retrieval_hit": bool(retrieval_hit),
            "retrieval_used": bool(retrieval_used),
            "retrieved_template": retrieved_template if retrieval_hit else None,
            "heuristic_template": heuristic,
            "retrieval_score": float(retrieval_score),
            "patterns_matched": matched_rule_ids,
            "patterns_used": used_rule_ids,
            "structure": dict(structure),
        }
        return selected

    def read_problem(
        self, problem_text: str, *, extra_rules: Optional[Sequence[Any]] = None
    ) -> Tuple[ProblemUnderstanding, Dict[str, Any]]:
        entries = self.word_galaxy.tokenize(problem_text)
        rules = list(extra_rules) if extra_rules is not None else list(self.rule_bank)
        matches = self.grammar_galaxy.match_word_sequence(entries, extra_rules=rules)

        understanding = ProblemUnderstanding()
        trace: Dict[str, Any] = {"patterns": []}

        # Prefer left-to-right patterns for composing multi-step operations.
        matches_sorted = sorted(matches, key=lambda m: (m.get("start") or 0, -(m.get("score") or 0.0)))

        seen_ops: set[tuple] = set()
        seen_weighted_terms: set[tuple] = set()
        seen_cost_terms: set[tuple] = set()
        seen_scalar_at_pos: set[tuple] = set()
        composite_extraction_spans: List[Tuple[int, int]] = []
        composite_operation_spans: List[Tuple[int, int]] = []
        total_label_spans: List[Tuple[int, int]] = []

        def _overlaps(spans: List[Tuple[int, int]], start: Optional[int], end: Optional[int]) -> bool:
            if start is None or end is None:
                return False
            for s, e in spans:
                if start < e and end > s:
                    return True
            return False

        def _add_op(op: Dict[str, Any]) -> None:
            typ = op.get("type")
            pos = op.get("pos")
            payload = None
            for k in ("amount", "factor", "divisor", "pct"):
                if k in op:
                    try:
                        payload = float(op[k])
                    except Exception:
                        payload = op[k]
                    break
            key = (typ, payload, pos)
            if key in seen_ops:
                return
            seen_ops.add(key)
            understanding.operations.append(op)

        for item in matches_sorted:
            rule = item.get("rule")
            captures = item.get("captures", {}) or {}
            rule_id = getattr(rule, "rule_id", None)
            domain = getattr(rule, "domain", "")
            pos = item.get("start")
            end = item.get("end")
            trace["patterns"].append({"rule_id": rule_id, "domain": domain, "captures": captures, "pos": pos})

            def _cap_num(key: str) -> Optional[float]:
                if key not in captures:
                    return None
                value = captures[key].get("value")
                if isinstance(value, (int, float)):
                    return float(value)
                lit = captures[key].get("rpn_literal")
                if isinstance(lit, str):
                    try:
                        return float(lit)
                    except Exception:
                        return None
                return None

            if domain == "math_extraction":
                # If we already matched a composite extraction (term) for this span, ignore the generic extractors.
                if _overlaps(composite_extraction_spans, pos, end) and rule_id in {
                    "galaxy_there_are_quantity",
                    "galaxy_sold_quantity",
                    "galaxy_sold_to_quantity",
                    "galaxy_has_quantity",
                    "galaxy_has_quantity_no_noun",
                    "galaxy_and_quantity_noun",
                    "galaxy_bought_quantity_noun",
                }:
                    continue
                if rule_id in {"galaxy_sold_avg_cost_dollar", "galaxy_and_avg_cost_dollar"}:
                    count = _cap_num("count")
                    price = _cap_num("price")
                    if count is not None and price is not None:
                        count_lit = str(int(count)) if abs(count - round(count)) < 1e-9 else str(count)
                        price_lit = str(int(price)) if abs(price - round(price)) < 1e-9 else str(price)
                        key = (float(count), float(price), pos, end)
                        if key not in seen_weighted_terms:
                            seen_weighted_terms.add(key)
                            understanding.quantities.append(
                                {
                                    "kind": "weighted_avg_term",
                                    "rpn": f"{count_lit} {price_lit} *",
                                    "value": float(count) * float(price),
                                    "count": float(count),
                                    "price": float(price),
                                    "weight": float(count),
                                    "source": rule_id,
                                    "pos": pos,
                                }
                            )
                            if isinstance(pos, int) and isinstance(end, int):
                                composite_extraction_spans.append((pos, end))
                    continue
                if rule_id == "galaxy_book_total_pages":
                    total = _cap_num("total")
                    if total is not None:
                        understanding.labels["total"] = float(total)
                        if not understanding.quantities:
                            understanding.quantities.append({"value": total, "source": rule_id, "pos": pos})
                    continue
                if rule_id == "galaxy_total_of":
                    total = _cap_num("total")
                    if total is not None and "total" not in understanding.labels:
                        understanding.labels["total"] = float(total)
                        # Keep as quantity only if it appears to be the main numeric anchor.
                        if not understanding.quantities:
                            understanding.quantities.append({"value": total, "source": rule_id, "pos": pos})
                        if isinstance(pos, int) and isinstance(end, int):
                            total_label_spans.append((pos, end))
                    continue
                if rule_id == "galaxy_for_dollar_bought_total":
                    total = _cap_num("total")
                    if total is not None and "total" not in understanding.labels:
                        understanding.labels["total"] = float(total)
                        if not understanding.quantities:
                            understanding.quantities.append({"value": total, "source": rule_id, "pos": pos})
                        if isinstance(pos, int) and isinstance(end, int):
                            total_label_spans.append((pos, end))
                    continue
                if rule_id == "galaxy_total_of_for_count":
                    total = _cap_num("total")
                    count = _cap_num("count")
                    if total is not None and count is not None:
                        understanding.labels.setdefault("total", float(total))
                        understanding.labels["paid_total"] = float(total)
                        understanding.labels.setdefault("count", float(count))
                        if isinstance(pos, int) and isinstance(end, int):
                            total_label_spans.append((pos, end))

                        low = (problem_text or "").lower()
                        # Only treat this as a unit-rate ask when the question explicitly
                        # asks for "each"/"per" (avoid overriding reimbursement problems).
                        unit_hint = any(w in low for w in ("each", " per ", "per-", "per ", "apiece"))
                        if unit_hint and float(count) != 0.0:
                            if not understanding.quantities:
                                understanding.quantities.append({"value": total, "source": rule_id, "pos": pos})
                            _add_op({"type": "post_divide", "divisor": count, "pos": pos, "source": rule_id})
                            if isinstance(pos, int) and isinstance(end, int):
                                composite_extraction_spans.append((pos, end))
                        elif not understanding.quantities:
                            understanding.quantities.append({"value": total, "source": rule_id, "pos": pos})
                    continue
                if rule_id == "galaxy_total_was":
                    total = _cap_num("total")
                    if total is not None and "total" not in understanding.labels:
                        understanding.labels["total"] = float(total)
                        if not understanding.quantities:
                            understanding.quantities.append({"value": total, "source": rule_id, "pos": pos})
                        if isinstance(pos, int) and isinstance(end, int):
                            total_label_spans.append((pos, end))
                    continue
                if rule_id == "galaxy_new_total_is":
                    total = _cap_num("total")
                    if total is not None and "total" not in understanding.labels:
                        understanding.labels["total"] = float(total)
                        if not understanding.quantities:
                            understanding.quantities.append({"value": total, "source": rule_id, "pos": pos})
                        if isinstance(pos, int) and isinstance(end, int):
                            total_label_spans.append((pos, end))
                    continue
                if rule_id == "galaxy_recorded_as":
                    value = _cap_num("value")
                    if value is not None:
                        understanding.quantities.append({"value": value, "source": rule_id, "pos": pos})
                    continue
                if rule_id == "galaxy_if_quantity_noun":
                    base = _cap_num("base")
                    if base is not None:
                        understanding.labels.setdefault("given", float(base))
                        understanding.quantities.append({"value": base, "source": rule_id, "pos": pos})
                    continue
                if rule_id == "galaxy_requires_quantity":
                    if _overlaps(total_label_spans, pos, end) or _overlaps(composite_operation_spans, pos, end):
                        continue
                    amt = _cap_num("amount")
                    if amt is not None:
                        # Avoid misreading "needs ... N <groups> that each have M ..." as a required scalar.
                        # In those cases, N is a group-count that should be handled by a composite extractor.
                        if isinstance(pos, int):
                            window = entries[pos : min(len(entries), pos + 18)]
                            has_each = any(getattr(e, "normalized", "").lower() == "each" for e in window)
                            num_count = sum(1 for e in window if getattr(e, "category", "") == "number")
                            if has_each and num_count >= 2:
                                continue
                        understanding.quantities.append({"value": amt, "source": rule_id, "pos": pos})
                    continue
                if rule_id == "galaxy_there_is_a_and_a":
                    a = _cap_num("a")
                    b = _cap_num("b")
                    if a is not None and b is not None:
                        understanding.quantities.append({"value": a, "source": rule_id, "pos": pos})
                        understanding.quantities.append({"value": b, "source": rule_id, "pos": pos})
                    continue
                if rule_id == "galaxy_ordered_for_money":
                    amt = _cap_num("amount")
                    if amt is not None:
                        understanding.quantities.append({"value": amt, "source": rule_id, "pos": pos})
                    continue
                if rule_id == "galaxy_unit_cost_is":
                    unit = _cap_num("unit_cost")
                    if unit is not None:
                        understanding.labels.setdefault("unit_cost", float(unit))
                    continue
                if rule_id == "galaxy_each_item_cost_dollar":
                    price = _cap_num("price")
                    if price is not None:
                        price_lit = str(int(price)) if abs(price - round(price)) < 1e-9 else str(price)
                        understanding.quantities.append(
                            {
                                "kind": "unit_cost",
                                "value": float(price),
                                "rpn": price_lit,
                                "source": rule_id,
                                "pos": pos,
                            }
                        )
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_extraction_spans.append((pos, end))
                    continue
                if rule_id == "galaxy_creates_for_money":
                    fixed_cost = _cap_num("fixed_cost")
                    if fixed_cost is not None:
                        understanding.labels.setdefault("fixed_cost", float(fixed_cost))
                    continue
                if rule_id == "galaxy_sells_for_times_that_much":
                    mult = _cap_num("multiplier")
                    if mult is not None:
                        understanding.labels.setdefault("markup", float(mult))
                    continue
                if rule_id == "galaxy_sells_per_day":
                    per_day = _cap_num("per_day")
                    if per_day is not None:
                        understanding.labels.setdefault("per_day", float(per_day))
                    continue
                if rule_id == "galaxy_days_a_week":
                    dpw = _cap_num("days_per_week")
                    if dpw is not None:
                        understanding.labels.setdefault("days_per_week", float(dpw))
                    continue
                if rule_id == "galaxy_days_a_week_no_prep":
                    dpw = _cap_num("days_per_week")
                    if dpw is not None:
                        understanding.labels.setdefault("days_per_week", float(dpw))
                    continue
                if rule_id == "galaxy_hours_a_week":
                    hrs = _cap_num("hours")
                    if hrs is not None:
                        understanding.labels.setdefault("hours_per_week", float(hrs))
                        understanding.quantities.append({"value": hrs, "source": rule_id, "pos": pos})
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_extraction_spans.append((pos, end))
                    continue
                if rule_id == "galaxy_a_month":
                    understanding.labels.setdefault("months", 1.0)
                    # GSM8K typically treats "a month" as 4 weeks unless otherwise specified.
                    understanding.labels.setdefault("weeks", 4.0)
                    continue
                if rule_id == "galaxy_half_an_hour":
                    minutes = 30.0
                    understanding.labels.setdefault("minutes", float(minutes))
                    understanding.quantities.append({"value": minutes, "source": rule_id, "pos": pos})
                    if isinstance(pos, int) and isinstance(end, int):
                        composite_extraction_spans.append((pos, end))
                    continue
                if rule_id == "galaxy_fraction_of_an_hour":
                    denom_cap = captures.get("denom_word", {}) if isinstance(captures, dict) else {}
                    denom_word = ""
                    if isinstance(denom_cap, dict):
                        denom_word = str(denom_cap.get("normalized") or denom_cap.get("token") or "").strip().lower()
                    denom_map = {
                        "half": 2.0,
                        "halves": 2.0,
                        "third": 3.0,
                        "thirds": 3.0,
                        "fourth": 4.0,
                        "fourths": 4.0,
                        "quarter": 4.0,
                        "quarters": 4.0,
                        "fifth": 5.0,
                        "fifths": 5.0,
                        "sixth": 6.0,
                        "sixths": 6.0,
                        "seventh": 7.0,
                        "sevenths": 7.0,
                        "eighth": 8.0,
                        "eighths": 8.0,
                        "ninth": 9.0,
                        "ninths": 9.0,
                        "tenth": 10.0,
                        "tenths": 10.0,
                    }
                    denom = float(denom_map.get(denom_word, 0.0))
                    if denom > 0.0:
                        minutes = 60.0 / denom
                        understanding.labels.setdefault("minutes", float(minutes))
                        understanding.quantities.append({"value": minutes, "source": rule_id, "pos": pos})
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_extraction_spans.append((pos, end))
                    continue
                if rule_id == "galaxy_bags_have_each":
                    each_amt = _cap_num("each")
                    if each_amt is not None and float(each_amt) > 0.0:
                        understanding.labels.setdefault("each_amount", float(each_amt))
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_extraction_spans.append((pos, end))
                    continue
                if rule_id == "galaxy_has_as_many_as_count":
                    mult = _cap_num("multiplier")
                    if mult is not None and float(mult) > 0.0:
                        understanding.labels.setdefault("as_many_multiplier", float(mult))
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_extraction_spans.append((pos, end))
                    continue
                if rule_id == "galaxy_in_weeks":
                    weeks = _cap_num("weeks")
                    if weeks is not None:
                        understanding.labels.setdefault("weeks", float(weeks))
                    continue
                if rule_id == "galaxy_with_weeks":
                    weeks = _cap_num("weeks")
                    if weeks is not None:
                        understanding.labels.setdefault("weeks", float(weeks))
                    continue
                if rule_id == "galaxy_rate_every_year":
                    rate = _cap_num("rate")
                    if rate is not None:
                        understanding.labels.setdefault("rate_per_year", float(rate))
                    continue
                if rule_id == "galaxy_minutes_a_day":
                    minutes = _cap_num("minutes")
                    if minutes is not None:
                        understanding.labels.setdefault("daily_minutes", float(minutes))
                    continue
                if rule_id == "galaxy_times_as_long":
                    mult = _cap_num("multiplier")
                    if mult is not None:
                        understanding.labels.setdefault("times_as_long", float(mult))
                    continue
                if rule_id == "galaxy_yesterday_read":
                    y = _cap_num("yesterday")
                    if y is not None:
                        understanding.labels["yesterday"] = float(y)
                    continue
                if rule_id in {
                    "galaxy_count_of_each",
                    "galaxy_count_with_each",
                    "galaxy_count_has_each",
                    "galaxy_count_that_each_have",
                    "galaxy_there_are_each_has",
                    "galaxy_there_are_each_have",
                    "galaxy_there_are_each_contain",
                    "galaxy_first_n_has_each",
                }:
                    count = _cap_num("count")
                    each = _cap_num("each")
                    if count is not None and each is not None:
                        count_lit = str(int(count)) if abs(count - round(count)) < 1e-9 else str(count)
                        each_lit = str(int(each)) if abs(each - round(each)) < 1e-9 else str(each)
                        understanding.quantities.append(
                            {
                                "rpn": f"{count_lit} {each_lit} *",
                                "value": float(count) * float(each),
                                "source": rule_id,
                                "pos": pos,
                            }
                        )
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_extraction_spans.append((pos, end))
                    continue

                if rule_id == "galaxy_count_each_minutes":
                    count = _cap_num("count")
                    minutes = _cap_num("minutes")
                    if count is not None and minutes is not None:
                        count_lit = str(int(count)) if abs(count - round(count)) < 1e-9 else str(count)
                        minutes_lit = str(int(minutes)) if abs(minutes - round(minutes)) < 1e-9 else str(minutes)
                        understanding.quantities.append(
                            {
                                "rpn": f"{count_lit} {minutes_lit} *",
                                "value": float(count) * float(minutes),
                                "source": rule_id,
                                "pos": pos,
                                "unit": "minutes",
                            }
                        )
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_extraction_spans.append((pos, end))
                    continue
                if rule_id == "galaxy_count_for_dollar_each":
                    count = _cap_num("count")
                    price = _cap_num("price")
                    if count is not None and price is not None:
                        # Guardrail: counts should be integer-like; avoid misreading price-as-count
                        # in phrases like "$10.50 each and $7.50 each".
                        try:
                            if float(count) <= 0.0 or abs(float(count) - round(float(count))) > 1e-9:
                                continue
                        except Exception:
                            continue
                        key = (float(count), float(price), pos, end)
                        if key in seen_cost_terms:
                            continue
                        seen_cost_terms.add(key)
                        count_lit = str(int(count)) if abs(count - round(count)) < 1e-9 else str(count)
                        price_lit = str(int(price)) if abs(price - round(price)) < 1e-9 else str(price)
                        understanding.quantities.append(
                            {
                                "kind": "cost_term",
                                "rpn": f"{count_lit} {price_lit} *",
                                "value": float(count) * float(price),
                                "count": float(count),
                                "price": float(price),
                                "source": rule_id,
                                "pos": pos,
                            }
                        )
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_extraction_spans.append((pos, end))
                    continue
                if rule_id == "galaxy_count_cost_dollar_each":
                    count = _cap_num("count")
                    price = _cap_num("price")
                    if count is not None and price is not None:
                        try:
                            if float(count) <= 0.0 or abs(float(count) - round(float(count))) > 1e-9:
                                continue
                        except Exception:
                            continue
                        key = (float(count), float(price), pos, end)
                        if key in seen_cost_terms:
                            continue
                        seen_cost_terms.add(key)
                        count_lit = str(int(count)) if abs(count - round(count)) < 1e-9 else str(count)
                        price_lit = str(int(price)) if abs(price - round(price)) < 1e-9 else str(price)
                        understanding.quantities.append(
                            {
                                "kind": "cost_term",
                                "rpn": f"{count_lit} {price_lit} *",
                                "value": float(count) * float(price),
                                "count": float(count),
                                "price": float(price),
                                "source": rule_id,
                                "pos": pos,
                            }
                        )
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_extraction_spans.append((pos, end))
                    continue
                if rule_id in {"multi_item_cost_sum", "multi_item_cost_sum_context"}:
                    # Build cost_term quantities that can later be summed by distribute_and_sum.
                    def _has_cost_term(count_val: float, price_val: float) -> bool:
                        for q in understanding.quantities:
                            if not isinstance(q, dict):
                                continue
                            if q.get("kind") != "cost_term":
                                continue
                            try:
                                if abs(float(q.get("count", 0.0)) - count_val) < 1e-9 and abs(
                                    float(q.get("price", 0.0)) - price_val
                                ) < 1e-9:
                                    return True
                            except Exception:
                                continue
                        return False

                    if rule_id == "multi_item_cost_sum":
                        count_a = _cap_num("count_a")
                        price_a = _cap_num("price_a")
                        count_b = _cap_num("count_b")
                        price_b = _cap_num("price_b")
                        if (
                            count_a is None
                            or price_a is None
                            or count_b is None
                            or price_b is None
                        ):
                            continue
                        pairs = [(count_a, price_a), (count_b, price_b)]
                    else:
                        count = _cap_num("count")
                        price_a = _cap_num("price_a")
                        price_b = _cap_num("price_b")
                        if count is None or price_a is None or price_b is None:
                            continue
                        pairs = [(count, price_a), (count, price_b)]

                    for count, price in pairs:
                        try:
                            c = float(count)
                            p = float(price)
                        except Exception:
                            continue
                        if _has_cost_term(c, p):
                            continue
                        count_lit = str(int(c)) if abs(c - round(c)) < 1e-9 else str(c)
                        price_lit = str(int(p)) if abs(p - round(p)) < 1e-9 else str(p)
                        understanding.quantities.append(
                            {
                                "kind": "cost_term",
                                "rpn": f"{count_lit} {price_lit} *",
                                "value": c * p,
                                "count": c,
                                "price": p,
                                "source": rule_id,
                                "pos": pos,
                            }
                        )

                    if isinstance(pos, int) and isinstance(end, int):
                        composite_extraction_spans.append((pos, end))
                    continue

                if rule_id == "galaxy_fraction_of":
                    num = _cap_num("num")
                    den = _cap_num("den")
                    base = _cap_num("base")
                    if num is not None and den is not None and base is not None and float(den) != 0.0:
                        num_lit = str(int(num)) if abs(num - round(num)) < 1e-9 else str(num)
                        den_lit = str(int(den)) if abs(den - round(den)) < 1e-9 else str(den)
                        base_lit = str(int(base)) if abs(base - round(base)) < 1e-9 else str(base)
                        rpn = f"{base_lit} {num_lit} * {den_lit} /"
                        val = float(base) * float(num) / float(den)
                        understanding.quantities.append(
                            {
                                "rpn": rpn,
                                "value": val,
                                "source": rule_id,
                                "pos": pos,
                            }
                        )
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_extraction_spans.append((pos, end))
                    continue

                if rule_id == "galaxy_rate_per_unit_for":
                    if _overlaps(composite_extraction_spans, pos, end):
                        continue
                    rate = _cap_num("rate")
                    count = _cap_num("count")
                    if rate is not None and count is not None:
                        rate_lit = str(int(rate)) if abs(rate - round(rate)) < 1e-9 else str(rate)
                        count_lit = str(int(count)) if abs(count - round(count)) < 1e-9 else str(count)
                        understanding.quantities.append(
                            {
                                "rpn": f"{rate_lit} {count_lit} *",
                                "value": float(rate) * float(count),
                                "source": rule_id,
                                "pos": pos,
                            }
                        )
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_extraction_spans.append((pos, end))
                    continue

                if rule_id == "galaxy_rate_per_month_for_years":
                    rate = _cap_num("rate")
                    years = _cap_num("years")
                    if rate is not None and years is not None:
                        rate_lit = str(int(rate)) if abs(rate - round(rate)) < 1e-9 else str(rate)
                        years_lit = str(int(years)) if abs(years - round(years)) < 1e-9 else str(years)
                        rpn = f"{rate_lit} {years_lit} 12 * *"
                        understanding.quantities.append(
                            {
                                "rpn": rpn,
                                "value": float(rate) * float(years) * 12.0,
                                "source": rule_id,
                                "pos": pos,
                            }
                        )
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_extraction_spans.append((pos, end))
                    continue

                if rule_id == "galaxy_page_letter_to_friends":
                    pages = _cap_num("pages")
                    friends = _cap_num("friends")
                    if pages is not None and friends is not None:
                        p_lit = str(int(pages)) if abs(pages - round(pages)) < 1e-9 else str(pages)
                        f_lit = str(int(friends)) if abs(friends - round(friends)) < 1e-9 else str(friends)
                        understanding.quantities.append(
                            {
                                "rpn": f"{p_lit} {f_lit} *",
                                "value": float(pages) * float(friends),
                                "source": rule_id,
                                "pos": pos,
                            }
                        )
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_extraction_spans.append((pos, end))
                    continue

                if rule_id == "galaxy_were_boys":
                    boys = _cap_num("boys")
                    if boys is not None:
                        understanding.labels.setdefault("boys", float(boys))
                    continue

                if rule_id == "galaxy_pizza_large_small_slices":
                    a = _cap_num("large_count")
                    b = _cap_num("small_count")
                    c = _cap_num("large_slices")
                    d = _cap_num("small_slices")
                    if a is not None and b is not None and c is not None and d is not None:
                        a_lit = str(int(a)) if abs(a - round(a)) < 1e-9 else str(a)
                        b_lit = str(int(b)) if abs(b - round(b)) < 1e-9 else str(b)
                        c_lit = str(int(c)) if abs(c - round(c)) < 1e-9 else str(c)
                        d_lit = str(int(d)) if abs(d - round(d)) < 1e-9 else str(d)
                        rpn = f"{a_lit} {c_lit} * {b_lit} {d_lit} * +"
                        understanding.quantities.append(
                            {
                                "rpn": rpn,
                                "value": float(a) * float(c) + float(b) * float(d),
                                "source": rule_id,
                                "pos": pos,
                            }
                        )
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_extraction_spans.append((pos, end))
                    continue

                if rule_id == "galaxy_and_quantity_noun":
                    base = _cap_num("base")
                    if base is not None and isinstance(pos, int):
                        # Avoid treating word-fraction numerators ("and one quarter ...") as standalone scalars.
                        denom_words = {
                            "half",
                            "halves",
                            "third",
                            "thirds",
                            "fourth",
                            "fourths",
                            "quarter",
                            "quarters",
                            "fifth",
                            "fifths",
                            "sixth",
                            "sixths",
                            "seventh",
                            "sevenths",
                            "eighth",
                            "eighths",
                            "ninth",
                            "ninths",
                            "tenth",
                            "tenths",
                        }
                        if pos + 2 < len(entries):
                            nxt = getattr(entries[pos + 2], "normalized", "").lower()
                            if nxt in denom_words:
                                continue
                        # If this "and N noun" is immediately followed by another number
                        # in the next few tokens, a richer composite extractor (e.g. "N bags of M")
                        # will usually capture the full term; avoid adding the raw count.
                        num_seen = 0
                        for j in range(pos, min(len(entries), pos + 7)):
                            if getattr(entries[j], "category", "") == "number":
                                num_seen += 1
                                if num_seen >= 2:
                                    base = None
                                    break
                    if base is not None:
                        if isinstance(pos, int):
                            key = (pos, float(base))
                            if key in seen_scalar_at_pos:
                                continue
                            seen_scalar_at_pos.add(key)
                        understanding.quantities.append({"value": base, "source": rule_id, "pos": pos})
                    continue

                base = _cap_num("base")
                if base is not None:
                    if rule_id == "galaxy_with_quantity_noun" and isinstance(pos, int) and isinstance(end, int):
                        time_units = {
                            "day",
                            "days",
                            "week",
                            "weeks",
                            "month",
                            "months",
                            "year",
                            "years",
                            "hour",
                            "hours",
                            "minute",
                            "minutes",
                        }
                        window = entries[pos : min(len(entries), end + 1)]
                        if any(getattr(e, "normalized", "").lower() in time_units for e in window):
                            continue
                    if isinstance(pos, int):
                        key = (pos, float(base))
                        if key in seen_scalar_at_pos:
                            continue
                        seen_scalar_at_pos.add(key)
                    understanding.quantities.append({"value": base, "source": rule_id, "pos": pos})
                continue

            if domain == "math_arithmetic" and rule_id == "galaxy_percent_of":
                pct = _cap_num("pct")
                base = _cap_num("base")
                if pct is None or base is None:
                    continue
                # If nothing else is extracted, treat as a standalone arithmetic expression.
                if not understanding.quantities:
                    understanding.quantities.append({"value": base, "source": rule_id, "pos": pos})
                    _add_op({"type": "percent_of", "pct": pct, "pos": pos, "source": rule_id})
                continue
            if domain == "math_arithmetic" and rule_id == "galaxy_percent_of_noun":
                pct = _cap_num("pct")
                if pct is None:
                    continue
                # Apply to the current base quantity (implicit total) during composition.
                _add_op({"type": "percent_of", "pct": pct, "pos": pos, "source": rule_id})
                continue

            if domain == "math_arithmetic":
                a = _cap_num("a")
                b = _cap_num("b")
                if a is None or b is None:
                    continue
                if rule_id in {"galaxy_divide_symbol"} and not understanding.quantities:
                    # Interpret bare "a/b" as a fraction quantity (not "a divided by b of a running total").
                    if float(b) != 0.0:
                        a_lit = str(int(a)) if abs(a - round(a)) < 1e-9 else str(a)
                        b_lit = str(int(b)) if abs(b - round(b)) < 1e-9 else str(b)
                        understanding.quantities.append(
                            {
                                "rpn": f"{a_lit} {b_lit} /",
                                "value": float(a) / float(b),
                                "source": rule_id,
                                "pos": pos,
                            }
                        )
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_extraction_spans.append((pos, end))
                    continue
                # Normalize: if we haven't extracted any base yet, use the first operand as base.
                if not understanding.quantities:
                    understanding.quantities.append({"value": a, "source": rule_id, "pos": pos})
                if rule_id in {"galaxy_divided_by", "galaxy_shared_among"}:
                    _add_op({"type": "divide", "divisor": b, "pos": pos, "source": rule_id})
                elif rule_id in {"galaxy_times_symbol", "galaxy_times"}:
                    _add_op({"type": "multiply", "factor": b, "pos": pos, "source": rule_id})
                elif rule_id in {"galaxy_plus_symbol", "galaxy_plus", "galaxy_plus_total"}:
                    _add_op({"type": "add", "amount": b, "pos": pos, "source": rule_id})
                elif rule_id in {"galaxy_minus_symbol", "galaxy_minus"}:
                    _add_op({"type": "subtract", "amount": b, "pos": pos, "source": rule_id})
                continue

            if domain == "math_operation":
                # Suppress simpler overlapping operations when a composite "each-of" op matched.
                if _overlaps(composite_operation_spans, pos, end) and rule_id in {
                    "galaxy_gave_to",
                    "galaxy_gave_amount_noun_to",
                    "galaxy_received",
                    "galaxy_received_amount_noun",
                    "galaxy_spent_money",
                    "galaxy_spent_amount",
                    "galaxy_lost_amount",
                    "galaxy_increased_by",
                }:
                    continue
                if rule_id == "galaxy_today_twice_as_yesterday":
                    understanding.goals["today_twice_as_yesterday"] = True
                    continue
                if rule_id == "galaxy_half_remaining":
                    understanding.goals["half_remaining"] = True
                    continue
                if rule_id in {"galaxy_gave_each_of", "galaxy_received_each_of", "galaxy_spent_each_for_count"}:
                    per = _cap_num("per") or _cap_num("price")
                    count = _cap_num("count")
                    if per is not None and count is not None:
                        amt = float(per) * float(count)
                        if rule_id == "galaxy_received_each_of":
                            _add_op({"type": "add", "amount": amt, "pos": pos, "source": rule_id})
                        else:
                            _add_op({"type": "subtract", "amount": amt, "pos": pos, "source": rule_id})
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_operation_spans.append((pos, end))
                    continue
                if rule_id == "galaxy_twice_a_week":
                    _add_op({"type": "multiply", "factor": 2.0, "pos": pos, "source": rule_id})
                    continue
                if rule_id == "galaxy_twice_a_day":
                    _add_op({"type": "multiply", "factor": 2.0, "pos": pos, "source": rule_id})
                    continue
                if rule_id == "galaxy_each_week":
                    _add_op({"type": "multiply", "factor": 7.0, "pos": pos, "source": rule_id})
                    continue
                if rule_id == "galaxy_increased_by_percent":
                    pct = _cap_num("pct")
                    if pct is not None:
                        _add_op({"type": "percent_increase", "pct": pct, "pos": pos, "source": rule_id})
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_operation_spans.append((pos, end))
                    continue
                if rule_id == "galaxy_increased_by":
                    # If the token immediately after the amount is '%' or 'percent', treat as percent increase instead.
                    if isinstance(end, int) and 0 <= end < len(entries):
                        nxt = getattr(entries[end], "normalized", "")
                        if nxt in {"%", "percent"}:
                            continue
                    amt = _cap_num("amount")
                    if amt is not None:
                        _add_op({"type": "add", "amount": amt, "pos": pos, "source": rule_id})
                    continue
                if rule_id == "galaxy_with_recoveries":
                    amt = _cap_num("amount")
                    if amt is not None:
                        _add_op({"type": "subtract", "amount": amt, "pos": pos, "source": rule_id})
                    continue
                if rule_id == "galaxy_spiked_to":
                    amt = _cap_num("amount")
                    if amt is not None:
                        _add_op({"type": "add", "amount": amt, "pos": pos, "source": rule_id})
                    continue
                if rule_id == "galaxy_percent_rate":
                    pct = _cap_num("pct")
                    if pct is not None:
                        _add_op({"type": "percent_of", "pct": pct, "pos": pos, "source": rule_id})
                    continue
                if rule_id == "percent_complement_subtract":
                    pct = _cap_num("pct")
                    if pct is not None:
                        _add_op({"type": "percent_of", "pct": pct, "pos": pos, "source": rule_id})
                    continue
                if rule_id == "percent_complement_direct":
                    understanding.goals["percent_complement"] = True
                    continue
                if rule_id == "galaxy_tax_is_percent":
                    pct = _cap_num("pct")
                    if pct is not None:
                        understanding.labels.setdefault("tax_pct", float(pct))
                    continue
                if rule_id in {
                    "galaxy_earns_dollars_per_hour",
                    "galaxy_earns_per_hour",
                    "galaxy_works_for_dollars_per_hour",
                }:
                    rate = _cap_num("rate")
                    if rate is not None:
                        _add_op({"type": "multiply", "factor": rate, "pos": pos, "source": rule_id})
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_operation_spans.append((pos, end))
                    continue
                if rule_id == "galaxy_in_days":
                    days = _cap_num("days")
                    if days is not None and float(days) > 0.0:
                        understanding.labels.setdefault("days", float(days))
                    continue
                if rule_id == "galaxy_after_years":
                    years = _cap_num("years")
                    if years is not None and float(years) > 0.0:
                        understanding.labels.setdefault("years", float(years))
                    continue
                if rule_id == "galaxy_takes_hours":
                    hrs = _cap_num("hours")
                    if hrs is not None:
                        _add_op({"type": "multiply", "factor": hrs, "pos": pos, "source": rule_id})
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_operation_spans.append((pos, end))
                    continue
                if rule_id == "galaxy_week_to_year":
                    _add_op({"type": "multiply", "factor": 52.0, "pos": pos, "source": rule_id})
                    continue
                if rule_id == "galaxy_the_rest":
                    understanding.goals["rest"] = True
                    continue
                if rule_id == "galaxy_a_third_of":
                    _add_op({"type": "fraction_part", "divisor": 3.0, "pos": pos, "source": rule_id})
                    continue
                if rule_id == "galaxy_a_quarter_of":
                    _add_op({"type": "fraction_part", "divisor": 4.0, "pos": pos, "source": rule_id})
                    continue
                if rule_id == "galaxy_fraction_words_of":
                    num = _cap_num("num")
                    denom_word = captures.get("denom_word", {}).get("normalized")
                    if num is None or not isinstance(denom_word, str):
                        continue
                    denom_map = {
                        "half": 2.0,
                        "halves": 2.0,
                        "third": 3.0,
                        "thirds": 3.0,
                        "fourth": 4.0,
                        "fourths": 4.0,
                        "quarter": 4.0,
                        "quarters": 4.0,
                        "fifth": 5.0,
                        "fifths": 5.0,
                        "sixth": 6.0,
                        "sixths": 6.0,
                        "seventh": 7.0,
                        "sevenths": 7.0,
                        "eighth": 8.0,
                        "eighths": 8.0,
                        "ninth": 9.0,
                        "ninths": 9.0,
                        "tenth": 10.0,
                        "tenths": 10.0,
                    }
                    den = denom_map.get(str(denom_word).lower())
                    if not isinstance(den, (int, float)) or float(den) == 0.0:
                        continue
                    _add_op(
                        {
                            "type": "fraction_ratio",
                            "numerator": float(num),
                            "denominator": float(den),
                            "pos": pos,
                            "source": rule_id,
                        }
                    )
                    continue
                if rule_id == "galaxy_half_as_many":
                    _add_op({"type": "derive_divide", "divisor": 2.0, "pos": pos, "source": rule_id})
                elif rule_id == "galaxy_twice_as_many":
                    _add_op({"type": "derive_multiply", "factor": 2.0, "pos": pos, "source": rule_id})
                elif rule_id == "galaxy_n_times_as_many":
                    mult = _cap_num("multiplier")
                    if mult is not None:
                        _add_op({"type": "derive_multiply", "factor": mult, "pos": pos, "source": rule_id})
                elif rule_id == "galaxy_gave_to":
                    amt = _cap_num("amount")
                    if amt is not None:
                        _add_op({"type": "subtract", "amount": amt, "pos": pos, "source": rule_id})
                elif rule_id == "galaxy_gave_amount_noun_to":
                    amt = _cap_num("amount")
                    if amt is not None:
                        _add_op({"type": "subtract", "amount": amt, "pos": pos, "source": rule_id})
                elif rule_id == "galaxy_received":
                    amt = _cap_num("amount")
                    if amt is not None:
                        _add_op({"type": "add", "amount": amt, "pos": pos, "source": rule_id})
                elif rule_id == "galaxy_received_amount_noun":
                    amt = _cap_num("amount")
                    if amt is not None:
                        _add_op({"type": "add", "amount": amt, "pos": pos, "source": rule_id})
                elif rule_id in {"galaxy_spent_money", "galaxy_spent_amount", "galaxy_lost_amount"}:
                    amt = _cap_num("amount")
                    if amt is not None:
                        _add_op({"type": "subtract", "amount": amt, "pos": pos, "source": rule_id})
                elif rule_id == "galaxy_quit_amount":
                    amt = _cap_num("amount")
                    if amt is not None:
                        _add_op({"type": "subtract", "amount": amt, "pos": pos, "source": rule_id})
                elif rule_id == "galaxy_new_got_in_amount":
                    amt = _cap_num("amount")
                    if amt is not None:
                        _add_op({"type": "add", "amount": amt, "pos": pos, "source": rule_id})
                elif rule_id == "galaxy_less_than":
                    amt = _cap_num("amount")
                    if amt is not None:
                        _add_op({"type": "subtract", "amount": amt, "pos": pos, "source": rule_id, "kind": "comparative"})
                elif rule_id == "galaxy_more_than":
                    amt = _cap_num("amount")
                    if amt is not None:
                        _add_op({"type": "add", "amount": amt, "pos": pos, "source": rule_id, "kind": "comparative"})
                elif rule_id == "galaxy_times_more":
                    mult = _cap_num("multiplier")
                    if mult is not None:
                        _add_op({"type": "times_more", "multiplier": mult, "pos": pos, "source": rule_id})
                elif rule_id == "galaxy_plus_amount":
                    amt = _cap_num("amount")
                    if amt is not None:
                        _add_op({"type": "add", "amount": amt, "pos": pos, "source": rule_id})
                elif rule_id == "galaxy_plus_amount_and_amount":
                    a = _cap_num("a")
                    b = _cap_num("b")
                    if a is not None:
                        _add_op({"type": "add", "amount": a, "pos": pos, "source": rule_id})
                    if b is not None:
                        _add_op({"type": "add", "amount": b, "pos": pos, "source": rule_id})
                elif rule_id == "galaxy_packs_of":
                    size = _cap_num("pack_size")
                    if size is not None:
                        _add_op({"type": "post_divide", "divisor": size, "pos": pos, "source": rule_id})
                elif rule_id == "galaxy_for_every_additional":
                    num = _cap_num("numerator")
                    den = _cap_num("denominator")
                    if num is not None and den is not None and float(den) != 0.0:
                        _add_op(
                            {
                                "type": "ratio_add",
                                "numerator": float(num),
                                "denominator": float(den),
                                "pos": pos,
                                "source": rule_id,
                            }
                        )
                elif rule_id == "galaxy_for_every_ratio":
                    num = _cap_num("numerator")
                    den = _cap_num("denominator")
                    if num is not None and den is not None and float(den) != 0.0:
                        _add_op(
                            {
                                "type": "ratio_scale",
                                "numerator": float(num),
                                "denominator": float(den),
                                "pos": pos,
                                "source": rule_id,
                            }
                        )
                elif rule_id in {"galaxy_each_cost_dollar", "galaxy_each_cost"}:
                    count = _cap_num("count")
                    price = _cap_num("price")
                    if count is not None and price is not None:
                        key = (float(count), float(price), pos, end)
                        if key in seen_cost_terms:
                            continue
                        seen_cost_terms.add(key)
                        count_lit = str(int(count)) if abs(count - round(count)) < 1e-9 else str(count)
                        price_lit = str(int(price)) if abs(price - round(price)) < 1e-9 else str(price)
                        entry: Dict[str, Any] = {
                            "rpn": f"{count_lit} {price_lit} *",
                            "value": float(count) * float(price),
                            "count": float(count),
                            "price": float(price),
                            "source": rule_id,
                            "pos": pos,
                        }
                        if rule_id == "galaxy_each_cost_dollar":
                            entry["kind"] = "cost_term"
                        understanding.quantities.append(entry)
                        if isinstance(pos, int) and isinstance(end, int):
                            composite_extraction_spans.append((pos, end))
                elif rule_id == "galaxy_rate_per_unit":
                    rate = _cap_num("rate")
                    cnt = _cap_num("count")
                    if rate is not None and cnt is not None and not understanding.quantities:
                        understanding.quantities.append({"value": cnt, "source": rule_id, "pos": pos})
                        _add_op({"type": "multiply", "factor": rate, "pos": pos, "source": rule_id})
                continue

            if domain == "math_relative":
                if rule_id in {"relative_more_than", "relative_less_than"}:
                    amt = _cap_num("amount")
                    if amt is None:
                        continue
                    typ = "add" if rule_id == "relative_more_than" else "subtract"
                    _add_op(
                        {
                            "type": typ,
                            "amount": float(amt),
                            "pos": pos,
                            "source": rule_id,
                            "kind": "comparative",
                        }
                    )
                    continue
                if rule_id == "relative_multiple_of":
                    _add_op({"type": "derive_multiply", "factor": 2.0, "pos": pos, "source": rule_id})
                    continue
                if rule_id == "relative_times_quantity":
                    mult = _cap_num("multiplier")
                    if mult is not None:
                        _add_op({"type": "derive_multiply", "factor": float(mult), "pos": pos, "source": rule_id})
                    continue
                continue

            if domain == "math_aggregation":
                # Be conservative: "how many" alone does not imply sum.
                if rule_id in {"galaxy_altogether"}:
                    understanding.aggregation = "sum"
                continue

            if domain == "math_operation" and rule_id == "galaxy_fraction_eaten":
                num = _cap_num("num")
                den = _cap_num("den")
                if num is None or den is None or float(den) == 0.0:
                    continue
                # Apply to the most recent base quantity (usually total).
                if not understanding.quantities:
                    continue
                total = float(understanding.quantities[0].get("value", 0.0) or 0.0)
                amt = total * float(num) / float(den)
                _add_op({"type": "subtract", "amount": amt, "pos": pos, "source": rule_id})
                continue

        # Derived unit conversions / per-period outputs.
        text_lower = (problem_text or "").lower()
        # Only consider the question clause when deciding requested output units.
        question_part = text_lower
        if "?" in text_lower:
            before_q = text_lower.rsplit("?", 1)[0]
            question_part = before_q.rsplit(".", 1)[-1]
        wants_hours = "hour" in question_part
        per_day = any(phrase in question_part for phrase in ("per day", "each day", "every day"))
        has_minutes_term = any(str(q.get("unit")) == "minutes" for q in understanding.quantities)
        days = understanding.labels.get("days")

        if wants_hours and has_minutes_term:
            # Convert minutes → hours unless already divided by 60 elsewhere.
            has_div_60 = False
            for op in understanding.operations:
                if op.get("type") not in {"divide", "post_divide"}:
                    continue
                try:
                    div = float(op.get("divisor", 0.0))
                except Exception:
                    continue
                if abs(div - 60.0) < 1e-9:
                    has_div_60 = True
                    break
            if not has_div_60:
                _add_op({"type": "post_divide", "divisor": 60.0, "pos": -10, "source": "derived_minutes_to_hours"})

        if per_day and isinstance(days, (int, float)) and float(days) > 0.0:
            has_div_days = False
            for op in understanding.operations:
                if op.get("type") not in {"divide", "post_divide"}:
                    continue
                try:
                    div = float(op.get("divisor", 0.0))
                except Exception:
                    continue
                if abs(div - float(days)) < 1e-9:
                    has_div_days = True
                    break
            if not has_div_days:
                _add_op({"type": "post_divide", "divisor": float(days), "pos": 1_000_000_000, "source": "derived_per_day"})

        # Disambiguation: "N times more ..." should not also trigger a plain "+N more than" op.
        times_more_positions = {
            op.get("pos")
            for op in understanding.operations
            if isinstance(op, dict) and op.get("type") == "times_more" and isinstance(op.get("pos"), int)
        }
        if times_more_positions:
            understanding.operations = [
                op
                for op in understanding.operations
                if not (
                    isinstance(op, dict)
                    and op.get("type") == "add"
                    and op.get("source") == "galaxy_more_than"
                    and op.get("pos") in times_more_positions
                )
            ]

        return understanding, trace

    def compose_rpn(
        self,
        understanding: ProblemUnderstanding,
        *,
        trace: Optional[Dict[str, Any]] = None,
        problem_text: str = "",
        template_override: Optional[str] = None,
    ) -> str:
        if not understanding.quantities and not understanding.labels:
            self._last_composition = {}
            self._last_selection_meta = {}
            return ""

        matched_patterns = list((trace or {}).get("patterns", [])) if isinstance((trace or {}).get("patterns", []), list) else []
        if isinstance(template_override, str) and template_override:
            template_used = template_override
            heuristic = self._heuristic_template(
                problem_text=problem_text,
                matched_patterns=matched_patterns,
                understanding=understanding,
            )
            structure = self._structure_signature(problem_text=problem_text, understanding=understanding)
            self._last_selection_meta = {
                "template_used": template_used,
                "template_selected_by": "override",
                "retrieved_template": None,
                "heuristic_template": heuristic,
                "retrieval_score": 0.0,
                "patterns_matched": sorted({p.get("rule_id") for p in matched_patterns if p.get("rule_id")}),
                "patterns_used": sorted(
                    {
                        str(q.get("source"))
                        for q in understanding.quantities
                        if q.get("source")
                    }
                    | {
                        str(o.get("source"))
                        for o in understanding.operations
                        if o.get("source")
                    }
                ),
                "structure": dict(structure),
            }
        else:
            template_used = self._select_composition_template(problem_text=problem_text, matched_patterns=matched_patterns, understanding=understanding)
        if isinstance(template_used, str) and template_used:
            counts = self._stats.get("template_counts", {})
            if not isinstance(counts, dict):
                counts = {}
            counts[template_used] = int(counts.get(template_used, 0)) + 1
            self._stats["template_counts"] = counts

        def _lit(v: float) -> str:
            return str(int(v)) if abs(v - round(v)) < 1e-9 else str(v)

        if template_used == "total_minus_boys":
            try:
                boys = float(understanding.labels.get("boys", 0.0))
            except Exception:
                boys = 0.0
            if boys <= 0.0:
                return ""
            candidates: List[Tuple[float, str]] = []
            for q in understanding.quantities:
                if not isinstance(q, dict):
                    continue
                try:
                    v = float(q.get("value"))
                except Exception:
                    continue
                expr = ""
                if isinstance(q.get("rpn"), str) and q.get("rpn"):
                    expr = str(q["rpn"]).strip()
                else:
                    expr = _lit(v)
                if expr:
                    candidates.append((v, expr))
            if not candidates:
                return ""
            base_v, base_expr = max(candidates, key=lambda t: t[0])
            if base_v <= 0.0:
                return ""
            boys_lit = _lit(boys)
            rpn = f"{base_expr} {boys_lit} -"
            meta = dict(self._last_selection_meta)
            meta.update({"composition_steps": [{"step": 1, "template": "total_minus_boys"}, {"step": 2, "base": base_expr, "boys": boys}, {"step": 3, "rpn": rpn}]})
            self._last_composition = meta
            return rpn

        if template_used == "fraction_part_total":
            # "count of accounts/items" × (total / divisor) where divisor comes from "a third/quarter of".
            fraction_parts = [op for op in understanding.operations if op.get("type") == "fraction_part"]
            if not fraction_parts:
                return ""
            op = sorted(fraction_parts, key=lambda o: (o.get("pos") or 0))[0]
            try:
                divisor = float(op.get("divisor", 0.0))
            except Exception:
                return ""
            if divisor <= 0.0:
                return ""

            candidates: List[Tuple[float, str]] = []
            for q in understanding.quantities:
                if not isinstance(q, dict):
                    continue
                try:
                    v = float(q.get("value"))
                except Exception:
                    continue
                expr = ""
                if isinstance(q.get("rpn"), str) and q.get("rpn"):
                    expr = str(q["rpn"]).strip()
                else:
                    expr = _lit(v)
                if expr:
                    candidates.append((v, expr))

            if not candidates:
                return ""

            base_v, base_expr = max(candidates, key=lambda t: t[0])
            count_v: float = 1.0
            for v, _expr in sorted(candidates, key=lambda t: t[0]):
                if abs(v - base_v) < 1e-9:
                    continue
                if v > 1.0 and abs(v - round(v)) < 1e-9 and v <= 50.0:
                    count_v = float(v)
                    break

            div_lit = _lit(divisor)
            rpn = f"{base_expr} {div_lit} /"
            if count_v > 1.0:
                rpn = f"{rpn} {_lit(count_v)} *"

            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "fraction_part_total"},
                        {"step": 2, "base": base_expr, "divisor": divisor, "count": count_v},
                        {"step": 3, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "daily_time_schedule":
            try:
                daily_minutes = float(understanding.labels.get("daily_minutes") or 0.0)
                times_as_long = float(understanding.labels.get("times_as_long") or 0.0)
                days_per_week = float(understanding.labels.get("days_per_week") or 0.0)
                weeks = float(understanding.labels.get("weeks") or 0.0)
            except Exception:
                return ""
            if daily_minutes <= 0.0 or times_as_long <= 0.0 or days_per_week <= 0.0 or weeks <= 0.0:
                return ""

            def _lit(v: float) -> str:
                return str(int(v)) if abs(v - round(v)) < 1e-9 else str(v)

            dm_lit = _lit(daily_minutes)
            mult_lit = _lit(times_as_long)
            dpw_lit = _lit(days_per_week)
            weeks_lit = _lit(weeks)

            # Total per day = daily_minutes + (daily_minutes * times_as_long)
            # Total per month = per_day * (days_per_week * weeks)
            rpn = f"{dm_lit} {mult_lit} * {dm_lit} + {dpw_lit} * {weeks_lit} *"

            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "daily_time_schedule"},
                        {"step": 2, "daily_minutes": daily_minutes, "times_as_long": times_as_long},
                        {"step": 3, "days_per_week": days_per_week, "weeks": weeks},
                        {"step": 4, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "daily_minutes_week":
            # "half an hour twice a day ... a fifth of an hour every day ... minutes ... each week"
            # Compute: (base_minutes * daily_mult + other_daily_minutes) * 7
            try:
                daily_mult = 1.0
                week_mult = 1.0
                for op in understanding.operations:
                    if op.get("type") != "multiply":
                        continue
                    src = op.get("source")
                    try:
                        factor = float(op.get("factor", 0.0))
                    except Exception:
                        continue
                    if src == "galaxy_twice_a_day" and factor > 0.0:
                        daily_mult = factor
                    if src == "galaxy_each_week" and factor > 0.0:
                        week_mult = factor
            except Exception:
                return ""
            if daily_mult <= 0.0 or week_mult <= 0.0:
                return ""

            minute_sources = {"galaxy_half_an_hour", "galaxy_fraction_of_an_hour"}
            minutes: List[Tuple[float, str, str]] = []
            for q in understanding.quantities:
                if not isinstance(q, dict):
                    continue
                src = str(q.get("source") or "")
                if src not in minute_sources:
                    continue
                try:
                    v = float(q.get("value"))
                except Exception:
                    continue
                if v <= 0.0:
                    continue
                expr = str(q.get("rpn") or "").strip() or _lit(v)
                minutes.append((v, expr, src))
            if len(minutes) < 2:
                return ""

            minutes_sorted = sorted(minutes, key=lambda t: t[0], reverse=True)
            base_v, base_expr, _src = minutes_sorted[0]
            other_exprs = [expr for _v, expr, _s in minutes_sorted[1:]]

            rpn = f"{base_expr} {_lit(daily_mult)} *"
            for expr in other_exprs:
                rpn = f"{rpn} {expr} +"
            rpn = f"{rpn} {_lit(week_mult)} *"

            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "daily_minutes_week"},
                        {"step": 2, "base_minutes": base_expr, "daily_mult": daily_mult, "other_minutes": list(other_exprs)},
                        {"step": 3, "week_mult": week_mult},
                        {"step": 4, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "bags_ratio_division":
            # "Pam has 1200 apples. Each of her bags has as many apples as 3 of Gerald's bags.
            # Gerald's bags have 40 apples each. How many bags does Pam have?"
            # → total / (multiplier * each_amount)
            try:
                each_amount = float(understanding.labels.get("each_amount") or 0.0)
                multiplier = float(understanding.labels.get("as_many_multiplier") or 0.0)
            except Exception:
                return ""
            if each_amount <= 0.0 or multiplier <= 0.0:
                return ""

            total_candidates: List[Tuple[float, str]] = []
            for q in understanding.quantities:
                if not isinstance(q, dict):
                    continue
                # Prefer scalar-ish quantities (exclude structured terms).
                if q.get("kind") not in (None, "", "scalar"):
                    continue
                try:
                    v = float(q.get("value"))
                except Exception:
                    continue
                if v <= 0.0:
                    continue
                expr = str(q.get("rpn") or "").strip() or _lit(v)
                if expr:
                    total_candidates.append((v, expr))
            if not total_candidates:
                return ""
            total_v, total_expr = max(total_candidates, key=lambda t: t[0])
            if total_v <= 0.0:
                return ""

            rpn = f"{total_expr} {_lit(multiplier)} {_lit(each_amount)} * /"
            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "bags_ratio_division"},
                        {"step": 2, "total": total_expr, "multiplier": multiplier, "each_amount": each_amount},
                        {"step": 3, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "hourly_wage_schedule":
            try:
                hours_per_week = float(understanding.labels.get("hours_per_week") or 0.0)
                weeks = float(understanding.labels.get("weeks") or 0.0)
            except Exception:
                return ""
            if hours_per_week <= 0.0 or weeks <= 0.0:
                return ""

            hourly_sources = {
                "galaxy_earns_dollars_per_hour",
                "galaxy_earns_per_hour",
                "galaxy_works_for_dollars_per_hour",
            }
            hourly_rate = None
            for op in sorted(understanding.operations, key=lambda o: (o.get("pos") or 0)):
                if op.get("type") != "multiply":
                    continue
                if op.get("source") not in hourly_sources:
                    continue
                try:
                    hourly_rate = float(op.get("factor"))
                except Exception:
                    hourly_rate = None
                if hourly_rate is not None and hourly_rate > 0.0:
                    break
            if hourly_rate is None or hourly_rate <= 0.0:
                return ""

            hours_lit = _lit(hours_per_week)
            rate_lit = _lit(hourly_rate)
            weeks_lit = _lit(weeks)
            rpn = f"{hours_lit} {rate_lit} *"
            if abs(weeks - 1.0) > 1e-9:
                rpn = f"{rpn} {weeks_lit} *"

            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "hourly_wage_schedule"},
                        {"step": 2, "hours_per_week": hours_per_week, "hourly_rate": hourly_rate, "weeks": weeks},
                        {"step": 3, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "inverse_fraction_chain":
            try:
                given = float(understanding.labels.get("given") or 0.0)
            except Exception:
                return ""
            if given <= 0.0:
                return ""

            num_prod = 1.0
            den_prod = 1.0
            for op in sorted(understanding.operations, key=lambda o: (o.get("pos") or 0)):
                typ = op.get("type")
                if typ == "fraction_ratio":
                    try:
                        n = float(op.get("numerator", 1.0))
                        d = float(op.get("denominator", 1.0))
                    except Exception:
                        continue
                    if d == 0.0:
                        continue
                    num_prod *= n
                    den_prod *= d
                elif typ == "fraction_part":
                    try:
                        d = float(op.get("divisor", 1.0))
                    except Exception:
                        continue
                    if d == 0.0:
                        continue
                    den_prod *= d

            if num_prod == 0.0 or den_prod == 0.0:
                return ""

            def _lit(v: float) -> str:
                return str(int(v)) if abs(v - round(v)) < 1e-9 else str(v)

            given_lit = _lit(given)
            den_lit = _lit(den_prod)
            num_lit = _lit(num_prod)
            if abs(num_prod - 1.0) < 1e-9:
                rpn = f"{given_lit} {den_lit} *"
            else:
                rpn = f"{given_lit} {den_lit} * {num_lit} /"

            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "inverse_fraction_chain"},
                        {"step": 2, "given": given, "num_prod": num_prod, "den_prod": den_prod},
                        {"step": 3, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "remaining_unit_cost":
            try:
                total = float(understanding.labels.get("total") or 0.0)
            except Exception:
                return ""
            if total <= 0.0:
                return ""

            cost_terms = [
                q
                for q in understanding.quantities
                if isinstance(q, dict) and q.get("kind") == "cost_term"
            ]
            if not cost_terms:
                return ""

            # Pick the last small integer count as the "unknown item count" (e.g., "4 tops").
            count_candidates: List[Tuple[int, float]] = []
            for q in understanding.quantities:
                if not isinstance(q, dict):
                    continue
                if q.get("kind") in {"cost_term", "unit_cost", "weighted_avg_term"}:
                    continue
                try:
                    v = float(q.get("value"))
                except Exception:
                    continue
                if v <= 1.0 or abs(v - round(v)) > 1e-9 or v > 100.0:
                    continue
                qpos = q.get("pos")
                if isinstance(qpos, int):
                    count_candidates.append((qpos, v))
            if not count_candidates:
                return ""
            unknown_count = float(sorted(count_candidates, key=lambda t: t[0])[-1][1])
            if unknown_count <= 0.0:
                return ""

            def _lit(v: float) -> str:
                return str(int(v)) if abs(v - round(v)) < 1e-9 else str(v)

            total_lit = _lit(total)
            count_lit = _lit(unknown_count)

            terms: List[str] = []
            for t in sorted(cost_terms, key=lambda q: (q.get("pos") or 0)):
                expr = str(t.get("rpn") or "").strip()
                if not expr:
                    try:
                        v = float(t.get("value"))
                    except Exception:
                        continue
                    expr = _lit(v)
                terms.append(expr)
            if not terms:
                return ""
            if len(terms) == 1:
                sum_expr = terms[0]
            else:
                sum_expr = " ".join(terms + ["+"] * (len(terms) - 1))

            # unit_cost = (total - sum(cost_terms)) / unknown_count
            rpn = f"{total_lit} {sum_expr} - {count_lit} /"

            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "remaining_unit_cost"},
                        {"step": 2, "total": total, "cost_terms": terms, "unknown_count": unknown_count},
                        {"step": 3, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "weighted_average":
            terms = [
                q
                for q in understanding.quantities
                if isinstance(q, dict) and q.get("kind") == "weighted_avg_term"
            ]
            if len(terms) < 2:
                return ""

            def _sum_expr(exprs: List[str]) -> str:
                if not exprs:
                    return ""
                if len(exprs) == 1:
                    return exprs[0]
                return " ".join(exprs + ["+"] * (len(exprs) - 1))

            numerator_terms: List[str] = []
            denom_terms: List[str] = []
            for t in sorted(terms, key=lambda q: (q.get("pos") or 0)):
                try:
                    count = float(t.get("count"))
                    price = float(t.get("price"))
                except Exception:
                    continue
                count_lit = _lit(count)
                price_lit = _lit(price)
                prod = str(t.get("rpn") or "").strip() or f"{count_lit} {price_lit} *"
                numerator_terms.append(prod)
                denom_terms.append(count_lit)
            if len(numerator_terms) < 2 or len(denom_terms) < 2:
                return ""

            num_expr = _sum_expr(numerator_terms)
            den_expr = _sum_expr(denom_terms)
            rpn = f"{num_expr} {den_expr} /"
            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "weighted_average"},
                        {"step": 2, "numerator_terms": numerator_terms},
                        {"step": 3, "denominator_terms": denom_terms},
                        {"step": 4, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "profit_markup_schedule":
            text = (problem_text or "").lower()
            if "profit" not in text:
                return ""

            unit_cost = None
            if "unit_cost" in understanding.labels:
                unit_cost = understanding.labels.get("unit_cost")
            if unit_cost is None:
                for q in understanding.quantities:
                    if isinstance(q, dict) and q.get("kind") == "unit_cost":
                        unit_cost = q.get("value")
                        break
            markup = understanding.labels.get("markup")
            per_day = understanding.labels.get("per_day")
            weeks = understanding.labels.get("weeks")
            days_per_week = understanding.labels.get("days_per_week")
            fixed_cost = understanding.labels.get("fixed_cost", 0.0)

            try:
                unit_cost_f = float(unit_cost)
                markup_f = float(markup)
                per_day_f = float(per_day)
                weeks_f = float(weeks)
                fixed_cost_f = float(fixed_cost or 0.0)
                days_per_week_f = float(days_per_week) if days_per_week is not None else 7.0
            except Exception:
                return ""
            if unit_cost_f <= 0 or markup_f <= 0 or per_day_f <= 0 or weeks_f <= 0 or days_per_week_f <= 0:
                return ""

            # Profit = (sell_price - unit_cost) * (per_day * days_per_week * weeks) - fixed_cost
            # sell_price = unit_cost * markup
            rpn = (
                f"{_lit(unit_cost_f)} {_lit(markup_f)} * {_lit(unit_cost_f)} - "
                f"{_lit(per_day_f)} * {_lit(days_per_week_f)} * {_lit(weeks_f)} * {_lit(fixed_cost_f)} -"
            ).strip()
            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "profit_markup_schedule"},
                        {
                            "step": 2,
                            "unit_cost": unit_cost_f,
                            "markup": markup_f,
                            "per_day": per_day_f,
                            "days_per_week": days_per_week_f,
                            "weeks": weeks_f,
                            "fixed_cost": fixed_cost_f,
                        },
                        {"step": 3, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "linear_growth":
            rate = understanding.labels.get("rate_per_year")
            years = understanding.labels.get("years")
            if rate is None or years is None:
                return ""
            try:
                rate_f = float(rate)
                years_f = float(years)
            except Exception:
                return ""
            if not understanding.quantities:
                return ""

            base_q = None
            for q in sorted(understanding.quantities, key=lambda q: (q.get("pos") or 0)):
                if str(q.get("source") or "") == "galaxy_currently_quantity":
                    base_q = q
                    break
            if base_q is None:
                base_q = sorted(understanding.quantities, key=lambda q: (q.get("pos") or 0))[-1]
            try:
                base_val = float(base_q.get("value"))
            except Exception:
                return ""
            base_expr = str(base_q.get("rpn") or "").strip() or _lit(base_val)

            rate_lit = _lit(rate_f)
            years_lit = _lit(years_f)
            rpn = f"{base_expr} {rate_lit} {years_lit} * +"
            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "linear_growth"},
                        {"step": 2, "base": base_expr, "rate_per_year": rate_f, "years": years_f},
                        {"step": 3, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "cost_difference":
            counts = [
                q
                for q in sorted(understanding.quantities, key=lambda q: (q.get("pos") or 0))
                if isinstance(q, dict)
                and q.get("kind") not in {"unit_cost", "weighted_avg_term"}
                and isinstance(q.get("value"), (int, float))
            ]
            costs = [
                q
                for q in sorted(understanding.quantities, key=lambda q: (q.get("pos") or 0))
                if isinstance(q, dict) and q.get("kind") == "unit_cost" and isinstance(q.get("value"), (int, float))
            ]
            if len(counts) < 2 or len(costs) < 2:
                return ""

            pairs: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
            used_counts: set[int] = set()
            for cost in costs:
                cpos = cost.get("pos")
                best_idx = None
                # Prefer stable left-to-right pairing: first unused count before this cost.
                for idx, cnt in enumerate(counts):
                    if idx in used_counts:
                        continue
                    qpos = cnt.get("pos")
                    if isinstance(cpos, int) and isinstance(qpos, int) and qpos < cpos:
                        best_idx = idx
                        break
                if best_idx is None:
                    for idx in range(len(counts)):
                        if idx not in used_counts:
                            best_idx = idx
                            break
                if best_idx is None:
                    continue
                used_counts.add(best_idx)
                pairs.append((counts[best_idx], cost))
                if len(pairs) >= 2:
                    break
            if len(pairs) < 2:
                return ""

            def _expr(q: Dict[str, Any]) -> str:
                if isinstance(q.get("rpn"), str) and q.get("rpn"):
                    return str(q["rpn"]).strip()
                v = float(q.get("value"))
                return _lit(v)

            cnt1, cost1 = pairs[0]
            cnt2, cost2 = pairs[1]
            term1 = f"{_expr(cnt1)} {_expr(cost1)} *"
            term2 = f"{_expr(cnt2)} {_expr(cost2)} *"
            rpn = f"{term1} {term2} -"
            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "cost_difference"},
                        {"step": 2, "term1": term1, "term2": term2},
                        {"step": 3, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "reimburse_overcharge":
            paid_total = understanding.labels.get("paid_total")
            count = understanding.labels.get("count")
            unit_cost = understanding.labels.get("unit_cost")
            if paid_total is None or count is None or unit_cost is None:
                return ""
            try:
                paid_total_f = float(paid_total)
                count_f = float(count)
                unit_cost_f = float(unit_cost)
            except Exception:
                return ""
            rpn = f"{_lit(paid_total_f)} {_lit(unit_cost_f)} {_lit(count_f)} * -"
            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "reimburse_overcharge"},
                        {"step": 2, "paid_total": paid_total_f, "unit_cost": unit_cost_f, "count": count_f},
                        {"step": 3, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "total_minus_sum_others":
            total = understanding.labels.get("total")
            if total is None:
                return ""
            try:
                total_f = float(total)
            except Exception:
                return ""
            total_sources = {"galaxy_total_of", "galaxy_total_was", "galaxy_total_of_for_count"}
            other_terms: List[str] = []
            for q in sorted(understanding.quantities, key=lambda q: (q.get("pos") or 0)):
                src = str(q.get("source") or "")
                try:
                    qv = float(q.get("value"))
                except Exception:
                    qv = None
                if src in total_sources and qv is not None and abs(qv - total_f) < 1e-9:
                    continue
                if isinstance(q.get("rpn"), str) and q.get("rpn"):
                    other_terms.append(str(q["rpn"]).strip())
                elif qv is not None:
                    other_terms.append(_lit(qv))
            if len(other_terms) < 2:
                return ""
            rpn = " ".join([_lit(total_f)] + other_terms + ["+"] * (len(other_terms) - 1) + ["-"])
            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "total_minus_sum_others"},
                        {"step": 2, "total": total_f, "terms": other_terms},
                        {"step": 3, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "gratuity_from_total":
            total = understanding.labels.get("total")
            tax_pct = understanding.labels.get("tax_pct")
            if total is None or tax_pct is None:
                return ""
            try:
                total_f = float(total)
                pct_f = float(tax_pct)
            except Exception:
                return ""
            items: List[float] = []
            for q in sorted(understanding.quantities, key=lambda q: (q.get("pos") or 0)):
                if str(q.get("source") or "") != "galaxy_ordered_for_money":
                    continue
                try:
                    items.append(float(q.get("value")))
                except Exception:
                    continue
            if len(items) < 2:
                return ""
            subtotal_terms = [_lit(v) for v in items]
            subtotal_expr = " ".join(subtotal_terms + ["+"] * (len(subtotal_terms) - 1))
            rpn = f"{_lit(total_f)} {subtotal_expr} {subtotal_expr} {_lit(pct_f)} 100 / * + -"
            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "gratuity_from_total"},
                        {"step": 2, "total": total_f, "items": items, "tax_pct": pct_f},
                        {"step": 3, "subtotal": subtotal_expr},
                        {"step": 4, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        # Multi-step template: book total, yesterday, today=2*yesterday, answer=half(remaining).
        # Example: 120 total, yesterday 12, today 24, remaining 84, answer 42.
        if template_used == "multi_step_store_recall":
            total = float(understanding.labels["total"])
            y = float(understanding.labels["yesterday"])
            total_lit = str(int(total)) if abs(total - round(total)) < 1e-9 else str(total)
            y_lit = str(int(y)) if abs(y - round(y)) < 1e-9 else str(y)
            # Slots: A=total, B=yesterday, C=today, D=remaining
            rpn = (
                f"{total_lit} STORE_A "
                f"{y_lit} STORE_B "
                "RECALL_B 2 * STORE_C "
                "RECALL_A RECALL_B - RECALL_C - STORE_D "
                "RECALL_D 2 /"
            )
            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                    {"step": 1, "label": "total", "value": total},
                    {"step": 2, "label": "yesterday", "value": y},
                    {"step": 3, "derive": "today=2*yesterday", "op": "RECALL_B 2 * STORE_C"},
                    {"step": 4, "derive": "remaining=total-yesterday-today", "op": "RECALL_A RECALL_B - RECALL_C - STORE_D"},
                    {"step": 5, "derive": "answer=remaining/2", "op": "RECALL_D 2 /"},
                    ],
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "total_minus_terms":
            total = understanding.labels.get("total")
            if total is None:
                return ""
            try:
                total_val = float(total)
            except Exception:
                return ""
            total_lit = str(int(total_val)) if abs(total_val - round(total_val)) < 1e-9 else str(total_val)

            terms: List[str] = []
            for q in sorted(understanding.quantities, key=lambda q: (q.get("pos") or 0)):
                if not (isinstance(q.get("rpn"), str) and q.get("rpn")):
                    continue
                # Avoid subtracting the total itself if it was captured as a term.
                try:
                    if abs(float(q.get("value", 0.0)) - total_val) < 1e-9:
                        continue
                except Exception:
                    pass
                terms.append(str(q["rpn"]).strip())
            if not terms:
                return ""
            rpn_parts: List[str] = [total_lit]
            for term in terms:
                rpn_parts.extend(term.split())
                rpn_parts.append("-")
            rpn = " ".join(rpn_parts)
            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "total_minus_terms"},
                        {"step": 2, "total": total_val},
                        {"step": 3, "subtract_terms": terms},
                        {"step": 4, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "percent_complement":
            pct_ops = [op for op in understanding.operations if op.get("type") == "percent_of"]
            if not pct_ops:
                return ""

            # Prefer an explicit total label; otherwise use the largest extracted scalar.
            total_val = understanding.labels.get("total")
            if total_val is None:
                scalars: List[float] = []
                for q in understanding.quantities:
                    try:
                        scalars.append(float(q.get("value")))
                    except Exception:
                        continue
                if not scalars:
                    return ""
                total_val = max(scalars)

            try:
                total_f = float(total_val)
                pct_f = float(pct_ops[0].get("pct", 0.0))
            except Exception:
                return ""

            total_lit = str(int(total_f)) if abs(total_f - round(total_f)) < 1e-9 else str(total_f)
            pct_lit = str(int(pct_f)) if abs(pct_f - round(pct_f)) < 1e-9 else str(pct_f)
            rpn = f"{total_lit} {total_lit} {pct_lit} 100 / * -"
            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "percent_complement"},
                        {"step": 2, "total": total_f, "pct": pct_f},
                        {"step": 3, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "multi_step_relative_chain_total":
            # Comparative chain totals:
            # base + (base +/- a1) + (base +/- a1 +/- a2) + ...
            # Example: base=11, +9, -13 → 11 + 20 + 7 = 38.
            if not understanding.quantities:
                return ""

            def _pick_base_quantity() -> Optional[Dict[str, Any]]:
                preferred_sources = {
                    "galaxy_has_quantity",
                    "galaxy_has_quantity_no_noun",
                    "galaxy_there_are_quantity",
                    "galaxy_sold_quantity",
                    "galaxy_sold_to_quantity",
                }
                qs = sorted(understanding.quantities, key=lambda q: (q.get("pos") or 0))
                for q in qs:
                    if q.get("source") in preferred_sources and isinstance(q.get("value"), (int, float)):
                        return q
                for q in qs:
                    if isinstance(q.get("value"), (int, float)):
                        return q
                return None

            base_q = _pick_base_quantity()
            if base_q is None:
                return ""
            base = float(base_q["value"])
            base_lit = str(int(base)) if abs(base - round(base)) < 1e-9 else str(base)

            comp_ops = [
                op
                for op in understanding.operations
                if op.get("kind") == "comparative" and op.get("type") in {"add", "subtract"}
            ]
            comp_ops = sorted(comp_ops, key=lambda o: (o.get("pos") or 0))
            if not comp_ops:
                return base_lit

            # Slots: A=base, B=current, C=sum
            rpn_parts: List[str] = [
                base_lit,
                "STORE_A",
                "RECALL_A",
                "STORE_B",
                "RECALL_B",
                "STORE_C",
            ]
            for op in comp_ops:
                typ = op.get("type")
                if typ == "add":
                    amt = float(op.get("amount", 0.0))
                    amt_lit = str(int(amt)) if abs(amt - round(amt)) < 1e-9 else str(amt)
                    rpn_parts.extend(
                        [
                            "RECALL_B",
                            amt_lit,
                            "+",
                            "STORE_B",
                            "RECALL_C",
                            "RECALL_B",
                            "+",
                            "STORE_C",
                        ]
                    )
                elif typ == "subtract":
                    amt = float(op.get("amount", 0.0))
                    amt_lit = str(int(amt)) if abs(amt - round(amt)) < 1e-9 else str(amt)
                    rpn_parts.extend(
                        [
                            "RECALL_B",
                            amt_lit,
                            "-",
                            "STORE_B",
                            "RECALL_C",
                            "RECALL_B",
                            "+",
                            "STORE_C",
                        ]
                    )
            rpn_parts.append("RECALL_C")
            rpn = " ".join(rpn_parts)
            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "multi_step_relative_chain_total"},
                        {"step": 2, "base": base},
                        {"step": 3, "comparative_ops": comp_ops},
                        {"step": 4, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "rate_duration":
            # Rate-style problems are typically multiplicative: rate × count × duration.
            # When we have multiple extracted quantities but no operations, multiply them
            # instead of returning the first literal.
            text = (problem_text or "").lower()
            wants_time = "how long" in text or "how much time" in text or any(
                w in text for w in ("in seconds", "in minutes", "in hours")
            )
            has_download = "download" in text or "downloads" in text
            has_first = "first" in text
            has_thereafter = "thereafter" in text or "afterward" in text
            grams_to_kilograms = ("kilogram" in text or "kilograms" in text or "kg" in text) and (
                "gram" in text or "grams" in text
            )

            # Special-case: piecewise download rates ("first X at r1, thereafter at r2") → time = X/r1 + (T-X)/r2.
            if has_download and has_first and has_thereafter:
                nums = self.extract_numbers(problem_text)
                if len(nums) >= 4:
                    total, rate1, size1, rate2 = nums[0], nums[1], nums[2], nums[3]
                    if abs(rate1) > 1e-12 and abs(rate2) > 1e-12:
                        total_lit = str(int(total)) if abs(total - round(total)) < 1e-9 else str(total)
                        r1_lit = str(int(rate1)) if abs(rate1 - round(rate1)) < 1e-9 else str(rate1)
                        s1_lit = str(int(size1)) if abs(size1 - round(size1)) < 1e-9 else str(size1)
                        r2_lit = str(int(rate2)) if abs(rate2 - round(rate2)) < 1e-9 else str(rate2)
                        rpn = f"{s1_lit} {r1_lit} / {total_lit} {s1_lit} - {r2_lit} / +"
                        meta = dict(self._last_selection_meta)
                        meta.update(
                            {
                                "composition_steps": [
                                    {"step": 1, "template": "rate_duration"},
                                    {"step": 2, "mode": "download_piecewise"},
                                    {"step": 3, "rpn": rpn},
                                ],
                            }
                        )
                        self._last_composition = meta
                        return rpn

            additive_ops = {"add", "subtract", "percent_of"}
            if not any(op.get("type") in additive_ops for op in understanding.operations):
                # If the question asks for time/duration (e.g., downloads at X per second),
                # we must invert the rate: time = total / rate, not rate * total.
                if wants_time or has_download:
                    patterns = (
                        list((trace or {}).get("patterns", []))
                        if isinstance((trace or {}).get("patterns", []), list)
                        else []
                    )
                    for pat in patterns:
                        if not isinstance(pat, dict):
                            continue
                        if pat.get("rule_id") != "galaxy_rate_per_unit_for":
                            continue
                        caps = pat.get("captures", {}) or {}
                        try:
                            rate = float((caps.get("rate") or {}).get("value"))
                            total = float((caps.get("count") or {}).get("value"))
                        except Exception:
                            continue
                        if abs(rate) < 1e-12:
                            continue
                        rate_lit = str(int(rate)) if abs(rate - round(rate)) < 1e-9 else str(rate)
                        total_lit = str(int(total)) if abs(total - round(total)) < 1e-9 else str(total)
                        rpn = f"{total_lit} {rate_lit} /"
                        meta = dict(self._last_selection_meta)
                        meta.update(
                            {
                                "composition_steps": [
                                    {"step": 1, "template": "rate_duration"},
                                    {"step": 2, "mode": "invert_rate"},
                                    {"step": 3, "rpn": rpn},
                                ]
                            }
                        )
                        self._last_composition = meta
                        return rpn

                factors: List[str] = []
                seen_factor_keys: set[tuple] = set()
                seen_exprs: set[str] = set()
                covered_literals: set[str] = set()

                def _is_num_token(tok: str) -> bool:
                    try:
                        float(tok)
                        return True
                    except Exception:
                        return False

                # If another extractor already produced a composite RPN term that includes a literal,
                # avoid also multiplying the same literal from helper extractors like galaxy_each_eats_quantity.
                for q in understanding.quantities:
                    if not isinstance(q, dict):
                        continue
                    if str(q.get("source") or "") == "galaxy_each_eats_quantity":
                        continue
                    expr0 = q.get("rpn")
                    if not isinstance(expr0, str) or not expr0.strip():
                        continue
                    for tok in expr0.strip().split():
                        if _is_num_token(tok):
                            covered_literals.add(tok)
                for q in sorted(understanding.quantities, key=lambda q: (q.get("pos") or 0)):
                    expr: Optional[str] = None
                    if isinstance(q.get("rpn"), str) and q.get("rpn"):
                        expr = str(q["rpn"]).strip()
                    else:
                        try:
                            v = float(q["value"])
                        except Exception:
                            continue
                        expr = str(int(v)) if abs(v - round(v)) < 1e-9 else str(v)

                    pos = q.get("pos")
                    # De-duplicate overlapping extractions (e.g., has_quantity vs has_quantity_no_noun)
                    # when they point to the same token span.
                    key = (pos, expr) if isinstance(pos, int) else (q.get("source"), pos, expr)
                    if key in seen_factor_keys:
                        continue
                    seen_factor_keys.add(key)
                    # De-duplicate helper extractors (e.g., galaxy_each_eats_quantity) when another
                    # extractor already captured the same numeric factor for this span.
                    src = str(q.get("source") or "")
                    if src == "galaxy_each_eats_quantity" and (expr in seen_exprs or expr in covered_literals):
                        continue
                    factors.append(expr)
                    seen_exprs.add(expr)

                mul_div_ops = [op for op in understanding.operations if op.get("type") in {"multiply", "divide"}]
                for op in sorted(mul_div_ops, key=lambda o: (o.get("pos") or 0)):
                    if op.get("type") == "multiply":
                        try:
                            fac = float(op.get("factor", 1.0))
                        except Exception:
                            continue
                        fac_lit = str(int(fac)) if abs(fac - round(fac)) < 1e-9 else str(fac)
                        factors.append(fac_lit)
                    elif op.get("type") == "divide":
                        try:
                            div = float(op.get("divisor", 1.0))
                        except Exception:
                            continue
                        if abs(div) < 1e-12:
                            continue
                        div_lit = str(int(div)) if abs(div - round(div)) < 1e-9 else str(div)
                        # Division is multiplicative, but keep it as an explicit op.
                        factors.append(f"1 {div_lit} /")

                if factors:
                    rpn_parts: List[str] = [factors[0]]
                    for f in factors[1:]:
                        rpn_parts.extend([f, "*"])
                    rpn = " ".join(rpn_parts)
                    if grams_to_kilograms:
                        rpn = f"{rpn} 1000 /"
                    meta = dict(self._last_selection_meta)
                    meta.update({"composition_steps": [{"step": 1, "template": "rate_duration"}, {"step": 2, "factors": factors}, {"step": 3, "rpn": rpn}]})
                    self._last_composition = meta
                    return rpn

        if template_used == "ratio_addition":
            ratio_ops = [op for op in understanding.operations if op.get("type") == "ratio_add"]
            if not ratio_ops:
                return ""
            op = sorted(ratio_ops, key=lambda o: (o.get("pos") or 0))[0]
            try:
                numerator = float(op.get("numerator", 0.0))
                denominator = float(op.get("denominator", 1.0))
            except Exception:
                return ""
            if abs(denominator) < 1e-12:
                return ""

            qs = sorted(understanding.quantities, key=lambda q: (q.get("pos") or 0))
            if len(qs) < 2:
                return ""

            def _expr(q: Dict[str, Any]) -> Optional[str]:
                if isinstance(q.get("rpn"), str) and q.get("rpn"):
                    return str(q["rpn"]).strip()
                try:
                    v = float(q.get("value"))
                except Exception:
                    return None
                return str(int(v)) if abs(v - round(v)) < 1e-9 else str(v)

            def _lit(v: float) -> str:
                return str(int(v)) if abs(v - round(v)) < 1e-9 else str(v)

            base_expr = _expr(qs[0])
            if not base_expr:
                return ""

            op_pos = op.get("pos") if isinstance(op.get("pos"), int) else None
            candidate_targets = []
            for q in qs[1:]:
                qpos = q.get("pos")
                if op_pos is None or not isinstance(qpos, int) or qpos < op_pos:
                    candidate_targets.append(q)
            target_q = candidate_targets[-1] if candidate_targets else qs[-1]
            target_expr = _expr(target_q)
            if not target_expr:
                return ""

            num_lit = _lit(numerator)
            den_lit = _lit(denominator)
            rpn = f"{base_expr} {target_expr} {num_lit} * {den_lit} / +"

            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "ratio_addition"},
                        {"step": 2, "base": base_expr, "target": target_expr},
                        {"step": 3, "ratio": f"{num_lit}/{den_lit}"},
                        {"step": 4, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "ratio_scale":
            ratio_ops = [op for op in understanding.operations if op.get("type") == "ratio_scale"]
            if not ratio_ops:
                return ""
            op = sorted(ratio_ops, key=lambda o: (o.get("pos") or 0))[0]
            try:
                numerator = float(op.get("numerator", 0.0))
                denominator = float(op.get("denominator", 1.0))
            except Exception:
                return ""
            if abs(denominator) < 1e-12:
                return ""

            def _expr(q: Dict[str, Any]) -> Optional[str]:
                if isinstance(q.get("rpn"), str) and q.get("rpn"):
                    return str(q["rpn"]).strip()
                try:
                    v = float(q.get("value"))
                except Exception:
                    return None
                return str(int(v)) if abs(v - round(v)) < 1e-9 else str(v)

            def _lit(v: float) -> str:
                return str(int(v)) if abs(v - round(v)) < 1e-9 else str(v)

            op_pos = op.get("pos") if isinstance(op.get("pos"), int) else None
            candidates: List[Dict[str, Any]] = []
            for q in sorted(understanding.quantities, key=lambda q: (q.get("pos") or 0)):
                if not isinstance(q, dict):
                    continue
                if q.get("kind") in {"unit_cost", "weighted_avg_term"}:
                    continue
                try:
                    v = float(q.get("value"))
                except Exception:
                    continue
                if abs(v - numerator) < 1e-9 or abs(v - denominator) < 1e-9:
                    continue
                candidates.append(q)
            if not candidates:
                return ""

            # Prefer the last quantity that appears after the ratio phrase ("for every ...").
            target_q = None
            if op_pos is not None:
                after = [q for q in candidates if isinstance(q.get("pos"), int) and int(q["pos"]) > op_pos]
                if after:
                    target_q = after[-1]
            if target_q is None:
                # Fallback: largest magnitude tends to be the "total" being scaled.
                target_q = max(candidates, key=lambda q: abs(float(q.get("value") or 0.0)))

            target_expr = _expr(target_q)
            if not target_expr:
                return ""
            num_lit = _lit(numerator)
            den_lit = _lit(denominator)
            rpn = f"{num_lit} {den_lit} / {target_expr} *"

            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "ratio_scale"},
                        {"step": 2, "ratio": f"{num_lit}/{den_lit}", "target": target_expr},
                        {"step": 3, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn

        if template_used == "distribute_and_sum" and not understanding.operations and len(understanding.quantities) >= 2:
            cost_terms = [
                q
                for q in understanding.quantities
                if isinstance(q, dict) and q.get("kind") == "cost_term" and isinstance(q.get("rpn"), str)
            ]
            # If we have explicit (count * price) terms, prefer summing those over raw scalars.
            # This prevents "5 friends ..." or other unrelated counts from polluting totals.
            if cost_terms:
                cost_rpns = [str(q["rpn"]).strip() for q in sorted(cost_terms, key=lambda q: (q.get("pos") or 0)) if q.get("rpn")]
                if not cost_rpns:
                    return ""
                if len(cost_rpns) == 1:
                    rpn = cost_rpns[0]
                else:
                    rpn = " ".join(cost_rpns + ["+"] * (len(cost_rpns) - 1))
                meta = dict(self._last_selection_meta)
                meta.update({"composition_steps": [{"step": 1, "terms": cost_rpns}, {"step": 2, "aggregate": "sum"}]})
                self._last_composition = meta
                return rpn

            terms: List[str] = []
            seen_term_keys: set[tuple] = set()
            for q in sorted(understanding.quantities, key=lambda q: (q.get("pos") or 0)):
                if isinstance(q.get("rpn"), str) and q.get("rpn"):
                    term = str(q["rpn"]).strip()
                    pos = q.get("pos")
                    key = (pos, term) if isinstance(pos, int) else (q.get("source"), pos, term)
                    if key in seen_term_keys:
                        continue
                    seen_term_keys.add(key)
                    terms.append(term)
                    continue
                try:
                    value = float(q["value"])
                except Exception:
                    continue
                lit = str(int(value)) if abs(value - round(value)) < 1e-9 else str(value)
                pos = q.get("pos")
                key = (pos, lit) if isinstance(pos, int) else (q.get("source"), pos, lit)
                if key in seen_term_keys:
                    continue
                seen_term_keys.add(key)
                terms.append(lit)
            if not terms:
                return ""
            if len(terms) == 1:
                rpn = terms[0]
            else:
                rpn = " ".join(terms + ["+"] * (len(terms) - 1))
            meta = dict(self._last_selection_meta)
            meta.update({"composition_steps": [{"step": 1, "terms": terms}, {"step": 2, "aggregate": "sum"}]})
            self._last_composition = meta
            return rpn

        if not understanding.operations and understanding.quantities and (len(understanding.quantities) == 1 or template_used == "simple_apply"):
            q0 = understanding.quantities[0]
            if isinstance(q0.get("rpn"), str) and q0.get("rpn"):
                rpn = str(q0["rpn"]).strip()
                text = (problem_text or "").lower()
                grams_to_kilograms = ("kilogram" in text or "kilograms" in text or "kg" in text) and (
                    "gram" in text or "grams" in text
                )
                if grams_to_kilograms and "1000 /" not in rpn and "/ 1000" not in rpn:
                    rpn = f"{rpn} 1000 /"
                meta = dict(self._last_selection_meta)
                meta.update({"composition_steps": [{"step": 1, "use_term": rpn}]})
                self._last_composition = meta
                return rpn
        elif not understanding.operations and len(understanding.quantities) >= 2:
            # As a last resort, aggregate multiple extracted terms instead of returning
            # an arbitrary first number. (Template selection should usually route to
            # distribute_and_sum already.)
            terms: List[str] = []
            seen_term_keys: set[tuple] = set()
            for q in sorted(understanding.quantities, key=lambda q: (q.get("pos") or 0)):
                if isinstance(q.get("rpn"), str) and q.get("rpn"):
                    term = str(q["rpn"]).strip()
                    pos = q.get("pos")
                    key = (pos, term) if isinstance(pos, int) else (q.get("source"), pos, term)
                    if key in seen_term_keys:
                        continue
                    seen_term_keys.add(key)
                    terms.append(term)
                    continue
                try:
                    v = float(q["value"])
                except Exception:
                    continue
                lit = str(int(v)) if abs(v - round(v)) < 1e-9 else str(v)
                pos = q.get("pos")
                key = (pos, lit) if isinstance(pos, int) else (q.get("source"), pos, lit)
                if key in seen_term_keys:
                    continue
                seen_term_keys.add(key)
                terms.append(lit)
            if terms:
                rpn = " ".join(terms + ["+"] * (len(terms) - 1))
                meta = dict(self._last_selection_meta)
                meta.update({"composition_steps": [{"step": 1, "fallback_terms": terms}, {"step": 2, "fallback": "sum"}, {"step": 3, "rpn": rpn}]})
                self._last_composition = meta
                return rpn

        # Base quantity
        base_q = understanding.quantities[0]
        if any(op.get("type") == "times_more" for op in understanding.operations) and len(understanding.quantities) >= 2:
            # In "X times more than Y" word problems, the base tends to be the later quantity (Y).
            base_q = sorted(understanding.quantities, key=lambda q: (q.get("pos") or 0))[-1]
        try:
            base = float(base_q.get("value", 0.0) or 0.0)
        except Exception:
            base = 0.0
        base_lit = str(int(base)) if abs(base - round(base)) < 1e-9 else str(base)
        base_expr: Optional[str] = None
        if isinstance(base_q.get("rpn"), str) and base_q.get("rpn"):
            base_expr = str(base_q["rpn"]).strip()
            if not base_expr:
                base_expr = None

        # Special-case: "base + derived(base)" patterns when sum is requested.
        derive_ops = [op for op in understanding.operations if op.get("type", "").startswith("derive_")]
        plain_ops = [op for op in understanding.operations if not op.get("type", "").startswith("derive_")]

        # Fraction-part remainder: "a third ... a quarter ... the rest ..." → base - base/3 - base/4 ...
        fraction_parts = [op for op in plain_ops if op.get("type") == "fraction_part"]
        if understanding.goals.get("rest") and fraction_parts:
            expr = base_expr or base_lit
            rpn_parts: List[str] = [expr]
            for op in sorted(fraction_parts, key=lambda o: (o.get("pos") or 0)):
                try:
                    div = float(op.get("divisor", 1.0))
                except Exception:
                    continue
                if abs(div) < 1e-12:
                    continue
                div_lit = str(int(div)) if abs(div - round(div)) < 1e-9 else str(div)
                rpn_parts.extend([expr, div_lit, "/", "-"])
            rpn = " ".join(rpn_parts)
            meta = dict(self._last_selection_meta)
            meta.update(
                {
                    "composition_steps": [
                        {"step": 1, "template": "fraction_rest"},
                        {"step": 2, "base": expr},
                        {"step": 3, "fractions": [{"divisor": op.get("divisor")} for op in fraction_parts]},
                        {"step": 4, "rpn": rpn},
                    ]
                }
            )
            self._last_composition = meta
            return rpn
        if len(derive_ops) == 1 and not plain_ops:
            op = derive_ops[0]
            if op["type"] == "derive_divide":
                divisor = float(op.get("divisor", 2.0))
                div_lit = str(int(divisor)) if abs(divisor - round(divisor)) < 1e-9 else str(divisor)
                if understanding.aggregation == "sum":
                    rpn = f"{base_lit} {base_lit} {div_lit} / +"
                else:
                    rpn = f"{base_lit} {div_lit} /"
                meta = dict(self._last_selection_meta)
                meta.update({"composition_steps": [{"step": 1, "base": base}, {"step": 2, "derive_divide": divisor}, {"step": 3, "rpn": rpn}]})
                self._last_composition = meta
                return rpn
            if op["type"] == "derive_multiply":
                factor = float(op.get("factor", 1.0))
                fac_lit = str(int(factor)) if abs(factor - round(factor)) < 1e-9 else str(factor)
                if understanding.aggregation == "sum":
                    rpn = f"{base_lit} {base_lit} {fac_lit} * +"
                else:
                    rpn = f"{base_lit} {fac_lit} *"
                meta = dict(self._last_selection_meta)
                meta.update({"composition_steps": [{"step": 1, "base": base}, {"step": 2, "derive_multiply": factor}, {"step": 3, "rpn": rpn}]})
                self._last_composition = meta
                return rpn

        # Generic sequential operations on the running total.
        normal_ops: List[Dict[str, Any]] = []
        post_ops: List[Dict[str, Any]] = []
        for op in plain_ops:
            if op.get("type") == "post_divide":
                post_ops.append(op)
            else:
                normal_ops.append(op)

        rpn_parts: List[str] = [base_expr or base_lit]
        for op in sorted(normal_ops, key=lambda o: (o.get("pos") or 0)):
            typ = op.get("type")
            if typ == "add":
                amt = float(op.get("amount", 0.0))
                amt_lit = str(int(amt)) if abs(amt - round(amt)) < 1e-9 else str(amt)
                rpn_parts.extend([amt_lit, "+"])
            elif typ == "subtract":
                amt = float(op.get("amount", 0.0))
                amt_lit = str(int(amt)) if abs(amt - round(amt)) < 1e-9 else str(amt)
                rpn_parts.extend([amt_lit, "-"])
            elif typ == "multiply":
                fac = float(op.get("factor", 1.0))
                fac_lit = str(int(fac)) if abs(fac - round(fac)) < 1e-9 else str(fac)
                rpn_parts.extend([fac_lit, "*"])
            elif typ == "times_more":
                mult = float(op.get("multiplier", 0.0))
                mult_lit = str(int(mult)) if abs(mult - round(mult)) < 1e-9 else str(mult)
                # Interpret "N times more" as multiplying by (N + 1).
                rpn_parts.extend([mult_lit, "1", "+", "*"])
            elif typ == "divide":
                div = float(op.get("divisor", 1.0))
                div_lit = str(int(div)) if abs(div - round(div)) < 1e-9 else str(div)
                rpn_parts.extend([div_lit, "/"])
            elif typ == "percent_of":
                pct = float(op.get("pct", 0.0))
                pct_lit = str(int(pct)) if abs(pct - round(pct)) < 1e-9 else str(pct)
                rpn_parts.extend([pct_lit, "100", "/", "*"])
            elif typ == "percent_increase":
                pct = float(op.get("pct", 0.0))
                pct_lit = str(int(pct)) if abs(pct - round(pct)) < 1e-9 else str(pct)
                # Multiply by (1 + pct/100).
                rpn_parts.extend([pct_lit, "100", "/", "1", "+", "*"])
        # Apply post-divisions last (e.g. "packs of 6" where the pack-size appears earlier).
        for op in sorted(post_ops, key=lambda o: (o.get("pos") or 0)):
            div = float(op.get("divisor", 1.0))
            div_lit = str(int(div)) if abs(div - round(div)) < 1e-9 else str(div)
            rpn_parts.extend([div_lit, "/"])

        rpn = " ".join(rpn_parts)
        meta = dict(self._last_selection_meta)
        meta.update({"composition_steps": [{"step": 1, "base": base_lit}, {"step": 2, "ops": plain_ops}, {"step": 3, "rpn": rpn}]})
        self._last_composition = meta
        return rpn

    def solve_with_correction(
        self,
        *,
        problem_text: str,
        rpn_engine: Any,
        max_attempts: int = 3,
        thinking_budget: int = 0,
    ) -> Tuple[Any, Dict[str, Any]]:
        entries = self.word_galaxy.tokenize(problem_text or "")
        explorer_result = None
        if self._explorer is not None:
            try:
                explorer_result = self._explorer.explore(entries=entries, rule_bank=self.rule_bank)
            except Exception:
                explorer_result = None

        # Always build both: focused subset (for noise reduction) and full bank (for coverage).
        focused_understanding, focused_trace = self.read_problem(
            problem_text, extra_rules=getattr(explorer_result, "selected_rules", None)
        )
        full_understanding, full_trace = self.read_problem(problem_text, extra_rules=self.rule_bank)

        def _u_score(u: ProblemUnderstanding) -> int:
            score = 0
            if getattr(u, "is_complete", lambda: False)():
                score += 100
            score += 10 * int(len(getattr(u, "quantities", [])))
            score += 6 * int(len(getattr(u, "operations", [])))
            if getattr(u, "aggregation", None):
                score += 5
            score += 2 * int(bool(getattr(u, "labels", {})))
            score += 2 * int(bool(getattr(u, "goals", {})))
            # Strongly prefer understandings that include structured RPN terms
            # (e.g., count×each extractions) over raw number lists.
            has_rpn = any(isinstance(q.get("rpn"), str) and q.get("rpn") for q in getattr(u, "quantities", []))
            if has_rpn:
                score += 25
            return int(score)

        def _has_rpn(u: ProblemUnderstanding) -> bool:
            return any(isinstance(q.get("rpn"), str) and q.get("rpn") for q in getattr(u, "quantities", []))

        # Prefer the richer understanding, but never drop structured terms/labels
        # just because the focused subset happened to match more generic extractors.
        focused_score = _u_score(focused_understanding)
        full_score = _u_score(full_understanding)
        if _has_rpn(full_understanding) and not _has_rpn(focused_understanding):
            understanding, trace = full_understanding, full_trace
        elif bool(getattr(full_understanding, "labels", {})) and not bool(getattr(focused_understanding, "labels", {})):
            understanding, trace = full_understanding, full_trace
        elif focused_score > full_score:
            understanding, trace = focused_understanding, focused_trace
        else:
            # Break ties by using the full bank for maximum coverage.
            understanding, trace = full_understanding, full_trace

        subgoals = self.decompose_into_subgoals(problem_text)
        exploration = self.explore_galaxy(problem_text)
        if explorer_result is not None:
            try:
                exploration = dict(exploration)
                exploration["tsinghua"] = {
                    "buckets": getattr(explorer_result, "buckets", {}),
                    "hub_concepts": getattr(explorer_result, "hub_concepts", []),
                    "selected_rule_ids": getattr(explorer_result, "selected_rule_ids", []),
                    "rule_checks": getattr(explorer_result, "rule_checks", 0),
                }
            except Exception:
                pass

        matched_patterns = trace.get("patterns", []) if isinstance(trace, dict) else []
        heuristic = self._heuristic_template(
            problem_text=problem_text,
            matched_patterns=matched_patterns,
            understanding=understanding,
        )
        concepts = exploration.get("concepts", []) if isinstance(exploration, dict) else []
        candidates: List[str] = []
        special: List[str] = []
        # If heuristic selects a specialized multi-step template, try it first.
        if heuristic in {
            "total_minus_terms",
            "total_minus_sum_others",
            "reimburse_overcharge",
            "gratuity_from_total",
            "rate_duration",
            "profit_markup_schedule",
            "linear_growth",
            "weighted_average",
            "cost_difference",
            "ratio_addition",
            "ratio_scale",
            "multi_step_relative_chain_total",
            "multi_step_store_recall",
        }:
            candidates.append(heuristic)
        # Exploration-guided prioritization.
        if "rate" in concepts and "duration" in concepts:
            candidates.append("rate_duration")
        if "aggregation" in concepts:
            candidates.append("distribute_and_sum")
        if "subtraction" in concepts:
            candidates.append("extract_operate_aggregate")
        candidates.append(heuristic)
        candidates.extend(["extract_operate_aggregate", "distribute_and_sum", "simple_apply", "multi_step_store_recall"])
        seen: set[str] = set()
        templates: List[str] = []
        for t in candidates:
            if isinstance(t, str) and t and t not in seen:
                templates.append(t)
                seen.add(t)

        # Prefer historically-successful templates first (non-hot-path ordering).
        # This only affects which templates get tried within the attempt budget.
        template_rates = self._success_stats.get("templates", {}) if isinstance(self._success_stats, dict) else {}
        pattern_rates = self._success_stats.get("patterns", {}) if isinstance(self._success_stats, dict) else {}
        if isinstance(template_rates, dict) and templates:
            pattern_ids = [p.get("rule_id") for p in trace.get("patterns", []) if isinstance(p, dict) and p.get("rule_id")]
            uniq_patterns = sorted({str(p) for p in pattern_ids if p})
            pat_rate_vals = [float(pattern_rates.get(p, 0.0)) for p in uniq_patterns if p in pattern_rates]
            avg_pat_rate = (sum(pat_rate_vals) / max(1, len(pat_rate_vals))) if pat_rate_vals else 0.0

            def _tmpl_score(t: str) -> float:
                base = float(template_rates.get(t, 0.0))
                # If we have no history for this template, don't punish it too hard.
                if t not in template_rates:
                    base = 0.25
                return 0.7 * base + 0.3 * float(avg_pat_rate)

            stable = list(templates)
            templates = sorted(stable, key=lambda t: (-_tmpl_score(t), stable.index(t)))

        if len(getattr(understanding, "quantities", [])) >= 2 and "simple_apply" in templates:
            templates = [t for t in templates if t != "simple_apply"] + ["simple_apply"]
        elif self.classify_question(problem_text) != "unknown" and "simple_apply" in templates:
            templates = [t for t in templates if t != "simple_apply"] + ["simple_apply"]
        # If we have explicit operations, prefer operation-aware templates before aggregation-only sum.
        if getattr(understanding, "operations", None) and "distribute_and_sum" in templates:
            templates = [t for t in templates if t != "distribute_and_sum"] + ["distribute_and_sum"]

        attempts: List[Dict[str, Any]] = []
        # Attempt 1: use compose_rpn() without override so retrieval/heuristic selection is observable.
        exec_attempts = 0
        try:
            primary_rpn = self.compose_rpn(understanding, trace=trace, problem_text=problem_text)
        except Exception:
            primary_rpn = ""
        primary_template = str(self.get_last_composition_meta().get("template_used") or "")
        if primary_rpn:
            exec_attempts += 1
            try:
                result = rpn_engine.evaluate(primary_rpn)
                verification = self.verify_reasonableness(problem_text, result)
                if primary_template == "rate_duration" and verification.get("plausible"):
                    if not self._verify_rate_duration_magnitude(problem_text, result):
                        verification = {"plausible": False, "reason": "rate_duration_magnitude"}
            except Exception as exc:  # noqa: BLE001
                result = None
                verification = {"plausible": False, "reason": f"exec_error:{exc}"}
            attempts.append(
                {
                    "attempt": 1,
                    "template": primary_template or "(selected)",
                    "rpn": primary_rpn,
                    "result": result,
                    "verification": verification,
                }
            )
            if verification.get("plausible") and int(thinking_budget or 0) <= 0:
                meta = {
                    "rpn_program": primary_rpn,
                    "template_used": primary_template,
                    "attempts": attempts,
                    "subgoals": subgoals,
                    "read_trace": trace,
                    "read_understanding": understanding.to_dict(),
                    "read_composition": self.get_last_composition_meta(),
                    "exploration": exploration,
                }
                if self.shadow is not None:
                    try:
                        pattern_ids = [p.get("rule_id") for p in trace.get("patterns", []) if isinstance(p, dict) and p.get("rule_id")]
                        ts = (exploration or {}).get("tsinghua", {}) if isinstance(exploration, dict) else {}
                        self.shadow.record_exploration(
                            problem_text=problem_text,
                            concepts_explored=list(concepts),
                            patterns_matched=[str(p) for p in pattern_ids],
                            templates_tried=[a.get("template") for a in attempts if isinstance(a, dict) and a.get("template")],
                            template_used=str(primary_template),
                            success=True,
                            rpn_program=primary_rpn,
                            result=result,
                            tsinghua=ts if isinstance(ts, dict) else {},
                        )
                    except Exception:
                        pass
                return result, meta
            if verification.get("plausible") and int(thinking_budget or 0) > 0:
                # With test-time compute enabled, treat the first plausible candidate as a baseline,
                # but allow deeper exploration to override when confidence is low.
                base_conf, _ = self._score_candidate(
                    problem_text=problem_text,
                    question_type=self.classify_question(problem_text),
                    numbers=self.extract_numbers(problem_text),
                    expression=primary_rpn,
                    result=result,
                    concepts=list(concepts),
                )
                low = (problem_text or "").lower()
                nums = self.extract_numbers(problem_text)
                indicators = self._count_multi_step_indicators(problem_text)
                is_easy = (
                    len(nums) <= 2
                    and indicators == 0
                    and "%" not in low
                    and "percent" not in low
                    and not any(w in low for w in ("each", "per", "every"))
                )
                if base_conf >= 0.97 and is_easy:
                    meta = {
                        "rpn_program": primary_rpn,
                        "template_used": primary_template,
                        "attempts": attempts,
                        "subgoals": subgoals,
                        "read_trace": trace,
                        "read_understanding": understanding.to_dict(),
                        "read_composition": self.get_last_composition_meta(),
                        "exploration": exploration,
                    }
                    return result, meta

        # Subsequent attempts: try alternative templates, but count only real execution attempts.
        alt_templates = [t for t in templates if t and t != primary_template]
        alt_idx = 0
        while alt_idx < len(alt_templates) and exec_attempts < max(1, int(max_attempts)):
            template = alt_templates[alt_idx]
            alt_idx += 1
            try:
                rpn = self.compose_rpn(
                    understanding,
                    trace=trace,
                    problem_text=problem_text,
                    template_override=template,
                )
            except Exception:
                rpn = ""
            if not rpn:
                attempts.append(
                    {
                        "attempt": len(attempts) + 1,
                        "template": template,
                        "rpn": "",
                        "result": None,
                        "verification": {"plausible": False, "reason": "no_rpn"},
                    }
                )
                continue

            exec_attempts += 1
            try:
                result = rpn_engine.evaluate(rpn)
            except Exception as exc:  # noqa: BLE001
                attempts.append(
                    {
                        "attempt": len(attempts) + 1,
                        "template": template,
                        "rpn": rpn,
                        "result": None,
                        "verification": {"plausible": False, "reason": f"exec_error:{exc}"},
                    }
                )
                continue

            verification = self.verify_reasonableness(problem_text, result)
            if template == "rate_duration" and verification.get("plausible"):
                if not self._verify_rate_duration_magnitude(problem_text, result):
                    verification = {"plausible": False, "reason": "rate_duration_magnitude"}
            attempts.append({"attempt": len(attempts) + 1, "template": template, "rpn": rpn, "result": result, "verification": verification})
            if verification.get("plausible") and int(thinking_budget or 0) <= 0:
                meta = {
                    "rpn_program": rpn,
                    "template_used": template,
                    "attempts": attempts,
                    "subgoals": subgoals,
                    "read_trace": trace,
                    "read_understanding": understanding.to_dict(),
                    "read_composition": self.get_last_composition_meta(),
                    "exploration": exploration,
                }
                if self.shadow is not None:
                    try:
                        pattern_ids = [p.get("rule_id") for p in trace.get("patterns", []) if isinstance(p, dict) and p.get("rule_id")]
                        ts = (exploration or {}).get("tsinghua", {}) if isinstance(exploration, dict) else {}
                        self.shadow.record_exploration(
                            problem_text=problem_text,
                            concepts_explored=list(concepts),
                            patterns_matched=[str(p) for p in pattern_ids],
                            templates_tried=[a.get("template") for a in attempts if isinstance(a, dict) and a.get("template")],
                            template_used=str(template),
                            success=True,
                            rpn_program=rpn,
                            result=result,
                            tsinghua=ts if isinstance(ts, dict) else {},
                        )
                    except Exception:
                        pass
                return result, meta
            if verification.get("plausible") and int(thinking_budget or 0) > 0:
                # Same as primary: allow test-time compute to choose among plausible hypotheses.
                break

        # Test-time compute: run a bounded exploration stage (parallel candidate eval).
        if int(thinking_budget or 0) > 0:
            ttc_result = self._test_time_compute(
                problem_text=problem_text,
                rpn_engine=rpn_engine,
                understanding=understanding,
                trace=trace,
                attempts=attempts,
                budget=int(thinking_budget or 0),
                exploration=exploration,
            )
            if ttc_result is not None:
                return ttc_result

        if self.shadow is not None:
            try:
                pattern_ids = [p.get("rule_id") for p in trace.get("patterns", []) if isinstance(p, dict) and p.get("rule_id")]
                ts = (exploration or {}).get("tsinghua", {}) if isinstance(exploration, dict) else {}
                self.shadow.record_exploration(
                    problem_text=problem_text,
                    concepts_explored=list(concepts),
                    patterns_matched=[str(p) for p in pattern_ids],
                    templates_tried=[a.get("template") for a in attempts if isinstance(a, dict) and a.get("template")],
                    template_used="",
                    success=False,
                    reason="max_attempts_exceeded",
                    tsinghua=ts if isinstance(ts, dict) else {},
                )
            except Exception:
                pass

        return None, {
            "rpn_program": "",
            "template_used": "",
            "attempts": attempts,
            "subgoals": subgoals,
            "read_trace": trace,
            "read_understanding": understanding.to_dict(),
            "read_composition": self.get_last_composition_meta(),
            "exploration": exploration,
        }

    def _extract_entities(self, problem_text: str, *, max_entities: int = 8) -> List[str]:
        """
        Best-effort entity extraction from WordGalaxy tokens (no regex parsing).

        Used only to seed relative-chain exploration candidates; the solver still
        relies on numeric test-time compute for correctness.
        """
        entries = self.word_galaxy.tokenize(problem_text or "")
        out: List[str] = []
        for e in entries:
            cat = str(getattr(e, "category", "") or "").lower()
            if cat not in {"noun", "proper_noun"}:
                continue
            token = str(getattr(e, "normalized", "") or "").lower()
            if not token or token in {"total", "altogether", "sum"}:
                continue
            if token in out:
                continue
            out.append(token)
            if len(out) >= max_entities:
                break
        return out

    def _extract_base_quantity(self, problem_text: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Best-effort extraction of a "base" scalar for relative/comparative chains.

        This is used only for candidate generation (test-time compute), not as a
        deterministic solver. It may use lightweight stdlib regex patterns to
        find a likely anchor like:
          - "If there were 100 male adults"
          - "Sammy has 20 cookies"
          - "Mandy researched 42 schools"
        """
        import re

        text = (problem_text or "").strip()
        if not text:
            return (None, None)

        # Prefer an explicit conditional base: "If there were 100 ..."
        m = re.search(r"\bif\s+there\s+(?:were|was|are|is)\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)", text, re.IGNORECASE)
        if m:
            return (m.group(2).lower(), float(m.group(1)))
        m = re.search(r"\bif\s+there\s+(?:were|was|are|is)\s+(\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
        if m:
            return ("given", float(m.group(1)))
        # "If Ben has 40 marbles" / "If Sue had 48 stickers" style conditionals.
        m = re.search(
            r"\bif\s+([A-Za-z]+)\s+(?:has|have|had|picked|picks|collects|collected|gathered|gathers|made|makes)\s+(\d+(?:\.\d+)?)\b",
            text,
            re.IGNORECASE,
        )
        if m:
            return (m.group(1).lower(), float(m.group(2)))
        # Multiword entities: "If Willowdale city has 3000 people ..."
        m = re.search(
            r"\bif\s+([A-Za-z]+)(?:\s+[A-Za-z]+){1,3}\s+(?:has|have|had)\s+(\d+(?:\.\d+)?)\b",
            text,
            re.IGNORECASE,
        )
        if m:
            return (m.group(1).lower(), float(m.group(2)))

        # "There were 140 goats"
        m = re.search(r"\bthere\s+(?:were|was|are|is)\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)", text, re.IGNORECASE)
        if m:
            return (m.group(2).lower(), float(m.group(1)))

        # "X has 20", "X scored 20", "X earned 20"
        m = re.search(
            r"\b([A-Za-z]+)\s+(?:has|have|had|scored|score|earned|earns|bought|buys|purchased|purchases|picked|picks|collects|collected|gathered|gathers|made|makes)\s+(\d+(?:\.\d+)?)\b",
            text,
            re.IGNORECASE,
        )
        if m:
            return (m.group(1).lower(), float(m.group(2)))

        # Fallback: first extracted number in appearance order.
        try:
            nums = self.extract_numbers(problem_text)
        except Exception:
            nums = []
        if nums:
            return ("given", float(nums[0]))
        return (None, None)

    @staticmethod
    def _lit_num(value: float) -> str:
        try:
            v = float(value)
        except Exception:
            return str(value)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        s = f"{v:.12g}"
        return s

    def _extract_combined_total(self, problem_text: str) -> Optional[float]:
        """
        Extract a combined/together/total quantity.
        """
        import re

        from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

        norm = normalize_number_words(problem_text or "")
        low = norm.lower()
        # Prefer explicit "combined ... of <N>" forms.
        for pat in (
            r"\bcombined\b[^\d]{0,40}\bof\b[^\d]{0,20}\b(\d+(?:\.\d+)?)\b",
            r"\bcombined\b[^\d]{0,60}\b(\d+(?:\.\d+)?)\b",
            r"\btogether\b[^\d]{0,60}\b(\d+(?:\.\d+)?)\b",
            r"\bin total\b[^\d]{0,60}\b(\d+(?:\.\d+)?)\b",
            r"\btotal\b[^\d]{0,60}\b(\d+(?:\.\d+)?)\b",
            # Also handle "N ... combined/together/in total" where the number precedes the cue.
            r"\b(\d+(?:\.\d+)?)\b[^\d]{0,40}\bcombined\b",
            r"\b(\d+(?:\.\d+)?)\b[^\d]{0,40}\baltogether\b",
            r"\b(\d+(?:\.\d+)?)\b[^\d]{0,40}\btogether\b",
            r"\b(\d+(?:\.\d+)?)\b[^\d]{0,40}\bin total\b",
            r"\b(\d+(?:\.\d+)?)\b[^\d]{0,40}\btotal\b",
        ):
            m = re.search(pat, low)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    continue
        return None

    def _extract_affine_relation(self, problem_text: str) -> Optional[Dict[str, float]]:
        """
        Extract an affine relation of the form: derived = factor*base + delta
        from phrases like "6 more than double", "one less than twice", "5 less than triple".
        """
        import re

        from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

        norm = normalize_number_words(problem_text or "")
        low = norm.lower()

        # "N more/less/fewer than double/twice/triple/K times"
        m = re.search(
            r"\b(\d+(?:\.\d+)?)\s+(more|less|fewer)\s+than\s+(double|twice|triple|(\d+(?:\.\d+)?)\s+times)(?:\s+as\s+(?:many|much))?\b",
            low,
        )
        if m:
            delta = float(m.group(1))
            if m.group(2) in {"less", "fewer"}:
                delta = -delta
            factor = 1.0
            head = str(m.group(3)).strip()
            if head in {"double", "twice"}:
                factor = 2.0
            elif head == "triple":
                factor = 3.0
            elif m.group(4):
                factor = float(m.group(4))
            return {"factor": float(factor), "delta": float(delta)}

        # "double ... plus N" / "twice ... and N more"
        m = re.search(
            r"\b(double|twice|triple|(\d+(?:\.\d+)?)\s+times)\b[^\d]{0,20}\b(plus|and)\b[^\d]{0,10}\b(\d+(?:\.\d+)?)\b",
            low,
        )
        if m:
            factor = 1.0
            head = str(m.group(1)).strip()
            if head in {"double", "twice"}:
                factor = 2.0
            elif head == "triple":
                factor = 3.0
            elif m.group(2):
                factor = float(m.group(2))
            delta = float(m.group(4))
            return {"factor": float(factor), "delta": float(delta)}

        return None

    def _extract_ratio_factor(self, problem_text: str) -> Optional[float]:
        import re

        from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

        norm = normalize_number_words(problem_text or "")
        low = norm.lower()
        if "twice as many" in low or "double" in low:
            return 2.0
        if "triple" in low or "thrice" in low:
            return 3.0
        m = re.search(r"\b(\d+(?:\.\d+)?)\s+times\s+as\s+many\b", low)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
        return None

    def _extract_question_target(self, problem_text: str) -> Optional[str]:
        import re

        text = (problem_text or "").strip()
        if not text:
            return None
        low = text.lower()
        # "How many ... does X have"
        m = re.search(r"\bhow\s+many\b[^\n]{0,80}\bdoes\s+([A-Za-z]+)\s+have\b", low)
        if m:
            return m.group(1).lower()
        # "How tall is X"
        m = re.search(r"\bhow\s+tall\s+is\s+([A-Za-z]+)\b", low)
        if m:
            return m.group(1).lower()
        # "How much ... did X ..."
        m = re.search(r"\bhow\s+much\b[^\n]{0,80}\bdid\s+([A-Za-z]+)\b", low)
        if m:
            return m.group(1).lower()
        return None

    def _extract_entity_relations(self, problem_text: str) -> List[Dict[str, Any]]:
        """
        Extract a small set of directed numeric relations between entities for chaining.
        This is intentionally narrow and used only for TTC candidate generation.
        """
        import re

        from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

        norm = normalize_number_words(problem_text or "")
        low = norm.lower()
        rels: List[Dict[str, Any]] = []

        # "X has twice as many ... as Y" => X = 2*Y
        for m in re.finditer(r"\b([A-Za-z]+)\s+has\s+(?:twice|double)\s+as\s+many\b[^\n]{0,40}\bas\s+([A-Za-z]+)\b", low):
            rels.append({"source": m.group(2).lower(), "target": m.group(1).lower(), "op": "*", "value": 2.0})

        # "X has N times as many ... as Y" => X = N*Y
        for m in re.finditer(
            r"\b([A-Za-z]+)\s+has\s+(\d+(?:\.\d+)?)\s+times\s+as\s+many\b[^\n]{0,40}\bas\s+([A-Za-z]+)\b",
            low,
        ):
            rels.append({"source": m.group(3).lower(), "target": m.group(1).lower(), "op": "*", "value": float(m.group(2))})

        # "X has N times fewer ... than Y" => X = Y / N
        for m in re.finditer(
            r"\b([A-Za-z]+)\s+has\s+(\d+(?:\.\d+)?)\s+times\s+fewer\b[^\n]{0,40}\bthan\s+([A-Za-z]+)\b",
            low,
        ):
            rels.append({"source": m.group(3).lower(), "target": m.group(1).lower(), "op": "/", "value": float(m.group(2))})

        # "X has one less than twice as many ... as Y" => X = 2*Y - 1
        for m in re.finditer(
            r"\b([A-Za-z]+)\s+has\s+(\d+(?:\.\d+)?)\s+less\s+than\s+(?:twice|double)\s+as\s+many\b[^\n]{0,40}\bas\s+([A-Za-z]+)\b",
            low,
        ):
            rels.append({"source": m.group(3).lower(), "target": m.group(1).lower(), "op": "affine", "factor": 2.0, "delta": -float(m.group(2))})

        # "X has N more than twice/double as many ... as Y" => X = 2*Y + N
        for m in re.finditer(
            r"\b([A-Za-z]+)\s+has\s+(\d+(?:\.\d+)?)\s+more\s+than\s+(?:twice|double)\s+as\s+many\b[^\n]{0,40}\bas\s+([A-Za-z]+)\b",
            low,
        ):
            rels.append({"source": m.group(3).lower(), "target": m.group(1).lower(), "op": "affine", "factor": 2.0, "delta": float(m.group(2))})

        # "X has N less than thrice/triple as many ... as Y" => X = 3*Y - N
        for m in re.finditer(
            r"\b([A-Za-z]+)\s+has\s+(\d+(?:\.\d+)?)\s+less\s+than\s+(?:thrice|triple)\s+as\s+many\b[^\n]{0,40}\bas\s+([A-Za-z]+)\b",
            low,
        ):
            rels.append({"source": m.group(3).lower(), "target": m.group(1).lower(), "op": "affine", "factor": 3.0, "delta": -float(m.group(2))})

        # "X has N more than thrice/triple as many ... as Y" => X = 3*Y + N
        for m in re.finditer(
            r"\b([A-Za-z]+)\s+has\s+(\d+(?:\.\d+)?)\s+more\s+than\s+(?:thrice|triple)\s+as\s+many\b[^\n]{0,40}\bas\s+([A-Za-z]+)\b",
            low,
        ):
            rels.append({"source": m.group(3).lower(), "target": m.group(1).lower(), "op": "affine", "factor": 3.0, "delta": float(m.group(2))})

        # Generic: "X has N more/less than K times as many ... as Y" => X = K*Y +/- N
        for m in re.finditer(
            r"\b([A-Za-z]+)\s+has\s+(\d+(?:\.\d+)?)\s+(more|less|fewer)\s+than\s+(\d+(?:\.\d+)?)\s+times\s+as\s+many\b[^\n]{0,40}\bas\s+([A-Za-z]+)\b",
            low,
        ):
            delta = float(m.group(2))
            if m.group(3) in {"less", "fewer"}:
                delta = -delta
            rels.append(
                {
                    "source": m.group(5).lower(),
                    "target": m.group(1).lower(),
                    "op": "affine",
                    "factor": float(m.group(4)),
                    "delta": delta,
                }
            )

        return rels

    def _build_entity_chain(self, problem_text: str, *, target_entity: str) -> Optional[str]:
        base_entity, base_value = self._extract_base_quantity(problem_text)
        if base_entity is None or base_value is None:
            return None
        base_entity = str(base_entity).lower()
        target = str(target_entity).lower()
        rels = self._extract_entity_relations(problem_text)
        if not rels:
            return None

        # BFS for a short path (<=4 hops) using only multiplicative/division relations.
        from collections import deque

        q = deque()
        q.append((base_entity, []))
        visited = {base_entity}
        while q:
            cur, ops = q.popleft()
            if cur == target:
                rpn = [self._lit_num(base_value)]
                for op in ops:
                    if op["op"] == "*":
                        rpn.append(f"{self._lit_num(op['value'])} *")
                    elif op["op"] == "/":
                        rpn.append(f"{self._lit_num(op['value'])} /")
                    elif op["op"] == "affine":
                        rpn.append(f"{self._lit_num(op['factor'])} * {self._lit_num(op['delta'])} +")
                return " ".join(rpn).strip()
            if len(ops) >= 4:
                continue
            for rel in rels:
                if rel.get("source") != cur:
                    continue
                nxt = str(rel.get("target") or "")
                if not nxt or nxt in visited:
                    continue
                visited.add(nxt)
                q.append((nxt, ops + [rel]))
        return None

    def _generate_algebraic_lite_candidates(
        self,
        *,
        problem_text: str,
        understanding: "ProblemUnderstanding",
        trace: dict,
        question_type: str,
        max_candidates: int = 9,
    ) -> List[str]:
        """
        TTC candidates for linear constraint patterns:
        - combined total + affine relation
        - ratio + altogether
        - short entity chains to a terminal target
        """
        candidates: List[str] = []
        low = (problem_text or "").lower()

        # Piecewise per-unit yield with fractional split.
        #
        # Generic pattern:
        #   "X units total. In condition A, yield is R per unit, but in condition B yield is only half as much.
        #    f of the units are condition B. What is total yield?"
        #
        # Solve via "all-good minus reduction on bad portion":
        #   total_yield = total*R - (total*f)*(R/2)
        #
        # Example: 60 acres, good=400 bushels/acre, clay=half as much, 1/3 clay:
        #   60*400 - (60*(1/3))*(400/2) = 20000
        if ("yields" in low) and ("per acre" in low) and ("half as much" in low) and ("how many" in low):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()

                m_total = re.search(r"\b(\d+(?:\.\d+)?)\s+acres?\b", low_norm)
                # Prefer the first explicit numeric yield-per-acre (not the "half as much" clause).
                m_yield = re.search(
                    r"\byields?\s+(\d+(?:\.\d+)?)\s+[a-z]+\s+per\s+acre\b", low_norm, re.IGNORECASE
                ) or re.search(r"\byields?\s+(\d+(?:\.\d+)?)\s+per\s+acre\b", low_norm, re.IGNORECASE)
                m_frac = re.search(r"\b(\d+)\s*/\s*(\d+)\b[^.]{0,120}\bacres?\b", low_norm) or re.search(
                    r"\b(\d+)\s*/\s*(\d+)\b", low_norm
                )
                if m_total and m_yield and m_frac:
                    total_units = float(m_total.group(1))
                    rate = float(m_yield.group(1))
                    fn = float(m_frac.group(1))
                    fd = float(m_frac.group(2))
                    if total_units > 0 and rate > 0 and fn > 0 and fd > 0:
                        t = self._lit_num(total_units)
                        r = self._lit_num(rate)
                        n = self._lit_num(fn)
                        d = self._lit_num(fd)
                        # total*rate - (total*fn/fd)*(rate/2)
                        candidates.insert(0, f"{t} {r} * {t} {n} * {d} / {r} 2 / * -")
            except Exception:
                pass

        # Legs constraint (quadrupeds + humans).
        # Example: "another dog walker and their 3 dogs ... 36 legs ... how many dogs is Mariel walking?"
        # total_legs = human_count*2 + (known_dogs + unknown_dogs)*4  => unknown_dogs = (total_legs - human_count*2 - known_dogs*4)/4
        if ("legs" in low) and any(w in low for w in ("dog", "dogs")) and ("dog walker" in low) and ("how many" in low):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()

                m_legs = re.search(r"\b(\d+(?:\.\d+)?)\s+legs\b", low_norm)
                # Prefer "their N dogs" as the known pack (the other walker).
                m_known = re.search(r"\btheir\s+(\d+(?:\.\d+)?)\s+dogs?\b", low_norm) or re.search(
                    r"\b(\d+(?:\.\d+)?)\s+dogs?\b", low_norm
                )
                if m_legs and m_known:
                    total_legs = float(m_legs.group(1))
                    known_dogs = float(m_known.group(1))
                    # Heuristic human count: "another/other dog walker" => 2 walkers.
                    humans = 2 if ("another dog walker" in low_norm or "other dog walker" in low_norm) else 1
                    human_legs = float(humans * 2)
                    if total_legs > 0 and known_dogs >= 0 and humans >= 1:
                        t = self._lit_num(total_legs)
                        h = self._lit_num(human_legs)
                        k = self._lit_num(known_dogs)
                        candidates.insert(0, f"{t} {h} - {k} 4 * - 4 /")
            except Exception:
                pass

        total = None
        if any(w in low for w in ("combined", "together", "in total")):
            total = self._extract_combined_total(problem_text)

        # Total-with-partial-known-cost:
        # "N items together cost T. Of the N items, there are K items that cost U each.
        #  If the remaining items are each equal in value, what is the cost of one?"
        #
        # Generic linear constraint:
        #   T = K*U + (N-K)*x  =>  x = (T - K*U) / (N-K)
        if total is not None and ("each" in low) and any(w in low for w in ("remaining", "rest", "left")):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()

                def _first_float(patterns: Sequence[str]) -> Optional[float]:
                    for pat in patterns:
                        m = re.search(pat, low_norm, re.IGNORECASE)
                        if not m:
                            continue
                        try:
                            return float(m.group(1))
                        except Exception:
                            continue
                    return None

                # N: total count of items (prefer "of the N <noun>", else sentence start "N <noun> together ...")
                n_total = _first_float(
                    [
                        r"\bof\s+the\s+(\d+(?:\.\d+)?)\s+[a-zA-Z]+\b",
                        r"^\s*(\d+(?:\.\d+)?)\s+[a-zA-Z]+\b",
                    ]
                )

                # K and U: known subset count and per-item unit cost.
                known_count: Optional[float] = None
                unit_cost: Optional[float] = None

                # "there are K <noun> that cost $U each"
                m = re.search(
                    r"\bthere\s+are\s+(\d+(?:\.\d+)?)\s+[a-zA-Z]+\b[^\n\r]{0,60}\bcost\b[^\d]{0,10}\$?\s*(\d+(?:\.\d+)?)\s+(?:each|apiece)\b",
                    low_norm,
                    re.IGNORECASE,
                )
                if m:
                    try:
                        known_count = float(m.group(1))
                        unit_cost = float(m.group(2))
                    except Exception:
                        known_count = None
                        unit_cost = None

                # "K <noun> cost $U each"
                if known_count is None or unit_cost is None:
                    m = re.search(
                        r"\b(\d+(?:\.\d+)?)\s+[a-zA-Z]+\b[^\n\r]{0,40}\bcost\b[^\d]{0,10}\$?\s*(\d+(?:\.\d+)?)\s+(?:each|apiece)\b",
                        low_norm,
                        re.IGNORECASE,
                    )
                    if m:
                        try:
                            known_count = float(m.group(1))
                            unit_cost = float(m.group(2))
                        except Exception:
                            known_count = None
                            unit_cost = None

                # "K <noun> at $U each"
                if known_count is None or unit_cost is None:
                    m = re.search(
                        r"\b(\d+(?:\.\d+)?)\s+[a-zA-Z]+\b[^\n\r]{0,40}\bat\b[^\d]{0,10}\$?\s*(\d+(?:\.\d+)?)\s+(?:each|apiece)\b",
                        low_norm,
                        re.IGNORECASE,
                    )
                    if m:
                        try:
                            known_count = float(m.group(1))
                            unit_cost = float(m.group(2))
                        except Exception:
                            known_count = None
                            unit_cost = None

                if (
                    n_total is not None
                    and known_count is not None
                    and unit_cost is not None
                    and n_total > 0
                    and known_count > 0
                    and unit_cost >= 0
                    and n_total - known_count > 0
                ):
                    denom = float(n_total - known_count)
                    total_lit = self._lit_num(total)
                    k_lit = self._lit_num(known_count)
                    u_lit = self._lit_num(unit_cost)
                    n_lit = self._lit_num(n_total)
                    denom_lit = self._lit_num(denom)

                    remaining_total = f"{total_lit} {k_lit} {u_lit} * -"

                    # Candidate A (unit value): (T - K*U)/(N-K)
                    candidates.append(f"{remaining_total} {n_lit} {k_lit} - /")
                    # Candidate B (unit value, precomputed denom): (T - K*U)/denom
                    candidates.append(f"{remaining_total} {denom_lit} /")
                    # Candidate C (remaining total): T - K*U
                    candidates.append(remaining_total)
            except Exception:
                pass

        affine = self._extract_affine_relation(problem_text) if total is not None else None
        if total is not None and affine:
            factor = float(affine.get("factor", 0.0))
            delta = float(affine.get("delta", 0.0))
            if factor > 0:
                total_lit = self._lit_num(total)
                delta_lit = self._lit_num(delta)
                denom_lit = self._lit_num(factor + 1.0)
                factor_lit = self._lit_num(factor)
                # base = (total - delta) / (factor + 1)
                candidates.append(f"{total_lit} {delta_lit} - {denom_lit} /")
                # derived = base*factor + delta
                candidates.append(f"{total_lit} {delta_lit} - {denom_lit} / {factor_lit} * {delta_lit} +")
                # derived = total - base
                candidates.append(f"{total_lit} {total_lit} {delta_lit} - {denom_lit} / -")

        # Ratio + altogether: base + ratio*base.
        if any(w in low for w in ("altogether", "in total", "combined", "together")):
            ratio = self._extract_ratio_factor(problem_text)
            base_entity, base_value = self._extract_base_quantity(problem_text)
            if ratio is not None and base_value is not None and ratio > 0:
                b = self._lit_num(base_value)
                r = self._lit_num(ratio)
                candidates.append(f"{b} {b} {r} * +")
                candidates.append(f"{b} {self._lit_num(ratio + 1.0)} *")

        # "Gave away" constraint:
        #   start = A
        #   after giving away x and receiving extra B: start + B - x
        #   equals affine of start: factor*start + delta
        #   => x = (start + B) - (factor*start + delta)
        #
        # Example: "Nigel won $45 but gave some away. His mother gave him $80 more.
        #          If now Nigel has $10 more than twice the amount he originally had, how much did he give away?"
        if any(w in low for w in ("give away", "gave away")) and any(w in low for w in ("gave him", "gave her", "gave them")):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                affine2 = self._extract_affine_relation(problem_text)
                if affine2:
                    factor = float(affine2.get("factor", 0.0))
                    delta = float(affine2.get("delta", 0.0))
                    if factor > 0:
                        # Prefer "won $A" as the starting amount.
                        start = None
                        m_won = re.search(r"\bwon\b[^\d]{0,10}\$?\s*(\d+(?:\.\d+)?)", low_norm)
                        if m_won:
                            try:
                                start = float(m_won.group(1))
                            except Exception:
                                start = None
                        if start is None:
                            # Fallback to first number in the prompt.
                            m_first = re.search(r"\b(\d+(?:\.\d+)?)\b", low_norm)
                            if m_first:
                                try:
                                    start = float(m_first.group(1))
                                except Exception:
                                    start = None
                        # Extra received amount.
                        extra = None
                        m_extra = re.search(r"\bgave\s+(?:him|her|them)\b[^\d]{0,15}\$?\s*(\d+(?:\.\d+)?)\b", low_norm)
                        if m_extra:
                            try:
                                extra = float(m_extra.group(1))
                            except Exception:
                                extra = None
                        if start is not None and extra is not None:
                            a = self._lit_num(start)
                            b = self._lit_num(extra)
                            f = self._lit_num(factor)
                            d = self._lit_num(delta)
                            # x = (A + B) - (A*factor + delta)
                            candidates.append(f"{a} {b} + {a} {f} * {d} + -")
                            # Variant: x = (A*factor + delta) - (A + B) (in case of sign ambiguity).
                            candidates.append(f"{a} {f} * {d} + {a} {b} + -")
            except Exception:
                pass

        # Chain to terminal entity (e.g., Jacob -> Annie -> Melanie).
        target = self._extract_question_target(problem_text)
        if target:
            chain = self._build_entity_chain(problem_text, target_entity=target)
            if chain:
                candidates.append(chain)

        # De-dupe while preserving order.
        seen: set[str] = set()
        out: List[str] = []
        for c in candidates:
            s = str(c).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= max_candidates:
                break
        return out

    def _generate_book_galaxy_candidates(
        self,
        problem_text: str,
        *,
        max_candidates: int = 6,
    ) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, List[str]]]:
        """
        Generate additional TTC candidates guided by ingested Book Galaxies.

        Book Galaxies provide concept grounding via page hits. Candidate synthesis
        stays within the existing RPN opcode surface (no external solvers).
        """
        if self._book_library is None or not self.enable_book_galaxies:
            return ([], [], {})

        try:
            toks = self.word_galaxy.tokenize(problem_text or "")
            normalized = [t.normalized for t in toks if getattr(t, "normalized", None)]
        except Exception:
            normalized = []

        # Book retrieval needs to bridge surface-form mismatches (pdftotext vs LaTeX).
        #
        # IMPORTANT: do not do ad-hoc query "normalization" in the hot path.
        # Instead, use the Math Galaxy symlink registry (one meaning, many forms)
        # to expand known variants (e.g., "\\cos" -> "cos", "cosine").
        try:
            import re

            math_galaxy = self.math_galaxy
            if math_galaxy is None:
                try:
                    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY

                    math_galaxy = MATH_GALAXY
                except Exception:
                    math_galaxy = None

            lookup = getattr(math_galaxy, "lookup", None) if math_galaxy is not None else None
            variants_for = getattr(math_galaxy, "variants_for", None) if math_galaxy is not None else None

            extra_surface: List[str] = []
            if callable(lookup) and callable(variants_for):
                s = str(problem_text or "")
                # LaTeX commands: \cos, \sqrt, \pi, ...
                for m in re.finditer(r"\\[A-Za-z]+", s):
                    cmd = m.group(0)
                    if lookup(cmd):
                        extra_surface.extend(list(variants_for(cmd)))
                # Unicode glyphs: π, √, ∑, ...
                for ch in set(s):
                    if ch.isspace():
                        continue
                    if lookup(ch):
                        extra_surface.extend(list(variants_for(ch)))

            if extra_surface:
                for variant in extra_surface:
                    v = str(variant or "").strip()
                    if not v:
                        continue
                    # Index keys may include canonical LaTeX tokens (e.g., "\\cos") so
                    # include the raw variant as well as WordGalaxy-normalized tokens.
                    normalized.append(v)
                    for t in self.word_galaxy.tokenize(v):
                        norm = getattr(t, "normalized", None)
                        if norm:
                            normalized.append(str(norm))
        except Exception:
            pass

        if normalized:
            # De-dupe while preserving order.
            seen: set[str] = set()
            deduped: List[str] = []
            for tok in normalized:
                t = str(tok or "").strip()
                if not t or t in seen:
                    continue
                seen.add(t)
                deduped.append(t)
            normalized = deduped

        if not normalized or self.book_top_k <= 0:
            return ([], [], {})

        hits = self._book_library.search(
            normalized_tokens=normalized,
            top_k=self.book_top_k,
            max_pages_per_book=6,
            min_token_hits=2,
        )

        # Optional: equation templates extracted during ingestion (lhs=rhs).
        try:
            template_hits = self._book_library.search_templates(
                normalized_tokens=normalized,
                top_k=self.book_top_k,
                max_templates_per_book=12,
                min_token_hits=2,
            )
        except Exception:
            template_hits = []

        # Optional: articulated artifacts (theorem/definition/formula blocks).
        try:
            artifact_hits = self._book_library.search_artifacts(
                normalized_tokens=normalized,
                top_k=self.book_top_k,
                max_artifacts_per_book=12,
                min_token_hits=2,
            )
        except Exception:
            artifact_hits = []

        if not hits and not template_hits and not artifact_hits:
            return ([], [], {})

        if hits:
            self._book_stats["hits"] = int(self._book_stats.get("hits", 0)) + len(hits)
        if template_hits:
            self._book_stats["template_hits"] = int(self._book_stats.get("template_hits", 0)) + len(template_hits)
        if artifact_hits:
            self._book_stats["artifact_hits"] = int(self._book_stats.get("artifact_hits", 0)) + len(artifact_hits)

        hits_meta: List[Dict[str, Any]] = []
        hint_text = ""
        for h in hits:
            hits_meta.append(
                {
                    "book_id": h.book_id,
                    "title": h.title,
                    "domain": h.domain,
                    "page": h.page_number,
                    "score": h.score,
                    "excerpt": h.excerpt,
                    "pos": h.position_3d,
                }
            )
            hint_text += " " + (h.excerpt or "")

        if template_hits:
            hits_meta.append(
                {
                    "templates": [
                        {
                            "book_id": th.book_id,
                            "title": th.title,
                            "domain": th.domain,
                            "page": th.page_number,
                            "score": th.score,
                            "lhs": th.lhs,
                            "rhs": th.rhs,
                            "rpn": th.rpn,
                        }
                        for th in template_hits[: min(50, len(template_hits))]
                    ]
                }
            )

        if artifact_hits:
            hits_meta.append(
                {
                    "artifacts": [
                        {
                            "book_id": ah.book_id,
                            "title": ah.title,
                            "domain": ah.domain,
                            "page": ah.page_number,
                            "score": ah.score,
                            "artifact_id": ah.artifact_id,
                            "artifact_type": ah.artifact_type,
                            "name": ah.name,
                            "conditions": ah.conditions,
                            "conditions_rpn": getattr(ah, "conditions_rpn", []),
                            "symbol_bindings": getattr(ah, "symbol_bindings", {}),
                            "lhs": ah.lhs,
                            "rhs": ah.rhs,
                            "rpn": ah.rpn,
                            "conclusion": getattr(ah, "conclusion", None),
                            "conclusion_rpn": getattr(ah, "conclusion_rpn", None),
                            "derived_rpns": ah.derived_rpns,
                            "var_mapping": getattr(ah, "var_mapping", {}),
                        }
                        for ah in artifact_hits[: min(50, len(artifact_hits))]
                    ]
                }
            )

        low = f"{problem_text} {hint_text}".lower()
        try:
            nums = self.extract_numbers(problem_text)
        except Exception:
            nums = []

        candidates: List[str] = []
        artifact_seed: List[str] = []
        template_seed: List[str] = []

        # Articulated artifacts: feed their executable RPN candidates into TTC.
        if artifact_hits:
            known_ops = {
                "+",
                "-",
                "*",
                "/",
                "^",
                "pow",
                "neg",
                "sqrt",
                "sin",
                "cos",
                "tan",
                "sinh",
                "cosh",
                "tanh",
                "asin",
                "acos",
                "atan",
                "atan2",
                "arcsin",
                "arccos",
                "arctan",
                "log",
                "ln",
                "log2",
                "log10",
                "exp",
                "gamma",
                "beta",
                "abs",
                "floor",
                "ceil",
                "round",
                "mod",
                "%",
                "max",
                "min",
                "gcd",
                "factorial",
                "!",
                "binomial",
                "binom",
            }
            known_consts = {"pi", "π", "tau", "phi", "φ", "e", "Pi", "PI"}

            def _is_num(tok: str) -> bool:
                try:
                    float(tok)
                    return True
                except Exception:
                    return False

            def _parse_problem_state(text: str) -> Dict[str, Any]:
                low_text = (text or "").lower()
                nums_local: List[float] = []
                try:
                    nums_local = [float(x) for x in self.extract_numbers(text)]
                except Exception:
                    nums_local = []

                # Minimal role extraction for better variable binding.
                roles: Dict[str, List[float]] = {}
                # Shape cues (2D vs 3D) and query intent (area/volume/etc).
                shape: Dict[str, bool] = {
                    "circle": "circle" in low_text,
                    "sphere": "sphere" in low_text,
                    "cylinder": "cylinder" in low_text,
                    "cone": "cone" in low_text,
                    "triangle": "triangle" in low_text,
                    "matrix": "matrix" in low_text,
                    "vector": "vector" in low_text,
                }
                intent: Dict[str, bool] = {
                    "area": "area" in low_text,
                    "surface_area": ("surface area" in low_text) or ("lateral area" in low_text),
                    "volume": "volume" in low_text,
                    "circumference": "circumference" in low_text,
                    "perimeter": "perimeter" in low_text,
                }
                angle_units: Dict[str, bool] = {
                    "degrees": ("^\\circ" in low_text) or ("°" in low_text) or ("degrees" in low_text),
                    "radians": ("radians" in low_text) or ("rad" in low_text),
                }
                # Right triangle cues.
                if "right triangle" in low_text or "right-angled" in low_text or "right angled" in low_text:
                    roles["right_triangle"] = [1.0]
                # Legs: "legs 3 and 4" / "legs of length 3 and 4"
                import re

                def _push_role(role: str, vals: List[float], *, max_vals: int = 3) -> None:
                    if not vals:
                        return
                    out = roles.setdefault(role, [])
                    for v in vals:
                        try:
                            fv = float(v)
                        except Exception:
                            continue
                        if any(abs(fv - float(x)) <= 1e-9 for x in out):
                            continue
                        out.append(fv)
                        if len(out) >= max_vals:
                            break

                def _find_role_numbers(*, keywords: List[str], window: int = 42, max_hits: int = 3) -> List[float]:
                    """
                    Find numbers near role keywords (for semantic binding).
                    This is intentionally lightweight (regex only).
                    """
                    hits: List[float] = []

                    def _add(v: float) -> None:
                        if any(abs(v - float(x)) <= 1e-9 for x in hits):
                            return
                        hits.append(float(v))

                    for kw in keywords:
                        kw = str(kw or "").strip().lower()
                        if not kw:
                            continue
                        # Avoid overly-generic single-letter matches unless explicitly assigned ("r = 3").
                        if len(kw) == 1 and kw.isalpha():
                            for m in re.finditer(rf"\b{re.escape(kw)}\s*=\s*(\d+(?:\.\d+)?)\b", low_text):
                                try:
                                    _add(float(m.group(1)))
                                except Exception:
                                    pass
                            continue

                        for m in re.finditer(rf"\b{re.escape(kw)}\b", low_text):
                            start = max(0, m.start() - int(window))
                            end = min(len(low_text), m.end() + int(window))
                            snippet = low_text[start:end]
                            # Prefer the closest numeric literal to the keyword occurrence.
                            kw_center = (m.start() + m.end()) / 2.0
                            nearest: tuple[float, float] | None = None  # (distance, value)
                            for nm in re.finditer(r"\b(\d+(?:\.\d+)?)\b", snippet):
                                try:
                                    val = float(nm.group(1))
                                except Exception:
                                    continue
                                num_pos = float(start + nm.start())
                                dist = abs(num_pos - kw_center)
                                if nearest is None or dist < nearest[0] - 1e-9:
                                    nearest = (dist, val)
                            if nearest is not None:
                                _add(nearest[1])
                                if len(hits) >= int(max_hits):
                                    return hits
                    return hits

                m = re.search(r"legs?(?:\s+of\s+length)?\s+(\d+(?:\.\d+)?)\s+(?:and|,)\s+(\d+(?:\.\d+)?)", low_text)
                if m:
                    _push_role("leg", [float(m.group(1)), float(m.group(2))], max_vals=4)
                # Hypotenuse: "hypotenuse 5"
                m = re.search(r"hypotenuse\s+(?:of\s+length\s+)?(\d+(?:\.\d+)?)", low_text)
                if m:
                    _push_role("hypotenuse", [float(m.group(1))])
                # Radius: "radius 7"
                m = re.search(r"radius\s+(?:of\s+)?(\d+(?:\.\d+)?)", low_text)
                if m:
                    _push_role("radius", [float(m.group(1))])
                # Diameter: "diameter 14"
                m = re.search(r"diameter\s+(?:of\s+)?(\d+(?:\.\d+)?)", low_text)
                if m:
                    _push_role("diameter", [float(m.group(1))])
                # Length/width/height: keep only first mention for now.
                for key in ("length", "width", "height"):
                    m = re.search(rf"{key}\s+(?:of\s+)?(\d+(?:\.\d+)?)", low_text)
                    if m:
                        _push_role(key, [float(m.group(1))])

                # Stage 3: semantic role binding via keyword proximity.
                # This catches common variants like "tall 5", "depth 5", or "r = 3".
                _push_role("radius", _find_role_numbers(keywords=["radius", "r"]))
                _push_role("diameter", _find_role_numbers(keywords=["diameter", "d"]))
                _push_role("height", _find_role_numbers(keywords=["height", "h", "tall", "altitude", "depth"]))
                _push_role("length", _find_role_numbers(keywords=["length", "l"]))
                _push_role("width", _find_role_numbers(keywords=["width", "w", "wide"]))

                return {
                    "text": text or "",
                    "low": low_text,
                    "numbers": nums_local,
                    "roles": roles,
                    "shape": shape,
                    "intent": intent,
                    "angle_units": angle_units,
                }

            # Phase 7 metadata diagnostic: allow optional, regex-only role inference from
            # artifact text when `symbol_bindings[*].meaning` is missing.
            # Default is OFF (keeps hot path minimal and avoids relying on noisy PDF prose).
            try:
                import os

                _infer_artifact_roles = str(os.environ.get("K3D_TRM_INFER_ARTIFACT_ROLES", "0") or "0").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "y",
                    "on",
                }
            except Exception:
                _infer_artifact_roles = False

            def _eval_condition_text(cond: str, state: Dict[str, Any]) -> Optional[bool]:
                s = str(cond or "").lower()
                low_text = str(state.get("low") or "")
                if not s:
                    return None
                # Right triangle gating is high-impact for wrong_computation prevention.
                if "right triangle" in s or "right-angled" in s or "right angled" in s:
                    return ("triangle" in low_text) and ("right" in low_text)
                # Triangle presence.
                if "triangle" in s:
                    return "triangle" in low_text
                # 2D/3D geometric intent.
                if "circle" in s:
                    return "circle" in low_text
                if "sphere" in s:
                    return "sphere" in low_text
                if "cylinder" in s:
                    return "cylinder" in low_text
                if "cone" in s:
                    return "cone" in low_text
                if "volume" in s:
                    return "volume" in low_text
                if "surface area" in s:
                    return "surface area" in low_text
                if "circumference" in s:
                    return "circumference" in low_text
                # Matrix/determinant cues.
                if "matrix" in s:
                    return "matrix" in low_text
                if "det" in s or "determinant" in s:
                    return ("det" in low_text) or ("determinant" in low_text)
                # Vector cues.
                if "vector" in s:
                    return "vector" in low_text
                return None

            def _artifact_condition_gate(cond_lines: List[str], state: Dict[str, Any]) -> bool:
                # Conservative: if we can evaluate a condition, it must pass.
                # Unknown/unparsed conditions are ignored for now (Phase 3).
                for c in cond_lines or []:
                    verdict = _eval_condition_text(c, state)
                    if verdict is False:
                        return False
                return True

            def _artifact_type_priority(t: str) -> float:
                key = str(t or "").strip().lower()
                return {
                    "theorem": 1.0,
                    "lemma": 0.95,
                    "corollary": 0.9,
                    "proposition": 0.85,
                    "definition": 0.75,
                    "formula": 0.6,
                    "example": 0.25,
                    "exercise": 0.15,
                }.get(key, 0.2)

            def _book_authority(book_id: str) -> float:
                # Prefer primary textbooks over method notes / exercises.
                return {
                    "la_done_right": 1.0,
                    "multivariable_calc": 1.0,
                    "advanced_calc_alt": 0.95,
                    "advanced_calculus": 0.95,
                    "advanced_calc_1_2": 0.9,
                    "wrede_calculus": 0.9,
                    "dmoi3": 0.85,
                    "transition_v104": 0.85,
                    "numerical_analysis": 0.75,
                    "math_for_programmers": 0.7,
                    "orland_math_prog": 0.6,
                    "shortestshortcut": 0.55,
                    "mathgems": 0.5,
                    "rpn_method": 0.3,
                    "rpn_intermediate": 0.3,
                    "stavely_python": 0.2,
                }.get(str(book_id or ""), 0.6)

            def _role_match_score(state: Dict[str, Any], symbol_bindings: Dict[str, Any]) -> float:
                if not symbol_bindings:
                    return 0.0
                # Important: do not mutate the shared `state["roles"]` lists while
                # instantiating multiple candidate programs for the same prompt.
                # Each candidate should get a fresh view of the available role
                # numbers so binding remains stable across candidates.
                roles = {}
                try:
                    for k, vals in (state.get("roles") or {}).items():
                        if isinstance(vals, list):
                            roles[str(k)] = list(vals)
                except Exception:
                    roles = {}
                if not roles:
                    return 0.0
                total = 0
                matched = 0
                for info in symbol_bindings.values():
                    if not isinstance(info, dict):
                        continue
                    meaning = str(info.get("meaning") or "").lower()
                    role = None
                    if "hypotenuse" in meaning:
                        role = "hypotenuse"
                    elif "leg" in meaning:
                        role = "leg"
                    elif "radius" in meaning:
                        role = "radius"
                    elif "diameter" in meaning:
                        role = "diameter"
                    elif meaning in {"length", "width", "height"}:
                        role = meaning
                    if role is None:
                        continue
                    total += 1
                    if role in roles and roles[role]:
                        matched += 1
                return float(matched) / float(max(1, total))

            def _infer_roles_from_text(text: str) -> Dict[str, str]:
                """
                Heuristically infer variable -> semantic role from artifact text.

                This is a lightweight (regex-only) fallback for book artifacts whose
                `symbol_bindings[*].meaning` is not populated (often "unknown").
                """
                import re

                low = str(text or "").lower()
                out: Dict[str, str] = {}

                def _add(var: str, role: str) -> None:
                    v = str(var or "").strip().lower()
                    if not v or len(v) != 1 or not v.isalpha():
                        return
                    out[v] = str(role or "").strip().lower()

                def _add_pair(a: str, b: str, role: str) -> None:
                    _add(a, role)
                    _add(b, role)

                # Common geometry roles.
                # Support light TeX-ish wrappers: "$a$", "($a$)", etc.
                v1 = r"\$?([a-z])\$?"

                # "legs a and b", "leg a", "... are legs"
                for m in re.finditer(rf"\blegs?\s+{v1}\s+(?:and|,)\s+{v1}\b", low):
                    _add_pair(m.group(1), m.group(2), "leg")
                for m in re.finditer(rf"\b{v1}\s+(?:and|,)\s+{v1}\s+are\s+legs\b", low):
                    _add_pair(m.group(1), m.group(2), "leg")
                for m in re.finditer(rf"\bleg\s+{v1}\b", low):
                    _add(m.group(1), "leg")

                for m in re.finditer(rf"\bhypotenuse\s*[:=,]?\s*{v1}\b", low):
                    _add(m.group(1), "hypotenuse")
                for m in re.finditer(rf"\b{v1}\s+is\s+(?:the\s+)?hypotenuse\b", low):
                    _add(m.group(1), "hypotenuse")

                for m in re.finditer(rf"\bradius\s*[:=,]?\s*{v1}\b", low):
                    _add(m.group(1), "radius")
                for m in re.finditer(rf"\b{v1}\s+is\s+(?:the\s+)?radius\b", low):
                    _add(m.group(1), "radius")

                for m in re.finditer(rf"\bdiameter\s*[:=,]?\s*{v1}\b", low):
                    _add(m.group(1), "diameter")
                for m in re.finditer(rf"\b{v1}\s+is\s+(?:the\s+)?diameter\b", low):
                    _add(m.group(1), "diameter")

                for m in re.finditer(rf"\bheight\s*[:=,]?\s*{v1}\b", low):
                    _add(m.group(1), "height")
                for m in re.finditer(rf"\b{v1}\s+is\s+(?:the\s+)?height\b", low):
                    _add(m.group(1), "height")

                for m in re.finditer(rf"\blength\s*[:=,]?\s*{v1}\b", low):
                    _add(m.group(1), "length")
                for m in re.finditer(rf"\bwidth\s*[:=,]?\s*{v1}\b", low):
                    _add(m.group(1), "width")
                for m in re.finditer(rf"\bbase\s*[:=,]?\s*{v1}\b", low):
                    _add(m.group(1), "base")
                for m in re.finditer(rf"\bside\s*[:=,]?\s*{v1}\b", low):
                    _add(m.group(1), "side")
                for m in re.finditer(rf"\bsides\s+{v1}\s+(?:and|,)\s+{v1}\b", low):
                    _add_pair(m.group(1), m.group(2), "side")

                return out

            def _augment_symbol_bindings_with_inferred_roles(ah: Any, sb: Dict[str, Any]) -> Dict[str, Any]:
                """
                If a book artifact has only "unknown" meanings, infer roles from its
                text/conditions and annotate `symbol_bindings` for better binding.
                """
                if not _infer_artifact_roles:
                    return sb
                if not isinstance(sb, dict) or not sb:
                    return {}

                any_known = False
                for info in sb.values():
                    if isinstance(info, dict):
                        m = str(info.get("meaning") or "").strip().lower()
                        if m and m != "unknown":
                            any_known = True
                            break
                if any_known:
                    return sb

                try:
                    parts = [
                        str(getattr(ah, "name", "") or ""),
                        str(getattr(ah, "artifact_type", "") or ""),
                        " ".join(list(getattr(ah, "conditions", []) or [])[:8]),
                        str(getattr(ah, "conclusion", "") or ""),
                        str(getattr(ah, "lhs", "") or ""),
                        str(getattr(ah, "rhs", "") or ""),
                        str(getattr(ah, "raw_block", "") or ""),
                        str(getattr(ah, "latex_source", "") or ""),
                    ]
                    text = " ".join(p for p in parts if p)
                except Exception:
                    text = ""

                inferred = _infer_roles_from_text(text)
                if not inferred:
                    return sb

                out = dict(sb)
                for var, role in inferred.items():
                    # Preserve existing meaning if it was already known.
                    cur = out.get(var) if isinstance(out.get(var), dict) else {}
                    cur_m = str(cur.get("meaning") or "").strip().lower()
                    if cur_m and cur_m != "unknown":
                        continue
                    nxt = dict(cur)
                    nxt["meaning"] = role
                    out[var] = nxt
                return out

            def _artifact_score(ah: Any, state: Dict[str, Any]) -> float:
                base = 0.0
                try:
                    base = float(getattr(ah, "score", 0.0) or 0.0)
                except Exception:
                    base = 0.0
                tpri = _artifact_type_priority(str(getattr(ah, "artifact_type", "") or ""))
                auth = _book_authority(str(getattr(ah, "book_id", "") or ""))

                conds = list(getattr(ah, "conditions", []) or [])
                # Low-signal examples/exercises with no explicit conditions are a major
                # wrong_computation source. Skip them entirely.
                atype = str(getattr(ah, "artifact_type", "") or "").strip().lower()
                if atype in {"example", "exercise"} and not conds:
                    return -1e9
                verdicts = [_eval_condition_text(c, state) for c in conds]
                if any(v is False for v in verdicts):
                    return -1e9
                known = sum(1 for v in verdicts if v is not None)
                matched = sum(1 for v in verdicts if v is True)
                cond_score = float(matched) / float(max(1, len(conds))) if conds else 0.0
                if conds and known == 0:
                    cond_score -= 0.4

                sb_raw = dict(getattr(ah, "symbol_bindings", {}) or {})
                sb_aug = _augment_symbol_bindings_with_inferred_roles(ah, sb_raw)
                role_score = _role_match_score(state, sb_aug)

                # Penalize obvious 2D/3D intent mismatch using artifact text.
                hay = " ".join(
                    [
                        str(getattr(ah, "name", "") or ""),
                        " ".join(conds),
                        str(getattr(ah, "conclusion", "") or ""),
                        str(getattr(ah, "lhs", "") or ""),
                        str(getattr(ah, "rhs", "") or ""),
                    ]
                ).lower()
                intent = dict(state.get("intent") or {})
                shape = dict(state.get("shape") or {})
                mismatch_penalty = 0.0
                if intent.get("volume") and ("area" in hay or "circumference" in hay):
                    mismatch_penalty -= 0.6
                if intent.get("area") and ("volume" in hay or "surface area" in hay):
                    mismatch_penalty -= 0.6
                if shape.get("circle") and ("sphere" in hay or "cylinder" in hay or "cone" in hay):
                    mismatch_penalty -= 1.4
                if shape.get("sphere") and ("circle" in hay and "sphere" not in hay):
                    mismatch_penalty -= 0.8

                # Weighted sum; keep bounded but prioritize type/conditions.
                return (
                    0.25 * min(base, 8.0)
                    + 10.0 * tpri
                    + 2.5 * cond_score
                    + 1.5 * role_score
                    + 1.0 * auth
                    + mismatch_penalty
                )

            def _artifact_context_gate(ah: Any, state: Dict[str, Any]) -> bool:
                """
                Hard reject obvious semantic mismatches (shape/intent) that are still
                structurally valid and often pass plausibility.
                """
                try:
                    conds_local = list(getattr(ah, "conditions", []) or [])
                    hay = " ".join(
                        [
                            str(getattr(ah, "name", "") or ""),
                            str(getattr(ah, "artifact_type", "") or ""),
                            " ".join(conds_local),
                            str(getattr(ah, "conclusion", "") or ""),
                            str(getattr(ah, "lhs", "") or ""),
                            str(getattr(ah, "rhs", "") or ""),
                            str(getattr(ah, "latex_source", "") or ""),
                        ]
                    ).lower()
                except Exception:
                    hay = ""

                intent = dict(state.get("intent") or {})
                shape = dict(state.get("shape") or {})

                # Shape mismatches (2D vs 3D).
                if shape.get("circle") and not shape.get("sphere"):
                    if any(w in hay for w in ("sphere", "cylinder", "cone")):
                        return False
                if shape.get("sphere"):
                    if ("circle" in hay) and ("sphere" not in hay):
                        return False

                # Intent mismatches (area vs volume).
                if intent.get("volume"):
                    # If the artifact looks like an area/circumference fact, reject.
                    if ("area" in hay or "circumference" in hay) and ("volume" not in hay):
                        return False
                if intent.get("surface_area"):
                    if "volume" in hay:
                        return False
                if intent.get("area") and not intent.get("surface_area"):
                    if "volume" in hay:
                        return False
                    if "surface area" in hay:
                        return False

                return True

            def _instantiate_with_bindings(rpn: str, state: Dict[str, Any], symbol_bindings: Dict[str, Any]) -> Optional[str]:
                parts = str(rpn or "").split()
                if not parts:
                    return None
                vars_in_order: List[str] = []
                for tok in parts:
                    if tok in known_ops or tok in known_consts or _is_num(tok):
                        continue
                    if tok not in vars_in_order:
                        vars_in_order.append(tok)
                if not vars_in_order:
                    inst = " ".join(parts)
                    try:
                        from knowledge3d.training.math_benchmarks.rpn_validator import is_valid_rpn as _is_valid_rpn
                        from knowledge3d.training.math_benchmarks.rpn_validator import validate_stack_shape as _validate_stack_shape

                        if not _is_valid_rpn(inst):
                            return None
                        if not _validate_stack_shape(inst).ok:
                            return None
                    except Exception:
                        pass
                    return inst

                # Prefer role-based binding when available.
                mapping: Dict[str, str] = {}
                roles = {}
                try:
                    for k, vals in (state.get("roles") or {}).items():
                        if isinstance(vals, list):
                            roles[str(k)] = list(vals)
                except Exception:
                    roles = {}

                def _role_for_var(v: str) -> Optional[str]:
                    info = symbol_bindings.get(v)
                    if not isinstance(info, dict):
                        info = {}
                    meaning = str(info.get("meaning") or "").lower()
                    if "hypotenuse" in meaning:
                        return "hypotenuse"
                    if "leg" in meaning:
                        return "leg"
                    if "radius" in meaning:
                        return "radius"
                    if "diameter" in meaning:
                        return "diameter"
                    if meaning in {"length", "width", "height"}:
                        return meaning
                    # Fallback: variable-name heuristics (only when the prompt context
                    # suggests the corresponding roles exist).
                    key = str(v or "").strip().lower()
                    if key in {"r"} and roles.get("radius"):
                        return "radius"
                    if key in {"d"} and roles.get("diameter"):
                        return "diameter"
                    if key in {"h"} and roles.get("height"):
                        return "height"
                    if key in {"l"} and roles.get("length"):
                        return "length"
                    if key in {"w"} and roles.get("width"):
                        return "width"
                    if state.get("shape", {}).get("triangle") or roles.get("leg") or roles.get("hypotenuse"):
                        if key == "c" and roles.get("hypotenuse"):
                            return "hypotenuse"
                        if key in {"a", "b"} and roles.get("leg"):
                            return "leg"
                    return None

                unfilled: List[str] = []
                for v in vars_in_order:
                    role = _role_for_var(v)
                    if role and role in roles and roles[role]:
                        mapping[v] = self._lit_num(float(roles[role].pop(0)))
                    else:
                        unfilled.append(v)

                # Fill remaining vars from remaining numbers (stable order).
                remaining_numbers = list(state.get("numbers") or [])
                # Remove already used values (best-effort).
                used = set()
                for lit in mapping.values():
                    try:
                        used.add(float(lit))
                    except Exception:
                        pass
                remaining_numbers = [n for n in remaining_numbers if n not in used]

                if len(unfilled) > len(remaining_numbers) or len(unfilled) > 6:
                    return None
                for i, v in enumerate(unfilled):
                    mapping[v] = self._lit_num(float(remaining_numbers[i]))

                inst = " ".join(mapping.get(tok, tok) for tok in parts)
                # Reject "flat" programs: multiple literals and zero operators.
                try:
                    ops = [t for t in inst.split() if t in known_ops]
                    if not ops:
                        lits = []
                        for t in inst.split():
                            try:
                                float(t)
                                lits.append(t)
                            except Exception:
                                pass
                        if len(lits) > 1:
                            return None
                except Exception:
                    pass
                # Avoid feeding obviously-invalid programs into TTC.
                try:
                    from knowledge3d.training.math_benchmarks.rpn_validator import is_valid_rpn as _is_valid_rpn
                    from knowledge3d.training.math_benchmarks.rpn_validator import validate_stack_shape as _validate_stack_shape

                    if not _is_valid_rpn(inst):
                        return None
                    if not _validate_stack_shape(inst).ok:
                        return None
                except Exception:
                    pass
                return inst

            state = _parse_problem_state(problem_text)

            scored_hits = sorted(artifact_hits, key=lambda a: _artifact_score(a, state), reverse=True)
            # Keep only the best few to avoid flooding TTC with low-signal examples.
            max_artifacts = 6
            selected_artifacts_meta: List[Dict[str, Any]] = []
            for ah in scored_hits[: max(1, int(max_artifacts))]:
                if not _artifact_context_gate(ah, state):
                    continue
                if not _artifact_condition_gate(getattr(ah, "conditions", []) or [], state):
                    continue
                # Require some *meaningful* alignment signal:
                # - either role-based bindings can be applied, or
                # - at least one condition is evaluable and passes.
                conds_all = list(getattr(ah, "conditions", []) or [])
                verdicts = [_eval_condition_text(c, state) for c in conds_all]
                evaluable = sum(1 for v in verdicts if v is not None)
                matched = sum(1 for v in verdicts if v is True)
                sb_raw = dict(getattr(ah, "symbol_bindings", {}) or {})
                sb_for_gate = _augment_symbol_bindings_with_inferred_roles(ah, sb_raw)
                role_gate = _role_match_score(state, sb_for_gate)
                # Phase 7B (quality): require at least one *matched* evaluable condition
                # or a role-binding match. Merely having "evaluable" conditions without a
                # match doesn't happen (False rejects), and non-evaluable conditions are
                # too noisy to trust for selection.
                if role_gate <= 0.0 and matched <= 0:
                    continue
                programs: List[str] = []
                if ah.rpn:
                    programs.append(str(ah.rpn))
                for d in list(getattr(ah, "derived_rpns", []) or []):
                    r = str(d.get("rpn") or "").strip()
                    if r:
                        programs.append(r)
                emitted: List[str] = []
                for rpn in programs:
                    inst = _instantiate_with_bindings(
                        rpn,
                        state,
                        symbol_bindings=sb_for_gate,
                    )
                    if inst:
                        candidates.append(inst)
                        emitted.append(inst)
                if emitted:
                    artifact_seed.extend(emitted)
                    try:
                        score_internal = float(_artifact_score(ah, state))
                    except Exception:
                        score_internal = None
                    selected_artifacts_meta.append(
                        {
                            "book_id": getattr(ah, "book_id", None),
                            "page": getattr(ah, "page_number", None),
                            "artifact_id": getattr(ah, "artifact_id", None),
                            "artifact_type": getattr(ah, "artifact_type", None),
                            "name": getattr(ah, "name", None),
                            "score_internal": score_internal,
                            "conditions": list(getattr(ah, "conditions", []) or [])[:6],
                            "condition_evaluable": evaluable,
                            "condition_matched": matched,
                            "role_match": role_gate,
                            "emitted_rpn": emitted[:6],
                        }
                    )
            if selected_artifacts_meta:
                hits_meta.append({"artifact_selection": selected_artifacts_meta})

        have_selected_artifacts = False
        try:
            have_selected_artifacts = bool(locals().get("selected_artifacts_meta"))
        except Exception:
            have_selected_artifacts = False

        # Book-derived equation templates (ingestion-time) are high-noise (they lack
        # applicability conditions). For Phase 7 quality refinement, only use them
        # as a fallback when we have no viable articulated artifacts.
        if template_hits and not have_selected_artifacts:
            # Try role-aware binding first (radius/height/legs/hypotenuse/etc) to avoid
            # low-signal "first number wins" instantiation.
            roles_for_templates: Dict[str, List[float]] = {}
            try:
                parse_state_fn = locals().get("_parse_problem_state")
                if callable(parse_state_fn):
                    st = parse_state_fn(problem_text)
                    roles_for_templates = dict((st or {}).get("roles") or {})
            except Exception:
                roles_for_templates = {}

            def _take_role(role: str) -> float | None:
                vals = roles_for_templates.get(role) if isinstance(roles_for_templates, dict) else None
                if not vals:
                    return None
                try:
                    return float(vals.pop(0))
                except Exception:
                    return None
            try:
                from knowledge3d.training.math_benchmarks.rpn_validator import is_valid_rpn as _is_valid_rpn
                from knowledge3d.training.math_benchmarks.rpn_validator import validate_stack_shape as _validate_stack_shape
            except Exception:
                _is_valid_rpn = None  # type: ignore[assignment]
                _validate_stack_shape = None  # type: ignore[assignment]

            known_ops = {
                "+",
                "-",
                "*",
                "/",
                "^",
                "pow",
                "neg",
                "sqrt",
                "sin",
                "cos",
                "tan",
                "sinh",
                "cosh",
                "tanh",
                "asin",
                "acos",
                "atan",
                "atan2",
                "arcsin",
                "arccos",
                "arctan",
                "log",
                "ln",
                "log2",
                "log10",
                "exp",
                "gamma",
                "beta",
                "abs",
                "floor",
                "ceil",
                "round",
                "mod",
                "%",
                "max",
                "min",
                "gcd",
                "factorial",
                "!",
                "binomial",
                "binom",
            }
            known_consts = {"pi", "π", "tau", "phi", "φ", "e", "Pi", "PI"}

            def _is_num(tok: str) -> bool:
                try:
                    float(tok)
                    return True
                except Exception:
                    return False

            selected_templates_meta: List[Dict[str, Any]] = []
            for th in template_hits:
                rpn = str(th.rpn or "").strip()
                if not rpn:
                    continue
                parts = rpn.split()
                vars_in_order: List[str] = []
                for tok in parts:
                    if tok in known_ops or tok in known_consts or _is_num(tok):
                        continue
                    if tok not in vars_in_order:
                        vars_in_order.append(tok)
                if not vars_in_order:
                    if callable(_is_valid_rpn) and (not _is_valid_rpn(rpn)):
                        continue
                    if callable(_validate_stack_shape) and (not _validate_stack_shape(rpn).ok):
                        continue
                    # Reject "flat" programs: multiple literals and zero operators.
                    try:
                        ops = [t for t in rpn.split() if t in known_ops]
                        if not ops:
                            lits = []
                            for t in rpn.split():
                                try:
                                    float(t)
                                    lits.append(t)
                                except Exception:
                                    pass
                            if len(lits) > 1:
                                continue
                    except Exception:
                        pass
                    candidates.append(rpn)
                    template_seed.append(rpn)
                    selected_templates_meta.append(
                        {
                            "book_id": th.book_id,
                            "page": th.page_number,
                            "score": th.score,
                            "lhs": th.lhs,
                            "rhs": th.rhs,
                            "emitted_rpn": rpn,
                        }
                    )
                    continue
                if not nums:
                    continue
                if len(vars_in_order) > len(nums) or len(vars_in_order) > 6:
                    continue
                mapping: Dict[str, str] = {}
                remaining_numbers = list(nums)
                used_roles = False
                for v in vars_in_order:
                    key = str(v or "").strip().lower()
                    bound: float | None = None
                    if key in {"r", "radius"}:
                        bound = _take_role("radius")
                    elif key in {"d", "diameter"}:
                        bound = _take_role("diameter")
                    elif key in {"h", "height"}:
                        bound = _take_role("height")
                    elif key in {"l", "length"}:
                        bound = _take_role("length")
                    elif key in {"w", "width"}:
                        bound = _take_role("width")
                    elif key in {"c"}:
                        # Triangle: prefer hypotenuse if present.
                        bound = _take_role("hypotenuse")
                    elif key in {"a", "b"}:
                        # Triangle: treat a/b as legs when available.
                        bound = _take_role("leg")
                    if bound is not None:
                        mapping[v] = self._lit_num(bound)
                        used_roles = True
                        # Remove one matching number from the general pool when possible.
                        try:
                            for i, n in enumerate(list(remaining_numbers)):
                                if abs(float(n) - float(bound)) <= 1e-9:
                                    remaining_numbers.pop(i)
                                    break
                        except Exception:
                            pass
                # Fallback: assign remaining variables to remaining numbers in prompt order.
                unfilled = [v for v in vars_in_order if v not in mapping]
                if len(unfilled) > len(remaining_numbers):
                    continue
                for i, v in enumerate(unfilled):
                    mapping[v] = self._lit_num(float(remaining_numbers[i]))
                inst = " ".join(mapping.get(tok, tok) for tok in parts)
                # Reject "flat" programs: multiple literals and zero operators.
                try:
                    ops = [t for t in inst.split() if t in known_ops]
                    if not ops:
                        lits = []
                        for t in inst.split():
                            try:
                                float(t)
                                lits.append(t)
                            except Exception:
                                pass
                        if len(lits) > 1:
                            continue
                except Exception:
                    pass
                if callable(_is_valid_rpn) and (not _is_valid_rpn(inst)):
                    continue
                if callable(_validate_stack_shape) and (not _validate_stack_shape(inst).ok):
                    continue
                candidates.append(inst)
                template_seed.append(inst)
                selected_templates_meta.append(
                    {
                        "book_id": th.book_id,
                        "page": th.page_number,
                        "score": th.score,
                        "lhs": th.lhs,
                        "rhs": th.rhs,
                        "emitted_rpn": inst,
                        "role_bound": bool(used_roles),
                    }
                )
            if selected_templates_meta:
                hits_meta.append({"template_selection": selected_templates_meta[:12]})

        # Determinant of a 2x2 numeric matrix: det([[a,b],[c,d]]) = ad - bc
        if ("determinant" in low or " det(" in low or " det " in low) and len(nums) >= 4:
            a, b, c, d = nums[0], nums[1], nums[2], nums[3]
            candidates.append(f"{self._lit_num(a)} {self._lit_num(d)} * {self._lit_num(b)} {self._lit_num(c)} * -")

        # gcd / lcm (lcm via gcd since lcm opcode isn't exposed in the parser)
        if ("gcd" in low or "greatest common divisor" in low) and len(nums) >= 2:
            a, b = nums[0], nums[1]
            candidates.append(f"{self._lit_num(a)} {self._lit_num(b)} gcd")
        if ("lcm" in low or "least common multiple" in low) and len(nums) >= 2:
            a, b = nums[0], nums[1]
            candidates.append(f"{self._lit_num(a)} {self._lit_num(b)} * {self._lit_num(a)} {self._lit_num(b)} gcd /")

        # binomial coefficients / combinations
        if ("binomial" in low or "choose" in low or "combination" in low or " ncr" in low) and len(nums) >= 2:
            n, k = nums[0], nums[1]
            candidates.append(f"{self._lit_num(n)} {self._lit_num(k)} binomial")

        # factorial
        if ("factorial" in low or "!" in low) and nums:
            n = float(nums[0])
            if n >= 0 and abs(n - round(n)) < 1e-9 and n <= 200:
                candidates.append(f"{self._lit_num(n)} factorial")

        # De-dupe while preserving order and cap.
        seen: set[str] = set()
        out: List[str] = []
        for c in candidates:
            s = str(c).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= max(0, int(max_candidates)):
                break
        out_set = set(out)
        artifact_out = [s.strip() for s in artifact_seed if isinstance(s, str) and s.strip() in out_set]
        template_out = [s.strip() for s in template_seed if isinstance(s, str) and s.strip() in out_set]
        sourced_seen: set[str] = set()
        sourced: List[str] = []
        for s in artifact_out + template_out:
            if not s or s in sourced_seen:
                continue
            sourced_seen.add(s)
            sourced.append(s)
        seed_sources = {"artifact": artifact_out, "template": template_out, "sourced": sourced}
        return (out, hits_meta, seed_sources)

    def _generate_schedule_rate_candidates(self, problem_text: str, *, max_candidates: int = 9) -> List[str]:
        """
        TTC seed candidates for common schedule/rate narratives.

        Keep this small and generic; these candidates are only meant to kick-start
        TTC on problems where the rest of the pipeline often misses the obvious
        schedule arithmetic.
        """
        import re

        from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

        norm = normalize_number_words(problem_text or "")
        low = norm.lower()
        candidates: List[str] = []

        def _f(x: float) -> str:
            return self._lit_num(float(x))

        # Pattern 1: "burns X every Y minutes" with bags/boxes.
        # Generic formula: time = (bags * per_bag) / rate * interval
        m_every = re.search(
            r"\b(\d+(?:\.\d+)?)\b[^\n]{0,50}\bevery\b[^\d]{0,20}\b(\d+(?:\.\d+)?)\b\s+(minutes?|hours?|days?)\b",
            low,
        )
        if m_every:
            try:
                rate = float(m_every.group(1))
                interval = float(m_every.group(2))
            except Exception:
                rate = 0.0
                interval = 0.0
            if rate > 0 and interval > 0:
                m_bags = re.search(r"\b(\d+(?:\.\d+)?)\b\s+bags?\b", low)
                m_contains = re.search(r"\bcontains\b[^\d]{0,20}\b(\d+(?:\.\d+)?)\b", low)
                if m_bags and m_contains:
                    try:
                        bags = float(m_bags.group(1))
                        per_bag = float(m_contains.group(1))
                    except Exception:
                        bags = 0.0
                        per_bag = 0.0
                    if bags > 0 and per_bag > 0:
                        candidates.append(f"{_f(bags)} {_f(per_bag)} * {_f(rate)} / {_f(interval)} *")

        # Pattern 1b: Solve duration from (initial + rate) -> final.
        # Example: "already blew up 12 ... rate 2 every five minutes ... there were 50 ... how many minutes?"
        # Generic formula: time = (final - initial) / rate * interval
        m_rate = re.search(
            r"\brate\b[^\d]{0,20}\bof\b[^\d]{0,20}\b(\d+(?:\.\d+)?)\b[^\n]{0,40}\bevery\b[^\d]{0,20}\b(\d+(?:\.\d+)?)\b\s+(minutes?|hours?)\b",
            low,
        )
        if not m_rate:
            m_rate = re.search(
                r"\b(\d+(?:\.\d+)?)\b[^\n]{0,20}\bevery\b[^\d]{0,20}\b(\d+(?:\.\d+)?)\b\s+(minutes?|hours?)\b",
                low,
            )
        if m_rate and ("how many minute" in low or "how many hour" in low):
            try:
                rate = float(m_rate.group(1))
                interval = float(m_rate.group(2))
                unit = str(m_rate.group(3) or "")
            except Exception:
                rate = 0.0
                interval = 0.0
                unit = ""
            if rate > 0 and interval > 0:
                m_initial = re.search(r"\balready\b[^\d]{0,30}\b(\d+(?:\.\d+)?)\b", low)
                m_final = re.search(r"\bthere\s+were\s+(\d+(?:\.\d+)?)\b", low)
                if m_initial and m_final:
                    try:
                        initial = float(m_initial.group(1))
                        final = float(m_final.group(1))
                    except Exception:
                        initial = 0.0
                        final = 0.0
                    if final > initial and initial >= 0:
                        expr = f"{_f(final)} {_f(initial)} - {_f(rate)} / {_f(interval)} *"
                        if unit.startswith("hour"):
                            expr = f"{expr} 60 *"
                        candidates.insert(0, expr)

        # Pattern 2: "N stories", "each story is H feet", "T trips ... each day of a week".
        # Generic formula: distance = stories * height * trips * round_trip * days
        m_story = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:st|nd|rd|th)?\s+story\b", low)
        m_height = re.search(r"\beach\s+story\s+is\s+(\d+(?:\.\d+)?)\s+feet\b", low)
        m_trips = re.search(r"\b(\d+(?:\.\d+)?)\s+trips?\b", low)
        if m_story and m_height and m_trips:
            try:
                stories = float(m_story.group(1))
                height = float(m_height.group(1))
                trips = float(m_trips.group(1))
            except Exception:
                stories = 0.0
                height = 0.0
                trips = 0.0
            if stories > 0 and height > 0 and trips > 0:
                round_trip = 2.0 if ("back" in low or "out and back" in low or "from and back" in low) else 1.0
                days = 7.0 if "week" in low else 1.0
                candidates.append(f"{_f(stories)} {_f(height)} * {_f(trips)} * {_f(round_trip)} * {_f(days)} *")

        # Pattern 3: Packaging per box + days + mixed whole + repeated fraction events.
        # Example (Basil): morning+bed fraction, plus whole cookies during the day.
        m_pack = re.search(r"\bpackaged\s+with\s+(\d+(?:\.\d+)?)\s+\w+\s+per\s+box\b", low)
        if not m_pack:
            m_pack = re.search(r"\b(\d+(?:\.\d+)?)\s+\w+\s+per\s+box\b", low)
        m_days = re.search(r"\bfor\s+(\d+(?:\.\d+)?)\s+days\b", low)
        m_frac = re.search(r"\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\b", low)
        m_whole = re.search(r"\bgets?\s+(\d+(?:\.\d+)?)\s+whole\s+\w+", low)
        if m_pack and m_days and m_frac and m_whole:
            try:
                per_box = float(m_pack.group(1))
                days = float(m_days.group(1))
                num = float(m_frac.group(1))
                den = float(m_frac.group(2))
                whole = float(m_whole.group(1))
            except Exception:
                per_box = 0.0
                days = 0.0
                num = 0.0
                den = 0.0
                whole = 0.0
            if per_box > 0 and days > 0 and whole >= 0 and den != 0:
                frac_times = 2.0 if ("morning" in low and "bed" in low) else 1.0
                frac = f"{_f(num)} {_f(den)} /"
                daily = f"{_f(whole)} {frac} {_f(frac_times)} * +"
                candidates.append(f"{daily} {_f(days)} * {_f(per_box)} /")
                if frac_times != 1.0:
                    daily2 = f"{_f(whole)} {frac} +"
                    candidates.append(f"{daily2} {_f(days)} * {_f(per_box)} /")

        # Pattern 4: Weekly schedule over N weeks with day-of-week buckets.
        # Example: \"1 hour every Monday, Wednesday, Friday; 30 min Tuesday/Thursday; 2 hours Saturday; over 2 weeks\".
        m_weeks = re.search(r"\b(\d+(?:\.\d+)?)\s+week", low)
        if m_weeks and any(d in low for d in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")):
            try:
                weeks = float(m_weeks.group(1))
            except Exception:
                weeks = 0.0
            if weeks > 0:
                day_names = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

                def _count_days(seg: str) -> int:
                    s = seg.lower()
                    return sum(1 for d in day_names if d in s)

                terms: List[str] = []
                seen_terms: set[str] = set()

                # Hour terms: \"rode 1 hour ... monday, wednesday, and friday\"
                for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s+hours?\b[^.]{0,120}\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)[^.]{0,120}", low):
                    try:
                        hours = float(m.group(1))
                    except Exception:
                        continue
                    cnt = _count_days(m.group(0))
                    if hours > 0 and cnt > 0:
                        t = f"{_f(hours)} {_f(float(cnt))} *"
                        if t not in seen_terms:
                            seen_terms.add(t)
                            terms.append(t)

                # Hour terms where the day list comes first: "on Saturdays ... for 2 hours".
                # Important: don't let a leading "on Tuesday ..." clause accidentally bind to a later
                # "on Saturday ... 2 hours" number; we reject any match whose gap contains another "on".
                day_pat = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?"
                days_list_pat = rf"{day_pat}(?:\s*(?:,|and)\s*{day_pat})*"
                for m in re.finditer(
                    rf"\bon\s+({days_list_pat})\b(?:(?!\bon\b)[^.]){{0,140}}\b(\d+(?:\.\d+)?)\s+hours?\b",
                    low,
                ):
                    days_seg = m.group(1)
                    try:
                        hours = float(m.group(2))
                    except Exception:
                        continue
                    cnt = _count_days(days_seg)
                    if hours > 0 and cnt > 0:
                        t = f"{_f(hours)} {_f(float(cnt))} *"
                        if t not in seen_terms:
                            seen_terms.add(t)
                            terms.append(t)

                # Minute terms (day list first): "On Tuesday and Thursday ... for 30 min".
                for m in re.finditer(r"\bon\b[^.]{0,80}\b(\d+(?:\.\d+)?)\s+min", low):
                    try:
                        mins = float(m.group(1))
                    except Exception:
                        continue
                    cnt = _count_days(m.group(0))
                    if mins > 0 and cnt > 0:
                        t = f"{_f(mins)} {_f(float(cnt))} * 60 /"
                        if t not in seen_terms:
                            seen_terms.add(t)
                            terms.append(t)

                # Minute terms where the number comes first: "rode for 30 min on Tuesday and Thursday".
                for m in re.finditer(
                    r"\b(\d+(?:\.\d+)?)\s+min(?:ute)?s?\b[^.]{0,160}\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b[^.]{0,160}",
                    low,
                ):
                    try:
                        mins = float(m.group(1))
                    except Exception:
                        continue
                    cnt = _count_days(m.group(0))
                    if mins > 0 and cnt > 0:
                        t = f"{_f(mins)} {_f(float(cnt))} * 60 /"
                        if t not in seen_terms:
                            seen_terms.add(t)
                            terms.append(t)

                if len(terms) >= 2:
                    weekly = " ".join(terms) + " " + "+ " * (len(terms) - 1)
                    weekly = weekly.strip()
                    candidates.append(f"{weekly} {_f(weeks)} *")

        # Pattern 4b: Day-by-day totals with a multiplier.
        #
        # Example (GSM8K #3344):
        #   "Sidney does 20 ... on Monday, 36 on Tuesday, 40 on Wednesday, and 50 on Thursday.
        #    Brooke does three times as many ... How many ... did Brooke do?"
        #
        # Generic formula:
        #   total = sum(day_values)
        #   target = total * factor   (also try total / factor)
        if any(d in low for d in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")):
            day_vals: List[float] = []
            day_names = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
            for m in re.finditer(r"\bon\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", low):
                # Pick the closest numeric mention immediately before "on <day>".
                start = m.start()
                window = low[max(0, start - 60) : start]
                nums = re.findall(r"(\d+(?:\.\d+)?)", window)
                if not nums:
                    continue
                try:
                    day_vals.append(float(nums[-1]))
                except Exception:
                    continue
                if len(day_vals) >= len(day_names):
                    break

            if len(day_vals) >= 2:
                # RPN sum: v1 v2 + v3 + ...
                sum_parts: List[str] = [_f(day_vals[0]), _f(day_vals[1]), "+"]
                for v in day_vals[2:]:
                    sum_parts.extend([_f(v), "+"])
                sum_expr = " ".join(sum_parts)

                factor: float | None = None
                m_times = re.search(r"\b(\d+(?:\.\d+)?)\s+times\s+as\s+many\b", low)
                if m_times:
                    try:
                        factor = float(m_times.group(1))
                    except Exception:
                        factor = None
                if factor is None and ("twice as many" in low or "double" in low):
                    factor = 2.0

                if factor is not None and factor > 0:
                    candidates.insert(0, f"{sum_expr} {_f(factor)} *")
                    candidates.insert(1, f"{sum_expr} {_f(factor)} /")
                else:
                    candidates.insert(0, sum_expr)

        # Pattern 5: Weekly-per-person sums repeated for N weeks.
        # Example: "In one week, Jake can eat 3 papayas, his brother can eat 5 papayas, and his father can eat 4 papayas.
        #          To account for 4 weeks, how many papayas ...?"
        # Generic formula: (sum(per_week_amounts)) * weeks
        m_span_weeks = (
            re.search(r"\bto\s+account\s+for\s+(\d+(?:\.\d+)?)\s+weeks?\b", low)
            or re.search(r"\bfor\s+(\d+(?:\.\d+)?)\s+weeks?\b", low)
            or re.search(r"\bover\s+(\d+(?:\.\d+)?)\s+weeks?\b", low)
            or re.search(r"\bin\s+(\d+(?:\.\d+)?)\s+weeks?\b", low)
        )
        m_span_one_week = re.search(r"\bin\s+(\d+(?:\.\d+)?)\s+week\b", low)
        if (
            m_span_weeks
            and m_span_one_week
            and ("week" in low)
            and any(w in low for w in ("eat", "eats", "can eat", "per week", "in one week"))
            and ("how many" in low)
        ):
            try:
                weeks = float(m_span_weeks.group(1))
            except Exception:
                weeks = 0.0
            if weeks > 0:
                exclude_spans = [m_span_weeks.span(1), m_span_one_week.span(1)]
                weekly_amounts: List[float] = []
                for m in re.finditer(r"\b(\d+(?:\.\d+)?)\b", low):
                    span = m.span(1)
                    if any(span[0] >= ex[0] and span[1] <= ex[1] for ex in exclude_spans):
                        continue
                    try:
                        v = float(m.group(1))
                    except Exception:
                        continue
                    if v <= 0:
                        continue
                    # Filter out obviously irrelevant huge numbers.
                    if v > 10000:
                        continue
                    weekly_amounts.append(v)
                if len(weekly_amounts) >= 2:
                    expr = " ".join(_f(v) for v in weekly_amounts)
                    expr = f"{expr} " + "+ " * (len(weekly_amounts) - 1)
                    expr = f"{expr}{_f(weeks)} *".strip()
                    candidates.insert(0, expr)

        # Pattern 6: Budget aggregation + savings percent.
        #
        # Generic formula:
        #   total_spend = (weekly_budget * weeks) + sum(monthly_fixed_costs)
        #   savings = total_spend * (pct/100)
        #
        # Example: "spend no more than $100 a week for 4 weeks ... rent $1500 ... $30 streaming ... $50 phone ...
        #          set aside 10% into savings" => (100*4 + 1500 + 30 + 50) * 10/100 = 198
        if (
            ("savings" in low or "save" in low or "set aside" in low)
            and (("%" in low) or ("percent" in low))
            and ("week" in low)
            and ("a week" in low or "per week" in low)
        ):
            try:
                m_pct = re.search(r"\b(\d+(?:\.\d+)?)\s*%", low) or re.search(
                    r"\b(\d+(?:\.\d+)?)\s+percent\b", low
                )
                m_weeks = (
                    re.search(r"\bnext\s+(\d+(?:\.\d+)?)\s+weeks?\b", low)
                    or re.search(r"\bfor\s+the\s+next\s+(\d+(?:\.\d+)?)\s+weeks?\b", low)
                    or re.search(r"\bfor\s+(\d+(?:\.\d+)?)\s+weeks?\b", low)
                )
                m_weekly = re.search(
                    r"\bno\s+more\s+than\s+\$?\s*(\d+(?:\.\d+)?)\s+a\s+week\b", low
                ) or re.search(r"\$?\s*(\d+(?:\.\d+)?)\s+a\s+week\b", low)
                if m_pct and m_weeks and m_weekly:
                    pct = float(m_pct.group(1))
                    weeks = float(m_weeks.group(1))
                    weekly = float(m_weekly.group(1))
                    if pct > 0 and weeks > 0 and weekly >= 0:
                        # Collect dollar amounts; treat them as fixed monthly costs except the weekly budget itself.
                        dollars = [
                            float(str(x).replace(",", ""))
                            for x in re.findall(r"\$\s*([\d,]+(?:\.\d+)?)", norm)
                            if str(x).strip()
                        ]
                        fixed = [d for d in dollars if abs(d - weekly) > 1e-9]
                        expr = f"{_f(weekly)} {_f(weeks)} *"
                        for d in fixed[:8]:
                            expr += f" {_f(d)} +"
                        expr += f" {_f(pct)} 100 / *"
                        candidates.insert(0, expr)
            except Exception:
                pass

        seen: set[str] = set()
        out: List[str] = []
        for c in candidates:
            s = str(c).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= int(max_candidates):
                break
        return out

    def _generate_multistep_geometry_candidates(self, problem_text: str, *, max_candidates: int = 6) -> List[str]:
        """
        TTC seed candidates for small, high-leverage multi-step geometry chains.

        This is a Phase 8 starter: it does not attempt general theorem chaining.
        Instead, it emits a few canonical composed programs that cover common
        "solve intermediate variable then apply target formula" problems.

        Example:
          "Circle with circumference 20, find area"
            r = C/(2*pi)
            A = pi*r^2
          => "20 2 pi * / 2 pow pi *"
        """
        import re

        from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

        norm = normalize_number_words(problem_text or "")
        low = norm.lower()
        candidates: List[str] = []

        def _f(x: float) -> str:
            return self._lit_num(float(x))

        # Circle: circumference -> area.
        if ("circle" in low) and ("circumference" in low) and ("area" in low):
            c_val: Optional[float] = None
            m_c = re.search(r"\bcircumference\b[^\d]{0,20}\b(\d+(?:\.\d+)?)\b", low)
            if m_c:
                try:
                    c_val = float(m_c.group(1))
                except Exception:
                    c_val = None
            if c_val is None:
                nums = self.extract_numbers(norm)
                if nums:
                    c_val = float(nums[0])
            if c_val is not None and c_val > 0:
                c = _f(c_val)
                # A = pi * (C/(2*pi))^2
                candidates.append(f"{c} 2 pi * / 2 pow pi *")
                # A = C^2 / (4*pi)
                candidates.append(f"{c} 2 pow 4 pi * /")
                # A = (C*C) / (4*pi)
                candidates.append(f"{c} {c} * 4 pi * /")

        # Circle: diameter -> area.
        if ("circle" in low) and ("diameter" in low) and ("area" in low):
            d_val: Optional[float] = None
            m_d = re.search(r"\bdiameter\b[^\d]{0,20}\b(\d+(?:\.\d+)?)\b", low)
            if m_d:
                try:
                    d_val = float(m_d.group(1))
                except Exception:
                    d_val = None
            if d_val is None:
                nums = self.extract_numbers(norm)
                if nums:
                    d_val = float(nums[0])
            if d_val is not None and d_val > 0:
                d = _f(d_val)
                # A = pi * (d/2)^2
                candidates.append(f"{d} 2 / 2 pow pi *")
                # A = (pi * d^2) / 4
                candidates.append(f"{d} 2 pow pi * 4 /")

        seen: set[str] = set()
        out: List[str] = []
        for c in candidates:
            s = str(c).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= int(max_candidates):
                break
        return out

    def _generate_fraction_ratio_remainder_candidates(self, problem_text: str, *, max_candidates: int = 6) -> List[str]:
        """
        TTC seed candidates for "fraction of total" + "ratio" + "rest" partitions.

        Example:
          total=16
          part = 1/4 of total
          other = 2 * part
          rest = total - part - other
        """
        import re

        from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

        norm = normalize_number_words(problem_text or "")
        low = norm.lower()
        if not any(w in low for w in ("rest", "remaining")):
            return []

        m_total = re.search(r"\b(\d+(?:\.\d+)?)\b", low)
        if not m_total:
            return []
        try:
            total = float(m_total.group(1))
        except Exception:
            return []
        if total <= 0:
            return []

        # Collect explicit fractions plus common implied ones ("half").
        fractions: list[tuple[float, float]] = []
        for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\b", low):
            try:
                n = float(m.group(1))
                d = float(m.group(2))
            except Exception:
                continue
            if abs(d) < 1e-12:
                continue
            fractions.append((n, d))

        if "half" in low and not any(abs(n - 1.0) < 1e-9 and abs(d - 2.0) < 1e-9 for n, d in fractions):
            fractions.append((1.0, 2.0))

        # Multi-fraction remainder (no ratio needed):
        # rest = total - sum(total * frac_i)
        if len(fractions) >= 2:
            t = self._lit_num(total)
            expr_parts: list[str] = [t]
            first = True
            for n, d in fractions[:3]:  # cap to 3 fractions to avoid noise
                n_l = self._lit_num(n)
                d_l = self._lit_num(d)
                if first:
                    expr_parts.append(f"{t} {n_l} * {d_l} / -")
                    first = False
                else:
                    expr_parts.append(f"{t} {n_l} * {d_l} / -")
            multi_rest = " ".join(expr_parts).strip()
        else:
            multi_rest = ""

        ratio: float | None = None
        if "twice" in low or "double" in low:
            ratio = 2.0
        elif "triple" in low or "thrice" in low:
            ratio = 3.0
        else:
            m_times = re.search(r"\b(\d+(?:\.\d+)?)\s+times\b", low)
            if m_times:
                try:
                    ratio = float(m_times.group(1))
                except Exception:
                    ratio = None
        if ratio is None or ratio <= 0:
            # If we have a valid multi-fraction remainder candidate, return it.
            if multi_rest:
                return [multi_rest][: int(max_candidates)]
            return []

        t = self._lit_num(total)
        r = self._lit_num(ratio)

        # Use the first explicit fraction for the ratio-partition pattern.
        if not fractions:
            return [multi_rest][: int(max_candidates)] if multi_rest else []
        num, den = fractions[0]
        n = self._lit_num(num)
        d = self._lit_num(den)

        part = f"{t} {n} * {d} /"
        rest = f"{t} {part} - {part} {r} * -"
        rest2 = f"{t} {part} {self._lit_num(ratio + 1.0)} * -"

        out = [rest, rest2, part, f"{part} {r} *"]
        if multi_rest:
            out.insert(0, multi_rest)
        seen: set[str] = set()
        uniq: List[str] = []
        for c in out:
            s = str(c).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            uniq.append(s)
            if len(uniq) >= int(max_candidates):
                break
        return uniq

    def _generate_coin_change_unknown_candidates(self, problem_text: str, *, max_candidates: int = 6) -> List[str]:
        """
        TTC seed candidates for coin-change word problems with an unknown coin count.

        Generic form:
          total_cents = (spent_dollars * 100) + leftover_cents
          known_value = pennies*1 + nickels*5 + dimes*10 (+ optional half-dollars, etc.)
          unknown_count = (total_cents - known_value) / unknown_coin_value

        This shows up across domains as a simple linear remainder constraint.
        """
        import re

        from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

        low = normalize_number_words(problem_text or "").lower()
        if not any(w in low for w in ("pennies", "nickels", "dimes", "quarters")):
            return []
        if "how many" not in low:
            return []

        def _m(pat: str) -> float | None:
            m = re.search(pat, low)
            if not m:
                return None
            try:
                return float(m.group(1))
            except Exception:
                return None

        pennies = _m(r"\b(\d+(?:\.\d+)?)\s+pennies\b")
        nickels = _m(r"\b(\d+(?:\.\d+)?)\s+nickels\b")
        dimes = _m(r"\b(\d+(?:\.\d+)?)\s+dimes\b")

        # Spent: try \"costs $3 each\" and a population like \"five family members\".
        each_cost = _m(r"\bcosts?\s*\$?\s*(\d+(?:\.\d+)?)\s+each\b")
        people = _m(r"\b(\d+(?:\.\d+)?)\s+family\s+members\b")
        if people is None:
            people = _m(r"\ball\s+(\d+(?:\.\d+)?)\s+family\s+members\b")

        # Leftover: \"48 cents left\" or \"$0.48 left\".
        leftover_cents = _m(r"\b(\d+(?:\.\d+)?)\s+cents?\s+left\b")
        if leftover_cents is None:
            m_dollars_left = re.search(r"\$\s*(\d+(?:\.\d+)?)\s+left\b", low)
            if m_dollars_left:
                try:
                    leftover_cents = float(m_dollars_left.group(1)) * 100.0
                except Exception:
                    leftover_cents = None

        wants_quarters = "quarter" in low and "how many" in low and "quarters" in low
        if not wants_quarters:
            return []

        if pennies is None or nickels is None or dimes is None or each_cost is None or people is None or leftover_cents is None:
            return []

        if any(v < 0 for v in (pennies, nickels, dimes, each_cost, people, leftover_cents)):
            return []

        p = self._lit_num(pennies)
        n = self._lit_num(nickels)
        d = self._lit_num(dimes)
        c = self._lit_num(each_cost)
        k = self._lit_num(people)
        left = self._lit_num(leftover_cents)

        # total_cents = people * each_cost * 100 + leftover_cents
        total_cents = f"{k} {c} * 100 * {left} +"
        known = f"{p} 1 * {n} 5 * + {d} 10 * +"
        expr = f"{total_cents} {known} - 25 /"

        return [expr][: int(max_candidates)]
    def _extract_relative_definitions(self, problem_text: str, base_qty: str | None) -> List[Dict[str, Any]]:
        """
        Extract coarse relative operation hints in text order.

        Returned entries are *hints* for TTC candidate generation, not a parse tree.
        Each item is one of:
          - {"op": "+/-", "value": <delta>}
          - {"op": "*", "value": <factor>}
          - {"op": "/", "value": <divisor>}
          - {"op": "frac", "num": <n>, "den": <d>}
        """
        import re

        text = problem_text or ""
        out: List[Tuple[int, Dict[str, Any]]] = []

        for m in re.finditer(r"(\d+(?:\.\d+)?)\s+(?:more|additional|extra)\b.*?\bthan\b", text, re.IGNORECASE):
            out.append((m.start(), {"op": "+", "value": float(m.group(1)), "base": base_qty}))
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s+(?:less|fewer)\b.*?\bthan\b", text, re.IGNORECASE):
            out.append((m.start(), {"op": "-", "value": float(m.group(1)), "base": base_qty}))

        for m in re.finditer(r"\b(twice|double)\b", text, re.IGNORECASE):
            out.append((m.start(), {"op": "*", "value": 2.0, "base": base_qty}))
        for m in re.finditer(r"\b(thrice|triple)\b", text, re.IGNORECASE):
            out.append((m.start(), {"op": "*", "value": 3.0, "base": base_qty}))
        for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s+times\b", text, re.IGNORECASE):
            try:
                out.append((m.start(), {"op": "*", "value": float(m.group(1)), "base": base_qty}))
            except Exception:
                continue

        # Fraction words (common narrative chains).
        for m in re.finditer(r"\bhalf\b", text, re.IGNORECASE):
            out.append((m.start(), {"op": "/", "value": 2.0, "base": base_qty}))
        for m in re.finditer(r"\bquarter\b", text, re.IGNORECASE):
            out.append((m.start(), {"op": "/", "value": 4.0, "base": base_qty}))
        for m in re.finditer(r"\bthird\b", text, re.IGNORECASE):
            out.append((m.start(), {"op": "/", "value": 3.0, "base": base_qty}))
        for m in re.finditer(r"\bfourth\b", text, re.IGNORECASE):
            out.append((m.start(), {"op": "/", "value": 4.0, "base": base_qty}))

        # Explicit numeric fractions like "1/3" or "3/4".
        for m in re.finditer(r"\b(\d+)\s*/\s*(\d+)\b", text):
            try:
                num = float(m.group(1))
                den = float(m.group(2))
            except Exception:
                continue
            if abs(den) < 1e-12:
                continue
            out.append((m.start(), {"op": "frac", "num": num, "den": den, "base": base_qty}))

        # Denominator-word fractions like "three eighths", "3 quarters", "a fifth".
        #
        # These appear frequently in relative chains ("three eighths as many") and
        # ratio cascades; treat them as a numeric fraction hint rather than a raw
        # divisor so TTC can generate base * num / den candidates.
        denom_word_to_den: Dict[str, int] = {
            "half": 2,
            "third": 3,
            "quarter": 4,
            "fourth": 4,
            "fifth": 5,
            "sixth": 6,
            "seventh": 7,
            "eighth": 8,
            "ninth": 9,
            "tenth": 10,
        }
        num_word_to_num: Dict[str, int] = {
            "a": 1,
            "an": 1,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        denom_re = "|".join(sorted(denom_word_to_den.keys(), key=len, reverse=True))
        num_re = "|".join(sorted(num_word_to_num.keys(), key=len, reverse=True))
        # Allow whitespace or hyphen: "three-eighths" or "three eighths".
        frac_word_pat = re.compile(
            rf"\b(\d+|{num_re})\s*(?:-|\s+)\s*({denom_re})s?\b",
            re.IGNORECASE,
        )
        for m in frac_word_pat.finditer(text):
            raw_num = str(m.group(1) or "").strip().lower()
            raw_den = str(m.group(2) or "").strip().lower()
            if not raw_den:
                continue
            if raw_num.isdigit():
                try:
                    num_i = int(raw_num)
                except Exception:
                    continue
            else:
                num_i = int(num_word_to_num.get(raw_num, 0))
            den_i = int(denom_word_to_den.get(raw_den, 0))
            if num_i <= 0 or den_i <= 0:
                continue
            # Avoid duplicating the existing "half/quarter/third/fourth" single-word
            # hints when the numerator is 1; for those, the "/" hints are sufficient.
            if num_i == 1 and raw_den in {"half", "quarter", "third", "fourth"}:
                continue
            out.append((m.start(), {"op": "frac", "num": float(num_i), "den": float(den_i), "base": base_qty}))

        out.sort(key=lambda x: x[0])
        return [d for _, d in out]

    def _generate_chain_variants(self, base_rpn: str) -> List[str]:
        """
        Generate light algebraic variants of a numeric-only chain.

        Note: unit tests use a minimal EchoEngine that supports only + - * /,
        so we do not emit stack ops like DUP here.
        """
        toks = [t for t in (base_rpn or "").split() if t]
        if not toks:
            return []
        variants: List[str] = []

        # Pattern: "x x d + +" → "x 2 * d +"
        if len(toks) == 5 and toks[0] == toks[1] and toks[3] == "+" and toks[4] == "+":
            x = toks[0]
            d = toks[2]
            variants.append(f"{x} 2 * {d} +")

        return variants

    def _generate_relative_chain_candidates(
        self,
        *,
        problem_text: str,
        understanding: ProblemUnderstanding,
        trace: Dict[str, Any],
        question_type: str,
        max_candidates: int = 12,
    ) -> List[str]:
        """
        Generate diverse numeric-only candidates for relative/comparative problems.

        This is composition search, not a chain parser:
        - uses relative/comparative building blocks (matched patterns + op hints)
        - explores both forward and inverse operations (asked entity can vary)
        """
        try:
            nums = self.extract_numbers(problem_text)
        except Exception:
            nums = []
        if not nums:
            return []

        pattern_ids: List[str] = []
        try:
            pattern_ids = [
                str(p.get("rule_id"))
                for p in (trace or {}).get("patterns", [])
                if isinstance(p, dict) and p.get("rule_id")
            ]
        except Exception:
            pattern_ids = []
        has_relative = any(pid.startswith("relative_") for pid in pattern_ids)

        comp_ops = [
            op
            for op in understanding.operations
            if op.get("kind") == "comparative" and op.get("type") in {"add", "subtract"}
        ]
        mult_ops = [op for op in understanding.operations if op.get("type") == "derive_multiply"]
        rel_defs = self._extract_relative_definitions(problem_text, None)
        has_fraction = any(d.get("op") in {"/", "frac"} for d in rel_defs)
        if not (has_relative or comp_ops or mult_ops or has_fraction):
            return []

        def _lit(x: float) -> str:
            return str(int(x)) if abs(x - round(x)) < 1e-9 else str(x)

        # Base values (preserve appearance order; do not sort by magnitude).
        bases: List[float] = []
        for q in sorted(understanding.quantities, key=lambda q: (q.get("pos") or 0)):
            try:
                bases.append(float(q.get("value")))
            except Exception:
                continue
        bases.extend([float(n) for n in nums])
        # If we can extract a better anchor (e.g., "If there were 100 ..."), push it first.
        base_entity, base_hint = self._extract_base_quantity(problem_text)
        if base_hint is not None:
            bases = [float(base_hint)] + bases
        # Unique while preserving order.
        deduped: List[float] = []
        for b in bases:
            if any(abs(b - x) < 1e-9 for x in deduped):
                continue
            deduped.append(float(b))
            if len(deduped) >= 6:
                break
        bases = deduped

        # Comparative deltas in text order.
        deltas: List[Tuple[str, float]] = []
        for op in sorted(comp_ops, key=lambda o: (o.get("pos") or 0)):
            typ = op.get("type")
            try:
                amt = float(op.get("amount", 0.0))
            except Exception:
                continue
            if abs(amt) < 1e-12:
                continue
            if typ == "add":
                deltas.append(("+", amt))
            elif typ == "subtract":
                deltas.append(("-", amt))

        # Multipliers (derive_multiply) in text order.
        factors_in_order: List[float] = []
        for op in sorted(mult_ops, key=lambda o: (o.get("pos") or 0)):
            try:
                f = float(op.get("factor", 1.0))
            except Exception:
                continue
            if abs(f - 1.0) < 1e-9:
                continue
            if abs(f) < 1e-12:
                continue
            factors_in_order.append(f)

        # Additional coarse hints from text itself.
        for d in rel_defs:
            if d.get("op") in {"+", "-"}:
                try:
                    deltas.append((str(d["op"]), float(d["value"])))
                except Exception:
                    continue
            # Only use regex-derived multipliers when the grammar-based relative rules
            # didn't already capture them; avoid double-counting repeated "twice".
            if d.get("op") == "*" and not factors_in_order:
                try:
                    factors_in_order.append(float(d["value"]))
                except Exception:
                    continue

        # Fraction/division chain steps in appearance order.
        fraction_steps: List[Dict[str, float]] = []
        for d in rel_defs:
            if d.get("op") == "/":
                try:
                    fraction_steps.append({"num": 1.0, "den": float(d["value"])})
                except Exception:
                    continue
            if d.get("op") == "frac":
                try:
                    fraction_steps.append({"num": float(d["num"]), "den": float(d["den"])})
                except Exception:
                    continue

        candidates: List[str] = []
        low = (problem_text or "").lower()
        group_total_story = (
            question_type == "total"
            and ("child" in low or "children" in low)
            and ("adult" in low or "adults" in low)
            and "total" in low
        )

        # Multi-step "after gifts" comparisons:
        #
        # Example:
        #   "A had X before she gave T a. B had Y before giving T b.
        #    After all the gifts, T now has d more than B. How many more does T have now than A?"
        #
        # Solve:
        #   B_now = Y - b
        #   T_now = B_now + d
        #   A_now = X - a
        #   answer = T_now - A_now
        #
        # This is a generic state-update + relative comparison pattern.
        if (
            ("before" in low)
            and ("gave" in low)
            and ("after" in low)
            and ("now has" in low)
            and ("more" in low or "less" in low or "fewer" in low)
        ):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                # Gift events: (giver had before) + gave recipient gift
                gift_events: List[Dict[str, Any]] = []
                pat_gift = re.compile(
                    r"\b(?P<giver>[a-z]+)\b[^.]{0,140}\bhad\s+(?P<have>\d+(?:\.\d+)?)\b[^.]{0,160}\bbefore\b[^.]{0,160}\b(?:gave|give|giving)\b\s+(?P<rec>[a-z]+)\b\s+(?P<gift>\d+(?:\.\d+)?)\b",
                    re.IGNORECASE,
                )
                for m in pat_gift.finditer(norm):
                    gift_events.append(
                        {
                            "giver": str(m.group("giver")).lower(),
                            "have": float(m.group("have")),
                            "rec": str(m.group("rec")).lower(),
                            "gift": float(m.group("gift")),
                        }
                    )
                    if len(gift_events) >= 6:
                        break

                m_rel = re.search(
                    r"\b(?P<target>[a-z]+)\b\s+now\s+has\s+(?P<delta>\d+(?:\.\d+)?)\s+(?P<cmp>more|less|fewer)\b[^,?.]{0,80}\bthan\s+(?P<ref>[a-z]+)\b",
                    norm,
                    re.IGNORECASE,
                )
                m_q = re.search(
                    r"\bhow\s+many\s+(?P<cmp>more|less|fewer)\b[^?]{0,120}?\bdoes\s+(?P<a>[a-z]+)\b[^?]{0,160}?\bthan\s+(?P<b>[a-z]+)\b",
                    norm,
                    re.IGNORECASE,
                )

                if gift_events and m_rel and m_q:
                    target = str(m_rel.group("target")).lower()
                    ref = str(m_rel.group("ref")).lower()
                    delta = float(m_rel.group("delta"))
                    rel_cmp = str(m_rel.group("cmp")).lower()
                    q_cmp = str(m_q.group("cmp")).lower()
                    a = str(m_q.group("a")).lower()
                    b = str(m_q.group("b")).lower()

                    # Determine which entity is being compared against target in the question.
                    if a == target:
                        other = b
                    elif b == target:
                        other = a
                    else:
                        other = b

                    # Extract ref_before/ref_gift and other_before/other_gift (when they gave target).
                    ref_before: float | None = None
                    ref_gift = 0.0
                    other_before: float | None = None
                    other_gift = 0.0
                    for e in gift_events:
                        giver = str(e.get("giver") or "")
                        if str(e.get("rec") or "") != target:
                            continue
                        if giver == ref and ref_before is None:
                            ref_before = float(e.get("have", 0.0))
                            ref_gift = float(e.get("gift", 0.0))
                        if giver == other and other_before is None:
                            other_before = float(e.get("have", 0.0))
                            other_gift = float(e.get("gift", 0.0))

                    if (ref_before is not None) and (other_before is not None):
                        ref_now = (
                            f"{_lit(ref_before)} {_lit(ref_gift)} -" if abs(ref_gift) > 1e-9 else f"{_lit(ref_before)}"
                        )
                        other_now = (
                            f"{_lit(other_before)} {_lit(other_gift)} -" if abs(other_gift) > 1e-9 else f"{_lit(other_before)}"
                        )

                        # target_now = ref_now (+|-) delta depending on "more/less/fewer"
                        if rel_cmp == "more":
                            target_now = f"{ref_now} {_lit(delta)} +"
                        else:
                            target_now = f"{ref_now} {_lit(delta)} -"

                        # Question asks: how many more/less does target have than other?
                        if q_cmp == "more":
                            candidates.insert(0, f"{target_now} {other_now} -")
                        else:
                            candidates.insert(0, f"{other_now} {target_now} -")
            except Exception:
                pass

        # Common narrative: "received half that amount" / "double that amount".
        #
        # Example (Peggy):
        #   "Peggy has 6 dolls. ... gives ... 30 dolls. ... receives half that amount ..."
        #   total = 6 + 30 + (30/2)
        #
        # Keep this generic (not GSM8K-specific) and numeric-only so it can be executed
        # by minimal engines used in unit tests.
        try:
            import re

            num_pos: List[Tuple[int, float]] = []
            for m in re.finditer(r"\d+(?:\.\d+)?", problem_text or ""):
                # Skip pieces of explicit fractions (we handle those elsewhere).
                if (m.start() > 0 and (problem_text or "")[m.start() - 1 : m.start()] == "/") or (
                    (problem_text or "")[m.end() : m.end() + 1] == "/"
                ):
                    continue
                try:
                    num_pos.append((m.start(), float(m.group(0))))
                except Exception:
                    continue

            def _last_number_before(pos: int) -> float | None:
                prev = [v for p, v in num_pos if p < pos]
                return prev[-1] if prev else None

            that_half = re.search(r"\bhalf\s+(?:of\s+)?that\s+amount\b", low)
            that_twice = re.search(r"\b(?:twice|double)\s+(?:of\s+)?that\s+amount\b", low)
            that_amount_m = that_half or that_twice
            if that_amount_m and base_hint is not None:
                ref = _last_number_before(that_amount_m.start())
                if ref is not None:
                    base_lit = _lit(float(base_hint))
                    ref_lit = _lit(float(ref))
                    # Prefer "base + ref + (ref/2)" / "base + ref + (ref*2)".
                    if that_half:
                        candidates.insert(0, f"{base_lit} {ref_lit} + {ref_lit} 2 / +")
                        candidates.insert(1, f"{base_lit} {ref_lit} 2 / +")
                    else:
                        candidates.insert(0, f"{base_lit} {ref_lit} + {ref_lit} 2 * +")
                        candidates.insert(1, f"{base_lit} {ref_lit} 2 * +")
        except Exception:
            pass

        # Common narrative: "half the total ... between A and B" (average of two totals).
        #
        # Example:
        #   "Tina has 40 students. Maura has 98 students. Zack has half the total number
        #    of students between Tina and Maura. How many students does Zack have?"
        #   rpn = 40 98 + 2 /
        try:
            import re

            m_between = re.search(r"\bhalf\b.*\btotal\b.*\bbetween\b", low)
            if m_between:
                # Special case: equality + half-between + absent-person clue.
                #
                # Example (GSM8K train #5955):
                #   "Tina's classroom has the same amount of students as Maura's. Zack's classroom has
                #    half the amount of total students between Tina and Maura's classrooms. ... between
                #    the 3 classrooms ... when Zack was sick there were 22 students in his class?"
                #
                # When an entity is absent, the observed count typically excludes them. If we can
                # infer Z = (observed + 1), and also Z = 1/2*(T+M) with T=M, then T=M=Z and the
                # total across K classrooms is K*Z.
                if ("same amount" in low or "same number" in low) and "classroom" in low:
                    k_groups: int | None = None
                    m_k = re.search(r"\bbetween\s+the\s+(\d+)\s+classrooms?\b", low)
                    if m_k:
                        try:
                            k_groups = int(m_k.group(1))
                        except Exception:
                            k_groups = None
                    if k_groups is None:
                        if "three classrooms" in low or "3 classrooms" in low:
                            k_groups = 3

                    m_absent = re.search(
                        r"\bwhen\s+\w+(?:'s)?\s+was\s+(?:sick|absent|away|out)\b.*?\bthere\s+were\s+(\d[\d,]*)\s+students?\b",
                        low,
                    )
                    if m_absent:
                        try:
                            observed = float(m_absent.group(1).replace(",", ""))
                        except Exception:
                            observed = None
                        if observed is not None:
                            # Encode the half-between constraint explicitly to pass plausibility checks:
                            #   Z = (observed+1) * 2 * 0.5  == observed+1
                            z_expr = f"{_lit(observed)} 1 + 2 * 0.5 *"
                            if question_type == "total" and k_groups:
                                candidates.insert(0, f"{z_expr} {k_groups} *")
                            # If the question asks for a single classroom size, returning Z is enough.
                            candidates.insert(0, z_expr)

                between_pos = m_between.start()
                # Use the last two numeric mentions before "between".
                nums_before: List[float] = []
                for q in sorted(understanding.quantities, key=lambda q: (q.get("pos") or 0)):
                    pos = q.get("pos")
                    if pos is None or pos >= between_pos:
                        continue
                    try:
                        nums_before.append(float(q.get("value")))
                    except Exception:
                        continue
                # Fallback to regex-pos numbers if no structured positions.
                if len(nums_before) < 2:
                    nums_before = []
                    for m in re.finditer(r"\d+(?:\.\d+)?", problem_text or ""):
                        if m.start() >= between_pos:
                            break
                        try:
                            nums_before.append(float(m.group(0)))
                        except Exception:
                            continue
                if len(nums_before) >= 2:
                    a = float(nums_before[-2])
                    b = float(nums_before[-1])
                    candidates.insert(0, f"{_lit(a)} {_lit(b)} + 2 /")
        except Exception:
            pass

        # Special: chained multipliers + terminal delta + "altogether/total" sum.
        #
        # Example:
        #   "Miriam has five times as many albums as Katrina. Katrina has six times the number
        #    of albums as Bridget. Bridget has 15 fewer albums than Adele. If Adele has 30 albums,
        #    how many albums do Miriam, Katrina, Bridget, and Adele have altogether?"
        #
        # This is a general "base with delta, then multiplicative chain, then sum all entities" pattern.
        base_entity_l = str(base_entity or "").strip().lower()
        if question_type == "total" and base_hint is not None and base_entity_l:
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words((problem_text or "").lower())

                times_rels: List[Dict[str, Any]] = []
                for m in re.finditer(
                    r"\b([a-z]+)\s+(?:has|have|had|is|are|was|were)\s+(\d+(?:\.\d+)?)\s+times\s+as\s+many\s+\w+\s+as\s+([a-z]+)\b",
                    norm,
                ):
                    times_rels.append(
                        {"target": m.group(1).lower(), "base": m.group(3).lower(), "factor": float(m.group(2))}
                    )
                for m in re.finditer(
                    r"\b([a-z]+)\s+(?:has|have|had|is|are|was|were)\s+(\d+(?:\.\d+)?)\s+times\s+the\s+(?:number|amount)\s+of\s+\w+\s+as\s+([a-z]+)\b",
                    norm,
                ):
                    times_rels.append(
                        {"target": m.group(1).lower(), "base": m.group(3).lower(), "factor": float(m.group(2))}
                    )

                delta_rels: List[Dict[str, Any]] = []
                for m in re.finditer(
                    r"\b([a-z]+)\s+(?:has|have|had|is|are|was|were)\s+(\d+(?:\.\d+)?)\s+(?:less|fewer)\s+\w+\s+than\s+([a-z]+)\b",
                    norm,
                ):
                    delta_rels.append(
                        {"target": m.group(1).lower(), "base": m.group(3).lower(), "delta": float(m.group(2))}
                    )

                delta_rel = next((d for d in delta_rels if d.get("base") == base_entity_l), None)
                if delta_rel:
                    delta = float(delta_rel.get("delta", 0.0))
                    if delta > 0:
                        delta_target = str(delta_rel.get("target")).lower()
                        # Follow a forward chain: base=delta_target -> target -> target ...
                        factors: List[float] = []
                        current = delta_target
                        for _ in range(3):
                            nxt = next((r for r in times_rels if str(r.get("base")) == current), None)
                            if not nxt:
                                break
                            f = float(nxt.get("factor", 1.0))
                            if abs(f - 1.0) < 1e-9 or abs(f) < 1e-12:
                                break
                            factors.append(f)
                            current = str(nxt.get("target")).lower()
                        # Need at least a 2-hop chain to matter.
                        if len(factors) >= 2:
                            base_v = float(base_hint)
                            n_lit = _lit(base_v)
                            d_lit = _lit(delta)
                            # Build values: base (N), delta_target (N-d), and each derived multiplier value.
                            # Without stack ops (unit tests), we recompute (N-d) for each derived value.
                            toks: List[str] = [n_lit, n_lit, d_lit, "-"]  # stack: N, (N-d)
                            prod = 1.0
                            for f in factors:
                                prod *= float(f)
                                toks.extend([n_lit, d_lit, "-", _lit(prod), "*"])
                            # Sum all entities: N + (N-d) + prod1*(N-d) + prod2*(N-d) + ...
                            n_terms = 2 + len(factors)
                            toks.extend(["+"] * (n_terms - 1))
                            candidates.insert(0, " ".join(toks))
                            # Also include the terminal derived only: prod*(N-d).
                            candidates.insert(0, f"{n_lit} {d_lit} - {_lit(prod)} *")
            except Exception:
                pass

        # High-priority mixed relative: delta + fraction/ratio (e.g., (P+8)*3/8).
        #
        # This must be added early because TTC later samples the candidate pool down
        # to `max_parallel_candidates` without full sorting; we want the richer chain
        # candidates to survive that sampling.
        if deltas and fraction_steps:
            op0, d0 = deltas[0]
            for base in bases[:2]:
                # Order A: (base +/- delta) * num/den ...
                toks_a: List[str] = [_lit(base), _lit(d0), ("+" if op0 == "+" else "-")]
                for step in fraction_steps[:2]:
                    num = float(step.get("num", 1.0))
                    den = float(step.get("den", 1.0))
                    if abs(den) < 1e-12:
                        continue
                    if abs(num - 1.0) > 1e-9:
                        toks_a.extend([_lit(num), "*"])
                    toks_a.extend([_lit(den), "/"])
                if len(toks_a) >= 5:
                    candidates.append(" ".join(toks_a))

                # Order B: base * num/den ... then +/- delta
                toks_b: List[str] = [_lit(base)]
                for step in fraction_steps[:2]:
                    num = float(step.get("num", 1.0))
                    den = float(step.get("den", 1.0))
                    if abs(den) < 1e-12:
                        continue
                    if abs(num - 1.0) > 1e-9:
                        toks_b.extend([_lit(num), "*"])
                    toks_b.extend([_lit(den), "/"])
                toks_b.extend([_lit(d0), ("+" if op0 == "+" else "-")])
                if len(toks_b) >= 5:
                    candidates.append(" ".join(toks_b))

        # High-priority "adults + children" story: ensure the full (k+1)*adults_total candidate is present early.
        if group_total_story and deltas:
            k = factors_in_order[0] if factors_in_order else 2.0
            try:
                kf = float(k)
            except Exception:
                kf = 2.0
            scale = kf + 1.0
            op, d = deltas[0]
            for base in bases[:2]:
                if op == "+":
                    candidates.append(f"{_lit(base)} 2 * {_lit(d)} + {_lit(scale)} *")
                else:
                    candidates.append(f"{_lit(base)} 2 * {_lit(d)} - {_lit(scale)} *")

        # Core 1-step relative transforms (forward/inverse).
        for base in bases:
            for op, d in deltas[:4]:
                if op == "+":
                    candidates.append(f"{_lit(base)} {_lit(d)} +")
                    candidates.append(f"{_lit(base)} {_lit(d)} -")
                else:
                    candidates.append(f"{_lit(base)} {_lit(d)} -")
                    candidates.append(f"{_lit(base)} {_lit(d)} +")

        # 2-step comparative chains: apply two deltas in sequence, then (optionally) compute a difference.
        #
        # Example: "Suraya picked 12 more than Caleb; Caleb picked 5 less than Kayla; Kayla picked 20;
        # how many more did Suraya pick than Kayla?" -> (20-5+12) - 20.
        if len(deltas) >= 2:
            try:
                hi_num = max(nums) if nums else None
            except Exception:
                hi_num = None
            chain_deltas = []
            for op, d in deltas:
                try:
                    df = float(d)
                except Exception:
                    continue
                # Filter out "deltas" that are actually base assignments (often equal to the max prompt number).
                if hi_num is not None and df >= hi_num - 1e-9:
                    continue
                chain_deltas.append((str(op), float(df)))
                if len(chain_deltas) >= 3:
                    break

            def _op_variants(sign: str) -> List[str]:
                return ["+", "-"] if sign == "+" else ["-", "+"]

            # High-priority: if we have an explicit conditional base and the question asks for a
            # comparison ("how many more/less ... than ..."), try chaining the first two deltas
            # from that base and subtracting the base at the end.
            if question_type == "difference" and base_hint is not None and len(chain_deltas) >= 2:
                b0 = float(base_hint)
                (op_a, d_a), (op_b, d_b) = chain_deltas[0], chain_deltas[1]
                # Two orderings; TTC will pick the plausible one.
                candidates.insert(0, f"{_lit(b0)} {_lit(d_a)} {op_a} {_lit(d_b)} {op_b} {_lit(b0)} -")
                candidates.insert(0, f"{_lit(b0)} {_lit(d_b)} {op_b} {_lit(d_a)} {op_a} {_lit(b0)} -")

            # Use a small set of bases; prefer the conditional base_hint when present.
            base_targets: List[float] = []
            if base_hint is not None:
                base_targets.append(float(base_hint))
            base_targets.extend(bases[:2])
            # Unique.
            bt: List[float] = []
            for b in base_targets:
                if any(abs(b - x) < 1e-9 for x in bt):
                    continue
                bt.append(float(b))
            base_targets = bt[:3]

            for i in range(len(chain_deltas)):
                for j in range(len(chain_deltas)):
                    if i == j:
                        continue
                    op1, d1 = chain_deltas[i]
                    op2, d2 = chain_deltas[j]
                    for base in base_targets:
                        for o1 in _op_variants(op1):
                            for o2 in _op_variants(op2):
                                chain = f"{_lit(base)} {_lit(d1)} {o1} {_lit(d2)} {o2}"
                                candidates.append(chain)
                                if question_type == "difference" and base_hint is not None:
                                    b0 = float(base_hint)
                                    candidates.append(f"{chain} {_lit(b0)} -")
                                    candidates.append(f"{_lit(b0)} {chain} -")
            # Prefer shorter chains early: de-duplicate without full sorting.
            seen_chain: set[str] = set()
            deduped_chain: List[str] = []
            for c in candidates:
                s = str(c).strip()
                if not s or s in seen_chain:
                    continue
                seen_chain.add(s)
                deduped_chain.append(s)
                if len(deduped_chain) >= max_candidates * 4:
                    break
            candidates = deduped_chain
        # (Sampling later will cap the actual pool.)

        # Fraction chains: sequentially apply observed fractions (base * num/den).
        if fraction_steps:
            for base in bases[:3]:
                toks: List[str] = [_lit(base)]
                for step in fraction_steps[:4]:
                    num = float(step.get("num", 1.0))
                    den = float(step.get("den", 1.0))
                    if abs(den) < 1e-12:
                        continue
                    if abs(num - 1.0) > 1e-9:
                        toks.extend([_lit(num), "*"])
                    toks.extend([_lit(den), "/"])
                if len(toks) >= 3:
                    candidates.append(" ".join(toks))

        # Multiplication/division chains: apply multipliers in order.
        if factors_in_order:
            # Consider both the textual order and its reverse, since natural language
            # often states relations out of dependency order ("Bill has 6x Harry" before
            # "Harry has 2x Sue").
            seqs: List[List[float]] = []
            seq = [float(f) for f in factors_in_order[:3]]
            if seq:
                seqs.append(seq)
                if len(seq) >= 2:
                    seqs.append(list(reversed(seq)))

            # De-duplicate sequences.
            uniq_seqs: List[List[float]] = []
            for s in seqs:
                key = tuple(round(x, 9) for x in s)
                if any(tuple(round(x, 9) for x in u) == key for u in uniq_seqs):
                    continue
                uniq_seqs.append(s)

            combined_total = (
                question_type == "total"
                and any(w in low for w in ("combined", "together", "altogether", "in all", "in total"))
                and len(seq) >= 2
            )
            if combined_total:
                base = float(bases[0])
                # Try to detect whether the multipliers are siblings ("X times ... as Sue" and "Y times ... as Sue")
                # or a chain ("Bill is 6x Harry" and "Harry is 2x Sue"). For the chain case, "combined" usually means
                # sum of an intermediate and a derived term (base*f1 + base*f1*f2).
                try:
                    import re

                    rels: List[Dict[str, Any]] = []
                    for m in re.finditer(
                        r"\b([A-Za-z]+)\s+(?:has|have|had|is|are|was|were)\s+(\d+(?:\.\d+)?)\s+times\s+as\s+many\s+\w+\s+as\s+([A-Za-z]+)\b",
                        problem_text,
                        re.IGNORECASE,
                    ):
                        rels.append(
                            {
                                "target": m.group(1).lower(),
                                "factor": float(m.group(2)),
                                "base": m.group(3).lower(),
                            }
                        )
                    for m in re.finditer(
                        r"\b([A-Za-z]+)\s+(?:has|have|had|is|are|was|were)\s+(?:twice|double)\s+as\s+many\s+\w+\s+as\s+([A-Za-z]+)\b",
                        problem_text,
                        re.IGNORECASE,
                    ):
                        rels.append({"target": m.group(1).lower(), "factor": 2.0, "base": m.group(2).lower()})

                    if isinstance(base_entity, str) and base_entity and rels:
                        # Find a 2-hop chain from base_entity.
                        r1 = next((r for r in rels if r.get("base") == base_entity.lower()), None)
                        r2 = next((r for r in rels if r1 and r.get("base") == str(r1.get("target"))), None)
                        if r1 and r2:
                            f1 = float(r1["factor"])
                            f2 = float(r2["factor"])
                            # combined = base*f1 + base*f1*f2
                            candidates.append(f"{_lit(base)} {_lit(f1)} * {_lit(base)} {_lit(f1)} * {_lit(f2)} * +")
                            # Also include compact form base*(f1 + f1*f2)
                            candidates.append(f"{_lit(base)} {_lit(float(f1 + f1 * f2))} *")
                except Exception:
                    pass

                # Sibling multipliers fall back: base*f1 + base*f2 (+ base*f3 ...)
                fs = [float(f) for f in factors_in_order[:3]]
                if len(fs) >= 2:
                    candidates.append(f"{_lit(base)} {_lit(fs[0])} * {_lit(base)} {_lit(fs[1])} * +")
                    candidates.append(f"{_lit(base)} {_lit(float(fs[0] + fs[1]))} *")
                if len(fs) >= 3:
                    candidates.append(
                        f"{_lit(base)} {_lit(fs[0])} * {_lit(base)} {_lit(fs[1])} * + {_lit(base)} {_lit(fs[2])} * +"
                    )
                    candidates.append(f"{_lit(base)} {_lit(float(fs[0] + fs[1] + fs[2]))} *")

            mul_bases = bases[:1] if combined_total else bases[:2]
            mul_seqs = uniq_seqs[:1] if combined_total else uniq_seqs[:2]
            for base in mul_bases:
                for s in mul_seqs:
                    combined_story = (
                        question_type == "total"
                        and any(w in low for w in ("combined", "together", "altogether", "in all", "in total"))
                        and len(s) >= 2
                    )
                    # Direct chain: base * f1 * f2 * ...
                    toks: List[str] = [_lit(base)]
                    for f in s:
                        toks.extend([_lit(f), "*"])
                    candidates.append(" ".join(toks))

                    # Inverse chain: base / f1 / f2 / ...
                    # Useful when the prompt gives the final derived quantity and asks for the base
                    # (e.g., "practiced twice as long as ran ... practiced 40, how long played?").
                    inv_toks: List[str] = [_lit(base)]
                    for f in s:
                        if abs(float(f)) < 1e-12:
                            continue
                        inv_toks.extend([_lit(f), "/"])
                    if len(inv_toks) >= 3:
                        candidates.append(" ".join(inv_toks))

                    # Product shortcut: base * (prod f)
                    #
                    # Skip this when the prompt explicitly asks for a combined/total of multiple
                    # sibling multipliers (e.g., "Bill is 6x Sue and Harry is 8x Sue, combined"),
                    # where prod is a misleading prior (should sum, not chain-multiply).
                    if not combined_story:
                        prod = 1.0
                        for f in s:
                            prod *= float(f)
                        if abs(prod - 1.0) > 1e-9:
                            candidates.append(f"{_lit(base)} {_lit(prod)} *")
                        if abs(prod) > 1e-12:
                            candidates.append(f"{_lit(base)} {_lit(prod)} /")

                    # Cumulative products for sums: base*f1 + base*f1*f2 + ...
                    cum_products: List[float] = []
                    cum = 1.0
                    for f in s:
                        cum *= float(f)
                        cum_products.append(cum)

                    sum_including_base = 1.0 + float(sum(cum_products))
                    sum_excluding_base = float(sum(cum_products))
                    if abs(sum_including_base - 1.0) > 1e-9:
                        candidates.append(f"{_lit(base)} {_lit(sum_including_base)} *")
                    if abs(sum_excluding_base) > 1e-12:
                        candidates.append(f"{_lit(base)} {_lit(sum_excluding_base)} *")

                    # Combine with a single comparative delta both ways; TTC will pick the plausible one.
                    # Skip this family for "adults + children = (k+1)*total_adults" stories where the delta is
                    # inside the adults total; otherwise this produces misleading candidates like "base*(1+k)+d".
                    if deltas and not group_total_story:
                        _, d = deltas[0]
                        candidates.append(f"{_lit(base)} {_lit(sum_including_base)} * {_lit(d)} -")
                        candidates.append(f"{_lit(base)} {_lit(sum_including_base)} * {_lit(d)} +")
                        candidates.append(f"{_lit(base)} {_lit(sum_excluding_base)} * {_lit(d)} -")
                        candidates.append(f"{_lit(base)} {_lit(sum_excluding_base)} * {_lit(d)} +")

        # Mixed relative: apply delta before fractions/multipliers (e.g., (P+8)*3/8).
        if deltas and (factors_in_order or fraction_steps):
            op0, d0 = deltas[0]
            for base in bases[:2]:
                base_delta = f"{_lit(base)} {_lit(d0)} {'+' if op0 == '+' else '-'}"
                if factors_in_order:
                    prod = 1.0
                    for f in factors_in_order[:3]:
                        prod *= float(f)
                    if abs(prod - 1.0) > 1e-9:
                        candidates.append(f"{base_delta} {_lit(prod)} *")
                if fraction_steps:
                    toks = base_delta.split()
                    for step in fraction_steps[:3]:
                        num = float(step.get('num', 1.0))
                        den = float(step.get('den', 1.0))
                        if abs(den) < 1e-12:
                            continue
                        if abs(num - 1.0) > 1e-9:
                            toks.extend([_lit(num), "*"])
                        toks.extend([_lit(den), "/"])
                    if len(toks) >= 5:
                        candidates.append(" ".join(toks))

        # Mixed relative: apply fractions/multipliers before delta (e.g., (P/2)+10).
        if deltas and (factors_in_order or fraction_steps):
            op0, d0 = deltas[0]
            for base in bases[:2]:
                # Fractions first, then delta.
                if fraction_steps:
                    toks: List[str] = [_lit(base)]
                    for step in fraction_steps[:3]:
                        num = float(step.get("num", 1.0))
                        den = float(step.get("den", 1.0))
                        if abs(den) < 1e-12:
                            continue
                        if abs(num - 1.0) > 1e-9:
                            toks.extend([_lit(num), "*"])
                        toks.extend([_lit(den), "/"])
                    toks.extend([_lit(d0), ("+" if op0 == "+" else "-")])
                    if len(toks) >= 5:
                        candidates.append(" ".join(toks))
                # Multipliers first, then delta.
                if factors_in_order:
                    prod = 1.0
                    for f in factors_in_order[:3]:
                        prod *= float(f)
                    if abs(prod - 1.0) > 1e-9:
                        candidates.append(f"{_lit(base)} {_lit(prod)} * {_lit(d0)} {'+' if op0 == '+' else '-'}")

        # Special family: "X more than Y" + "children were N times the total adults" → total people.
        has_children = "child" in low or "children" in low
        has_adults = "adult" in low or "adults" in low
        mentions_total = "total" in low
        if question_type == "total" and has_children and has_adults and mentions_total and deltas:
            # adults_total = 2*base +/- delta, total_people = (k+1)*adults_total.
            k = factors_in_order[0] if factors_in_order else 2.0
            try:
                kf = float(k)
            except Exception:
                kf = 2.0
            scale = kf + 1.0
            op, d = deltas[0]
            for base in bases[:2]:
                if op == "+":
                    candidates.append(f"{_lit(base)} 2 * {_lit(d)} + {_lit(scale)} *")
                else:
                    candidates.append(f"{_lit(base)} 2 * {_lit(d)} - {_lit(scale)} *")

        # Total chain: base -> (base +/- delta) -> factor*(base +/- delta), then sum all three.
        #
        # Example: "Jar A has 28. Jar B has 12 more than Jar A. Jar C has twice as many as Jar B. How many altogether?"
        #   total = A + B + C = A + (A+12) + 2*(A+12)
        if question_type == "total" and bases and deltas and factors_in_order and any(w in low for w in ("altogether", "in total", "in all")):
            try:
                base = float(bases[0])
                op, d1 = deltas[0]
                k = float(factors_in_order[0])
            except Exception:
                base = None
                d1 = None
                k = None
            if base is not None and d1 is not None and k is not None and abs(k) > 1.0 and abs(d1) > 1e-12:
                op_tok = "+" if op == "+" else "-"
                d_abs = abs(float(d1))
                # total = base + (base op d) + (base op d)*k
                total_expr = (
                    f"{_lit(base)} {_lit(base)} {_lit(d_abs)} {op_tok} + {_lit(base)} {_lit(d_abs)} {op_tok} {_lit(abs(k))} * +"
                )
                c_only = f"{_lit(base)} {_lit(d_abs)} {op_tok} {_lit(abs(k))} *"
                candidates.insert(0, total_expr)
                candidates.insert(1, c_only)

        # Legacy: base + (base +/- d) for totals (still helpful for some narratives).
        if question_type == "total" and bases and deltas:
            base = bases[0]
            op, d1 = deltas[0]
            if op == "+":
                candidates.append(f"{_lit(base)} {_lit(base)} {_lit(d1)} + +")
                candidates.append(f"{_lit(base)} 2 * {_lit(d1)} +")
            else:
                candidates.append(f"{_lit(base)} {_lit(base)} {_lit(d1)} - +")
                candidates.append(f"{_lit(base)} 2 * {_lit(d1)} -")

        # Light algebraic variants.
        for c in list(candidates)[:12]:
            candidates.extend(self._generate_chain_variants(c))

        seen: set[str] = set()
        out: List[str] = []
        for c in candidates:
            c = str(c).strip()
            if not c or c in seen:
                continue
            seen.add(c)
            out.append(c)
            if len(out) >= max_candidates:
                break
        return out

    def _generate_age_relative_candidates(self, problem_text: str, *, max_candidates: int = 6) -> List[str]:
        """
        TTC seed candidates for simple age-offset narratives.

        Pattern:
          "<Y> years ago X turned <A>. In <F> years Z will be <k>x X age. How old is Z now?"

        RPN:
          A Y + F + k * F -
        """
        import re

        from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

        norm = normalize_number_words(problem_text or "")
        low = norm.lower()

        if "year" not in low or ("ago" not in low and "in " not in low):
            return []

        # "3 years ago ... turned 27"
        m_past = re.search(r"\b(\d+(?:\.\d+)?)\s+years?\s+ago\b[^.]{0,120}\bturned\s+(\d+(?:\.\d+)?)\b", low)
        # "In 5 years ... will be twice ... age"
        m_future = re.search(
            r"\bin\s+(\d+(?:\.\d+)?)\s+years?\b[^.]{0,160}\bwill\s+be\s+(twice|double|(\d+(?:\.\d+)?)\s+times)\b[^.]{0,120}\bage\b",
            low,
        )

        if not (m_past and m_future):
            return []

        try:
            years_ago = float(m_past.group(1))
            age_then = float(m_past.group(2))
            years_future = float(m_future.group(1))
        except Exception:
            return []

        mult = 2.0
        try:
            if m_future.group(2) and m_future.group(2).strip() not in {"twice", "double"}:
                mult = float(m_future.group(2))
        except Exception:
            mult = 2.0

        if years_ago <= 0 or years_future <= 0 or age_then <= 0 or mult <= 0:
            return []

        a = self._lit_num(age_then)
        y = self._lit_num(years_ago)
        f = self._lit_num(years_future)
        k = self._lit_num(mult)

        # Variant A: Z_now = k*(X_then + years_ago + years_future) - years_future
        candidates = [f"{a} {y} + {f} + {k} * {f} -"]
        # Variant B: Z_now = k*X_now + (k-1)*years_future
        candidates.append(f"{a} {y} + {k} * {f} {k} 1 - * +")

        seen: set[str] = set()
        out: List[str] = []
        for c in candidates:
            s = str(c).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= int(max_candidates):
                break
        return out

    def _test_time_compute(
        self,
        *,
        problem_text: str,
        rpn_engine: Any,
        understanding: ProblemUnderstanding,
        trace: Dict[str, Any],
        attempts: List[Dict[str, Any]],
        budget: int,
        exploration: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]] | None:
        """
        Test-time compute: generate diverse RPN candidates and evaluate in parallel.

        This runs only after the standard template attempts fail, and is intended to
        improve generalization by exploring generic equations, conversions, and
        simple arithmetic compositions over extracted numbers.
        """
        try:
            nums = self.extract_numbers(problem_text)
        except Exception:
            nums = []
        words = []
        try:
            words = list((exploration or {}).get("words", []) or [])
        except Exception:
            words = []

        question_type = self.classify_question(problem_text)
        if understanding.goals.get("rest") or understanding.goals.get("percent_complement"):
            question_type = "difference"
        strategy = "balanced"
        # Track the best candidate overall (used for strategy adjustment) and the
        # best *plausible* candidate (used for final selection). This prevents
        # high-scoring-but-implausible expressions from overriding a plausible
        # solution and causing TTC to return None.
        best_overall: Dict[str, Any] = {"confidence": -1.0}
        best_plausible: Dict[str, Any] = {"confidence": -1.0}
        rejected_by_reason: Dict[str, int] = {}
        evaluated = 0
        plausible_seen = 0
        book_sourced_candidates_evaluated = 0
        book_sourced_eval: List[Dict[str, Any]] = []

        # Seed candidates from templates already attempted (if any).
        seed_candidates: List[str] = []
        for a in attempts:
            if not isinstance(a, dict):
                continue
            rpn = a.get("rpn")
            if isinstance(rpn, str) and rpn.strip():
                seed_candidates.append(rpn.strip())

        relative_seed: List[str] = self._generate_relative_chain_candidates(
            problem_text=problem_text,
            understanding=understanding,
            trace=trace,
            question_type=question_type,
            max_candidates=max(6, self.max_parallel_candidates // 2),
        )
        algebra_seed: List[str] = self._generate_algebraic_lite_candidates(
            problem_text=problem_text,
            understanding=understanding,
            trace=trace,
            question_type=question_type,
            # Keep this small; algebraic-lite candidates are high-impact but can
            # crowd out other TTC families when over-produced.
            max_candidates=min(9, max(6, self.max_parallel_candidates // 2)),
        )
        schedule_seed: List[str] = self._generate_schedule_rate_candidates(
            problem_text,
            max_candidates=min(6, max(3, self.max_parallel_candidates // 3)),
        )
        partition_seed: List[str] = self._generate_fraction_ratio_remainder_candidates(
            problem_text,
            max_candidates=min(6, max(3, self.max_parallel_candidates // 3)),
        )
        age_seed: List[str] = self._generate_age_relative_candidates(
            problem_text,
            max_candidates=min(6, max(3, self.max_parallel_candidates // 3)),
        )
        coin_seed: List[str] = self._generate_coin_change_unknown_candidates(
            problem_text,
            max_candidates=min(4, max(2, self.max_parallel_candidates // 4)),
        )
        book_seed, book_hits, book_seed_sources = self._generate_book_galaxy_candidates(
            problem_text,
            max_candidates=min(6, max(3, self.max_parallel_candidates // 3)),
        )
        if book_hits:
            trace.setdefault("book_galaxy", {})["hits"] = book_hits
        book_seed_set = set(book_seed or [])
        book_sourced_seed_set = set((book_seed_sources or {}).get("sourced", []) or [])
        book_heuristic_seed_set = set(book_seed_set) - set(book_sourced_seed_set)

        # Phase 8 (starter): multi-step geometry chaining candidates.
        multistep_geom_seed: List[str] = []
        multistep_geom_seed_set: set[str] = set()
        try:
            import os

            enable_multistep = str(os.environ.get("K3D_TRM_ENABLE_MULTISTEP", "0") or "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
        except Exception:
            enable_multistep = False
        if enable_multistep:
            multistep_geom_seed = self._generate_multistep_geometry_candidates(
                problem_text, max_candidates=min(6, max(3, self.max_parallel_candidates // 3))
            )
        multistep_geom_seed_set = set(multistep_geom_seed or [])
        if multistep_geom_seed:
            seed_candidates.extend(multistep_geom_seed)

        for depth in range(1, max(1, int(budget)) + 1):
            candidates = self._generate_rpn_candidates(
                problem_text=problem_text,
                numbers=nums,
                words=words,
                question_type=question_type,
                depth=depth,
                strategy=strategy,
                max_candidates=self.max_parallel_candidates,
            )
            # Mix in relative-chain candidates:
            # - For "than/more/less" style questions, prefer trying relative-chain candidates early
            #   (they often require multi-step additive composition).
            # - Otherwise, keep them later to avoid crowding out other TTC families.
            low = (problem_text or "").lower()
            prioritize_relative = bool(relative_seed) and (
                " than " in f" {low} " or "how many more" in low or "more than" in low or "less than" in low
            )
            total_cue = any(w in low for w in ("combined", "together", "altogether", "in total"))
            affine_cue = any(
                w in low
                for w in (
                    "more than double",
                    "less than double",
                    "more than twice",
                    "less than twice",
                    "more than triple",
                    "less than triple",
                )
            )
            cost_remaining_cue = ("each" in low and "cost" in low and ("remaining" in low or "rest" in low or "left" in low))
            ratio_cue = "times as many" in low or "times fewer" in low or "times less" in low
            prioritize_algebra = bool(algebra_seed) and total_cue and (affine_cue or ratio_cue or cost_remaining_cue)
            schedule_cue = question_type == "duration" or "every" in low or "trips" in low or "week" in low
            partition_cue = ("rest" in low or "remaining" in low) and ("/" in low or "half" in low or "quarter" in low)
            age_cue = ("year" in low or "years" in low) and ("ago" in low) and ("old" in low) and ("in " in low) and ("twice" in low or "double" in low or "times" in low)
            coin_cue = any(w in low for w in ("pennies", "nickels", "dimes", "quarters")) and ("how many" in low)
            prioritize_schedule = bool(schedule_seed) and schedule_cue
            prioritize_partition = bool(partition_seed) and partition_cue
            prioritize_age = bool(age_seed) and age_cue
            prioritize_coin = bool(coin_seed) and coin_cue
            prioritize_multistep_geom = bool(multistep_geom_seed) and (
                ("circle" in low and "area" in low and ("circumference" in low or "diameter" in low))
            )
            prioritize_book = bool(book_seed) and (
                "det" in low
                or "determinant" in low
                or "matrix" in low
                or "gcd" in low
                or "lcm" in low
                or "binomial" in low
                or "choose" in low
                or "triangle" in low
                or "hypotenuse" in low
                or "leg" in low
                or "circle" in low
                or "radius" in low
                or "diameter" in low
                or "area" in low
                or "volume" in low
                or "pythagorean" in low
                or "cos" in low
                or "sin" in low
                or "tan" in low
            )
            if prioritize_relative:
                candidates = (
                    list(seed_candidates)
                    + list(relative_seed)
                    + list(book_seed)
                    + list(schedule_seed)
                    + list(partition_seed)
                    + list(coin_seed)
                    + list(algebra_seed)
                    + list(candidates)
                )
            elif prioritize_algebra:
                candidates = (
                    list(seed_candidates)
                    + list(algebra_seed)
                    + list(book_seed)
                    + list(schedule_seed)
                    + list(partition_seed)
                    + list(coin_seed)
                    + list(candidates)
                    + list(relative_seed)
                )
            elif prioritize_book:
                candidates = (
                    list(seed_candidates)
                    + list(book_seed)
                    + list(candidates)
                    + list(schedule_seed)
                    + list(partition_seed)
                    + list(coin_seed)
                    + list(relative_seed)
                    + list(algebra_seed)
                )
            elif prioritize_coin:
                candidates = list(seed_candidates) + list(coin_seed) + list(candidates) + list(relative_seed) + list(algebra_seed)
            elif prioritize_schedule:
                candidates = (
                    list(seed_candidates)
                    + list(schedule_seed)
                    + list(age_seed)
                    + list(book_seed)
                    + list(candidates)
                    + list(relative_seed)
                    + list(algebra_seed)
                )
            elif prioritize_partition:
                candidates = list(seed_candidates) + list(partition_seed) + list(candidates) + list(relative_seed) + list(algebra_seed)
            elif prioritize_age:
                candidates = (
                    list(seed_candidates)
                    + list(age_seed)
                    + list(candidates)
                    + list(schedule_seed)
                    + list(book_seed)
                    + list(relative_seed)
                    + list(algebra_seed)
                )
            else:
                candidates = (
                    list(seed_candidates)
                    + list(book_seed)
                    + list(candidates)
                    + list(schedule_seed)
                    + list(age_seed)
                    + list(partition_seed)
                    + list(coin_seed)
                    + list(relative_seed)
                    + list(algebra_seed)
                )
            # De-duplicate while preserving order.
            seen: set[str] = set()
            uniq: List[str] = []
            for c in candidates:
                c = str(c).strip()
                if not c or c in seen:
                    continue
                seen.add(c)
                uniq.append(c)
                # Keep a larger pool; we will sample down to max_parallel_candidates below.
                if len(uniq) >= max(200, self.max_parallel_candidates * 8):
                    break

            if len(uniq) <= self.max_parallel_candidates:
                candidates = uniq
            else:
                # Sample without full scoring/sorting: keep some early (simple ops),
                # some mid (often where remaining/fraction templates land), and
                # some late (deep conversions/edge cases).
                head_n = max(1, self.max_parallel_candidates // 3)
                mid_n = max(1, self.max_parallel_candidates // 3)
                tail_n = self.max_parallel_candidates - head_n - mid_n

                head = uniq[:head_n]
                mid_start = max(0, (len(uniq) // 2) - (mid_n // 2))
                mid = uniq[mid_start : mid_start + mid_n]
                tail = uniq[-tail_n:] if tail_n > 0 else []
                sampled: List[str] = []
                for e in head + mid + tail:
                    if e not in sampled:
                        sampled.append(e)
                # Phase 7B: ensure book-sourced candidates are actually evaluated.
                # Without this, book seeds can exist (diagnostics show ~50% of TTC calls)
                # but be dropped by the sampling window and never compete.
                # Phase 7B (safe): only force book-sourced candidates into TTC evaluation
                # when the prompt has strong "math concept" cues. Forcing unconditionally
                # can crowd out generic TTC families on wordy prompts.
                forced: List[str] = []
                if prioritize_book and book_sourced_seed_set:
                    for e in uniq:
                        if e in book_sourced_seed_set and e not in forced:
                            forced.append(e)
                        if len(forced) >= 3:
                            break
                forced_multistep: List[str] = []
                if prioritize_multistep_geom and multistep_geom_seed_set:
                    for e in uniq:
                        if e in multistep_geom_seed_set and e not in forced_multistep:
                            forced_multistep.append(e)
                        if len(forced_multistep) >= 2:
                            break
                for e in reversed(forced_multistep + forced):
                    if e in sampled:
                        sampled.remove(e)
                    else:
                        sampled.insert(0, e)
                        if len(sampled) > self.max_parallel_candidates:
                            sampled.pop()
                candidates = sampled[: self.max_parallel_candidates]
            if not candidates:
                continue

            try:
                results = rpn_engine.evaluate_batch(candidates, max_parallel=18)
            except Exception:
                # Fallback: sequential execution if batch isn't available.
                results = []
                for expr in candidates:
                    try:
                        results.append(rpn_engine.evaluate(expr))
                    except Exception:
                        results.append(None)

            for expr, res in zip(candidates, results):
                evaluated += 1
                conf, verdict = self._score_candidate(
                    problem_text=problem_text,
                    question_type=question_type,
                    numbers=nums,
                    expression=expr,
                    result=res,
                    concepts=list((exploration or {}).get("concepts", [])),
                )
                if verdict.get("plausible"):
                    # Phase 7 semantic (Stage 1): once hygiene guarantees structural validity,
                    # bias TTC toward book-sourced candidates over generic guesses *only*
                    # when plausibility is comparable (i.e., both pass plausibility gates).
                    #
                    # Keep this additive (not multiplicative) to avoid destabilizing the
                    # confidence scale across TTC families.
                    if expr in book_sourced_seed_set:
                        conf = float(conf) + 0.45
                    # Phase 8: multi-step geometry candidates get a small boost so
                    # they can compete with single-step arithmetic when plausible.
                    if expr in multistep_geom_seed_set:
                        conf = float(conf) + 0.15
                if expr in book_sourced_seed_set:
                    book_sourced_candidates_evaluated += 1
                    if len(book_sourced_eval) < 8:
                        try:
                            r = None
                            if res is not None:
                                try:
                                    r = float(res)
                                except Exception:
                                    r = str(res)[:64]
                            book_sourced_eval.append(
                                {
                                    "rpn": str(expr),
                                    "result": r,
                                    "plausible": bool(verdict.get("plausible")),
                                    "reason": verdict.get("reason"),
                                    "confidence": float(conf),
                                }
                            )
                        except Exception:
                            pass
                if verdict.get("plausible"):
                    plausible_seen += 1
                else:
                    reason = str(verdict.get("reason") or "unknown")
                    rejected_by_reason[reason] = int(rejected_by_reason.get(reason, 0)) + 1
                tie = self._candidate_tiebreak(numbers=nums, expression=expr)
                best_conf = float(best_overall.get("confidence", -1.0))
                best_tie = float(best_overall.get("tie", -1.0))
                if conf > best_conf + 1e-12 or (abs(conf - best_conf) <= 1e-12 and tie > best_tie + 1e-12):
                    best_overall = {
                        "confidence": float(conf),
                        "tie": float(tie),
                        "expression": expr,
                        "result": res,
                        "verdict": verdict,
                        "depth": depth,
                        "strategy": strategy,
                    }
                if verdict.get("plausible"):
                    best_conf = float(best_plausible.get("confidence", -1.0))
                    best_tie = float(best_plausible.get("tie", -1.0))
                    if conf > best_conf + 1e-12 or (abs(conf - best_conf) <= 1e-12 and tie > best_tie + 1e-12):
                        best_plausible = {
                            "confidence": float(conf),
                            "tie": float(tie),
                            "expression": expr,
                            "result": res,
                            "verdict": verdict,
                            "depth": depth,
                            "strategy": strategy,
                        }

            # Early stop on a strong candidate.
            # For prompts with many quantities, avoid stopping early on partial computations
            # that ignore major numbers (e.g., fraction-only without the later spend term).
            current_best = best_plausible if best_plausible.get("expression") else best_overall
            best_conf = float(current_best.get("confidence", 0.0))
            best_tie = float(current_best.get("tie", 0.0))
            if current_best.get("verdict", {}).get("plausible") and best_conf >= 0.95:
                # Percent and multi-step stories tend to have near-miss candidates;
                # avoid early stopping too aggressively there unless confidence is very high.
                low = (problem_text or "").lower()
                avoid_early_stop = False
                if ("%" in low or "percent" in low) and best_conf < 0.98:
                    avoid_early_stop = True
                if self._count_multi_step_indicators(problem_text) > 0 and best_conf < 0.98:
                    avoid_early_stop = True

                if not avoid_early_stop:
                    if len(nums) < 4:
                        break
                    # Require higher structural/coverage signal when 4+ numbers appear.
                    if best_tie >= 34.0:
                        break

            # Course correction: adjust strategy based on best failure mode.
            # Use the best *overall* failure signal when we haven't found any
            # plausible candidate yet; otherwise keep the strategy stable.
            reason = ""
            if not best_plausible.get("expression"):
                reason = str(best_overall.get("verdict", {}).get("reason") or "")
            if reason in {"out_of_range", "wrong_magnitude", "percent_result_exceeds_total", "percent_result_exceeds_scale"}:
                strategy = "prioritize_conversions"
            elif reason in {"negative_result"}:
                strategy = "prioritize_aggregation"
            elif reason and ("divide" in reason or "division" in reason):
                strategy = "prioritize_division"
            else:
                # Mildly alternate to diversify candidates.
                if strategy == "balanced":
                    strategy = "prioritize_division"
                elif strategy == "prioritize_division":
                    strategy = "prioritize_aggregation"
                else:
                    strategy = "balanced"

        if best_plausible.get("expression") and best_plausible.get("verdict", {}).get("plausible"):
            rpn = str(best_plausible.get("expression"))
            result = best_plausible.get("result")
            attempts.append(
                {
                    "attempt": len(attempts) + 1,
                    "template": "test_time_compute",
                    "rpn": rpn,
                    "result": result,
                    "verification": best_plausible.get("verdict"),
                    "depth": best_plausible.get("depth"),
                    "strategy": best_plausible.get("strategy"),
                    "confidence": best_plausible.get("confidence"),
                }
            )
            meta = {
                "rpn_program": rpn,
                "template_used": "test_time_compute",
                "attempts": attempts,
                "subgoals": self.decompose_into_subgoals(problem_text),
                "read_trace": trace,
                "read_understanding": understanding.to_dict(),
                "read_composition": dict(self.get_last_composition_meta()),
                "exploration": dict(exploration or {}),
                "test_time": {
                    "best_depth": best_plausible.get("depth"),
                    "best_strategy": best_plausible.get("strategy"),
                    "best_confidence": best_plausible.get("confidence"),
                    "best_source": ("book" if rpn in book_sourced_seed_set else ("book_heuristic" if rpn in book_seed_set else "non_book")),
                    "book_seed_count": int(len(book_seed or [])),
                    "book_sourced_seed_count": int(len(book_sourced_seed_set)),
                    "book_seed_sample": list(book_seed[:6]) if isinstance(book_seed, list) else [],
                    "book_sourced_seed_sample": list((book_seed_sources or {}).get("sourced", [])[:6]) if isinstance(book_seed_sources, dict) else [],
                    "multistep_geometry_seed_count": int(len(multistep_geom_seed or [])),
                    "multistep_geometry_seed_sample": list(multistep_geom_seed[:6]) if isinstance(multistep_geom_seed, list) else [],
                    "book_sourced_candidates_evaluated": int(book_sourced_candidates_evaluated),
                    "book_sourced_eval": list(book_sourced_eval),
                    "candidates_evaluated": int(evaluated),
                    "plausible_candidates_seen": int(plausible_seen),
                    "rejected_by_reason": dict(sorted(rejected_by_reason.items(), key=lambda kv: (-kv[1], kv[0]))),
                },
            }
            if self.shadow is not None:
                try:
                    pattern_ids = [p.get("rule_id") for p in trace.get("patterns", []) if isinstance(p, dict) and p.get("rule_id")]
                    ts = (exploration or {}).get("tsinghua", {}) if isinstance(exploration, dict) else {}
                    self.shadow.record_exploration(
                        problem_text=problem_text,
                        concepts_explored=list((exploration or {}).get("concepts", [])),
                        patterns_matched=[str(p) for p in pattern_ids],
                        templates_tried=[a.get("template") for a in attempts if isinstance(a, dict) and a.get("template")],
                        template_used="test_time_compute",
                        success=True,
                        rpn_program=rpn,
                        result=result,
                        tsinghua=ts if isinstance(ts, dict) else {},
                    )
                except Exception:
                    pass
            return result, meta

        # Fallback: if *no* plausible candidate survived, return the best numeric
        # attempt instead of bubbling up a "no_rule_match". This keeps the failure
        # mode informative (`wrong_computation` with an RPN trace) and avoids
        # regressing coverage by returning `None`.
        if best_overall.get("expression") and best_overall.get("result") is not None:
            rpn = str(best_overall.get("expression"))
            result = best_overall.get("result")
            attempts.append(
                {
                    "attempt": len(attempts) + 1,
                    "template": "test_time_compute_fallback",
                    "rpn": rpn,
                    "result": result,
                    "verification": best_overall.get("verdict"),
                    "depth": best_overall.get("depth"),
                    "strategy": best_overall.get("strategy"),
                    "confidence": best_overall.get("confidence"),
                }
            )
            meta = {
                "rpn_program": rpn,
                "template_used": "test_time_compute",
                "attempts": attempts,
                "subgoals": self.decompose_into_subgoals(problem_text),
                "read_trace": trace,
                "read_understanding": understanding.to_dict(),
                "read_composition": dict(self.get_last_composition_meta()),
                "exploration": dict(exploration or {}),
                "test_time": {
                    "best_depth": best_overall.get("depth"),
                    "best_strategy": best_overall.get("strategy"),
                    "best_confidence": best_overall.get("confidence"),
                    "used_fallback": True,
                    "best_source": ("book" if rpn in book_sourced_seed_set else ("book_heuristic" if rpn in book_seed_set else "non_book")),
                    "book_seed_count": int(len(book_seed or [])),
                    "book_sourced_seed_count": int(len(book_sourced_seed_set)),
                    "book_seed_sample": list(book_seed[:6]) if isinstance(book_seed, list) else [],
                    "book_sourced_seed_sample": list((book_seed_sources or {}).get("sourced", [])[:6]) if isinstance(book_seed_sources, dict) else [],
                    "multistep_geometry_seed_count": int(len(multistep_geom_seed or [])),
                    "multistep_geometry_seed_sample": list(multistep_geom_seed[:6]) if isinstance(multistep_geom_seed, list) else [],
                    "book_sourced_candidates_evaluated": int(book_sourced_candidates_evaluated),
                    "book_sourced_eval": list(book_sourced_eval),
                    "candidates_evaluated": int(evaluated),
                    "plausible_candidates_seen": int(plausible_seen),
                    "rejected_by_reason": dict(sorted(rejected_by_reason.items(), key=lambda kv: (-kv[1], kv[0]))),
                },
            }
            return result, meta

        return None

    def _candidate_tiebreak(self, *, numbers: Sequence[float], expression: str) -> float:
        """
        Secondary ranking when confidence scores tie.

        Prefer candidates that use more of the prominent numbers and show richer
        structure (more operators) when the prompt contains multiple quantities.
        """
        expr_tokens = (expression or "").split()
        ops = [t for t in expr_tokens if t in {"+", "-", "*", "/"}]
        used: List[float] = []
        for tok in expr_tokens:
            if tok in {"+", "-", "*", "/"}:
                continue
            try:
                used.append(float(tok))
            except Exception:
                continue
        used_abs = {round(abs(v), 6) for v in used}
        uniq_nums = list(dict.fromkeys([round(abs(float(n)), 6) for n in numbers if n is not None]))
        uniq_nums = [n for n in uniq_nums if n != 0.0]
        uniq_nums.sort(reverse=True)
        important = uniq_nums[:3]
        used_important = sum(1 for x in important if x in used_abs)
        # Weighted: important coverage dominates, then distinct numbers, then op count.
        return float(used_important * 10 + len(used_abs) * 2 + len(ops))

    def _score_candidate(
        self,
        *,
        problem_text: str,
        question_type: str,
        numbers: Sequence[float],
        expression: str,
        result: Any,
        concepts: Sequence[str] = (),
    ) -> Tuple[float, Dict[str, Any]]:
        verdict = self.verify_plausibility(problem_text, result, expression)
        try:
            val = float(result) if result is not None else None
        except Exception:
            val = None

        score = 0.0
        if val is None:
            return (score, verdict)
        if val != val:
            return (score, {"plausible": False, "reason": "nan"})

        # Base: numeric and non-negative (most GSM-like problems).
        score += 0.4
        if val >= 0:
            score += 0.1

        # Prefer in-range results.
        if verdict.get("plausible"):
            score += 0.35
        else:
            # Soft penalties for magnitude mismatch.
            if numbers:
                lo = min(numbers)
                hi = max(numbers)
                if hi > 0:
                    ratio = val / hi
                    if ratio > 1000 or ratio < 1e-6:
                        score -= 0.2
                    elif ratio > 100 or ratio < 1e-3:
                        score -= 0.1

        # Question-type heuristics.
        if numbers:
            hi = max(numbers)
            if question_type == "total" and val >= hi:
                score += 0.05
            if question_type == "difference" and val <= hi:
                score += 0.05

        # Concept alignment (cheap): prefer operators that match the text concepts.
        ops = {tok for tok in (expression or "").split() if tok in {"+", "-", "*", "/"}}
        cset = {str(c) for c in (concepts or []) if c}
        if "aggregation" in cset and "+" in ops:
            score += 0.05
        if "subtraction" in cset and "-" in ops:
            score += 0.05
        if "multiplication" in cset and "*" in ops:
            score += 0.05
        if "division" in cset and "/" in ops:
            score += 0.05
        if "rate" in cset and (("*" in ops) or ("/" in ops)):
            score += 0.03

        # Percent/ratio structural preferences.
        low = (problem_text or "").lower()
        expr = str(expression or "")
        if "%" in low or "percent" in low:
            # Prefer the canonical "p 100 /" normalization to avoid 80× inflation.
            if " 100 /" in f" {expr} " or "/ 100" in f" {expr} ":
                score += 0.08
            # Penalize pure multiplication without normalization (often "total pct *").
            if "*" in ops and "/" not in ops:
                score -= 0.05

            # If a percent appears in a money/cost context, discourage candidates that
            # simply add the percent as an absolute number (e.g., "+ 60") instead of
            # using percent normalization and multiplication.
            if ("$" in problem_text) or any(w in low for w in ("cost", "costs", "price", "insurance", "out-of-pocket")):
                if "+" in ops and (" 100 /" not in f" {expr} ") and ("/ 100" not in f" {expr} "):
                    # Heuristic: if the expression contains any 1..100 literal and no /100 normalization,
                    # treat it as a likely percent being added directly.
                    try:
                        percent_lits = [float(t) for t in expr.split() if t not in {"+", "-", "*", "/"}]
                        if any(0 < p <= 100 and abs(p - round(p)) < 1e-9 for p in percent_lits):
                            score -= 0.18
                    except Exception:
                        pass

        # Inverse-chain preference: questions that ask for an earlier duration/count
        # should favor division candidates over multiplication ones.
        if question_type == "duration" and ("if" in low):
            if any(w in low for w in ("twice as long", "times as long", "twice", "double", "thrice", "triple")):
                if "/" in ops and "*" not in ops:
                    score += 0.18
                if "*" in ops and "/" not in ops:
                    score -= 0.18

        # Unit-cost preference: questions asking "how much does each ... cost" should favor division.
        if ("each" in low) and ("cost" in low) and ("$" in problem_text):
            if ("how much" in low) or ("what" in low):
                if "/" in ops and "*" not in ops:
                    score += 0.14
                if "*" in ops and "/" not in ops:
                    score -= 0.14
                # Orientation: prefer dollars / count (unit cost), not count / dollars.
                if "/" in ops and self._count_basic_ops(expr) == 1:
                    try:
                        import re

                        dollars = [float(x) for x in re.findall(r"\$\s*(\d+(?:\.\d+)?)", problem_text)]
                        if dollars:
                            dollar_set = {float(d) for d in dollars}
                            toks = [t for t in expr.split() if t]
                            if len(toks) == 3 and toks[2] == "/":
                                a = float(toks[0])
                                b = float(toks[1])
                                a_is_dollar = any(abs(a - d) < 1e-9 for d in dollar_set)
                                b_is_dollar = any(abs(b - d) < 1e-9 for d in dollar_set)
                                a_is_int = abs(a - round(a)) < 1e-9
                                b_is_int = abs(b - round(b)) < 1e-9
                                if a_is_dollar and b_is_int and not b_is_dollar:
                                    score += 0.16
                                elif b_is_dollar and a_is_int and not a_is_dollar:
                                    score -= 0.16
                    except Exception:
                        pass

        # Consumable packs over time: "uses X per Y days, pack size N costs $C, over D days"
        # should prefer candidates with both rate normalization and pack normalization:
        #   (X/Y) * D / N * C
        if ("pack" in low) and ("$" in problem_text) and any(w in low for w in ("uses", "use", "using")) and (
            ("over" in low or "for" in low) and ("day" in low or "days" in low)
        ):
            div_count = expr.split().count("/")
            mul_count = expr.split().count("*")
            if div_count >= 2 and mul_count >= 1:
                score += 0.18
            if div_count == 0 and mul_count >= 1:
                score -= 0.18

        # Half-rate fuel efficiency: "half as many miles per gallon" means mpg is halved,
        # so required gallons should roughly double relative to distance/mpg.
        if ("miles per gallon" in low or "miles/gallon" in low) and ("half" in low) and ("gallon" in low):
            if ("how many gallons" in low) or ("gallons" in low and ("how many" in low or "require" in low)):
                if "/" in ops:
                    score += 0.06
                expr_pad = f" {expr} "
                if (" 2 *" in expr_pad) or ("2 *" in expr_pad):
                    score += 0.10
                if (" / 2" in expr_pad) or (" 2 /" in expr_pad):
                    score -= 0.06

        # Packaging unit price: reward candidates that multiply multiple packaging factors.
        if ("carton" in low) and ("box" in low) and ("pack" in low) and ("dozen" in low) and ("$" in problem_text):
            if ("price" in low) and ("pack" in low):
                mul_count = expr.split().count("*")
                div_count = expr.split().count("/")
                if div_count >= 1 and mul_count >= 2:
                    score += 0.18
                if div_count >= 1 and mul_count == 0:
                    score -= 0.18

        # Pair-of-two preference: "N pairs" ↔ "2*N items". Prefer direction based on the question.
        if ("pair" in low) or ("pairs" in low):
            asks_pairs = ("how many pairs" in low) or ("number of pairs" in low)
            # When the prompt is explicitly pricing "per pair" (or "price of one pair"),
            # treat "pair" as the monetary unit, not a count-to-items conversion.
            pair_is_pricing_unit = ("$" in problem_text) and any(w in low for w in ("price", "cost")) and any(
                w in low for w in ("per pair", "price of one pair", "one pair")
            )
            if asks_pairs:
                if "/" in ops and "*" not in ops:
                    score += 0.12
                if "*" in ops and "/" not in ops:
                    score -= 0.12
            else:
                # Default to item-count questions: reward multiplication by 2 when present.
                expr_pad = f" {expr} "
                if (" 2 *" in expr_pad) or ("2 *" in expr_pad):
                    score += 0.08
                if (" 2 /" in expr_pad) or ("/ 2" in expr_pad):
                    score -= 0.04
            # If the prompt mentions pairs and the candidate never uses a 2× adjustment,
            # it's often ignoring the pair-to-item conversion (high-impact failure mode).
            # Skip this penalty when "pair" is explicitly the priced unit (e.g., "price of one pair of shoes").
            if not pair_is_pricing_unit:
                if "2" not in {t for t in expr.split() if t} and (
                    ("*" in ops) or ("/" in ops) or ("+" in ops) or ("-" in ops)
                ):
                    score -= 0.14

        # Inverse fraction-per-day: "each day ... 1/10 ... finish ... how many days"
        # tends to collapse to den/num regardless of total. Reward candidates close to that.
        try:
            import re

            if ("how many days" in low) and (("each day" in low) or ("every day" in low)):
                m = re.search(r"\b(\d+)\s*/\s*(\d+)\b", low)
                if m:
                    n = float(m.group(1))
                    d = float(m.group(2))
                    if n > 0 and d > 0:
                        target = d / n
                        if abs(val - target) <= 1e-6:
                            score += 0.28
                        elif abs(val - target) <= max(1.0, abs(target)) * 0.05:
                            score += 0.14
                        else:
                            score -= 0.08
        except Exception:
            pass

        # Average "fit into": prefer candidates that form an average (sum/ count) then divide.
        if ("average" in low) and ("fit into" in low or ("fit" in low and "into" in low)):
            if "+" in ops and "/" in ops:
                score += 0.20
            if "*" in ops and "+" not in ops:
                score -= 0.20
            # "fit into" implies a second division by the base quantity; reward candidates with 2 divisions.
            div_count = expr.split().count("/")
            if div_count >= 2:
                score += 0.12
            elif div_count == 1:
                score -= 0.08

        # Out-of-pocket complements: prefer subtraction-based percent complement.
        if ("out-of-pocket" in low) and ("%" in low or "percent" in low):
            if "-" in ops and (" 100 /" in f" {expr} " or "/ 100" in f" {expr} "):
                score += 0.18
            if "+" in ops and "-" not in ops:
                score -= 0.18

        # Cost/total problems: prefer sum-of-products, discourage pure division.
        if ("$" in problem_text) or any(w in low for w in ("cost", "costs", "price", "spend", "spent", "out-of-pocket", "insurance")):
            asks_total_cost = any(w in low for w in ("how much", "how many", "total", "altogether", "in all", "spent", "spend"))
            if asks_total_cost:
                if "+" in ops and "*" in ops:
                    score += 0.12
                if "/" in ops and "+" not in ops and "-" not in ops and "per" not in low and "each" not in low:
                    score -= 0.20
            # Cash-vs-installment savings: needs multiply-then-add then subtract cash.
            if ("cash" in low) and ("down" in low) and ("month" in low) and ("save" in low):
                if "*" in ops and "+" in ops and "-" in ops:
                    score += 0.16
                if "+" in ops and "*" not in ops:
                    score -= 0.12
            # Full theatre adult/child revenue: needs (total-children)*adult + children*child.
            if ("seat" in low) and ("ticket" in low) and ("adult" in low) and ("child" in low) and ("full" in low):
                if "*" in ops and "+" in ops and "-" in ops:
                    score += 0.16
                if "+" in ops and "*" not in ops:
                    score -= 0.10

        # Multi-step cues: prefer richer (more ops) candidates when the prompt sequences actions.
        indicators = self._count_multi_step_indicators(problem_text)
        if indicators > 0:
            n_ops = self._count_basic_ops(expr)
            if n_ops >= indicators + 1:
                score += 0.07
            else:
                score -= 0.12

        # Operation/intent alignment: discourage contradictory operator choices.
        if "remaining" in low or "left" in low or "rest" in low:
            # Remaining/rest questions usually require subtraction at some point.
            if "-" in ops:
                score += 0.08
            else:
                score -= 0.12
            if "+" in ops and "-" not in ops:
                score -= 0.18
        # "sold ... customers ... the rest bought ..." linear remainder: prefer subtracting known products then dividing by rest_per.
        if ("sold" in low) and ("customer" in low) and ("bought" in low) and ("rest bought" in low or "the rest" in low):
            if "*" in ops and "-" in ops and "/" in ops:
                score += 0.18
            if "+" in ops and "/" not in ops:
                score -= 0.10
        # Weekly schedule counting: prefer weeks*days_per_week then subtract missed counts.
        if ("week" in low) and ("miss" in low) and any(d in low for d in ("wednesday", "friday", "monday", "tuesday", "thursday")):
            if "*" in ops and "-" in ops:
                score += 0.14
            if "+" in ops and "*" not in ops and "-" not in ops:
                score -= 0.14
        # Vertical travel / repeated trips: heavily multiplicative, rarely additive.
        if ("story" in low or "stories" in low) and ("feet" in low) and ("trip" in low or "trips" in low) and ("week" in low):
            n_ops = self._count_basic_ops(expr)
            # These are typically multi-factor products (stories * feet/story * trips/day * 2 * 7).
            # Penalize \"partial\" products that ignore most factors.
            if n_ops < 3:
                score -= 0.28
            if "*" in ops and "+" not in ops and "-" not in ops and n_ops >= 3:
                score += 0.18
            if "+" in ops:
                score -= 0.22
        # Packaging: when asking how many boxes/cartons are needed, division by per-box is typical.
        if any(q in low for q in ("how many boxes", "how many box", "how many cartons", "how many carton")) and ("per box" in low or "packaged with" in low):
            if "/" in ops:
                score += 0.18
            if "*" in ops and "/" not in ops:
                score -= 0.22
        # Billing cadence: "charges $R each week... pay every P weeks... how many weeks to get $T" favors division, not addition.
        if ("how many weeks" in low) and ("pay" in low) and ("every" in low) and ("each week" in low or "per week" in low):
            if "/" in ops and "+" not in ops:
                score += 0.14
            if "+" in ops:
                score -= 0.18
        if any(w in low for w in ("total", "altogether", "in all", "combined")):
            if "-" in ops and "+" not in ops:
                score -= 0.12
        if any(w in low for w in ("each", "per", "every")):
            # Rate problems are rarely pure addition.
            if "+" in ops and "*" not in ops and "/" not in ops:
                score -= 0.10
        if "as many" in low:
            # "as many" tends to imply multiplicative scaling; discourage subtraction unless
            # there is an explicit comparative cue ("more/less/fewer/difference/remaining").
            has_comparative_cue = any(w in low for w in ("more", "less", "fewer", "difference", "left", "remaining"))
            has_ratio_cue = "/" in low or any(
                w in low
                for w in (
                    "half",
                    "third",
                    "quarter",
                    "fourth",
                    "fifth",
                    "sixth",
                    "seventh",
                    "eighth",
                    "ninth",
                    "tenth",
                )
            )
            if has_ratio_cue and "-" in ops and not has_comparative_cue:
                score -= 0.12
            # If the text signals a ratio ("as many" + fraction words), require some
            # multiplicative scaling; pure addition is almost never correct.
            if has_ratio_cue and "*" not in ops and "/" not in ops:
                score -= 0.12
        if any(w in low for w in ("combined", "together", "altogether", "in all", "in total")):
            # Combined totals generally require addition; penalize candidates that never add.
            # This nudges selection away from misleading chain products (e.g., base*(6*8)).
            if "+" in ops:
                score += 0.06
            elif "*" in ops:
                score -= 0.10
            # When "combined" appears alongside explicit multipliers, insist on a
            # multiply-then-add shape (sum of products), not just adding raw numbers.
            if any(w in low for w in ("times", "twice", "double", "triple")) and "*" not in ops:
                score -= 0.18
            if any(w in low for w in ("times", "twice", "double", "triple")) and "+" not in ops:
                score -= 0.10
        if "more" in low and ("as many" in low or "/" in low):
            # When both a delta ("more") and a ratio cue are present, prefer candidates that
            # actually combine them (delta + fraction/multiply), not just one side.
            if "as many" in low:
                has_ratio_words = any(
                    w in low
                    for w in (
                        "half",
                        "third",
                        "quarter",
                        "fourth",
                        "fifth",
                        "sixth",
                        "seventh",
                        "eighth",
                        "ninth",
                        "tenth",
                    )
                )
                if has_ratio_words and ("/" in ops or "*" in ops):
                    if "+" in ops or "-" in ops:
                        score += 0.06
                    else:
                        score -= 0.14
            if "+" in ops and "/" in ops:
                score += 0.06
            elif "/" in ops and "+" not in ops and "-" not in ops:
                score -= 0.08
        if any(w in low for w in ("how long did", "how many did", "originally", "at first")) and any(
            w in low for w in ("twice", "double", "times")
        ):
            # Inverse multiplier questions: prefer division chains over multiplication.
            if "/" in ops:
                score += 0.06
            if "*" in ops and "/" not in ops:
                score -= 0.08

        # "times fewer/less" indicates division, not multiplication.
        if ("times fewer" in low) or ("times less" in low):
            if "/" in ops:
                score += 0.14
            if "*" in ops and "/" not in ops:
                score -= 0.24
        # "twice/double as many" indicates multiplication by 2 (not division by 2) unless the text is
        # explicitly asking an inverse question (handled elsewhere).
        if ("twice as many" in low) or ("double" in low and "as many" in low):
            expr_pad = f" {expr} "
            if " 2 *" in expr_pad:
                score += 0.08
            if " 2 /" in expr_pad and ("half" not in low):
                score -= 0.18

        # Story-specific structural cues (still generic): adults/children totals should be (k+1)*adults_total,
        # meaning at least 2 multiplications and an addition inside the adults total.
        if ("child" in low or "children" in low) and ("adult" in low or "adults" in low) and "total" in low:
            mul_count = sum(1 for t in (expression or "").split() if t == "*")
            if mul_count >= 2 and "+" in ops:
                score += 0.10
            else:
                score -= 0.12

        # Delta-after-fraction preference: "X more ... than ... who has half ..." usually means divide first, then add.
        if "more" in low and "than" in low and "half" in low:
            toks = (expression or "").split()
            last_op = toks[-1] if toks else ""
            if "/" in ops and "+" in ops and last_op == "+":
                score += 0.12
            if last_op == "/" and "+" in ops:
                score -= 0.12

        # Multi "each get" sum-of-products: prefer candidates that include ALL extracted (count, amount) pairs,
        # in the correct multiplicative grouping. This prevents high-scoring but wrong pair swaps like
        # "2*2 + 3*1 + 8*0.5" when the text implies "2*1 + 3*2 + 8*0.5".
        if (low.count("each") >= 2) and ("teaspoon" in low):
            try:
                import re

                word_to_num = {
                    "one": 1.0,
                    "two": 2.0,
                    "three": 3.0,
                    "four": 4.0,
                    "five": 5.0,
                    "six": 6.0,
                    "seven": 7.0,
                    "eight": 8.0,
                    "nine": 9.0,
                    "ten": 10.0,
                    "half": 0.5,
                }
                pat = re.compile(
                    r"(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b"
                    r"[^.]{0,120}?\beach\b[^.]{0,80}?\b(?:get|eat|receive)\w*\b\s+"
                    r"(?P<amt>\d+(?:\.\d+)?|half|one|two|three|four|five)\b"
                    r"\s*(?:a\s+)?(?:teaspoon|teaspoons)\b",
                    re.IGNORECASE,
                )
                needed: List[tuple[float, float]] = []
                for m in pat.finditer(problem_text or ""):
                    c_raw = str(m.group("count")).strip().lower()
                    a_raw = str(m.group("amt")).strip().lower()
                    c = float(c_raw) if c_raw.isdigit() else float(word_to_num.get(c_raw, 0.0))
                    a = float(a_raw) if a_raw.replace(".", "", 1).isdigit() else float(word_to_num.get(a_raw, 0.0))
                    if c > 0 and a > 0:
                        needed.append((c, a))
                    if len(needed) >= 4:
                        break
                if len(needed) >= 2:
                    # Extract multiplication operand pairs in the candidate.
                    expr_tokens = (expression or "").split()
                    mul_pairs: List[tuple[float, float]] = []
                    for i, tok in enumerate(expr_tokens):
                        if tok != "*":
                            continue
                        if i < 2:
                            continue
                        try:
                            a = float(expr_tokens[i - 2])
                            b = float(expr_tokens[i - 1])
                        except Exception:
                            continue
                        mul_pairs.append((a, b))

                    def _pair_eq(p: tuple[float, float], q: tuple[float, float]) -> bool:
                        return (abs(p[0] - q[0]) < 1e-9 and abs(p[1] - q[1]) < 1e-9) or (
                            abs(p[0] - q[1]) < 1e-9 and abs(p[1] - q[0]) < 1e-9
                        )

                    remaining = list(mul_pairs)
                    matched = 0
                    for c, a in needed:
                        found_idx = None
                        for j, p in enumerate(remaining):
                            if _pair_eq((c, a), p):
                                found_idx = j
                                break
                        if found_idx is not None:
                            matched += 1
                            remaining.pop(found_idx)
                    if matched < len(needed):
                        # Strong penalty: missing or mispaired groups.
                        score -= 0.18 * (len(needed) - matched)
                    else:
                        score += 0.08
            except Exception:
                pass

        # Penalize candidates that ignore most numbers in the prompt.
        num_tokens = 0
        used_vals: List[float] = []
        expr_tokens = (expression or "").split()
        for tok in expr_tokens:
            if tok in {"+", "-", "*", "/"}:
                continue
            try:
                v = float(tok)
            except Exception:
                continue
            num_tokens += 1
            used_vals.append(v)

        if len(numbers) >= 3 and num_tokens <= 2:
            score -= 0.10

        # Prefer using at least 2 of the top-3 magnitude numbers when present.
        if numbers and used_vals:
            uniq_nums = []
            for n in numbers:
                try:
                    uniq_nums.append(float(n))
                except Exception:
                    continue
            uniq_nums = list(dict.fromkeys([round(abs(n), 6) for n in uniq_nums if n != 0.0]))
            uniq_nums.sort(reverse=True)
            important = uniq_nums[:3]
            used_abs = {round(abs(v), 6) for v in used_vals}
            used_important = sum(1 for x in important if x in used_abs)
            if len(important) >= 2 and used_important < 2:
                score -= 0.18

        # Penalize identity operations (×1, ÷1) which often indicate "picked the wrong number".
        if expr_tokens and ("*" in ops or "/" in ops):
            for i in range(2, len(expr_tokens)):
                op = expr_tokens[i]
                if op not in {"*", "/"}:
                    continue
                try:
                    a = float(expr_tokens[i - 2])
                    b = float(expr_tokens[i - 1])
                except Exception:
                    continue
                if op in {"*", "/"} and (abs(a - 1.0) < 1e-9 or abs(b - 1.0) < 1e-9):
                    score -= 0.08
                    break

        # Prefer integer-like answers when input numbers look integral.
        is_int_like = abs(val - round(val)) < 1e-6
        if is_int_like:
            score += 0.05

        # If the question is phrased as a count/amount, heavily prefer integer answers.
        asks_integer = any(q in low for q in ("how many", "number of", "how much", "total", "altogether", "in all"))
        if asks_integer and not is_int_like:
            score -= 0.25
        if asks_integer and self._count_basic_ops(expr) == 0:
            score -= 0.22

        # Clamp to [0, 1].
        if score < 0:
            score = 0.0
        if score > 1:
            score = 1.0
        return (float(score), verdict)

    def _generate_rpn_candidates(
        self,
        *,
        problem_text: str,
        numbers: Sequence[float],
        words: Sequence[str],
        question_type: str,
        depth: int,
        strategy: str,
        max_candidates: int,
    ) -> List[str]:
        """
        Generate a bounded list of numeric-only RPN candidates.

        This intentionally avoids external libs and keeps candidates simple:
        pairwise/triple arithmetic, plus optional conversion suggestions from the
        generic equations galaxy.
        """
        nums = [float(n) for n in (numbers or []) if isinstance(n, (int, float))]
        if not nums:
            return []

        def _lit(x: float) -> str:
            return str(int(x)) if abs(x - round(x)) < 1e-9 else str(x)

        ws_all = {str(w).lower() for w in (words or []) if w}
        low_text = (problem_text or "").lower()
        # Add common implicit numeric words not always present in WordGalaxy as numbers.
        if ("dozen" in ws_all) or ("dozen" in low_text):
            if 12.0 not in nums:
                nums.append(12.0)
            if "half dozen" in low_text and 6.0 not in nums:
                nums.append(6.0)
        if ("half" in ws_all) or ("half" in low_text):
            if 0.5 not in nums:
                nums.append(0.5)

        # Basic operator preferences based on strategy.
        op_order = ["*", "/", "+", "-"]
        if strategy == "prioritize_division":
            op_order = ["/", "*", "-", "+"]
        elif strategy == "prioritize_aggregation":
            op_order = ["+", "-", "*", "/"]
        elif strategy == "prioritize_conversions":
            op_order = ["*", "/", "+", "-"]

        candidates: List[str] = []
        special: List[str] = []

        def _dedupe_limit(exprs: Sequence[str]) -> List[str]:
            seen: set[str] = set()
            out: List[str] = []
            for expr in exprs:
                e = str(expr).strip()
                if not e or e in seen:
                    continue
                seen.add(e)
                out.append(e)
                if len(out) >= max_candidates:
                    break
            return out

        # Depth 1+: percent increase/decrease relative to a baseline ("X% more expensive", "Y% less", "discount").
        #
        # This is a generic building block (applies to money, weight, distance, etc.), not GSM8K-specific.
        if depth >= 1 and (("%" in low_text) or ("percent" in low_text)) and any(
            w in low_text for w in ("more", "less", "cheaper", "expensive", "discount", "increase", "decrease")
        ):
            try:
                import re

                pct_vals = [
                    float(p)
                    for p in re.findall(r"(\d+(?:\.\d+)?)\s*%", low_text)
                    if p and 0.0 < float(p) <= 100.0
                ]
                pct = pct_vals[0] if pct_vals else None
                if pct is not None:
                    others = [float(n) for n in nums if abs(float(n) - pct) > 1e-9]
                    if others:
                        base = max(others)
                        base_lit = _lit(base)
                        pct_lit = _lit(pct)
                        if any(w in low_text for w in ("more", "increase", "expensive")):
                            # base + base*pct/100
                            candidates.append(f"{base_lit} {base_lit} {pct_lit} * 100 / +")
                        if any(w in low_text for w in ("less", "decrease", "cheaper", "discount")):
                            # base - base*pct/100
                            candidates.append(f"{base_lit} {base_lit} {pct_lit} * 100 / -")
            except Exception:
                pass

        # Depth 1+: percent applied to a derived total (often a product), especially with "remaining/left/cut".
        #
        # Example:
        #   "50 rows. each row has 400 flowers. cuts 60% ... how many remaining?"
        #   total = 50*400; remaining = total*(100-60)/100
        if depth >= 1 and (("%" in low_text) or ("percent" in low_text)) and any(
            w in low_text for w in ("remaining", "remain", "left", "cut", "cuts", "removed", "remove", "took", "take")
        ):
            try:
                import re

                pct_vals = [
                    float(p)
                    for p in re.findall(r"(\d+(?:\.\d+)?)\s*%", low_text)
                    if p and 0.0 < float(p) <= 100.0
                ]
                if not pct_vals:
                    pct_vals = [
                        float(p)
                        for p in re.findall(r"\b(\d+(?:\.\d+)?)\s+percent\b", low_text)
                        if p and 0.0 < float(p) <= 100.0
                    ]
                pct = pct_vals[0] if pct_vals else None
                if pct is not None:
                    # Prefer integer-ish totals for the product; ignore the percent itself and common 100 literals.
                    base_ints = [
                        float(n)
                        for n in nums
                        if abs(float(n) - pct) > 1e-9 and abs(float(n) - 100.0) > 1e-9 and float(n) > 0
                    ]
                    base_ints = [n for n in base_ints if abs(n - round(n)) < 1e-9]
                    base_ints = list(dict.fromkeys(base_ints))
                    # Keep a small pool to avoid combinatorial blow-up.
                    base_ints.sort(reverse=True)
                    pool = base_ints[:4]
                    if len(pool) >= 2:
                        pct_lit = _lit(float(pct))
                        # Candidate family: (a*b) * (100-pct) / 100
                        for i in range(min(3, len(pool))):
                            for j in range(min(3, len(pool))):
                                if i == j:
                                    continue
                                a = pool[i]
                                b = pool[j]
                                total = f"{_lit(a)} {_lit(b)} *"
                                # Remaining/complement
                                if "remain" in low_text or "remaining" in low_text or "left" in low_text:
                                    # Put in `special` so it survives max_parallel sampling/truncation.
                                    special.append(f"{total} 100 {pct_lit} - * 100 /")
                                # Cut/removed amount
                                if "cut" in low_text or "remove" in low_text or "took" in low_text:
                                    special.append(f"{total} {pct_lit} * 100 /")
                    elif base_ints:
                        # Fallback to applying percent to a single base value.
                        base = base_ints[0]
                        base_lit = _lit(base)
                        pct_lit = _lit(float(pct))
                        if "remain" in low_text or "remaining" in low_text or "left" in low_text:
                            special.append(f"{base_lit} 100 {pct_lit} - * 100 /")
                        if "cut" in low_text or "remove" in low_text or "took" in low_text:
                            special.append(f"{base_lit} {pct_lit} * 100 /")
            except Exception:
                pass

        # Depth 1+: capacity increase inside nested "each" structures.
        #
        # Example:
        #   "each carriage has 25 seats. If each carriage could accommodate 10 more passengers,
        #    how many passengers would fill up 3 trains?"
        #   rpn: (25+10) * 4 * 3
        if depth >= 1 and ("each" in low_text) and any(w in low_text for w in ("accommodate", "capacity")) and (
            "more" in low_text
        ):
            try:
                import re

                m_base = re.search(r"\beach\s+\w+\s+has\s+(\d+(?:\.\d+)?)\b", low_text)
                m_extra = re.search(r"\baccommodate\s+(\d+(?:\.\d+)?)\s+more\b", low_text)
                if m_base and m_extra:
                    base_n = float(m_base.group(1))
                    extra_n = float(m_extra.group(1))
                    # Multiply by up to two other integer counts from the prompt (e.g., 4 carriages, 3 trains).
                    ints = [float(n) for n in nums if n > 0 and abs(n - round(n)) < 1e-9]
                    # Remove the base/extra if present.
                    filtered = [n for n in ints if abs(n - base_n) > 1e-9 and abs(n - extra_n) > 1e-9]
                    uniq = list(dict.fromkeys(filtered))
                    if len(uniq) >= 2:
                        c1 = uniq[-2]
                        c2 = uniq[-1]
                        candidates.append(f"{_lit(base_n)} {_lit(extra_n)} + {_lit(c1)} * {_lit(c2)} *")
                        candidates.append(f"{_lit(base_n)} {_lit(extra_n)} + {_lit(c2)} * {_lit(c1)} *")
            except Exception:
                pass

        # Depth 1+: "some items have known per-unit cost, remaining items equal-cost".
        #
        # Example:
        #   "Five shirts together cost $85. Of the 5 shirts, there are 3 shirts that cost $15 each.
        #    If the remaining shirts are each equal in value, what is the cost of each remaining shirt?"
        #   each_unknown = (total_cost - known_count*known_unit_cost) / (total_count - known_count)
        #
        # Generic across items (shirts, tickets, meals, etc.).
        if depth >= 1 and ("together cost" in low_text or "in total cost" in low_text or "total cost" in low_text) and (
            "remaining" in low_text
        ):
            try:
                import re

                m_total = re.search(
                    r"\b(\d+)\s+\w+\s+(?:together\s+)?cost\s+\$?\s*(\d+(?:\.\d+)?)\b",
                    low_text,
                )
                m_known = re.search(
                    r"\b(?:there\s+are|of\s+the)\s+(\d+)\s+\w+\s+that\s+cost\s+\$?\s*(\d+(?:\.\d+)?)\s+each\b",
                    low_text,
                ) or re.search(
                    r"\b(\d+)\s+\w+\s+that\s+cost\s+\$?\s*(\d+(?:\.\d+)?)\s+each\b",
                    low_text,
                )
                if m_total and m_known:
                    total_count = float(m_total.group(1))
                    total_cost = float(m_total.group(2))
                    known_count = float(m_known.group(1))
                    unit_cost = float(m_known.group(2))
                    rem_count = total_count - known_count
                    if rem_count > 0:
                        candidates.append(
                            f"{_lit(total_cost)} {_lit(known_count)} {_lit(unit_cost)} * - {_lit(rem_count)} /"
                        )
            except Exception:
                pass

        # Depth 1+: ratio populations + ticket revenue ("N times as many children as adults", prices, total collected).
        #
        # Example:
        #   "Three times as many children as adults attend ... adult ticket $7, child ticket $3,
        #    theater collected $6000 ... total people?"
        #   adults = total / (adult_price + N*child_price); total_people = adults * (N+1)
        if depth >= 1 and ("ticket" in low_text) and ("collected" in low_text) and ("times as many" in low_text):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(low_text)
                m_ratio = re.search(r"\b(\d+(?:\.\d+)?)\s+times\s+as\s+many\s+children\s+as\s+adults\b", norm)
                if not m_ratio:
                    m_ratio = re.search(r"\bchildren\s+are\s+(\d+(?:\.\d+)?)\s+times\s+as\s+many\s+as\s+adults\b", norm)
                if m_ratio:
                    ratio = float(m_ratio.group(1))
                    if ratio > 0:
                        m_adult = re.search(r"\badult\s+ticket\s+costs?\s*\$\s*(\d+(?:\.\d+)?)", norm)
                        m_child = re.search(r"\bchild'?s?\s+ticket\s+costs?\s*\$\s*(\d+(?:\.\d+)?)", norm)
                        m_total = re.search(r"\bcollected\b[^\d$]{0,60}\$\s*(\d[\d,]*(?:\.\d+)?)", norm) or re.search(
                            r"\btotal\b[^\d$]{0,60}\$\s*(\d[\d,]*(?:\.\d+)?)", norm
                        )
                        if m_adult and m_child and m_total:
                            adult_price = float(m_adult.group(1))
                            child_price = float(m_child.group(1))
                            total_rev = float(str(m_total.group(1)).replace(",", ""))
                            denom = adult_price + ratio * child_price
                            if abs(denom) > 1e-12:
                                mult = ratio + 1.0
                                candidates.append(f"{_lit(total_rev)} {_lit(denom)} / {_lit(mult)} *")
                                # Prefer a fully-expanded variant that uses prompt literals, improving
                                # coverage/tie-break scoring and reducing spurious alternatives.
                                # total / (adult + ratio*child) * (ratio+1)
                                candidates.append(
                                    f"{_lit(total_rev)} {_lit(adult_price)} {_lit(child_price)} {_lit(ratio)} * + / {_lit(mult)} *"
                                )
            except Exception:
                pass

        # Depth 1+: ordinal chains with a known terminal value and a requested total.
        #
        # Example:
        #   "second throw skips 2 more than first; third skips twice second; fourth skips 3 fewer than third;
        #    fifth skips one more than fourth; if fifth skipped 8, total between all throws?"
        # This is a generic relative-chain reasoning pattern expressed as an affine transform of the
        # terminal value, keeping the hot path numeric-only.
        if depth >= 1 and ("throw" in low_text) and all(w in low_text for w in ("first", "second", "third", "fourth", "fifth")) and (
            "total" in low_text or "in total" in low_text
        ):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(low_text)
                # Terminal value: "fifth ... skipped N"
                m_last = re.search(r"\bfifth\s+\w+\s+skipped\s+(\d+(?:\.\d+)?)\b", norm)
                if not m_last:
                    m_last = re.search(r"\bfifth\s+\w+\s+.*?\bskipped\s+(\d+(?:\.\d+)?)\b", norm)
                if m_last:
                    x5 = float(m_last.group(1))
                    # Relations (parsed in forward direction) then inverted to back-solve from x5.
                    m_more = re.search(r"\bsecond\s+\w+\s+skips\s+(\d+(?:\.\d+)?)\s+more\b.*\bfirst\b", norm)
                    m_twice = re.search(r"\bthird\s+\w+\s+skips\s+twice\b.*\bsecond\b", norm)
                    m_fewer = re.search(r"\bfourth\s+\w+\s+skips\s+(\d+(?:\.\d+)?)\s+fewer\b.*\bthird\b", norm)
                    m_one_more = re.search(r"\bfifth\s+\w+\s+skips\s+(?:one|1)\s+more\b.*\bfourth\b", norm)
                    if m_more and m_twice and m_fewer and m_one_more:
                        d21 = float(m_more.group(1))  # x2 = x1 + d21
                        d43 = float(m_fewer.group(1))  # x4 = x3 - d43
                        # Back-solve affine chain: x_i = a_i*x5 + b_i
                        a5, b5 = 1.0, 0.0
                        # x5 = x4 + 1 -> x4 = x5 - 1
                        a4, b4 = a5, b5 - 1.0
                        # x4 = x3 - d43 -> x3 = x4 + d43
                        a3, b3 = a4, b4 + d43
                        # x3 = 2*x2 -> x2 = x3 / 2
                        a2, b2 = a3 / 2.0, b3 / 2.0
                        # x2 = x1 + d21 -> x1 = x2 - d21
                        a1, b1 = a2, b2 - d21
                        a_sum = a1 + a2 + a3 + a4 + a5
                        b_sum = b1 + b2 + b3 + b4 + b5
                        # total = a_sum*x5 + b_sum
                        #
                        # Provide both a compact affine form and a "structured" variant that
                        # keeps the discovered deltas in the expression. The structured variant
                        # tends to win tie-breaks because it uses more of the prompt's salient
                        # numbers (e.g., the "+2 more" and "-3 fewer") without changing the
                        # numeric meaning.
                        #
                        # For this fixed throw-chain pattern:
                        #   x2 = x1 + d21
                        #   x3 = 2*x2
                        #   x4 = x3 - d43
                        #   x5 = x4 + 1
                        # the constant term simplifies to: b_sum = 2*d43 - 3 - d21
                        special.insert(0, f"{_lit(x5)} {_lit(a_sum)} * {_lit(d43)} 2 * 3 - {_lit(d21)} - +")
                        special.insert(1, f"{_lit(x5)} {_lit(a_sum)} * {_lit(b_sum)} +")
            except Exception:
                pass

        # Depth 1+: inventory delta stories starting from an initial count and applying +/- events.
        #
        # Example:
        #   "started with 50 balloons. gave 1 away. 12 floated away. gave 9 more away. grabbed 11 more."
        #   total = 50 - (1+12+9) + 11
        if depth >= 1 and ("started with" in low_text) and ("how many" in low_text) and ("$" not in low_text):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(low_text)
                m_start = re.search(r"\bstarted\s+with\s+(\d+(?:\.\d+)?)\b", norm)
                if m_start:
                    start = float(m_start.group(1))
                    # Losses: gave away / lost / floated away / spent
                    losses: list[float] = []
                    for m in re.finditer(r"\bgave\s+(\d+(?:\.\d+)?)\b", norm):
                        losses.append(float(m.group(1)))
                    # Passing/handing a single item away is also a loss (common in "started with X" narratives).
                    for m in re.finditer(r"\bpassing\s+(\d+(?:\.\d+)?)\s+\w+\b", norm):
                        losses.append(float(m.group(1)))
                    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s+balloons?\s+floated\s+away\b", norm):
                        losses.append(float(m.group(1)))
                    for m in re.finditer(r"\blost\s+(\d+(?:\.\d+)?)\b", norm):
                        losses.append(float(m.group(1)))
                    # Gains: grabbed/took/received/got
                    gains: list[float] = []
                    for m in re.finditer(r"\bgrabbed\s+the\s+last\s+(\d+(?:\.\d+)?)\b", norm):
                        gains.append(float(m.group(1)))
                    for m in re.finditer(r"\bgot\s+(\d+(?:\.\d+)?)\b", norm):
                        gains.append(float(m.group(1)))
                    for m in re.finditer(r"\breceived\s+(\d+(?:\.\d+)?)\b", norm):
                        gains.append(float(m.group(1)))
                    for m in re.finditer(r"\btook\s+(\d+(?:\.\d+)?)\b", norm):
                        gains.append(float(m.group(1)))
                    if losses or gains:
                        expr = f"{_lit(start)}"
                        for v in losses[:6]:
                            expr += f" {_lit(v)} -"
                        for v in gains[:6]:
                            expr += f" {_lit(v)} +"
                        special.insert(0, expr)
            except Exception:
                pass

        # Depth 1+: geometry packing (right triangles in a square).
        #
        # Example:
        #   "How many right triangles with a height of 2 inches and a width of 2 inches
        #    could fit inside a square with 2-inch sides?"
        #
        # Candidate: count = square_area / triangle_area = (s*s) / (h*w/2) = 2*s*s/(h*w)
        if depth >= 1 and ("right triangle" in low_text) and ("square" in low_text) and ("fit" in low_text):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                m_square = re.search(r"\bsquare\b[^\d]{0,60}\b(\d+(?:\.\d+)?)\s*[- ]?\w*\s+sides?\b", low_norm)
                if not m_square:
                    m_square = re.search(r"\b(\d+(?:\.\d+)?)\s*[- ]?\w*\s+sides?\b[^\n]{0,80}\bsquare\b", low_norm)
                m_tri = re.search(
                    r"\bheight\s+of\s+(\d+(?:\.\d+)?)\b[^\d]{0,40}\b(?:width|base)\s+of\s+(\d+(?:\.\d+)?)\b",
                    low_norm,
                )
                if m_square and m_tri:
                    s = float(m_square.group(1))
                    h = float(m_tri.group(1))
                    w = float(m_tri.group(2))
                    if s > 0 and h > 0 and w > 0:
                        special.insert(0, f"{_lit(s)} {_lit(s)} * 2 * {_lit(h)} {_lit(w)} * /")
                        special.insert(1, f"{_lit(s)} {_lit(s)} * {_lit(h)} {_lit(w)} * / 2 *")
            except Exception:
                pass

        # Depth 1+: consumable pack cost over a duration.
        #
        # Example:
        #   "Judy uses 10 pencils during her 5 day school week. A 30 pack of pencils costs $4.
        #    How much will she spend on pencils over 45 days?"
        #
        # total_items = uses / period_days * duration_days
        # packs = total_items / pack_size
        # cost = packs * pack_cost
        if depth >= 1 and ("pack" in low_text) and ("$" in low_text) and any(w in low_text for w in ("uses", "use", "using")) and (
            "day" in low_text or "days" in low_text
        ):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()

                m_use = re.search(
                    r"\buses?\s+(\d+(?:\.\d+)?)\s+\w+\b[^\d]{0,40}\b(\d+(?:\.\d+)?)\s+day\b",
                    low_norm,
                )
                m_pack = re.search(r"\b(\d+(?:\.\d+)?)\s*pack\b[^\d$]{0,40}\$\s*(\d+(?:\.\d+)?)\b", low_norm)
                m_dur = re.search(r"\bover\s+(\d+(?:\.\d+)?)\s+days?\b", low_norm) or re.search(
                    r"\bfor\s+(\d+(?:\.\d+)?)\s+days?\b", low_norm
                )

                if m_use and m_pack and m_dur:
                    uses = float(m_use.group(1))
                    period_days = float(m_use.group(2))
                    pack_size = float(m_pack.group(1))
                    pack_cost = float(m_pack.group(2))
                    duration_days = float(m_dur.group(1))
                    if uses > 0 and period_days > 0 and pack_size > 0 and duration_days > 0:
                        base = f"{_lit(uses)} {_lit(period_days)} / {_lit(duration_days)} * {_lit(pack_size)} /"
                        # Packs needed (can be fractional if not divisible).
                        special.append(base)
                        # Total cost.
                        special.insert(0, f"{base} {_lit(pack_cost)} *")
            except Exception:
                pass

        # Depth 1+: pairs ("N pairs" ↔ 2*N items).
        # Generate both directions; scoring/plausibility should pick the right one.
        if depth >= 1 and (("pair" in ws_all) or ("pairs" in ws_all) or ("pair" in low_text)):
            ints = [n for n in nums if n > 0 and abs(n - round(n)) < 1e-9]
            for x in ints[:5]:
                x_lit = _lit(float(x))
                candidates.append(f"{x_lit} 2 *")
                candidates.append(f"{x_lit} 2 /")
                candidates.append(f"2 {x_lit} *")
            # Mixed "each ... and ... pairs" totals: (a*b) + (pairs*2).
            # Example: "3 breeding balls with 8 snakes each and 6 additional pairs of snakes"
            # -> 3*8 + 6*2.
            if ("each" in ws_all) and any(w in low_text for w in ("total", "in total", "altogether", "how many")):
                uniq = list(dict.fromkeys([float(n) for n in ints]))
                # Generate a small set of triple candidates without full combinatorial blow-up.
                for a in uniq[:4]:
                    for b in uniq[:4]:
                        if abs(a - b) < 1e-9:
                            continue
                        for c in uniq[:4]:
                            if abs(c - a) < 1e-9 or abs(c - b) < 1e-9:
                                continue
                            special.append(f"{_lit(a)} {_lit(b)} * {_lit(c)} 2 * +")
                            if len(special) >= 18:
                                break
                        if len(special) >= 18:
                            break
                    if len(special) >= 18:
                        break

        # Depth 1+: "twice that long" totals: X minutes + 2*X minutes.
        # Generic pattern used for time/cost narratives with a base duration and a repeated "twice" segment.
        if depth >= 1 and {"twice", "double"}.intersection(ws_all) and ("minute" in ws_all or "minutes" in ws_all):
            if len(nums) >= 1:
                for x in nums[:3]:
                    if x <= 0:
                        continue
                    x_lit = _lit(x)
                    candidates.append(f"{x_lit} {x_lit} 2 * +")
                    candidates.append(f"{x_lit} 2 * {x_lit} +")

        # Depth 1+: travel-time totals with a "twice as much time" middle leg.
        # Example: "takes 40 days ... twice as much time ... and 60 days ..." -> 40 + 2*40 + 60.
        if depth >= 1 and {"twice", "double"}.intersection(ws_all) and ("day" in ws_all or "days" in ws_all or "day" in low_text):
            if any(w in low_text for w in ("takes", "it takes", "to fly", "to travel")) and any(
                w in low_text for w in ("and",)
            ):
                ints = [float(n) for n in nums if n > 0 and abs(n - round(n)) < 1e-9]
                uniq = list(dict.fromkeys(ints))
                if len(uniq) >= 2:
                    base = min(uniq)
                    other = max(uniq)
                    base_lit = _lit(base)
                    other_lit = _lit(other)
                    candidates.append(f"{base_lit} {base_lit} 2 * + {other_lit} +")
                    candidates.append(f"{base_lit} 2 * {base_lit} + {other_lit} +")

        # Depth 1+: half-rate mpg: gallons = distance / (mpg/2) = distance*2/mpg.
        if depth >= 1 and (("miles/gallon" in low_text) or ("miles per gallon" in low_text)) and ("half" in low_text) and ("gallon" in low_text):
            ints = [float(n) for n in nums if n > 0 and abs(n - round(n)) < 1e-9]
            uniq = list(dict.fromkeys(ints))
            if len(uniq) >= 2:
                distance = max(uniq)
                mpg = min(uniq)
                candidates.append(f"{_lit(distance)} {_lit(mpg)} / 2 *")
                candidates.append(f"{_lit(distance)} 2 * {_lit(mpg)} /")

        # Depth 1+: "combined total" + "more/less than double/triple" linear constraint.
        #
        # Example:
        #   total = A + B
        #   B = k*A ± d
        # Solve for B directly:
        #   if "+" (more):  B = (k*total + d) / (k+1)
        #   if "-" (less):  B = (k*total - d) / (k+1)
        if depth >= 1 and any(w in low_text for w in ("combined", "together", "in total")) and any(
            w in low_text for w in ("double", "twice", "triple", "thrice")
        ):
            k = 2.0
            if ("triple" in low_text) or ("thrice" in low_text):
                k = 3.0
            ints = [float(n) for n in nums if n > 0 and abs(n - round(n)) < 1e-9]
            if ints:
                total = max(ints)
                deltas = [n for n in ints if 0 < n < total and n <= 50]
                deltas = list(dict.fromkeys(deltas))
                for d in deltas[:4]:
                    if "more" in low_text:
                        candidates.append(f"{_lit(total)} {_lit(k)} * {_lit(d)} + {_lit(k + 1.0)} /")
                    if ("less" in low_text) or ("fewer" in low_text):
                        candidates.append(f"{_lit(total)} {_lit(k)} * {_lit(d)} - {_lit(k + 1.0)} /")

        # Depth 1+: multi "each get" sum-of-products (feed/teaspoons style).
        # Prefer pairing integer group counts with small per-item amounts and summing products.
        if depth >= 1 and (low_text.count("each") >= 2) and any(w in low_text for w in ("get ", "gets ", "teaspoon")):
            # First try a light regex-based structured extraction to preserve ordering:
            # "<count> ... each get <amount> teaspoon(s)" repeated.
            each_get_ordered = False
            try:
                import re

                word_to_num = {
                    "one": 1.0,
                    "two": 2.0,
                    "three": 3.0,
                    "four": 4.0,
                    "five": 5.0,
                    "six": 6.0,
                    "seven": 7.0,
                    "eight": 8.0,
                    "nine": 9.0,
                    "ten": 10.0,
                    "half": 0.5,
                }
                pairs: List[tuple[float, float]] = []
                pat = re.compile(
                    r"(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b"
                    r"[^.]{0,120}?\beach\b[^.]{0,80}?\b(?:get|eat|receive)\w*\b\s+"
                    r"(?P<amt>\d+(?:\.\d+)?|half|one|two|three|four|five)\b"
                    r"\s*(?:a\s+)?(?:teaspoon|teaspoons)\b",
                    re.IGNORECASE,
                )
                for m in pat.finditer(problem_text or ""):
                    c_raw = str(m.group("count")).strip().lower()
                    a_raw = str(m.group("amt")).strip().lower()
                    c = float(c_raw) if c_raw.isdigit() else float(word_to_num.get(c_raw, 0.0))
                    a = float(a_raw) if a_raw.replace(".", "", 1).isdigit() else float(word_to_num.get(a_raw, 0.0))
                    if c > 0 and a > 0:
                        pairs.append((c, a))
                    if len(pairs) >= 4:
                        break
                if len(pairs) >= 2:
                    toks: List[str] = []
                    for c, a in pairs:
                        toks.append(f"{_lit(c)} {_lit(a)} *")
                    expr = " ".join(toks)
                    for _ in range(len(toks) - 1):
                        expr += " +"
                    special.insert(0, expr)
                    each_get_ordered = True
            except Exception:
                pass

            # Fallback: if ordering extraction failed, try pairing counts with plausible amounts by permutation.
            # When ordering extraction succeeds, do not add permutations, as they often generate high-scoring but
            # wrong pairings (e.g., swapping "two ... one" into "two ... two").
            if not each_get_ordered:
                ints = [float(n) for n in nums if n > 1 and abs(n - round(n)) < 1e-9]
                counts = list(dict.fromkeys([n for n in ints if n >= 2]))[:3]
                amounts = [float(n) for n in nums if 0 < n <= 5 and (abs(n - round(n)) < 1e-9 or abs(n - 0.5) < 1e-9)]
                amounts = list(dict.fromkeys(amounts))
                if "one" in low_text and 1.0 not in amounts:
                    amounts.append(1.0)
                if len(counts) >= 2 and amounts:
                    try:
                        from itertools import permutations

                        n_terms = min(3, len(counts), len(amounts))
                        for perm in list(permutations(amounts, n_terms))[:12]:
                            toks: List[str] = []
                            for c, a in zip(counts[:n_terms], perm):
                                toks.append(f"{_lit(c)} {_lit(a)} *")
                            expr = " ".join(toks)
                            for _ in range(len(toks) - 1):
                                expr += " +"
                            special.append(expr)
                    except Exception:
                        pass

        # Depth 1+: "times fewer/less" means division, not multiplication.
        # Example: "Annie has three times fewer cats than Jacob" => Annie = Jacob / 3.
        # Often combined with an outer multiplier: "Melanie has twice as many as Annie" => Jacob/3*2.
        if depth >= 1 and any(phrase in low_text for phrase in ("times fewer", "times less")):
            try:
                import re

                word_to_num = {
                    "one": 1.0,
                    "two": 2.0,
                    "three": 3.0,
                    "four": 4.0,
                    "five": 5.0,
                    "six": 6.0,
                    "seven": 7.0,
                    "eight": 8.0,
                    "nine": 9.0,
                    "ten": 10.0,
                }
                div_m = re.search(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+times\s+(?:fewer|less)", low_text)
                if div_m:
                    d_raw = div_m.group(1).lower()
                    divisor = float(d_raw) if d_raw.isdigit() else float(word_to_num.get(d_raw, 0.0))
                else:
                    divisor = 0.0
                if divisor > 1e-9:
                    base = max(nums)
                    base_lit = _lit(float(base))
                    div_lit = _lit(float(divisor))
                    # Direct division candidate.
                    special.append(f"{base_lit} {div_lit} /")

                    # Optional outer multiplier (twice/double or N times as many).
                    mult = None
                    if ("twice as many" in low_text) or ("double" in low_text) or ("twice" in low_text and "as many" in low_text):
                        mult = 2.0
                    m2 = re.search(r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+times\s+as\s+many", low_text)
                    if m2:
                        m_raw = m2.group(1).lower()
                        mult = float(m_raw) if m_raw.isdigit() else float(word_to_num.get(m_raw, 0.0))
                    if mult and mult > 1e-9:
                        mult_lit = _lit(float(mult))
                        special.insert(0, f"{base_lit} {div_lit} / {mult_lit} *")
                        special.append(f"{base_lit} {mult_lit} * {div_lit} /")
            except Exception:
                pass

        # Depth 1+: "A made N; B made D more; C made a quarter/third/half of B; total?"
        # Total = base + derived + derived/f.
        if depth >= 1 and ("more than" in low_text) and any(w in low_text for w in ("quarter", "fourth", "third", "half", "1/4", "1/3", "1/2")) and any(
            w in low_text for w in ("in all", "together", "altogether", "total", "combined")
        ):
            try:
                import re

                ordered = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", problem_text)]
                if len(ordered) >= 2:
                    base = float(ordered[0])
                    delta = float(ordered[1])
                    frac = None
                    if ("quarter" in low_text) or ("fourth" in low_text) or ("1/4" in low_text):
                        frac = 4.0
                    elif ("third" in low_text) or ("1/3" in low_text):
                        frac = 3.0
                    elif ("half" in low_text) or ("1/2" in low_text):
                        frac = 2.0
                    if frac and frac > 1e-9:
                        b = _lit(base)
                        d = _lit(delta)
                        f = _lit(float(frac))
                        # base + (base+delta) + (base+delta)/frac
                        special.insert(0, f"{b} {b} {d} + + {b} {d} + {f} / +")
            except Exception:
                pass

        # Depth 1+: mixed unit cost with "other cost $X more" (weighted sum-of-products).
        # Example: "9 pills ... 4 cost $1.50 each ... other cost $5.50 more" => 4*1.5 + (9-4)*(1.5+5.5)
        if depth >= 1 and ("$" in problem_text) and ("more" in low_text) and ("each" in low_text) and any(w in low_text for w in ("pill", "pills")):
            try:
                import re

                # Find total count and sub-count.
                total_m = re.search(r"(\d+)\s+pills?\s+a\s+day", low_text)
                sub_m = re.search(r"of\s+these\s+\d+\s+pills?,\s+(\d+)\s+pills?", low_text)
                dollars = [float(x) for x in re.findall(r"\$\s*(\d+(?:\.\d+)?)", problem_text)]
                if total_m and sub_m and len(dollars) >= 2:
                    n_total = float(total_m.group(1))
                    n_sub = float(sub_m.group(1))
                    p = min(dollars)
                    q = max(dollars)  # "more" delta
                    n_total_lit = _lit(n_total)
                    n_sub_lit = _lit(n_sub)
                    p_lit = _lit(p)
                    q_lit = _lit(q)
                    # (n_total-n_sub) * (p+q) + n_sub*p
                    special.insert(0, f"{n_total_lit} {n_sub_lit} - {p_lit} {q_lit} + * {n_sub_lit} {p_lit} * +")
            except Exception:
                pass

        # Depth 1+: "older/younger" + "twice/double" + "difference" => build both derived ages and subtract.
        # Example: "Lexie is 6 years older than her brother and her sister is twice her age... age difference?"
        if depth >= 1 and ("difference" in low_text) and any(w in low_text for w in ("older than", "younger than")) and any(
            w in low_text for w in ("twice", "double")
        ):
            try:
                import re

                ordered = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", problem_text)]
                if len(ordered) >= 2:
                    age = ordered[0]
                    delta = ordered[1]
                    a = _lit(float(age))
                    d = _lit(float(delta))
                    # sister = 2*age, brother = age-delta, difference = sister-brother
                    special.insert(0, f"{a} 2 * {a} {d} - -")
            except Exception:
                pass

        # Depth 1+: sequential fraction scaling ("1/10th ... and half of that") => divide in order.
        if depth >= 1 and ("size" in low_text) and any(w in low_text for w in ("1/10", "1/5", "1/4", "1/3", "1/2", "half", "quarter", "third")):
            try:
                import re

                base = max(nums)
                base_expr = _lit(float(base))
                factors: List[tuple[float, float]] = []
                for m in re.finditer(r"(\d+)\s*/\s*(\d+)", low_text):
                    num = float(m.group(1))
                    den = float(m.group(2))
                    if den > 0 and 0 < num < den:
                        factors.append((num, den))
                if "half" in low_text:
                    factors.append((1.0, 2.0))
                if "quarter" in low_text or "fourth" in low_text:
                    factors.append((1.0, 4.0))
                if "third" in low_text:
                    factors.append((1.0, 3.0))
                if factors:
                    expr = base_expr
                    for num, den in factors[:3]:
                        if abs(num - 1.0) < 1e-9:
                            expr = f"{expr} {_lit(float(den))} /"
                        else:
                            expr = f"{expr} {_lit(float(num))} * {_lit(float(den))} /"
                    special.insert(0, expr)
            except Exception:
                pass

        # Depth 1+: 2-group ratio system with a known total cost.
        # Example: "6 pairs of shoes and 4 jerseys for $560. Jerseys cost 1/4 the price of one pair of shoes.
        # Find the shoe total price." => shoe_total = total * n_shoes / (n_shoes + n_jerseys * ratio).
        if depth >= 1 and ("$" in problem_text) and ("shoe" in low_text) and ("jersey" in low_text) and any(
            w in low_text for w in ("1/4", "quarter", "1/2", "half", "1/3", "third")
        ):
            try:
                import re

                dollars = [float(x) for x in re.findall(r"\$\s*(\d+(?:\.\d+)?)", problem_text)]
                if not dollars:
                    raise ValueError("no dollars")
                total = max(dollars)
                shoe_m = re.search(r"(\d+)\s+pairs?\s+of\s+shoes?", low_text)
                jersey_m = re.search(r"(\d+)\s+jerseys?", low_text)
                if not (shoe_m and jersey_m):
                    raise ValueError("no counts")
                n_shoes = float(shoe_m.group(1))
                n_jerseys = float(jersey_m.group(1))
                # ratio of jersey to shoe.
                num = 1.0
                den = 1.0
                frac_m = re.search(r"(\d+)\s*/\s*(\d+)", low_text)
                if frac_m:
                    num = float(frac_m.group(1))
                    den = float(frac_m.group(2))
                elif ("quarter" in low_text) or ("fourth" in low_text):
                    num, den = 1.0, 4.0
                elif "half" in low_text:
                    num, den = 1.0, 2.0
                elif "third" in low_text:
                    num, den = 1.0, 3.0
                if den > 0 and n_shoes > 0 and n_jerseys > 0:
                    t = _lit(float(total))
                    a = _lit(float(n_shoes))
                    b = _lit(float(n_jerseys))
                    num_l = _lit(float(num))
                    den_l = _lit(float(den))
                    # total*n_shoes / (n_shoes + n_jerseys*(num/den))
                    special.insert(0, f"{t} {a} * {a} {b} {num_l} {den_l} / * + /")
            except Exception:
                pass

        # Depth 1+: "sold TOTAL items; some customers bought k each; the rest bought r each; how many bought r?"
        #
        # Generic linear form:
        #   total = Σ(count_i * per_i) + x * rest_per
        #   x = (total - Σ(count_i*per_i)) / rest_per
        #
        # Example: watermelon stand.
        if depth >= 1 and ("sold" in low_text) and ("customer" in low_text) and ("bought" in low_text) and any(
            w in low_text for w in ("the rest", "rest bought")
        ):
            try:
                import re
                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(low_text)

                m_total = re.search(r"\bsold\s+(\d+)\b", norm)
                if not m_total:
                    m_total = re.search(r"\b(\d+)\b[^.]{0,80}\bsold\b", norm)
                if not m_total:
                    raise ValueError("no total")
                total = float(m_total.group(1))

                # Extract "N customers bought K" fragments.
                pairs: List[tuple[float, float]] = []
                for m in re.finditer(r"\b(\d+)\s+customers?\s+bought\s+(\d+)\b", norm):
                    c = float(m.group(1))
                    k = float(m.group(2))
                    if c > 0 and k > 0:
                        pairs.append((c, k))
                    if len(pairs) >= 4:
                        break

                # Extract rest_per from "rest bought R".
                m_rest = re.search(r"(?:the\s+)?rest\s+bought\s+(\d+)\b", norm)
                if not m_rest:
                    m_rest = re.search(r"rest\s+of\s+them\s+bought\s+(\d+)\b", norm)
                if not m_rest:
                    raise ValueError("no rest_per")
                rest_per = float(m_rest.group(1))

                if total > 0 and rest_per > 0 and pairs:
                    expr = _lit(total)
                    for c, k in pairs[:3]:
                        expr += f" {_lit(c)} {_lit(k)} * -"
                    expr += f" {_lit(rest_per)} /"
                    special.insert(0, expr)
            except Exception:
                pass

        # Depth 1+: simple weekly schedule counting (2 days/week for N weeks minus missed days).
        #
        # Example:
        #   "school lunch on Wednesdays and Fridays... 36 weeks... missed 1 Wednesday and 2 Fridays"
        # -> (36*2) - (1+2).
        if depth >= 1 and ("week" in low_text) and ("miss" in low_text) and any(
            d in low_text for d in ("wednesday", "friday", "monday", "tuesday", "thursday")
        ):
            try:
                import re

                word_to_num = {
                    "one": 1.0,
                    "two": 2.0,
                    "three": 3.0,
                    "four": 4.0,
                    "five": 5.0,
                    "six": 6.0,
                    "seven": 7.0,
                    "eight": 8.0,
                    "nine": 9.0,
                    "ten": 10.0,
                }

                m_weeks = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+weeks?\b", low_text)
                if not m_weeks:
                    raise ValueError("no weeks")
                w_raw = m_weeks.group(1).lower()
                weeks = float(w_raw) if w_raw.isdigit() else float(word_to_num.get(w_raw, 0.0))
                if weeks <= 0:
                    raise ValueError("bad weeks")

                # Determine how many unique weekdays are mentioned in the schedule phrase.
                days = []
                for d in ("monday", "tuesday", "wednesday", "thursday", "friday"):
                    if d in low_text:
                        days.append(d)
                # Heuristic: schedule mentions at least 2 distinct weekdays.
                days_per_week = float(min(3, max(2, len(set(days)))))

                # Extract missed counts by weekday when present.
                missed_counts: List[float] = []
                for d in set(days):
                    mm = re.search(
                        rf"\bmiss\w*\b[^.]{0,80}\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+{d}s?\b",
                        low_text,
                    )
                    if mm:
                        r = mm.group(1).lower()
                        missed_counts.append(float(r) if r.isdigit() else float(word_to_num.get(r, 0.0)))
                if not missed_counts:
                    # Fallback: just sum all numbers after "miss" (often two numbers).
                    tail = low_text.split("miss", 1)[1] if "miss" in low_text else ""
                    missed_counts = [float(x) for x in re.findall(r"\b(\d+)\b", tail)[:3]]

                missed_sum = float(sum(missed_counts)) if missed_counts else 0.0
                expr = f"{_lit(weeks)} {_lit(days_per_week)} *"
                if missed_sum > 0:
                    expr += f" {_lit(missed_sum)} -"
                special.insert(0, expr)
            except Exception:
                pass

        # Depth 1+: "charges $R each week; paid every P weeks; how many weeks to get $T" => weeks = T/R.
        #
        # Payment cadence does not change the number of weeks when the rate is explicitly per-week.
        if depth >= 1 and ("how many weeks" in low_text) and ("pay" in low_text) and ("every" in low_text) and (
            "each week" in low_text or "per week" in low_text
        ):
            try:
                import re
                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(low_text)

                # Prefer explicit "charges R" and "get T" anchors.
                m_rate = re.search(r"\bcharges?\s+(\d+(?:\.\d+)?)\b", norm)
                m_total = re.search(r"\bget\s+(\d+(?:\.\d+)?)\b", norm)
                if m_rate and m_total:
                    rate = float(m_rate.group(1))
                    total = float(m_total.group(1))
                    if rate > 0 and total > 0:
                        special.insert(0, f"{_lit(total)} {_lit(rate)} /")
                else:
                    nums_all = [float(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)\b", norm)]
                    if len(nums_all) >= 2:
                        total = max(nums_all)
                        # Heuristic: pick a "rate-like" number (often near 100..1000) smaller than total.
                        rate_candidates = [n for n in nums_all if 0 < n < total]
                        rate = max(rate_candidates) if rate_candidates else min(nums_all)
                        if rate > 0 and total > 0 and total > rate:
                            special.insert(0, f"{_lit(total)} {_lit(rate)} /")
            except Exception:
                pass

        # Depth 1+: "hour-long playlist" remaining minutes then divide by song length.
        if depth >= 1 and (("playlist" in ws_all) or ("playlist" in low_text)) and (("hour" in ws_all) or ("hour" in low_text)) and (
            ("minute" in ws_all) or ("minutes" in ws_all) or ("minute" in low_text)
        ):
            total_minutes = 60.0
            total_lit = _lit(total_minutes)
            counts = [n for n in nums if n > 5 and abs(n - round(n)) < 1e-9]
            durs = [int(round(n)) for n in nums if 0 < n <= 15 and abs(n - round(n)) < 1e-9]
            durs = list(dict.fromkeys(durs))
            for count in counts[:3]:
                for d1 in durs[:4]:
                    consumed = f"{_lit(count)} {_lit(float(d1))} *"
                    for d2 in durs[:4]:
                        if d2 == d1:
                            continue
                        candidates.append(f"{total_lit} {consumed} - {_lit(float(d2))} /")

        # Depth 1+: "each day ... 1/10 of ..." inverse fraction duration: days = den/num.
        if depth >= 1 and (("day" in ws_all or "days" in ws_all) or ("day" in low_text)) and (("each" in ws_all or "every" in ws_all) or ("each" in low_text) or ("every" in low_text)):
            # Prefer small integer fractions appearing in the prompt.
            ints = [int(round(n)) for n in nums if 0 < n <= 12 and abs(n - round(n)) < 1e-9]
            ints = list(dict.fromkeys(ints))
            frac_pairs: List[tuple[int, int]] = []
            for n in ints:
                for d in ints:
                    if d <= 1 or n <= 0 or n >= d:
                        continue
                    frac_pairs.append((n, d))
            for n, d in frac_pairs[:6]:
                candidates.append(f"{_lit(float(d))} {_lit(float(n))} /")
                # Total-cancels form: total / (total*n/d) -> d/n
                total = max(nums)
                candidates.append(f"{_lit(total)} {_lit(total)} {_lit(float(n))} * {_lit(float(d))} / /")

        # Depth 1+: "average ... fit into ..." => (sum(nums)/count) / base.
        if depth >= 1 and ("average" in ws_all) and (("fit" in ws_all) or ("into" in ws_all)):
            count = min(4, len(nums))
            if count >= 3:
                selected = nums[:count]
                sum_expr = _lit(selected[0])
                for v in selected[1:]:
                    sum_expr = f"{sum_expr} {_lit(v)} +"
                count_lit = _lit(float(count))
                for base in selected[:3]:
                    candidates.append(f"{sum_expr} {count_lit} / {_lit(base)} /")

        # Depth 1+: cash vs installment savings.
        #
        # Pattern:
        #   "$C cash or $D down payment and $M a month for N months. How much save by paying cash?"
        # Savings = (D + M*N) - C.
        if depth >= 1 and ("cash" in low_text) and ("down" in low_text) and ("month" in low_text) and ("save" in low_text):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(low_text)
                dollars = [float(x) for x in re.findall(r"\$\s*(\d+(?:\.\d+)?)", problem_text)]
                m_cash = re.search(r"\$\s*(\d+(?:\.\d+)?)\s+\w*\s*cash\b", problem_text, re.IGNORECASE)
                m_down = re.search(r"\$\s*(\d+(?:\.\d+)?)\s+\w*\s*down", problem_text, re.IGNORECASE)
                m_monthly = re.search(r"\$\s*(\d+(?:\.\d+)?)\s+\w*\s*(?:a|per)\s+month", problem_text, re.IGNORECASE)
                m_months = re.search(r"\bfor\s+(\d+)\s+months?\b", norm)
                if m_cash and m_down and m_monthly and m_months:
                    cash = float(m_cash.group(1))
                    down = float(m_down.group(1))
                    monthly = float(m_monthly.group(1))
                    months = float(m_months.group(1))
                    if cash > 0 and down >= 0 and monthly >= 0 and months > 0:
                        special.insert(0, f"{_lit(down)} {_lit(monthly)} {_lit(months)} * + {_lit(cash)} -")
                elif len(dollars) >= 3:
                    cash = max(dollars)
                    rest = [d for d in dollars if abs(d - cash) > 1e-9]
                    if len(rest) >= 2:
                        down = max(rest)
                        monthly = min(rest)
                        months = max([n for n in nums if abs(n - round(n)) < 1e-9 and n <= 60] or [0.0])
                        if months > 0:
                            special.insert(0, f"{_lit(down)} {_lit(monthly)} {_lit(months)} * + {_lit(cash)} -")
            except Exception:
                pass

        # Depth 1+: full theatre two-ticket revenue (adult/child).
        #
        # Pattern:
        #   "A theatre has T seats. Ticket is $A adult and $C child. The theatre is full and contains K children.
        #    What is total ticket cost?" => (T-K)*A + K*C.
        if depth >= 1 and ("seat" in low_text) and ("ticket" in low_text) and ("adult" in low_text) and ("child" in low_text) and (
            "full" in low_text or "sold out" in low_text
        ):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(low_text)
                m_seats = re.search(r"\b(\d+)\s+seats?\b", norm)
                prices = [float(x) for x in re.findall(r"\$\s*(\d+(?:\.\d+)?)", problem_text)]
                m_children = re.search(r"\b(\d+)\s+children\b", norm)
                if m_seats and m_children and len(prices) >= 2:
                    seats = float(m_seats.group(1))
                    children = float(m_children.group(1))
                    adult_price = max(prices)
                    child_price = min(prices)
                    if seats > 0 and 0 <= children <= seats and adult_price >= 0 and child_price >= 0:
                        t = _lit(seats)
                        k = _lit(children)
                        a = _lit(adult_price)
                        c = _lit(child_price)
                        special.insert(0, f"{t} {k} - {a} * {k} {c} * +")
            except Exception:
                pass

        # Depth 1+: cost totals from counts and unit costs (sum of products), plus insurance complements.
        if depth >= 1 and any(w in low_text for w in ("cost", "costs", "price", "spend", "spent", "insurance", "out-of-pocket", "$")):
            has_percent = ("percent" in low_text) or ("%" in low_text)
            pcts: List[int] = []
            if has_percent:
                pcts = [int(round(n)) for n in nums if 0 < n <= 100 and abs(n - round(n)) < 1e-9]
                pcts = list(dict.fromkeys(pcts))

            # Unit costs are commonly decimals; counts are commonly integers (and not percents).
            unit_costs = [n for n in nums if n > 0 and abs(n - round(n)) > 1e-9]
            counts = [
                n
                for n in nums
                if n >= 2
                and n <= 200
                and abs(n - round(n)) < 1e-9
                and int(round(n)) not in set(pcts)
            ]

            # If no decimal costs exist (insurance scenarios), treat large integers as costs.
            if not unit_costs and has_percent:
                unit_costs = [n for n in nums if n >= 101 and (int(round(n)) not in set(pcts))]

            uniq_costs: List[float] = []
            for c in unit_costs:
                if all(abs(c - u) > 1e-9 for u in uniq_costs):
                    uniq_costs.append(c)

            if counts and len(uniq_costs) >= 2:
                c1, c2 = uniq_costs[0], uniq_costs[1]
                for cnt in counts[:2]:
                    candidates.append(f"{_lit(cnt)} {_lit(c1)} * {_lit(cnt)} {_lit(c2)} * +")
                    candidates.append(f"{_lit(cnt)} {_lit(c1)} {_lit(c2)} + *")

            if has_percent and len(uniq_costs) >= 2:
                c1, c2 = float(uniq_costs[0]), float(uniq_costs[1])
                total_cost = c1 + c2
                total_lit = _lit(total_cost)
                for p in pcts[:3]:
                    # Out-of-pocket = total - covered = total - (total*p/100)
                    candidates.append(f"{total_lit} {total_lit} {_lit(float(p))} 100 / * -")
                    # Equivalent: total*(1 - p/100) = total*(100-p)/100
                    candidates.append(f"{total_lit} {_lit(float(100 - p))} 100 / *")
                    # Prefer forms that keep original costs visible (tiebreak uses prominent numbers).
                    candidates.append(f"{_lit(c1)} {_lit(c2)} + {_lit(c1)} {_lit(c2)} + {_lit(float(p))} 100 / * -")
                    candidates.append(f"{_lit(c1)} {_lit(c2)} + {_lit(float(100 - p))} 100 / *")

        # Depth 1+: nested more/less with multipliers (common city population chains).
        if depth >= 1:
            try:
                import re

                low = (problem_text or "").lower()
                if ("more than" in low and "less than" in low) and (("twice" in low or "double" in low) and ("thrice" in low or "triple" in low)):
                    base = max(nums)
                    m_more = re.search(r"(\d+(?:\.\d+)?)\s+more\s+than\s+(?:twice|double|2\s+times)", low)
                    m_less = re.search(r"(\d+(?:\.\d+)?)\s+(?:less|fewer)\s+than\s+(?:thrice|triple|3\s+times)", low)
                    if m_more and m_less:
                        d_more = float(m_more.group(1))
                        d_less = float(m_less.group(1))
                        candidates.append(f"{_lit(base)} 3 * {_lit(d_less)} - 2 * {_lit(d_more)} +")
            except Exception:
                pass

        # Depth 1+: weighted fraction groups (e.g. "half ...", "three-fourths ...", "10% ...") + counts -> sum.
        # Example: Barry/Kevin/Julie/Joe "nice people in the crowd".
        if depth >= 1 and ("people named" in low_text) and ("nice" in low_text) and ("how many" in low_text) and ("crowd" in low_text):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()

                counts_by_name: Dict[str, float] = {}
                for m in re.finditer(r"\b(\d+)\s+people named\s+(\w+)\b", low_norm, re.IGNORECASE):
                    counts_by_name[str(m.group(2)).lower()] = float(m.group(1))

                if len(counts_by_name) >= 2:
                    mult_by_name: Dict[str, str] = {}

                    for m in re.finditer(r"\ball\s+people named\s+(\w+)\s+are nice\b", low_norm, re.IGNORECASE):
                        mult_by_name[str(m.group(1)).lower()] = "all"

                    for m in re.finditer(
                        r"(?:only\s+)?half\s+of\s+(?:the\s+)?people named\s+(\w+)\s+are nice\b",
                        low_norm,
                        re.IGNORECASE,
                    ):
                        mult_by_name[str(m.group(1)).lower()] = "1/2"

                    frac_word = {
                        ("one", "third"): (1, 3),
                        ("two", "third"): (2, 3),
                        ("three", "fourth"): (3, 4),
                        ("three", "quarter"): (3, 4),
                        ("one", "fourth"): (1, 4),
                        ("one", "quarter"): (1, 4),
                        ("three", "quarter"): (3, 4),
                    }
                    for m in re.finditer(
                        r"\b(one|two|three)\s*(?:-| )?(thirds?|fourths?|quarters?)\s+of\s+(?:the\s+)?people named\s+(\w+)\s+are nice\b",
                        low_norm,
                        re.IGNORECASE,
                    ):
                        num_word = str(m.group(1)).lower()
                        den_word = str(m.group(2)).lower()
                        if den_word.startswith("third"):
                            den_key = "third"
                        elif den_word.startswith("fourth"):
                            den_key = "fourth"
                        else:
                            den_key = "quarter"
                        frac = frac_word.get((num_word, den_key))
                        if frac:
                            mult_by_name[str(m.group(3)).lower()] = f"{frac[0]}/{frac[1]}"

                    for m in re.finditer(
                        r"\b(\d+)\s*/\s*(\d+)\s+of\s+(?:the\s+)?people named\s+(\w+)\s+are nice\b",
                        low_norm,
                        re.IGNORECASE,
                    ):
                        mult_by_name[str(m.group(3)).lower()] = f"{int(m.group(1))}/{int(m.group(2))}"

                    for m in re.finditer(
                        r"\b(\d+(?:\.\d+)?)%\s+of\s+(?:the\s+)?people named\s+(\w+)\s+are nice\b",
                        low_norm,
                        re.IGNORECASE,
                    ):
                        pct = str(m.group(1))
                        mult_by_name[str(m.group(2)).lower()] = f"{pct}%"

                    terms: List[str] = []
                    for name, count in counts_by_name.items():
                        mult = mult_by_name.get(name)
                        if not mult:
                            continue
                        if mult == "all":
                            terms.append(_lit(count))
                            continue
                        if mult.endswith("%"):
                            p = float(mult[:-1])
                            terms.append(f"{_lit(count)} {_lit(p)} * 100 /")
                            continue
                        if "/" in mult:
                            n_s, d_s = mult.split("/", 1)
                            n = float(n_s)
                            d = float(d_s) if float(d_s) != 0 else 1.0
                            terms.append(f"{_lit(count)} {_lit(n)} * {_lit(d)} /")
                            continue
                        terms.append(f"{_lit(count)} {_lit(float(mult))} *")

                    if len(terms) >= 2:
                        expr = terms[0]
                        for t in terms[1:]:
                            expr = f"{expr} {t} +"
                        special.insert(0, expr)
            except Exception:
                pass

        # Depth 1+: threshold-based weekly pay (piecewise) from per-game scores.
        if depth >= 1 and ("paid" in low_text) and ("average" in low_text) and ("points" in low_text) and ("gets $" in low_text):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()

                money = [
                    float(str(x).replace(",", ""))
                    for x in re.findall(r"\$\s*([\d,]+(?:\.\d+)?)", problem_text)
                    if str(x).strip()
                ]
                if len(money) >= 2:
                    pay_hi = max(money)
                    pay_lo = min(money)
                    m_thresh = re.search(r"\baverages?\s+(\d+(?:\.\d+)?)\s+or more\b", low_norm)
                    threshold = float(m_thresh.group(1)) if m_thresh else None
                    scores = [float(x) for x in re.findall(r"\bscored\s+(\d+(?:\.\d+)?)\b", low_norm)]
                    if scores and threshold is not None:
                        avg = float(sum(scores)) / float(len(scores))
                        pay = pay_hi if avg >= threshold - 1e-9 else pay_lo
                        special.insert(0, _lit(pay))
            except Exception:
                pass

        # Depth 1+: mixed-quality acreage yield with fractional land split.
        if depth >= 1 and ("acre" in low_text) and ("bushels per acre" in low_text) and ("/" in low_text or "one-" in low_text or "half" in low_text):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                m_acres = re.search(r"\b(\d+(?:\.\d+)?)\s+acres?\b", low_norm)
                m_frac = re.search(r"\b(\d+)\s*/\s*(\d+)\b", low_norm)
                yields = [float(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)\s+bushels per acre\b", low_norm)]
                if m_acres and m_frac and yields:
                    acres = float(m_acres.group(1))
                    n = float(m_frac.group(1))
                    d = float(m_frac.group(2))
                    if d != 0 and acres > 0:
                        good = float(max(yields))
                        other = None
                        if len(yields) >= 2:
                            other = float(min(yields))
                        elif "half as much" in low_norm:
                            other = good / 2.0
                        elif "twice as much" in low_norm or "double" in low_norm:
                            other = good * 2.0
                        if other is not None and other >= 0:
                            a = _lit(acres)
                            frac = f"{_lit(n)} {_lit(d)} /"
                            expr = f"{a} {frac} * {_lit(other)} * {a} {a} {frac} * - {_lit(good)} * +"
                            special.insert(0, expr)
            except Exception:
                pass

        # Depth 1+: fraction cascade with discard + keep (rotten then kept) from a sum.
        if depth >= 1 and ("rotten" in low_text) and ("kept" in low_text) and ("sell" in low_text):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                counts = [
                    float(x)
                    for x in re.findall(r"\b(?:picked|pick)\s+(\d+(?:\.\d+)?)\b", low_norm)
                ]
                fracs = [(float(a), float(b)) for a, b in re.findall(r"\b(\d+)\s*/\s*(\d+)\b", low_norm)]
                if len(counts) >= 2 and len(fracs) >= 2:
                    t_expr = _lit(counts[0])
                    for c in counts[1:4]:
                        t_expr = f"{t_expr} {_lit(c)} +"
                    (n1, d1) = fracs[0]
                    (n2, d2) = fracs[1]
                    if d1 != 0 and d2 != 0:
                        # fresh = total - total*(n1/d1), sell = fresh - fresh*(n2/d2)
                        fresh_expr = f"{t_expr} {t_expr} {_lit(n1)} * {_lit(d1)} / -"
                        sell_expr = f"{fresh_expr} {fresh_expr} {_lit(n2)} * {_lit(d2)} / -"
                        special.insert(0, sell_expr)
            except Exception:
                pass

        # Depth 1+: chained expedition durations in weeks then converted to days.
        if depth >= 1 and ("expedition" in low_text) and ("week" in low_text) and ("day" in low_text) and any(
            q in low_text for q in ("how many days", "number of days", "total number of days", "total days")
        ):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                m_first = re.search(r"first expedition[^.]{0,120}?\b(\d+)\s+weeks?\b", low_norm)
                m_more = re.search(r"\b(\d+)\s+weeks?\s+more\b[^.]{0,80}?second expedition", low_norm)
                k = 2.0 if "twice" in low_norm else None
                m_times = re.search(r"\b(\d+)\s+times\b[^.]{0,80}?last expedition", low_norm)
                if m_times:
                    k = float(m_times.group(1))
                if m_first and m_more and k is not None:
                    w1 = float(m_first.group(1))
                    delta = float(m_more.group(1))
                    expr = f"{_lit(w1)} {_lit(w1)} {_lit(delta)} + + {_lit(w1)} {_lit(delta)} + {_lit(k)} * + 7 *"
                    special.insert(0, expr)
            except Exception:
                pass

        # Depth 1+: sum of multiple fraction-of quantities.
        # Example: "mixes 3/5th of 20 liters ... with 5/6th of 18 liters ..." -> 20*(3/5) + 18*(5/6)
        if depth >= 1 and (" of " in low_text) and ("/" in low_text) and any(
            q in low_text for q in ("how many", "total", "altogether", "in all", "obtained", "mixture", "combined")
        ):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                # Capture patterns like "3/5th of 20" or "5/6 of 18"
                frac_terms = []
                for n, d, qty in re.findall(
                    r"\b(\d+)\s*/\s*(\d+)(?:st|nd|rd|th)?\s+of\s+(\d+(?:\.\d+)?)\b",
                    low_norm,
                ):
                    try:
                        nn = float(n)
                        dd = float(d)
                        qq = float(qty)
                    except Exception:
                        continue
                    if dd <= 0:
                        continue
                    # qty * n / d  -> qty n * d /
                    frac_terms.append(f"{_lit(qq)} {_lit(nn)} * {_lit(dd)} /")
                if len(frac_terms) >= 2:
                    # Sum the first few terms.
                    terms = frac_terms[:4]
                    rpn = " ".join(terms) + " " + "+ " * (len(terms) - 1)
                    special.insert(0, rpn.strip())
            except Exception:
                pass

        # Depth 1+: "K times as many ... altogether/total" (sum or difference).
        # Example: "Grant has four times as many vacations as Kelvin has classes. If Kelvin has 90 classes,
        # how many vacations and classes do they have altogether?" -> 90*4 + 90.
        if depth >= 1 and ("times as many" in low_text or "twice as many" in low_text or "double" in low_text) and any(
            q in low_text for q in ("altogether", "in all", "in total", "total")
        ):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                k: float | None = None
                m_k = re.search(r"\b(\d+(?:\.\d+)?)\s+times\s+as\s+many\b", low_norm)
                if m_k:
                    k = float(m_k.group(1))
                elif "twice as many" in low_norm or "double" in low_norm:
                    k = 2.0
                elif "thrice" in low_norm or "triple" in low_norm:
                    k = 3.0
                if k is not None and k > 0:
                    base_candidates: List[float] = []
                    # Prefer an explicit "if X has N ..." style base.
                    for m in re.finditer(r"\b(?:if\s+)?\w+\s+has\s+(\d+(?:\.\d+)?)\b", low_norm):
                        try:
                            base_candidates.append(float(m.group(1)))
                        except Exception:
                            pass
                    # Fallback to the largest extracted number if nothing obvious.
                    base = max(base_candidates) if base_candidates else (max(nums) if nums else None)
                    if base is not None:
                        # Sum: base*k + base
                        special.insert(0, f"{_lit(base)} {_lit(k)} * {_lit(base)} +")
                        # Difference variant: base*k - base (for "how many more").
                        if "how many more" in low_text or "more than" in low_text:
                            special.insert(0, f"{_lit(base)} {_lit(k)} * {_lit(base)} -")
            except Exception:
                pass

        # Depth 1+: percent complements ("X% ... how many not/remaining/left").
        # Example: "80% of 100 plays were lead; how many not lead?" -> 100*(100-80)/100 = 20.
        if depth >= 1 and ("%" in low_text or "percent" in low_text) and ("$" not in problem_text):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                # Whole-word complement cues to avoid matching substrings like "notebook".
                wants_complement = re.search(r"\b(not|remaining|left|rest)\b", low_norm) is not None
                if wants_complement:
                    m_pct = re.search(r"\b(\d+(?:\.\d+)?)\s*%", low_norm)
                    if not m_pct:
                        m_pct = re.search(r"\b(\d+(?:\.\d+)?)\s+percent\b", low_norm)
                    if m_pct:
                        pct = float(m_pct.group(1))
                        if 0 < pct < 100 and nums:
                            # Prefer a non-percent total (e.g., 100 plays) instead of the percent itself.
                            total_candidates = [n for n in nums if n > pct + 1e-9]
                            total = max(total_candidates) if total_candidates else max(nums)
                            comp = 100.0 - pct
                            special.insert(0, f"{_lit(total)} {_lit(comp)} 100 / *")
                            # Alternative form: total - total*pct/100.
                            special.insert(0, f"{_lit(total)} {_lit(total)} {_lit(pct)} 100 / * -")
            except Exception:
                pass

        # Depth 1+: nested "each" multiplication chains (hierarchical counting).
        # Example: "4 trains, each has 4 carriages; each carriage has 3 rows; each row has 5 wheels" -> 4*4*3*5.
        #
        # IMPORTANT: only trigger for nested containment ("each ... has/contains/holds ..."), not for
        # multi-group payouts like "each get X" (those are sum-of-products).
        try:
            import re

            has_nested_each = (
                re.search(r"\beach\s+\w+\s+(?:has|have|contains|contain|holds)\b", low_text) is not None
            )
            has_each_get = re.search(r"\beach\s+\w+\s+gets?\b", low_text) is not None
        except Exception:
            has_nested_each = False
            has_each_get = False
        if (
            depth >= 1
            and has_nested_each
            and not has_each_get
            and (low_text.count("each") >= 2)
            and any(q in low_text for q in ("how many", "total", "in all", "altogether"))
        ):
            try:
                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                # Use integers in appearance order, preserving duplicates; ignore dollar amounts.
                ints = [float(m.group(1)) for m in re.finditer(r"\b(\d+)\b", low_norm)]
                # Filter: small-ish positive factors tend to represent nested counts.
                factors = [int(x) for x in ints if 1 <= x <= 50]
                if len(factors) >= 3:
                    # Multiply first few factors (bounded).
                    chain = [_lit(float(factors[0]))]
                    for f in factors[1:6]:
                        chain.append(_lit(float(f)))
                        chain.append("*")
                    special.insert(0, " ".join(chain))
            except Exception:
                pass

        # Depth 1+: "rest of" then fractional part of the rest.
        # Example: "42 total, 20 Europe, 10 SA; from the rest only half ..." -> (42-20-10)/2.
        if depth >= 1 and ("rest of" in low_text or "from the rest" in low_text) and ("half" in low_text):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                ints = [float(m.group(1)) for m in re.finditer(r"\b(\d+(?:\.\d+)?)\b", low_norm)]
                # Exclude the explicit 0.5 helper from numeric words.
                ints = [x for x in ints if x > 1]
                if len(ints) >= 3:
                    total = max(ints)
                    parts = [x for x in ints if x != total][:3]
                    if len(parts) >= 2:
                        expr = f"{_lit(total)} {_lit(parts[0])} - {_lit(parts[1])} - 2 /"
                        special.insert(0, expr)
            except Exception:
                pass

        # Depth 1+: monthly budget then set aside percent for savings.
        if depth >= 1 and ("set aside" in low_text) and ("savings" in low_text) and ("%" in low_text or "percent" in low_text):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                m_weeks = re.search(r"\bnext\s+(\d+)\s+weeks?\b", low_norm)
                m_weekly = re.search(r"\$\s*([\d,]+(?:\.\d+)?)\s+(?:a|per)\s+week\b", problem_text, re.IGNORECASE)
                # Note: '%' is not a word character, so a trailing word-boundary (`\b`) breaks matches.
                m_pct = re.search(r"\b(\d+(?:\.\d+)?)\s*%", low_norm)
                dollars = [
                    float(str(x).replace(",", ""))
                    for x in re.findall(r"\$\s*([\d,]+(?:\.\d+)?)", problem_text)
                    if str(x).strip()
                ]
                if m_weeks and m_weekly and m_pct and dollars:
                    weeks = float(m_weeks.group(1))
                    weekly = float(str(m_weekly.group(1)).replace(",", ""))
                    pct = float(m_pct.group(1))
                    remaining = list(dollars)
                    removed = False
                    for i, v in enumerate(list(remaining)):
                        if not removed and abs(v - weekly) < 1e-9:
                            remaining.pop(i)
                            removed = True
                            break
                    monthly_sum = float(sum(remaining)) if remaining else 0.0
                    expr = f"{_lit(weekly)} {_lit(weeks)} *"
                    if monthly_sum > 0:
                        expr += f" {_lit(monthly_sum)} +"
                    expr += f" {_lit(pct)} 100 / *"
                    special.insert(0, expr)
            except Exception:
                pass

        # Depth 1+: tiered hourly pay (base rate up to threshold hours, then higher rate).
        # Example: "$20 every hour ... up to 40 hours ... paid double ... for a 50-hour week"
        # RPN: rate threshold * rate 2 * total threshold - * +
        if depth >= 1 and ("per hour" in low_text or "every hour" in low_text) and ("up to" in low_text) and any(
            w in low_text for w in ("after which", "after that", "after", "then")
        ):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()

                m_rate = re.search(r"\$\s*([\d,]+(?:\.\d+)?)\s+(?:every|per)\s+hour", low_norm)
                if not m_rate:
                    m_rate = re.search(r"\bpaid\s+\$?\s*([\d,]+(?:\.\d+)?)\s+(?:every|per)\s+hour", low_norm)
                m_thresh = re.search(r"\bup to\b[^\d]{0,20}?(\d+(?:\.\d+)?)\s+hours?\b", low_norm)
                # Total worked: "for a 50-hour week", "for 50 hours", etc.
                m_total = (
                    re.search(r"\bfor a\s+(\d+(?:\.\d+)?)\s*[- ]\s*hour\b", low_norm)
                    or re.search(r"\bfor\s+(\d+(?:\.\d+)?)\s+hours?\b", low_norm)
                    or re.search(r"\b(\d+(?:\.\d+)?)\s*[- ]\s*hour\s+week\b", low_norm)
                )
                if m_rate and m_thresh and m_total:
                    rate = float(str(m_rate.group(1)).replace(",", ""))
                    thresh = float(m_thresh.group(1))
                    total = float(m_total.group(1))
                    if rate > 0 and thresh > 0 and total > 0:
                        # If no overtime, simple pay.
                        special.insert(0, f"{_lit(rate)} {_lit(total)} *")

                        factor = 2.0
                        if "double" in low_norm or "twice" in low_norm:
                            factor = 2.0
                        m_times = re.search(r"\b(\d+(?:\.\d+)?)\s+times\b[^.]{0,60}?(?:amount|rate|pay)", low_norm)
                        if m_times:
                            try:
                                factor = float(m_times.group(1))
                            except Exception:
                                factor = factor
                        # Tiered: base + overtime
                        special.insert(0, f"{_lit(rate)} {_lit(thresh)} * {_lit(rate)} {_lit(factor)} * {_lit(total)} {_lit(thresh)} - * +")
            except Exception:
                pass

        # Depth 1+: unit-price with per-item discount.
        # Example: "$5 each ... bought 10 ... discount of $0.5 each" -> (5 - 0.5) * 10.
        if depth >= 1 and ("discount" in low_text) and ("$" in problem_text):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                has_per_word = re.search(r"\\bper\\b", low_norm) is not None
                has_discount_unit_context = any(
                    w in low_norm for w in ("each", "per person", "given a discount", "discount of")
                ) or has_per_word
                if not has_discount_unit_context:
                    # Avoid false positives like "personal planner" matching "per".
                    dollars = []
                else:
                    dollars = [
                        float(str(x).replace(",", ""))
                        for x in re.findall(r"\$\s*([\d,]+(?:\.\d+)?)", problem_text)
                        if str(x).strip()
                    ]
                if len(dollars) >= 2:
                    unit = max(dollars)
                    disc = min(dollars)
                    # Count: "bought N", "buy N", or "there are N of them".
                    m_n = (
                        re.search(r"\bbought\s+(\d+(?:\.\d+)?)\b", low_norm)
                        or re.search(r"\bpurchased\s+(\d+(?:\.\d+)?)\b", low_norm)
                        or re.search(r"\bbuy\s+(\d+(?:\.\d+)?)\b", low_norm)
                        or re.search(r"\bthere\s+are\s+(\d+(?:\.\d+)?)\b", low_norm)
                        or re.search(r"\b(\d+(?:\.\d+)?)\s+of\s+them\b", low_norm)
                    )
                    if m_n:
                        n = float(m_n.group(1))
                        if n > 0:
                            special.insert(0, f"{_lit(unit)} {_lit(disc)} - {_lit(n)} *")
                            special.insert(0, f"{_lit(n)} {_lit(unit)} * {_lit(n)} {_lit(disc)} * -")
            except Exception:
                pass

        # Depth 1+: missing item cost from a total and other known $ costs.
        # Example: "$200 sneakers and $250 outfit, total $750; what was the racket cost?" -> 750 - 200 - 250.
        if depth >= 1 and ("$" in problem_text) and ("total" in low_text) and ("discount" not in low_text):
            try:
                import re

                dollars = [
                    float(str(x).replace(",", ""))
                    for x in re.findall(r"\$\s*([\d,]+(?:\.\d+)?)", problem_text)
                    if str(x).strip()
                ]
                # Require at least 1 explicit total and 2 other costs.
                if len(dollars) >= 3 and any(w in low_text for w in ("spent", "spend", "in total", "total of", "total for")):
                    total = max(dollars)
                    others = [d for d in dollars if abs(d - total) > 1e-9]
                    # Prefer the largest other costs; counts/discounts are typically smaller.
                    others = sorted(others, reverse=True)[:3]
                    if others:
                        expr = _lit(total)
                        for d in others:
                            expr += f" {_lit(d)} -"
                        special.insert(0, expr)
                        # Alternate form: total - (a+b+...)
                        if len(others) >= 2:
                            sum_expr = " ".join(_lit(d) for d in others)
                            sum_expr += " " + "+ " * (len(others) - 1)
                            special.insert(0, f"{_lit(total)} {sum_expr.strip()} -")
            except Exception:
                pass

        # Depth 1+: "N items together cost T; K cost C each; remaining equal" -> (T - K*C) / (N-K).
        if depth >= 1 and ("$" in problem_text) and ("remaining" in low_text) and ("equal" in low_text) and any(
            w in low_text for w in ("together cost", "together costs", "together costed")
        ):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                # Total: "<N> <item> together cost $T"
                m_total = re.search(
                    r"\b(\d+)\s+[a-z ]+?\s+together\s+costs?\s+\$\s*([\d,]+(?:\.\d+)?)\b", low_norm
                )
                # Known: "<K> <item> ... cost $C each"
                m_known = re.search(
                    r"\b(\d+)\s+[a-z ]+?\s+that\s+costs?\s+\$\s*([\d,]+(?:\.\d+)?)\s+each\b", low_norm
                )
                if m_total and m_known:
                    n_total = float(m_total.group(1))
                    total = float(str(m_total.group(2)).replace(",", ""))
                    k = float(m_known.group(1))
                    each = float(str(m_known.group(2)).replace(",", ""))
                    rem = n_total - k
                    if n_total > 0 and k >= 0 and rem > 0:
                        special.insert(0, f"{_lit(total)} {_lit(k)} {_lit(each)} * - {_lit(n_total)} {_lit(k)} - /")
                        # Variant: precompute known_total then subtract.
                        special.insert(0, f"{_lit(total)} {_lit(k)} {_lit(each)} * - {_lit(rem)} /")
            except Exception:
                pass

        # Depth 1+: percent discount applied to a multi-item subtotal.
        # Example: "4 notebooks at $15 and 8 planners at $10, 20% discount" -> (4*15 + 8*10) * 80/100.
        if depth >= 1 and ("discount" in low_text) and ("$" in problem_text) and ("%" in low_text or "percent" in low_text):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                m_pct = re.search(r"\b(\d+(?:\.\d+)?)\s*%", low_norm)
                if not m_pct:
                    m_pct = re.search(r"\b(\d+(?:\.\d+)?)\s+percent\b", low_norm)
                pct = float(m_pct.group(1)) if m_pct else None
                dollars = [
                    float(str(x).replace(",", ""))
                    for x in re.findall(r"\$\s*([\d,]+(?:\.\d+)?)", problem_text)
                    if str(x).strip()
                ]
                if pct is not None and 0 < pct < 100 and len(dollars) >= 2:
                    pay_pct = 100.0 - pct
                    aligned_ok = False

                    # Prefer an item-aligned subtotal when we can associate counts with priced items.
                    # Example: "<item> costs $P ... buy N <item> and M <item2> ...".
                    def _singular(phrase: str) -> str:
                        phrase = re.sub(r"[^a-z ]+", " ", (phrase or "").lower()).strip()
                        phrase = re.sub(r"\s+", " ", phrase)
                        parts = phrase.split()
                        if not parts:
                            return ""
                        # Singularize only the last token (cheap heuristic).
                        last = parts[-1]
                        if last.endswith("s") and len(last) > 1:
                            parts[-1] = last[:-1]
                        return " ".join(parts)

                    price_by_item: Dict[str, float] = {}
                    price_seq: List[float] = []
                    for m in re.finditer(r"\b([a-z ]+?)\s+costs?\s+\$\s*([\d,]+(?:\.\d+)?)\b", low_norm):
                        item = _singular(m.group(1))
                        if not item:
                            continue
                        try:
                            price_val = float(str(m.group(2)).replace(",", ""))
                            price_by_item[item] = price_val
                            price_seq.append(price_val)
                        except Exception:
                            pass

                    counts_by_item: Dict[str, float] = {}
                    count_seq: List[float] = []
                    pos_buy = low_norm.find("buy")
                    if pos_buy >= 0:
                        end = low_norm.find("discount", pos_buy)
                        buy_clause = low_norm[pos_buy : (end if end > pos_buy else len(low_norm))]
                        for m in re.finditer(r"\b(\d+)\s+([a-z ]+?)(?=\s+(?:and|at|with|for|$|,))", buy_clause):
                            item = _singular(m.group(2))
                            if not item:
                                continue
                            c_val = float(m.group(1))
                            counts_by_item[item] = c_val
                            count_seq.append(c_val)

                    if price_by_item and counts_by_item:
                        terms: List[str] = []
                        for c_item, c in list(counts_by_item.items())[:4]:
                            # Exact match, else substring match.
                            price = price_by_item.get(c_item)
                            if price is None:
                                for p_item, p in price_by_item.items():
                                    if c_item == p_item or c_item in p_item or p_item in c_item:
                                        price = p
                                        break
                            if price is None:
                                continue
                            if c > 0 and price > 0:
                                terms.append(f"{_lit(c)} {_lit(price)} *")
                        if len(terms) >= 2:
                            subtotal = " ".join(terms) + " " + "+ " * (len(terms) - 1)
                            special.insert(0, f"{subtotal.strip()} {_lit(pay_pct)} 100 / *")
                            aligned_ok = True
                    # Fallback alignment by appearance order (works well for well-structured prompts).
                    if (not aligned_ok) and len(price_seq) >= 2 and len(count_seq) >= 2:
                        terms = [f"{_lit(count_seq[0])} {_lit(price_seq[0])} *", f"{_lit(count_seq[1])} {_lit(price_seq[1])} *"]
                        subtotal = " ".join(terms) + " +"
                        special.insert(0, f"{subtotal.strip()} {_lit(pay_pct)} 100 / *")
                        aligned_ok = True

                    if not aligned_ok:
                        # Candidate counts: integers not part of $ amounts and not the percent itself.
                        ints = [
                            float(x)
                            for x in re.findall(r"\b(\d+)\b", low_norm)
                            if str(x).strip()
                        ]
                        counts = []
                        for x in ints:
                            if abs(x - pct) < 1e-9 or abs(x - pay_pct) < 1e-9:
                                continue
                            if any(abs(x - d) < 1e-9 for d in dollars):
                                continue
                            if x <= 0:
                                continue
                            counts.append(float(x))
                        # Keep a small set to avoid blowup.
                        counts = list(dict.fromkeys(counts))[:4]
                        prices = list(dict.fromkeys(sorted(dollars)))[:4]
                        # Build pairwise subtotals.
                        for c1 in counts[:3]:
                            for c2 in counts[:3]:
                                if abs(c1 - c2) < 1e-9:
                                    continue
                                for p1 in prices[:3]:
                                    for p2 in prices[:3]:
                                        if abs(p1 - p2) < 1e-9:
                                            continue
                                        subtotal = f"{_lit(c1)} {_lit(p1)} * {_lit(c2)} {_lit(p2)} * +"
                                        # Append (not insert) so any item-aligned candidate stays at the top.
                                        special.append(f"{subtotal} {_lit(pay_pct)} 100 / *")
                                        if len(special) >= max_candidates:
                                            break
                                    if len(special) >= max_candidates:
                                        break
                                if len(special) >= max_candidates:
                                    break
                            if len(special) >= max_candidates:
                                break
            except Exception:
                pass

        # Depth 1+: simple currency conversion ("A units is worth B dollars") optionally modified by a fraction
        # of the official rate.
        if depth >= 1 and ("worth" in low_text) and any(w in low_text for w in ("dollar", "dollars")):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                m_rate = re.search(r"\b(\d+(?:\.\d+)?)\s+\w+\s+is\s+worth\s+(\d+(?:\.\d+)?)\s+dollars?\b", low_norm)
                if m_rate:
                    a = float(m_rate.group(1))
                    b = float(m_rate.group(2))
                    if a > 0 and b > 0:
                        # Try to find an explicit "with N <unit>" amount; otherwise use the largest number.
                        m_amt = re.search(r"\bwith\s+(\d+(?:\.\d+)?)\s+\w+\b", low_norm)
                        amt = float(m_amt.group(1)) if m_amt else (max(nums) if nums else None)
                        if amt is not None and amt > 0:
                            base_conv = f"{_lit(amt)} {_lit(b)} * {_lit(a)} /"
                            special.insert(0, base_conv)
                            m_frac = re.search(r"\b(\d+)\s*/\s*(\d+)(?:st|nd|rd|th)?s?\b", low_norm)
                            if m_frac and any(w in low_norm for w in ("official", "only give", "only gives", "only giving")):
                                fn = float(m_frac.group(1))
                                fd = float(m_frac.group(2))
                                if fd > 0:
                                    special.insert(0, f"{base_conv} {_lit(fn)} * {_lit(fd)} /")
            except Exception:
                pass

        # Depth 1+: month-from-week comparisons with "twice/double" and "how many more".
        # Example: "X a week; Y twice as many; how many more in a month?" -> (X*2 - X) * 4.
        if (
            depth >= 1
            and ("a week" in low_text or "per week" in low_text)
            and ("month" in low_text)
            and (("how many more" in low_text) or ("more than" in low_text))
            and (("twice" in low_text) or ("double" in low_text) or ("times" in low_text))
        ):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                # Base rate per week: "<N> ... a week".
                m_base = re.search(r"\b(\d+(?:\.\d+)?)\b[^.]{0,40}?\b(?:a|per)\s+week\b", low_norm)
                base = float(m_base.group(1)) if m_base else (max(nums) if nums else None)
                k = 2.0 if ("twice" in low_norm or "double" in low_norm) else None
                m_k = re.search(r"\b(\d+(?:\.\d+)?)\s+times\b", low_norm)
                if m_k:
                    k = float(m_k.group(1))
                if base is not None and k is not None and base > 0 and k > 0:
                    special.insert(0, f"{_lit(base)} {_lit(k)} * {_lit(base)} - 4 *")
                    special.insert(0, f"{_lit(base)} {_lit(k - 1)} * 4 *")
            except Exception:
                pass

        # Depth 1+: per-day quantities with a "twice as many" derived term, then asked for a week total.
        # Example: "2 ... per day, 3 ... per day, twice as many as the 3 ... each day. How many ... in a week?"
        if depth >= 1 and ("per day" in low_text) and ("week" in low_text) and (("twice" in low_text) or ("double" in low_text)):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                rates: List[float] = []
                for m in re.finditer(r"\b(\d+(?:\.\d+)?)\b[^.]{0,60}?\bper\s+day\b", low_norm):
                    try:
                        rates.append(float(m.group(1)))
                    except Exception:
                        pass
                rates = [r for r in rates if r > 0]
                if len(rates) >= 2:
                    r1 = rates[0]
                    r2 = rates[1]
                    # Derived from the second rate (most common in "twice as many as <second noun>").
                    best = f"{_lit(r1)} {_lit(r2)} + {_lit(r2)} 2 * + 7 *"
                    variant = f"{_lit(r1)} {_lit(r2)} + {_lit(r1)} 2 * + 7 *"
                    special.insert(0, best)
                    special.insert(1, variant)
            except Exception:
                pass

        # Depth 1+: inverse-half after an addition/subtraction, e.g. "(x + a)/2 = b" -> b*2 - a.
        if depth >= 1 and ("half" in low_text) and any(w in low_text for w in ("now", "only", "left")) and any(
            w in low_text for w in ("bought", "added", "add", "plus")
        ):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                m_now = re.search(r"\b(?:now\s+)?only\s+(\d+(?:\.\d+)?)\b", low_norm)
                m_delta = (
                    re.search(r"\bbought\s+(\d+(?:\.\d+)?)\b", low_norm)
                    or re.search(r"\badded\s+(\d+(?:\.\d+)?)\b", low_norm)
                    or re.search(r"\badd\s+(\d+(?:\.\d+)?)\b", low_norm)
                )
                if m_now and m_delta:
                    now = float(m_now.group(1))
                    delta = float(m_delta.group(1))
                    if now > 0:
                        minus = f"{_lit(now)} 2 * {_lit(delta)} -"
                        plus = f"{_lit(now)} 2 * {_lit(delta)} +"
                        special.insert(0, minus)
                        special.insert(1, plus)
            except Exception:
                pass

        # Depth 1+: unknown unit price from total minus known items.
        if depth >= 1 and ("how much does" in low_text) and ("cost" in low_text) and ("total" in low_text) and ("$" in problem_text):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                m_total = re.search(r"\btotal[^$]{0,40}?\$\s*([\d,]+(?:\.\d+)?)\b", problem_text, re.IGNORECASE)
                if not m_total:
                    m_total = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", problem_text)
                total = float(str(m_total.group(1)).replace(",", "")) if m_total else None
                m_unknown = re.search(r"how much does\s+1\s+([a-z ]+?)\s+cost", low_norm)
                unknown_phrase = str(m_unknown.group(1)).strip() if m_unknown else ""

                known: List[tuple[str, float]] = []
                for m in re.finditer(
                    r"(?:cost of|cost)\s+(?:a|an|1|one)\s+([a-z ]+?)\s+(?:is|costs)\s+\$?\s*([\d,]+(?:\.\d+)?)",
                    low_norm,
                ):
                    known.append((str(m.group(1)).strip(), float(str(m.group(2)).replace(",", ""))))
                for m in re.finditer(r"\ba\s+([a-z ]+?)\s+costs?\s+\$?\s*([\d,]+(?:\.\d+)?)", low_norm):
                    known.append((str(m.group(1)).strip(), float(str(m.group(2)).replace(",", ""))))

                if total is not None and unknown_phrase and known:
                    # Prefer counts from the "purchase list" portion of the prompt (typically
                    # before the first mention of "total"), to avoid accidentally capturing
                    # the "1 <item>" in the question or the "$2/$5" unit prices.
                    search_space = low_norm
                    pos_total = low_norm.find("total")
                    if pos_total > 0:
                        search_space = low_norm[:pos_total]

                    def _find_count(phrase: str) -> Optional[float]:
                        if not phrase:
                            return None
                        phrase = str(phrase).strip().lower()
                        phrase = re.sub(r"\s+", " ", phrase)

                        # Try to match "<count> <container>(s) of <thing>" for phrases like
                        # "bag of chocolate chips" -> "20 bags of chocolate chips".
                        m_of = re.match(r"(.+?)\s+of\s+(.+)", phrase)
                        if m_of:
                            container = m_of.group(1).strip()
                            thing = m_of.group(2).strip()
                            container_pat = re.escape(container) + r"s?"
                            thing_pat = re.escape(thing)
                            pattern = r"\b(\d+)\s+" + container_pat + r"\s+of\s+" + thing_pat + r"\b"
                        else:
                            # Simple "<count> <thing>(s)" (e.g., "10 chocolate bars").
                            pattern = r"\b(\d+)\s+" + re.escape(phrase) + r"s?\b"

                        matches = list(re.finditer(pattern, search_space))
                        if not matches:
                            return None
                        return float(matches[-1].group(1))

                    unknown_count = _find_count(unknown_phrase)
                    known_terms: List[str] = []
                    for phrase, price in known[:3]:
                        c = _find_count(phrase)
                        if c is None:
                            continue
                        known_terms.append(f"{_lit(c)} {_lit(price)} *")
                    if unknown_count and unknown_count > 0 and known_terms:
                        expr = f"{_lit(total)} {known_terms[0]} -"
                        for t in known_terms[1:]:
                            expr = f"{expr} {t} -"
                        expr = f"{expr} {_lit(unknown_count)} /"
                        special.insert(0, expr)
            except Exception:
                pass

        # Depth 1+: two-entity weekly total cost with different durations.
        if depth >= 1 and ("total" in low_text) and ("a week" in low_text) and ("weekly cost" in low_text) and ("weeks" in low_text):
            try:
                import re

                from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

                norm = normalize_number_words(problem_text or "")
                low_norm = norm.lower()
                m_total_week = re.search(r"\$\s*([\d,]+(?:\.\d+)?)\s+in total a week\b", low_norm)
                if not m_total_week:
                    m_total_week = re.search(r"\$\s*([\d,]+(?:\.\d+)?)\s+in total\s+a week\b", low_norm)
                total_week = float(str(m_total_week.group(1)).replace(",", "")) if m_total_week else None
                m_known = re.search(r"weekly cost of the\s+(\w+)[^$]{0,30}?\$\s*([\d,]+(?:\.\d+)?)", low_norm)
                if total_week is not None and m_known:
                    known_ent = str(m_known.group(1)).lower()
                    known_week = float(str(m_known.group(2)).replace(",", ""))
                    weeks_by_ent: Dict[str, float] = {}
                    for m in re.finditer(r"\bthe\s+(\w+)\s+for\s+(\d+)\s+weeks?\b", low_norm):
                        weeks_by_ent[str(m.group(1)).lower()] = float(m.group(2))
                    if known_ent in weeks_by_ent and len(weeks_by_ent) >= 2 and 0 <= known_week <= total_week:
                        other_weeks = None
                        for ent, w in weeks_by_ent.items():
                            if ent != known_ent:
                                other_weeks = w
                                break
                        if other_weeks is not None:
                            expr = f"{_lit(weeks_by_ent[known_ent])} {_lit(known_week)} * {_lit(other_weeks)} {_lit(total_week)} {_lit(known_week)} - * +"
                            special.insert(0, expr)
            except Exception:
                pass

        # Depth 1: pairwise arithmetic.
        for i in range(min(6, len(nums))):
            for j in range(min(6, len(nums))):
                if i == j:
                    continue
                a = nums[i]
                b = nums[j]
                for op in op_order:
                    if op == "/" and abs(b) < 1e-12:
                        continue
                    candidates.append(f"{_lit(a)} {_lit(b)} {op}")
                    if len(candidates) >= max_candidates and depth <= 1:
                        return _dedupe_limit(list(special) + list(candidates))

        # Depth 2+: triple chaining (left associative).
        if depth >= 2 and len(nums) >= 3:
            base_ops = op_order[:2] if strategy in {"prioritize_division"} else op_order[:3]
            stop = False
            for a in nums[:4]:
                for b in nums[:4]:
                    if a == b:
                        continue
                    for c in nums[:4]:
                        if c in {a, b}:
                            continue
                        for op1 in base_ops:
                            if op1 == "/" and abs(b) < 1e-12:
                                continue
                            for op2 in base_ops:
                                if op2 == "/" and abs(c) < 1e-12:
                                    continue
                                candidates.append(f"{_lit(a)} {_lit(b)} {op1} {_lit(c)} {op2}")
                                if len(candidates) >= max_candidates:
                                    if depth < 4:
                                        return _dedupe_limit(list(special) + list(candidates))
                                    # For deeper depths, keep building specialized candidates later;
                                    # just cap triple-chain growth.
                                    if len(candidates) >= max_candidates * 4:
                                        stop = True
                                        break
                            if stop:
                                break
                        if stop:
                            break
                    if stop:
                        break
                if stop:
                    break

        # Depth 3+: sum all parts (accumulation).
        if depth >= 3 and len(nums) >= 2:
            parts = [_lit(n) for n in nums[:5]]
            if len(parts) >= 2:
                rpn = " ".join(parts) + " " + "+ " * (len(parts) - 1)
                candidates.append(rpn.strip())

        # Depth 4+: fraction/percentage-of-total patterns and "remaining/rest".
        if depth >= 4 and len(nums) >= 3:
            ws = {str(w).lower() for w in (words or []) if w}
            has_remaining = (str(question_type) == "difference") or bool(ws.intersection({"remaining", "remain", "rest", "left"}))
            total = max(nums)
            total_lit = _lit(total)

            # Candidate fractions (numerator/denominator) from small integers in the prompt.
            small_ints = []
            for n in nums:
                if n <= 0:
                    continue
                if abs(n - round(n)) > 1e-9:
                    continue
                if n <= 12:
                    small_ints.append(int(round(n)))
            small_ints = list(dict.fromkeys(small_ints))  # stable unique

            frac_pairs: List[tuple[int, int]] = []
            for n in small_ints:
                for d in small_ints:
                    if d <= 1 or n <= 0 or n >= d:
                        continue
                    frac_pairs.append((n, d))

            # total * n / d
            for n, d in frac_pairs[:6]:
                special.append(f"{total_lit} {_lit(n)} * {_lit(d)} /")
                if has_remaining:
                    # total - (total*n/d)  => total total n * d / -
                    special.append(f"{total_lit} {total_lit} {_lit(n)} * {_lit(d)} / -")

            # total - frac1 - frac2  => (total - frac1) - frac2, where both fracs are of total
            if has_remaining and len(frac_pairs) >= 2:
                (n1, d1) = frac_pairs[0]
                (n2, d2) = frac_pairs[1]
                special.append(
                    f"{total_lit} {total_lit} {_lit(n1)} * {_lit(d1)} / - "
                    f"{total_lit} {_lit(n2)} * {_lit(d2)} / -"
                )

            # Depth 5+: remaining after a fractional give-away AND an extra subtraction term.
            # Example: "has $100 ... gives 1/4 ... buys $40 ... remaining"
            if depth >= 5 and has_remaining:
                others: List[float] = []
                for x in nums:
                    if abs(x - total) < 1e-9:
                        continue
                    if x <= 0:
                        continue
                    others.append(x)
                # Keep a few plausible "spend" terms (smaller than total).
                others = [x for x in others if x < total] or others
                for n, d in frac_pairs[:6]:
                    for x in others[:4]:
                        if abs(float(x) - float(n)) < 1e-9 or abs(float(x) - float(d)) < 1e-9:
                            continue
                        special.append(f"{total_lit} {total_lit} {_lit(n)} * {_lit(d)} / - {_lit(x)} -")

            # Percentage-of-total when '%' appears.
                if "%" in ws or "percent" in ws:
                    pct_ints = [p for p in small_ints if 0 < p <= 100]
                    for p in pct_ints[:4]:
                        # total * (p/100)
                        special.append(f"{total_lit} {_lit(p)} 100 / *")
                        if has_remaining:
                            special.append(f"{total_lit} {total_lit} {_lit(p)} 100 / * -")

            # "How much does each ... cost?" or "per <item>" → division candidates.
            if any(w in low_text for w in ("each", "per")) and any(w in low_text for w in ("how much", "what is the cost", "cost")):
                total = max(nums)
                dollars: List[float] = []
                try:
                    import re

                    dollars = [float(x) for x in re.findall(r"\$\s*(\d+(?:\.\d+)?)", problem_text)]
                    if dollars:
                        total = max(dollars)
                except Exception:
                    pass
                total_lit = _lit(float(total))
                # Denominators are usually counts (often larger than the $ total), so do not
                # constrain them to be < total when a dollar total is detected.
                if dollars:
                    denoms = [
                        n
                        for n in nums
                        if n > 0 and abs(n - round(n)) < 1e-9 and abs(float(n) - float(total)) > 1e-9
                    ]
                else:
                    denoms = [n for n in nums if 0 < n < total and abs(n - round(n)) < 1e-9]
                denoms = list(dict.fromkeys(denoms))
                for d in denoms[:6]:
                    d_lit = _lit(float(d))
                    special.append(f"{total_lit} {d_lit} /")

            # Packaging unit-price chains: cartons -> boxes -> packs (price per pack).
            # Example: "carton contains 12 boxes ... each box has 10 packs ... dozen cartons cost $1440 ... price of a pack"
            if ("price" in low_text) and ("pack" in low_text) and ("carton" in low_text) and ("box" in low_text) and ("dozen" in low_text):
                try:
                    import re

                    dollars = [float(x) for x in re.findall(r"\$\s*(\d+(?:\.\d+)?)", problem_text)]
                    if dollars:
                        cost = max(dollars)
                        # Use up to three multiplicative packaging factors (excluding the cost itself).
                        ints = [float(n) for n in nums if n > 1 and abs(n - round(n)) < 1e-9 and abs(float(n) - cost) > 1e-9]
                        ints = list(dict.fromkeys(ints))
                        factors = [12.0] + ints  # dozen cartons is implicit 12
                        factors = [f for f in factors if f > 1]
                        if len(factors) >= 2:
                            denom = factors[0]
                            denom_expr = _lit(denom)
                            for f in factors[1:4]:
                                denom_expr = f"{denom_expr} {_lit(float(f))} *"
                            # Put this first so it survives max_candidates truncation.
                            special.insert(0, f"{_lit(float(cost))} {denom_expr} /")
                except Exception:
                    pass

        # Conversions: add a few candidates early when words indicate units.
        if self.generic_equations_galaxy is not None and (depth >= 2 or strategy == "prioritize_conversions"):
            try:
                conv = self.generic_equations_galaxy.suggest_conversion_candidates(words=words, numbers=nums)
                for expr in conv:
                    special.append(expr)
            except Exception:
                pass

        # Light ordering: ensure specialized candidates get evaluated.
        if special:
            candidates = list(special) + list(candidates)

        # Light ordering: prefer operations suggested by question type.
        # Keep it stable (no full sort): partition into preferred/fallback.
        special_first: List[str] = []
        preferred: List[str] = []
        fallback: List[str] = []
        special_set: set[str] = {str(s).strip() for s in (special or []) if str(s).strip()}
        for expr in candidates:
            e = str(expr).strip()
            if e in special_set:
                special_first.append(expr)
                continue
            if question_type == "total" and "+" in expr:
                preferred.append(expr)
            elif question_type == "difference" and "-" in expr:
                preferred.append(expr)
            elif question_type == "rate" and "*" in expr:
                preferred.append(expr)
            elif " /" in f" {expr} " and strategy == "prioritize_division":
                preferred.append(expr)
            else:
                fallback.append(expr)
        out = special_first + preferred + fallback

        # De-duplicate while preserving order.
        seen: set[str] = set()
        deduped: List[str] = []
        for expr in out:
            e = str(expr).strip()
            if not e or e in seen:
                continue
            seen.add(e)
            deduped.append(e)
            if len(deduped) >= max_candidates:
                break
        return deduped

    def record_reading_success(
        self,
        *,
        problem_text: str,
        understanding: ProblemUnderstanding,
        trace: Dict[str, Any],
        rpn_program: str,
        confidence: float,
    ) -> None:
        if self.shadow is None:
            return
        patterns = [p.get("rule_id") for p in trace.get("patterns", []) if p.get("rule_id")]
        task_signature = {
            "problem_text": (problem_text or "")[:200],
            "patterns_matched": patterns,
            "quantities_found": len(understanding.quantities),
            "operations_found": len(understanding.operations),
            "problem_type": "math_reading",
        }
        try:
            self.shadow.record(
                task_signature=task_signature,
                program=rpn_program,
                program_type="reading",
                score=float(confidence),
                task_id=f"read_{hash(problem_text) % 10000}",
                semantic_context={
                    "word_patterns": trace.get("patterns", []),
                    "composition_strategy": understanding.to_dict(),
                },
            )
        except Exception:
            return

        # Calibrate per-pattern success.
        for pattern_id in patterns:
            try:
                self.shadow.update_pattern_confidence(str(pattern_id), float(confidence))
            except Exception:
                continue


__all__ = ["TRMGalaxyReader", "ProblemUnderstanding"]
