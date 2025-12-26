"""
Generic Equations Galaxy - cross-domain fundamentals for test-time compute.

These entries are intentionally NOT dataset-specific (e.g., not GSM8K-only).
They represent universal relationships that appear across physics, geometry,
economics, and everyday word problems, and can be instantiated into RPN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


GENERIC_EQUATIONS: Dict[str, Dict[str, Any]] = {
    # Rate & time
    "rate_time_distance": {
        "formula": "distance = rate × time",
        "rpn": "rate time *",
        "domains": ["physics", "math", "economics"],
        "isomorphic_to": [
            "money = rate × duration",
            "work = power × time",
            "distance = speed × time",
        ],
    },
    # Work & energy
    "work_rate_time": {
        "formula": "work = rate × time",
        "rpn": "rate time *",
        "domains": ["physics", "economics", "biology"],
        "isomorphic_to": [
            "earnings = wage × hours",
            "production = output_rate × time",
            "growth = growth_rate × time",
        ],
    },
    # Area
    "area_rectangle": {
        "formula": "area = length × width",
        "rpn": "length width *",
        "domains": ["geometry", "physics"],
        "isomorphic_to": [
            "cost = price × quantity",
            "total = count × value",
        ],
    },
    # Conversions
    "unit_conversion": {
        "formula": "target = source × conversion_factor",
        "rpn": "source factor *",
        "domains": ["physics", "chemistry", "math"],
        "examples": [
            "kilograms = grams × 0.001",
            "hours = minutes × (1/60)",
            "dollars = cents × 0.01",
        ],
    },
    # Distribution
    "fair_share": {
        "formula": "share = total / count",
        "rpn": "total count /",
        "domains": ["math", "economics"],
        "isomorphic_to": [
            "average = sum / count",
            "rate = total / time",
        ],
    },
    # Accumulation
    "total_from_parts": {
        "formula": "total = sum(parts)",
        "rpn": "part1 part2 + part3 + ...",
        "domains": ["math", "physics", "economics"],
    },
    # Difference
    "remaining": {
        "formula": "remaining = total - used",
        "rpn": "total used -",
        "domains": ["math", "economics", "physics"],
    },
}


@dataclass(frozen=True)
class GenericEquation:
    equation_id: str
    formula: str
    rpn: str
    domains: List[str] = field(default_factory=list)
    isomorphic_to: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


class GenericEquationGalaxy:
    """
    Lightweight "galaxy" of generic equations for test-time compute.

    This is used as a cross-domain knowledge source during exploration. It is not
    a solver; callers should instantiate equations into numeric RPN candidates.
    """

    def __init__(self, equations: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        src = equations or GENERIC_EQUATIONS
        self._equations: Dict[str, GenericEquation] = {}
        for eq_id, data in src.items():
            self._equations[str(eq_id)] = GenericEquation(
                equation_id=str(eq_id),
                formula=str(data.get("formula", "")),
                rpn=str(data.get("rpn", "")),
                domains=[str(d) for d in (data.get("domains") or [])],
                isomorphic_to=[str(x) for x in (data.get("isomorphic_to") or [])],
                examples=[str(x) for x in (data.get("examples") or [])],
            )

    def __len__(self) -> int:
        return len(self._equations)

    def get(self, equation_id: str) -> Optional[GenericEquation]:
        return self._equations.get(str(equation_id))

    def all(self) -> List[GenericEquation]:
        return list(self._equations.values())

    def query_by_domain(self, domain: str) -> List[GenericEquation]:
        d = str(domain)
        return [e for e in self._equations.values() if d in e.domains]

    def query_any_domain(self, domains: Sequence[str]) -> List[GenericEquation]:
        allowed = {str(d) for d in (domains or [])}
        if not allowed:
            return []
        return [e for e in self._equations.values() if any(d in allowed for d in e.domains)]

    def keywords(self) -> List[str]:
        out: List[str] = []
        for eq in self._equations.values():
            out.extend(eq.domains)
        return sorted(set(out))

    def suggest_conversion_candidates(
        self,
        *,
        words: Sequence[str],
        numbers: Sequence[float],
    ) -> List[str]:
        """
        Return RPN candidates for common unit conversions when unit words appear.

        This is intentionally small and general; it is a bridge until the
        full conversion graph exists in the Reality Galaxy.
        """
        ws = {str(w).lower() for w in (words or []) if w}
        nums = [float(n) for n in (numbers or []) if isinstance(n, (int, float))]
        if not nums:
            return []

        cands: List[str] = []

        def _add_mul(x: float, factor: float) -> None:
            cands.append(f"{x} {factor} *")

        def _add_div(x: float, denom: float) -> None:
            if denom != 0:
                cands.append(f"{x} {denom} /")

        for x in nums[:3]:
            # money
            if "cents" in ws and ("dollars" in ws or "$" in ws):
                _add_mul(x, 0.01)
            if "dollars" in ws and "cents" in ws:
                _add_mul(x, 100.0)

            # time
            if "minutes" in ws and "hours" in ws:
                _add_div(x, 60.0)
            if "hours" in ws and "minutes" in ws:
                _add_mul(x, 60.0)
            if "seconds" in ws and "minutes" in ws:
                _add_div(x, 60.0)
            if "minutes" in ws and "seconds" in ws:
                _add_mul(x, 60.0)

            # days/weeks
            if "weeks" in ws and "days" in ws:
                _add_mul(x, 7.0)
            if "days" in ws and "weeks" in ws:
                _add_div(x, 7.0)

        return cands


GENERIC_EQUATION_GALAXY = GenericEquationGalaxy()


__all__ = ["GENERIC_EQUATIONS", "GenericEquation", "GenericEquationGalaxy", "GENERIC_EQUATION_GALAXY"]

