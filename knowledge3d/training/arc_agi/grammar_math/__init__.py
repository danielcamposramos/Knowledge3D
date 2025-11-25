"""Math grammar rules (arithmetic, algebra, calculus, linear algebra, geometry, statistics, logic)."""

from __future__ import annotations

from typing import List

from .arithmetic import ARITHMETIC_RULES
from .algebra import ALGEBRA_RULES
from .calculus import CALCULUS_RULES
from .linear_algebra import LINEAR_ALGEBRA_RULES
from .geometry import GEOMETRY_RULES
from .statistics import STATISTICS_RULES
from .logic import LOGIC_RULES


def get_math_rules() -> List:
    rules = []
    rules.extend(ARITHMETIC_RULES)
    rules.extend(ALGEBRA_RULES)
    rules.extend(CALCULUS_RULES)
    rules.extend(LINEAR_ALGEBRA_RULES)
    rules.extend(GEOMETRY_RULES)
    rules.extend(STATISTICS_RULES)
    rules.extend(LOGIC_RULES)
    return rules


__all__ = ["get_math_rules"]
