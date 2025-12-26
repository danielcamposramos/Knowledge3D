"""
Select solving strategy based on problem classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SolvingStrategy:
    strategy_name: str
    steps: List[str]
    rpn_chains: List[str]
    required_vars: List[str]


STRATEGIES: Dict[tuple, SolvingStrategy] = {
    ("quadratic", "solve"): SolvingStrategy(
        strategy_name="quadratic_formula",
        steps=[
            "Compute discriminant",
            "Apply quadratic formula",
        ],
        rpn_chains=[
            "{b} 2 pow {a} {c} * 4 * - STORE_DISC",
            "{b} neg RECALL_DISC sqrt + {a} 2 * /",
            "{b} neg RECALL_DISC sqrt - {a} 2 * /",
        ],
        required_vars=["a", "b", "c"],
    ),
    ("quadratic", "factor"): SolvingStrategy(
        strategy_name="quadratic_factor",
        steps=["Find roots and return"],
        rpn_chains=[
            "{b} 2 pow {a} {c} * 4 * - sqrt STORE_SQRT_DISC",
            "{b} neg RECALL_SQRT_DISC + {a} 2 * /",
            "{b} neg RECALL_SQRT_DISC - {a} 2 * /",
        ],
        required_vars=["a", "b", "c"],
    ),
    ("linear", "solve"): SolvingStrategy(
        strategy_name="linear_isolate",
        steps=["Isolate variable"],
        rpn_chains=["{c} {b} - {a} /"],
        required_vars=["a", "b", "c"],
    ),
    ("system", "solve"): SolvingStrategy(
        strategy_name="system_cramer",
        steps=["Solve 2x2 system via Cramer"],
        rpn_chains=[
            "{c1} {b2} * {c2} {b1} * - {a1} {b2} * {a2} {b1} * - /",
            "{a1} {c2} * {a2} {c1} * - {a1} {b2} * {a2} {b1} * - /",
        ],
        required_vars=["a1", "b1", "c1", "a2", "b2", "c2"],
    ),
    ("sequence", "sum"): SolvingStrategy(
        strategy_name="sum_integers",
        steps=["Sum of first n integers"],
        rpn_chains=["{n} {n} 1 + * 2 /"],
        required_vars=["n"],
    ),
    ("sequence", "arithmetic"): SolvingStrategy(
        strategy_name="arithmetic_sequence",
        steps=["nth term", "sum"],
        rpn_chains=["{a} {n} 1 - {d} * +", "{n} {a} 2 * {n} 1 - {d} * + * 2 /"],
        required_vars=["a", "d", "n"],
    ),
    ("sequence", "geometric"): SolvingStrategy(
        strategy_name="geometric_sequence",
        steps=["nth term", "sum"],
        rpn_chains=["{a} {r} {n} 1 - pow *", "{a} 1 {r} {n} pow - * 1 {r} - /"],
        required_vars=["a", "r", "n"],
    ),
    ("number_theory", "gcd"): SolvingStrategy(
        strategy_name="euclidean_gcd",
        steps=["Compute gcd"],
        rpn_chains=["{a} {b} gcd"],
        required_vars=["a", "b"],
    ),
    ("number_theory", "lcm"): SolvingStrategy(
        strategy_name="lcm_from_gcd",
        steps=["LCM via gcd"],
        rpn_chains=["{a} {b} * {a} {b} gcd /"],
        required_vars=["a", "b"],
    ),
    ("number_theory", "modular"): SolvingStrategy(
        strategy_name="modular_arithmetic",
        steps=["Compute mod"],
        rpn_chains=["{a} {b} mod"],
        required_vars=["a", "b"],
    ),
    ("combinatorics", "count"): SolvingStrategy(
        strategy_name="combinatorics_basic",
        steps=["Combination default"],
        rpn_chains=["{n} {k} binomial"],
        required_vars=["n", "k"],
    ),
    ("expression", "evaluate"): SolvingStrategy(
        strategy_name="expression_eval",
        steps=["Direct evaluation"],
        rpn_chains=[""],
        required_vars=[],
    ),
}


class StrategySelector:
    def select(self, classification) -> Optional[SolvingStrategy]:
        key = (classification.problem_type, classification.subtype)
        return STRATEGIES.get(key, STRATEGIES.get(("expression", "evaluate")))

    def get_all_strategies(self) -> Dict[tuple, SolvingStrategy]:
        return STRATEGIES.copy()
