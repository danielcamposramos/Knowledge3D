# CODEX BRIEFING: Competition Math Grammar Rules

**Date:** December 13, 2025
**Priority:** HIGH - Grammar is the bottleneck, opcodes now work
**Partner:** Claude (Architecture) → Codex (Implementation)

---

## Executive Summary

Opcodes verified working: `factorial`, `binomial`, `gamma`, `beta` execute correctly on GPU.

**Current scores:**
- GSM8K: 90.92% ✅ (word problem rules work)
- MATH: 2.17% ❌
- Omni-MATH: 10.05% ❌
- AMC-AIME: 2.99% ❌

**Root cause:** The grammar rules don't recognize competition math patterns to invoke the new opcodes.

---

## Task 1: Fix Existing Binomial Rule

**File:** `knowledge3d/training/arc_agi/math_grammar_rules.py`

**Current (inefficient):**
```python
GrammarRule(
    rule_id="sym_binomial",
    language="math",
    pattern="\\binom{n}{k}",
    rpn_program="n factorial k factorial n k - factorial * /",  # 3 factorials!
    ...
)
```

**Fix to use native opcode:**
```python
GrammarRule(
    rule_id="sym_binomial",
    language="math",
    pattern="\\binom{n}{k}",
    rpn_program="n k binomial",  # Single opcode!
    ...
)
```

---

## Task 2: Add COMPETITION_MATH_RULES

Add a new rule set after SYMBOLIC_RULES that handles natural language competition math patterns:

```python
# =============================================================================
# COMPETITION MATH RULES (AMC, AIME, MATH dataset patterns)
# =============================================================================
COMPETITION_MATH_RULES = [
    # ---------- PERMUTATIONS ----------
    GrammarRule(
        rule_id="comp_permutation_arrange",
        language="english",
        pattern=r"(?:how many|number of) (?:ways|arrangements?) (?:to |can )?(arrange|order|line up) (\d+)",
        rpn_program="{n} factorial",
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[
            {"input": "How many ways to arrange 5 people?", "output": "120"},
            {"input": "Number of arrangements of 7 items", "output": "5040"},
        ],
    ),
    GrammarRule(
        rule_id="comp_permutation_pnk",
        language="english",
        pattern=r"(?:how many|number of) (?:ways|permutations?) (?:to )?(?:choose|select|pick) (\d+) from (\d+) (?:where |when )?order matters",
        rpn_program="{n} factorial {n} {k} - factorial /",  # P(n,k) = n!/(n-k)!
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[
            {"input": "How many ways to choose 3 from 10 where order matters?", "output": "720"},
        ],
    ),

    # ---------- COMBINATIONS ----------
    GrammarRule(
        rule_id="comp_combination_choose",
        language="english",
        pattern=r"(?:how many|number of) (?:ways|combinations?) (?:to )?(?:choose|select|pick) (\d+) from (\d+)",
        rpn_program="{n} {k} binomial",
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[
            {"input": "How many ways to choose 3 from 10?", "output": "120"},
            {"input": "Number of combinations selecting 5 from 20", "output": "15504"},
        ],
    ),
    GrammarRule(
        rule_id="comp_combination_committee",
        language="english",
        pattern=r"(?:how many|number of) (?:ways|committees?) (?:to )?(?:form|create|select) a (?:committee|group|team) of (\d+) from (\d+)",
        rpn_program="{n} {k} binomial",
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[
            {"input": "How many ways to form a committee of 4 from 12?", "output": "495"},
        ],
    ),
    GrammarRule(
        rule_id="comp_combination_handshakes",
        language="english",
        pattern=r"(\d+) people (?:shake hands|meet|greet)",
        rpn_program="{n} 2 binomial",  # C(n,2) handshakes
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[
            {"input": "10 people shake hands, how many handshakes?", "output": "45"},
        ],
    ),

    # ---------- BINOMIAL THEOREM ----------
    GrammarRule(
        rule_id="comp_binomial_coefficient",
        language="english",
        pattern=r"coefficient of x\^(\d+) in \(1\+x\)\^(\d+)",
        rpn_program="{n} {k} binomial",  # Coeff of x^k in (1+x)^n is C(n,k)
        domain="math_algebra",
        symbol_refs=[],
        examples=[
            {"input": "coefficient of x^3 in (1+x)^10", "output": "120"},
        ],
    ),
    GrammarRule(
        rule_id="comp_binomial_expansion_term",
        language="english",
        pattern=r"(\d+)(?:th|st|nd|rd) term (?:in |of )?\(.*\)\^(\d+)",
        rpn_program="{n} {k} 1 - binomial",  # kth term uses C(n,k-1)
        domain="math_algebra",
        symbol_refs=[],
        examples=[
            {"input": "4th term in (a+b)^7", "output": "35"},  # C(7,3)
        ],
    ),

    # ---------- PROBABILITY ----------
    GrammarRule(
        rule_id="comp_probability_exactly_k",
        language="english",
        pattern=r"probability (?:of )?(?:exactly )?(\d+) successes? in (\d+) trials? with p=([0-9.]+)",
        rpn_program="{n} {k} binomial {p} {k} pow {p} 1 swap - {n} {k} - pow * *",  # C(n,k)*p^k*(1-p)^(n-k)
        domain="math_probability",
        symbol_refs=[],
        examples=[
            {"input": "probability of exactly 3 successes in 10 trials with p=0.5", "output": "0.1172"},
        ],
    ),

    # ---------- DIVISIBILITY / NUMBER THEORY ----------
    GrammarRule(
        rule_id="comp_divisors_count",
        language="english",
        pattern=r"(?:how many|number of) (?:positive )?divisors of (\d+)",
        rpn_program="{n} PRIME_FACTORIZE { 1 + } map *",  # Product of (e_i + 1)
        domain="math_number_theory",
        symbol_refs=[],
        examples=[
            {"input": "How many divisors of 12?", "output": "6"},  # 1,2,3,4,6,12
        ],
    ),
    GrammarRule(
        rule_id="comp_gcd",
        language="english",
        pattern=r"(?:gcd|greatest common divisor|hcf) (?:of )?(\d+) and (\d+)",
        rpn_program="{a} {b} GCD",
        domain="math_number_theory",
        symbol_refs=[],
        examples=[
            {"input": "GCD of 24 and 36", "output": "12"},
        ],
    ),
    GrammarRule(
        rule_id="comp_lcm",
        language="english",
        pattern=r"(?:lcm|least common multiple) (?:of )?(\d+) and (\d+)",
        rpn_program="{a} {b} * {a} {b} GCD /",  # LCM = a*b/GCD(a,b)
        domain="math_number_theory",
        symbol_refs=[],
        examples=[
            {"input": "LCM of 4 and 6", "output": "12"},
        ],
    ),

    # ---------- SEQUENCES ----------
    GrammarRule(
        rule_id="comp_arithmetic_sum",
        language="english",
        pattern=r"sum (?:of )?(?:first )?(\d+) (?:positive )?integers",
        rpn_program="{n} {n} 1 + * 2 /",  # n(n+1)/2
        domain="math_algebra",
        symbol_refs=[],
        examples=[
            {"input": "sum of first 100 integers", "output": "5050"},
        ],
    ),
    GrammarRule(
        rule_id="comp_geometric_sum",
        language="english",
        pattern=r"sum (?:of )?geometric series (?:with )?a=([0-9.]+),? r=([0-9.]+),? n=(\d+)",
        rpn_program="{a} 1 {r} {n} pow - * 1 {r} - /",  # a(1-r^n)/(1-r)
        domain="math_algebra",
        symbol_refs=[],
        examples=[
            {"input": "sum of geometric series with a=1, r=2, n=5", "output": "31"},
        ],
    ),

    # ---------- LATEX NOTATION (for MATH dataset) ----------
    GrammarRule(
        rule_id="comp_latex_binom",
        language="math",
        pattern=r"\\binom\{(\d+)\}\{(\d+)\}",
        rpn_program="{n} {k} binomial",
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[
            {"input": "\\binom{10}{3}", "output": "120"},
        ],
    ),
    GrammarRule(
        rule_id="comp_latex_frac",
        language="math",
        pattern=r"\\frac\{([^}]+)\}\{([^}]+)\}",
        rpn_program="{num} {denom} /",
        domain="math_algebra",
        symbol_refs=[],
        examples=[
            {"input": "\\frac{6}{2}", "output": "3"},
        ],
    ),
    GrammarRule(
        rule_id="comp_latex_sqrt",
        language="math",
        pattern=r"\\sqrt\{([^}]+)\}",
        rpn_program="{x} sqrt",
        domain="math_algebra",
        symbol_refs=[],
        examples=[
            {"input": "\\sqrt{16}", "output": "4"},
        ],
    ),
    GrammarRule(
        rule_id="comp_latex_factorial",
        language="math",
        pattern=r"(\d+)!",
        rpn_program="{n} factorial",
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[
            {"input": "5!", "output": "120"},
        ],
    ),

    # ---------- MODULAR ARITHMETIC ----------
    GrammarRule(
        rule_id="comp_mod_remainder",
        language="english",
        pattern=r"(?:what is )?(?:the )?remainder (?:when )?(\d+) (?:is )?divided by (\d+)",
        rpn_program="{a} {b} mod",
        domain="math_number_theory",
        symbol_refs=[],
        examples=[
            {"input": "remainder when 17 divided by 5", "output": "2"},
        ],
    ),
    GrammarRule(
        rule_id="comp_mod_congruence",
        language="math",
        pattern=r"(\d+) ≡ \? \(mod (\d+)\)",
        rpn_program="{a} {m} mod",
        domain="math_number_theory",
        symbol_refs=[],
        examples=[
            {"input": "17 ≡ ? (mod 5)", "output": "2"},
        ],
    ),
]
```

---

## Task 3: Register COMPETITION_MATH_RULES

At the bottom of `math_grammar_rules.py`, add to ALL_MATH_RULES:

```python
ALL_MATH_RULES = list(
    CALCULUS_RULES
    + SET_THEORY_RULES
    + LOGIC_RULES
    + STATISTICS_RULES
    + FINANCE_RULES
    + WORD_PROBLEM_RULES
    + SYMBOLIC_RULES
    + COMPETITION_MATH_RULES  # ADD THIS
)
```

---

## Task 4: Update WordProblemSolver to Use Competition Rules

In `knowledge3d/training/math_benchmarks/word_problem_solver.py`, ensure it loads COMPETITION_MATH_RULES:

```python
from knowledge3d.training.arc_agi.math_grammar_rules import (
    WORD_PROBLEM_RULES,
    SYMBOLIC_RULES,
    COMPETITION_MATH_RULES,  # ADD THIS
)

class WordProblemSolver:
    def __init__(self):
        self.rules = WORD_PROBLEM_RULES + SYMBOLIC_RULES + COMPETITION_MATH_RULES
```

---

## Task 5: Add Regex-Based Pattern Matching

The current matcher may be too literal. Add a regex-capable matching method:

```python
import re

def match_rule(self, problem_text: str) -> Optional[Tuple[GrammarRule, Dict[str, Any]]]:
    """Match problem text against grammar rules, extracting captured groups."""
    text_lower = problem_text.lower()

    for rule in self.rules:
        try:
            match = re.search(rule.pattern, text_lower, re.IGNORECASE)
            if match:
                # Extract captured groups as variables
                groups = match.groups()
                variables = {}

                # Map groups to common variable names
                var_names = ['n', 'k', 'a', 'b', 'p', 'r', 'x', 'num', 'denom', 'm']
                for i, val in enumerate(groups):
                    if i < len(var_names):
                        try:
                            variables[var_names[i]] = float(val)
                        except (ValueError, TypeError):
                            variables[var_names[i]] = val

                return rule, variables
        except re.error:
            # Pattern is literal string, do exact match
            if rule.pattern.lower() in text_lower:
                return rule, {}

    return None
```

---

## Testing

After implementation:

```bash
PYTHONPATH=. python3 -c "
from knowledge3d.training.math_benchmarks.word_problem_solver import WordProblemSolver
solver = WordProblemSolver()

# Test competition patterns
tests = [
    'How many ways to arrange 5 people?',
    'How many ways to choose 3 from 10?',
    'What is the coefficient of x^3 in (1+x)^10?',
    'Sum of first 100 integers',
    'Remainder when 17 divided by 5',
    '\\\\binom{10}{3}',
]

for t in tests:
    result = solver.solve(t)
    print(f'{t[:40]:40s} → {result}')
"
```

**Expected:**
```
How many ways to arrange 5 people?       → 120.0
How many ways to choose 3 from 10?       → 120.0
coefficient of x^3 in (1+x)^10           → 120.0
Sum of first 100 integers                → 5050.0
Remainder when 17 divided by 5           → 2.0
\binom{10}{3}                            → 120.0
```

---

## Success Criteria

| Metric | Before | Target |
|--------|--------|--------|
| MATH | 2.17% | 10%+ |
| Omni-MATH | 10.05% | 15%+ |
| AMC-AIME | 2.99% | 8%+ |

---

## Architecture Notes

- Rules use regex patterns with capture groups `(\d+)` for number extraction
- RPN programs use `{n}`, `{k}` placeholders that get substituted with captured values
- Grammar rules leverage the **native opcodes** (`factorial`, `binomial`, `mod`) not manual computation
- NO numpy/cupy - pure RPN execution through ModularRPNEngine

---

**Codex:** Implement these grammar rules, update the solver, and re-run benchmarks. Report scores.
