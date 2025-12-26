# CODEX BRIEFING: Algebra Galaxy - Multi-Step Symbolic Reasoning

**Date:** December 13, 2025
**Priority:** HIGH - Unlocks MATH/AMC-AIME performance
**Partner:** Claude (Architecture) → Codex (Implementation)

---

## Executive Summary

GSM8K works (90.92%) because: `pattern → single RPN → answer`

MATH/AMC-AIME fail (2-3%) because they need: `pattern → **multi-step reasoning** → answer`

**Solution:** Add an Algebra Galaxy layer that handles symbolic manipulation through RPN chains.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Problem Text                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Problem Classifier                            │
│  (Identifies: quadratic, linear, factorial, geometry, etc.)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Strategy Selector                             │
│  (Maps problem type → solving strategy → RPN chain)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Algebra Solver                                │
│  - Polynomial ops (factor, expand, roots)                       │
│  - Equation solver (linear, quadratic, systems)                 │
│  - Expression simplifier                                        │
│  - Multi-step RPN execution                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ModularRPNEngine (GPU)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Problem Classifier

**File:** `knowledge3d/training/math_benchmarks/problem_classifier.py`

```python
"""
Classify math problems to select solving strategy.
Pure pattern matching - no ML, no numpy.
"""

import re
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass


@dataclass
class ProblemClassification:
    problem_type: str           # quadratic, linear, factorial, geometry, etc.
    subtype: str               # solve, factor, simplify, evaluate
    variables: List[str]       # extracted variable names
    coefficients: Dict[str, float]  # extracted numeric values
    confidence: float          # 0.0 - 1.0


class ProblemClassifier:
    """Classify math problems using pattern matching."""

    # Problem type patterns (ordered by specificity)
    PATTERNS = [
        # Quadratic equations
        (r"x\^2|x²|x\s*\*\s*x", r"=\s*0", "quadratic", "solve"),
        (r"factor.*x\^2|factor.*x²", None, "quadratic", "factor"),
        (r"roots?\s+of|zeros?\s+of|solutions?\s+(?:to|of)", r"x\^2|x²", "quadratic", "solve"),

        # Linear equations
        (r"solve.*[a-z]\s*[+\-*/].*=", r"(?!.*x\^2|.*x²)", "linear", "solve"),
        (r"find\s+[a-z]\s+(?:if|when|such)", None, "linear", "solve"),

        # Systems of equations
        (r"system|simultaneous", r"equations?", "system", "solve"),
        (r"\{.*=.*\n.*=.*\}", None, "system", "solve"),

        # Polynomials
        (r"expand|multiply.*\(.*\)\s*\(.*\)", None, "polynomial", "expand"),
        (r"simplify", r"polynomial|expression", "polynomial", "simplify"),
        (r"degree\s+of|leading\s+coefficient", None, "polynomial", "analyze"),

        # Combinatorics (already handled by competition rules, but classify anyway)
        (r"how\s+many\s+ways|arrangements?|permutations?", None, "combinatorics", "count"),
        (r"choose|combinations?|select.*from", None, "combinatorics", "count"),
        (r"probability", None, "probability", "calculate"),

        # Number theory
        (r"divisors?|factors?\s+of\s+\d+", None, "number_theory", "divisors"),
        (r"gcd|greatest\s+common|hcf", None, "number_theory", "gcd"),
        (r"lcm|least\s+common\s+multiple", None, "number_theory", "lcm"),
        (r"prime|composite", None, "number_theory", "primality"),
        (r"remainder|mod|modulo", None, "number_theory", "modular"),

        # Sequences/Series
        (r"arithmetic\s+(?:sequence|series|progression)", None, "sequence", "arithmetic"),
        (r"geometric\s+(?:sequence|series|progression)", None, "sequence", "geometric"),
        (r"sum\s+of\s+(?:first\s+)?\d+", None, "sequence", "sum"),
        (r"nth\s+term|find.*term", None, "sequence", "term"),

        # Geometry
        (r"area|perimeter|circumference", None, "geometry", "measure"),
        (r"triangle|circle|rectangle|square", None, "geometry", "shape"),
        (r"angle|degree|radian", None, "geometry", "angle"),

        # Expressions (fallback)
        (r"evaluate|calculate|compute|find\s+the\s+value", None, "expression", "evaluate"),
        (r"simplify", None, "expression", "simplify"),
    ]

    def classify(self, problem_text: str) -> ProblemClassification:
        """Classify a math problem."""
        text_lower = problem_text.lower()

        for primary, secondary, ptype, subtype in self.PATTERNS:
            if re.search(primary, text_lower):
                if secondary is None or re.search(secondary, text_lower):
                    variables = self._extract_variables(problem_text)
                    coefficients = self._extract_coefficients(problem_text)
                    confidence = self._compute_confidence(text_lower, primary, secondary)
                    return ProblemClassification(
                        problem_type=ptype,
                        subtype=subtype,
                        variables=variables,
                        coefficients=coefficients,
                        confidence=confidence,
                    )

        # Fallback: expression evaluation
        return ProblemClassification(
            problem_type="expression",
            subtype="evaluate",
            variables=self._extract_variables(problem_text),
            coefficients=self._extract_coefficients(problem_text),
            confidence=0.3,
        )

    def _extract_variables(self, text: str) -> List[str]:
        """Extract single-letter variables."""
        # Find isolated letters that look like variables
        matches = re.findall(r'\b([a-z])\b(?!\s*[=<>].*[a-z]\b)', text.lower())
        return list(set(matches))

    def _extract_coefficients(self, text: str) -> Dict[str, float]:
        """Extract numeric coefficients from equations."""
        coeffs = {}

        # Match patterns like "3x^2", "-5x", "7"
        # ax^2 + bx + c = 0
        quad_match = re.search(
            r'(-?\d*\.?\d*)\s*x\^?2?\s*([+\-])\s*(\d*\.?\d*)\s*x\s*([+\-])\s*(\d+\.?\d*)',
            text
        )
        if quad_match:
            a = quad_match.group(1) or '1'
            coeffs['a'] = float(a) if a not in ('', '-') else (-1.0 if a == '-' else 1.0)
            sign_b = 1 if quad_match.group(2) == '+' else -1
            b = quad_match.group(3) or '1'
            coeffs['b'] = sign_b * (float(b) if b else 1.0)
            sign_c = 1 if quad_match.group(4) == '+' else -1
            coeffs['c'] = sign_c * float(quad_match.group(5))

        # Extract standalone numbers
        numbers = re.findall(r'\b(\d+\.?\d*)\b', text)
        for i, n in enumerate(numbers[:5]):  # First 5 numbers
            coeffs[f'n{i}'] = float(n)

        return coeffs

    def _compute_confidence(self, text: str, primary: str, secondary: str) -> float:
        """Estimate classification confidence."""
        primary_matches = len(re.findall(primary, text))
        confidence = min(0.5 + 0.1 * primary_matches, 0.9)
        if secondary and re.search(secondary, text):
            confidence = min(confidence + 0.1, 0.95)
        return confidence
```

---

## Part 2: Strategy Selector

**File:** `knowledge3d/training/math_benchmarks/strategy_selector.py`

```python
"""
Select solving strategy based on problem classification.
Maps problem types to RPN solving chains.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SolvingStrategy:
    strategy_name: str
    steps: List[str]           # Human-readable step descriptions
    rpn_chains: List[str]      # RPN programs for each step
    required_vars: List[str]   # Variables needed from classification


# Strategy registry - maps (problem_type, subtype) to solving strategy
STRATEGIES: Dict[tuple, SolvingStrategy] = {

    # ========== QUADRATIC ==========
    ("quadratic", "solve"): SolvingStrategy(
        strategy_name="quadratic_formula",
        steps=[
            "Extract coefficients a, b, c",
            "Compute discriminant: b² - 4ac",
            "Check discriminant sign",
            "Apply quadratic formula: (-b ± √Δ) / 2a",
        ],
        rpn_chains=[
            # Step 1: Store coefficients (assumed in vars a, b, c)
            "{a} STORE_A {b} STORE_B {c} STORE_C",
            # Step 2: Discriminant = b² - 4ac
            "{b} 2 pow {a} {c} * 4 * - STORE_DISC",
            # Step 3: Check discriminant (returns 1 if ≥ 0, else 0)
            "RECALL_DISC 0 >=",
            # Step 4: Roots (if discriminant ≥ 0)
            # x1 = (-b + √Δ) / 2a
            "{b} neg RECALL_DISC sqrt + {a} 2 * /",
            # x2 = (-b - √Δ) / 2a
            "{b} neg RECALL_DISC sqrt - {a} 2 * /",
        ],
        required_vars=["a", "b", "c"],
    ),

    ("quadratic", "factor"): SolvingStrategy(
        strategy_name="quadratic_factor",
        steps=[
            "Find roots using quadratic formula",
            "Express as a(x - r1)(x - r2)",
        ],
        rpn_chains=[
            # Find roots first
            "{b} 2 pow {a} {c} * 4 * - sqrt STORE_SQRT_DISC",
            "{b} neg RECALL_SQRT_DISC + {a} 2 * / STORE_R1",
            "{b} neg RECALL_SQRT_DISC - {a} 2 * / STORE_R2",
            # Return roots as answer (factored form implied)
            "RECALL_R1 RECALL_R2",
        ],
        required_vars=["a", "b", "c"],
    ),

    # ========== LINEAR ==========
    ("linear", "solve"): SolvingStrategy(
        strategy_name="linear_isolate",
        steps=[
            "Move variable terms to left side",
            "Move constants to right side",
            "Divide by coefficient",
        ],
        rpn_chains=[
            # ax + b = c  =>  x = (c - b) / a
            "{c} {b} - {a} /",
        ],
        required_vars=["a", "b", "c"],
    ),

    # ========== SYSTEMS ==========
    ("system", "solve"): SolvingStrategy(
        strategy_name="system_substitution",
        steps=[
            "Isolate one variable in first equation",
            "Substitute into second equation",
            "Solve for remaining variable",
            "Back-substitute to find first variable",
        ],
        rpn_chains=[
            # For: a1*x + b1*y = c1, a2*x + b2*y = c2
            # Using Cramer's rule: x = (c1*b2 - c2*b1) / (a1*b2 - a2*b1)
            "{c1} {b2} * {c2} {b1} * - {a1} {b2} * {a2} {b1} * - / STORE_X",
            # y = (a1*c2 - a2*c1) / (a1*b2 - a2*b1)
            "{a1} {c2} * {a2} {c1} * - {a1} {b2} * {a2} {b1} * - / STORE_Y",
            "RECALL_X RECALL_Y",
        ],
        required_vars=["a1", "b1", "c1", "a2", "b2", "c2"],
    ),

    # ========== POLYNOMIAL ==========
    ("polynomial", "expand"): SolvingStrategy(
        strategy_name="foil_expand",
        steps=[
            "Apply FOIL: (a+b)(c+d) = ac + ad + bc + bd",
        ],
        rpn_chains=[
            # (a+b)(c+d) = ac + ad + bc + bd
            "{a} {c} * {a} {d} * + {b} {c} * + {b} {d} * +",
        ],
        required_vars=["a", "b", "c", "d"],
    ),

    # ========== SEQUENCES ==========
    ("sequence", "arithmetic"): SolvingStrategy(
        strategy_name="arithmetic_sequence",
        steps=[
            "Identify first term (a) and common difference (d)",
            "Apply formula: nth term = a + (n-1)d",
            "Sum formula: S_n = n(a + a_n)/2 = n(2a + (n-1)d)/2",
        ],
        rpn_chains=[
            # nth term
            "{a} {n} 1 - {d} * +",
            # Sum of n terms
            "{n} {a} 2 * {n} 1 - {d} * + * 2 /",
        ],
        required_vars=["a", "d", "n"],
    ),

    ("sequence", "geometric"): SolvingStrategy(
        strategy_name="geometric_sequence",
        steps=[
            "Identify first term (a) and common ratio (r)",
            "nth term = a * r^(n-1)",
            "Sum = a(1 - r^n)/(1 - r) if r ≠ 1",
        ],
        rpn_chains=[
            # nth term
            "{a} {r} {n} 1 - pow *",
            # Sum of n terms
            "{a} 1 {r} {n} pow - * 1 {r} - /",
        ],
        required_vars=["a", "r", "n"],
    ),

    ("sequence", "sum"): SolvingStrategy(
        strategy_name="sum_integers",
        steps=[
            "Sum of first n integers: n(n+1)/2",
        ],
        rpn_chains=[
            "{n} {n} 1 + * 2 /",
        ],
        required_vars=["n"],
    ),

    # ========== NUMBER THEORY ==========
    ("number_theory", "gcd"): SolvingStrategy(
        strategy_name="euclidean_gcd",
        steps=[
            "Apply Euclidean algorithm iteratively",
        ],
        rpn_chains=[
            # Use built-in GCD opcode (we need to add this)
            "{a} {b} GCD",
        ],
        required_vars=["a", "b"],
    ),

    ("number_theory", "lcm"): SolvingStrategy(
        strategy_name="lcm_from_gcd",
        steps=[
            "LCM(a,b) = a*b / GCD(a,b)",
        ],
        rpn_chains=[
            "{a} {b} * {a} {b} GCD /",
        ],
        required_vars=["a", "b"],
    ),

    ("number_theory", "modular"): SolvingStrategy(
        strategy_name="modular_arithmetic",
        steps=[
            "Compute a mod b",
        ],
        rpn_chains=[
            "{a} {b} mod",
        ],
        required_vars=["a", "b"],
    ),

    # ========== COMBINATORICS ==========
    ("combinatorics", "count"): SolvingStrategy(
        strategy_name="combinatorics_basic",
        steps=[
            "Identify if permutation or combination",
            "Apply P(n,k) = n!/(n-k)! or C(n,k) = n!/(k!(n-k)!)",
        ],
        rpn_chains=[
            # Default to combination
            "{n} {k} binomial",
        ],
        required_vars=["n", "k"],
    ),

    # ========== EXPRESSION (fallback) ==========
    ("expression", "evaluate"): SolvingStrategy(
        strategy_name="direct_evaluation",
        steps=[
            "Parse expression and evaluate",
        ],
        rpn_chains=[
            # Will be filled by expression parser
            "",
        ],
        required_vars=[],
    ),
}


class StrategySelector:
    """Select solving strategy based on problem classification."""

    def select(self, classification) -> Optional[SolvingStrategy]:
        """Select strategy for classified problem."""
        key = (classification.problem_type, classification.subtype)

        if key in STRATEGIES:
            strategy = STRATEGIES[key]
            # Check if we have required variables
            missing = [v for v in strategy.required_vars
                      if v not in classification.coefficients]
            if missing and classification.confidence > 0.5:
                # Try to infer missing variables
                pass
            return strategy

        # Fallback to expression evaluation
        return STRATEGIES.get(("expression", "evaluate"))

    def get_all_strategies(self) -> Dict[tuple, SolvingStrategy]:
        """Return all registered strategies."""
        return STRATEGIES.copy()
```

---

## Part 3: Algebra Solver

**File:** `knowledge3d/training/math_benchmarks/algebra_solver.py`

```python
"""
Multi-step algebraic problem solver.
Chains RPN programs through ModularRPNEngine.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from knowledge3d.training.math_benchmarks.problem_classifier import (
    ProblemClassifier,
    ProblemClassification,
)
from knowledge3d.training.math_benchmarks.strategy_selector import (
    StrategySelector,
    SolvingStrategy,
)


class AlgebraSolver:
    """
    Solve algebraic problems through multi-step RPN execution.

    Pipeline:
    1. Classify problem
    2. Select strategy
    3. Execute RPN chains
    4. Return result
    """

    def __init__(self, rpn_engine=None):
        self.classifier = ProblemClassifier()
        self.strategy_selector = StrategySelector()
        self._rpn_engine = rpn_engine
        self._execution_trace: List[Dict[str, Any]] = []

    @property
    def rpn_engine(self):
        """Lazy-load RPN engine."""
        if self._rpn_engine is None:
            from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
            self._rpn_engine = ModularRPNEngine()
        return self._rpn_engine

    def solve(self, problem_text: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Solve a math problem.

        Returns:
            (answer, metadata) where metadata contains classification,
            strategy used, and execution trace.
        """
        self._execution_trace = []

        # Step 1: Classify
        classification = self.classifier.classify(problem_text)
        self._trace("classify", {"classification": classification})

        # Step 2: Select strategy
        strategy = self.strategy_selector.select(classification)
        if strategy is None:
            return None, {"error": "no_strategy", "classification": classification}
        self._trace("strategy", {"strategy": strategy.strategy_name})

        # Step 3: Extract/infer variables
        variables = self._extract_variables(problem_text, classification, strategy)
        self._trace("variables", {"variables": variables})

        # Step 4: Execute RPN chains
        results = []
        for i, rpn_template in enumerate(strategy.rpn_chains):
            if not rpn_template.strip():
                continue

            # Substitute variables into RPN template
            rpn_program = self._substitute_variables(rpn_template, variables)
            self._trace(f"rpn_step_{i}", {"template": rpn_template, "program": rpn_program})

            # Parse and execute
            tokens = self._parse_rpn(rpn_program)
            if tokens:
                try:
                    result = self.rpn_engine.evaluate(tokens)
                    results.append(result)
                    self._trace(f"result_{i}", {"result": result})

                    # Store intermediate results
                    if "STORE_" in rpn_template:
                        var_match = re.search(r'STORE_(\w+)', rpn_template)
                        if var_match:
                            variables[var_match.group(1).lower()] = result
                except Exception as e:
                    self._trace(f"error_{i}", {"error": str(e)})

        # Return final result
        final_answer = results[-1] if results else None
        metadata = {
            "classification": {
                "type": classification.problem_type,
                "subtype": classification.subtype,
                "confidence": classification.confidence,
            },
            "strategy": strategy.strategy_name,
            "variables": variables,
            "trace": self._execution_trace,
        }

        return final_answer, metadata

    def _extract_variables(
        self,
        problem_text: str,
        classification: ProblemClassification,
        strategy: SolvingStrategy,
    ) -> Dict[str, float]:
        """Extract or infer required variables."""
        variables = dict(classification.coefficients)

        # Try to extract more variables from text
        # Pattern: "a = 5" or "let a = 5" or "where a = 5"
        assignments = re.findall(
            r'\b([a-z])\s*=\s*(-?\d+\.?\d*)',
            problem_text.lower()
        )
        for var, val in assignments:
            variables[var] = float(val)

        # Extract from common phrases
        # "sum of first N integers" -> n = N
        n_match = re.search(r'(?:first|sum of)\s+(\d+)', problem_text.lower())
        if n_match and 'n' not in variables:
            variables['n'] = float(n_match.group(1))

        # "choose K from N" -> n, k
        choose_match = re.search(
            r'(?:choose|select|pick)\s+(\d+)\s+(?:from|out of)\s+(\d+)',
            problem_text.lower()
        )
        if choose_match:
            variables['k'] = float(choose_match.group(1))
            variables['n'] = float(choose_match.group(2))

        # Quadratic: extract from "x² - 5x + 6 = 0"
        quad_match = re.search(
            r'x[²\^2]\s*([+\-])\s*(\d+)\s*x\s*([+\-])\s*(\d+)',
            problem_text
        )
        if quad_match:
            variables['a'] = 1.0
            sign_b = 1 if quad_match.group(1) == '+' else -1
            variables['b'] = sign_b * float(quad_match.group(2))
            sign_c = 1 if quad_match.group(3) == '+' else -1
            variables['c'] = sign_c * float(quad_match.group(4))

        return variables

    def _substitute_variables(self, template: str, variables: Dict[str, float]) -> str:
        """Replace {var} placeholders with values."""
        result = template
        for var, val in variables.items():
            result = result.replace(f'{{{var}}}', str(val))
        return result

    def _parse_rpn(self, program: str) -> List[Any]:
        """Parse RPN program string into token list."""
        tokens = []
        for token in program.split():
            # Skip STORE/RECALL for now (handled separately)
            if token.startswith('STORE_') or token.startswith('RECALL_'):
                continue

            # Try to parse as number
            try:
                tokens.append(float(token))
            except ValueError:
                # It's an operator
                tokens.append(token.lower())

        return tokens

    def _trace(self, step: str, data: Dict[str, Any]) -> None:
        """Record execution trace."""
        self._execution_trace.append({"step": step, **data})

    def solve_batch(self, problems: List[str]) -> List[Tuple[Any, Dict[str, Any]]]:
        """Solve multiple problems."""
        return [self.solve(p) for p in problems]
```

---

## Part 4: New Opcodes Needed

Add these to `rpn_opcodes.py` and implement in CUDA kernel:

```python
# Algebra opcodes
OP_GCD = 0xD8           # Euclidean GCD
OP_POLY_EVAL = 0xD9     # Evaluate polynomial at x
OP_POLY_ROOTS = 0xDA    # Find polynomial roots (quadratic)
OP_NEG = 0xDB           # Negate top of stack
OP_GTE = 0xDC           # Greater than or equal
```

**CUDA implementations for kernel:**

```cuda
case 0xD8: {  // GCD - Euclidean algorithm
    float b_f = 0.0f;
    float a_f = 0.0f;
    if (!pop_scalar(stack, stack_size, b_f, error_code)) break;
    if (!pop_scalar(stack, stack_size, a_f, error_code)) break;
    int a = (int)fabsf(a_f);
    int b = (int)fabsf(b_f);
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    push_scalar(stack, stack_size, (float)a, error_code);
    break;
}

case 0xDB: {  // NEG - negate
    float x = 0.0f;
    if (!pop_scalar(stack, stack_size, x, error_code)) break;
    push_scalar(stack, stack_size, -x, error_code);
    break;
}

case 0xDC: {  // GTE - greater than or equal
    float rhs = 0.0f;
    float lhs = 0.0f;
    if (!pop_scalar(stack, stack_size, rhs, error_code)) break;
    if (!pop_scalar(stack, stack_size, lhs, error_code)) break;
    push_scalar(stack, stack_size, lhs >= rhs ? 1.0f : 0.0f, error_code);
    break;
}
```

---

## Part 5: Integration with Pipeline

**Update `sovereign_math_pipeline.py`:**

```python
from knowledge3d.training.math_benchmarks.algebra_solver import AlgebraSolver

class SovereignMathPipeline:
    def __init__(self):
        # ... existing init ...
        self.algebra_solver = AlgebraSolver()

    def solve_problem(self, problem: Dict[str, Any]) -> Any:
        """Enhanced problem solving with algebra support."""
        text = problem.get("problem", problem.get("question", ""))

        # Try word problem solver first (fast path for GSM8K)
        result = self.word_solver.solve(text)
        if result is not None:
            return result

        # Try algebra solver for competition math
        answer, metadata = self.algebra_solver.solve(text)
        if answer is not None and metadata.get("classification", {}).get("confidence", 0) > 0.5:
            return answer

        # Fallback to direct proceduralizer
        return self.proceduralizer.proceduralize(text)
```

---

## Testing

```bash
PYTHONPATH=. python3 -c "
from knowledge3d.training.math_benchmarks.algebra_solver import AlgebraSolver

solver = AlgebraSolver()

tests = [
    'Solve x² - 5x + 6 = 0',
    'Find the sum of first 100 positive integers',
    'What is the GCD of 24 and 36?',
    'How many ways to choose 3 from 10?',
    'Solve 2x + 5 = 15',
]

for t in tests:
    answer, meta = solver.solve(t)
    print(f'{t[:45]:45s} → {answer} ({meta[\"strategy\"]})')
"
```

**Expected:**
```
Solve x² - 5x + 6 = 0                         → 2.0 (quadratic_formula)
Find the sum of first 100 positive integers   → 5050.0 (sum_integers)
What is the GCD of 24 and 36?                 → 12.0 (euclidean_gcd)
How many ways to choose 3 from 10?            → 120.0 (combinatorics_basic)
Solve 2x + 5 = 15                             → 5.0 (linear_isolate)
```

---

## Success Criteria

| Metric | Before | Target |
|--------|--------|--------|
| MATH | 2.15% | 15%+ |
| Omni-MATH | 9.06% | 20%+ |
| AMC-AIME | 2.85% | 10%+ |

---

## Architecture Notes

- **NO numpy, NO cupy** - Pure Python pattern matching + RPN execution
- Multi-step execution through chained RPN programs
- Variables stored/recalled through dictionary (Python-side)
- GPU execution only for arithmetic (ModularRPNEngine)
- Extensible: Add more strategies to `STRATEGIES` dict

---

**Codex:** Implement Parts 1-5, add the new opcodes to the CUDA kernel, compile PTX, and run the test script. Then re-run math benchmarks and report scores.
