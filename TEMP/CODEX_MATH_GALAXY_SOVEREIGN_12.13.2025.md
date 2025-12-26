# CODEX BRIEFING: Math Galaxy - SOVEREIGN Approach

**Date:** December 13, 2025
**Priority:** CRITICAL - Course correction
**Partner:** Claude (Architecture) → Codex (Implementation)

---

## What Went Wrong

We treated K3D as a "fancy calculator with Python preprocessing":
```
Input → Python regex → Extract coefficients → Build RPN externally → GPU
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         WRONG. This is NOT sovereign. External preprocessing.
```

**The architecture says (CLAUDE.md, Dual Client Reality):**
- Everything is **procedural RPN + metadata**
- Character Galaxy has glyphs with meanings
- Grammar Galaxy has transformation rules
- The Galaxy IS the model's knowledge
- NO external preprocessing

---

## The Correct Approach

```
Input → Grammar Galaxy (pattern match) → RPN composition → GPU execution
        Math Galaxy (symbols with RPN templates)

        The Galaxy stores logic. Weights = rules. Memory = symbols.
```

---

## Task 1: Populate Math Symbol Galaxy

**File:** `knowledge3d/training/arc_agi/math_symbol_galaxy.py`

Add LaTeX symbols as Galaxy entries with RPN templates:

```python
"""
Math Symbol Galaxy - LaTeX symbols with RPN meanings.

Each symbol is a Galaxy entry that the model can "see" and compose into RPN.
This is sovereign: the symbols ARE the model's math knowledge.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MathSymbol:
    """A math symbol with its RPN meaning."""
    symbol: str              # The symbol/command (e.g., "\\frac", "!", "^")
    category: str            # operator, function, constant, delimiter
    arity: int               # Number of arguments (0=constant, 1=unary, 2=binary)
    rpn_template: str        # RPN program template with {0}, {1} placeholders
    precedence: int          # For infix operators (higher = binds tighter)
    associativity: str       # "left", "right", or "none"
    description: str         # Human-readable description


# =============================================================================
# MATH SYMBOL GALAXY - Core entries
# =============================================================================

MATH_SYMBOLS = [
    # ===== ARITHMETIC OPERATORS =====
    MathSymbol(
        symbol="+",
        category="operator",
        arity=2,
        rpn_template="{0} {1} add",
        precedence=1,
        associativity="left",
        description="Addition",
    ),
    MathSymbol(
        symbol="-",
        category="operator",
        arity=2,
        rpn_template="{0} {1} sub",
        precedence=1,
        associativity="left",
        description="Subtraction",
    ),
    MathSymbol(
        symbol="*",
        category="operator",
        arity=2,
        rpn_template="{0} {1} mul",
        precedence=2,
        associativity="left",
        description="Multiplication",
    ),
    MathSymbol(
        symbol="/",
        category="operator",
        arity=2,
        rpn_template="{0} {1} div",
        precedence=2,
        associativity="left",
        description="Division",
    ),
    MathSymbol(
        symbol="^",
        category="operator",
        arity=2,
        rpn_template="{0} {1} pow",
        precedence=3,
        associativity="right",
        description="Exponentiation",
    ),
    MathSymbol(
        symbol="!",
        category="operator",
        arity=1,
        rpn_template="{0} factorial",
        precedence=4,
        associativity="left",
        description="Factorial (postfix)",
    ),
    MathSymbol(
        symbol="%",
        category="operator",
        arity=2,
        rpn_template="{0} {1} mod",
        precedence=2,
        associativity="left",
        description="Modulo",
    ),

    # ===== LATEX COMMANDS =====
    MathSymbol(
        symbol="\\frac",
        category="function",
        arity=2,
        rpn_template="{0} {1} div",
        precedence=0,
        associativity="none",
        description="Fraction a/b",
    ),
    MathSymbol(
        symbol="\\binom",
        category="function",
        arity=2,
        rpn_template="{0} {1} binomial",
        precedence=0,
        associativity="none",
        description="Binomial coefficient C(n,k)",
    ),
    MathSymbol(
        symbol="\\sqrt",
        category="function",
        arity=1,
        rpn_template="{0} sqrt",
        precedence=0,
        associativity="none",
        description="Square root",
    ),
    MathSymbol(
        symbol="\\sin",
        category="function",
        arity=1,
        rpn_template="{0} sin",
        precedence=0,
        associativity="none",
        description="Sine",
    ),
    MathSymbol(
        symbol="\\cos",
        category="function",
        arity=1,
        rpn_template="{0} cos",
        precedence=0,
        associativity="none",
        description="Cosine",
    ),
    MathSymbol(
        symbol="\\tan",
        category="function",
        arity=1,
        rpn_template="{0} tan",
        precedence=0,
        associativity="none",
        description="Tangent",
    ),
    MathSymbol(
        symbol="\\log",
        category="function",
        arity=1,
        rpn_template="{0} log",
        precedence=0,
        associativity="none",
        description="Natural logarithm",
    ),
    MathSymbol(
        symbol="\\ln",
        category="function",
        arity=1,
        rpn_template="{0} log",
        precedence=0,
        associativity="none",
        description="Natural logarithm",
    ),
    MathSymbol(
        symbol="\\exp",
        category="function",
        arity=1,
        rpn_template="{0} exp",
        precedence=0,
        associativity="none",
        description="Exponential e^x",
    ),
    MathSymbol(
        symbol="\\abs",
        category="function",
        arity=1,
        rpn_template="{0} abs",
        precedence=0,
        associativity="none",
        description="Absolute value",
    ),
    MathSymbol(
        symbol="\\floor",
        category="function",
        arity=1,
        rpn_template="{0} floor",
        precedence=0,
        associativity="none",
        description="Floor function",
    ),
    MathSymbol(
        symbol="\\ceil",
        category="function",
        arity=1,
        rpn_template="{0} ceil",
        precedence=0,
        associativity="none",
        description="Ceiling function",
    ),
    MathSymbol(
        symbol="\\gcd",
        category="function",
        arity=2,
        rpn_template="{0} {1} gcd",
        precedence=0,
        associativity="none",
        description="Greatest common divisor",
    ),
    MathSymbol(
        symbol="\\lcm",
        category="function",
        arity=2,
        rpn_template="{0} {1} mul {0} {1} gcd div",  # lcm(a,b) = a*b/gcd(a,b)
        precedence=0,
        associativity="none",
        description="Least common multiple",
    ),

    # ===== CONSTANTS =====
    MathSymbol(
        symbol="\\pi",
        category="constant",
        arity=0,
        rpn_template="3.14159265358979",
        precedence=0,
        associativity="none",
        description="Pi",
    ),
    MathSymbol(
        symbol="e",
        category="constant",
        arity=0,
        rpn_template="2.71828182845905",
        precedence=0,
        associativity="none",
        description="Euler's number",
    ),

    # ===== COMPARISON =====
    MathSymbol(
        symbol="=",
        category="relation",
        arity=2,
        rpn_template="{0} {1} eq",
        precedence=0,
        associativity="none",
        description="Equality",
    ),
    MathSymbol(
        symbol=">",
        category="relation",
        arity=2,
        rpn_template="{0} {1} gt",
        precedence=0,
        associativity="none",
        description="Greater than",
    ),
    MathSymbol(
        symbol="<",
        category="relation",
        arity=2,
        rpn_template="{0} {1} lt",
        precedence=0,
        associativity="none",
        description="Less than",
    ),
    MathSymbol(
        symbol="\\geq",
        category="relation",
        arity=2,
        rpn_template="{0} {1} gte",
        precedence=0,
        associativity="none",
        description="Greater than or equal",
    ),
    MathSymbol(
        symbol="\\leq",
        category="relation",
        arity=2,
        rpn_template="{0} {1} lte",
        precedence=0,
        associativity="none",
        description="Less than or equal",
    ),
]


class MathSymbolGalaxy:
    """
    Galaxy of math symbols with RPN meanings.

    The model "sees" symbols and looks them up here to compose RPN programs.
    This IS the model's math knowledge - sovereign, no external preprocessing.
    """

    def __init__(self):
        self._symbols = {s.symbol: s for s in MATH_SYMBOLS}
        self._by_category = {}
        for s in MATH_SYMBOLS:
            self._by_category.setdefault(s.category, []).append(s)

    def lookup(self, symbol: str) -> Optional[MathSymbol]:
        """Look up a symbol's RPN meaning."""
        return self._symbols.get(symbol)

    def get_rpn_template(self, symbol: str) -> Optional[str]:
        """Get RPN template for a symbol."""
        s = self.lookup(symbol)
        return s.rpn_template if s else None

    def compose_rpn(self, symbol: str, *args) -> str:
        """Compose RPN program from symbol and arguments."""
        s = self.lookup(symbol)
        if not s:
            return ""
        template = s.rpn_template
        for i, arg in enumerate(args):
            template = template.replace(f"{{{i}}}", str(arg))
        return template

    def all_symbols(self) -> List[MathSymbol]:
        """Return all registered symbols."""
        return list(self._symbols.values())

    def symbols_by_category(self, category: str) -> List[MathSymbol]:
        """Return symbols in a category."""
        return self._by_category.get(category, [])


# Global instance
MATH_GALAXY = MathSymbolGalaxy()
```

---

## Task 2: Grammar Rules That Reference Math Galaxy

**Update:** `knowledge3d/training/arc_agi/math_grammar_rules.py`

Grammar rules should compose RPN by referencing the Math Galaxy:

```python
from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY

# Grammar rules that use Galaxy symbols
SOVEREIGN_MATH_RULES = [
    GrammarRule(
        rule_id="latex_frac",
        language="math",
        pattern=r"\\frac\{(\d+)\}\{(\d+)\}",
        rpn_program=lambda m: MATH_GALAXY.compose_rpn("\\frac", m.group(1), m.group(2)),
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "\\frac{24}{4}", "output": "6"}],
    ),
    GrammarRule(
        rule_id="latex_binom",
        language="math",
        pattern=r"\\binom\{(\d+)\}\{(\d+)\}",
        rpn_program=lambda m: MATH_GALAXY.compose_rpn("\\binom", m.group(1), m.group(2)),
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[{"input": "\\binom{10}{3}", "output": "120"}],
    ),
    GrammarRule(
        rule_id="latex_sqrt",
        language="math",
        pattern=r"\\sqrt\{(\d+)\}",
        rpn_program=lambda m: MATH_GALAXY.compose_rpn("\\sqrt", m.group(1)),
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "\\sqrt{16}", "output": "4"}],
    ),
    GrammarRule(
        rule_id="factorial",
        language="math",
        pattern=r"(\d+)!",
        rpn_program=lambda m: MATH_GALAXY.compose_rpn("!", m.group(1)),
        domain="math_combinatorics",
        symbol_refs=[],
        examples=[{"input": "5!", "output": "120"}],
    ),
    GrammarRule(
        rule_id="power",
        language="math",
        pattern=r"(\d+)\^(\d+)",
        rpn_program=lambda m: MATH_GALAXY.compose_rpn("^", m.group(1), m.group(2)),
        domain="math_arithmetic",
        symbol_refs=[],
        examples=[{"input": "2^10", "output": "1024"}],
    ),
]
```

---

## Task 3: Sovereign Expression Composer

**File:** `knowledge3d/training/math_benchmarks/sovereign_composer.py`

Replace the external Python preprocessing with Galaxy-based composition:

```python
"""
Sovereign Expression Composer - Composes RPN from input using Galaxy.

NO external preprocessing. The Galaxy IS the model's knowledge.
"""

import re
from typing import List, Any, Optional

from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY, MathSymbol


class SovereignComposer:
    """
    Composes RPN programs from input expressions using the Math Galaxy.

    This is sovereign: symbols are looked up in the Galaxy, not processed
    externally. The Galaxy stores the logic.
    """

    def __init__(self):
        self.galaxy = MATH_GALAXY

    def compose(self, expression: str) -> List[Any]:
        """
        Compose RPN program from expression using Galaxy lookups.

        The model "sees" the expression, matches symbols against Galaxy,
        and composes RPN. No external preprocessing.
        """
        tokens = self._tokenize(expression)
        return self._to_rpn(tokens)

    def _tokenize(self, expr: str) -> List[str]:
        """Tokenize expression into symbols the Galaxy understands."""
        tokens = []
        i = 0

        while i < len(expr):
            # Skip whitespace
            if expr[i].isspace():
                i += 1
                continue

            # LaTeX commands: \frac, \binom, \sqrt, etc.
            if expr[i] == '\\':
                # Find end of command
                j = i + 1
                while j < len(expr) and expr[j].isalpha():
                    j += 1
                cmd = expr[i:j]

                # Check if Galaxy knows this symbol
                if self.galaxy.lookup(cmd):
                    tokens.append(cmd)
                    i = j

                    # Handle arguments in braces
                    while i < len(expr) and expr[i] == '{':
                        # Extract argument
                        brace_count = 1
                        start = i + 1
                        i += 1
                        while i < len(expr) and brace_count > 0:
                            if expr[i] == '{':
                                brace_count += 1
                            elif expr[i] == '}':
                                brace_count -= 1
                            i += 1
                        arg = expr[start:i-1]
                        # Recursively tokenize argument
                        tokens.append(('ARG', arg))
                    continue
                else:
                    i = j
                    continue

            # Numbers
            if expr[i].isdigit() or (expr[i] == '.' and i+1 < len(expr) and expr[i+1].isdigit()):
                j = i
                while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                tokens.append(expr[i:j])
                i = j
                continue

            # Operators and other symbols
            sym = expr[i]
            if self.galaxy.lookup(sym):
                tokens.append(sym)
            i += 1

        return tokens

    def _to_rpn(self, tokens: List) -> List[Any]:
        """
        Convert tokens to RPN using Galaxy symbol information.

        Uses precedence and associativity from Galaxy entries.
        """
        output = []
        op_stack = []

        i = 0
        while i < len(tokens):
            tok = tokens[i]

            # Handle argument tuples
            if isinstance(tok, tuple) and tok[0] == 'ARG':
                # Recursively compose the argument
                arg_tokens = self._tokenize(tok[1])
                arg_rpn = self._to_rpn(arg_tokens)
                output.extend(arg_rpn)
                i += 1
                continue

            # Number
            if isinstance(tok, str) and tok[0].isdigit():
                output.append(float(tok))
                i += 1
                continue

            # Symbol lookup in Galaxy
            sym = self.galaxy.lookup(tok)
            if sym:
                if sym.category == 'constant':
                    # Constants become their RPN value
                    output.append(float(sym.rpn_template))
                elif sym.category == 'function':
                    # Functions go on operator stack
                    op_stack.append((tok, sym))
                elif sym.category == 'operator':
                    # Handle operator precedence
                    while (op_stack and
                           isinstance(op_stack[-1], tuple) and
                           op_stack[-1][1].category == 'operator' and
                           ((sym.associativity == 'left' and sym.precedence <= op_stack[-1][1].precedence) or
                            (sym.associativity == 'right' and sym.precedence < op_stack[-1][1].precedence))):
                        _, s = op_stack.pop()
                        output.append(s.rpn_template.split()[-1])  # Get opcode name
                    op_stack.append((tok, sym))
                elif tok == '(':
                    op_stack.append(tok)
                elif tok == ')':
                    while op_stack and op_stack[-1] != '(':
                        item = op_stack.pop()
                        if isinstance(item, tuple):
                            output.append(item[1].rpn_template.split()[-1])
                    if op_stack:
                        op_stack.pop()  # Remove '('

            i += 1

        # Pop remaining operators
        while op_stack:
            item = op_stack.pop()
            if isinstance(item, tuple):
                output.append(item[1].rpn_template.split()[-1])

        return output
```

---

## Task 4: Wire Into Pipeline

**Update:** `knowledge3d/training/math_benchmarks/sovereign_math_pipeline.py`

```python
from knowledge3d.training.math_benchmarks.sovereign_composer import SovereignComposer

class SovereignMathPipeline:
    def __init__(self):
        # ... existing ...
        self.composer = SovereignComposer()

    def solve_problem(self, problem: Dict[str, Any]) -> Any:
        text = problem.get("problem", problem.get("question", ""))

        # SOVEREIGN: Compose RPN from input using Galaxy
        rpn_tokens = self.composer.compose(text)

        if rpn_tokens:
            # Execute on GPU
            result = self.rpn_engine.evaluate(rpn_tokens)
            if result is not None:
                return result

        # Fallback to word problem solver (which also uses Galaxy rules)
        return self.word_solver.solve(text)
```

---

## Task 5: Clean Up External Preprocessing

**DELETE or deprecate:**
- `algebra_solver.py` external preprocessing methods
- `problem_classifier.py` external regex extraction
- Any Python code that does LaTeX normalization outside the Galaxy

The Galaxy IS the preprocessing. Symbols have meanings. Grammar rules compose.

---

## Testing

```bash
PYTHONPATH=. python3 -c "
from knowledge3d.training.math_benchmarks.sovereign_composer import SovereignComposer
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

composer = SovereignComposer()
engine = ModularRPNEngine()

tests = [
    '\\\\frac{24}{4}',
    '\\\\binom{10}{3}',
    '5!',
    '2^10',
    '\\\\sqrt{16}',
]

for expr in tests:
    rpn = composer.compose(expr)
    result = engine.evaluate(rpn)
    print(f'{expr:20s} → RPN: {rpn} → {result}')
"
```

---

## Success Criteria

1. Math symbols are Galaxy entries with RPN templates
2. Grammar rules compose RPN by Galaxy lookup
3. NO external Python preprocessing of expressions
4. The Galaxy IS the model's math knowledge

---

**Codex:** Implement Tasks 1-5. The Galaxy stores logic. Weights = rules. This is sovereign.
