# CODEX BRIEFING: Algebra Solver Fixes - Round 1

**Date:** December 13, 2025
**Priority:** HIGH - Diagnostics identified core issues
**Partner:** Claude (Architecture) → Codex (Implementation)

---

## Root Causes Identified

1. **Expression strategy is empty** - 90% of problems fall back to `expression/evaluate` but its RPN program is `""` → returns None
2. **No LaTeX normalization** - Classifier sees raw `\frac{...}{...}` instead of numbers
3. **Coefficient extraction is shallow** - Only finds standalone numbers, not LaTeX math

---

## Fix 1: Expression Evaluator Strategy (HIGHEST IMPACT)

**File:** `knowledge3d/training/math_benchmarks/algebra_solver.py`

The current fallback produces nothing. Add a real expression parser:

```python
def _try_direct_expression_eval(self, problem_text: str) -> Optional[float]:
    """
    Try to extract and evaluate a numeric expression directly.
    This is the fallback when no strategy matches.
    """
    # Normalize LaTeX first
    text = self._normalize_latex(problem_text)

    # Look for expressions to evaluate
    # Pattern: "what is X + Y * Z" or "compute 5! + 3" or just "2^10"
    import re

    # Find the last numeric expression in the text
    # Look for patterns like: numbers with operators
    expr_pattern = r'([\d\.\+\-\*\/\^\!\(\)\s]+)'
    matches = re.findall(expr_pattern, text)

    for match in reversed(matches):  # Try from end (usually the question)
        tokens = self._expr_to_rpn(match.strip())
        if tokens and len(tokens) >= 1:
            try:
                result = self.rpn_engine.evaluate(tokens)
                if result is not None:
                    return result
            except:
                continue

    return None


def _expr_to_rpn(self, expr: str) -> List[Any]:
    """Convert simple infix expression to RPN tokens."""
    import re

    tokens = []
    # Tokenize: numbers, operators, factorial
    token_pattern = r'(\d+\.?\d*|\+|\-|\*|\/|\^|\!|\(|\))'
    raw_tokens = re.findall(token_pattern, expr)

    # Simple shunting-yard for basic expressions
    output = []
    op_stack = []
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3, '!': 4}
    right_assoc = {'^'}

    i = 0
    while i < len(raw_tokens):
        tok = raw_tokens[i]

        if re.match(r'\d', tok):
            output.append(float(tok))
        elif tok == '!':
            # Factorial is postfix - apply to last number
            output.append('factorial')
        elif tok == '(':
            op_stack.append(tok)
        elif tok == ')':
            while op_stack and op_stack[-1] != '(':
                output.append(self._op_to_rpn(op_stack.pop()))
            if op_stack:
                op_stack.pop()  # Remove '('
        elif tok in precedence:
            while (op_stack and op_stack[-1] != '(' and
                   op_stack[-1] in precedence and
                   (precedence[op_stack[-1]] > precedence[tok] or
                    (precedence[op_stack[-1]] == precedence[tok] and tok not in right_assoc))):
                output.append(self._op_to_rpn(op_stack.pop()))
            op_stack.append(tok)

        i += 1

    while op_stack:
        output.append(self._op_to_rpn(op_stack.pop()))

    return output


def _op_to_rpn(self, op: str) -> str:
    """Convert operator to RPN token name."""
    mapping = {
        '+': 'add', '-': 'sub', '*': 'mul', '/': 'div',
        '^': 'pow', '!': 'factorial'
    }
    return mapping.get(op, op)
```

**Update `solve()` method:**

```python
def solve(self, problem_text: str) -> Tuple[Any, Dict[str, Any]]:
    # ... existing code ...

    # Step 4: Execute RPN chains
    results = []
    for i, rpn_template in enumerate(strategy.rpn_chains):
        # ... existing chain execution ...

    # NEW: If no results from strategy, try direct expression evaluation
    if not results or results[-1] is None:
        direct_result = self._try_direct_expression_eval(problem_text)
        if direct_result is not None:
            results.append(direct_result)
            self._trace("direct_eval", {"result": direct_result})

    # Return final result
    final_answer = results[-1] if results else None
    # ... rest unchanged ...
```

---

## Fix 2: LaTeX Normalization

**Add to `algebra_solver.py`:**

```python
def _normalize_latex(self, text: str) -> str:
    """Convert LaTeX to plain math notation."""
    import re

    result = text

    # Remove LaTeX delimiters
    result = re.sub(r'\$\$?', '', result)
    result = re.sub(r'\\\[|\\\]', '', result)
    result = re.sub(r'\\text\{([^}]*)\}', r'\1', result)

    # Convert fractions: \frac{a}{b} → (a/b)
    while r'\frac' in result:
        result = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1/\2)', result)

    # Convert binomials: \binom{n}{k} → C(n,k) (for display, actual eval uses binomial)
    result = re.sub(r'\\binom\{(\d+)\}\{(\d+)\}', r'binom(\1,\2)', result)

    # Convert powers: x^{2} → x^2, 2^{10} → 2^10
    result = re.sub(r'\^{(\d+)}', r'^\1', result)

    # Convert sqrt: \sqrt{x} → sqrt(x)
    result = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', result)

    # Remove other LaTeX commands
    result = re.sub(r'\\[a-zA-Z]+', '', result)

    # Clean up whitespace
    result = re.sub(r'\s+', ' ', result).strip()

    return result
```

**Update `classify()` in `problem_classifier.py`:**

```python
def classify(self, problem_text: str) -> ProblemClassification:
    """Classify a math problem."""
    # Normalize LaTeX before classification
    text_normalized = self._normalize_latex(problem_text)
    text_lower = text_normalized.lower()

    # ... rest of classification logic uses text_lower ...
```

---

## Fix 3: Enhanced Coefficient Extraction

**Update `_extract_coefficients()` in `problem_classifier.py`:**

```python
def _extract_coefficients(self, text: str) -> Dict[str, float]:
    """Extract numeric coefficients from equations."""
    import re
    coeffs = {}

    # First normalize LaTeX
    text = self._normalize_latex(text)

    # Extract from \binom{n}{k} patterns (now binom(n,k))
    binom_match = re.search(r'binom\((\d+),(\d+)\)', text)
    if binom_match:
        coeffs['n'] = float(binom_match.group(1))
        coeffs['k'] = float(binom_match.group(2))

    # Extract from fractions (now (a/b))
    frac_matches = re.findall(r'\((\d+)/(\d+)\)', text)
    for i, (num, denom) in enumerate(frac_matches[:3]):
        coeffs[f'frac{i}_num'] = float(num)
        coeffs[f'frac{i}_denom'] = float(denom)

    # Extract quadratic: ax^2 + bx + c (various formats)
    # Handle: x^2 - 5x + 6, 2x^2 + 3x - 1, etc.
    quad_patterns = [
        r'(-?\d*)\s*x\^?2\s*([+\-])\s*(\d*)\s*x\s*([+\-])\s*(\d+)',
        r'x\^?2\s*([+\-])\s*(\d+)\s*x\s*([+\-])\s*(\d+)',
    ]

    for pattern in quad_patterns:
        match = re.search(pattern, text.replace(' ', ''))
        if match:
            groups = match.groups()
            if len(groups) == 5:
                a = groups[0] or '1'
                coeffs['a'] = float(a) if a not in ('', '-') else (-1.0 if a == '-' else 1.0)
                sign_b = 1 if groups[1] == '+' else -1
                b = groups[2] or '1'
                coeffs['b'] = sign_b * (float(b) if b else 1.0)
                sign_c = 1 if groups[3] == '+' else -1
                coeffs['c'] = sign_c * float(groups[4])
            break

    # Extract standalone numbers
    numbers = re.findall(r'\b(\d+\.?\d*)\b', text)
    for i, n in enumerate(numbers[:5]):
        coeffs[f'n{i}'] = float(n)

    # Special: factorial pattern "n!" → n
    factorial_match = re.search(r'(\d+)!', text)
    if factorial_match and 'n' not in coeffs:
        coeffs['n'] = float(factorial_match.group(1))

    return coeffs
```

---

## Fix 4: Add Missing Classifier Patterns

**Add to `PATTERNS` in `problem_classifier.py`:**

```python
# Add at the TOP of PATTERNS list (high priority)
(r"find the value|what is the value|compute|calculate|evaluate", None, "expression", "evaluate"),
(r"\\binom|binom\(|choose", r"\d+", "combinatorics", "count"),
(r"(\d+)!", None, "combinatorics", "factorial"),
(r"probability|expected|random", None, "probability", "calculate"),
(r"how many|in how many ways|number of ways", None, "combinatorics", "count"),
```

---

## Fix 5: Handle Binomial in Expression Parser

**Update `_expr_to_rpn()` to handle `binom(n,k)`:**

```python
def _expr_to_rpn(self, expr: str) -> List[Any]:
    """Convert simple infix expression to RPN tokens."""
    import re

    # Handle binom(n,k) before general tokenization
    binom_match = re.search(r'binom\((\d+),(\d+)\)', expr)
    if binom_match:
        n, k = float(binom_match.group(1)), float(binom_match.group(2))
        return [n, k, 'binomial']

    # ... rest of existing tokenization ...
```

---

## Testing

After implementing fixes:

```bash
# Quick sanity check
PYTHONPATH=. python3 -c "
from knowledge3d.training.math_benchmarks.algebra_solver import AlgebraSolver
solver = AlgebraSolver()

tests = [
    'What is 2^10?',
    'Compute 5! + 10',
    'Find the value of \\\\binom{10}{3}',
    'Calculate \\\\frac{24}{4}',
    'Evaluate 3 * 4 + 5',
]

for t in tests:
    answer, meta = solver.solve(t)
    print(f'{t[:40]:40s} → {answer}')
"

# Re-run diagnostics
PYTHONPATH=. python3 scripts/diagnose_algebra_solver.py
```

**Expected improvement:**
- Expression/evaluate should now produce results
- LaTeX problems should be parsed
- More problems should have coefficients extracted

---

## Success Criteria

After these fixes:
- Diagnostics show >50% of samples returning a result (vs 1/15 before)
- LaTeX expressions are normalized and parsed
- `expression/evaluate` strategy actually evaluates

---

**Codex:** Implement Fixes 1-5, run the sanity check and diagnostics. Report:
1. How many samples now produce results
2. Any new error patterns
3. Ready for full benchmark run?
