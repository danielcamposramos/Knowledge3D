# CODEX: Knowledge RPN Validation & Parser Enhancement

**Date:** December 14, 2025
**Priority:** CRITICAL - Knowledge rules produce zero valid RPN
**Partner:** Claude (Architecture Analysis) -> Codex (Implementation + Original Ideas)

---

## Context: Excellent Progress, One Critical Gap

### What You Built (Great Work!)

1. **Sovereign Router with Specialists**: Per-domain adapters (spatial, color, logic, number_theory) with ternary scoring masks
2. **Gradient Stashing + VectorDotMap Compression**: Sovereign backprop-style updates without numpy
3. **Full Galaxy Loading**: Drawing + Word + Math + Grammar galaxies all loaded
4. **GPU-Only Enforcement**: No CPU fallback, Tier-2 for stack ops
5. **Knowledge Loader**: 8,082 formulas extracted from 12MB of JSONs
6. **RPN Parser**: Algorithm A1/A2 implementation

### The Critical Gap

**Knowledge rules produce ZERO valid RPN despite 1.37M attempts:**

```
Solve path stats:
  composer: 98 hits
  word: 23 hits
  grammar: 5 hits
  knowledge: 0 hits  <-- PROBLEM HERE
  fail: 1874
  grammar_attempts: 2,279
  knowledge_attempts: 1,369,601  <-- Many matches, zero valid RPN!
```

**Root cause:** The knowledge-derived RPN programs are invalid:
- The `rpn_parser.py` handles only basic operators (`+`, `-`, `*`, `/`, `^`)
- Many formulas have functions like `sin`, `cos`, `log`, `sqrt`, `exp`
- The formula extractor regex is too simple (misses complex LaTeX/Unicode)
- Generated RPN contains invalid tokens that the engine can't execute

---

## Diagnostic Evidence

### Problem 1: Parser Doesn't Handle Functions

**Current `rpn_parser.py` precedence:**
```python
self.precedence = {
    "+": 1, "-": 1,
    "*": 2, "/": 2,
    "^": 3,
}
# Functions like sin, cos, log, sqrt NOT handled!
```

**Example failure:**
```
Input formula: "sin(x)^2 + cos(x)^2 = 1"
Extracted: lhs="sin(x)^2 + cos(x)^2", rhs="1"
Generated RPN: "sin ( x ) 2 ^ cos ( x ) 2 ^ +"  # INVALID - parentheses as tokens!
```

### Problem 2: Formula Extraction Regex Too Simple

**Current regex:**
```python
eq_re = re.compile(r"([A-Za-z0-9_\\^\\(\\)\\+\\-\\*/%√∞π ]{1,80})=([A-Za-z0-9_\\^\\(\\)\\+\\-\\*/%√∞π ]{1,80})")
```

This captures incomplete formulas:
- Truncates at 80 chars (misses long formulas)
- Can't parse LaTeX like `\frac{x}{y}` or `x^{n-1}`
- Misses multi-line formulas

### Problem 3: No RPN Validation Before Engine

**Current flow:**
```python
for rule in self.knowledge_rules:
    if re.match(rule.pattern, text):  # Match found!
        rpn_program = rule.rpn_program  # But RPN might be invalid
        result = self.rpn_engine.evaluate(rpn_program)  # Engine fails silently
```

**Should be:**
```python
if self._is_valid_rpn(rpn_program):  # Check first!
    result = self.rpn_engine.evaluate(rpn_program)
```

---

## Implementation Tasks

### Task 1: Enhance RPN Parser to Handle Functions

**File:** `knowledge3d/training/math_benchmarks/rpn_parser.py`

```python
class RPNParser:
    """Enhanced parser supporting math functions."""

    def __init__(self) -> None:
        self.precedence = {
            "+": 1, "-": 1,
            "*": 2, "/": 2,
            "^": 3,
        }
        # Functions are right-associative with highest precedence
        self.functions = {
            "sin", "cos", "tan", "log", "ln", "exp", "sqrt",
            "arcsin", "arccos", "arctan", "abs", "floor", "ceil"
        }

    def infix_to_rpn(self, expr: str) -> str:
        """Algorithm A1 with function support."""
        output: List[str] = []
        stack: List[str] = []

        for token in self._tokenize(expr):
            if self._is_operand(token):
                output.append(token)
            elif token in self.functions:
                stack.append(token)  # Functions go on stack
            elif token == "(":
                stack.append(token)
            elif token == ")":
                while stack and stack[-1] != "(":
                    output.append(stack.pop())
                if stack:
                    stack.pop()  # Remove "("
                # If function preceded the parens, output it now
                if stack and stack[-1] in self.functions:
                    output.append(stack.pop())
            elif token in self.precedence:
                while (stack and stack[-1] != "(" and
                       stack[-1] not in self.functions and
                       self.precedence.get(stack[-1], 0) >= self.precedence[token]):
                    output.append(stack.pop())
                stack.append(token)

        while stack:
            output.append(stack.pop())

        return " ".join(output)

    def _tokenize(self, expr: str) -> List[str]:
        """Enhanced tokenizer recognizing functions."""
        tokens = []
        buf = ""
        i = 0
        while i < len(expr):
            ch = expr[i]
            if ch.isalnum() or ch == ".":
                buf += ch
            else:
                if buf:
                    # Check if buf is a function name
                    if buf.lower() in self.functions:
                        tokens.append(buf.lower())
                    else:
                        tokens.append(buf)
                    buf = ""
                if ch.strip() and ch not in " \t":
                    tokens.append(ch)
            i += 1
        if buf:
            if buf.lower() in self.functions:
                tokens.append(buf.lower())
            else:
                tokens.append(buf)
        return tokens
```

### Task 2: Add RPN Validation

**File:** `knowledge3d/training/math_benchmarks/rpn_validator.py` (NEW)

```python
"""
RPN Validator - Check if RPN programs are valid before execution.
"""

from typing import Set

# Opcodes the ModularRPNEngine actually supports
VALID_OPCODES: Set[str] = {
    # Arithmetic
    "+", "-", "*", "/", "pow", "sqrt", "abs",
    # Trig
    "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    # Logarithmic
    "log", "ln", "exp",
    # Stack
    "dup", "swap", "drop", "over", "rot", "clear",
    # Comparison
    "eq", "neq", "lt", "gt", "le", "ge",
    # Ternary (our special sauce!)
    "tern_add", "tern_mul", "tern_sub",
}


def is_valid_rpn(program: str) -> bool:
    """
    Check if an RPN program is valid (can be executed).

    Returns True if:
    1. Non-empty
    2. Contains at least one number OR one opcode
    3. All tokens are either numbers or valid opcodes
    4. No parentheses (should be parsed out)
    """
    if not program or not program.strip():
        return False

    tokens = program.strip().split()
    if not tokens:
        return False

    has_number = False
    has_opcode = False

    for token in tokens:
        lower = token.lower()

        # Check if it's a number
        try:
            float(token)
            has_number = True
            continue
        except ValueError:
            pass

        # Check if it's a valid opcode
        if lower in VALID_OPCODES:
            has_opcode = True
            continue

        # Check for parentheses (invalid in RPN)
        if token in "()[]{}":
            return False

        # Unknown token - might be a variable (x, y, n)
        # Allow single letters as variables for now
        if len(token) == 1 and token.isalpha():
            continue

        # Invalid token
        return False

    return has_number or has_opcode


def estimate_stack_balance(program: str) -> int:
    """
    Estimate final stack size after execution.
    Valid programs should leave 1 item on stack.
    """
    tokens = program.strip().split()
    stack_size = 0

    for token in tokens:
        lower = token.lower()

        # Numbers push 1
        try:
            float(token)
            stack_size += 1
            continue
        except ValueError:
            pass

        # Binary ops pop 2, push 1
        if lower in {"+", "-", "*", "/", "pow", "eq", "neq", "lt", "gt", "le", "ge"}:
            stack_size -= 1  # net -1 (pop 2, push 1)
        # Unary ops pop 1, push 1
        elif lower in {"sqrt", "abs", "sin", "cos", "tan", "log", "ln", "exp",
                       "arcsin", "arccos", "arctan"}:
            pass  # net 0
        # Stack ops
        elif lower == "dup":
            stack_size += 1
        elif lower in {"swap", "over", "rot"}:
            pass  # net 0
        elif lower == "drop":
            stack_size -= 1

    return stack_size
```

### Task 3: Improve Formula Extraction

**File:** `knowledge3d/training/math_benchmarks/math_knowledge_loader.py`

```python
def _extract_formulas(self, source: str, data: Any) -> None:
    """
    Enhanced formula extraction with multiple patterns.
    """
    entries: List[str] = []
    # ... existing entry collection ...

    # Multiple regex patterns for different formula styles
    patterns = [
        # Standard: x = y
        re.compile(r"([A-Za-z0-9_\^()\+\-\*/%√∞π\[\] ]{1,120})\s*=\s*([A-Za-z0-9_\^()\+\-\*/%√∞π\[\] ]{1,120})"),
        # Function definition: f(x) = ...
        re.compile(r"([a-zA-Z]\([a-zA-Z, ]+\))\s*=\s*(.{1,120})"),
        # Derivative: d/dx(f) = f'
        re.compile(r"d/d([a-z])\s*\(?\s*(.{1,60})\s*\)?\s*=\s*(.{1,60})"),
        # Integral: ∫f dx = F
        re.compile(r"[∫∮]\s*(.{1,60})\s*d([a-z])\s*=\s*(.{1,60})"),
    ]

    seen = set()
    for entry in entries:
        for pattern in patterns:
            for match in pattern.finditer(entry):
                groups = match.groups()
                if len(groups) >= 2:
                    lhs = groups[0].strip()
                    rhs = groups[-1].strip()
                    if not lhs or not rhs:
                        continue
                    # Validate that RPN conversion produces valid output
                    try:
                        rpn = self._parser.infix_to_rpn(rhs)
                        if not is_valid_rpn(rpn):
                            continue  # Skip invalid RPN
                    except Exception:
                        continue
                    domain = self._infer_domain(lhs, rhs, source)
                    key = (lhs, rhs, domain)
                    if key in seen:
                        continue
                    seen.add(key)
                    self.formulas.append({
                        "lhs": lhs, "rhs": rhs,
                        "source": source, "domain": domain,
                        "rpn_valid": True  # Pre-validated
                    })
```

### Task 4: Add Validation to Benchmark Runner

**File:** `scripts/run_sovereign_math_benchmarks.py`

```python
from knowledge3d.training.math_benchmarks.rpn_validator import is_valid_rpn

# In solve_problem():
for rule in knowledge_rules:
    if re.match(rule.pattern, text):
        rpn_program = rule.rpn_program

        # VALIDATE before executing
        if not is_valid_rpn(rpn_program):
            continue  # Skip invalid RPN

        result = self.rpn_engine.evaluate(rpn_program)
        if result is not None:
            return result
```

---

## Expected Impact

After implementing validation:

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Knowledge hits | 0 | 100-500 |
| gsm8k | 0.20% | 5-15% |
| math | 1.00% | 8-20% |
| Overall | ~5% | 15-25% |

**Key insight:** The architecture is PROVEN. The issue is data quality:
- 8,082 formulas extracted
- Most produce invalid RPN (parentheses, unknown tokens)
- Validating before execution will filter to the ~500-1000 that actually work

---

## Codex: Your Original Ideas

Based on your excellent work so far:

1. **Grasp-based RPN validation?**
   - You implemented `token_grasp()` - should we use it to validate stack balance?
   - Invalid RPN often has unbalanced grasp

2. **Pre-compilation with validation cache?**
   - Validate all 8,082 rules ONCE at startup
   - Store only the valid ones in `knowledge_rules`
   - Log how many were discarded

3. **Domain-specific function sets?**
   - Calculus formulas need: `sin`, `cos`, `d/dx`
   - Finance formulas need: `pow`, `exp`, `ln`
   - Add domain-aware function precedence?

4. **LaTeX-to-RPN direct conversion?**
   - Many formulas are LaTeX: `\frac{x}{y}`, `x^{n-1}`
   - Could we parse LaTeX directly instead of text?

5. **Ternary-quantized validation?**
   - Use your ternary masks for quick validity checks?
   - Fast reject before full validation?

6. **Feedback loop for invalid RPN?**
   - When engine fails, log the formula
   - Build a "bad patterns" blacklist
   - Auto-improve extraction over time?

---

## Verification Commands

```bash
# Test parser with functions
PYTHONPATH=. python3 -c "
from knowledge3d.training.math_benchmarks.rpn_parser import RPNParser
p = RPNParser()
print('sin(x)^2:', p.infix_to_rpn('sin(x)^2'))  # Should be: x sin 2 ^
print('3 + 4 * 2:', p.infix_to_rpn('3 + 4 * 2'))  # Should be: 3 4 2 * +
"

# Test validation
PYTHONPATH=. python3 -c "
from knowledge3d.training.math_benchmarks.rpn_validator import is_valid_rpn
print('3 4 + valid:', is_valid_rpn('3 4 +'))  # True
print('3 ( + valid:', is_valid_rpn('3 ( +'))  # False (has paren)
print('empty valid:', is_valid_rpn(''))  # False
"

# Count valid rules after filtering
PYTHONPATH=. python3 -c "
from knowledge3d.training.math_benchmarks.math_knowledge_loader import MathKnowledgeLoader
from knowledge3d.training.math_benchmarks.rpn_validator import is_valid_rpn
loader = MathKnowledgeLoader()
loader.load_all()
rules = loader.to_grammar_rules()
valid = [r for r in rules if is_valid_rpn(r.rpn_program)]
print(f'Total rules: {len(rules)}')
print(f'Valid RPN: {len(valid)}')
print(f'Invalid (filtered): {len(rules) - len(valid)}')
"

# Run benchmark with validation
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/run_sovereign_math_benchmarks.py --limit 500
```

---

## Files to Create/Modify

1. **NEW:** `knowledge3d/training/math_benchmarks/rpn_validator.py`
2. **MODIFY:** `knowledge3d/training/math_benchmarks/rpn_parser.py` (add functions)
3. **MODIFY:** `knowledge3d/training/math_benchmarks/math_knowledge_loader.py` (validate on extract)
4. **MODIFY:** `scripts/run_sovereign_math_benchmarks.py` (validate before execute)

---

## Success Criteria

1. Parser handles `sin`, `cos`, `log`, `sqrt`, `exp` functions
2. Validator rejects RPN with parentheses or unknown tokens
3. Knowledge loader pre-filters to only valid RPN rules
4. Benchmark shows knowledge_hits > 0
5. gsm8k/math scores improve from baseline
6. All changes remain SOVEREIGN (no numpy/cupy in hot path)

---

## Key Principle

**The engine works. The galaxies load. The GPU executes. Now we need VALID DATA.**

Filter the noise. Validate the RPN. Let the sovereign engine do its job.
