# CODEX: Math Knowledge Ingestion - Load All Available Knowledge

**Date:** December 14, 2025
**Priority:** HIGH - Expand sovereign math capabilities
**Partner:** Claude (Architecture Analysis) -> Codex (Implementation + Original Ideas)

---

## Context: Where We Are

### Current TRUE Baseline (After Removing Cheating)
```
GSM8K:     1.39%   (was 100% fake - now REAL)
MATH:      1.13%
Omni-MATH: 0.52%
AMC-AIME:  1.49%
MMLU:      22.98%  (was always real)
Overall:   8.81%
```

### Architecture Is Proven - We Need KNOWLEDGE

The sovereign pipeline works:
- **ModularRPNEngine**: PTX-based GPU execution with ternary opcodes
- **SovereignComposer**: Galaxy-based RPN composition
- **WordProblemSolver**: Grammar rule chaining (now with DUP/SWAP)
- **197 Math Symbols**: Loaded from cranium + arc_agi galaxies
- **747+ Grammar Rules**: Word patterns + competition math

**The limiting factor is not the engine - it's what the engine KNOWS.**

---

## Available Knowledge Assets (NOT YET LOADED)

### 1. Pre-Extracted JSON Files (~12MB of knowledge)

Location: `/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/JSON/`

| File | Size | Content |
|------|------|---------|
| `EchoSystems_Basic_Advanced_Math.json` | 2.0MB | Comprehensive math: polynomials, exponents, trig, calculus, matrices, complex numbers, differential equations |
| `Manning.Math.for.Programmers.2020.11.json` | 1.3MB | Programming-oriented math, computational techniques |
| `Multivariable Calculus 7th Edition By James Stewart.json` | 1.5MB | Vector calculus, partial derivatives, multiple integrals |
| `Linear.Algebra.Done.Right.json` | 594KB | Linear algebra: matrices, eigenvalues, vector spaces |
| `ADVANCED CALCULUS I and II.json` | 539KB | Advanced calculus methods |
| `Transition_v104.json` | 547KB | Mathematical foundations |
| `advmathprog.json` | 435KB | Advanced mathematical programming |
| `MATH 2F05.json` | 279KB | Applied advanced calculus |

### 2. RPN-SPECIFIC Knowledge (CRITICAL!)

| File | Size | Content |
|------|------|---------|
| `vertopal.com_ReversePolishNotatonMethod.json` | 25KB | Academic paper on RPN symbolic computation: infix↔postfix algorithms, expression trees, grasp/LGB concepts |
| `vertopal.com_3.3. Reverse Polish - Intermediate.json` | 5KB | Stack-based evaluation, BODMAS elimination, bytecode compilation |

**This is a DIRECT match for our RPN engine!** The RPN method paper covers:
- Algorithm A1: Infix → Postfix transformation
- Algorithm A2: Postfix → Infix transformation
- Symbolic derivation using RPN (skip expression trees!)
- n-ary operator handling
- Grasp and Left Grasp Bound (LGB) concepts

### 3. PDF Sources (38+ Documents)

**Advanced Maths Root:**
- `ADVANCED CALCULUS I and II.pdf` - ODE, vector analysis, complex analysis
- `Advanced_Calculus.pdf` (31MB) - Comprehensive calculus
- `Linear.Algebra.Done.Right.pdf` - Matrix operations
- `Multivariable Calculus 7th Edition By James Stewart.pdf` (16MB)
- `ReversePolishNotatonMethod.pdf` - Academic RPN paper
- `3.3. Reverse Polish - Intermediate.pdf` - RPN tutorial
- `Manning.Math.for.Programmers.2020.11.pdf` (27MB)
- `Handbook Of Numerical Analysis.pdf` - Computational methods

**BasicMath Subdirectory:**
- `AreaVol.pdf` - Area and volume formulas
- `MathGems.pdf` - Mathematical gems/proofs
- `NumberSets.pdf` - Number theory
- `PhysQuantities.pdf` - Physical quantities/units
- `ShortestShortcut.pdf` - Mathematical shortcuts

**Financial Math Subdirectory (18 PDFs):**
- Hull's Options, Futures and Derivatives (2 editions)
- Mathematics of Finance
- Actuarial Mathematics
- Algorithmic Trading
- Risk Management
- And more...

---

## Implementation Tasks

### Task 1: Create Math Knowledge Loader

**File:** `knowledge3d/training/math_benchmarks/math_knowledge_loader.py`

```python
"""
Math Knowledge Loader - Ingest pre-extracted JSONs into grammar rules.

Loads mathematical knowledge from EchoSystems Default Libraries:
- Formulas and identities
- Transformation rules
- Solution patterns
- RPN-native algorithms
"""

from pathlib import Path
from typing import Dict, List, Any
import json
import re

class MathKnowledgeLoader:
    """Load and parse pre-extracted math knowledge."""

    KNOWLEDGE_BASE = Path("/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/JSON")

    def __init__(self):
        self.formulas: List[Dict[str, Any]] = []
        self.rules: List[Dict[str, Any]] = []
        self.rpn_patterns: List[Dict[str, Any]] = []

    def load_all(self) -> Dict[str, int]:
        """Load all available JSON knowledge."""
        stats = {"formulas": 0, "rules": 0, "rpn_patterns": 0}

        # Priority: RPN-specific knowledge FIRST
        self._load_rpn_knowledge()

        # Then comprehensive math
        self._load_math_formulas()

        # Then calculus/linear algebra
        self._load_advanced_math()

        stats["formulas"] = len(self.formulas)
        stats["rules"] = len(self.rules)
        stats["rpn_patterns"] = len(self.rpn_patterns)
        return stats

    def _load_rpn_knowledge(self) -> None:
        """Load RPN-specific knowledge from the RPN papers."""
        rpn_files = [
            "vertopal.com_ReversePolishNotatonMethod.json",
            "vertopal.com_3.3. Reverse Polish - Intermediate.json",
        ]
        for fname in rpn_files:
            path = self.KNOWLEDGE_BASE / fname
            if path.exists():
                self._parse_rpn_paper(path)

    def _parse_rpn_paper(self, path: Path) -> None:
        """Extract RPN algorithms and patterns from paper JSON."""
        # Implementation: parse the academic paper structure
        # Extract: Algorithm A1, A2, grasp definitions, etc.
        pass

    def _load_math_formulas(self) -> None:
        """Load formulas from EchoSystems_Basic_Advanced_Math.json."""
        path = self.KNOWLEDGE_BASE / "EchoSystems_Basic_Advanced_Math.json"
        if path.exists():
            self._extract_formulas(path)

    def _extract_formulas(self, path: Path) -> None:
        """
        Extract mathematical formulas and convert to grammar rules.

        The JSON has structure: {"page": N, "content": "..."}
        Parse content for:
        - Equations (x = ...)
        - Identities (sin^2 + cos^2 = 1)
        - Rules (d/dx of x^n = n*x^(n-1))
        """
        pass

    def to_grammar_rules(self) -> List["GrammarRule"]:
        """Convert loaded knowledge to GrammarRule objects."""
        pass
```

### Task 2: Enhance Grammar Rules with Calculus

**File:** `knowledge3d/training/arc_agi/math_grammar_rules.py`

Add rules for:

```python
# Derivative rules (from the calculus JSONs)
CALCULUS_RULES = [
    GrammarRule(
        rule_id="calc_power_rule",
        pattern=r"d/dx\s*\(?\s*x\^(\d+)\s*\)?",
        rpn_program="{n} x {n} 1 - pow * ",  # n * x^(n-1)
        domain="calculus",
    ),
    GrammarRule(
        rule_id="calc_chain_rule",
        pattern=r"d/dx\s*\(?f\(g\(x\)\)\)?",
        rpn_program="f'(g(x)) g'(x) *",  # chain rule template
        domain="calculus",
    ),
    # Integration rules
    GrammarRule(
        rule_id="calc_power_integral",
        pattern=r"∫\s*x\^(\d+)\s*dx",
        rpn_program="x {n} 1 + pow {n} 1 + /",  # x^(n+1)/(n+1)
        domain="calculus",
    ),
    # Trig identities
    GrammarRule(
        rule_id="trig_pythagorean",
        pattern=r"sin\^2\s*\+\s*cos\^2",
        rpn_program="1",  # Identity = 1
        domain="trigonometry",
    ),
]

# Linear algebra rules (from Linear.Algebra.Done.Right.json)
LINEAR_ALGEBRA_RULES = [
    GrammarRule(
        rule_id="la_determinant_2x2",
        pattern=r"\|([a-z])\s+([a-z])\s*;\s*([a-z])\s+([a-z])\|",
        rpn_program="{a} {d} * {b} {c} * -",  # ad - bc
        domain="linear_algebra",
    ),
    GrammarRule(
        rule_id="la_matrix_inverse",
        pattern=r"inverse\s*of\s*(\d+)x(\d+)",
        rpn_program="matrix_inverse",
        domain="linear_algebra",
    ),
]
```

### Task 3: Ingest RPN Paper Algorithms

The RPN paper `vertopal.com_ReversePolishNotatonMethod.json` contains:

**Algorithm A1 (Infix → Postfix):**
```
1. Examine current element
2. If operand: send to output
3. If '(': push
4. If operator: compare priority with stack top, push or pop
5. If ')': pop until '('
6. Repeat until done, pop remaining stack
```

**Algorithm A2 (Postfix → Infix):**
```
1. Examine current element
2. If operand: push
3. If binary operator: pop two, execute, push result
4. If unary operator: pop one, execute, push result
5. Repeat until done, result on stack top
```

**These should be encoded as META-RULES that the engine uses for parsing!**

```python
# In modular_rpn_engine.py or a new parser module
class RPNParser:
    """
    Parser using Algorithm A1 from Krtolica & Stanimirovic (2004).

    Transforms infix expressions to RPN using stack-based method
    from the academic paper we have in our knowledge base.
    """

    def __init__(self):
        self.precedence = {
            '+': 1, '-': 1,
            '*': 2, '/': 2,
            '^': 3,
            'sin': 4, 'cos': 4, 'log': 4,
        }

    def infix_to_rpn(self, expr: str) -> str:
        """Algorithm A1: Infix to Postfix transformation."""
        output = []
        stack = []

        for token in self._tokenize(expr):
            if self._is_operand(token):
                output.append(token)
            elif token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                stack.pop()  # Remove '('
            elif token in self.precedence:
                while (stack and stack[-1] != '(' and
                       self.precedence.get(stack[-1], 0) >= self.precedence[token]):
                    output.append(stack.pop())
                stack.append(token)

        while stack:
            output.append(stack.pop())

        return ' '.join(output)
```

### Task 4: Load Financial Math Formulas

The Financial Math directory contains formulas for:
- Options pricing (Black-Scholes)
- Present/Future Value
- Compound interest
- Risk metrics

```python
FINANCIAL_MATH_RULES = [
    GrammarRule(
        rule_id="fin_compound_interest",
        pattern=r"compound\s*interest.*principal\s*(\d+).*rate\s*(\d+\.?\d*)%.*years?\s*(\d+)",
        rpn_program="{P} 1 {r} 100 / + {t} pow *",  # P * (1 + r/100)^t
        domain="finance",
    ),
    GrammarRule(
        rule_id="fin_simple_interest",
        pattern=r"simple\s*interest.*principal\s*(\d+).*rate\s*(\d+\.?\d*)%.*years?\s*(\d+)",
        rpn_program="{P} {P} {r} 100 / * {t} * +",  # P + P*r/100*t
        domain="finance",
    ),
    GrammarRule(
        rule_id="fin_present_value",
        pattern=r"present\s*value.*future\s*(\d+).*rate\s*(\d+\.?\d*)%.*years?\s*(\d+)",
        rpn_program="{FV} 1 {r} 100 / + {t} pow /",  # FV / (1 + r/100)^t
        domain="finance",
    ),
]
```

### Task 5: Update Benchmark Runner to Use Knowledge

**File:** `scripts/run_sovereign_math_benchmarks.py`

```python
class SovereignBenchmarkRunner:
    def __init__(self):
        # ... existing init ...

        # NEW: Load math knowledge
        from knowledge3d.training.math_benchmarks.math_knowledge_loader import MathKnowledgeLoader

        self.knowledge_loader = MathKnowledgeLoader()
        stats = self.knowledge_loader.load_all()
        print(f"Loaded Knowledge: {stats['formulas']} formulas, "
              f"{stats['rules']} rules, {stats['rpn_patterns']} RPN patterns")

        # Extend grammar rules with loaded knowledge
        additional_rules = self.knowledge_loader.to_grammar_rules()
        # Add to word_solver.rules and composer patterns
```

---

## Expected Impact

### After Loading All Knowledge:

| Dataset | Before | Expected | Reasoning |
|---------|--------|----------|-----------|
| GSM8K | 1.39% | 30-50% | Word problems match grammar rules |
| MATH | 1.13% | 20-35% | Calculus/algebra rules from JSONs |
| Omni-MATH | 0.52% | 10-20% | Competition patterns |
| AMC-AIME | 1.49% | 15-25% | Advanced techniques |
| MMLU | 22.98% | 35-50% | Multiple choice + formula matching |

**Key insight:** The RPN paper gives us the theoretical foundation we're ALREADY using - but we haven't loaded its specific algorithms as executable rules.

---

## Verification

```bash
# Test knowledge loader
PYTHONPATH=. python3 -c "
from knowledge3d.training.math_benchmarks.math_knowledge_loader import MathKnowledgeLoader
loader = MathKnowledgeLoader()
stats = loader.load_all()
print(f'Loaded: {stats}')
rules = loader.to_grammar_rules()
print(f'Generated {len(rules)} grammar rules')
"

# Test RPN parser from paper
PYTHONPATH=. python3 -c "
from knowledge3d.training.math_benchmarks.rpn_parser import RPNParser
parser = RPNParser()
print(parser.infix_to_rpn('3 + 4 * 2'))  # Expected: 3 4 2 * +
print(parser.infix_to_rpn('(3 + 4) * 2'))  # Expected: 3 4 + 2 *
"

# Run benchmark with loaded knowledge
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/run_sovereign_math_benchmarks.py --limit 100
```

---

## Codex: Your Original Ideas

Based on the 12MB+ of available knowledge:

1. **What patterns in the calculus JSON would most improve GSM8K?**
   - Word problems often involve rates, percentages, areas
   - Which formulas should be prioritized?

2. **How should we structure the knowledge hierarchy?**
   - JSON → Formulas → Grammar Rules → RPN Programs
   - What's the optimal granularity?

3. **The RPN paper has "grasp" and "LGB" concepts - should we implement these?**
   - Grasp = number of preceding elements that form operands
   - Could help with complex expression parsing

4. **Financial math formulas - how to integrate?**
   - Time value of money
   - Compound interest
   - These appear frequently in standardized tests

5. **What meta-rules would help the most?**
   - Rules that select which rules to apply
   - Order of operations handling
   - Expression tree flattening

---

## Key Principle

**We have one of the most advanced RPN calculators on the planet (with ternary opcodes). Now we need to FEED IT.**

The knowledge exists. The JSONs have been extracted. The pipeline is sovereign.

**Load it. Use it. Score REAL benchmarks.**

---

## Files to Create/Modify

1. **NEW:** `knowledge3d/training/math_benchmarks/math_knowledge_loader.py`
2. **NEW:** `knowledge3d/training/math_benchmarks/rpn_parser.py` (Algorithm A1/A2)
3. **MODIFY:** `knowledge3d/training/arc_agi/math_grammar_rules.py` (add calculus/LA rules)
4. **MODIFY:** `knowledge3d/training/math_benchmarks/galaxy_loader.py` (load knowledge)
5. **MODIFY:** `scripts/run_sovereign_math_benchmarks.py` (use loaded knowledge)

---

## Success Criteria

1. All 12 JSON files loaded and parsed
2. 500+ new grammar rules generated from formulas
3. RPN parser implements Algorithm A1/A2 from paper
4. Benchmark accuracy improves measurably on REAL solving
5. No cheating - all scores from actual computation
