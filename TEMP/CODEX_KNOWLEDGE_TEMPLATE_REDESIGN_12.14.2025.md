# CODEX: Knowledge Template Redesign - From Literal to Parametric

**Date:** December 14, 2025
**Priority:** CRITICAL - Knowledge path fundamentally broken
**Partner:** Claude (Architecture Analysis) -> Codex (Implementation + Original Ideas)

---

## Root Cause Found

The knowledge rules produce ZERO hits because **the design is fundamentally wrong**:

### Current (Broken) Approach

```python
# Formula extracted: "A = P(1 + r)^t"
# Generated pattern: "A\ =\ P\(1\ \+\ r\)\^t"  # LITERAL escaped string!
# Generated RPN: "P 1 r + t pow *"  # Uses variables, not numbers!

# Problem text: "Calculate compound interest on $1000 at 5% for 3 years"
# Pattern match: NEVER - the text doesn't contain literal "A = P(1 + r)^t"
```

### Evidence from Diagnostic

```
Pattern: \(\ \ x\ 2...               # Escaped garbage
Pattern: 0\ rather\ than\ n...       # English text, not math!
Pattern: I\ \ A\ i...                # Random letters
```

The extractor is capturing **random text** from PDFs, not formulas. And even valid formulas become **literal patterns** that can't match problem numbers.

---

## Correct Approach: Parametric Templates

Knowledge rules should be **templates with capture groups** that extract numbers from problem text:

### Example 1: Compound Interest

**Formula:** `A = P(1 + r)^t`

**Should become:**
```python
GrammarRule(
    rule_id="fin_compound_interest",
    pattern=r"(\d+\.?\d*)\s*(?:dollars?|principal).*?(\d+\.?\d*)\s*%.*?(\d+)\s*years?",
    rpn_template="{0} 1 {1} 100 / + {2} pow *",  # P * (1 + r/100)^t
    domain="math_finance",
)
```

**Match:** "Calculate compound interest on **1000** dollars at **5**% for **3** years"
- Captures: `[1000, 5, 3]`
- RPN: `1000 1 5 100 / + 3 pow *` = `1000 * 1.05^3` = `1157.625`

### Example 2: Pythagorean Theorem

**Formula:** `a^2 + b^2 = c^2`

**Should become:**
```python
GrammarRule(
    rule_id="geo_pythagorean",
    pattern=r"(\d+\.?\d*)\s*(?:squared|²).*?(\d+\.?\d*)\s*(?:squared|²)",
    rpn_template="{0} 2 pow {1} 2 pow + sqrt",  # sqrt(a² + b²)
    domain="math_geometry",
)
```

### Example 3: Simple Arithmetic (GSM8K)

**Pattern for "X times Y":**
```python
GrammarRule(
    rule_id="arith_times",
    pattern=r"(\d+\.?\d*)\s*(?:times|×|multiplied by)\s*(\d+\.?\d*)",
    rpn_template="{0} {1} *",
    domain="math_arithmetic",
)
```

---

## Implementation Tasks

### Task 1: Create Curated Template Library

Instead of extracting from garbage PDFs, create a **curated library** of working templates:

**File:** `knowledge3d/training/math_benchmarks/math_templates.py`

```python
"""
Curated math templates with parametric patterns.

Each template:
1. Has a regex pattern with capture groups for numbers
2. Has an RPN template with {0}, {1}, etc. placeholders
3. Is domain-tagged for routing
"""

from typing import List, Dict, Any
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule


def get_arithmetic_templates() -> List[GrammarRule]:
    """Templates for basic arithmetic (GSM8K)."""
    return [
        # Addition
        GrammarRule(
            rule_id="arith_add_total",
            pattern=r"(\d+\.?\d*)\s*(?:and|plus|\+|added to)\s*(\d+\.?\d*)",
            rpn_program="{0} {1} +",
            domain="math_arithmetic",
        ),
        # Multiplication
        GrammarRule(
            rule_id="arith_mul_times",
            pattern=r"(\d+\.?\d*)\s*(?:times|×|multiplied by)\s*(\d+\.?\d*)",
            rpn_program="{0} {1} *",
            domain="math_arithmetic",
        ),
        # Division (half, third, quarter)
        GrammarRule(
            rule_id="arith_div_half",
            pattern=r"half\s*(?:of|as many as)?\s*(\d+\.?\d*)",
            rpn_program="{0} 2 /",
            domain="math_arithmetic",
        ),
        GrammarRule(
            rule_id="arith_div_third",
            pattern=r"(?:one third|a third)\s*(?:of)?\s*(\d+\.?\d*)",
            rpn_program="{0} 3 /",
            domain="math_arithmetic",
        ),
        # Percentage
        GrammarRule(
            rule_id="arith_percent_of",
            pattern=r"(\d+\.?\d*)\s*%\s*(?:of)\s*(\d+\.?\d*)",
            rpn_program="{1} {0} 100 / *",  # base * (pct/100)
            domain="math_arithmetic",
        ),
        # Remaining after spending
        GrammarRule(
            rule_id="arith_remaining",
            pattern=r"(\d+\.?\d*).*?(?:spent|used|gave away)\s*(\d+\.?\d*)",
            rpn_program="{0} {1} -",
            domain="math_arithmetic",
        ),
        # Total with half (Natalia pattern)
        GrammarRule(
            rule_id="arith_total_half",
            pattern=r"(\d+\.?\d*).*?half\s*(?:as many|that).*?(?:altogether|total|in all)",
            rpn_program="{0} DUP 2 / +",  # X + X/2
            domain="math_arithmetic",
        ),
    ]


def get_finance_templates() -> List[GrammarRule]:
    """Templates for financial math."""
    return [
        # Compound interest
        GrammarRule(
            rule_id="fin_compound",
            pattern=r"(\d+\.?\d*)\s*(?:dollars?|principal).*?(\d+\.?\d*)\s*%.*?(\d+)\s*years?",
            rpn_program="{0} 1 {1} 100 / + {2} pow *",
            domain="math_finance",
        ),
        # Simple interest
        GrammarRule(
            rule_id="fin_simple",
            pattern=r"simple\s*interest.*?(\d+\.?\d*).*?(\d+\.?\d*)\s*%.*?(\d+)",
            rpn_program="{0} {0} {1} 100 / * {2} * +",  # P + P*r*t
            domain="math_finance",
        ),
    ]


def get_geometry_templates() -> List[GrammarRule]:
    """Templates for geometry."""
    return [
        # Rectangle area
        GrammarRule(
            rule_id="geo_rect_area",
            pattern=r"(?:rectangle|room).*?(\d+\.?\d*)\s*(?:by|×|x)\s*(\d+\.?\d*)",
            rpn_program="{0} {1} *",
            domain="math_geometry",
        ),
        # Circle area
        GrammarRule(
            rule_id="geo_circle_area",
            pattern=r"circle.*?radius\s*(?:of|is)?\s*(\d+\.?\d*)",
            rpn_program="{0} 2 pow 3.14159 *",  # π * r²
            domain="math_geometry",
        ),
        # Pythagorean
        GrammarRule(
            rule_id="geo_pythagorean",
            pattern=r"(\d+\.?\d*).*?(\d+\.?\d*).*?hypotenuse",
            rpn_program="{0} 2 pow {1} 2 pow + sqrt",
            domain="math_geometry",
        ),
    ]


def get_all_templates() -> List[GrammarRule]:
    """Get all curated templates."""
    templates = []
    templates.extend(get_arithmetic_templates())
    templates.extend(get_finance_templates())
    templates.extend(get_geometry_templates())
    return templates
```

### Task 2: Template-Aware Evaluation

**File:** `scripts/run_sovereign_math_benchmarks.py`

```python
def _apply_template(self, rule: GrammarRule, text: str) -> Optional[float]:
    """Apply a parametric template to problem text."""
    try:
        match = re.search(rule.pattern, text, re.IGNORECASE)
        if not match:
            return None

        # Extract captured numbers
        captures = match.groups()

        # Build RPN by substituting captures into template
        rpn_program = rule.rpn_program
        for i, val in enumerate(captures):
            rpn_program = rpn_program.replace(f"{{{i}}}", str(val))

        # Validate and execute
        if not is_valid_rpn(rpn_program):
            return None

        return self.rpn_engine.evaluate(rpn_program)
    except Exception:
        return None
```

### Task 3: Integrate Templates into Benchmark

```python
# In __init__:
from knowledge3d.training.math_benchmarks.math_templates import get_all_templates
self.template_rules = get_all_templates()

# In solve_problem():
# Try templates FIRST (high quality, curated)
for rule in self.template_rules:
    result = self._apply_template(rule, text)
    if result is not None:
        self._solve_stats["template"] += 1
        return result

# Then try word solver, composer, etc.
```

---

## Why This Works

| Approach | Problem | Solution |
|----------|---------|----------|
| **Literal patterns** | `"A = P(1+r)^t"` never matches problem text | **Regex capture groups** extract numbers from natural language |
| **Variable RPN** | `P 1 r + t pow *` has no values | **Template substitution** replaces `{0}`, `{1}` with captured numbers |
| **PDF garbage** | Extracts random text | **Curated library** of working templates |
| **Zero hits** | Pattern never matches | **Fuzzy patterns** match word problems |

---

## Expected Impact

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Template hits | 0 | 200-500 |
| gsm8k | 0.20% | 15-30% |
| math | 0.20% | 10-20% |
| Overall | ~5% | 15-25% |

**The key insight:** GSM8K problems are word problems with patterns like:
- "Natalia sold **48** clips... half as many... altogether"
- "John has **$100**, spends **$30**..."

These MATCH our templates if we use **capture groups**, not literal patterns.

---

## Codex: Your Original Ideas

1. **Auto-generate templates from working examples?**
   - When word solver succeeds, extract the pattern
   - Add to template library automatically

2. **Template ranking by domain?**
   - Use your domain specialists to score templates
   - Finance specialist boosts finance templates

3. **Multi-step template chaining?**
   - Some problems need: "calculate X, then apply Y"
   - Chain template results through scratchpad

4. **Template coverage analysis?**
   - Log which templates hit, which miss
   - Identify gaps in coverage

5. **Ternary-accelerated pattern matching?**
   - Use ternary signatures for fast template filtering
   - Skip templates that can't match

---

## Files to Create/Modify

1. **NEW:** `knowledge3d/training/math_benchmarks/math_templates.py`
2. **MODIFY:** `scripts/run_sovereign_math_benchmarks.py` (add template path)
3. **OPTIONAL:** Keep `math_knowledge_loader.py` for future proper PDF extraction

---

## Success Criteria

1. Template library has 20+ curated templates
2. `_apply_template()` correctly substitutes captures into RPN
3. Benchmark shows template hits > 0
4. gsm8k score improves to 10%+
5. All templates remain SOVEREIGN (no numpy/cupy)

---

## Key Principle

**Quality over quantity.** 20 working templates > 8,000 garbage extracts.

The PDF knowledge can be revisited later with proper extraction. For now, curated templates will prove the architecture works.
