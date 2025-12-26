# CODEX: Template Coverage Expansion - From 4 Hits to 200+

**Date:** December 14, 2025
**Priority:** CRITICAL - Template system works but coverage is 0.8%
**Partner:** Claude (Architecture Analysis) -> Codex (Implementation + Original Ideas)

---

## Current State

Templates ARE WORKING:
- Natalia problem: Pattern matches, RPN executes, answer = 72 (correct!)
- But: Only 4 template hits out of 500 problems (0.8% coverage)
- Solve stats: template: 4, composer: 58, word: 20, knowledge: 1, fail: 1917

**Root Cause:** Patterns are too specific. GSM8K uses varied natural language.

---

## The Problem: Pattern Rigidity

Current pattern for "half altogether":
```python
r"(\d+\.?\d*).*?half\s*(?:as many|that many|as much).*?(?:altogether|total|in all)"
```

This ONLY matches if:
1. Number appears before "half"
2. "half" is followed by "as many/that many/as much"
3. Text ends with "altogether/total/in all"

But GSM8K problems say things like:
- "sold **48** clips... **half** that many... **how many** did she sell" (no "altogether")
- "**half** of the **48**..." (number after "half")
- "combined total" / "in total" / "all together" (different endings)

---

## Solution: Pattern Variant Explosion

### Task 1: Create Synonym-Rich Patterns

For EACH operation, create multiple pattern variants covering synonyms:

**File:** `knowledge3d/training/math_benchmarks/math_templates.py`

```python
def get_expanded_templates() -> List[GrammarRule]:
    """Expanded templates with synonym variants."""
    templates = []

    # === HALF PATTERNS (Multiple Variants) ===
    half_endings = [
        r"(?:altogether|total|in all|combined|in total|all together)",
        r"(?:how many|what is|find)",
        r"(?:did (?:she|he|they) (?:sell|make|have|get))",
    ]
    half_middles = [
        r"half\s*(?:as many|as much|that many|that much|of that)",
        r"(?:sold|made|had|got)\s*half",
    ]

    for middle in half_middles:
        for ending in half_endings:
            templates.append(GrammarRule(
                rule_id=f"gsm_half_v{len(templates)}",
                language="math",
                pattern=rf"(\d+\.?\d*).*?{middle}.*?{ending}",
                rpn_program="{0} DUP 2 / +",
                domain="math_arithmetic",
            ))

    # === PURCHASE/COST PATTERNS ===
    templates.extend([
        # "Each X costs $Y, bought Z" → Y * Z
        GrammarRule(
            rule_id="gsm_cost_each_bought",
            language="math",
            pattern=r"(?:each|per|every)\s*.*?(?:costs?|\$)\s*(\d+\.?\d*).*?(?:bought|purchased|got|buys?)\s*(\d+)",
            rpn_program="{0} {1} *",
            domain="math_arithmetic",
        ),
        # "$X each, Y items" → X * Y
        GrammarRule(
            rule_id="gsm_price_quantity",
            language="math",
            pattern=r"\$?(\d+\.?\d*)\s*(?:each|per|apiece).*?(\d+)\s*(?:items?|pieces?|units?|of them)",
            rpn_program="{0} {1} *",
            domain="math_arithmetic",
        ),
        # "bought X at $Y each" → X * Y
        GrammarRule(
            rule_id="gsm_bought_at_price",
            language="math",
            pattern=r"(?:bought|purchased|got)\s*(\d+).*?\$?(\d+\.?\d*)\s*(?:each|per|apiece)",
            rpn_program="{0} {1} *",
            domain="math_arithmetic",
        ),
    ])

    # === PER-DAY/WEEK/MONTH PATTERNS ===
    templates.extend([
        # "X per day for Y days" → X * Y
        GrammarRule(
            rule_id="gsm_per_day_for_days",
            language="math",
            pattern=r"(\d+\.?\d*)\s*(?:per|each|a|every)\s*day.*?(?:for)?\s*(\d+)\s*days?",
            rpn_program="{0} {1} *",
            domain="math_arithmetic",
        ),
        # "X a week for Y weeks" → X * Y
        GrammarRule(
            rule_id="gsm_per_week_for_weeks",
            language="math",
            pattern=r"(\d+\.?\d*)\s*(?:per|a|each|every)\s*week.*?(?:for)?\s*(\d+)\s*weeks?",
            rpn_program="{0} {1} *",
            domain="math_arithmetic",
        ),
        # "X hours a day, Y days" → X * Y
        GrammarRule(
            rule_id="gsm_hours_per_day",
            language="math",
            pattern=r"(\d+\.?\d*)\s*hours?\s*(?:a|per|each)\s*day.*?(\d+)\s*days?",
            rpn_program="{0} {1} *",
            domain="math_arithmetic",
        ),
    ])

    # === GAIN/LOSS PATTERNS ===
    templates.extend([
        # "has X, gets Y more" → X + Y
        GrammarRule(
            rule_id="gsm_has_gets_more",
            language="math",
            pattern=r"(?:has|have|had|starts? with)\s*(\d+\.?\d*).*?(?:gets?|gains?|receives?|finds?|earns?)\s*(\d+\.?\d*)\s*(?:more)?",
            rpn_program="{0} {1} +",
            domain="math_arithmetic",
        ),
        # "has X, loses Y" → X - Y
        GrammarRule(
            rule_id="gsm_has_loses",
            language="math",
            pattern=r"(?:has|have|had)\s*(\d+\.?\d*).*?(?:loses?|lost|gives? away|spent)\s*(\d+\.?\d*)",
            rpn_program="{0} {1} -",
            domain="math_arithmetic",
        ),
        # "X birds, Y fly away" → X - Y
        GrammarRule(
            rule_id="gsm_quantity_removed",
            language="math",
            pattern=r"(?:there (?:are|were)|has|have)\s*(\d+).*?(\d+)\s*(?:fly away|flew away|leave|left|are removed|are taken|were taken)",
            rpn_program="{0} {1} -",
            domain="math_arithmetic",
        ),
    ])

    # === MULTIPLICATION WITH "TIMES" ===
    templates.extend([
        # "X times as many as Y" → X * Y (note: X is the multiplier!)
        GrammarRule(
            rule_id="gsm_x_times_as_many_as_y",
            language="math",
            pattern=r"(\d+)\s*times\s*(?:as many|as much)\s*(?:as)?\s*.*?(\d+)",
            rpn_program="{0} {1} *",
            domain="math_arithmetic",
        ),
        # "Y has Z, X has N times that" → Z * N
        GrammarRule(
            rule_id="gsm_n_times_that",
            language="math",
            pattern=r"(\d+).*?(\d+)\s*times\s*(?:that|as much|as many|more)",
            rpn_program="{0} {1} *",
            domain="math_arithmetic",
        ),
    ])

    # === DISTRIBUTION PATTERNS ===
    templates.extend([
        # "X each for Y people" → X * Y
        GrammarRule(
            rule_id="gsm_each_for_people",
            language="math",
            pattern=r"(\d+\.?\d*)\s*(?:each|apiece).*?(?:for|to|among)?\s*(\d+)\s*(?:people|children|students|friends?|members?|guests?)",
            rpn_program="{0} {1} *",
            domain="math_arithmetic",
        ),
        # "Y people get X each" → X * Y
        GrammarRule(
            rule_id="gsm_people_get_each",
            language="math",
            pattern=r"(\d+)\s*(?:people|children|students|friends?).*?(?:get|receive|have)\s*(\d+\.?\d*)\s*(?:each|apiece)",
            rpn_program="{0} {1} *",
            domain="math_arithmetic",
        ),
    ])

    # === MULTI-STEP: A then B then C ===
    templates.extend([
        # "had X, got Y more, then Z more" → X + Y + Z
        GrammarRule(
            rule_id="gsm_add_three",
            language="math",
            pattern=r"(?:had|has|starts? with)\s*(\d+\.?\d*).*?(?:got|gets|gains?|receives?)\s*(\d+\.?\d*).*?(?:then|and|also).*?(\d+\.?\d*)\s*more",
            rpn_program="{0} {1} + {2} +",
            domain="math_arithmetic",
        ),
        # "X, minus Y, minus Z" → X - Y - Z
        GrammarRule(
            rule_id="gsm_sub_three",
            language="math",
            pattern=r"(\d+\.?\d*).*?(?:loses?|spent|gave)\s*(\d+\.?\d*).*?(?:then|and|also).*?(?:loses?|spent|gave)\s*(\d+\.?\d*)",
            rpn_program="{0} {1} - {2} -",
            domain="math_arithmetic",
        ),
    ])

    # === AGE PROBLEMS ===
    templates.extend([
        # "X years older than Y who is Z" → Z + X
        GrammarRule(
            rule_id="gsm_years_older",
            language="math",
            pattern=r"(\d+)\s*years?\s*older.*?(?:who is|is)\s*(\d+)",
            rpn_program="{1} {0} +",
            domain="math_arithmetic",
        ),
        # "X years younger than Y who is Z" → Z - X
        GrammarRule(
            rule_id="gsm_years_younger",
            language="math",
            pattern=r"(\d+)\s*years?\s*younger.*?(?:who is|is)\s*(\d+)",
            rpn_program="{1} {0} -",
            domain="math_arithmetic",
        ),
        # "In X years, will be Y" → Y - X (current age)
        GrammarRule(
            rule_id="gsm_in_years_will_be",
            language="math",
            pattern=r"[Ii]n\s*(\d+)\s*years?.*?(?:will be|be)\s*(\d+)",
            rpn_program="{1} {0} -",
            domain="math_arithmetic",
        ),
    ])

    # === WORK/RATE PROBLEMS ===
    templates.extend([
        # "X pages in Y hours" (rate)
        GrammarRule(
            rule_id="gsm_pages_in_hours",
            language="math",
            pattern=r"(\d+)\s*pages?.*?(?:in|per|every)\s*(\d+)\s*hours?",
            rpn_program="{0} {1} /",
            domain="math_arithmetic",
        ),
        # "earns $X per hour, works Y hours" → X * Y
        GrammarRule(
            rule_id="gsm_hourly_rate",
            language="math",
            pattern=r"(?:earns?|makes?|gets?)\s*\$?(\d+\.?\d*)\s*(?:per|an?|each)\s*hour.*?(?:works?|worked)\s*(\d+\.?\d*)\s*hours?",
            rpn_program="{0} {1} *",
            domain="math_arithmetic",
        ),
    ])

    # === REMAINING/LEFT OVER PATTERNS ===
    templates.extend([
        # "has X, uses Y, how many left" → X - Y
        GrammarRule(
            rule_id="gsm_uses_left",
            language="math",
            pattern=r"(?:has|have|had)\s*(\d+\.?\d*).*?(?:uses?|used|eats?|ate|gives?|gave)\s*(\d+\.?\d*).*?(?:left|remaining|remain)",
            rpn_program="{0} {1} -",
            domain="math_arithmetic",
        ),
        # "X total, Y are red, how many not red" → X - Y
        GrammarRule(
            rule_id="gsm_total_minus_some",
            language="math",
            pattern=r"(\d+)\s*(?:total|in all).*?(\d+)\s*(?:are|is|were).*?(?:how many|what)",
            rpn_program="{0} {1} -",
            domain="math_arithmetic",
        ),
    ])

    return templates
```

### Task 2: Add Number Word Recognition

**File:** `knowledge3d/training/math_benchmarks/number_words.py`

```python
"""
Convert number words to digits for template matching.
"""

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1000000,
    # Special words
    "dozen": 12, "score": 20, "half": 0.5, "quarter": 0.25,
    "pair": 2, "couple": 2, "few": 3, "several": 5,
}

def normalize_number_words(text: str) -> str:
    """Replace number words with digits for template matching."""
    import re
    result = text.lower()

    # Handle compound numbers: "twenty-five" → "25"
    for tens_word, tens_val in [("twenty", 20), ("thirty", 30), ("forty", 40),
                                  ("fifty", 50), ("sixty", 60), ("seventy", 70),
                                  ("eighty", 80), ("ninety", 90)]:
        for ones_word, ones_val in [("one", 1), ("two", 2), ("three", 3),
                                     ("four", 4), ("five", 5), ("six", 6),
                                     ("seven", 7), ("eight", 8), ("nine", 9)]:
            compound = f"{tens_word}[-\\s]?{ones_word}"
            result = re.sub(compound, str(tens_val + ones_val), result)

    # Handle simple words
    for word, val in NUMBER_WORDS.items():
        result = re.sub(rf"\b{word}\b", str(val), result)

    return result
```

**Integrate into `_apply_template()`:**

```python
def _apply_template(self, rule, text: str):
    """Apply a parametric template rule to text."""
    import re as _re
    from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

    try:
        # Try original text first
        m = _re.search(rule.pattern, text, _re.IGNORECASE | _re.DOTALL)

        # If no match, try with number words normalized
        if not m:
            normalized = normalize_number_words(text)
            m = _re.search(rule.pattern, normalized, _re.IGNORECASE | _re.DOTALL)

        if not m:
            return None

        rpn_program = rule.rpn_program
        for idx, group in enumerate(m.groups()):
            clean = str(group).replace(",", "").strip()
            rpn_program = rpn_program.replace(f"{{{idx}}}", clean)
        if not is_valid_rpn(rpn_program):
            return None
        return self.engine.evaluate(rpn_program)
    except Exception:
        return None
```

### Task 3: Template Priority Scoring

Templates should be tried in order of specificity:

```python
def get_all_templates() -> List[GrammarRule]:
    """Get all templates sorted by specificity (most specific first)."""
    templates = []
    templates.extend(get_gsm8k_templates())
    templates.extend(get_expanded_templates())

    # Sort by pattern length (longer = more specific)
    templates.sort(key=lambda r: len(r.pattern), reverse=True)
    return templates
```

### Task 4: Pattern Debugging Mode

Add debugging to identify which problems ALMOST match:

```python
def _debug_template_matches(self, text: str) -> None:
    """Debug: show partial matches for analysis."""
    import re as _re

    for rule in self.template_rules[:20]:  # Top 20 templates
        # Check if ANY part of pattern matches
        pattern_parts = rule.pattern.split(".*?")
        matches = []
        for part in pattern_parts:
            if _re.search(part, text, _re.IGNORECASE):
                matches.append(part)

        if matches and len(matches) < len(pattern_parts):
            print(f"  PARTIAL: {rule.rule_id} matched {len(matches)}/{len(pattern_parts)} parts")
```

---

## Expected Impact

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Template patterns | 22 | 80+ |
| Template hits | 4 | 100-200 |
| gsm8k | 0.20% | 15-25% |
| Overall | 4.80% | 12-20% |

---

## Key Principles

1. **Synonym Explosion**: Every operation needs 3-5 pattern variants
2. **Number Word Normalization**: "twelve" → "12" before matching
3. **Specificity Ordering**: Try specific patterns before generic ones
4. **Multi-step Support**: Patterns for "X then Y then Z" operations

---

## Success Criteria

1. Template library has 80+ patterns
2. Number word normalization works
3. Template hits > 100 in 500-sample benchmark
4. gsm8k score reaches 15%+
5. All templates remain SOVEREIGN (no numpy/cupy)

---

## Your Original Ideas, Codex

Consider:
1. **Template generation from successful word solver results?**
2. **Pattern mining from GSM8K solution annotations (`<<3*4=12>>`)?**
3. **Template chaining via scratchpad for multi-step?**
4. **Ternary-accelerated pattern pre-filtering?**
5. **Domain-aware template selection (finance vs geometry)?**

---

## Files to Modify

1. **EXPAND:** `knowledge3d/training/math_benchmarks/math_templates.py`
2. **CREATE:** `knowledge3d/training/math_benchmarks/number_words.py`
3. **MODIFY:** `scripts/run_sovereign_math_benchmarks.py` (_apply_template with normalization)

---

## Implementation Order

1. Add 40+ new pattern variants to math_templates.py
2. Create number_words.py with normalization
3. Integrate normalization into _apply_template()
4. Sort templates by specificity
5. Run benchmark, measure improvement
6. Iterate based on partial match analysis

**The architecture is proven - templates work! Now we need COVERAGE.**
