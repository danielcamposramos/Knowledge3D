# CLAUDE → CODEX: Phase 3 - Galaxy Coverage Expansion

**Date:** December 15, 2025
**Priority:** HIGH - Coverage is the bottleneck
**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

---

## Paradigm Reminder (from CLAUDE.md)

**Before implementing, internalize these principles:**

### Galaxy-First Design
> "Ask: 'Should this be in Galaxy or hardcoded?' (Answer: Galaxy)"
> "Patterns → Grammar Galaxy rules"
> "Knowledge → procedural programs in Galaxy Universe"

### TRM Navigation Patterns
> "Design for TRM to LEARN, not hardcode logic"
> "TRM should navigate, combine, create (not execute fixed rules)"
> "Enable shadow copy enhancement (learning from success)"

### Sovereignty (Hot Path)
> "NO CPU preprocessing (use Galaxy navigation instead)"
> "PTX kernels + Galaxy Universe + RPN programs = sovereign"

---

## Current Status

### What's Working
| Test | Result | Method |
|------|--------|--------|
| `5!` = 120 | PASS | factorial rule |
| `25% of 80` = 20 | PASS | percent rule |
| **Natalia's clips = 72** | PASS | **galaxy_read** |

### What's Broken
| Issue | Impact | Root Cause |
|-------|--------|------------|
| `24 / 4` returns None | Division fails | Placeholder `{0} {1} /` not filled |
| 68% no_rule_match | Low coverage | Not enough Galaxy-aware rules |

### Failure Analysis
```
wrong_computation: 17% (down from 50% - FIXED!)
no_rule_match: 68% (NEW bottleneck)
```

---

## Phase 3 Objectives

### 3.1 Fix Placeholder Bug in Division Patterns

**Problem:** Division rule returns template with unfilled placeholders.

**Current (broken):**
```python
GrammarRule(
    rule_id="division",
    pattern=r"(\d+)\s*/\s*(\d+)",
    rpn_program="{0} {1} /",  # Placeholders not filled!
)
```

**Fix:** Use lambda to fill placeholders from match groups.

```python
GrammarRule(
    rule_id="division",
    pattern=r"(\d+)\s*/\s*(\d+)",
    rpn_program=lambda m: f"{m.group(1)} {m.group(2)} /",  # Fills from match
)
```

**Or use `{g0}` `{g1}` syntax** that TRM Navigator already handles:
```python
rpn_program="{g0} {g1} /",  # Navigator replaces {gN} with match.group(N)
```

### 3.2 Expand Galaxy-Aware Rules (Word Sequence)

**Principle:** Don't add more regex patterns. Add more **word_sequence** rules that TRM can learn to navigate.

**File:** `knowledge3d/training/arc_agi/math_grammar_rules.py`

Add to `GALAXY_AWARE_RULES`:

```python
GALAXY_AWARE_RULES = [
    # === EXTRACTION PATTERNS ===
    # "[entity] has/had/owns [N] [items]" → extract base quantity
    GrammarRule(
        rule_id="galaxy_has_quantity",
        pattern="word_sequence",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "proper_noun"},
                {"word_in": ["has", "had", "owns", "bought", "sold", "made"]},
                {"category": "number", "capture": "base"},
                {"category": "noun"},
            ],
        },
        rpn_program=lambda ctx: str(ctx.get("base", 0)),
        domain="math_extraction",
    ),

    # "[N] divided by [M]" or "[N] / [M]"
    GrammarRule(
        rule_id="galaxy_divided_by",
        pattern="word_sequence",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "dividend"},
                {"word_in": ["divided", "/"]},
                {"word_in": ["by", ""]},
                {"category": "number", "capture": "divisor"},
            ],
        },
        rpn_program=lambda ctx: f"{ctx['dividend']} {ctx['divisor']} /",
        domain="math_arithmetic",
    ),

    # "[N] times [M]" or "[N] * [M]"
    GrammarRule(
        rule_id="galaxy_times",
        pattern="word_sequence",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "a"},
                {"word_in": ["times", "*", "multiplied"]},
                {"category": "number", "capture": "b"},
            ],
        },
        rpn_program=lambda ctx: f"{ctx['a']} {ctx['b']} *",
        domain="math_arithmetic",
    ),

    # "[N] plus [M]" or "[N] + [M]"
    GrammarRule(
        rule_id="galaxy_plus",
        pattern="word_sequence",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "a"},
                {"word_in": ["plus", "+", "and"]},
                {"category": "number", "capture": "b"},
            ],
        },
        rpn_program=lambda ctx: f"{ctx['a']} {ctx['b']} +",
        domain="math_arithmetic",
    ),

    # "[N] minus [M]" or "[N] - [M]"
    GrammarRule(
        rule_id="galaxy_minus",
        pattern="word_sequence",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "a"},
                {"word_in": ["minus", "-", "less"]},
                {"category": "number", "capture": "b"},
            ],
        },
        rpn_program=lambda ctx: f"{ctx['a']} {ctx['b']} -",
        domain="math_arithmetic",
    ),

    # === OPERATION PATTERNS (GSM8K) ===
    # "twice as many" → * 2
    GrammarRule(
        rule_id="galaxy_twice_as_many",
        pattern="word_sequence",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word": "twice"},
                {"word": "as"},
                {"word": "many"},
            ],
        },
        rpn_program=lambda ctx: f"{ctx.get('base', 0)} 2 *",
        domain="math_operation",
    ),

    # "three times as many" → * 3
    GrammarRule(
        rule_id="galaxy_n_times_as_many",
        pattern="word_sequence",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"category": "number", "capture": "multiplier"},
                {"word": "times"},
                {"word": "as"},
                {"word": "many"},
            ],
        },
        rpn_program=lambda ctx: f"{ctx.get('base', 0)} {ctx['multiplier']} *",
        domain="math_operation",
    ),

    # "gave [N] to" → subtract
    GrammarRule(
        rule_id="galaxy_gave_to",
        pattern="word_sequence",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word_in": ["gave", "gives", "give"]},
                {"category": "number", "capture": "amount"},
                {"word": "to"},
            ],
        },
        rpn_program=lambda ctx: f"{ctx['amount']} -",
        domain="math_operation",
    ),

    # "received [N]" → add
    GrammarRule(
        rule_id="galaxy_received",
        pattern="word_sequence",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word_in": ["received", "gets", "got", "gained"]},
                {"category": "number", "capture": "amount"},
            ],
        },
        rpn_program=lambda ctx: f"{ctx['amount']} +",
        domain="math_operation",
    ),

    # === AGGREGATION PATTERNS ===
    # "how many ... altogether/total/in all"
    GrammarRule(
        rule_id="galaxy_how_many_total",
        pattern="word_sequence",
        semantics={
            "pattern_type": "word_sequence",
            "word_pattern": [
                {"word": "how"},
                {"word": "many"},
            ],
        },
        rpn_program=lambda ctx: "",  # Signals: aggregate results
        domain="math_aggregation",
    ),
]
```

### 3.3 Enhance TRMGalaxyReader Composition

**File:** `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`

The reader should compose RPN by:
1. Finding base quantities (extraction patterns)
2. Applying operations in sequence
3. Aggregating at the end

```python
def compose_rpn(self, understanding: ProblemUnderstanding) -> str:
    """
    Compose RPN from structured understanding.

    TRM learns HOW to compose - this is the framework.
    """
    rpn_parts = []

    # Step 1: Add base quantities
    for qty in understanding.quantities:
        rpn_parts.append(str(qty["value"]))

    # Step 2: Apply each operation
    for op in understanding.operations:
        if op["type"] == "multiply":
            rpn_parts.append(f"{op['factor']} *")
        elif op["type"] == "divide":
            rpn_parts.append(f"{op['divisor']} /")
        elif op["type"] == "add":
            rpn_parts.append(f"{op['amount']} +")
        elif op["type"] == "subtract":
            rpn_parts.append(f"{op['amount']} -")

    # Step 3: Aggregation (sum multiple quantities)
    if len(understanding.quantities) > 1 and understanding.aggregation == "sum":
        # Already have quantities on stack, just add them
        for _ in range(len(understanding.quantities) - 1):
            rpn_parts.append("+")

    return " ".join(rpn_parts)
```

### 3.4 Shadow Copy Records Reading Success

**Principle:** TRM should LEARN to read. Shadow copy must record successful Galaxy reading.

**File:** `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`

```python
def record_reading_success(
    self,
    problem_text: str,
    understanding: ProblemUnderstanding,
    trace: Dict,
    rpn_program: str,
    confidence: float,
) -> None:
    """
    Record successful reading to shadow copy.

    This enables TRM to learn which word patterns work.
    """
    if self.shadow is None:
        return

    task_signature = {
        "problem_text": problem_text[:200],
        "patterns_matched": [p["rule_id"] for p in trace.get("patterns", [])],
        "quantities_found": len(understanding.quantities),
        "operations_found": len(understanding.operations),
    }

    self.shadow.record(
        task_signature=task_signature,
        program=rpn_program,
        program_type="reading",  # Special type for reading discoveries
        score=confidence,
        task_id=f"read_{hash(problem_text) % 10000}",
        semantic_context={
            "word_patterns": trace.get("patterns", []),
            "composition_strategy": understanding.to_dict(),
        },
    )
```

---

## Implementation Checklist

### Phase 3.1: Fix Placeholder Bug (Hour 1)
- [ ] Find all rules with `{0}` `{1}` placeholders
- [ ] Convert to `{g0}` `{g1}` OR lambda functions
- [ ] Test: `24 / 4` → 6.0

### Phase 3.2: Add Galaxy-Aware Rules (Hours 2-4)
- [ ] Add `galaxy_divided_by` word_sequence rule
- [ ] Add `galaxy_times` word_sequence rule
- [ ] Add `galaxy_plus`, `galaxy_minus` rules
- [ ] Add `galaxy_gave_to`, `galaxy_received` rules
- [ ] Add `galaxy_twice_as_many`, `galaxy_n_times_as_many`
- [ ] Wire into GALAXY_AWARE_RULES list

### Phase 3.3: Enhance Composition (Hours 4-6)
- [ ] Update `compose_rpn()` to handle operation sequences
- [ ] Handle multiple quantities with aggregation
- [ ] Test: Multi-step word problems

### Phase 3.4: Shadow Copy Reading (Hour 7)
- [ ] Implement `record_reading_success()` in TRMGalaxyReader
- [ ] Wire shadow copy into reader
- [ ] Test: Reading discoveries appear in shadow copy

### Phase 3.5: Benchmark (Hour 8)
- [ ] Run: `--use-trm-navigator --datasets gsm8k --max-problems 100`
- [ ] Verify: `no_rule_match` < 50% (down from 68%)
- [ ] Verify: GSM8K accuracy > 10%

---

## Success Criteria

### Functional
- [ ] `24 / 4` → 6.0 (placeholder fixed)
- [ ] `24 divided by 4` → 6.0 (word_sequence rule)
- [ ] Natalia's clips → 72 (still works)
- [ ] Multi-operation problems compose correctly

### Coverage
- [ ] `no_rule_match` < 50% (down from 68%)
- [ ] GSM8K accuracy > 10% (up from 2%)

### Learning
- [ ] Shadow copy records reading successes
- [ ] Pattern confidence updates on Galaxy read
- [ ] Reading discoveries committed to Grammar Galaxy

---

## Architecture Validation

Before considering Phase 3 complete, verify:

### Sovereignty Check
```bash
# No numpy in hot path
grep -r "import numpy" knowledge3d/training/math_benchmarks/trm_galaxy_reader.py
# Should return NOTHING
```

### Galaxy-First Check
- [ ] New patterns are word_sequence rules (not regex)
- [ ] Patterns stored in Grammar Galaxy (not hardcoded)
- [ ] TRM navigates Galaxy to read (not external preprocessing)

### Shadow Copy Check
- [ ] Reading success recorded with `program_type="reading"`
- [ ] Word patterns stored in semantic_context
- [ ] Pattern confidence updated

---

## Example: Full Flow

**Problem:** "John has 15 apples. He gave 3 to Mary. How many does he have left?"

**Step 1: Word Galaxy Tokenization**
```
["John", "has", "15", "apples", "He", "gave", "3", "to", "Mary", ...]
↓ Word Galaxy lookup
[proper_noun, verb, number(15), noun, pronoun, verb, number(3), prep, proper_noun, ...]
```

**Step 2: Grammar Galaxy Pattern Matching**
```
Pattern: galaxy_has_quantity
  Match: "John has 15 apples"
  Capture: base=15

Pattern: galaxy_gave_to
  Match: "gave 3 to"
  Capture: amount=3, operation=subtract
```

**Step 3: Compose RPN**
```
understanding.quantities = [{value: 15}]
understanding.operations = [{type: "subtract", amount: 3}]

RPN: "15 3 -"
```

**Step 4: Execute & Record**
```
result = engine.evaluate("15 3 -") → 12.0
shadow.record(program_type="reading", ...)
```

---

## Final Directive

**Codex, follow the paradigm:**

1. **Galaxy-First:** Add word_sequence rules, not regex patterns
2. **TRM Learns:** Shadow copy records reading successes
3. **Sovereign:** No numpy/external preprocessing in hot path
4. **Fix Before Expand:** Fix placeholder bug first, then add coverage

**The Galaxy reading paradigm is PROVEN (Natalia = 72). Now expand coverage.**

---

**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

**Status:** Ready for implementation
**Priority:** HIGH - Coverage is the accuracy bottleneck
