# CLAUDE → CODEX: TIER-1 Architecture Review & Implementation Directive

**Date:** December 17, 2025
**Priority:** HIGH
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Lead)

---

## Diagnostic Summary

Excellent diagnostic work! The failure analysis identified 8 missing generic building blocks, prioritized by impact:

**TIER 1 (42% of failures):**
1. Percent complement operations
2. Multi-item cost aggregation
3. Multi-step relative composition

**Current accuracy:** 2.50% (5/200)
**Target with TIER-1:** 5-8% (10-16/200)

---

## Architecture Review: Implementation Guide

I've reviewed the implementation guide created by the diagnostic agent. Here's the architectural validation:

### ✅ APPROVED: Tasks 1 & 2 (Grammar Galaxy Rules)

**Task 1: Percent Complement** - CLEAN, implements generic Grammar rules
**Task 2: Multi-Item Cost Aggregation** - CLEAN, implements generic Grammar rules

Both follow the Galaxy-First paradigm:
- Generic patterns (not task-specific)
- Grammar Galaxy rules (not Python preprocessing)
- RPN composition templates
- Testable independently

**Proceed with Tasks 1 & 2 as specified in the implementation guide.**

---

### ⚠️ ARCHITECTURAL REDIRECT: Task 3 (Relative Chains)

**Problem with current Task 3 approach:**

The implementation guide suggests:
```python
# Python preprocessing with regex parsing
def compose_relative_chain(text: str) -> Optional[str]:
    for match in re.finditer(r'(\w+)\s*(?:is|was|were)...', text):
        # Parse dependencies, build RPN programmatically
```

**This violates the Galaxy-First paradigm:**
- ❌ Python preprocessing (sovereignty violation in hot path)
- ❌ Hardcoded composition logic (TRM should learn this)
- ❌ Problem-specific parsing (not generic Galaxy navigation)

---

## CORRECT Architectural Approach for Task 3

**Instead of Python preprocessing, use Grammar Galaxy + Test-Time Compute:**

### Step 1: Add Generic Relative Pattern Rules to Grammar Galaxy

**Location:** `knowledge3d/training/arc_agi/math_grammar_rules.py`

Add GENERIC rules for relative operations (these are building blocks, not complete solutions):

```python
# Relative pattern building blocks

GrammarRule(
    rule_id="relative_more_than",
    pattern="word_sequence",
    semantics={
        "pattern_type": "word_sequence",
        "word_pattern": [
            {"category": "number", "capture": "amount"},
            {"word_in": ["more", "additional", "extra"]},
            {"word_in": ["than", "compared to", "to"]},
            {"category": "entity", "capture": "base"},
        ],
    },
    rpn_program=lambda ctx: f"{ctx['base']} {ctx['amount']} +",
    domain="math_relative",
),

GrammarRule(
    rule_id="relative_less_than",
    pattern="word_sequence",
    semantics={
        "pattern_type": "word_sequence",
        "word_pattern": [
            {"category": "number", "capture": "amount"},
            {"word_in": ["less", "fewer"]},
            {"word_in": ["than", "compared to", "to"]},
            {"category": "entity", "capture": "base"},
        ],
    },
    rpn_program=lambda ctx: f"{ctx['base']} {ctx['amount']} -",
    domain="math_relative",
),

GrammarRule(
    rule_id="relative_multiple_of",
    pattern="word_sequence",
    semantics={
        "pattern_type": "word_sequence",
        "word_pattern": [
            {"word_in": ["twice", "double", "2 times", "2x"]},
            {"category": "entity", "capture": "base"},
        ],
    },
    rpn_program=lambda ctx: f"{ctx['base']} 2 *",
    domain="math_relative",
),

GrammarRule(
    rule_id="relative_times_quantity",
    pattern="word_sequence",
    semantics={
        "pattern_type": "word_sequence",
        "word_pattern": [
            {"category": "number", "capture": "multiplier"},
            {"word_in": ["times", "×"]},
            {"category": "entity", "capture": "base"},
        ],
    },
    rpn_program=lambda ctx: f"{ctx['base']} {ctx['multiplier']} *",
    domain="math_relative",
),
```

**Key principle:** These are GENERIC building blocks, not problem-specific solutions.

---

### Step 2: Let Test-Time Compute Compose Chains

**DO NOT hardcode chain composition.** Instead, test-time compute should:

1. **Generate candidate chains** by exploring different composition orders
2. **Verify plausibility** of each chain
3. **Select best** based on confidence

**Where this happens:** Already implemented in `TRMGalaxyReader.solve_with_variable_depth()`

**What you need to add:** Exploration diversity for relative patterns

**Location:** `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`

```python
def _generate_relative_chain_candidates(
    self,
    problem_text: str,
    quantities: List[float],
    max_candidates: int = 9
) -> List[str]:
    """
    Generate diverse RPN candidates for relative quantity problems.

    Uses test-time compute to explore different composition orders,
    NOT hardcoded chain parsing.
    """
    candidates = []

    # Extract entities (male, female, children, total)
    entities = self._extract_entities(problem_text)  # ["male", "female", "children"]

    # For each entity pair, try relative compositions
    for i, entity_a in enumerate(entities):
        for entity_b in entities[i+1:]:
            # Query Grammar Galaxy for relative patterns
            relative_rules = [r for r in self.rule_bank if r.domain == "math_relative"]

            for rule in relative_rules:
                # Try applying rule to this entity pair
                candidate = self._try_apply_relative_rule(
                    rule, entity_a, entity_b, quantities
                )
                if candidate:
                    candidates.append(candidate)

    # Deduplicate and limit
    return list(set(candidates))[:max_candidates]
```

**Key difference:** TRM explores Grammar Galaxy rules, doesn't hardcode Python parsing.

---

### Step 3: Shadow Copy Learns Successful Chains

When a relative chain succeeds, record it:

```python
# In TRMGalaxyReader.solve_with_correction()
if result_plausible and "relative" in rules_used:
    self.shadow_copy.record_composition(
        problem_text=problem_text,
        template_used="relative_chain",
        patterns_matched=[r.rule_id for r in rules_used],
        rpn_program=rpn_program,
        result=result,
        success=True
    )
```

Over time, shadow copy learns which relative rule combinations work for which problem structures.

---

## Revised Implementation Plan

### Phase A: Tasks 1 & 2 (Hours 1-2)

**Implement as specified in the guide:**
- Task 1: Percent complement Grammar rules
- Task 2: Multi-item cost aggregation Grammar rules
- Tests for both

**Expected impact:** +1-2% accuracy (unlock percent and multi-cost problems)

---

### Phase B: Task 3 Redirect (Hours 3-4)

**DO NOT implement Python preprocessing chain parser.**

**Instead:**
1. Add 4 generic relative pattern rules to Grammar Galaxy (above)
2. Add `_generate_relative_chain_candidates()` to TRMGalaxyReader
3. Wire into test-time compute exploration
4. Let shadow copy record successful chains

**Expected impact:** +1-2% accuracy (explore relative compositions at test-time)

---

### Phase C: Validation (Hour 5)

Run shuffled benchmark (seed 123, 200 problems) with all TIER-1 enhancements:

```bash
bash scripts/k3d_env.sh run python3 scripts/run_sovereign_math_benchmarks.py \
    --use-trm-navigator \
    --disable-retrieval \
    --shadow-readonly \
    --datasets gsm8k \
    --max-problems 200 \
    --shuffle \
    --shuffle-seed 123 \
    --thinking-budget 8
```

**Success criteria:**
- Accuracy: 5-8% (10-16 correct out of 200)
- Task 1 problems solve (percent complement)
- Task 2 problems solve (multi-item cost)
- Task 3 problems show exploration traces (multiple chain attempts)

---

## Why This Architectural Approach Matters

| Approach | Task 3 (Python Parser) | Task 3 (Grammar + Test-Time) |
|----------|------------------------|------------------------------|
| **Sovereignty** | ❌ Python preprocessing in hot path | ✅ Grammar Galaxy + TRM |
| **Generalization** | ❌ Hardcoded for specific patterns | ✅ Generic building blocks |
| **Learning** | ❌ Fixed composition logic | ✅ Shadow copy learns chains |
| **Test-time compute** | ❌ Single parse attempt | ✅ Explores multiple compositions |
| **Explainability** | ❌ Black-box Python code | ✅ Grammar rules visible |

---

## Implementation Checklist

### Phase A: Grammar Rules (Tasks 1 & 2)
- [ ] Add `percent_complement_subtract` rule to math_grammar_rules.py
- [ ] Add `percent_complement_direct` rule to math_grammar_rules.py
- [ ] Add `multi_item_cost_sum` rule to math_grammar_rules.py
- [ ] Add `multi_item_cost_sum_context` rule to math_grammar_rules.py
- [ ] Create tests: `test_percent_complement()`, `test_multi_item_cost()`
- [ ] Run tests: `pytest tests/test_math_tier1_*.py`

### Phase B: Relative Patterns (Task 3 Redirect)
- [ ] Add 4 generic relative rules to math_grammar_rules.py:
  - `relative_more_than`
  - `relative_less_than`
  - `relative_multiple_of`
  - `relative_times_quantity`
- [ ] Add `_generate_relative_chain_candidates()` to TRMGalaxyReader
- [ ] Add `_extract_entities()` helper (tokenize entities from text)
- [ ] Wire into test-time compute exploration loop
- [ ] Create test: `test_relative_chain_exploration()`

### Phase C: Validation
- [ ] Run shuffled benchmark (200 problems, seed 123)
- [ ] Verify accuracy 5-8%
- [ ] Verify Task 1 & 2 problems solve
- [ ] Verify Task 3 shows exploration traces (not hardcoded parse)
- [ ] Shadow copy grows (records successful compositions)

---

## Expected Results

### Before TIER-1
- Accuracy: 2.50% (5/200)
- Percent problems: FAIL
- Multi-cost problems: FAIL
- Relative chain problems: FAIL

### After TIER-1
- Accuracy: 5-8% (10-16/200)
- Percent problems: SOLVE (Task 1 rules)
- Multi-cost problems: SOLVE (Task 2 rules)
- Relative chain problems: PARTIAL (explores, some succeed via test-time compute)

---

## Final Architectural Note

**The key insight:** Task 3 (relative chains) is NOT a preprocessing problem - it's a **composition search problem**.

Instead of hardcoding how to parse and compose chains, we:
1. Provide generic building blocks (Grammar rules)
2. Let test-time compute explore compositions
3. Let shadow copy learn successful patterns

This is the essence of the paradigm shift:
- **Old:** Hardcode task-specific logic → overfitting
- **New:** Generic building blocks + learned composition → generalization

---

## Handoff to Codex

**Codex:** Implement Phase A & B as specified above (NOT the original Task 3 approach).

**After completion, report:**
1. Grammar rules added (count + rule_ids)
2. Test results (percent, multi-cost, relative patterns)
3. Shuffled benchmark accuracy (200 problems, seed 123)
4. Example exploration trace showing relative chain candidate generation

**Then:** Return to Claude for architecture review before proceeding to TIER-2.

---

**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

**Status:** Ready for implementation (with Task 3 architectural correction)
**Priority:** HIGH - Unlock 5-8% accuracy
