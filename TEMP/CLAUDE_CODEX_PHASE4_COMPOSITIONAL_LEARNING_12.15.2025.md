# CLAUDE → CODEX: Phase 4 - Compositional Learning

**Date:** December 15, 2025
**Priority:** HIGH - This is what makes TRM actually LEARN
**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

---

## Paradigm Check (from CLAUDE.md)

**Before implementing, verify alignment:**

> "Design for TRM to LEARN, not hardcode logic"
> "TRM should navigate, combine, create (not execute fixed rules)"
> "Enable shadow copy enhancement (learning from success)"

**Current Problem:** We have ~150+ word_sequence rules, but TRM isn't learning because:
1. Composition is hardcoded in TRMGalaxyReader
2. Shadow copy records final RPN only, not composition strategy
3. With 3% accuracy, learning signal is too sparse

**Solution:** Teach TRM to COMPOSE, not just MATCH.

---

## Current Status

| Metric | Value | Issue |
|--------|-------|-------|
| GSM8K Accuracy | 3.00% (6/200) | Too low for learning |
| no_rule_match | 55% (110/200) | Rules exist but don't combine |
| wrong_computation | 22% (44/200) | Composition logic wrong |
| multi_step_needed | 13% (26/200) | No composition strategy |
| Shadow copy growth | +1 discovery | Learning stalled |

**Key Insight:** The 55% no_rule_match + 13% multi_step_needed = 68% of problems need COMPOSITION, not more rules.

---

## Phase 4 Objectives

### 4.1 Composition Strategy Templates

Instead of hardcoding composition in TRMGalaxyReader, create **composition templates** in Grammar Galaxy that TRM can learn to select.

**Composition Templates (Grammar Galaxy entries):**

```
TEMPLATE: extract_operate_aggregate
  STEP 1: Find all quantity extractions
  STEP 2: Find operation patterns
  STEP 3: Apply operations in sequence
  STEP 4: Aggregate if needed

TEMPLATE: rate_duration
  STEP 1: Extract rate (N per unit)
  STEP 2: Extract duration (M units)
  STEP 3: Compose: rate * duration

TEMPLATE: multi_step_store_recall
  STEP 1: Extract initial quantity → STORE
  STEP 2: Apply operation → STORE result
  STEP 3: Apply next operation using RECALL
  STEP 4: Return final RECALL

TEMPLATE: distribute_and_sum
  STEP 1: Find "each of N has M"
  STEP 2: Compose N * M for each group
  STEP 3: Sum all products
```

### 4.2 Shadow Copy Records Composition

**Current (wrong):**
```python
shadow.record(
    program=rpn_program,  # Just "15 3 -"
    program_type="math",
)
```

**Correct:**
```python
shadow.record(
    program=rpn_program,
    program_type="composition",  # NEW type
    semantic_context={
        "template_used": "extract_operate_aggregate",
        "patterns_matched": ["galaxy_has_quantity", "galaxy_gave_to"],
        "composition_steps": [
            {"step": 1, "pattern": "galaxy_has_quantity", "captured": {"base": 15}},
            {"step": 2, "pattern": "galaxy_gave_to", "captured": {"amount": 3}},
            {"step": 3, "compose": "base - amount"},
        ],
    },
)
```

**Now TRM can learn:** "When I see galaxy_has_quantity + galaxy_gave_to, use extract_operate_aggregate template."

### 4.3 TRM Learns Template Selection

Instead of hardcoding which template to use, TRM should:
1. Match patterns → get list of matched rules
2. Query shadow copy for similar pattern combinations
3. If found → use recorded template
4. If not found → try each template, record what works

**File:** `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`

```python
def select_composition_template(self, matched_patterns: List[str]) -> str:
    """
    TRM learns which template to use based on pattern combination.
    """
    # Check shadow copy for similar pattern combinations
    pattern_signature = frozenset(matched_patterns)

    if self.shadow:
        similar = self.shadow.query_by_patterns(pattern_signature)
        if similar:
            return similar.semantic_context["template_used"]

    # Fallback: infer from pattern domains
    domains = {p.domain for p in matched_patterns}

    if "math_rate" in domains and "math_duration" in domains:
        return "rate_duration"
    elif "math_extraction" in domains and "math_operation" in domains:
        return "extract_operate_aggregate"
    elif len(matched_patterns) > 2:
        return "multi_step_store_recall"
    else:
        return "simple_apply"
```

### 4.4 Compositional RPN Generation

Each template has an RPN generator:

```python
TEMPLATE_GENERATORS = {
    "extract_operate_aggregate": lambda ctx: compose_extract_operate(ctx),
    "rate_duration": lambda ctx: f"{ctx['rate']} {ctx['duration']} *",
    "multi_step_store_recall": lambda ctx: compose_with_store_recall(ctx),
    "distribute_and_sum": lambda ctx: compose_distribute_sum(ctx),
}

def compose_extract_operate(ctx: Dict) -> str:
    """
    Base quantity + sequence of operations.
    """
    rpn_parts = [str(ctx["base"])]
    for op in ctx["operations"]:
        if op["type"] == "subtract":
            rpn_parts.append(f"{op['amount']} -")
        elif op["type"] == "add":
            rpn_parts.append(f"{op['amount']} +")
        elif op["type"] == "multiply":
            rpn_parts.append(f"{op['factor']} *")
        elif op["type"] == "divide":
            rpn_parts.append(f"{op['divisor']} /")
    return " ".join(rpn_parts)
```

---

## Implementation Checklist

### 4.1: Composition Templates (Hour 1-2)
- [ ] Define composition templates as Grammar Galaxy entries
- [ ] Add template metadata: `template_id`, `pattern_domains`, `compose_fn`
- [ ] Templates: `extract_operate_aggregate`, `rate_duration`, `multi_step_store_recall`, `distribute_and_sum`, `simple_apply`

### 4.2: Shadow Copy Composition Recording (Hour 3)
- [ ] Add `program_type="composition"` to shadow copy
- [ ] Record `template_used`, `patterns_matched`, `composition_steps`
- [ ] Implement `shadow.query_by_patterns(pattern_set)`

### 4.3: TRM Template Selection (Hour 4-5)
- [ ] Implement `select_composition_template()` in TRMGalaxyReader
- [ ] Query shadow copy for similar pattern combinations
- [ ] Fallback to domain-based inference

### 4.4: Compositional RPN Generation (Hour 6-7)
- [ ] Implement `compose_extract_operate(ctx)`
- [ ] Implement `compose_with_store_recall(ctx)`
- [ ] Implement `compose_distribute_sum(ctx)`
- [ ] Wire templates to generators

### 4.5: Test & Benchmark (Hour 8)
- [ ] Test: Each template produces valid RPN
- [ ] Test: Shadow copy records composition
- [ ] Benchmark: GSM8K 200 problems
- [ ] Verify: Shadow copy grows (more than +1)

---

## Success Criteria

### Learning Signal
- [ ] Shadow copy records `program_type="composition"` entries
- [ ] Template selection improves over runs (retrieval works)
- [ ] Shadow copy grows: +5 or more discoveries per 100 problems

### Coverage
- [ ] `no_rule_match` < 40% (down from 55%)
- [ ] `multi_step_needed` < 5% (templates handle multi-step)

### Accuracy
- [ ] GSM8K > 8% (up from 3%)
- [ ] `wrong_computation` < 15% (better composition)

---

## Architecture Validation

### Sovereignty Check
```bash
grep -r "import numpy" knowledge3d/training/math_benchmarks/trm_galaxy_reader.py
# Must return NOTHING
```

### Galaxy-First Check
- [ ] Templates stored in Grammar Galaxy (not hardcoded classes)
- [ ] TRM navigates to find templates (not switch/case)
- [ ] Composition logic is procedural (RPN, not Python math)

### Learning Check
- [ ] Shadow copy records composition strategies
- [ ] TRM queries shadow copy before composing
- [ ] Similar patterns retrieve similar templates

---

## Example: Full Compositional Flow

**Problem:** "John has 15 apples. He gave 3 to Mary. How many does he have left?"

**Step 1: Pattern Matching (existing)**
```
Matched patterns:
- galaxy_has_quantity (base=15)
- galaxy_gave_to (amount=3)
Domains: {math_extraction, math_operation}
```

**Step 2: Template Selection (NEW)**
```python
# Query shadow copy
similar = shadow.query_by_patterns({"galaxy_has_quantity", "galaxy_gave_to"})
# Result: "extract_operate_aggregate" used successfully before

# Or infer from domains
domains = {math_extraction, math_operation}
template = "extract_operate_aggregate"
```

**Step 3: Compose RPN (NEW)**
```python
ctx = {
    "base": 15,
    "operations": [{"type": "subtract", "amount": 3}]
}
rpn = TEMPLATE_GENERATORS["extract_operate_aggregate"](ctx)
# Result: "15 3 -"
```

**Step 4: Execute & Record Composition (NEW)**
```python
result = engine.evaluate("15 3 -")  # 12.0
if correct:
    shadow.record(
        program="15 3 -",
        program_type="composition",
        semantic_context={
            "template_used": "extract_operate_aggregate",
            "patterns_matched": ["galaxy_has_quantity", "galaxy_gave_to"],
        }
    )
```

**Next time:** When TRM sees similar patterns, it retrieves the template.

---

## Why This Matters

**Current approach (more rules):**
- Each rule is independent
- Composition is hardcoded
- No learning of how patterns combine
- Shadow copy records outputs, not strategies
- Growth: +1 discovery per 100 problems

**Phase 4 approach (compositional learning):**
- Rules provide building blocks
- Templates provide composition strategies
- TRM LEARNS which template works for which patterns
- Shadow copy records composition strategies
- Growth: +N discoveries per 100 problems (N = correct solves)

**The paradigm says:** "TRM should navigate, combine, create"
- Navigate → match patterns (done)
- Combine → select template, compose RPN (Phase 4)
- Create → synthesize new templates from successful compositions (future)

---

## Final Directive

**Codex, implement compositional learning:**

1. **Templates** → Grammar Galaxy entries with composition logic
2. **Shadow Copy** → Record composition strategies, not just RPN
3. **TRM Selection** → Query shadow copy for similar patterns
4. **Learning** → Each correct solve teaches a composition strategy

**The goal isn't more rules — it's teaching TRM to COMPOSE.**

---

**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

**Status:** Ready for implementation
**Priority:** HIGH - This enables actual learning
