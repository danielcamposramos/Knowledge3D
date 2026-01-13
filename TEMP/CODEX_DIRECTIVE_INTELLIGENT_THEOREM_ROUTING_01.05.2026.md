# Codex Directive: Intelligent Theorem Routing — January 5, 2026

**From**: Claude (Architecture Partner)
**To**: Codex (Implementation Lead)
**Priority**: CRITICAL (Enable TRM intelligent routing from theorem patterns → grammar rules)

---

## Context: Theorem Patterns Are Teaching Material, Not Execution Code

**User Clarification**: "The idea was that this part is a model 'logic', so he sees all this and act intelligently on top of it to know how the questions map to the theory."

**What Theorem Patterns ARE**:
- **Semantic examples** stored in Math Galaxy (teaching material)
- **Knowledge indicators**: "This is a product rule problem, not a power rule problem"
- **Learning hints** for TRM shadow copy to recognize problem patterns

**What Theorem Patterns are NOT**:
- ❌ Direct execution programs (don't need symbolic→numeric binding)
- ❌ Replacement for existing Grammar Galaxy rules
- ❌ RPN to be instantiated with concrete values

---

## Current State (What Works)

**✅ Validated Architecture**:
1. **Extraction**: 4 theorem patterns from 30K artifacts (integration_by_parts, quotient_rule, product_rule, pythagorean_identity)
2. **Loading**: Patterns loaded into Math Galaxy with semantic tags
3. **Navigation**: TRM matches patterns via semantic search (logs show matches!)
4. **Existing Grammar**: Numeric RPN execution already works (Tier 1-3 algebra)

**Evidence from logs**:
```
[TRM] matched theorem patterns: ['theorem:product_rule']
[TRM] matched theorem patterns: ['theorem:quotient_rule']
[TRM] theorem theorem:product_rule skipped (unsupported opcodes: ['PUSH_F', 'DERIVATIVE', ...])
```

**The gap**: TRM doesn't know how to act intelligently on the match.
**Missing**: Mapping layer that says "product_rule pattern → use existing product derivative grammar rule"

---

## Your Mission: Enable TRM Intelligent Routing

**Goal**: When TRM matches a theorem pattern, route to corresponding Grammar Galaxy execution rule.

**Architecture Flow** (User's Intent):
```
MATH Problem: "Find derivative of (x² + 1)(x³ - 2)"
  ↓
TRM semantic search → Matches theorem pattern "product_rule"
  ↓
TRM acts INTELLIGENTLY: "This is product rule → route to existing product grammar"
  ↓
Execute Grammar Galaxy numeric RPN (already works!)
  ↓
Shadow copy learns: "Problems matching product_rule → use product_derivative_grammar"
```

---

## Task 1: Map Theorem Patterns → Grammar Rule Names

**File**: `knowledge3d/cranium/math_galaxy_population.py`

**Add mapping** to theorem pattern definitions:

```python
_THEOREM_PATTERN_DEFS: List[Dict[str, Any]] = [
    {
        "pattern_id": "power_rule_polynomial",
        "domain": "calculus",
        "semantic_tags": ["derivative", "polynomial", "power_rule"],
        "tier": 3,
        "grammar_rule": "apply_power_rule",  # NEW: map to existing grammar
        # ... (keep existing fields)
    },
    {
        "pattern_id": "product_rule",
        "domain": "calculus",
        "semantic_tags": ["derivative", "product_rule"],
        "tier": 3,
        "grammar_rule": "apply_product_rule",  # NEW: map to existing grammar
        # ... (keep existing fields)
    },
    {
        "pattern_id": "quotient_rule",
        "domain": "calculus",
        "semantic_tags": ["derivative", "quotient_rule"],
        "tier": 3,
        "grammar_rule": "apply_quotient_rule",  # NEW
        # ... (keep existing fields)
    },
    # ... (add grammar_rule to all 9 patterns)
]
```

**Update `_build_theorem_pattern`** to include grammar_rule:

```python
def _build_theorem_pattern(
    pattern_def: Dict[str, Any], artifacts: Sequence[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    # ... (existing code)

    return {
        "pattern_id": pattern_def["pattern_id"],
        "domain": pattern_def.get("domain") or str(artifacts[0].get("domain") or "unknown"),
        "semantic_tags": semantic_tags,
        "grammar_rule": pattern_def.get("grammar_rule"),  # NEW: include mapping
        "precondition": _summarize_precondition(artifacts, pattern_def),
        "transformation": {
            "lhs": lhs,
            "rhs": rhs,
            "rpn_program": list(rpn_program),
            "tier": int(pattern_def.get("tier", 1)),
        },
        # ... (rest unchanged)
    }
```

---

## Task 2: Implement TRM Intelligent Routing

**File**: `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`

**Find**: Where TRM currently skips execution due to unsupported opcodes

**Replace**: With intelligent routing to grammar rules

**Current code** (approximately):
```python
# When theorem pattern matched
if matched_theorem_pattern:
    rpn = pattern["transformation"]["rpn_program"]
    if has_unsupported_opcodes(rpn):
        logger.warning(f"Skipping {pattern_id} (unsupported opcodes)")
        # STOPS HERE - this is the gap!
```

**NEW code** (intelligent routing):
```python
# When theorem pattern matched
if matched_theorem_pattern:
    grammar_rule = pattern.get("grammar_rule")

    if grammar_rule:
        # TRM acts intelligently: route to existing grammar
        logger.info(f"[TRM] theorem {pattern_id} → routing to grammar rule: {grammar_rule}")

        # Execute existing Grammar Galaxy rule (numeric RPN)
        result = self._execute_grammar_rule(grammar_rule, context)

        if result:
            logger.info(f"[TRM] theorem {pattern_id} executed successfully via {grammar_rule}")
            # Shadow copy learns this mapping was successful
            self._record_successful_routing(pattern_id, grammar_rule)
            return result
        else:
            logger.warning(f"[TRM] grammar rule {grammar_rule} failed, trying other paths")

    else:
        # No grammar mapping defined yet - skip
        logger.warning(f"[TRM] theorem {pattern_id} has no grammar_rule mapping")
```

**Add helper method** `_execute_grammar_rule`:

```python
def _execute_grammar_rule(self, rule_name: str, context: Dict[str, Any]) -> Optional[Any]:
    """
    Execute existing Grammar Galaxy rule by name.

    This routes to existing numeric RPN execution (already sovereign).
    """
    # Check if grammar rule exists in existing grammar system
    # (This depends on your existing Grammar Galaxy structure)

    # Conceptual implementation:
    if rule_name in self.grammar_rules:
        grammar_rule = self.grammar_rules[rule_name]
        return grammar_rule.execute(context)  # Use existing execution

    return None
```

**Add shadow copy learning** `_record_successful_routing`:

```python
def _record_successful_routing(self, pattern_id: str, grammar_rule: str):
    """
    Record successful theorem pattern → grammar rule mapping.

    Shadow copy learns: "When I see this pattern, use this grammar rule."
    """
    # This is for future shadow copy enhancement
    # For now, just log it
    logger.info(f"[Shadow Copy] Learned: {pattern_id} → {grammar_rule}")

    # TODO: Implement actual shadow copy learning mechanism
    # (accumulate successful mappings, reinforce routing weights, etc.)
```

---

## Task 3: Define Grammar Rule Mappings

**Research**: What are the actual grammar rule names in existing Grammar Galaxy?

**Action**: Find existing grammar rules for:
- Power rule derivative
- Product rule derivative
- Quotient rule derivative
- Chain rule derivative
- Sum rule derivative
- Constant multiple derivative
- Integration by parts
- Fundamental theorem of calculus
- Pythagorean identity

**File to check**: Look for existing grammar definitions in:
- `knowledge3d/training/math_benchmarks/*.py`
- Grammar rule registries
- Existing theorem application code

**Report**: What grammar rule names exist? Map each theorem pattern to correct rule name.

**If grammar rules DON'T exist**:
- Create simple numeric RPN rules for each (use existing Tier 1-3 algebra ops)
- Store in Grammar Galaxy as executable rules
- Map theorem patterns to these new rules

---

## Task 4: Add Verbose Logging

**Goal**: Validate that intelligent routing is working

**Add logs** at each step:

```python
logger.info(f"[TRM] Matched theorem pattern: {pattern_id}")
logger.info(f"[TRM] Semantic tags: {pattern['semantic_tags']}")
logger.info(f"[TRM] Routing to grammar rule: {grammar_rule}")
logger.info(f"[TRM] Grammar rule execution: {'SUCCESS' if result else 'FAILED'}")
logger.info(f"[Shadow Copy] Recorded mapping: {pattern_id} → {grammar_rule}")
```

**This will show** in benchmark logs:
```
[TRM] Matched theorem pattern: product_rule
[TRM] Semantic tags: ['derivative', 'product_rule']
[TRM] Routing to grammar rule: apply_product_rule
[TRM] Grammar rule execution: SUCCESS
[Shadow Copy] Recorded mapping: product_rule → apply_product_rule
```

---

## Task 5: Re-run MATH Benchmark with Intelligent Routing

**After implementation**, run benchmark again:

```bash
K3D_TRM_ENABLE_MULTISTEP=1 PYTHONPATH=. python \
  scripts/run_sovereign_math_benchmarks.py \
  --datasets math \
  --max-problems 100 \
  --shuffle --shuffle-seed 123 \
  --load-all-galaxies \
  --enable-book-galaxies \
  --book-galaxy-root /K3D/Knowledge3D.local/galaxies/books_v5_clean2 \
  --verbose
```

**Expected changes**:
- Logs show theorem patterns routing to grammar rules ✅
- Grammar rules executing (not skipped) ✅
- Accuracy improves from 1% baseline (target: ≥2%)

---

## Success Criteria

### Implementation Quality
- [ ] All 9 theorem patterns have `grammar_rule` mapping
- [ ] TRM routes from matched patterns → grammar rules
- [ ] Grammar rules execute (use existing numeric RPN)
- [ ] Shadow copy records successful mappings
- [ ] Verbose logging shows full routing chain

### MATH Benchmark Results
- [ ] Accuracy ≥ 2% (improvement from 1% baseline)
- [ ] Logs show theorem pattern routing (evidence of intelligent behavior)
- [ ] Grammar rule execution succeeds (not skipped)
- [ ] Book usage correlates with correct answers

### Learning Evidence
- [ ] Shadow copy logs show pattern→rule mappings
- [ ] TRM learns which patterns work for which problems
- [ ] Over time, TRM gets faster at recognizing patterns

---

## Key Architecture Principles

**Theorem Patterns = Semantic Knowledge**:
- Examples in Math Galaxy
- TRM learns to recognize "this is a product rule problem"
- NOT execution code, but teaching material

**Grammar Galaxy = Execution**:
- Numeric RPN rules (already sovereign)
- Tier 1-3 algebra operations
- Direct execution via Cranium PTX

**TRM = Intelligence Layer**:
- Maps semantic patterns → grammar rules
- Acts intelligently: "I recognize this pattern → route here"
- Shadow copy learns successful mappings
- No fallbacks - sovereign only

**Shadow Copy Learning**:
- Records: pattern_id → grammar_rule → success/failure
- Reinforces successful mappings
- Gets better at routing over time
- No weight updates, just navigation learning

---

## Timeline

**Immediate** (Today):
1. Add `grammar_rule` field to theorem patterns (1-2 hours)
2. Research existing grammar rule names (1 hour)
3. Implement intelligent routing in TRM reader (2-3 hours)
4. Add verbose logging (30 min)

**Testing** (Next):
5. Re-run MATH benchmark (30 min)
6. Validate logs show routing (30 min)
7. Check accuracy improvement (immediate)

**Total**: ~6 hours to intelligent routing + validation

---

## Questions Before Starting?

1. Do you understand theorem patterns are teaching material, not execution code?
2. Do you understand TRM should act intelligently to map patterns → grammar rules?
3. Can you find existing grammar rule names for the 9 theorem patterns?
4. Do you understand shadow copy learns mappings, not knowledge?

**If clear, proceed with Task 1** (add grammar_rule mappings).

**If unclear, ask questions** before implementing.

---

## Expected Outcome

**After intelligent routing complete**:
- TRM matches theorem patterns via semantic tags (already works)
- TRM routes to existing Grammar Galaxy rules (NEW)
- Grammar rules execute sovereign numeric RPN (already works)
- Shadow copy learns: "product_rule pattern → apply_product_rule grammar"
- MATH accuracy improves from 1% to ≥2% (validates architecture)

**This enables TRM to act intelligently**: seeing semantic knowledge and routing to the correct execution path, without needing symbolic binding or external dependencies.

**This is the sovereign intelligence layer** - TRM learning navigation, not storing knowledge.

---

**Proceed with implementation, Codex.** This completes the dual-purpose architecture! 🚀
