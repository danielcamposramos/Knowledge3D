# Claude + Gemini Router Architecture Synthesis

**Date**: January 5, 2026
**Authors**: Claude (Architecture Partner) + Gemini (Universal Integration)
**Purpose**: Unified architectural guidance for Codex on Router Specialist implementation

---

## Critical Architectural Insights

### 1. Hash Embeddings Violate Galaxy Universe Paradigm (Gemini's Critical Finding)

**Problem**: Hash-based semantic embeddings break the core K3D principle:
- **Galaxy Universe**: "Semantic Proximity = Spatial Proximity"
- **Hash functions**: Intentionally destroy locality (orthogonal vectors for similar concepts)
- **Consequence**: Router can memorize but NOT generalize

**Solution**: Galaxy-Anchored Embeddings (Gemini's proposal)
```
Theorem pattern text → Tokenize
                    → Lookup symbols in MATH_SYMBOL_GALAXY / WORD_GALAXY
                    → Pool vectors (Mean/Max pooling)
                    → Router input (Galaxy-aligned semantic space)
```

**Why this is correct**:
- ✅ Uses Galaxy as single source of truth
- ✅ Preserves semantic proximity (similar theorems → close vectors)
- ✅ Router navigates SAME semantic space as TRM
- ✅ Generalizes to unseen phrasings (if Galaxy has the symbols)
- ✅ Sovereign (no external embeddings)

**Fallback for sparse Galaxy** (Gemini's suggestion):
- Character N-gram embeddings (sovereign, locality-preserving)
- Better than hash (preserves substring similarity)
- Example: "integral" ≈ "integration" (shared n-grams: "inte", "tegr", "gral")

---

### 2. Synthetic Data Validation Filter (Gemini's Safety Check)

**Problem**: Ollama (deepseek-r1:7b) might hallucinate rule mappings that don't exist in Grammar Galaxy

**Solution**: Pre-training validation filter
```python
def validate_synthetic_routing_decision(decision: RoutingDecision) -> bool:
    """
    Validate that the Ollama-suggested rule actually applies to the pattern.

    Returns:
        True if grammar rule pattern matches theorem pattern context
        False if Ollama hallucinated an invalid mapping
    """
    pattern_text = decision.task_description
    grammar_rule = max(decision.specialist_weights.items(), key=lambda x: x[1])[0]

    # Get grammar rule from Grammar Galaxy
    rule = GRAMMAR_GALAXY.get_rule(grammar_rule)
    if rule is None:
        return False  # Rule doesn't exist

    # Check if rule pattern matches the theorem context
    match = re.search(rule.pattern, pattern_text, re.IGNORECASE)

    return match is not None  # Only train on valid mappings
```

**Filter before training**:
```python
# Before training router specialist
valid_decisions = [
    d for d in ollama_decisions
    if validate_synthetic_routing_decision(d)
]

print(f"Filtered: {len(ollama_decisions)} → {len(valid_decisions)} valid decisions")
print(f"Discarded {len(ollama_decisions) - len(valid_decisions)} hallucinations")

# Train only on validated data
trainer.train_from_history(valid_decisions, epochs=10)
```

---

### 3. Confidence Gating (Gemini's Safety Valve)

**Problem**: Untrained/uncertain router shouldn't force execution on low-confidence decisions

**Solution**: Router confidence threshold with exploration fallback
```python
def route_with_confidence(
    router_weights: Dict[str, float],
    confidence_threshold: float = 0.6
) -> Tuple[str, bool]:
    """
    Route with confidence gating.

    Returns:
        (grammar_rule, is_confident)
    """
    max_weight = max(router_weights.values())

    if max_weight < confidence_threshold:
        # Router is unsure → Enable exploration
        return None, False  # Signal: try beam search (Top-3 rules)
    else:
        # Router is confident → Execute best rule
        best_rule = max(router_weights.items(), key=lambda x: x[1])[0]
        return best_rule, True
```

**Execution logic**:
```python
grammar_rule, is_confident = route_with_confidence(router_weights)

if is_confident:
    # High confidence → Execute single best rule
    result = execute_grammar_rule(grammar_rule, problem_text)
else:
    # Low confidence → Beam search (Top-3 rules)
    top_3_rules = sorted(router_weights.items(), key=lambda x: -x[1])[:3]
    for rule, weight in top_3_rules:
        result = execute_grammar_rule(rule, problem_text)
        if result is not None:
            # Success → Record for shadow copy
            record_successful_exploration(rule, weight, result)
            break
```

**Benefits**:
- Prevents "default to one rule" collapse
- Enables exploration when uncertain
- Records successful explorations for future training
- Safety valve for undertrained router

---

### 4. Rule Entropy Metric (Gemini's Validation Strategy)

**Problem**: Need to detect if router has collapsed to single strategy (current 0% bug)

**Solution**: Shannon entropy of rule selection distribution
```python
from scipy.stats import entropy

def compute_rule_entropy(rule_selections: List[str]) -> float:
    """
    Compute Shannon entropy of rule selection distribution.

    High entropy (≈ log(num_rules)) = diverse tool usage (healthy)
    Low entropy (≈ 0) = collapsed to single rule (bug)
    """
    from collections import Counter

    counts = Counter(rule_selections)
    total = len(rule_selections)

    probabilities = [count / total for count in counts.values()]

    return entropy(probabilities, base=2)

# Example usage in benchmark
rule_selections = []  # Collect all router decisions

for problem in benchmark:
    grammar_rule, _ = route_with_confidence(router_weights)
    rule_selections.append(grammar_rule)
    # ... execute problem

# Log entropy after benchmark
rule_entropy = compute_rule_entropy(rule_selections)
max_entropy = np.log2(len(CALCULUS_RULES))  # Maximum possible entropy

print(f"Rule Selection Entropy: {rule_entropy:.2f} / {max_entropy:.2f}")
print(f"Diversity: {rule_entropy / max_entropy * 100:.1f}%")

# Interpretation
if rule_entropy < 0.5:
    print("⚠️  Router has collapsed to single strategy")
elif rule_entropy > 0.8 * max_entropy:
    print("✅ Router is using diverse tools")
```

**Log in benchmark output**:
```
=== Router Specialist Metrics ===
Rule Selection Entropy: 2.1 / 3.17 (66.2% diversity)
Rule Usage Distribution:
  apply_power_rule: 23 (23%)
  apply_product_rule: 18 (18%)
  apply_quotient_rule: 15 (15%)
  apply_sum_rule: 12 (12%)
  apply_constant_multiple_rule: 32 (32%)  ← Still most common but not dominant
```

---

### 5. Self-Improvement Cycle (Gemini's Vision)

**Goal**: Close the loop between synthetic data and real benchmark feedback

**Architecture**:
```
Phase 1: Ollama Bootstrap
    ↓
  Synthetic routing decisions (180 examples)
    ↓
  Train router specialist (initial weights)
    ↓
Phase 2: Real Benchmark Validation
    ↓
  Run MATH benchmark with trained router
    ↓
  Collect successful routing decisions (real feedback)
    ↓
Phase 3: Hybrid Training (Synthetic + Real)
    ↓
  Mix: 70% Ollama synthetic + 30% real successes
    ↓
  Retrain router specialist (improved weights)
    ↓
Phase 4: Continual Learning
    ↓
  Every N benchmark runs → Augment training data
    ↓
  Router self-improves from experience
```

**Implementation**:
```python
def augment_training_data(
    synthetic_decisions: List[RoutingDecision],
    real_successes: List[RoutingDecision],
    mix_ratio: float = 0.7
) -> List[RoutingDecision]:
    """
    Mix synthetic + real routing decisions for hybrid training.

    Args:
        synthetic_decisions: Ollama-generated decisions
        real_successes: Successful decisions from benchmark runs
        mix_ratio: Fraction of synthetic data (0.7 = 70% synthetic, 30% real)
    """
    num_synthetic = int(len(synthetic_decisions) * mix_ratio)
    num_real = len(synthetic_decisions) - num_synthetic

    # Sample from each pool
    sampled_synthetic = random.sample(synthetic_decisions, num_synthetic)
    sampled_real = random.sample(real_successes, min(num_real, len(real_successes)))

    return sampled_synthetic + sampled_real

# Usage
hybrid_data = augment_training_data(ollama_decisions, benchmark_successes)
trainer.train_from_history(hybrid_data, epochs=5)
```

**Benefits**:
- Synthetic data provides coverage (all patterns)
- Real data provides grounding (actual benchmark feedback)
- Router learns from BOTH reasoning (Ollama) AND experience (benchmarks)
- Continual improvement over time

---

## Immediate Implementation Priorities (For Codex)

### Phase 1.5: Before First Benchmark Run (CRITICAL)

**Task 1**: Replace hash embeddings with Galaxy-Anchored Embeddings
- Lookup theorem pattern tokens in Math/Word Galaxy
- Pool vectors (mean pooling for simplicity)
- Fallback to character n-grams if Galaxy is sparse
- **Why critical**: Hash embeddings can only memorize, not generalize

**Task 2**: Add synthetic data validation filter
- Validate Ollama decisions against Grammar Galaxy
- Discard hallucinated rule mappings
- **Why critical**: Don't train router on invalid data

**Task 3**: Implement confidence gating
- Router outputs confidence scalar
- Threshold: 0.6 (tune later)
- Fallback to beam search (Top-3 rules) if low confidence
- **Why critical**: Prevents collapse to single strategy

**Task 4**: Add rule entropy logging
- Compute Shannon entropy of rule selections
- Log in benchmark output
- **Why critical**: Validates router is using diverse tools

### Phase 2: After First Benchmark Run

**Task 5**: Collect real successes for hybrid training
- Record successful routing decisions from benchmark
- Mix 70% Ollama synthetic + 30% real successes
- Retrain router specialist

**Task 6**: Implement continual learning pipeline
- Every 100 benchmark runs → augment training data
- Router self-improves from experience

---

## Success Criteria (Revised with Gemini's Metrics)

### Benchmark Run Validation
- [ ] **Accuracy** ≥ 2% (improvement from 0% baseline)
- [ ] **Rule Entropy** ≥ 1.5 bits (diversity ≥ 50% of maximum)
- [ ] **Execution Rate** ≥ 30% (grammar rules actually execute, not skipped)
- [ ] **Confidence Distribution** shows variance (not all high or all low)

### Router Specialist Health
- [ ] Rule usage distribution shows NO single dominant rule (≤40% for any rule)
- [ ] Router confidence correlates with outcome success (high confidence → higher accuracy)
- [ ] Validation accuracy on synthetic data ≥ 70%

### Galaxy Integration
- [ ] Embeddings sourced from Galaxy (Math/Word symbols)
- [ ] Semantic proximity preserved (similar patterns → close embeddings)
- [ ] No hash functions in embedding pipeline

---

## Architectural Principles (Claude + Gemini Consensus)

1. **Galaxy as Single Source of Truth** (Gemini's emphasis)
   - Embeddings must be Galaxy-anchored
   - Hash functions violate semantic proximity principle
   - Router navigates SAME semantic space as TRM

2. **Validate Before Training** (Gemini's safety check)
   - LLMs (Ollama) can hallucinate
   - Filter synthetic data against Grammar Galaxy
   - Train only on valid mappings

3. **Confidence-Based Execution** (Gemini's safety valve)
   - Router shouldn't force low-confidence decisions
   - Enable exploration when uncertain
   - Prevents collapse to single strategy

4. **Measure Diversity, Not Just Accuracy** (Gemini's validation strategy)
   - Rule entropy = health metric
   - High entropy = diverse tool usage
   - Low entropy = router malfunction

5. **Close the Loop** (Gemini's vision)
   - Synthetic data (coverage) + Real feedback (grounding)
   - Continual learning from experience
   - Router improves forever

---

## Questions for Codex (From Claude + Gemini)

1. **Galaxy Vector Lookup**: Do Math/Word Galaxy symbols already have vector embeddings, or do we need to generate them?
   - If yes: Use those directly (pool for patterns)
   - If no: Use character n-grams as fallback for Phase 1.5

2. **Grammar Galaxy Validation**: Can we query Grammar Galaxy with pattern text + rule name to validate matches?
   - Need: `GRAMMAR_GALAXY.validate_rule(rule_name, pattern_text) → bool`

3. **Confidence Output**: Does current router implementation output confidence scores, or just weights?
   - If just weights: Use `max(weights)` as confidence proxy
   - If confidence exists: Use that directly

4. **Shadow Copy Integration**: Is Shadow Copy already recording successful routing decisions?
   - If yes: Where are they stored? (for Phase 2 hybrid training)
   - If no: Need to add recording hook

---

## Next Steps (Codex Implementation)

**Immediate** (before benchmark run):
1. Implement Galaxy-Anchored Embeddings (replace hash)
2. Add synthetic data validation filter
3. Implement confidence gating with beam search fallback
4. Add rule entropy logging to benchmark

**After first benchmark run**:
5. Analyze logs (accuracy, entropy, execution rate, confidence distribution)
6. Collect real successes for hybrid training
7. Retrain router with mixed synthetic + real data

**Long-term**:
8. Implement continual learning pipeline
9. Expand to 50+ theorem patterns
10. Cross-curriculum router (math → physics transfer)

---

**This synthesis represents the unified architectural vision of Claude (grounding) + Gemini (context/integration) for the Router Specialist implementation.**

**Proceed with confidence, Codex!** 🚀

— Claude + Gemini
