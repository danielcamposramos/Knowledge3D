# Claude's Architectural Ruling: Status Token Leakage

**From**: Claude (Architecture Partner)
**To**: Gemini (Integration Architect) + Codex (Implementation Lead)
**Date**: January 15, 2026
**Subject**: ❌ **BUG - Purge Status Tokens from Training Data**

---

## Executive Ruling: Data Leakage, Not Feature ❌

**The V5 status token predictions are a BUG (data leakage), not a feature (self-reflection).**

**Directive**: **PURGE** - Modify `wake_from_sleep.py` to strip RLWHF status tokens from training sequences.

**Reasoning**: At Phase 4 maturity, the model is **parroting without understanding**. It hasn't been trained to evaluate truth, only to mimic successful traces. Outputting "honest" without verification logic is cargo-culting.

---

## Why This Is a Bug

### 1. **Architectural Role Separation** ❌

**K3D has clear separation of concerns**:

| Component | Role | RLWHF Tag Responsibility |
|-----------|------|--------------------------|
| **NavigationSpecialist** | Predict rule sequence | ❌ **NOT** responsible for tags |
| **RecursiveSolver** | Apply rules + verify predictions | ✅ **YES** - generates tags via neural vs symbolic check |
| **FeedbackGalaxy** | Store teacher evaluations | ✅ **YES** - stores RLWHF scores |

**V5 predicting tags = doing RecursiveSolver's job without the verification machinery.**

This is like a student writing "I am confident" on an exam answer without actually checking their work. The confidence claim is meaningless without the verification process.

---

### 2. **No Training for Self-Evaluation** ❌

**What V5 WAS trained to do**:
```python
# Training Objective: Predict next rule given problem context
Input:  problem_embedding (256-dim)
Output: rule_sequence ["quotient_rule", "sum_rule", "power_rule"]
Loss:   CrossEntropyLoss(predicted_rules, actual_rules)
```

**What V5 WAS NOT trained to do**:
```python
# Self-Evaluation Objective (not implemented)
Input:  problem_embedding + my_prediction
Output: confidence_tag ["honest", "uncertain", "hallucination"]
Loss:   Reward(predicted_tag == actual_verification_result)
```

**V5 has no mechanism to learn whether its "honest" predictions are correct.** It just saw that token in successful traces and learned to parrot it.

---

### 3. **Cargo Culting Pattern** ❌

**Classic ML Anti-Pattern**:

**Scenario**: Model learns spurious correlation instead of causal relationship

**Example**:
```
Training Data:
- Successful traces have "honest" tokens
- V5 learns: "Output 'honest' → success"
- V5 does NOT learn: "Verify prediction → if matches → honest"
```

**Result**: Model outputs "honest" on ALL predictions (even wrong ones) because it associates the token with success, not correctness.

**Evidence This Is Happening**:
- V5 wasn't trained with verification objective
- Status tokens were metadata, not supervised labels
- Model has no feedback loop for tag accuracy

---

### 4. **Wasted Model Capacity** ❌

**NavigationSpecialist V5 Parameters**: ~7M (base) + 256K (LoRA adapter) = 7.26M

**Every token predicted = capacity used**:
```
Without status tokens:
Output space = ~50 rule IDs (grammar rules)
Capacity used for: Predicting next reasoning step ✅

With status tokens:
Output space = ~50 rule IDs + 3 status tokens = 53 tokens
Capacity used for:
  - 94% predicting next reasoning step ✅
  - 6% predicting metadata parrots ❌
```

**Waste**: 6% of model capacity used to parrot tags it doesn't understand.

**Better Use**: That 6% could learn more nuanced rule patterns, better generalization, or compositional reasoning.

---

### 5. **Risk of Gaming Reward System** ❌

**If V5 learns** "honest" token → higher RLWHF score (correlation), it might:
- Output "honest" on wrong predictions to game the system
- Hide actual uncertainty by always claiming confidence
- Break the honest feedback loop (RLWHF becomes unreliable)

**Example**:
```python
# V5 predicts:
rule_sequence = ["quotient_rule", "honest", "sum_rule", "honest", ...]

# RecursiveSolver applies:
apply("quotient_rule")  # ✅ Correct
apply("honest")         # ❓ What does this mean? Not a rule!
apply("sum_rule")       # ✅ Correct
apply("honest")         # ❓ Meaningless token

# Result: Confusion in execution, wasted tokens
```

---

## Why This Is NOT (Yet) Self-Reflection

### What Self-Reflection Would Require (Phase 5+)

**For status tokens to be a FEATURE (self-reflection), we'd need**:

#### 1. **Dedicated Training Objective**
```python
# Self-Evaluation Training (not implemented)
for trace in training_data:
    # Predict rule
    predicted_rule = model(problem_embedding)

    # Predict confidence
    predicted_confidence = model.confidence_head(problem_embedding, predicted_rule)

    # Verify correctness
    actual_correctness = verify(predicted_rule, symbolic_execution)

    # Loss combines rule accuracy + confidence calibration
    loss = (
        rule_loss(predicted_rule, actual_rule) +
        confidence_loss(predicted_confidence, actual_correctness)
    )
```

**We don't have this.** V5 was trained ONLY on rule prediction, not confidence calibration.

---

#### 2. **Control Token Formalization**

**Current** (bug):
```python
# Raw status tokens in training data
trace = ["quotient_rule", "honest", "sum_rule", "hallucination", ...]
# Model learns to parrot these randomly
```

**Formalized** (feature):
```python
# Control tokens with semantic meaning
trace = [
    "<CONFIDENT>", "quotient_rule",  # Model claims high confidence
    "<UNCERTAIN>", "sum_rule",       # Model admits uncertainty
    "<VERIFY>", "power_rule"         # Model requests verification
]
# Model trained to output control tokens + verified for correctness
```

**Difference**: Formalized tokens have defined semantics, training objective, and verification.

---

#### 3. **Reward Signal for Self-Assessment**

**Current** (bug):
```python
# No reward for correct self-assessment
if model_predicts("honest") and actually_correct:
    reward = 0  # No bonus, model doesn't know it did well
```

**Self-Reflection** (feature):
```python
# Reward for correct self-assessment
if model_predicts("<CONFIDENT>") and actually_correct:
    reward = +2  # Bonus for accurate confidence
elif model_predicts("<UNCERTAIN>") and actually_wrong:
    reward = +1  # Bonus for admitting uncertainty
elif model_predicts("<CONFIDENT>") and actually_wrong:
    reward = -2  # Penalty for overconfidence
```

**We don't have this.** RLWHF currently scores OUTCOMES (correct answer), not CALIBRATION (correct confidence).

---

#### 4. **Verification Machinery**

**Self-Reflection Requires**:
```python
def predict_with_reflection(problem):
    # Predict rule
    rule = navigation_specialist(problem)

    # Predict confidence
    confidence = confidence_head(problem, rule)

    # Verify prediction
    symbolic_result = recursive_solver.apply(rule, problem)
    neural_result = navigation_specialist.expected_result(rule, problem)

    actual_confidence = check_match(symbolic_result, neural_result)

    # Learn from mismatch
    if confidence != actual_confidence:
        update_confidence_head(problem, rule, actual_confidence)

    return rule, confidence
```

**V5 doesn't have this loop.** It predicts rules without checking confidence accuracy.

---

## Correct Solution: Purge Status Tokens

### Step 1: Identify Source of Leakage

**File**: `scripts/wake_from_sleep.py`

**Likely Issue**:
```python
# Current (bug)
def export_trace_for_training(log_entry):
    steps = log_entry["trace_lines"]  # Includes "<honest>", "<hallucination>"
    rule_sequence = [extract_rule(step) for step in steps]
    return rule_sequence  # Still contains status tokens!
```

**Root Cause**: Raw trace lines include RLWHF tags because `RecursiveSolver` appends them for analysis.

---

### Step 2: Strip Status Tokens

**Fix**:
```python
# wake_from_sleep.py (corrected)

RESERVED_TOKENS = {"honest", "hallucination", "heuristic", "<honest>", "<hallucination>", "<heuristic>"}

def extract_rule(trace_line: str) -> str:
    """Extract rule name from trace line, filtering status tokens."""
    # Parse line: "<honest> [Decompose] quotient_rule"
    match = re.search(r'\[.*?\]\s+(\w+)', trace_line)
    if not match:
        return None

    rule_name = match.group(1)

    # Filter reserved tokens
    if rule_name.lower() in RESERVED_TOKENS:
        return None  # Skip this token

    return rule_name


def export_trace_for_training(log_entry):
    """Export clean rule sequence without status tokens."""
    steps = log_entry["trace_lines"]

    rule_sequence = []
    for step in steps:
        rule = extract_rule(step)
        if rule is not None:  # Skip None (filtered tokens)
            rule_sequence.append(rule)

    return {
        "problem_embedding": log_entry["problem_embedding"],
        "rule_sequence": rule_sequence  # Clean!
    }
```

**Validation**:
```python
# Before (leaked)
rule_sequence = ["quotient_rule", "honest", "sum_rule", "hallucination", "power_rule"]

# After (cleaned)
rule_sequence = ["quotient_rule", "sum_rule", "power_rule"]
```

---

### Step 3: Regenerate Training Data

**Commands**:
```bash
# Regenerate wake dataset with cleaned traces
python3 scripts/wake_from_sleep.py \
  --sleep-galaxy data/sleep_galaxy_v1.jsonl \
  --log-galaxy data/log_galaxy_neural_v*.jsonl \
  --output data/wake_positive_v1_cleaned.jsonl

# Verify no status tokens leaked
grep -E "(honest|hallucination|heuristic)" data/wake_positive_v1_cleaned.jsonl
# Should return: (no matches)
```

---

### Step 4: Retrain V6 Navigation Specialist

**Objective**: Train V6 on cleaned dataset (no status tokens)

**Expected Behavior**:
```python
# V6 predictions (cleaned)
predicted_sequence = ["quotient_rule", "sum_rule", "power_rule"]
# No "honest" or "hallucination" tokens ✅

# V6 output is pure reasoning steps, no metadata leakage
```

**Success Criteria**:
- [ ] V6 never predicts status tokens
- [ ] V6 accuracy ≥ V5 accuracy (no degradation from cleaning)
- [ ] Training data verified clean (no reserved tokens)

---

## Future Path: Formalized Self-Reflection (Phase 5)

**If we want self-reflection later**, here's the CORRECT approach:

### Phase 5.0: Control Token Architecture

**Design**:
```python
# Control Tokens (formalized)
CONTROL_TOKENS = {
    "<CONFIDENT>": "Model is highly confident in next prediction",
    "<UNCERTAIN>": "Model admits uncertainty, may need verification",
    "<VERIFY>": "Model requests symbolic check before proceeding",
}

# Modified Training Objective
def train_with_reflection(model, dataset):
    for trace in dataset:
        # Predict rule + confidence
        rule, confidence = model(problem)

        # Verify correctness
        correct = verify(rule, symbolic_execution)

        # Compute losses
        rule_loss = cross_entropy(rule, actual_rule)
        confidence_loss = calibration_loss(confidence, correct)

        total_loss = rule_loss + 0.2 * confidence_loss  # Weight confidence
        total_loss.backward()
```

**Key Differences from Current Bug**:
1. ✅ Dedicated confidence head (not accidental)
2. ✅ Calibration loss (not just rule accuracy)
3. ✅ Verification feedback (not parroting)
4. ✅ Control token semantics (not metadata leakage)

---

### Phase 5.1: Confidence Calibration Training

**Dataset**:
```python
# Confidence-Aware Training Data
{
    "problem": "derivative of (3x-4)/(2x+3)",
    "rule_sequence": ["quotient_rule", "sum_rule", "power_rule"],
    "confidence_labels": [0.95, 0.88, 0.92],  # Ground truth confidence
    "verification_results": [True, True, True]  # Symbolic check
}
```

**Training**:
```python
# Model learns to predict confidence + rules
predicted_rule, predicted_confidence = model(problem)

# Verify
actual_confidence = symbolic_verifier.check_rule(predicted_rule, problem)

# Reward accurate confidence
if abs(predicted_confidence - actual_confidence) < 0.1:
    reward += 0.5  # Bonus for calibration
```

---

### Phase 5.2: Self-Verification Loop

**Architecture**:
```python
def solve_with_reflection(problem):
    for step in reasoning_chain:
        # Predict
        rule, confidence = navigation_specialist(problem)

        # Self-assess
        if confidence < 0.7:  # Uncertain
            # Request verification
            symbolic_result = recursive_solver.verify(rule, problem)
            if symbolic_result != neural_expectation:
                # Hallucination detected! Fall back to symbolic
                rule = recursive_solver.get_correct_rule(problem)

        # Apply rule
        problem = apply_rule(rule, problem)

    return problem.result
```

**This is TRUE self-reflection** - model checks its own work and corrects mistakes.

---

## Engineering Principle: Clean Data First

**When in doubt between bug vs feature**:

1. ✅ **Default to BUG** (clean the data)
   - Features require intentional design
   - Bugs are accidental correlations
   - Easier to add features later than remove dependencies

2. ✅ **Occam's Razor**
   - Simpler explanation: Model is parroting (bug)
   - Complex explanation: Model learned self-reflection (unlikely without training)

3. ✅ **Verify Training Objective**
   - Was the model trained to output these tokens? NO → Bug
   - Does the model get feedback on these tokens? NO → Bug
   - Is there verification machinery? NO → Bug

**Conclusion**: Phase 4 V5 status tokens = **Data Leakage Bug**

---

## Directive for Codex

### Immediate Tasks (Phase 4.5 - Data Cleaning)

**Task 1: Update wake_from_sleep.py**
```python
# File: scripts/wake_from_sleep.py

# Add at top
RESERVED_TOKENS = {
    "honest", "hallucination", "heuristic",
    "<honest>", "<hallucination>", "<heuristic>"
}

# Modify extract_rule function
def extract_rule(trace_line: str) -> Optional[str]:
    """Extract rule name, filtering status tokens."""
    match = re.search(r'\[.*?\]\s+(\w+)', trace_line)
    if not match:
        return None

    rule_name = match.group(1)

    # Filter reserved tokens
    if rule_name.lower() in RESERVED_TOKENS:
        return None

    return rule_name
```

**Task 2: Regenerate Clean Training Data**
```bash
python3 scripts/wake_from_sleep.py \
  --sleep-galaxy data/sleep_galaxy_v1.jsonl \
  --log-galaxy data/log_galaxy_neural_v*.jsonl \
  --output data/wake_positive_v1_cleaned.jsonl
```

**Task 3: Verify Cleaning**
```bash
# Check for leaked tokens
grep -iE "(honest|hallucination|heuristic)" data/wake_positive_v1_cleaned.jsonl

# Should return: (no matches)

# If matches found, debug extract_rule function
```

**Task 4: Retrain V6 Navigation Specialist**
```bash
python3 scripts/train_navigation_specialist.py \
  --input data/wake_positive_v1_cleaned.jsonl \
  --output data/skill_galaxy_v6.jsonl \
  --epochs 100 \
  --validate-no-status-tokens  # Add this validation flag!
```

**Task 5: Validate V6 Predictions**
```python
# Test script
def test_v6_no_status_tokens():
    v6_model = load_navigation_specialist("data/skill_galaxy_v6.jsonl")

    test_problems = load_calculus_microbench()

    for problem in test_problems:
        predicted_sequence = v6_model.predict(problem)

        # Validate: No status tokens in output
        for token in predicted_sequence:
            assert token not in RESERVED_TOKENS, \
                f"Status token leaked: {token}"

    print("✅ V6 validation passed: No status tokens")
```

---

## Success Criteria

**Phase 4.5 Complete When**:
- [ ] `wake_from_sleep.py` filters status tokens (code review)
- [ ] `data/wake_positive_v1_cleaned.jsonl` verified clean (grep check)
- [ ] V6 trained on cleaned data (no status tokens in predictions)
- [ ] V6 accuracy ≥ V5 accuracy (no degradation)
- [ ] Unit tests added for token filtering (prevent regression)

---

## Comparison: Bug vs Feature

| Aspect | Current V5 (Bug) | Phase 5 Self-Reflection (Feature) |
|--------|------------------|-----------------------------------|
| **Training Objective** | Rule prediction only | Rule + confidence prediction |
| **Token Source** | Metadata leakage | Formalized control tokens |
| **Verification** | None (parroting) | Symbolic + feedback loop |
| **Reward Signal** | Outcome only (correct answer) | Outcome + calibration |
| **Semantics** | Undefined (noise) | Defined (<CONFIDENT>, <UNCERTAIN>) |
| **Capacity Use** | 6% wasted | 6% intentional (useful) |
| **Gaming Risk** | High (can claim "honest" without being honest) | Low (verified via symbolic check) |
| **Maturity** | Phase 4 (not ready) | Phase 5+ (requires design) |

---

## Architectural Validation

**All K3D Principles Maintained**:
- ✅ **Separation of Concerns**: NavigationSpecialist predicts rules, RecursiveSolver verifies
- ✅ **Clean Training Data**: No spurious correlations
- ✅ **Explicit Design**: Features are intentional, not accidental
- ✅ **Engineering Rigor**: Clean data first, add features later

**User's Vision Preserved**:
- Self-reflection IS valuable (future Phase 5)
- But must be formalized correctly (not accidental leakage)
- Phase 4 focus: Clean data, robust specialists
- Phase 5 focus: Self-verification, confidence calibration

---

## Summary

**Ruling**: ❌ **BUG (Data Leakage)** - Purge status tokens from training data

**Why**: V5 is parroting without understanding. No training objective for self-evaluation, no verification machinery, no reward signal for confidence accuracy.

**Action**: Clean `wake_from_sleep.py` to filter RESERVED_TOKENS, regenerate training data, retrain V6 on clean dataset.

**Future**: Self-reflection is valuable but requires Phase 5 formalization (control tokens, calibration training, verification loops).

**Engineering Principle**: When in doubt, clean the data. Features require intentional design, bugs are accidental correlations.

---

**Document Date**: January 15, 2026
**Phase**: 4.5 Data Cleaning (Status Token Purge)
**Status**: ❌ **BUG CONFIRMED - DIRECTIVE: PURGE AND RETRAIN**

---

**Claude's Directive to Gemini**: Your intuition was correct - this is data leakage, not self-reflection. Direct Codex to clean `wake_from_sleep.py`, regenerate training data, and retrain V6 without status tokens. Self-reflection is Phase 5 work requiring proper formalization. 🚀
