# Claude's Architectural Guidance: Complexity Metric for Oracle Mutation

**From**: Claude (Architecture Partner)
**To**: User + Gemini + Codex
**Date**: January 15, 2026
**Subject**: ✅ **Complexity Metric Design - Per-Source Baseline Recommended**

---

## Executive Summary

**Issue**: GSM8K complexity scores remain negative despite improved verification (27/60 = 45%) because micro-templates produce short RPN (2 ops) while base GSM8K produces longer programs.

**Root Cause**: Comparing **conceptually simple bootstrap examples** (micro-templates) against **executionally complex full problems** (base GSM8K).

**Architectural Ruling**: ✅ **Option 2 (Per-Source Baseline)** is the correct solution. Option 1 (artificially inflating step counts) would be gaming the metric.

**Better Approach**: Track **mutation DELTA** (steps added by mutation) separately from absolute complexity.

---

## Why Current Metric Is Misleading

### The Conflation Problem

**Current Complexity Calculation**:
```python
complexity = mut_steps - base_steps
```

**What This Measures**: Absolute difference between mutated problem and base problem

**The Issue**: Compares apples to oranges

**Example**:
```python
# Micro-template (bootstrap example)
Problem: "John has 5 apples. Mary gives him 3 more. How many altogether?"
RPN: [5, 3, +]  # 2 ops
mut_steps = 2

# Base GSM8K (full problem)
Problem: "Natalia sold clips to 48 friends in April, half as many in May..."
RPN: [48, 0.5, *, +, ...]  # 5 ops
base_steps = 5

# Complexity
complexity = 2 - 5 = -3  # NEGATIVE!
```

**But**: The micro-template is CORRECTLY solved with 2 ops. It's a simple problem by design (bootstrap example for training).

**The micro-template ISN'T less complex than base GSM8K - it's a DIFFERENT problem type.**

---

### What We're Really Measuring

**Two Different Dimensions**:

| Dimension | Micro-Templates | Base GSM8K |
|-----------|----------------|------------|
| **Conceptual Complexity** | Simple (by design) | Moderate-Hard |
| **Execution Complexity** | 2 ops (correct) | 5+ ops (correct) |
| **Purpose** | Bootstrap examples | Benchmark targets |

**Comparison Mistake**: Measuring execution complexity of bootstrap examples against benchmark targets.

**Analogy**: Comparing "2 + 3 = ?" (elementary school) to "Solve quadratic equation" (high school) and saying the first is "less complex" - true, but they're for different learning stages!

---

## Why Option 1 (Inflate Steps) Is Wrong ❌

**Proposal**: Extend GSM8K solver to generate 3-4 ops for micro-templates (even when 2 ops suffice).

**Why This Is Bad Architecture**:

### 1. Gaming the Metric ❌
```python
# Honest (current)
Problem: "5 + 3 = ?"
RPN: [5, 3, +]  # 2 ops (correct)

# Inflated (proposed)
Problem: "5 + 3 = ?"
RPN: [5, DUP, 3, SWAP, +, NOP]  # 6 ops (padding!)
```

**Result**: Higher complexity score, but no actual increase in reasoning depth.

### 2. Wastes Execution Resources ❌
- Extra operations consume PTX cycles
- Longer RPN programs use more VRAM
- No benefit to reasoning quality

### 3. Misleads Training ❌
- Model learns: "Simple problems need complex execution"
- Inefficient patterns reinforced
- Contradicts K3D's principle: "Simplest solution that works"

### 4. Technical Debt ❌
- Solver becomes inconsistent (sometimes inflates, sometimes doesn't)
- Hard to maintain (which problems get padding?)
- Breaks compositional reasoning (can't reuse inflated patterns)

**Architectural Principle Violation**: **Don't fake complexity to satisfy a metric. Fix the metric instead.**

---

## Why Option 2 (Per-Source Baseline) Is Better ✅

**Proposal**: Compare GSM8K mutations to GSM8K micro-template baseline, not full GSM8K.

**Why This Is Good Architecture**:

### 1. Honest Measurement ✅
```python
# Compare within same problem class
micro_template_baseline = 2 ops  # Average for micro-templates
mutated_template = 3 ops         # After mutation

complexity = 3 - 2 = +1  # Positive! (mutation added 1 op)
```

**Result**: Measures whether mutation INCREASED complexity (the actual goal).

### 2. Per-Source Fairness ✅
```python
# Different baselines for different sources
complexity_by_source = {
    "gsm8k_micro": mut_steps - gsm8k_micro_baseline,
    "calc_micro": mut_steps - calc_micro_baseline,
    "gsm8k_full": mut_steps - gsm8k_full_baseline
}
```

**Result**: Micro-templates compared to micro-template baseline, full problems compared to full problem baseline.

### 3. Measures Mutation Effectiveness ✅
```python
# Did mutation succeed in adding complexity?
if complexity > 0:
    print("Mutation added reasoning steps ✅")
else:
    print("Mutation simplified or failed ❌")
```

**Result**: Directly measures mutation quality.

---

## Better Approach: Mutation DELTA Metric

**Instead of absolute complexity, track mutation impact**:

### Mutation DELTA Calculation

```python
def calculate_mutation_delta(original_problem, mutated_problem):
    """Measure complexity added by mutation."""

    # Solve original
    original_steps = solver.solve(original_problem).step_count

    # Solve mutated
    mutated_steps = solver.solve(mutated_problem).step_count

    # Delta = steps added by mutation
    mutation_delta = mutated_steps - original_steps

    return {
        "original_steps": original_steps,
        "mutated_steps": mutated_steps,
        "mutation_delta": mutation_delta,  # Key metric!
        "percent_increase": (mutation_delta / original_steps) * 100
    }
```

**Example**:
```python
# Original micro-template
Problem: "John has 5 apples. Mary gives 3. How many altogether?"
RPN: [5, 3, +]  # 2 ops

# Mutated (added "remaining" clause)
Problem: "John has 5 apples. Mary gives 3. Then John eats 1. How many remaining altogether?"
RPN: [5, 3, +, 1, -]  # 4 ops

# Mutation Delta
mutation_delta = 4 - 2 = +2 ops
percent_increase = (2 / 2) * 100 = +100%

print("Mutation added 2 reasoning steps (100% increase) ✅")
```

**Benefits**:
- ✅ Measures mutation effectiveness directly
- ✅ Doesn't compare across problem types
- ✅ Positive delta = successful mutation
- ✅ Scales consistently across all sources

---

## Recommended Implementation

### Step 1: Track Per-Source Baselines

```python
# Compute baselines per source
def compute_source_baselines(verification_results):
    """Calculate average step count per source."""

    baselines = {}

    for source in ["gsm8k_micro", "gsm8k_full", "calc_micro"]:
        source_problems = [r for r in verification_results if r["source"] == source]
        avg_steps = sum(p["base_steps"] for p in source_problems) / len(source_problems)
        baselines[source] = avg_steps

    return baselines

# Example output
baselines = {
    "gsm8k_micro": 2.1,   # Micro-templates are simple
    "gsm8k_full": 5.8,    # Full problems are complex
    "calc_micro": 3.2     # Calculus templates moderate
}
```

---

### Step 2: Calculate Relative Complexity

```python
def calculate_relative_complexity(problem_result, baselines):
    """Calculate complexity relative to source baseline."""

    source = problem_result["source"]
    baseline = baselines[source]

    mut_steps = problem_result["mut_steps"]

    # Relative to source baseline
    relative_complexity = mut_steps - baseline

    return {
        "source": source,
        "baseline": baseline,
        "mut_steps": mut_steps,
        "relative_complexity": relative_complexity,  # Can be negative!
        "exceeds_baseline": relative_complexity > 0
    }
```

**Example**:
```python
# GSM8K micro-template
result = {
    "source": "gsm8k_micro",
    "mut_steps": 3
}

relative = calculate_relative_complexity(result, baselines)
# relative_complexity = 3 - 2.1 = +0.9 ✅ (exceeds micro baseline)

# GSM8K full problem
result = {
    "source": "gsm8k_full",
    "mut_steps": 4
}

relative = calculate_relative_complexity(result, baselines)
# relative_complexity = 4 - 5.8 = -1.8 ❌ (simpler than full baseline)
```

---

### Step 3: Track Mutation DELTA (Preferred)

```python
def calculate_mutation_metrics(mutation_result):
    """Calculate mutation effectiveness metrics."""

    original_steps = mutation_result["original_steps"]  # Before mutation
    mutated_steps = mutation_result["mutated_steps"]     # After mutation

    mutation_delta = mutated_steps - original_steps

    return {
        "original_steps": original_steps,
        "mutated_steps": mutated_steps,
        "mutation_delta": mutation_delta,
        "percent_increase": (mutation_delta / max(original_steps, 1)) * 100,
        "successful_mutation": mutation_delta > 0
    }
```

**Summary Statistics**:
```python
# Aggregate mutation effectiveness
mutation_results = [calculate_mutation_metrics(r) for r in all_results]

summary = {
    "total_mutations": len(mutation_results),
    "successful_mutations": sum(1 for r in mutation_results if r["successful_mutation"]),
    "avg_delta": np.mean([r["mutation_delta"] for r in mutation_results]),
    "avg_percent_increase": np.mean([r["percent_increase"] for r in mutation_results]),
    "median_delta": np.median([r["mutation_delta"] for r in mutation_results])
}

print(f"Mutation Effectiveness:")
print(f"  Success Rate: {summary['successful_mutations']} / {summary['total_mutations']}")
print(f"  Avg Steps Added: {summary['avg_delta']:.2f}")
print(f"  Avg Increase: {summary['avg_percent_increase']:.1f}%")
```

---

## Updated Reporting Format

### Current Format (Misleading)
```json
{
  "source": "gsm8k",
  "avg_complexity": -2.3,  // Negative! Looks bad
  "verified": 27,
  "failed": 33
}
```

**Problem**: Negative complexity misleads about mutation quality.

---

### Recommended Format (Clear)
```json
{
  "source": "gsm8k_micro",
  "baseline_steps": 2.1,
  "avg_mutated_steps": 3.2,
  "avg_mutation_delta": +1.1,  // Positive! Mutation working
  "avg_percent_increase": "+52%",
  "successful_mutations": "27/60 (45%)",
  "failed_mutations": "33/60 (55%)"
}
```

**Benefits**:
- ✅ Clear that mutation adds complexity (+1.1 steps)
- ✅ Shows percent increase (+52%)
- ✅ Separates verification success from complexity metric
- ✅ Baseline visible for context

---

## Architectural Principles Validation

### 1. Honest Metrics ✅
- Don't inflate step counts to game the metric
- Measure what mutation actually achieves (DELTA)
- Per-source baselines prevent apples-to-oranges

### 2. Separation of Concerns ✅
- **Complexity**: How many reasoning steps (metric)
- **Verification**: Does the solver get correct answer (quality)
- **Mutation**: Did mutation add complexity (effectiveness)

### 3. Composability ✅
- Mutation DELTA works for all problem sources
- Relative complexity adapts to source baseline
- Metrics scale consistently

### 4. Engineering Rigor ✅
- Fix the metric, don't fake the data
- Measure effectiveness directly (DELTA)
- Track per-source baselines (context-aware)

---

## Connection to Phase 4.1: DataGeneration Specialist

**Current**: Manual oracle mutation with complexity verification

**Phase 4.1 Vision**: **Learned mutation specialist**

**Architecture**:
```python
class MutationSpecialist(nn.Module):
    """Learns to generate complex training problems."""

    def forward(self, original_problem_embedding):
        # Predict mutation that adds complexity
        mutation_ops = self.mutation_head(original_problem_embedding)
        return mutation_ops

# Training Objective
for original_problem in dataset:
    # Predict mutation
    mutation = mutation_specialist(original_problem)

    # Apply mutation
    mutated_problem = apply_mutation(original_problem, mutation)

    # Measure effectiveness
    delta = calculate_mutation_delta(original_problem, mutated_problem)

    # Reward positive delta
    reward = delta["mutation_delta"] if delta["successful_mutation"] else -1

    # Train specialist
    loss = -reward  # Maximize delta
    loss.backward()
```

**Mutation DELTA is the training signal** for learned mutation!

**The oracle mutation system is bootstrapping Phase 4.1** - we're learning which mutations are effective so the specialist can mimic them.

---

## Directive for Implementation

### Immediate Tasks

**Task 1: Add Mutation DELTA Tracking**
```python
# File: scripts/oracle_mutate.py (or verification script)

def track_mutation_delta(original_problem, mutated_problem):
    """Track steps added by mutation."""

    # Solve original
    original_result = solver.solve(original_problem)

    # Solve mutated
    mutated_result = solver.solve(mutated_problem)

    return {
        "original_steps": len(original_result.rpn),
        "mutated_steps": len(mutated_result.rpn),
        "mutation_delta": len(mutated_result.rpn) - len(original_result.rpn),
        "verification_success": mutated_result.correct
    }
```

**Task 2: Update Reporting**
```python
# File: scripts/generate_oracle_report.py

def generate_mutation_report(results):
    """Generate report with per-source baselines and mutation deltas."""

    by_source = {}

    for source in ["gsm8k_micro", "gsm8k_full", "calc_micro"]:
        source_results = [r for r in results if r["source"] == source]

        by_source[source] = {
            "baseline_steps": np.mean([r["original_steps"] for r in source_results]),
            "avg_mutated_steps": np.mean([r["mutated_steps"] for r in source_results]),
            "avg_mutation_delta": np.mean([r["mutation_delta"] for r in source_results]),
            "successful_mutations": sum(1 for r in source_results if r["mutation_delta"] > 0),
            "total_mutations": len(source_results),
            "verification_rate": sum(1 for r in source_results if r["verification_success"]) / len(source_results)
        }

    return by_source
```

**Task 3: Validate Metrics**
```python
# Sanity check
report = generate_mutation_report(results)

for source, metrics in report.items():
    print(f"{source}:")
    print(f"  Baseline: {metrics['baseline_steps']:.2f} steps")
    print(f"  Mutated: {metrics['avg_mutated_steps']:.2f} steps")
    print(f"  Delta: {metrics['avg_mutation_delta']:+.2f} steps")
    print(f"  Success Rate: {metrics['successful_mutations']}/{metrics['total_mutations']}")
    print(f"  Verification: {metrics['verification_rate']:.1%}")
```

---

## Success Criteria

**Metrics Fixed When**:
- [ ] Mutation DELTA tracked for all problems
- [ ] Per-source baselines computed
- [ ] Reporting shows positive delta for successful mutations
- [ ] No artificial step inflation (solver stays honest)
- [ ] GSM8K micro-templates show positive delta (mutation working)

---

## Summary

**Issue**: Complexity metric compares bootstrap examples (micro-templates) to full problems (base GSM8K), producing misleading negative scores.

**Root Cause**: Conflating conceptual complexity (problem difficulty) with execution complexity (solver steps).

**Solution**: ✅ **Track Mutation DELTA** (steps added by mutation) instead of absolute complexity.

**Recommended Approach**:
1. Track original steps vs mutated steps
2. Calculate mutation_delta = mutated - original
3. Report per-source baselines for context
4. Positive delta = successful mutation

**NOT Recommended**: ❌ Artificially inflate step counts to game the metric.

**Phase 4.1 Connection**: Mutation DELTA becomes the training signal for learned mutation specialist.

**Architectural Principle**: Fix the metric, don't fake the data. Measure effectiveness directly.

---

**Document Date**: January 15, 2026
**Context**: Oracle Mutation System (Phase 4.1 Bootstrap)
**Status**: ✅ **GUIDANCE PROVIDED - IMPLEMENT MUTATION DELTA METRIC**

---

**Claude's Directive**: Your GSM8K verification improvement (45%) is excellent progress! The complexity metric just needs adjustment - track MUTATION DELTA (steps added by mutation) instead of comparing to unrelated base problems. This honest metric will become the training signal for Phase 4.1 learned mutation specialist. 🚀
