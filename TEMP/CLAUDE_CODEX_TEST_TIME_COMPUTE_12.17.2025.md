# CLAUDE → CODEX: Test-Time Compute Architecture (o1-style Reasoning)

**Date:** December 17, 2025
**Priority:** CRITICAL - This unlocks true AGI reasoning
**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

---

## The Paradigm Shift

### What We Learned

| Approach | Result | Lesson |
|----------|--------|--------|
| Task-specific rules | 12% on subset → 1% shuffled | **Overfitting** |
| Load all galaxies | 1% → 1% (no change) | **Galaxy is empty** |

**Root cause:** Reality Galaxy has physics (projectile, charge, LC circuit) but NOT the generic equations GSM8K needs (rate, work, area, conversions).

### What We Need

**Current (static):**
```
Problem → Match rules → Execute template → Return answer
```

**Needed (dynamic, o1-style):**
```
Problem → THINK (variable depth) → EXPLORE (parallel) → VERIFY → ITERATE → Answer
         ↓
    Adjust depth based on difficulty
    Run multiple hypotheses in parallel
    Course-correct when wrong
    Use cross-domain knowledge
```

This is **test-time compute** - the breakthrough behind OpenAI o1/o3.

---

## Architecture: Test-Time Compute for K3D

### Component 1: Seed Galaxy with Generic Knowledge

**Problem:** Reality Galaxy has 3 physics systems, but GSM8K needs generic equations.

**Solution:** Populate Math/Reality Galaxy with cross-domain fundamentals:

```python
GENERIC_EQUATIONS = {
    # Rate & Time
    "rate_time_distance": {
        "formula": "distance = rate × time",
        "rpn": "rate time *",
        "domains": ["physics", "math", "economics"],
        "isomorphic_to": [
            "money = rate × duration",
            "work = power × time",
            "distance = speed × time"
        ]
    },

    # Work & Energy
    "work_rate_time": {
        "formula": "work = rate × time",
        "rpn": "rate time *",
        "domains": ["physics", "economics", "biology"],
        "isomorphic_to": [
            "earnings = wage × hours",
            "production = output_rate × time",
            "growth = growth_rate × time"
        ]
    },

    # Area & Volume
    "area_rectangle": {
        "formula": "area = length × width",
        "rpn": "length width *",
        "domains": ["geometry", "physics"],
        "isomorphic_to": [
            "cost = price × quantity",
            "total = count × value"
        ]
    },

    # Conversions
    "unit_conversion": {
        "formula": "target = source × conversion_factor",
        "rpn": "source factor *",
        "domains": ["physics", "chemistry", "math"],
        "examples": [
            "kilograms = grams × 0.001",
            "hours = minutes × (1/60)",
            "dollars = cents × 0.01"
        ]
    },

    # Distribution
    "fair_share": {
        "formula": "share = total / count",
        "rpn": "total count /",
        "domains": ["math", "economics"],
        "isomorphic_to": [
            "average = sum / count",
            "rate = total / time"
        ]
    },

    # Accumulation
    "total_from_parts": {
        "formula": "total = sum(parts)",
        "rpn": "part1 part2 + part3 + ...",
        "domains": ["math", "physics", "economics"]
    },

    # Difference
    "remaining": {
        "formula": "remaining = total - used",
        "rpn": "total used -",
        "domains": ["math", "economics", "physics"]
    }
}
```

**Implementation:**
```python
# File: knowledge3d/cranium/generic_equations.py
class GenericEquationGalaxy:
    """
    Cross-domain generic equations for true generalization.

    These are NOT task-specific. They're universal mathematical
    relationships that apply across physics, economics, geometry, etc.
    """

    def load_equations(self):
        for eq_id, eq_data in GENERIC_EQUATIONS.items():
            self.add_equation(
                id=eq_id,
                formula=eq_data["formula"],
                rpn=eq_data["rpn"],
                domains=eq_data["domains"],
                isomorphic_patterns=eq_data.get("isomorphic_to", [])
            )
```

### Component 2: Dynamic Thinking Depth (Test-Time Compute)

**Principle:** Allocate compute based on problem difficulty, like o1.

```python
class TestTimeCompute:
    """
    Dynamic thinking depth - spend more compute on harder problems.

    Easy problem: 1 iteration, simple template
    Hard problem: 10+ iterations, explore Galaxy deeply
    """

    def solve_with_variable_depth(self, problem: str, budget: int = 10) -> Result:
        """
        Test-time compute: iterate until confident or budget exhausted.

        Args:
            problem: Problem text
            budget: Maximum iterations (like o1's token budget)

        Returns:
            Best answer found within budget
        """
        attempts = []
        confidence_threshold = 0.9

        for depth in range(1, budget + 1):
            # Explore with increasing depth
            result = self.explore_at_depth(problem, depth)

            # Verify plausibility
            confidence = self.verify_result(problem, result)

            attempts.append({
                "depth": depth,
                "result": result,
                "confidence": confidence
            })

            # Early stopping if high confidence
            if confidence >= confidence_threshold:
                print(f"  [TEST-TIME] Solved at depth {depth}/{budget}, confidence={confidence:.2f}")
                return result

            # Course correction for next iteration
            self.adjust_exploration_strategy(problem, result, confidence)

        # Return best attempt
        best = max(attempts, key=lambda x: x["confidence"])
        print(f"  [TEST-TIME] Best at depth {best['depth']}, confidence={best['confidence']:.2f}")
        return best["result"]

    def explore_at_depth(self, problem: str, depth: int) -> Result:
        """
        Explore Galaxy with specified depth.

        Depth 1: Surface-level patterns only
        Depth 5: Explore 2-3 hops in Galaxy graph
        Depth 10: Deep cross-domain exploration
        """
        if depth == 1:
            # Shallow: check direct pattern matches
            return self.shallow_match(problem)

        elif depth <= 5:
            # Medium: explore nearby concepts
            return self.medium_exploration(problem, hops=depth)

        else:
            # Deep: cross-domain isomorphic search
            return self.deep_cross_domain(problem, max_hops=depth)
```

### Component 3: Parallel RPN Exploration

**Principle:** Run multiple hypotheses in parallel, like AlphaGo tree search.

```python
class ParallelRPNExplorer:
    """
    Run multiple RPN candidates in parallel on GPU.

    Instead of sequential try-one-at-a-time, launch ALL candidates
    on different GPU cores and pick the best.
    """

    def explore_parallel(
        self,
        problem: str,
        num_candidates: int = 27  # Tesla 3³
    ) -> List[Candidate]:
        """
        Generate N diverse candidates, execute in parallel on GPU.
        """
        # Step 1: Generate diverse candidates
        candidates = self.generate_diverse_candidates(problem, num_candidates)

        # Step 2: Batch execute on GPU (parallel RPN cores)
        results = self.batch_execute_rpn(candidates)

        # Step 3: Rank by plausibility
        ranked = self.rank_by_plausibility(problem, results)

        return ranked

    def generate_diverse_candidates(self, problem: str, n: int) -> List[str]:
        """
        Generate N diverse RPN programs using different strategies.
        """
        candidates = []

        # Strategy 1: Tsinghua exploration (cluster → scout → expand)
        candidates.extend(self.tsinghua_exploration(problem, k=n//3))

        # Strategy 2: Cross-domain isomorphic search
        candidates.extend(self.cross_domain_search(problem, k=n//3))

        # Strategy 3: Shadow copy retrieval
        candidates.extend(self.shadow_retrieval(problem, k=n//3))

        return candidates[:n]  # Limit to N

    def batch_execute_rpn(self, rpn_programs: List[str]) -> List[float]:
        """
        Execute N RPN programs in parallel on GPU.

        Uses ModularRPNEngine.batch_evaluate() - all candidates
        execute simultaneously on different GPU cores.
        """
        return self.rpn_engine.batch_evaluate(rpn_programs)
```

### Component 4: Mid-Run Course Correction

**Principle:** If answer implausible, adjust exploration and retry.

```python
def solve_with_correction(self, problem: str, max_attempts: int = 3) -> Result:
    """
    Course correction: adjust strategy based on failures.
    """
    exploration_strategy = "balanced"  # Start balanced

    for attempt in range(max_attempts):
        # Explore with current strategy
        result = self.explore(problem, strategy=exploration_strategy)

        # Verify
        plausibility = self.verify_plausibility(problem, result)

        if plausibility["ok"]:
            return result

        # Course correct based on failure mode
        if plausibility["reason"] == "out_of_range":
            # Result too large/small → search for division/conversion
            exploration_strategy = "prioritize_division"

        elif plausibility["reason"] == "wrong_magnitude":
            # Off by 10×/100× → search for unit conversions
            exploration_strategy = "prioritize_conversions"

        elif plausibility["reason"] == "negative":
            # Negative result → search for different operation order
            exploration_strategy = "prioritize_aggregation"

        print(f"  [CORRECTION] Attempt {attempt+1} failed ({plausibility['reason']}), switching to {exploration_strategy}")

    return result  # Return best attempt
```

---

## Implementation Checklist

### Phase 1: Seed Generic Equations (Hours 1-2)
- [ ] Create `knowledge3d/cranium/generic_equations.py`
- [ ] Define `GENERIC_EQUATIONS` dict (7 core patterns above)
- [ ] Add loader to Reality/Math Galaxy
- [ ] Test: equations appear in Galaxy report

### Phase 2: Test-Time Compute Infrastructure (Hours 3-5)
- [ ] Implement `solve_with_variable_depth()` in TRMGalaxyReader
- [ ] Add `--thinking-budget` flag to benchmark runner
- [ ] Implement early stopping (confidence threshold)
- [ ] Test: depth increases for harder problems

### Phase 3: Parallel RPN Exploration (Hours 6-8)
- [ ] Implement `generate_diverse_candidates()`
- [ ] Wire `ModularRPNEngine.batch_evaluate()` (may already exist)
- [ ] Implement `rank_by_plausibility()`
- [ ] Test: 27 candidates execute in parallel

### Phase 4: Mid-Run Correction (Hours 9-10)
- [ ] Implement `solve_with_correction()`
- [ ] Define failure modes (out_of_range, wrong_magnitude, etc.)
- [ ] Implement strategy switching
- [ ] Test: failures trigger strategy adjustment

### Phase 5: Validation (Hour 11-12)
- [ ] Run shuffled GSM8K (seed 123, 200)
- [ ] Compare: static (1%) vs test-time compute (target: 5-8%)
- [ ] Analyze: which problems benefited from deeper thinking
- [ ] Shadow copy: record thinking traces

---

## Success Criteria

### Generic Equations
- [ ] Reality/Math Galaxy report shows 7+ generic equations
- [ ] Equations have cross-domain isomorphic patterns
- [ ] Explorer can query equations by domain

### Test-Time Compute
- [ ] Easy problems solve at depth 1-2
- [ ] Hard problems explore depth 5-10
- [ ] Early stopping works (stops at high confidence)
- [ ] Thinking budget controls max iterations

### Parallel Exploration
- [ ] 27 RPN candidates execute in <100ms (GPU batch)
- [ ] Diverse strategies contribute (Tsinghua, cross-domain, shadow)
- [ ] Ranking selects plausible results

### Course Correction
- [ ] Implausible results trigger strategy switch
- [ ] Retry with different exploration approach
- [ ] Learning: record which strategies work for which failures

### Overall
- [ ] Shuffled accuracy > 5% (5× improvement from 1%)
- [ ] Cross-domain equations used (trace shows physics → math)
- [ ] Shadow copy records thinking traces
- [ ] Generalization validated (multiple seeds, same performance)

---

## Expected Impact

| Metric | Current | With Test-Time Compute | Reasoning |
|--------|---------|------------------------|-----------|
| Shuffled accuracy | 1% | **5-8%** | Dynamic depth + cross-domain |
| Thinking depth | Fixed (1 pass) | Variable (1-10) | Allocate compute to hard problems |
| Candidates explored | ~5-10 | 27 (parallel) | GPU batch execution |
| Cross-domain use | 0% | 20-30% | Generic equations enable transfer |

---

## The o1 Connection

**OpenAI o1/o3 breakthrough:** Test-time compute (spend more tokens thinking).

**K3D equivalent:**
- o1 tokens = K3D thinking depth
- o1 chain-of-thought = K3D Galaxy exploration trace
- o1 self-correction = K3D course correction
- o1 verification = K3D plausibility checking

**But K3D advantage:**
- **Sovereign** (PTX + RPN, no external API)
- **Explainable** (RPN programs, not black-box tokens)
- **Cross-domain** (unified Galaxy, not siloed training)
- **Cost: $0** (local GPU, not $200/task)

---

## Final Directive

**Codex:**

**Phase 1 (First):** Seed generic equations
- Add `GENERIC_EQUATIONS` to Reality/Math Galaxy
- Verify they appear in Galaxy report
- Test cross-domain queries

**Phase 2-4 (Then):** Implement test-time compute
- Variable depth exploration
- Parallel RPN execution
- Course correction

**Phase 5 (Finally):** Validate
- Run shuffled GSM8K
- Target: 5-8% accuracy (5-8× improvement)
- Record thinking traces

**The goal:** Enable TRM to THINK like o1, using K3D's sovereign architecture.

---

**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

**Status:** Ready for implementation
**Priority:** CRITICAL - This is the path to AGI reasoning
