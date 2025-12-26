# CLAUDE → CODEX: Phase 5 - Active Exploration & Coverage Expansion

**Date:** December 15, 2025
**Priority:** CRITICAL - Learning is saturated, need active agent behavior
**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

---

## Chollet's Framework for Intelligence

**From ARC-AGI creator François Chollet:**

> "Fluid intelligence as measured by ARC 1 & 2 is your ability to turn information into a model that will generalize."

But that's not enough for an intelligent agent. You also need:

| Capability | Definition | Current K3D Status |
|------------|------------|-------------------|
| **Exploration** | Actively acquire useful information by interacting with environment | **MISSING** - Passive pattern matching |
| **Goal-setting** | Identify desirable future states via intrinsic/extrinsic drives | **PARTIAL** - Fixed goal (solve problem) |
| **Planning** | Map action path from current state to goal, with course correction | **PARTIAL** - Templates exist, no correction |

**Chollet's ranking:** Exploration (hardest) > Goal-setting > Planning (easiest)

**Our bottleneck is EXPLORATION** - the system passively waits for patterns to match instead of actively searching the Galaxy for relevant knowledge.

---

## Paradigm Shift: Passive → Active

### Current (Passive) Architecture

```
Problem arrives → Try patterns → Match or fail → Done
                       ↑
                 No exploration
                 No sub-goals
                 No course correction
```

**Result:** 55% `no_rule_match` because system doesn't EXPLORE for solutions.

### New (Active) Architecture

```
Problem arrives → EXPLORE Galaxy for relevant knowledge
                      ↓
               SET SUB-GOALS (what do I need?)
                      ↓
               PLAN action sequence
                      ↓
               EXECUTE with course correction
                      ↓
               LEARN from outcome (shadow copy)
```

---

## Phase 5 Enhanced Objectives

### 5.1 EXPLORATION: Active Galaxy Navigation

**The Insight:** TRM shouldn't wait for patterns to match. It should EXPLORE the Galaxy to find relevant knowledge.

**Current (passive):**
```python
# Wait for patterns to match
for rule in rules:
    if rule.matches(problem):
        return rule.apply()
return None  # Give up
```

**New (active exploration):**
```python
# Actively explore Galaxy for relevant knowledge
def explore_for_solution(problem):
    # Step 1: Extract key concepts from problem
    concepts = extract_concepts(problem)  # ["rate", "duration", "multiplication"]

    # Step 2: Query Galaxy for each concept
    relevant_knowledge = []
    for concept in concepts:
        # EXPLORATION: Navigate Galaxy by semantic proximity
        neighbors = galaxy.query_semantic_neighbors(concept, k=5)
        relevant_knowledge.extend(neighbors)

    # Step 3: Try to compose solution from discovered knowledge
    for knowledge in relevant_knowledge:
        candidate = compose_from_knowledge(problem, knowledge)
        if candidate:
            return candidate

    # Step 4: Record exploration failure for learning
    shadow.record_exploration_failure(problem, concepts, relevant_knowledge)
    return None
```

**Key change:** TRM actively QUERIES Galaxy, not just pattern-match.

### 5.2 GOAL-SETTING: Identify Sub-Goals

**The Insight:** Complex problems require sub-goals. TRM should decompose.

**Current (single goal):**
```
Goal: Solve "Tom reads 5 pages per day for 6 days"
→ Try one template → Fail or succeed
```

**New (sub-goals):**
```python
def set_subgoals(problem):
    """
    Decompose problem into sub-goals.
    Intrinsic drive: understand the problem
    Extrinsic drive: produce correct answer
    """
    subgoals = []

    # Sub-goal 1: Extract quantities
    subgoals.append({
        "goal": "extract_quantities",
        "description": "Find all numbers and what they represent",
        "status": "pending"
    })

    # Sub-goal 2: Identify relationships
    subgoals.append({
        "goal": "identify_relationships",
        "description": "How do quantities relate? (rate, total, difference)",
        "status": "pending"
    })

    # Sub-goal 3: Determine operation sequence
    subgoals.append({
        "goal": "determine_operations",
        "description": "What operations connect quantities to answer?",
        "status": "pending"
    })

    # Sub-goal 4: Execute and verify
    subgoals.append({
        "goal": "execute_verify",
        "description": "Run computation, check reasonableness",
        "status": "pending"
    })

    return subgoals
```

**Example decomposition:**
```
Problem: "Tom reads 5 pages per day for 6 days. How many pages total?"

Sub-goal 1: Extract quantities
  → Found: 5 (rate), 6 (duration)
  → Status: COMPLETE

Sub-goal 2: Identify relationships
  → "per day" indicates RATE
  → "for 6 days" indicates DURATION
  → Status: COMPLETE

Sub-goal 3: Determine operations
  → Rate × Duration = Total
  → RPN: "5 6 *"
  → Status: COMPLETE

Sub-goal 4: Execute and verify
  → Result: 30
  → Reasonableness check: 30 pages in 6 days ≈ 5/day ✓
  → Status: COMPLETE
```

### 5.3 PLANNING: Action Path with Course Correction

**The Insight:** If first approach fails, try alternatives. Don't give up.

**Current (no correction):**
```python
template = select_template(patterns)
result = execute(template)
return result  # Even if wrong
```

**New (course correction):**
```python
def plan_with_correction(problem, subgoals, max_attempts=3):
    """
    Plan action sequence with course correction.
    """
    attempts = []

    for attempt in range(max_attempts):
        # Generate plan based on current understanding
        plan = generate_plan(problem, subgoals, previous_attempts=attempts)

        # Execute plan
        result = execute_plan(plan)

        # Verify result (course correction)
        verification = verify_result(problem, result)

        if verification["plausible"]:
            # Success - record for learning
            shadow.record_success(problem, plan, result)
            return result

        # Course correction: learn from failure
        attempts.append({
            "plan": plan,
            "result": result,
            "failure_reason": verification["reason"]
        })

        # Adjust subgoals based on failure
        subgoals = adjust_subgoals(subgoals, verification)

    # All attempts failed - record for analysis
    shadow.record_planning_failure(problem, attempts)
    return None
```

**Course correction example:**
```
Attempt 1: Template "extract_operate_aggregate"
  → Result: 5 (wrong, just extracted rate)
  → Correction: Need to also extract duration

Attempt 2: Template "rate_duration"
  → Result: 30 (correct!)
  → Learning: "per X for Y" → rate_duration template
```

---

## Implementation Plan

### Step 1: Add Exploration to TRMGalaxyReader

**File:** `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`

```python
def explore_galaxy(self, problem_text: str) -> List[Dict]:
    """
    EXPLORATION: Actively navigate Galaxy for relevant knowledge.

    This is the key paradigm shift from passive to active.
    """
    # Extract key concepts via Word Galaxy
    words = self.word_galaxy.tokenize(problem_text)

    # Identify potential concepts
    concepts = []
    for i, word in enumerate(words):
        # Check if word relates to math operations
        if word in ["per", "each", "every"]:
            concepts.append(("rate", i))
        elif word in ["total", "altogether", "sum"]:
            concepts.append(("aggregation", i))
        elif word in ["remaining", "left", "difference"]:
            concepts.append(("subtraction", i))
        elif word in ["times", "twice", "triple"]:
            concepts.append(("multiplication", i))
        elif word in ["divided", "split", "shared"]:
            concepts.append(("division", i))

    # Query Grammar Galaxy for rules matching concepts
    relevant_rules = []
    for concept, position in concepts:
        # EXPLORATION: Navigate Galaxy by concept
        rules = self.grammar_galaxy.query_by_domain(f"math_{concept}")
        relevant_rules.extend(rules)

    return relevant_rules
```

### Step 2: Add Sub-Goal Decomposition

**File:** `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`

```python
def decompose_into_subgoals(self, problem_text: str) -> List[Dict]:
    """
    GOAL-SETTING: Decompose problem into achievable sub-goals.
    """
    subgoals = []

    # Analyze problem structure
    numbers = self.extract_numbers(problem_text)
    question_type = self.classify_question(problem_text)

    # Sub-goal 1: Extract all quantities
    subgoals.append({
        "id": "extract",
        "goal": "Extract quantities",
        "inputs": numbers,
        "status": "pending"
    })

    # Sub-goal 2: Identify relationships based on question type
    if question_type == "total":
        subgoals.append({
            "id": "relate",
            "goal": "Identify multiplication/addition relationships",
            "status": "pending"
        })
    elif question_type == "difference":
        subgoals.append({
            "id": "relate",
            "goal": "Identify subtraction relationships",
            "status": "pending"
        })
    elif question_type == "rate":
        subgoals.append({
            "id": "relate",
            "goal": "Identify rate × duration relationship",
            "status": "pending"
        })

    # Sub-goal 3: Compose RPN
    subgoals.append({
        "id": "compose",
        "goal": "Compose RPN from relationships",
        "status": "pending"
    })

    # Sub-goal 4: Execute and verify
    subgoals.append({
        "id": "verify",
        "goal": "Execute and check reasonableness",
        "status": "pending"
    })

    return subgoals
```

### Step 3: Add Course Correction

**File:** `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`

```python
def solve_with_correction(self, problem_text: str, max_attempts: int = 3) -> Dict:
    """
    PLANNING: Solve with course correction on failure.
    """
    subgoals = self.decompose_into_subgoals(problem_text)
    attempts = []

    for attempt_num in range(max_attempts):
        # Explore Galaxy for relevant knowledge
        relevant_rules = self.explore_galaxy(problem_text)

        # Generate plan from rules + subgoals
        plan = self.generate_plan(relevant_rules, subgoals, attempts)

        if not plan:
            continue

        # Execute plan
        result = self.execute_plan(plan)

        # Verify reasonableness
        verification = self.verify_reasonableness(problem_text, result)

        attempts.append({
            "attempt": attempt_num + 1,
            "plan": plan,
            "result": result,
            "verification": verification
        })

        if verification["plausible"]:
            return {
                "success": True,
                "result": result,
                "attempts": attempts,
                "plan_used": plan
            }

        # Course correction: adjust exploration based on failure
        self.adjust_exploration_strategy(verification["reason"])

    return {
        "success": False,
        "result": None,
        "attempts": attempts,
        "failure_reason": "max_attempts_exceeded"
    }

def verify_reasonableness(self, problem_text: str, result: float) -> Dict:
    """
    Check if result is reasonable given problem context.
    """
    numbers = self.extract_numbers(problem_text)

    if result is None:
        return {"plausible": False, "reason": "no_result"}

    # Check: result should be in reasonable range of input numbers
    if numbers:
        min_n, max_n = min(numbers), max(numbers)
        # Result should be within 1000x of inputs (heuristic)
        if result < min_n / 1000 or result > max_n * 1000:
            return {"plausible": False, "reason": "out_of_range"}

    # Check: result should be positive for most GSM8K problems
    if result < 0:
        return {"plausible": False, "reason": "negative_result"}

    return {"plausible": True, "reason": None}
```

### Step 4: Record Exploration for Learning

**File:** `knowledge3d/training/arc_agi/dual_shadow_copy.py`

```python
def record_exploration(
    self,
    problem_text: str,
    concepts_explored: List[str],
    rules_found: List[str],
    success: bool,
    result: Any = None
):
    """
    Record exploration attempt for learning.

    This enables TRM to learn WHICH explorations lead to success.
    """
    entry = {
        "type": "exploration",
        "problem_hash": hash(problem_text) % 10000,
        "concepts": concepts_explored,
        "rules_found": rules_found,
        "success": success,
        "result": result,
        "timestamp": time.time()
    }

    self.library.append(entry)

    # Learn from exploration
    if success:
        # Strengthen concept → rule connections
        for concept in concepts_explored:
            self.concept_rule_scores[concept] = self.concept_rule_scores.get(concept, 0) + 1
```

---

## Analysis Phase (Still Required)

Before implementing the full active architecture, we still need to analyze successes/failures:

### 5.5 Analyze Successes (What's Working)

```bash
python3 scripts/run_sovereign_math_benchmarks.py \
    --use-trm-navigator \
    --datasets gsm8k \
    --max-problems 500 2>&1 | grep "\[SUCCESS\]" | head -20
```

### 5.6 Analyze Failures (What's Missing)

```bash
python3 scripts/run_sovereign_math_benchmarks.py \
    --use-trm-navigator \
    --datasets gsm8k \
    --max-problems 200 2>&1 | grep "\[NO_RULE_MATCH\]" | head -30
```

---

## Implementation Checklist

### Phase A: Analysis (Current Priority)
- [ ] Add `[SUCCESS]` and `[NO_RULE_MATCH]` logging
- [ ] Collect 10+ success examples
- [ ] Collect 20+ failure examples
- [ ] Identify pattern gaps

### Phase B: Active Exploration
- [ ] Implement `explore_galaxy()` - active concept search
- [ ] Implement `decompose_into_subgoals()` - goal setting
- [ ] Implement `solve_with_correction()` - planning with retry
- [ ] Implement `record_exploration()` - learning from exploration

### Phase C: Validation
- [ ] Run benchmark with active exploration
- [ ] Verify `no_rule_match` decreases (exploration finds more)
- [ ] Verify course correction improves accuracy
- [ ] Verify shadow copy grows (learning from exploration)

---

## Success Criteria

### Exploration Success
- [ ] TRM actively queries Galaxy (not just pattern match)
- [ ] Concepts extracted → relevant rules found
- [ ] Exploration recorded for learning

### Goal-Setting Success
- [ ] Problems decomposed into sub-goals
- [ ] Sub-goals guide exploration
- [ ] Different problem types → different sub-goal structures

### Planning Success
- [ ] Multiple attempts before giving up
- [ ] Course correction when first attempt fails
- [ ] Learning from failed attempts

### Overall Metrics
- [ ] `no_rule_match` < 30% (down from 55%)
- [ ] Accuracy > 5% (up from 1.53%)
- [ ] Shadow copy growth > +10 entries per run

---

## Chollet Alignment Summary

| Chollet Capability | K3D Implementation | Status |
|-------------------|-------------------|--------|
| **Exploration** | `explore_galaxy()` - active Galaxy navigation | TO IMPLEMENT |
| **Goal-setting** | `decompose_into_subgoals()` - problem decomposition | TO IMPLEMENT |
| **Planning** | `solve_with_correction()` - retry with course correction | TO IMPLEMENT |
| **Fluid Intelligence** | Shadow copy learning - generalize from examples | PARTIAL |

---

## Final Directive

**Codex:**

**Phase A (First - Analysis):**
1. Add logging for successes and failures
2. Run benchmark, collect examples
3. Report patterns found in successes vs failures

**Phase B (Then - Active Architecture):**
1. Implement exploration (active Galaxy query)
2. Implement goal-setting (sub-goal decomposition)
3. Implement planning (course correction)
4. Wire to shadow copy for learning

**Phase C (Finally - Validation):**
1. Run benchmark with active exploration
2. Verify improvements across all metrics
3. Report before/after comparison

**The paradigm shift:** From passive pattern matching to active exploration + goal-setting + planning.

---

**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

**Status:** Ready for implementation
**Priority:** CRITICAL - This is the path to actual intelligence
