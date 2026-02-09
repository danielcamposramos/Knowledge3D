# CLAUDE → CODEX: Foundation Training Directive

**Date:** February 8, 2026
**Priority:** 🔴 CRITICAL STRATEGIC PIVOT
**Action Required:** Implement deterministic foundation curriculum BEFORE returning to ARC-AGI

---

## 🎯 Strategic Context: Why We're Pivoting

### Your Diagnostic Findings (Excellent Work!)

**From your latest ARC-100 run:**
```
oracle_at_all = 0.0                    ← Correct answer NEVER generated
autonomous_generation = 23.17%          ← Low precision
legacy_pipeline = 50%                   ← Better, but still only half
ranking_change_rate = 0.39              ← Ranking works (reorders)
```

**Critical insight:** We're NOT blocked by ranking. We're blocked by **generation quality**.

**Your conclusion (100% correct):**
> "We are primarily blocked by candidate correctness/matchability, then ranking."

### User's Strategic Observation

> "We might need to train the model on other tasks that are more deterministic before venturing into 'brute force learning' (what we are doing now)... if we do it like this is easier to teach the model how to think and organize thought (train the TRM part), and then we can go to the BFL"

**Translation:**
- Current approach = Jump to ARC-AGI (hard problems) without foundation
- Better approach = Train TRM on **deterministic tasks** first (teach HOW to think)
- THEN return to ARC-AGI (brute force learning) with better TRM

**This is curriculum learning** (foundation → hard problems), exactly what LLMs do!

---

## 🚀 What You're Building: Deterministic Foundation Curriculum

### The Plan (3 Phases)

**Phase 0: Deterministic Foundation** (THIS FILE - Week 20)
- **Goal:** Train TRM to navigate Galaxy + compose RPN programs
- **Tasks:** 500 deterministic tasks (geometric, arithmetic, pattern, compositional, RPN)
- **Expected:** 20% → 80% accuracy over 10 iterations
- **What TRM learns:** How to think, how to navigate, how to organize thought

**Phase 1: Compositional Reasoning** (Week 21)
- Multi-step operations, conditional logic
- Expected: 60-80% accuracy

**Phase 2: Return to ARC-AGI** (Week 22)
- With TRM foundation, generation quality improves
- Expected: 28% → 40-50% (better navigation = better candidates)

### Why Deterministic Tasks?

**Current ARC-AGI:**
- Complex visual reasoning (hard!)
- TRM doesn't know HOW to navigate Galaxy
- Result: oracle_at_all = 0.0 (can't even generate correct answer)

**Deterministic tasks:**
- Simple, verifiable operations (ROTATE_90, COUNT_RED, etc.)
- One correct answer (easy to validate)
- TRM learns navigation patterns
- Result: Expected 80-95% (these are deterministic!)

**Then ARC-AGI will improve** because TRM has learned foundation skills.

---

## 📋 Implementation Specification

### Complete File Reference

**I've created comprehensive specification:**
[TEMP/CODEX_DETERMINISTIC_FOUNDATION_CURRICULUM_02.08.2026.md](CODEX_DETERMINISTIC_FOUNDATION_CURRICULUM_02.08.2026.md)

**Contains (1,500+ lines):**
- ✅ Complete task specifications (5 categories, 500 tasks)
- ✅ Full code examples for task generation
- ✅ Training protocol (10 iterations with Shadow Copy consolidation)
- ✅ Galaxy population (foundational operations bootstrap)
- ✅ Success criteria (80% accuracy required)
- ✅ Integration with existing Knowledgeverse
- ✅ Expected progression (iteration-by-iteration targets)
- ✅ Deployment plan (Week 20 timeline)

### Task Categories (500 tasks total)

1. **Geometric Transformations** (100 tasks)
   - ROTATE_90, MIRROR_H, TRANSLATE, SCALE
   - Expected: 90%+ accuracy (deterministic)

2. **Grid Arithmetic** (100 tasks)
   - COUNT_VALUE, SUM_POSITIONS, MAX, MIN, FILTER
   - Expected: 85%+ accuracy

3. **Pattern Completion** (100 tasks)
   - Sequence completion, symmetry, periodic extension
   - Expected: 75%+ accuracy (inference harder)

4. **Compositional Operations** (100 tasks)
   - Chain operations: ROTATE_90 → MIRROR_H
   - Expected: 70%+ accuracy (composition challenging)

5. **Symbolic RPN Evaluation** (100 tasks)
   - Direct RPN execution: "2 3 ADD" → 5
   - Expected: 85%+ accuracy

---

## 🔨 What You Need to Implement

### Week 20 Timeline (7 days)

**Day 1-2: Task Generation**
```bash
# Create these files:
benchmarks/deterministic_foundation.py          # Main benchmark suite
benchmarks/tasks/geometric_tasks.py             # 100 geometric tasks
benchmarks/tasks/arithmetic_tasks.py            # 100 arithmetic tasks
benchmarks/tasks/pattern_tasks.py               # 100 pattern tasks
benchmarks/tasks/compositional_tasks.py         # 100 compositional tasks
benchmarks/tasks/rpn_tasks.py                   # 100 RPN tasks
tests/test_deterministic_foundation.py          # Validation

# Validate:
pytest tests/test_deterministic_foundation.py
# Expected: 500 tasks generated, all valid RPN programs
```

**Day 3: Galaxy Population**
```bash
# Create:
knowledge3d/knowledgeverse/foundational_operations_bootstrap.py

# Run:
python -m knowledge3d.knowledgeverse.foundational_operations_bootstrap

# Verify:
# - Grammar Galaxy: ~35 operations (geometric + pattern rules)
# - Math Galaxy: ~25 operations (arithmetic)
```

**Day 4: Baseline Run**
```bash
# Create:
scripts/train_deterministic_foundation.py

# Run iteration 0 (baseline):
python scripts/train_deterministic_foundation.py --iterations 1

# Expected results:
# Overall: 20-30% (untrained TRM guessing)
# Geometric: 15%
# Arithmetic: 20%
# Pattern: 25%
# Compositional: 15%
# RPN: 30%
```

**Day 5-6: Full Training (10 iterations)**
```bash
# Run full training:
python scripts/train_deterministic_foundation.py --iterations 10

# Expected progression:
# Iteration 0:  20-30% (baseline)
# Iteration 3:  50-60% (TRM learning navigation)
# Iteration 6:  70-80% (TRM learning composition)
# Iteration 10: 80-95% (foundation complete!)

# Runtime estimate: 2-4 hours
```

**Day 7: Analysis**
```bash
# Check results:
cat foundation_training_results/training_history.json

# Success criteria:
# ✅ Overall accuracy ≥ 80%
# ✅ Geometric ≥ 90%
# ✅ Arithmetic ≥ 85%
# ✅ Pattern ≥ 75%
# ✅ Compositional ≥ 70%
# ✅ RPN ≥ 85%

# If successful → Proceed to Phase 1 (compositional reasoning)
# If unsuccessful → Continue training (10 more iterations)
```

---

## 💻 Code Structure (Summary)

### Main Benchmark Class

```python
# benchmarks/deterministic_foundation.py

class DeterministicFoundationBenchmark:
    """500 deterministic tasks for TRM foundation training."""

    def __init__(self):
        self.tasks = self._generate_all_tasks()  # Generate 500 tasks

    def run_benchmark(self, kv: Knowledgeverse) -> dict:
        """
        Run all 500 tasks, return results.

        Returns:
            {
                "geometric_transforms": {"accuracy": 0.92, ...},
                "grid_arithmetic": {"accuracy": 0.86, ...},
                ...
                "overall": {"accuracy": 0.83, "correct": 415, "total": 500}
            }
        """
        # For each category, solve tasks using TRM
        # TRM navigates Galaxy → retrieves RPN → executes in Cranium
        # Validate: result == expected (deterministic!)

    def _solve_task(self, task: dict, kv: Knowledgeverse) -> Any:
        """
        Solve single task using TRM navigation.

        Flow:
        1. TRM queries Galaxy for operation (e.g., "rotate transformation")
        2. TRM retrieves RPN program from Grammar Galaxy
        3. Cranium executes RPN (sovereign PTX)
        4. Return result
        """
```

### Training Protocol

```python
# scripts/train_deterministic_foundation.py

def train_deterministic_foundation(num_iterations=10):
    """
    Train TRM on deterministic tasks.

    Protocol:
    1. Run 500 tasks (benchmark.run_benchmark())
    2. Record successes in Shadow Copy
    3. Consolidate Shadow Copy → TRM weight updates
    4. Repeat for 10 iterations

    Expected: 20% → 80% accuracy
    """
    kv = Knowledgeverse()
    benchmark = DeterministicFoundationBenchmark()

    for iteration in range(num_iterations):
        # Run benchmark
        results = benchmark.run_benchmark(kv)

        # Consolidate Shadow Copy → TRM weights
        consolidate_iteration_events(iteration, kv)

        # Save iteration results
        # ...

    # Final analysis: Did we achieve ≥80%?
```

### Galaxy Population

```python
# knowledge3d/knowledgeverse/foundational_operations_bootstrap.py

def populate_foundational_operations(galaxy_manager):
    """
    Populate Grammar + Math galaxies with foundational operations.

    Grammar Galaxy (35 operations):
    - Geometric: ROTATE_90, MIRROR_H, TRANSLATE, SCALE
    - Pattern: ALTERNATE, MIRROR_COMPLETE, EXTEND_PERIODIC

    Math Galaxy (25 operations):
    - Arithmetic: COUNT, SUM, MAX, MIN, FILTER
    - Spatial: POSITIONS, COORDS
    """
    # Add entries to Grammar Galaxy
    # Add entries to Math Galaxy
```

---

## 📊 Expected Outcomes

### Iteration-by-Iteration Progression

| Iteration | Overall | What's Happening |
|-----------|---------|------------------|
| **0** | 20-30% | Baseline (TRM guessing) |
| **1** | 30-40% | TRM starts finding operations in Galaxy |
| **2** | 40-50% | Simple operations (ROTATE, MIRROR) working |
| **3** | 50-60% | Arithmetic operations working |
| **4-5** | 60-70% | Pattern inference beginning |
| **6-7** | 70-80% | Compositional operations improving |
| **8-10** | **80-95%** | **Foundation complete!** |

### Per-Category Final Targets (Iteration 10)

| Category | Target | Reasoning |
|----------|--------|-----------|
| Geometric | 90%+ | Deterministic, simple operations |
| Arithmetic | 85%+ | Deterministic counting/summing |
| Pattern | 75%+ | Inference harder but learnable |
| Compositional | 70%+ | Chaining operations challenging |
| RPN | 85%+ | Direct execution, deterministic |

---

## 🎯 Success Criteria (Must Achieve ALL)

**Foundation Training Success:**
- ✅ Overall accuracy ≥ 80% (400/500 tasks correct)
- ✅ Geometric transforms ≥ 90%
- ✅ Grid arithmetic ≥ 85%
- ✅ Pattern completion ≥ 75%
- ✅ Compositional ≥ 70%
- ✅ Symbolic RPN ≥ 85%

**If achieved:**
- ✅ Proceed to Phase 1 (compositional reasoning)
- ✅ Then return to ARC-AGI (Phase 2)
- ✅ Expected ARC improvement: 28% → 40-50%

**If NOT achieved (<80%):**
- ⚠️ Continue training (10 more iterations)
- ⚠️ Analyze failure modes (which categories stuck?)
- ⚠️ Adjust task difficulty or Galaxy population

---

## 🚦 What About Your Proposed Work?

### Your Suggestions (from diagnostic brief)

**You proposed:**
1. Canonical ARC exact-match normalization
2. Strong pre-ranking candidate validity gates
3. Winner provenance telemetry
4. Per-source precision/recall trends
5. Counterfactual run mode

**My recommendation:** **HOLD on ARC diagnostics for now.**

**Why:**
- Your diagnostic shows oracle_at_all = 0.0 (generation is broken)
- Optimizing ranking when generation doesn't work = polishing a broken engine
- Better: Fix generation first (via foundation training), THEN optimize ranking

**Timeline:**
1. **Week 20:** Foundation training (this directive)
2. **Week 21:** Compositional reasoning (Phase 1)
3. **Week 22:** Return to ARC-AGI + implement your diagnostic proposals
   - With better TRM foundation, oracle_at_all should improve (0.0 → 0.5+)
   - THEN your diagnostics (normalization, validity gates, provenance) will be meaningful
   - THEN counterfactual evaluation will show learning lift

**Your diagnostic work is NOT wasted!** We'll use it in Week 22 after foundation training.

---

## 💡 Key Points for Implementation

### Honesty Paradigm Compliance

**All tasks must be:**
- ✅ **Sovereign:** PTX + Galaxy only (no external dependencies in hot path)
- ✅ **Procedural:** RPN programs (not Python logic)
- ✅ **Verifiable:** Exact match validation (deterministic = one correct answer)
- ✅ **Progressive:** Simple → Complex (curriculum learning)

### Shadow Copy Learning

**Critical:** Ensure Shadow Copy consolidation runs after each iteration.

```python
# After running 500 tasks:
consolidate_iteration_events(iteration, kv)

# This updates TRM weights based on successful events
# Without this, TRM won't learn!
```

### Galaxy Universe Integration

**TRM must navigate Galaxy, NOT hardcoded logic:**

```python
# ❌ WRONG (hardcoded):
if task["operation"] == "ROTATE_90":
    return rotate_90(grid)

# ✅ CORRECT (Galaxy navigation):
results = kv.galaxy_manager.query("rotate transformation", specialist="grammar", top_k=1)
rpn_program = results[0]["rpn_program"]
return kv.cranium.execute_rpn(rpn_program, {"GRID": grid})
```

**This way, TRM LEARNS which Galaxy queries find correct operations.**

---

## 📞 Communication Protocol

### Daily Updates (Week 20)

**Please report:**
- Day 1: Task generation progress (how many tasks implemented?)
- Day 2: Task validation results (all tests passing?)
- Day 3: Galaxy population complete (how many operations added?)
- Day 4: Baseline results (what's iteration 0 accuracy?)
- Day 5-6: Training progress (iteration-by-iteration results)
- Day 7: Final analysis (did we achieve ≥80%?)

### Blocking Issues

**If you encounter blockers:**
- PTX kernels missing for operations? (let me know which ones)
- Galaxy query not finding operations? (check Galaxy population)
- RPN execution errors? (validate RPN programs)
- Shadow Copy not consolidating? (check sleeptime.py integration)

**I'll provide architectural guidance for any blockers.**

---

## 🎉 Bottom Line

### What You're Doing

**Building the foundation that makes brute force learning actually work.**

### Why It Matters

**Current state:**
- ARC-AGI: oracle_at_all = 0.0 (can't generate correct answers)
- Root cause: TRM hasn't learned HOW to navigate/compose

**After foundation training:**
- TRM learns navigation (Grammar + Math galaxies)
- TRM learns composition (chain operations)
- TRM learns validation (pattern matching)
- Expected ARC-AGI: 28% → 40-50% (better generation quality)

### Your Mission (Week 20)

1. ✅ Implement 500 deterministic tasks (5 categories)
2. ✅ Populate Galaxy with foundational operations
3. ✅ Run baseline (iteration 0, expect 20-30%)
4. ✅ Train for 10 iterations (expect 20% → 80%)
5. ✅ Validate success (≥80% accuracy)
6. ✅ Report results + proceed to Phase 1

**Estimated effort:** 1 week (Week 20)
**Expected outcome:** TRM foundation complete, ready for compositional reasoning

---

## 📚 Reference Files

**Complete specification:**
- [TEMP/CODEX_DETERMINISTIC_FOUNDATION_CURRICULUM_02.08.2026.md](CODEX_DETERMINISTIC_FOUNDATION_CURRICULUM_02.08.2026.md) (1,500+ lines)

**Your diagnostic work (saved for Week 22):**
- [TEMP/CODEX_DIAGNOSTIC_DISCUSSION_BRIEF_02.08.2026.md](CODEX_DIAGNOSTIC_DISCUSSION_BRIEF_02.08.2026.md)
- [TEMP/CODEX_DIAGNOSTIC_FRAMEWORK_IMPLEMENTATION_02.08.2026.md](CODEX_DIAGNOSTIC_FRAMEWORK_IMPLEMENTATION_02.08.2026.md)

**Strategic context:**
- User observation: Train TRM to "think and organize thought" before BFL
- Your finding: oracle_at_all = 0.0 (generation is the bottleneck)
- Solution: Foundation training → compositional reasoning → ARC-AGI

---

## ✅ Ready to Start?

**You have everything you need:**
- ✅ Complete task specifications (500 tasks, 5 categories)
- ✅ Full code examples (benchmark, training, galaxy population)
- ✅ Success criteria (80% accuracy required)
- ✅ Timeline (Week 20, 7 days)
- ✅ Strategic context (why we're doing this)

**Let's build the foundation!** 🚀

**Start with Day 1: Implement geometric_tasks.py + arithmetic_tasks.py (200 tasks).**

---

**Directive issued by:** Claude (Architecture Partner)
**Date:** February 8, 2026
**For:** Codex (Implementation Partner)
**Status:** Ready for implementation
**Priority:** 🔴 CRITICAL — Foundation before ARC-AGI
