# CODEX: Progressive Curriculum — Child to Athlete

**Date:** February 8, 2026
**Authors:** Claude (Architecture) + Codex (Implementation) + User (Strategic Vision)
**Status:** 🔴 CRITICAL ENHANCEMENT — Difficulty Scheduling
**Context:** Deterministic foundation saturates at 100% (too easy after bootstrap)

---

## 🎯 The "Child Learning to Walk" Analogy

### User's Strategic Vision

> "A child learning to walk can not run a marathon"

**Translation to TRM training:**

| Human Development | TRM Training | Current Status | What's Needed |
|-------------------|--------------|----------------|---------------|
| **Infant (crawling)** | Learn basic operations exist | ✅ Bootstrap complete (63 ops) | ✅ Done |
| **Child (walking)** | Execute operations with direct names | ✅ Stage A saturated (100%) | Need progressive difficulty |
| **Teen (running)** | Infer operations from descriptions | ❌ Not trained yet | Stage B-C (this file!) |
| **Athlete (marathon)** | Multi-step reasoning with noise | ❌ Not trained yet | Stage D + ARC-AGI |

**Critical insight (Codex):**
> "The current deterministic suite saturates quickly after bootstrap (expected). The missing piece to make this behave like 'child learning to walk' is difficulty scheduling, not more static tasks."

**Problem:**
- Current tasks: "ROTATE_90" → TRM queries "rotate_90_cw" (exact match, 100% immediately)
- No learning curve: Like giving a child a bicycle instead of teaching them to walk first

**Solution:**
- Progressive curriculum with difficulty gates
- Stage A → B → C → D progression
- Each stage requires NEW skills, not just more tasks

---

## 📊 Current State Analysis

### Codex's Implementation (Excellent!)

**What works:**
- ✅ 500 deterministic tasks generated
- ✅ 63 foundational operations bootstrapped (35 Grammar + 28 Math)
- ✅ Training driver with Shadow Copy consolidation
- ✅ All tests passing (13 tests)

**What saturates:**
- ⚠️ **initial=1.000 final=1.000** (100% from iteration 0!)
- ⚠️ No learning curve (TRM doesn't need to improve)
- ⚠️ Tasks too easy after bootstrap (direct op name matching)

**Why this happened:**
```python
# Current task (Stage A - too easy):
task = {
    "operation": "ROTATE_90",  # Direct operation name
    "input": grid,
    "expected": rotated_grid,
}

# TRM solution:
query = "ROTATE_90"  # Exact match in Grammar Galaxy
ops = galaxy.query(query)  # Finds "rotate_90_cw" immediately
# Result: 100% accuracy, no learning needed
```

**User's diagnosis (via analogy):**
- This is like teaching a child to walk by carrying them
- They never learn to balance, coordinate, problem-solve
- When we ask them to run (ARC-AGI), they fall

---

## 🏗️ Progressive Curriculum Architecture

### 4-Stage Difficulty Progression

**Metaphor mapping:**

| Stage | Human Analogy | TRM Skill | Difficulty | Success Criteria |
|-------|---------------|-----------|------------|------------------|
| **A: Standing** | Child stands with support | Execute ops with direct names | Easy (100% after bootstrap) | ≥95% accuracy |
| **B: Walking** | Child walks independently | Infer ops from descriptions | Medium (60-80% initial) | ≥85% accuracy |
| **C: Running** | Teen runs with obstacles | Choose correct op among distractors, chain operations | Hard (40-60% initial) | ≥75% accuracy |
| **D: Marathon** | Athlete completes marathon | Multi-step reasoning with noise, sparse context | Very Hard (20-40% initial) | ≥65% accuracy |

**Progression gates:**
- A → B: When Stage A achieves ≥95% for 3 consecutive iterations
- B → C: When Stage B achieves ≥85% for 3 consecutive iterations
- C → D: When Stage C achieves ≥75% for 3 consecutive iterations
- D → ARC-AGI: When Stage D achieves ≥65% for 3 consecutive iterations

---

## 🔬 Stage Specifications

### Stage A: Standing (Direct Op Names) — ✅ COMPLETE

**What TRM learns:** Operations exist and can be executed

**Task format:**
```python
{
    "operation": "ROTATE_90",  # Direct, exact operation name
    "input": grid,
    "expected": rotated_grid,
    "hint": "rotate_90_cw",  # Galaxy entry ID (makes it trivial)
}
```

**TRM challenge:** **None** (exact match query)

**Current status:** ✅ **Saturated at 100%** (bootstrap provides all ops)

**Progression gate:** ≥95% for 3 iterations → **Already met, advance to Stage B**

---

### Stage B: Walking (Alias-Only Prompts) — 🔴 IMPLEMENT NEXT

**What TRM learns:** Infer operations from natural descriptions

**User's analogy:** Child walks independently (no support/direct names)

**Task format:**
```python
{
    "description": "turn the grid 90 degrees clockwise",  # Alias, NOT exact op name
    "input": grid,
    "expected": rotated_grid,
    # NO "operation" field (TRM must infer!)
}
```

**TRM challenge:**
1. Parse description → identify operation type ("turn" = rotate)
2. Query Galaxy with inferred concept ("rotate transformation")
3. Retrieve candidate operations (might get ROTATE_90, ROTATE_180, etc.)
4. Filter by description ("90 degrees clockwise" → ROTATE_90)
5. Execute selected operation

**Why this is harder:**
- No exact match (must infer semantic meaning)
- Requires understanding synonyms ("turn" = "rotate", "flip" = "mirror")
- Tests TRM's Galaxy navigation (can it find ops from descriptions?)

**Example tasks:**

| Description | Operation | Expected Accuracy (Initial) |
|-------------|-----------|----------------------------|
| "turn the grid 90 degrees clockwise" | ROTATE_90 | 70% |
| "flip the grid horizontally" | MIRROR_H | 75% |
| "count how many cells are red" | COUNT_VALUE (red=1) | 80% |
| "make the grid twice as big" | SCALE_2 | 60% |
| "slide the grid 1 unit right" | TRANSLATE_1_0 | 65% |

**Expected progression:**
- Iteration 0: 60-70% (TRM struggles with inference)
- Iteration 3: 75-80% (TRM learns common aliases)
- Iteration 5: 85%+ (gate met, advance to Stage C)

**Implementation:**

```python
# benchmarks/tasks/stage_b_alias_tasks.py - NEW FILE

def generate_stage_b_tasks(num_tasks: int = 500) -> List[Dict]:
    """
    Generate tasks with alias-only prompts (no direct op names).

    TRM must infer operation from natural description.
    """
    tasks = []

    # Rotation aliases
    rotation_aliases = [
        ("turn the grid 90 degrees clockwise", "ROTATE_90"),
        ("rotate the grid a quarter turn to the right", "ROTATE_90"),
        ("spin the grid 90 degrees", "ROTATE_90"),
        ("turn the grid 180 degrees", "ROTATE_180"),
        ("flip the grid upside down", "ROTATE_180"),
        ("turn the grid 270 degrees clockwise", "ROTATE_270"),
        ("rotate the grid three-quarters to the right", "ROTATE_270"),
    ]

    for i, (description, operation) in enumerate(rotation_aliases * 15):  # 105 rotation tasks
        size = [2, 3, 4, 5][i % 4]
        grid = np.random.randint(0, 3, (size, size))

        angle = int(operation.split('_')[1])
        expected = rotate_grid(grid, angle)

        tasks.append({
            "category": "stage_b_walking",
            "subcategory": "rotation_alias",
            "task_id": f"stage_b_rot_{i:03d}",
            "description": description,  # Natural language (no exact op name!)
            "input": grid,
            "expected": expected,
            "ground_truth_operation": operation,  # For validation only (not given to TRM)
        })

    # Mirror aliases
    mirror_aliases = [
        ("flip the grid horizontally", "MIRROR_H"),
        ("reflect the grid left-right", "MIRROR_H"),
        ("mirror the grid across vertical axis", "MIRROR_H"),
        ("flip the grid vertically", "MIRROR_V"),
        ("reflect the grid top-bottom", "MIRROR_V"),
        ("mirror the grid across horizontal axis", "MIRROR_V"),
    ]

    for i, (description, operation) in enumerate(mirror_aliases * 18):  # 108 mirror tasks
        size = [2, 3, 4, 5][i % 4]
        grid = np.random.randint(0, 3, (size, size))

        axis = operation.split('_')[1]
        expected = mirror_grid(grid, axis)

        tasks.append({
            "category": "stage_b_walking",
            "subcategory": "mirror_alias",
            "task_id": f"stage_b_mir_{i:03d}",
            "description": description,
            "input": grid,
            "expected": expected,
            "ground_truth_operation": operation,
        })

    # Arithmetic aliases (similar structure)
    # Translation aliases
    # Scale aliases
    # ... (total 500 tasks)

    return tasks
```

**Success criteria:** ≥85% for 3 iterations → Advance to Stage C

---

### Stage C: Running (Distractors + Compositional Chains) — 🔴 IMPLEMENT AFTER STAGE B

**What TRM learns:** Choose correct operation among plausible alternatives, chain multi-step operations

**User's analogy:** Teen runs with obstacles (must dodge, navigate, coordinate)

**Task format (with distractors):**
```python
{
    "description": "turn the grid 90 degrees clockwise",
    "input": grid,
    "expected": rotated_grid,
    "distractors": [
        "ROTATE_180",  # Plausible (rotation) but wrong angle
        "ROTATE_270",  # Plausible but opposite direction
        "MIRROR_H",    # Not rotation but visual transform
    ],
    # TRM must choose ROTATE_90 from 4 options (correct + 3 distractors)
}
```

**Task format (compositional chains):**
```python
{
    "description": "rotate 90 degrees clockwise, then flip horizontally",
    "input": grid,
    "expected": composed_grid,
    "operations": ["ROTATE_90", "MIRROR_H"],  # Ground truth (not given to TRM)
    # TRM must:
    # 1. Parse multi-step description
    # 2. Infer two operations
    # 3. Compose into RPN: "GRID ROTATE_90 MIRROR_H"
    # 4. Execute composed program
}
```

**TRM challenge:**
1. **Disambiguation:** Choose correct op when multiple plausible (requires understanding nuance)
2. **Composition:** Parse multi-step descriptions into sequential operations
3. **Validation:** Intermediate results must be correct (chaining errors compound)

**Why this is harder:**
- Distractors force TRM to differentiate (not just retrieve first match)
- Compositional chains require planning (must parse "then" → sequential ops)
- Tests TRM's reasoning (can it handle multi-step problems?)

**Example tasks:**

| Description | Correct Op | Distractors | Expected Accuracy (Initial) |
|-------------|------------|-------------|----------------------------|
| "turn 90 degrees clockwise" | ROTATE_90 | ROTATE_180, ROTATE_270, MIRROR_H | 50% |
| "count cells greater than 5" | FILTER_GT (threshold=5) | COUNT_ALL, MAX, SUM | 45% |
| "rotate 90°, then mirror horizontally" | [ROTATE_90, MIRROR_H] | (compositional) | 40% |
| "flip vertically, then count red cells" | [MIRROR_V, COUNT_VALUE] | (compositional) | 35% |

**Expected progression:**
- Iteration 0: 40-50% (TRM confused by distractors)
- Iteration 3: 55-65% (TRM learns to differentiate)
- Iteration 5: 70-75% (TRM handles composition)
- Iteration 8: 75%+ (gate met, advance to Stage D)

**Implementation:**

```python
# benchmarks/tasks/stage_c_running_tasks.py - NEW FILE

def generate_stage_c_tasks(num_tasks: int = 500) -> List[Dict]:
    """
    Generate tasks with distractors and compositional chains.

    250 distractor tasks (choose correct op among 3-5 candidates)
    250 compositional tasks (chain 2-3 operations)
    """
    tasks = []

    # Distractor tasks (250)
    distractor_sets = {
        "ROTATE_90": ["ROTATE_180", "ROTATE_270", "MIRROR_H"],
        "MIRROR_H": ["MIRROR_V", "ROTATE_180", "TRANSLATE_0_1"],
        "COUNT_VALUE": ["COUNT_ALL", "SUM_POSITIONS", "MAX"],
        "SCALE_2": ["SCALE_3", "TRANSLATE_1_1", "ROTATE_90"],
        # ... more distractor sets
    }

    for correct_op, distractors in distractor_sets.items():
        for i in range(50):  # 50 tasks per operation
            size = [2, 3, 4][i % 3]
            grid = np.random.randint(0, 3, (size, size))

            # Generate description (alias for correct op)
            description = get_alias_for_operation(correct_op, variation=i)

            # Apply correct operation for expected output
            expected = apply_operation(grid, correct_op)

            tasks.append({
                "category": "stage_c_running",
                "subcategory": "distractor",
                "task_id": f"stage_c_dist_{len(tasks):03d}",
                "description": description,
                "input": grid,
                "expected": expected,
                "correct_operation": correct_op,
                "distractors": distractors,  # TRM sees these as plausible options
            })

    # Compositional tasks (250)
    composition_templates = [
        (["ROTATE_90", "MIRROR_H"], "rotate 90 degrees clockwise, then flip horizontally"),
        (["MIRROR_V", "ROTATE_180"], "flip vertically, then turn 180 degrees"),
        (["TRANSLATE_1_0", "MIRROR_H"], "slide 1 unit right, then mirror horizontally"),
        (["ROTATE_90", "SCALE_2"], "turn 90 degrees, then make twice as big"),
        (["COUNT_VALUE", "FILTER_GT"], "count red cells, then filter values greater than result"),
        # ... more templates
    ]

    for operations, description in composition_templates:
        for i in range(50):  # 50 tasks per template
            size = [2, 3, 4][i % 3]
            grid = np.random.randint(0, 3, (size, size))

            # Apply operations sequentially
            expected = grid
            for op in operations:
                expected = apply_operation(expected, op)

            tasks.append({
                "category": "stage_c_running",
                "subcategory": "compositional",
                "task_id": f"stage_c_comp_{len(tasks):03d}",
                "description": description,
                "input": grid,
                "expected": expected,
                "operations": operations,  # Ground truth (not given to TRM)
            })

    return tasks
```

**Success criteria:** ≥75% for 3 iterations → Advance to Stage D

---

### Stage D: Marathon (Sparse Context + Noisy Phrasing) — 🔴 IMPLEMENT AFTER STAGE C

**What TRM learns:** Robust reasoning with incomplete information and linguistic variation

**User's analogy:** Athlete completes marathon (endurance, adaptation, robustness)

**Task format (sparse context):**
```python
{
    "description": "transform the grid",  # Vague! TRM must infer from examples
    "examples": [
        {"input": grid1, "output": rotated_grid1},
        {"input": grid2, "output": rotated_grid2},
    ],  # TRM must infer: these are ROTATE_90 examples
    "input": grid3,
    "expected": rotated_grid3,
    # NO explicit description of operation (must infer from examples)
}
```

**Task format (noisy phrasing):**
```python
{
    "description": "could you maybe rotate this grid about 90 degrees or so to the right?",
    # Noisy: "could you maybe", "or so", "to the right" (vs "clockwise")
    "input": grid,
    "expected": rotated_grid,
    # TRM must extract core meaning despite noise
}
```

**TRM challenge:**
1. **Induction:** Infer operation from input/output examples (no explicit description)
2. **Robustness:** Parse noisy language ("could you maybe", "or so", "to the right")
3. **Abstraction:** Generalize from few examples to new inputs

**Why this is hardest:**
- Sparse context = TRM can't rely on explicit descriptions
- Noisy phrasing = TRM must extract signal from noise (like real-world instructions)
- Tests TRM's abstraction (can it infer rules from examples, like ARC-AGI?)

**Example tasks:**

| Description/Examples | Operation | Expected Accuracy (Initial) |
|---------------------|-----------|----------------------------|
| Examples: [grid → rotated] × 3 | Infer: ROTATE_90 | 25% |
| "could you maybe flip this horizontally?" | MIRROR_H | 30% |
| "I think we should turn it like 90 degrees" | ROTATE_90 | 35% |
| Examples: [grid → counted] × 2, description: "do the same" | COUNT_VALUE | 20% |

**Expected progression:**
- Iteration 0: 20-30% (TRM struggles with vague descriptions)
- Iteration 5: 40-50% (TRM learns to infer from examples)
- Iteration 10: 55-65% (TRM robust to noise)
- Iteration 15: 65%+ (gate met, ready for ARC-AGI!)

**Implementation:**

```python
# benchmarks/tasks/stage_d_marathon_tasks.py - NEW FILE

def generate_stage_d_tasks(num_tasks: int = 500) -> List[Dict]:
    """
    Generate tasks with sparse context and noisy phrasing.

    250 sparse context tasks (infer from examples)
    250 noisy phrasing tasks (extract signal from noise)
    """
    tasks = []

    # Sparse context tasks (250)
    operations = ["ROTATE_90", "MIRROR_H", "COUNT_VALUE", "SCALE_2", "TRANSLATE_1_0"]

    for op in operations:
        for i in range(50):  # 50 tasks per operation
            # Generate 2-3 example input/output pairs
            examples = []
            for _ in range(2 + (i % 2)):  # 2 or 3 examples
                size = [2, 3][i % 2]
                example_input = np.random.randint(0, 3, (size, size))
                example_output = apply_operation(example_input, op)
                examples.append({"input": example_input, "output": example_output})

            # Generate test input (TRM must apply inferred op)
            test_input = np.random.randint(0, 3, (size, size))
            expected = apply_operation(test_input, op)

            tasks.append({
                "category": "stage_d_marathon",
                "subcategory": "sparse_context",
                "task_id": f"stage_d_sparse_{len(tasks):03d}",
                "description": "transform the grid like the examples",  # Vague!
                "examples": examples,
                "input": test_input,
                "expected": expected,
                "ground_truth_operation": op,
            })

    # Noisy phrasing tasks (250)
    noisy_templates = {
        "ROTATE_90": [
            "could you maybe rotate this grid about 90 degrees or so to the right?",
            "I think we should turn it like 90 degrees clockwise",
            "umm... can you spin this grid a quarter turn?",
            "rotate-ish 90 degrees I guess",
        ],
        "MIRROR_H": [
            "could you flip this horizontally please?",
            "I want to mirror it left-right maybe",
            "umm... reflect across the vertical axis?",
            "flip-wise horizontal I think",
        ],
        # ... more noisy templates for each operation
    }

    for op, templates in noisy_templates.items():
        for i in range(50):  # 50 tasks per operation
            size = [2, 3, 4][i % 3]
            grid = np.random.randint(0, 3, (size, size))
            expected = apply_operation(grid, op)

            description = templates[i % len(templates)]

            tasks.append({
                "category": "stage_d_marathon",
                "subcategory": "noisy_phrasing",
                "task_id": f"stage_d_noisy_{len(tasks):03d}",
                "description": description,
                "input": grid,
                "expected": expected,
                "ground_truth_operation": op,
            })

    return tasks
```

**Success criteria:** ≥65% for 3 iterations → **Ready for ARC-AGI!**

---

## 🎓 Progression Gates & Curriculum Flow

### Automatic Stage Advancement

```python
# scripts/train_progressive_curriculum.py - ENHANCED VERSION

def train_progressive_curriculum(max_iterations: int = 50):
    """
    Train TRM with progressive curriculum and automatic stage advancement.

    Stages:
    - A: Standing (direct op names) → 95% for 3 iters → advance
    - B: Walking (alias prompts) → 85% for 3 iters → advance
    - C: Running (distractors + composition) → 75% for 3 iters → advance
    - D: Marathon (sparse + noisy) → 65% for 3 iters → ARC-AGI ready!
    """
    kv = Knowledgeverse()

    stages = {
        "A_standing": {
            "benchmark": DeterministicFoundationBenchmark(),
            "gate": 0.95,
            "consecutive_required": 3,
        },
        "B_walking": {
            "benchmark": StageBWalkingBenchmark(),
            "gate": 0.85,
            "consecutive_required": 3,
        },
        "C_running": {
            "benchmark": StageCRunningBenchmark(),
            "gate": 0.75,
            "consecutive_required": 3,
        },
        "D_marathon": {
            "benchmark": StageDMarathonBenchmark(),
            "gate": 0.65,
            "consecutive_required": 3,
        },
    }

    current_stage = "A_standing"
    consecutive_successes = 0
    training_history = []

    for iteration in range(max_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1} — Stage: {current_stage}")
        print(f"{'='*60}")

        # Run current stage benchmark
        stage_config = stages[current_stage]
        results = stage_config["benchmark"].run_benchmark(kv)

        accuracy = results["overall"]["accuracy"]
        print(f"Accuracy: {accuracy:.1%} (gate: {stage_config['gate']:.1%})")

        # Consolidate Shadow Copy
        consolidate_iteration_events(iteration, kv)

        # Check progression gate
        if accuracy >= stage_config["gate"]:
            consecutive_successes += 1
            print(f"Gate progress: {consecutive_successes}/{stage_config['consecutive_required']}")

            if consecutive_successes >= stage_config["consecutive_required"]:
                # Advance to next stage
                stage_names = list(stages.keys())
                current_idx = stage_names.index(current_stage)

                if current_idx + 1 < len(stage_names):
                    next_stage = stage_names[current_idx + 1]
                    print(f"\n🎉 STAGE COMPLETE! Advancing: {current_stage} → {next_stage}")
                    current_stage = next_stage
                    consecutive_successes = 0
                else:
                    print(f"\n🏆 ALL STAGES COMPLETE! TRM ready for ARC-AGI!")
                    break
        else:
            consecutive_successes = 0
            print(f"Below gate, resetting progress counter")

        # Record iteration
        training_history.append({
            "iteration": iteration,
            "stage": current_stage,
            "accuracy": accuracy,
            "gate": stage_config["gate"],
            "consecutive_successes": consecutive_successes,
        })

    return training_history
```

### Expected Timeline

**Optimistic path:**
- Stage A: 0 iterations (already saturated at 100%)
- Stage B: 5-8 iterations (alias learning)
- Stage C: 8-12 iterations (distractor + composition)
- Stage D: 12-20 iterations (sparse + noisy)
- **Total:** 25-40 iterations (1-2 weeks runtime)

**Realistic path:**
- Stage A: 1-2 iterations (verify saturation)
- Stage B: 8-12 iterations (alias harder than expected)
- Stage C: 12-18 iterations (composition challenging)
- Stage D: 18-30 iterations (robustness requires more training)
- **Total:** 39-62 iterations (2-3 weeks runtime)

---

## 🎯 Mapping to ARC-AGI Performance

### Expected ARC-AGI Improvements

**After each stage:**

| Stage Complete | TRM Skill Gained | Expected ARC-AGI Accuracy |
|----------------|------------------|--------------------------|
| **Baseline (no training)** | None | 20% (random patterns) |
| **A: Standing** | Execute direct ops | 25% (slight improvement) |
| **B: Walking** | Infer ops from descriptions | **35-40%** (better pattern matching) |
| **C: Running** | Composition + disambiguation | **45-55%** (multi-step reasoning) |
| **D: Marathon** | Robust inference from examples | **55-70%** (approaching human-level!) |

**Why progressive curriculum helps ARC-AGI:**

1. **Stage B (Alias)** → Better pattern matching
   - ARC tasks describe transformations implicitly (visual examples)
   - TRM learns to infer operations from descriptions/examples
   - Direct transfer: "What transformation turns this into that?"

2. **Stage C (Composition)** → Multi-step reasoning
   - Many ARC tasks require chained operations (rotate → filter → mirror)
   - TRM learns compositional reasoning
   - Direct transfer: Parse complex visual transformations into steps

3. **Stage D (Sparse + Noisy)** → Few-shot learning
   - ARC provides 2-3 example pairs (sparse context!)
   - TRM learns to infer rules from few examples
   - Direct transfer: Exactly what ARC requires (induction from examples)

**The "marathon" becomes achievable** when you've trained for it progressively.

---

## 💡 Enhanced Ideas (Claude + Codex Synthesis)

### Codex's Original Proposals (Excellent!)

1. ✅ Alias-only prompts (Stage B)
2. ✅ Distractor candidates (Stage C)
3. ✅ Longer compositional chains (Stage C)
4. ✅ Noisy phrasing (Stage D)

### Claude's Enhancements

5. **Staged progression gates** (A → B → C → D with automatic advancement)
   - TRM can't skip stages (must master walking before running)
   - Prevents overfitting to hard tasks without foundation

6. **Sparse context tasks** (Stage D)
   - Infer operation from input/output examples only
   - Directly mimics ARC-AGI format (few-shot learning)
   - Tests TRM's inductive reasoning

7. **Metaphor-driven design** (child → athlete)
   - Each stage maps to human development
   - Makes curriculum progression intuitive
   - Clear success criteria (when is TRM "ready to run"?)

8. **Gradual distractor introduction** (Stage C)
   - Start: 1 correct + 2 distractors (33% chance if guessing)
   - Mid: 1 correct + 3 distractors (25% chance)
   - End: 1 correct + 4 distractors (20% chance)
   - Forces TRM to develop strong differentiation

9. **Compositional depth progression** (Stage C)
   - Start: 2-step chains (ROTATE → MIRROR)
   - Mid: 3-step chains (ROTATE → MIRROR → COUNT)
   - End: 4-step chains (ROTATE → MIRROR → FILTER → COUNT)
   - Tests TRM's planning horizon

10. **Linguistic variation** (Stage B-D)
    - Stage B: Clean aliases ("flip horizontally")
    - Stage C: Casual phrasing ("flip it left-right")
    - Stage D: Noisy phrasing ("umm... flip-ish horizontally I guess?")
    - Builds robustness to real-world variation

### Combined Innovation: Difficulty Knobs

**Each stage has tunable difficulty parameters:**

```python
# Stage B: Alias difficulty
stage_b_config = {
    "alias_diversity": 5,  # How many synonyms per operation
    "semantic_distance": "medium",  # How far aliases deviate from op name
    # "flip" = close, "reflect" = medium, "make symmetrical" = far
}

# Stage C: Distractor difficulty
stage_c_config = {
    "num_distractors": 3,  # How many wrong options
    "distractor_plausibility": "high",  # How plausible distractors are
    # ROTATE_90 distractors: low = MIRROR_H, high = ROTATE_180
    "composition_length": 2,  # How many ops to chain
}

# Stage D: Noise difficulty
stage_d_config = {
    "example_count": 2,  # How many examples for sparse context
    "noise_level": "medium",  # How much linguistic noise
    # low = "rotate 90", medium = "turn it 90 degrees", high = "umm maybe rotate-ish 90?"
}
```

**This allows fine-tuning if stages are too hard/easy.**

---

## 🚀 Implementation Roadmap

### Week 21: Stage B (Walking)

**Day 1-2: Task generation**
```bash
# Implement:
benchmarks/tasks/stage_b_alias_tasks.py  # 500 alias-only tasks

# Validate:
pytest tests/test_stage_b_walking.py
```

**Day 3-5: Training**
```bash
# Run Stage B training:
python scripts/train_progressive_curriculum.py --start-stage B

# Expected:
# Iteration 0: 60-70% (TRM struggles with inference)
# Iteration 5: 80-85% (gate approaching)
# Iteration 8: 85%+ (gate met, advance to Stage C)
```

**Day 6-7: Analysis + ARC-AGI probe**
```bash
# After Stage B complete, probe ARC-AGI:
python benchmarks/arc_agi_2.py --num-tasks 100

# Expected improvement:
# Before Stage B: 28% (baseline)
# After Stage B: 35-40% (+7-12% improvement!)
```

### Week 22: Stage C (Running)

**Day 1-2: Task generation**
```bash
benchmarks/tasks/stage_c_running_tasks.py  # 500 distractor + compositional tasks
```

**Day 3-6: Training**
```bash
# Expected:
# Iteration 0: 40-50%
# Iteration 8: 70-75%
# Iteration 12: 75%+ (gate met, advance to Stage D)
```

**Day 7: ARC-AGI probe**
```bash
# Expected improvement:
# After Stage C: 45-55% (+10% from Stage B)
```

### Week 23-24: Stage D (Marathon)

**Day 1-3: Task generation**
```bash
benchmarks/tasks/stage_d_marathon_tasks.py  # 500 sparse + noisy tasks
```

**Day 4-10: Training**
```bash
# Expected:
# Iteration 0: 20-30%
# Iteration 10: 50-60%
# Iteration 20: 65%+ (gate met, TRM ready!)
```

**Day 11-14: Final ARC-AGI validation**
```bash
# Run full ARC-AGI benchmark:
python benchmarks/arc_agi_2.py --num-tasks 400 --enriched

# Expected final performance:
# After Stage D: 55-70% (!!!)
# Improvement from baseline: +27-42%
```

---

## 📊 Success Metrics

### Per-Stage Validation

**Stage B success:**
- ✅ Achieves 85%+ accuracy on alias tasks
- ✅ ARC-AGI improves by +7-12% (28% → 35-40%)
- ✅ TRM learns synonyms ("flip" = "mirror", "turn" = "rotate")

**Stage C success:**
- ✅ Achieves 75%+ accuracy on distractor + compositional tasks
- ✅ ARC-AGI improves by +10% (35-40% → 45-55%)
- ✅ TRM differentiates similar operations (ROTATE_90 vs ROTATE_180)
- ✅ TRM chains 2-3 operations correctly

**Stage D success:**
- ✅ Achieves 65%+ accuracy on sparse + noisy tasks
- ✅ ARC-AGI improves by +10-15% (45-55% → 55-70%)
- ✅ TRM infers operations from 2-3 examples (few-shot learning!)
- ✅ TRM robust to linguistic noise

**Overall curriculum success:**
- ✅ ARC-AGI: 28% → **55-70%** (+27-42% improvement!)
- ✅ TRM demonstrates progressive learning (not saturation)
- ✅ TRM ready for brute force learning (Phase 2)

---

## 🎉 Bottom Line for Codex

### What You're Building Next

**Stage B: Walking (Alias-Only Prompts)**
- 500 tasks with natural descriptions (no direct op names)
- TRM must infer operations from descriptions
- Expected: 60% → 85% over 5-8 iterations

**Then Stages C & D:**
- Stage C: Distractors + composition (choose correct, chain ops)
- Stage D: Sparse context + noise (infer from examples, robust parsing)

### Why This Fixes the Saturation

**Current problem:**
- Tasks too easy (100% from iteration 0)
- TRM doesn't learn (no difficulty curve)

**Progressive curriculum solution:**
- Each stage requires NEW skills (not just more tasks)
- Stage B: Inference (not trivial!)
- Stage C: Disambiguation + composition (hard!)
- Stage D: Few-shot learning (very hard!)

**Result:**
- Learning curve: 60% → 85% → 75% → 65% (progressive mastery)
- ARC-AGI improvement: 28% → 55-70% (after full curriculum)
- TRM foundation complete, ready for brute force learning

### User's Analogy (Your North Star)

> "A child learning to walk can not run a marathon"

**You're building the training program** that takes TRM from crawling (Stage A) to marathon-ready (Stage D).

**Each stage = developmental milestone:**
- Standing → Walking → Running → Marathon
- Direct ops → Aliases → Composition → Sparse inference

**When Stage D completes, TRM can "run the marathon" (ARC-AGI with 55-70% accuracy).**

---

## 🚦 Immediate Next Steps

**For Codex (Week 21):**

1. **Implement Stage B task generator**
   - File: `benchmarks/tasks/stage_b_alias_tasks.py`
   - 500 tasks with alias-only descriptions
   - Validate: All tasks solvable with current Galaxy operations

2. **Create Stage B benchmark**
   - File: `benchmarks/stage_b_walking.py`
   - Integrate with progressive curriculum training

3. **Enhance training driver**
   - File: `scripts/train_progressive_curriculum.py`
   - Add automatic stage advancement (gates + consecutive successes)

4. **Run Stage B training**
   - Expected: 5-8 iterations to 85% gate
   - Report: Iteration-by-iteration progression

5. **ARC-AGI probe**
   - After Stage B complete, run ARC-AGI
   - Expected: 28% → 35-40% improvement

**Timeline:** Week 21 (7 days)

**Let's teach TRM to walk before asking it to run the marathon!** 🚀

---

**Document prepared by:** Claude (Architecture) + Codex (Implementation Insights) + User (Strategic Vision)
**Date:** February 8, 2026
**Status:** Ready for Stage B implementation
**Analogy credit:** User ("child learning to walk")
