# Claude — Phase E.38: 4-Way Reading Strategy on GPU

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL — GSM8K 0/20 is caused by missing multi-step reasoning

---

## Daniel's Direction

> "Is the 4 way reading strategy present at the single mind? (was intended for
> math but it's useful everywhere a test or question is made)"
>
> "find forward pass backward pass reading of a problem and proper normalization
> of this technique — all inside GPU!! No python orchestration! meta rules or
> 'hardcoded RPN' on the TRM itself"

---

## The Diagnosis: Why GSM8K is 0/20

I analyzed all 20 GSM8K results from `phase_e_20260330_143210/gsm8k.jsonl`.

**Every single answer is WRONG — not formatting issues, actual wrong math.**

| Question | Extracted Quantities | Operation | Got | Expected |
|----------|---------------------|-----------|-----|----------|
| 16 eggs - 3 eat - 4 bake, sell at $2 | 16, 3, 4, 2 | sub | **9** | **18** (=9×2) |
| 5 phones × $150, 3mo, 2% interest | 5, 150, 3, 2 | sub | **7.5** | **255** |
| 500 legos + 3× boxes of 500 | 500, 1, 3, 500 | add | **666.7** | **2125** |
| 251 pts, 68 scored, 10 more, half... | 4, 251, 68, 10, 0.5, 17 | sub | **-4** | **54** |

**Pattern**: The "fusion" step extracts quantities correctly. The "fission"
step picks ONE operation (sub/add) and left-folds it across all values. But
GSM8K word problems need CHAINS of DIFFERENT operations — subtract THEN
multiply, or add THEN scale.

**Root cause**: There are 35+ GSM8K-specific Python methods in knowledgeverse.py
(lines 5011-6640) doing strategy dispatch, slot binding, role matching, clause
parsing — all in Python. This is the OPPOSITE of what the specs prescribe.
The reasoning should be TRM navigation through Grammar Galaxy rules, not
Python string manipulation.

---

## The 4-Way Reading Strategy

This is NOT a GSM8K-specific technique. It's a universal reading comprehension
strategy that the TRM should apply to ANY input requiring multi-step reasoning.

### Pass 1: Forward Read (Perceive — Extract Entities)

Read the problem start-to-end. Extract:
- **Quantities** with their units and roles: "16 eggs", "$2 per egg", "3 months"
- **Entities** and their relationships: "Janet", "ducks", "farmers' market"
- **Actions/verbs** that imply operations: "eats" → subtract, "sells" → multiply by rate

This is the TRM's normal **Perceive** step in the game loop. The frame arrives
at the Tablet, the TRM perceives it, navigates to relevant Galaxy entries.

**On GPU**: Already partially working — the "Navigator fusion quantities"
trace shows quantities are extracted. But the entity-action binding is in Python.

### Pass 2: Backward Read (Navigate — Trace from Goal)

Read from the QUESTION backwards to the quantities:
- **Goal identification**: "How much money?" → goal = total_earnings (Number Galaxy)
- **Goal type**: monetary amount → needs quantity × price pattern
- **Dependency trace**: total_earnings ← remainder × price ← (initial - consumed) × price

This is the TRM's **Navigate** step — but navigating BACKWARDS from the goal
star to the input stars, using Grammar Galaxy transformation rules.

**On GPU**: This is where the breakdown happens. The current code does forward-only
navigation. It finds "sub" as the top operation and stops. The backward pass from
"how much money" → "price × quantity" pattern is done in Python string matching
(`_gsm8k_goal_typing`, `_gsm8k_role_value_map`, etc.) — 500+ lines of Python
doing what Grammar Galaxy rules should do.

### Pass 3: Operation Chain Construction (Reason — Build RPN Program)

With forward entities and backward goal trace, construct the computation:

```
Step 1: remainder = initial - eat - bake    → 16 3 - 4 -        = 9
Step 2: earnings = remainder × price        → 9 2 *              = 18
```

This is an RPN program composed from Grammar Galaxy rules:
- Rule "subtract_consumed": `PUSH initial PUSH consumed SUB` (pattern: "eats X for Y")
- Rule "multiply_by_rate": `PUSH quantity PUSH rate MUL` (pattern: "sells at $X per Y")
- Composition: chain Step 1 output as Step 2 input

**On GPU**: The `_gsm8k_left_fold_program` only applies ONE operation.
The chain construction should be TRM composing Grammar rules, not Python
selecting a strategy label.

### Pass 4: Normalization (Decide — Validate and Emit)

Sanity-check the answer:
- **Dimensional analysis**: "dollars" output from "eggs × dollars/egg" — consistent
- **Magnitude check**: 18 is reasonable for daily earnings from eggs
- **Ternary signal**: +1 if consistent, 0 if uncertain, -1 if contradictory

This is the TRM's **Decide** step — the Halting Gate checks convergence.

**On GPU**: The Halting Gate exists but doesn't do dimensional validation.
This should be Grammar Galaxy meta-rules that verify unit consistency.

---

## Spec Grounding

### Adaptive Reasoning Budget Spec §6.2 (Recursive Sub-Task Decomposition)

The spec ALREADY defines this pattern:

```
function ADAPTIVE_REASON(query, budget, depth):
    result = SWARM_REASON(query, min(budget, T_decomp))
    if result.σ == +1:
        return result  // Direct resolution succeeded

    // Budget exceeded → decompose
    subtasks = DECOMPOSE(query, result.partial_knowledge)
    for each subtask in subtasks:
        sub_result = ADAPTIVE_REASON(subtask, B_sub, depth + 1)
        PERSIST_AS_STAR(subtask, sub_result)

    composed = COMPOSE(sub_results, query)
    return composed
```

GSM8K "16 eggs, eat 3, bake 4, sell at $2, how much?" should decompose:
- Sub-task A: remainder = 16 - 3 - 4 = 9 (σ=+1, shallow)
- Sub-task B: earnings = 9 × 2 = 18 (σ=+1, shallow, depends on A)
- Compose: answer = 18

### Adaptive Reasoning Budget Spec §6.3 (Mathematical Decomposition)

> "Break a proof into lemmas. Each lemma is a sub-task with its own budget.
> Lemma dependencies form a DAG."

This is EXACTLY what GSM8K needs. Each arithmetic step is a "lemma."

### RPN Opcode Registry (Budget Control Recipes)

> Budget state stored in STORE/RECALL registers 60-67
> Sub-task decomposition: `OP_RECALL depth → OP_PUSH D_max → OP_CMP → OP_BRANCH`

The RPN primitives for sub-task decomposition ALREADY EXIST.

### Three Brain System §3.3 (Core Reasoning Operations)

> "RPN stack machine execution (stack-based VM)"
> "Recursive refinement (iterative convergence)"

The 4-way reading is recursive refinement — the TRM applies multiple passes
of the game loop to the same input.

---

## What Must Change

### Remove (Sovereignty Debt — ~600 lines of Python)

These 35+ Python methods in `knowledgeverse.py` do what Grammar Galaxy rules
and TRM navigation should do:

- `_gsm8k_left_fold_program` (line 5019) — builds RPN with ONE operation
- `_gsm8k_slot_role_names` (line 5039) — Python slot analysis
- `_gsm8k_quantity_role_candidates` (line 5223) — Python role binding
- `_gsm8k_role_text_overlap` (line 5375) — Python text matching
- `_gsm8k_template_program` (line 6030) — Python program generation
- `_gsm8k_decomposition_preview` (line 6404) — Python decomposition
- `_gsm8k_decomposition_result` (line 6598) — Python result extraction
- All the strategy dispatch: forward_chain, backward_chain, clause_chain,
  goal_adjusted_chain, hierarchical_sum, etc. — these are Grammar Galaxy
  transformation rules, not Python if/elif chains

### Add (Grammar Galaxy Rules + TRM Meta-Rules)

**Grammar Galaxy entries** for word problem decomposition patterns:

```
Star: "word_problem_multi_step_decomposition"
RPN Meta-Rule:
  PERCEIVE_QUANTITIES     → forward read, extract numbers + units
  PERCEIVE_GOAL           → backward read from question to goal type
  BUILD_DEPENDENCY_DAG    → trace which quantities feed which operations
  COMPOSE_OPERATION_CHAIN → chain operations as multi-step RPN program
  EVALUATE_CHAIN          → execute the composed RPN
  VALIDATE_UNITS          → dimensional analysis check
```

**Specific Grammar rules** for common word problem patterns:

```
Star: "pattern_consume_then_rate"
Pattern: [initial] - [consumed₁] - [consumed₂] ... × [rate]
RPN:    PUSH initial PUSH consumed₁ SUB PUSH consumed₂ SUB ... PUSH rate MUL
Example: "16 eggs - 3 eaten - 4 baked × $2/egg = $18"

Star: "pattern_unit_price_total"
Pattern: [quantity] × [unit_price]
RPN:    PUSH quantity PUSH unit_price MUL
Example: "5 phones × $150 = $750"

Star: "pattern_percentage_increment"
Pattern: [base] + [base] × [rate] × [periods]
RPN:    PUSH base PUSH base PUSH rate MUL PUSH periods MUL ADD
Example: "$750 + $750 × 0.02 × 3 = $795"
```

These go in Grammar Galaxy as procedural rules. The TRM navigates to them
during the backward-read pass when it identifies the goal type.

### Wire (TRM Game Loop)

The 4-way reading maps directly to the existing game loop:

```
TRM Tick 1 (Forward Read):
  Perceive → extract quantities, entities, actions from input
  Navigate → find matching quantity stars in Number Galaxy
  Store intermediate: quantity_list, entity_roles

TRM Tick 2 (Backward Read):
  Perceive → re-read the QUESTION part of the input
  Navigate → find goal type star in Grammar Galaxy
  Navigate → trace dependency from goal back to quantities
  Store intermediate: dependency_dag, operation_chain

TRM Tick 3 (Operation Chain):
  Reason → Nine-Chain Swarm processes sub-tasks in dependency order
  Worker 0: Step 1 of chain (e.g., subtract consumed)
  Worker 1: Step 2 of chain (e.g., multiply by rate) — waits for Worker 0
  Compose → chain sub-results

TRM Tick 4 (Normalization):
  Decide → Halting Gate checks:
    - Units consistent? (Grammar meta-rule)
    - Magnitude reasonable? (Reality Galaxy bounds)
    - All quantities consumed? (no orphan numbers)
  Act → emit answer or request more ticks
```

This is NOT new architecture — it's the EXISTING game loop running
MULTIPLE TICKS on the same input, which is exactly what the Adaptive
Reasoning Budget spec prescribes for σ=0 or σ=-1 knowledge.

---

## Universality: Not Just GSM8K

Daniel said: "was intended for math but it's useful everywhere a test or
question is made."

The 4-way reading applies to:

| Benchmark | Forward Read | Backward Read | Chain | Normalize |
|-----------|-------------|---------------|-------|-----------|
| **GSM8K** | Extract quantities + verbs | "How much?" → goal type | Multi-step arithmetic | Unit check |
| **MMLU** | Extract topic + options | "Which is correct?" → goal = select | Navigate Galaxy for each option | Confidence ranking |
| **LHE** | Extract claim + context | "Is this true?" → goal = verify | Multi-hop graph traversal | Contradiction check |
| **ARC-AGI-2** | Extract grid patterns | "What's the output?" → goal = transform | Grammar rule composition | Pattern validation |
| **ARC3** | Extract frame + goal frame | "Which action?" → goal = navigate | Spatial delta → action mapping | Position check |

The Grammar Galaxy rules for 4-way reading are UNIVERSAL meta-rules, not
benchmark-specific code paths. The same "perceive → trace goal → build chain →
validate" pattern handles all input types through the same TRM game loop.

---

## Execution Sequence

1. **E.38a**: Add Grammar Galaxy entries for word problem decomposition patterns
   (consume_then_rate, unit_price_total, percentage_increment, etc.)
2. **E.38b**: Wire TRM to run multiple ticks per input (forward + backward + chain)
   using the Adaptive Reasoning Budget's existing decomposition mechanism
3. **E.38c**: Replace `_gsm8k_left_fold_program` with Grammar-rule-composed
   multi-step RPN programs
4. **E.38d**: Add unit/dimensional validation as Grammar meta-rules
5. **E.38e**: Remove Python GSM8K strategy dispatch methods (the 35+ methods)
6. **E.38f**: Run GSM8K benchmark — target > 30% (same as MMLU baseline)

---

## Success Criteria

- [ ] 4-way reading is Grammar Galaxy meta-rules + TRM multi-tick, NOT Python
- [ ] Word problems decompose into sub-task DAGs (per ARB spec §6.2)
- [ ] Multi-step arithmetic chains produce correct results (16-3-4=9, 9×2=18)
- [ ] GSM8K score > 0/20 (target: match or exceed MMLU 30% baseline)
- [ ] Same 4-way reading pattern applied to MMLU, LHE (not GSM8K-specific)
- [ ] Python GSM8K methods reduced (target: remove strategy dispatch entirely)
- [ ] All reasoning on GPU — no Python string matching for goal/role analysis
