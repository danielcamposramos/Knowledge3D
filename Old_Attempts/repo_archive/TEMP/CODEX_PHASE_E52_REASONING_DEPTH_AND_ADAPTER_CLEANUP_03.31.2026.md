# Codex — Phase E.52: Reasoning Depth via Galaxy Stars + Adapter Cleanup

**Date:** 2026-03-31
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** HIGH — E.51 lifecycle works; now deepen reasoning quality

---

## Context

E.51 is working: warm boot, clean shutdown, idle consolidation, GPU sleep
kernels wired, universal path live. The answer=2.125 on the validation
prompt tells us the PLUMBING is correct but the Galaxy doesn't yet contain
enough reasoning knowledge to handle diverse math. The symlink execution
chain routes to the math specialist correctly — but once there, the
specialist has limited material to work with.

There are two structural issues remaining:

1. **`_infer_query_mode` + `_select_gpu_profile` IS task-type routing by
   another name.** It returns `"MATH_TASK"` / `"ARC_TASK"` / `"LHE_TASK"`
   and then branches to select reasoning programs. The Galaxy's symlinks
   should do this — meaning stars say "this involves math" through their
   grammar_refs, and Jarvis dispatches accordingly. The current code
   infers the mode in Python and picks the reasoning program in Python.

2. **36 `_gsm8k_*` methods (~2,500 lines) contain the actual math
   reasoning logic** — operation chain construction, template binding,
   role-to-slot mapping, decomposition. This IS knowledge. It needs to
   become Galaxy stars with RPN programs so the symlink chain can use it.

---

## Part A: Migrate Math Reasoning Python → Galaxy Stars

### What To Migrate

The `_gsm8k_*` methods contain three kinds of knowledge:

**1. Operation chain construction** (`_gsm8k_left_fold_program`,
`_gsm8k_template_program`, `_gsm8k_decomposition_result`):
These build RPN programs from parsed word problems. The LOGIC of "subtraction
followed by multiplication" should be an RPN chain template stored as a
Grammar Galaxy star with symlinks to math_operations.

**Target**: Create Grammar Galaxy stars for common operation chain patterns:
- `operation_chain_left_fold` — sequential L→R evaluation
- `operation_chain_nested` — operations with grouping
- `operation_chain_ratio` — "X times as many" pattern
- `operation_chain_difference_then_multiply` — subtract then scale

Each star has `grammar_refs` → [math_operations] so Jarvis knows to
dispatch the math specialist when these are found.

**2. Slot-role disambiguation** (`_gsm8k_quantity_role_candidates`,
`_gsm8k_semantic_slot_score`, `_gsm8k_role_text_overlap`):
These figure out which number plays which role (initial count, amount lost,
multiplier). This is ENTITY EXTRACTION — it should use the
`forward_entity_extraction` reasoning strategy star that already exists.

**Target**: Extend the existing `forward_entity_extraction` star with
additional symlinks to quantity-specific extraction patterns. Create new
stars:
- `quantity_role_initial` — "had N", "started with N"
- `quantity_role_delta` — "lost N", "gave N", "bought N"
- `quantity_role_multiplier` — "N times", "double", "triple"

**3. Template matching** (`_gsm8k_template_slot_bindings`,
`_gsm8k_pattern_structural_score`):
These match word problems to known templates. Templates ARE Grammar Galaxy
stars — they should live there, not in Python methods.

**Target**: Migrate the 20 existing math templates (already in Galaxy from
E.38) to carry their slot-binding logic as RPN programs in the star's
`meaning_rpn` field, rather than in Python.

### How To Migrate (Pattern)

For each `_gsm8k_*` method being migrated:

1. **Extract the knowledge** — what rule/pattern does this method encode?
2. **Create a Galaxy star** in House JSONL with:
   - `meaning_rpn`: The pattern as an RPN program
   - `grammar_refs`: Symlinks to relevant domain stars
   - `meta_refs`: Symlinks to reasoning strategy stars
3. **Wire into symlink chain** — ensure the star is findable by Galaxy
   search when a relevant query arrives
4. **Remove the Python method** ONLY after the star is live and tested

Do NOT remove all 36 methods at once. Migrate in batches:
- Batch 1: Operation chain patterns (5-8 methods → 4-6 stars)
- Batch 2: Entity extraction patterns (5-8 methods → 3-5 stars)
- Batch 3: Template binding (3-5 methods → extend existing template stars)

After each batch: run the live validation (not isolated tests) to verify
the symlink chain still produces correct answers.

---

## Part B: Clean Benchmark Adapters (Stop Passing task["type"])

### Current State

Benchmark adapters pass explicit type hints:
```python
# benchmarks/mmlu.py:258
task_result = self.kv.execute_task(task={"type": "MMLU_TASK", ...})

# benchmarks/last_humanity_exam.py:285
task_result = self.kv.execute_task(task={"type": task_type, ...})

# benchmarks/arc_agi_3.py:1260
{"type": "ARC_TASK", ...}
```

Since `_infer_query_mode` now infers from content, these type hints are
mostly ignored — but they're architectural debt. A live system doesn't
know what benchmark is asking the question.

### Target

Each benchmark adapter becomes a pure I/O wrapper:
- Convert external format → `{"query": ..., "options": [...]}` (universal)
- Call `kv.execute_task(task=...)` with NO type hint
- Convert universal result → external format

### Changes

**mmlu.py**: Remove `"type": "MMLU_TASK"`. The query text + options list
is enough for `_infer_query_mode` to figure out it's a multiple-choice
question. The subject/domain goes in `route={"specialist": ...}` if
needed, NOT in task type.

**last_humanity_exam.py**: Remove `"type": task_type`. Same — the
elimination-style question + options list is sufficient.

**math_competitions.py**: Remove `"type": "MATH_TASK"`. The math content
in the query text triggers math inference.

**arc_agi_3.py**: Remove `"type": "ARC_TASK"`. The grid payload structure
triggers ARC inference via `_looks_like_arc_payload`.

**arc_agi_2_adapter.py**: Same — remove `"type": "ARC_TASK"`.

### Verification

After removing type hints, run the full benchmark to verify that
`_infer_query_mode` correctly infers the mode from content alone.
Expected: identical routing (same query_mode in results) because the
inference logic already exists.

---

## Part C: Reduce Python Branching in Scoring/Halting

### Current State

76 `if task_type ==` branches remain in knowledgeverse.py. Key offenders:

- `_halting_gate_converged` (line 13389): Different thresholds per task type
- `_select_gpu_profile` (line 8116): Different reasoning programs per mode
- Scoring functions: Different alpha/weights per task type

### Target (Incremental)

These thresholds and weights should be **meta-rule stars** (Layer 4) in
the Galaxy. When the TRM finds a math problem, the meta-rule star says
"use gap_threshold=0.04" — not a Python constant.

This is a LARGER migration that will take multiple phases. For E.52,
start with the most impactful:

**Halting gate thresholds** → Create 4 meta-rule stars:
- `halting_threshold_elimination` (for choice-based: LHE, MMLU)
- `halting_threshold_math` (for numeric answers)
- `halting_threshold_spatial` (for grid/visual tasks)
- `halting_threshold_default` (general queries)

Each star's `meaning_rpn` contains the threshold values. The halting gate
reads the star's values instead of branching on task_type.

---

## Part D: Naming Cleanup (Remaining _gsm8k_ References)

### Current State

141 `_gsm8k_` references remain in knowledgeverse.py. Many are method
names, some are field references. The E.50 rename was partial.

### Target

Rename ALL remaining `_gsm8k_*` → `_math_*` (or more specific meaning
names). This is mechanical — just rename. The methods themselves get
migrated to Galaxy stars in Part A over time.

Naming convention per FOUNDATIONAL_KNOWLEDGE_SPECIFICATION §0:
- `_gsm8k_left_fold_program` → `_math_left_fold_program`
- `_gsm8k_template_program` → `_math_template_program`
- `_gsm8k_quantity_role_candidates` → `_math_quantity_role_candidates`
- `_gsm8k_slot_value` → `_math_slot_value`
- etc.

Mechanical rename, full boot test after.

---

## Execution Order

| Part | Priority | Risk |
|------|----------|------|
| D: Rename _gsm8k_ → _math_ | FIRST | Low — mechanical rename |
| B: Clean benchmark adapters | SECOND | Low — type hints already ignored |
| A: Migrate math reasoning → stars | THIRD | Medium — requires careful testing |
| C: Meta-rule stars for thresholds | FOURTH | Medium — incremental |

Start with D (quick win, reduces confusion) then B (removes dead hints),
then A (deepens reasoning quality), then C (incremental sovereignty push).

---

## Success Criteria

- [ ] Zero `_gsm8k_*` method names in knowledgeverse.py (all `_math_*`)
- [ ] Zero `"type": "..."_TASK"` in benchmark adapters (pure content inference)
- [ ] 4+ new operation chain pattern stars in Grammar Galaxy
- [ ] 3+ new quantity role stars in Grammar Galaxy
- [ ] Symlink chain produces correct answer for "Janet's ducks" (answer=18)
- [ ] Full benchmark runs with NO type hints, same or better accuracy
- [ ] `_infer_query_mode` correctly routes all suites from content alone
- [ ] 4 halting threshold meta-rule stars created
- [ ] `if task_type ==` count reduced from 76 to <40

---

## The Bigger Picture

Daniel's vision: "If we clear all this nonsense following the
proceduralization with proper symlink and dedup logic the architecture
defines, we'll end up with a system that can solve any task we aim at
its live interface."

E.51 cleared the lifecycle nonsense. E.52 clears the reasoning-routing
nonsense. Each `_gsm8k_*` method migrated to a Galaxy star is one less
line of Python deciding HOW to solve — and one more piece of knowledge
the TRM can navigate to, compose with, and use for ANY input, not just
GSM8K.

The math accuracy problem (answer=2.125) is NOT a pipeline problem — the
pipeline works. It's a KNOWLEDGE problem: the Galaxy needs richer
reasoning stars so the symlink chain has more material to compose from.
That's what Part A delivers.
