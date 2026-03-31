# Codex — Phase E.50: Knowledge Migration + Dead Code Purge

**Date:** 2026-03-31
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL — we've been trying to do this for 6 months

---

## Daniel's Exact Words

> "The idea on the strategy should be not delete — migrate to symlinked
> meaning stars, metadata and RPN instructions, all symlinked (if it uses
> math — symlink to it). Delete only fallbacks and old orchestration that
> has been migrated."

> "Why are we naming gsm8k? This is math problems, math questions. We are
> not a benchmark solver tool! We are the next generation of AI, a new
> paradigm that works. One Knowledgeverse, no duplicate information. Why
> develop math solving strategy for gsm8k and then take a 0 score at IMO
> because of naming?"

> "We've proved but always instead of advancing we craft more and more
> runners and python!"

> "Take out anything that's not a live game system. Fallbacks and old
> wrong attempts cloud the path."

> "The boot is expected to be slow, do not run isolated tests! Without
> all layers the symlink is broken!"

---

## The Strategy: MIGRATE Then Delete

**DO NOT just delete.** The 41 `_gsm8k_*` methods contain real
mathematical knowledge: quantity extraction, operation chain patterns,
semantic role binding, multi-step decomposition. That knowledge must be
MIGRATED to Galaxy stars with symlinks and RPN programs. Then — and only
then — delete the Python shell that held it.

**What gets migrated:** The KNOWLEDGE inside Python methods → Galaxy
stars (Layer 2 meaning, Layer 3 rules, Layer 4 meta-rules), with
symlinks to Math/Grammar/Number Galaxy, and RPN programs expressing
the computation.

**What gets deleted immediately:** Fallback paths, benchmark-specific
naming, old orchestration where the Galaxy path already works, dead
variables, commented-out code.

---

## Part A: Rename — Zero Benchmark Names in Knowledge

**This is why IMO scores 0.** The math knowledge is locked behind
`_gsm8k_*` names. The TRM navigates to "math" concepts, not "gsm8k"
concepts. Same knowledge, different benchmark, zero score — because
of naming.

### In `knowledgeverse.py`:

| Current name | Renamed to | Why |
|-------------|-----------|-----|
| `GPU_GSM8K_TARGET_GALAXIES` | `GPU_WORD_PROBLEM_TARGET_GALAXIES` | Word problems are universal |
| `_gsm8k_reasoning_strategy_rows` | `_math_reasoning_strategy_rows` | Math reasoning, not GSM8K reasoning |
| `_gsm8k_strategy_weight` | `_math_strategy_weight` | Universal |
| `_gsm8k_halting_thresholds` | `_math_halting_thresholds` | Universal |
| `_is_reasoning_strategy_entry` | Keep (already generic) | Already correctly named |
| `gsm8k_worker_N` | `math_worker_N` | Universal workers |
| Any star named `gsm8k_*` | `math_*` or `word_problem_*` | Universal |

### In benchmark wrappers:

Benchmark wrappers (`benchmarks/gsm8k.py`, `benchmarks/imo_bench.py`,
etc.) CAN mention the benchmark name — they're I/O adapters. But they
must route through the SAME universal path:

```python
# benchmarks/gsm8k.py — thin I/O wrapper
task = {"type": "MATH_TASK", "query": question_text}  # MATH_TASK, not GSM8K_TASK
result = kv.execute_task(task=task, specialist="auto")

# benchmarks/imo_bench.py — thin I/O wrapper
task = {"type": "MATH_TASK", "query": problem_text}  # SAME MATH_TASK
result = kv.execute_task(task=task, specialist="auto")
```

Same type, same path, same Galaxy, same reasoning. The only difference
is the I/O format of the input/output.

### In House JSONL:

Any star with `gsm8k` in its name, category, or tags must be renamed:
- `gsm8k_forward_chain` → `math_forward_entity_extraction`
- `gsm8k_backward_chain` → `math_backward_goal_tracing`
- Category: `gsm8k_strategy` → `math_strategy`
- Tags: `["gsm8k"]` → `["math", "word_problem", "arithmetic"]`

---

## Part B: Migrate `_gsm8k_*` Knowledge to Galaxy Stars

For each of the 41 `_gsm8k_*` methods, extract the KNOWLEDGE and
express it as a Galaxy star. Group by what they actually do:

### Group 1: Quantity Extraction (→ Layer 3 Rule Stars)

These methods extract numbers and their semantic roles from text:

| Python method | Knowledge to extract | Star name |
|--------------|---------------------|-----------|
| `_gsm8k_quantity_role_candidates` | How to find numbers in text + their roles (initial, part, rate) | `math_quantity_role_extraction` |
| `_gsm8k_quantity_role_rows` | Semantic role patterns (price×quantity, total−parts) | `math_semantic_role_patterns` |
| `_gsm8k_slot_role_names` | Slot taxonomy: initial, part, rate, total, count | `math_slot_taxonomy` |
| `_gsm8k_quantity_snippet` | Context window around a number | `math_quantity_context_window` |
| `_gsm8k_text_tokens` | Tokenization for math text | (merge into existing Word Galaxy) |

**Star format:**
```jsonl
{"name": "math_quantity_role_extraction", "galaxy": "Grammar", "layer": 3,
 "meaning": "Extract numeric quantities from text and classify their semantic roles: initial value, part to subtract, rate to multiply, total to compute",
 "rpn_program": "TEXT TOKENIZE FOREACH IF NUMBER THEN CONTEXT_WINDOW ROLE_CLASSIFY STORE",
 "symlinks": ["math_semantic_role_patterns", "math_slot_taxonomy", "number_galaxy_entry"],
 "category": "math_reasoning", "tags": ["math", "word_problem", "quantity_extraction"]}
```

### Group 2: Operation Chain Construction (→ Layer 3-4 Stars)

These methods build multi-step arithmetic from extracted quantities:

| Python method | Knowledge | Star name |
|--------------|-----------|-----------|
| `_gsm8k_template_program` | How to chain operations (sub, mul, add) into sequence | `math_operation_chain_builder` |
| `_gsm8k_template_slot_bindings` | How to bind quantities to operation slots | `math_slot_binding` |
| `_gsm8k_left_fold_program` | Left-fold arithmetic pattern (a op b op c) | `math_left_fold_pattern` |
| `_gsm8k_operator_token` | Map operation names to RPN operators | (merge into Grammar Galaxy operation rules) |
| `_gsm8k_product_tokens` | Multiplication chain pattern | `math_multiplication_chain` |
| `_gsm8k_sum_token_rows` | Addition chain pattern | `math_addition_chain` |
| `_gsm8k_decomposition_preview` | Preview of multi-step decomposition | `math_decomposition_strategy` |
| `_gsm8k_decomposition_result` | Execute multi-step decomposition | `math_decomposition_execution` |

### Group 3: Pattern Matching + Scoring (→ Layer 4 Meta-Rules)

These methods decide WHICH operation pattern fits:

| Python method | Knowledge | Star name |
|--------------|-----------|-----------|
| `_gsm8k_execution_pattern_score` | Score how well a pattern fits the problem | `meta_math_pattern_fitness` |
| `_gsm8k_pattern_structural_score` | Structural cues (keywords → operations) | `meta_math_structural_cues` |
| `_gsm8k_operation_role_match_score` | How well roles match operations | `meta_math_role_operation_match` |
| `_gsm8k_operation_disambiguation_bonus` | Disambiguate between similar patterns | `meta_math_pattern_disambiguation` |
| `_gsm8k_semantic_slot_score` | Score semantic fit of slot binding | `meta_math_semantic_slot_fitness` |

### Group 4: Context + Execution (→ Merge into E.49 Jarvis path)

| Python method | Knowledge | Destination |
|--------------|-----------|------------|
| `_gsm8k_word_problem_context` (425 lines!) | Full word problem analysis | Split into 5-8 stars above |
| `_gsm8k_execution_context` | Execution setup | Part of Jarvis dispatch (E.49) |
| `_gsm8k_execution_trace` | Trace recording | Part of sleep-time learning |
| `_gsm8k_numeric_entry_value` | Number parsing from Galaxy entry | Merge into Number Galaxy |

### Group 5: Consensus/Override (→ Delete after E.49)

| Python method | Action |
|--------------|--------|
| `_gsm8k_consensus_record` | Delete: Python voting, should be Halting Gate |
| `_gsm8k_structural_override_priority` | Delete: Python override, should be defeasible logic |
| `_gsm8k_structural_override_record` | Delete: Python override, should be defeasible logic |
| `_gsm8k_preview_candidate_id` | Delete: diagnostic formatting |

---

## Part C: Migrate ARC3 Perception to Galaxy Stars

Same principle for `arc_agi_3.py` — the perception functions contain
knowledge about game mechanics that should be Galaxy stars:

| Python function | Knowledge | Star name |
|----------------|-----------|-----------|
| `_movement_budget_snapshot` | How to read a movement bar from status area | `perception_movement_budget_reading` |
| `_lives_remaining` | How to count lives from red indicators | `perception_lives_indicator_reading` |
| `_frame_state` | How to classify frame as gameplay/transition | `perception_frame_state_classification` |
| `_flash_semantics` | What color flashes mean (failure/success) | `perception_flash_feedback_interpretation` |
| `_should_force_reset` | When to strategically reset | `meta_strategic_reset_decision` |
| `_select_mechanic_target` | How to choose which game target to approach | `meta_game_target_selection` |
| `_exploration_order` | How to explore when path is unknown | `meta_exploration_strategy` |

After migration: slim `arc_agi_3.py` to I/O adapter + spatial bridge.

---

## Part D: Fix Inline Consolidation (Briefs Lost on Restart)

`briefs_consolidated=0` because `pending_recent_briefs=0` — the briefs
were lost when the system unloaded between the query phase and the sleep
phase. In an always-on live system, this never happens: the briefs are
still in VRAM when consolidation runs inline.

**The bug:** The benchmark runner (or test script) boots the system,
runs queries (which create briefs), then SHUTS DOWN the Knowledgeverse.
When sleep consolidation runs on a fresh instance, the briefs are gone.

**The fix:** Consolidation must run BEFORE the system unloads. The
always-on pattern is:

```
Boot → Load all knowledge → Run queries → Briefs accumulate in memory
→ Idle detected (or explicit consolidation call) → Sleep consolidation
runs WITH briefs still in memory → Briefs processed → Galaxy strengthened
→ Checkpoint saved → System continues (or shuts down cleanly)
```

**Implementation:**
1. In `run_full_benchmark.py` and `run_arc3_agent.py`: call
   `kv.jarvis_sleep_consolidation()` BEFORE `kv` is garbage collected
   or the script exits.
2. The sleep call must happen on the SAME Knowledgeverse instance that
   accumulated the briefs — not a new instance.
3. Verify: after the call, `briefs_consolidated > 0` in the summary.
4. Only THEN save checkpoint and exit.

**Why this matters:** Without consolidation, the system never learns.
It answers queries, forgets the traces, starts fresh next time. The
whole point of the always-on paradigm is that experience accumulates.

### Success Criteria (Consolidation)

- [ ] Sleep consolidation called BEFORE system unloads
- [ ] `briefs_consolidated > 0` after a benchmark run
- [ ] Same KV instance that ran queries also runs consolidation
- [ ] Checkpoint saved AFTER consolidation (not before)

---

## Part E: Delete (Only What's Been Migrated or Is Dead)

**Delete immediately** (no knowledge to migrate):
1. All `fallback` paths — no fallbacks, we fail and fix
2. `ptx_fallback_rate` metric — there is no fallback rate
3. `_fallback_gpu_buffer_signature_base` — fix GPU buffers, don't fall back
4. All `"status": "fallback"` route entries
5. `foundational_operations_bootstrap.py` — if knowledge is in House JSONL
6. Commented-out code blocks
7. Unused imports and variables

**Delete after migration** (Part B/C must happen first):
1. Each `_gsm8k_*` method → delete AFTER its knowledge is a Galaxy star
2. Each ARC3 perception function → delete AFTER its knowledge is a star
3. `_exploration_fallback` → delete (it's a fallback)

---

## Part E: Verify Cross-Benchmark Math Works

After renaming and migration, verify that the SAME math path serves
ALL math benchmarks:

```bash
# All of these must route through the SAME Galaxy path:
python scripts/run_full_benchmark.py --gsm8k-count 5
python scripts/run_full_benchmark.py --imo-count 5
python scripts/run_full_benchmark.py --math-count 5
```

Check the logs: all three should show the same route galaxies, same
reasoning_strategies hits, same Jarvis dispatch. The only difference
is the I/O format. If GSM8K routes differently than IMO, the naming
fix isn't complete.

---

## Testing: FULL BOOT ONLY

**NEVER run isolated tests.** The system is hyper-modular but
inter-dependent. Without all layers loaded, symlinks are dangling
pointers. The boot IS slow because it loads ALL knowledge — that's
correct behavior, not a bug to optimize around.

1. Full boot with ALL knowledge
2. Full test suite
3. GSM8K 20, IMO 5, Math 5 — verify same path, same reasoning
4. ARC3 diagnostic — verify game knowledge stars reachable

---

## Success Criteria

- [ ] Zero benchmark names in knowledge code (no `gsm8k` in method/star/variable names)
- [ ] `GPU_GSM8K_TARGET_GALAXIES` → `GPU_WORD_PROBLEM_TARGET_GALAXIES` (or unified)
- [ ] All `gsm8k_worker_N` → `math_worker_N`
- [ ] 15+ new meaning-named math Galaxy stars from migrated Python knowledge
- [ ] Each migrated star has symlinks to Math/Grammar/Number galaxies
- [ ] Each migrated star has RPN program expressing the computation
- [ ] Python methods deleted AFTER their knowledge is migrated
- [ ] All fallback paths deleted
- [ ] `foundational_operations_bootstrap.py` deleted (if knowledge in House)
- [ ] GSM8K and IMO route through the SAME math reasoning path
- [ ] Full boot test suite passes (no isolated tests)
- [ ] `knowledgeverse.py` reduced by 2,000+ lines (migrated code removed)

---

## The Six-Month Pattern We Must Break

This migration has been on the table since October 2025. Every time:
1. We identify Python orchestration that should be Galaxy knowledge
2. Instead of migrating, we add MORE Python to make the benchmark work
3. The new Python works for that ONE benchmark but not others
4. Benchmark-naming locks the knowledge behind one test surface
5. Other benchmarks score 0 on the same underlying capability

**This time:** Migrate the knowledge. Delete the Python. Name by meaning.
One brain, one Galaxy, all benchmarks. No more Python orchestration
growth. The architecture works — we proved it. Now trust it.
