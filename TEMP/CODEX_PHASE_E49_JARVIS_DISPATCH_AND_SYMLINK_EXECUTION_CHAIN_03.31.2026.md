# Codex — Phase E.49: Jarvis Dispatch + Symlink Execution Chain

**Date:** 2026-03-31
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL — this is THE gap between "finds the word" and "solves the problem"

---

## Daniel's Exact Words

> "The TRM finding the word is correct. Now, Jarvis must dispatch the
> execution layer based on the decision (math or not! This is a single
> brain using reasoning chains that execute in RPN stacks). The TRM
> through Jarvis must then decide to execute by knowing through the
> symlink what to concatenate into a LoRA-like specialist that spawns
> a math core to do the work in a master-worker-worker/worker fashion
> and layered."

> "We do not duplicate things, we reference and include metadata. This
> is a meaning-centric brain with semantic gravity cohered by meaning
> as its clustering rule."

> "When I said inline I meant while the reasoning stars are still loaded
> — not a new process. This is an always-on live system."

---

## The Diagnosed Failure (GSM8K 0/20)

From the E.48 diagnostic run, here is what happens on every GSM8K problem:

```
Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes
muffins with 4. She sells the remainder at $2 per egg. How much does
she make?

WHAT HAPPENS NOW:
1. TRM finds meaning star: "en_two-way" (similarity=0.87)     ← CORRECT
2. Route includes: reasoning_strategies, game_mechanics, etc.  ← CORRECT
3. Navigator extracts quantities: 16, 3, 4, 2                 ← CORRECT
4. GSM8K worker_0 forward_chain → 9  (16-4-3=9)              ← PARTIAL (intermediate)
5. GSM8K worker_1 backward_chain → -7                         ← WRONG
6. Winning star: synset_00233925_a from Language               ← WRONG DISPATCH
7. Answer: 9                                                   ← WRONG (should be 18)

WHAT SHOULD HAPPEN:
1. TRM finds meaning star "sells remainder at price" → meaning understood
2. Jarvis reads symlinks on that star:
   - symlink → "subtraction" (operation type)
   - symlink → "multiplication" (operation type)
   - symlink → "multi_step_arithmetic" (meta-rule)
3. Jarvis dispatches math specialist with FULL chain:
   Step 1: 16 - 3 = 13 (eggs after breakfast)      STORE R0
   Step 2: R0 - 4 = 9  (eggs after muffins)        STORE R1
   Step 3: R1 × 2 = 18 (dollars at market)          STORE R2
4. Math core executes all 3 steps via RPN stack
5. Result: 18                                        ← CORRECT
```

The system correctly finds the word (step 1) and extracts quantities
(step 3). But it does ONE subtraction (16-4-3=9) and stops. It doesn't
continue to 9×2=18 because:

- **Jarvis doesn't follow the symlinks** to identify "this needs multi-step math"
- **No specialist dispatched** — the winning star comes from Language, not Math
- **No RPN chain constructed** — just a single fission operation
- **The reasoning_strategies stars exist** but aren't USED in the execution path

---

## What The Specs Say

### TRM_SPECIALIST_MATRYOSHKA_ARCHITECTURE.md §1-2

```
NavigatorSpecialist (router — this is Jarvis)
├── MathSpecialist (master)
│   ├── BasicMathSpecialist (arithmetic)
│   └── ...
```

> "Everything is a specialist, including routers. Specialists can spawn
> sub-specialists autonomously."

Jarvis IS Worker 8 in the Nine-Chain Swarm (AVATAR_EMBODIMENT_SPECIFICATION.md
line 507). It's the meta-specialist/coordinator. Its job is to look at what
the TRM found and decide which specialists to activate.

### HYPER_PARALLEL_PROCESSING.md §2 — Cross-Core Register Communication

```
Core 0 (arithmetic specialist):
  STORE_0 <- 3 x 3 = 9 (sessions per week)

Core 1 (rate specialist):
  RECALL_0 -> 9
  STORE_1 <- 9 x 60 = 540 (meters per week)
```

THIS IS THE EXACT PATTERN GSM8K NEEDS. Core 0 computes intermediate
result, STOREs it. Core 1 RECALLs it, continues. The multi-step chain
is built by cross-core register communication.

### FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md §1.4 — Layer 3 Rules

Rules are executable RPN programs. The `forward_entity_extraction` star
(reasoning_strategies.jsonl) says to extract entities. The
`operation_chain_construction` star says to build a multi-step RPN chain.
These are Layer 3 stars that Jarvis should navigate to and execute.

---

## The Implementation Gap

The composed head pipeline currently:

1. Embeds the query text → 16-dim vector
2. Searches Galaxy for nearest neighbors by embedding similarity
3. Picks the TOP-1 star with highest composed score
4. Returns that star's associated answer/program

**What's missing: the symlink follow-through.**

When the top star is found (meaning star), its symlinks should be
followed to find the EXECUTION stars. The execution stars contain
RPN programs. Those programs should be loaded into swarm worker stacks
and executed.

The flow should be:

```
Query → Embed → Galaxy search → Find meaning star (TOP-1)
                                    │
                                    ├── symlinks: ["subtraction", "multiplication", ...]
                                    │
                                    ▼
                            Jarvis reads symlinks
                                    │
                                    ├── "subtraction" → Grammar Galaxy rule star
                                    ├── "multiplication" → Grammar Galaxy rule star
                                    ├── "multi_step_arithmetic" → Meta-Rule star
                                    │
                                    ▼
                            Dispatch to Math Specialist
                                    │
                                    ├── Worker 0: forward extraction
                                    │   RPN: 16 3 SUB → STORE R0 (=13)
                                    │
                                    ├── Worker 1: RECALL R0, continue
                                    │   RPN: R0 4 SUB → STORE R1 (=9)
                                    │
                                    ├── Worker 2: RECALL R1, continue
                                    │   RPN: R1 2 MUL → STORE R2 (=18)
                                    │
                                    ▼
                            Halting Gate: R2 = 18
                                    │
                                    ▼
                            Answer: 18
```

---

## What Codex Must Implement

### Step 1: Symlink Follow-Through in Composed Head

After the composed head finds the top-1 meaning star, follow its symlinks
to discover execution-relevant stars.

**Where:** In `_select_composed_head_candidate` (or the method that
produces the final answer from the top candidate).

**Logic:**
```
top_star = composed_head_result
symlink_ids = top_star.get("symlinks", [])

# Look up each symlinked star in the Galaxy catalog
execution_stars = []
for symlink_id in symlink_ids:
    linked_star = galaxy_catalog.get(symlink_id)
    if linked_star is None:
        continue
    # Is this star a Rule (Layer 3) or Meta-Rule (Layer 4)?
    layer = linked_star.get("layer", 2)
    if layer >= 3:
        execution_stars.append(linked_star)

# If execution stars found → dispatch to specialist
if execution_stars:
    # Jarvis dispatch: determine task type from execution stars
    task_types = set(star.get("category", "") for star in execution_stars)
    if any("math" in t or "arithmetic" in t for t in task_types):
        # Dispatch to math specialist
        result = jarvis_dispatch_math(quantities, execution_stars)
```

**CRITICAL:** This is NOT new Python orchestration. This is Jarvis
(Worker 8) reading Galaxy metadata (symlinks) to decide which
specialists to activate. The metadata IS in the Galaxy. The dispatch
IS internal. The execution IS on RPN stacks.

### Step 2: RPN Chain Construction from Quantities + Operations

The GSM8K worker already extracts quantities (16, 3, 4, 2) and identifies
operation patterns. What it doesn't do is build the FULL chain.

**Current:** Single fission → 16-4-3=9 (stops)
**Needed:** Multi-step chain → (16-3-4)×2=18

The quantities are bound to semantic roles:
- 16 = initial (total eggs)
- 3 = part (breakfast)
- 4 = part (muffins)
- 2 = rate (price per egg)

The operation chain should be constructed from the GOAL backward:
1. Goal: total earnings (asked question)
2. Earnings = remainder × rate
3. Remainder = initial - part_1 - part_2
4. Therefore: (16 - 3 - 4) × 2 = 18

This is EXACTLY what the `backward_goal_tracing` reasoning strategy star
describes. And `operation_chain_construction` says to build the multi-step
RPN chain. These stars EXIST but are not being FOLLOWED.

**Implementation:**
When Jarvis identifies "multi-step arithmetic" from the symlinks:

```
# Build RPN chain from semantic slot bindings
chain = []
# Step 1: Subtract parts from initial
chain.append(f"{initial} {part_1} SUB STORE R0")
# Step 2: Subtract more parts
chain.append(f"RECALL R0 {part_2} SUB STORE R1")
# Step 3: Multiply by rate
chain.append(f"RECALL R1 {rate} MUL STORE R2")
# Execute chain
result = rpn_execute(chain)
```

### Step 3: Jarvis as Worker 8 Coordination

Per AVATAR_EMBODIMENT_SPECIFICATION.md line 507: Worker 8 is the
meta specialist (Jarvis coordinator).

Jarvis's role in the Nine-Chain Swarm:
1. **Observe** what the other 8 workers found (meaning stars, patterns)
2. **Follow symlinks** from found stars to identify execution requirements
3. **Dispatch** appropriate specialist(s) based on symlink metadata
4. **Track** all worker results as they come in
5. **Report** to TRM for final decision via Halting Gate

Currently, the swarm workers independently search the Galaxy and return
candidates. Jarvis (Worker 8) should be the one that COORDINATES:

- Worker 0 (math): "I found quantities 16, 3, 4, 2"
- Worker 1 (grammar): "I found operation pattern: subtract-then-multiply"
- Jarvis (Worker 8): "Math + Grammar say multi-step arithmetic. Building
  chain: (16-3-4)×2. Dispatching to math core for execution."

### Step 4: Make Reasoning Strategy Stars Part of the Execution Path

The 5 reasoning strategy stars exist in the Galaxy and are boosted in
scoring. But scoring boost ≠ execution. The stars contain RPN programs:

```
forward_entity_extraction:
  rpn: "TOKENS FOREACH IF NOUN THEN GALAXY_LOOKUP IF NUMBER THEN NUM_BIND"

backward_goal_tracing:
  rpn: "QUESTION_ENTITY GOAL_BIND DEPS REVERSE_TRACE"

operation_chain_construction:
  rpn: "DEPS TOPO_SORT FOREACH OPERATION_BIND STORE CHAIN"
```

These RPN programs should be EXECUTED when the star is found, not just
scored. When the composed head finds `operation_chain_construction` as a
high-scoring candidate, its RPN program should be loaded into a worker
stack and executed with the current problem's data.

**This is the key architectural bridge:** Stars are not just "answers to
return." Layer 3-4 stars are PROGRAMS to EXECUTE. Finding them means
RUNNING them.

---

## What NOT To Do

- Do NOT add more Python orchestration methods for GSM8K
- Do NOT add new `_gsm8k_*` Python methods
- Do NOT build a separate "chain builder" in Python
- The chain construction IS the RPN program in the star
- The dispatch IS Jarvis reading symlinks
- The execution IS workers running RPN stacks
- Everything stays INSIDE the Galaxy, INSIDE the pipeline

---

## Verification: GSM8K First 5

After implementing symlink follow-through and Jarvis dispatch, re-run
the first 5 GSM8K problems. Expected improvement:

| # | Question | Current | Expected |
|---|----------|---------|----------|
| 0 | Janet's ducks (16-3-4)×2 | 9 (single step) | 18 (full chain) |
| 1-4 | Other word problems | 0/4 | >0 (any improvement) |

Even 1/5 correct would prove the symlink execution path works.

---

## Grounding: Exact Spec References

| Spec | Section | What It Says |
|------|---------|-------------|
| TRM_SPECIALIST_MATRYOSHKA_ARCHITECTURE.md | §1 | Specialists are fractal, router IS a specialist (Jarvis) |
| AVATAR_EMBODIMENT_SPECIFICATION.md | §7.3, line 507 | Worker 8 = meta specialist (Jarvis coordinator) |
| HYPER_PARALLEL_PROCESSING.md | §2 | Cross-core STORE/RECALL for multi-step chains |
| HYPER_PARALLEL_PROCESSING.md | §4 | RPN stacks are native parallelization substrate |
| FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md | §1.4 | Layer 3 = executable RPN transformation rules |
| FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md | §1.5 | Layer 4 = meta-rules (WHEN to apply strategies) |

---

## Success Criteria

- [ ] Symlink follow-through: when top-1 star found, its symlinks are resolved
- [ ] Layer 3-4 stars discovered via symlinks are flagged as "execution stars"
- [ ] Jarvis (Worker 8) reads execution star metadata to determine dispatch
- [ ] Math specialist dispatched for multi-step arithmetic problems
- [ ] RPN chain built from quantities + operation types (not single fission)
- [ ] Cross-core STORE/RECALL used for intermediate results
- [ ] GSM8K score improves from 0/20 (any > 0 proves the path)
- [ ] reasoning_strategies stars are EXECUTED, not just scored
- [ ] No new Python `_gsm8k_*` methods (Jarvis dispatches, not Python)
- [ ] All execution stays inside Galaxy + swarm pipeline

---

## CRITICAL NAMING CORRECTION (Addendum)

**ZERO benchmark names in knowledge code.** Daniel: "Why are we naming
gsm8k? This is math problems, math questions. We are not a benchmark
solver tool! One Knowledgeverse, no duplicate information."

Any method, star, variable, or galaxy named `gsm8k` must be renamed to
its MEANING:

| Wrong (benchmark-named) | Right (meaning-named) |
|--------------------------|----------------------|
| `_gsm8k_reasoning_strategy_rows` | `_math_reasoning_strategy_rows` |
| `_gsm8k_word_problem_context` | `_math_word_problem_context` |
| `_gsm8k_template_program` | `_math_operation_chain_program` |
| `_gsm8k_quantity_role_candidates` | `_math_quantity_role_candidates` |
| `GPU_GSM8K_TARGET_GALAXIES` | `GPU_MATH_WORD_PROBLEM_TARGET_GALAXIES` (or just use the same as MATH) |
| `gsm8k_worker_0` | `math_worker_0` |
| `proceduralized_gsm8k_train_10` | absorbed into Math Galaxy (no benchmark prefix) |

The SAME math reasoning that solves GSM8K must solve IMO, MMLU math,
physics problems, any word problem. If it's named `gsm8k`, the TRM
won't navigate to it for IMO. If it's named `math`, it works everywhere.

Benchmark WRAPPERS (thin I/O adapters like `benchmarks/gsm8k.py`) can
mention the benchmark — they're I/O. Knowledge CANNOT.

---

## The Single Lesson

Daniel said it clearly: **we do not duplicate things, we reference and
include metadata.**

The meaning star "sells remainder at price" doesn't need to CONTAIN
the arithmetic. It symlinks to "subtraction" and "multiplication" stars
in Grammar Galaxy. Those stars contain the RPN programs. Jarvis reads
the symlinks and dispatches. The math core executes. One brain, many
workers, all live, all internal.

The current pipeline finds the meaning (correct) but doesn't follow the
references to the execution layer. Fix that one gap and the whole
architecture comes alive.
