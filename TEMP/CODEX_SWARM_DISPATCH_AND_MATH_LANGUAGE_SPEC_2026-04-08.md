# Codex Direction: Swarm Worker Wiring — Math and Language Benchmarks

**Date:** 2026-04-08
**Authority:** docs/vocabulary/MATH_CORE_SPECIFICATION.md §2, docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md,
              CODEX.md § "GRE Specialist Kernels" (top priority)
**Depends on:** CAS/SAS tiered wiring complete ✅, WINE/execute_task migration complete ✅
**Goal:** Wire all 9 GRE specialists permanently into the swarm so math and language benchmarks
         score non-trivially through the sovereign pipeline.

---

## Architectural Principle: All Nine Always Run

The nine-chain swarm is the internal cognitive swarm of ONE sovereign AI (the superdotados
model: gifted individuals think through multiple parallel cognitive channels simultaneously).

**All nine workers ALWAYS run on EVERY problem — no exceptions.**

Why: Galaxy knowledge is connected by symlinks across all domains. A geometry worker on an
MMLU history question might traverse a spatial analogy symlink that leads to the right answer.
A resonance worker on an ARC grid might find a structural pattern from the Grammar Galaxy that
no spatial worker found. Excluding workers by surface kind kills these cross-domain discoveries.

**Surface kind does NOT select workers. It does NOT exclude workers.**

Surface kind informs:

1. **Halting gate scoring weights** — `gre_multimodal_halting_gate` weighs each worker's
   confidence contribution; surface kind adjusts these weights so the most relevant workers
   count more toward convergence
2. **Domain hint in route** — helps workers start their LED-A* navigation from the right
   Galaxy neighborhood, but they are FREE to traverse all galaxies via symlinks
3. **Result extraction** — how Python reads the result from the halting gate winner
   (grid for GAME_2D, numeric for MATH, letter for QUESTION)

---

## Fixed Worker Identity: One Permanent Assignment Per Slot

Each of the 9 swarm workers has a FIXED GRE kernel identity that NEVER changes between tasks.
These are the worker identities from CODEX.md's GRE table, permanently assigned:

```text
Worker 0 → gre_atomic_fission_fusion   # ALWAYS: decomposes ANY problem into sub-goals first
Worker 1 → gre_resonance_field         # ALWAYS: broad semantic field across all galaxies
Worker 2 → gre_vector_resonator        # ALWAYS: embedding similarity search
Worker 3 → gre_arc_reasoner            # ALWAYS: structural/spatial pattern recognition
Worker 4 → gre_geometry_router         # ALWAYS: geometric and coordinate reasoning
Worker 5 → gre_graph_crystallizer      # ALWAYS: multi-hop graph traversal (LED-A* chains)
Worker 6 → gre_temporal_reasoning      # ALWAYS: sequential/temporal logic chains
Worker 7 → gre_fractal_emitter         # ALWAYS: recursive, fractal, self-similar patterns
Worker 8 → gre_embedding_extractor     # ALWAYS: extracts dense input embedding for resonance
```

These 9 workers run in parallel on every task. The halting gate collects all 9 candidates
and converges on the winner. A geometry worker finding an unexpected answer on a language
task is a FEATURE — that's a cross-domain insight via symlinks.

---

## Worker 0 — The Decomposer (Applies to ALL Problems)

`gre_atomic_fission_fusion` at slot 0 is architecturally special: it runs FIRST and its
output (the decomposed sub-goal structure) is available to ALL other workers via shared VRAM
STORE/RECALL registers before they compute their candidates.

**Why decomposition applies to everything:**

- Math problem: "Mary has 5 apples, eats 2, how many left?" → sub-goals: {initial=5, delta=-2, result=?}
- MMLU question: "Which element has atomic number 6?" → sub-goals: {lookup=periodic_table, key=6, field=symbol}
- ARC grid: given training pairs → sub-goals: {transform_type=?, input=grid, output=?}
- LHE multi-hop: "X causes Y which causes Z, what does X cause?" → sub-goals: chain[X→Y, Y→Z, X→?]

In all cases, the decomposed sub-goal structure gives ALL other workers a better-structured
problem to navigate from. Workers that don't use the decomposition simply ignore it and
navigate from the raw input embedding instead.

**Decomposer execution order:**
Worker 0 launches first (or in a preliminary pass), stores sub-goals in STORE registers
60-67 (reserved for decomposition, per ADAPTIVE_REASONING_BUDGET_SPECIFICATION.md §A.2).
Workers 1-8 launch in parallel and may RECALL from those registers.

---

## Halting Gate Weights by Surface Kind

The halting gate (`gre_multimodal_halting_gate.cu`) already supports weighted agreement.
Surface kind adjusts which workers' confidence scores count more heavily toward the
convergence threshold — it does NOT exclude any worker from running.

```python
# Halting gate weight vectors (confidence multiplier per worker slot)
# All workers run; these weights affect convergence scoring only
HALTING_WEIGHTS = {
    SURFACE_KIND_GAME_2D: [
        1.0,  # worker 0: decomposer (spatial sub-goal structure)
        0.8,  # worker 1: resonance field (may find spatial analogies)
        1.2,  # worker 2: vector resonator (nearest training-pair embedding)
        2.0,  # worker 3: arc_reasoner (primary spatial specialist)
        2.0,  # worker 4: geometry_router (primary geometric specialist)
        1.0,  # worker 5: graph_crystallizer (multi-hop Grammar rules)
        0.6,  # worker 6: temporal reasoning (less central for spatial)
        1.5,  # worker 7: fractal_emitter (recursive spatial patterns)
        0.8,  # worker 8: embedding_extractor
    ],
    SURFACE_KIND_MATH: [
        2.0,  # worker 0: decomposer (CRITICAL for math word problems)
        1.0,  # worker 1: resonance field (analogous solved problems)
        1.5,  # worker 2: vector resonator (math symbol matching)
        0.8,  # worker 3: arc_reasoner (spatial math diagrams)
        1.5,  # worker 4: geometry_router (coordinate/geometric math)
        1.0,  # worker 5: graph_crystallizer (equation chain traversal)
        2.0,  # worker 6: temporal_reasoning (multi-step calculation sequence)
        0.6,  # worker 7: fractal_emitter (recursive formulas e.g. factorials)
        1.2,  # worker 8: embedding_extractor
    ],
    SURFACE_KIND_QUESTION: [
        1.5,  # worker 0: decomposer (question structure decomposition)
        2.0,  # worker 1: resonance field (PRIMARY for broad factual search)
        2.0,  # worker 2: vector resonator (answer candidate similarity)
        0.8,  # worker 3: arc_reasoner (structural question pattern)
        0.6,  # worker 4: geometry_router (less relevant for text questions)
        2.0,  # worker 5: graph_crystallizer (CRITICAL for LHE multi-hop)
        1.0,  # worker 6: temporal_reasoning (causal chain questions)
        0.5,  # worker 7: fractal_emitter (rarely relevant)
        1.5,  # worker 8: embedding_extractor
    ],
    # default: uniform weights (1.0 for all) when surface kind unknown
}
```

Pass the weight vector into `gre_multimodal_halting_gate` at launch. If the halting gate
currently takes a uniform agreement threshold, add a `worker_weights: float[9]` parameter.

---

## Micro-Specialist Layer: Worker-Worker Tier (MISSING — Wire This First)

**Authority:** `docs/vocabulary/MATH_CORE_SPECIFICATION.md §2.1–2.3`

The three-level hierarchy is fully specified in the vocabulary but only two levels are wired:

```text
Level 3 — Master:            TRM (trm_step_fused.ptx) — one sovereign entity
Level 2 — Workers:           Nine-chain internal swarm — 9 GRE specialists (wired ✅)
Level 1 — Micro-specialists: Tier-1 math cores spawned by workers (NOT YET WIRED ❌)
```

**Level 1 is the cheap parallel labor.** Each swarm worker focuses on high-level
Galaxy navigation and candidate selection (its specialty). The repetitive, cheap
sub-computations — evaluating each sub-goal, checking each graph hop candidate,
scoring each embedding similarity — are delegated to Tier-1 micro-specialists running
in parallel underneath the worker.

### Allocation Rule

> Each worker receives at least one micro-specialist. Workers can claim more from the
> GPU pool based on free SM capacity — never by problem complexity alone.
> (See feedback: "GPU dispatch: scale by FREE GPU resources, not cap by question complexity")

At boot, query GPU SM count and pre-allocate the Tier-1 pool:

```python
# In Knowledgeverse.boot() — after galaxy load:
import cupy as cp   # only at boot, not in hot path

sm_count = cp.cuda.Device().attributes["MultiProcessorCount"]  # RTX 3070 = 46
tier1_pool_size = int(sm_count * 10 * 0.66)  # 66% of capacity for Tier-1 (spec §2.3)
# RTX 3070: 46 × 10 × 0.66 ≈ 303 Tier-1 micro-specialist slots

self._micro_pool = MicroSpecialistPool(
    pool_size=tier1_pool_size,
    engine=LightweightRPNEngine,   # Tier-1: fast, cheap, bounded ops
    stack_depth=69,
    max_program_length=69,
)
```

`MicroSpecialistPool` — new file: `knowledge3d/cranium/ptx_runtime/micro_specialist_pool.py`:

- Thread-safe atomic slot acquisition (`acquire(n) → list[int]`)
- Batch Tier-1 program execution (`run_batch(slots, programs)`)
- Slot release back to pool after worker collects results
- Pool stats: `slots_used`, `slots_free`, `peak_utilization`

### How Workers Spawn Micro-Specialists

Each GRE worker's RPN program includes a `MICRO_SPAWN n` step. Codex defines this as
an RPN macro (Stage 1 in the opcode admission pipeline — not a new PTX kernel yet,
per `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §6`):

```text
MICRO_SPAWN n     → acquires n slots from pool, returns [slot_0 ... slot_n-1] on stack
MICRO_RUN         → executes the program at each slot in parallel (fan-out)
MICRO_COLLECT     → fan-in: reduction over slot results into single value on stack
MICRO_RELEASE     → returns slots to pool
```

These four macros are Stage 1 surface expansions in `ModularRPNEngine` — they expand to
existing `OP_STORE`/`OP_RECALL`/`OP_LOOP` + pool API calls. No new PTX kernel needed.

### Per-Worker Micro-Specialist Patterns

**Worker 0 — `gre_atomic_fission_fusion` (Decomposer):**

Spawns one micro-specialist per sub-goal, evaluates them in parallel:

```text
# Main worker: decompose
gre_fission_launch       # → n_goals on stack
STORE 60                 # save n_goals for other workers
MICRO_SPAWN n_goals      # claim n_goals Tier-1 slots
# Each micro-specialist slot i gets sub-goal i:
MICRO_RUN [
  RECALL goal_i          # sub-goal operands
  OP_SEMANTIC_RESOLVE    # 0x23A — resolve symbol → value
  cas_push_const         # operand a
  cas_push_const         # operand b
  RECALL op_i            # operation (+, -, *, /)
  cas_build
  OP_CANONICALIZE        # 0x238 — constant fold
]
MICRO_COLLECT SUM        # aggregate: partial results ready for worker 6
MICRO_RELEASE
```

**Worker 5 — `gre_graph_crystallizer` (Multi-hop):**

Spawns one micro-specialist per hop candidate, traverses in parallel:

```text
RECALL 60                # n_goals from decomposer
MICRO_SPAWN n_goals
MICRO_RUN [
  gre_crystallizer_step  # one LED-A* hop per micro-specialist
  OP_SEMANTIC_EQUIV      # 0x23D — is this hop an answer candidate?
]
MICRO_COLLECT MAX        # highest-confidence hop wins
MICRO_RELEASE
```

**Worker 1 — `gre_resonance_field` (Broad search):**

Spawns micro-specialists per Galaxy neighborhood chunk:

```text
PUSH n_galaxy_chunks     # partition all loaded galaxies into chunks
MICRO_SPAWN n_chunks
MICRO_RUN [
  gre_resonance_step     # semantic field over one galaxy chunk
]
MICRO_COLLECT MAX        # best resonance match across all chunks
MICRO_RELEASE
```

**Worker 2 — `gre_vector_resonator` (Embedding similarity):**

Spawns micro-specialists per top-k candidate:

```text
PUSH top_k               # e.g., 16 candidates from initial embedding probe
MICRO_SPAWN top_k
MICRO_RUN [
  OP_SEMANTIC_RESOLVE    # 0x23A — resolve candidate to value
  cosine_sim             # Tier-1 dot product
]
MICRO_COLLECT ARGMAX     # highest-similarity candidate
MICRO_RELEASE
```

### Pool Saturation Policy

If the pool has fewer free slots than requested, the worker takes what is available and
runs the rest sequentially. This degrades gracefully — never blocks. Report `slots_used /
slots_free` in the halting gate metadata so sleep-time consolidation can detect saturation
and advise Phase D scaling.

---

## CAS/SAS in Worker RPN Programs

Workers emit RPN programs containing CAS/SAS opcodes. TieredRPNEngine routes to the correct
tier. ALL workers can use ANY CAS/SAS opcode — the ops are universally available.

**Worker 0 (`gre_atomic_fission_fusion`) — Decomposer:**

```text
# Any problem type
gre_fission_launch             # decomposes input into (n_goals, goal_base, op_base)
STORE 60                       # n_goals → register 60 (shared)
STORE 61                       # goal_base → register 61 (shared)
STORE 62                       # op_base → register 62 (shared)
```

**Worker 6 (`gre_temporal_reasoning`) — sequences the decomposed sub-goals for MATH:**

```text
RECALL 60                      # n_goals
RECALL 61                      # goal_base
LOOP n_goals
  OP_SEMANTIC_RESOLVE          # 0x23A — resolve operand symbol → float value
  cas_push_const               # operand a
  cas_push_const               # operand b
  RECALL 62                    # operation opcode
  cas_build                    # build CAS STAR node
  OP_CANONICALIZE              # 0x238 — normalize (folds constants: 5-2 → 3)
  STORE sub_goal_result        # available to halting gate as candidate
ENDLOOP
```

**Worker 5 (`gre_graph_crystallizer`) — traverses multi-hop chains for QUESTION:**

```text
RECALL 60                      # n_goals from decomposer
RECALL 61                      # goal_base
LOOP n_goals
  gre_crystallizer_step        # one LED-A* hop through Reality/Word Galaxy
  OP_SEMANTIC_EQUIV            # 0x23D — is this hop's result an answer candidate?
  BRANCH_IF answer_found
ENDLOOP
```

**Worker 2 (`gre_vector_resonator`) — symbol matching for MATH:**

```text
RECALL question_embedding      # from worker 8's STORE
gre_resonator_launch           # cosine sim over Math Galaxy
LOOP top_k
  OP_SEMANTIC_RESOLVE          # 0x23A — bind symbol_id → value (PI, E, G...)
  STORE symbol_slot
ENDLOOP
```

---

## Grammar Rules for Math (Boot-time Seeding)

`sas_grammar_bootstrap.py` seeds 7 algebraic rules. These must load at boot:

```python
# In Knowledgeverse.boot() or load_all_galaxies():
from knowledge3d.cranium.kernels.sas_grammar_bootstrap import seed_math_grammar_rules
from knowledge3d.cranium.kernels.sas_symbol_bootstrap import seed_sas_symbols
seed_sas_symbols(self.gpu_context)          # PI, E, G, c, h, ħ, k_B, N_A, e, ε₀, μ₀
seed_math_grammar_rules(self.galaxy_manager)  # commutativity, identity, power rules
```

Without these, `OP_RULE_SELECT` finds nothing and `OP_SEMANTIC_RESOLVE` returns 0.0 for all
symbols. Add both calls if not already present.

---

## Tier-3 Executor Unification — Deferred to Phase D

CAS/SAS Tier-3 programs (containing `OP_CANONICALIZE` or `OP_CONTEXTUAL_REWRITE`) run on
the modular kernel. This is GPU hot-path, not Python fallback. Correct behavior.

The gap (AdvancedRPNEngine not absorbing CAS/SAS) only matters when programs mix CAS ops
with `OP_TRM_*` ops in a single RPN program — needed for Phase D (TRM game loop).
Do not implement this now.

---

## Galaxy Coverage Requirement

ALL galaxies must be loaded at boot (symlink constraint). Verify Knowledgeverse.boot() loads:

```text
Drawing, Character, Word, Number, Grammar, Math, Reality, Audio, 3DObjects, Tool
```

Fail loudly if any are missing. Silent partial loads break symlinks mid-traversal.

---

## Tests

```bash
# Verify all 9 workers run on ALL surface kinds (no exclusions)
bash scripts/k3d_env.sh run -e k3d-cranium \
  env PYTHONPATH=$(pwd) pytest -q tests/test_swarm_always_nine.py

# Verify halting gate receives correct weight vector per surface kind
bash scripts/k3d_env.sh run -e k3d-cranium \
  env PYTHONPATH=$(pwd) pytest -q tests/test_halting_gate_weights.py

# Worker 0 decomposes correctly on math, question, AND game inputs
bash scripts/k3d_env.sh run -e k3d-cranium \
  env PYTHONPATH=$(pwd) pytest -q tests/test_decomposer_universal.py

# Math benchmark (20 tasks)
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/math_competitions.py --max-tasks 20 \
  --summary-output /tmp/math_swarm_summary.json

# MMLU benchmark (20 tasks)
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/mmlu.py --max-tasks 20 \
  --summary-output /tmp/mmlu_swarm_summary.json
```

`test_swarm_always_nine.py` must verify:
1. For GAME_2D task: exactly 9 workers launched, `gre_arc_reasoner` present (slot 3)
2. For MATH task: exactly 9 workers launched, `gre_atomic_fission_fusion` present (slot 0)
3. For QUESTION task: exactly 9 workers launched, `gre_graph_crystallizer` present (slot 5)
4. No worker is ever excluded by surface kind — the count is always 9

`test_decomposer_universal.py` must verify:
1. Worker 0 sub-goal decomposition runs on a math word problem → produces n_goals > 0
2. Worker 0 decomposition runs on an MMLU question → produces n_goals > 0
3. Worker 0 decomposition runs on an ARC grid pair input → produces n_goals > 0
4. STORE registers 60-62 are populated after worker 0 completes, before workers 1-8 read

---

## Report Back

Write `TEMP/CODEX_TO_CLAUDE_SWARM_DISPATCH_REPORT_2026-04-08.md` with:

1. Where fixed worker assignments are now defined (file + line)
2. How halting gate weight vectors are passed per surface kind
3. Grammar/symbol bootstrap confirmed at boot (yes/no)
4. Math benchmark: `tasks=20, correct=K, score=X%`
5. MMLU benchmark: `tasks=20, correct=K, score=X%`
6. All tests passing: command + count
7. Honest gaps (expected: some workers still stub-launching — report which)

---

## What NOT to Do

- Do NOT exclude workers by surface kind — all nine ALWAYS run
- Do NOT make worker identity dynamic or swappable per task — fixed permanent slots
- Do NOT add Python reasoning to compensate for workers not finding answers
- Do NOT implement Tier-3 unification — deferred to Phase D
- Do NOT wait for encyclopedia ingest — Grammar Galaxy has enough now to show improvement
- Do NOT benchmark-name any knowledge — meaning-named always
  (FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md §0)
