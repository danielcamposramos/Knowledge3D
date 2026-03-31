# Codex — Phase E.47: IMO Bench Integration + Reasoning Pipeline Push

**Date:** 2026-03-31
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** HIGH — broadening the benchmark surface AND fixing the reasoning pipeline

---

## Part A: Add IMO Bench to Existing Benchmark Suite

### What Is IMO Bench?

IMO Bench (https://imobench.github.io/) is a DeepMind benchmark suite for
olympiad-level mathematical reasoning, vetted by IMO medalists:

| Sub-benchmark | Problems | Focus |
|---------------|----------|-------|
| **IMO-AnswerBench** | 400 | Correct answers (Algebra, Combinatorics, Geometry, Number Theory) |
| **IMO-ProofBench** | 60 | Rigorous proof writing (graded 0-7) |
| **IMO-GradingBench** | 1000 | Grading mathematical solutions (4-way: Correct/Almost/Partial/Incorrect) |

Four difficulty levels: pre-IMO, IMO-Easy, IMO-Medium, IMO-Hard.

Source: https://github.com/google-deepmind/superhuman/tree/main/imobench/

### Integration Pattern

Follow the SAME pattern as `benchmarks/math_competitions.py` and
`benchmarks/gsm8k.py`:

1. **New file**: `benchmarks/imo_bench.py`
   - Class `IMOBenchmark` following the UnifiedMathBenchmark pattern
   - Thin wrapper: load JSONL/JSON dataset → route through `kv.execute_task()`
     with `MATH_TASK` type
   - Answer extraction: LaTeX comparison (same as math_competitions)
   - Support `--imo-count N` for sampling

2. **Dataset download**: Clone or fetch from the GitHub repo above into
   `/K3D/Knowledge3D.local/datasets/imo_bench/`
   - IMO-AnswerBench problems (400 items)
   - Optionally IMO-ProofBench (60 items, for later)

3. **Add to `run_full_benchmark.py`**:
   - Import alongside MMLU, GSM8K, LHE, ARC2, ARC3
   - New CLI arg: `--imo-count` (default: 20 for quick runs)
   - Log to `imo_bench.jsonl` in the run directory
   - Include in summary output

4. **Do NOT build a special solver path**: IMO problems are math questions.
   They go through the SAME `MATH_TASK` pipeline as everything else. The
   Galaxy has Math + Grammar stars. The composed head pipeline reasons.
   Same path, harder questions. That's the point.

### Success Criteria (IMO Bench)

- [ ] `benchmarks/imo_bench.py` exists with `IMOBenchmark` class
- [ ] Dataset downloaded to `/K3D/Knowledge3D.local/datasets/imo_bench/`
- [ ] `run_full_benchmark.py` includes `--imo-count` arg
- [ ] 20-problem quick run produces valid JSONL log
- [ ] Score reported alongside other benchmarks in summary
- [ ] No new Python orchestration for solving (uses existing MATH_TASK path)

---

## Part B: E.40 — 2D Platform Game Knowledge Corpus (100+ Stars)

The E.40 spec already exists in TEMP. Key points:

### What Codex Must Do

1. **Research** comprehensive 2D platform game mechanics:
   - Movement systems (gravity, jumping, ladders, swimming)
   - Puzzle mechanics (switches, keys/locks, pressure plates, teleporters)
   - Enemy patterns (patrol, chase, flee, projectile)
   - Level design patterns (linear, hub, metroidvania, scrolling)
   - Visual encoding (color = type, size = importance, animation = state)
   - State machines (idle, moving, falling, attacking, dead)
   - Resource systems (health, lives, score, inventory, fuel/energy)
   - Control schemes (D-pad, 2-button, analog, touch)

2. **Produce 100+ meaning-first stars** in `game_mechanics.jsonl`:
   - Each star = universal concept (not LS20-specific)
   - Symlinks to existing Galaxy entries (Grammar, Reality, Drawing)
   - Layer 2 (Meaning) focus — what the concept IS, not how it looks
   - Bidirectional symlinks (per feedback norm)

3. **Load at boot** alongside existing game knowledge (currently 121 stars)

### Success Criteria (E.40)

- [ ] 100+ new meaning-first game knowledge stars
- [ ] Loaded at boot (total game stars > 221)
- [ ] Universal concepts, not game-specific
- [ ] Symlinked to Grammar/Reality/Drawing galaxies

---

## Part C: E.37 — Hardware-Adaptive CPU Parallelization

The E.37 spec already exists in TEMP. Key points:

### What Codex Must Do

1. **Detect hardware**: CPU cores, cache sizes, available RAM
2. **Cache profile**: Write to `/K3D/Knowledge3D.local/hardware_profile.json`
3. **Parallelize `run_full_benchmark.py`**:
   - MMLU, GSM8K, LHE, ARC2, ARC3, IMO each in a subprocess
   - Share single Knowledgeverse instance (or fork after boot)
   - Adaptive: fewer cores → fewer parallel suites
   - Streaming progress from all suites

### Success Criteria (E.37)

- [ ] Hardware profile detected and cached
- [ ] 5+ benchmark suites run in parallel
- [ ] Wall-clock time reduced vs sequential
- [ ] Adapts on migration to different hardware

---

## Part D: E.38 — 4-Way Reading Strategy on GPU (GSM8K Fix)

The E.38 spec already exists in TEMP. This is the CRITICAL one for
reasoning quality across ALL benchmarks. Key points:

### What Codex Must Do

The GSM8K 0/20 failure (and weak IMO/math performance generally) is
because the pipeline does ONE operation where multi-step chains are
needed. The 4-way reading strategy:

1. **Forward read**: Extract entities and quantities from the problem
   text in order. Each entity → Galaxy star lookup. Each quantity →
   Number Galaxy binding.

2. **Backward read**: Start from the GOAL (what is asked) and trace
   backward to identify which operations and quantities are needed.
   This is the "what do we need to compute?" pass.

3. **Operation chain construction**: Build a multi-step RPN program
   from the extracted operations. NOT a single left-fold — a DAG of
   dependent computations. Use the Adaptive Reasoning Budget spec
   (§6.2 Recursive Sub-Task Decomposition) for this.

4. **Normalization/validation**: Verify intermediate results are
   plausible (units match, magnitudes reasonable).

### Where This Lives in the Architecture

Per the specs:
- **Grammar Galaxy**: Transformation rules (RPN) that express operations
  like "multiply", "subtract", "divide-by". These ARE the reading
  strategies.
- **Meta-Rules (Layer 4)**: WHEN to apply which reading strategy. The
  4-way strategy is a meta-rule: "For word problems, apply forward+backward
  reading before constructing the operation chain."
- **Adaptive Reasoning Budget**: §6.2 already defines recursive sub-task
  decomposition. GSM8K needs this: break "calculate the total cost" into
  sub-tasks "find unit price" + "multiply by quantity" + "subtract discount."

### Implementation Path

These must be **Grammar Galaxy stars + Meta-Rule stars**, NOT Python methods:

```
Star: "forward_entity_extraction"
  layer: 3 (Rule)
  meaning: "Read problem left-to-right. For each noun, check Galaxy for
           matching meaning star. For each number, bind to Number Galaxy."
  rpn: "TOKENS FOREACH IF NOUN THEN GALAXY_LOOKUP IF NUMBER THEN NUM_BIND"
  symlinks: → backward_goal_tracing, operation_chain_construction

Star: "backward_goal_tracing"
  layer: 3 (Rule)
  meaning: "Identify the question (what is asked). Trace backward from
           goal through dependencies to identify required operations."
  rpn: "QUESTION_ENTITY GOAL_BIND DEPS REVERSE_TRACE"
  symlinks: → forward_entity_extraction, operation_chain_construction

Star: "operation_chain_construction"
  layer: 3 (Rule)
  meaning: "From extracted entities and goal dependencies, build a
           multi-step RPN program. Each step is one arithmetic operation.
           Steps chain via STORE/RECALL registers."
  rpn: "DEPS TOPO_SORT FOREACH OPERATION_BIND STORE CHAIN"
  symlinks: → forward_entity_extraction, backward_goal_tracing

Star: "four_way_reading_meta_rule"
  layer: 4 (Meta-Rule)
  meaning: "For word problems requiring multi-step arithmetic: apply
           forward extraction, then backward goal tracing, then chain
           construction, then validation. This sequence ensures all
           quantities are bound before operations are attempted."
  rpn: "IF WORD_PROBLEM THEN forward_entity_extraction CALL
        backward_goal_tracing CALL operation_chain_construction CALL
        result_validation CALL"
  symlinks: → forward_entity_extraction, backward_goal_tracing,
             operation_chain_construction
```

These stars go into the House JSONL and are loaded at boot. The TRM
navigates to them when it encounters a math word problem. The composed
head pipeline ALREADY supports multi-step reasoning via the Adaptive
Reasoning Budget — the stars just need to exist so the TRM can find
and compose them.

### The 35+ Python `_gsm8k_*` Methods

The ~600 lines of Python GSM8K methods in `knowledgeverse.py` (lines
5011-6640) do IN PYTHON what these Grammar Galaxy rules should do ON GPU.
Don't delete them yet — but the new stars should be the PRIMARY path.
When the Galaxy path works, the Python methods become dead code that
can be removed in a later cleanup.

### Success Criteria (E.38)

- [ ] 4-way reading strategy stars in House JSONL (loaded at boot)
- [ ] Meta-rule star links the 4 steps in correct sequence
- [ ] GSM8K problems route through Grammar Galaxy for operation chain
- [ ] Multi-step arithmetic works (not single left-fold)
- [ ] GSM8K score improves from 0/20 (any non-zero is progress)
- [ ] Math benchmark score maintained (20/20 must not regress)
- [ ] IMO Bench produces non-zero scores through same path

---

## Execution Order

Codex should execute these in parallel where possible:

| Task | Dependencies | Can Parallel? |
|------|-------------|---------------|
| IMO Bench integration | None (new benchmark, additive) | YES |
| E.40 game knowledge | None (new stars, additive) | YES |
| E.37 parallelization | IMO Bench (needs it in the suite) | AFTER IMO |
| E.38 4-way reading | None (new Galaxy stars + meta-rules) | YES |
| ARC3 (E.46+) | Continue in background | YES |

Suggested: start IMO + E.40 + E.38 in parallel. Add E.37 after IMO
is integrated.

---

## Architectural Note: Why This Matters

The hardcoded LS20 script was a symptom. The disease is that the
pipeline's reasoning capability is shallow:

- **ARC3**: pathfinder works (E.45) but strategy selection is blind
- **GSM8K**: single-operation fission where multi-step chains needed
- **Math**: 20/20 on synthetic guard set, untested on real MATH dataset
- **MMLU**: 23% baseline, mostly from knowledge retrieval not reasoning
- **LHE**: 9/100, multi-hop graph crystallizer needed

ALL of these share the same root cause: **the Grammar Galaxy is too
thin**. The procedural rules that EXPRESS reasoning strategies don't
exist as navigable stars. The TRM can't navigate to "backward goal
tracing" if that concept isn't a star in the Galaxy.

E.38 adds the reasoning strategy stars.
E.40 adds the domain knowledge stars.
IMO Bench adds a harder measurement surface.
E.37 makes the whole suite run faster.

This is the "knowledge curation and proceduralization" Daniel called
for — not tuning Python heuristics, but building the Galaxy that the
TRM navigates.
