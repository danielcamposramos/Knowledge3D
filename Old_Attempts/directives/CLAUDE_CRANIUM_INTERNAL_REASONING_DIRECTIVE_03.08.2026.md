# Claude Architecture Directive: Cranium-Internal Reasoning -- Move LHE Out of Python Entirely

**Date:** March 8, 2026
**From:** Claude (Architecture) + Daniel (Direction)
**To:** Codex (Implementation)
**Context:** LHE at 3/10 (then regressed to 2/10). Despite MeaningAtom layer, Galaxy bootstrap entries, and sovereign RPN scoring, the workers are still Python loops on CPU. Daniel identified the fundamental problem: the bridges should be INTERNAL. The TRM should execute reasoning WITHOUT leaving the GPU.

---

## Daniel's Insight

"We MUST move out from python entirely, those bridges should be internal? can it be called from inside (TRM)? you see what I mean (no CPU calls = fast execution)"

This is the deepest architectural correction yet. It's not about making workers use MeaningAtoms instead of regex. It's about eliminating the Python worker layer entirely and having the TRM/Cranium execute reasoning inside the GPU context.

---

## The Current Architecture Problem

### What Actually Happens (Slow, CPU-Heavy)

```
1. Python daemon receives question                    [CPU]
2. Python four-pass decomposition                     [CPU]
3. Python builds parse_bundle dict                    [CPU]
4. Python queries Galaxy (via Python Knowledgeverse)  [CPU -> GPU -> CPU round-trip]
5. Python builds MeaningAtom list from dicts          [CPU]
6. Python FormulaReasoningWorker iterates atoms       [CPU loop]
7. Python ConceptMatchingWorker iterates atoms        [CPU loop]
8. Python ProceduralExecutionWorker runs SA cipher    [CPU heavy]
9. Python EvidenceSynthesisWorker iterates fields     [CPU loop]
10. Python collects proposals into dict               [CPU]
11. Python builds RPN expression strings              [CPU]
12. GPU evaluates RPN batch (scoring)                 [GPU -- ~42us]
13. Python picks best result                          [CPU]
```

Steps 4-11 are the bottleneck. The GPU does ~42us of actual work (step 12). Everything else is Python loops on CPU.

### What the Spec Describes (Fast, GPU-Internal)

From THREE_BRAIN_SYSTEM_SPECIFICATION.md Section 3.4:

```assembly
PUSH "neuron"              # Push query to stack
LOAD_GALAXY embedding      # Load embedding vector from Galaxy
PUSH 10                    # Top-K parameter
CALL find_similar          # Find 10 most similar nodes
CALL pathfind_to_answer    # Navigate to answer node
RECALL answer_text         # Retrieve text data
OUTPUT                     # Return to user

# Execution Trace (42us TOTAL):
```

This is the spec's vision: the ENTIRE reasoning chain runs as RPN opcodes inside the Cranium. `LOAD_GALAXY` reads from VRAM. `CALL find_similar` runs on GPU. `RECALL` retrieves from GPU memory. 42us total. No Python loops.

### The Gap

The spec describes `LOAD_GALAXY`, `CALL find_similar`, `RECALL` as RPN opcodes that execute inside the Cranium's PTX context. But the current implementation:

1. `LOAD_GALAXY` is not implemented as an opcode in ModularRPNEngine
2. `find_similar` is not implemented as a callable RPN function
3. Galaxy queries go through Python `Knowledgeverse.query()` -> Python `trm_navigator.navigate_and_compose()` -> Python loops

The "bridges" (procedural_drawing_bridge, etc.) exist but they're Python-to-GPU bridges -- they marshal data from Python to GPU and back. Daniel's point: these should be INTERNAL bridges that the TRM calls without leaving the GPU context.

---

## The Target Architecture

### Phase A: RPN Programs That Navigate Galaxy

The first step is not to implement new PTX kernels. It's to make the existing `evaluate_batch` do MORE work per GPU call by composing longer RPN programs that encode the full reasoning chain.

**Current:** Python builds many small RPN expressions, each evaluated on GPU separately:
```python
# Python builds these strings:
expressions = [
    f"{support:.6f} {worker_bonus:.6f} + {triangulation:.6f} + ...",  # candidate 1
    f"{support:.6f} {worker_bonus:.6f} + {triangulation:.6f} + ...",  # candidate 2
    # ...
]
# GPU evaluates them
scores = engine.evaluate_batch(expressions)
```

**Target:** Python builds ONE large RPN program that encodes the full reasoning pipeline:
```python
# Instead of Python iterating MeaningAtoms in a loop,
# compose an RPN program that does the iteration on GPU:
program = """
    # For each evidence entry (pre-loaded as STORE slots):
    RECALL evidence_0_concept_ref
    RECALL evidence_0_confidence
    RECALL goal_domain
    eq                          # does evidence domain match goal?
    RECALL evidence_0_rpn
    # ... evaluate sub-program ...
    # Store intermediate score
    STORE score_0

    RECALL evidence_1_concept_ref
    RECALL evidence_1_confidence
    RECALL goal_domain
    eq
    # ...
    STORE score_1

    # Select best
    RECALL score_0
    RECALL score_1
    max
"""
result = engine.evaluate(program)
```

This is still using the EXISTING `ModularRPNEngine.evaluate()`. No new PTX kernels needed. But the reasoning logic moves from Python loops into RPN programs that execute on GPU.

### Phase B: LOAD_GALAXY Opcode

Add `LOAD_GALAXY` as a real opcode in the RPN engine. When executed, it reads a Galaxy entry from VRAM by ID and pushes its fields onto the stack.

This requires:
1. Galaxy entries indexed by ID in a GPU-accessible buffer (they already are -- the augmentation snapshot is loaded into VRAM)
2. A new opcode (`LOAD_GALAXY = 0xE0`) that takes an entry ID from the stack, looks up the entry in the VRAM buffer, and pushes its embedding/confidence/domain onto the stack
3. The RPN program can then operate on the Galaxy data without returning to Python

```assembly
# RPN program that queries Galaxy internally:
PUSH "gamma_matrices"          # concept name
LOAD_GALAXY                    # loads entry from VRAM -> pushes embedding + metadata
DUP                            # duplicate for later
PUSH "trace_formula"           # second concept
LOAD_GALAXY                    # load second entry
dot                            # cosine similarity between entries
PUSH 0.7                       # threshold
gt                             # are they related?
# ... branch based on result ...
```

### Phase C: TRM-Internal Specialist Routing

The TRM specialist tree (SpecialistBase with spawn_child, route, LoRA deltas) should route queries INSIDE the GPU context. Instead of Python calling `specialist_router.py` to decide which worker to use, the TRM's forward pass outputs routing weights that select which RPN program to execute.

```
Current:
  Python specialist_router.route(query) -> returns specialist name
  Python selects worker based on name
  Python worker runs on CPU

Target:
  TRM forward pass (GPU) -> outputs routing vector [formula: 0.8, concept: 0.1, ...]
  Routing vector selects RPN program ID from Grammar Galaxy (GPU lookup)
  Selected RPN program executes on GPU (Cranium)
  No Python routing logic
```

### Phase D: Full Cranium-Internal Pipeline

The end state from the spec:

```
Question text -> tokenized into embedding (GPU)
  -> TRM forward pass selects specialist + skeleton (GPU)
  -> Skeleton's RPN program executes:
     -> LOAD_GALAXY queries evidence entries (GPU, VRAM)
     -> Composition operations on evidence (GPU, RPN stack)
     -> STORE/RECALL for intermediate results (GPU registers)
     -> Score computation (GPU, RPN arithmetic)
  -> Best result returned to Python (single GPU->CPU transfer)
```

Total GPU time: ~42-100us (Cranium specification target).
Total CPU time: question input + result output only.

---

## Concrete Implementation Steps

### Step 1: Compose Full Reasoning Chain as Single RPN Program (No New Opcodes)

**What changes:** Instead of Python workers iterating evidence and building proposal lists, compose ONE RPN program per LHE question that encodes the full reasoning logic.

**How:** The daemon's `_dispatch_lhe_task` pre-loads evidence into STORE slots (numbered registers), then builds a single RPN program:

```python
def _compose_lhe_reasoning_program(self, evidence_rows, meaning_atoms, goal):
    """Build one RPN program that evaluates all candidates on GPU."""
    program_parts = []

    # Pre-store evidence data as numbered slots
    for i, atom in enumerate(meaning_atoms[:18]):  # max 18 (Tesla resonance)
        program_parts.append(f"{atom.confidence:.6f} store r{i}_conf")
        # domain match score
        domain_match = 1.0 if atom.domain == goal.get("domain", "") else 0.0
        program_parts.append(f"{domain_match:.6f} store r{i}_domain")
        # subject match score
        subject_match = 1.0 if atom.subject in str(goal.get("raw", "")).lower() else 0.0
        program_parts.append(f"{subject_match:.6f} store r{i}_subject")

    # Scoring logic as RPN (replaces Python loop)
    for i in range(min(len(meaning_atoms), 18)):
        program_parts.append(f"recall r{i}_conf recall r{i}_domain 0.5 * + recall r{i}_subject 0.3 * + store score_{i}")

    # Find max score
    if meaning_atoms:
        program_parts.append("recall score_0")
        for i in range(1, min(len(meaning_atoms), 18)):
            program_parts.append(f"recall score_{i} max")

    return " ".join(program_parts)
```

This moves the entire scoring loop from Python to GPU. One `engine.evaluate()` call replaces Python iteration.

**Impact:** Evidence iteration, domain matching, confidence weighting, score comparison -- all on GPU in one call. Python only prepares the program and reads the result.

### Step 2: Add LOAD_GALAXY Opcode

**What changes:** New opcode in ModularRPNEngine that reads a Galaxy entry from the VRAM augmentation buffer.

**Prerequisites:**
- The augmentation snapshot entries need to be indexed by numeric ID in a GPU-accessible array
- Each entry needs its key fields (confidence, domain hash, subject hash, embedding slice) in a flat struct

**Opcode semantics:**
```
LOAD_GALAXY: pop entry_index from stack, push (confidence, domain_hash, subject_hash) as 3 values
```

This enables RPN programs to query Galaxy without returning to Python.

### Step 3: Compile Reasoning Skeletons to RPN Programs at Boot

**What changes:** At daemon startup, compile each reasoning skeleton (CoT, Elimination, Dimensional Analysis, etc.) into a concrete RPN program template. Store these in Grammar Galaxy as executable entries.

**How:** Each skeleton becomes a parameterized RPN program:

```python
# Grammar Galaxy entry:
{
    "id": "skeleton_cot",
    "rpn_program": """
        # Chain of Thought: decompose -> solve each -> chain
        recall step_count
        0 store current_step
        # Loop: for each step, load evidence, score, accumulate
        recall current_step recall step_count lt
        {
            recall current_step LOAD_GALAXY
            # ... evaluate sub-step ...
            recall current_step 1 + store current_step
        } loop
        # Return accumulated score
    """,
    "condition_rpn": "recall goal_kind 'sequential' eq",
}
```

At runtime, the TRM selects a skeleton by evaluating `condition_rpn` on GPU, then executes `rpn_program` on GPU. No Python skeleton selection.

### Step 4: TRM Forward Pass Outputs Routing Vector

**What changes:** The TRM's forward pass (2-layer SwiGLU MLP, already exists as PTX kernel) outputs a routing vector instead of just an embedding. The routing vector selects which skeleton's RPN program to execute.

**This requires:** Modifying the TRM output layer to include routing logits. The matryoshka specialist hierarchy already has routing_bias -- this makes routing happen on GPU instead of in Python.

---

## What This Eliminates

| Component | Status | After Phase A | After Phase D |
|-----------|--------|---------------|---------------|
| Python FormulaReasoningWorker | CPU loop | RPN program on GPU | Eliminated |
| Python ConceptMatchingWorker | CPU loop | RPN program on GPU | Eliminated |
| Python ProceduralExecutionWorker | CPU heavy (SA cipher) | Partially on GPU | RPN cipher solver |
| Python EvidenceSynthesisWorker | CPU loop | RPN program on GPU | Eliminated |
| Python skeleton selection | Keyword matching | RPN condition eval | TRM routing vector |
| Python proposal collection | Dict manipulation | GPU STORE/RECALL | GPU-internal |
| Python score fusion | Dict + sort | GPU evaluate_batch | Single GPU program |

---

## Performance Target

From THREE_BRAIN_SYSTEM_SPECIFICATION.md Section 3.3:

> Target latency: <100us per operation (sub-frame at 10,000 fps)
> Zero external dependencies (sovereignty principle)
> Memory-efficient (<2KB working memory per operation)

The current LHE path takes seconds (Python loops). The spec target is <100us. Phase A (composing full reasoning as single RPN programs) should bring this to milliseconds. Phases B-D should approach the spec target.

---

## Constraints

1. **ARC 10/10 and Math 20/20 must not regress.** These paths should NOT be touched.
2. **Start with Step 1** (compose reasoning as single RPN program). This requires NO new opcodes, NO new PTX kernels -- only changing HOW the RPN programs are built.
3. **The existing ModularRPNEngine.evaluate_batch() is the execution substrate.** Build on it, don't replace it.
4. **STORE/RECALL already work.** Use them for intermediate results within a single RPN program.
5. **18 parallel instances already exist** (Tesla 3-6-9 resonance). Use them for parallel candidate evaluation.

---

## The Principle

Daniel's insight maps directly to the Three-Brain System spec:

- **Cranium** = reasoning engine, ~42us per operation, GPU registers
- **Galaxy** = active memory, ~5us per access, GPU VRAM
- **House** = persistent memory, ~5ms per access, SSD

The spec says Cranium READS from Galaxy at 5us. The current implementation reads from Galaxy through Python at ~5ms (1000x slower). The fix is not to optimize Python -- it's to eliminate Python from the reasoning path entirely.

The Cranium's RPN Execution Engine already has the opcodes: PUSH, POP, STORE, RECALL, arithmetic, logic, control flow (BRANCH, LOOP). What's missing is Galaxy access (LOAD_GALAXY) and full-chain composition (encoding the entire reasoning pipeline as one RPN program instead of many small ones).

Step 1 doesn't even need LOAD_GALAXY. It just needs longer RPN programs that encode the worker logic currently written in Python. The GPU already has all the arithmetic and control flow opcodes. Python pre-loads evidence data into STORE slots, then the RPN program does everything else.

Daniel: "no CPU calls = fast execution." The RPN stack is the reasoning engine. Let it reason.
