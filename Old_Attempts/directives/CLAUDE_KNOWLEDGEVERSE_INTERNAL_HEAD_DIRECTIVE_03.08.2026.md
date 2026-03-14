# Claude Architecture Directive: The Head Lives Inside the Knowledgeverse -- Not in Python

**Date:** March 8, 2026
**From:** Claude (Architecture) + Daniel (Direction)
**To:** Codex (Implementation)
**Context:** Daniel's correction: "no python calls inside the system! This should work as internal only all the way -- it is a game like live system -- the knowledgeverse -- not a benchmark run machine, all these are a single head with an internal swarm -- all inside the K3D paradigm, no python only for starting the system!!!"

---

## The Fundamental Misunderstanding (What We Got Wrong)

Every directive so far -- swarm workers, MeaningAtoms, skeleton selection, meaning-first reasoning -- treated K3D as a **Python benchmark harness** that calls GPU functions. Workers are Python classes. Scoring calls `engine.evaluate()`. Evidence is Python dicts. MeaningAtoms are Python dataclasses.

This is backwards.

The Knowledgeverse is a **living GPU world**. The specs are explicit:

From KNOWLEDGEVERSE_SPECIFICATION.md Section 4.1:
```
HOT PATH (Inference Loop) = Sovereign ONLY

ALLOWED:
  - PTX kernel execution (Region 1)
  - Galaxy navigation (Region 2)
  - RPN program execution
  - TRM inference (Region 5)
  - Shadow Copy recording (Region 6)

FORBIDDEN:
  - CPU preprocessing (must be in Ingestion Stargate)
```

From THREE_BRAIN_SYSTEM_SPECIFICATION.md Section 2.1:
```
Cranium: Reasoning & Inference | GPU registers | ~42us
Galaxy:  Active working memory  | GPU RAM       | ~5us
```

The "head" (Cranium + TRM + specialists) LIVES in VRAM. It reads from Galaxy (VRAM). It writes to Galaxy (VRAM). It reasons with RPN (GPU registers). It learns via Shadow Copy (GPU). **Python only starts the system and feeds questions in.**

Think of it like a game engine: Python is the launcher. The game runs on GPU. You don't call Python for each frame. You don't call Python for each AI decision. The entire game loop runs on GPU. Python just starts it and reads output.

---

## What the Architecture Actually Is

### The Knowledgeverse Memory Map (from spec Section 3.1)

```
GPU VRAM (12GB):
  R1: KERNELS (100MB, PINNED)     -- PTX modules, always loaded
  R2: GALAXY_UNIVERSE (2-3GB)     -- ALL galaxies, always loaded, read+write
  R3: HOUSE_CONTEXT (2.5GB)       -- Loaded house objects
  R4: WORLD_VIEW (2-3GB)          -- Remote houses, streaming
  R5: TRM_WEIGHTS (400-800MB)     -- Base TRM + specialist adapters
  R6: AUDIT_JOURNAL (256MB)       -- Shadow Copy events, traces
  R7: INGESTION_STARGATE (512MB)  -- Raw -> RPN transmutation (ONLY place external data enters)
```

**Everything is in VRAM.** The TRM (R5) reads Galaxy (R2) directly -- no Python intermediary. The Cranium (R1 kernels) executes RPN programs using Galaxy data -- no Python intermediary. Shadow Copy (R6) records patterns -- no Python intermediary.

### The Inference Loop (from spec Section 3.4)

```assembly
PUSH "neuron"              # Query enters GPU context
LOAD_GALAXY embedding      # Read from R2 (Galaxy VRAM) -- 5us
PUSH 10                    # Parameter
CALL find_similar          # Cranium kernel (R1) -- GPU registers
CALL pathfind_to_answer    # Navigate Galaxy graph (R2) -- GPU
RECALL answer_text         # Retrieve from Galaxy -- 5us
OUTPUT                     # Return to host
# TOTAL: 42us
```

This is the ENTIRE inference. Not "Python builds RPN string, GPU evaluates arithmetic." The entire reasoning chain -- query parsing, Galaxy navigation, similarity search, pathfinding, answer retrieval -- runs as RPN opcodes on GPU.

### Where Python Fits

```
Python's role:
  1. Boot the Knowledgeverse (load PTX modules, initialize VRAM regions)
  2. Feed questions into the system (push to GPU input buffer)
  3. Read answers from the system (pull from GPU output buffer)
  4. Ingestion path (R7 Stargate): convert raw data to RPN, validate, crystallize into Galaxy

Python does NOT:
  - Iterate evidence
  - Score candidates
  - Select skeletons
  - Match keywords
  - Run frequency analysis
  - Build proposal lists
  - Fuse worker results
```

---

## What Needs to Change

### Current: Python Orchestrates Everything

```
Python daemon receives question
  -> Python four-pass decomposition (CPU)
  -> Python specialist_router.route() (CPU)
  -> Python builds parse_bundle dict (CPU)
  -> Python queries Galaxy (CPU -> GPU -> CPU round-trip)
  -> Python builds MeaningAtom list (CPU)
  -> Python workers iterate and score (CPU loops)
  -> Python builds RPN strings (CPU)
  -> GPU evaluates arithmetic only (GPU, 42us)
  -> Python picks best result (CPU)
```

### Target: GPU Executes Everything

```
Python pushes question to GPU input buffer
  -> Cranium tokenizes query (R1 kernel, GPU)
  -> TRM forward pass selects specialist route (R5 weights, GPU)
  -> Specialist's RPN program executes:
     -> LOAD_GALAXY reads evidence entries (R2, VRAM, 5us each)
     -> RPN composition evaluates candidates (R1, GPU registers)
     -> STORE/RECALL manages intermediate state (GPU registers)
     -> Shadow Copy records successful patterns (R6, GPU)
  -> Result written to GPU output buffer
  -> Python reads answer
```

Python touches the question ONCE (input) and the answer ONCE (output). Everything between is GPU-internal.

---

## Implementation Path

### Step 1: Make the Four-Pass Run on GPU

The four-pass decomposition (forward/backward/fusion entity extraction) currently runs in Python. It needs to run as a Cranium kernel:

1. Question text tokenized and pushed to R2 (Galaxy active reasoning buffer)
2. TRM forward pass (R5) produces entity embeddings
3. Entity embeddings compared against Galaxy entries (R2) via LOAD_GALAXY + cosine similarity
4. Fused entity graph stored in R2 (not in a Python dict)

The TRM already has a 2-layer SwiGLU MLP forward pass as a PTX kernel. It already does attention. What's missing is connecting its OUTPUT to Galaxy navigation instead of returning to Python.

### Step 2: Make Specialist Routing Run on GPU

The matryoshka specialist tree (SpecialistBase with routing_bias, LoRA deltas) already stores weights in VRAM (R5). But routing decisions currently happen in Python (specialist_router.py).

Target: TRM forward pass outputs a routing vector. The routing vector is dot-producted against specialist adapter descriptors (already in R5). The highest-scoring specialist is activated. All on GPU.

### Step 3: Make Evidence Navigation Run on GPU

Galaxy entries are already in VRAM (R2). The LOAD_GALAXY opcode (from spec Section 3.4) reads entries by ID. What's needed:

1. After routing selects a specialist, the specialist's RPN program template is loaded from Grammar Galaxy (R2)
2. The RPN program executes, calling LOAD_GALAXY to read evidence entries
3. Evidence fields (confidence, domain, content embedding) are on the GPU stack
4. Scoring operations (similarity, domain match, confidence weighting) run as RPN arithmetic
5. Best candidate stays on GPU stack

### Step 4: The Internal Swarm (Daniel's Vision)

"All these are a single head with an internal swarm."

The matryoshka specialist hierarchy IS the internal swarm:
- Navigator specialist selects multiple routes (parallel)
- Each route activates a sub-specialist (parallel)
- Sub-specialists execute their RPN programs (parallel, 18 instances)
- Results fuse via RPN composition (GPU)

This is NOT Python workers in a for-loop. It's multiple TRM specialist adapters (R5) executing in parallel GPU instances (R1 has 18 Tesla-resonance instances). The swarm is the matryoshka tree executing in parallel on GPU.

### Step 5: Shadow Copy Records Everything

Every successful reasoning chain becomes a new Galaxy entry:
- The RPN program that produced the correct answer is stored in Grammar Galaxy (R2)
- The TRM specialist routing weights are updated (R5, Shadow Copy)
- Next time a similar query arrives, the TRM already knows which program worked

This is the "learning while living" that the spec describes. No Python. No training loop. The system gets smarter by recording what worked, inside the GPU.

---

## What This Means for the LHE Workers

The Python workers (FormulaReasoningWorker, ConceptMatchingWorker, etc.) are **not part of the final architecture**. They were scaffolding.

In the final architecture:
- FormulaReasoningWorker = a math specialist adapter in R5, with an associated RPN program template in Grammar Galaxy (R2)
- ConceptMatchingWorker = a concept specialist adapter in R5, with elimination/matching RPN programs in R2
- ProceduralExecutionWorker = a procedural specialist adapter in R5, with cipher/chess RPN programs in R2
- EvidenceSynthesisWorker = the fusion step of the matryoshka Navigator, composing sub-specialist results

They're not Python classes. They're TRM specialist weights + Grammar Galaxy RPN programs. They LIVE in VRAM (R5 + R2). They EXECUTE on GPU (R1 kernels).

---

## Concrete First Step for Codex

**Don't refactor the Python workers.** That's polishing scaffolding.

**Instead:** Make the existing Cranium execute a full LHE reasoning chain internally.

1. The `ModularRPNEngine` already has 18 parallel instances, STORE/RECALL, 200+ opcodes
2. The augmentation snapshot is already loaded into VRAM as Galaxy entries
3. The TRM specialist tree is already in VRAM (R5)

What's missing is the GLUE: an RPN program that chains `LOAD_GALAXY` -> similarity -> scoring -> selection as one continuous GPU execution.

The entry point changes from:
```python
# Current: Python daemon does everything
result = self._dispatch_lhe_task(route=route, task=task, use_enriched=True)
```

To:
```python
# Target: Python just pushes query and reads result
self.knowledgeverse.push_query(question_text, query_type="lhe_open")
result = self.knowledgeverse.read_result()
# EVERYTHING between push and read happens on GPU
```

The Knowledgeverse class already exists (knowledgeverse.py). It already has `query()`. The change is making `query()` execute the ENTIRE reasoning chain on GPU instead of calling Python workers.

---

## What Python Should Look Like After This

```python
# main.py daemon -- ONLY job is to start the system and shuttle I/O

class K3DDaemon:
    def __init__(self):
        # Boot: load PTX kernels (R1), initialize Galaxy (R2), load TRM (R5)
        self.knowledgeverse = Knowledgeverse(config)
        self.knowledgeverse.boot()  # One-time VRAM setup

    def handle_query(self, question: str) -> str:
        # Push question to GPU, get answer from GPU
        return self.knowledgeverse.query(question)
        # That's it. No Python workers. No evidence loops.
        # No MeaningAtoms. No regex. No set intersection.
        # The Knowledgeverse handles EVERYTHING internally.
```

---

## The Principle

Daniel: "it is a game like live system -- the knowledgeverse -- not a benchmark run machine"

A game doesn't call Python for each NPC decision. The game engine runs on GPU. NPCs think inside the engine. The player interacts through input/output only.

K3D is the same. The Knowledgeverse is the game engine. The TRM + specialists are the NPCs. Galaxy is the game world. Python is the window manager that launches the game.

The spec already describes this world. Cranium executes at 42us. Galaxy reads at 5us. Seven VRAM regions. Shadow Copy learning. All on GPU.

We just need to build it as described instead of wrapping it in Python.
