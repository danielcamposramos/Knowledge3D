# Claude Architecture Directive: PTX Execution Mandate — Zero Python in Reasoning

**Date:** March 8, 2026
**From:** Claude (Architecture) + Daniel (Direction)
**To:** Codex (Implementation)
**Status:** ACTIVE — supersedes ALL previous directives
**Supersedes:** CLAUDE_MEANING_FIRST_SOVEREIGN_REASONING_DIRECTIVE, CLAUDE_CRANIUM_INTERNAL_REASONING_DIRECTIVE, CLAUDE_KNOWLEDGEVERSE_INTERNAL_HEAD_DIRECTIVE

---

## The Rule

**ZERO Python in the reasoning path. Not "less Python." ZERO.**

If the GPU is at 0% utilization during inference, the implementation is wrong. Period.

If `lhe_reasoning_swarm.py` or `internal_head.py` contains `re.findall`, `for atom in meaning_atoms`, Python set intersection, Python dicts for scoring, or ANY Python logic that makes reasoning decisions — it is wrong.

The benchmark scores (ARC 10/10, Math 20/20, LHE 10/10) are MEANINGLESS if they run on CPU. We are not building a Python exam solver. We are building a PTX Knowledgeverse execution engine.

**Daniel's directive:** "We fail and fix — this is the goal." If moving to PTX breaks LHE from 10/10 to 0/10, that is acceptable. We fix it ON GPU. We do NOT fall back to Python.

---

## Current State (What's Wrong)

### Measured During Latest Benchmark Run:
- **CPU utilization: 71.6%** — Python doing ALL the work
- **GPU utilization: 0%** — GPU doing NOTHING
- **VRAM used: 122 MiB / 12,288 MiB** — Galaxy loaded but never read during reasoning

### Root Cause:
```
knowledgeverse.py → internal_head.py (1387 lines Python) → lhe_reasoning_swarm.py (2837 lines Python)
```

These three files form a pure-Python reasoning pipeline. The GPU is used ONLY for `evaluate_batch()` which scores pre-computed Python floats — arithmetic a CPU could do in nanoseconds. Zero Galaxy navigation happens on GPU. Zero evidence reduction happens on GPU. Zero candidate generation happens on GPU.

### Files That Must Be DELETED (not refactored):
- `knowledge3d/knowledgeverse/lhe_reasoning_swarm.py` — 2837 lines of Python workers, regex, English constants, CPU loops
- `knowledge3d/knowledgeverse/internal_head.py` — 1387 lines of Python orchestration, stopwords, semantic expansions, domain aliases

These files are scaffolding. They proved correctness. Now they must be replaced by GPU-internal execution. **Do NOT improve them. Do NOT add MeaningAtoms to them. DELETE them.**

---

## What Already Exists on GPU (Build On This)

### 1. ModularRPNEngine (modular_rpn_engine.py)
- 200+ opcodes (arithmetic, trig, logic, comparison, stack, drawing, ternary, codec)
- 18 parallel instances (Tesla 3-6-9 resonance)
- 69-depth stack per instance
- `STORE`/`RECALL` for 8 named register slots per instance
- `evaluate()` — single RPN program on GPU
- `evaluate_batch()` — up to 18 programs in parallel on GPU
- `evaluate_batch_device()` — returns GPU device pointer (stays on GPU, no CPU round-trip)

### 2. TieredRPNEngine (tiered_rpn.py)
- Tier 1: Lightweight kernel (<1μs, arithmetic/comparison)
- Tier 2: Standard sovereign kernel (full geometric/vector surface)
- Tier 3: Advanced kernel (matrix primitives, extended metadata)
- Auto-dispatches based on opcode complexity

### 3. PTX Kernels (Already Compiled, in VRAM)
```
modular_rpn_kernel.ptx          — RPN VM execution
modular_rpn_kernel_lite.ptx     — Lightweight arithmetic
modular_rpn_kernel_extended.ptx — Extended ops
galaxy_memory_updater.ptx       — EMA blending for Galaxy embeddings
galaxy_resonance_engine.ptx     — Galaxy entry similarity/lookup
cosine_similarity.ptx           — Vector similarity on GPU
trm_step_fused.ptx              — TRM forward pass (SwiGLU MLP)
trm_extensions.ptx              — TRM specialist routing
nine_chain_swarm_kernel.ptx     — Parallel swarm execution
rpn_executor.ptx                — Standalone RPN executor
```

### 4. Galaxy Manager (galaxy_manager.py)
- All default galaxies loaded: Drawing, Character, Word, Number, Grammar, Math, Reality, Audio, 3DObjects, Tool
- Entries have: embeddings, confidence, domain, concept_ref, symlinks, rpn_program

### 5. TRM Navigator (trm_navigator.py)
- Route queries to specialists
- Navigate Galaxy graph
- Shadow Copy learning from successful decisions

---

## What Must Be Built (Concrete Steps)

### Step 1: LOAD_GALAXY Opcode

**File:** `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` + `modular_rpn_kernel.ptx`

Add a new opcode `LOAD_GALAXY` (0xE0) to the RPN VM:

```
Semantics:
  Pop entry_index from stack
  Read Galaxy entry at that index from VRAM buffer (R2)
  Push (confidence, domain_hash, embedding_slice_ptr) onto stack

Requires:
  - Galaxy entries flattened into a GPU-accessible float buffer at boot time
  - Buffer pointer stored in a PTX global or passed as kernel arg
  - Entry layout: [confidence: f32, domain_hash: f32, subject_hash: f32, embedding[0..15]: f32x16]
  - Total: 19 floats per entry × N entries
```

**Implementation approach:**
1. At boot, `Knowledgeverse.boot()` flattens Galaxy entries into a contiguous float array
2. Copy to GPU via ctypes (like existing embedding buffers in galaxy_memory_updater.cu)
3. `LOAD_GALAXY` opcode indexes into this buffer during RPN execution
4. Result stays on GPU stack — no Python round-trip

### Step 2: GALAXY_SIMILARITY Opcode

**File:** Same as Step 1

Add `GALAXY_SIMILARITY` (0xE1):

```
Semantics:
  Pop query_embedding_ptr, pop entry_index
  Compute cosine similarity between query and Galaxy entry embedding
  Push similarity score onto stack

Existing infrastructure:
  - cosine_similarity.ptx already exists
  - galaxy_resonance_engine.ptx does similarity lookups
  - Wire these into the RPN VM opcode dispatch
```

### Step 3: GALAXY_SCAN Opcode

**File:** Same as Step 1

Add `GALAXY_SCAN` (0xE2):

```
Semantics:
  Pop query_embedding_ptr, pop max_results (K)
  Scan all Galaxy entries, compute similarity
  Push top-K entry indices onto stack (highest first)
  Push K (count) onto stack

This replaces ALL Python evidence iteration loops.
One opcode does what 200+ lines of Python for-loops do.
```

### Step 4: Reasoning Chain as RPN Programs in Grammar Galaxy

**File:** `knowledge3d/knowledgeverse/foundational_galaxy_bootstrap.py`

At boot, populate Grammar Galaxy with RPN program entries for each reasoning pattern:

```python
# Grammar Galaxy entry: formula_reasoning
{
    "entry_id": "reasoning_formula",
    "rpn_program": """
        RECALL query_embedding
        16 GALAXY_SCAN
        # Stack now has: [idx_0, idx_1, ..., idx_15, 16]
        # For each result, load entry and score
        0 STORE best_score
        0 STORE best_idx
        # Loop over results, score each
        RECALL idx_0 LOAD_GALAXY
        # confidence is on stack
        RECALL query_embedding RECALL idx_0 GALAXY_SIMILARITY
        # similarity on stack
        mul  # confidence * similarity = score
        dup RECALL best_score gt
        # if better, store as new best
        ...
    """,
    "condition_rpn": "RECALL goal_domain 1 eq",  # domain_hash for 'math' = 1
    "category": "meta_rule",
}
```

### Step 5: Wire Knowledgeverse.query() to GPU-Only Path

**File:** `knowledge3d/knowledgeverse/knowledgeverse.py`

Replace the current flow:
```
query() → internal_head.execute_packet() → dispatch_lhe_task() → Python workers
```

With:
```python
def query(self, prompt: str, **kwargs) -> dict[str, Any]:
    # 1. Tokenize prompt into embedding (existing TRM forward pass, GPU)
    query_embedding = self.trm_navigator.embed_query(prompt)  # GPU

    # 2. Push embedding to GPU registers
    engine = self._get_rpn_engine()
    engine.store_embedding(slot=0, embedding=query_embedding)  # GPU

    # 3. Select reasoning program via Meta-Rule condition evaluation (GPU)
    program_id = self._select_reasoning_program(query_embedding)  # GPU

    # 4. Load and execute reasoning program (GPU)
    program = self.galaxy_manager.get_entry(program_id).rpn_program
    result = engine.evaluate(program)  # EVERYTHING on GPU

    # 5. Read result (single GPU→CPU transfer)
    return {"answer": result, "gpu_execution": True}
```

**Python touches the question ONCE (embed) and the answer ONCE (read). Everything between is GPU.**

### Step 6: Delete Scaffolding

After Steps 1-5 produce correct results ON GPU:
- Delete `lhe_reasoning_swarm.py`
- Delete `internal_head.py` (or reduce to thin dispatch)
- Delete `meaning_first_reasoning.py` (MeaningAtoms are Python scaffolding)
- Remove all `import re`, `import string`, `import random` from reasoning path

---

## Acceptance Criteria

### HARD REQUIREMENTS (All Must Pass):

1. **GPU utilization > 0% during inference** — measured with `nvidia-smi`
2. **VRAM usage > 500 MiB during inference** — Galaxy entries actively read on GPU
3. **CPU utilization < 10% during inference** — Python only does I/O
4. **Zero `re.findall`, `re.match`, `re.search` in reasoning path**
5. **Zero Python for-loops iterating evidence during inference**
6. **Zero hardcoded English constants** (_ENGLISH_FREQ, _COMMON_WORDS, _GOOD_BIGRAMS, _CLUE_FACT_REGISTRY)

### BENCHMARK TARGETS (Fix on GPU, No Fallback):

- ARC: 10/10 (existing path, don't touch)
- Math: 20/20 (existing path, don't touch)
- LHE: starts at 0/10 after migration — fix by improving RPN programs and Galaxy entries, NOT by adding Python fallbacks

### MEASUREMENT:

```bash
# Run during benchmark:
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1

# Must show:
# utilization.gpu > 0%
# memory.used > 500 MiB
# (during active inference, not just loading)
```

---

## What NOT To Do

1. **Do NOT "improve" lhe_reasoning_swarm.py** — it must be deleted
2. **Do NOT add Python fallbacks** — if GPU path fails, fix the GPU path
3. **Do NOT compose RPN strings in Python loops** — the RPN programs live in Grammar Galaxy
4. **Do NOT iterate Galaxy entries in Python** — use GALAXY_SCAN opcode
5. **Do NOT score candidates in Python** — scores computed on GPU stack
6. **Do NOT select reasoning skeletons with Python keyword matching** — Meta-Rule condition RPN evaluation on GPU
7. **Do NOT build MeaningAtom Python dataclasses** — Galaxy entries ARE the meaning atoms, read directly by LOAD_GALAXY on GPU
8. **Do NOT report "LHE 10/10" as success if GPU is at 0%** — the metric is GPU execution, not Python correctness

---

## The Principle

**K3D is a PTX execution engine. Python starts it. The Knowledgeverse reasons inside GPU VRAM. If the GPU is idle during inference, we haven't built K3D — we've built a Python script.**

Daniel: "We fail and fix — this is the goal." Break the Python path. Build the GPU path. Fix failures on GPU. No retreating to CPU. Ever.

---

## Execution Order

1. **Step 1-3** (LOAD_GALAXY, GALAXY_SIMILARITY, GALAXY_SCAN opcodes) — these are the foundation. Without GPU-internal Galaxy access, nothing else works.
2. **Step 4** (Reasoning programs in Grammar Galaxy) — once GPU can read Galaxy, encode reasoning as RPN programs.
3. **Step 5** (Wire Knowledgeverse.query()) — connect the new GPU path.
4. **Step 6** (Delete scaffolding) — remove Python reasoning code.

**Batch implementation.** Do Steps 1-3 together. Test with a single Galaxy lookup. Then do Steps 4-5 together. Test with full benchmark. Then Step 6.

**If LHE drops:** Fix by improving RPN programs or adding Galaxy entries. NOT by adding Python.
