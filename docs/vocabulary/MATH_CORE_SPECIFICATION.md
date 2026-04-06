# Math Core Specification — Tiered RPN Engines

**Version**: 1.0  
**Status**: Draft (Phase G/H architecture, Reality Enabler aligned)  
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)  
**Date**: November 2025

---

## Abstract

This specification defines the **Math Core** architecture that underpins all sovereign computation in K3D. It formalizes:

- The **Modular RPN Engine** as the core execution surface (18 stacks, 69-line programs).
- A tiered **math core** hierarchy (simple, mid, high complexity) with a worker‑worker → worker → master pattern.
- How multiple math cores can be instantiated, routed, and composed to balance speed vs capability.

This spec sits beneath Reality Enabler, Adaptive Procedural Compression, and all higher-level specialists: every nontrivial computation in K3D is ultimately an RPN program executed by one or more math cores.

---

## 1. Core RPN Engine

### 1.1 Modular RPN Engine

The **Modular RPN Engine** (`ModularRPNEngine` in `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`) is the canonical scalar/vector RPN executor:

- Backed by `modular_rpn_kernel.ptx` via `TieredRPNEngine` bridge.
- Provides:
  - Tokenization and compilation of RPN expressions into opcodes + literals.
  - GPU execution of a single program (`execute_single`) or batches (`execute_batch`).
- Architectural constants:
  - `_INSTANCE_COUNT = 18` — up to 18 parallel instances.
  - `_STACK_MAX = 69` — each instance supports programs up to 69 instructions with internal stack usage sized accordingly.

**Key property**: All higher-level math cores (simple, mid, high) are **configurations and routing patterns** on top of this same RPN execution surface.

### 1.2 Shared Opcode Surface

The engine shares a unified opcode set (`rpn_opcodes.py`), including:

- Tier‑1 arithmetic and math (`OP_ADD`, `OP_SUB`, `OP_MUL`, `OP_DIV`, `OP_SQRT`, `OP_EXP`, `OP_LOG`, trigonometry, comparisons, stack ops).
- Tier‑2 cooperative/linear algebra (`OP_MATVEC_F32`, `OP_MATMUL_SMALL`, reductions, clustering ops).
- Tier‑3 TRM integration (`OP_TRM_MATVEC_*`, `OP_TRM_SWIGLU_*`, etc.).
- Procedural drawing, temporal reasoning, and clustering extensions.

This specification treats them as a **single programmable substrate**; different math cores simply expose different subsets and routing policies.

---

## 2. Tiered Math Core Model

### 2.1 Three Tiers

To balance speed and capability, math is handled by three conceptual tiers of math cores:

- **Simple Core (worker‑worker)**:
  - Purpose: very fast, low-cost operations – basic arithmetic, small vector ops, simple predicates.
  - Typical opcode subset: Tier‑1 arithmetic/stack ops, small portions of Tier‑2 where latency is critical.
  - Analogy: asking a junior worker to handle routine calculations.

- **Mid Core (worker)**:
  - Purpose: moderate complexity – matrix‑vector multiplies, small matrix ops, clustering helpers, basic calculus-style primitives.
  - Opcode subset: full Tier‑1 + most Tier‑2 (e.g., `OP_MATVEC_F32`, `OP_MEMCPY_F32`, `OP_FILL_F32`, `OP_VEC_BLEND`, reductions).
  - Analogy: experienced engineer handling typical workloads.

- **High Core (master)**:
  - Purpose: high‑complexity operations – TRM coupling, symbolic-like primitives, quantum helpers, heavy multivariate ops.
  - Opcode subset: all tiers, including TRM integration ops (`OP_TRM_*`), quantum and programmable ops (`OP_SYMBOLIC_DIFF`, `OP_GRADIENT`, `OP_SERIES_SUM`, etc.).
  - Analogy: asking a PhD to do a full derivation or train a specialist – powerful but should be reserved for when it matters.

All three tiers share:
- The same 18-instance, 69-instruction limits from `ModularRPNEngine`.
- The same PTX backend; tiering is about **how** we schedule and which opcodes we allow, not about separate executors.

### 2.2 Worker–Worker → Worker → Master Pattern

Math cores are organized as a **fan‑in hierarchy**:

- Many **simple cores** (worker‑worker) perform:
  - Scalar ops, small vector ops, local filters.
  - Pre‑aggregation, initial norms, cheap probes.

- Fewer **mid cores** (worker):
  - Consume simple cores’ outputs (e.g., partial sums, local Jacobians),
  - Execute moderate ops (matrix multiplies, reductions, vectorized non‑linearities),
  - Prepare compact summary states/gradients.

- One or a small number of **high cores** (master):
  - Consume mid‑level summaries,
  - Execute TRM‑coupled or symbolic operations,
  - Make global decisions (e.g., commit/reject updates, choose new specialists).

This hierarchy is conceptual and implemented via:
- Multiple `ModularRPNEngine` instances (simple/mid/high core pools),
- Or via configuration of `TieredRPNEngine` / `AdvancedRPNEngine` and `RPNMathCore` (for Tier‑3 math).

### 2.3 Instantiation and Scaling

**CRITICAL PARADIGM:** Math Cores are **instantiable templates**, not fixed resources.

The **18-instance baseline** (`_INSTANCE_COUNT = 18`) represents **one instantiation pattern**, not a hard limit. Implementations SHOULD:

- **Spawn Math Cores dynamically** based on GPU capacity and workload:
  - Query GPU hardware (SM count, VRAM, warp capacity)
  - Calculate max concurrent cores: `sm_count × cores_per_sm`
  - Instantiate on-demand, deallocate when idle

- **Scale to GPU hardware limits**:
  - **Consumer GPUs** (RTX 3070): 46 SMs → 460+ concurrent cores
  - **Enthusiast GPUs** (RTX 4090): 128 SMs → 1,280+ concurrent cores
  - **Datacenter GPUs** (H100): 132 SMs → 2,640+ concurrent cores

- **Resource allocation per core**:
  - Stack state: 69 lines × 4 bytes = 276 bytes per core
  - Metadata: ~2 KB per core (instance ID, tier, history)
  - **Total overhead negligible**: 10,000 cores = 22 MB

- **Dynamic lifecycle management**:
  - Spawn cores on first request (lazy instantiation)
  - Pool idle cores for reuse (avoid allocation overhead)
  - Deallocate after timeout (prevent memory fragmentation)
  - Monitor GPU utilization, scale up/down dynamically

**Architectural Constants (Per Core Instance):**
- `STACK_DEPTH = 69` (Tesla 6-9 resonance)
- `TERNARY_OPS = {SIGN, TQUANT, TCMP}` (Setun heritage)
- `MAX_PROGRAM_LENGTH = 69` instructions

**Scaling Philosophy:**
- **Tier-1 (Simple):** 66% of cores for high-frequency operations
- **Tier-2 (Mid):** 22% of cores for moderate complexity
- **Tier-3 (High):** 11% of cores for chaotic/quantum systems

From the perspective of higher layers (swarm, Reality Enabler, compression), a "math core" is a **single logical service** that internally fans out to simple/mid/high RPN workers and **scales horizontally** to match GPU parallelism.

---

## 3. Math Core and Existing Modules

This spec stitches together several existing modules as views on the same math fabric:

- **Core scalar/vector engine**:
  - `ModularRPNEngine` (this spec) – 18 instances, 69-instruction programs.

- **Tier‑3 math helpers**:
  - `RPNMathCore` (`rpn_math_core.py`) – wraps `AdvancedRPNEngine` to drive Tier‑3 opcodes (e.g., TRM matvec, vector norms, small matmul).

- **Opcode surface**:
  - `rpn_opcodes.py` – defines Tier‑1/Tier‑2/Tier‑3 opcodes, including:
    - basic math, clustering, matrix ops, quantum helpers,
    - programmable ops (branch/loop/store/recall, divergence/curl/laplacian),
    - procedural drawing and temporal reasoning.

Taken together, these define the **Math Core substrate**. The three-tier core hierarchy is a scheduling and routing discipline on top of this substrate.

### 3.1 Bridges and Tiers (Annotation)

Existing bridges already implement a simple/mid/high partition:

- **Tier‑1 / Simple core (worker‑worker)**:
  - `LightweightRPNEngine` (`bridges/lightweight_rpn.py`):
    - MAX_INSTANCES = 18, STACK_DEPTH = 69.
    - `SUPPORTED_OPS` is a small subset: basic arithmetic (`+ - * /`), elementary math (`sqrt exp log sin cos tan`), simple comparisons (`gt lt eq max min`), and basic stack ops (`dup swap drop`).
    - Target use: ultra‑fast, latency‑critical operations and very small programs.

- **Tier‑2 / Mid core (worker)**:
  - Standard sovereign RPN (exposed via `ModularRPNEngine` in `bridges/sovereign_bridges.py` and used from `ptx_runtime.modular_rpn_engine`):
    - Full Tier‑1 + most Tier‑2 opcodes (vector ops, clustering helpers, small matvec).
    - Target use: general workloads where 18 instances × 69 instructions is sufficient.

- **Tier‑3 / High core (master)**:
  - `AdvancedRPNEngine` (`bridges/advanced_rpn.py`) and `RPNMathCore` (`ptx_runtime/rpn_math_core.py`):
    - Extend the standard instance layout to track type/shape metadata and handle matrix/tensor ops via TRM‑style opcodes (`OP_TRM_*`, `OP_MATMUL_SMALL`, etc.).
    - Target use: high‑complexity routines (TRM integration, multi-step matmul/vec pipelines, symbolic‑like primitives).

- **Tier selection and routing**:
  - `TieredRPNEngine` (`bridges/tiered_rpn.py`) is the orchestrator:
    - Examines program opcodes and dispatches to Tier‑1, Tier‑2, or Tier‑3 (`LightweightRPNEngine`, standard sovereign RPN, `AdvancedRPNEngine`).
    - Maintains per‑instance `last_tier` and counters, effectively implementing the worker‑worker → worker → master pattern for a single logical math core.

These bridges are the concrete realization of the abstract math core tiers described in this spec.

### 3.2 Specialists and Adapters on the Math Core

Adaptive swarm specialists (as described in the Phase H documentation) sit **on top** of the math core:

- Each specialist uses math cores as its numeric execution layer:
  - Builds domain‑specific RPN programs (procedural libraries) over the opcode surface.
  - Calls into Tier‑1/2/3 engines via `TieredRPNEngine` or helper wrappers (e.g., `RPNMathCore`).
- Specialists also carry **adapter weights** (e.g., LoRA‑style adapters over TRM/router weights) and follow a **shadow copy → validate → commit** pattern:
  - Candidate adapter updates are evaluated using RPN‑driven metrics and ternary gates (worse/unknown/better).
  - Only validated adapters are promoted to “main” weights.

Thus, specialists self‑improve at two levels:
- **Procedural level**: evolving RPN programs and procedural stars stored in Galaxy/House.
- **Weight level**: adapter updates that alter how and when RPN programs are invoked.

The math core does not change—specialists are compositions and policies built on top of it.

---

## 4. FOV-Driven LOD for Both Clients

Math cores are part of K3D’s broader **LOD strategy** (see SPATIAL_UI_ARCHITECTURE_SPECIFICATION):

- **Human client LOD**:
  - Uses camera FOV/frustum and room-specific budgets to choose geometric detail (coarse/medium/full meshes, texture resolution).
  - Reality Enabler scenes follow the same pattern: distant simulations can be rendered as centroids or low-poly approximations while keeping high-detail states in Galaxy only.

- **AI client LOD (Matryoshka tiers)**:
  - Uses Matryoshka embeddings + PD04 programs to choose dimensionality (64/128/512/2048D) per query or simulation step.
  - Simple cores typically operate on ultrafast/fast tiers (64/128D); mid cores on 128/512D; high cores on 512/2048D when necessary.
  - Field-of-view for AI is expressed as spatial + semantic neighborhoods (query radius + k‑NN), then the math core chooses the appropriate tier for each operation.

Because both clients share the same nodes and Matryoshka tiers, “zooming in” (spatially or semantically) increases LOD in **both** geometry and math:
- Humans see more detailed meshes and textures.
- AI math cores switch to higher‑dimensional embeddings and richer programs.

## 5. Portability to Other Backends

While this spec describes a CUDA/PTX implementation, the **Math Core contract is backend-agnostic**:

- Normative parts:
  - RPN program structure (stack semantics, opcode meanings).
  - Opcode registry in `rpn_opcodes.py` as the canonical math vocabulary.
  - Tiered math core model (simple/mid/high) and Matryoshka/PD04 LOD behavior.

- Non-normative parts (example implementation):
  - Current use of PTX kernels and CUDA Driver API.
  - Concrete bridges (`LightweightRPNEngine`, `AdvancedRPNEngine`, `TieredRPNEngine`) and their performance characteristics.

Future backends (e.g., Vulkan compute, WebGPU, Metal, FPGA/ASIC implementations, or even transformer-style modules that emulate RPN semantics) **MAY** be used, provided they:
- implement the same RPN execution model and opcode semantics,
- preserve determinism guarantees where required (especially for PD04 decompression and Reality Enabler laws),
- honour the same tiering and Matryoshka LOD behavior from the perspective of higher-level specs.

This mirrors LLM diversity: the **storage and semantics specifications** (RPN programs, opcodes, Matryoshka embeddings) are stable; different engines can implement them as long as they respect the contract.

## 6. Tesla 3-6-9 and Setun Ternary Heritage

K3D Math Cores embody two foundational principles:

### 6.1 Tesla's 3-6-9 Harmonic Pattern

**Architectural Resonance:**
- **Stack Depth:** 69 lines (contains literal 6 and 9)
  - Digital root: 6 + 9 = 15 → 1 + 5 = 6
  - Product: 6 × 9 = 54 → 5 + 4 = 9
- **Instance Count:** 18 baseline (divisible by 3, 6, 9)
  - 18 / 3 = 6 (ternary resonance)
  - Digital root: 1 + 8 = 9
- **Ternary Base:** {-1, 0, +1} (three-state balanced logic)

**Design Philosophy:** Tesla observed that 3, 6, and 9 form the fundamental pattern of the universe. K3D honors this through stack depth, instance multiples, and ternary operations.

### 6.2 Setun Ternary Computer Legacy

**Historical Context:**
- Setun (USSR, 1958): World's only mass-produced ternary computer
- **Balanced ternary:** {-1, 0, +1} instead of binary {0, 1}
- Advantages: More efficient for signed arithmetic, natural zero representation
- Abandoned due to tooling/ecosystem, not technical limits

**K3D Resurrection:**
```
Setun Innovation          K3D Implementation
├─ Ternary logic          ├─ SIGN, TQUANT, TCMP opcodes
├─ Balanced representation├─ {-1, 0, +1} for physics directions
├─ Efficient arithmetic   ├─ Semantic clarity (charge, comparison)
└─ 50% fewer "trits"      └─ GPU-friendly PTX kernels
```

**Why Ternary for Physics:**
- **Direction:** Velocity signs, charge polarity, force vectors
- **Comparison:** Less than / Equal / Greater than (single opcode)
- **Classification:** Underdamped / Critical / Overdamped (natural encoding)
- **Semantic clarity:** {-1, 0, +1} maps directly to physical meaning

**Performance:**
- CPU: Ternary ops may be slower (binary hardware dominance)
- GPU: Ternary ops often faster (parallel three-way classification)
- **Trade-off:** Semantic clarity outweighs raw CPU speed for physics grounding

## 7. Conformance

An implementation conforms to the Math Core Specification v2.0 if:

1. It uses the shared opcode surface from `rpn_opcodes.py` as the canonical math vocabulary.
2. It exposes a `ModularRPNEngine`-compatible core (instantiable template, 69‑instruction programs per core).
3. It implements **dynamic core spawning** that scales to GPU hardware limits (not fixed at 18 instances).
4. It implements a simple/mid/high tiering strategy for math workloads, even if tiers share the same underlying PTX executor.
5. It documents which kernels / bridges (e.g., `TieredRPNEngine`, `AdvancedRPNEngine`, `RPNMathCore`) correspond to each tier in its deployment.
6. It implements **ternary operations** (SIGN, TQUANT, TCMP) following Setun balanced ternary semantics.
7. It honors **Tesla 3-6-9 resonance** in architectural constants (stack depth 69, instance multiples of 3/6/9).

**Version 2.0 Changes:**
- Added dynamic instantiation requirement (Section 2.3)
- Documented Tesla 3-6-9 and Setun ternary heritage (Section 6)
- Upgraded conformance to require scalable spawning (not fixed 18 cores)

Future revisions MAY add normative complexity heuristics, explicit routing API, and multi-GPU federation protocols.
