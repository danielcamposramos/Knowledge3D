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

Implementations MAY:

- Instantiate **multiple math core pools**:
  - e.g., 2× high, 4× mid, 12× simple cores mapped to the 18 instance slots and/or multiple GPU streams.
  - Use routing logic (e.g., based on expression complexity, length, or required opcodes) to dispatch each RPN program to an appropriate core tier.

- Use **complexity heuristics** (e.g., estimated FLOPs, opcode types, vector sizes) to:
  - keep simple operations on simple cores for minimal latency,
  - reserve high cores for TRM integration, symbolic ops, or Reality Enabler heavy physics/chem/biological routines.

From the perspective of higher layers (swarm, Reality Enabler, compression), a “math core” is a **single logical service** that internally fans out to simple/mid/high RPN workers.

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

This mirrors LLM diversity: the **storage and semantics standards** (RPN programs, opcodes, Matryoshka embeddings) are stable; different engines can implement them as long as they respect the contract.

## 6. Conformance

An implementation conforms to the Math Core Specification if:

1. It uses the shared opcode surface from `rpn_opcodes.py` as the canonical math vocabulary.
2. It exposes a `ModularRPNEngine`-compatible core (18 instances, 69‑instruction programs).
3. It implements a simple/mid/high tiering strategy for math workloads, even if tiers share the same underlying PTX executor.
4. It documents which kernels / bridges (e.g., `TieredRPNEngine`, `AdvancedRPNEngine`, `RPNMathCore`) correspond to each tier in its deployment.

Future revisions MAY add a normative complexity heuristic and explicit routing API; for now, tiering is an architectural pattern rather than a hard-coded interface.
