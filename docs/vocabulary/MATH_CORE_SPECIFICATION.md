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

---

## 4. Conformance

An implementation conforms to the Math Core Specification if:

1. It uses the shared opcode surface from `rpn_opcodes.py` as the canonical math vocabulary.
2. It exposes a `ModularRPNEngine`-compatible core (18 instances, 69‑instruction programs).
3. It implements a simple/mid/high tiering strategy for math workloads, even if tiers share the same underlying PTX executor.
4. It documents which kernels / bridges (e.g., `TieredRPNEngine`, `AdvancedRPNEngine`, `RPNMathCore`) correspond to each tier in its deployment.

Future revisions MAY add a normative complexity heuristic and explicit routing API; for now, tiering is an architectural pattern rather than a hard-coded interface.

