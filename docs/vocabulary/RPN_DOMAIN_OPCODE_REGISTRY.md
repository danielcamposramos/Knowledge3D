# RPN Domain Opcode Registry — Physics, Chemistry, Biology (Initial Sketch)

**Version**: 0.1 (Draft, non-normative)  
**Status**: Design sketch (grounded in existing opcodes)  
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)  
**Date**: November 2025

---

## Abstract

This document sketches a **minimal, domain-oriented RPN opcode registry** for Reality Enabler domains (physics, chemistry, biology), grounded in the existing `rpn_opcodes.py` surface. It does not introduce new opcodes yet; instead, it:

- Groups existing opcodes into domain-specific roles.
- Identifies where physics/chemistry/biology semantics are constructed via **programs**, not new primitives.
- Provides a roadmap for future opcode extensions once the math core spec and Reality Enabler are stable.

---

## 1. Philosophy

1. **Programs before opcodes**: Prefer to build domain semantics as RPN programs over the existing math surface instead of immediately adding domain-specific opcodes.
2. **Shared math substrate**: Physics, chemistry, and biology share the same `rpn_opcodes.py` math core; Reality Enabler composes these into dual-program stars.
3. **Tiered execution via math cores**: Simple/mid/high math cores (see `MATH_CORE_SPECIFICATION.md`) decide where and how programs run, not which math is available.

---

## 2. Physics-Oriented Opcodes (Using Existing Surface)

Without introducing new opcodes, physics primitives can be described via:

- **Vector & matrix ops**:
  - `OP_VEC_L2_NORM`, `OP_VEC_NORMALIZE`, `OP_VEC_BLEND`
  - `OP_DOT_PRODUCT`, `OP_CROSS_PRODUCT`, `OP_OUTER_PRODUCT`
  - `OP_MATVEC_F32`, `OP_MATMUL_SMALL`

- **Calculus / field operators**:
  - `OP_DIVERGENCE`, `OP_CURL`, `OP_LAPLACIAN`

- **Temporal reasoning**:
  - `OP_TEMPORAL_COHERENCE`, `OP_TEMPORAL_MASK`, `OP_TEMPORAL_AGGREGATE`

- **Programmability**:
  - `OP_BRANCH`, `OP_LOOP`, `OP_STORE`, `OP_RECALL`, `OP_LIMIT`, `OP_SERIES_SUM`, `OP_SERIES_PRODUCT`.

**Example program roles** (to be stored in `behavior_rpn`):

- Kinematics update:
  - Use addition/multiplication, vector ops, and temporal aggregation to implement `x_{t+1} = x_t + v_t·Δt`, `v_{t+1} = v_t + a_t·Δt`.

- Force accumulation:
  - Use dot/cross/outer products and reductions to sum forces from neighbors and compute accelerations.

- Field evaluation:
  - Use `OP_DIVERGENCE`, `OP_CURL`, `OP_LAPLACIAN` over grid‑encoded states to approximate PDE operators (e.g., Navier–Stokes, Poisson).

In Reality Enabler terms, a **physics galaxy** star’s `behavior_rpn` is a composition of these existing ops, scheduled across simple/mid/high math cores depending on complexity.

### 2.1 Current Physics Implementations (Reference)

The following small systems in `knowledge3d/cranium/physics_demo.py` and `physics_galaxy.py` use this surface today:

- **ConstantAcceleration1D**:  
  - Law: `v_{t+1} = v_t + a·dt`, `x_{t+1} = x_t + v_{t+1}·dt`.  
  - RPN usage: scalar `+`/`*` via `ModularRPNEngine` (Tier‑1/2).

- **HarmonicOscillator1D**:  
  - Law: `x'' + ω²x = 0` rewritten as `v' = -ω²x`, `x' = v`.  
  - RPN usage: integration steps in RPN; state‑dependent acceleration computed in host, ready to be inlined into RPN if needed.

- **Orbital2D**:  
  - Law: `a = -μ r / |r|³` for central gravitational force in 2D.  
  - RPN usage: per‑component integration for `(vx, vy, x, y)`; radius/energy checks in tests demonstrate physical plausibility.

- **Heat1D**:  
  - Law: 1D heat diffusion `T_i^{n+1} = T_i^n + α·dt/dx²·(T_{i+1}^n − 2T_i^n + T_{i−1}^n)`.  
  - RPN usage: integration step delegated to RPN; stencil computed in host code, with future work to move more of the stencil math into pure RPN if desired.

- **Heat2D**:  
   - Law: 2D heat diffusion with 5-point stencil over a rectangular grid.  
   - RPN usage: similar pattern to Heat1D; the 2D Laplacian is assembled host-side, and the integration step `T_{i,j}^{n+1} = T_{i,j}^n + dT_{i,j}` is executed via scalar RPN programs per cell.

All of these reuse the shared RPN math surface; they are examples of how to encode ODEs and simple PDEs without adding new opcodes.

---

## 3. Chemistry-Oriented Opcodes (Via Math + Graph Programs)

Chemistry semantics can be expressed through:

- **Graph-like operations using existing math**:
  - Use `OP_SET_UNION`, `OP_SET_INTERSECTION`, `OP_SET_DIFFERENCE`, `OP_SET_CARTESIAN` for combinatorial aspects (bond possibilities, reaction templates).
  - Use `OP_CLUSTER_ASSIGN`, `OP_COSINE_SIM_BATCH` to cluster or compare molecular embeddings.

- **Linear algebra + nonlinearity**:
  - Property prediction and simple reaction scoring can be implemented as small MLP programs using `OP_MATVEC_F32`, `OP_VECTOR_RELU`, `OP_VECTOR_SIGMOID`.

- **Quantum-ish helpers**:
  - `OP_QUANTUM_SUPERPOSE`, `OP_QUANTUM_MEASURE`, `OP_QUANTUM_PHASE` and related ops can be used to approximate orbital-style reasoning when needed (without claiming physical accuracy).

**Role in Reality Enabler**:

- Atoms and functional groups:
  - `behavior_rpn` encodes valence and allowed bond patterns using set and comparison ops.

- Molecules and materials:
  - `behavior_rpn` uses vector/matrix ops and clustering to predict coarse properties (polarity, stability, simple reactivity classes).

Again, no new opcodes are needed for a first pass; chemistry semantics are programs on existing math, executed via math cores.

---

## 4. Biology-Oriented Opcodes (Growth, Networks, Evolution)

Biology relies on:

- **Fractal and geometric kernels** (already implemented via bridges):
  - `FractalEmitter` and drawing opcodes allow L‑system / fractal growth representation at the visual level.

- **Graph and clustering ops**:
  - `OP_CLUSTER_ASSIGN`, `OP_VEC_BLEND`, `OP_SET_*` ops to build and update connectivity graphs (e.g., neural networks, signaling pathways).

- **Temporal reasoning**:
  - `OP_TEMPORAL_COHERENCE`, `OP_TEMPORAL_AGGREGATE` to model stability and temporal integration.

- **Programmability opcodes**:
  - `OP_BRANCH`, `OP_LOOP`, `OP_STORE`, `OP_RECALL` for cell‑automaton rules and multi-step growth programs.

These tools are sufficient to encode:

- Growth rules (L‑systems + loops + branch conditions).
- Simple cellular automata for consolidation/pruning.
- Network formation and strengthening/weakening over time.

---

## 5. Capability Classification (A/B/C)

Every multimodal or domain feature proposal MUST be classified into one of three capability classes before implementation begins:

### Class A: Executable Now
- Can be expressed with current opcodes and bridges.
- Examples: drawing primitives, STORE/RECALL-driven temporal state, vector/matrix transforms, ternary routing, simple procedural animation.
- **Action**: Compose Galaxy recipes immediately. No new opcodes needed.

### Class B: Representable Now, Kernel Later
- Can be encoded immediately as Galaxy recipes or Grammar macros using existing opcodes.
- Deserves a dedicated opcode only after usage frequency and performance justify it.
- Examples: L-system expansion, spectrogram pipelines, triplanar mapping, mesh extrusion from 2D contours, boid update rules.
- **Action**: Encode as recipes/macros first. Monitor usage. Promote to opcode only when justified (see Section 6).

### Class C: Research / Not Yet Admitted
- Requires new data structures, too many host-side assumptions, or unproven performance.
- Examples: full volumetric remeshing pipelines, robust production-grade cloth/fluids, heavy differentiable rendering loops.
- **Action**: Document as research target. Do not encode or plan implementation until prerequisites are resolved.

**Rule**: Every new multimodal proposal must state which class it belongs to.

---

## 6. Opcode Admission and Promotion Pipeline

The principle from Section 1 ("programs before opcodes") is formalized here as a four-stage pipeline:

### Stage 0: Galaxy Recipe
- Store the technique as Grammar/Reality/Drawing/Audio knowledge only.
- No new opcode. Prove semantic usefulness first.

### Stage 1: Macro Surface
- Add a stable macro or token expansion in the high-level compiler (`ModularRPNEngine`).
- Examples: `STORE_X`, `RECALL_DISC`, future `SPECTROGRAM_MACRO`.
- Still no new kernel.

### Stage 2: Opcode Candidate
- Measure repeated usage across Galaxy recipes and tool-nodes (see `KNOWLEDGEVERSE_SPECIFICATION.md`).
- Profile cost of recipe/macro execution.
- Show that a dedicated opcode reduces complexity materially.

### Stage 3: PTX Kernel Admission
- Add kernel only after ALL of the following are satisfied:

| Criterion | Requirement |
|-----------|-------------|
| **Recipe exists** | Already expressed as composition of existing ops |
| **Frequency is high** | Appears in many tool-nodes or object recipes |
| **Speedup is meaningful** | Not 5%, but enough to matter architecturally |
| **Semantics are stable** | Same meaning across use cases |
| **Graph complexity reduced** | Fewer references, cleaner composition, easier TRM routing |
| **Test coverage** | Sovereignty tests + functional tests |
| **Sovereignty review** | Pure PTX, no CPU fallbacks, no external dependencies |

### Budget Control Recipes (Class B — Stage 0-1)

The **Adaptive Reasoning Budget** (see [ADAPTIVE_REASONING_BUDGET_SPECIFICATION.md](ADAPTIVE_REASONING_BUDGET_SPECIFICATION.md) §A.2) is expressed entirely as Class B recipes using existing opcodes — no new kernel admission required:

- **Budget state** stored in STORE/RECALL registers 60-67 (budget_total, budget_remaining, budget_min, ternary_signal, decomposition_depth, sub_task_count, priority, watermark_level)
- **Budget computation**: `OP_PUSH B_base → OP_PUSH 2 → OP_PUSH 1 → OP_RECALL signal → OP_SUB → OP_POW → OP_MUL` = B(q) = B_base × 2^(1−σ)
- **Budget enforcement**: `OP_RECALL budget_remaining → OP_PUSH 0 → OP_CMP → OP_BRANCH halt_if_zero`
- **Sub-task decomposition**: `OP_RECALL depth → OP_PUSH D_max → OP_CMP → OP_BRANCH decompose_or_serialize`

These recipes demonstrate the "programs before opcodes" principle: budget governance composes from existing RPN primitives. If profiling shows budget computation as a bottleneck (unlikely — budget logic runs once per TRM tick, not per-element), promotion to Stage 2 is warranted.

### Multimodal Target Opcodes (Stage 2 Candidates)

The following are valid design targets currently at Stage 0-1. They MUST pass the promotion pipeline before receiving PTX kernels:

**3D Generation Domain** (Stage 0):
- `OP_MESH_EXTRUDE`, `OP_BOOLEAN_UNION`, `OP_BOOLEAN_SUBTRACT`, `OP_BOOLEAN_INTERSECT`
- `OP_LSYSTEM_STEP`, `OP_MARCHING_CUBES`, `OP_NOISE_3D`

**Signal Domain** (Stage 0):
- `OP_FFT_FORWARD`, `OP_FFT_INVERSE`
- `OP_AUDIO_TO_SPECTROGRAM`, `OP_SPECTROGRAM_TO_AUDIO`
- `OP_FREQUENCY_FILTER`

**Image Generation Domain** (Stage 0):
- `OP_GRADIENT_BLEND`, `OP_TEXTURE_SAMPLE`, `OP_CONVOLUTION_2D`

**Temporal Domain** (Stage 0):
- `OP_TEMPORAL_SEQUENCE`, `OP_TEMPORAL_INTERPOLATE`, `OP_PHYSICS_INTEGRATE`

---

## 7. Ternary-Ready Register Semantics (March 2026)

The [Hyper-Parallel Processing](HYPER_PARALLEL_PROCESSING.md) paradigm (§6) establishes that all RPN register values MUST be representable with **value + confidence + polarity** — either natively (future ternary hardware: balanced ternary −1/0/+1) or by convention (current binary hardware: explicit encoding).

**Implications for RPN opcodes:**
- `OP_STORE` / `OP_RECALL` registers carry ternary-ready triples, not bare scalars
- Cross-core register sharing (hyper-parallel specialist swarm) uses the same STORE/RECALL mechanism
- "Uncertain / no evidence" (0 in balanced ternary) is a first-class value, not a sentinel or NaN
- When ternary hardware accelerators arrive, RPN programs migrate by changing the hardware mapping, not the opcode semantics

**Current implementation**: Binary encoding with explicit confidence fields. Future ternary hardware maps each trit natively.

---

## 8. Future Extensions

Once the math core tiering and Reality Enabler galaxies are stable, the promotion pipeline (Section 6) governs admission of new opcodes. Candidates include:

- Physics-specific shortcuts (e.g., `OP_FORCE_ACCUM`, `OP_INTEGRATE_EULER`), if profiling shows strong benefit.
- Chemistry-specific combinators (e.g., `OP_APPLY_REACTION_RULE`) built on top of set/graph ops.
- Biology-specific pattern operators (e.g., `OP_BRANCH_LSYSTEM`) for more compact growth programs.

All candidates MUST pass the Stage 0-3 pipeline documented above.

---

## 8. References

- `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`  
- `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`  
- `knowledge3d/cranium/ptx_runtime/rpn_math_core.py`  
- `docs/vocabulary/MATH_CORE_SPECIFICATION.md`  
- `docs/vocabulary/REALITY_ENABLER_SPECIFICATION.md`  
