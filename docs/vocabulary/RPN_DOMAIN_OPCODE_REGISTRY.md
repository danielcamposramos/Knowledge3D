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

## 5. Future Extensions (Placeholder)

Once the math core tiering and Reality Enabler galaxies are stable, we can consider **new opcodes** where justified by repeated patterns:

- Physics-specific shortcuts (e.g., `OP_FORCE_ACCUM`, `OP_INTEGRATE_EULER`), if profiling shows strong benefit.
- Chemistry-specific combinators (e.g., `OP_APPLY_REACTION_RULE`) built on top of set/graph ops.
- Biology-specific pattern operators (e.g., `OP_BRANCH_LSYSTEM`) for more compact growth programs.

Any new opcode MUST:
- Be definable as a composition of existing primitives (for verification).
- Demonstrate a measurable performance or clarity benefit vs programs alone.
- Be added to `rpn_opcodes.py` and documented here with domain semantics.

---

## 6. References

- `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`  
- `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`  
- `knowledge3d/cranium/ptx_runtime/rpn_math_core.py`  
- `docs/vocabulary/MATH_CORE_SPECIFICATION.md`  
- `docs/vocabulary/REALITY_ENABLER_SPECIFICATION.md`  
