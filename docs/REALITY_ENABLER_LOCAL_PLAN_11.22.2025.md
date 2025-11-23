# Reality Enabler — Local System Plan (11.22.2025)

**Status**: Planning document (local PTX+RPN implementation)  
**Scope**: From current architecture → minimal, working local system (no Vulkan/WebGPU/Fog yet)

---

## 1. Phase Roadmap to a Local Working System

### Phase 0 — Baseline Sanity & Guardrails

- Run core tests locally (`pytest -q` or at least cranium/PTX suites).  
- Enforce sovereign flags: `K3D_PTX_STRICT=1`, `K3D_FORCE_PTX_FUSE=1`.  
- Treat vocabulary docs as constraints:
  - `MATH_CORE_SPECIFICATION.md`
  - `RPN_MATHEMATICAL_FOUNDATIONS.md`
  - `ADAPTIVE_PROCEDURAL_COMPRESSION_SPECIFICATION.md`
  - `REALITY_ENABLER_SPECIFICATION.md`

### Phase 1 — Math Core Hardening (Tiered RPN Stack)

- Normalize all math entry points onto `TieredRPNEngine`:
  - Tier‑1: `LightweightRPNEngine` (simple core, fast scalar/vector ops).
  - Tier‑2: standard sovereign RPN (via `ModularRPNEngine` bridge).
  - Tier‑3: `AdvancedRPNEngine` + `RPNMathCore` (matrix/TRM ops).
- Add tests that explicitly exercise:
  - Tier‑1 opcode subset and latency.
  - Tier‑2 scalar/vector math correctness.
  - Tier‑3 matrix/TRM paths via `RPNMathCore`.
  - Routing heuristics in `TieredRPNEngine` (programs that hit each tier).
- Declare this as the **Math Core** per `MATH_CORE_SPECIFICATION.md`.

### Phase 2 — PD04 + Matryoshka as Universal LOD

- Extend PD04 usage beyond text/character embeddings:
  - Simulation states (physics, chem, bio) must be representable as Matryoshka vectors with PD04 programs.
  - TRM embeddings must be compressible and reconstructable at 64/128/512/2048D.  
- Add tests:
  - PD04 round-trips for “simulation-like” vectors (e.g., small trajectories or field slices).
  - Fidelity thresholds verified in this new context.
- Document in code which components:
  - **Store** state via PD04 (House/Galaxy),
  - **Cache** transient embeddings,
  - **Use** PD04 as simulation LOD (Reality Enabler).

### Phase 3 — Minimal Physics Reality Enabler Demo

- Define a tiny **Physics Galaxy**:
  - Example: 2-body system (sun + planet) or pendulum.  
- Implement `reality_*` nodes:
  - `reality_atom` / `reality_system` for the chosen scenario.  
  - `visual_rpn`: simple body geometry + trajectory visualization using drawing opcodes/FractalEmitter.  
  - `behavior_rpn`: kinematic update as RPN (Euler or similar) over existing math opcodes.
- Wire to math cores:
  - Use `TieredRPNEngine` to execute step programs and write state back into Galaxy as Matryoshka+PD04 embeddings.
- Tests:
  - Unit: one simulation step matches a Python reference integrator (within tolerance).  
  - Integration: 100-step run completes under ~1s and produces coherent orbits/trajectories.

### Phase 4 — Minimal Chemistry & Biology Examples

- Chemistry:
  - Select a tiny subset (e.g., a few molecules from RDKit/QM9).  
  - Create `reality_atom` nodes for 2–3 elements (H, C, O).  
  - Create `reality_molecule` nodes (e.g., H₂O, CH₄) with `component_refs` to atoms/bonds.  
  - `behavior_rpn`: encode simple valence checks or a toy property (e.g., “is polar?”).
- Biology:
  - Implement an L-system/CA growth rule in Knowledge Gardens:
    - `visual_rpn`: L-system drawing for a small tree/branch structure.
    - `behavior_rpn`: growth update rule over discrete steps.
- Tests:
  - Chemistry: compare `behavior_rpn` valence/property outcomes to a small Python checker.  
  - Biology: verify growth sequence matches the chosen L-system for a few iterations.

### Phase 5 — House/Galaxy Integration + SleepTime Plausibility

- Integrate demos end‑to‑end:
  - Physics Lab room with live simulation from Physics Galaxy.  
  - Chemistry/Biology artifacts in Library/Knowledge Gardens.  
- Extend SleepTime:
  - During consolidation, call `law_rpn` / `behavior_rpn` for `reality_*` nodes to check:
    - basic physics/chem/bio plausibility,
    - whether scenes can be crystallized to House or must be quarantined/Museumed.  
  - Optionally lower Matryoshka tier on write while preserving invariants.
- Tests:
  - End-to-end SleepTime test where plausible scenes are persisted, impossible scenes are rejected or archived.

### Phase 6 — Viewer + Tablet Loop Closure

- Viewer:
  - Add a minimal “Reality Lab” entry point:
    - Avatar can see the physics demo running in a House room.  
    - Molecule and growth demos visualized in Library/Knowledge Gardens.  
  - Ensure viewer reads `extras.k3d` reality metadata and draws auxiliary visuals (orbits, bonds, growth overlays).
- Tablet:
  - Implement basic actions:
    - Start/stop simulation (invokes one or more RPN steps via math cores).  
    - Increase/decrease LOD (select Matryoshka tier) around focused nodes.  
- Smoke test:
  - Launch viewer + live server + minimal pipeline.
  - Walk avatar into Lab/Gardens; start simulations; see and verify updates.

### Phase 7 — Small Dataset Ingestion

- Curate **small** local datasets:
  - Physics: a short OpenFOAM/Bullet trace or hand‑crafted state sequences.  
  - Chemistry: ~100 molecules via RDKit/SMILES.  
  - Biology: one or two small growth or network examples.  
- Build ingestion tools:
  - Convert these slices into:
    - `reality_*` nodes,
    - PD04-compressed embeddings,
    - optionally training/evaluation pairs for later specialist fine‑tuning.  
- Tests:
  - Ingestion round-trip: run ingestion, then query Galaxy to confirm node counts, types, and a couple of `behavior_rpn` outputs.

### Phase 8 — Metrics & Stabilization

- Measure:
  - Latency for math cores (Tier‑1/2/3) on representative programs.  
  - Single‑tick simulation latency for the physics demo.  
  - Memory footprint for Galaxy with minimal Reality Enabler galaxies loaded.  
- Stabilize docs and examples:
  - Add a short “Reality Enabler Local Demo” guide in `docs/`.  
  - Ensure vocabulary specs and plan reflect the actual implementation state.
- Only after this: tackle **portability** (Vulkan/WebGPU) and **FOG deployment**:
  - e.g., math core over portals / remote Reality Enabler engines, keeping RPN+PD04+glTF contracts intact.

---

## 2. Cranium, RPN Core, and Specialists (Design Commit)

### 2.1 RPN Core + Atomic Knowledge as Sufficient Foundation

We commit to the following architectural stance:

- The **RPN math core + atomic knowledge (procedural stars)** are sufficient as the computation + storage substrate:
  - RPN provides the logic language (stack machine semantics, opcodes).  
  - PD04/Matryoshka + procedural glyph/star programs provide compressed, regenerable knowledge units.  
- The **Cranium** is treated as an orchestrator and adapter over this substrate:
  - It does not hide its own opaque “knowledge”; instead, it reads/writes Galaxy/House and manipulates RPN programs and PD04 embeddings.

The old “tripled fission/fusion multi-modal concept” remains conceptually valid, but is now grounded in:
- Math cores (Tier‑1/2/3) for all numeric work.  
- Procedural dual-program stars (visual_rpn + behavior_rpn) as the atomic knowledge units in the Galaxy/House.

### 2.2 Base Model + Specialists (Adapters) — Dual Self-Improvement

We adopt a **base-model + specialists** pattern, aligned with previous adaptive swarm work:

- **Base model**:
  - Holds the core reasoning logic and routing patterns (e.g., TRM + router, as specified in Phase H docs).  
  - Interprets multi-modal inputs and decides which specialists/math cores to invoke.  
  - Lives as a compact weight set plus a small RPN program library (e.g., common routines).

- **Specialists as adapters**:
  - Each specialist is:
    - A LoRA-style adapter (or similar low-rank/adapter head) over the base model’s weights, and  
    - A set of **RPN programs** and domain-specific procedures (e.g., procedural drawing, physics/chem/bio laws, compression schemes).  
  - Examples: OCR specialist, audio specialist, physics specialist, chemistry specialist, Reality Enabler domain specialists.

- **Dual self-improvement loops**:
  1. **RPN-level (procedural)**:
     - Specialists can refine / add new RPN programs and procedural stars (e.g., better physics integrator, refined drawing program).
     - Changes are immediately representable in Galaxy/House as updated programs and PD04 embeddings.
  2. **Weight-level (shadow copy adapters)**:
     - Specialists train via shadow adapters:
       - New LoRA/adapter weights are trained and validated against baselines.  
       - Ternary validation/halting gates (worse/unknown/better) decide whether to commit or discard updates.  
     - Shadow → main promotion only occurs when objective metrics show improvement.

This keeps execution and logic in sync:
- Execution paths live in the math core and RPN programs.  
- Logic lives both in:
  - explicit RPN procedures, and  
  - adapter weights guiding how/when to apply those procedures.

### 2.3 Impact on Documentation (Summary)

This plan assumes and reinforces:

- **RPN_MATHEMATICAL_FOUNDATIONS.md**: RPN is the core semantics; math cores and Reality Enabler use it as their execution language.
- **MATH_CORE_SPECIFICATION.md**: Math cores provide tiered, RPN-based execution; specialists call into math cores, not bespoke math code.
- **ADAPTIVE_PROCEDURAL_COMPRESSION_SPECIFICATION.md**: PD04/Matryoshka apply to all embeddings, including those used by Reality Enabler and TRM/specialists.
- **REALITY_ENABLER_SPECIFICATION.md**: Reality Enabler domain specialists are implemented as `reality_*` nodes + RPN programs, with math cores executing their laws.
- **CRANIUM_* docs**: Cranium acts as the coordinator/router and adapter host over this substrate (future edits will gradually align these docs to the RPN-centric implementation).

Future work will update individual documentation files as implementation converges, but this plan is the guiding contract:  
**Base model + adapters + procedural stars, with self-improvement at both RPN and weight levels, all grounded in the Math Core and Galaxy/House memory architecture.**

