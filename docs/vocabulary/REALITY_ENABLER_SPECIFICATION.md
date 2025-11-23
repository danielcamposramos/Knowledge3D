# Reality Enabler Specification — Universal Procedural Fabric

**Version**: 1.0  
**Status**: Draft (Post-Atomic Breakthrough)  
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)  
**Date**: November 2025

---

## Abstract

The **Reality Enabler** extends K3D from static spatial knowledge representation to **procedural reality understanding**. Instead of wrapping external simulators (physics, chemistry, biology, and other domains), K3D **transmutes** their behavior into sovereign PTX kernels and dual-program stars. Each reality node stores:

- `visual_rpn`: how the object appears in 3D (form)
- `behavior_rpn` / `meaning_rpn`: how it behaves under domain laws (meaning)
- Matryoshka embeddings + PD04 programs: compressed state for search and LOD

This specification defines how Reality Enabler galaxies, nodes, and procedures integrate with:
- The **Three-Brain System** (Cranium / Galaxy / House),
- The **Dual-Client Contract** (human vs AI perception),
- The **Spatial UI Architecture** (rooms, Galaxy Universe),
- The **Adaptive Procedural Compression** stack (PD04 + ternary codecs).

---

## 1. Motivation and Scope

### 1.1 From Integration to Transmutation

Traditional plans treated K3D as a wrapper around external simulators (e.g., Bullet, OpenFOAM, GROMACS). Reality Enabler adopts a stricter sovereign position:

- External engines and datasets are used **offline** to inspire and train
  domain-specialist kernels and RPN programs.
- At runtime, only **sovereign PTX kernels** and **procedural programs** execute.

**Scope of this spec**:
- Physics (rigid/soft bodies, simple fluids)
- Chemistry (atoms, molecules, materials, basic reactions)
- Biology (growth patterns, networks)
- Extensible to any computable domain (weather, astro, economics) using the same fabric.

### 1.2 Design Principles

1. **Procedural-First**: Laws and behaviors are programs (`meaning_rpn`), not tables.
2. **Stacked Galaxies**: Higher-level realities are compositions (symlinks) of lower-level atoms.
3. **Matryoshka Simulation**: Same embedding tiers (64–2048D) serve as simulation LOD.
4. **Dual-Client Compliance**: Humans and AI see/execute the same stars.
5. **Sovereign Execution**: All runtime computation uses PTX kernels; external engines never run in hot paths.

---

## 2. Reality Enabler Galaxies

Reality Enabler extends the **Galaxy Universe** (see SPATIAL_UI_ARCHITECTURE_SPECIFICATION §4) with domain/specialist galaxies whose stars are procedural simulations.

### 2.1 Galaxy Types

- **Physics Galaxy**:
  - Stars: forces, constraints, integrators, simple bodies, fields.
  - Behavior: `behavior_rpn` executed via `ModularRPNEngine`, `VectorResonator`, `GraphCrystallizer`, and `WorldModelBridge` (dynamic meshes, temporal coherence).
  - Visuals: `visual_rpn` + drawing kernels (`rpn_executor.ptx`, `FractalEmitter`) render orbits, trajectories, fields.
  - Current seeds (local demo layer):
    - 1D constant acceleration and 1D harmonic oscillator (`ConstantAcceleration1D`, `HarmonicOscillator1D` in `knowledge3d/cranium/physics_demo.py`).
    - 2D central-force orbit (`Orbital2D` in `physics_demo.py`).
    - 1D and 2D heat diffusion (`Heat1D`, `Heat2D` in `physics_demo.py`).
    - Procedural storage via `PhysicsGalaxy` (`knowledge3d/cranium/physics_galaxy.py`) using PD0x/ProceduralGalaxy as a physics‑specific galaxy root.

- **Chemistry Galaxy**:
  - Stars: `reality_atom` (H, O, C, …), `reality_molecule`, `reality_material`.
  - Composition: `component_refs` symlink to lower-tier nodes (atoms/bonds); no duplication of embeddings.
  - Behavior: valence rules, simple reactions, property predictors in `behavior_rpn`, trained from tools like RDKit/OpenMM/GROMACS but distilled into sovereign programs.

- **Biology / Growth Galaxies**:
  - Stars: growth rules (L-systems, cellular automata), tissues, organs, networks.
  - Behavior: `meaning_rpn` expresses growth and interaction with environment (resources, signals).
  - Execution: `FractalEmitter` and `GraphCrystallizer` implement spatial/graph evolution; `TemporalReasoning` handles time-series effects.

- **Other Domain Galaxies** (optional):
  - Weather, astrophysics, economics, etc., follow the same pattern: define reality atoms, stacks, and behavior programs.

### 2.2 Stacked Compositional Galaxies

Reality Enabler galaxies mirror the existing text stack (letters → words → phrases):

- **Atoms → Molecules → Materials → Scenes**:
  - Atoms: `reality_atom` nodes (physics/chem/bio).
  - Molecules/parts: `reality_molecule` nodes with `component_refs` to atoms/bonds.
  - Materials/organs/subsystems: `reality_material` / `reality_system` nodes with `component_refs` to molecules/parts.
  - Scenes/experiments/worlds: top-level stars referencing many subsystems and a `law_rpn` that orchestrates them.

- **Symlink-Style Composition**:
  - Higher tiers MUST refer to lower tiers via references (`component_refs`, `letter_refs`, `word_refs`) instead of copying embeddings or programs.
  - Updating an atomic star (e.g., mass of Hydrogen, a stroke primitive) automatically improves all dependent composites.

### 2.3 Matryoshka Simulation LOD

Reality Enabler reuses the **Adaptive Procedural Compression** spec:

- PD04 programs + Matryoshka tiers `{64, 128, 512, 2048}` encode:
  - 64D: coarse plausibility (is this stable/valid?).
  - 128/512D: structural/Newtonian behavior (rigid bodies, basic reactions, coarse growth).
  - 2048D: high fidelity (fluids, soft bodies, detailed pathways).
- These embeddings are **regenerable from procedures** and serve as:
  - Search keys (find “similar” behaviors or states),
  - LOD selectors for simulation (ternary/Matryoshka router chooses tier based on focus/budget).

The ternary codec and Matryoshka logic (see ADAPTIVE_PROCEDURAL_COMPRESSION_SPECIFICATION) underlie PD04 programs and ensure lightning-fast encode/decode of simulation states and laws on GPU.

---

## 3. Reality Nodes — Dual-Program Stars

### 3.1 Node Types

Reality Enabler extends the K3D Node archetypes (see K3D_NODE_SPECIFICATION §1.3) with:

- `reality_atom` — indivisible domain primitives (e.g., H, gravitational point mass, cell type)
- `reality_molecule` — structured composites (molecules, mechanisms, small subsystems)
- `reality_material` — bulk/continuum-level entities (materials, tissues, fields)
- `reality_system` — full systems (experiments, scenes, organisms, weather cells, orbital systems)

### 3.2 Required Fields

Every reality node MUST provide:

- **Semantic Identity**:
  - Domain invariants (mass, charge, valence, phase, lattice type, organism class, etc.).
  - Units and reference frames made explicit in metadata.

- **Procedural Programs**:
  - `visual_rpn`: how to draw/render at each LOD (atoms as spheres/clouds, bonds as cylinders, fields as volumes/lines).
  - `behavior_rpn` / `meaning_rpn`: how the object behaves under domain laws:
    - physics: force accumulation, simple integrators, constraints;
    - chemistry: reaction templates, bond changes, phase changes;
    - biology: growth/decay, specialization, signaling.
  - Optional `law_rpn`: localized constraints (e.g., conservation, stability bounds) used during SleepTime/House materialization.

- **Compositional References**:
  - `component_refs`: pointers to lower-tier nodes (atoms, molecules, parts).
  - For mixed scenes, nodes MAY also reference text/diagram nodes (e.g., formula diagrams; see drawing grammars galaxy).

- **Embeddings & Compression**:
  - Matryoshka tiers `{64, 128, 512, 2048}` regenerated from procedures and encoded as PD04 programs.
  - PD04 programs themselves MAY be stored in `extras.k3d` alongside RPN code; ternary kernels handle on-GPU decode.

### 3.3 Execution Model

At runtime:

- Human clients:
  - Render `visual_rpn` results as geometry via the spatial UI (Library, Workshop, Bathtub).
  - Perceive simulations as evolving 3D scenes (or diagrams).

- AI clients:
  - Execute `behavior_rpn` / `meaning_rpn` via:
    - `ModularRPNEngine` (core RPN VM),
    - `VectorResonator` (vector blending, similarity, field composition),
    - `GraphCrystallizer` (constraint and neighborhood aggregation),
    - `TemporalReasoning` and `WorldModelBridge` (time evolution, dynamic mesh updates),
    - `GalaxyResonanceEngine` and `GalaxyMemoryUpdater` (state writing to Galaxy).
  - Use PD04 programs + ternary codecs to load/store simulation states at appropriate LOD.

Dual-client contract: both clients operate on the same node data; the only difference is how they view/execute it.

---

## 4. Integration with the Three-Brain System

### 4.1 Cranium (Reasoning)

- Hosts the PTX kernel suite that executes reality RPN programs:
  - RPN execution (`ModularRPNEngine`),
  - vector/field blending (`VectorResonator`),
  - graph aggregation (`GraphCrystallizer`),
  - temporal/world modeling (`WorldModelBridge`, `TemporalReasoning`).
- Uses ternary/Matryoshka logic to select simulation fidelity given:
  - user focus (e.g., avatar gaze, room context),
  - GPU budget,
  - Error tolerances from `law_rpn`.

### 4.2 Galaxy (Active Reality Memory)

- Stores reality stars in their respective galaxies (physics, chemistry, biology, etc.).
- Maintains:
  - current simulation states (embeddings + PD04 programs),
  - structural graphs (who interacts with whom),
  - spatial positions (where in the Galaxy Universe reality nodes live).
- Cross-galaxy queries (see SPATIAL_UI_ARCHITECTURE_SPECIFICATION §4.4) can mix reality galaxies with text/visual/audio/reasoning galaxies (e.g., “find molecules similar to this and scenes where they appear”).

### 4.3 House (Persistent Reality Artifacts)

- SleepTime consolidation (see SLEEPTIME_PROTOCOL_SPECIFICATION) MUST:
  - validate candidate scenes with Reality Enabler specialists using `law_rpn` (basic physics/chem/bio plausibility checks),
  - crystallize validated scenes as House GLBs (rooms, objects, Knowledge Gardens artifacts),
  - optionally archive rejected or experimental artifacts to Museum.
- Reality artifacts are standard House objects:
  - a “Lab” room with running experiments,
  - a “Material Library” shelf with stable materials,
  - a “Cell Garden” representing growth experiments.

---

## 5. Dual-Client Contract Implications

Reality Enabler reinforces the Dual-Client Contract:

- Every simulation visualized in the House or Galaxy MUST:
  - correspond to a reality node with executable `behavior_rpn`,
  - allow AI to re-run or approximate the same behavior on demand.
- Action buffers (288-byte contract) MAY include simulation intents:
  - “start/stop simulation in this region,”
  - “increase LOD around this node,”
  - “materialize stable snapshot into House.”
- Human-visible anomalies (e.g., impossible geometry) SHOULD be traceable back to:
  - a specific set of reality nodes and `law_rpn` programs,
  - enabling debugging and refinement of domain specialists.

---

## 6. Conformance

An implementation conforms to the Reality Enabler Specification if it:

1. Implements at least one **Reality Enabler galaxy** (physics, chemistry, or biology) with dual-program stars and Matryoshka+PD04 embeddings.
2. Represents reality objects as compositional `reality_*` nodes with:
   - `visual_rpn`, `behavior_rpn` / `meaning_rpn`,
   - `component_refs` to lower-tier nodes (symlink-style),
   - regenerated Matryoshka embeddings encoded via PD04 programs.
3. Executes reality programs using sovereign PTX kernels (no external engine calls in hot paths).
4. Exposes simulations through the Dual-Client Contract:
   - humans see evolving geometry in House/Galaxy,
   - AI executes the same programs via Cranium kernels.
5. Integrates with SleepTime consolidation so that:
   - physically/chemically/biologically plausible scenes are crystallized to House,
   - invalid scenes are either corrected or archived.

### 6.1 Backend Portability and Example Implementation

This specification is intentionally **storage- and semantics-first**:

- Normative:
  - Node shapes (`reality_*` types, `visual_rpn`, `behavior_rpn`/`law_rpn`, `component_refs`).
  - Galaxy structure (specialist galaxies, stacked composition, Matryoshka+PD04 usage).
  - Dual-client contract (same stars for humans and AI).
- Non-normative:
  - The current CUDA/PTX implementation of math cores and bridges.

Other engines and backends (e.g., Vulkan compute, WebGPU, Metal, CPU SIMD, or even transformer-based modules that implement the same `behavior_rpn` semantics) MAY be used, as long as they:
- respect the RPN and opcode semantics defined by the Math Core and RPN foundation docs,
- preserve determinism guarantees for laws and PD04 decompression where required,
- continue to honour the dual-client and Three-Brain contracts.

K3D’s PTX+RPN stack is therefore a **reference implementation**, not a constraint on the standard. The standards live in the glTF+`extras.k3d` schemas, Matryoshka/PD04 programs, and RPN law semantics; multiple heterogeneous engines can coexist above that substrate, just as multiple LLM architectures coexist above common data formats today.

---

## 7. References

- `docs/vocabulary/SPATIAL_UI_ARCHITECTURE_SPECIFICATION.md`  
- `docs/vocabulary/K3D_NODE_SPECIFICATION.md`  
- `docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md`  
- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md`  
- `docs/vocabulary/ADAPTIVE_PROCEDURAL_COMPRESSION_SPECIFICATION.md`  
- `docs/vocabulary/MATH_CORE_SPECIFICATION.md`  
- `docs/Reality_Enabler.md`  
