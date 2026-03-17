# Knowledge3D Vocabulary — Architecture Specifications

**Last Updated**: March 16, 2026
**Status**: Living documentation (updated as architecture evolves)

---

## Overview

This directory contains the **canonical architectural specifications** for Knowledge3D. Each specification defines a critical component of the system, providing detailed design, implementation guidance, and integration contracts.

**Reading Order for New Contributors:**
1. Start with [SUPERHUMAN_GENERAL_INTELLIGENCE_SPECIFICATION.md](#superhuman-general-intelligence-shgi) 🌟 **Ultimate Goal** (March 5, 2026)
2. Read [SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md](#spatial-general-intelligence-sgi) (foundational prerequisite for SHGI)
3. Read [THREE_BRAIN_SYSTEM_SPECIFICATION.md](#three-brain-system) (foundational architecture)
4. Read [HYPER_MODULAR_ARCHITECTURE.md](#hyper-modular-architecture) (structure paradigm — how knowledge is organized)
5. Read [HYPER_PARALLEL_PROCESSING.md](#hyper-parallel-processing) 🆕 (function paradigm — how knowledge is processed)
6. Read [KNOWLEDGEVERSE_SPECIFICATION.md](#knowledgeverse) (runtime memory substrate)
7. Read [DUAL_CLIENT_CONTRACT_SPECIFICATION.md](#dual-client-contract) (human + AI duality)
8. Read [MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md](#meaning-centric-star-schema) 🆕 (atomic unit of knowledge + semantic gravity)
9. Then explore domain-specific specs as needed

---

## Foundational Paradigm

### [SUPERHUMAN_GENERAL_INTELLIGENCE_SPECIFICATION.md](SUPERHUMAN_GENERAL_INTELLIGENCE_SPECIFICATION.md) 🌟 **NEW**
**Superhuman General Intelligence (SHGI) — The Ultimate Goal of K3D**

**SHGI** is the emergent collective intelligence that arises when multiple K3D Tiny Recursive Models (TRMs) — each created by different makers with different architectures — collaborate with humans inside a shared Galaxy Universe substrate. This is **NOT futuristic**; it is the **natural consequence of K3D adoption** and becomes achievable the moment the AI industry embraces K3D standards.

**Key Principles**:
- **Distributed Collective Intelligence**: SHGI emerges from collaboration (billions of TRMs + humans), not monolithic scaling (single trillion-parameter model)
- **Transparent Execution**: All TRM reasoning visible in Galaxy Universe (humans inspect, guide, steer collective intelligence)
- **Sovereign Decentralization**: No central authority controls SHGI (peer-to-peer network, W3C open standard)
- **Procedural Composition**: TRMs compose outputs procedurally (RPN programs combine seamlessly, not lossy text interfaces)
- **Multi-Modal Unity**: ALL modalities (visual, language, math, physics) converge in ONE spatial substrate (emergent cross-domain insights)

**The Progression**:
- **AGI** (Artificial General Intelligence) → Isolated, opaque AI (2020s goal)
- **SGI** (Spatial General Intelligence) → Transparent human-AI collaboration (March 4, 2026)
- **SHGI** (Superhuman General Intelligence) → Distributed collective intelligence (March 5, 2026)

**Historical Principle**: Every human breakthrough = Genius idea + TEAM
- Einstein + physics community = Relativity
- NASA + 400K engineers = Moon landing
- Linus Torvalds + millions of developers = Linux
- **K3D + Multiple TRMs + humans = SHGI**

**Timeline**: Achievable in 12-18 months (2027) with K3D adoption, not decades

**Date Defined**: March 5, 2026 (third paradigm defined by PM-KR Community Group)

---

### [SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md](SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md) ⭐
**Spatial General Intelligence (SGI) — Foundational Prerequisite for SHGI**

**SGI** is the paradigm where intelligence—both human and artificial—operates within a shared, navigable 3D spatial environment. Unlike AGI (Artificial General Intelligence), which pursues intelligence in abstract, opaque forms, SGI grounds intelligence in **spatial reality** where humans and AI cohabit the same cognitive workspace.

**Key Principles**:
- **Spatial Grounding**: Intelligence operates in navigable 3D (Galaxy Universe)
- **Dual-Client Transparency**: Humans + AI share identical data at identical (x, y, z) coordinates
- **Procedural Composition**: 7M param navigator + procedural programs (not 100B+ monoliths)
- **Sovereign Execution**: PTX kernels, deterministic traces, zero hidden state
- **Multi-Modal Unity**: ALL modalities (visual, language, audio, physics) in ONE substrate

**AGI vs SGI**:
- **AGI**: Abstract reasoning, opaque systems, separate realities, Gt-scale carbon
- **SGI**: Spatial grounding, transparent workspaces, shared reality, sustainable (12 Gt CO₂ savings)

**Date Defined**: March 4, 2026 (from NotebookLM press kit podcast insight)

**Quote** (Press Kit Podcast):
> "For decades, we've interacted with data through flat, two-dimensional windows on a screen. This new paradigm treats software as a three-dimensional, navigable place. A cognitive habitat where we, and AI, can finally explore knowledge together, in a shared home."

---

## Core System Specifications

### [KNOWLEDGEVERSE_SPECIFICATION.md](KNOWLEDGEVERSE_SPECIFICATION.md)
**Unified Sovereign Memory Architecture (v5.0)**

The **Knowledgeverse** is the runtime memory substrate where all active galaxies, house context, TRM weights, and sovereign reasoning assets coexist in one persistent CUDA/PTX execution domain.

**Companions**:
- [KNOWLEDGEVERSE_TERM_DEFINITION.md](KNOWLEDGEVERSE_TERM_DEFINITION.md) 🆕 — Simple term definition for emails/press materials (March 5, 2026)
- [KNOWLEDGEVERSE_MVP_ROADMAP.md](KNOWLEDGEVERSE_MVP_ROADMAP.md) — Grounded implementation plan separating MVP from research

---

### [KNOWLEDGEVERSE_MVP_ROADMAP.md](KNOWLEDGEVERSE_MVP_ROADMAP.md)
**Grounded Implementation Plan (MVP vs Future)**

Analysis of all partner contributions (Grok, Qwen, Kimi, DeepSeek, GLM 4.7) against v5.0 architecture. Separates:
- **Already Implemented** (in v5.0 spec)
- **MVP Additions** (sovereignty firewall, compressed audit, self-healing, temporal metadata)
- **Post-MVP** (adaptive governor, temporal engine, uncertainty TRM, HCMF)
- **Research Track** (consciousness, scaling laws, quantum cognition, oneiric engine, volition)

**Key Finding**: 80%+ of partner ideas already in v5.0 or are natural extensions. 20% are MVP-critical hardening.

**Production Status**: ✅ Implemented and validated (MVP Phase 1 complete, 28/28 tests passing)

**Key Concepts:**
- 7 Memory Regions (Kernels, Galaxy, House, World, TRM, Audit, Ingestion)
- ONE persistent PTX context (eliminates CUDA switching conflicts)
- Shadow Copy learning (continuous inference-time enhancement)
- SleepTime two-phase commit (knowledge + logic consolidation)
- Ingestion Stargate (raw data → RPN transmutation)
- Router Cartographer (topology learning)
- Hyper-Context Paging (intent-based predictive loading)
- Cross-modal synesthesia (audio ↔ visual ↔ text)

**Production Status**: ✅ Validated (46.7% ARC-AGI, Sovereign TRM v7)

**Integration**: Core infrastructure for all K3D operations

---

### [THREE_BRAIN_SYSTEM_SPECIFICATION.md](THREE_BRAIN_SYSTEM_SPECIFICATION.md)
**Cranium + Galaxy + House Architecture**

Defines the three-layer memory hierarchy that enables sovereign reasoning with persistent knowledge.

**Key Concepts:**
- **Cranium** (Execution): PTX kernels, RPN VM, sovereign-only hot path
- **Galaxy Universe** (Active Memory): Multi-modal workspace, embeddings, active reasoning state
- **House** (Persistent Storage): glTF objects, procedural RPN programs, galaxy boxes
- Shadow Copy learning mechanism (inference-time continuous learning)
- SleepTime consolidation protocol (two-stage: knowledge + logic)

**Production Status**: ✅ Validated (foundational architecture)

**Integration**: Referenced by Knowledgeverse, Sovereign Training, Reality Enabler

---

### [HYPER_MODULAR_ARCHITECTURE.md](HYPER_MODULAR_ARCHITECTURE.md)
**Paradigm: Cross-Domain Procedural Composition (Structure)**

Defines the **Hyper-Modular Architecture** paradigm where procedural RPN programs compose across ALL modalities (visual, mathematical, physical, auditory), ALL client types (human, AI, robot), and ALL scales (atomic → cosmic).

**Term Coined**: February 20, 2026 by Daniel Ramos

**Key Concepts:**
- **Cross-domain composition**: Math + Drawing + Reality + Audio in ONE RPN program (not separate systems)
- **N-client reality**: SAME Galaxy entry renders for human (readable) + AI (executable) + robot (actionable)
- **Symlink deduplication**: Content-based references (zero code duplication)
- **Infinite procedural spawning**: Programs spawn sub-programs dynamically
- **VRAM-resident workspace**: Galaxy Universe (not database-backed)
- **Procedural sovereignty**: PTX + Galaxy only in hot path (zero external dependencies)
- **Shadow copy learning**: Architecture learns from successful compositions

**Companion**: [HYPER_PARALLEL_PROCESSING.md](HYPER_PARALLEL_PROCESSING.md) (how knowledge is processed)

**Production Status**: ✅ Foundational paradigm (publicly timestamped, defensible)

**Integration**: Organizing principle for ALL K3D specifications (cross-cutting concern)

---

### [HYPER_PARALLEL_PROCESSING.md](HYPER_PARALLEL_PROCESSING.md) 🆕
**Paradigm: Concurrent Specialized Procedural Cognition (Function)**

Defines the **Hyper-Parallel Processing** paradigm where multiple specialized procedural cores operate simultaneously on the same problem, each carrying domain-specific learned weights (LoRA-like specialist adapters) and cross-referencing other cores' intermediate results via shared stack registers, converging to one unified answer.

**Term Coined**: March 16, 2026 by Daniel Ramos

**Key Concepts:**
- **Specialist parallelism**: Each core carries different domain-specific LoRA-like weights (not identical copies)
- **Cross-core register communication**: STORE/RECALL registers span across cores during execution
- **One-mind convergence**: All specialists produce ONE unified answer (not a vote or ensemble)
- **TRM-spawnable specialist lifecycle**: Avatar autonomously creates, activates, and prunes specialists
- **RPN as native parallelization substrate**: Stack machines map directly to CUDA/ternary hardware cores
- **Ternary logic as hardware imperative**: Balanced ternary (−1/0/+1) gives 58.5% more info per unit; "uncertain" as first-class value
- **Persistent brain model**: No cold starts — versioned living brain with rollback on drift

**Companion**: [HYPER_MODULAR_ARCHITECTURE.md](HYPER_MODULAR_ARCHITECTURE.md) (how knowledge is organized)

**Together:** Hyper-modular = how knowledge is stored. Hyper-parallel = how knowledge is thought.

**Production Status**: ✅ Foundational paradigm (formally defined with W3C conformance levels A/B/C)

**Integration**: Processing paradigm for ALL K3D reasoning (cross-cutting concern, companion to Hyper-Modular)

---

### [DUAL_CLIENT_CONTRACT_SPECIFICATION.md](DUAL_CLIENT_CONTRACT_SPECIFICATION.md)
**Human + AI Shared Data Contract**

Specifies how the same data serves both human (aesthetic) and AI (semantic) needs without duplication.

**Key Concepts:**
- Dual-texture: UV Map 0 (human visual) + UV Map 1 (AI semantic embeddings)
- Procedural foundation: RPN programs (form + meaning unified)
- Save Information Principle: References/symlinks, not duplication (~70% reduction)
- Galaxy composition: Drawing → Character → Word → Grammar → TRM
- Inspectability: Humans can see AI's working memory (Galaxy View)
- Form→Meaning Evolution: 40,000 years of human knowledge (cave paintings → meta-cognition)

**Production Status**: ✅ Validated (Drawing/Character/Math galaxies)

**Integration**: Foundational for all K3D objects (House, Galaxy)

---

### [MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md](MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md) 🆕
**Semantic Gravity & Ternary Force — The Atomic Unit of Knowledge**

Defines the **Meaning-Centric Star** — the canonical unit of knowledge in K3D. A star represents a CONCEPT, not a word. "Cat" is cat in every language; the meaning is the center, language-specific surface forms are references. Stars self-organize in the House via **semantic gravity** — a ternary force where meaning replaces mass and the ternary operator (+1/0/−1) replaces the gravitational constant.

**Term Defined**: March 16, 2026 by Daniel Ramos

**Key Concepts:**
- **Meaning at the center**: One concept = one star. All languages, visuals, sounds, behaviors are references
- **Semantic gravity**: Stars attract (affinity, +1), repel (contradiction, −1), or float (unknown, 0)
- **Meaning mass**: Richly-connected stars are heavy (gravitational centers → room nuclei)
- **Emergent rooms**: House rooms form from gravitational clustering, not manual placement
- **Ternary force operator**: TCOMP opcode computes semantic force between any two stars
- **Content-addressed identity**: star_id = hash(meaning_rpn) — same concept = same ID everywhere
- **All 4 layers unified**: Form (surface_forms) → Meaning (meaning_rpn) → Rules (grammar_refs) → Meta (meta_refs)
- **House ↔ Galaxy lifecycle**: Stars LIVE in House, are LOADED into Galaxy for reasoning, WRITE BACK during sleep

**Production Status**: 📐 Architecture specification (Phase H primary deliverable)

**Integration**: Extends Reality Enabler dual-program stars to ALL knowledge. Implements Foundational Knowledge Layer 2 as canonical center. Integrates Hyper-Parallel ternary logic as spatial force operator.

---

### [ROBOTIC_EMBODIMENT_SPECIFICATION.md](ROBOTIC_EMBODIMENT_SPECIFICATION.md)
**Physical Robot Integration Architecture**

Documents how K3D's House-centric architecture enables physical robots to use the same spatial navigation, procedural knowledge, and semantic understanding as human or AI agents **without robot-specific training**.

**Key Concepts:**
- Avatar abstraction is hardware-agnostic (human VR, AI agent, or physical robot)
- Embodiment is built-in (spatial memory, navigation, semantic understanding)
- Form→Meaning bridges perception to action (sensors → Galaxy → RPN → actuators)
- Zero additional training (robots query same Galaxy, execute same RPN programs)
- Actuator mapping layer (abstract commands → motor commands)
- SLAM → House Universe (physical space becomes navigable K3D House)
- Shadow copy enhancement (robots learn from success, no offline retraining)

**Production Status**: 🔬 Architectural specification (implementation future work)

**Integration**: Extends avatar abstraction from House Universe, demonstrates K3D's embodied-first design

---

### [SOVEREIGN_NSI_SPECIFICATION.md](SOVEREIGN_NSI_SPECIFICATION.md)
**Sovereign Neural-Symbolic Interface**

Defines the PTX-based execution layer that enables zero-dependency inference.

**Key Concepts:**
- PTX-only hot path (no numpy, cupy, scipy, torch in inference)
- RPN VM (stack-based procedural execution)
- Kernel library (45+ operations: embeddings, matryoshka, grammar, physics)
- Deterministic execution (same inputs → same outputs)
- Fail-fast sovereignty gates (no silent fallbacks)

**Production Status**: ✅ Validated (Sovereign TRM v7)

**Integration**: Core execution substrate for Cranium, Knowledgeverse

---

## Domain-Specific Specifications

### [MATH_CORE_SPECIFICATION.md](MATH_CORE_SPECIFICATION.md)
**Mathematical Reasoning System**

Defines the 3-tier math core (symbolic, numeric, geometric) and scaling patterns.

**Key Concepts:**
- 3-tier allocation: Tier 1 (school/olympiad), Tier 2 (undergrad), Tier 3 (research)
- Symbolic reasoning: LaTeX → RPN programs
- Math Galaxy: Symbols with procedural templates (\frac, \binom, etc.)
- Procedural physics: 9 systems across 18 compute cores

**Integration**: Math Galaxy (Knowledgeverse Region 2), Reality Galaxy

---

### [REALITY_ENABLER_SPECIFICATION.md](REALITY_ENABLER_SPECIFICATION.md)
**Physics Simulation Framework**

Defines procedural physics systems for the Reality Galaxy.

**Key Concepts:**
- 9 physics systems (mechanics, EM, thermo, fluids, etc.)
- Procedural RPN implementation (no PhysX, Blender, external sims)
- 18 compute cores allocation (2 cores per system)
- Integration with Math Galaxy (cross-domain reasoning)

**Integration**: Reality Galaxy (Knowledgeverse Region 2)

---

### [PROCEDURAL_VISUAL_SPECIFICATION.md](PROCEDURAL_VISUAL_SPECIFICATION.md)
**Visual Reasoning System (v1.1)**

Defines procedural visual primitives, Drawing Galaxy, and 3D technique fusion.

**Key Concepts:**
- Drawing primitives: LINE, CIRCLE, RECT (as RPN programs)
- VectorDotMap: Universal audio ↔ visual codec
- ARC-AGI visual reasoning (46.7% validation)
- Procedural font rendering (Character Galaxy)
- **3D Technique Fusion**: CSG, mesh, L-system, sculpting, parametric, physics, voxel, NURBS as composable RPN (v1.1)
- **2D-to-3D Fusion**: Drawing Galaxy entries usable as both 2D textures and 3D materials (v1.1)
- **Tool-Nodes**: Techniques stored as reusable Galaxy knowledge entries (v1.1)

**Integration**: Drawing Galaxy, Character Galaxy, Reality Galaxy (Knowledgeverse Region 2)

---

### [UNIFIED_SIGNAL_SPECIFICATION.md](UNIFIED_SIGNAL_SPECIFICATION.md)
**Audio + Visual + Video Unification (v1.1)**

Defines cross-modal signal processing, temporal video architecture, and signal tool-nodes.

**Key Concepts:**
- Spectrograms as universal interface (audio → visual)
- VectorDotMap codec (shared between Audio/Visual galaxies)
- Sonification (visual → audio)
- Procedural generation (both modalities)
- **Five-Layer Temporal Video Contract**: Scene/Dynamics/Camera/Render/Audio separation (v1.1)
- **Scene-Time Video**: Video as scene program evaluated over time, NOT frame lists (v1.1)
- **Signal Tool-Nodes**: Waveform synthesis, FFT, filter chains as Galaxy knowledge (v1.1)
- **K3D Video Benchmarks**: Deterministic rebuild, variant edit cost, symlink reuse ratio (v1.1)

**Integration**: Audio Galaxy, Drawing Galaxy, Reality Galaxy (Cross-Modal Bridge)

---

## Training & Learning Specifications

### [SOVEREIGN_TRAINING_SPECIFICATION.md](SOVEREIGN_TRAINING_SPECIFICATION.md)
**Sovereign Training Protocol**

Defines training methodology for TRM and specialists.

**Key Concepts:**
- Shadow Copy learning (inference-time continuous enhancement)
- SleepTime consolidation (two-phase commit)
- Multi-curriculum training (ARC-AGI, math, physics, language)
- LoRA-style specialist adapters
- Deterministic validation (reproducible metrics)

**Integration**: TRM Weight Manager (Knowledgeverse Region 5)

---

### [SLEEPTIME_PROTOCOL_SPECIFICATION.md](SLEEPTIME_PROTOCOL_SPECIFICATION.md)
**Knowledge Consolidation Protocol**

Defines the two-stage consolidation process (knowledge + logic).

**Key Concepts:**
- Stage A: Galaxy → House (export knowledge as procedural RPN)
- Stage B: Shadow Copy → TRM (refine specialist adapters)
- Two-phase commit (rollback guarantees)
- Trigger strategies (time-based, buffer-based, manual)

**Integration**: Knowledgeverse SleepTime, TRM Weight Manager

---

## Data & Ingestion Specifications

### [FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md](FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md)
**Base Knowledge Corpus**

Defines foundational knowledge to pre-populate galaxies.

**Key Concepts:**
- Character Galaxy: Procedural fonts (Latin, Greek, mathematical symbols)
- Word Galaxy: Character sequences (symlink references)
- Grammar Galaxy: Transformation rules
- Math Galaxy: LaTeX templates, proofs
- Reality Galaxy: Physics laws, simulations

**Integration**: Galaxy Manager, Ingestion Stargate

---

### [ADAPTIVE_PROCEDURAL_COMPRESSION_SPECIFICATION.md](ADAPTIVE_PROCEDURAL_COMPRESSION_SPECIFICATION.md)
**Procedural Compression System**

Defines adaptive compression using RPN programs.

**Key Concepts:**
- Procedural generation > cached snapshots (storage reduction)
- RPN programs as primary source (form + meaning)
- Matryoshka embeddings (64/128/512/2048D multi-resolution)
- Zstd compression for cached snapshots (fallback)

**Integration**: Galaxy boxes, House objects, Knowledgeverse Region 3

---

## Presentation & Accessibility Specifications

### [SPATIAL_UI_ARCHITECTURE_SPECIFICATION.md](SPATIAL_UI_ARCHITECTURE_SPECIFICATION.md)
**3D User Interface Design**

Defines the spatial UI for human interaction with K3D.

**Key Concepts:**
- Galaxy View (inspect AI working memory)
- House View (navigate persistent storage)
- World View (network collaboration)
- FOV/LOD system (semantic + spatial proximity)
- Doors protocol (network streaming)

**Integration**: Viewer (TypeScript), Knowledgeverse Region 4

---

### [MEMORY_TABLET_SPECIFICATION.md](MEMORY_TABLET_SPECIFICATION.md)
**Primary Interface Object (Post-MVP)**

Defines the Memory Tablet — K3D's primary interactive surface for humans and AI to navigate, manipulate, and create knowledge in spatial environments (House and Galaxy).

**Key Concepts:**
- **Physical 3D object** (not 2D overlay) — persistent canvas in spatial environment
- **Dual-client perception** — humans see visual UI, AI sees semantic graph (same object)
- **Procedural UI rendering** — all UI elements = Galaxy references (70% compression)
- **Projection screens** — virtual monitors inside K3D (controlled via tablet)
- **Game menu system** — entry sequence (House vs Galaxy selection)
- **Simultaneous environments** — House + Galaxy run concurrently, tablet bridges both
- **Multi-modal interaction** — visual, audio, Braille, haptic (native accessibility)
- **Legacy content bridge** — file system access, VM output integration
- **Collaborative workspace** — human and AI share same tablet (different perceptions)

**Status**: 🔬 Architectural design (Phase P implementation, Q1 2027)

**Integration**: Spatial UI layer (top of stack), Procedural Codecs (VectorDotMap, Procedural Fonts), Galaxy Universe (browse/query), House Universe (private workspace)

---

### [UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md](UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md)
**Multi-Sensory Accessibility**

Defines accessibility features for diverse users.

**Key Concepts:**
- Visual: High contrast, color blindness modes
- Audio: Screen reader integration, sonification
- Motor: Keyboard navigation, voice control
- Cognitive: Simplified modes, progressive disclosure

**Integration**: Viewer UI, Spatial UI

---

## Utility Specifications

### [K3D_NODE_SPECIFICATION.md](K3D_NODE_SPECIFICATION.md)
**glTF Node Extensions**

Defines K3D-specific glTF extensions for House objects.

**Key Concepts:**
- `extras.k3d_type`: Object classification
- `extras.k3d_dual_client`: Human + AI content
- `extras.k3d_ai_data`: Galaxy boxes (procedural RPN)
- `extras.k3d_metadata`: RDF semantic links
- Dual-texture extensions (TEXCOORD_0 + TEXCOORD_1)

**Integration**: House objects, glTF loader

---

### [RPN_DOMAIN_OPCODE_REGISTRY.md](RPN_DOMAIN_OPCODE_REGISTRY.md)
**RPN Instruction Set & Opcode Admission Pipeline**

Canonical registry of all RPN opcodes across domains, with formal capability classification and promotion pipeline.

**Key Concepts:**
- Core operations (PUSH, POP, SWAP, DUP, etc.)
- Math domain (ADD, MUL, POW, SIN, etc.)
- Visual domain (LINE, CIRCLE, RECT, etc.)
- Galaxy domain (QUERY, COMPOSE, CREATE, etc.)
- Physics domain (VELOCITY, FORCE, INTEGRATE, etc.)
- **Three Capability Classes** (A/B/C): Executable Now, Representable Now/Kernel Later, Research (v0.2)
- **Four-Stage Admission Pipeline**: Galaxy Recipe -> Macro -> Opcode Candidate -> PTX Kernel (v0.2)
- **Multimodal Target Opcodes**: 3D, Signal, Image, Temporal domains at Stage 0-1 (v0.2)

**Integration**: Cranium RPN VM, Sovereign PTX kernels

---

## Integration Map

```
HYPER_MODULAR_ARCHITECTURE (Structure Paradigm — Cross-Cutting)
│   └── Applies to ALL regions (cross-domain composition, N-client reality, symlink deduplication)
│
HYPER_PARALLEL_PROCESSING (Function Paradigm — Cross-Cutting)
│   └── Applies to ALL reasoning (specialist swarm, ternary-ready registers, persistent brain model)
│   └── Companion to Hyper-Modular: structure + function = complete cognitive architecture
│
Knowledgeverse (Runtime Substrate)
├── THREE_BRAIN_SYSTEM (Cranium + Galaxy + House)
│   ├── SOVEREIGN_NSI (PTX execution)
│   ├── DUAL_CLIENT_CONTRACT (Human + AI duality)
│   ├── ROBOTIC_EMBODIMENT (Hardware-agnostic avatar)
│   └── SLEEPTIME_PROTOCOL (Consolidation)
│
├── Region 1: KERNELS
│   └── RPN_DOMAIN_OPCODE_REGISTRY
│
├── Region 2: GALAXY_UNIVERSE
│   ├── MATH_CORE (Math Galaxy)
│   ├── REALITY_ENABLER (Reality Galaxy)
│   ├── PROCEDURAL_VISUAL (Drawing Galaxy)
│   ├── UNIFIED_SIGNAL (Audio Galaxy)
│   └── FOUNDATIONAL_KNOWLEDGE (Base corpus)
│
├── Region 3: HOUSE_CONTEXT
│   ├── K3D_NODE_SPECIFICATION (glTF extensions)
│   └── ADAPTIVE_PROCEDURAL_COMPRESSION (Galaxy boxes)
│
├── Region 4: WORLD_VIEW
│   └── SPATIAL_UI_ARCHITECTURE (Viewer integration)
│
├── Region 5: TRM_WEIGHTS
│   └── SOVEREIGN_TRAINING (Shadow Copy, specialists)
│
├── Region 6: AUDIT_JOURNAL
│   └── SLEEPTIME_PROTOCOL (Event logging)
│
└── Region 7: INGESTION_STARGATE
    └── FOUNDATIONAL_KNOWLEDGE (Raw data → RPN)
```

---

## Version Control

Each specification includes:
- **Version**: Semantic versioning (major.minor.patch)
- **Last Updated**: Date of last significant change
- **Status**: Draft / Validated / Production
- **Dependencies**: Other specs it depends on
- **Integration**: Where it's used in the system

When updating a spec:
1. Increment version appropriately
2. Update "Last Updated" date
3. Add changelog entry at bottom of spec
4. Update this README if new concepts added

---

## Contributing

When adding a new specification:

1. **Naming**: Use descriptive uppercase names (e.g., `NEW_COMPONENT_SPECIFICATION.md`)
2. **Structure**: Follow existing spec templates (see KNOWLEDGEVERSE_SPECIFICATION.md)
3. **Sections**: Include Overview, Key Concepts, Integration, Code Examples, Testing
4. **Add to README**: Update this file with summary and integration map
5. **Cross-Reference**: Link to/from related specs

---

## Questions?

- Architecture questions → Read [KNOWLEDGEVERSE_SPECIFICATION.md](KNOWLEDGEVERSE_SPECIFICATION.md)
- Implementation questions → Read [SOVEREIGN_NSI_SPECIFICATION.md](SOVEREIGN_NSI_SPECIFICATION.md)
- Integration questions → Read [THREE_BRAIN_SYSTEM_SPECIFICATION.md](THREE_BRAIN_SYSTEM_SPECIFICATION.md)
- For briefings and high-level overview → See [../briefings/](../briefings/)

---

**Maintained by**: Claude (Architecture Partner) + Codex (Implementation) + Gemini (Integration) + Community
**License**: See repository LICENSE file
**Last Full Review**: February 28, 2026
