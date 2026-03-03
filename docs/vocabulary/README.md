# Knowledge3D Vocabulary — Architecture Specifications

**Last Updated**: March 4, 2026
**Status**: Living documentation (updated as architecture evolves)

---

## Overview

This directory contains the **canonical architectural specifications** for Knowledge3D. Each specification defines a critical component of the system, providing detailed design, implementation guidance, and integration contracts.

**Reading Order for New Contributors:**
1. Start with [SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md](#spatial-general-intelligence-sgi) ⭐ **NEW: Foundational Goal**
2. Read [THREE_BRAIN_SYSTEM_SPECIFICATION.md](#three-brain-system) (foundational architecture)
3. Read [HYPER_MODULAR_ARCHITECTURE.md](#hyper-modular-architecture) (organizing paradigm)
4. Read [KNOWLEDGEVERSE_SPECIFICATION.md](#knowledgeverse) (runtime memory substrate)
5. Read [DUAL_CLIENT_CONTRACT_SPECIFICATION.md](#dual-client-contract) (human + AI duality)
6. Then explore domain-specific specs as needed

---

## Foundational Paradigm

### [SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md](SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md) ⭐ **NEW**
**Spatial General Intelligence (SGI) — The Goal of PM-KR/K3D**

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

**Companion**: [KNOWLEDGEVERSE_MVP_ROADMAP.md](KNOWLEDGEVERSE_MVP_ROADMAP.md) — Grounded implementation plan separating MVP from research

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
**Paradigm: Cross-Domain Procedural Composition**

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

**Comparison to Traditional Modular Programming:**
- Traditional: Functions compose within ONE domain (e.g., image processing → image processing)
- Hyper-Modular: Galaxy entries compose across ALL domains (e.g., Math Galaxy → Drawing Galaxy → Reality Galaxy)

**15-dimension comparison table** showing distinction from traditional modular, microservices, OOP, Unix pipes, knowledge graphs, TerraVision

**Production Status**: ✅ Foundational paradigm (publicly timestamped, defensible)

**Integration**: Organizing principle for ALL K3D specifications (cross-cutting concern)

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
**Visual Reasoning System**

Defines procedural visual primitives and the Drawing Galaxy.

**Key Concepts:**
- Drawing primitives: LINE, CIRCLE, RECT (as RPN programs)
- VectorDotMap: Universal audio ↔ visual codec
- ARC-AGI visual reasoning (46.7% validation)
- Procedural font rendering (Character Galaxy)

**Integration**: Drawing Galaxy, Character Galaxy (Knowledgeverse Region 2)

---

### [UNIFIED_SIGNAL_SPECIFICATION.md](UNIFIED_SIGNAL_SPECIFICATION.md)
**Audio + Visual Unification**

Defines cross-modal signal processing for synesthesia.

**Key Concepts:**
- Spectrograms as universal interface (audio → visual)
- VectorDotMap codec (shared between Audio/Visual galaxies)
- Sonification (visual → audio)
- Procedural generation (both modalities)

**Integration**: Audio Galaxy, Drawing Galaxy (Cross-Modal Bridge)

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
**RPN Instruction Set**

Canonical registry of all RPN opcodes across domains.

**Key Concepts:**
- Core operations (PUSH, POP, SWAP, DUP, etc.)
- Math domain (ADD, MUL, POW, SIN, etc.)
- Visual domain (LINE, CIRCLE, RECT, etc.)
- Galaxy domain (QUERY, COMPOSE, CREATE, etc.)
- Physics domain (VELOCITY, FORCE, INTEGRATE, etc.)

**Integration**: Cranium RPN VM, Sovereign PTX kernels

---

## Integration Map

```
HYPER_MODULAR_ARCHITECTURE (Organizing Paradigm — Cross-Cutting)
│   └── Applies to ALL regions (cross-domain composition, N-client reality, symlink deduplication)
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
