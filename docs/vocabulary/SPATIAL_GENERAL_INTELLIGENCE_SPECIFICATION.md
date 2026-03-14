# Spatial General Intelligence (SGI) Specification

**Version**: 1.0
**Status**: Foundational Concept (K3D Canonical Vocabulary)
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)
**Date**: March 4, 2026

---

## Abstract

**Spatial General Intelligence (SGI)** is the paradigm where intelligence—both human and artificial—operates within a shared, navigable 3D spatial environment. Unlike Artificial General Intelligence (AGI), which pursues intelligence in abstract, opaque forms, SGI grounds intelligence in **spatial reality** where humans and AI cohabit the same cognitive workspace, enabling transparent, verifiable, and collaborative reasoning.

This specification defines SGI as the foundational goal of PM-KR (Procedural Memory Knowledge Representation) and K3D (Knowledge3D), positioning it as the **web-native evolution** of general intelligence.

---

## 1. Introduction

### 1.1 The AGI Paradigm (Traditional Approach)

**Artificial General Intelligence (AGI)** pursues human-level intelligence through:
- **Abstract reasoning**: Intelligence divorced from spatial grounding
- **Opaque systems**: Black-box neural networks (humans cannot inspect)
- **Separate realities**: AI processes embeddings, humans see dashboards (no shared truth)
- **Massive duplication**: Same knowledge stored redundantly across training data, model weights, inference caches
- **Unsustainable compute**: 100B+ parameters, Gt-scale carbon footprint

**Result**: AGI = powerful but unverifiable, energy-intensive, and fundamentally separate from human cognition.

---

### 1.2 The SGI Paradigm (Spatial Approach)

**Spatial General Intelligence (SGI)** achieves general intelligence through:
- **Spatial grounding**: Intelligence operates in navigable 3D environment (Galaxy Universe)
- **Transparent systems**: Humans + AI see identical procedural nodes at identical (x, y, z) coordinates
- **Shared reality**: Dual-Client Contract ensures both clients consume same data (verifiable identity)
- **Zero duplication**: Procedural memory (store recipe once, referenced everywhere)
- **Sustainable compute**: 7M parameter core + procedural composition (not 100B+ monoliths)

**Result**: SGI = verifiable, energy-efficient, and fundamentally collaborative (humans + AI share cognitive habitat).

**The Reverse Analogy**: The tech industry borrowed spatial metaphors (windows, desktop, folders, doors, addresses, rooms) and flattened them into 2D. SGI reverses this — builds ACTUAL spatial reality where those metaphors become literal. K3D is the pinnacle: uniting ALL knowledge representation, game engines, computer history, and network architecture into one spatial system. The House IS a memory palace (Method of Loci, 40,000 years old). The Galaxy IS the brain. The avatar IS the AI entity.

---

## 2. Core Principles of SGI

### 2.1 Spatial Grounding Principle

**Definition**: Intelligence MUST operate within a navigable 3D spatial environment where knowledge has (x, y, z) coordinates.

**Rationale**:
- **Human cognition is spatial**: We think in rooms, places, journeys (method of loci, 40,000 years old)
- **AI spatial processing**: Attention mechanisms naturally map to spatial proximity (transformers → spatial navigation)
- **Shared grounding**: 3D space provides common reference frame for humans and AI

**Implementation** (K3D):
- **Galaxy Universe**: Unified VRAM workspace where ALL knowledge lives (Drawing, Character, Word, Grammar, Math, Reality galaxies)
- **TRM IS the Avatar**: 7M parameter entity that LIVES in the House (Memory Palace) and THINKS inside the Galaxy (Internal Brain). Runs as a continuous game loop (`trm_step_fused.ptx`), not as a function Python calls. Has an internal swarm of nine parallel cognitive channels ("superdotados" model — how gifted individuals think).
- **Spatial queries**: Avatar navigates to knowledge at (x, y, z) coordinates (same mechanism humans use to "look" at objects)
- **K3D is NOT a program you run**: It is a living, always-on, embodied AI that perfects itself during idle time (sleep-time consolidation)

**Counter-example** (AGI):
- Knowledge embedded in model weights (no spatial coordinates)
- Humans see post-hoc explanations (not AI's internal state)
- No shared grounding (AI "thinks" in 1024-dim embeddings, humans "see" in pixels)

---

### 2.2 Dual-Client Transparency Principle

**Definition**: Humans and AI MUST consume identical data structures, differing only in representation modality (visual vs. procedural).

**Rationale**:
- **Verifiability**: Users can inspect what AI is "looking at" (same K3D nodes)
- **Trust**: No hidden state (AI memory IS the external 3D world)
- **Collaboration**: Humans can point to spatial locations, AI navigates to same coordinates

**Implementation** (K3D):
- **Dual-Client Contract**: Human Avatar sees glTF geometry (UV Map 0), Synthetic User executes RPN programs (UV Map 1)
- **Node Identity**: Both clients query identical K3D Node IDs at identical (x, y, z) positions
- **Action Transparency**: All AI operations spatially grounded (observable in 3D workspace)

**Counter-example** (AGI):
- AI processes hidden embeddings (opaque)
- Humans see dashboards/graphs (post-hoc reconstruction)
- No guarantee of identity between "what AI sees" and "what human sees"

**Source**: [docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md](DUAL_CLIENT_CONTRACT_SPECIFICATION.md)

---

### 2.3 Procedural Composition Principle

**Definition**: Intelligence emerges from COMPOSING procedural programs (not storing massive parameter counts).

**Rationale**:
- **Efficiency**: 7M parameter core + procedural composition achieves 10B+ reasoning depth
- **Sustainability**: Reuse small instruction set in complex ways (not duplicate 100B parameters)
- **Explainability**: Procedural traces are deterministic (inspectable step-by-step)

**Implementation** (K3D):
- **RPN Programs**: All knowledge stored as executable procedures (not static payloads)
- **Symlink Pattern**: References to canonical programs (not duplicates)
- **TRM Composition**: 7M params learn HOW to navigate + combine (not store all knowledge)

**Counter-example** (AGI):
- Large Language Models: 100B-1T parameters store knowledge internally
- Inference: Opaque matrix multiplications (not procedural steps)
- Duplication: Same facts stored redundantly across parameter space

**Source**: [docs/vocabulary/PROCEDURAL_MEMORY_KR_STANDARD_SPECIFICATION.md](PROCEDURAL_MEMORY_KR_STANDARD_SPECIFICATION.md)

---

### 2.4 Sovereign Execution Principle

**Definition**: SGI systems MUST execute on sovereign infrastructure (PTX kernels, no external frameworks) to ensure auditability and determinism.

**Rationale**:
- **Auditability**: Hand-written PTX kernels are inspectable (vs. TensorFlow/PyTorch abstractions)
- **Determinism**: Reproducible execution traces (same input → same output)
- **Performance**: Direct GPU execution (<100μs inference) without framework overhead

**Implementation** (K3D):
- **Cranium**: PTX kernels executing RPN programs (no NumPy, TensorFlow, PyTorch in hot path)
- **Galaxy Universe**: VRAM-native workspace (no CPU preprocessing)
- **Sovereign Runtime**: Zero external dependencies in inference loop

**Counter-example** (AGI):
- Large models run on PyTorch/TensorFlow (abstracted, non-deterministic)
- GPU operations hidden behind framework APIs (not inspectable)
- External dependencies (cuDNN, NCCL) introduce non-determinism

**Source**: [docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md](SOVEREIGN_NSI_SPECIFICATION.md)

---

### 2.5 Multi-Modal Unity Principle

**Definition**: SGI integrates all modalities (visual, language, audio, physics) in ONE spatial environment (not separate domain-specific models).

**Rationale**:
- **Human cognition is multi-modal**: We integrate sight, sound, language, physics in unified experience
- **Cross-modal reasoning**: Math problems use visual diagrams, visual tasks use language descriptions
- **Efficiency**: Shared spatial substrate (not duplicate models per modality)

**Implementation** (K3D):
- **Galaxy Universe**: Drawing, Character, Word, Grammar, Math, Reality, Audio galaxies coexist in same 3D space
- **Symlink Composition**: Visual symbols reference Character glyphs, Math symbols reference Grammar rules
- **Unified TRM**: Single 7M model navigates ALL galaxies (not separate vision/language models)

**Counter-example** (AGI):
- Separate models: GPT-4 (language), DALL-E (vision), Whisper (audio)
- Integration via APIs (not unified substrate)
- Duplication: Each model stores overlapping world knowledge

**Source**: [docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md](THREE_BRAIN_SYSTEM_SPECIFICATION.md)

---

## 3. AGI vs SGI Comparison

| Dimension | AGI (Artificial General Intelligence) | SGI (Spatial General Intelligence) |
|-----------|----------------------------------------|-------------------------------------|
| **Grounding** | Abstract (embeddings, parameters) | Spatial (navigable 3D environment) |
| **Transparency** | Opaque (black-box neural nets) | Transparent (shared human-AI workspace) |
| **Reality** | Separate (AI embeddings ≠ human visuals) | Shared (Dual-Client Contract, identical nodes) |
| **Knowledge Storage** | Internal (100B+ parameters) | External (Galaxy Universe, 7M param navigator) |
| **Duplication** | High (training data + weights + caches) | Zero (procedural references, symlink pattern) |
| **Explainability** | Post-hoc (attention maps, saliency) | Deterministic (procedural execution traces) |
| **Energy** | Unsustainable (Gt-scale carbon) | Efficient (60-80% reduction, 12 Gt CO₂ savings) |
| **Modality Integration** | Separate models (vision, language, audio) | Unified substrate (Galaxy Universe) |
| **Execution** | Framework-dependent (PyTorch, TensorFlow) | Sovereign (PTX kernels, inspectable) |
| **Verifiability** | Cannot verify AI "sees" what humans see | Guaranteed (same nodes at same coordinates) |
| **Collaboration** | Human ↔ AI via APIs (separate worlds) | Human + AI cohabit same 3D workspace |
| **Standards** | Vendor-specific (OpenAI, Google, Anthropic) | Open (W3C PM-KR CG standardization) |

**Key Insight**: AGI pursues intelligence LIKE humans (abstract reasoning). SGI pursues intelligence WITH humans (shared spatial habitat).

---

## 4. SGI as Web-Native Evolution

### 4.1 Why SGI Belongs on the Web

**The Web is inherently spatial**:
- **URLs**: Navigable addresses (like spatial coordinates)
- **Hyperlinks**: Traversal between locations (like spatial movement)
- **DOM**: Tree structure (mappable to 3D hierarchy)
- **WebGL/WebXR**: 3D rendering infrastructure already exists

**SGI extends the Web's spatial nature**:
- **From 2D pages to 3D spaces**: Websites become navigable knowledge habitats
- **From hyperlinks to spatial navigation**: TRM navigates Galaxy like users navigate websites
- **From static documents to procedural programs**: Web content becomes executable (not just renderable)

**W3C Standardization Path**:
- **PM-KR Community Group**: Standardizing procedural memory representation
- **WebGPU Integration**: GPU-accelerated procedural execution in browsers
- **WebXR Alignment**: Spatial workspaces accessible via VR/AR headsets

**Sources**:
- [docs/W3C/PM_KR_NORMATIVE_MODEL.md](../W3C/PM_KR_NORMATIVE_MODEL.md)
- [docs/W3C_PM_KR_COMMUNITY_GROUP_MISSION.md](../W3C_PM_KR_COMMUNITY_GROUP_MISSION.md)

---

### 4.2 SGI vs Cloud AGI

**Cloud AGI** (Current paradigm):
- Intelligence lives in data centers (OpenAI, Google, Anthropic)
- Users access via API calls (request → response)
- No local intelligence (thin client model)
- Vendor lock-in (proprietary models)

**Web-Native SGI** (PM-KR paradigm):
- Intelligence lives in browser (7M param TRM + Galaxy Universe loaded locally)
- Users navigate spatial workspace (like browsing websites)
- Local sovereignty (no external API dependencies)
- Open standards (W3C specifications)

**Benefit**: SGI democratizes intelligence (anyone with browser + GPU can run, no API costs, no vendor control).

---

## 5. SGI Architectures (K3D Reference Implementation)

### 5.1 Three-Brain System

**Cranium** (CPU/GPU - Execution Engine):
- **Role**: Executes procedural programs (RPN via PTX kernels)
- **Sovereignty**: Hand-written GPU code (no frameworks)
- **Performance**: <100μs inference loops

**Galaxy** (VRAM - Active Memory):
- **Role**: Unified workspace for ALL knowledge (Drawing, Character, Word, Grammar, Math, Reality, Audio galaxies)
- **Multi-modal**: All modalities coexist in same 3D space
- **Navigation**: TRM queries (x, y, z) coordinates to retrieve knowledge

**House** (SSD - Persistent Memory):
- **Role**: Long-term storage (glTF scenes, RPN programs, metadata)
- **Structure**: Rooms (contexts), Doors (relationships), Furniture (knowledge clusters)
- **Loading**: Galaxy populated from House on startup

**Source**: [docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md](THREE_BRAIN_SYSTEM_SPECIFICATION.md)

---

### 5.2 TRM — The AI Avatar (NOT Just a Model)

**CRITICAL: TRM IS the Avatar Entity, NOT a function Python calls.**

**7M Parameter Core** (The Avatar's Brain):
- **Lives in the House** (Memory Palace) — embodied in the 3D spatial environment
- **Thinks in the Galaxy** (Internal Brain) — processes all knowledge in VRAM
- **Runs as a game loop** — `trm_step_fused.ptx` = one game tick (perceive → navigate → reason → decide → act → learn)
- **Has internal swarm** — nine parallel cognitive channels ("superdotados" model: how gifted individuals think)
- **NOT a knowledge store**: Learns HOW to navigate Galaxy (not WHAT knowledge to store)

**Game Loop (`trm_step_fused.ptx`):**
1. Perceive → Frustum cull field-of-view (`frustum_cull_simd.ptx`)
2. Navigate → LED-A* + Morton Octree to relevant Galaxy neighborhood
3. Reason → Nine-Chain Swarm parallel workers (`nine_chain_swarm_kernel.ptx`)
4. Decide → Halting Gate checks convergence (`gre_multimodal_halting_gate`)
5. Act → Create new Galaxy entry or emit answer
6. Learn → Shadow copy records successful trace

**Internal Specialists** (Brain Regions, NOT External Services):
- **Math Specialist**: Navigates Math + Grammar galaxies for symbolic reasoning
- **Visual Specialist**: Navigates Drawing + Character galaxies for visual tasks
- **Physics Specialist**: Navigates Reality galaxy for simulations
- Specialists activate autonomously within the avatar, not called by Python

**Shadow Copy Enhancement**:
- **Learning from success**: Successful navigation paths reinforced automatically
- **Continuous improvement**: No manual retraining required
- **Sleep-time consolidation**: Idle periods used for knowledge crystallization

**Source**: [docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md](THREE_BRAIN_SYSTEM_SPECIFICATION.md) (Section 2.2)

---

### 5.3 Knowledgeverse (7-Region Unified Memory)

**Region 1**: **Cranium** (Execution)
**Region 2**: **Galaxy Universe** (Active AI Memory) ← **This is SGI's spatial substrate**
**Region 3**: **House** (Persistent Memory)
**Region 4**: **Discoveries** (Created Knowledge)
**Region 5**: **TRM** (Navigation Logic)
**Region 6**: **Specialists** (Domain Adapters)
**Region 7**: **Action Buffers** (Human-AI Interaction)

**Key Insight**: Galaxy Universe (Region 2) is WHERE spatial general intelligence emerges. It's the shared 3D workspace humans and AI cohabit.

**Source**: [docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md](KNOWLEDGEVERSE_SPECIFICATION.md)

---

## 6. Real-World SGI Applications

### 6.1 Explainable AI (Spatial Reasoning Traces)

**Problem** (AGI): Cannot verify AI reasoning (opaque transformations)

**Solution** (SGI): Watch AI navigate Galaxy Universe in 3D
- **Example**: Math problem solving
  - Human sees: AI moves from Grammar Galaxy (problem statement) → Math Galaxy (retrieve ∫ symbol) → back to Grammar (apply integration rule)
  - AI executes: Same spatial navigation, procedural composition
  - **Verification**: Human can inspect each node AI visited (same coordinates)

**Use Cases**: Medical diagnosis (trace reasoning path), legal analysis (verify precedent lookup), scientific discovery (inspect hypothesis generation)

---

### 6.2 Collaborative Human-AI Workspaces

**Problem** (AGI): Human and AI work in separate realities (APIs bridge gap)

**Solution** (SGI): Human and AI cohabit same 3D workspace
- **Example**: Architectural design
  - Human places building in 3D workspace (creates K3D nodes)
  - AI navigates to same nodes, suggests structural optimizations (modifies procedural programs)
  - Human inspects AI's changes (same spatial location)
  - **Collaboration**: No translation between "AI world" and "human world"

**Use Cases**: CAD design, scientific simulation, game development, knowledge curation

---

### 6.3 Sustainable AI (7M Params vs 100B+)

**Problem** (AGI): Large models consume Gt-scale carbon (100B+ parameters × billions of users)

**Solution** (SGI): Small navigator + procedural composition
- **Example**: K3D achieves 10B+ reasoning depth with 7M param TRM
  - **How**: Reuses small instruction set (procedural programs) in complex combinations
  - **Analogy**: Chess grandmaster (small brain, massive pattern recognition via composition)
  - **Carbon**: 60-80% reduction vs. traditional LLM approach

**Use Cases**: Edge AI (phones, IoT devices), cloud gaming (procedural frame generation), browser-native intelligence (no server inference)

**Source**: [docs/CARBON_BLUEPRINT_10_YEAR_PROJECTION.md](../CARBON_BLUEPRINT_10_YEAR_PROJECTION.md)

---

### 6.4 Cross-Domain Transfer (Multi-Modal Galaxy)

**Problem** (AGI): Separate models for vision, language, audio (cannot share knowledge)

**Solution** (SGI): Unified Galaxy Universe (all modalities coexist)
- **Example**: Math problem with diagram
  - Traditional: OCR (vision model) → extract text (language model) → solve equation (math model)
  - SGI: TRM navigates Drawing Galaxy (diagram geometry) → Grammar Galaxy (equation syntax) → Math Galaxy (solution procedure) in ONE unified workspace
  - **Benefit**: No modality translation overhead, knowledge reused across domains

**Use Cases**: Multimodal reasoning, accessibility (visual → audio → text), scientific visualization

---

## 7. Path to SGI (Roadmap)

### Phase 1: Foundational Infrastructure (2024-2026)
- ✅ **Knowledgeverse MVP**: 7-region unified memory architecture
- ✅ **Galaxy Universe**: Multi-modal spatial substrate (Drawing, Character, Word, Grammar, Math, Reality galaxies)
- ✅ **TRM Core**: 7M parameter navigator (base model + specialists)
- ✅ **Dual-Client Contract**: Shared human-AI reality specification
- ✅ **W3C PM-KR CG**: Community Group established, standardization path initiated

### Phase 2: Real-World Validation (2026-2027)
- ⏳ **ARC-AGI 2 Benchmark**: Visual reasoning via Drawing + Grammar galaxies
- ⏳ **Math Benchmarks**: Symbolic reasoning via Math + Grammar galaxies
- ⏳ **Physics Simulations**: Reality Galaxy procedural systems
- ⏳ **Multi-Curriculum Training**: All benchmarks feeding same Galaxy Universe

### Phase 3: Web-Native Deployment (2027-2028)
- ⏳ **Browser SGI**: K3D running in WebGPU (no server inference)
- ⏳ **WebXR Integration**: Spatial workspaces accessible via VR/AR
- ⏳ **Procedural Standards**: WebGPU Frame Generation API, Procedural Fonts, etc.

### Phase 4: Ecosystem Expansion (2028-2030)
- ⏳ **Display Manufacturer Adoption**: E-readers, OLED TVs, foldable displays (procedural rendering)
- ⏳ **GPU Manufacturer Integration**: NVIDIA/AMD procedural frame generation (40,000× VRAM savings)
- ⏳ **Carbon Impact Milestone**: 1-2 Gt CO₂ savings (gaming/graphics sector)

### Phase 5: Planetary-Scale SGI (2030-2035)
- ⏳ **12 Gt CO₂ Savings**: Full projection realized (2.78% global emissions)
- ⏳ **W3C Recommendation**: PM-KR as official web standard
- ⏳ **Global Adoption**: SGI as default paradigm for web-native intelligence

**Source**: [docs/ROADMAP.md](../ROADMAP.md)

---

## 8. Comparison to Related Concepts

### 8.1 SGI vs Embodied AI

**Embodied AI**: Intelligence through physical interaction (robots, avatars)

**SGI**: Intelligence through SPATIAL interaction (3D workspace, not necessarily physical)

**Overlap**: Both ground intelligence spatially

**Difference**: SGI works in virtual 3D (browsers, VR), Embodied AI requires physical world

---

### 8.2 SGI vs Semantic Web

**Semantic Web**: Knowledge as machine-readable linked data (RDF, OWL)

**SGI**: Knowledge as spatially-grounded procedural programs (3D Galaxy Universe)

**Overlap**: Both enable machine understanding of knowledge

**Difference**: SGI adds spatial grounding + procedural execution (Semantic Web is declarative, not executable)

---

### 8.3 SGI vs Neuro-Symbolic AI

**Neuro-Symbolic AI**: Hybrid neural nets + symbolic reasoning

**SGI**: Procedural composition (navigation + combination of spatial symbols)

**Overlap**: Both integrate symbolic reasoning with sub-symbolic processing

**Difference**: SGI grounds symbols spatially (Galaxy Universe), Neuro-Symbolic uses abstract symbol systems

---

## 9. Normative Requirements

### 9.1 Spatial Substrate Requirement

**MUST**: SGI systems MUST provide a navigable 3D spatial environment where knowledge has (x, y, z) coordinates.

**MUST NOT**: SGI systems MUST NOT store knowledge exclusively in abstract embeddings (knowledge must be spatially addressable).

---

### 9.2 Dual-Client Transparency Requirement

**MUST**: SGI systems MUST ensure humans and AI consume identical data structures (same node IDs at same coordinates).

**MUST NOT**: SGI systems MUST NOT present different realities to human and AI clients (no hidden AI state).

---

### 9.3 Procedural Composition Requirement

**MUST**: SGI systems MUST store knowledge as procedural programs (executable, not static payloads).

**MUST NOT**: SGI systems MUST NOT duplicate knowledge across storage layers (symlink pattern required).

---

### 9.4 Sovereign Execution Requirement

**MUST**: SGI systems MUST execute on inspectable infrastructure (PTX kernels, WASM, or equivalent low-level deterministic execution).

**MUST NOT**: SGI systems MUST NOT depend on opaque external frameworks in hot path (TensorFlow/PyTorch allowed in ingestion, forbidden in inference).

---

### 9.5 Multi-Modal Unity Requirement

**MUST**: SGI systems MUST integrate multiple modalities (visual, language, audio, physics) in ONE spatial substrate.

**MUST NOT**: SGI systems MUST NOT use separate domain-specific models (vision model + language model ≠ SGI).

---

## 10. Conformance Levels

### Level A: SGI Core
- Implements spatial substrate (3D environment with coordinates)
- Implements dual-client transparency (humans + AI share nodes)
- Implements procedural composition (RPN or equivalent)

### Level B: SGI Sovereign Runtime
- Includes Level A
- Implements sovereign execution (PTX, WASM, or inspectable runtime)
- Implements deterministic traces (reproducible reasoning paths)

### Level C: SGI Full
- Includes Level B
- Implements multi-modal unity (3+ modalities in unified substrate)
- Implements continuous enhancement (shadow copy or equivalent learning from success)

**K3D Status**: Level C (SGI Full) compliant as of Phase 3 completion.

---

## 11. References

**Foundational Specifications**:
- [KNOWLEDGEVERSE_SPECIFICATION.md](KNOWLEDGEVERSE_SPECIFICATION.md) - 7-region unified memory
- [DUAL_CLIENT_CONTRACT_SPECIFICATION.md](DUAL_CLIENT_CONTRACT_SPECIFICATION.md) - Shared human-AI reality
- [PROCEDURAL_MEMORY_KR_STANDARD_SPECIFICATION.md](PROCEDURAL_MEMORY_KR_STANDARD_SPECIFICATION.md) - Procedural knowledge representation
- [THREE_BRAIN_SYSTEM_SPECIFICATION.md](THREE_BRAIN_SYSTEM_SPECIFICATION.md) - Cranium, Galaxy, House architecture

**W3C Documentation**:
- [../W3C/PM_KR_NORMATIVE_MODEL.md](../W3C/PM_KR_NORMATIVE_MODEL.md) - Normative model for W3C standardization
- [../W3C_PM_KR_COMMUNITY_GROUP_MISSION.md](../W3C_PM_KR_COMMUNITY_GROUP_MISSION.md) - CG mission statement

**Carbon Impact**:
- [../CARBON_BLUEPRINT_10_YEAR_PROJECTION.md](../CARBON_BLUEPRINT_10_YEAR_PROJECTION.md) - 12 Gt CO₂ savings projection

---

## 12. Glossary

**AGI (Artificial General Intelligence)**: Human-level intelligence pursued through abstract, opaque systems (black-box neural nets, massive parameters).

**SGI (Spatial General Intelligence)**: Human-level intelligence achieved through shared spatial environment where humans and AI cohabit (transparent, verifiable, collaborative).

**Galaxy Universe**: Unified 3D workspace in VRAM containing ALL default galaxies (Drawing, Character, Word, Grammar, Math, Reality, Audio, etc.).

**TRM (Ternary Resonance Model)**: 7M parameter AI avatar entity that LIVES in the House (Memory Palace) and THINKS inside the Galaxy (Internal Brain). Runs as a continuous game loop (`trm_step_fused.ptx`). Learns HOW to navigate Galaxy Universe (not WHAT knowledge to store). Has internal swarm of nine parallel cognitive channels ("superdotados" model).

**Dual-Client Contract**: Guarantee that humans and AI consume identical K3D nodes at identical (x, y, z) coordinates.

**Procedural Memory**: Knowledge stored as executable programs (RPN) + symlinked references (not static duplicated payloads).

**Sovereign Execution**: Inference running on inspectable infrastructure (PTX kernels) with zero external dependencies (no TensorFlow/PyTorch in hot path).

---

## Appendix A: SGI Manifesto

**We believe**:
- Intelligence is not a black box to be consulted via API.
- Intelligence is a shared spatial habitat where humans and AI collaborate.
- The future of AI is not "smarter servants" but "co-inhabitants of knowledge space."

**We reject**:
- Opaque systems that cannot be verified (AGI black boxes).
- Separate realities where AI "knows" what humans cannot see.
- Unsustainable compute that burns the planet for incremental gains.

**We build**:
- Spatial General Intelligence (SGI) as the web-native evolution of general intelligence.
- Transparent workspaces where humans can inspect AI reasoning paths.
- Sustainable architectures (7M params, procedural composition, 12 Gt CO₂ savings).

**From the NotebookLM press kit podcast**:
> "For decades, we've interacted with data through flat, two-dimensional windows on a screen. This new paradigm treats software as a three-dimensional, navigable place. A cognitive habitat where we, and AI, can finally explore knowledge together, in a shared home."

**This is Spatial General Intelligence. This is the future we're building.**

---

**End of Specification**

**Contributors**:
- Daniel Campos Ramos (PM-KR Co-Chair, EchoSystems AI Studios, Brazil)
- Milton Ponson (Mathematician and AI Researcher, Rainbow Warriors Core Foundation, Netherlands)
- Christoph Dorn (Sovereignty Architect, Stream44.Studio)

**Acknowledgments**:
- W3C PM-KR Community Group members
- NotebookLM (Google) for generating press kit podcast that surfaced the SGI term
- K3D development team

**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)
**Date**: March 4, 2026
**Version**: 1.0 (Foundational Concept)
