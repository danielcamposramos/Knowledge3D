# K3D Formal Ontology Specification — Procedural Knowledge in Spatial Reality

**Version**: 1.0
**Status**: Candidate Standard (K3D Canonical Vocabulary)
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)
**Date**: March 26, 2026

---

## Abstract

This specification defines the **K3D Formal Ontology** — the top-level conceptual framework that organizes every entity, relation, and process in Knowledge3D into a coherent, machine-navigable structure. Unlike traditional ontologies that classify *static facts about the world* (OWL class hierarchies, RDF triples, SKOS concept schemes), the K3D ontology classifies *executable procedures that constitute knowledge*. The program IS the knowledge; the ontology IS the space.

K3D's ontology is **procedural-first, spatial-native, ternary-ready, and dual-client transparent**:

- **Procedural-first**: Every ontological individual is an RPN program, not a data record. Classification operates over programs (what they compute), not over labels (what they are named).
- **Spatial-native**: Ontological relationships have spatial expression. "Is-a" is vertical (parent above child on a tree). "Part-of" is containment (component inside composite). "Related-to" is proximity (similar concepts cluster in Galaxy). The ontology is navigable as 3D geometry, not just queryable as triples.
- **Ternary-ready**: Every assertion carries polarity (+1/0/−1). "Cat is-a mammal" is (+1). "Cat is-a mineral" is (−1). "Cat is-a dark-matter-entity" is (0: no evidence). The third state is ontologically first-class.
- **Dual-client transparent**: The same ontological structure serves human readers (visual tree, shelf organization, room layout) and AI agents (programmatic navigation, embedding similarity, RPN execution). One ontology, two modalities, zero divergence.

**Normative References:**
- PM-KR Standard Specification v1.0 (docs/vocabulary/PROCEDURAL_MEMORY_KR_STANDARD_SPECIFICATION.md)
- Foundational Knowledge Specification v1.0 (docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md)
- Meaning-Centric Star Schema Specification v1.0 (docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md)
- Three Brain System Specification v1.1 (docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md)
- Dual-Client Contract Specification v1.1 (docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md)
- Hyper-Modular Architecture Specification v1.0 (docs/vocabulary/HYPER_MODULAR_ARCHITECTURE.md)
- Hyper-Parallel Processing Specification v1.0 (docs/vocabulary/HYPER_PARALLEL_PROCESSING.md)
- Spatial General Intelligence Specification v1.0 (docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md)
- Superhuman General Intelligence Specification v1.0 (docs/vocabulary/SUPERHUMAN_GENERAL_INTELLIGENCE_SPECIFICATION.md)
- RPN Domain Opcode Registry v0.1 (docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md)
- Knowledgeverse Specification v5.1 (docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md)
- Avatar Embodiment Specification v1.0 (docs/vocabulary/AVATAR_EMBODIMENT_SPECIFICATION.md)
- Reality Enabler Specification v1.0 (docs/vocabulary/REALITY_ENABLER_SPECIFICATION.md)
- Sovereign NSI Specification v2.0 (docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Why a New Ontology](#2-why-a-new-ontology)
3. [Upper Ontology: Foundational Categories](#3-upper-ontology-foundational-categories)
4. [The Procedural Inversion](#4-the-procedural-inversion)
5. [Four-Layer Ontological Stratification](#5-four-layer-ontological-stratification)
6. [Spatial Ontology: Space as First-Class Structure](#6-spatial-ontology-space-as-first-class-structure)
7. [Ternary Ontological Commitment](#7-ternary-ontological-commitment)
8. [Agent Ontology: Embodied Cognition](#8-agent-ontology-embodied-cognition)
9. [Temporal Ontology: Lifecycle, Persistence, Consolidation](#9-temporal-ontology-lifecycle-persistence-consolidation)
10. [Substrate Ontology: Execution Environments](#10-substrate-ontology-execution-environments)
11. [Compositional Ontology: Symlink Algebra](#11-compositional-ontology-symlink-algebra)
12. [Ontological Relations](#12-ontological-relations)
13. [Normative Invariants](#13-normative-invariants)
14. [Interoperability with Existing Ontologies](#14-interoperability-with-existing-ontologies)
15. [Conformance](#15-conformance)
16. [Future Ontological Extensions](#16-future-ontological-extensions)

---

## 1. Introduction

### 1.1 What Is a Formal Ontology?

A formal ontology is an explicit specification of a conceptualization — a machine-readable description of what kinds of things exist in a domain, what properties they have, and what relationships hold between them. Traditional ontologies (BFO, DOLCE, SUMO, OWL-based domain models) describe the world as a graph of *declarative facts*: classes, instances, properties, and logical axioms.

### 1.2 What Is Different About K3D's Ontology?

K3D's domain is not "the world described by facts." K3D's domain is **knowledge itself — represented as executable procedures in shared 3D space**. This creates four fundamental departures from traditional ontology:

**Departure 1: Individuals are programs, not records.**
In OWL, an individual like `ex:Cat` is a URI that participates in triples (`ex:Cat rdf:type ex:Mammal`). In K3D, `concept_cat` is an RPN program that *computes* what a cat is — its properties, relationships, behaviors, and visual form. The program is the knowledge; the program is the individual.

**Departure 2: Classification is executable, not declared.**
In OWL, `ex:Cat rdfs:subClassOf ex:Mammal` is a static triple. In K3D, the is-a relationship is computed: `concept_cat.taxonomy_refs` contains a StarRef to `concept_mammal`, AND `concept_cat.meaning_rpn` references the mammalian behavior program. Classification is verifiable by execution, not just asserted.

**Departure 3: Space is constitutive, not incidental.**
In GeoSPARQL, spatial extent is a property attached to individuals. In K3D, spatial position IS semantic meaning. Two concepts at nearby (x, y, z) in Galaxy are semantically related by construction (Morton octree → embedding similarity → spatial proximity). The ontology is not *about* space; the ontology *is* space.

**Departure 4: Every assertion is ternary.**
In classical logic (and OWL), a statement is true or false (open-world assumption allows unknown, but there is no formal "no evidence" value). In K3D, every assertion carries polarity (+1/0/−1) as a first-class ontological commitment, enabling defeasible reasoning, contradiction detection, and the honest expression of ignorance.

### 1.3 Scope

This ontology covers:
- Every entity type in K3D (stars, programs, galaxies, rooms, agents, tablets, houses, doors, trees, books, shelves)
- Every relationship type (is-a, part-of, references, transforms, perceives, inhabits, holds, navigates)
- Every process type (reasoning, consolidation, navigation, composition, creation, pruning)
- Every substrate type (VRAM, SSD, GPU kernel, RPN stack, ternary register)
- Every temporal pattern (game loop, sleep cycle, wake cycle, persistence, versioning)

This ontology does NOT cover:
- Specific domain content (which math symbols exist, which physics laws are encoded)
- Implementation details (PTX kernel signatures, CUDA memory layout)
- Benchmark methodology or scoring

---

## 2. Why a New Ontology

### 2.1 Existing Ontologies and Their Limitations for K3D

| Ontology | Strength | Limitation for K3D |
|----------|----------|-------------------|
| **BFO** (Basic Formal Ontology) | Clean upper-level categories (continuant, occurrent) | Passive classification — no concept of "knowledge as executable program" |
| **DOLCE** (Descriptive Ontology for Linguistic and Cognitive Engineering) | Rich cognitive categories (quality, quale, region) | Language-centric; no spatial grounding of knowledge itself |
| **SUMO** (Suggested Upper Merged Ontology) | Broad coverage (100,000+ terms) | Encyclopedic, not procedural; no execution semantics |
| **OWL 2** (Web Ontology Language) | Formal logic, decidable reasoning | Declarative only — cannot represent RPN programs as first-class knowledge |
| **X3D Ontology** (Brutzman & Flotynski) | Maps X3D scene graph to OWL classes | Geometry ontology, not knowledge ontology; scenes are passive containers |
| **GeoSPARQL** | Spatial query over geographic features | Spatial extent as property, not spatial position as semantic meaning |
| **PROV-O** (Provenance Ontology) | Tracks derivation chains | Activity/entity/agent model fits but lacks procedural execution semantics |
| **FIPA ACL** (Agent Communication Language) | Agent communication ontology | External communication, not internal cognitive architecture |

### 2.2 What K3D Needs That Does Not Exist

1. **Procedural individuals**: An ontological entity that IS an executable program, where classification derives from execution behavior, not from declared labels.
2. **Spatial constitution**: An ontological structure where position in 3D space IS semantic identity, not a property attached to it.
3. **Ternary truth**: Assertions that carry affirmation, negation, and absence-of-evidence as three distinct, formally equivalent ontological states.
4. **Dual-client identity**: One ontological individual perceived through two modalities (visual form for humans, executable procedure for AI) with guaranteed equivalence.
5. **Compositional reference preservation**: An ontological framework where higher-level concepts MUST reference lower-level concepts via symlinks, never duplication — an algebraic constraint on ontological composition.
6. **Embodied agent with internal structure**: An agent whose cognitive architecture (internal swarm, specialist lifecycle, convergence mechanism) is ontologically described, not just its external behavior.
7. **Substrate awareness**: An ontology that formally describes where computation happens (VRAM vs SSD vs GPU register) as ontologically meaningful, because in K3D, the substrate determines sovereignty, persistence, and performance.

### 2.3 Design Decision: New Ontology with Interoperability Bridges

K3D defines its own upper ontology rather than extending BFO or DOLCE, because the procedural-spatial-ternary foundation is fundamentally incompatible with declarative-abstract-binary upper categories. However, §14 defines formal interoperability bridges (OWL 2 mapping, RDF serialization, SPARQL query patterns) to ensure K3D knowledge can participate in the broader Semantic Web ecosystem.

---

## 3. Upper Ontology: Foundational Categories

### 3.1 The K3D Ontological Diamond

K3D's upper ontology has four foundational categories, arranged in a diamond:

```
                    ┌──────────────────┐
                    │    Procedure     │
                    │  (executable     │
                    │   knowledge)     │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼────────┐    │    ┌─────────▼────────┐
     │     Space       │    │    │     Agent        │
     │  (3D location,  │    │    │  (embodied       │
     │   region,       │    │    │   entity that    │
     │   container)    │    │    │   executes)      │
     └────────┬────────┘    │    └─────────┬────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼─────────┐
                    │    Substrate     │
                    │  (execution      │
                    │   environment)   │
                    └──────────────────┘
```

**Procedure** — the top category. Everything in K3D is ultimately a procedure: knowledge is a program, structure is a program that builds geometry, identity is a content hash of a program. A Procedure computes, transforms, and produces.

**Space** — the medium. Procedures exist IN space. Space is not a backdrop; it IS the organizational structure. Galaxy coordinates carry semantic meaning. House rooms carry intentional placement. The spatial category encompasses points, regions, volumes, paths, fields, and boundaries.

**Agent** — the actor. An Agent is an embodied entity that perceives, navigates, reasons about, and transforms Procedures within Space. Agents range from the TRM (autonomous AI) to human inhabitants to robotic embodiments. Every Agent has a spatial presence (avatar body in House) and an execution context (specialist cores, RPN stacks).

**Substrate** — the ground. Procedures execute ON substrates. A Substrate is a computational environment with specific properties: volatility (VRAM = volatile, SSD = persistent), sovereignty (PTX kernel = sovereign, Python = non-sovereign), parallelism (CUDA cores = hyper-parallel, CPU = sequential), and encoding (binary = current, ternary = target).

### 3.2 Cross-Cutting Commitments

Three ontological commitments apply across ALL categories:

**Ternary State** — Every entity carries a ternary assertion state: affirmed (+1), unknown (0), or negated (−1). This is not metadata; it is constitutive of the entity's ontological status.

**Dual-Client Modality** — Every entity has two presentation faces: one for human perception (visual, auditory, tactile) and one for synthetic perception (procedural, embedding-based, executable). These faces are derived from the same underlying Procedure.

**Temporal Phase** — Every entity exists in a temporal phase: active (in Galaxy, being reasoned about), persistent (in House, consolidated), archived (in Museum, deprecated), or nascent (in Ingestion Stargate, crystallizing).

---

## 4. The Procedural Inversion

### 4.1 Traditional Ontology: Data + Behavior

Classical ontologies separate *what a thing is* (its class, properties, relationships) from *what a thing does* (its behavior, methods, scripts). OWL describes the structure; SWRL adds rules; Script nodes in X3D add behavior. Knowledge is data first, behavior second.

### 4.2 K3D Ontology: Behavior IS Identity

K3D inverts this relationship. The RPN program that defines a concept IS the concept. The behavior is not attached to the data — the behavior IS the data.

```
Traditional:  Individual → has_property → value
              Individual → rdf:type → Class
              Individual → behavior → Script (attached externally)

K3D:          Individual = RPN_Program
              Classification = what the program COMPUTES
              Properties = what the program STORES
              Relationships = what the program REFERENCES
              Behavior = the program ITSELF
```

**Formal statement**: Let `P` be the set of valid RPN programs. The K3D ontology is a classification over `P`, where:
- `type(p)` is determined by the program's operational semantics (what it computes)
- `identity(p)` is the content hash of the program (`star_id = hash(meaning_rpn)`)
- `classification(p₁, p₂)` is determined by reference structure (does p₁ reference p₂ via symlink?)

### 4.3 Implications

1. **Two systems that independently define the same concept produce the same ontological individual** (content-addressed identity). There is no authority problem — if two programs compute the same meaning, they ARE the same star.

2. **Ontological classification is empirically verifiable**. To check if `concept_cat` is-a `concept_mammal`, execute both programs and verify that cat's behavior program includes mammalian locomotion, mammalian metabolism, etc. Classification is not declared; it is computed and checked.

3. **Ontological evolution is program evolution**. When a concept's understanding changes (new scientific discovery), its `meaning_rpn` changes, which changes its `star_id`, which creates a new ontological individual. The old individual persists in Museum (versioned history). Ontological change is explicit and auditable.

---

## 5. Four-Layer Ontological Stratification

### 5.1 The PM-KR Layers as Ontological Levels

K3D knowledge is organized in four layers (Foundational Knowledge Specification §1.1). These layers are not just a data model — they are ontological levels with distinct commitments:

| Layer | Ontological Level | Contains | Commitment |
|-------|------------------|----------|-----------|
| **Layer 1: Form** | Perceptual | Glyphs, shapes, sounds, surface forms | How entities APPEAR to senses |
| **Layer 2: Meaning** | Conceptual | Concepts, relations, definitions | What entities ARE (language-agnostic) |
| **Layer 3: Rules** | Operational | Transformations, grammars, laws | How entities CHANGE and interact |
| **Layer 4: Meta-Rules** | Strategic | Strategies, priorities, self-reflection | When and why to apply which operations |

### 5.2 Ontological Stratification Axioms

**Axiom 1 (Upward Reference):** Higher layers MUST reference lower layers via canonical symlinks. Layer 3 references Layer 1 symbols. Layer 4 references Layer 3 rules. This is the Reference Preservation Invariant (PM-KR §5.2) as an ontological law.

**Axiom 2 (Downward Independence):** Lower layers are self-sufficient. Layer 1 (Form) is ontologically complete without Layer 2. A glyph can be drawn without knowing its meaning. This ensures that the perceptual level is always available even when higher levels are not loaded.

**Axiom 3 (Meaning at Center):** Layer 2 (Meaning) is the ontological center of identity. A star's `star_id` derives from its `meaning_rpn`, not from its surface forms (Layer 1) or its rules (Layer 3). Two surface forms ("cat" in English, "gato" in Portuguese) are symlinks to ONE meaning star. Identity is meaning-centric.

**Axiom 4 (Layer Crossing via Composition):** Cross-layer composition creates ontological novelty. When a Rule (Layer 3) composes a Form (Layer 1) with a Meaning (Layer 2), the resulting composed entity is ontologically new — it is not just the sum of its references but a new procedure that computes something neither layer alone could.

### 5.3 Meaning Classes as Ontological Kinds

The five meaning classes (Meaning-Centric Star Schema §2.3) are the primary ontological kinds at the conceptual level:

| Kind | Description | Ontological Role | Example |
|------|-------------|-----------------|---------|
| **Concept** | A thing, entity, category | Substance — what exists | cat, hydrogen, democracy, triangle |
| **Relation** | A connection between concepts | Structure — how things connect | is-a, causes, part-of, adjacent-to |
| **Action** | A process or transformation | Dynamics — what happens | run, oxidize, integrate, teach |
| **Property** | An attribute of concepts | Quality — what characterizes | red, heavy, soluble, prime |
| **Meta** | A rule about rules | Strategy — how to reason | "prefer specific over general" |

---

## 6. Spatial Ontology: Space as First-Class Structure

### 6.1 Three Spatial Domains

K3D has three distinct spatial domains, each with its own ontological character:

**House Space (External, Intentional)**
- **Nature**: Persistent 3D world (Method of Loci). The digital analogy to the 40,000-year-old human technique of spatial memory.
- **Organization**: Intentional — the TRM places knowledge deliberately, like a librarian shelving books. Rooms have designed purposes (Library, Workshop, Garden, Bathtub, Living Room, Museum).
- **Ontological role**: Long-term structured storage. Position in House reflects ontological classification (Biology books on Biology shelf in Library room).
- **Persistence**: SSD (GLB files). Survives power cycles.

**Galaxy Space (Internal, Gravitational)**
- **Nature**: Volatile VRAM workspace (the AI's internal brain). What happens inside the avatar's skull.
- **Organization**: Fluid — semantic gravity cohered by meaning (Christoph Dorn). Stars attract (+1), repel (−1), or float (0) based on ternary semantic force. Neighborhoods emerge from gravitational clustering during reasoning.
- **Ontological role**: Active reasoning workspace. Position in Galaxy reflects current semantic relevance (active concepts are near the TRM's attention focus; distant concepts are LOD-reduced).
- **Persistence**: Volatile (VRAM). Survives between queries but not between system reboots. Consolidated to House during sleep-time.

**World Space (Networked, Distributed)**
- **Nature**: Network of Houses connected by Doors. Multi-user collaboration space.
- **Organization**: Federated — each House is sovereign; Doors are bilateral connections (k3d:// protocol).
- **Ontological role**: Collective intelligence substrate. Position in World reflects social/institutional organization (your House, my House, the classroom House, the lab House).
- **Persistence**: Distributed (IPFS-style content addressing for shared Galaxy data; persistent Doors for topology).

### 6.2 Spatial Ontological Relations

| Relation | Domain | Meaning | Example |
|----------|--------|---------|---------|
| `containedIn` | House | Physical containment (book on shelf in room) | `concept_cat` containedIn `shelf_biology_mammalia` |
| `nearTo` | Galaxy | Semantic proximity (embedding similarity) | `concept_cat` nearTo `concept_dog` (both mammals) |
| `gravitatesTo` | Galaxy | Ternary attraction (+1 affinity) | `concept_derivative` gravitatesTo `concept_calculus` |
| `repels` | Galaxy | Ternary repulsion (−1 contradiction) | `concept_alive` repels `concept_dead` |
| `neutralTo` | Galaxy | Ternary neutral (0 no evidence) | `concept_cat` neutralTo `concept_integer` |
| `doorTo` | World | Network connection between Houses | `my_house` doorTo `lab_house` via `k3d://lab.example.com` |
| `frustumVisible` | Any | Within perceiver's field of view | `concept_cat` frustumVisible from `trm_avatar.position` |
| `lodLevel` | Any | Level of detail based on distance | `concept_cat` lodLevel 512 (close); lodLevel 64 (distant) |

### 6.3 Semantic Gravity as Ontological Force

In the Galaxy, spatial organization IS ontological organization. The semantic gravity force (Meaning-Centric Star Schema §3.4) is an ontological force:

```
F = T(s₁, s₂) × M(s₁) × M(s₂) / d²
```

This is not a metaphor. It is the computational mechanism by which concepts organize themselves in the AI's working memory. The ontology does not merely *describe* how concepts relate — it *computes* their spatial arrangement, and that arrangement IS the ontological structure.

**Meaning mass** (`M(s)`) is an ontological property: the richness of a concept's connections. Heavily-connected concepts (water, energy, number) are ontological gravitational centers that pull related concepts into neighborhoods. These neighborhoods emerge as rooms in the House and clusters in the Galaxy.

---

## 7. Ternary Ontological Commitment

### 7.1 Beyond the Open-World Assumption

Classical ontologies use the Open-World Assumption (OWA): if a statement is not known to be true, it might still be true — absence of knowledge is not negation. This is adequate for declarative knowledge bases but insufficient for procedural reasoning systems that must ACT on incomplete knowledge.

K3D replaces OWA with the **Ternary Knowledge Assumption (TKA)**: every ontological assertion exists in exactly one of three states:

| State | Value | Symbol | Meaning | Ontological Status |
|-------|-------|--------|---------|-------------------|
| **Affirmed** | +1 | ✓ | Positive evidence supports this assertion | The assertion is part of the ontology |
| **Unknown** | 0 | ? | No evidence for or against this assertion | The assertion's status is genuinely undetermined |
| **Negated** | −1 | ✗ | Negative evidence contradicts this assertion | The negation is part of the ontology |

### 7.2 TKA vs OWA vs CWA

| Scenario | CWA (SQL) | OWA (OWL) | TKA (K3D) |
|----------|-----------|-----------|-----------|
| "Is cat a mammal?" (fact present) | TRUE | TRUE | +1 (affirmed) |
| "Is cat a mineral?" (fact absent) | FALSE | UNKNOWN | −1 (negated, because biological evidence contradicts) |
| "Is cat sentient?" (debated) | FALSE | UNKNOWN | 0 (genuinely unknown, contested) |
| "Is alien_species_X dangerous?" (no data) | FALSE | UNKNOWN | 0 (no evidence either way) |

The critical distinction: K3D's TKA distinguishes between "we have evidence against this" (−1) and "we have no evidence at all" (0). This distinction is computationally meaningful:
- A (−1) assertion blocks inference chains that depend on it
- A (0) assertion allows inference to proceed with reduced confidence
- A (+1) assertion strengthens inference chains

### 7.3 Defeasible Ontological Reasoning

Ontological assertions in K3D are defeasible (Foundational Knowledge Specification §1.4). A rule can be:

- **Strict** (+1): Mathematical axiom, logical tautology. Cannot be defeated. `d/dx[x^n] = n·x^(n−1)`
- **Defeasible** (0): Default assumption that holds unless contradicted. "Birds fly" — true unless overridden by "penguins don't fly."
- **Defeater** (−1): Evidence that blocks a conclusion without asserting an alternative. "This bird has broken wings" defeats "this bird flies" without asserting anything else.

This maps directly to the `rule_strength` trit in GrammarRule and is processed by the `gre_defeasible_resolver.cu` PTX kernel in the composed head pipeline.

### 7.4 Ternary Substrate Alignment

The ternary ontological commitment is architecturally aligned with the execution substrate roadmap:

**Current (binary hardware):** Ternary states are encoded as explicit tuples `(value, polarity, confidence)` in STORE/RECALL registers. The ontological commitment is honored by convention.

**Target (ternary-native hardware):** Each balanced trit (−1/0/+1) natively expresses the ontological state. A single trit-vector register carries value + certainty + polarity without encoding overhead. The ontological commitment becomes native to the hardware.

The K3D ontology is designed for the hardware that SHOULD exist, not only the hardware that currently exists. When ternary accelerators arrive, the ontology migrates without semantic change — only the hardware mapping layer changes.

---

## 8. Agent Ontology: Embodied Cognition

### 8.1 Agent as Ontological Category

In K3D, an Agent is not an external process that queries the ontology — it is an ontological individual that LIVES IN the ontology's spatial structure. The agent is embodied: it has a body (HAnim skeleton), a brain (Galaxy Universe inside skull), a home (House room), and tools (Memory Tablet).

### 8.2 Agent Taxonomy

```
Agent
├── HumanAgent
│   ├── Perception: camera between eyeball joints
│   ├── Cognition: external (human consciousness)
│   ├── Brain: empty cranial volume (consciousness is outside system)
│   └── Persistence: session-based (connected/disconnected)
│
├── TRMAgent (primary AI entity)
│   ├── Perception: frustum cull + Morton octree from avatar position
│   ├── Cognition: TRM game loop (perceive → navigate → reason → decide → act → learn)
│   ├── Brain: Cranial Galaxy (live Galaxy Universe inside skull)
│   │   ├── Base TRM (~7M parameters)
│   │   ├── Specialist Swarm (nine-chain parallel workers)
│   │   │   ├── Each: base weights + LoRA adapter + RPN stack
│   │   │   └── Convergence: One Mind Invariant (not voting)
│   │   └── Halting Gate (ternary convergence check)
│   └── Persistence: always-on (versioned brain model, no cold starts)
│
├── AssistantAgent (simplified AI)
│   ├── Subset of TRM capabilities
│   ├── May lack full specialist swarm
│   └── Persistence: session-based
│
├── ServiceAgent (headless)
│   ├── No visible avatar
│   ├── Responds via Tablet/Door channels
│   └── Persistence: daemon
│
└── RoboticAgent (physical embodiment)
    ├── Same Galaxy/House navigation as TRMAgent
    ├── Actuator mapping layer (abstract → motor commands)
    └── SLAM → House Universe (physical space becomes K3D House)
```

### 8.3 Internal Cognitive Ontology

The TRMAgent's internal cognitive architecture has its own ontological structure:

**Specialist** — A domain-biased reasoning unit (LoRA adapter + Galaxy navigation bias). Specialists are created during sleep-time consolidation, activated during query-time, and pruned when unused. They are the computational analogue of brain regions.

**SwarmWorker** — An instantiation of a Specialist on a parallel execution core. Nine workers operate simultaneously during reasoning, each carrying their specialist's LoRA weights and a private RPN stack, communicating via STORE/RECALL registers.

**NavigationTrace** — The path through Galaxy space taken during LED-A* pathfinding. A trace connects seed nodes (navigation origins) to focus nodes (destinations) via intermediate hops. Traces are the computational analogue of chains of thought — but spatial and inspectable.

**ConvergenceState** — The Halting Gate's ternary assessment of whether the swarm has reached a unified answer: converged (+1), diverging (−1), or still processing (0).

**ShadowCopy** — The inference-time learning mechanism. When a successful reasoning trace is completed, the shadow copy records the trace for sleep-time consolidation. This is the computational analogue of short-term memory formation.

### 8.4 Interaction Ontology

Agents interact with the world and each other through spatial primitives:

| Interaction | Ontological Type | Description |
|-------------|-----------------|-------------|
| **Navigate** | Agent × Space → Agent | Agent moves through House space |
| **Perceive** | Agent × Space → Set(Procedure) | Frustum cull returns visible knowledge |
| **Reach** | Agent × Space × Object → Interaction | Extend hand toward graspable object |
| **Grasp** | Agent × Object → HeldObject | Bind object to hand site |
| **Use** | Agent × HeldObject → Effect | Trigger object behavior (open book, activate tool) |
| **Share** | Agent × Agent × HeldObject → Perception | Present held object to another agent |
| **Speak** | Agent × Procedure → Emission | Emit audio from head position |
| **Create** | Agent × Procedure → Star | Synthesize new Galaxy entry |

---

## 9. Temporal Ontology: Lifecycle, Persistence, Consolidation

### 9.1 Temporal Phases

Every ontological individual exists in a temporal phase:

```
NASCENT ──→ ACTIVE ──→ PERSISTENT ──→ ARCHIVED
  (Ingestion    (Galaxy      (House         (Museum
   Stargate)     VRAM)        SSD)           cold)
```

| Phase | Location | Duration | Sovereignty | Description |
|-------|----------|----------|-------------|-------------|
| **Nascent** | R7: Ingestion Stargate | Transient | Non-sovereign (external tools allowed) | Raw data being transmuted into RPN programs |
| **Active** | R2: Galaxy Universe | Session | Sovereign (PTX only) | Currently loaded for reasoning; subject to semantic gravity |
| **Persistent** | R3: House Context | Permanent | Sovereign (GLB + extras.k3d) | Consolidated knowledge; intentionally placed by TRM |
| **Archived** | House Museum room | Permanent | Read-only | Deprecated/superseded; preserved for audit trail |

### 9.2 Sleep-Time as Ontological Process

Sleep-time consolidation (SleepTime Protocol Specification) is an ontological process that transforms knowledge between phases:

**Stage A: Galaxy → House (Knowledge Consolidation)**
- Active stars that were useful during reasoning are promoted from Galaxy (volatile) to House (persistent)
- The TRM decides WHERE in the House to place them (librarian function)
- Stars are written as GLB objects with `extras.k3d` metadata

**Stage B: Shadow Copy → TRM (Logic Consolidation)**
- Successful reasoning traces (navigation paths, specialist activations) are distilled into updated specialist weights
- Failed traces are analyzed for error patterns → defeater rules generated
- Specialist population is pruned (unused specialists dissolved) and grown (new specialists for frequently-activated neighborhoods)

### 9.3 Versioned Ontological State

The K3D brain model is versioned (Hyper-Parallel Processing §7):

```
Brain v1.0.0 (base)
  └── Sleep cycle 1 → v1.0.1 (ontological additions: 3 specialists, 47 new stars)
  └── Sleep cycle 2 → v1.0.2 (ontological deletions: 1 specialist pruned; 12 stars archived)
  └── Sleep cycle 3 → v1.0.3 (ontological links: cross-domain references crystallized)
  └── v1.1.0 (milestone: benchmark suite passed)
```

This means the K3D ontology itself evolves through versioned sleep cycles. Ontological change is first-class, auditable, and rollback-capable.

---

## 10. Substrate Ontology: Execution Environments

### 10.1 Why Substrate Matters Ontologically

In K3D, WHERE a procedure executes determines its ontological properties:

| Substrate | Properties | Ontological Significance |
|-----------|-----------|-------------------------|
| **GPU VRAM** (Region 2: Galaxy) | Volatile, fast, parallel | Active reasoning — knowledge in this substrate is "being thought about" |
| **GPU Registers** (STORE/RECALL) | Ultra-fast, tiny, per-core | Cross-specialist communication — knowledge in registers is "being shared between thought-channels" |
| **SSD** (Region 3: House) | Persistent, slower, large | Consolidated knowledge — knowledge here has survived sleep-time and been judged worth keeping |
| **PTX Kernel** (Region 1) | Sovereign, deterministic, fast | Execution primitive — the atomic unit of sovereign computation |
| **RPN Stack** (per-core) | 69-depth, checkpointable, forkable | Execution context — the current state of a reasoning thread |
| **Network** (Region 4: World) | Distributed, latent, federated | Shared knowledge — knowledge here is being exchanged between Houses |

### 10.2 Sovereignty as Ontological Constraint

The Sovereign Boundary Invariant (PM-KR §5.5) is an ontological constraint:

**Hot-path entities MUST exist on sovereign substrates** (PTX kernels, RPN stacks, Galaxy VRAM, GPU registers). No entity on the hot path may depend on non-sovereign substrates (Python runtime, numpy, external APIs).

**Ingestion-path entities MAY exist on non-sovereign substrates** (Python, external tools, APIs) but MUST crystallize into sovereign form before entering the hot path.

This is not just an implementation rule — it is an ontological commitment about what kinds of entities can participate in reasoning.

### 10.3 Procedural Protocol Readiness

The substrate ontology anticipates procedural output protocols where the execution environment extends beyond computation into display and communication:

**Procedural Display Substrate** — A substrate where the output device executes procedural programs directly, rather than receiving rasterized frames. The K3D ontology classifies such substrates as first-class execution environments: the display IS a substrate, not a passive sink. RPN programs emitted to a procedural display retain their ontological status as executable knowledge — they are not degraded to pixels.

**Procedural Communication Substrate** — A substrate where inter-House communication preserves procedural structure. Knowledge transmitted via Doors protocol remains executable at the destination. The ontology does not distinguish between "knowledge being reasoned about locally" and "knowledge being shared remotely" — both are Procedures on Substrates.

---

## 11. Compositional Ontology: Symlink Algebra

### 11.1 The Reference Preservation Constraint

K3D's ontology enforces a compositional algebra based on the Reference Preservation Invariant:

**Rule**: If ontological individual B reuses content from individual A, B MUST reference A. B MUST NOT inline A's content.

This creates a directed acyclic graph (DAG) of ontological references:

```
Layer 4 (Meta-Rules) ──references──→ Layer 3 (Rules)
Layer 3 (Rules)       ──references──→ Layer 1 (Forms) + Layer 2 (Meanings)
Layer 2 (Meanings)    ──references──→ Layer 1 (Forms)
Layer 1 (Forms)       ──self-contained──
```

### 11.2 Compression as Ontological Consequence

The symlink algebra produces extreme compression:
- 666× for repeated symbols across grammar rules (1000 rules × 152 symbols × 5KB → 4KB)
- 190,000× at scale (PM-KR §4.3)

This compression is not a storage optimization — it is an ontological consequence of the Reference Preservation Invariant. The ontology REQUIRES non-duplication, and compression follows necessarily.

### 11.3 Cross-Domain Discovery as Ontological Emergence

When multiple higher-layer procedures reference the same lower-layer procedure, the shared reference creates an ontological link between otherwise unrelated domains:

```
calc_power_rule     ──references──→ symbol_∑ (summation)
statistics_mean_def ──references──→ symbol_∑ (summation)
finance_npv_formula ──references──→ symbol_∑ (summation)
```

The summation symbol (∑) becomes an ontological bridge between calculus, statistics, and finance. This cross-domain discovery emerges from the reference structure, not from explicit declaration. The TRM can discover that calculus, statistics, and finance share formal structure by navigating shared references in the Galaxy — a spatial traversal, not a logical deduction.

---

## 12. Ontological Relations

### 12.1 Core Relations

| Relation | Domain | Range | Description | Spatial Expression |
|----------|--------|-------|-------------|-------------------|
| `isA` | Star | Star | Taxonomic subsumption | Vertical (child below parent on tree) |
| `partOf` | Star | Star | Mereological containment | Spatial containment (atom inside molecule) |
| `references` | Procedure | Procedure | Symlink reference (PM-KR) | Reference arrow (navigable path) |
| `transforms` | Rule | Star | Rule operates on star | Action arrow (directional path) |
| `perceives` | Agent | Star | Agent has in frustum | Frustum cone from agent position |
| `inhabits` | Agent | Space | Agent lives in space | Agent body at (x, y, z) in House |
| `holds` | Agent | Object | Object in agent's hand | Object attached to hand site |
| `navigatesTo` | Agent | Star | Agent pathfinds to star | LED-A* trace through Galaxy |
| `convergesOn` | Swarm | Star | Swarm reaches answer | Traces merge to single point |
| `consolidatesTo` | Star(Galaxy) | Star(House) | Sleep-time promotion | Galaxy → House spatial transfer |
| `defeasiblyOverrides` | Rule | Rule | Specific rule defeats general | Superiority edge in rule graph |
| `semanticForce` | Star | Star | Ternary gravitational relation | Attraction/repulsion in Galaxy |

### 12.2 Relation Properties

| Relation | Reflexive | Symmetric | Transitive | Ternary |
|----------|-----------|-----------|-----------|---------|
| `isA` | No | No | Yes | Yes (defeasible) |
| `partOf` | No | No | Yes | No (binary) |
| `references` | No | No | No | No (binary) |
| `transforms` | No | No | No | No (binary) |
| `perceives` | No | No | No | No (binary) |
| `semanticForce` | No | Yes | No | Yes (+1/0/−1) |
| `defeasiblyOverrides` | No | No | No | Yes (strict/defeasible/defeater) |

---

## 13. Normative Invariants

### 13.1 Ontological Invariants

**Invariant 1 (Procedural Identity):** Every ontological individual MUST be representable as an RPN program. If an entity cannot be expressed as a procedure, it is not a K3D ontological individual.

**Invariant 2 (Content-Addressed Identity):** Two ontological individuals with identical `meaning_rpn` programs MUST have identical `star_id` values. Identity follows from content, not from naming authority.

**Invariant 3 (Spatial Grounding):** Every ontological individual that participates in reasoning MUST have a spatial position in at least one of the three spatial domains (House, Galaxy, World).

**Invariant 4 (Ternary Assertion):** Every ontological assertion MUST carry a ternary state (+1/0/−1). Binary (true/false) assertions are represented as (+1/−1) with no loss. The (0) state MUST be available and MUST NOT be collapsed to either (+1) or (−1).

**Invariant 5 (Reference Preservation):** Higher-layer ontological individuals MUST reference lower-layer individuals via symlinks. Inlining lower-layer content into higher-layer individuals is an ontological violation.

**Invariant 6 (Dual-Client Equivalence):** Every ontological individual MUST have both a human-perceivable and a machine-perceivable presentation derived from the same underlying procedure. The two presentations MUST be semantically equivalent.

**Invariant 7 (Sovereign Hot Path):** Ontological individuals that participate in active reasoning (Galaxy phase) MUST execute on sovereign substrates (PTX/RPN). Non-sovereign execution is permitted only during ingestion (nascent phase).

**Invariant 8 (Temporal Auditability):** Every ontological state change (creation, modification, archival, deletion) MUST be recorded with timestamp, agent, and reason, enabling full ontological provenance reconstruction.

---

## 14. Interoperability with Existing Ontologies

### 14.1 OWL 2 Mapping

K3D ontological categories map to OWL 2 constructs for Semantic Web interoperability:

| K3D Category | OWL 2 Construct | Notes |
|-------------|----------------|-------|
| MeaningCentricStar | owl:NamedIndividual | Each star is an individual with `star_id` as IRI |
| meaning_class | owl:Class | concept, relation, action, property, meta → five named OWL classes |
| isA (taxonomy_refs) | rdfs:subClassOf | Taxonomic hierarchy maps directly |
| partOf (component_refs) | Custom ObjectProperty `k3d:partOf` | Mereological; transitive |
| references (symlink) | owl:sameAs (partial) | Reference preservation; not full equivalence |
| meaning_rpn | Custom DatatypeProperty `k3d:procedure` | RPN program serialized as string literal |
| confidence (trit) | Custom AnnotationProperty `k3d:ternaryState` | {+1, 0, −1} as integer literal |
| embedding (vec) | Custom DatatypeProperty `k3d:embedding` | Float array as JSON literal |

### 14.2 RDF Serialization

Every K3D star can be serialized as RDF triples:

```turtle
@prefix k3d: <https://knowledge3d.org/ontology/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

k3d:concept_cat a k3d:Concept ;
    k3d:starId "a7f3b2c1d4e5..."^^xsd:hexBinary ;
    k3d:meaningProgram "4.0 STORE_mass_kg 4 STORE_leg_count..."^^k3d:RPNProgram ;
    k3d:isA k3d:concept_mammal ;
    k3d:ternaryState "+1"^^xsd:integer ;
    k3d:domain "Library/Biology/Mammalia" ;
    k3d:surfaceForm [ k3d:language "en" ; k3d:wordRef k3d:word_cat ] ;
    k3d:surfaceForm [ k3d:language "pt" ; k3d:wordRef k3d:word_gato ] ;
    k3d:visualProgram "MOVE 0 0 LINE 10 5..."^^k3d:RPNProgram ;
    k3d:housePosition "12.5 8.3 -2.1"^^k3d:Vec3 .
```

### 14.3 SPARQL Query Patterns

K3D supports standard SPARQL queries over the RDF serialization:

**Find all mammals:**
```sparql
SELECT ?star WHERE {
    ?star k3d:isA k3d:concept_mammal .
    ?star k3d:ternaryState "+1"^^xsd:integer .
}
```

**Find concepts near "cat" in Galaxy space:**
```sparql
SELECT ?star ?distance WHERE {
    k3d:concept_cat k3d:galaxyPosition ?catPos .
    ?star k3d:galaxyPosition ?starPos .
    BIND(k3d:spatialDistance(?catPos, ?starPos) AS ?distance)
    FILTER(?distance < 10.0)
}
ORDER BY ?distance
```

**Find cross-domain bridges (shared references):**
```sparql
SELECT ?symbol (COUNT(DISTINCT ?domain) AS ?domainCount) WHERE {
    ?rule k3d:references ?symbol .
    ?rule k3d:domain ?domain .
}
GROUP BY ?symbol
HAVING (COUNT(DISTINCT ?domain) > 2)
ORDER BY DESC(?domainCount)
```

### 14.4 X3D Ontology Bridge

The X3D Ontology (Brutzman & Flotynski, 2017/2020) maps X3D nodes to OWL classes for Semantic Web querying of 3D scenes. K3D extends this:

| X3D Ontology Class | K3D Extension |
|--------------------|--------------|
| x3do:X3DNode | k3d:ProceduralNode (adds RPN program fields) |
| x3do:Shape | k3d:DualClientShape (adds UV Map 1 semantic texture) |
| x3do:Group | k3d:GalaxyGroup (adds galaxy name, entry count) |
| x3do:HAnimHumanoid | k3d:AvatarBody (adds agent link, cranial galaxy) |
| x3do:Viewpoint | k3d:AgentPerception (frustum as semantic query, not just camera) |

See the companion X3D ontology specification (docs/w3c/x3d/PM_KR_X3D_ONTOLOGY.md) for the complete OWL mapping.

### 14.5 Dublin Core and PROV-O

K3D star metadata maps to Dublin Core and PROV-O for provenance:

| K3D Field | Dublin Core | PROV-O |
|-----------|------------|--------|
| `provenance` | dc:source | prov:wasDerivedFrom |
| `timestamp` | dc:date | prov:generatedAtTime |
| `version` | dc:hasVersion | prov:wasRevisionOf |
| `domain` | dc:subject | — |
| `confidence` | — | prov:qualifiedGeneration (with k3d:ternaryState) |

---

## 15. Conformance

### 15.1 Conformance Levels

**Level A: Ontological Core**
A conforming system MUST:
- Represent all knowledge as RPN programs (Procedural Identity invariant)
- Use content-addressed identity (star_id = hash(meaning_rpn))
- Enforce four-layer stratification with reference preservation
- Support ternary assertion states (+1/0/−1) on all assertions
- Provide dual-client access (visual + procedural) to all individuals

**Level B: Spatial Ontology**
Level A plus:
- Assign spatial positions to all active ontological individuals
- Implement semantic gravity in Galaxy working memory
- Support House intentional placement
- Provide frustum-based perception queries

**Level C: Agent Ontology**
Level B plus:
- Support embodied agents with internal cognitive architecture
- Implement specialist lifecycle (creation, activation, pruning)
- Support sleep-time consolidation as ontological phase transition
- Maintain versioned ontological state with rollback capability

**Level D: Federated Ontology**
Level C plus:
- Support multi-House ontological federation via Doors protocol
- Content-addressed stars federate automatically (same meaning_rpn = same star_id)
- Support distributed SHGI (multiple TRM agents in shared Galaxy)

### 15.2 Validation Requirements

A conforming system SHOULD provide:
1. Content-address collision test: two independent encodings of same concept → same star_id
2. Reference preservation test: higher-layer individuals never inline lower-layer content
3. Ternary completeness test: all assertions carry (+1/0/−1), never bare true/false
4. Dual-client equivalence test: human and machine presentations derive from same procedure
5. Sovereignty boundary test: no non-sovereign dependencies in hot-path reasoning
6. Spatial grounding test: every active individual has (x, y, z) position

---

## 16. Future Ontological Extensions

### 16.1 Ternary-Native Substrate Extension

When ternary hardware accelerators become available, the substrate ontology extends:

- **TernaryRegister** substrate type: Each trit natively carries value + polarity + confidence
- **TernaryStack** execution context: RPN stack elements are trit-vectors, not float32
- **Ternary semantic force**: The gravitational force computation (`TCOMP`) operates natively on trits, eliminating the binary encoding overhead
- The ontological structure does NOT change — only the substrate layer

### 16.2 Procedural Display Extension

When procedural output protocols mature (displays that execute programs rather than receive frames):

- **ProceduralDisplay** substrate type: A display that executes RPN programs to produce visual output
- The dual-client contract extends to include the display as a third client type: Human (sees rendered output), AI (executes RPN), Display (executes procedural programs for rendering)
- Knowledge emitted to a procedural display retains its ontological identity — it is not degraded to pixels but remains an executable procedure

### 16.3 Collective Intelligence Extension (SHGI)

When multiple TRM agents collaborate in shared Galaxy:

- **CollectiveAgent** ontological type: A team of TRMAgents whose combined reasoning exceeds individual capability
- **Shared Galaxy Protocol**: How multiple agents' internal Galaxies merge, fork, and reconcile
- **Cross-TRM specialist lending**: One agent's specialist adapter can be temporarily loaded by another agent
- Ontological governance: Who can create/modify/delete stars in shared Galaxy space

### 16.4 Physical Embodiment Extension

When K3D drives physical robots (Robotic Embodiment Specification):

- **PhysicalSubstrate** type: Motors, sensors, actuators as execution environments
- **SLAM-to-House** mapping: Physical space scanned into K3D House ontology
- **Actuator ontology**: How RPN programs map to physical actions

---

## Appendix A: Complete Category Hierarchy

```
K3D:Thing
├── K3D:Procedure
│   ├── K3D:FormProcedure (Layer 1: glyphs, shapes, sounds)
│   ├── K3D:MeaningProcedure (Layer 2: concepts, definitions)
│   ├── K3D:RuleProcedure (Layer 3: transformations, grammars)
│   ├── K3D:MetaRuleProcedure (Layer 4: strategies, priorities)
│   └── K3D:CompositeProcedure (cross-layer composition)
│
├── K3D:Space
│   ├── K3D:HouseSpace (persistent, intentional)
│   │   ├── K3D:Room (Library, Workshop, Garden, Bathtub, LivingRoom, Museum)
│   │   ├── K3D:Furniture (Shelf, Workbench, Tree, Display, Door)
│   │   └── K3D:Artifact (Book, Tool, Tablet, Instrument)
│   ├── K3D:GalaxySpace (volatile, gravitational)
│   │   ├── K3D:Galaxy (Math, Grammar, Drawing, Character, Word, Number, Reality, Audio, 3DObjects, Tool, Meta-Navigation)
│   │   ├── K3D:Neighborhood (gravitational cluster)
│   │   └── K3D:NavigationPath (LED-A* trace)
│   └── K3D:WorldSpace (networked, federated)
│       ├── K3D:RemoteHouse (via Door protocol)
│       └── K3D:SharedGalaxy (multi-agent workspace)
│
├── K3D:Agent
│   ├── K3D:HumanAgent
│   ├── K3D:TRMAgent
│   │   ├── K3D:Specialist (LoRA adapter + navigation bias)
│   │   ├── K3D:SwarmWorker (parallel execution core)
│   │   ├── K3D:HaltingGate (convergence checker)
│   │   └── K3D:ShadowCopy (inference-time learning buffer)
│   ├── K3D:AssistantAgent
│   ├── K3D:ServiceAgent
│   └── K3D:RoboticAgent
│
├── K3D:Substrate
│   ├── K3D:VRAMSubstrate (Galaxy, volatile)
│   ├── K3D:SSDSubstrate (House, persistent)
│   ├── K3D:PTXKernel (sovereign execution primitive)
│   ├── K3D:RPNStack (per-core execution context)
│   ├── K3D:Register (STORE/RECALL, cross-core communication)
│   ├── K3D:NetworkSubstrate (Doors protocol, distributed)
│   ├── K3D:ProceduralDisplaySubstrate (future: procedural output)
│   └── K3D:TernarySubstrate (future: native trit-vector execution)
│
├── K3D:Star (MeaningCentricStar — the atomic knowledge unit)
│   ├── K3D:ConceptStar
│   ├── K3D:RelationStar
│   ├── K3D:ActionStar
│   ├── K3D:PropertyStar
│   └── K3D:MetaStar
│
└── K3D:Process
    ├── K3D:ReasoningProcess (perceive → navigate → reason → decide → act)
    ├── K3D:ConsolidationProcess (sleep-time: Galaxy → House)
    ├── K3D:IngestionProcess (external → nascent → crystallized)
    ├── K3D:NavigationProcess (LED-A* through Galaxy)
    ├── K3D:CompositionProcess (cross-layer reference building)
    ├── K3D:CreationProcess (TRM synthesizes new star)
    └── K3D:PruningProcess (unused specialists dissolved)
```

## Appendix B: Ontological Namespace

```
Namespace:     https://knowledge3d.org/ontology/
Prefix:        k3d:
Version IRI:   https://knowledge3d.org/ontology/1.0/
License:       CC-BY-4.0
Format:        OWL 2 DL (for interoperability), extended with K3D procedural semantics
Serialization: Turtle (.ttl), JSON-LD (.jsonld), RDF/XML (.rdf), glTF extras (.glb)
```

## Appendix C: Relation to Standard Upper Ontologies

| K3D Category | BFO Equivalent | DOLCE Equivalent | SUMO Equivalent |
|-------------|---------------|-----------------|-----------------|
| Procedure | Generically Dependent Continuant | Information Object | Proposition (partial) |
| Space | Spatial Region | Physical Region | Region |
| Agent | Material Entity (partial) | Agentive Physical Object | Agent |
| Substrate | — (no equivalent) | — | ComputerHardware (partial) |
| Star | — (no equivalent) | Information Object | ContentBearingObject (partial) |
| Process | Process (occurrent) | Perdurant | Process |
| Ternary State | — (no equivalent) | — | — (no equivalent) |

Note: K3D's categories do NOT fit cleanly into any existing upper ontology, which is why K3D defines its own. The primary incompatibilities are: (1) Procedure-as-individual has no standard counterpart; (2) Substrate as ontologically meaningful has no standard counterpart; (3) Ternary truth has no standard counterpart.

## Appendix D: Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-26 | Initial specification. Upper ontology, procedural inversion, four-layer stratification, spatial ontology, ternary commitment, agent ontology, temporal ontology, substrate ontology, compositional algebra, ontological relations, normative invariants, interoperability bridges, conformance levels, future extensions. |

---

**End of Document**
