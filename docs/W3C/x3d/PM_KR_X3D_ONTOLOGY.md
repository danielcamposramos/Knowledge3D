# PM-KR X3D Ontology — OWL 2 Formalization of Procedural Knowledge Scenes

**Version**: 0.1 (Initial Draft)
**Status**: W3C PM-KR Community Group Working Draft
**Date**: March 26, 2026
**Authors**: PM-KR Community Group (Daniel Campos Ramos, Chair; Milton Ponson, Co-Chair)
**Liaison**: Web3D Consortium (Don Brutzman, Advisory Committee Representative)
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Reference Implementation)

**Normative References**:
- ISO/IEC 19775-1:2023 (X3D Architecture and Base Components, Version 4.0)
- ISO/IEC 19774:2019 (HAnim — Humanoid Animation, Version 2.0)
- OWL 2 Web Ontology Language (W3C Recommendation, 2012-12-11)
- RDF 1.1 Concepts and Abstract Syntax (W3C Recommendation, 2014-02-25)
- SPARQL 1.1 Query Language (W3C Recommendation, 2013-03-21)
- X3D Ontology for Semantic Web (Brutzman, 2020; Flotynski, 2019)
- GeoSPARQL 1.0 (OGC 11-052r4, spatial query extension)
- PROV-O: The PROV Ontology (W3C Recommendation, 2013-04-30)
- PM-KR X3D Procedural Memory Component v0.1 (docs/w3c/x3d/PM_KR_X3D_PROCEDURAL_MEMORY_COMPONENT.md)
- PM-KR X3D Avatar Embodiment Specification v0.1 (docs/w3c/x3d/PM_KR_X3D_AVATAR_SPECIFICATION.md)
- K3D Formal Ontology Specification v1.0 (docs/vocabulary/FORMAL_ONTOLOGY_SPECIFICATION.md)
- PM-KR Technology Specification v1.0 (docs/vocabulary/PROCEDURAL_MEMORY_KR_STANDARD_SPECIFICATION.md)
- Foundational Knowledge Specification v1.0 (docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md)
- Dual-Client Contract Specification v1.1 (docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md)
- Meaning-Centric Star Schema Specification v1.0 (docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md)
- Sovereign NSI Specification v2.0 (docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md)
- RPN Domain Opcode Registry v0.1 (docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Scope and Design Rationale](#2-scope-and-design-rationale)
3. [Namespace Architecture](#3-namespace-architecture)
4. [Relationship to the X3D Ontology](#4-relationship-to-the-x3d-ontology)
5. [Upper Ontology: OWL 2 Formalization](#5-upper-ontology-owl-2-formalization)
6. [Procedural Knowledge Classes](#6-procedural-knowledge-classes)
7. [Spatial Ontology Classes](#7-spatial-ontology-classes)
8. [Agent Ontology Classes](#8-agent-ontology-classes)
9. [Substrate Ontology Classes](#9-substrate-ontology-classes)
10. [Object Properties](#10-object-properties)
11. [Datatype Properties](#11-datatype-properties)
12. [Annotation Properties](#12-annotation-properties)
13. [Ternary Assertion Encoding](#13-ternary-assertion-encoding)
14. [Class Restrictions and Axioms](#14-class-restrictions-and-axioms)
15. [SPARQL Query Patterns](#15-sparql-query-patterns)
16. [RDF/Turtle Serialization Examples](#16-rdfturtle-serialization-examples)
17. [glTF-to-RDF Extraction](#17-gltf-to-rdf-extraction)
18. [Interoperability Bridges](#18-interoperability-bridges)
19. [Procedural Protocol and Display Readiness](#19-procedural-protocol-and-display-readiness)
20. [Conformance](#20-conformance)
21. [Examples](#21-examples)

---

## 1. Introduction

### 1.1 Purpose

This document defines the **PM-KR X3D Ontology** — an OWL 2 DL formalization that maps the K3D Formal Ontology (docs/vocabulary/FORMAL_ONTOLOGY_SPECIFICATION.md) and the PM-KR X3D node types (docs/w3c/x3d/PM_KR_X3D_PROCEDURAL_MEMORY_COMPONENT.md) into a Semantic Web ontology. It enables:

1. **SPARQL querying** of procedural knowledge scenes — find all concepts in a Galaxy, discover cross-domain bridges through shared symlink references, identify navigation traces through spatial regions.
2. **OWL 2 reasoning** over procedural knowledge — subsumption, classification, and consistency checking over the four-layer PM-KR hierarchy.
3. **Linked Data publication** of K3D knowledge — every MeaningCentricStar, every Galaxy, every House room can be dereferenced as RDF on the Semantic Web.
4. **Interoperability** with existing X3D Ontology tooling — tools that already process X3D-to-OWL can extend to PM-KR nodes.
5. **Federated knowledge discovery** — SPARQL federation across multiple K3D Houses published as RDF endpoints.

### 1.2 Relationship to the K3D Formal Ontology

The K3D Formal Ontology (docs/vocabulary/FORMAL_ONTOLOGY_SPECIFICATION.md) defines the **conceptual framework**: the Upper Ontology Diamond (Procedure, Space, Agent, Substrate), the four-layer stratification, ternary commitment, and normative invariants.

This X3D Ontology is the **formal serialization** of that framework in OWL 2 DL. It translates each K3D ontological category into an OWL 2 class, each ontological relation into an OWL 2 property, and each invariant into an OWL 2 axiom or SHACL constraint. Where the K3D Formal Ontology says "what kinds of things exist," this specification says "how to encode them as RDF triples so that OWL reasoners and SPARQL engines can process them."

### 1.3 Terminology

The keywords **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**, **SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY**, and **OPTIONAL** in this document are to be interpreted as described in RFC 2119.

| Term | Definition | Source |
|------|-----------|--------|
| **X3D Ontology** | OWL 2 representation of X3D Architecture nodes, fields, and scene structure | Brutzman (2020), Flotynski (2019) |
| **PM-KR Ontology** | This specification — OWL 2 extension for procedural knowledge in X3D scenes | This document |
| **K3D Formal Ontology** | The conceptual upper ontology (not OWL-encoded) | docs/vocabulary/FORMAL_ONTOLOGY_SPECIFICATION.md |
| **Named Individual** | An OWL 2 entity with a unique IRI; in K3D, each MeaningCentricStar is a Named Individual | OWL 2 Spec |
| **Content-Addressed IRI** | An IRI derived from the content hash of a star's meaning_rpn program | K3D Formal Ontology §4.2 |
| **Ternary Annotation** | An OWL 2 annotation property encoding the +1/0/−1 assertion state | This document §13 |
| **Procedural Literal** | An RDF literal of type `k3d:RPNProgram` carrying an executable RPN program | This document §11 |

---

## 2. Scope and Design Rationale

### 2.1 What This Ontology Defines

1. **OWL 2 DL class hierarchy** mapping the K3D Upper Ontology Diamond and all subcategories.
2. **Object properties** for the 12 ontological relations (isA, partOf, references, transforms, perceives, inhabits, holds, navigatesTo, convergesOn, consolidatesTo, defeasiblyOverrides, semanticForce).
3. **Datatype properties** for procedural content (RPN programs, embeddings, positions, timestamps).
4. **Annotation properties** for ternary assertion state, dual-client modality, and provenance.
5. **OWL 2 axioms** encoding the 8 normative invariants as formal restrictions.
6. **SPARQL query patterns** for common K3D operations (Galaxy navigation, cross-domain bridge discovery, sleep-time provenance).
7. **Extraction rules** from glTF `extras.k3d` to RDF triples.

### 2.2 What This Ontology Does Not Define

- Specific RPN opcodes or their semantics (see RPN Domain Opcode Registry).
- GPU kernel implementations or CUDA memory layout.
- Runtime execution protocols (game loop, sleep-time consolidation, swarm dispatch).
- Network protocols for multi-House federation (Doors protocol).
- Benchmark scoring or evaluation criteria.

### 2.3 Design Rationale: OWL 2 DL (Not OWL 2 Full)

This ontology targets **OWL 2 DL** (Description Logics profile) to ensure decidable reasoning. Key constraints:

- Classes and individuals are strictly separated (no metaclass patterns).
- All property domains and ranges are declared.
- Cardinality restrictions use only non-negative integers.
- No property chains of unbounded length.

The DL profile guarantees that standard OWL reasoners (Pellet, HermiT, ELK) can perform subsumption checking, consistency validation, and instance classification over K3D knowledge graphs.

### 2.4 Relationship to X3D Extension Mechanisms

The X3D Ontology (Brutzman, 2020) maps X3D Architecture nodes to OWL 2 classes using a systematic naming convention: each X3D node type becomes an OWL class in the `x3do:` namespace. PM-KR extends this by:

1. Importing `x3do:` as a dependency — PM-KR classes that correspond to X3D nodes are declared as subclasses of their `x3do:` counterparts.
2. Adding the `k3d:` namespace for all PM-KR-specific classes, properties, and individuals.
3. Preserving the X3D Ontology's field-to-property mapping convention — every X3D field maps to an OWL property; PM-KR adds new fields as new properties.

---

## 3. Namespace Architecture

### 3.1 Namespace Declarations

```turtle
@prefix k3d:    <https://knowledge3d.org/ontology/> .
@prefix k3ds:   <https://knowledge3d.org/ontology/spatial/> .
@prefix k3da:   <https://knowledge3d.org/ontology/agent/> .
@prefix k3dsub: <https://knowledge3d.org/ontology/substrate/> .
@prefix x3do:   <https://www.web3d.org/x3d/ontology/> .
@prefix hanim:  <https://www.web3d.org/x3d/ontology/hanim/> .
@prefix owl:    <http://www.w3.org/2002/07/owl#> .
@prefix rdfs:   <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd:    <http://www.w3.org/2001/XMLSchema#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix prov:   <http://www.w3.org/ns/prov#> .
@prefix geo:    <http://www.opengis.net/ont/geosparql#> .
@prefix skos:   <http://www.w3.org/2004/02/skos/core#> .
```

### 3.2 Namespace Design Rationale

| Prefix | Scope | Rationale |
|--------|-------|-----------|
| `k3d:` | Core ontology — procedures, stars, layers, processes | Primary namespace for all PM-KR knowledge representation classes |
| `k3ds:` | Spatial ontology — House, Galaxy, World, rooms, regions | Separated because spatial relations have distinct formal properties (metric, topological) |
| `k3da:` | Agent ontology — TRM, specialists, swarm, cognitive states | Separated because agent classes carry behavioral contracts that differ from knowledge classes |
| `k3dsub:` | Substrate ontology — VRAM, SSD, PTX, RPN stack, registers | Separated because substrate types have hardware-specific properties orthogonal to knowledge content |
| `x3do:` | X3D Ontology (imported) — base X3D scene graph classes | Established namespace; PM-KR extends, does not replace |
| `hanim:` | HAnim Ontology (imported) — humanoid skeleton classes | Established namespace; avatar embodiment extends HAnim |

### 3.3 Ontology Header

```turtle
<https://knowledge3d.org/ontology/>
    a owl:Ontology ;
    owl:versionIRI <https://knowledge3d.org/ontology/0.1/> ;
    owl:imports <https://www.web3d.org/x3d/ontology/> ;
    owl:imports <https://www.web3d.org/x3d/ontology/hanim/> ;
    rdfs:label "PM-KR X3D Ontology"@en ;
    rdfs:comment """OWL 2 DL formalization of the K3D Formal Ontology for
    procedural knowledge representation in X3D scenes. Extends the X3D Ontology
    (Brutzman & Flotynski) with classes for executable knowledge, spatial
    cognition, embodied agents, and execution substrates."""@en ;
    dcterms:creator "PM-KR Community Group" ;
    dcterms:created "2026-03-26"^^xsd:date ;
    dcterms:license <https://creativecommons.org/licenses/by/4.0/> .
```

---

## 4. Relationship to the X3D Ontology

### 4.1 The X3D Ontology (Brutzman & Flotynski)

The X3D Ontology maps the X3D Architecture (ISO/IEC 19775-1) to OWL 2. Its class hierarchy mirrors the X3D abstract type system:

```
x3do:X3DNode
├── x3do:X3DChildNode
│   ├── x3do:X3DGroupingNode (Transform, Group, Switch, etc.)
│   ├── x3do:X3DShapeNode (Shape)
│   ├── x3do:X3DLightNode (DirectionalLight, PointLight, etc.)
│   └── x3do:X3DSensorNode (TouchSensor, ProximitySensor, etc.)
├── x3do:X3DGeometryNode (IndexedFaceSet, Sphere, etc.)
├── x3do:X3DAppearanceNode (Appearance, Material, etc.)
└── x3do:X3DMetadataNode (MetadataString, MetadataFloat, etc.)
```

Each X3D field becomes an OWL datatype or object property. Each ROUTE becomes an OWL object property connecting output to input. Each scene graph parent-child relationship is an OWL object property (`x3do:hasChild`).

### 4.2 PM-KR Extension Points

PM-KR extends the X3D Ontology at six specific points, each corresponding to a node type defined in the PM-KR X3D Procedural Memory Component:

| X3D Ontology Class | PM-KR Extension | Rationale |
|--------------------|----------------|-----------|
| `x3do:X3DChildNode` | `k3d:X3DProceduralNode` (abstract) | Base for all PM-KR nodes; adds layer, program, canonical ID |
| `x3do:X3DGroupingNode` | `k3d:GalaxyGroup` | Galaxy as a semantic grouping with domain, entry count, substrate |
| `x3do:X3DChildNode` | `k3d:MeaningCentricStar` | The atomic knowledge unit; carries RPN programs, symlinks, embedding |
| `x3do:X3DGroupingNode` | `k3d:HouseRoom` | Room in the Memory Palace; intentional organization container |
| `hanim:HAnimHumanoid` | `k3da:AvatarBody` | HAnim humanoid with Cranial Galaxy and cognitive architecture |
| `x3do:X3DChildNode` | `k3da:AgentEntity` | Autonomous reasoning entity with game loop semantics |

### 4.3 Import Semantics

```turtle
# PM-KR imports and extends the X3D Ontology
<https://knowledge3d.org/ontology/>
    owl:imports <https://www.web3d.org/x3d/ontology/> .

# All PM-KR abstract node types are subclasses of x3do classes
k3d:X3DProceduralNode rdfs:subClassOf x3do:X3DChildNode .
```

This ensures that any OWL reasoner that loads the PM-KR ontology also loads the X3D Ontology, and all PM-KR individuals are valid X3D scene graph members.

---

## 5. Upper Ontology: OWL 2 Formalization

### 5.1 The K3D Ontological Diamond in OWL 2

The K3D Formal Ontology defines four foundational categories: **Procedure**, **Space**, **Agent**, **Substrate**. These map to four disjoint top-level OWL 2 classes:

```turtle
# ── Top-Level Classes (The Ontological Diamond) ──

k3d:Thing a owl:Class ;
    rdfs:label "K3D Thing"@en ;
    rdfs:comment "Top-level class for all K3D ontological individuals."@en .

k3d:Procedure a owl:Class ;
    rdfs:subClassOf k3d:Thing ;
    rdfs:label "Procedure"@en ;
    rdfs:comment """Executable knowledge — an RPN program that IS a knowledge
    individual. Classification derives from what the program computes, not
    from declared labels."""@en .

k3ds:Space a owl:Class ;
    rdfs:subClassOf k3d:Thing ;
    rdfs:label "Space"@en ;
    rdfs:comment """Spatial domain where procedures exist and agents act.
    Position in space IS semantic meaning, not a property attached to it."""@en .

k3da:Agent a owl:Class ;
    rdfs:subClassOf k3d:Thing ;
    rdfs:label "Agent"@en ;
    rdfs:comment """Embodied entity that perceives, navigates, reasons about,
    and transforms Procedures within Space."""@en .

k3dsub:Substrate a owl:Class ;
    rdfs:subClassOf k3d:Thing ;
    rdfs:label "Substrate"@en ;
    rdfs:comment """Execution environment with specific properties: volatility,
    sovereignty, parallelism, encoding. WHERE computation happens is
    ontologically meaningful."""@en .

# Disjointness axiom — the four categories are mutually exclusive
[] a owl:AllDisjointClasses ;
    owl:members ( k3d:Procedure k3ds:Space k3da:Agent k3dsub:Substrate ) .
```

### 5.2 Cross-Cutting Commitment Classes

Three ontological commitments apply across all four categories:

```turtle
# ── Ternary State (applies to all assertions) ──

k3d:TernaryAssertion a owl:Class ;
    rdfs:label "Ternary Assertion"@en ;
    rdfs:comment """An assertion carrying polarity: affirmed (+1),
    unknown (0), or negated (−1). Replaces classical OWA."""@en .

# ── Dual-Client Modality ──

k3d:DualClientEntity a owl:Class ;
    rdfs:label "Dual-Client Entity"@en ;
    rdfs:comment """An entity with two presentation faces: human-perceivable
    (visual/auditory) and machine-perceivable (procedural/executable),
    both derived from the same underlying Procedure."""@en .

# ── Temporal Phase ──

k3d:TemporalPhase a owl:Class ;
    owl:oneOf ( k3d:Nascent k3d:Active k3d:Persistent k3d:Archived ) ;
    rdfs:label "Temporal Phase"@en ;
    rdfs:comment "Lifecycle phase of an ontological individual."@en .

k3d:Nascent a owl:NamedIndividual, k3d:TemporalPhase ;
    rdfs:label "Nascent"@en ;
    rdfs:comment "Ingestion Stargate — raw data being transmuted into RPN."@en .

k3d:Active a owl:NamedIndividual, k3d:TemporalPhase ;
    rdfs:label "Active"@en ;
    rdfs:comment "Galaxy VRAM — currently loaded for reasoning."@en .

k3d:Persistent a owl:NamedIndividual, k3d:TemporalPhase ;
    rdfs:label "Persistent"@en ;
    rdfs:comment "House SSD — consolidated, intentionally placed."@en .

k3d:Archived a owl:NamedIndividual, k3d:TemporalPhase ;
    rdfs:label "Archived"@en ;
    rdfs:comment "Museum — deprecated, preserved for audit."@en .
```

---

## 6. Procedural Knowledge Classes

### 6.1 The Star as OWL Individual

The `MeaningCentricStar` is the atomic knowledge unit in K3D. In OWL 2, each star is a Named Individual whose IRI is derived from its content-addressed star_id:

```turtle
k3d:MeaningCentricStar a owl:Class ;
    rdfs:subClassOf k3d:Procedure, k3d:DualClientEntity ;
    rdfs:subClassOf x3do:X3DChildNode ;
    rdfs:label "Meaning-Centric Star"@en ;
    rdfs:comment """The atomic knowledge unit. An RPN program with
    content-addressed identity (star_id = hash(meaning_rpn)),
    four-layer classification, dual-client presentation, and
    ternary assertion state."""@en .
```

### 6.2 Meaning Class Hierarchy (Five Ontological Kinds)

```turtle
k3d:ConceptStar a owl:Class ;
    rdfs:subClassOf k3d:MeaningCentricStar ;
    rdfs:label "Concept Star"@en ;
    rdfs:comment "A thing, entity, or category (substance)."@en .

k3d:RelationStar a owl:Class ;
    rdfs:subClassOf k3d:MeaningCentricStar ;
    rdfs:label "Relation Star"@en ;
    rdfs:comment "A connection between concepts (structure)."@en .

k3d:ActionStar a owl:Class ;
    rdfs:subClassOf k3d:MeaningCentricStar ;
    rdfs:label "Action Star"@en ;
    rdfs:comment "A process or transformation (dynamics)."@en .

k3d:PropertyStar a owl:Class ;
    rdfs:subClassOf k3d:MeaningCentricStar ;
    rdfs:label "Property Star"@en ;
    rdfs:comment "An attribute of concepts (quality)."@en .

k3d:MetaStar a owl:Class ;
    rdfs:subClassOf k3d:MeaningCentricStar ;
    rdfs:label "Meta Star"@en ;
    rdfs:comment "A rule about rules (strategy)."@en .

# Disjoint and covering — every star is exactly one kind
[] a owl:AllDisjointClasses ;
    owl:members ( k3d:ConceptStar k3d:RelationStar k3d:ActionStar
                  k3d:PropertyStar k3d:MetaStar ) .

k3d:MeaningCentricStar owl:equivalentClass [
    a owl:Class ;
    owl:unionOf ( k3d:ConceptStar k3d:RelationStar k3d:ActionStar
                  k3d:PropertyStar k3d:MetaStar )
] .
```

### 6.3 Four-Layer Procedure Classes

The PM-KR four-layer hierarchy maps to OWL subclasses of `k3d:Procedure`:

```turtle
k3d:FormProcedure a owl:Class ;
    rdfs:subClassOf k3d:Procedure ;
    rdfs:label "Form Procedure (Layer 1)"@en ;
    rdfs:comment """Perceptual level: glyphs, shapes, sounds, surface forms.
    How entities APPEAR to senses. Self-contained — no upward
    dependencies."""@en .

k3d:MeaningProcedure a owl:Class ;
    rdfs:subClassOf k3d:Procedure ;
    rdfs:label "Meaning Procedure (Layer 2)"@en ;
    rdfs:comment """Conceptual level: concepts, definitions, relations.
    What entities ARE, language-agnostic. Identity center —
    star_id derives from meaning_rpn."""@en .

k3d:RuleProcedure a owl:Class ;
    rdfs:subClassOf k3d:Procedure ;
    rdfs:label "Rule Procedure (Layer 3)"@en ;
    rdfs:comment """Operational level: transformations, grammars, laws.
    How entities CHANGE and interact. MUST reference Layer 1 and
    Layer 2 via symlinks."""@en .

k3d:MetaRuleProcedure a owl:Class ;
    rdfs:subClassOf k3d:Procedure ;
    rdfs:label "Meta-Rule Procedure (Layer 4)"@en ;
    rdfs:comment """Strategic level: strategies, priorities, self-reflection.
    When and why to apply which operations. MUST reference Layer 3
    via symlinks."""@en .

k3d:CompositeProcedure a owl:Class ;
    rdfs:subClassOf k3d:Procedure ;
    rdfs:label "Composite Procedure"@en ;
    rdfs:comment """Cross-layer composition that creates ontological novelty.
    References procedures from multiple layers."""@en .
```

### 6.4 Galaxy as OWL Class

```turtle
k3d:Galaxy a owl:Class ;
    rdfs:subClassOf k3ds:GalaxySpace ;
    rdfs:subClassOf x3do:X3DGroupingNode ;
    rdfs:label "Galaxy"@en ;
    rdfs:comment """A named domain collection of MeaningCentricStars in VRAM.
    Galaxies are simultaneously loaded — no selection or swapping."""@en .

# Named Galaxy individuals (the default set)
k3d:DrawingGalaxy  a owl:NamedIndividual, k3d:Galaxy ;
    rdfs:label "Drawing Galaxy"@en .
k3d:CharacterGalaxy a owl:NamedIndividual, k3d:Galaxy ;
    rdfs:label "Character Galaxy"@en .
k3d:WordGalaxy     a owl:NamedIndividual, k3d:Galaxy ;
    rdfs:label "Word Galaxy"@en .
k3d:NumberGalaxy   a owl:NamedIndividual, k3d:Galaxy ;
    rdfs:label "Number Galaxy"@en .
k3d:GrammarGalaxy  a owl:NamedIndividual, k3d:Galaxy ;
    rdfs:label "Grammar Galaxy"@en .
k3d:MathGalaxy     a owl:NamedIndividual, k3d:Galaxy ;
    rdfs:label "Math Galaxy"@en .
k3d:RealityGalaxy  a owl:NamedIndividual, k3d:Galaxy ;
    rdfs:label "Reality Galaxy"@en .
k3d:AudioGalaxy    a owl:NamedIndividual, k3d:Galaxy ;
    rdfs:label "Audio Galaxy"@en .
k3d:ObjectsGalaxy  a owl:NamedIndividual, k3d:Galaxy ;
    rdfs:label "3D Objects Galaxy"@en .
k3d:ToolGalaxy     a owl:NamedIndividual, k3d:Galaxy ;
    rdfs:label "Tool Galaxy"@en .
```

---

## 7. Spatial Ontology Classes

### 7.1 Three Spatial Domains

```turtle
# ── House Space (Intentional, Persistent) ──

k3ds:HouseSpace a owl:Class ;
    rdfs:subClassOf k3ds:Space ;
    rdfs:label "House Space"@en ;
    rdfs:comment """Persistent 3D Memory Palace. The digital analogy to the
    40,000-year-old Method of Loci. Shared spatial reality for humans AND AI.
    Organization is intentional — the TRM places knowledge deliberately,
    like a librarian shelving books."""@en .

k3ds:Room a owl:Class ;
    rdfs:subClassOf k3ds:HouseSpace ;
    rdfs:label "Room"@en ;
    rdfs:comment "A named spatial partition within the House."@en .

k3ds:Library     a owl:Class ; rdfs:subClassOf k3ds:Room .
k3ds:Workshop    a owl:Class ; rdfs:subClassOf k3ds:Room .
k3ds:Garden      a owl:Class ; rdfs:subClassOf k3ds:Room .
k3ds:Bathtub     a owl:Class ; rdfs:subClassOf k3ds:Room .
k3ds:LivingRoom  a owl:Class ; rdfs:subClassOf k3ds:Room .
k3ds:Museum      a owl:Class ; rdfs:subClassOf k3ds:Room .

k3ds:Furniture a owl:Class ;
    rdfs:subClassOf k3ds:HouseSpace ;
    rdfs:label "Furniture"@en ;
    rdfs:comment "A spatial container within a room (shelf, workbench, tree, display)."@en .

k3ds:Artifact a owl:Class ;
    rdfs:subClassOf k3ds:HouseSpace ;
    rdfs:label "Artifact"@en ;
    rdfs:comment """An object placed on furniture. Books, tools, tablets,
    instruments. A Book's CONTENTS are a Galaxy when loaded."""@en .

# ── Galaxy Space (Gravitational, Volatile) ──

k3ds:GalaxySpace a owl:Class ;
    rdfs:subClassOf k3ds:Space ;
    rdfs:label "Galaxy Space"@en ;
    rdfs:comment """Volatile VRAM workspace — the AI's internal brain.
    Organization is fluid: semantic gravity cohered by meaning.
    Stars attract (+1), repel (−1), or float (0)."""@en .

k3ds:Neighborhood a owl:Class ;
    rdfs:subClassOf k3ds:GalaxySpace ;
    rdfs:label "Neighborhood"@en ;
    rdfs:comment "Gravitational cluster of semantically related stars."@en .

k3ds:NavigationPath a owl:Class ;
    rdfs:subClassOf k3ds:GalaxySpace ;
    rdfs:label "Navigation Path"@en ;
    rdfs:comment "LED-A* trace connecting seed to focus through Galaxy."@en .

# ── World Space (Networked, Federated) ──

k3ds:WorldSpace a owl:Class ;
    rdfs:subClassOf k3ds:Space ;
    rdfs:label "World Space"@en ;
    rdfs:comment """Network of Houses connected by Doors. Multi-user
    collaboration space. Each House is sovereign; Doors are bilateral."""@en .

k3ds:RemoteHouse a owl:Class ;
    rdfs:subClassOf k3ds:WorldSpace ;
    rdfs:label "Remote House"@en ;
    rdfs:comment "A House reachable via Door protocol (k3d:// URI)."@en .

k3ds:SharedGalaxy a owl:Class ;
    rdfs:subClassOf k3ds:WorldSpace ;
    rdfs:label "Shared Galaxy"@en ;
    rdfs:comment "Multi-agent workspace for collective reasoning."@en .
```

### 7.2 Spatial Relation Properties

```turtle
k3ds:containedIn a owl:ObjectProperty, owl:TransitiveProperty ;
    rdfs:domain k3d:Thing ;
    rdfs:range k3ds:HouseSpace ;
    rdfs:label "contained in"@en ;
    rdfs:comment "Physical containment in House space (book on shelf in room)."@en .

k3ds:nearTo a owl:ObjectProperty, owl:SymmetricProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range k3d:MeaningCentricStar ;
    rdfs:label "near to"@en ;
    rdfs:comment "Semantic proximity in Galaxy space (embedding similarity)."@en .

k3ds:gravitatesTo a owl:ObjectProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range k3d:MeaningCentricStar ;
    rdfs:label "gravitates to"@en ;
    rdfs:comment "Ternary attraction (+1 affinity) in Galaxy."@en .

k3ds:repels a owl:ObjectProperty, owl:SymmetricProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range k3d:MeaningCentricStar ;
    rdfs:label "repels"@en ;
    rdfs:comment "Ternary repulsion (−1 contradiction) in Galaxy."@en .

k3ds:doorTo a owl:ObjectProperty, owl:SymmetricProperty ;
    rdfs:domain k3ds:HouseSpace ;
    rdfs:range k3ds:HouseSpace ;
    rdfs:label "door to"@en ;
    rdfs:comment "Bilateral network connection between Houses."@en .

k3ds:frustumVisible a owl:ObjectProperty ;
    rdfs:domain k3da:Agent ;
    rdfs:range k3d:Thing ;
    rdfs:label "frustum visible"@en ;
    rdfs:comment "Within the perceiving agent's field of view."@en .
```

### 7.3 Semantic Gravity Formalization

The ternary semantic force between stars is modeled as a reified relation:

```turtle
k3d:SemanticForce a owl:Class ;
    rdfs:label "Semantic Force"@en ;
    rdfs:comment """Reified ternary gravitational relation between two stars.
    F = T(s₁,s₂) × M(s₁) × M(s₂) / d²
    where T is ternary polarity, M is meaning mass, d is distance."""@en .

k3d:forceSource a owl:ObjectProperty ;
    rdfs:domain k3d:SemanticForce ;
    rdfs:range k3d:MeaningCentricStar .

k3d:forceTarget a owl:ObjectProperty ;
    rdfs:domain k3d:SemanticForce ;
    rdfs:range k3d:MeaningCentricStar .

k3d:forcePolarity a owl:DatatypeProperty ;
    rdfs:domain k3d:SemanticForce ;
    rdfs:range k3d:TritValue ;
    rdfs:comment "Ternary polarity: +1 (attraction), 0 (neutral), −1 (repulsion)."@en .

k3d:forceMagnitude a owl:DatatypeProperty ;
    rdfs:domain k3d:SemanticForce ;
    rdfs:range xsd:float ;
    rdfs:comment "Computed magnitude of the gravitational force."@en .
```

---

## 8. Agent Ontology Classes

### 8.1 Agent Taxonomy

```turtle
k3da:HumanAgent a owl:Class ;
    rdfs:subClassOf k3da:Agent ;
    rdfs:label "Human Agent"@en ;
    rdfs:comment """Human inhabitant. Perception via camera. Cognition is
    external (human consciousness). Empty cranial volume."""@en .

k3da:TRMAgent a owl:Class ;
    rdfs:subClassOf k3da:Agent ;
    rdfs:label "TRM Agent"@en ;
    rdfs:comment """Primary AI entity. Lives in House, thinks in Galaxy,
    runs as game loop. ~7M parameters + specialist swarm. TRM IS the
    avatar — not a function Python calls."""@en .

k3da:AssistantAgent a owl:Class ;
    rdfs:subClassOf k3da:Agent ;
    rdfs:label "Assistant Agent"@en ;
    rdfs:comment "Simplified AI with subset of TRM capabilities."@en .

k3da:ServiceAgent a owl:Class ;
    rdfs:subClassOf k3da:Agent ;
    rdfs:label "Service Agent"@en ;
    rdfs:comment "Headless agent responding via Tablet/Door channels."@en .

k3da:RoboticAgent a owl:Class ;
    rdfs:subClassOf k3da:Agent ;
    rdfs:label "Robotic Agent"@en ;
    rdfs:comment """Physical embodiment with actuator mapping.
    SLAM → House Universe (physical space becomes K3D House)."""@en .

# Disjoint agent types
[] a owl:AllDisjointClasses ;
    owl:members ( k3da:HumanAgent k3da:TRMAgent k3da:AssistantAgent
                  k3da:ServiceAgent k3da:RoboticAgent ) .
```

### 8.2 Internal Cognitive Architecture

```turtle
k3da:Specialist a owl:Class ;
    rdfs:subClassOf k3da:Agent ;
    rdfs:label "Specialist"@en ;
    rdfs:comment """Domain-biased reasoning unit (LoRA adapter + Galaxy
    navigation bias). Created during sleep-time, activated during query,
    pruned when unused. Computational analogue of brain regions."""@en .

k3da:SwarmWorker a owl:Class ;
    rdfs:label "Swarm Worker"@en ;
    rdfs:comment """Instantiation of a Specialist on a parallel execution core.
    Nine workers operate simultaneously, communicating via STORE/RECALL
    registers."""@en .

k3da:HaltingGate a owl:Class ;
    rdfs:label "Halting Gate"@en ;
    rdfs:comment """Ternary convergence checker. Converged (+1),
    diverging (−1), still processing (0)."""@en .

k3da:ShadowCopy a owl:Class ;
    rdfs:label "Shadow Copy"@en ;
    rdfs:comment """Inference-time learning buffer. Records successful
    reasoning traces for sleep-time consolidation."""@en .

k3da:NavigationTrace a owl:Class ;
    rdfs:label "Navigation Trace"@en ;
    rdfs:comment """LED-A* path through Galaxy space. Connects seed nodes
    to focus nodes via intermediate hops. Computational analogue of
    chains of thought — spatial and inspectable."""@en .

k3da:CranialGalaxy a owl:Class ;
    rdfs:subClassOf k3ds:GalaxySpace ;
    rdfs:label "Cranial Galaxy"@en ;
    rdfs:comment """Live Galaxy Universe inside an avatar's skull volume.
    Bounded grouping node with Galaxy-to-skull coordinate mapping."""@en .

# TRM Agent composition restrictions
k3da:TRMAgent rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty k3da:hasCranialGalaxy ;
    owl:qualifiedCardinality "1"^^xsd:nonNegativeInteger ;
    owl:onClass k3da:CranialGalaxy
] .

k3da:TRMAgent rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty k3da:hasHaltingGate ;
    owl:qualifiedCardinality "1"^^xsd:nonNegativeInteger ;
    owl:onClass k3da:HaltingGate
] .

k3da:TRMAgent rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty k3da:hasSwarmWorker ;
    owl:minQualifiedCardinality "1"^^xsd:nonNegativeInteger ;
    owl:onClass k3da:SwarmWorker
] .
```

### 8.3 Interaction Ontology

```turtle
k3da:Interaction a owl:Class ;
    rdfs:label "Interaction"@en ;
    rdfs:comment "Spatial action between agent and world or agents."@en .

k3da:Navigate    a owl:Class ; rdfs:subClassOf k3da:Interaction .
k3da:Perceive    a owl:Class ; rdfs:subClassOf k3da:Interaction .
k3da:Reach       a owl:Class ; rdfs:subClassOf k3da:Interaction .
k3da:Grasp       a owl:Class ; rdfs:subClassOf k3da:Interaction .
k3da:Use         a owl:Class ; rdfs:subClassOf k3da:Interaction .
k3da:Share       a owl:Class ; rdfs:subClassOf k3da:Interaction .
k3da:Speak       a owl:Class ; rdfs:subClassOf k3da:Interaction .
k3da:Create      a owl:Class ; rdfs:subClassOf k3da:Interaction .

k3da:interactionAgent a owl:ObjectProperty ;
    rdfs:domain k3da:Interaction ;
    rdfs:range k3da:Agent .

k3da:interactionTarget a owl:ObjectProperty ;
    rdfs:domain k3da:Interaction ;
    rdfs:range k3d:Thing .

k3da:interactionResult a owl:ObjectProperty ;
    rdfs:domain k3da:Interaction ;
    rdfs:range k3d:Thing .
```

---

## 9. Substrate Ontology Classes

### 9.1 Execution Environment Taxonomy

```turtle
k3dsub:VRAMSubstrate a owl:Class ;
    rdfs:subClassOf k3dsub:Substrate ;
    rdfs:label "VRAM Substrate"@en ;
    rdfs:comment "Galaxy Universe. Volatile, fast, parallel."@en .

k3dsub:SSDSubstrate a owl:Class ;
    rdfs:subClassOf k3dsub:Substrate ;
    rdfs:label "SSD Substrate"@en ;
    rdfs:comment "House persistence. Non-volatile, large."@en .

k3dsub:PTXKernel a owl:Class ;
    rdfs:subClassOf k3dsub:Substrate ;
    rdfs:label "PTX Kernel"@en ;
    rdfs:comment "Sovereign execution primitive. Deterministic, inspectable."@en .

k3dsub:RPNStack a owl:Class ;
    rdfs:subClassOf k3dsub:Substrate ;
    rdfs:label "RPN Stack"@en ;
    rdfs:comment """Per-core execution context. 69-depth, checkpointable,
    forkable."""@en .

k3dsub:Register a owl:Class ;
    rdfs:subClassOf k3dsub:Substrate ;
    rdfs:label "Register"@en ;
    rdfs:comment "STORE/RECALL. Cross-specialist communication."@en .

k3dsub:NetworkSubstrate a owl:Class ;
    rdfs:subClassOf k3dsub:Substrate ;
    rdfs:label "Network Substrate"@en ;
    rdfs:comment "Doors protocol. Distributed, federated."@en .

k3dsub:ProceduralDisplaySubstrate a owl:Class ;
    rdfs:subClassOf k3dsub:Substrate ;
    rdfs:label "Procedural Display Substrate"@en ;
    rdfs:comment """Future: output device that executes procedural programs
    directly rather than receiving rasterized frames. Knowledge emitted
    to a procedural display retains ontological identity as executable
    procedure — not degraded to pixels."""@en .

k3dsub:TernarySubstrate a owl:Class ;
    rdfs:subClassOf k3dsub:Substrate ;
    rdfs:label "Ternary Substrate"@en ;
    rdfs:comment """Future: native trit-vector execution environment where
    each balanced trit (−1/0/+1) carries value + certainty + polarity
    without binary encoding overhead. The ontological commitment becomes
    native to the hardware."""@en .
```

### 9.2 Substrate Properties

```turtle
k3dsub:isVolatile a owl:DatatypeProperty ;
    rdfs:domain k3dsub:Substrate ;
    rdfs:range xsd:boolean ;
    rdfs:comment "Whether data survives power cycles."@en .

k3dsub:isSovereign a owl:DatatypeProperty ;
    rdfs:domain k3dsub:Substrate ;
    rdfs:range xsd:boolean ;
    rdfs:comment "Whether execution requires only self-contained primitives."@en .

k3dsub:parallelism a owl:DatatypeProperty ;
    rdfs:domain k3dsub:Substrate ;
    rdfs:range xsd:nonNegativeInteger ;
    rdfs:comment "Number of parallel execution lanes."@en .

k3dsub:encoding a owl:DatatypeProperty ;
    rdfs:domain k3dsub:Substrate ;
    rdfs:range xsd:string ;
    rdfs:comment """Encoding type: 'binary' (current) or 'ternary' (target).
    When ternary accelerators arrive, the ontology migrates without
    semantic change — only the substrate encoding layer changes."""@en .

k3dsub:executesOn a owl:ObjectProperty ;
    rdfs:domain k3d:Procedure ;
    rdfs:range k3dsub:Substrate ;
    rdfs:label "executes on"@en ;
    rdfs:comment "Links a procedure to its execution substrate."@en .
```

---

## 10. Object Properties

### 10.1 Core Ontological Relations

```turtle
# ── Taxonomic ──

k3d:isA a owl:ObjectProperty, owl:TransitiveProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range k3d:MeaningCentricStar ;
    rdfs:label "is a"@en ;
    rdfs:comment """Taxonomic subsumption. Spatial expression: vertical
    (child below parent on ontological tree). Defeasible — carries
    ternary state."""@en .

# ── Mereological ──

k3d:partOf a owl:ObjectProperty, owl:TransitiveProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range k3d:MeaningCentricStar ;
    rdfs:label "part of"@en ;
    rdfs:comment """Mereological containment. Spatial expression:
    component inside composite (atom inside molecule)."""@en .

# ── Referential (Symlink) ──

k3d:references a owl:ObjectProperty ;
    rdfs:domain k3d:Procedure ;
    rdfs:range k3d:Procedure ;
    rdfs:label "references"@en ;
    rdfs:comment """Canonical symlink reference. Higher-layer procedures
    MUST reference lower-layer via this property. Inlining is an
    ontological violation."""@en .

# ── Transformative ──

k3d:transforms a owl:ObjectProperty ;
    rdfs:domain k3d:RuleProcedure ;
    rdfs:range k3d:MeaningCentricStar ;
    rdfs:label "transforms"@en ;
    rdfs:comment "Rule operates on star. Directional path."@en .

# ── Perceptual ──

k3d:perceives a owl:ObjectProperty ;
    rdfs:domain k3da:Agent ;
    rdfs:range k3d:MeaningCentricStar ;
    rdfs:label "perceives"@en ;
    rdfs:comment "Agent has star in frustum. Dynamic — changes with position."@en .

# ── Spatial ──

k3d:inhabits a owl:ObjectProperty ;
    rdfs:domain k3da:Agent ;
    rdfs:range k3ds:Space ;
    rdfs:label "inhabits"@en ;
    rdfs:comment "Agent lives in space. Body at (x,y,z) in House."@en .

k3d:holds a owl:ObjectProperty ;
    rdfs:domain k3da:Agent ;
    rdfs:range k3ds:Artifact ;
    rdfs:label "holds"@en ;
    rdfs:comment "Object attached to agent's hand site."@en .

# ── Navigation ──

k3d:navigatesTo a owl:ObjectProperty ;
    rdfs:domain k3da:Agent ;
    rdfs:range k3d:MeaningCentricStar ;
    rdfs:label "navigates to"@en ;
    rdfs:comment "LED-A* pathfinding trace through Galaxy."@en .

# ── Convergence ──

k3d:convergesOn a owl:ObjectProperty ;
    rdfs:domain k3da:SwarmWorker ;
    rdfs:range k3d:MeaningCentricStar ;
    rdfs:label "converges on"@en ;
    rdfs:comment "Swarm traces merge to single answer point."@en .

# ── Consolidation ──

k3d:consolidatesTo a owl:ObjectProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range k3d:MeaningCentricStar ;
    rdfs:label "consolidates to"@en ;
    rdfs:comment "Sleep-time promotion: Galaxy → House."@en .

# ── Defeasible ──

k3d:defeasiblyOverrides a owl:ObjectProperty ;
    rdfs:domain k3d:RuleProcedure ;
    rdfs:range k3d:RuleProcedure ;
    rdfs:label "defeasibly overrides"@en ;
    rdfs:comment """Specific rule defeats general rule. Superiority edge
    in defeasible rule graph."""@en .

# ── Galaxy Membership ──

k3d:memberOf a owl:ObjectProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range k3d:Galaxy ;
    rdfs:label "member of"@en ;
    rdfs:comment "Star belongs to a named Galaxy domain."@en .
```

---

## 11. Datatype Properties

### 11.1 Custom Datatypes

```turtle
# ── RPN Program Literal ──

k3d:RPNProgram a rdfs:Datatype ;
    rdfs:label "RPN Program"@en ;
    rdfs:comment """An RPN instruction sequence encoded as a space-delimited
    string of opcodes and operands. Executable on a K3D stack machine."""@en .

# ── Trit Value ──

k3d:TritValue a rdfs:Datatype ;
    owl:equivalentClass [
        a rdfs:Datatype ;
        owl:onDatatype xsd:integer ;
        owl:withRestrictions (
            [ xsd:minInclusive "-1"^^xsd:integer ]
            [ xsd:maxInclusive "1"^^xsd:integer ]
        )
    ] ;
    rdfs:label "Trit Value"@en ;
    rdfs:comment "+1 (affirmed), 0 (unknown), −1 (negated)."@en .

# ── 3D Position ──

k3d:Vec3 a rdfs:Datatype ;
    rdfs:label "3D Vector"@en ;
    rdfs:comment "Space-delimited 'x y z' float triple."@en .

# ── Embedding Vector ──

k3d:EmbeddingVector a rdfs:Datatype ;
    rdfs:label "Embedding Vector"@en ;
    rdfs:comment "JSON array of floats representing semantic embedding."@en .
```

### 11.2 Star Datatype Properties

```turtle
k3d:starId a owl:DatatypeProperty, owl:FunctionalProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range xsd:hexBinary ;
    rdfs:label "star ID"@en ;
    rdfs:comment """Content-addressed identifier: hash(meaning_rpn).
    Two systems defining the same concept produce the same star_id."""@en .

k3d:meaningProgram a owl:DatatypeProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range k3d:RPNProgram ;
    rdfs:label "meaning program"@en ;
    rdfs:comment "The RPN program that IS the star's meaning (Layer 2)."@en .

k3d:visualProgram a owl:DatatypeProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range k3d:RPNProgram ;
    rdfs:label "visual program"@en ;
    rdfs:comment "The RPN program for human-visible form (Layer 1)."@en .

k3d:behaviorProgram a owl:DatatypeProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range k3d:RPNProgram ;
    rdfs:label "behavior program"@en ;
    rdfs:comment "The RPN program for executable behavior."@en .

k3d:embedding a owl:DatatypeProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range k3d:EmbeddingVector ;
    rdfs:label "embedding"@en ;
    rdfs:comment "Semantic embedding vector for Galaxy positioning."@en .

k3d:housePosition a owl:DatatypeProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range k3d:Vec3 ;
    rdfs:label "house position"@en ;
    rdfs:comment "Intentional placement in House space."@en .

k3d:galaxyPosition a owl:DatatypeProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range k3d:Vec3 ;
    rdfs:label "galaxy position"@en ;
    rdfs:comment "Gravitational position in Galaxy working memory."@en .

k3d:meaningMass a owl:DatatypeProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range xsd:float ;
    rdfs:label "meaning mass"@en ;
    rdfs:comment """Richness of connections. Heavily-connected concepts have
    high mass and act as gravitational centers."""@en .

k3d:lodLevel a owl:DatatypeProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range xsd:nonNegativeInteger ;
    rdfs:label "LOD level"@en ;
    rdfs:comment "Level of detail (512 = close, 64 = distant)."@en .

k3d:layer a owl:DatatypeProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range xsd:nonNegativeInteger ;
    rdfs:label "layer"@en ;
    rdfs:comment "PM-KR layer: 1=Form, 2=Meaning, 3=Rules, 4=Meta-Rules."@en .

k3d:confidence a owl:DatatypeProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range xsd:float ;
    rdfs:label "confidence"@en ;
    rdfs:comment "Confidence weight [0.0, 1.0]."@en .

k3d:domain a owl:DatatypeProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range xsd:string ;
    rdfs:label "domain path"@en ;
    rdfs:comment "House organization path (e.g., Library/Biology/Mammalia)."@en .
```

### 11.3 Surface Form Properties

```turtle
k3d:SurfaceForm a owl:Class ;
    rdfs:label "Surface Form"@en ;
    rdfs:comment """A language-specific rendering of a meaning star.
    Multiple surface forms reference one star. Identity is at the
    meaning level, not the surface level."""@en .

k3d:hasSurfaceForm a owl:ObjectProperty ;
    rdfs:domain k3d:MeaningCentricStar ;
    rdfs:range k3d:SurfaceForm .

k3d:language a owl:DatatypeProperty ;
    rdfs:domain k3d:SurfaceForm ;
    rdfs:range xsd:language ;
    rdfs:comment "BCP 47 language tag."@en .

k3d:wordRef a owl:ObjectProperty ;
    rdfs:domain k3d:SurfaceForm ;
    rdfs:range k3d:MeaningCentricStar ;
    rdfs:comment "Reference to the Word Galaxy entry (symlink, not copy)."@en .
```

---

## 12. Annotation Properties

### 12.1 Ternary Assertion Annotations

```turtle
k3d:ternaryState a owl:AnnotationProperty ;
    rdfs:range k3d:TritValue ;
    rdfs:label "ternary state"@en ;
    rdfs:comment """+1 (affirmed), 0 (unknown), −1 (negated).
    Applies to any OWL axiom or assertion to encode ternary polarity.
    This replaces OWA for K3D knowledge."""@en .

k3d:ruleStrength a owl:AnnotationProperty ;
    rdfs:range k3d:TritValue ;
    rdfs:label "rule strength"@en ;
    rdfs:comment """+1 (strict — cannot be defeated), 0 (defeasible —
    default that holds unless contradicted), −1 (defeater — blocks
    conclusion without asserting alternative)."""@en .
```

### 12.2 Provenance Annotations

```turtle
k3d:createdBy a owl:AnnotationProperty ;
    rdfs:subPropertyOf prov:wasGeneratedBy ;
    rdfs:label "created by"@en ;
    rdfs:comment "Agent that created or modified this individual."@en .

k3d:createdAt a owl:AnnotationProperty ;
    rdfs:subPropertyOf prov:generatedAtTime ;
    rdfs:range xsd:dateTime ;
    rdfs:label "created at"@en .

k3d:version a owl:AnnotationProperty ;
    rdfs:subPropertyOf dcterms:hasVersion ;
    rdfs:range xsd:string ;
    rdfs:label "version"@en ;
    rdfs:comment "Brain model version (e.g., v1.0.3)."@en .

k3d:sleepCycle a owl:AnnotationProperty ;
    rdfs:range xsd:nonNegativeInteger ;
    rdfs:label "sleep cycle"@en ;
    rdfs:comment "Which consolidation cycle produced this individual."@en .

k3d:provenance a owl:AnnotationProperty ;
    rdfs:subPropertyOf dcterms:source ;
    rdfs:label "provenance"@en ;
    rdfs:comment "External source from which this knowledge was ingested."@en .
```

### 12.3 Dual-Client Annotations

```turtle
k3d:humanPresentation a owl:AnnotationProperty ;
    rdfs:label "human presentation"@en ;
    rdfs:comment """Description of how this individual appears to human
    perception (UV Map 0 channel)."""@en .

k3d:machinePresentation a owl:AnnotationProperty ;
    rdfs:label "machine presentation"@en ;
    rdfs:comment """Description of how this individual appears to machine
    perception (UV Map 1 / RGBA channel encoding)."""@en .
```

---

## 13. Ternary Assertion Encoding

### 13.1 The Problem: OWL 2 is Binary

OWL 2 follows classical logic: a statement is either entailed or not. The Open-World Assumption means absence of knowledge is not negation — but there is no formal third state. K3D requires three distinct, ontologically equivalent states: affirmed (+1), unknown (0), negated (−1).

### 13.2 Encoding Strategy: Reified Assertions with Ternary Annotation

Every K3D assertion that carries ternary polarity is encoded as a reified statement with a `k3d:ternaryState` annotation:

```turtle
# Standard OWL: "cat is-a mammal" — binary, no polarity
k3d:concept_cat k3d:isA k3d:concept_mammal .

# K3D Ternary: "cat is-a mammal" with polarity +1 (affirmed)
_:assertion1 a k3d:TernaryAssertion ;
    rdf:subject k3d:concept_cat ;
    rdf:predicate k3d:isA ;
    rdf:object k3d:concept_mammal ;
    k3d:ternaryState "+1"^^k3d:TritValue .

# K3D Ternary: "cat is-a mineral" with polarity −1 (negated)
_:assertion2 a k3d:TernaryAssertion ;
    rdf:subject k3d:concept_cat ;
    rdf:predicate k3d:isA ;
    rdf:object k3d:concept_mineral ;
    k3d:ternaryState "-1"^^k3d:TritValue .

# K3D Ternary: "cat is-a sentient" with polarity 0 (genuinely unknown)
_:assertion3 a k3d:TernaryAssertion ;
    rdf:subject k3d:concept_cat ;
    rdf:predicate k3d:isA ;
    rdf:object k3d:concept_sentient ;
    k3d:ternaryState "0"^^k3d:TritValue .
```

### 13.3 Compact Encoding for Affirmed Assertions

When an assertion is affirmed (+1), the reification MAY be omitted. The simple triple form is assumed to carry +1 polarity:

```turtle
# This simple triple implies k3d:ternaryState "+1"
k3d:concept_cat k3d:isA k3d:concept_mammal .
```

Negated (−1) and unknown (0) assertions MUST always use the reified form.

### 13.4 Ternary-Native Substrate Alignment

On current binary hardware, the reification encoding works but carries overhead (4 triples per assertion instead of 1). When ternary-native substrates become available:

- Each balanced trit natively represents the polarity without reification.
- The RDF serialization can be optimized to a single triple with an embedded trit literal.
- The ontological semantics remain unchanged — only the serialization efficiency improves.

The PM-KR ontology is designed for the hardware that SHOULD exist, deployed on the hardware that currently exists.

---

## 14. Class Restrictions and Axioms

### 14.1 Invariant 1: Procedural Identity

Every `MeaningCentricStar` MUST have exactly one `meaningProgram`:

```turtle
k3d:MeaningCentricStar rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty k3d:meaningProgram ;
    owl:qualifiedCardinality "1"^^xsd:nonNegativeInteger ;
    owl:onDataRange k3d:RPNProgram
] .
```

### 14.2 Invariant 2: Content-Addressed Identity

```turtle
# Functional property — each star has exactly one starId
k3d:starId a owl:FunctionalProperty .

# Inverse functional — each starId identifies exactly one star
k3d:starId a owl:InverseFunctionalProperty .
```

This axiom pair ensures that `star_id` is a bijection: no two stars share an ID, and no star has multiple IDs.

### 14.3 Invariant 3: Spatial Grounding

Every active star MUST have at least one spatial position:

```turtle
# Active stars must have Galaxy position
k3d:MeaningCentricStar rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty k3d:temporalPhase ;
    owl:hasValue k3d:Active
] ;
rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty k3d:galaxyPosition ;
    owl:minCardinality "1"^^xsd:nonNegativeInteger
] .
```

### 14.4 Invariant 5: Reference Preservation (Stratification Axiom)

Rules (Layer 3) MUST reference Forms (Layer 1) or Meanings (Layer 2):

```turtle
k3d:RuleProcedure rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty k3d:references ;
    owl:someValuesFrom [
        a owl:Class ;
        owl:unionOf ( k3d:FormProcedure k3d:MeaningProcedure )
    ]
] .

k3d:MetaRuleProcedure rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty k3d:references ;
    owl:someValuesFrom k3d:RuleProcedure
] .
```

### 14.5 Invariant 6: Dual-Client Equivalence

Every star MUST have both a visual program AND a meaning program:

```turtle
k3d:MeaningCentricStar rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty k3d:visualProgram ;
    owl:minCardinality "1"^^xsd:nonNegativeInteger
] .

k3d:MeaningCentricStar rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty k3d:meaningProgram ;
    owl:minCardinality "1"^^xsd:nonNegativeInteger
] .
```

### 14.6 Invariant 7: Sovereign Hot Path

Active procedures MUST execute on sovereign substrates:

```turtle
# Sovereign substrates
k3dsub:SovereignSubstrate a owl:Class ;
    owl:equivalentClass [
        a owl:Class ;
        owl:unionOf ( k3dsub:PTXKernel k3dsub:RPNStack
                      k3dsub:VRAMSubstrate k3dsub:Register
                      k3dsub:TernarySubstrate )
    ] .

# Active procedures execute on sovereign substrates only
# (SHACL constraint — not expressible in pure OWL 2 DL)
# See §14.7 for SHACL encoding
```

### 14.7 SHACL Constraints for Non-DL Invariants

Some K3D invariants cannot be expressed in OWL 2 DL. These are encoded as SHACL shapes for validation:

```turtle
@prefix sh: <http://www.w3.org/ns/shacl#> .

# Sovereignty constraint: active procedures must be on sovereign substrates
k3d:SovereignHotPathShape a sh:NodeShape ;
    sh:targetClass k3d:MeaningCentricStar ;
    sh:property [
        sh:path k3d:temporalPhase ;
        sh:hasValue k3d:Active ;
        sh:message "Active star detected" ;
    ] ;
    sh:property [
        sh:path k3dsub:executesOn ;
        sh:class k3dsub:SovereignSubstrate ;
        sh:message "Active star MUST execute on sovereign substrate." ;
        sh:severity sh:Violation ;
    ] .

# Reference preservation: no inlining across layers
k3d:ReferencePreservationShape a sh:NodeShape ;
    sh:targetClass k3d:RuleProcedure ;
    sh:sparql [
        sh:message "Rule procedure MUST NOT inline lower-layer content." ;
        sh:select """
            SELECT $this WHERE {
                $this k3d:layer ?layer .
                $this k3d:references ?ref .
                ?ref k3d:layer ?refLayer .
                FILTER (?refLayer >= ?layer)
            }
        """ ;
    ] .

# Ternary completeness: no bare true/false
k3d:TernaryCompletenessShape a sh:NodeShape ;
    sh:targetClass k3d:TernaryAssertion ;
    sh:property [
        sh:path k3d:ternaryState ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:datatype k3d:TritValue ;
        sh:message "Every ternary assertion MUST carry exactly one trit value." ;
    ] .
```

---

## 15. SPARQL Query Patterns

### 15.1 Galaxy Navigation: Find Stars Near a Concept

```sparql
PREFIX k3d: <https://knowledge3d.org/ontology/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?star ?label ?distance WHERE {
    k3d:concept_derivative k3d:galaxyPosition ?seedPos .
    ?star a k3d:MeaningCentricStar ;
          k3d:galaxyPosition ?starPos ;
          rdfs:label ?label .
    BIND(k3d:spatialDistance(?seedPos, ?starPos) AS ?distance)
    FILTER(?distance < 5.0)
    FILTER(?star != k3d:concept_derivative)
}
ORDER BY ?distance
LIMIT 24
```

### 15.2 Cross-Domain Bridge Discovery

Find symbols referenced by procedures in 3+ different domains — the ontological bridges that connect otherwise unrelated fields:

```sparql
PREFIX k3d: <https://knowledge3d.org/ontology/>

SELECT ?symbol ?label (COUNT(DISTINCT ?domain) AS ?domainCount)
       (GROUP_CONCAT(DISTINCT ?domain; separator=", ") AS ?domains)
WHERE {
    ?rule a k3d:RuleProcedure ;
          k3d:references ?symbol ;
          k3d:domain ?domain .
    ?symbol rdfs:label ?label .
}
GROUP BY ?symbol ?label
HAVING (COUNT(DISTINCT ?domain) > 2)
ORDER BY DESC(?domainCount)
```

### 15.3 Sleep-Time Provenance: Trace Knowledge Evolution

```sparql
PREFIX k3d: <https://knowledge3d.org/ontology/>
PREFIX prov: <http://www.w3.org/ns/prov#>

SELECT ?star ?label ?cycle ?version ?createdAt WHERE {
    ?star a k3d:MeaningCentricStar ;
          rdfs:label ?label ;
          k3d:sleepCycle ?cycle ;
          k3d:version ?version ;
          k3d:createdAt ?createdAt .
}
ORDER BY ?star ?cycle
```

### 15.4 Agent Perception Query: What Does the TRM See?

```sparql
PREFIX k3d: <https://knowledge3d.org/ontology/>
PREFIX k3da: <https://knowledge3d.org/ontology/agent/>

SELECT ?star ?label ?lodLevel WHERE {
    ?agent a k3da:TRMAgent ;
           k3d:perceives ?star .
    ?star rdfs:label ?label ;
          k3d:lodLevel ?lodLevel .
}
ORDER BY DESC(?lodLevel)
```

### 15.5 Defeasible Reasoning: Find Override Chains

```sparql
PREFIX k3d: <https://knowledge3d.org/ontology/>

SELECT ?specific ?general ?strength WHERE {
    ?specific k3d:defeasiblyOverrides ?general ;
              k3d:ruleStrength ?strength .
    ?general rdfs:label ?generalLabel .
    ?specific rdfs:label ?specificLabel .
}
ORDER BY ?general
```

### 15.6 Taxonomic Closure: All Ancestors of a Concept

```sparql
PREFIX k3d: <https://knowledge3d.org/ontology/>

SELECT ?ancestor ?label WHERE {
    k3d:concept_cat k3d:isA+ ?ancestor .
    ?ancestor rdfs:label ?label .
}
```

### 15.7 Federated Query: Search Across Multiple Houses

```sparql
PREFIX k3d: <https://knowledge3d.org/ontology/>

SELECT ?house ?star ?label WHERE {
    SERVICE <k3d://lab.example.com/sparql> {
        ?star a k3d:ConceptStar ;
              rdfs:label ?label ;
              k3d:domain ?domain .
        FILTER(STRSTARTS(?domain, "Library/Physics"))
        BIND("lab" AS ?house)
    }
    UNION
    SERVICE <k3d://classroom.example.com/sparql> {
        ?star a k3d:ConceptStar ;
              rdfs:label ?label ;
              k3d:domain ?domain .
        FILTER(STRSTARTS(?domain, "Library/Physics"))
        BIND("classroom" AS ?house)
    }
}
```

---

## 16. RDF/Turtle Serialization Examples

### 16.1 Complete Star Example: Concept "Cat"

```turtle
@prefix k3d: <https://knowledge3d.org/ontology/> .
@prefix k3ds: <https://knowledge3d.org/ontology/spatial/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix dcterms: <http://purl.org/dc/terms/> .

k3d:concept_cat a k3d:ConceptStar ;
    rdfs:label "cat"@en , "gato"@pt , "Katze"@de , "猫"@ja ;
    k3d:starId "a7f3b2c1d4e5f6a7b8c9d0e1f2a3b4c5"^^xsd:hexBinary ;
    k3d:meaningProgram """
        4.0 STORE_mass_kg
        4 STORE_leg_count
        1 STORE_domesticated
        RECALL_mammal_program EXEC
        RECALL_predator_instinct EXEC
        RECALL_feline_locomotion EXEC
    """^^k3d:RPNProgram ;
    k3d:visualProgram """
        MOVE 0 0
        RECALL_feline_body_outline EXEC
        RECALL_whisker_pattern EXEC
        RECALL_ear_triangles EXEC
    """^^k3d:RPNProgram ;
    k3d:behaviorProgram """
        SENSE_PREY ? IF
            RECALL_stalk_pattern EXEC
        ELSE
            RECALL_rest_pattern EXEC
        THEN
    """^^k3d:RPNProgram ;
    k3d:layer "2"^^xsd:nonNegativeInteger ;
    k3d:confidence "0.95"^^xsd:float ;
    k3d:meaningMass "47.3"^^xsd:float ;
    k3d:embedding "[0.23, -0.41, 0.87, ...]"^^k3d:EmbeddingVector ;
    k3d:housePosition "12.5 8.3 -2.1"^^k3d:Vec3 ;
    k3d:galaxyPosition "3.2 -1.7 0.8"^^k3d:Vec3 ;
    k3d:domain "Library/Biology/Mammalia" ;
    k3d:isA k3d:concept_mammal ;
    k3d:isA k3d:concept_pet ;
    k3d:memberOf k3d:RealityGalaxy ;
    k3d:temporalPhase k3d:Persistent ;
    k3d:hasSurfaceForm [
        k3d:language "en"^^xsd:language ;
        k3d:wordRef k3d:word_cat
    ] ;
    k3d:hasSurfaceForm [
        k3d:language "pt"^^xsd:language ;
        k3d:wordRef k3d:word_gato
    ] ;
    k3d:createdBy k3d:agent_trm_primary ;
    k3d:createdAt "2026-03-15T14:30:00Z"^^xsd:dateTime ;
    k3d:version "1.0.2" ;
    k3d:sleepCycle "47"^^xsd:nonNegativeInteger ;
    k3d:ternaryState "+1"^^k3d:TritValue ;
    k3ds:containedIn k3d:shelf_biology_mammalia .
```

### 16.2 Grammar Rule Example: Defeasible Reasoning

```turtle
k3d:rule_birds_fly a k3d:RuleProcedure ;
    rdfs:label "birds fly"@en ;
    k3d:layer "3"^^xsd:nonNegativeInteger ;
    k3d:references k3d:concept_bird ;
    k3d:references k3d:action_fly ;
    k3d:transforms k3d:concept_bird ;
    k3d:meaningProgram """
        RECALL_bird_test EXEC
        ? IF
            RECALL_flight_capability STORE_result
        THEN
    """^^k3d:RPNProgram ;
    k3d:ruleStrength "0"^^k3d:TritValue ;
    k3d:ternaryState "+1"^^k3d:TritValue .

k3d:rule_penguins_no_fly a k3d:RuleProcedure ;
    rdfs:label "penguins do not fly"@en ;
    k3d:layer "3"^^xsd:nonNegativeInteger ;
    k3d:references k3d:concept_penguin ;
    k3d:references k3d:action_fly ;
    k3d:transforms k3d:concept_penguin ;
    k3d:defeasiblyOverrides k3d:rule_birds_fly ;
    k3d:ruleStrength "+1"^^k3d:TritValue ;
    k3d:ternaryState "+1"^^k3d:TritValue .
```

### 16.3 TRM Agent Example

```turtle
@prefix k3da: <https://knowledge3d.org/ontology/agent/> .
@prefix k3dsub: <https://knowledge3d.org/ontology/substrate/> .

k3d:agent_trm_primary a k3da:TRMAgent ;
    rdfs:label "Primary TRM"@en ;
    k3d:inhabits k3d:house_primary ;
    k3da:hasCranialGalaxy k3d:cranial_galaxy_primary ;
    k3da:hasHaltingGate k3d:halting_gate_primary ;
    k3da:hasSwarmWorker k3d:worker_chat ,
                        k3d:worker_math ,
                        k3d:worker_grammar ,
                        k3d:worker_visual ,
                        k3d:worker_reality ,
                        k3d:worker_spatial ,
                        k3d:worker_logic ,
                        k3d:worker_language ,
                        k3d:worker_meta ;
    k3dsub:executesOn k3dsub:rtx3070_vram ;
    k3d:version "1.0.47" ;
    k3d:temporalPhase k3d:Active .

k3d:worker_math a k3da:SwarmWorker ;
    rdfs:label "Math Specialist Worker"@en ;
    k3da:specialistDomain k3d:MathGalaxy ;
    k3d:confidence "0.87"^^xsd:float .

k3d:cranial_galaxy_primary a k3da:CranialGalaxy ;
    rdfs:label "Primary Cranial Galaxy"@en ;
    k3d:starCount "247889"^^xsd:nonNegativeInteger ;
    k3d:galaxyCount "19"^^xsd:nonNegativeInteger .
```

---

## 17. glTF-to-RDF Extraction

### 17.1 Extraction Principle

K3D knowledge persists in glTF files with `extras.k3d` metadata blocks. The PM-KR ontology defines extraction rules that transform these blocks into RDF triples, enabling SPARQL querying of House content stored on disk.

### 17.2 Mapping Rules

| glTF `extras.k3d` Field | RDF Property | Type |
|--------------------------|-------------|------|
| `star_id` | `k3d:starId` | `xsd:hexBinary` |
| `meaning_class` | `rdf:type` (subclass of `k3d:MeaningCentricStar`) | OWL Class |
| `meaning_rpn` | `k3d:meaningProgram` | `k3d:RPNProgram` |
| `visual_rpn` | `k3d:visualProgram` | `k3d:RPNProgram` |
| `behavior_rpn` | `k3d:behaviorProgram` | `k3d:RPNProgram` |
| `taxonomy_refs[]` | `k3d:isA` | Object Property |
| `component_refs[]` | `k3d:partOf` | Object Property |
| `symlink_refs[]` | `k3d:references` | Object Property |
| `embedding` | `k3d:embedding` | `k3d:EmbeddingVector` |
| `position` | `k3d:housePosition` | `k3d:Vec3` |
| `layer` | `k3d:layer` | `xsd:nonNegativeInteger` |
| `confidence` | `k3d:confidence` | `xsd:float` |
| `trit` | `k3d:ternaryState` | `k3d:TritValue` |
| `domain` | `k3d:domain` | `xsd:string` |
| `surface_forms[].lang` | `k3d:language` (via `k3d:hasSurfaceForm`) | `xsd:language` |
| `surface_forms[].word_ref` | `k3d:wordRef` (via `k3d:hasSurfaceForm`) | Object Property |
| `provenance` | `k3d:provenance` | `xsd:string` |
| `version` | `k3d:version` | `xsd:string` |

### 17.3 Extraction Algorithm (Pseudocode)

```
FOR each node N in glTF scene:
    IF N has extras.k3d:
        star_uri = k3d: + extras.k3d.star_id
        EMIT (star_uri, rdf:type, meaning_class_to_owl(extras.k3d.meaning_class))
        EMIT (star_uri, k3d:starId, extras.k3d.star_id)
        EMIT (star_uri, k3d:meaningProgram, extras.k3d.meaning_rpn)
        FOR each taxonomy_ref in extras.k3d.taxonomy_refs:
            EMIT (star_uri, k3d:isA, k3d: + taxonomy_ref)
        FOR each symlink_ref in extras.k3d.symlink_refs:
            EMIT (star_uri, k3d:references, k3d: + symlink_ref)
        # ... (remaining fields per mapping table)
```

A conforming implementation SHOULD provide a command-line tool or API endpoint that performs this extraction, producing a Turtle, N-Triples, or JSON-LD file from a glTF House.

---

## 18. Interoperability Bridges

### 18.1 X3D Ontology Alignment

| X3D Ontology (`x3do:`) | PM-KR Ontology (`k3d:`) | Alignment |
|------------------------|------------------------|-----------|
| `x3do:X3DNode` | `k3d:Thing` | `k3d:Thing rdfs:subClassOf x3do:X3DNode` (PM-KR individuals are X3D nodes) |
| `x3do:X3DChildNode` | `k3d:MeaningCentricStar` | Direct subclass — stars are scene graph children |
| `x3do:X3DGroupingNode` | `k3d:Galaxy` | Direct subclass — galaxies are grouping containers |
| `x3do:Shape` | `k3d:DualClientEntity` | PM-KR extends Shape with dual UV-map contract |
| `x3do:hasChild` | `k3d:memberOf` (inverse direction) | Galaxy "has child" stars ↔ star "member of" Galaxy |
| `x3do:X3DMetadataNode` | `k3d:ternaryState` | K3D uses annotation properties instead of metadata nodes |
| `hanim:HAnimHumanoid` | `k3da:AvatarBody` | PM-KR Avatar extends HAnim Humanoid |

### 18.2 BFO Alignment (Informative)

For systems that use the Basic Formal Ontology, K3D provides an informative (non-normative) alignment:

| BFO Category | K3D Counterpart | Notes |
|-------------|----------------|-------|
| `bfo:Continuant` | `k3d:MeaningCentricStar` | Stars persist through time (version history) |
| `bfo:Occurrent` | `k3d:Process` | K3D processes have temporal extent |
| `bfo:SpatialRegion` | `k3ds:Space` | K3D spaces have stronger semantics (position = meaning) |
| `bfo:Quality` | `k3d:PropertyStar` | K3D qualities are RPN programs, not qualia |
| `bfo:Role` | — | K3D does not separate role from bearer; the program IS both |
| `bfo:Function` | `k3d:Procedure` | K3D unifies function with entity (Procedural Inversion) |

### 18.3 DOLCE Alignment (Informative)

| DOLCE Category | K3D Counterpart | Notes |
|---------------|----------------|-------|
| `dolce:Endurant` | `k3d:MeaningCentricStar` | Stars are "wholly present at each time" |
| `dolce:Perdurant` | `k3d:Process` | Reasoning, consolidation, navigation |
| `dolce:Quality` | `k3d:PropertyStar` | RPN-based qualities |
| `dolce:Region` | `k3ds:Space` | House/Galaxy/World spaces |
| `dolce:Quale` | — | K3D replaces qualia with RPN execution semantics |
| `dolce:AgentivePhysicalObject` | `k3da:Agent` | K3D agents are spatial + cognitive |

### 18.4 Dublin Core Mapping

```turtle
# Star provenance maps to Dublin Core
k3d:domain rdfs:subPropertyOf dcterms:subject .
k3d:provenance rdfs:subPropertyOf dcterms:source .
k3d:createdAt rdfs:subPropertyOf dcterms:created .
k3d:version rdfs:subPropertyOf dcterms:hasVersion .
```

### 18.5 PROV-O Integration

```turtle
# Stars are PROV Entities
k3d:MeaningCentricStar rdfs:subClassOf prov:Entity .

# Agents are PROV Agents
k3da:Agent rdfs:subClassOf prov:Agent .

# Processes are PROV Activities
k3d:Process rdfs:subClassOf prov:Activity .

# Consolidation is derivation
k3d:consolidatesTo rdfs:subPropertyOf prov:wasDerivedFrom .

# Creation is generation
k3d:createdBy rdfs:subPropertyOf prov:wasGeneratedBy .
```

### 18.6 SKOS Alignment for Vocabularies

```turtle
# Stars with surface forms map to SKOS Concepts
k3d:ConceptStar rdfs:subClassOf skos:Concept .

# Surface forms map to SKOS labels
k3d:hasSurfaceForm rdfs:subPropertyOf skos:altLabel .

# Galaxy membership maps to SKOS Collection
k3d:Galaxy rdfs:subClassOf skos:Collection .

# isA maps to SKOS broader/narrower
k3d:isA rdfs:subPropertyOf skos:broader .
```

### 18.7 GeoSPARQL Alignment

K3D spatial queries can interoperate with GeoSPARQL endpoints:

```turtle
# K3D spaces have GeoSPARQL geometry
k3ds:HouseSpace rdfs:subClassOf geo:Feature .

k3d:housePosition rdfs:subPropertyOf geo:hasGeometry .

# K3D spatial relations extend GeoSPARQL
k3ds:containedIn rdfs:subPropertyOf geo:sfWithin .
```

---

## 19. Procedural Protocol and Display Readiness

### 19.1 Ontological Status of Procedural Outputs

A core PM-KR principle is that knowledge emitted to an output channel retains its ontological identity. When a star's visual program is sent to a display, it is not "rendered into pixels and gone" — the program remains a procedure, and the display is a substrate.

This section formalizes the ontological implications for future substrates where outputs are procedural rather than rasterized.

### 19.2 Procedural Display Substrate

```turtle
k3dsub:ProceduralDisplaySubstrate rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty k3dsub:isSovereign ;
    owl:hasValue true
] .

k3dsub:ProceduralDisplaySubstrate rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty k3dsub:encoding ;
    owl:hasValue "procedural"
] .
```

A procedural display:
- Receives RPN programs, not rasterized frames.
- Executes programs locally, producing visual output from procedural instructions.
- Preserves the ontological identity of emitted knowledge — a displayed star is still a `k3d:MeaningCentricStar`, queryable via SPARQL, navigable by agents.
- Enables dual-client rendering at the hardware level: the same procedural program drives both human-visible output AND machine-executable semantics.

### 19.3 Procedural Protocol Substrate

```turtle
k3dsub:ProceduralProtocolSubstrate a owl:Class ;
    rdfs:subClassOf k3dsub:NetworkSubstrate ;
    rdfs:label "Procedural Protocol Substrate"@en ;
    rdfs:comment """Communication channel where transmitted knowledge retains
    procedural structure. RPN programs sent between Houses via Doors arrive
    as executable procedures, not serialized data."""@en .
```

Knowledge transmitted via a procedural protocol is ontologically indistinguishable from knowledge processed locally. The ontology makes no formal distinction between "this star is in my Galaxy" and "this star arrived via Door" — both are Procedures on Substrates.

### 19.4 Ternary-Native Display and Protocol

When ternary substrates mature, procedural displays and protocols gain native ternary expression:

- **Ternary procedural display**: Each pixel/voxel carries a trit-vector of (value, certainty, polarity) rather than binary RGB. A rendered concept visually communicates its ternary state.
- **Ternary procedural protocol**: Each transmitted RPN instruction carries native trit encoding. Inter-House communication preserves ternary semantics without binary encoding overhead.

The PM-KR ontology already models these substrates (`k3dsub:TernarySubstrate`). When hardware arrives, the ontology extends with display-specific and protocol-specific subclasses — no structural changes to the upper ontology.

---

## 20. Conformance

### 20.1 Ontology Producer Conformance

An **ontology producer** is a system that generates RDF triples conforming to this specification.

**Level 1: Core Producer**
A conforming Level 1 producer MUST:
- Use the `k3d:` namespace for all PM-KR classes and properties.
- Assign every star a `k3d:starId` computed as content hash of `k3d:meaningProgram`.
- Classify every star into exactly one of the five meaning classes.
- Assign `k3d:layer` values consistent with the four-layer hierarchy.
- Encode `k3d:meaningProgram` and `k3d:visualProgram` as `k3d:RPNProgram` literals.
- Produce valid OWL 2 DL — no metaclass patterns, all properties with declared domains/ranges.

**Level 2: Spatial Producer**
Level 1 plus:
- Assign `k3d:housePosition` and/or `k3d:galaxyPosition` to all active stars.
- Emit `k3ds:containedIn` triples for House-organized knowledge.
- Emit `k3d:memberOf` triples for Galaxy membership.
- Encode semantic force relations using the `k3d:SemanticForce` reification.

**Level 3: Agent Producer**
Level 2 plus:
- Emit `k3da:TRMAgent` individuals with required cardinality restrictions (cranial galaxy, halting gate, swarm workers).
- Emit `k3da:Interaction` individuals for agent actions.
- Emit PROV-O provenance triples (`prov:wasGeneratedBy`, `prov:generatedAtTime`).
- Encode ternary assertions using the reification pattern (§13).

### 20.2 Ontology Consumer Conformance

An **ontology consumer** is a system that reads and processes RDF triples conforming to this specification.

**Level 1: Core Consumer**
A conforming Level 1 consumer MUST:
- Recognize all `k3d:` classes and properties defined in §§5–12.
- Resolve `k3d:starId` as content-addressed identity.
- Execute SPARQL queries using the patterns defined in §15.

**Level 2: Spatial Consumer**
Level 1 plus:
- Interpret spatial positions for 3D visualization or spatial querying.
- Support the `k3ds:containedIn`, `k3ds:nearTo`, and `k3ds:gravitatesTo` properties.

**Level 3: Reasoning Consumer**
Level 2 plus:
- Process `k3d:TernaryAssertion` reifications.
- Apply `k3d:defeasiblyOverrides` chains in query answering.
- Validate SHACL constraints (§14.7) on loaded knowledge graphs.

### 20.3 Validation Test Suite

A conforming implementation SHOULD pass the following validation tests:

| Test | Description | Validates |
|------|-------------|-----------|
| T1 | Round-trip: glTF → RDF → SPARQL query → expected results | Extraction (§17) |
| T2 | OWL consistency check: no logical contradictions | Class hierarchy (§§5–9) |
| T3 | Cardinality: every star has exactly one starId, meaningProgram | Invariants 1–2 (§14) |
| T4 | Reference preservation: no rule inlines form content | Invariant 5 (§14.4) |
| T5 | Ternary completeness: no assertion without trit value | Invariant 4 (§14.7) |
| T6 | SPARQL patterns: all §15 queries return expected results on test dataset | Query patterns |
| T7 | Namespace: all triples use declared prefixes, no undefined URIs | Namespace (§3) |
| T8 | SHACL: sovereignty constraint passes for active stars | Invariant 7 (§14.7) |

---

## 21. Examples

### 21.1 Complete X3D Scene with RDF Annotation

This example shows an X3D scene with PM-KR knowledge nodes, where each node's `extras.k3d` metadata maps to the RDF representation defined in this ontology:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<X3D profile="ProceduralMemoryInterchange" version="4.0"
     xmlns:k3d="https://knowledge3d.org/ontology/">

  <head>
    <meta name="ontology" content="https://knowledge3d.org/ontology/0.1/"/>
    <meta name="description" content="Mathematics Galaxy fragment with defeasible rules"/>
  </head>

  <Scene>
    <!-- Galaxy Group: Math Galaxy -->
    <GalaxyGroup DEF="MathGalaxy"
                 galaxyName="Math"
                 entryCount="4"
                 starDomain="Math">

      <!-- Layer 1: Form — the summation symbol ∑ -->
      <ProceduralMemoryNode DEF="symbol_summation"
          canonicalId="b3f2a1c4d5e6f7a8b9c0d1e2f3a4b5c6"
          pmkrLayer="1"
          visualProgram="MOVE 0 12 LINE 8 12 LINE 4 0 LINE 0 12
                         MOVE 0 12 LINE 4 24 LINE 8 24"
          meaningProgram="RECALL_range EXEC RECALL_body EXEC
                          STORE_accumulator LOOP_SUM">
        <Shape>
          <Appearance>
            <Material diffuseColor="0.1 0.1 0.8"/>
          </Appearance>
          <Text string='"∑"'>
            <FontStyle size="2.0"/>
          </Text>
        </Shape>
      </ProceduralMemoryNode>

      <!-- Layer 2: Meaning — the concept of summation -->
      <ProceduralMemoryNode DEF="concept_summation"
          canonicalId="c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9"
          pmkrLayer="2"
          meaningProgram="RECALL_range_iterator EXEC
                          RECALL_accumulation_op EXEC
                          RECALL_convergence_check EXEC"
          symlinkRefs="symbol_summation">
        <!-- References Layer 1 symbol via symlink, not duplication -->
      </ProceduralMemoryNode>

      <!-- Layer 3: Rule — finite series summation formula -->
      <ProceduralMemoryNode DEF="rule_finite_series"
          canonicalId="d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0"
          pmkrLayer="3"
          meaningProgram="RECALL_n RECALL_first RECALL_last
                          ADD MUL 2 DIV"
          symlinkRefs="concept_summation"
          ruleStrength="1">
        <!-- Strict rule: Gauss's formula n(a+l)/2 -->
      </ProceduralMemoryNode>

      <!-- Layer 4: Meta-Rule — when to apply series vs iteration -->
      <ProceduralMemoryNode DEF="meta_series_strategy"
          canonicalId="e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1"
          pmkrLayer="4"
          meaningProgram="RECALL_is_arithmetic ? IF
                            RECALL_rule_finite_series EXEC
                          ELSE
                            RECALL_iterative_sum EXEC
                          THEN"
          symlinkRefs="rule_finite_series">
        <!-- Strategy: prefer closed-form when sequence is arithmetic -->
      </ProceduralMemoryNode>

    </GalaxyGroup>
  </Scene>
</X3D>
```

### 21.2 Corresponding RDF/Turtle

The same knowledge as RDF triples, following the extraction rules of §17:

```turtle
@prefix k3d: <https://knowledge3d.org/ontology/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# ── Layer 1: Form ──
k3d:symbol_summation a k3d:FormProcedure, k3d:ConceptStar ;
    rdfs:label "∑"@en ;
    k3d:starId "b3f2a1c4d5e6f7a8b9c0d1e2f3a4b5c6"^^xsd:hexBinary ;
    k3d:layer "1"^^xsd:nonNegativeInteger ;
    k3d:visualProgram """MOVE 0 12 LINE 8 12 LINE 4 0 LINE 0 12
                         MOVE 0 12 LINE 4 24 LINE 8 24"""^^k3d:RPNProgram ;
    k3d:meaningProgram """RECALL_range EXEC RECALL_body EXEC
                          STORE_accumulator LOOP_SUM"""^^k3d:RPNProgram ;
    k3d:memberOf k3d:MathGalaxy ;
    k3d:ternaryState "+1"^^k3d:TritValue .

# ── Layer 2: Meaning ──
k3d:concept_summation a k3d:MeaningProcedure, k3d:ConceptStar ;
    rdfs:label "summation"@en ;
    k3d:starId "c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9"^^xsd:hexBinary ;
    k3d:layer "2"^^xsd:nonNegativeInteger ;
    k3d:meaningProgram """RECALL_range_iterator EXEC
                          RECALL_accumulation_op EXEC
                          RECALL_convergence_check EXEC"""^^k3d:RPNProgram ;
    k3d:references k3d:symbol_summation ;
    k3d:memberOf k3d:MathGalaxy ;
    k3d:ternaryState "+1"^^k3d:TritValue .

# ── Layer 3: Rule ──
k3d:rule_finite_series a k3d:RuleProcedure ;
    rdfs:label "finite series summation (Gauss)"@en ;
    k3d:starId "d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0"^^xsd:hexBinary ;
    k3d:layer "3"^^xsd:nonNegativeInteger ;
    k3d:meaningProgram "RECALL_n RECALL_first RECALL_last ADD MUL 2 DIV"^^k3d:RPNProgram ;
    k3d:references k3d:concept_summation ;
    k3d:transforms k3d:concept_summation ;
    k3d:ruleStrength "+1"^^k3d:TritValue ;
    k3d:memberOf k3d:MathGalaxy ;
    k3d:ternaryState "+1"^^k3d:TritValue .

# ── Layer 4: Meta-Rule ──
k3d:meta_series_strategy a k3d:MetaRuleProcedure ;
    rdfs:label "series vs iteration strategy"@en ;
    k3d:starId "e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1"^^xsd:hexBinary ;
    k3d:layer "4"^^xsd:nonNegativeInteger ;
    k3d:meaningProgram """RECALL_is_arithmetic ? IF
                            RECALL_rule_finite_series EXEC
                          ELSE
                            RECALL_iterative_sum EXEC
                          THEN"""^^k3d:RPNProgram ;
    k3d:references k3d:rule_finite_series ;
    k3d:memberOf k3d:MathGalaxy ;
    k3d:ternaryState "+1"^^k3d:TritValue .
```

### 21.3 SPARQL Query Over the Example

```sparql
# Find all cross-layer references in the Math Galaxy
PREFIX k3d: <https://knowledge3d.org/ontology/>

SELECT ?source ?sourceLayer ?target ?targetLayer WHERE {
    ?source k3d:references ?target ;
            k3d:layer ?sourceLayer ;
            k3d:memberOf k3d:MathGalaxy .
    ?target k3d:layer ?targetLayer .
    FILTER(?sourceLayer > ?targetLayer)
}
ORDER BY ?sourceLayer

# Expected results:
# concept_summation (L2) → symbol_summation (L1)
# rule_finite_series (L3) → concept_summation (L2)
# meta_series_strategy (L4) → rule_finite_series (L3)
```

---

## Appendix A: Complete OWL 2 Class Hierarchy

```
k3d:Thing
├── k3d:Procedure
│   ├── k3d:FormProcedure                    (Layer 1)
│   ├── k3d:MeaningProcedure                 (Layer 2)
│   ├── k3d:RuleProcedure                    (Layer 3)
│   ├── k3d:MetaRuleProcedure                (Layer 4)
│   └── k3d:CompositeProcedure               (cross-layer)
│
├── k3d:MeaningCentricStar                   ⊆ k3d:Procedure ∩ k3d:DualClientEntity
│   ├── k3d:ConceptStar
│   ├── k3d:RelationStar
│   ├── k3d:ActionStar
│   ├── k3d:PropertyStar
│   └── k3d:MetaStar
│
├── k3ds:Space
│   ├── k3ds:HouseSpace                      (intentional, persistent)
│   │   ├── k3ds:Room
│   │   │   ├── k3ds:Library
│   │   │   ├── k3ds:Workshop
│   │   │   ├── k3ds:Garden
│   │   │   ├── k3ds:Bathtub
│   │   │   ├── k3ds:LivingRoom
│   │   │   └── k3ds:Museum
│   │   ├── k3ds:Furniture
│   │   └── k3ds:Artifact
│   ├── k3ds:GalaxySpace                     (gravitational, volatile)
│   │   ├── k3d:Galaxy
│   │   ├── k3ds:Neighborhood
│   │   └── k3ds:NavigationPath
│   └── k3ds:WorldSpace                      (networked, federated)
│       ├── k3ds:RemoteHouse
│       └── k3ds:SharedGalaxy
│
├── k3da:Agent
│   ├── k3da:HumanAgent
│   ├── k3da:TRMAgent
│   ├── k3da:AssistantAgent
│   ├── k3da:ServiceAgent
│   └── k3da:RoboticAgent
│
├── k3dsub:Substrate
│   ├── k3dsub:VRAMSubstrate
│   ├── k3dsub:SSDSubstrate
│   ├── k3dsub:PTXKernel
│   ├── k3dsub:RPNStack
│   ├── k3dsub:Register
│   ├── k3dsub:NetworkSubstrate
│   ├── k3dsub:ProceduralDisplaySubstrate    (future)
│   └── k3dsub:TernarySubstrate              (future)
│
├── k3d:Process
│   ├── k3d:ReasoningProcess
│   ├── k3d:ConsolidationProcess
│   ├── k3d:IngestionProcess
│   ├── k3d:NavigationProcess
│   ├── k3d:CompositionProcess
│   ├── k3d:CreationProcess
│   └── k3d:PruningProcess
│
├── k3d:SemanticForce                        (reified ternary force)
├── k3d:TernaryAssertion                     (reified ternary statement)
├── k3d:SurfaceForm                          (language-specific rendering)
├── k3d:DualClientEntity                     (mixin — two presentation faces)
├── k3d:TemporalPhase                        (enum: Nascent, Active, Persistent, Archived)
│
├── k3da:Specialist                          (LoRA adapter + navigation bias)
├── k3da:SwarmWorker                         (parallel execution core)
├── k3da:HaltingGate                         (ternary convergence)
├── k3da:ShadowCopy                          (learning buffer)
├── k3da:NavigationTrace                     (LED-A* path)
├── k3da:CranialGalaxy                       ⊆ k3ds:GalaxySpace
│
└── k3da:Interaction
    ├── k3da:Navigate
    ├── k3da:Perceive
    ├── k3da:Reach
    ├── k3da:Grasp
    ├── k3da:Use
    ├── k3da:Share
    ├── k3da:Speak
    └── k3da:Create
```

---

## Appendix B: Namespace URI Summary

| Prefix | Full URI | Content |
|--------|---------|---------|
| `k3d:` | `https://knowledge3d.org/ontology/` | Core classes, properties, datatypes |
| `k3ds:` | `https://knowledge3d.org/ontology/spatial/` | Spatial domain classes and properties |
| `k3da:` | `https://knowledge3d.org/ontology/agent/` | Agent classes and cognitive architecture |
| `k3dsub:` | `https://knowledge3d.org/ontology/substrate/` | Execution environment classes |
| `x3do:` | `https://www.web3d.org/x3d/ontology/` | X3D Ontology (imported) |
| `hanim:` | `https://www.web3d.org/x3d/ontology/hanim/` | HAnim Ontology (imported) |

---

## Appendix C: Property Summary

### Object Properties

| Property | Domain | Range | Characteristics |
|----------|--------|-------|----------------|
| `k3d:isA` | Star | Star | Transitive |
| `k3d:partOf` | Star | Star | Transitive |
| `k3d:references` | Procedure | Procedure | — |
| `k3d:transforms` | RuleProcedure | Star | — |
| `k3d:perceives` | Agent | Star | — |
| `k3d:inhabits` | Agent | Space | — |
| `k3d:holds` | Agent | Artifact | — |
| `k3d:navigatesTo` | Agent | Star | — |
| `k3d:convergesOn` | SwarmWorker | Star | — |
| `k3d:consolidatesTo` | Star | Star | — |
| `k3d:defeasiblyOverrides` | RuleProcedure | RuleProcedure | — |
| `k3d:memberOf` | Star | Galaxy | — |
| `k3ds:containedIn` | Thing | HouseSpace | Transitive |
| `k3ds:nearTo` | Star | Star | Symmetric |
| `k3ds:gravitatesTo` | Star | Star | — |
| `k3ds:repels` | Star | Star | Symmetric |
| `k3ds:doorTo` | HouseSpace | HouseSpace | Symmetric |
| `k3ds:frustumVisible` | Agent | Thing | — |
| `k3dsub:executesOn` | Procedure | Substrate | — |

### Datatype Properties

| Property | Domain | Range |
|----------|--------|-------|
| `k3d:starId` | Star | `xsd:hexBinary` |
| `k3d:meaningProgram` | Star | `k3d:RPNProgram` |
| `k3d:visualProgram` | Star | `k3d:RPNProgram` |
| `k3d:behaviorProgram` | Star | `k3d:RPNProgram` |
| `k3d:embedding` | Star | `k3d:EmbeddingVector` |
| `k3d:housePosition` | Star | `k3d:Vec3` |
| `k3d:galaxyPosition` | Star | `k3d:Vec3` |
| `k3d:meaningMass` | Star | `xsd:float` |
| `k3d:lodLevel` | Star | `xsd:nonNegativeInteger` |
| `k3d:layer` | Star | `xsd:nonNegativeInteger` |
| `k3d:confidence` | Star | `xsd:float` |
| `k3d:domain` | Star | `xsd:string` |

### Annotation Properties

| Property | Purpose |
|----------|---------|
| `k3d:ternaryState` | +1/0/−1 polarity on assertions |
| `k3d:ruleStrength` | +1/0/−1 defeasibility of rules |
| `k3d:createdBy` | Provenance: creating agent |
| `k3d:createdAt` | Provenance: creation timestamp |
| `k3d:version` | Brain model version |
| `k3d:sleepCycle` | Consolidation cycle number |
| `k3d:provenance` | External source reference |
| `k3d:humanPresentation` | UV Map 0 description |
| `k3d:machinePresentation` | UV Map 1 description |
