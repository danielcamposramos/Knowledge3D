# PM-KR Procedural Memory Component for X3D

**Version**: 0.1 (Initial Draft)
**Status**: W3C PM-KR Community Group Working Draft
**Date**: March 26, 2026
**Authors**: PM-KR Community Group (Daniel Campos Ramos, Chair; Milton Ponson, Co-Chair)
**Liaison**: Web3D Consortium (Don Brutzman, Advisory Committee Representative)
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Reference Implementation)

**Normative References**:
- ISO/IEC 19775-1:2023 (X3D Architecture and Base Components, Version 4.0)
- ISO/IEC 19776-1 (X3D XML Encoding)
- ISO/IEC 19774 (HAnim Humanoid Animation)
- PM-KR Technology Specification v1.0 (docs/vocabulary/PROCEDURAL_MEMORY_KR_STANDARD_SPECIFICATION.md)
- Foundational Knowledge Specification v1.0 (docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md)
- Dual-Client Contract Specification v1.1 (docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md)
- Knowledgeverse Specification v5.1 (docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md)
- Sovereign NSI Specification (docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md)
- RPN Domain Opcode Registry v0.1 (docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md)
- Spatial General Intelligence Specification (docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Scope and Design Rationale](#2-scope-and-design-rationale)
3. [Concepts](#3-concepts)
4. [Component Definition: ProceduralMemory](#4-component-definition-proceduralmemory)
5. [Abstract Node Types](#5-abstract-node-types)
6. [Concrete Node Reference](#6-concrete-node-reference)
7. [RPN Program Encoding](#7-rpn-program-encoding)
8. [Symlink Reference System](#8-symlink-reference-system)
9. [Dual-Client Rendering Contract](#9-dual-client-rendering-contract)
10. [Component Definition: KnowledgeNavigation](#10-component-definition-knowledgenavigation)
11. [ProceduralMemoryInterchange Profile](#11-proceduralmemoryinterchange-profile)
12. [glTF Interoperability](#12-gltf-interoperability)
13. [Conformance](#13-conformance)
14. [Relationship to Existing X3D Components](#14-relationship-to-existing-x3d-components)
15. [Examples](#15-examples)

---

## 1. Introduction

### 1.1 Purpose

This document defines two new X3D components and one new X3D profile that extend the X3D Architecture (ISO/IEC 19775-1:2023) with **procedural memory knowledge representation** capabilities:

- **ProceduralMemory component**: Introduces node types for representing knowledge as executable procedures organized in a four-layer compositional hierarchy (Form, Meaning, Rules, Meta-Rules), connected by canonical references instead of data duplication.

- **KnowledgeNavigation component**: Introduces node types for spatial knowledge indexing, graph-based pathfinding, and autonomous agent embodiment within the scene graph.

- **ProceduralMemoryInterchange profile**: Bundles these components with existing X3D components to define a conformance target for procedural knowledge interchange between systems.

### 1.2 Motivation

X3D provides a comprehensive architecture for declarative 3D scene representation with scripting capabilities (Script nodes, ROUTE event wiring, Scene Access Interface). The X3D Ontology (Brutzman & Flotynski, 2020) demonstrates that X3D scenes can be formalized for semantic web querying via OWL 2.

However, X3D's current architecture treats knowledge as **passive scene data with behavior attached via scripts**. Script nodes are escape hatches from the declarative model --- they add behavior to otherwise inert geometry.

PM-KR inverts this relationship: **knowledge IS executable programs**. A mathematical symbol is not geometry with metadata --- it is an RPN program that draws itself (form) AND carries its mathematical semantics (meaning). Rules are programs that operate on those programs. Strategy is programs that select among rules. The four-layer stack (Form, Meaning, Rules, Meta-Rules) composes upward through canonical references, achieving 69--666x compression while preserving full reconstructibility.

This extension brings PM-KR's procedural-first knowledge representation into the X3D ecosystem, enabling:

1. **Knowledge interchange**: X3D scenes where knowledge nodes carry executable procedures, not just geometry and metadata.
2. **Dual-client rendering**: The same node serves human viewers (visual rendering via UV Map 0) and machine consumers (procedural execution via UV Map 1) with guaranteed identity.
3. **Compositional compression**: Cross-node symlink references eliminate payload duplication across the four-layer hierarchy.
4. **Autonomous agent embodiment**: An X3D scene can contain a spatially-grounded agent entity that navigates and reasons within the scene graph.
5. **Interoperability with X3D Ontology**: PM-KR layer metadata maps cleanly to OWL 2 classes, enabling SPARQL queries over procedural knowledge structures.

### 1.3 Terminology

The keywords **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**, **SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY**, and **OPTIONAL** in this document are to be interpreted as described in RFC 2119.

| Term | Definition | Source |
|------|-----------|--------|
| **Procedural Memory** | Knowledge stored as executable procedures, not static duplicated payloads | PM-KR Spec §3 |
| **Canonical Source** | The single authoritative procedural representation of a symbol, object, or rule | PM-KR Spec §3 |
| **Symlink Reference** | Lightweight reference from one node to an already canonical node | PM-KR Spec §3 |
| **Form** | Procedural representation of structure and appearance (Layer 1) | Foundational Knowledge Spec §1.2 |
| **Meaning** | Semantic behavior, interpretation, or executable transformation semantics (Layer 2) | Foundational Knowledge Spec §1.3 |
| **Rules** | Executable transformation programs referencing lower layers (Layer 3) | Foundational Knowledge Spec §1.4 |
| **Meta-Rules** | Strategy and control over rule selection and consolidation (Layer 4) | Foundational Knowledge Spec §1.5 |
| **RPN Program** | A sequence of opcodes executed on a stack machine (Reverse Polish Notation) | RPN Domain Opcode Registry §1 |
| **Dual-Client** | Architecture where human and synthetic users consume the same node truth | Dual-Client Contract §1.3 |
| **Galaxy** | A named collection of knowledge nodes in a domain (e.g., Math Galaxy, Grammar Galaxy) | Knowledgeverse Spec §2.1 |
| **House** | A persistent 3D spatial environment (Memory Palace) containing rooms, objects, and knowledge | Three Brain System Spec |
| **Sovereignty** | Execution using only self-contained, inspectable primitives (no external framework dependencies in the hot path) | Sovereign NSI Spec §1.2 |

---

## 2. Scope and Design Rationale

### 2.1 What This Extension Defines

This extension defines X3D node types, field semantics, and conformance criteria for:

1. **Procedural knowledge nodes** with four-layer classification (Form, Meaning, Rules, Meta-Rules).
2. **Executable program fields** that carry RPN instruction sequences as first-class node data.
3. **Canonical reference fields** that implement the symlink compression pattern.
4. **Dual-client texture contracts** for simultaneous human and machine consumption.
5. **Knowledge galaxy grouping** for organizing procedural nodes into named domain collections.
6. **Spatial knowledge indexing** for octree-based spatial queries over knowledge nodes.
7. **Agent embodiment** for representing autonomous reasoning entities within the scene.

### 2.2 What This Extension Does Not Define

- Specific GPU kernel implementations (sovereign execution is an implementation concern, not a scene format concern).
- Training or learning procedures (sleep-time consolidation is a runtime behavior, not a scene description).
- Specific RPN opcode assignments (the opcode registry is normatively referenced but not redefined here).
- Network protocols for multi-user collaboration (Doors protocol is out of scope for scene format).
- Benchmark or evaluation procedures.

### 2.3 Design Rationale: Why Both Component and Profile

**X3D Component** is the correct mechanism for introducing new node types, abstract interfaces, and field semantics. PM-KR requires two components because the concerns are separable:

- **ProceduralMemory**: The knowledge representation layer (nodes, layers, references, programs). An implementation could support this component without supporting spatial navigation --- useful for pure knowledge interchange (e.g., exporting a Grammar Galaxy as an X3D file).

- **KnowledgeNavigation**: The spatial indexing and agent embodiment layer (octrees, pathfinding, agents). An implementation could use standard X3D navigation without PM-KR knowledge nodes --- but this component enables the "world IS the memory" paradigm where spatial position carries semantic meaning.

**X3D Profile** is the correct mechanism for bundling these components with existing X3D components into a conformance target. The ProceduralMemoryInterchange profile specifies which existing X3D capabilities (Grouping, Shape, Texturing, Metadata, etc.) are required alongside the new PM-KR components.

### 2.4 Relationship to X3D Extension Mechanisms

This specification uses the **formal component extension** path (ISO/IEC 9973 registration), not the PROTO/EXTERNPROTO author-level extension path. The rationale:

1. PM-KR nodes carry semantic contracts (canonicality, reference preservation, dual-client equivalence) that PROTO cannot enforce.
2. The four-layer hierarchy requires abstract type inheritance that PROTO does not support.
3. Conformance testing requires component-level support declarations.

However, a **PROTO-based reference implementation** SHOULD be provided for immediate experimentation in existing X3D browsers, prior to formal component registration. See Section 15.

---

## 3. Concepts

### 3.1 Four-Layer Knowledge Architecture

PM-KR represents knowledge across four hierarchical layers, each progressively more abstract:

```
Layer 4: META-RULES (Strategy / Control)
    | when and why to apply
    v
Layer 3: RULES (Transformation / Grammar)
    | how to transform
    v
Layer 2: MEANING (Semantics / Behavior)
    | what it means
    v
Layer 1: FORM (Primitives / Appearance)
    | how it looks and sounds
```

**Critical constraint**: Higher layers MUST reference lower layers via canonical symlink references. Higher layers MUST NOT duplicate lower-layer payload data. This is the **Reference Preservation Invariant** (PM-KR Spec §5.2).

**Rationale**: Without reference preservation, a grammar rule that uses 20 mathematical symbols would store 20 copies of each symbol's procedural program. With reference preservation, it stores 20 lightweight references (4 bytes each). At scale (1,000 rules x 152 symbols), this yields 190,000x compression (PM-KR Spec §4.3, Foundational Knowledge Spec §2.2).

**Cross-domain discovery** emerges naturally: when multiple rules across different domains (calculus, statistics, finance) reference the same canonical symbol (e.g., summation ∑), the system discovers cross-domain connections through shared references.

### 3.2 Procedural Primacy

In PM-KR, **the program IS the knowledge**. This differs fundamentally from X3D's current model where geometry is declarative data and behavior is added via Script nodes:

| Aspect | X3D Current Model | PM-KR Extension |
|--------|-------------------|-----------------|
| Knowledge storage | Geometry + metadata fields | Executable RPN programs |
| Behavior | Script nodes (JavaScript/Java) | RPN programs in node fields |
| Composition | ROUTE event wiring | Symlink references across layers |
| Identity | Node DEF name | Canonical ID + layer classification |
| Compression | DEF/USE within scene | Cross-layer symlink references |
| Execution model | Event cascade + script callbacks | Stack machine (RPN) |

**PM-KR does NOT replace X3D's existing model.** Procedural knowledge nodes coexist with standard X3D geometry nodes in the same scene graph. A ProceduralMemoryNode can be a child of a standard Transform node, positioned in 3D space alongside standard Shape nodes.

### 3.3 Dual-Client Contract

Every PM-KR node in the scene graph serves two clients simultaneously:

**Human Client** perceives the node through visual rendering:
- UV Map 0 texture: high-resolution aesthetic rendering (512x512+ RGB)
- Standard X3D Appearance/Material for visual presentation
- Readable fonts, proper layout, interactive highlights

**Synthetic Client** perceives the node through procedural execution:
- UV Map 1 texture: compressed semantic data (256x256)
- RPN program fields for executable knowledge
- Embedding vectors for spatial similarity queries

**Guaranteed identity** (Dual-Client Contract §2.1): Both clients query the same X3D node at the same (x, y, z) position and receive data derived from the same canonical procedural source.

### 3.4 Spatial Semantics

In PM-KR scenes, **spatial position carries semantic meaning**. Nodes that are semantically similar are spatially proximate. This is not a visualization convenience --- it is a computational invariant:

- **Morton octree indexing** maps semantic embeddings to 3D positions via space-filling curves.
- **Frustum culling** on the knowledge graph is semantically meaningful: what is "in view" is what is "relevant."
- **Level of Detail (LOD)** on knowledge nodes controls reasoning depth: distant (irrelevant) knowledge is processed at lower resolution.
- **Pathfinding** through the knowledge graph is spatial navigation through the scene.

This means an X3D browser that supports spatial queries (proximity, frustum, LOD) over PM-KR nodes is performing **knowledge retrieval**, not just scene rendering.

---

## 4. Component Definition: ProceduralMemory

### 4.1 Component Name

`ProceduralMemory`

### 4.2 Component Overview

The ProceduralMemory component provides node types for representing knowledge as executable procedures organized in a four-layer compositional hierarchy, connected by canonical references.

### 4.3 Component Levels

**Table 4.1 --- ProceduralMemory component support levels**

| Level | Prerequisites | Nodes Added | Description |
|-------|--------------|-------------|-------------|
| 1 | Core:1, Grouping:1, Metadata:1 | ProceduralFormNode, ProceduralMeaningNode, RPNProgram, CanonicalReference | Basic four-layer nodes with RPN programs and references |
| 2 | Core:1, Grouping:1, Metadata:1, Texturing:1 | ProceduralRulesNode, ProceduralMetaRulesNode, DualClientTexture, GalaxyGroup | Full four-layer hierarchy with dual-client textures and galaxy grouping |
| 3 | Core:1, Grouping:2, Metadata:1, Texturing:2, Shape:2 | ProceduralCompositeNode, DefeasibleRule, CompressionProgram | Defeasible logic, procedural compression (PD04), composite nodes |

---

## 5. Abstract Node Types

### 5.1 X3DProceduralNode

Base abstract type for all PM-KR procedural knowledge nodes. Extends X3DChildNode (usable as children in grouping nodes) and X3DMetadataObject (carries metadata).

```
X3DProceduralNode : X3DChildNode, X3DBoundedObject {
  SFString [in,out] canonicalId     ""           # Stable unique identifier (PM-KR §6)
  SFString [in,out] layer           "form"       # One of: "form", "meaning", "rules", "meta_rules"
  SFString [in,out] galaxy          ""           # Galaxy domain name (e.g., "Math", "Grammar", "Drawing")
  SFString [in,out] domain          ""           # Sub-domain within galaxy (e.g., "calculus", "morphology")
  SFFloat  [in,out] confidence      1.0          [0,1]   # Source confidence
  SFString [in,out] provenance      ""           # Provenance URI or description
  SFString [in,out] version         "1.0"        # Versioned namespace identifier
  SFTime   [in,out] timestamp       0            # Creation or last-modification time
  SFNode   [in,out] formProgram     NULL         [RPNProgram]      # Layer 1: procedural form
  SFNode   [in,out] meaningProgram  NULL         [RPNProgram]      # Layer 2: procedural meaning
  MFNode   [in,out] canonicalRefs   []           [CanonicalReference]  # Symlink references to canonical nodes
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
  SFNode   [in,out] embedding       NULL         [MetadataFloat]   # Semantic embedding vector (for spatial indexing)
  SFBool   [in,out] bboxDisplay     FALSE
  SFBool   [in,out] visible         TRUE
  SFVec3f  []       bboxCenter      0 0 0        (-inf,inf)
  SFVec3f  []       bboxSize        -1 -1 -1     [0,inf) or -1 -1 -1
}
```

**Behavioral rules:**

1. The `canonicalId` field MUST be unique within its versioned namespace. Two nodes with the same `canonicalId` and `version` MUST represent the same canonical concept.

2. The `layer` field constrains which program fields are normatively required:
   - `"form"`: `formProgram` MUST be present; `meaningProgram` is OPTIONAL.
   - `"meaning"`: `meaningProgram` MUST be present; `formProgram` is OPTIONAL (MAY reference a form node via `canonicalRefs`).
   - `"rules"`: `meaningProgram` MUST be present (the rule IS a transformation program); `formProgram` is OPTIONAL.
   - `"meta_rules"`: `meaningProgram` MUST be present (the meta-rule IS a strategy program); `formProgram` is OPTIONAL.

3. The `canonicalRefs` field contains references to other PM-KR nodes. Per the Reference Preservation Invariant (PM-KR Spec §5.2), if this node reuses content from another canonical node, it MUST reference that node via `canonicalRefs` rather than duplicating the content in its own program fields.

4. The `embedding` field, when present, MUST contain a MetadataFloat node whose `value` field holds the semantic embedding vector. The embedding dimension is not constrained by this specification but MUST be consistent across all nodes in a scene intended for spatial similarity queries.

5. The `galaxy` field groups nodes into named domain collections. Standard galaxy names defined by PM-KR are: `"Drawing"`, `"Character"`, `"Word"`, `"Number"`, `"Grammar"`, `"Math"`, `"Reality"`, `"Audio"`, `"3DObjects"`, `"Tool"`. Implementations MAY define additional galaxy names.

### 5.2 X3DKnowledgeLayerNode

Abstract type for nodes that represent a specific layer in the four-layer hierarchy. Adds layer-specific fields.

```
X3DKnowledgeLayerNode : X3DProceduralNode {
  # Inherits all fields from X3DProceduralNode
  # Constrains 'layer' to a specific value (enforced per concrete type)
}
```

---

## 6. Concrete Node Reference

### 6.1 ProceduralFormNode

Represents a Layer 1 (Form) knowledge node. Canonical procedural primitive: a glyph, shape, sound waveform, or other sensory-level procedure.

```
ProceduralFormNode : X3DKnowledgeLayerNode {
  # layer field is implicitly "form" and MUST NOT be set to any other value
  SFString [in,out] canonicalId     ""
  SFString [in,out] galaxy          ""
  SFString [in,out] domain          ""
  SFFloat  [in,out] confidence      1.0          [0,1]
  SFString [in,out] provenance      ""
  SFString [in,out] version         "1.0"
  SFTime   [in,out] timestamp       0
  SFNode   [in,out] formProgram     NULL         [RPNProgram]      # REQUIRED for form nodes
  SFNode   [in,out] meaningProgram  NULL         [RPNProgram]      # OPTIONAL
  MFNode   [in,out] canonicalRefs   []           [CanonicalReference]
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
  SFNode   [in,out] embedding       NULL         [MetadataFloat]
  SFNode   [in,out] appearance      NULL         [X3DAppearanceNode]  # Visual rendering for human client
  SFNode   [in,out] dualTexture     NULL         [DualClientTexture]  # Dual-client texture pair
  SFBool   [in,out] bboxDisplay     FALSE
  SFBool   [in,out] visible         TRUE
  SFVec3f  []       bboxCenter      0 0 0        (-inf,inf)
  SFVec3f  []       bboxSize        -1 -1 -1     [0,inf) or -1 -1 -1
}
```

**Behavioral rules:**

1. `formProgram` MUST be present and non-NULL. A form node without a form program is invalid.

2. The `formProgram` RPN program, when executed, MUST produce a visual or auditory output that constitutes the node's canonical form. For visual forms, this is a sequence of drawing commands (MOVE, LINE, ARC, BEZIER, CLOSE, STROKE, FILL). For auditory forms, this is a sequence of audio synthesis commands.

3. If `appearance` is present, it provides X3D-standard visual rendering for human clients. The appearance SHOULD be derived from the `formProgram` output but MAY be independently authored for aesthetic purposes.

4. Form nodes are canonical sources. Other nodes that reference this form MUST use `CanonicalReference` nodes pointing to this node's `canonicalId`, not duplicate the `formProgram` content.

**Example --- Summation Symbol (U+2211):**
```xml
<ProceduralFormNode DEF="CHAR_2211"
    canonicalId="char:U+2211"
    galaxy="Character"
    domain="math_symbol"
    confidence="1.0"
    version="1.0">
  <RPNProgram containerField="formProgram"
      opcodes="32 8 MOVE 8 32 LINE 32 56 LINE 56 56 LINE 32 32 MOVE 8 32 LINE 32 8 LINE STROKE"/>
  <MetadataFloat containerField="embedding"
      name="embedding16"
      value="0.82 -0.15 0.43 0.91 -0.27 0.56 0.18 -0.64 0.33 0.77 -0.41 0.22 0.68 -0.09 0.51 0.37"/>
  <MetadataSet containerField="metadata">
    <MetadataString name="unicode" value='"U+2211"'/>
    <MetadataString name="name" value='"N-ARY SUMMATION"'/>
    <MetadataString name="script" value='"Common"'/>
    <MetadataString name="languages" value='"en" "pt" "de" "fr" "zh" "ja"'/>
  </MetadataSet>
</ProceduralFormNode>
```

### 6.2 ProceduralMeaningNode

Represents a Layer 2 (Meaning) knowledge node. Semantic composition over form references.

```
ProceduralMeaningNode : X3DKnowledgeLayerNode {
  # layer field is implicitly "meaning" and MUST NOT be set to any other value
  SFString [in,out] canonicalId     ""
  SFString [in,out] galaxy          ""
  SFString [in,out] domain          ""
  SFFloat  [in,out] confidence      1.0          [0,1]
  SFString [in,out] provenance      ""
  SFString [in,out] version         "1.0"
  SFTime   [in,out] timestamp       0
  SFNode   [in,out] formProgram     NULL         [RPNProgram]      # OPTIONAL (may reference form nodes)
  SFNode   [in,out] meaningProgram  NULL         [RPNProgram]      # REQUIRED for meaning nodes
  MFNode   [in,out] canonicalRefs   []           [CanonicalReference]  # MUST reference Layer 1 sources
  MFString [in,out] charRefs        []           # Character Galaxy canonical IDs (Layer 1 symlinks)
  MFString [in,out] symbolRefs      []           # Symbol canonical IDs (Layer 1 symlinks)
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
  SFNode   [in,out] embedding       NULL         [MetadataFloat]
  SFNode   [in,out] dualTexture     NULL         [DualClientTexture]
  SFBool   [in,out] bboxDisplay     FALSE
  SFBool   [in,out] visible         TRUE
  SFVec3f  []       bboxCenter      0 0 0        (-inf,inf)
  SFVec3f  []       bboxSize        -1 -1 -1     [0,inf) or -1 -1 -1
}
```

**Behavioral rules:**

1. `meaningProgram` MUST be present and non-NULL. A meaning node without a meaning program is invalid.

2. The `meaningProgram` RPN program encodes the semantic behavior of this concept. For a word, this might be a program that, given a context, produces the word's contribution to meaning. For a mathematical concept, this might be the operational definition (e.g., summation = iterated addition over an index range).

3. `charRefs` and `symbolRefs` are convenience fields that contain canonical IDs of Layer 1 nodes. These are symlink references. The referenced nodes MUST exist as ProceduralFormNode instances elsewhere in the scene (or in a referenced external scene via Inline). Implementations MUST NOT resolve these references by inlining the referenced node's program data.

4. A meaning node MUST NOT duplicate Layer 1 visual data. If a meaning node needs to display a glyph, it MUST reference the canonical form node.

**Example --- Word "derivative" referencing characters:**
```xml
<ProceduralMeaningNode DEF="WORD_DERIVATIVE"
    canonicalId="word:en:derivative"
    galaxy="Word"
    domain="calculus"
    confidence="1.0"
    charRefs='"char:U+0064" "char:U+0065" "char:U+0072" "char:U+0069"
              "char:U+0076" "char:U+0061" "char:U+0074" "char:U+0069"
              "char:U+0076" "char:U+0065"'>
  <RPNProgram containerField="meaningProgram"
      opcodes="LOAD_CONTEXT RECALL_FUNCTION RECALL_VARIABLE LIMIT_DELTA APPLY_QUOTIENT NORMALIZE"/>
  <MetadataSet containerField="metadata">
    <MetadataString name="definition" value='"Rate of change of a function with respect to a variable"'/>
    <MetadataString name="domains" value='"calculus" "analysis" "physics"'/>
  </MetadataSet>
</ProceduralMeaningNode>
```

### 6.3 ProceduralRulesNode

Represents a Layer 3 (Rules) knowledge node. Executable transformation program referencing lower layers.

```
ProceduralRulesNode : X3DKnowledgeLayerNode {
  # layer field is implicitly "rules" and MUST NOT be set to any other value
  SFString [in,out] canonicalId     ""
  SFString [in,out] galaxy          ""
  SFString [in,out] domain          ""
  SFFloat  [in,out] confidence      1.0          [0,1]
  SFString [in,out] provenance      ""
  SFString [in,out] version         "1.0"
  SFTime   [in,out] timestamp       0
  SFNode   [in,out] formProgram     NULL         [RPNProgram]
  SFNode   [in,out] meaningProgram  NULL         [RPNProgram]      # REQUIRED: the rule IS a transform program
  MFNode   [in,out] canonicalRefs   []           [CanonicalReference]
  MFString [in,out] symbolRefs      []           # Layer 1 canonical IDs used by this rule
  MFString [in,out] wordRefs        []           # Layer 2 canonical IDs used by this rule
  MFString [in,out] ruleRefs        []           # Layer 3 canonical IDs (for rule chaining)
  SFString [in,out] pattern         ""           # Input pattern this rule matches
  SFString [in,out] language        ""           # Language scope (e.g., "en", "pt", "*" for universal)
  SFInt32  [in,out] ruleStrength    0            [-1,1]  # Trit: +1=strict, 0=defeasible, -1=defeater
  MFString [in,out] superiorTo      []           # Canonical IDs of rules this rule defeats
  SFFloat  [in,out] trustWeight     1.0          [0,1]   # Source trust weight
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
  SFNode   [in,out] embedding       NULL         [MetadataFloat]
  SFBool   [in,out] bboxDisplay     FALSE
  SFBool   [in,out] visible         TRUE
  SFVec3f  []       bboxCenter      0 0 0        (-inf,inf)
  SFVec3f  []       bboxSize        -1 -1 -1     [0,inf) or -1 -1 -1
}
```

**Behavioral rules:**

1. `meaningProgram` MUST be present. The rule's meaning program IS the transformation: given input matching `pattern`, execute this program to produce the transformation output.

2. `ruleStrength` encodes defeasible logic classification (Foundational Knowledge Spec §1.4, Sovereign NSI Spec §4.3):
   - `+1` (strict): Cannot be defeated. Mathematical axioms, verified facts.
   - `0` (defeasible): Default for most rules. Can be overridden by superior evidence.
   - `-1` (defeater): Blocks conclusions without proving alternatives.

3. `superiorTo` lists canonical IDs of other ProceduralRulesNode instances that this rule defeats when they conflict. This implements the superiority relation from defeasible logic.

4. `symbolRefs` and `wordRefs` are symlink references to Layer 1 and Layer 2 nodes. The Reference Preservation Invariant applies: these MUST be references, not inlined copies.

**Example --- Grammar rule for derivative notation:**
```xml
<ProceduralRulesNode DEF="RULE_LEIBNIZ_NOTATION"
    canonicalId="rule:calculus:leibniz_derivative"
    galaxy="Grammar"
    domain="calculus"
    pattern="d/dx f(x)"
    language="*"
    ruleStrength="1"
    symbolRefs='"char:U+2202" "char:U+002F"'
    wordRefs='"word:en:derivative" "word:en:function"'>
  <RPNProgram containerField="meaningProgram"
      opcodes="RECALL_NUMERATOR RECALL_DENOMINATOR APPLY_QUOTIENT PARTIAL_DERIVATIVE"/>
  <MetadataSet containerField="metadata">
    <MetadataString name="notation_type" value='"Leibniz"'/>
    <MetadataString name="alternatives" value='"rule:calculus:lagrange_derivative" "rule:calculus:newton_derivative"'/>
  </MetadataSet>
</ProceduralRulesNode>
```

### 6.4 ProceduralMetaRulesNode

Represents a Layer 4 (Meta-Rules) knowledge node. Strategy and control over rule selection.

```
ProceduralMetaRulesNode : X3DKnowledgeLayerNode {
  # layer field is implicitly "meta_rules" and MUST NOT be set to any other value
  SFString [in,out] canonicalId     ""
  SFString [in,out] galaxy          ""
  SFString [in,out] domain          ""
  SFFloat  [in,out] confidence      1.0          [0,1]
  SFString [in,out] provenance      ""
  SFString [in,out] version         "1.0"
  SFTime   [in,out] timestamp       0
  SFNode   [in,out] formProgram     NULL         [RPNProgram]
  SFNode   [in,out] meaningProgram  NULL         [RPNProgram]      # REQUIRED: the meta-rule IS a strategy program
  MFNode   [in,out] canonicalRefs   []           [CanonicalReference]
  MFString [in,out] ruleRefs        []           # Layer 3 canonical IDs this meta-rule governs
  SFString [in,out] category        ""           # e.g., "eloquence", "pedagogy", "self_reflection", "storytelling"
  SFNode   [in,out] condition       NULL         [RPNProgram]      # When to apply (RPN predicate)
  SFFloat  [in,out] priority        1.0          [0,inf)           # For consolidation ordering
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
  SFNode   [in,out] embedding       NULL         [MetadataFloat]
  SFBool   [in,out] bboxDisplay     FALSE
  SFBool   [in,out] visible         TRUE
  SFVec3f  []       bboxCenter      0 0 0        (-inf,inf)
  SFVec3f  []       bboxSize        -1 -1 -1     [0,inf) or -1 -1 -1
}
```

**Behavioral rules:**

1. `meaningProgram` MUST be present. The meta-rule's meaning program defines the strategy: what action to take when the `condition` predicate evaluates to true.

2. `condition`, when present, is an RPNProgram that evaluates to a boolean (or ternary) value. If the condition evaluates to true (or positive trit), the meta-rule's `meaningProgram` is applicable.

3. `ruleRefs` identifies the Layer 3 rules that this meta-rule governs. A meta-rule's strategy applies to the selection, ordering, or suppression of the referenced rules.

4. `priority` determines ordering when multiple meta-rules apply simultaneously. Higher values indicate higher priority.

### 6.5 RPNProgram

A node that encodes an executable RPN (Reverse Polish Notation) program.

```
RPNProgram : X3DNode {
  SFString [in,out] opcodes         ""           # Space-separated RPN instruction sequence
  MFFloat  [in,out] constants       []           # Numeric constants referenced by the program
  MFString [in,out] stringConstants []           # String constants referenced by the program
  SFInt32  [in,out] stackDepth      69           [1,256]  # Maximum stack depth
  SFInt32  [in,out] registerCount   16           [0,256]  # STORE/RECALL registers available
  SFString [in,out] tier            "standard"   # One of: "lite", "standard", "extended"
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
}
```

**Behavioral rules:**

1. The `opcodes` field contains a space-separated sequence of RPN tokens. Tokens are either:
   - Numeric literals (parsed as float constants pushed to stack)
   - Named opcodes from the RPN opcode registry (e.g., `MOVE`, `LINE`, `ADD`, `MUL`, `STORE`, `RECALL`, `BRANCH`)
   - Constant references (`$0`, `$1`, etc.) indexing into the `constants` array
   - String constant references (`@0`, `@1`, etc.) indexing into `stringConstants`

2. `tier` classifies the program's opcode requirements:
   - `"lite"` (0x00--0x3F): Basic arithmetic, stack manipulation, comparison
   - `"standard"` (0x40--0x9F): Drawing, vector operations, trigonometry, branching
   - `"extended"` (0xA0--0xFF): Ternary logic, quantum-inspired ops, advanced math

3. Execution semantics: push-down stack machine with STORE/RECALL registers for inter-operation communication. Stack underflow MUST be detected and reported as an error. Stack overflow beyond `stackDepth` MUST be detected and reported as an error.

4. Programs MUST be deterministic: given the same input stack state, the same program MUST produce the same output stack state on any conformant implementation.

### 6.6 CanonicalReference

A lightweight reference from one PM-KR node to another canonical node. Implements the symlink pattern.

```
CanonicalReference : X3DNode {
  SFString [in,out] targetId        ""           # Canonical ID of the referenced node
  SFString [in,out] targetLayer     ""           # Layer of the referenced node ("form", "meaning", "rules", "meta_rules")
  SFString [in,out] role            ""           # Semantic role of this reference (e.g., "char_ref", "symbol_ref", "rule_ref")
  SFString [in,out] targetVersion   ""           # Version constraint (empty = latest)
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
}
```

**Behavioral rules:**

1. `targetId` MUST match the `canonicalId` of exactly one X3DProceduralNode in the scene (or in a scene reachable via X3D Inline). If the target cannot be resolved, the implementation MUST report a reference resolution error (fail-fast, per PM-KR Spec §5.3).

2. Implementations MUST NOT resolve a CanonicalReference by copying the target node's program data into the referencing node. The reference is the representation --- it is not a cache hint.

3. `role` is informational and aids tooling. Standard roles: `"char_ref"`, `"symbol_ref"`, `"word_ref"`, `"rule_ref"`, `"component_ref"`.

### 6.7 DualClientTexture

A node that pairs two textures for simultaneous human and machine consumption.

```
DualClientTexture : X3DNode {
  SFNode   [in,out] humanTexture    NULL         [X3DTexture2DNode]  # UV Map 0: human-optimized
  SFNode   [in,out] machineTexture  NULL         [X3DTexture2DNode]  # UV Map 1: machine-optimized
  SFString [in,out] machineEncoding "text_as_image"  # Encoding type for machine texture
  SFFloat  [in,out] machineFidelity 0.97         [0,1]  # Expected OCR/decode fidelity
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
}
```

**Behavioral rules:**

1. `humanTexture` SHOULD be a high-resolution texture (512x512 or larger) optimized for visual perception: readable fonts (14--18pt equivalent), proper layout, aesthetic color.

2. `machineTexture` SHOULD be a compressed texture (256x256 or smaller) optimized for machine consumption: maximum information density, small fonts (6--8pt), structured layout.

3. Both textures MUST represent the same underlying knowledge content. They differ only in representation modality, not in semantic content.

4. `machineEncoding` indicates how the machine texture should be decoded. Standard values: `"text_as_image"` (OCR-style decode), `"embedding_grid"` (direct float extraction), `"rpn_encoded"` (RPN program encoded as pixel values).

### 6.8 GalaxyGroup

A grouping node that represents a named knowledge galaxy (domain collection).

```
GalaxyGroup : X3DGroupingNode {
  MFNode   [in]     addChildren                  [X3DChildNode]
  MFNode   [in]     removeChildren               [X3DChildNode]
  MFNode   [in,out] children        []           [X3DChildNode]
  SFString [in,out] galaxyName      ""           # Galaxy name (e.g., "Math", "Grammar")
  SFString [in,out] galaxyType      "default"    # One of: "default", "custom", "meta"
  SFInt32  [in,out] entryCount      0            [0,inf)  # Number of knowledge entries
  SFFloat  [in,out] loadPriority    1.0          [0,inf)  # Priority for memory management
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
  SFBool   [in,out] bboxDisplay     FALSE
  SFBool   [in,out] visible         TRUE
  SFVec3f  []       bboxCenter      0 0 0        (-inf,inf)
  SFVec3f  []       bboxSize        -1 -1 -1     [0,inf) or -1 -1 -1
}
```

**Behavioral rules:**

1. All ProceduralFormNode, ProceduralMeaningNode, ProceduralRulesNode, and ProceduralMetaRulesNode children of a GalaxyGroup inherit the `galaxyName` as their default galaxy if their own `galaxy` field is empty.

2. Standard galaxy names for `galaxyType="default"`:
   - `"Drawing"`: Visual primitives (LINE, CIRCLE, RECT as RPN programs)
   - `"Character"`: Glyphs with font/language/pronunciation/meaning
   - `"Word"`: Character sequences (symlinked references to Character galaxy)
   - `"Number"`: Numeric representations
   - `"Grammar"`: Transformation rules (RPN) + context metadata
   - `"Math"`: Symbols with RPN templates
   - `"Reality"`: Physics/chemistry/biology procedural systems
   - `"Audio"`: Temporal patterns, spectrograms
   - `"3DObjects"`: 3D mesh primitives
   - `"Tool"`: Meta-programs and utilities

3. `galaxyType="meta"` identifies the Meta-Navigation Galaxy, which stores routing topology learned during operation.

### 6.9 DefeasibleRule (Level 3)

A specialized rules node with explicit defeasible logic semantics aligned with the Sovereign NSI Spec §4.3 triple-stage pipeline.

```
DefeasibleRule : ProceduralRulesNode {
  # Inherits ALL fields from ProceduralRulesNode
  SFString [in,out] defeasibleStage  "intra_path"  # One of: "early_gate", "intra_path", "final_resolution"
  SFString [in,out] proofTag         ""             # Proof tag for halting gate verification
  SFNode   [in,out] strictChain      NULL           [RPNProgram]  # Strict support accumulation program
  SFNode   [in,out] defeasibleChain  NULL           [RPNProgram]  # Defeasible support accumulation program
}
```

**Behavioral rules:**

1. `defeasibleStage` identifies where in the triple-stage pipeline this rule participates:
   - `"early_gate"`: Path-level pruning of defeated reasoning paths.
   - `"intra_path"`: Candidate-level conflict resolution within a path.
   - `"final_resolution"`: Cross-path verdict production with proof tags.

2. The `ruleStrength`, `superiorTo`, and `trustWeight` fields inherited from ProceduralRulesNode define the defeasible logic relationships. The ternary encoding (+1, 0, -1) aligns with RPN ternary opcodes TADD (0x70), TMUL (0x71), TNOT (0x72), TCOMP (0x73), TPACK (0x75).

### 6.10 CompressionProgram (Level 3)

A node representing a PD04 adaptive procedural compression program for embeddings.

```
CompressionProgram : X3DNode {
  SFInt32  [in,out] dimension        128          [64,2048]  # Target embedding dimension
  SFInt32  [in,out] prototypeIndex   0            [0,inf)    # Dictionary prototype index
  MFFloat  [in,out] deltaIndices     []           # Sparse delta: index positions
  MFFloat  [in,out] deltaValues      []           # Sparse delta: values at those positions
  SFFloat  [in,out] fidelity         0.999        [0,1]      # Expected reconstruction fidelity (cosine)
  SFString [in,out] qualityLevel     "fast"       # One of: "ultrafast", "fast", "balanced", "maximum"
  SFNode   [in,out] metadata         NULL         [X3DMetadataObject]
}
```

**Behavioral rules:**

1. Reconstruction follows the PD04 program structure (Adaptive Procedural Compression Spec §1.2):
   ```
   LOAD_PROTOTYPE(prototypeIndex) → ADD_DELTA(deltaIndices, deltaValues) → NORMALIZE
   ```

2. `dimension` constrains the output dimensionality. Standard values: 64 (ultrafast, ~100 bytes), 128 (fast, ~115 bytes), 512 (balanced, ~340 bytes), 2048 (maximum, ~700 bytes).

3. `fidelity` is the expected cosine similarity between the original embedding and the reconstructed embedding. Implementations MUST verify this during validation.

---

## 7. RPN Program Encoding

### 7.1 Opcode Tiers

RPN programs use a tiered opcode system (RPN Domain Opcode Registry §5):

**Tier 1 --- Lite (0x00--0x3F):**
Basic arithmetic (ADD, SUB, MUL, DIV, MOD, POW), stack manipulation (DUP, SWAP, ROT, DROP), comparison (EQ, LT, GT, LE, GE), constants (PUSH, PI, E).

**Tier 2 --- Standard (0x40--0x9F):**
Drawing (MOVE, LINE, ARC, BEZIER, CLOSE, STROKE, FILL), vector operations (DOT_PRODUCT, CROSS_PRODUCT, VEC_NORMALIZE, VEC_BLEND), trigonometry (SIN, COS, TAN, ATAN2), control flow (BRANCH, LOOP, CALL, RET), memory (STORE, RECALL, LOAD_GALAXY, SAVE_GALAXY).

**Tier 3 --- Extended (0xA0--0xFF):**
Ternary logic (TADD 0x70, TMUL 0x71, TNOT 0x72, TCOMP 0x73, TQUANT 0x74, TPACK 0x75, TUNPACK 0x76), calculus operators (DIVERGENCE, CURL, LAPLACIAN), set operations (SET_UNION, SET_INTERSECTION, SET_DIFFERENCE), quantum-inspired (QUANTUM_SUPERPOSE, QUANTUM_MEASURE).

### 7.2 Encoding in X3D

RPN programs are encoded in the `opcodes` field of RPNProgram as a space-separated string of tokens. This encoding is chosen for human readability in XML and compatibility with X3D's SFString field type.

**Example:**
```xml
<RPNProgram opcodes="32 8 MOVE 8 32 LINE 32 56 LINE STROKE"
            tier="standard"
            stackDepth="69"/>
```

**Binary encoding**: For the X3D Compressed Binary Encoding (ISO/IEC 19776-3), RPN programs MAY be encoded as a byte sequence using the opcode numeric values directly, prefixed by a 4-byte program length.

### 7.3 Programs Before Opcodes

Per the RPN Domain Opcode Registry §1: "Prefer to build domain semantics as RPN programs over the existing math surface instead of immediately adding domain-specific opcodes." This principle applies to X3D extensions:

- New domain semantics (physics simulations, chemical reactions, biological processes) SHOULD be expressed as RPN programs using existing opcodes (Class A or Class B in the opcode registry).
- New opcodes SHOULD only be introduced after the opcode admission pipeline (Registry §6) demonstrates frequency, speedup, stable semantics, and sovereignty compliance.

---

## 8. Symlink Reference System

### 8.1 Reference Resolution

CanonicalReference nodes form a directed acyclic graph (DAG) across the four-layer hierarchy. Resolution follows these rules:

1. **Scope**: References resolve within the current scene first, then in scenes loaded via X3D Inline nodes, in document order.

2. **Versioning**: If `targetVersion` is specified, only nodes matching that version are candidates. If empty, the latest version in scope is selected.

3. **Fail-fast**: Unresolvable references MUST cause an explicit error. Implementations MUST NOT silently ignore broken references or substitute default values.

4. **Acyclicity**: Reference graphs SHOULD be acyclic. Implementations MUST detect cycles and report them as errors.

### 8.2 Compression Metrics

Implementations SHOULD report reference compression metrics:

- **Reference count**: Total CanonicalReference nodes in the scene.
- **Unique targets**: Number of distinct canonical IDs referenced.
- **Estimated savings**: Bytes saved by reference vs. inline duplication.

These metrics enable the Auditability Invariant (PM-KR Spec §5.6).

---

## 9. Dual-Client Rendering Contract

### 9.1 Texture Mapping Convention

PM-KR nodes that carry visual representation use the dual-texture convention from the Dual-Client Contract Specification §2.3:

- **UV Map 0 (texCoord index 0)**: Human-optimized texture. Standard X3D TextureCoordinate mapping.
- **UV Map 1 (texCoord index 1)**: Machine-optimized texture. Same geometry, different information density.

An X3D browser rendering for a human client uses UV Map 0. A machine client (or X3D browser in "AI mode") uses UV Map 1.

### 9.2 Identity Guarantee

For any PM-KR node at position (x, y, z) in the scene:
- The human client rendering (UV Map 0 + Appearance) and the machine client data (UV Map 1 + RPNProgram + embedding) MUST derive from the same canonical source.
- Modification of the canonical source MUST update both representations atomically.
- SHA-256 verification: `hash(human_node_data)` and `hash(machine_node_data)` need not be equal (different representations), but both MUST be derivable from `hash(canonical_source)`.

---

## 10. Component Definition: KnowledgeNavigation

### 10.1 Component Name

`KnowledgeNavigation`

### 10.2 Component Overview

The KnowledgeNavigation component provides node types for spatial knowledge indexing, graph-based pathfinding, and autonomous agent embodiment within X3D scenes containing PM-KR knowledge nodes.

### 10.3 Component Levels

**Table 10.1 --- KnowledgeNavigation component support levels**

| Level | Prerequisites | Nodes Added | Description |
|-------|--------------|-------------|-------------|
| 1 | Core:1, Grouping:1, ProceduralMemory:1 | SpatialKnowledgeIndex, KnowledgeQuery | Spatial indexing and basic queries over knowledge nodes |
| 2 | Core:1, Grouping:2, ProceduralMemory:2, Navigation:1 | KnowledgePathfinder, KnowledgeFrustum, KnowledgeLOD | Graph navigation, frustum culling, and LOD for knowledge |
| 3 | Core:2, Grouping:2, ProceduralMemory:2, Navigation:2 | AgentEntity, AgentSwarm, AgentMemoryPalace | Autonomous agent embodiment and swarm reasoning |

### 10.4 SpatialKnowledgeIndex (Level 1)

Provides spatial indexing over PM-KR nodes using octree-based structures (Morton code mapping).

```
SpatialKnowledgeIndex : X3DChildNode {
  MFNode   [in,out] knowledgeNodes  []           [X3DProceduralNode]  # Nodes to index
  SFString [in,out] indexType       "morton"     # One of: "morton", "kdtree", "bvh"
  SFInt32  [in,out] maxDepth        8            [1,16]   # Octree depth
  SFFloat  [in,out] cellSize        1.0          (0,inf)  # Leaf cell size
  SFBool   [out]    isBuilt         FALSE        # TRUE after index construction
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
}
```

### 10.5 KnowledgeQuery (Level 1)

A sensor-like node that queries the spatial knowledge index and outputs matching nodes.

```
KnowledgeQuery : X3DSensorNode {
  SFVec3f  [in,out] queryPosition   0 0 0        (-inf,inf)  # Query center
  SFFloat  [in,out] queryRadius     5.0          (0,inf)     # Search radius
  SFInt32  [in,out] maxResults      128          [1,4096]    # Maximum returned nodes
  MFString [in,out] galaxyFilter    []           # Restrict to these galaxies (empty = all)
  SFFloat  [in,out] similarityThreshold 0.18     [0,1]       # Minimum embedding similarity
  MFNode   [out]    results         []           [X3DProceduralNode]  # Matching nodes
  SFInt32  [out]    resultCount     0            [0,inf)     # Number of results
  SFBool   [in,out] enabled         TRUE
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
}
```

### 10.6 KnowledgePathfinder (Level 2)

Graph-based pathfinding over the knowledge node graph, implementing LED-A* (Lazy-Expanding A* on Dependency-Dense Graphs).

```
KnowledgePathfinder : X3DChildNode {
  SFNode   [in,out] index           NULL         [SpatialKnowledgeIndex]  # Spatial index to navigate
  SFVec3f  [in,out] startPosition   0 0 0        (-inf,inf)
  SFVec3f  [in,out] goalPosition    0 0 0        (-inf,inf)
  SFFloat  [in,out] alpha           0.35         [0,1]       # Geometric cost weight
  SFFloat  [in,out] beta            0.65         [0,1]       # Semantic cost weight
  SFInt32  [in,out] maxPathLength   128          [1,4096]    # Maximum path hops
  MFNode   [out]    pathNodes       []           [X3DProceduralNode]  # Ordered path result
  SFNode   [out]    focusNode       NULL         [X3DProceduralNode]  # Best destination node
  SFInt32  [out]    pathLength      0            [0,inf)
  SFBool   [in,out] enabled         TRUE
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
}
```

**Behavioral rules:**

1. Pathfinding computes the shortest path through the knowledge graph where edge cost combines geometric distance (`alpha` weight) and semantic distance (`beta` weight). Per the LED-A* kernel specification: `edge_cost = alpha * geometric_cost + beta * semantic_cost`.

2. `focusNode` outputs the terminal knowledge node (best destination) found by the pathfinder. This node receives `led_focus: 1.0` priority in candidate ranking.

### 10.7 KnowledgeFrustum (Level 2)

Frustum culling for knowledge nodes --- filters visible (relevant) knowledge based on a query viewpoint.

```
KnowledgeFrustum : X3DChildNode {
  SFVec3f  [in,out] viewPosition    0 0 0        (-inf,inf)  # Observer position
  SFVec3f  [in,out] viewDirection   0 0 -1       (-inf,inf)  # Look direction
  SFFloat  [in,out] fieldOfView     1.5708       (0,3.14159) # FOV in radians
  SFFloat  [in,out] nearDistance    0.1           (0,inf)     # Near clipping
  SFFloat  [in,out] farDistance     1000.0        (0,inf)     # Far clipping
  MFNode   [in,out] candidates      []           [X3DProceduralNode]  # Input candidates
  MFNode   [out]    visibleNodes    []           [X3DProceduralNode]  # Visible output
  SFInt32  [out]    visibleCount    0            [0,inf)
  SFBool   [in,out] enabled         TRUE
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
}
```

### 10.8 KnowledgeLOD (Level 2)

Level-of-detail control for knowledge nodes. Controls reasoning depth based on semantic distance.

```
KnowledgeLOD : X3DChildNode {
  MFNode   [in,out] candidates      []           [X3DProceduralNode]  # Input candidates
  SFFloat  [in,out] saliencyThreshold 0.3        [0,1]       # Minimum saliency for inclusion
  SFInt32  [in,out] focusLevel      3            [0,8]       # LOD level considered "in focus"
  MFNode   [out]    filteredNodes   []           [X3DProceduralNode]  # LOD-filtered output
  MFInt32  [out]    lodLevels       []           # LOD level per output node
  MFFloat  [out]    saliencies      []           # Saliency score per output node
  SFBool   [in,out] enabled         TRUE
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
}
```

### 10.9 AgentEntity (Level 3)

An autonomous reasoning agent embodied in the X3D scene. Represents the TRM avatar concept from the Three Brain System Specification.

```
AgentEntity : X3DChildNode, X3DBoundedObject {
  SFString [in,out] agentId         ""           # Unique agent identifier
  SFString [in,out] agentType       "trm"        # Agent architecture type
  SFVec3f  [in,out] position        0 0 0        (-inf,inf)  # Current position in scene
  SFRotation [in,out] orientation   0 0 1 0      # Current orientation
  SFNode   [in,out] body            NULL         [X3DChildNode]     # Visual representation (HAnim or geometry)
  SFNode   [in,out] internalState   NULL         [GalaxyGroup]      # Agent's internal knowledge (Galaxy)
  SFNode   [in,out] memoryPalace    NULL         [AgentMemoryPalace] # Long-term memory (House)
  SFInt32  [in,out] parameterCount  7000000      [0,inf)    # Model parameter count
  SFInt32  [in,out] maxRecursionSteps 9          [1,100]    # Max refinement iterations
  SFFloat  [in,out] convergenceThreshold 0.01    (0,inf)    # Convergence epsilon
  SFBool   [in,out] isActive        TRUE         # Whether agent is running its game loop
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
  SFBool   [in,out] bboxDisplay     FALSE
  SFBool   [in,out] visible         TRUE
  SFVec3f  []       bboxCenter      0 0 0        (-inf,inf)
  SFVec3f  []       bboxSize        -1 -1 -1     [0,inf) or -1 -1 -1
}
```

**Behavioral rules:**

1. An AgentEntity represents an autonomous cognitive entity that lives IN the scene, not an external process querying the scene. Per the Three Brain System Specification: "TRM IS the Avatar --- lives in House, thinks in Galaxy, runs as game loop."

2. `internalState` references a GalaxyGroup containing the agent's active working memory. This is the "Galaxy" in the Three Brain System: volatile, always loaded, multi-modal.

3. `memoryPalace` references the agent's long-term persistent memory structure. This is the "House" in the Three Brain System: persistent, spatially organized via Method of Loci.

4. When `isActive` is TRUE, the agent executes a continuous perception-reasoning-action loop:
   - Perceive: Frustum cull visible knowledge from current position
   - Navigate: Pathfind to relevant knowledge neighborhoods
   - Reason: Execute swarm workers on candidate knowledge
   - Decide: Check convergence (halting gate)
   - Act: Create new knowledge or emit answer
   - Learn: Record successful traces (shadow copy)

5. `maxRecursionSteps` and `convergenceThreshold` control the recursive refinement behavior (Sovereign NSI Spec §4.2): the agent iterates until the delta between refinement steps falls below the threshold, or the maximum step count is reached.

### 10.10 AgentSwarm (Level 3)

A group of parallel specialist reasoning workers within an agent (Hyper-Parallel Processing Specification).

```
AgentSwarm : X3DChildNode {
  SFNode   [in,out] agent           NULL         [AgentEntity]     # Parent agent
  SFInt32  [in,out] workerCount     9            [1,64]            # Number of parallel workers
  MFString [in,out] specialistNames []           # Named specialists (e.g., "math", "grammar", "visual")
  SFString [in,out] convergenceMode "one_mind"   # One of: "one_mind", "voting", "best_of_n"
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
}
```

**Behavioral rules:**

1. `convergenceMode="one_mind"` (Hyper-Parallel Processing Spec default): Workers communicate during execution via shared registers, producing one unified answer through convergence. NOT voting, NOT averaging.

2. Each specialist carries domain-specific learned parameters (LoRA-like adapters) that bias navigation toward relevant Galaxy neighborhoods. The specialist IS a spatial bias in Galaxy navigation, not a separate model.

### 10.11 AgentMemoryPalace (Level 3)

The persistent long-term memory structure for an agent, organized as a spatial environment (Method of Loci).

```
AgentMemoryPalace : X3DGroupingNode {
  MFNode   [in]     addChildren                  [X3DChildNode]
  MFNode   [in]     removeChildren               [X3DChildNode]
  MFNode   [in,out] children        []           [X3DChildNode]   # Rooms, objects, knowledge
  SFString [in,out] palaceId        ""           # Unique palace identifier
  SFString [in,out] persistencePath ""           # SSD/disk path for persistence
  SFInt32  [in,out] entryCount      0            [0,inf)          # Total knowledge entries
  SFInt32  [in,out] galaxyCount     0            [0,inf)          # Number of loaded galaxies
  SFNode   [in,out] metadata        NULL         [X3DMetadataObject]
  SFBool   [in,out] bboxDisplay     FALSE
  SFBool   [in,out] visible         TRUE
  SFVec3f  []       bboxCenter      0 0 0        (-inf,inf)
  SFVec3f  []       bboxSize        -1 -1 -1     [0,inf) or -1 -1 -1
}
```

---

## 11. ProceduralMemoryInterchange Profile

### 11.1 Profile Name

`ProceduralMemoryInterchange`

### 11.2 Purpose

This profile defines the minimum X3D capabilities required to author, view, and exchange procedural knowledge scenes. It targets knowledge interchange between systems that implement PM-KR, analogous to how the CADInterchange profile targets engineering data exchange.

### 11.3 Component Table

**Table 11.1 --- ProceduralMemoryInterchange profile components**

| Component | Level | Rationale |
|-----------|-------|-----------|
| Core | 1 | Metadata nodes (MetadataFloat, MetadataString, MetadataSet) |
| Grouping | 2 | Transform, Group, Switch (spatial organization of knowledge) |
| Rendering | 3 | Coordinate, IndexedFaceSet (geometry for dual-client rendering) |
| Shape | 2 | Shape, Appearance, Material, PhysicalMaterial (visual presentation) |
| Texturing | 2 | ImageTexture, TextureCoordinate, MultiTexture (dual-texture support) |
| Navigation | 1 | Viewpoint, NavigationInfo (scene navigation) |
| Networking | 2 | Inline, Anchor (external scene references for distributed knowledge) |
| Lighting | 1 | DirectionalLight (basic scene illumination) |
| Environmental effects | 1 | Background (scene environment) |
| Interpolation | 2 | PositionInterpolator, OrientationInterpolator (animation) |
| **ProceduralMemory** | **2** | **Full four-layer hierarchy, dual-client textures, galaxy grouping** |
| **KnowledgeNavigation** | **1** | **Spatial indexing and basic knowledge queries** |

### 11.4 Extended Profile: ProceduralMemoryImmersive

For implementations supporting autonomous agents and full spatial reasoning:

| Component | Level | Rationale |
|-----------|-------|-----------|
| *All from ProceduralMemoryInterchange* | *same* | *base requirements* |
| Grouping | 3 | StaticGroup (optimization for large knowledge scenes) |
| ProceduralMemory | 3 | Defeasible logic, compression programs |
| KnowledgeNavigation | 3 | Agent embodiment, swarm reasoning, memory palace |
| Humanoid animation | 1 | HAnim for agent visual embodiment |
| Sound | 1 | AudioClip (audio galaxy support) |
| Scripting | 1 | Script (for custom agent behaviors during prototyping) |

### 11.5 XML Header

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 4.0//EN"
          "http://www.web3d.org/specifications/x3d-4.0.dtd">
<X3D version='4.0' profile='ProceduralMemoryInterchange'>
  <head>
    <component name='ProceduralMemory' level='2'/>
    <component name='KnowledgeNavigation' level='1'/>
    <meta name='title' content='MathGalaxy.x3d'/>
    <meta name='description' content='Math Galaxy with 152 symbol form nodes and 200 grammar rules'/>
    <meta name='generator' content='K3D PM-KR Exporter v1.0'/>
    <meta name='pmkr:conformance' content='Level B (Sovereign Runtime)'/>
  </head>
  <Scene>
    <!-- Scene content -->
  </Scene>
</X3D>
```

---

## 12. glTF Interoperability

### 12.1 Mapping to glTF Extensions

Since X3D 4.0 supports glTF integration, PM-KR nodes map to glTF via the `extras.k3d` extension pattern already defined in the K3D implementation:

```json
{
  "nodes": [{
    "name": "char_U+2211",
    "translation": [10.5, 23.1, -5.3],
    "extras": {
      "k3d": {
        "canonicalId": "char:U+2211",
        "layer": "form",
        "galaxy": "Character",
        "formProgram": "32 8 MOVE 8 32 LINE 32 56 LINE STROKE",
        "embedding16": [0.82, -0.15, 0.43, ...],
        "semantic": {
          "rdf_subject": "http://unicode.org/U+2211",
          "rdf_predicate": "rdf:type",
          "rdf_object": "pm-kr:ProceduralFormNode"
        }
      }
    }
  }]
}
```

### 12.2 X3D-to-glTF Round-Trip

PM-KR X3D scenes SHOULD be exportable to glTF with full PM-KR metadata preservation via `extras.k3d`. The round-trip MUST preserve:
- Canonical IDs and layer classifications
- RPN program opcodes
- Canonical references (as canonical ID strings)
- Embedding vectors
- Galaxy grouping
- Dual-client texture assignments

### 12.3 X3D Ontology Integration

PM-KR node types map to OWL 2 classes for SPARQL querying, extending the X3D Ontology (Brutzman & Flotynski, 2020):

```turtle
@prefix pmkr: <http://www.w3.org/ns/pmkr#> .
@prefix x3d:  <http://www.web3d.org/specifications/x3d-ontology#> .

pmkr:ProceduralFormNode rdfs:subClassOf x3d:X3DChildNode .
pmkr:ProceduralMeaningNode rdfs:subClassOf x3d:X3DChildNode .
pmkr:ProceduralRulesNode rdfs:subClassOf x3d:X3DChildNode .
pmkr:ProceduralMetaRulesNode rdfs:subClassOf x3d:X3DChildNode .

pmkr:canonicalId rdfs:domain pmkr:X3DProceduralNode ;
                 rdfs:range  xsd:string .

pmkr:layer rdfs:domain pmkr:X3DProceduralNode ;
           rdfs:range  [ owl:oneOf ("form" "meaning" "rules" "meta_rules") ] .

pmkr:referencesCanonical rdfs:domain pmkr:X3DProceduralNode ;
                         rdfs:range  pmkr:X3DProceduralNode .
```

**Example SPARQL query** --- find all rules that reference the summation symbol:
```sparql
SELECT ?rule ?pattern WHERE {
  ?rule a pmkr:ProceduralRulesNode ;
        pmkr:symbolRefs "char:U+2211" ;
        pmkr:pattern ?pattern .
}
```

---

## 13. Conformance

### 13.1 PM-KR Conformance Levels in X3D Context

Aligning PM-KR Spec §7 conformance levels with X3D component levels:

| PM-KR Level | X3D ProceduralMemory Level | Requirements |
|-------------|---------------------------|-------------|
| **A (Core)** | Level 1 | Layer model, canonicality, reference preservation, deterministic reconstruction |
| **B (Sovereign Runtime)** | Level 2 | Level A + RPN program execution, dual-client rendering, fail-fast on missing references |
| **C (Auditable Production)** | Level 3 | Level B + provenance tracking, compression metrics, defeasible logic, conformance test artifacts |

### 13.2 Conformance Tests

An implementation claiming ProceduralMemory component support MUST pass tests for:

1. **Canonical deduplication**: Two nodes with the same `canonicalId` and `version` MUST be treated as the same concept.
2. **Reference resolution**: All CanonicalReference `targetId` values resolve to existing nodes. Broken references cause explicit errors.
3. **Deterministic reconstruction**: Given a scene with RPN programs, execution produces identical output across runs.
4. **Layer constraint enforcement**: `formProgram` is required for form nodes, `meaningProgram` for meaning/rules/meta_rules nodes.
5. **Dual-client consistency**: Human and machine textures in DualClientTexture derive from the same canonical source.
6. **Symlink preservation**: Implementations do not inline referenced content during save/load cycles.

### 13.3 PROTO Fallback

For X3D browsers that do not natively support the ProceduralMemory component, a PROTO-based reference implementation SHOULD be provided. This allows immediate experimentation, though PROTO cannot enforce all behavioral rules (particularly fail-fast reference resolution and layer constraint enforcement).

---

## 14. Relationship to Existing X3D Components

### 14.1 Core Component

PM-KR nodes extend X3DNode and use Metadata nodes extensively. All PM-KR nodes carry the standard `metadata` field. MetadataFloat is used for embedding vectors, MetadataString for provenance and classification, MetadataSet for structured metadata trees.

### 14.2 Grouping Component

GalaxyGroup extends X3DGroupingNode, using Transform for spatial positioning of knowledge nodes. The existing Group, Transform, and Switch nodes work with PM-KR nodes as children.

### 14.3 Shape and Texturing Components

ProceduralFormNode carries an optional `appearance` field (X3DAppearanceNode) for human client rendering. DualClientTexture uses X3DTexture2DNode for both human and machine textures. MultiTexture from Texturing component Level 2 enables simultaneous dual-texture rendering.

### 14.4 Navigation Component

Viewpoint and NavigationInfo from the standard Navigation component work with KnowledgeNavigation for spatial scene navigation. KnowledgeFrustum extends the frustum culling concept from rendering to knowledge retrieval.

### 14.5 Scripting Component

During prototyping, Script nodes MAY implement PM-KR behavioral rules (reference resolution, layer validation) in JavaScript or Java. For production, these behaviors SHOULD be implemented natively in the browser's component support.

### 14.6 Humanoid Animation Component (HAnim)

AgentEntity's `body` field MAY reference an HAnim HumanoidRoot node for visual agent embodiment. This enables the TRM avatar to have a visible, animated body in the scene.

### 14.7 Sound Component

Audio Galaxy entries (ProceduralFormNode with `galaxy="Audio"`) carry RPN programs that produce audio waveforms. The X3D 4.0 Sound component (BufferAudioSource, SpatialSound) provides the rendering pipeline for these audio forms.

---

## 15. Examples

### 15.1 Minimal Form Node Scene

```xml
<?xml version="1.0" encoding="UTF-8"?>
<X3D version='4.0' profile='ProceduralMemoryInterchange'>
  <head>
    <component name='ProceduralMemory' level='1'/>
    <meta name='title' content='SingleFormNode.x3d'/>
  </head>
  <Scene>
    <Transform translation='0 0 0'>
      <ProceduralFormNode DEF='CHAR_PLUS'
          canonicalId='char:U+002B'
          galaxy='Character'
          domain='arithmetic'
          confidence='1.0'>
        <RPNProgram containerField='formProgram'
            opcodes='16 32 MOVE 48 32 LINE 32 16 MOVE 32 48 LINE STROKE'
            tier='standard'/>
        <MetadataFloat containerField='embedding'
            name='embedding16'
            value='0.5 0.3 -0.2 0.8 0.1 -0.4 0.6 0.2 -0.1 0.7 0.4 -0.3 0.5 0.1 -0.2 0.9'/>
      </ProceduralFormNode>
    </Transform>
  </Scene>
</X3D>
```

### 15.2 Cross-Layer Reference Scene

```xml
<?xml version="1.0" encoding="UTF-8"?>
<X3D version='4.0' profile='ProceduralMemoryInterchange'>
  <head>
    <component name='ProceduralMemory' level='2'/>
    <meta name='title' content='CrossLayerRefs.x3d'/>
  </head>
  <Scene>
    <!-- Galaxy: Character (Layer 1 Form nodes) -->
    <GalaxyGroup galaxyName='Character' galaxyType='default'>
      <Transform translation='0 0 0'>
        <ProceduralFormNode DEF='CHAR_D' canonicalId='char:U+0064' galaxy='Character'>
          <RPNProgram containerField='formProgram' opcodes='8 8 MOVE 8 56 LINE ARC 24 56 8 8 CLOSE FILL'/>
        </ProceduralFormNode>
      </Transform>
      <!-- ... more character form nodes ... -->
    </GalaxyGroup>

    <!-- Galaxy: Word (Layer 2 Meaning nodes) -->
    <GalaxyGroup galaxyName='Word' galaxyType='default'>
      <Transform translation='10 0 0'>
        <ProceduralMeaningNode DEF='WORD_DOG'
            canonicalId='word:en:dog'
            galaxy='Word'
            domain='animal'
            charRefs='"char:U+0064" "char:U+006F" "char:U+0067"'>
          <RPNProgram containerField='meaningProgram'
              opcodes='LOAD_CONTEXT RECALL_ANIMAL MATCH_DOMESTIC STORE'/>
          <!-- No formProgram duplication: character glyphs referenced via charRefs -->
        </ProceduralMeaningNode>
      </Transform>
    </GalaxyGroup>

    <!-- Galaxy: Grammar (Layer 3 Rules nodes) -->
    <GalaxyGroup galaxyName='Grammar' galaxyType='default'>
      <Transform translation='20 0 0'>
        <ProceduralRulesNode DEF='RULE_PLURAL_S'
            canonicalId='rule:en:plural_regular_s'
            galaxy='Grammar'
            domain='morphology'
            pattern='NOUN + s'
            language='en'
            ruleStrength='0'
            wordRefs='"word:en:dog"'
            symbolRefs='"char:U+0073"'>
          <RPNProgram containerField='meaningProgram'
              opcodes='RECALL_WORD DUP LENGTH 1 SUB RECALL_CHAR_S APPEND STORE'/>
        </ProceduralRulesNode>
      </Transform>
    </GalaxyGroup>
  </Scene>
</X3D>
```

### 15.3 Agent Entity Scene

```xml
<?xml version="1.0" encoding="UTF-8"?>
<X3D version='4.0' profile='ProceduralMemoryInterchange'>
  <head>
    <component name='ProceduralMemory' level='2'/>
    <component name='KnowledgeNavigation' level='3'/>
    <component name='H-Anim' level='1'/>
    <meta name='title' content='AgentInHouse.x3d'/>
  </head>
  <Scene>
    <!-- The House (Memory Palace) -->
    <AgentMemoryPalace DEF='MY_HOUSE' palaceId='house:default' entryCount='247889' galaxyCount='19'>
      <!-- Library Room -->
      <Transform translation='0 0 0'>
        <GalaxyGroup galaxyName='Math' galaxyType='default'>
          <!-- ... math knowledge nodes ... -->
        </GalaxyGroup>
      </Transform>
      <!-- Knowledge Garden -->
      <Transform translation='50 0 0'>
        <GalaxyGroup galaxyName='Reality' galaxyType='default'>
          <!-- ... reality knowledge nodes ... -->
        </GalaxyGroup>
      </Transform>
    </AgentMemoryPalace>

    <!-- The Agent (TRM Avatar) -->
    <AgentEntity DEF='TRM_AVATAR'
        agentId='trm:primary'
        position='5 0 5'
        parameterCount='7000000'
        maxRecursionSteps='9'
        convergenceThreshold='0.0001'
        isActive='true'>
      <AgentMemoryPalace USE='MY_HOUSE' containerField='memoryPalace'/>
      <!-- Agent's internal Galaxy (working memory) -->
      <GalaxyGroup containerField='internalState' galaxyName='WorkingMemory' galaxyType='meta'>
        <!-- Active reasoning state, loaded at runtime -->
      </GalaxyGroup>
      <!-- Agent's swarm workers -->
      <AgentSwarm workerCount='9'
          specialistNames='"math" "grammar" "visual" "chat" "physics" "logic" "spatial" "temporal" "meta"'
          convergenceMode='one_mind'/>
    </AgentEntity>

    <!-- Spatial index for knowledge queries -->
    <SpatialKnowledgeIndex DEF='KNOWLEDGE_INDEX' indexType='morton' maxDepth='8'/>

    <!-- Knowledge pathfinder -->
    <KnowledgePathfinder DEF='LED_PATHFINDER' alpha='0.35' beta='0.65' maxPathLength='128'>
      <SpatialKnowledgeIndex USE='KNOWLEDGE_INDEX' containerField='index'/>
    </KnowledgePathfinder>
  </Scene>
</X3D>
```

---

## Appendix A: PM-KR Invariants Mapped to X3D Conformance

| PM-KR Invariant | X3D Enforcement | Test |
|-----------------|-----------------|------|
| Canonicality (§5.1) | `canonicalId` + `version` uniqueness | No two nodes share same ID+version |
| Reference Preservation (§5.2) | CanonicalReference resolution | References resolve, content not inlined |
| Deterministic Reconstruction (§5.3) | RPNProgram determinism | Same input → same output |
| Dual-Client Equivalence (§5.4) | DualClientTexture identity | Both textures from same source |
| Sovereign Boundary (§5.5) | Implementation concern (not scene format) | N/A for interchange |
| Auditability (§5.6) | Metadata provenance fields | Provenance and timestamp present |

---

## Appendix B: Opcode Quick Reference

Subset of RPN opcodes relevant to X3D PM-KR scenes:

| Opcode | Hex | Tier | Description |
|--------|-----|------|-------------|
| PUSH | 0x01 | Lite | Push constant to stack |
| ADD | 0x02 | Lite | a + b |
| SUB | 0x03 | Lite | a - b |
| MUL | 0x04 | Lite | a * b |
| DIV | 0x05 | Lite | a / b |
| DUP | 0x10 | Lite | Duplicate top of stack |
| SWAP | 0x11 | Lite | Swap top two elements |
| STORE | 0x20 | Lite | Store to register |
| RECALL | 0x21 | Lite | Recall from register |
| MOVE | 0x64 | Standard | Move pen to (x, y) |
| LINE | 0x65 | Standard | Draw line to (x, y) |
| ARC | 0x66 | Standard | Draw arc |
| BEZIER | 0x67 | Standard | Draw cubic Bezier |
| CLOSE | 0x69 | Standard | Close path |
| STROKE | 0x6A | Standard | Render stroke |
| FILL | 0x6B | Standard | Render fill |
| BRANCH | 0x80 | Standard | Conditional branch |
| LOOP | 0x81 | Standard | Loop construct |
| DOT_PRODUCT | 0x50 | Standard | Vector dot product |
| TADD | 0x70 | Extended | Ternary addition |
| TMUL | 0x71 | Extended | Ternary multiplication |
| TNOT | 0x72 | Extended | Ternary negation |
| TPACK | 0x75 | Extended | Pack (D, d) trit pair |

See RPN Domain Opcode Registry (docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md) for the complete registry.

---

## Appendix C: Document History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-03-26 | Initial draft. ProceduralMemory and KnowledgeNavigation components defined. ProceduralMemoryInterchange profile defined. |

---

**End of Document**

*This specification is a working draft of the PM-KR Community Group (W3C). It is intended for review by the Web3D Consortium and the broader standards community. Feedback should be directed to the PM-KR Community Group mailing list.*
