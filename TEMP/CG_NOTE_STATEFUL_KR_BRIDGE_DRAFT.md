# Stateful Knowledge Representation Bridge

**W3C Semantic Agent Communication Community Group — CG Note (Draft)**

**Status:** Non-Normative
**Version:** 0.2 (K3D Vocabulary Aligned)
**Date:** December 2025
**Editors:** [CG Contributors]

---

## Abstract

This document describes a non-normative bridge enabling interoperable communication between autonomous agents and stateful knowledge representation (KR) systems. The bridge connects AgentIDL interaction semantics with graph-based, spatial/3D, and versioned knowledge stores while preserving identity, provenance, and semantic traceability.

Agents do not share global memory. Instead, they exchange artifacts via published resources with dereferenceable URIs, append-only semantic logs, and validated snapshots. This architecture is informed by the **Dual Client Reality** principle: the same data structures serve both human visualization and AI reasoning.

---

## 1. Scope

### 1.1 In Scope

- Minimal model for agent identity, capabilities, intents, and delegations
- Stateful KR resources: graph stores, spatial/3D coordinate systems, versioned repositories
- Semantic logs with provenance and replay capability
- Procedural resource encoding (RPN programs as canonical representation)
- JSON-LD contexts, RDF/Turtle serialization, SHACL shapes for validation
- Hypermedia discovery via well-known entry URIs
- Lightweight conformance guidance for implementers

### 1.2 Out of Scope

- Transport protocols (HTTP, WebSocket, MQTT — left to implementers)
- Internal agent reasoning or planning algorithms
- Blockchain or distributed ledger implementations
- Economic models, tokenomics, or payment systems
- Domain-specific ontologies (medical, legal, industrial)

---

## 2. Minimal Model

The bridge defines core classes inspired by the **Three-Brain System** pattern (reasoning engine + active memory + persistent storage):

| Class | Description | Architectural Pattern |
|-------|-------------|----------------------|
| `sacg:Agent` | Autonomous entity with identity and capabilities | Reasoning layer |
| `sacg:Identity` | Agent identifier; optionally DID/VC-bound | — |
| `sacg:Capability` | Declared action an agent can perform | RPN opcode capabilities |
| `sacg:Intent` | Requested action from one agent to another | Action buffer |
| `sacg:ExecutionRecord` | Immutable record of an executed intent | Consolidation log entry |
| `sacg:SemanticLog` | Append-only ledger of execution records | Consolidation log |
| `sacg:Resource` | Stateful KR artifact (graph, spatial, versioned) | Knowledge Node |
| `sacg:Snapshot` | Immutable point-in-time state of a resource | Persistent artifact |
| `sacg:Version` | Named snapshot in a versioned resource history | — |
| `sacg:Diff` | Delta between two snapshots | — |
| `sacg:Merge` | Reconciliation of divergent version branches | — |
| `sacg:Provenance` | Origin and transformation history metadata | Node provenance facet |
| `sacg:AccessControl` | Permission constraints on resource operations | — |

### 2.1 Class Relationships

```
Agent --has--> Identity
Agent --declares--> Capability[]
Agent --issues--> Intent
Agent --grants--> Delegation
Intent --produces--> ExecutionRecord
ExecutionRecord --appended-to--> SemanticLog
ExecutionRecord --modifies--> Resource
Resource --has--> Snapshot[]
Resource --encoded-via--> ProceduralProgram (RPN)
Snapshot --tracked-by--> Version
Version --compared-via--> Diff
Diff --reconciled-by--> Merge
Resource --annotated-with--> Provenance
Resource --governed-by--> AccessControl
```

---

## 3. Knowledge Systems

### 3.1 No Shared Global Memory

Agents operate independently. There is no shared mutable state. Communication occurs exclusively through:

1. **Published Resources** — Agents publish snapshots at dereferenceable URIs
2. **Intent Exchange** — Agents request actions via structured intents
3. **Semantic Logs** — Append-only records provide audit trail

This follows a **tiered memory hierarchy** pattern: active (volatile) → persistent (durable).

### 3.2 Supported KR Paradigms

| Paradigm | Description | Serialization | Example |
|----------|-------------|---------------|---------|
| **Graph** | RDF triples, property graphs, knowledge graphs | Turtle, JSON-LD, TriG | Neo4j, Jena, RDFLib |
| **Spatial/3D** | Euclidean coordinates, meshes, embeddings | glTF 2.0 + `extras` | K3D Node, USD |
| **Versioned** | Git-style history with branches, commits, merges | JSON-LD | DVC, LakeFS |
| **Procedural** | Executable RPN programs generating resources | RPN bytecode | Drawing systems |

### 3.3 Dual Client Reality

Resources SHOULD support the **Dual Client Reality** principle:

- **Human perception**: Visual geometry, textures, 3D spatial layout
- **AI perception**: Semantic embeddings, graph topology, executable programs

Both clients operate on the **same data**; only rendering differs.

### 3.4 Resource Exchange Pattern

```
┌─────────┐    Intent     ┌─────────┐
│ Agent A │──────────────>│ Agent B │
└────┬────┘               └────┬────┘
     │                         │
     │  publish(snapshot)      │  dereference(URI)
     v                         v
┌─────────────────────────────────────┐
│         Published Resource          │
│  (graph | spatial | versioned)      │
│  URI: https://example.org/res/123   │
│                                     │
│  Dual representation:               │
│  - Visual: glTF geometry            │
│  - Semantic: extras embeddings      │
└─────────────────────────────────────┘
```

---

## 4. Resource Structure (Spatial Node Pattern)

### 4.1 Minimal Resource Facets

Bridge-compatible resources SHOULD include:

| Facet | Description | Required |
|-------|-------------|----------|
| `position` | Spatial coordinates (x, y, z) | For spatial resources |
| `embedding` | Vector representation (variable dim) | RECOMMENDED |
| `semantic` | RDF subject/predicate/object | REQUIRED |
| `provenance` | Origin, timestamps, author | REQUIRED |
| `procedural` | RPN program generating the resource | OPTIONAL |

### 4.2 Matryoshka Embeddings (LOD)

Resources MAY use **variable-dimension embeddings** for Level of Detail:

| Dimension | Purpose | Use Case |
|-----------|---------|----------|
| 64D | Coarse plausibility | Quick similarity check |
| 128D | Structural similarity | Clustering |
| 512D | Balanced fidelity | Default operations |
| 2048D | Maximum detail | Fine-grained reasoning |

Embeddings at lower dimensions SHOULD be prefixes of higher-dimension embeddings (Matryoshka property).

### 4.3 Procedural Encoding

Resources MAY be stored as **procedural programs** rather than static data:

```json
{
  "@type": "sacg:ProceduralResource",
  "procedural_program": "0.5 0.5 0.3 CIRCLE 0.2 FILL",
  "regenerable": true,
  "compression_ratio": 12.5
}
```

This enables:
- **Compression**: 12–80× vs static data
- **Resolution independence**: Render at any LOD
- **Dual client**: Same program renders visual geometry AND generates embeddings

---

## 5. Interfaces and Serialization

### 5.1 JSON-LD Context

All bridge artifacts use a shared JSON-LD context:

```json
{
  "@context": {
    "sacg": "https://w3id.org/sacg#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "Agent": "sacg:Agent",
    "Identity": "sacg:Identity",
    "Capability": "sacg:Capability",
    "Intent": "sacg:Intent",
    "ExecutionRecord": "sacg:ExecutionRecord",
    "SemanticLog": "sacg:SemanticLog",
    "Resource": "sacg:Resource",
    "Snapshot": "sacg:Snapshot",
    "Provenance": "sacg:Provenance",
    "ProceduralProgram": "sacg:ProceduralProgram",
    "embedding": {
      "@id": "sacg:embedding",
      "@type": "@json"
    },
    "position": {
      "@id": "sacg:position",
      "@type": "@json"
    },
    "issuedBy": { "@id": "sacg:issuedBy", "@type": "@id" },
    "targetAgent": { "@id": "sacg:targetAgent", "@type": "@id" },
    "targetResource": { "@id": "sacg:targetResource", "@type": "@id" },
    "timestamp": { "@id": "sacg:timestamp", "@type": "xsd:dateTime" },
    "action": "sacg:action",
    "coordinates": "sacg:coordinates",
    "matryoshkaDim": "sacg:matryoshkaDim"
  }
}
```

### 5.2 glTF Serialization (Spatial Resources)

Spatial resources SHOULD serialize as glTF 2.0 with `extras` extension:

```json
{
  "nodes": [{
    "name": "KnowledgeNode_001",
    "translation": [1.5, 2.0, -3.0],
    "extras": {
      "bridge": {
        "embedding": {
          "dim": 512,
          "values_ref": "bufferView:3",
          "matryoshka": true
        },
        "semantic": {
          "subject": "https://example.org/concept/rotation",
          "predicate": "rdf:type",
          "object": "https://example.org/ontology#Transformation"
        },
        "provenance": {
          "source": "https://example.org/agents/reasoner-a",
          "timestamp": "2025-12-11T14:00:00Z"
        },
        "procedural": {
          "rpn_program": "ROTATE_90_CW APPLY",
          "regenerable": true
        }
      }
    }
  }]
}
```

### 5.3 Hypermedia Discovery

Agents discover bridge endpoints via well-known URI:

```
GET /.well-known/sacg-bridge
```

Response:
```json
{
  "@context": "https://w3id.org/sacg/context.jsonld",
  "@type": "sacg:BridgeEntryPoint",
  "agentDirectory": "https://example.org/agents/",
  "resourceCatalog": "https://example.org/resources/",
  "semanticLog": "https://example.org/log/",
  "supportedParadigms": ["graph", "spatial", "versioned", "procedural"],
  "matryoshkaDimensions": [64, 128, 512, 2048]
}
```

---

## 6. Semantic Logs (Consolidation Pattern)

### 6.1 Properties

Semantic logs follow a **consolidation protocol** pattern:

- **Append-only** — Records cannot be modified or deleted
- **Provenance-linked** — Each record references issuing agent and source intent
- **Replayable** — Log can reconstruct resource state at any point
- **ACID-compliant** — Atomic writes, consistency, isolation, durability

### 6.2 Log Entry Structure

```json
{
  "@context": "https://w3id.org/sacg/context.jsonld",
  "@type": "ExecutionRecord",
  "@id": "https://example.org/log/entry/42",
  "timestamp": "2025-12-11T14:30:00Z",
  "issuedBy": "https://example.org/agents/agent-a",
  "intent": {
    "@type": "Intent",
    "action": "UpdateResource",
    "targetResource": "https://example.org/resources/graph-001"
  },
  "result": "Success",
  "stateChange": {
    "before": "https://example.org/resources/graph-001/snapshot/v2",
    "after": "https://example.org/resources/graph-001/snapshot/v3"
  },
  "provenance": {
    "derivedFrom": "https://example.org/resources/source-data",
    "generatedBy": "https://example.org/agents/agent-a",
    "proceduralProgram": "TRANSFORM NORMALIZE STORE"
  },
  "metrics": {
    "latencyMs": 8.3,
    "nodesAffected": 1247
  }
}
```

### 6.3 Consolidation Triggers

Implementations MAY consolidate semantic logs based on:

| Trigger | Condition | Action |
|---------|-----------|--------|
| Time-based | Every N hours | Archive log entries |
| Event-based | N intent executions | Snapshot current state |
| Memory pressure | Resource utilization > threshold | Flush to persistent storage |

---

## 7. Interoperability Test Corpus

### 7.1 Fixture Components

| Component | Description | Format |
|-----------|-------------|--------|
| `agentidl-spec.idl` | AgentIDL interface definitions | WebIDL |
| `sacg-shapes.ttl` | SHACL shapes for all core classes | Turtle |
| `sample-log.jsonld` | 10-entry semantic log with varied intents | JSON-LD |
| `graph-snapshot.trig` | Sample graph resource snapshot | TriG |
| `spatial-snapshot.glb` | Sample 3D resource with extensions | glTF Binary |
| `procedural-resource.json` | RPN-encoded procedural resource | JSON-LD |
| `versioned-history.json` | Git-style version tree | JSON-LD |

### 7.2 Validation Procedure

1. Parse `sacg-shapes.ttl` into SHACL processor
2. Load test resource (graph, spatial, procedural, or versioned)
3. Validate resource against shapes
4. Replay `sample-log.jsonld` entries
5. Verify final state matches expected snapshot
6. For procedural resources: Execute RPN program, compare output

---

## 8. Conformance Guidance

This is a non-normative CG Note. The following guidance helps implementers achieve interoperability:

### 8.1 Bridge-Compatible Implementation

An implementation is considered **bridge-compatible** if it:

1. **Emits SHACL-valid resources** — All published resources pass validation against `sacg-shapes.ttl`
2. **Exposes entry URI** — Provides `/.well-known/sacg-bridge` discovery endpoint
3. **Supports semantic log format** — Accepts and emits `ExecutionRecord` entries per Section 6
4. **Uses dereferenceable URIs** — All resource and agent identifiers resolve to representations

### 8.2 Dual Client Conformance (Recommended)

Implementations SHOULD support dual client reality:

1. **Visual representation** — Resources renderable as 3D geometry or 2D visualization
2. **Semantic representation** — Resources include embeddings and graph metadata
3. **Same source** — Both representations derive from the same canonical data

### 8.3 Optional Extensions

- DID/VC identity binding
- SPARQL endpoint for graph resources
- GeoSPARQL for spatial queries
- Git-compatible versioning operations
- RPN execution engine for procedural resources
- Matryoshka embedding support (multi-resolution)

---

## 9. Non-Goals

The following topics are explicitly out of scope:

- **Transport protocols** — Use HTTP, WebSocket, MQTT, or any suitable protocol
- **Internal reasoning** — Agent planning, inference engines, internal architecture
- **Blockchains** — Distributed consensus mechanisms
- **Economic systems** — Tokens, payments, incentive structures
- **Domain ontologies** — Healthcare, finance, manufacturing vocabularies

---

## Appendix A: SHACL Shapes (Excerpt)

```turtle
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix sacg: <https://w3id.org/sacg#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

sacg:AgentShape a sh:NodeShape ;
    sh:targetClass sacg:Agent ;
    sh:property [
        sh:path sacg:hasIdentity ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class sacg:Identity ;
    ] ;
    sh:property [
        sh:path sacg:declaresCapability ;
        sh:class sacg:Capability ;
    ] .

sacg:ExecutionRecordShape a sh:NodeShape ;
    sh:targetClass sacg:ExecutionRecord ;
    sh:property [
        sh:path sacg:timestamp ;
        sh:datatype xsd:dateTime ;
        sh:minCount 1 ;
    ] ;
    sh:property [
        sh:path sacg:issuedBy ;
        sh:class sacg:Agent ;
        sh:minCount 1 ;
    ] ;
    sh:property [
        sh:path sacg:intent ;
        sh:class sacg:Intent ;
        sh:minCount 1 ;
    ] .

sacg:ResourceShape a sh:NodeShape ;
    sh:targetClass sacg:Resource ;
    sh:property [
        sh:path sacg:hasSnapshot ;
        sh:class sacg:Snapshot ;
    ] ;
    sh:property [
        sh:path sacg:hasProvenance ;
        sh:class sacg:Provenance ;
        sh:minCount 1 ;
    ] .

sacg:SpatialResourceShape a sh:NodeShape ;
    sh:targetClass sacg:SpatialResource ;
    sh:property [
        sh:path sacg:position ;
        sh:datatype xsd:string ;
        sh:pattern "^\\{.*\\}$" ;
    ] ;
    sh:property [
        sh:path sacg:embedding ;
        sh:minCount 0 ;
    ] ;
    sh:property [
        sh:path sacg:matryoshkaDim ;
        sh:in ( 64 128 512 2048 ) ;
    ] .

sacg:ProceduralResourceShape a sh:NodeShape ;
    sh:targetClass sacg:ProceduralResource ;
    sh:property [
        sh:path sacg:rpnProgram ;
        sh:datatype xsd:string ;
        sh:minCount 1 ;
    ] ;
    sh:property [
        sh:path sacg:regenerable ;
        sh:datatype xsd:boolean ;
    ] .
```

---

## Appendix B: Complete JSON-LD Examples

### B.1 Agent Declaration

```json
{
  "@context": "https://w3id.org/sacg/context.jsonld",
  "@type": "Agent",
  "@id": "https://example.org/agents/spatial-processor",
  "hasIdentity": {
    "@type": "Identity",
    "identifier": "did:example:spatial-processor-001",
    "verificationMethod": "did:example:spatial-processor-001#key-1"
  },
  "declaresCapability": [
    {
      "@type": "Capability",
      "action": "TransformSpatialResource",
      "inputType": "sacg:SpatialResource",
      "outputType": "sacg:SpatialResource",
      "rpnOpcodes": ["ROTATE", "SCALE", "TRANSLATE", "CROP"]
    },
    {
      "@type": "Capability",
      "action": "QueryGraph",
      "inputType": "xsd:string",
      "outputType": "sacg:Snapshot"
    }
  ],
  "memoryModel": {
    "activeCapacity": "200MB",
    "persistentStorage": "glTF-GLB",
    "consolidationProtocol": "append-only-log"
  }
}
```

### B.2 Spatial Resource with Extensions

```json
{
  "@context": [
    "https://w3id.org/sacg/context.jsonld",
    {
      "geo": "http://www.opengis.net/ont/geosparql#",
      "Point3D": "sacg:Point3D",
      "x": "sacg:x",
      "y": "sacg:y",
      "z": "sacg:z"
    }
  ],
  "@type": "SpatialResource",
  "@id": "https://example.org/resources/knowledge-node-001",
  "position": {
    "x": 1.5,
    "y": 2.0,
    "z": -3.0
  },
  "embedding": {
    "matryoshkaDim": 512,
    "values": [0.1, 0.5, -0.3, "...truncated..."],
    "regenerableFrom": "procedural"
  },
  "semantic": {
    "subject": "https://example.org/concept/rotation",
    "predicate": "rdf:type",
    "object": "https://example.org/ontology#Transformation"
  },
  "procedural": {
    "@type": "ProceduralProgram",
    "rpnProgram": "INPUT DUP ROTATE_90_CW NORMALIZE OUTPUT",
    "regenerable": true,
    "compressionRatio": 24.5
  },
  "hasSnapshot": {
    "@type": "Snapshot",
    "@id": "https://example.org/resources/knowledge-node-001/snapshot/v1",
    "timestamp": "2025-12-11T10:00:00Z",
    "format": "glTF-GLB",
    "byteSize": 2048
  },
  "hasProvenance": {
    "@type": "Provenance",
    "generatedBy": "https://example.org/agents/reasoner-a",
    "generatedAt": "2025-12-11T10:00:00Z",
    "derivedFrom": "https://example.org/datasets/task-001"
  }
}
```

### B.3 Intent and Execution Record

```json
{
  "@context": "https://w3id.org/sacg/context.jsonld",
  "@type": "Intent",
  "@id": "https://example.org/intents/req-456",
  "issuedBy": "https://example.org/agents/coordinator",
  "targetAgent": "https://example.org/agents/spatial-processor",
  "action": "TransformSpatialResource",
  "targetResource": "https://example.org/resources/knowledge-node-001",
  "parameters": {
    "operation": "Rotate",
    "rpnProgram": "ROTATE_90_CW",
    "axis": "Z",
    "angleDegrees": 90
  },
  "requestedLOD": {
    "matryoshkaDim": 512,
    "purpose": "transformation"
  }
}
```

Resulting execution record:

```json
{
  "@context": "https://w3id.org/sacg/context.jsonld",
  "@type": "ExecutionRecord",
  "@id": "https://example.org/log/entry/789",
  "timestamp": "2025-12-11T10:05:00Z",
  "issuedBy": "https://example.org/agents/spatial-processor",
  "intent": "https://example.org/intents/req-456",
  "result": "Success",
  "stateChange": {
    "before": "https://example.org/resources/knowledge-node-001/snapshot/v1",
    "after": "https://example.org/resources/knowledge-node-001/snapshot/v2"
  },
  "metrics": {
    "latencyMs": 4.2,
    "kernelsExecuted": 3
  }
}
```

---

## Appendix C: RDF/Turtle Example

```turtle
@prefix sacg: <https://w3id.org/sacg#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ex: <https://example.org/> .

ex:agents/graph-analyst a sacg:Agent ;
    sacg:hasIdentity [
        a sacg:Identity ;
        sacg:identifier "did:example:graph-analyst-001"
    ] ;
    sacg:declaresCapability [
        a sacg:Capability ;
        sacg:action "QueryGraph" ;
        sacg:inputType xsd:string ;
        sacg:outputType sacg:Snapshot
    ] .

ex:resources/knowledge-node-001 a sacg:SpatialResource ;
    sacg:position "{\"x\": 1.5, \"y\": 2.0, \"z\": -3.0}" ;
    sacg:matryoshkaDim 512 ;
    sacg:rpnProgram "ROTATE_90_CW NORMALIZE" ;
    sacg:regenerable true ;
    sacg:hasSnapshot ex:resources/knowledge-node-001/snapshot/v3 ;
    sacg:hasProvenance [
        a sacg:Provenance ;
        sacg:generatedBy ex:agents/spatial-processor ;
        sacg:generatedAt "2025-12-10T08:00:00Z"^^xsd:dateTime
    ] .

ex:log/entry/100 a sacg:ExecutionRecord ;
    sacg:timestamp "2025-12-11T14:00:00Z"^^xsd:dateTime ;
    sacg:issuedBy ex:agents/graph-analyst ;
    sacg:intent [
        a sacg:Intent ;
        sacg:action "QueryGraph" ;
        sacg:targetResource ex:resources/knowledge-node-001
    ] ;
    sacg:result "Success" ;
    sacg:latencyMs 4.2 .
```

---

## References

### W3C Standards
- [W3C DID Core](https://www.w3.org/TR/did-core/)
- [W3C Verifiable Credentials](https://www.w3.org/TR/vc-data-model/)
- [SHACL Specification](https://www.w3.org/TR/shacl/)
- [JSON-LD 1.1](https://www.w3.org/TR/json-ld11/)
- [GeoSPARQL](https://www.ogc.org/standards/geosparql)
- [PROV-O Ontology](https://www.w3.org/TR/prov-o/)

### Implementation References
- [glTF 2.0 Specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
- [WebXR Device API](https://www.w3.org/TR/webxr/)

---

## Acknowledgments

This CG Note draft incorporates architectural patterns including:

- **Dual Client Reality**: Same data serves human visualization and AI reasoning
- **Three-Brain System**: Separation of reasoning, active memory, and persistent storage
- **Procedural Foundation**: Executable RPN programs as canonical representation
- **Consolidation Protocol**: ACID-compliant memory management
- **Matryoshka Embeddings**: Variable-dimension LOD for efficient processing

---

**End of CG Note Draft (v0.2)**
