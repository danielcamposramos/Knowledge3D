# K3D Node: Formal Specification

**Date:** February 10, 2026
**For:** Scientific Paper (W3C/Academic Publication)
**Purpose:** Formal definition of Knowledge3D nodes for interoperability and conformance

---

## 1. Abstract Definition

A **K3D node** is a structured data record representing a procedurally-encoded knowledge atom within the Knowledge3D spatial memory architecture. Each node:
- Has **stable identity** (unique ID/IRI)
- Contains **procedural payload** (RPN program for GPU-native execution)
- Includes **semantic metadata** (type, domain, provenance, confidence)
- Supports **spatial addressing** (3D coordinates in Galaxy Universe)
- Maintains **temporal provenance** (creation time, source, verification)
- Enables **multi-modal linking** (symlinks to other galaxies)

---

## 2. Required Fields (Minimal K3D Node)

### Core Schema (JSON/JSONLD)

```json
{
  "id": "string",              // REQUIRED: Unique identifier (IRI-compatible)
  "type": "string",            // REQUIRED: Node type (primitive, transformation, discovered_pattern, etc.)
  "rpn_program": "string",     // REQUIRED: RPN procedural definition (executable on Cranium PTX)
  "domain": "string",          // REQUIRED: Galaxy domain (drawing, grammar, math, reality, etc.)
  "provenance": {              // REQUIRED: Origin metadata
    "source": "string",        // Source type (canonical, discovered, derived, ingested)
    "timestamp": "ISO8601",    // Creation time
    "specialist": "string",    // Creating specialist (visual, math, physics, etc.)
    "confidence": float        // Confidence score [0.0, 1.0]
  },
  "payload": object,           // OPTIONAL: Additional type-specific data
  "links": [                   // OPTIONAL: Cross-galaxy symlinks
    {
      "target_id": "string",
      "target_galaxy": "string",
      "relation": "string"
    }
  ],
  "spatial": {                 // OPTIONAL: 3D coordinates in Galaxy Universe
    "x": float,
    "y": float,
    "z": float,
    "galaxy": "string"
  }
}
```

### Field Descriptions

#### id (REQUIRED)
- **Type:** `string`
- **Format:** IRI-compatible (can be prefixed with `k3d:`, `urn:k3d:`, or HTTP URI)
- **Uniqueness:** Globally unique within Knowledgeverse instance
- **Stability:** Immutable after creation (content-addressed or UUID)
- **Examples:**
  - `"line_primitive_01"` (canonical primitive)
  - `"discovered_pattern_arc_task_007_gen_3"` (discovered pattern)
  - `"k3d:drawing/transformations/rotation_90"` (IRI form)

#### type (REQUIRED)
- **Type:** `string`
- **Purpose:** Categorizes node for routing and interpretation
- **Standard Types:**
  - `"foundational_primitive"`: Core procedural building block (LINE, CIRCLE, etc.)
  - `"transformation"`: RPN rule for transforming grids/objects
  - `"discovered_pattern"`: Pattern learned from training data
  - `"canonical_rule"`: Grammar transformation rule
  - `"shape"`: Composite visual shape
  - `"scene"`: Collection of shapes with layout
  - `"derived_symbol"`: Mathematical symbol with computation template
- **Extensibility:** Custom types allowed, prefixed with `"custom:"`

#### rpn_program (REQUIRED)
- **Type:** `string`
- **Format:** Space-separated RPN tokens (Reverse Polish Notation)
- **Execution:** Runs on Cranium PTX kernels (GPU-native)
- **Sovereignty:** Must contain ONLY PTX-executable operations (no external library calls)
- **Examples:**
  - `"0 0 10 0 LINE"` (line from (0,0) to (10,0))
  - `"DUP ROT_90 OVERLAY"` (duplicate, rotate 90°, overlay)
  - `"2 SQRT 3.14159 MUL"` (√2 × π)
- **Validation:** Parser checks for valid PTX ops before insertion

#### domain (REQUIRED)
- **Type:** `string`
- **Purpose:** Identifies originating galaxy (namespace)
- **Standard Domains:**
  - `"drawing"`: Visual primitives, shapes, scenes
  - `"grammar"`: Language transformation rules
  - `"math"`: Mathematical symbols and operations
  - `"reality"`: Physics/chemistry/biology procedural systems
  - `"character"`: Unicode glyphs with font metadata
  - `"audio"`: Temporal sound patterns
  - `"3d_objects"`: Volumetric shapes with collision boundaries

#### provenance (REQUIRED)
Object containing origin metadata:

**provenance.source:**
- `"canonical"`: Hardcoded foundational knowledge (e.g., LINE primitive)
- `"discovered"`: Learned from training data (e.g., ARC pattern)
- `"derived"`: Computed from existing nodes (e.g., symlink composition)
- `"ingested"`: Imported from external data (e.g., PDF extraction)

**provenance.timestamp:**
- ISO 8601 format: `"2026-02-10T14:32:15Z"`
- UTC timezone preferred
- Used for temporal queries and audit trail

**provenance.specialist:**
- Creating specialist name: `"visual"`, `"math"`, `"physics"`, `"language"`
- Enables specialist-specific queries
- Supports Matryoshka routing (different specialists for different tasks)

**provenance.confidence:**
- Float in range [0.0, 1.0]
- 1.0 = canonical knowledge (verified)
- 0.8-0.9 = high-confidence discovery
- 0.5-0.7 = uncertain discovery
- <0.5 = speculative hypothesis
- Used for shadow copy learning (reinforce successful patterns)

---

## 3. Conformance Requirements

### 3.1 Identity Stability

**MUST:**
- Node IDs remain stable across serialization/deserialization
- Content-addressed IDs (e.g., SHA256 hash) are immutable
- UUID-based IDs persist in audit journal

**MUST NOT:**
- Change ID after node creation (breaks symlink references)
- Reuse IDs from deleted nodes (audit trail consistency)

### 3.2 RDF Mapping

K3D nodes map to RDF triples for semantic web interoperability:

```turtle
@prefix k3d: <http://knowledge3d.org/vocab#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<k3d:drawing/line_01>
  a k3d:FoundationalPrimitive ;
  k3d:type "foundational_primitive" ;
  k3d:domain "drawing" ;
  k3d:rpnProgram "0 0 10 0 LINE" ;
  k3d:provenanceSource "canonical" ;
  k3d:provenanceConfidence "1.0"^^xsd:float ;
  k3d:provenanceTimestamp "2026-02-10T14:32:15Z"^^xsd:dateTime ;
  k3d:provenanceSpecialist "visual" .
```

**Conversion Tool:**
```bash
python scripts/export_knowledgeverse_to_rdf.py \
  --input ../Knowledge3D.local/galaxies/ \
  --output k3d_knowledge_graph.ttl \
  --format turtle
```

### 3.3 Deterministic Addressing

**For reproducibility, K3D implements deterministic spatial addressing:**

1. **Content-Addressed Coordinates:**
   - `hash(rpn_program) → (x, y, z)` in Galaxy Universe
   - Same RPN program always maps to same location
   - Enables cache hits across sessions

2. **Semantic Proximity:**
   - Similar RPN programs have close coordinates
   - Uses LSH (Locality-Sensitive Hashing) for embedding
   - TRM learns to navigate semantically-close regions

3. **Galaxy Partitioning:**
   - Each galaxy occupies distinct 3D subspace
   - Drawing: (0-1000, 0-1000, 0-1000)
   - Grammar: (1000-2000, 0-1000, 0-1000)
   - Math: (2000-3000, 0-1000, 0-1000)
   - Prevents cross-domain collision

### 3.4 Substrate Portability

**K3D nodes must be portable across:**

1. **Storage Substrates:**
   - JSONL files (newline-delimited JSON)
   - SQLite database (compressed audit journal)
   - VRAM buffers (GPU-native CuPy arrays)
   - RDF triples (semantic web export)

2. **Execution Substrates:**
   - NVIDIA GPUs (CUDA/PTX)
   - AMD GPUs (ROCm/HIP, future)
   - Intel GPUs (SYCL, future)
   - CPU fallback (emergency only, not sovereign)

3. **Serialization Formats:**
   - JSON (human-readable, ingestion/export)
   - MessagePack (efficient binary, inter-process)
   - Protocol Buffers (typed schemas, versioning)
   - glTF 2.0 extensions (3D asset exchange)

**Export Example:**
```bash
# Export Drawing Galaxy to glTF 2.0
python scripts/export_galaxy_to_gltf.py \
  --galaxy Drawing \
  --output drawing_galaxy.gltf \
  --include-extensions K3D_procedural_shapes

# Import from glTF back to K3D
python scripts/import_gltf_to_galaxy.py \
  --input external_shapes.gltf \
  --galaxy Drawing \
  --confidence 0.7
```

---

## 4. Example Nodes (Concrete Instances)

### 4.1 Foundational Primitive (Drawing Galaxy)

```json
{
  "id": "line_primitive_01",
  "type": "foundational_primitive",
  "rpn_program": "0 0 10 0 LINE",
  "domain": "drawing",
  "provenance": {
    "source": "canonical",
    "timestamp": "2026-02-06T10:00:00Z",
    "specialist": "visual",
    "confidence": 1.0
  },
  "payload": {
    "description": "Horizontal line from origin to (10,0)",
    "primitive_type": "LINE",
    "parameters": ["x0", "y0", "x1", "y1"],
    "glTF_compatible": true
  },
  "spatial": {
    "x": 125.3,
    "y": 48.7,
    "z": 12.1,
    "galaxy": "Drawing"
  }
}
```

### 4.2 Discovered Pattern (ARC-AGI Task)

```json
{
  "id": "discovered_pattern_arc_007fbbfb_gen_12",
  "type": "discovered_pattern",
  "rpn_program": "GRID_IN DETECT_SYMMETRY AXIS_VERTICAL REFLECT OVERLAY",
  "domain": "grammar",
  "provenance": {
    "source": "discovered",
    "timestamp": "2026-02-09T12:45:32Z",
    "specialist": "visual",
    "confidence": 0.82,
    "discovery_context": {
      "task_id": "007fbbfb",
      "train_pair_index": 2,
      "generation_method": "ternary_contrastive_anti",
      "verified": false
    }
  },
  "payload": {
    "pattern_family": "symmetry",
    "transformation_type": "vertical_reflection",
    "success_count": 3,
    "failure_count": 1,
    "first_seen": "2026-02-09T12:45:32Z"
  },
  "links": [
    {
      "target_id": "reflect_transformation_canonical",
      "target_galaxy": "Drawing",
      "relation": "uses_primitive"
    }
  ],
  "spatial": {
    "x": 1523.8,
    "y": 782.1,
    "z": 94.3,
    "galaxy": "Grammar"
  }
}
```

### 4.3 Mathematical Symbol (Math Galaxy)

```json
{
  "id": "latex_frac_symbol",
  "type": "derived_symbol",
  "rpn_program": "NUMERATOR DENOMINATOR DIVIDE_LINE STACK_VERTICAL",
  "domain": "math",
  "provenance": {
    "source": "canonical",
    "timestamp": "2026-02-06T10:05:00Z",
    "specialist": "math",
    "confidence": 1.0
  },
  "payload": {
    "latex_command": "\\frac",
    "parameters": ["numerator", "denominator"],
    "visual_template": "fraction_vertical_bar",
    "semantic": "division_operator",
    "arity": 2
  },
  "links": [
    {
      "target_id": "line_primitive_01",
      "target_galaxy": "Drawing",
      "relation": "renders_using"
    },
    {
      "target_id": "division_operator",
      "target_galaxy": "Math",
      "relation": "semantic_equivalent"
    }
  ],
  "spatial": {
    "x": 2048.5,
    "y": 512.0,
    "z": 256.7,
    "galaxy": "Math"
  }
}
```

### 4.4 Physics System (Reality Galaxy)

```json
{
  "id": "projectile_motion_system",
  "type": "procedural_system",
  "rpn_program": "INIT_POS INIT_VEL GRAVITY_ACCEL TIME_STEP INTEGRATE_VERLET UPDATE_POS",
  "domain": "reality",
  "provenance": {
    "source": "canonical",
    "timestamp": "2026-02-06T11:00:00Z",
    "specialist": "physics",
    "confidence": 1.0
  },
  "payload": {
    "physics_domain": "classical_mechanics",
    "simulation_type": "particle_dynamics",
    "time_dependent": true,
    "parameters": {
      "g": -9.81,
      "dt": 0.016,
      "air_resistance": 0.0
    },
    "validation": "galileo_free_fall_experiment"
  },
  "spatial": {
    "x": 3125.7,
    "y": 1024.3,
    "z": 512.9,
    "galaxy": "Reality"
  }
}
```

---

## 5. Validation Rules

### 5.1 Schema Validation

**JSON Schema for K3D nodes:**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "K3D Node",
  "type": "object",
  "required": ["id", "type", "rpn_program", "domain", "provenance"],
  "properties": {
    "id": {
      "type": "string",
      "minLength": 1,
      "pattern": "^[a-zA-Z0-9_:/-]+$"
    },
    "type": {
      "type": "string",
      "minLength": 1
    },
    "rpn_program": {
      "type": "string",
      "minLength": 1
    },
    "domain": {
      "type": "string",
      "enum": ["drawing", "grammar", "math", "reality", "character", "audio", "3d_objects"]
    },
    "provenance": {
      "type": "object",
      "required": ["source", "timestamp", "specialist", "confidence"],
      "properties": {
        "source": {
          "type": "string",
          "enum": ["canonical", "discovered", "derived", "ingested"]
        },
        "timestamp": {
          "type": "string",
          "format": "date-time"
        },
        "specialist": {
          "type": "string"
        },
        "confidence": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        }
      }
    },
    "payload": {
      "type": "object"
    },
    "links": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["target_id", "target_galaxy", "relation"],
        "properties": {
          "target_id": {"type": "string"},
          "target_galaxy": {"type": "string"},
          "relation": {"type": "string"}
        }
      }
    },
    "spatial": {
      "type": "object",
      "required": ["x", "y", "z", "galaxy"],
      "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"},
        "z": {"type": "number"},
        "galaxy": {"type": "string"}
      }
    }
  }
}
```

**Validation Tool:**
```bash
python scripts/validate_knowledgeverse_schema.py \
  --galaxies ../Knowledge3D.local/galaxies/ \
  --schema docs/paper-evidence/k3d_node_schema.json \
  --strict
```

### 5.2 Sovereignty Validation

**Every RPN program must be PTX-executable:**

```python
# Sovereignty test
def validate_rpn_sovereignty(rpn_program: str) -> bool:
    """Check that RPN contains only PTX-executable operations."""
    forbidden_imports = ["numpy", "scipy", "torch", "tensorflow", "sklearn"]
    forbidden_ops = ["EXTERNAL_CALL", "PYTHON_EVAL", "LAZY_EMBED"]

    tokens = rpn_program.split()
    for token in tokens:
        if any(lib in token.lower() for lib in forbidden_imports):
            return False  # Sovereignty violation
        if token in forbidden_ops:
            return False  # Non-sovereign operation

    return True  # Sovereign RPN program
```

**Test Suite:**
```bash
pytest tests/test_sovereignty_validation.py -v
# Expected: All nodes in all galaxies pass sovereignty checks
```

---

## 6. Interoperability Mappings

### 6.1 glTF 2.0 Extension

K3D nodes map to glTF 2.0 assets with custom extension:

```json
{
  "asset": {"version": "2.0"},
  "extensions": {
    "K3D_procedural_shapes": {
      "nodes": [
        {
          "id": "line_primitive_01",
          "type": "foundational_primitive",
          "rpn_program": "0 0 10 0 LINE",
          "provenance": {
            "source": "canonical",
            "confidence": 1.0
          }
        }
      ]
    }
  },
  "meshes": [
    {
      "name": "line_mesh_01",
      "primitives": [
        {
          "attributes": {"POSITION": 0},
          "mode": 1
        }
      ],
      "extensions": {
        "K3D_procedural_shapes": {
          "rpn_program": "0 0 10 0 LINE"
        }
      }
    }
  ]
}
```

### 6.2 RDF/OWL Ontology

Full ontology available at: `docs/vocabulary/k3d_ontology.owl`

**Key Classes:**
- `k3d:Node` (superclass)
  - `k3d:FoundationalPrimitive`
  - `k3d:Transformation`
  - `k3d:DiscoveredPattern`
  - `k3d:DerivedSymbol`

**Key Properties:**
- `k3d:rpnProgram` (functional, required)
- `k3d:domain` (required)
- `k3d:provenanceSource` (required)
- `k3d:spatialCoordinate` (optional)

### 6.3 ARIA Metadata (Accessibility)

K3D nodes include ARIA metadata for screen reader support:

```json
{
  "id": "line_primitive_01",
  "aria": {
    "role": "img",
    "label": "Horizontal line from origin to point (10, 0)",
    "description": "Foundational drawing primitive representing a straight line segment",
    "live": "polite"
  },
  "braille": {
    "pattern": "⠿⠿⠿⠿",
    "description": "Tactile representation of horizontal line"
  }
}
```

---

## 7. Versioning and Evolution

### 7.1 Schema Versioning

```json
{
  "schema_version": "1.0.0",
  "id": "node_example",
  "type": "foundational_primitive",
  ...
}
```

**Backward Compatibility:**
- Version 1.x readers MUST accept 1.0 nodes
- New optional fields allowed in minor versions
- Breaking changes require major version bump

### 7.2 Node Lifecycle

1. **Creation**: Node inserted into galaxy with confidence 0.5-1.0
2. **Validation**: Shadow copy testing (success → confidence increase)
3. **Reinforcement**: Repeated successful use → confidence approaches 1.0
4. **Deprecation**: Confidence drops below 0.1 → marked for pruning
5. **Archival**: Moved to compressed audit journal (not deleted)

---

## 8. Conformance Levels

### Level 1: Minimal Conformance (REQUIRED)
- [x] Required fields present (id, type, rpn_program, domain, provenance)
- [x] Valid JSON syntax
- [x] RPN program is PTX-sovereign (no external calls)
- [x] Provenance confidence in [0.0, 1.0]

### Level 2: Standard Conformance (RECOMMENDED)
- [x] Level 1 requirements
- [x] Spatial coordinates assigned
- [x] JSON schema validation passes
- [x] Deterministic addressing (content-based IDs)

### Level 3: Full Conformance (OPTIONAL)
- [x] Level 2 requirements
- [x] RDF export available
- [x] glTF 2.0 compatible (if visual domain)
- [x] ARIA metadata (if user-facing)
- [x] Cross-galaxy symlinks documented

---

## 9. Reference Implementations

**Source Files:**
- `knowledge3d/knowledgeverse/galaxy_manager.py` (core node storage)
- `knowledge3d/knowledgeverse/drawing_galaxy.py` (Drawing node examples)
- `knowledge3d/knowledgeverse/grammar_galaxy.py` (Grammar node examples)
- `knowledge3d/knowledgeverse/reality_galaxy.py` (Physics node examples)

**Validation Scripts:**
- `scripts/validate_knowledgeverse_schema.py`
- `scripts/export_knowledgeverse_to_rdf.py`
- `scripts/export_galaxy_to_gltf.py`

**Test Suite:**
- `tests/test_knowledgeverse_schema.py`
- `tests/test_sovereignty_validation.py`
- `tests/test_rdf_export.py`

---

## 10. Contact and Governance

**Specification Maintainer:** Knowledge3D Project
**License:** Apache 2.0 (open specification)
**Patent Policy:** No patents filed (public prior art)
**Standards Body:** Submitted to W3C Spatial Data on the Web Interest Group

**For questions or clarifications:**
- GitHub: https://github.com/knowledge3d/knowledge3d
- Email: specs@knowledge3d.org (TBD)
- W3C Community Group: TBD

---

**Document Version:** 1.0.0
**Last Updated:** February 10, 2026
**Status:** STABLE (for scientific publication)
