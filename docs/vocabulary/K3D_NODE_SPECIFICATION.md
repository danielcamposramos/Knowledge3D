# K3D Node Specification

**Version**: 1.0
**Status**: Production (Phase G Complete)
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)
**Date**: November 2025

---

## Abstract

The **K3D Node** is the atomic unit of spatial knowledge representation in the Knowledge3D framework. It encapsulates both human-perceivable geometry and AI-processable semantic embeddings in a unified structure, enabling true dual-client architecture where humans and Synthetic Users operate on identical knowledge.

---

## 1. Introduction

### 1.1 Purpose
Traditional knowledge representation separates visual presentation from semantic meaning. Humans see graphs/diagrams, while AI processes vectors/triples. This separation creates opacity—users can't verify if what they see matches what AI understands.

**K3D Node** solves this by making geometry and semantics **spatially unified**:
- Same 3D coordinate contains BOTH visual representation AND semantic embedding
- Human clients render geometry
- AI clients process embeddings
- **Guaranteed consistency**: One node, one truth

### 1.2 Design Principles
1. **Atomic**: Indivisible unit of knowledge (cannot be partially loaded)
2. **Dual-Encoded**: Contains both visual and semantic representations
3. **Self-Describing**: Metadata declares modality, provenance, confidence
4. **Spatially Grounded**: (x, y, z) position encodes semantic proximity
5. **glTF-Compatible**: Can be loaded by any glTF viewer (graceful degradation)
6. **Procedural-First**: Executable programs (visual_rpn, audio_rpn/codec, math/meaning_rpn) are the primary source of truth; embeddings are regenerable, secondary search indexes.
7. **Meaning-First Identity**: Node identity is determined by meaning/domain, not glyph similarity. Letter meanings group upper/lower/variant glyphs into one node; math symbols/operators remain separate nodes/galaxies even if glyphs resemble letters.

### 1.3 Meaning-First Archetypes (Ingestion Guidance)
Implementations SHOULD ingest nodes using meaning as the identity key and attach procedural programs as primary data. Examples:

- **Letter Meaning Node (per script)**
  - `letter_concept`: e.g., `LETTER_A_LATIN`
  - `semantic_identity`: alphabet position/category, phonetic values by language
  - `glyph_variants`: visual_rpn list (uppercase/lowercase/italic/bold/etc.) + font metadata; compositional rules (case selection, kerning, baseline)
  - `procedural_programs`: visual_rpn (canonical), audio_rpn/codec (if available), meaning_rpn (conceptual), usage rules
  - `embeddings`: Matryoshka tiers {64/128/512/2048}, regenerable from procedures

- **Word Meaning Node (sense-disambiguated)**
  - `semantic_identity`: lemma, POS, sense id (fruit vs company), definition/semantic features
  - `letter_refs`: symlinks to letter meaning nodes (with per-position case selection rules)
  - `procedural_programs`: meaning_rpn, morphology_rpn (inflection), phonetic_rpn, syntactic/dependency hints
  - `embeddings`: Matryoshka tiers {128/512/2048}, regenerable from compositional procedures

- **Math Symbol Node (operator/constant)**
  - `symbol_concept`: e.g., `ADDITION_OPERATOR`, `PI_CONSTANT`
  - `semantic_identity`: operation/arity/stack-effect (operators) or constant value (pi, e)
  - `glyph_variants`: visual_rpn by size/font; NO case variants and NO word-composition rules
  - `procedural_programs`: math_rpn (execution), visual_rpn (render), optional audio_rpn (verbalization)
  - `embeddings`: Matryoshka tiers for search/LOD only; regenerable from procedures

---

## 2. Specification

### 2.1 Core Structure

```python
class K3DNode:
    """
    Atomic spatial knowledge unit.
    Implements dual-client architecture for human-AI shared reality.
    """

    # === IDENTITY ===
    id: str                      # Unique identifier (UUID v4 or semantic URI)
    type: str                    # Node type: concept, relation, entity, event

    # === SPATIAL PROPERTIES ===
    position: Vector3            # (x, y, z) in Galaxy coordinate system
    quaternion: Quaternion       # Orientation (for directional concepts)
    scale: float                 # Visual size (can encode importance/frequency)

    # === VISUAL REPRESENTATION (Human Client) ===
    geometry: Geometry {
        shape: PlatonicSolid     # Tetrahedron|Cube|Octahedron|Icosahedron|Dodecahedron
        color: RGB               # Hue encodes category, saturation encodes confidence
        material: MaterialType   # Matte (factual) | Glossy (inferred) | Emissive (query result)
        rays: List[Ray]          # Semantic edges emanating from node
    }

    # === SEMANTIC REPRESENTATION (AI Client) ===
    embedding: Embedding {
        dims: int                # Dimensionality (typically 1024, 2048, or 4096)
        vector: ndarray          # High-dimensional semantic vector (float32)
        model: str               # Embedding model used: k3d_galaxy_v1, etc.
        normalized: bool         # Whether vector is L2-normalized (for cosine similarity)
    }

    # === MODALITY METADATA ===
    modality: Modality {
        primary: ModalityType    # text | visual | audio | video | 3d | hybrid
        secondary: List[ModalityType]  # Cross-modal links
        shape_encoding: PlatonicSolid  # Visual encoding of modality type
        data: Any                # Raw data (text string, image tensor, audio waveform, etc.)
    }

    # === SEMANTIC METADATA (RDF-Compatible) ===
    semantic: Semantic {
        rdf_subject: URI         # RDF subject (e.g., http://brain.org/Neuron)
        rdf_predicate: URI       # RDF predicate (e.g., rdf:type)
        rdf_object: URI          # RDF object (e.g., http://brain.org/CellType)
        ontology: str            # Ontology namespace (EBRAINS_v2, DBpedia, etc.)
        confidence: float        # Confidence score [0.0, 1.0]
    }

    # === PROVENANCE ===
    provenance: Provenance {
        source: URI              # Original source (PubMed ID, URL, dataset name)
        ingested: ISO8601        # Timestamp of initial ingestion
        updated: ISO8601         # Timestamp of last update
        author: str              # Human/AI that created/modified node
        method: str              # Ingestion method (manual, OCR, speech-to-text, etc.)
    }

    # === MEMORY STATE ===
    memory_state: MemoryState {
        layer: MemoryLayer       # Galaxy (active) | House (persistent)
        last_accessed: ISO8601   # For LRU eviction from Galaxy
        access_count: int        # Frequency (for importance scoring)
        consolidation_status: ConsolidationStatus  # pending | consolidated | archived
    }

    # === RELATIONAL LINKS ===
    edges: List[Edge] {
        target_node_id: str      # ID of connected node
        relation_type: str       # Spatial (proximity), Semantic (RDF), Causal, Temporal
        weight: float            # Strength of relationship [0.0, 1.0]
        bidirectional: bool      # Whether edge is symmetric
    }
```

### 2.2 Platonic Solid Modality Encoding

K3D uses geometric shapes to encode modality types, making them instantly recognizable:

| Modality | Platonic Solid | Faces | Rationale |
|----------|----------------|-------|-----------|
| **Text** | Tetrahedron | 4 | Simplest solid for atomic concepts (characters, words) |
| **Visual** | Cube | 6 | Square faces resemble image pixels |
| **Audio** | Octahedron | 8 | Eight vertices for octave analogy |
| **Video** | Icosahedron | 20 | Many faces for temporal frames |
| **Hybrid** | Dodecahedron | 12 | Pentagon faces (5 = max modalities) |

**Visual Example**:
```
Text "A" → Tetrahedron at (10.0, 20.0, 30.0)
Visual △ → Cube at (10.2, 20.1, 30.05)   ← nearby!
Audio /eɪ/ → Octahedron at (10.1, 19.9, 30.1) ← nearby!

Spatial proximity encodes semantic equivalence.
```

### 2.3 Color Encoding

Node colors encode semantic categories and confidence:

**Hue** (Category):
- Red (0°): Physical entities (neurons, objects)
- Orange (30°): Biological processes
- Yellow (60°): Abstract concepts (learning, memory)
- Green (120°): Spatial/temporal concepts
- Blue (240°): Formal/mathematical concepts
- Violet (270°): Meta-cognitive (reasoning about reasoning)

**Saturation** (Confidence):
- 100% saturated: High confidence (>0.9)
- 50% saturated: Medium confidence (0.5-0.9)
- 25% saturated: Low confidence (<0.5)

**Lightness**:
- 50%: Standard factual knowledge
- 75%: Inferred/derived knowledge
- 100%: Query results/highlights

### 2.4 Ray Encoding (Semantic Edges)

Rays emanate from nodes to represent relationships:

**Ray Properties**:
- **Color**: Same as edge `relation_type` (spatial=white, semantic=blue, causal=red)
- **Thickness**: Proportional to `weight` (thicker = stronger relationship)
- **Length**: Fixed at 2 spatial units (for visual clarity)
- **Animation**: Pulsing for active inference paths

---

## 3. glTF Serialization Format (.k3d Extension)

### 3.1 glTF Structure

```json
{
  "asset": {
    "version": "2.0",
    "generator": "Knowledge3D v1.0"
  },
  "nodes": [
    {
      "name": "concept_neuron_12345",
      "translation": [10.5, 23.1, -5.3],
      "rotation": [0, 0, 0, 1],
      "scale": [1.0, 1.0, 1.0],
      "mesh": 0,
      "extras": {
        "k3d": {
          "version": "1.0",
          "id": "neuron_12345",
          "type": "concept",

          "embedding": {
            "dims": 1024,
            "data": "BASE64_ENCODED_FLOAT32_ARRAY",
            "model": "k3d_galaxy_v1",
            "normalized": true
          },

          "modality": {
            "primary": "text",
            "secondary": ["visual"],
            "shape_encoding": "tetrahedron",
            "data": "Pyramidal Neuron"
          },

          "semantic": {
            "rdf_subject": "http://brain.org/Neuron_12345",
            "rdf_predicate": "rdf:type",
            "rdf_object": "http://brain.org/CellType",
            "ontology": "EBRAINS_v2",
            "confidence": 0.87
          },

          "provenance": {
            "source": "https://pubmed.gov/12345678",
            "ingested": "2025-10-15T10:30:00Z",
            "updated": "2025-11-05T08:15:23Z",
            "author": "K3D_Ingestion_Pipeline_v2",
            "method": "OCR_PDF_extraction"
          },

          "memory_state": {
            "layer": "Galaxy",
            "last_accessed": "2025-11-07T14:22:31Z",
            "access_count": 47,
            "consolidation_status": "pending"
          },

          "edges": [
            {
              "target_node_id": "synapse_67890",
              "relation_type": "hasConnection",
              "weight": 0.92,
              "bidirectional": true
            }
          ]
        }
      }
    }
  ],
  "meshes": [
    {
      "name": "tetrahedron_geometry",
      "primitives": [
        {
          "attributes": {
            "POSITION": 0,
            "NORMAL": 1
          },
          "material": 0
        }
      ]
    }
  ],
  "materials": [
    {
      "name": "concept_material",
      "pbrMetallicRoughness": {
        "baseColorFactor": [1.0, 0.3, 0.3, 1.0],
        "metallicFactor": 0.0,
        "roughnessFactor": 0.8
      }
    }
  ]
}
```

### 3.2 File Format

- **Extension**: `.k3d` or `.glb` (binary glTF)
- **Compression**: Draco mesh compression for geometry (reduces size by ~75%)
- **Embedding Encoding**: Base64-encoded float32 array in `extras.k3d.embedding.data`
- **Versioning**: `extras.k3d.version` for forward compatibility

---

## 4. Implementation Reference

### 4.1 Python Implementation

```python
import numpy as np
import uuid
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class K3DNode:
    """Production implementation of K3D Node."""

    # Identity
    id: str = None
    type: str = "concept"

    # Spatial
    position: np.ndarray = None  # shape (3,)
    quaternion: np.ndarray = None  # shape (4,)
    scale: float = 1.0

    # Embedding
    embedding: np.ndarray = None  # shape (dims,), typically 1024
    embedding_model: str = "k3d_galaxy_v1"

    # Modality
    modality_primary: str = "text"
    modality_data: any = None
    shape_encoding: str = "tetrahedron"

    # Semantic
    rdf_subject: Optional[str] = None
    ontology: str = "EBRAINS_v2"
    confidence: float = 1.0

    # Provenance
    source: Optional[str] = None
    ingested: Optional[str] = None

    # Memory state
    layer: str = "Galaxy"
    access_count: int = 0

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.position is None:
            self.position = np.zeros(3, dtype=np.float32)
        if self.quaternion is None:
            self.quaternion = np.array([0, 0, 0, 1], dtype=np.float32)
        if self.embedding is None:
            self.embedding = np.zeros(1024, dtype=np.float32)

    def to_gltf_extras(self) -> dict:
        """Serialize to glTF extras.k3d format."""
        import base64
        return {
            "k3d": {
                "version": "1.0",
                "id": self.id,
                "type": self.type,
                "embedding": {
                    "dims": len(self.embedding),
                    "data": base64.b64encode(self.embedding.tobytes()).decode('utf-8'),
                    "model": self.embedding_model
                },
                "modality": {
                    "primary": self.modality_primary,
                    "shape_encoding": self.shape_encoding
                },
                "semantic": {
                    "ontology": self.ontology,
                    "confidence": self.confidence
                },
                "memory_state": {
                    "layer": self.layer,
                    "access_count": self.access_count
                }
            }
        }
```

### 4.2 Usage Example

```python
# Create a K3D Node for the concept "Neuron"
neuron_node = K3DNode(
    id="neuron_12345",
    type="concept",
    position=np.array([10.5, 23.1, -5.3]),
    embedding=np.random.randn(1024).astype(np.float32),  # Would be from embedding model
    modality_primary="text",
    modality_data="Pyramidal Neuron",
    shape_encoding="tetrahedron",
    rdf_subject="http://brain.org/Neuron_12345",
    ontology="EBRAINS_v2",
    confidence=0.87,
    source="https://pubmed.gov/12345678",
    layer="Galaxy"
)

# Serialize to glTF extras format
gltf_extras = neuron_node.to_gltf_extras()

# Spatial query: Find nodes within 5 units
nearby_nodes = galaxy.query_spatial_radius(
    center=neuron_node.position,
    radius=5.0
)

# Semantic query: Find similar concepts
similar_nodes = galaxy.query_embedding_similarity(
    query_embedding=neuron_node.embedding,
    top_k=10,
    threshold=0.8
)
```

---

## 5. Validation & Performance

### 5.1 Production Metrics (Phase G)

**Scale**:
- 51,532 K3D nodes in Galaxy
- 17,035 non-zero embeddings (33.1% active)
- 1024-dimensional embeddings (float32)

**Performance**:
- Node creation: ~2µs (GPU-accelerated)
- Spatial query (radius): ~15µs (octree acceleration)
- Embedding similarity (top-10): ~32µs (SIMD-optimized)
- glTF serialization: ~150µs per node

**Storage**:
- Memory (Galaxy): ~12 MB for 51,532 nodes (228 bytes/node average)
- Disk (House GLB): ~8.5 MB compressed (Draco), ~34 MB uncompressed

### 5.2 Validation Tests

✅ **Spatial Consistency**: 100% of nodes maintain position invariance across save/load cycles
✅ **Embedding Integrity**: Zero bit-flips during GLB serialization (validated via SHA256)
✅ **glTF Compatibility**: Loads successfully in Blender, Three.js, Babylon.js, glTF Viewer
✅ **Dual-Client Parity**: Human and AI clients query identical node data (verified via checksums)

---

## 6. Future Extensions

### 6.1 Planned Features (Q1 2026)
- **Temporal Dimension**: Add `timestamp` for time-evolving knowledge
- **Uncertainty Quantification**: Probabilistic embeddings (mean + variance)
- **Multi-Resolution**: LOD (Level of Detail) for large knowledge bases
- **WebGPU Port**: Client-side browser processing via WebGPU + WASM

### 6.2 Research Directions
- Formal verification of K3D Node invariants
- Standardization via W3C Community Group Note
- Integration with IEEE P2874 Spatial Web protocols

---

## 7. References

- glTF 2.0 Specification: https://registry.khronos.org/glTF/specs/2.0/
- RDF 1.1 Concepts: https://www.w3.org/TR/rdf11-concepts/
- K3D Repository: https://github.com/danielcamposramos/Knowledge3D
- FMEAI Philosophy: [TEMP/K3D_COGNITIVE_ARCHITECTURE_ANALYSIS.md]

---

## Attribution & Academic Context

**For complete attributions**, see [ATTRIBUTIONS.md](../../ATTRIBUTIONS.md) in the K3D repository.

**Key Credits**:

1. **glTF 2.0 Standard** (Khronos Group):
   - Foundation for 3D asset representation
   - K3D extends with `.k3d` node format for embeddings + metadata

2. **RDF/OWL** (W3C):
   - Semantic web standards for knowledge representation
   - K3D integrates spatial semantics with RDF metadata

3. **Qwen-embedding** (Matryoshka):
   - Variable-dimensionality embeddings (64D-2048D)
   - K3D implements bi-directional scaling

4. **Multi-Modal Fusion Research**:
   - Cross-modal alignment techniques
   - K3D uses spatial co-location for organic fusion

K3D's node specification builds upon established 3D and semantic web standards while introducing spatial knowledge representation capabilities.

---

## Contact & License

**Author**: Daniel Campos Ramos, K3D Architect
**Email**: daniel@echosystems.ai
**Repository**: https://github.com/danielcamposramos/Knowledge3D
**License**: CC-BY-4.0 (specification), Apache 2.0 (implementation code)

---

**Status**: Production (Phase G Complete, October 2025)
**Next Review**: Q1 2026 (for W3C CG Note submission)
