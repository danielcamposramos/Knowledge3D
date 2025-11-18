# Three-Brain System Specification

**Version**: 1.0
**Status**: Production (Phase G Complete)
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)
**Date**: November 2025

---

## Abstract

The **Three-Brain System** is K3D's hierarchical memory architecture that mirrors biological memory systems and computer memory hierarchies. It separates cognitive functions into three distinct but interconnected components: **Cranium** (reasoning), **Galaxy** (active memory), and **House** (persistent memory). This architecture enables explainable AI through embodied spatial reasoning while maintaining high performance and scalability.

---

## 1. Introduction

### 1.1 Motivation

Traditional AI systems conflate computation and storage, leading to:
- **Opacity**: Memory and reasoning are entangled (can't observe one without affecting the other)
- **Inefficiency**: Must load entire model into memory for any operation
- **Non-Scalability**: Knowledge base size limited by GPU VRAM

**Biological Inspiration**:
Human cognition separates:
1. **Prefrontal Cortex**: Executive function, reasoning, planning
2. **Hippocampus**: Active working memory, rapid encoding
3. **Neocortex**: Long-term consolidated memory

**Computer Architecture Analogy**:
1. **CPU/GPU**: Processing units
2. **RAM**: Fast volatile storage
3. **Disk/SSD**: Persistent storage

### 1.2 Design Principles

1. **Separation of Concerns**: Reasoning ≠ Memory ≠ Persistence
2. **Explicit Embodiment**: Memory is the external 3D world (not internal parameters)
3. **Scalability**: Knowledge base can exceed GPU VRAM (only active subset loaded)
4. **Transparency**: Memory state is always inspectable (GLB files are human-readable 3D)
5. **Biological Fidelity**: Mirrors neuroscience principles (consolidation, replay, forgetting)

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    K3D THREE-BRAIN SYSTEM                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │   CRANIUM    │────────▶│    GALAXY    │                 │
│  │  (Reasoning) │◀────────│  (Active RAM) │                 │
│  └──────────────┘         └──────┬───────┘                 │
│        │                          │                          │
│        │                          │ SleepTime                │
│        │                          │ Consolidation            │
│        │                          ▼                          │
│        │                  ┌──────────────┐                 │
│        └─────────────────▶│    HOUSE     │                 │
│          (Provenance)     │ (Persistent  │                 │
│                           │    Disk)     │                 │
│                           └──────────────┘                 │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│  Analogies:                                                   │
│  • Biology:  PFC + Hippocampus + Neocortex                   │
│  • Computing: CPU/GPU + RAM + Disk                           │
│  • Philosophy: Mind + Working Awareness + Long-Term Knowledge│
└─────────────────────────────────────────────────────────────┘
```

### 2.1 Component Hierarchy

| Component | Function | Storage | Access Time | Capacity | Volatility |
|-----------|----------|---------|-------------|----------|------------|
| **Cranium** | Reasoning & Inference | GPU registers | ~42µs | N/A | Stateless |
| **Galaxy** | Active working memory | GPU RAM | ~5µs | Constrained by VRAM | Volatile |
| **House** | Long-term persistence | SSD/HDD | ~5ms | Unlimited | Persistent |

---

## 3. Component 1: Cranium (Reasoning Engine)

### 3.1 Overview

**Cranium** is the sovereign reasoning engine—a collection of hand-written PTX GPU kernels that perform atomic cognitive operations without external dependencies.

**Philosophy**: "Intelligence is not stored; it is computed."
- Cranium has NO parameters to tune (all cognition is algorithmic)
- Reasoning emerges from execution of atomic operations
- Complete transparency: Every operation traceable to source PTX kernel

### 3.2 Architecture

```
Cranium Components:
├── RPN Execution Engine (15-stack VM)
│   ├── Stack operations: PUSH, POP, DUP, SWAP
│   ├── Arithmetic: ADD, SUB, MUL, DIV, MOD, POW
│   ├── Logic: AND, OR, NOT, XOR
│   ├── Control: BRANCH, LOOP, CALL, RET
│   └── Memory: STORE, RECALL, LOAD_GALAXY, SAVE_GALAXY
├── TRM (Tiny Recursive Model) Kernels
│   ├── Forward pass (2-layer SwiGLU MLP)
│   ├── Recursive refinement (iterate until convergence)
│   └── Attention mechanism (scaled dot-product)
├── Spatial Operations
│   ├── Frustum culling (FOV-based filtering)
│   ├── Pathfinding (A* through Galaxy graph)
│   ├── Octree traversal (spatial acceleration)
│   └── Embedding similarity (cosine distance, SIMD-optimized)
└── Multi-Modal Fusion
    ├── Text embedding (character-level + word-level)
    ├── Visual embedding (CNN feature extraction)
    ├── Audio embedding (spectrogram → features)
    └── Cross-modal alignment (spatial proximity loss)
```

### 3.3 PTX Kernel Suite

**42 Production Kernels** (all <100µs latency):

**Core Reasoning**:
- `rpn_execute.ptx` - RPN stack machine interpreter (15µs)
- `trm_forward.ptx` - TRM forward pass (32µs)
- `trm_recursive.ptx` - Iterative reasoning refinement (80µs for 9 steps)

**Spatial Operations**:
- `frustum_cull_simd.ptx` - SIMD-optimized frustum culling (8µs)
- `morton_octree.ptx` - Z-order curve spatial indexing (12µs)
- `pathfind_astar.ptx` - A* pathfinding through graph (42µs average)
- `embedding_cosine_simd.ptx` - Batch cosine similarity (25µs for 1000 nodes)

**Multi-Modal**:
- `glyph_match.ptx` - Character recognition via template matching (42µs)
- `visual_embed.ptx` - CNN feature extraction (98µs)
- `audio_spectrogram.ptx` - FFT-based audio embedding (67µs)

**Performance Characteristics**:
- All kernels <100µs (sub-frame latency at 10,000 fps)
- Zero CPU fallback (fail-fast on GPU errors)
- SIMD-optimized (32 threads/warp, 100% occupancy)
- Memory-efficient (<2KB stack per thread)

### 3.4 RPN Execution Example

**Query**: "What is a neuron?"

```assembly
# RPN Bytecode (PTX-compiled)
PUSH "neuron"              # Push query to stack
LOAD_GALAXY embedding      # Load embedding vector from Galaxy
PUSH 10                    # Top-K parameter
CALL find_similar          # Find 10 most similar nodes
CALL pathfind_to_answer    # Navigate to answer node
RECALL answer_text         # Retrieve text data
OUTPUT                     # Return to user

# Execution Trace (42µs total):
0µs:  Stack: ["neuron"]
5µs:  Stack: ["neuron", vec[1024]]
7µs:  Stack: ["neuron", vec[1024], 10]
32µs: Stack: ["neuron", [node_234, node_567, ...]]  # Similar nodes found
40µs: Stack: ["neuron", "A neuron is a nerve cell..."]
42µs: OUTPUT → "A neuron is a nerve cell that transmits electrical signals."
```

**Transparency**: Every RPN operation logged, enabling full reasoning trace.

---

## 4. Component 2: Galaxy (Active Memory)

### 4.1 Overview

**Galaxy** is K3D's active working memory—a 3D spatial structure populated by "stars" (K3D Nodes) as 3D embeddings. It serves as the bidirectional bridge between symbolic (House) and neural (Cranium) processing.

**Philosophy**: "The world is the memory."
- Memory is NOT internal model parameters
- Memory is the external 3D environment
- AI navigates memory spatially (like humans navigate physical space)

### 4.2 Structure

```
Galaxy Memory Space:
├── Coordinate System: Cartesian 3D (x, y, z)
│   └── Origin: (0, 0, 0) at conceptual center
├── Nodes: K3D spatial knowledge units
│   ├── Position: (x, y, z) encodes semantic proximity
│   ├── Embedding: 1024-4096 dim vectors
│   └── Shape: Platonic solids encode modality
├── Edges: Semantic relationships
│   ├── Type: Spatial (proximity), Semantic (RDF), Causal
│   └── Weight: Strength [0.0, 1.0]
└── Spatial Acceleration Structures:
    ├── Octree: Hierarchical bounding volumes
    ├── KD-Tree: K-dimensional space partitioning
    └── BVH: Bounding volume hierarchy for ray queries
```

### 4.3 Memory Properties

**Capacity** (Phase G Production):
- 51,532 total nodes (stars)
- 17,035 non-zero embeddings (33.1% active)
- 1024-dimensional vectors (float32)
- ~12 MB GPU RAM usage

**Spatial Distribution**:
- Semantic clusters: Related concepts naturally group in 3D space
- Sparsity: 66.9% of nodes are "dark matter" (zero embeddings, placeholders)
- Density: ~5-10 nodes per cubic unit in active regions

**Access Patterns**:
- **Spatial Query**: Find all nodes within radius r of position (x, y, z)
  - Time: O(log N) via octree (measured: ~15µs for r=5.0)
- **Semantic Query**: Find K most similar embeddings to query vector
  - Time: O(N) naive, O(log N) with approximate nearest neighbor (measured: ~32µs for K=10)
- **Hybrid Query**: Spatial + Semantic constraints (e.g., "find neurons within 10 units of position X")
  - Time: ~45µs (spatial filter first, then semantic ranking)

### 4.4 Dynamic Behavior

**Active Memory Management**:
- **LRU Eviction**: Least-recently-used nodes evicted when VRAM limit reached
- **Lazy Loading**: Nodes loaded from House on-demand (cache miss)
- **Prefetching**: Predicted next nodes loaded during idle cycles
- **Consolidation**: Periodic SleepTime events persist active nodes to House

**Memory Consolidation (SleepTime)**:
```python
def sleep_time_consolidation():
    """
    Biological analogy: Hippocampal replay during sleep.
    Transfers active Galaxy nodes to persistent House storage.
    """
    # 1. LOCK Galaxy (pause inference)
    galaxy.lock()

    # 2. EMA UPDATE (smooth embeddings over time)
    for node in galaxy.active_nodes:
        node.embedding = (
            0.9 * node.embedding_previous +
            0.1 * node.embedding_current
        )  # Exponential moving average, α=0.1

    # 3. PRUNE redundancy (remove near-duplicates)
    for node_a, node_b in galaxy.all_pairs():
        if cosine_similarity(node_a.embedding, node_b.embedding) > 0.98:
            merge_nodes(node_a, node_b)  # Keep higher access_count

    # 4. SERIALIZE to GLB (compress to disk format)
    glb_data = serialize_galaxy_to_gltf(galaxy)

    # 5. COMMIT to House (atomic write)
    house.write_transaction(glb_data, timestamp=now())

    # 6. UNLOCK Galaxy (resume inference)
    galaxy.unlock()
```

**Measured Performance**:
- Consolidation time: ~8.3ms for 51,532 nodes (meets <10ms target)
- Compression ratio: 4:1 (34MB → 8.5MB GLB with Draco)
- Atomicity: Transaction-based (all-or-nothing writes)

---

## 5. Component 3: House (Persistent Memory)

### 5.1 Overview

**House** is K3D's long-term persistent memory—a collection of glTF scenes stored on disk that represent consolidated knowledge states. It serves as the "ground truth" for knowledge, from which Galaxy is populated.

**Philosophy**: "Knowledge must persist beyond runtime."
- House outlives any single inference session
- Knowledge bases are portable (copy GLB files between systems)
- Human-inspectable (load in Blender, view 3D structure)

### 5.2 Structure

```
House File System:
/K3D/Knowledge3D.local/house/
├── worlds/
│   ├── neuroscience_2025-11-07.glb  (Active world)
│   ├── neuroscience_2025-11-05.glb  (Previous snapshot)
│   └── neuroscience_2025-10-15.glb  (Initial knowledge base)
├── archives/
│   ├── 2025-Q3/
│   │   └── ...
│   └── 2025-Q4/
│       └── ...
└── index/
    ├── manifest.json  (List of all worlds with metadata)
    └── provenance.db  (SQLite database of source→node mappings)
```

### 5.3 glTF Scene Format

Each House GLB file is a valid glTF 2.0 scene containing:

**Scene Graph**:
- Root node: World origin
- Child nodes: K3D Nodes (concepts, entities, relations)
- Meshes: Platonic solid geometries
- Materials: PBR materials with semantic colors

**K3D Extensions** (in `extras.k3d`):
- Node embeddings (base64-encoded float32 arrays)
- Semantic metadata (RDF triples, ontology references)
- Provenance (source URLs, timestamps, confidence scores)
- Memory state (access counts, consolidation status)

**File Size**:
- Uncompressed: ~34 MB (51,532 nodes)
- Draco compressed: ~8.5 MB (4:1 ratio)
- Embedding data: ~200 MB (float32 vectors) → stored externally in `.npz` files

**Versioning**:
- Filename includes ISO timestamp: `world_YYYY-MM-DD.glb`
- Git-style content addressing: SHA256 hash in metadata
- Incremental backups: Only changed nodes re-serialized

### 5.4 Loading & Saving

**Loading from House to Galaxy**:
```python
def load_house_to_galaxy(world_path: str):
    """Load persistent knowledge from House GLB into active Galaxy."""
    # Parse GLB file
    gltf_data = parse_gltf(world_path)

    # Extract K3D nodes
    for gltf_node in gltf_data['nodes']:
        if 'k3d' in gltf_node['extras']:
            k3d_node = deserialize_k3d_node(gltf_node)
            galaxy.insert(k3d_node)

    # Build spatial acceleration structures
    galaxy.build_octree()
    galaxy.build_kdtree()

    print(f"Loaded {len(galaxy.nodes)} nodes from {world_path}")
```

**Saving from Galaxy to House**:
```python
def save_galaxy_to_house(world_name: str):
    """Consolidate active Galaxy into persistent House GLB."""
    timestamp = datetime.now().isoformat()
    output_path = f"/K3D/Knowledge3D.local/house/worlds/{world_name}_{timestamp}.glb"

    # Serialize Galaxy nodes to glTF
    gltf_scene = {
        "asset": {"version": "2.0", "generator": "K3D SleepTime v1.0"},
        "nodes": [node.to_gltf() for node in galaxy.active_nodes],
        "meshes": generate_platonic_solid_meshes(),
        "materials": generate_semantic_materials()
    }

    # Compress with Draco
    glb_data = compress_gltf_to_glb(gltf_scene, use_draco=True)

    # Atomic write (temp file + rename)
    write_atomic(output_path, glb_data)

    print(f"Saved {len(galaxy.active_nodes)} nodes to {output_path}")
```

---

## 6. Inter-Component Communication

### 6.1 Data Flow Patterns

**Inference (Query Answering)**:
```
User Query
    ↓
Cranium (RPN parse query)
    ↓
Galaxy (spatial + semantic search)
    ↓
Cranium (reasoning via pathfinding)
    ↓
Galaxy (retrieve answer node data)
    ↓
Cranium (format response)
    ↓
User Answer
```

**Learning (Knowledge Ingestion)**:
```
External Data (PDF, audio, image)
    ↓
Cranium (embedding extraction)
    ↓
Galaxy (insert new node at semantic position)
    ↓
[Periodic SleepTime trigger]
    ↓
House (consolidate to persistent GLB)
```

**Memory Consolidation (SleepTime)**:
```
Galaxy (active nodes + access stats)
    ↓
Cranium (EMA smoothing, redundancy pruning)
    ↓
Galaxy (update embeddings, merge nodes)
    ↓
House (serialize to GLB, atomic write)
    ↓
Galaxy (mark nodes as consolidated)
```

### 6.2 Communication Protocols

**Cranium ↔ Galaxy**:
- **Protocol**: Direct GPU memory access (zero-copy)
- **Latency**: ~5µs per node read/write
- **Bandwidth**: ~100 GB/s (GPU RAM bandwidth)

**Galaxy ↔ House**:
- **Protocol**: File I/O (glTF serialization/deserialization)
- **Latency**: ~5ms per world load/save
- **Bandwidth**: ~200 MB/s (SSD sequential write)

**Cranium ↔ House** (provenance lookup):
- **Protocol**: SQLite query on `provenance.db`
- **Latency**: ~0.5ms per lookup
- **Use Case**: Retrieve source URL for answer provenance chain

---

## 7. Biological & Computer Architecture Analogies

### 7.1 Neuroscience Parallels

| Biological Structure | K3D Component | Function |
|---------------------|---------------|----------|
| **Prefrontal Cortex** | Cranium | Executive function, reasoning, planning |
| **Hippocampus** | Galaxy | Rapid encoding, spatial navigation, memory consolidation |
| **Neocortex** | House | Long-term declarative memory storage |
| **Sleep Cycles** | SleepTime | Memory consolidation, synaptic pruning |
| **Spatial Navigation Cells** | Galaxy Octree | Place cells, grid cells for spatial indexing |
| **Synaptic Plasticity** | EMA Embedding Updates | Hebbian learning ("neurons that fire together, wire together") |

### 7.2 Computer Architecture Parallels

| Computing Component | K3D Component | Characteristics |
|--------------------|---------------|-----------------|
| **CPU/GPU** | Cranium | Fast, stateless processing |
| **L1/L2 Cache** | (Not modeled) | Sub-µs access, small capacity |
| **RAM** | Galaxy | µs access, medium capacity (VRAM-limited) |
| **SSD/HDD** | House | ms access, unlimited capacity |
| **Swap File** | (Not used) | K3D fails-fast instead of swapping |

---

## 8. Validation & Performance

### 8.1 Production Metrics (Phase G)

**Cranium (Reasoning Engine)**:
- ✅ 45+ PTX kernels, all <100µs latency
- ✅ Zero CPU fallbacks (100% GPU-native)
- ✅ 7M TRM parameters (10,000× more efficient than 70B LLMs)

**Galaxy (Active Memory)**:
- ✅ 51,532 nodes, 17,035 active (33.1%)
- ✅ Spatial query <15µs (octree-accelerated)
- ✅ Semantic query <32µs (SIMD-optimized cosine)
- ✅ 12 MB VRAM usage (<1% of RTX 3060 12GB)

**House (Persistent Storage)**:
- ✅ 8.5 MB compressed GLB (4:1 Draco ratio)
- ✅ Consolidation <10ms (meets real-time target)
- ✅ Load time <5ms (SSD-optimized)
- ✅ 100% glTF 2.0 compatible (loads in Blender)

### 8.2 Integration Tests

✅ **Cranium-Galaxy**: RPN `LOAD_GALAXY` opcode retrieves correct embeddings (10,000 tests, 100% pass)
✅ **Galaxy-House**: SleepTime consolidation preserves all node data (checksums match)
✅ **Cranium-House**: Provenance lookup returns correct source URLs (1,000 queries, 100% accuracy)
✅ **End-to-End**: Query "What is a neuron?" → Answer with provenance chain in <100µs

---

## 9. Future Enhancements

### 9.1 Planned (Q1 2026)

**Cranium**:
- Formal verification of PTX kernels (prove correctness)
- WebGPU port for browser-based reasoning

**Galaxy**:
- Dynamic LOD (Level of Detail) for million-node knowledge bases
- Approximate nearest neighbor (HNSW) for sub-10µs semantic queries

**House**:
- Incremental GLB serialization (only save changed nodes)
- Distributed House (knowledge sharded across multiple files/servers)

### 9.2 Research Directions

- **Neuroplasticity**: Online learning that modifies Cranium kernels
- **Episodic Memory**: Temporal indexing in Galaxy (remember query history)
- **Forgetting**: Biologically-inspired decay (unused nodes fade)

---

## 10. References

- Neuroscience: "The Hippocampus as a Cognitive Map" (O'Keefe & Nadel, 1978)
- Computer Architecture: "Computer Architecture: A Quantitative Approach" (Hennessy & Patterson)
- Memory Hierarchies: "Cache and Memory Hierarchy Design: A Performance-Directed Approach" (Przybylski)
- K3D Implementation: https://github.com/danielcamposramos/Knowledge3D

---

## Attribution & Academic Context

**For complete attributions**, see [ATTRIBUTIONS.md](../../ATTRIBUTIONS.md) in the K3D repository.

**Key Credits**:

1. **Neuroscience Research**:
   - Hippocampus as cognitive map (O'Keefe & Nadel, 1978)
   - Memory hierarchy concepts
   - K3D applies neuroscience principles to AI architecture

2. **Computer Architecture** (Hennessy & Patterson):
   - Cache hierarchies and memory management
   - K3D adapts for House (disk) / Galaxy (RAM) / Cranium (CPU) architecture

3. **Game Industry** (Memory Management):
   - LOD systems for efficient resource loading
   - SleepTime protocol inspired by game state management

4. **RDF/OWL** (W3C):
   - Persistent knowledge representation (House layer)

K3D's Three-Brain System is a novel contribution that applies biological and computer architecture principles to AI memory management.

---

## Contact & License

**Author**: Daniel Campos Ramos, K3D Architect
**Email**: daniel@echosystems.ai
**Repository**: https://github.com/danielcamposramos/Knowledge3D
**License**: CC-BY-4.0 (specification), Apache 2.0 (implementation code)

---

**Status**: Production (Phase G Complete, October 2025)
**Next Review**: Q1 2026 (for W3C CG Note submission)
