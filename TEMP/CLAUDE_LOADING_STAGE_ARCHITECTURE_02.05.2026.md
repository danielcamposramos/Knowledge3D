# Loading Stage Architecture — Unified GPU Memory Arena

**Date**: February 5, 2026
**Author**: Claude (Architecture Partner)
**Status**: 🎯 **SPECIFICATION** (Ready for Codex Implementation)
**Priority**: **FOUNDATIONAL** - Solves CUDA context switching, enables Galaxy+House unification

---

## Executive Summary

The **Loading Stage** is a persistent GPU memory arena that solves the CUDA context switching problem while enabling seamless integration of Galaxy Universe (AI working memory) and House Universe (persistent storage). Inspired by game engine streaming architectures, it provides a unified context where all operations execute without context transfer conflicts.

**The Problem It Solves**:
```
Current Issue (Shadow Copy runs, external library conflicts):
  Operation A → Create CUDA context → Execute → Destroy context
  Operation B → Create CUDA context → ERROR: Previous context still active!

Root Cause:
  - PyTorch initializes its own CUDA context
  - Sovereign loader creates separate context
  - Conflict when both try to use GPU simultaneously
```

**The Solution**:
```
Loading Stage (Persistent Memory Arena):
  Startup → Allocate 70% of GPU VRAM (8.4GB on RTX 3060)
         → Create ONE persistent PTX context
         → Load everything into this context
         → All operations execute in SAME context
  Result → No context switching = No conflicts
```

**Key Innovation**: This architecture enables **dual-universe access** with **lazy loading** (like web browsers) and **semantic LOD** (like game engines), while maintaining **100% sovereignty** (PTX-only hot path).

---

## Architecture Overview

### The Two Universes (Dual-Client Reality)

```
┌─────────────────────────────────────────────────────────────────┐
│ CLIENT VIEW MODES (Humans AND AI)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ [Galaxy View] ←────── inspectable ──────→ [Galaxy View]        │
│   • AI's working memory (visible to humans for debugging!)     │
│   • Active reasoning state (embeddings, transformations)       │
│   • Hot cache (frequently used patterns)                       │
│   • Multi-modal workspace (Drawing, Math, Audio, Reality)      │
│                                                                 │
│ [House/World View] ←─── shared space ────→ [House/World View]  │
│   • Persistent storage (glTF objects = "holographic hard disk")│
│   • Objects within objects (galaxy boxes = offline AI data)    │
│   • Server-hosted houses (SaaS in spatial form)                │
│   • Library: Human section (books) + AI section (galaxy data)  │
│   • Network-accessible via "Doors" protocol                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Critical Insight**: Galaxy is NOT opaque to humans - they can inspect AI's working memory! This enables:
- Debugging AI reasoning (see what it's thinking)
- Teaching AI (correct mistakes by editing Galaxy)
- Collaboration (humans and AI share same 3D workspace)

### Loading Stage Memory Layout (RTX 3060: 12GB VRAM)

```
┌───────────────────────────────────────────────────────────────┐
│ GPU VRAM (12GB Total)                                         │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│ LOADING STAGE (8.4GB = 70% dedicated to K3D)                 │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Persistent PTX Context (ONE context for ALL operations)│ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │                                                         │ │
│ │ Region 1: PTX Kernels (100MB)                          │ │
│ │   • All Cranium operations (45+ kernels)               │ │
│ │   • RPN VM bytecode                                    │ │
│ │   • Loaded once at startup                             │ │
│ │                                                         │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │                                                         │ │
│ │ Region 2: Galaxy Universe (2GB)                        │ │
│ │   • Drawing Galaxy (visual primitives)                 │ │
│ │   • Character/Word Galaxy (language)                   │ │
│ │   • Grammar Galaxy (transformations)                   │ │
│ │   • Math Galaxy (symbolic reasoning)                   │ │
│ │   • Reality Galaxy (physics simulations)               │ │
│ │   • Audio Galaxy (temporal patterns)                   │ │
│ │   ↳ Active reasoning state (hot cache)                 │ │
│ │                                                         │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │                                                         │ │
│ │ Region 3: House Context (3GB)                          │ │
│ │   • Currently loaded House objects (lazy on-demand)    │ │
│ │   • Human-readable content (text, 3D meshes, images)   │ │
│ │   • AI data (galaxy boxes - serialized snapshots)      │ │
│ │   • LOD cache (centroids → medium → full detail)       │ │
│ │   ↳ LRU eviction policy (least recently used)          │ │
│ │                                                         │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │                                                         │ │
│ │ Region 4: World View Buffer (3.3GB)                    │ │
│ │   • Remote houses (via Doors - network streaming)      │ │
│ │   • Async download buffer (background loading)         │ │
│ │   • Collaboration workspace (multi-user sessions)      │ │
│ │   • Dynamic expansion (grows/shrinks with need)        │ │
│ │                                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│ RESERVED FOR OS (3.6GB = 30% for system + desktop rendering) │
│   • Integrated GPU rendering (desktop on iGPU)              │
│   • OS buffers and caches                                   │
│   • Headroom for system stability                           │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## K3D Object Container Format

### House Objects = Multi-Modal Self-Contained Packages

**Analogy**: Web pages downloaded by browser

```
Web Browsing:
  Navigate to URL → Browser downloads:
    - HTML (structure)
    - CSS (styling)
    - Images (content)
    - JavaScript (behavior)
  → Everything cached locally for instant access

K3D Equivalent:
  Navigate to House object → Loading Stage downloads:
    - glTF mesh (3D structure)
    - RDF metadata (semantic links)
    - RPN programs (procedural behavior)
    - Human content (text, images, audio)
    - AI data (galaxy snapshots, embeddings)
  → Everything in VRAM arena for instant reasoning
```

### Container Structure (glTF with K3D Extensions)

```json
{
  "asset": {
    "version": "2.0",
    "generator": "K3D House Builder"
  },
  "scene": 0,
  "scenes": [{"nodes": [0, 1, 2]}],
  "nodes": [
    {
      "name": "BookObject",
      "mesh": 0,
      "extras": {
        "k3d_type": "library_book",
        "k3d_dual_client": {
          "human_readable": {
            "title": "Calculus Foundations",
            "format": "markdown",
            "content_uri": "data/calculus_text.md"
          },
          "ai_executable": {
            "rpn_programs": ["data/calculus_proofs.rpn"],
            "galaxy_snapshot": "data/calculus_galaxy.bin"
          }
        },
        "k3d_ai_data": {
          "format": "compressed",
          "compression": "zstd",
          "payload_uri": "data/galaxy_box.bin",
          "contents": {
            "type": "galaxy_snapshot",
            "galaxies": ["Math", "Grammar"],
            "embeddings_dim": 512,
            "timestamp": "2026-02-05T12:00:00Z"
          }
        },
        "k3d_metadata": {
          "rdf": {
            "@context": "https://schema.org/",
            "@type": "Book",
            "name": "Calculus Foundations",
            "author": "Mathematical Sciences Library",
            "datePublished": "2025-11-15",
            "inLanguage": "en",
            "about": ["Calculus", "Mathematical Analysis"],
            "k3d_provenance": {
              "source": "https://example.com/calculus.pdf",
              "ingested": "2025-11-15T10:30:00Z",
              "checksum": "sha256:abc123..."
            }
          }
        }
      }
    }
  ],
  "meshes": [
    {
      "name": "BookMesh",
      "primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]
    }
  ]
}
```

**Key Components**:
1. **3D Geometry** (glTF standard): Visual representation for both clients
2. **k3d_dual_client**: Human-readable + AI-executable content
3. **k3d_ai_data**: Galaxy boxes (serialized AI state)
4. **k3d_metadata**: RDF semantic links + provenance

### Galaxy Boxes (AI Data in House Objects)

**Concept**: "Books for AI" - serialized galaxy snapshots stored in 3D objects

```
Galaxy Box Contents:
├── Format Metadata
│   ├── Compression (zstd, lz4, raw, or procedural)
│   ├── Timestamp (when saved)
│   └── Schema version (for compatibility)
│
├── Galaxy Snapshots
│   ├── Math Galaxy state (symbols + embeddings)
│   ├── Grammar Galaxy state (rules + patterns)
│   ├── Drawing Galaxy state (visual primitives)
│   └── ... (any galaxy can be serialized)
│
├── Embeddings
│   ├── Matryoshka vectors (64/128/512/2048D)
│   ├── Spatial coordinates (x, y, z)
│   └── Semantic metadata (tags, provenance)
│
├── TRM Weights (Optional)
│   ├── Learned patterns (shadow copy enhancements)
│   ├── Specialist adapters (math, visual, physics)
│   └── Checkpoint metadata (training history)
│
└── Reasoning Traces (Optional)
    ├── Archived solutions (successful reasoning paths)
    ├── Confidence scores (verified patterns)
    └── Failure modes (negative knowledge)
```

**Use Cases**:
1. **Save AI State**: Export current Galaxy Universe to House object
2. **Share Insights**: AI A learns something → saves to galaxy box → AI B loads it
3. **Library of Knowledge**: Collection of galaxy boxes = AI's "book collection"
4. **Archival**: Long-term storage of learned patterns (like human library)

---

## Loading Stage Core Components

### 1. LoadingStage Manager

**Purpose**: Manage persistent GPU memory arena and unified PTX context

```python
class LoadingStage:
    """
    Persistent GPU memory arena with unified PTX context.
    Solves CUDA context switching problem.
    Inspired by game engine memory management.
    """

    def __init__(self, gpu_id=0, reserve_percent=30):
        """
        Initialize Loading Stage with persistent PTX context.

        Args:
            gpu_id: GPU device ID (default 0)
            reserve_percent: Percentage of VRAM to reserve for OS (default 30%)
        """
        # Query GPU capacity
        self.gpu_id = gpu_id
        self.total_vram_gb = self._query_gpu_vram()
        self.reserve_gb = self.total_vram_gb * (reserve_percent / 100)
        self.arena_size_gb = self.total_vram_gb - self.reserve_gb

        print(f"[LoadingStage] GPU {gpu_id}: {self.total_vram_gb:.2f}GB total")
        print(f"[LoadingStage] Arena: {self.arena_size_gb:.2f}GB (Loading Stage)")
        print(f"[LoadingStage] Reserved: {self.reserve_gb:.2f}GB (OS + desktop)")

        # Create ONE persistent PTX context (THE KEY SOLUTION)
        self.ptx_context = self._create_persistent_context()

        # Allocate regions within arena
        self.regions = {
            'kernels': MemoryRegion(size_mb=100, name='PTX Kernels'),
            'galaxy': MemoryRegion(size_mb=2048, name='Galaxy Universe'),
            'house': MemoryRegion(size_mb=3072, name='House Context'),
            'world': MemoryRegion(size_mb=3276, name='World View Buffer')
        }

        # Managers
        self.galaxy_manager = GalaxyManager(self.regions['galaxy'])
        self.house_manager = HouseManager(self.regions['house'])
        self.door_manager = DoorManager(self.regions['world'])
        self.lod_manager = LODManager()

        # Object cache (lazy loading)
        self.loaded_objects = {}  # URI → components
        self.lru_cache = LRUCache(max_size_mb=self.regions['house'].size_mb)

    def _create_persistent_context(self):
        """
        Create ONE persistent PTX context for ALL operations.
        This is THE solution to context switching conflicts.
        """
        import ctypes
        from knowledge3d.cranium.sovereign import loader

        # Initialize sovereign loader with persistent context
        sovereign = loader.SovereignRPNEngine()

        # This context will be used for ALL operations:
        # - PTX kernel execution
        # - Galaxy operations
        # - House object loading
        # - Remote streaming (Doors)

        print("[LoadingStage] ✅ Persistent PTX context created")
        print("[LoadingStage] All operations will execute in SAME context")
        print("[LoadingStage] No more context switching conflicts!")

        return sovereign.context

    def load_object(self, object_uri, components=None):
        """
        Load House object into Loading Stage (lazy on-demand).
        Like web browser: download what you need, cache locally.

        Args:
            object_uri: Local path or network URL
                - Local: "house://local/library/calculus_book.glb"
                - Remote: "door://example.com/shared/physics_sim.glb"
            components: ['3d', 'ai_data', 'metadata', 'programs']
                If None, load all components

        Returns:
            dict with requested components
        """
        # Check cache first (LRU)
        if object_uri in self.loaded_objects:
            self.lru_cache.touch(object_uri)  # Update access time
            return self.loaded_objects[object_uri]

        # Not cached - load from source
        if object_uri.startswith('door://'):
            # Remote house (network via Doors protocol)
            obj_data = self.door_manager.fetch(object_uri)
        else:
            # Local house (file system)
            obj_data = self._load_local_glb(object_uri)

        # Extract components (lazy - only what's needed)
        extracted = {}

        if not components or '3d' in components:
            extracted['3d'] = self._extract_3d_mesh(obj_data)

        if not components or 'ai_data' in components:
            extracted['ai_data'] = self._extract_galaxy_box(obj_data)
            if extracted['ai_data']:
                # Deserialize galaxy snapshot into Region 2 (Galaxy Universe)
                self.galaxy_manager.load_snapshot(extracted['ai_data'])

        if not components or 'metadata' in components:
            extracted['metadata'] = self._extract_rdf(obj_data)

        if not components or 'programs' in components:
            extracted['programs'] = self._extract_rpn(obj_data)

        # Cache in Region 3 (House Context)
        if not self.lru_cache.has_space_for(obj_data):
            # Evict LRU to make room
            self.lru_cache.evict_lru()

        self.loaded_objects[object_uri] = extracted
        self.lru_cache.add(object_uri, obj_data)

        return extracted

    def _extract_galaxy_box(self, obj_data):
        """
        Extract AI data (galaxy box) from House object.

        Galaxy boxes are stored in glTF extras.k3d_ai_data:
        {
          "format": "compressed" | "raw" | "procedural",
          "compression": "zstd" | "lz4" | null,
          "payload_uri": "data/galaxy_box.bin",
          "contents": {
            "type": "galaxy_snapshot",
            "galaxies": ["Math", "Grammar"],
            ...
          }
        }
        """
        # Parse glTF extras
        if 'extras' not in obj_data:
            return None

        ai_data = obj_data['extras'].get('k3d_ai_data')
        if not ai_data:
            return None

        # Load payload (from URI or inline)
        if 'payload_uri' in ai_data:
            payload = self._load_binary(ai_data['payload_uri'])
        else:
            payload = ai_data['payload']

        # Decompress if needed
        if ai_data['format'] == 'compressed':
            import zstandard as zstd
            payload = zstd.decompress(payload)
        elif ai_data['format'] == 'procedural':
            # RPN program generates data on demand
            payload = self.rpn_engine.execute(ai_data['generator_program'])

        return {
            'metadata': ai_data['contents'],
            'data': payload
        }
```

### 2. GalaxyManager (Working Memory)

**Purpose**: Manage Galaxy Universe (Region 2) - AI's active reasoning state

```python
class GalaxyManager:
    """
    Manages Galaxy Universe within Loading Stage.
    Galaxy = AI's working memory (hot cache).
    """

    def __init__(self, region):
        self.region = region
        self.loaded_galaxies = {}  # name → galaxy instance
        self.access_times = {}  # name → last access timestamp

    def load_galaxy(self, galaxy_name, lod_level='high'):
        """
        Load galaxy into Region 2 (Galaxy Universe).

        Args:
            galaxy_name: 'Drawing', 'Math', 'Grammar', etc.
            lod_level: 'high', 'medium', 'low' (detail level)
        """
        if galaxy_name in self.loaded_galaxies:
            self.access_times[galaxy_name] = time.time()
            return self.loaded_galaxies[galaxy_name]

        # Check space
        if not self.region.has_space_for(galaxy_name):
            self._evict_lru_galaxy()

        # Load into Region 2 (within persistent PTX context)
        galaxy = self._load_galaxy_vram(galaxy_name, lod_level)
        self.loaded_galaxies[galaxy_name] = galaxy
        self.access_times[galaxy_name] = time.time()

        return galaxy

    def load_snapshot(self, galaxy_box_data):
        """
        Load galaxy snapshot from House object (galaxy box).
        Deserializes AI state into Region 2 (Galaxy Universe).
        """
        metadata = galaxy_box_data['metadata']
        data = galaxy_box_data['data']

        # Deserialize each galaxy
        for galaxy_name in metadata['galaxies']:
            galaxy_state = self._deserialize_galaxy(data, galaxy_name)
            self.loaded_galaxies[galaxy_name] = galaxy_state

        print(f"[GalaxyManager] Loaded snapshot: {metadata['galaxies']}")

    def save_snapshot(self, galaxy_names, compression='zstd'):
        """
        Save current Galaxy Universe state to serialized format.
        Can be stored in House object (galaxy box) for persistence.
        """
        snapshot = {
            'metadata': {
                'type': 'galaxy_snapshot',
                'galaxies': galaxy_names,
                'timestamp': time.time()
            },
            'data': {}
        }

        # Serialize each galaxy
        for name in galaxy_names:
            if name in self.loaded_galaxies:
                snapshot['data'][name] = self._serialize_galaxy(
                    self.loaded_galaxies[name]
                )

        # Compress if requested
        if compression == 'zstd':
            import zstandard as zstd
            cctx = zstd.ZstdCompressor(level=10)
            snapshot['data'] = cctx.compress(
                json.dumps(snapshot['data']).encode()
            )
            snapshot['metadata']['compression'] = 'zstd'

        return snapshot
```

### 3. HouseManager (Persistent Storage Cache)

**Purpose**: Manage House Context (Region 3) - cached House objects

```python
class HouseManager:
    """
    Manages House Context within Loading Stage.
    House = persistent storage cached in VRAM (Region 3).
    """

    def __init__(self, region):
        self.region = region
        self.loaded_houses = {}  # URI → house data
        self.lod_cache = {}  # URI → {low, medium, high} LOD levels

    def load_house(self, house_uri, lod_level='high'):
        """
        Load House object into Region 3 with LOD.

        LOD Levels:
        - 'low': Centroid + metadata only (minimal memory)
        - 'medium': Simplified mesh + essential data
        - 'high': Full detail mesh + all data
        """
        cache_key = f"{house_uri}:{lod_level}"

        if cache_key in self.lod_cache:
            return self.lod_cache[cache_key]

        # Load from source
        house_data = self._load_house_data(house_uri)

        # Apply LOD
        if lod_level == 'low':
            house_lod = self._extract_centroid(house_data)
        elif lod_level == 'medium':
            house_lod = self._simplify_mesh(house_data, target_ratio=0.3)
        else:  # high
            house_lod = house_data

        # Cache in Region 3
        self.lod_cache[cache_key] = house_lod
        return house_lod
```

### 4. LODManager (Semantic Proximity)

**Purpose**: Determine LOD levels based on semantic + spatial proximity

```python
class LODManager:
    """
    Level-of-Detail manager using semantic proximity.
    Not just spatial distance - semantic relevance matters!
    Inspired by game engines but extended for knowledge.
    """

    def get_lod_level(self, object_data, focus_point, focus_semantic):
        """
        Determine LOD based on:
        1. Spatial distance (FOV - field of view)
        2. Semantic proximity (is it related to current reasoning?)
        3. Recency (recently used = keep higher LOD)

        Args:
            object_data: Object metadata (position, embedding)
            focus_point: Current spatial focus (x, y, z)
            focus_semantic: Current reasoning context (embedding)

        Returns:
            'high', 'medium', or 'low'
        """
        # Spatial distance (Euclidean in 3D space)
        spatial_dist = np.linalg.norm(
            object_data['position'] - focus_point
        )

        # Semantic distance (cosine similarity)
        semantic_sim = np.dot(
            object_data['embedding'],
            focus_semantic
        ) / (np.linalg.norm(object_data['embedding']) *
             np.linalg.norm(focus_semantic))
        semantic_dist = 1 - semantic_sim  # Convert to distance

        # Recency weight (time since last access)
        recency_weight = time.time() - object_data.get('last_access', 0)
        recency_weight = min(recency_weight / 3600, 1.0)  # Normalize to [0,1]

        # Combined score (weighted heuristic)
        score = (0.4 * spatial_dist +
                 0.4 * semantic_dist +
                 0.2 * recency_weight)

        # Determine LOD
        if score < 0.3:
            return 'high'    # Very relevant - full detail
        elif score < 0.7:
            return 'medium'  # Somewhat relevant - reduced detail
        else:
            return 'low'     # Not relevant - centroid only
```

### 5. DoorManager (Network Streaming)

**Purpose**: Handle remote House loading via "Doors" protocol

```python
class DoorManager:
    """
    Manages network streaming of remote Houses.
    "Doors" = network protocol (like online game server).
    """

    def __init__(self, region):
        self.region = region
        self.active_streams = {}  # URI → download progress

    def fetch(self, remote_uri):
        """
        Fetch remote House via Doors protocol.

        Args:
            remote_uri: "door://example.com/shared/physics_sim.glb"

        Returns:
            House data (loaded into Region 4 - World View Buffer)
        """
        # Parse door:// URI
        parsed = urllib.parse.urlparse(remote_uri)
        host = parsed.netloc
        path = parsed.path

        # Fetch manifest first (list of objects + LODs)
        manifest = self._fetch_manifest(host, path)

        # Priority queue (immediate area first)
        priority_objects = self._prioritize(manifest)

        # Async download (don't block reasoning)
        for obj in priority_objects:
            self._async_download(obj, host)

        return manifest

    def _async_download(self, obj, host):
        """
        Download object in background (async streaming).
        Like online game: load world chunks while playing.
        """
        # Start background thread/task
        # Load into Region 4 (World View Buffer)
        # Update progress tracking
        pass
```

---

## Integration with Existing Infrastructure

### 1. Sovereign Loader Integration

**Existing**: `knowledge3d/cranium/sovereign/loader.py`
- Creates PTX context on demand
- Handles GPU memory allocation
- Provides ctypes CUDA API wrappers

**Integration**:
```python
# OLD (multiple contexts - causes conflicts)
def execute_kernel():
    context = create_cuda_context()  # New context each time
    run_ptx_kernel(context)
    destroy_cuda_context(context)

# NEW (unified Loading Stage context)
class LoadingStage:
    def __init__(self):
        # Create ONE persistent context (used by sovereign loader)
        self.ptx_context = SovereignRPNEngine().context

    def execute_kernel(self, kernel_name):
        # Use persistent context (no creation/destruction)
        run_ptx_kernel(self.ptx_context, kernel_name)
```

**Changes Needed**:
1. Modify sovereign loader to accept existing context (don't always create new)
2. Add context reuse flag: `SovereignRPNEngine(reuse_context=True)`
3. Expose context handle for LoadingStage to use

### 2. Galaxy Universe Integration

**Existing**: Multiple galaxy implementations
- `knowledge3d/cranium/reality_galaxy.py`
- `knowledge3d/ingestion/atomic/drawing_grammar_builder.py`
- `knowledge3d/training/arc_agi/grammar_galaxy.py`

**Integration**:
```python
# Galaxies now load into LoadingStage Region 2
loading_stage = LoadingStage()

# Load Drawing Galaxy
drawing_galaxy = loading_stage.galaxy_manager.load_galaxy('Drawing')

# Load Math Galaxy
math_galaxy = loading_stage.galaxy_manager.load_galaxy('Math')

# All galaxies share SAME PTX context (no conflicts!)
```

### 3. House Layer Integration

**Existing**: glTF/GLB loading
- `viewer/src/loadK3D.ts` (TypeScript viewer)
- House storage in `/K3D/Knowledge3D.local/house/`

**Integration**:
```python
# House objects now load into LoadingStage Region 3
loading_stage = LoadingStage()

# Load local house
book = loading_stage.load_object('house://local/library/calculus.glb')

# Load remote house (via Doors)
remote = loading_stage.load_object('door://example.com/shared/physics.glb')

# Both use SAME PTX context (seamless)
```

---

## Testing Strategy

### 1. Memory Allocation Tests

```python
def test_loading_stage_allocation():
    """Test Loading Stage allocates correct VRAM percentage."""
    ls = LoadingStage(reserve_percent=30)

    # Check allocation (70% of GPU)
    assert ls.arena_size_gb == ls.total_vram_gb * 0.7
    assert ls.reserve_gb == ls.total_vram_gb * 0.3

    # Check regions sum to arena size
    total_region_mb = sum(r.size_mb for r in ls.regions.values())
    assert abs(total_region_mb - ls.arena_size_gb * 1024) < 10  # Within 10MB

def test_persistent_context():
    """Test ONE context used for all operations."""
    ls = LoadingStage()

    # Execute multiple operations
    context_ids = []
    for i in range(10):
        ctx_id = ls.ptx_context.handle
        context_ids.append(ctx_id)

    # All operations use SAME context
    assert len(set(context_ids)) == 1, "Multiple contexts detected!"
```

### 2. Object Loading Tests

```python
def test_lazy_loading():
    """Test objects load on-demand (lazy)."""
    ls = LoadingStage()

    # Object not loaded initially
    assert 'house://test/book.glb' not in ls.loaded_objects

    # Load object
    obj = ls.load_object('house://test/book.glb')

    # Now cached
    assert 'house://test/book.glb' in ls.loaded_objects

    # Second load is instant (cached)
    import time
    start = time.time()
    obj2 = ls.load_object('house://test/book.glb')
    duration = time.time() - start

    assert duration < 0.001, "Cache miss (should be instant)"
    assert obj is obj2, "Different object returned (cache miss)"

def test_galaxy_box_extraction():
    """Test galaxy boxes extract AI data correctly."""
    ls = LoadingStage()

    # Load object with galaxy box
    obj = ls.load_object('house://test/ai_book.glb', components=['ai_data'])

    # Check AI data extracted
    assert 'ai_data' in obj
    assert obj['ai_data']['metadata']['type'] == 'galaxy_snapshot'

    # Check galaxy loaded into Region 2
    assert 'Math' in ls.galaxy_manager.loaded_galaxies
```

### 3. LOD/FOV Tests

```python
def test_lod_semantic_proximity():
    """Test LOD adjusts based on semantic relevance."""
    lod_mgr = LODManager()

    # Object semantically close
    obj_relevant = {
        'position': np.array([1, 1, 1]),
        'embedding': np.array([0.5, 0.5, 0.5]),
        'last_access': time.time()
    }
    focus_point = np.array([1, 1, 1])
    focus_semantic = np.array([0.5, 0.5, 0.5])

    lod = lod_mgr.get_lod_level(obj_relevant, focus_point, focus_semantic)
    assert lod == 'high', "Should be high LOD (semantically relevant)"

    # Object semantically far
    obj_irrelevant = {
        'position': np.array([100, 100, 100]),
        'embedding': np.array([-0.5, -0.5, -0.5]),
        'last_access': 0
    }

    lod = lod_mgr.get_lod_level(obj_irrelevant, focus_point, focus_semantic)
    assert lod == 'low', "Should be low LOD (semantically irrelevant)"
```

### 4. Network Streaming Tests (Doors)

```python
def test_remote_loading():
    """Test remote House loading via Doors protocol."""
    ls = LoadingStage()

    # Mock remote server
    with mock_door_server('door://test.com/house.glb'):
        obj = ls.load_object('door://test.com/house.glb')

    # Check loaded into Region 4 (World View Buffer)
    assert obj in ls.door_manager.active_streams

def test_async_streaming():
    """Test async loading doesn't block reasoning."""
    ls = LoadingStage()

    # Start async download (large file)
    ls.door_manager.fetch('door://test.com/large_house.glb')

    # Can still execute operations immediately
    result = ls.galaxy_manager.load_galaxy('Math')
    assert result is not None, "Reasoning blocked by download!"
```

### 5. Sovereignty Tests

```python
def test_no_context_switching():
    """Test no CUDA context switching occurs."""
    ls = LoadingStage()

    # Track context switches
    context_switches = 0
    original_create = cuda.cuCtxCreate

    def track_create(*args):
        nonlocal context_switches
        context_switches += 1
        return original_create(*args)

    cuda.cuCtxCreate = track_create

    # Execute many operations
    for i in range(100):
        ls.galaxy_manager.load_galaxy('Math')
        ls.house_manager.load_house('house://test/book.glb')
        ls.execute_kernel('some_kernel')

    # Only ONE context created (at initialization)
    assert context_switches == 1, f"Context switched {context_switches} times!"
```

---

## Success Criteria

### Phase 1: Core Infrastructure (Week 1)
- [ ] LoadingStage class implemented
- [ ] Persistent PTX context creation
- [ ] Region allocation (kernels, galaxy, house, world)
- [ ] Basic object loading (local glTF files)
- [ ] Tests: memory allocation, persistent context

### Phase 2: Galaxy Integration (Week 2)
- [ ] GalaxyManager implemented
- [ ] Galaxy loading into Region 2
- [ ] Galaxy box serialization/deserialization
- [ ] LRU eviction policy
- [ ] Tests: galaxy loading, snapshots, eviction

### Phase 3: LOD/FOV System (Week 3)
- [ ] LODManager implemented
- [ ] Semantic proximity calculation
- [ ] LOD level determination (high/medium/low)
- [ ] Dynamic LOD adjustment
- [ ] Tests: semantic proximity, LOD switching

### Phase 4: Doors Protocol (Week 4)
- [ ] DoorManager implemented
- [ ] Remote manifest fetching
- [ ] Async streaming (background loading)
- [ ] Priority queue (immediate area first)
- [ ] Tests: network loading, async streaming

### Phase 5: Integration & Validation (Week 5)
- [ ] Sovereign loader integration (context reuse)
- [ ] Galaxy Universe integration (all galaxies)
- [ ] House layer integration (glTF loading)
- [ ] End-to-end test (load remote house with galaxy box)
- [ ] Sovereignty test (no context switching)
- [ ] Performance benchmarks (loading times, memory usage)

---

## Performance Targets

| Metric | Target | Justification |
|--------|--------|---------------|
| **Context Creation** | 1 (at startup only) | Persistent context - no switching |
| **Object Load (cached)** | <1ms | LRU cache hit (VRAM access) |
| **Object Load (uncached)** | <100ms | Local glTF load + extraction |
| **Galaxy Load** | <500ms | Deserialize + upload to Region 2 |
| **LOD Switch** | <10ms | Re-mesh + upload (low → high) |
| **Remote Fetch (manifest)** | <200ms | Network round-trip + parse |
| **Memory Overhead** | <5% | Arena metadata + cache structures |

---

## Future Enhancements (Post-MVP)

### 1. Smart Prefetching
- Predict next objects based on current focus
- Preload in background (Region 4)
- Use TRM to guide prefetch decisions

### 2. Compression Optimization
- Evaluate compression algorithms (zstd, lz4, lzma)
- Adaptive compression (high LOD = less compression)
- Procedural generation (RPN programs as compressed data)

### 3. Multi-GPU Support
- Distribute regions across multiple GPUs
- Galaxy on GPU 0, House on GPU 1
- Unified context via NVLink/PCIe

### 4. Collaboration Features
- Shared World View Buffer (multi-user)
- Real-time synchronization (websockets)
- Conflict resolution (CRDTs for concurrent edits)

---

## Architectural Assessment

**Why This Design is Sound:**

1. **Proven Pattern**: Game engines use memory arenas + streaming for decades
2. **Unified Context**: Solves CUDA switching problem elegantly
3. **Lazy Loading**: Only load what's needed (efficiency)
4. **Semantic LOD**: Extends game engine LOD with AI reasoning
5. **Dual-Client**: Humans inspect AI working memory (transparency)
6. **Sovereignty**: PTX-only hot path maintained
7. **Scalability**: Doors protocol enables distributed houses

**Risks & Mitigations:**

| Risk | Mitigation |
|------|------------|
| **Memory pressure** (70% VRAM aggressive) | LRU eviction + dynamic expansion |
| **Serialization overhead** (galaxy boxes large) | Compression (zstd) + procedural generation |
| **Network latency** (Doors slow) | Async streaming + priority queue + prefetch |
| **LOD heuristics expensive** | Cache LOD decisions + spatial indexing |

---

## Conclusion

The Loading Stage architecture provides a **foundational solution** to the CUDA context switching problem while enabling **seamless Galaxy+House unification**. By adopting proven game engine patterns (memory arenas, streaming, LOD) and extending them with AI-specific features (semantic proximity, galaxy boxes), we achieve a system that is:

- ✅ **Efficient**: Lazy loading, LRU eviction, LOD optimization
- ✅ **Scalable**: Doors protocol for distributed houses
- ✅ **Transparent**: Dual-client inspectability (humans see AI's thoughts)
- ✅ **Sovereign**: PTX-only hot path (no framework overhead)
- ✅ **Production-Ready**: Based on battle-tested patterns

**This architecture is ready for Codex implementation.**

---

**Document Version**: 1.0
**Status**: 🚀 **READY FOR IMPLEMENTATION**
**Next Step**: Hand to Codex for Phase 1 (Core Infrastructure)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
