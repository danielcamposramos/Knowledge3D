# Phase H2: GPU Mesh Kernels + House Room Templates

**Date**: March 16, 2026
**From**: Claude (Architecture Partner)
**To**: Codex (Implementation)
**Priority**: CRITICAL — GPU kernels are the foundation, host-side is scaffolding
**Depends on**: Phase H1 (host-side mesh surface) — COMPLETE

---

## Directive

Phase H1 delivered a working host-side mesh construction surface. Now we need **real GPU kernels** backing these operations and **House room templates** that compose from them.

Two parallel tracks:
- **Track A**: Compile mesh_generators.cu → PTX, wire through a sovereign bridge, make every mesh generator callable on GPU
- **Track B**: Define House rooms as meaning-centric stars with `visual_rpn` construction programs using H1 opcodes

Daniel's mandate: **real opcodes in real kernels.** The host-side Python code from H1 stays as CPU fallback for ingestion/tests, but the sovereign path must go through GPU.

---

## Track A: GPU Mesh Kernel Pipeline

### A1. Expand mesh_generators.cu with All Shape Kernels

**File**: `knowledge3d/cranium/kernels/mesh_generators.cu`

The current file has `generate_uv_sphere_vertices` and `compute_face_normals`. Add kernels for ALL H1 generators:

```c
// Already exists:
__global__ void generate_uv_sphere_vertices(float* vertices, float* normals, float* uvs, float radius, int stacks, int slices);
__global__ void compute_face_normals(const float* vertices, const unsigned int* indices, float* face_normals, int triangle_count);

// ADD these kernels:
__global__ void generate_plane_vertices(float* vertices, float* normals, float* uvs, float width, float depth, int segments_w, int segments_d);
__global__ void generate_cube_vertices(float* vertices, float* normals, float* uvs, unsigned int* indices, float size);
__global__ void generate_cylinder_vertices(float* vertices, float* normals, float* uvs, float radius, float height, int segments);
__global__ void generate_cone_vertices(float* vertices, float* normals, float* uvs, float radius, float height, int segments);
__global__ void generate_torus_vertices(float* vertices, float* normals, float* uvs, float major_r, float minor_r, int major_seg, int minor_seg);
__global__ void generate_icosphere_vertices(float* vertices, float* normals, float* uvs, float radius, int subdivisions);
__global__ void mat4_transform_vertices(float* vertices, float* normals, const float* matrix, int vertex_count);
__global__ void generate_index_buffer_grid(unsigned int* indices, int rows, int cols);
```

**Implementation notes per kernel:**

**generate_plane_vertices**: One thread per vertex. Thread index → (row, col) → vertex position, normal (0,1,0), UV. Standard grid pattern. Grid dimension = `(segments_w + 1) * (segments_d + 1)` vertices.

**generate_cube_vertices**: 24 vertices (4 per face × 6 faces for per-face normals). Can be a single-warp kernel (24 threads) or table-driven. Each thread writes position + normal + UV from lookup tables. Index buffer is constant (36 indices for 12 triangles) — can be a device constant.

**generate_cylinder_vertices**: One thread per vertex on the side surface. `segments × 2` vertices (top ring + bottom ring). Caps added separately (fan pattern). Same trig pattern as UV sphere but with fixed Y.

**generate_cone_vertices**: Like cylinder but top ring collapses to apex. One thread per base ring vertex + apex.

**generate_torus_vertices**: One thread per vertex. Thread index → (major_seg, minor_seg) → nested trig. `major_segments × minor_segments` vertices total. Same pattern as the host-side `generate_torus()`.

**generate_icosphere_vertices**: This one is trickier on GPU due to recursive subdivision. Two approaches:
1. **Recommended**: Pre-compute icosahedron + subdivision on host (fast, 12→42→162→642 vertices), upload vertex buffer to GPU, then use `mat4_transform_vertices` kernel for transforms. The subdivision is a one-time operation.
2. **Alternative**: Iterative subdivision kernel with shared memory for midpoint cache. Only worth it if we generate many icospheres per frame.

**mat4_transform_vertices**: One thread per vertex. Reads 4×4 matrix from constant memory, applies to each vertex position and normal. This is the GPU version of `MAT4_APPLY`. High-frequency operation — every placed object uses this.

**generate_index_buffer_grid**: One thread per quad cell. Writes 6 indices (2 triangles) per cell. Used by plane, sphere, cylinder side surfaces.

### A2. Compile to PTX

**Build step** (add to existing compilation):
```bash
nvcc -ptx -arch=sm_86 \
  knowledge3d/cranium/kernels/mesh_generators.cu \
  -o knowledge3d/cranium/ptx/mesh_generators.ptx
```

Use the same `sm_` target as other kernels in the project (check existing Makefile/build script). The RTX 3070 is sm_86.

### A3. Sovereign Mesh Bridge

**File**: `knowledge3d/cranium/bridges/sovereign_mesh_bridge.py`

Follow the exact pattern of `ProceduralDrawingBridge` and `nine_chain_swarm_bridge.py`:

```python
from knowledge3d.cranium.sovereign import loader

class SovereignMeshBridge:
    """GPU-backed mesh generation via mesh_generators.ptx."""

    def __init__(self):
        ptx_path = Path(__file__).parent.parent / "ptx" / "mesh_generators.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(
                f"Mesh generators PTX not found: {ptx_path}. Compile with:\n"
                "  nvcc -ptx -arch=sm_86 "
                "knowledge3d/cranium/kernels/mesh_generators.cu "
                "-o knowledge3d/cranium/ptx/mesh_generators.ptx"
            )
        self._module = loader.load_module_from_file(str(ptx_path))
        self._kernels = {
            "uv_sphere": loader.get_function(self._module, "generate_uv_sphere_vertices"),
            "face_normals": loader.get_function(self._module, "compute_face_normals"),
            "plane": loader.get_function(self._module, "generate_plane_vertices"),
            "cube": loader.get_function(self._module, "generate_cube_vertices"),
            "cylinder": loader.get_function(self._module, "generate_cylinder_vertices"),
            "cone": loader.get_function(self._module, "generate_cone_vertices"),
            "torus": loader.get_function(self._module, "generate_torus_vertices"),
            "transform": loader.get_function(self._module, "mat4_transform_vertices"),
            "grid_indices": loader.get_function(self._module, "generate_index_buffer_grid"),
        }
        # Pre-allocate GPU buffers (max 65536 vertices)
        self._max_vertices = 65536
        self._d_vertices = loader.gpu_malloc(self._max_vertices * 3 * 4)  # float3
        self._d_normals = loader.gpu_malloc(self._max_vertices * 3 * 4)
        self._d_uvs = loader.gpu_malloc(self._max_vertices * 2 * 4)
        self._d_indices = loader.gpu_malloc(self._max_vertices * 6 * 4)  # max 6 indices per vertex
        self._d_matrix = loader.gpu_malloc(16 * 4)  # one mat4

    def generate_uv_sphere(self, radius, stacks, slices) -> MeshBuffer:
        """Generate UV sphere entirely on GPU, return host MeshBuffer."""
        ...  # launch kernel, copy back, construct MeshBuffer

    def generate_cube(self, size) -> MeshBuffer:
        ...

    def generate_plane(self, width, depth, seg_w, seg_d) -> MeshBuffer:
        ...

    def generate_cylinder(self, radius, height, segments, caps) -> MeshBuffer:
        ...

    def transform_mesh(self, mesh: MeshBuffer, matrix: Matrix4) -> MeshBuffer:
        """Upload mesh to GPU, apply mat4 transform kernel, copy back."""
        ...
```

**Key pattern**: Upload parameters → launch kernel → copy vertex/normal/UV/index buffers back → construct MeshBuffer. Same lifecycle as ProceduralDrawingBridge's rasterize path.

**Fallback**: If PTX not compiled (no GPU), fall back to host-side `mesh_opcodes.py` functions. This means tests work on CPU-only machines (like Codex sandbox) while GPU machines get sovereign execution.

### A4. Wire into MeshBridge (Automatic GPU/CPU Selection)

**File**: Modify `knowledge3d/cranium/bridges/mesh_bridge.py`

```python
class MeshBridge:
    def __init__(self):
        self.engine = MeshRPNEngine()  # CPU fallback (existing)
        self._sovereign: SovereignMeshBridge | None = None
        try:
            from .sovereign_mesh_bridge import SovereignMeshBridge
            self._sovereign = SovereignMeshBridge()
        except Exception:
            pass  # No GPU — use host engine

    def execute_rpn_program(self, program: str) -> MeshRenderResult:
        if self._sovereign is not None:
            return self._execute_on_gpu(program)
        return self._execute_on_cpu(program)
```

This gives us: GPU when available, CPU when not, same API. Same pattern the rest of K3D uses.

### A5. Tests for GPU Path

**File**: `tests/test_mesh_gpu.py`

```python
@pytest.mark.skipif(not has_cuda(), reason="No CUDA GPU")
def test_gpu_sphere_matches_cpu_sphere():
    cpu_mesh = generate_uv_sphere(1.0, 12, 16)
    gpu_mesh = SovereignMeshBridge().generate_uv_sphere(1.0, 12, 16)
    assert len(cpu_mesh.vertices) == len(gpu_mesh.vertices)
    for cpu_v, gpu_v in zip(cpu_mesh.vertices, gpu_mesh.vertices):
        assert cpu_v == pytest.approx(gpu_v, abs=1e-5)

@pytest.mark.skipif(not has_cuda(), reason="No CUDA GPU")
def test_gpu_transform_matches_cpu_transform():
    ...

@pytest.mark.skipif(not has_cuda(), reason="No CUDA GPU")
def test_gpu_cube_correct_topology():
    ...
```

Test invariant: GPU output must match CPU output within floating-point tolerance. This ensures the kernels are correct AND gives us a safety net for future kernel changes.

---

## Track B: House Room Templates as Meaning-Centric Stars

### B1. Room Template Stars

**File**: `knowledge3d/knowledgeverse/house_rooms.py`

Define the foundational House rooms as meaning-centric stars. Each room has a `visual_rpn` that constructs it using H1 mesh opcodes. The TRM will learn these programs and eventually compose variations.

```python
HOUSE_ROOMS: list[MeaningCentricStar] = [
    # The Library — where knowledge books live
    MeaningCentricStar(
        star_id="room_library",
        meaning_class="room",
        meaning_rpn="ROOM KNOWLEDGE BOOKS READING DOMAIN_CENTER",
        domain="House/Library",
        visual_rpn=(
            # Outer shell: 8×4×6 room
            "MESH_BEGIN 8.0 GEN_CUBE "
            "7.6 GEN_CUBE CSG_SUBTRACT "  # Hollow out
            # Door opening on front wall
            "MESH_BEGIN 1.2 GEN_CUBE "
            "0.5 1.0 0.3 MAT4_SCALE MAT4_APPLY "
            "4.0 0.0 0.0 MAT4_TRANSLATE MAT4_APPLY MESH_END "
            "CSG_SUBTRACT MESH_END"
        ),
        behavior_rpn="ROOM_ENTER LOAD_KNOWLEDGE_DOMAIN Library ACTIVATE_SHELVES",
        surface_forms={
            "en": SurfaceForm(word_ref="seed_word_en_library", char_refs=["char_l", ...]),
            "pt": SurfaceForm(word_ref="seed_word_pt_biblioteca", char_refs=[...]),
            "ja": SurfaceForm(word_ref="seed_word_ja_toshokan", char_refs=[...]),
        },
        house_position=(0.0, 0.0, 0.0),
        house_room="House/Library",
        confidence=1,
        polarity=1,
        taxonomy_refs=["concept_language", "concept_mathematics"],
        component_refs=["furniture_bookshelf", "furniture_desk", "furniture_chair"],
    ),

    # The Knowledge Garden — ontological trees with branches and leaves
    MeaningCentricStar(
        star_id="room_garden",
        meaning_class="room",
        meaning_rpn="ROOM GROWTH ONTOLOGY TREES EXPLORATION DOMAIN_CENTER",
        domain="House/Garden",
        visual_rpn=(
            # Open-air garden: ground plane + boundary walls (low)
            "MESH_BEGIN 20.0 20.0 4 4 GEN_PLANE "
            # Low perimeter walls
            "MESH_BEGIN 20.0 GEN_CUBE 19.0 GEN_CUBE CSG_SUBTRACT "
            "1.0 0.05 1.0 MAT4_SCALE MAT4_APPLY MESH_END "
            "CSG_UNION MESH_END"
        ),
        behavior_rpn="ROOM_ENTER LOAD_KNOWLEDGE_DOMAIN Ontology GROW_TREES",
        ...
    ),

    # The Workshop — tools the AI uses
    MeaningCentricStar(
        star_id="room_workshop",
        meaning_class="room",
        meaning_rpn="ROOM TOOLS CONSTRUCTION BUILD APPLY DOMAIN_CENTER",
        domain="House/Workshop",
        visual_rpn=(...),  # Room with workbench, tool racks
        ...
    ),

    # The Gallery — visual art and audio
    MeaningCentricStar(
        star_id="room_gallery",
        meaning_class="room",
        meaning_rpn="ROOM VISUAL AUDIO ART PERCEPTION DOMAIN_CENTER",
        domain="House/Gallery",
        visual_rpn=(...),  # Open hall with display walls
        ...
    ),

    # The Bathtub/Observatory — introspection portal
    MeaningCentricStar(
        star_id="room_bathtub",
        meaning_class="room",
        meaning_rpn="ROOM INTROSPECTION GALAXY META OBSERVE DOMAIN_CENTER",
        domain="House/Bathtub",
        visual_rpn=(...),  # Small room with central basin (the bathtub/portal)
        behavior_rpn="ROOM_ENTER INTROSPECT_MODE ACTIVATE_GALAXY_VIEW",
        ...
    ),
]
```

**Rooms to define** (minimum viable House):

| Room | Archetype | visual_rpn Pattern | Seed Stars Placed Here |
|------|-----------|-------------------|----------------------|
| **Library** | Enclosed room with shelves | CSG hollow cube + door | concept_mathematics, concept_physics, concept_chemistry, concept_biology, concept_language |
| **Knowledge Garden** | Open ground with low walls | Plane + low perimeter | concept_growth |
| **Workshop** | Enclosed room with workbench | CSG hollow cube + wide door | concept_tool |
| **Gallery** | Open hall with display walls | CSG wide box + archways | concept_visual_art, concept_sound |
| **Bathtub/Observatory** | Small room with central portal | CSG small cube + basin (lathe) | concept_self_reflection |

### B2. Furniture Templates

**File**: `knowledge3d/knowledgeverse/house_furniture.py`

Furniture pieces that populate rooms. Same pattern: meaning-centric stars with `visual_rpn`.

```python
HOUSE_FURNITURE: list[MeaningCentricStar] = [
    MeaningCentricStar(
        star_id="furniture_bookshelf",
        meaning_class="furniture",
        meaning_rpn="SHELF STORAGE BOOKS VERTICAL DOMAIN_CENTER",
        domain="House/Library/Furniture",
        visual_rpn=(
            # Bookshelf: back panel + 5 shelves + 2 side panels
            "MESH_BEGIN "
            # Back panel
            "MESH_BEGIN 1.0 GEN_CUBE "
            "2.0 0.02 0.8 MAT4_SCALE MAT4_APPLY MESH_END "
            # Bottom shelf
            "MESH_BEGIN 1.0 GEN_CUBE "
            "2.0 0.05 0.8 MAT4_SCALE MAT4_APPLY "
            "0.0 -0.75 0.0 MAT4_TRANSLATE MAT4_APPLY MESH_END CSG_UNION "
            # Shelf 2
            "MESH_BEGIN 1.0 GEN_CUBE "
            "2.0 0.05 0.8 MAT4_SCALE MAT4_APPLY "
            "0.0 -0.35 0.0 MAT4_TRANSLATE MAT4_APPLY MESH_END CSG_UNION "
            # Shelf 3
            "MESH_BEGIN 1.0 GEN_CUBE "
            "2.0 0.05 0.8 MAT4_SCALE MAT4_APPLY "
            "0.0 0.05 0.0 MAT4_TRANSLATE MAT4_APPLY MESH_END CSG_UNION "
            # Shelf 4
            "MESH_BEGIN 1.0 GEN_CUBE "
            "2.0 0.05 0.8 MAT4_SCALE MAT4_APPLY "
            "0.0 0.45 0.0 MAT4_TRANSLATE MAT4_APPLY MESH_END CSG_UNION "
            # Top shelf
            "MESH_BEGIN 1.0 GEN_CUBE "
            "2.0 0.05 0.8 MAT4_SCALE MAT4_APPLY "
            "0.0 0.85 0.0 MAT4_TRANSLATE MAT4_APPLY MESH_END CSG_UNION "
            # Left side panel
            "MESH_BEGIN 1.0 GEN_CUBE "
            "0.05 1.8 0.8 MAT4_SCALE MAT4_APPLY "
            "-1.0 0.0 0.0 MAT4_TRANSLATE MAT4_APPLY MESH_END CSG_UNION "
            # Right side panel
            "MESH_BEGIN 1.0 GEN_CUBE "
            "0.05 1.8 0.8 MAT4_SCALE MAT4_APPLY "
            "1.0 0.0 0.0 MAT4_TRANSLATE MAT4_APPLY MESH_END CSG_UNION "
            "MESH_END"
        ),
        house_position=(0.0, 0.0, 0.0),
        house_room="House/Library",
        confidence=1,
        polarity=1,
    ),

    MeaningCentricStar(
        star_id="furniture_desk",
        meaning_class="furniture",
        meaning_rpn="DESK SURFACE WORK HORIZONTAL DOMAIN_CENTER",
        domain="House/Library/Furniture",
        visual_rpn=(
            # Desk: tabletop + 4 legs
            "MESH_BEGIN "
            "MESH_BEGIN 1.0 GEN_CUBE "
            "1.5 0.05 0.8 MAT4_SCALE MAT4_APPLY "
            "0.0 0.75 0.0 MAT4_TRANSLATE MAT4_APPLY MESH_END "
            # Legs (4 cylinders)
            "MESH_BEGIN 0.03 0.75 8 1 GEN_CYLINDER "
            "-0.7 0.375 -0.35 MAT4_TRANSLATE MAT4_APPLY MESH_END CSG_UNION "
            "MESH_BEGIN 0.03 0.75 8 1 GEN_CYLINDER "
            "0.7 0.375 -0.35 MAT4_TRANSLATE MAT4_APPLY MESH_END CSG_UNION "
            "MESH_BEGIN 0.03 0.75 8 1 GEN_CYLINDER "
            "-0.7 0.375 0.35 MAT4_TRANSLATE MAT4_APPLY MESH_END CSG_UNION "
            "MESH_BEGIN 0.03 0.75 8 1 GEN_CYLINDER "
            "0.7 0.375 0.35 MAT4_TRANSLATE MAT4_APPLY MESH_END CSG_UNION "
            "MESH_END"
        ),
        house_position=(3.0, 0.0, 2.0),
        house_room="House/Library",
        ...
    ),

    MeaningCentricStar(
        star_id="furniture_bathtub",
        meaning_class="furniture",
        meaning_rpn="BATHTUB PORTAL INTROSPECTION VESSEL DOMAIN_CENTER",
        domain="House/Bathtub/Furniture",
        visual_rpn=(
            # Bathtub: lathe a basin profile
            "0.6 0.0 MOVE 0.7 0.1 LINE 0.7 0.4 LINE "
            "0.5 0.5 0.3 0.4 QUAD "
            "0.0 0.4 LINE "
            "16 LATHE"
        ),
        house_position=(0.0, 0.0, 0.0),
        house_room="House/Bathtub",
        ...
    ),

    MeaningCentricStar(
        star_id="furniture_knowledge_tree",
        meaning_class="furniture",
        meaning_rpn="TREE ONTOLOGY BRANCHES LEAVES GROWTH DOMAIN_CENTER",
        domain="House/Garden/Furniture",
        visual_rpn=(
            # Tree trunk (cylinder) + canopy (icosphere)
            "MESH_BEGIN "
            "MESH_BEGIN 0.15 2.0 8 1 GEN_CYLINDER MESH_END "
            "MESH_BEGIN 1.5 2 GEN_ICOSPHERE "
            "0.0 2.5 0.0 MAT4_TRANSLATE MAT4_APPLY MESH_END "
            "CSG_UNION MESH_END"
        ),
        house_position=(5.0, 0.0, 5.0),
        house_room="House/Garden",
        ...
    ),
]
```

### B3. Room + Furniture Composition

**File**: `knowledge3d/knowledgeverse/house_builder.py`

Composes rooms + furniture + seed star placement into a complete House.

```python
def build_house(manager: GalaxyManager) -> dict:
    """Compose all room templates + furniture + seed stars into House."""
    # 1. Store room templates
    for room in HOUSE_ROOMS:
        manager.store_meaning_star("House", room)

    # 2. Store furniture templates
    for furniture in HOUSE_FURNITURE:
        manager.store_meaning_star("House", furniture)

    # 3. Place seed stars in their rooms (from seed_stars.py)
    for star in SEED_STARS:
        # star.house_room already set (e.g., "Library/Mathematics")
        manager.store_meaning_star("House", star)

    # 4. Build mesh for each room (execute visual_rpn)
    bridge = MeshBridge()  # Uses GPU if available
    meshes = {}
    for room in HOUSE_ROOMS:
        meshes[room.star_id] = bridge.execute_rpn_program(room.visual_rpn)

    return {"rooms": len(HOUSE_ROOMS), "furniture": len(HOUSE_FURNITURE), "meshes": meshes}
```

### B4. Asset Proceduralizing Pipeline

**File**: Modify existing `glb_decomposer.py` + new script `scripts/proceduralize_assets.py`

Run the GLB decomposer on existing assets to populate the House with real objects:

```bash
# Proceduralize Khronos samples
python -m knowledge3d.tools.training_pipelines.glb_decomposer \
  /K3D/Knowledge3D.local/datasets/gltf_samples/Box.glb \
  --domain Workshop/Assets --output /K3D/Knowledge3D.local/galaxies/House_Assets.jsonl

# Proceduralize curated house objects
python -m knowledge3d.tools.training_pipelines.glb_decomposer \
  /K3D/Knowledge3D.local/datasets/gltf_house/GlamVelvetSofa.glb \
  --domain House/Library/Furniture --output /K3D/Knowledge3D.local/galaxies/House_Assets.jsonl
```

Batch script that processes all available GLBs:

```python
# scripts/proceduralize_assets.py
ASSET_SOURCES = [
    ("/K3D/Knowledge3D.local/datasets/gltf_samples/", "Workshop/Assets"),
    ("/K3D/Knowledge3D.local/datasets/gltf_curated/library/", "House/Library/Furniture"),
    ("/K3D/Knowledge3D.local/datasets/gltf_curated/office/", "House/Workshop/Furniture"),
    ("/K3D/Knowledge3D.local/datasets/gltf_curated/workshop/", "House/Workshop/Tools"),
    ("/K3D/Knowledge3D.local/datasets/gltf_house/", "House/Furniture"),
]
```

---

## Implementation Files Summary

### New Files
1. **`knowledge3d/cranium/bridges/sovereign_mesh_bridge.py`** — GPU mesh bridge (loads PTX, launches kernels, copies buffers)
2. **`knowledge3d/knowledgeverse/house_rooms.py`** — 5 room template stars
3. **`knowledge3d/knowledgeverse/house_furniture.py`** — Furniture template stars (bookshelf, desk, chair, bathtub, knowledge tree, workbench)
4. **`knowledge3d/knowledgeverse/house_builder.py`** — Composition: rooms + furniture + seed stars → House
5. **`scripts/proceduralize_assets.py`** — Batch GLB → star pipeline
6. **`tests/test_mesh_gpu.py`** — GPU mesh kernel correctness (skipif no CUDA)
7. **`tests/test_house_builder.py`** — House composition test (CPU, no GPU needed)

### Modified Files
8. **`knowledge3d/cranium/kernels/mesh_generators.cu`** — Add 7 new kernels (plane, cube, cylinder, cone, torus, transform, grid_indices)
9. **`knowledge3d/cranium/bridges/mesh_bridge.py`** — Auto-select GPU or CPU path
10. **`knowledge3d/knowledgeverse/__init__.py`** — Export new modules

### Build Step
11. **Compile**: `nvcc -ptx -arch=sm_86 knowledge3d/cranium/kernels/mesh_generators.cu -o knowledge3d/cranium/ptx/mesh_generators.ptx`

---

## Constraints

1. **Kernels are REAL** — every `.cu` function must be a working CUDA kernel, not a stub
2. **Bridge follows existing pattern** — use `loader.load_module_from_file()`, `loader.get_function()`, `loader.gpu_malloc()`, `loader.launch_kernel()` from `knowledge3d.cranium.sovereign.loader`
3. **CPU fallback** — if PTX not compiled, `MeshBridge` falls back to host-side `MeshRPNEngine`. Tests must pass either way.
4. **stdlib only** for house_rooms.py, house_furniture.py, house_builder.py — no numpy
5. **No hot-path changes** — benchmark pipeline untouched
6. **visual_rpn programs must be executable** — every room and furniture template `visual_rpn` must produce a valid MeshBuffer when fed to `MeshRPNEngine.evaluate()`
7. **All room/furniture stars must be valid MeaningCentricStar** — pass `to_dict()`/`from_dict()` round-trip, have en/pt/ja surface forms

## Benchmark Guard

```bash
# Existing — must stay pinned
pytest tests/test_gpu_math_query.py::test_math_first_twenty_problems_stay_green_on_gpu_path  # 20/20
pytest tests/test_meaning_star.py  # 7 passed
pytest tests/test_galaxy_manager_ptx_fallback.py  # 2 passed
pytest tests/test_mesh_opcodes.py  # 8 passed

# New
pytest tests/test_house_builder.py  # all green
pytest tests/test_mesh_gpu.py  # all green (or skip if no GPU)
```

## Success Criteria

1. `mesh_generators.ptx` compiles and loads via sovereign loader
2. GPU sphere vertices match CPU sphere vertices within 1e-5
3. GPU mat4 transform matches CPU transform within 1e-5
4. All 5 room template `visual_rpn` programs produce valid MeshBuffer (vertices > 0, triangles > 0, normals computed)
5. All furniture template `visual_rpn` programs produce valid MeshBuffer
6. `house_builder.build_house()` completes without error
7. GLB decomposer processes at least 5 Khronos samples → valid MeaningCentricStar entries
8. Existing tests green, no benchmark regression

## Order of Implementation

1. **Expand mesh_generators.cu** with all shape kernels
2. **Compile to PTX** (verify it loads)
3. **sovereign_mesh_bridge.py** — wire kernel launch + buffer copy
4. **test_mesh_gpu.py** — verify GPU matches CPU
5. **Update mesh_bridge.py** — auto GPU/CPU selection
6. **house_rooms.py** — 5 room templates with executable visual_rpn
7. **house_furniture.py** — furniture templates
8. **test_house_builder.py** — verify all visual_rpn programs execute
9. **house_builder.py** — composition logic
10. **proceduralize_assets.py** — batch decompose existing GLBs
11. **Benchmark guard** — full regression check
