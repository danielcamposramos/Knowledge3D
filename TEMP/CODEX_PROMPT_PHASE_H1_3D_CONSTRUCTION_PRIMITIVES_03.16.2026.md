# Phase H1: 3D Construction Primitives — Opcodes, Kernels, Composition

**Date**: March 16, 2026
**From**: Claude (Architecture Partner)
**To**: Codex (Implementation)
**Priority**: CRITICAL — this is the foundation for House construction
**Depends on**: Phase H0 (meaning-centric star schema) — COMPLETE

---

## Directive

Build the 3D construction opcode surface and kernels **bottom-up**, from vertex atoms to composed shapes. Adopt game industry patterns (OpenGL vertex/index buffers, Unreal/Quake BSP-style CSG, Blender-style extrude/lathe). Do NOT reinvent — reuse and adapt what the game and CAD industries have proven over decades.

**This is opcode + kernel work.** Not Galaxy recipes. Not macros. Real opcodes in `rpn_opcodes.py`, real dispatch in the RPN engine, real PTX kernels where needed for GPU-parallel mesh generation.

---

## Architecture Context

### What Exists (2D — COMPLETE)
- 17 drawing opcodes (0x64-0x78): MOVE, LINE, QUAD, CUBIC, ARC, CLOSE, STROKE, FILL, transforms, colors
- ProceduralDrawingBridge with 100μs sovereignty budget
- Procedural font infrastructure (Bezier decomposition → glyph programs)
- Gradient, filter, lighting opcodes (0xF3-0xFF)

### What Exists (3D — GALAXY ENTRIES ONLY, NO KERNELS)
- `objects_3d_galaxy.py`: RPN program *strings* for cubes, spheres, lathes, grids, cylinders, transforms, spatial queries
- These are Stage 0 recipes: stored text, NOT executable by the RPN engine
- No `GENERATE_CUBE_VERTICES`, `GENERATE_UV_SPHERE`, `MAKE_TRI_FACE`, `MAT4_*` in any kernel or engine dispatch
- The 3D Galaxy entries reference opcodes that don't exist yet

### What Exists (Assets)
- 71 Khronos sample GLBs in `Knowledge3D.local/datasets/gltf_samples/`
- Curated collections in `gltf_curated/{workshop, office, it, library}/`
- 26 procedurally-generated Reality GLBs in `output/gltf/`
- `memory_house.gltf` in viewer
- Download tool: `knowledge3d/tools/training_pipelines/download_gltf_samples.py`

### Available Opcode Ranges
Gaps available for 3D mesh opcodes (confirm no collision before assigning):
- 0x44-0x5F (large block, 28 slots)
- 0x84-0x8F (12 slots)
- 0x98-0x9F (8 slots)

**Recommendation**: Use 0x44-0x5F for the 3D mesh surface. Clean block, no collisions.

---

## Deliverables (Bottom-Up Order)

### Layer 0: Vertex/Face Atoms (opcodes + engine dispatch)

**File**: `rpn_opcodes.py` (new constants) + `modular_rpn_engine.py` (dispatch)

These are the ATOMS. Everything else composes from these. Game industry standard vertex/index buffer pattern (OpenGL, Vulkan, DirectX all use this).

```
OP_VERTEX3        0x44   # x y z → push vertex to vertex buffer
OP_NORMAL3        0x45   # nx ny nz → push normal to normal buffer
OP_UV2            0x46   # u v → push UV coordinate to UV buffer
OP_TRI_FACE       0x47   # i0 i1 i2 → push triangle face (3 vertex indices)
OP_QUAD_FACE      0x48   # i0 i1 i2 i3 → push quad face (4 indices, split to 2 tris)
OP_FACE_NORMAL    0x49   # i0 i1 i2 → compute face normal from cross product, push to normal buffer
OP_MESH_BEGIN     0x4A   # → initialize vertex/index/normal/UV buffers
OP_MESH_END       0x4B   # → finalize mesh, compute missing normals, pack buffers
```

**Implementation notes**:
- Vertex buffer = flat f32 array on stack or VRAM region (max 65536 vertices per mesh)
- Index buffer = flat u32 array (max 196608 indices = 65536 triangles)
- Normal buffer = same layout as vertex buffer
- UV buffer = flat f32 pairs
- `MESH_END` auto-computes normals for any face that didn't get explicit `FACE_NORMAL`
- This is the standard vertex/index buffer pattern every GPU API uses

### Layer 1: 3D Transform Opcodes

```
OP_MAT4_IDENTITY    0x4C   # → push 4x4 identity matrix
OP_MAT4_TRANSLATE   0x4D   # tx ty tz → push translation matrix
OP_MAT4_SCALE       0x4E   # sx sy sz → push scale matrix
OP_MAT4_ROTATE_X    0x4F   # angle_rad → push rotation matrix
OP_MAT4_ROTATE_Y    0x50   # angle_rad → push rotation matrix
OP_MAT4_ROTATE_Z    0x51   # angle_rad → push rotation matrix
OP_MAT4_MUL         0x52   # mat4_a mat4_b → push mat4_a × mat4_b
OP_MAT4_APPLY       0x53   # mat4 → apply transform to all vertices in current mesh buffer
```

**Implementation notes**:
- Standard 4x4 homogeneous transform matrices (column-major, OpenGL convention)
- `MAT4_MUL` = standard matrix multiplication (composes transforms)
- `MAT4_APPLY` = transform every vertex in the current mesh buffer by the matrix
- Reuse existing `OP_MATRIX_MULT` (0xAA) logic where possible, but these operate on the mesh vertex buffer specifically

### Layer 2: Parametric Shape Generators

Every game engine ships these. Unity has `GameObject.CreatePrimitive()`. Unreal has BSP brushes. Blender has `Add Mesh > ...`. We need the same.

```
OP_GEN_PLANE        0x54   # width depth segments_w segments_d → generate plane mesh
OP_GEN_CUBE         0x55   # size → generate cube mesh (6 faces, 8 vertices, 12 tris)
OP_GEN_UV_SPHERE    0x56   # radius stacks slices → generate UV sphere
OP_GEN_CYLINDER     0x57   # radius height segments caps → generate cylinder
OP_GEN_CONE         0x58   # radius height segments → generate cone
OP_GEN_TORUS        0x59   # major_r minor_r major_seg minor_seg → generate torus
OP_GEN_ICOSPHERE    0x5A   # radius subdivisions → generate icosphere (better topology than UV sphere)
```

**Implementation notes** (adopt from game industry):
- **Plane**: grid of quads, standard UV mapping (0,0)→(1,1). Same as Unity's Plane.
- **Cube**: 8 vertices, 6 faces (each split to 2 tris), per-face normals. Standard box mesh.
- **UV Sphere**: latitude/longitude rings. `stacks` rings × `slices` segments. Known singularity at poles — acceptable for knowledge visualization. Same as Blender's UV Sphere.
- **Cylinder**: top ring + bottom ring + side quads + optional cap faces. Same as Three.js `CylinderGeometry`.
- **Cone**: like cylinder but top radius = 0.
- **Torus**: nested rotation — minor circle rotated around major axis. Same as Three.js `TorusGeometry`.
- **Icosphere**: start from icosahedron, subdivide faces. Better triangle distribution than UV sphere. Same as Blender's Ico Sphere.
- All generators: write to current mesh buffer (between MESH_BEGIN / MESH_END), auto-compute normals, generate UVs.

### Layer 3: Constructive Solid Geometry (CSG)

The Quake/Unreal level editor pattern. Known since 1996. Used to carve rooms from solid volumes (exactly what we need for House construction).

```
OP_CSG_UNION        0x5B   # mesh_a mesh_b → mesh_result (combine volumes)
OP_CSG_SUBTRACT     0x5C   # mesh_a mesh_b → mesh_result (carve B from A)
OP_CSG_INTERSECT    0x5D   # mesh_a mesh_b → mesh_result (keep only overlap)
```

**Implementation notes**:
- Use BSP tree approach (Binary Space Partition — the Quake/Doom algorithm)
- Input: two triangle meshes from the mesh buffer
- Output: new triangle mesh written to mesh buffer
- CSG_SUBTRACT is the room builder: start with a solid cube, subtract door/window shapes
- This is Class B initially — implement as CPU-side (ingestion/sleep-time), profile, promote to PTX if hot-path usage justifies
- **Reference**: Evan Wallace's CSG.js algorithm (MIT licensed, well-understood, Three.js ecosystem)

### Layer 4: Profile-to-3D Operations

The Blender/Maya/3ds Max modeling toolkit. How you turn 2D drawing programs into 3D objects.

```
OP_EXTRUDE          0x5E   # 2d_path depth → extrude 2D contour to 3D mesh
OP_LATHE            0x5F   # 2d_profile segments → revolve profile around Y axis
```

**Implementation notes**:
- **EXTRUDE**: Takes the current 2D path (from Drawing opcodes MOVE/LINE/QUAD/CUBIC/ARC/CLOSE), creates front face + back face + side walls. This is the bridge from 2D drawing to 3D mesh. Standard extrusion algorithm: duplicate vertices at depth, connect edges.
- **LATHE** (aka revolve/revolution): Takes a 2D profile curve, rotates it N times around the Y axis to generate a surface of revolution. How you make a wine glass, a vase, a chess piece. Same as Blender's "Screw" modifier or Three.js `LatheGeometry`.
- Both consume the path from the 2D drawing buffer and write to the 3D mesh buffer
- This is the critical bridge: **same drawing primitives that build glyphs also build 3D objects**

---

## Implementation Files

### New Files

1. **`knowledge3d/cranium/ptx_runtime/mesh_opcodes.py`**
   - All opcode constants (0x44-0x5F)
   - Mesh buffer data structures (vertex/index/normal/UV arrays)
   - Pure Python mesh generation functions for each shape generator
   - CSG implementation (BSP-based, CPU-side for now)
   - Extrude/Lathe algorithms
   - **stdlib only** — no numpy, no trimesh, no external mesh libraries

2. **`knowledge3d/cranium/ptx_runtime/mesh_engine.py`**
   - Dispatch table wiring: opcode → mesh function
   - Integration with ModularRPNEngine (register mesh opcodes)
   - Mesh buffer management (begin/end lifecycle)

3. **`knowledge3d/cranium/kernels/mesh_generators.cu`** (PTX kernel)
   - GPU-parallel vertex generation for UV sphere, icosphere, torus
   - GPU-parallel normal computation
   - GPU-parallel CSG (when promoted from CPU)
   - Entry point per shape type, standard CUDA grid/block pattern

4. **`knowledge3d/cranium/bridges/mesh_bridge.py`**
   - Bridge between mesh_engine and mesh_generators.cu
   - Same pattern as ProceduralDrawingBridge
   - Sovereignty: 100μs budget, no CPU fallbacks in hot path (after kernel promotion)

5. **`knowledge3d/tools/training_pipelines/glb_decomposer.py`**
   - **Ingestion tool** (flexible, can use external libs)
   - Takes GLB files from `gltf_samples/`, `gltf_curated/`, `gltf_house/`
   - Decomposes meshes into RPN construction programs
   - Outputs meaning-centric stars with `visual_rpn` = the construction program
   - This is how existing assets inform TRM: it learns the procedural vocabulary by seeing how real objects are built

6. **`tests/test_mesh_opcodes.py`**
   - Vertex/face buffer lifecycle
   - Each shape generator produces valid mesh (vertex count, face count, normals)
   - CSG: union/subtract/intersect on known inputs
   - Extrude: 2D square → 3D box
   - Lathe: 2D profile → 3D solid of revolution
   - Transform: MAT4 applied correctly to vertices
   - Round-trip: generate mesh → export to dict → reconstruct mesh → compare

### Modified Files

7. **`rpn_opcodes.py`** — add new opcode constants (0x44-0x5F)
8. **`modular_rpn_engine.py`** — register mesh opcode dispatch
9. **`objects_3d_galaxy.py`** — update Galaxy entries to reference real opcodes (currently dead strings)
10. **`__init__.py`** — export new modules

---

## GLB Decomposer Detail (Asset → RPN Pipeline)

This is the ingestion path that turns downloaded 3D assets into procedural vocabulary.

### Input
GLB file (binary glTF 2.0) from Khronos samples or curated collections.

### Process
1. Parse GLB → extract mesh primitives (vertices, indices, normals, UVs)
2. Classify mesh topology: is this a deformed cube? cylinder? sphere? freeform?
3. For recognized primitives: emit parametric RPN (`SIZE GEN_CUBE` or `R STACKS SLICES GEN_UV_SPHERE`)
4. For freeform meshes: emit vertex-level RPN (`MESH_BEGIN` + vertex/face sequence + `MESH_END`)
5. Extract transforms from glTF node hierarchy → emit `MAT4_*` composition
6. Create `MeaningCentricStar` with:
   - `visual_rpn` = the construction program
   - `domain` = mapped to House room (e.g., "Workshop/Tools", "Library/Furniture")
   - `surface_forms` = name in available languages (from glTF `extras` or filename)
   - `house_room` = semantic room assignment
   - `house_position` = TBD (TRM places deliberately)

### Output
- Meaning-centric star entries loadable into Galaxy
- Procedural vocabulary that TRM can learn from: "this is how a chair is built", "this is how a bookshelf is built"

### Libraries (Ingestion — flexible, not hot path)
- `pygltflib` or `trimesh` for GLB parsing (ingestion only, not sovereign)
- Standard math for topology classification
- Output must be pure RPN + MeaningCentricStar (sovereign)

---

## Composition Examples (What This Enables)

### Example 1: Build a Bookshelf from Primitives
```rpn
MESH_BEGIN
  1.0 GEN_CUBE                         # base unit cube
  2.0 0.1 0.8 MAT4_SCALE MAT4_APPLY    # flatten to shelf plank

  # Copy and stack 5 shelves
  MESH_BEGIN
    2.0 0.1 0.8 MAT4_SCALE MAT4_APPLY
    0.0 0.0 0.0 MAT4_TRANSLATE MAT4_APPLY  # bottom shelf
  MESH_END
  MESH_BEGIN
    2.0 0.1 0.8 MAT4_SCALE MAT4_APPLY
    0.0 0.4 0.0 MAT4_TRANSLATE MAT4_APPLY  # shelf 2
  MESH_END
  # ... repeat for each shelf

  # Side panels
  MESH_BEGIN
    0.05 1.6 0.8 MAT4_SCALE MAT4_APPLY
    -1.0 0.0 0.0 MAT4_TRANSLATE MAT4_APPLY  # left side
  MESH_END
  MESH_BEGIN
    0.05 1.6 0.8 MAT4_SCALE MAT4_APPLY
    1.0 0.0 0.0 MAT4_TRANSLATE MAT4_APPLY   # right side
  MESH_END
MESH_END
```

### Example 2: Build a Room via CSG (Quake/Unreal Pattern)
```rpn
MESH_BEGIN
  10.0 GEN_CUBE                          # solid block
  MESH_BEGIN
    9.6 GEN_CUBE                         # inner cavity (slightly smaller)
    0.0 0.2 0.0 MAT4_TRANSLATE MAT4_APPLY  # raise floor
  MESH_END
  CSG_SUBTRACT                           # hollow out = room!

  # Carve a doorway
  MESH_BEGIN
    1.2 GEN_CUBE
    2.2 0.05 0.0 MAT4_SCALE MAT4_APPLY  # door-sized box
    5.0 0.0 0.0 MAT4_TRANSLATE MAT4_APPLY  # position at wall
  MESH_END
  CSG_SUBTRACT                           # carve door opening
MESH_END
```

### Example 3: Lathe a Vase from 2D Profile
```rpn
# Draw the profile (2D, same primitives as glyphs)
MOVE 0.3 0.0
LINE 0.4 0.2
QUAD 0.5 0.5 0.35 0.6    # belly curve
LINE 0.2 0.8
LINE 0.25 1.0             # neck
CLOSE

# Revolve to 3D
MESH_BEGIN
  24 LATHE                 # 24-segment revolution → 3D vase
MESH_END
```

This last example demonstrates the critical principle: **same MOVE/LINE/QUAD drawing primitives that build font glyphs also build 3D vase profiles.**

---

## Constraints

1. **stdlib only** for all hot-path code (mesh_opcodes.py, mesh_engine.py, mesh_bridge.py)
2. **No numpy, trimesh, scipy** in hot path — pure Python vertex math, then PTX kernels
3. **Ingestion tools** (glb_decomposer.py) CAN use external libs (pygltflib, trimesh) — they run once
4. **No hot-path changes** — existing benchmark pipeline untouched
5. **Backward compatible** — existing `objects_3d_galaxy.py` entries still loadable
6. **Opcode collision check** — verify 0x44-0x5F is clear before assigning (I checked, but double-confirm)
7. **Tests MUST pass** — existing test suite green + new mesh tests

## Benchmark Guard

Run after implementation:
```bash
# All must stay pinned
pytest tests/test_gpu_math_query.py::test_math_first_twenty_problems_stay_green_on_gpu_path  # 20/20
pytest tests/test_meaning_star.py  # 7 passed
pytest tests/test_galaxy_manager_ptx_fallback.py  # 2 passed
# New
pytest tests/test_mesh_opcodes.py  # all green
```

---

## Success Criteria

1. All 20 opcodes (0x44-0x5F) registered and dispatching in ModularRPNEngine
2. Each parametric generator produces geometrically correct mesh (vertex count, normal directions, UV mapping)
3. CSG subtract on two cubes produces a hollow box (room)
4. EXTRUDE on a square path produces a rectangular prism
5. LATHE on a semicircle profile produces a sphere-like mesh
6. GLB decomposer successfully processes at least 10 Khronos sample models → MeaningCentricStar entries
7. Existing tests green, no benchmark regression

---

## Game Industry References (Adopt, Don't Reinvent)

| Pattern | Source | Our Use |
|---------|--------|---------|
| Vertex/Index buffers | OpenGL, Vulkan, DirectX (since 1992) | Mesh buffer structure |
| Parametric primitives | Unity `CreatePrimitive`, Three.js geometries | Shape generators |
| CSG/BSP | Quake (1996), Unreal Editor, Source Engine | Room carving |
| Extrude/Lathe | Blender, Maya, 3ds Max, SolidWorks | 2D→3D bridge |
| UV sphere algorithm | Standard latitude/longitude rings | GEN_UV_SPHERE |
| Icosphere algorithm | Icosahedron subdivision (Blender) | GEN_ICOSPHERE |
| Torus algorithm | Nested rotation (Three.js TorusGeometry) | GEN_TORUS |
| Normal computation | Cross product of edge vectors (universal) | FACE_NORMAL |
| glTF 2.0 parsing | Khronos Group standard | GLB decomposer |

**Philosophy**: The game industry solved mesh generation 30 years ago. We adopt their vertex/face/normal patterns. Our contribution is making these **procedural RPN programs** that compose into meaning-centric stars — same atoms, different composition layer.

---

## Order of Implementation

1. **Opcodes + buffer structs** (mesh_opcodes.py) — the atoms
2. **Engine dispatch** (mesh_engine.py + modular_rpn_engine.py wiring)
3. **Shape generators** (plane, cube, sphere, cylinder first — most useful)
4. **Tests** for generators (vertex counts, normals, basic geometry validation)
5. **3D transforms** (MAT4_* family)
6. **CSG** (CPU-side first, BSP algorithm)
7. **Extrude + Lathe** (the 2D→3D bridge)
8. **GLB decomposer** (ingestion tool, last — builds on all above)
9. **Update objects_3d_galaxy.py** to reference real opcodes
10. **Benchmark guard** — full regression check

**Estimated scope**: ~1500-2000 lines new code + ~200 lines modified.
