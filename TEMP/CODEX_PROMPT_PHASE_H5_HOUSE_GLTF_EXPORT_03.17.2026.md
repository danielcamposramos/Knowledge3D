# Phase H5: House GLTF Export — First Visible House

**Spec Author:** Claude (Architecture Partner)
**Date:** March 17, 2026
**Depends on:** Phase H4 (populated House) COMPLETE
**Sovereignty:** Ingestion/export path (flexible). This is I/O, not hot path.

---

## Why This Matters

The House has 5 rooms, 6 furniture items, 5 books, and 20 knowledge tree nodes — all with executable `visual_rpn` programs. But no human has seen them. This phase makes the House visible by exporting it as a GLTF/GLB file that any 3D viewer can render.

This directly addresses Christoph's request: "I am trying to make practical progress towards having K3D render something a human can understand."

---

## Architecture Context

**Already exists:**
- `MeshBuffer` with vertices/triangles/normals/uvs (mesh_opcodes.py)
- `MeshBridge.execute_rpn_program()` producing MeshBuffer from visual_rpn
- `pygltflib` as dependency (used in glb_decomposer.py for import)
- `glb_ctypes_loader.save_stars_to_glb()` exporting metadata in `extras.k3d` (but NO geometry)
- Three.js viewer loading GLTF with `extras.k3d` metadata (viewer/src/loadK3D.ts)

**Missing:**
- `MeshBuffer` → GLTF geometry serialization
- House scene composition (rooms + furniture + books + tree as scene graph)
- GLB export with both geometry AND k3d metadata

---

## Deliverables

### Track A: MeshBuffer → GLTF Geometry Serialization
### Track B: House Scene Export
### Track C: Export Script + Viewer Integration

---

## Track A: MeshBuffer GLTF Serialization

### A1. Add `to_gltf_bytes()` to MeshBuffer

**File:** `knowledge3d/cranium/ptx_runtime/mesh_opcodes.py`

Add a method that serializes the mesh to binary buffers suitable for GLTF:

```python
def to_gltf_bytes(self) -> tuple[bytes, bytes, bytes, bytes]:
    """Return (position_bytes, normal_bytes, uv_bytes, index_bytes) as little-endian float32/uint32."""
```

This returns raw binary data. Position and normal are `float32 * 3 * vertex_count`. UVs are `float32 * 2 * vertex_count`. Indices are `uint32 * 3 * triangle_count`. Standard GLTF buffer layout.

Also compute and return bounding box (min/max) for the Accessor — GLTF requires this for POSITION.

### A2. Create `gltf_export.py`

**File:** `knowledge3d/tools/gltf_export.py`

Utility module that converts a MeshBuffer into pygltflib structures:

```python
def mesh_to_gltf_node(
    mesh: MeshBuffer,
    *,
    name: str = "",
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    extras: dict[str, object] | None = None,
) -> GltfNodeData:
    """Convert a MeshBuffer into GLTF node data (buffers, views, accessors, primitive, node)."""
```

This function:
1. Calls `mesh.to_gltf_bytes()` to get binary data
2. Creates pygltflib BufferView entries for position, normal, uv, indices
3. Creates Accessor entries with proper componentType, count, min/max
4. Creates a Primitive with attributes POSITION, NORMAL, TEXCOORD_0 and indices
5. Creates a Node with the mesh, optional translation, and `extras["k3d"]` metadata
6. Returns all the pieces as a dataclass so the caller can compose them into a scene

```python
@dataclass
class GltfNodeData:
    buffer_data: bytes  # concatenated binary data for this node
    buffer_views: list[BufferView]
    accessors: list[Accessor]
    mesh: Mesh
    node: Node
```

### A3. Scene composition function

```python
def compose_scene(nodes: list[GltfNodeData], *, asset_generator: str = "Knowledge3D") -> GLTF2:
    """Merge multiple GltfNodeData into a single GLTF2 scene."""
```

This function:
1. Concatenates all buffer_data into one binary blob
2. Adjusts BufferView byte offsets for the concatenated buffer
3. Adjusts Accessor indices to point to correct BufferViews
4. Creates a single Buffer, Scene, and returns GLTF2 ready to save

---

## Track B: House Scene Export

### B1. Create `export_house.py`

**File:** `knowledge3d/tools/export_house.py`

Main export function:

```python
def export_house_glb(
    output_path: Path,
    *,
    include_books: bool = True,
    include_tree: bool = True,
) -> dict[str, Any]:
    """Build and export the full House as a GLB file."""
```

This function:
1. Creates a MeshBridge
2. Iterates over HOUSE_ROOMS — executes each `visual_rpn`, gets MeshBuffer, converts to GltfNodeData with `house_position` as translation
3. Same for HOUSE_FURNITURE — positioned relative to their room
4. Same for HOUSE_BOOKS — positioned on shelves
5. Same for KNOWLEDGE_TREE_BRANCHES — already have `house_position`
6. Attaches `extras["k3d"]` to each node containing:
   - `star_id`, `meaning_class`, `domain`, `galaxy_ref` (if any)
   - `surface_forms` (human-readable labels)
   - `behavior_rpn` (what happens when activated)
7. Composes all nodes into a scene graph:
   - Root node "House"
   - Child nodes per room
   - Furniture/books/tree as children of their respective rooms
8. Saves as GLB via `gltf.save_binary(str(output_path))`
9. Returns summary: room count, furniture count, total vertices, file size

### B2. Room hierarchy

The scene graph should reflect the House structure:

```
House (root)
├── Library (position 0,0,0)
│   ├── Bookshelf
│   ├── Desk
│   ├── Chair
│   ├── Mathematics Primer (book)
│   ├── Language Foundations (book)
│   ├── Physics Handbook (book)
│   ├── Biology Atlas (book)
│   └── Tool Manual (book)
├── Garden (position 18,0,0)
│   ├── Knowledge Tree (trunk)
│   ├── Mathematics Branch
│   ├── Language Branch
│   ├── Physics Branch
│   ├── Biology Branch
│   ├── Tools Branch
│   └── 15 leaves
├── Workshop (position 36,0,0)
│   └── Workbench
├── Gallery (position 54,0,0)
└── Bathtub Observatory (position 72,0,0)
    └── Bathtub Basin
```

Each room node uses its `house_position` as translation. Furniture/books use their own `house_position` as LOCAL offset from the room origin (the positions in the data are already room-local for books/furniture, room-global for rooms themselves).

**Important:** Furniture `house_position` values are relative to the room. The export should add furniture as children of the room node, using their `house_position` directly. Room nodes are at the top level with their `house_position` as world translation. The tree nodes in Garden are already in world coordinates centered around (5,0,5) — make them children of the Garden node and subtract the Garden's position to get local coordinates.

---

## Track C: Export Script + Viewer Integration

### C1. CLI entry point

**File:** `scripts/export_house.py`

```python
def main(argv=None):
    parser = argparse.ArgumentParser(description="Export the K3D House as GLB.")
    parser.add_argument("--output", type=Path,
                        default=Path("viewer/public/house.glb"),
                        help="Output GLB path")
    args = parser.parse_args(argv)
    summary = export_house_glb(args.output)
    print(f"Exported House: {summary['rooms']} rooms, "
          f"{summary['total_vertices']} vertices, "
          f"{summary['file_size_kb']:.1f} KB -> {args.output}")
```

Default output is `viewer/public/house.glb` so the viewer can serve it directly.

### C2. Viewer integration note

The existing viewer's `loadK3DFromGLTF()` already handles GLTF loading with `extras.k3d`. No viewer changes needed for basic rendering. The Three.js GLTFLoader will render the geometry; the K3D metadata rides along in extras.

**Do NOT modify the viewer code.** Just place the GLB file where the viewer can serve it. The viewer already has a mechanism to load GLTF files from its public directory.

---

## Tests

### `tests/test_gltf_export.py`

```python
def test_mesh_buffer_to_gltf_bytes():
    """MeshBuffer serializes to valid binary layout."""
    mesh = generate_cube(1.0)
    pos, norms, uvs, indices = mesh.to_gltf_bytes()
    assert len(pos) == 24 * 3 * 4  # 24 verts * 3 floats * 4 bytes
    assert len(norms) == 24 * 3 * 4
    assert len(uvs) == 24 * 2 * 4  # 24 verts * 2 floats * 4 bytes
    assert len(indices) == 12 * 3 * 4  # 12 tris * 3 indices * 4 bytes

def test_mesh_to_gltf_node_creates_valid_structure():
    """Single mesh converts to GLTF node with correct buffer layout."""
    mesh = generate_cube(1.0)
    node_data = mesh_to_gltf_node(mesh, name="test_cube")
    assert len(node_data.buffer_views) == 4  # pos, normal, uv, indices
    assert len(node_data.accessors) == 4

def test_compose_scene_creates_valid_gltf():
    """Multiple nodes compose into a valid GLTF2 structure."""
    cube = generate_cube(1.0)
    sphere = generate_uv_sphere(0.5, 8, 12)
    nodes = [
        mesh_to_gltf_node(cube, name="cube"),
        mesh_to_gltf_node(sphere, name="sphere", translation=(3.0, 0.0, 0.0)),
    ]
    gltf = compose_scene(nodes)
    assert len(gltf.nodes) >= 2
    assert len(gltf.meshes) >= 2
    assert gltf.binary_blob() is not None

def test_export_house_produces_valid_glb(tmp_path):
    """Full House export creates a loadable GLB file."""
    output = tmp_path / "house.glb"
    summary = export_house_glb(output)
    assert output.exists()
    assert output.stat().st_size > 1000  # non-trivial size
    assert summary["rooms"] == 5
    assert summary["furniture"] >= 6
    assert summary["books"] >= 5
    # Verify it's a valid GLB by loading back
    pygltflib = pytest.importorskip("pygltflib")
    gltf = pygltflib.GLTF2().load(str(output))
    assert len(gltf.nodes) > 10  # rooms + furniture + books + tree

def test_exported_nodes_carry_k3d_metadata(tmp_path):
    """Each exported node should have extras.k3d with star metadata."""
    output = tmp_path / "house.glb"
    export_house_glb(output)
    pygltflib = pytest.importorskip("pygltflib")
    gltf = pygltflib.GLTF2().load(str(output))
    nodes_with_k3d = [n for n in gltf.nodes if n.extras and "k3d" in n.extras]
    assert len(nodes_with_k3d) >= 15  # rooms + furniture + books + some tree nodes
```

---

## Success Criteria

1. `MeshBuffer.to_gltf_bytes()` produces valid binary layout (float32 positions/normals/uvs, uint32 indices)
2. `mesh_to_gltf_node()` converts any MeshBuffer to a valid GLTF node
3. `compose_scene()` merges multiple nodes into a single GLTF2 with concatenated binary buffer
4. `export_house_glb()` produces a GLB file with:
   - All 5 rooms as geometry with correct positioning
   - All furniture, books, tree branches, and leaves
   - Scene graph hierarchy (rooms → children)
   - `extras["k3d"]` metadata on each node
5. The GLB opens in any GLTF viewer (Blender, three.js, etc.)
6. All existing tests pass, GPU math non-regression holds

---

## Files Changed/Created

| File | Action |
|------|--------|
| `knowledge3d/cranium/ptx_runtime/mesh_opcodes.py` | Add `to_gltf_bytes()` to MeshBuffer |
| `knowledge3d/tools/gltf_export.py` | **NEW** — mesh_to_gltf_node, compose_scene |
| `knowledge3d/tools/export_house.py` | **NEW** — export_house_glb |
| `scripts/export_house.py` | **NEW** — CLI entry point |
| `tests/test_gltf_export.py` | **NEW** — export tests |

---

## Architectural Note

This is the **Dual Client Contract** coming alive. The same procedural programs that the AI processes (RPN in Galaxy) now produce geometry that humans can see (GLTF in viewer). Same House, two clients. Form + Meaning, rendered for both.
