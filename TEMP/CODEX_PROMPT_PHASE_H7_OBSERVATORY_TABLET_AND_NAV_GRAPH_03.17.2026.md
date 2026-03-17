# Phase H7: Bathtub Observatory, Memory Tablet, and Navigation Graph

**Spec Author:** Claude (Architecture Partner)
**Date:** March 17, 2026
**Depends on:** Phase H6 (Doors, Workshop Tools, Gallery Displays) COMPLETE
**Sovereignty:** Ingestion/export path (flexible).

---

## Context

The House now has 50 GLTF nodes (260KB, 3929 vertices): 5 rooms, 6 furniture, 4 doors, 5 tools, 5 books, 4 displays, 20 knowledge tree nodes, plus the root. Four of five rooms are populated. The Bathtub Observatory has only its basin — it's the introspection/Galaxy-view room and needs observation instruments. The Memory Tablet (primary K3D interface object per `docs/vocabulary/MEMORY_TABLET_SPECIFICATION.md`) has no physical presence in the House yet. And while doors encode room connectivity in `behavior_rpn`, there's no traversable House-level navigation graph.

This phase fills the last empty room, instantiates the Memory Tablet as a 3D object, and creates a navigation graph from door connectivity.

---

## Deliverables

### Track A: Bathtub Observatory Instruments
### Track B: Memory Tablet 3D Object
### Track C: House Navigation Graph
### Track D: Re-export Updated House GLB

---

## Track A: Bathtub Observatory Instruments

### A1. Create `house_observatory.py`

**File:** `knowledge3d/knowledgeverse/house_observatory.py`

The Bathtub Observatory is the introspection room — where the TRM observes its own Galaxy Universe from "outside." The bathtub is already a portal/vessel (`PORTAL REST REFLECT`). Add observation instruments around it.

**3 observatory objects:**

| star_id | Description | Shape | Position (room-local) |
|---------|-------------|-------|-----------------------|
| `observatory_telescope` | Points "outward" toward Galaxy | Cylinder tube (long, narrow) + cone eyepiece | (0.0, 0.8, -2.0) |
| `observatory_prism` | Decomposes incoming knowledge into spectra | Triangular prism (extruded triangle) | (1.8, 0.3, -1.0) |
| `observatory_journal` | Records observations and reflections | Thin cube (like an open journal) | (-1.5, 0.3, 0.5) |

Each instrument has:
- `meaning_class = "instrument"`
- `house_room = "House/Bathtub"`
- `taxonomy_refs` pointing to relevant concept entries (self_reflection, growth, etc.)
- Multilingual surface forms

**Telescope shape hint:**
```python
visual_rpn = (
    "0.06 0.50 12 1 GEN_CYLINDER 0.0 0.25 0.0 MAT4_TRANSLATE MAT4_APPLY "
    "0.09 0.12 12 GEN_CONE 0.0 0.56 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
    "-0.35 MAT4_ROTATE_X MAT4_APPLY"  # tilt upward ~20 degrees
)
```

**Prism shape hint:** Use `MOVE`/`LINE`/`CLOSE` path operations to create a triangular profile, then extrude or lathe. Or approximate as a cube with rotation that suggests a triangular cross-section. Keep it simple.

**Journal shape:** Thin open-book shape similar to `_book_visual_rpn` from `house_books.py` but flatter (open, not closed).

### A2. Update Bathtub room `component_refs`

Add the 3 observatory instrument star_ids to the Bathtub room's `component_refs` in `house_rooms.py`.

---

## Track B: Memory Tablet 3D Object

### B1. Create `house_memory_tablet.py`

**File:** `knowledge3d/knowledgeverse/house_memory_tablet.py`

The Memory Tablet is K3D's primary interface object (see `docs/vocabulary/MEMORY_TABLET_SPECIFICATION.md`). In this phase, we instantiate it as a **physical 3D object** in the House. It's a thin rectangular slab with a slightly recessed screen area — like a tablet device.

**1 Memory Tablet object:**

| star_id | Description | Position |
|---------|-------------|----------|
| `memory_tablet` | Primary K3D interface object | House root level (not in any specific room) |

**Key design:**
- `meaning_class = "tablet"`
- `domain = "House/Interface"`
- `house_room = "House"` (top-level, not room-bound — the tablet moves with the avatar)
- `house_position = (0.0, 1.0, 2.0)` — starts near Library entrance, at hand height
- `behavior_rpn = "TABLET ACTIVATE BROWSE_GALAXY QUERY_KNOWLEDGE INSPECT_PROGRAMS"`

**Tablet shape:**
```python
# Thin slab with recessed screen
visual_rpn = (
    "1.0 GEN_CUBE 0.40 0.28 0.02 MAT4_SCALE MAT4_APPLY "   # outer body
    "1.0 GEN_CUBE 0.36 0.24 0.01 MAT4_SCALE MAT4_APPLY "   # screen recess
    "0.0 0.0 0.011 MAT4_TRANSLATE MAT4_APPLY CSG_SUBTRACT"  # offset slightly forward
)
```

**Surface forms:** Trilingual ("Memory Tablet" / "Tablete de Memoria" / "記憶タブレット")

**`taxonomy_refs`** should reference the concept entries that the tablet can browse:
```python
taxonomy_refs=[
    "concept_language", "concept_mathematics", "concept_visual_art",
    "concept_physics", "concept_biology", "concept_tool",
]
```

This is the entry point for ALL Galaxy browsing. The taxonomy_refs indicate which knowledge domains the tablet provides access to.

### B2. Register Memory Tablet in house_builder.py

Add the Memory Tablet to `build_house()`. It gets stored in the House galaxy like other objects but is NOT a child of any room in the scene graph — it's a sibling of rooms, direct child of the House root node.

---

## Track C: House Navigation Graph

### C1. Create `house_nav_graph.py`

**File:** `knowledge3d/knowledgeverse/house_nav_graph.py`

Build a traversable navigation graph from the existing door connectivity. This is a **build-time data structure** (not hot-path) that formalizes room-to-room adjacency for future TRM navigation.

```python
@dataclass
class HouseNavNode:
    """A navigable location in the House."""
    star_id: str
    house_room: str
    position: tuple[float, float, float]
    connected_to: list[str]  # star_ids of adjacent nav nodes

@dataclass
class HouseNavEdge:
    """A traversable connection between two nav nodes."""
    door_star_id: str
    from_node: str  # star_id of nav node
    to_node: str    # star_id of nav node
    cost: float     # traversal cost (default 1.0 for adjacent rooms)

@dataclass
class HouseNavGraph:
    """Complete House navigation graph."""
    nodes: dict[str, HouseNavNode]
    edges: list[HouseNavEdge]

    def neighbors(self, star_id: str) -> list[str]:
        """Return star_ids of nodes reachable from the given node."""

    def shortest_path(self, from_id: str, to_id: str) -> list[str]:
        """BFS shortest path between two nodes. Returns list of star_ids."""
```

### C2. Build the graph from existing data

```python
def build_house_nav_graph() -> HouseNavGraph:
    """Construct navigation graph from HOUSE_ROOMS and HOUSE_DOORS."""
```

This function:
1. Creates a `HouseNavNode` for each room in `HOUSE_ROOMS` using `star_id`, `house_room`, and `house_position`
2. Parses each door's `behavior_rpn` (format: `"DOOR_TRAVERSE CONNECT {room_a} {room_b}"`) to extract the two room references
3. Maps room references (e.g., `"House/Library"`) back to room star_ids
4. Creates bidirectional `HouseNavEdge` entries for each door
5. Populates each node's `connected_to` list

**Result for current House:**
```
room_library ←→ room_garden (via door_library_garden)
room_garden ←→ room_workshop (via door_garden_workshop)
room_workshop ←→ room_gallery (via door_workshop_gallery)
room_gallery ←→ room_bathtub (via door_gallery_bathtub)
```

This is a simple linear chain for now, but the graph structure supports any topology (branching corridors, shortcuts, etc.) as the House grows.

### C3. Export navigation graph as metadata

Add the navigation graph as metadata on the House root node in the GLTF export. This goes into `extras["k3d"]["nav_graph"]` on the House root node:

```python
{
    "k3d": {
        "star_id": "house_root",
        "meaning_class": "house",
        "nav_graph": {
            "nodes": ["room_library", "room_garden", "room_workshop", "room_gallery", "room_bathtub"],
            "edges": [
                {"door": "door_library_garden", "from": "room_library", "to": "room_garden"},
                {"door": "door_garden_workshop", "from": "room_garden", "to": "room_workshop"},
                ...
            ]
        }
    }
}
```

This enables any GLTF consumer (viewer, TRM, external tool) to understand House connectivity without parsing behavior_rpn.

---

## Track D: Re-export Updated House GLB

### D1. Update `export_house.py`

- Import and include observatory instruments in the export pipeline (children of Bathtub room node)
- Import and include Memory Tablet (child of House root node, NOT any room)
- Add `nav_graph` metadata to House root node extras
- Update summary dict with new counts

### D2. Re-generate `viewer/public/house.glb`

Run export to produce updated GLB. Expected growth: ~260KB to ~280-310KB with 4 new mesh objects (3 instruments + 1 tablet) plus nav_graph metadata.

---

## Tips for Codex

**Tip 1 — Tablet is NOT a room child.** In `export_house.py`, the Memory Tablet should be added as a direct child of the House root node (alongside rooms), NOT as a child of any room. It's the one object that transcends room boundaries.

**Tip 2 — Observatory positions are room-local.** The Bathtub room is at world (72, 0, 0). Observatory instruments use local coordinates relative to the room origin, like tools in the Workshop.

**Tip 3 — Nav graph uses existing data.** Don't create new connectivity — parse what's already in `HOUSE_DOORS`. Each door's `behavior_rpn` has the form `"DOOR_TRAVERSE CONNECT House/RoomA House/RoomB"`. Also `taxonomy_refs` on doors already contain `[room_a, room_b]` references.

**Tip 4 — BFS is sufficient for shortest_path.** The graph is small (5 nodes, 4 edges). No need for Dijkstra or A*. Simple BFS from collections.deque.

**Tip 5 — Prism shape.** If the `MOVE`/`LINE`/`CLOSE`/`LATHE` path ops are too complex for a prism, approximate with a cube rotated 45 degrees on one axis: `"1.0 GEN_CUBE 0.15 0.20 0.15 MAT4_SCALE MAT4_APPLY 0.7854 MAT4_ROTATE_Z MAT4_APPLY"` — a rotated cube looks prism-like. Or use two thin cubes as triangular approximation.

**Tip 6 — Follow the export pattern.** The new objects (observatory instruments + tablet) follow the exact same pattern as doors/tools/displays in export_house.py: execute visual_rpn → mesh_to_gltf_node → append to ordered_nodes with parent reference.

---

## Tests

### `tests/test_house_observatory.py`

```python
def test_observatory_instruments_in_bathtub():
    for instrument in OBSERVATORY_INSTRUMENTS:
        assert instrument.meaning_class == "instrument"
        assert instrument.house_room == "House/Bathtub"
        assert instrument.taxonomy_refs

def test_observatory_shapes_constructable():
    bridge = MeshBridge()
    for instrument in OBSERVATORY_INSTRUMENTS:
        result = bridge.execute_rpn_program(instrument.visual_rpn)
        assert result.mesh.vertices
        assert result.mesh.triangles
```

### `tests/test_house_memory_tablet.py`

```python
def test_memory_tablet_exists():
    assert MEMORY_TABLET.star_id == "memory_tablet"
    assert MEMORY_TABLET.meaning_class == "tablet"
    assert MEMORY_TABLET.house_room == "House"

def test_memory_tablet_shape_constructable():
    bridge = MeshBridge()
    result = bridge.execute_rpn_program(MEMORY_TABLET.visual_rpn)
    assert result.mesh.vertices
    assert result.mesh.triangles

def test_memory_tablet_references_all_domains():
    refs = set(MEMORY_TABLET.taxonomy_refs)
    assert "concept_language" in refs
    assert "concept_mathematics" in refs
```

### `tests/test_house_nav_graph.py`

```python
def test_nav_graph_has_all_rooms():
    graph = build_house_nav_graph()
    assert len(graph.nodes) == 5
    assert "room_library" in graph.nodes
    assert "room_bathtub" in graph.nodes

def test_nav_graph_edges_from_doors():
    graph = build_house_nav_graph()
    assert len(graph.edges) == 8  # 4 doors × 2 directions

def test_nav_graph_neighbors():
    graph = build_house_nav_graph()
    garden_neighbors = graph.neighbors("room_garden")
    assert "room_library" in garden_neighbors
    assert "room_workshop" in garden_neighbors

def test_nav_graph_shortest_path():
    graph = build_house_nav_graph()
    path = graph.shortest_path("room_library", "room_bathtub")
    assert path == ["room_library", "room_garden", "room_workshop", "room_gallery", "room_bathtub"]

def test_nav_graph_shortest_path_reverse():
    graph = build_house_nav_graph()
    path = graph.shortest_path("room_bathtub", "room_library")
    assert path == ["room_bathtub", "room_gallery", "room_workshop", "room_garden", "room_library"]
```

### Updated export test

```python
def test_exported_house_includes_observatory_and_tablet(tmp_path):
    output = tmp_path / "house.glb"
    summary = export_house_glb(output)
    assert summary["instruments"] >= 3
    assert summary["tablet"] >= 1
    # Verify nav_graph in root node extras
    pygltflib = pytest.importorskip("pygltflib")
    gltf = pygltflib.GLTF2().load(str(output))
    house_node = next(n for n in gltf.nodes if n.name == "House")
    assert "nav_graph" in house_node.extras["k3d"]
```

### Non-regression

All existing tests must pass. GPU math non-regression: `test_math_first_twenty_problems_stay_green_on_gpu_path`.

---

## Success Criteria

1. 3 observatory instruments in Bathtub room with constructable visual_rpn
2. 1 Memory Tablet as top-level House object (not room-bound)
3. Navigation graph built from door connectivity with BFS shortest_path
4. Nav graph metadata exported on House root GLTF node
5. All new objects included in GLTF export with proper hierarchy
6. Updated `house.glb` viewable in any GLTF viewer
7. All existing tests pass

---

## Files Changed/Created

| File | Action |
|------|--------|
| `knowledge3d/knowledgeverse/house_observatory.py` | **NEW** — 3 observatory instruments |
| `knowledge3d/knowledgeverse/house_memory_tablet.py` | **NEW** — Memory Tablet 3D object |
| `knowledge3d/knowledgeverse/house_nav_graph.py` | **NEW** — Navigation graph from doors |
| `knowledge3d/knowledgeverse/house_rooms.py` | Update Bathtub component_refs |
| `knowledge3d/knowledgeverse/house_builder.py` | Register instruments, tablet |
| `knowledge3d/tools/export_house.py` | Include new objects + nav_graph metadata |
| `knowledge3d/knowledgeverse/__init__.py` | Exports |
| `viewer/public/house.glb` | Re-generated with all objects |
| `tests/test_house_observatory.py` | **NEW** |
| `tests/test_house_memory_tablet.py` | **NEW** |
| `tests/test_house_nav_graph.py` | **NEW** |
| `tests/test_gltf_export.py` | Update export assertions |

---

## Architectural Note

After H7, the House physical construction is **feature complete**: all 5 rooms populated, all rooms connected, primary interface object instantiated, and navigation graph traversable. The House has a Library (knowledge storage), Garden (ontological growth), Workshop (tool operations), Gallery (perceptual displays), and Bathtub Observatory (introspection). The Memory Tablet bridges the House to the Galaxy Universe.

**What comes next:** Making the House LIVE — TRM navigating through doors using the nav graph, activating objects via behavior_rpn, and the Memory Tablet rendering Galaxy content. That's Phase D territory (TRM game loop migration).
