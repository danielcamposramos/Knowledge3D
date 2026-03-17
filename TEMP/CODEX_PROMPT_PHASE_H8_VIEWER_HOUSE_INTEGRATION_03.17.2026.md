# Phase H8: Viewer House Integration — Making the House Navigable

**Spec Author:** Claude (Architecture Partner)
**Date:** March 17, 2026
**Depends on:** Phase H7 (complete physical House) COMPLETE
**Sovereignty:** I/O path (viewer is Three.js, flexible).

---

## Context

The House is physically complete: 54 GLTF nodes, 278KB, 4167 vertices, 5 populated rooms connected by 4 doors, with a navigation graph and Memory Tablet. But two paradigms exist:

1. **House GLB** (from H5-H7): Each GLTF node carries `extras.k3d` with `star_id`, `meaning_class`, `surface_forms`, `behavior_rpn`, `taxonomy_refs`, etc. The House root node carries `nav_graph` metadata.

2. **Viewer** (`loadK3DFromGLTF`): Expects a flat K3D payload on a single mesh primitive's extras — `ids[]`, `vectors[]`, `embeddings[]`, `metadata[]`, `neighbors[]`. Point-cloud-first design.

These are complementary, not conflicting. The viewer already loads `memory_house.glb` and `knowledge_garden.glb` as decorative scene assets alongside point cloud data. The House GLB follows the scene asset pattern — it has actual geometry (rooms, furniture, doors), not a point cloud.

This phase adds a **House scene loader** to the viewer that understands per-node `extras.k3d` metadata, enables room-aware camera navigation via doors, and wires the existing DoorsApp to the House nav_graph.

---

## Deliverables

### Track A: House Scene Loader
### Track B: Room-Aware Camera Navigation
### Track C: DoorsApp Integration with Nav Graph
### Track D: House Object Tooltips

---

## Track A: House Scene Loader

### A1. Create `viewer/src/loadHouseScene.ts`

New loader that understands the House GLB's per-node `extras.k3d` schema. This is SEPARATE from the existing `loadK3DFromGLTF` which handles point cloud data.

```typescript
export interface HouseNode {
  starId: string;
  meaningClass: string;  // "room" | "furniture" | "door" | "tool_object" | "display" | "instrument" | "tablet"
  domain: string;
  houseRoom: string;
  housePosition: [number, number, number];
  surfaceForms: Record<string, { word_ref: string; char_refs: string[] }>;
  behaviorRpn: string;
  taxonomyRefs: string[];
  componentRefs: string[];
  visualRpn?: string;
  galaxyRef?: string;
  mesh: THREE.Mesh;  // the actual Three.js mesh from the GLTF scene
}

export interface HouseNavGraph {
  nodes: string[];  // room star_ids
  edges: Array<{ door: string; from: string; to: string; cost: number }>;
}

export interface LoadedHouseScene {
  root: THREE.Group;             // the full House scene graph
  nodesByStarId: Map<string, HouseNode>;
  rooms: HouseNode[];            // just the rooms
  doors: HouseNode[];            // just the doors
  navGraph: HouseNavGraph;       // from House root extras
  currentRoom: string;           // initially "room_library"
}
```

### A2. Loading logic

```typescript
export async function loadHouseScene(url: string): Promise<LoadedHouseScene> {
  // 1. Use GLTFLoader to load the House GLB
  // 2. Walk the GLTF scene graph (scene.traverse)
  // 3. For each node with extras?.k3d, extract HouseNode data
  // 4. Extract nav_graph from the "House" root node's extras.k3d.nav_graph
  // 5. Return LoadedHouseScene with everything indexed by star_id
}
```

**Key:** Walk `gltf.scene.traverse()`, check each `Object3D` for `userData.k3d` (Three.js puts GLTF extras into userData). Build the `nodesByStarId` map from there.

### A3. Integration with main.ts

In `main.ts`, add a `loadHouseScene()` call alongside or instead of the existing `loadHouse()` for `memory_house.glb`. The loaded scene gets added directly to the Three.js scene. No conversion to point cloud.

```typescript
// In main.ts, after scene setup:
const houseScene = await loadHouseScene('/house.glb');
scene.add(houseScene.root);
```

**TIP:** Three.js GLTFLoader already puts the geometry and scene hierarchy into `gltf.scene`. The mesh nodes are ready to render. The extras metadata rides along as `node.userData`. This is a thin extraction layer, not a conversion.

---

## Track B: Room-Aware Camera Navigation

### B1. Create `viewer/src/roomCamera.ts`

Camera controller that understands room boundaries and transitions between rooms via doors.

```typescript
export class RoomCamera {
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private currentRoom: HouseNode;
  private rooms: Map<string, HouseNode>;
  private transitioning: boolean = false;

  /** Smoothly move camera to center on the given room. */
  goToRoom(roomStarId: string): void {
    const room = this.rooms.get(roomStarId);
    if (!room) return;
    this.currentRoom = room;
    // Animate camera target to room's housePosition
    // Room center = housePosition; camera offset = (0, 5, 10) relative to room center
    // Use lerp over ~1 second for smooth transition
  }

  /** Called in animation loop. */
  update(delta: number): void {
    if (this.transitioning) {
      // Lerp camera position and target toward destination
      // When close enough, snap and set transitioning = false
    }
  }
}
```

### B2. Room AABB visualization (optional but helpful)

When the user is in a room, subtly dim or reduce opacity of objects in other rooms. This gives spatial context without hiding content.

```typescript
// For each non-current room, set its mesh materials to 30% opacity
// For current room, full opacity
```

This is a nice-to-have, not required. If complex, skip it.

### B3. Initial camera position

On load, camera starts centered on `room_library` (the first room, position `(0, 0, 0)`). Camera looks at the room center from above and slightly behind: `camera.position.set(0, 8, 12)`, `controls.target.set(0, 0, 0)`.

---

## Track C: DoorsApp Integration with Nav Graph

### C1. Feed nav_graph to DoorsApp

The existing `DoorsApp` in `apps.ts` already handles `doors_list` events and renders door buttons. Wire it to the House nav_graph:

```typescript
// When house scene loads, emit doors_list event to tablet:
const doorItems = houseScene.doors.map(door => {
  // Parse behavior_rpn to get connected rooms
  const tokens = door.behaviorRpn.split(' ');
  const roomA = tokens[2]; // "House/Library"
  const roomB = tokens[3]; // "House/Garden"
  return {
    label: door.surfaceForms.en?.word_ref || door.starId,
    address: `${roomA} <-> ${roomB}`,
    starId: door.starId,
    roomA,
    roomB,
  };
});
tablet.publish({ type: 'doors_list', payload: { items: doorItems } });
```

### C2. Handle door activation

When user clicks a door button in DoorsApp, determine which room to navigate to (the OTHER room from current), then call `roomCamera.goToRoom()`:

```typescript
// In main.ts local handler for 'openDoor' events:
case 'openDoor': {
  const door = houseScene.nodesByStarId.get(payload.starId);
  if (!door) break;
  // Parse which two rooms the door connects
  const tokens = door.behaviorRpn.split(' ');
  const rooms = [tokens[2], tokens[3]]; // House/Library, House/Garden
  // Go to the OTHER room (not the current one)
  const targetRoom = rooms.find(r => !r.endsWith(houseScene.currentRoom.replace('room_', '')));
  // Map House/Library -> room_library
  const targetStarId = roomStarIdFromHouseRoom(targetRoom);
  roomCamera.goToRoom(targetStarId);
  houseScene.currentRoom = targetStarId;
  break;
}
```

**TIP:** The nav_graph already has the adjacency data. Use `navGraph.edges` to find which rooms are reachable from the current room via which doors.

### C3. Door mesh raycasting (optional)

If time permits, add raycasting on door meshes — clicking a door mesh in the 3D scene (not just the tablet button) triggers room transition. Use `THREE.Raycaster` on click, check if intersected mesh has `userData.k3d.meaning_class === "door"`.

---

## Track D: House Object Tooltips

### D1. Hover tooltips for House objects

When the mouse hovers over a House object (room, furniture, tool, display, etc.), show a tooltip with the object's surface form in the current language.

```typescript
// In animation loop, after raycasting:
const intersects = raycaster.intersectObjects(houseScene.root.children, true);
if (intersects.length > 0) {
  const hit = intersects[0].object;
  const k3d = hit.userData?.k3d;
  if (k3d?.star_id) {
    const node = houseScene.nodesByStarId.get(k3d.star_id);
    if (node) {
      showTooltip(node.surfaceForms.en?.word_ref || node.starId);
    }
  }
}
```

### D2. Tooltip rendering

Use a simple DOM overlay div positioned at the mouse cursor. No need for a 3D label — DOM tooltips are more readable and lighter.

```typescript
// Create a div#house-tooltip in the HTML
// On hover: set innerHTML, position at mouse coords, display: block
// On unhover: display: none
```

---

## Tips for Codex

**Tip 1 — userData, not extras.** Three.js GLTFLoader puts GLTF `extras` into `Object3D.userData`. So `gltf.scene.traverse(node => { if (node.userData?.k3d) { ... } })` is how you access the K3D metadata after loading.

**Tip 2 — Don't modify loadK3D.ts.** The existing `loadK3DFromGLTF` handles point cloud payloads. The House loader is a SEPARATE path (`loadHouseScene.ts`). They coexist — the viewer can load both point cloud data AND the House scene simultaneously.

**Tip 3 — Scene graph is free.** The House GLB already has the scene hierarchy (House root → rooms → children). Three.js GLTFLoader reconstructs this as a Three.js Group tree. You get the hierarchy for free — just `scene.add(gltf.scene)`.

**Tip 4 — Room positions are world coordinates.** Room nodes have `house_position` as world translation (Library at 0,0,0; Garden at 18,0,0; etc.). Furniture/tools/displays have room-local positions. Three.js handles this correctly via parent-child transforms.

**Tip 5 — Keep DoorsApp backward compatible.** The existing DoorsApp expects `{ items: [{label, address}, ...] }`. Add `starId` to the item type but keep `label` and `address` so the existing UI works. The new behavior hooks into the `openDoor` event handler.

**Tip 6 — Camera transition.** Use `THREE.Vector3.lerp` in the animation loop for smooth camera movement. Transition time ~1 second. Don't use TWEEN libraries — keep it dependency-free.

**Tip 7 — GLB path.** The House GLB is at `viewer/public/house.glb`. The viewer serves files from `public/`, so the URL is just `/house.glb`.

---

## Tests

### `tests/test_viewer_house_scene.py` (Python-side validation)

These tests verify the House GLB has the correct structure for the viewer to consume:

```python
def test_house_glb_nodes_have_k3d_extras(tmp_path):
    """Every procedural node carries extras.k3d for the viewer."""
    output = tmp_path / "house.glb"
    export_house_glb(output)
    pygltflib = pytest.importorskip("pygltflib")
    gltf = pygltflib.GLTF2().load(str(output))
    k3d_nodes = [n for n in gltf.nodes if isinstance(n.extras, dict) and "k3d" in n.extras]
    assert len(k3d_nodes) >= 54  # all nodes + House root

def test_house_root_has_nav_graph(tmp_path):
    output = tmp_path / "house.glb"
    export_house_glb(output)
    pygltflib = pytest.importorskip("pygltflib")
    gltf = pygltflib.GLTF2().load(str(output))
    house_node = next(n for n in gltf.nodes if n.name == "House")
    nav = house_node.extras["k3d"]["nav_graph"]
    assert len(nav["nodes"]) == 5
    assert len(nav["edges"]) == 8

def test_door_nodes_have_behavior_rpn(tmp_path):
    """Viewer needs behavior_rpn to determine door connectivity."""
    output = tmp_path / "house.glb"
    export_house_glb(output)
    pygltflib = pytest.importorskip("pygltflib")
    gltf = pygltflib.GLTF2().load(str(output))
    doors = [n for n in gltf.nodes if isinstance(n.extras, dict) and n.extras.get("k3d", {}).get("meaning_class") == "door"]
    assert len(doors) == 4
    for door in doors:
        assert "DOOR_TRAVERSE CONNECT" in door.extras["k3d"]["behavior_rpn"]
```

### Viewer-side tests (TypeScript)

If the viewer has a test setup (vitest, jest), add:

```typescript
// test/loadHouseScene.test.ts
test('loadHouseScene extracts rooms from house.glb', async () => {
  const scene = await loadHouseScene('/house.glb');
  expect(scene.rooms.length).toBe(5);
  expect(scene.doors.length).toBe(4);
  expect(scene.navGraph.nodes.length).toBe(5);
  expect(scene.navGraph.edges.length).toBe(8);
});

test('nodesByStarId maps all nodes', async () => {
  const scene = await loadHouseScene('/house.glb');
  expect(scene.nodesByStarId.has('room_library')).toBe(true);
  expect(scene.nodesByStarId.has('memory_tablet')).toBe(true);
});
```

If no test infrastructure exists for the viewer, these tests can wait. The Python-side GLB validation tests are sufficient for this phase.

### Non-regression

All existing Python tests must pass. GPU math non-regression: `test_math_first_twenty_problems_stay_green_on_gpu_path`.

---

## Success Criteria

1. `loadHouseScene()` loads House GLB and extracts all HouseNode data from per-node `extras.k3d`
2. House scene renders in viewer with all 54 nodes visible as actual 3D geometry
3. Camera starts centered on Library room
4. DoorsApp shows 4 doors from nav_graph, clicking a door transitions camera to connected room
5. Hover tooltips show object names (surface forms) on mouse over
6. Existing viewer functionality (point cloud, tablet apps) unchanged
7. All existing Python tests pass

---

## Files Changed/Created

| File | Action |
|------|--------|
| `viewer/src/loadHouseScene.ts` | **NEW** — House scene loader extracting per-node extras.k3d |
| `viewer/src/roomCamera.ts` | **NEW** — Room-aware camera with smooth transitions |
| `viewer/src/main.ts` | Wire loadHouseScene, roomCamera, door events, tooltips |
| `viewer/src/apps.ts` | Update DoorsApp item type to include starId |
| `tests/test_viewer_house_scene.py` | **NEW** — Python-side GLB structure validation |
| `tests/test_gltf_export.py` | Add nav_graph + door structure assertions |

---

## Architectural Note

This phase bridges two paradigms: the **procedural House** (per-node metadata, scene hierarchy, room geometry) and the **semantic viewer** (point clouds, neighbor graphs, embeddings). They coexist — the viewer renders BOTH the House as a 3D scene and the Galaxy data as point clouds. The House is where the avatar LIVES (spatial reality); the Galaxy is what the avatar THINKS (semantic workspace). Same viewer, two rendering modes, unified by the Memory Tablet.

This is the Dual Client Contract in action: the same House GLB serves the human (visible rooms, clickable doors, readable tooltips) and the AI (nav_graph, behavior_rpn, taxonomy_refs). One artifact, two clients.
