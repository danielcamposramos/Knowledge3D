# Phase H11b: Living Room + HoloDesk — The Collaboration Space

**Spec Author:** Claude (Architecture Partner)
**Date:** March 17, 2026
**Depends on:** Phase H7 (nav graph), Phase H10 (behavior activation) COMPLETE
**Sovereignty:** Ingestion path (Python, flexible).
**Build:** Use `bash /K3D/Knowledge3D.local/envs/viewer-build/build.sh` for viewer. Standard pytest for Python.

---

## Context

The House currently has 5 rooms: Library, Garden, Workshop, Gallery, Bathtub. The original architectural vision (cross-section render) shows a **Living Room** as the central collaborative space — the room where the sofa faces floating holographic displays.

This phase adds:
1. **Room: Living Room** — the main collaborative space, positioned between Library and Garden
2. **Furniture: Sofa** — seating oriented toward the center of the room
3. **Furniture: HoloDesk** — a low center table that IS a planar 3D projection surface

The HoloDesk is architecturally significant. It completes K3D's **three projection surface geometries**:

| Surface | Location | Geometry | Purpose |
|---------|----------|----------|---------|
| Memory Tablet | House root | Flat slab | 2D interface (Canvas/DOM) |
| **HoloDesk** | **Living Room** | **Table + projection plane** | **3D collaborative (planar holographic)** |
| Bathtub Bubble | Bathtub | Sphere | 3D introspective (stellarium) |

The HoloDesk represents **EchoSystems HoloDesk** — a virtual 3D projection surface for augmented collaboration. In the K3D House, it's where shared 3D models float above the table for human and AI co-exploration. In future phases, this is where remote collaborators' avatars materialize for AR/VR sessions.

---

## Deliverables

### Track A: Living Room + Furniture Stars
### Track B: Door Connectivity
### Track C: Builder + Export Integration

---

## Track A: Living Room + Furniture Stars

### A1. Add Living Room to `house_rooms.py`

Insert `room_living` into `HOUSE_ROOMS`. The Living Room is positioned between the Library and the Garden — it's the central hub of the House.

```python
MeaningCentricStar(
    star_id="room_living",
    meaning_class="room",
    meaning_rpn="ROOM COLLABORATION SHARED PROJECTION DOMAIN_CENTER",
    domain="House/LivingRoom",
    visual_rpn=(
        # 10×10 room shell with door openings on two sides
        "10.0 GEN_CUBE 9.4 GEN_CUBE CSG_SUBTRACT "
        # Door opening toward Library (left wall)
        "1.0 GEN_CUBE 1.3 1.6 0.35 MAT4_SCALE MAT4_APPLY -5.0 -1.0 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_SUBTRACT "
        # Door opening toward Garden (right wall)
        "1.0 GEN_CUBE 1.3 1.6 0.35 MAT4_SCALE MAT4_APPLY 5.0 -1.0 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_SUBTRACT"
    ),
    behavior_rpn="ROOM_ENTER LOAD_KNOWLEDGE_DOMAIN COLLABORATION ACTIVATE_HOLODESK",
    surface_forms=surface_forms("living room", "sala de estar", "リビングルーム"),
    house_position=(-10.0, 0.0, 0.0),  # To the LEFT of Library (Library is at 0,0,0)
    house_room="House/LivingRoom",
    confidence=1,
    polarity=1,
    taxonomy_refs=["concept_visual_art", "concept_tool", "concept_language"],
    component_refs=[
        "furniture_sofa",
        "furniture_holodesk",
    ],
),
```

**Position note:** Library is at `(0,0,0)`. Living Room at `(-10,0,0)` places it to the left. This makes the Living Room the FIRST room — the entry point. Door between them at the shared wall.

### A2. Add Sofa to `house_furniture.py`

```python
MeaningCentricStar(
    star_id="furniture_sofa",
    meaning_class="furniture",
    meaning_rpn="SOFA SEATING COMFORT COLLABORATION SHARED DOMAIN_CENTER",
    domain="House/LivingRoom/Furniture",
    visual_rpn=(
        # Seat cushion: wide, low, deep
        "1.0 GEN_CUBE 2.4 0.45 1.0 MAT4_SCALE MAT4_APPLY "
        "0.0 0.3 0.0 MAT4_TRANSLATE MAT4_APPLY "
        # Backrest: tall thin slab behind seat
        "1.0 GEN_CUBE 2.4 0.7 0.2 MAT4_SCALE MAT4_APPLY "
        "0.0 0.8 -0.5 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
        # Left armrest
        "1.0 GEN_CUBE 0.15 0.55 0.9 MAT4_SCALE MAT4_APPLY "
        "-1.2 0.55 -0.05 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
        # Right armrest
        "1.0 GEN_CUBE 0.15 0.55 0.9 MAT4_SCALE MAT4_APPLY "
        "1.2 0.55 -0.05 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
    ),
    behavior_rpn="SUPPORT SEATED COLLABORATE",
    surface_forms=surface_forms("sofa", "sofa", "ソファ"),
    house_position=(0.0, 0.0, -2.5),  # Back wall of Living Room
    house_room="House/LivingRoom",
    confidence=1,
    polarity=1,
    taxonomy_refs=["concept_visual_art", "concept_language"],
),
```

### A3. Add HoloDesk to `house_furniture.py`

The HoloDesk is a LOW center table (coffee table height ~0.4m) with a flat projection surface on top. The table itself is simple; the magic is in its behavior_rpn and what it represents.

```python
MeaningCentricStar(
    star_id="furniture_holodesk",
    meaning_class="furniture",
    meaning_rpn="HOLODESK PROJECTION SURFACE COLLABORATION 3D AUGMENTED DOMAIN_CENTER",
    domain="House/LivingRoom/Furniture",
    visual_rpn=(
        # Table top: wide, thin, rectangular (the projection surface)
        "1.0 GEN_CUBE 1.6 0.04 0.9 MAT4_SCALE MAT4_APPLY "
        "0.0 0.42 0.0 MAT4_TRANSLATE MAT4_APPLY "
        # Four short legs
        "0.04 0.40 8 1 GEN_CYLINDER -0.7 0.20 -0.38 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
        "0.04 0.40 8 1 GEN_CYLINDER  0.7 0.20 -0.38 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
        "0.04 0.40 8 1 GEN_CYLINDER -0.7 0.20  0.38 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
        "0.04 0.40 8 1 GEN_CYLINDER  0.7 0.20  0.38 MAT4_TRANSLATE MAT4_APPLY CSG_UNION "
        # Thin glowing rim around the edge (slightly larger than top)
        "1.0 GEN_CUBE 1.68 0.02 0.98 MAT4_SCALE MAT4_APPLY "
        "1.0 GEN_CUBE 1.56 0.04 0.86 MAT4_SCALE MAT4_APPLY CSG_SUBTRACT "
        "0.0 0.45 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION"
    ),
    behavior_rpn="HOLODESK ACTIVATE PROJECT_3D COLLABORATE SHARE_MODELS",
    surface_forms=surface_forms("HoloDesk", "HoloMesa", "ホロデスク"),
    house_position=(0.0, 0.0, 0.0),  # Center of Living Room, in front of sofa
    house_room="House/LivingRoom",
    confidence=1,
    polarity=1,
    taxonomy_refs=[
        "concept_visual_art",
        "concept_tool",
        "concept_mathematics",
        "concept_language",
    ],
),
```

**Visual design:**
- Coffee table height (0.42m top surface) — low, in front of sofa
- Wide enough for shared viewing (1.6m × 0.9m)
- Thin legs (0.04 radius cylinders)
- Rim frame around the edge (CSG_SUBTRACT creates the hollow frame) — this is the "glowing edge" from the render
- The flat top IS the projection surface — in future phases, 3D content renders ABOVE this plane

---

## Track B: Door Connectivity

### B1. Add doors connecting Living Room

The Living Room connects to the Library (which connects to everything else). Add a new door to `house_doors.py`:

```python
_door(
    star_id="door_living_library",
    title_en="Living Room Library Door",
    title_pt="Porta Sala Biblioteca",
    title_ja="リビングと図書館の扉",
    room_a="House/LivingRoom",
    room_b="House/Library",
    house_position=(9.0, 0.0, 0.0),
    rotation_y=math.pi / 2.0,
),
```

Insert this as the FIRST door in `HOUSE_DOORS` so the nav graph starts from the Living Room.

### B2. Update nav graph traversal order

The room chain becomes: **Living Room → Library → Garden → Workshop → Gallery → Bathtub**

The Living Room is now the entry point — the first room the camera starts in.

---

## Track C: Builder + Export Integration

### C1. No changes to `house_builder.py` or `export_house.py`

The new room and furniture are added to `HOUSE_ROOMS` and `HOUSE_FURNITURE` lists. The builder already iterates those lists. The export already iterates those lists. **No builder/export code changes needed** — the lists drive everything.

### C2. Update `room_living` component_refs in `house_rooms.py`

Already specified in A1 above.

### C3. Update viewer default room

In `viewer/src/main.ts`, the default room is `room_library`. Update it to `room_living` so the viewer starts in the Living Room:

```typescript
// Change default room from 'room_library' to 'room_living'
roomCamera = new RoomCamera(camera, controls, loadedHouseScene.rooms, loadedHouseScene.currentRoom || 'room_living');
roomCamera.snapToRoom(loadedHouseScene.currentRoom || 'room_living');
```

---

## Track D: HoloDesk Behavior Activation

### D1. Add HOLODESK case to behavior interpreter

In `viewer/src/behavior/interpreter.ts`, add a case for the `HOLODESK` command:

```typescript
case 'HOLODESK':
  return { type: 'browse_galaxy' };  // For now, opens Galaxy mode like Tablet
```

This is intentionally simple for H11b. In future phases, `HOLODESK` will activate a specialized 3D projection mode distinct from the Tablet's 2D Galaxy browser. For now, it demonstrates the behavior_rpn activation pipeline.

---

## Tips for Codex

**Tip 1 — Room positioning.** Existing rooms use positive X offsets: Library(0), Garden(18), Workshop(36), Gallery(54), Bathtub(72). Living Room at (-10, 0, 0) places it LEFT of Library. This is correct — it's the entry space.

**Tip 2 — No builder changes.** The builder iterates `HOUSE_ROOMS` and `HOUSE_FURNITURE`. Adding to those lists is sufficient. No structural changes to `house_builder.py` or `export_house.py`.

**Tip 3 — HoloDesk visual_rpn uses same primitives.** GEN_CUBE + GEN_CYLINDER + CSG_SUBTRACT + CSG_UNION + MAT4_SCALE + MAT4_TRANSLATE. All ops already in the known_ops set (test_rpn_parity.py).

**Tip 4 — Door connectivity.** The `_door()` helper handles everything. Just add one call. The nav graph builder will automatically discover the new connection.

**Tip 5 — Default room change.** Grep for `room_library` in main.ts — there are likely 2 references (RoomCamera constructor fallback and snapToRoom fallback). Change both to `room_living`.

**Tip 6 — HoloDesk is semantically a PROJECTION SURFACE, not just furniture.** Its taxonomy_refs span 4 domains (visual art, tool, math, language) because it projects ALL knowledge. It's the Living Room's equivalent of the Observatory's telescope — but for shared, augmented viewing instead of solitary observation.

---

## Tests

### Python tests

Update `tests/test_house_rooms.py` (or create if needed):
```python
def test_living_room_exists():
    from knowledge3d.knowledgeverse.house_rooms import HOUSE_ROOMS
    living = [r for r in HOUSE_ROOMS if r.star_id == "room_living"]
    assert len(living) == 1
    assert living[0].house_room == "House/LivingRoom"

def test_living_room_has_holodesk():
    from knowledge3d.knowledgeverse.house_rooms import HOUSE_ROOMS
    living = [r for r in HOUSE_ROOMS if r.star_id == "room_living"][0]
    assert "furniture_holodesk" in living.component_refs
```

Update `tests/test_house_furniture.py` (or add to existing):
```python
def test_holodesk_furniture():
    from knowledge3d.knowledgeverse.house_furniture import HOUSE_FURNITURE
    holodesk = [f for f in HOUSE_FURNITURE if f.star_id == "furniture_holodesk"]
    assert len(holodesk) == 1
    assert holodesk[0].house_room == "House/LivingRoom"
    assert "HOLODESK" in holodesk[0].behavior_rpn

def test_sofa_furniture():
    from knowledge3d.knowledgeverse.house_furniture import HOUSE_FURNITURE
    sofa = [f for f in HOUSE_FURNITURE if f.star_id == "furniture_sofa"]
    assert len(sofa) == 1
    assert sofa[0].house_room == "House/LivingRoom"
```

### Existing tests

`test_rpn_parity.py` will automatically pick up the new visual_rpn programs (it iterates HOUSE_FURNITURE and HOUSE_ROOMS). All tokens must be in known_ops.

### Non-regression

All existing tests must pass. Rebuild GLB via export_house.py.

---

## Success Criteria

1. Living Room exists as `room_living` with position `(-10, 0, 0)`, connected to Library via door
2. Sofa exists as `furniture_sofa` in Living Room at back wall position
3. HoloDesk exists as `furniture_holodesk` in Living Room at center position
4. HoloDesk visual_rpn produces valid geometry (low table with rim frame)
5. Door `door_living_library` connects Living Room to Library
6. Nav graph traversal: Living Room → Library → Garden → Workshop → Gallery → Bathtub
7. Viewer starts in Living Room by default
8. Clicking HoloDesk activates browse_galaxy (via behavior_rpn)
9. All Python + viewer tests pass, TypeScript clean, build succeeds

---

## Files Changed/Created

| File | Action |
|------|--------|
| `knowledge3d/knowledgeverse/house_rooms.py` | Add `room_living` |
| `knowledge3d/knowledgeverse/house_furniture.py` | Add `furniture_sofa`, `furniture_holodesk` |
| `knowledge3d/knowledgeverse/house_doors.py` | Add `door_living_library` |
| `viewer/src/behavior/interpreter.ts` | Add `HOLODESK` case |
| `viewer/src/main.ts` | Change default room to `room_living` |
| `tests/test_house_rooms.py` | Add Living Room tests |
| `tests/test_house_furniture.py` | Add sofa + HoloDesk tests |

---

## Architectural Note: Three Projection Surfaces

The House now has three distinct projection surface geometries:

**Memory Tablet (2D Flat)**
- Geometry: Flat slab with recessed screen
- Projection: Canvas 2D texture → text, apps, DOM (H11 adds this)
- Purpose: Primary UI, always accessible, app-based interaction
- Analogy: A tablet/phone screen

**HoloDesk (3D Planar)**
- Geometry: Low table with flat projection surface
- Projection: Volumetric content rendered ABOVE the table plane
- Purpose: Shared 3D collaboration, model review, AR/VR bridging
- Analogy: Microsoft HoloLens shared workspace / holographic table
- Future: Where collaborators' avatars materialize for shared sessions

**Bathtub Bubble (3D Spherical)**
- Geometry: Sphere surrounding the bathtub
- Projection: Stellarium-like dome projection of Galaxy state
- Purpose: Introspective full-immersion view of knowledge space
- Analogy: Planetarium dome / stellarium / total immersion sphere
- Future: Sleep-time consolidation visualization (TRM's internal state projected)

These three surfaces = three ways to VIEW the same Galaxy data. The Dual-Client Contract extended to three geometric configurations. Each serves a different cognitive mode: focused work (Tablet), collaborative review (HoloDesk), and deep introspection (Bubble).
