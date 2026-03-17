# Phase H6: Doors, Workshop Tools, and Gallery Displays

**Spec Author:** Claude (Architecture Partner)
**Date:** March 17, 2026
**Depends on:** Phase H5 (House GLTF export) COMPLETE
**Sovereignty:** Ingestion/export path (flexible).

---

## Context

The House is built and visible (190KB GLB, 2789 vertices). Five rooms exist but they're isolated — no physical connections between them. The Workshop has a workbench but no tools. The Gallery has walls but no displays. This phase adds spatial connectivity and fills the remaining empty rooms.

---

## Deliverables

### Track A: Doors Between Rooms
### Track B: Workshop Tool Objects
### Track C: Gallery Display Objects
### Track D: Re-export Updated House GLB

---

## Track A: Doors Between Rooms

### A1. Create `house_doors.py`

**File:** `knowledge3d/knowledgeverse/house_doors.py`

Doors are MeaningCentricStars with `meaning_class = "door"`. Each door is a rectangular opening connecting two rooms. Physically, a door is a thin frame (cube with inner cube subtracted = doorframe shape).

```python
def _door(
    *,
    star_id: str,
    title_en: str, title_pt: str, title_ja: str,
    room_a: str,  # house_room of first room
    room_b: str,  # house_room of second room
    house_position: tuple[float, float, float],
    rotation_y: float = 0.0,  # radians, for doors along different axes
) -> MeaningCentricStar:
    frame_rpn = (
        "1.0 GEN_CUBE 0.2 2.2 1.4 MAT4_SCALE MAT4_APPLY "
        "1.0 GEN_CUBE 0.3 2.0 1.1 MAT4_SCALE MAT4_APPLY CSG_SUBTRACT"
    )
    if rotation_y:
        frame_rpn += f" {rotation_y:.4f} MAT4_ROTATE_Y MAT4_APPLY"
    return MeaningCentricStar(
        star_id=star_id,
        meaning_class="door",
        meaning_rpn=f"DOOR CONNECT {room_a.split('/')[-1].upper()} {room_b.split('/')[-1].upper()}",
        domain="House/Connectivity",
        visual_rpn=frame_rpn,
        behavior_rpn=f"DOOR_TRAVERSE CONNECT {room_a} {room_b}",
        surface_forms=surface_forms(title_en, title_pt, title_ja),
        house_position=house_position,
        house_room=room_a,  # primary room affiliation
        component_refs=[],
        taxonomy_refs=[],
        confidence=1,
        polarity=1,
    )
```

**4 doors connecting adjacent rooms:**

| star_id | Connects | Position (between rooms) |
|---------|----------|------------------------|
| `door_library_garden` | Library ↔ Garden | (9.0, 0.0, 0.0) — midpoint between Library(0,0,0) and Garden(18,0,0) |
| `door_garden_workshop` | Garden ↔ Workshop | (27.0, 0.0, 0.0) — midpoint between Garden(18,0,0) and Workshop(36,0,0) |
| `door_workshop_gallery` | Workshop ↔ Gallery | (45.0, 0.0, 0.0) — midpoint between Workshop(36,0,0) and Gallery(54,0,0) |
| `door_gallery_bathtub` | Gallery ↔ Bathtub | (63.0, 0.0, 0.0) — midpoint between Gallery(54,0,0) and Bathtub(72,0,0) |

Each door's `behavior_rpn` encodes which two rooms it connects, enabling future TRM navigation.

### A2. Register doors in house_builder.py and export

Add doors to `build_house()` and to the GLTF export pipeline in `export_house.py`.

---

## Track B: Workshop Tool Objects

### B1. Create `house_workshop_tools.py`

**File:** `knowledge3d/knowledgeverse/house_workshop_tools.py`

Physical tool objects placed on the Workshop workbench. Each is a simple 3D shape representing a tool category, with `taxonomy_refs` pointing to Tool Galaxy entries.

**5 tool objects:**

| star_id | Shape | Tool Galaxy Refs | Position on workbench |
|---------|-------|-----------------|----------------------|
| `tool_obj_hammer` | Cylinder handle + cube head | `tool_mathcore_tier1_scalar_worker_worker_v1` (basic operations) | (36.0, 1.05, -2.8) |
| `tool_obj_wrench` | L-shaped cube composition | `tool_geom_profile_prep_v1`, `tool_geom_bbox_crop_v1` (geometry tools) | (36.3, 1.05, -2.8) |
| `tool_obj_brush` | Thin cylinder + cone tip | `tool_paint_gradient_backdrop_v1`, `tool_paint_filter_stack_v1` (visual tools) | (36.6, 1.05, -2.8) |
| `tool_obj_tuning_fork` | Y-shaped cylinder composition | `tool_signal_audio_spectrogram_v1`, `tool_codec_audio_mdct_v1` (audio tools) | (36.9, 1.05, -2.8) |
| `tool_obj_lens` | Torus (magnifying shape) | `tool_codec_ternary_blocks_v1`, `tool_codec_video_dct8_grid_v1` (codec tools) | (37.2, 1.05, -2.8) |

Each tool has:
- `meaning_class = "tool_object"`
- `visual_rpn` producing a recognizable tool shape
- `taxonomy_refs` linking to Tool Galaxy entries
- Multilingual surface forms
- `house_room = "House/Workshop"`

Tool shapes should be simple and small (all dimensions under 0.3 units) — they sit ON the workbench surface.

### B2. Update Workshop room `component_refs`

Add tool object star_ids to the Workshop room's `component_refs` in `house_rooms.py`.

---

## Track C: Gallery Display Objects

### C1. Create `house_gallery_displays.py`

**File:** `knowledge3d/knowledgeverse/house_gallery_displays.py`

Display frames on Gallery walls. Each is a thin rectangular frame (cube with inner cube subtracted) mounted at eye level, containing a reference to what it displays.

**4 display frames:**

| star_id | Display Content | Wall Position | Taxonomy Refs |
|---------|----------------|---------------|---------------|
| `display_drawing_primitives` | Drawing Galaxy visual primitives | Left wall | `concept_visual_art`, drawing entries |
| `display_number_line` | Number Galaxy sequence | Back wall | `concept_mathematics`, num_* entries |
| `display_character_forms` | Character Galaxy multilingual glyphs | Right wall | `concept_language`, seed_word entries |
| `display_physics_forces` | Reality Galaxy force diagrams | Front wall | `concept_physics`, reality_dynamics entries |

Each display has:
- `meaning_class = "display"`
- `visual_rpn` = thin frame (cube scaled flat + inner subtract for the "canvas" area)
- `taxonomy_refs` pointing to the Galaxy entries it displays
- `house_room = "House/Gallery"`

Frame shape: `1.0 GEN_CUBE W H 0.04 MAT4_SCALE MAT4_APPLY 1.0 GEN_CUBE (W-0.06) (H-0.06) 0.05 MAT4_SCALE MAT4_APPLY CSG_SUBTRACT`

Vary dimensions slightly per frame (landscape vs portrait orientations).

### C2. Update Gallery room `component_refs`

Add display star_ids to the Gallery room's `component_refs` in `house_rooms.py`.

---

## Track D: Re-export Updated House GLB

### D1. Update `export_house.py`

Import and include doors, workshop tools, and gallery displays in the export pipeline. Each category follows the same pattern as furniture/books:
- Execute visual_rpn → MeshBuffer
- Convert to GltfNodeData with house_position as translation
- Attach as children of their respective room nodes

### D2. Re-generate `viewer/public/house.glb`

Run the export script to produce the updated House GLB with all new objects. The file should grow from ~190KB to ~250-300KB with the additional objects (13 new: 4 doors + 5 tools + 4 displays).

---

## Tests

### `tests/test_house_doors.py`

```python
def test_doors_connect_adjacent_rooms():
    for door in HOUSE_DOORS:
        assert door.meaning_class == "door"
        assert door.behavior_rpn.startswith("DOOR_TRAVERSE")
        assert "CONNECT" in door.behavior_rpn

def test_door_visual_rpn_produces_frame():
    bridge = MeshBridge()
    for door in HOUSE_DOORS:
        result = bridge.execute_rpn_program(door.visual_rpn)
        assert result.mesh.vertices
        assert result.mesh.triangles
```

### `tests/test_house_workshop_tools.py`

```python
def test_workshop_tools_reference_tool_galaxy():
    for tool in WORKSHOP_TOOLS:
        assert tool.meaning_class == "tool_object"
        assert tool.house_room == "House/Workshop"
        assert any(ref.startswith("tool_") for ref in tool.taxonomy_refs)

def test_workshop_tool_shapes_constructable():
    bridge = MeshBridge()
    for tool in WORKSHOP_TOOLS:
        result = bridge.execute_rpn_program(tool.visual_rpn)
        assert result.mesh.vertices
```

### `tests/test_house_gallery_displays.py`

```python
def test_gallery_displays_reference_knowledge():
    for display in GALLERY_DISPLAYS:
        assert display.meaning_class == "display"
        assert display.house_room == "House/Gallery"
        assert display.taxonomy_refs

def test_gallery_display_frames_constructable():
    bridge = MeshBridge()
    for display in GALLERY_DISPLAYS:
        result = bridge.execute_rpn_program(display.visual_rpn)
        assert result.mesh.vertices
```

### Updated export test

```python
def test_exported_house_includes_doors_and_tools(tmp_path):
    output = tmp_path / "house.glb"
    summary = export_house_glb(output)
    assert summary["doors"] >= 4
    assert summary["tools"] >= 5
    assert summary["displays"] >= 4
```

### Non-regression

All existing tests must pass. GPU math non-regression: `test_math_first_twenty_problems_stay_green_on_gpu_path`.

---

## Success Criteria

1. 4 doors connecting adjacent rooms with constructable doorframe visual_rpn
2. 5 tool objects on Workshop workbench referencing Tool Galaxy entries
3. 4 display frames on Gallery walls referencing Drawing/Number/Character/Physics entries
4. All new objects included in GLTF export with proper scene hierarchy
5. Updated `house.glb` viewable in any GLTF viewer
6. All existing tests pass

---

## Files Changed/Created

| File | Action |
|------|--------|
| `knowledge3d/knowledgeverse/house_doors.py` | **NEW** — 4 door objects |
| `knowledge3d/knowledgeverse/house_workshop_tools.py` | **NEW** — 5 tool objects |
| `knowledge3d/knowledgeverse/house_gallery_displays.py` | **NEW** — 4 display frames |
| `knowledge3d/knowledgeverse/house_rooms.py` | Update Workshop + Gallery component_refs |
| `knowledge3d/knowledgeverse/house_builder.py` | Register doors, tools, displays |
| `knowledge3d/tools/export_house.py` | Include new objects in export |
| `knowledge3d/knowledgeverse/__init__.py` | Exports |
| `viewer/public/house.glb` | Re-generated with all objects |
| `tests/test_house_doors.py` | **NEW** |
| `tests/test_house_workshop_tools.py` | **NEW** |
| `tests/test_house_gallery_displays.py` | **NEW** |
| `tests/test_gltf_export.py` | Update export assertions |
