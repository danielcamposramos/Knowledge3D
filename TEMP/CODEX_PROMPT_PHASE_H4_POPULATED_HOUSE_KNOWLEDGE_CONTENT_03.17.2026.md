# Phase H4: Populated House — Filling Containers with Real Knowledge

**Spec Author:** Claude (Architecture Partner)
**Date:** March 17, 2026
**Depends on:** Phase H3 (container objects + loadable Galaxies) COMPLETE
**Sovereignty:** Ingestion path (flexible). All content is Galaxy entries referencing existing foundational entries.

---

## Context

The House structure is built (5 rooms, 6 furniture, 5 books). The container pattern works (galaxy_ref → loadable Galaxy). Two content Galaxies exist (Mathematics Primer: 17 entries, Language Foundations: 17 entries).

Three books have `galaxy_ref` but no content Galaxy yet:
- `book_physics_handbook` → `Book/PhysicsHandbook` (empty)
- `book_biology_atlas` → `Book/BiologyAtlas` (empty)
- `book_tool_manual` → `Book/ToolManual` (empty)

The Reality Galaxy already has **46+ physics entries** (kinematics, dynamics, E&M, thermo), **5+ biology entries**, and **5+ chemistry entries**. The Tool Galaxy has **47 entries**. These are the EXISTING entries that book content should REFERENCE — not duplicate.

---

## Deliverables

### Track A: Physics Handbook Content Galaxy (~20 entries)
### Track B: Biology Atlas Content Galaxy (~15 entries)
### Track C: Tool Manual Content Galaxy (~15 entries)
### Track D: Knowledge Tree Ontological Structure

---

## Track A: Physics Handbook Content Galaxy

**File:** `knowledge3d/knowledgeverse/book_content_physics.py`

Follow the exact pattern from `book_content_mathematics.py`: a `_book_star()` helper, entries with `meaning_class` in {chapter, section, page}, domain `"Book/PhysicsHandbook"`.

**Structure (4 chapters, ~20 entries total):**

**Chapter 1: Motion** — references kinematics entries
```
physicsbook_ch1_motion
├── physicsbook_sec1_position_velocity
│   taxonomy_refs: ["reality_kinematics_position_update_euler",
│                   "reality_kinematics_velocity_update_euler",
│                   "reality_kinematics_average_velocity"]
├── physicsbook_sec1_acceleration
│   taxonomy_refs: ["reality_kinematics_acceleration_definition",
│                   "reality_kinematics_constant_acceleration"]
├── physicsbook_sec1_projectiles
│   taxonomy_refs: ["reality_kinematics_projectile_2d",
│                   "reality_kinematics_projectile_range",
│                   "reality_kinematics_projectile_time_of_flight"]
└── physicsbook_page_circular_motion
    taxonomy_refs: ["reality_kinematics_uniform_circular_speed",
                    "reality_kinematics_relative_velocity"]
```

**Chapter 2: Forces and Energy** — references dynamics entries
```
physicsbook_ch2_forces
├── physicsbook_sec2_newton_laws
│   taxonomy_refs: ["reality_dynamics_newton_second_law",
│                   "reality_dynamics_friction_force",
│                   "reality_dynamics_hooke_law"]
├── physicsbook_sec2_energy
│   taxonomy_refs: ["reality_dynamics_kinetic_energy",
│                   "reality_dynamics_potential_energy_gravity",
│                   "reality_dynamics_energy_conservation",
│                   "reality_dynamics_work", "reality_dynamics_power"]
├── physicsbook_sec2_momentum
│   taxonomy_refs: ["reality_dynamics_momentum_linear",
│                   "reality_dynamics_impulse",
│                   "reality_dynamics_momentum_conservation_1d"]
└── physicsbook_page_collisions
    taxonomy_refs: ["reality_dynamics_elastic_collision_1d",
                    "reality_dynamics_inelastic_collision_1d",
                    "reality_dynamics_center_of_mass_velocity"]
```

**Chapter 3: Electricity and Magnetism** — references E&M entries
```
physicsbook_ch3_em
├── physicsbook_sec3_electrostatics
│   taxonomy_refs: ["reality_em_coulomb_force",
│                   "reality_em_electric_field_point",
│                   "reality_em_electric_potential_point",
│                   "reality_em_gauss_law"]
├── physicsbook_sec3_circuits
│   taxonomy_refs: ["reality_em_ohm_current", "reality_em_ohm_voltage",
│                   "reality_em_resistor_power",
│                   "reality_em_series_resistance",
│                   "reality_em_parallel_resistance",
│                   "reality_em_capacitor_charge"]
└── physicsbook_page_induction
    taxonomy_refs: ["reality_em_faraday_law",
                    "reality_em_ampere_law",
                    "reality_em_lorentz_force"]
```

**Chapter 4: Heat and Thermodynamics** — references thermo entries
```
physicsbook_ch4_thermo
├── physicsbook_sec4_temperature
│   taxonomy_refs: ["reality_thermo_ideal_gas_temperature",
│                   "reality_thermo_ideal_gas_pressure",
│                   "reality_thermo_ideal_gas_volume",
│                   "reality_thermo_heat_capacity"]
├── physicsbook_sec4_energy_transfer
│   taxonomy_refs: ["reality_thermo_first_law",
│                   "reality_thermo_conduction_rate",
│                   "reality_thermo_phase_change_heat",
│                   "reality_thermo_thermal_energy"]
└── physicsbook_page_engines
    taxonomy_refs: ["reality_thermo_carnot_efficiency",
                    "reality_thermo_entropy_change"]
```

All entries use `grammar_refs` where applicable (e.g., `"sequential_computation"`, `"rate_application"`, `"comparison_delta"`).

---

## Track B: Biology Atlas Content Galaxy

**File:** `knowledge3d/knowledgeverse/book_content_biology.py`

Domain: `"Book/BiologyAtlas"`. ~15 entries across 3 chapters.

**Chapter 1: The Cell**
```
biobook_ch1_cell
├── biobook_sec1_cell_structure
│   taxonomy_refs: ["reality_biology_cell_theory",
│                   "reality_anchor_college_biology_core"]
├── biobook_sec1_genetics
│   taxonomy_refs: ["reality_biology_genetics_inheritance"]
└── biobook_page_dna_rna
    taxonomy_refs: ["reality_biology_cell_theory",
                    "reality_biology_genetics_inheritance"]
```

**Chapter 2: Organisms and Evolution**
```
biobook_ch2_organisms
├── biobook_sec2_evolution
│   taxonomy_refs: ["reality_biology_evolution_selection"]
├── biobook_sec2_homeostasis
│   taxonomy_refs: ["reality_biology_homeostasis"]
└── biobook_page_adaptation
    taxonomy_refs: ["reality_biology_evolution_selection",
                    "concept_growth"]
```

**Chapter 3: Ecology**
```
biobook_ch3_ecology
├── biobook_sec3_populations
│   taxonomy_refs: ["reality_biology_ecology_populations"]
├── biobook_sec3_energy_flow
│   taxonomy_refs: ["reality_biology_ecology_populations",
                    "reality_thermo_first_law"]
│   (cross-domain reference! Biology uses thermodynamics)
└── biobook_page_systems_view
    taxonomy_refs: ["concept_biology", "concept_physics", "concept_growth"]
```

**Note the cross-domain reference in sec3_energy_flow** — ecology references thermodynamics. This is the multi-modal Galaxy pattern in action: knowledge from different domains connects naturally.

---

## Track C: Tool Manual Content Galaxy

**File:** `knowledge3d/knowledgeverse/book_content_tools.py`

Domain: `"Book/ToolManual"`. ~15 entries across 3 chapters.

**Chapter 1: Math Core Tools**
```
toolbook_ch1_math
├── toolbook_sec1_scalar
│   taxonomy_refs: ["tool_mathcore_tier1_scalar_worker_worker_v1"]
├── toolbook_sec1_vector
│   taxonomy_refs: ["tool_mathcore_tier2_vector_worker_v1"]
├── toolbook_sec1_orchestration
│   taxonomy_refs: ["tool_mathcore_tier3_master_v1",
│                   "tool_mathcore_spawn_cascade_v1"]
└── toolbook_page_cascade_pattern
    taxonomy_refs: ["tool_mathcore_spawn_cascade_v1",
                    "concept_tool"]
```

**Chapter 2: Geometry and Construction**
```
toolbook_ch2_geometry
├── toolbook_sec2_profiles
│   taxonomy_refs: ["tool_geom_profile_prep_v1",
│                   "tool_geom_profile_lathe_mesh_v1",
│                   "tool_geom_profile_extrude_mesh_v1",
│                   "tool_geom_profile_sweep_mesh_v1"]
├── toolbook_sec2_bounding
│   taxonomy_refs: ["tool_geom_bbox_crop_v1"]
└── toolbook_page_mesh_fusion
    taxonomy_refs: ["tool_fusion_contour_to_mesh_v1",
                    "concept_visual_art"]
```

**Chapter 3: Media and Codecs**
```
toolbook_ch3_media
├── toolbook_sec3_visual
│   taxonomy_refs: ["tool_paint_gradient_backdrop_v1",
│                   "tool_paint_filter_stack_v1",
│                   "tool_paint_composite_edge_v1",
│                   "tool_paint_palette_contrastive_v1"]
├── toolbook_sec3_audio
│   taxonomy_refs: ["tool_signal_audio_spectrogram_v1",
│                   "tool_signal_spectrogram_surface_v1",
│                   "tool_codec_audio_mdct_v1"]
└── toolbook_page_encoding
    taxonomy_refs: ["tool_codec_ternary_blocks_v1",
                    "tool_codec_video_dct8_grid_v1"]
```

---

## Track D: Knowledge Tree Ontological Structure

The Garden has a `furniture_knowledge_tree` with `visual_rpn = "0.15 2.0 8 1 GEN_CYLINDER 1.5 1 GEN_ICOSPHERE ..."` — a plain trunk + sphere canopy. This needs to become a real ontological structure.

### D1. Create `house_knowledge_tree.py`

**File:** `knowledge3d/knowledgeverse/house_knowledge_tree.py`

Define tree branch nodes as MeaningCentricStars. Each branch is a 3D object (cylinder angled from trunk) that represents a knowledge domain:

```python
KNOWLEDGE_TREE_BRANCHES: list[MeaningCentricStar] = [
    # Main trunk (already exists as furniture_knowledge_tree)
    # Branch 1: Mathematics (angled right)
    MeaningCentricStar(
        star_id="tree_branch_mathematics",
        meaning_class="branch",
        meaning_rpn="BRANCH MATHEMATICS DOMAIN_LINK",
        domain="House/Garden/KnowledgeTree",
        visual_rpn=(
            "0.06 1.2 6 1 GEN_CYLINDER "
            "0.7854 MAT4_ROTATE_Z MAT4_APPLY "  # 45 degrees
            "0.0 1.6 0.4 MAT4_TRANSLATE MAT4_APPLY"
        ),
        galaxy_ref="",  # branches don't load galaxies, they're navigational
        taxonomy_refs=["concept_mathematics"],
        component_refs=["tree_leaf_numbers", "tree_leaf_operations", "tree_leaf_patterns"],
        house_position=(5.0, 1.6, 5.4),
        house_room="House/Garden",
        ...
    ),
    # Branch 2: Language (angled left)
    # Branch 3: Physics (angled forward)
    # Branch 4: Biology (angled back)
    # Branch 5: Tools (angled up-right)
]
```

Create **5 branches** (one per seed concept domain that has a Library book) and **15 leaf nodes** (3 per branch). Each leaf is a small sphere referencing a key concept from that domain:

```python
MeaningCentricStar(
    star_id="tree_leaf_numbers",
    meaning_class="leaf",
    meaning_rpn="LEAF NUMBERS COUNTING CARDINAL",
    domain="House/Garden/KnowledgeTree",
    visual_rpn="0.08 0 GEN_ICOSPHERE 0.85 2.1 0.7 MAT4_TRANSLATE MAT4_APPLY",
    taxonomy_refs=["concept_mathematics", "num_0", "num_1", "num_2"],
    ...
)
```

### D2. Update `furniture_knowledge_tree` in `house_furniture.py`

Add `component_refs` linking to the branch star_ids:

```python
component_refs=["tree_branch_mathematics", "tree_branch_language",
                "tree_branch_physics", "tree_branch_biology",
                "tree_branch_tools"],
```

### D3. Register tree branches in `house_builder.py`

Import `KNOWLEDGE_TREE_BRANCHES` and store them in the House Galaxy alongside other objects. Generate meshes for all branches and leaves.

---

## Registration

### Update `foundational_galaxy_bootstrap.py`

Extend `populate_book_galaxies()` to also populate:
- `Book/PhysicsHandbook` from `book_content_physics.PHYSICS_HANDBOOK_ENTRIES`
- `Book/BiologyAtlas` from `book_content_biology.BIOLOGY_ATLAS_ENTRIES`
- `Book/ToolManual` from `book_content_tools.TOOL_MANUAL_ENTRIES`

### Update `__init__.py`

Export `KNOWLEDGE_TREE_BRANCHES` and the new entry lists.

---

## Tests

### `tests/test_book_content_all.py`

```python
def test_physics_handbook_entries_reference_reality_galaxy():
    for star in PHYSICS_HANDBOOK_ENTRIES:
        assert star.domain == "Book/PhysicsHandbook"
        assert star.meaning_class in {"chapter", "section", "page"}
        # Physics entries should reference reality_* entries
        all_refs = star.taxonomy_refs + star.grammar_refs
        assert any(ref.startswith("reality_") or ref.startswith("concept_")
                    for ref in all_refs), f"{star.star_id} has no reality refs"

def test_biology_atlas_entries_reference_biology():
    for star in BIOLOGY_ATLAS_ENTRIES:
        assert star.domain == "Book/BiologyAtlas"
        assert star.meaning_class in {"chapter", "section", "page"}

def test_tool_manual_entries_reference_tools():
    for star in TOOL_MANUAL_ENTRIES:
        assert star.domain == "Book/ToolManual"
        assert star.meaning_class in {"chapter", "section", "page"}
        all_refs = star.taxonomy_refs
        assert any(ref.startswith("tool_") or ref.startswith("concept_")
                    for ref in all_refs), f"{star.star_id} has no tool refs"

def test_all_book_galaxies_load_on_demand(tmp_path):
    manager = GalaxyManager(storage_root=tmp_path / "galaxies")
    populate_book_galaxies(manager)
    build_house(manager)
    for book in HOUSE_BOOKS:
        galaxy = manager.load_galaxy_on_demand(book)
        assert galaxy is not None, f"{book.star_id} galaxy not loaded"
        assert len(galaxy.entries) > 0, f"{book.star_id} galaxy is empty"

def test_cross_domain_references_in_biology():
    """Biology ecology chapter references thermodynamics — multi-modal pattern."""
    ecology_sections = [s for s in BIOLOGY_ATLAS_ENTRIES
                        if "energy" in s.star_id.lower()]
    assert any("reality_thermo" in ref
               for s in ecology_sections for ref in s.taxonomy_refs)
```

### `tests/test_knowledge_tree.py`

```python
def test_knowledge_tree_branches_have_valid_rpn():
    bridge = MeshBridge()
    for branch in KNOWLEDGE_TREE_BRANCHES:
        assert branch.meaning_class in {"branch", "leaf"}
        result = bridge.execute_rpn_program(branch.visual_rpn)
        assert result.mesh.vertices

def test_knowledge_tree_branches_reference_domains():
    branch_domains = {b.star_id for b in KNOWLEDGE_TREE_BRANCHES
                      if b.meaning_class == "branch"}
    assert len(branch_domains) >= 5  # one per major domain

def test_knowledge_tree_leaves_reference_concepts():
    for leaf in KNOWLEDGE_TREE_BRANCHES:
        if leaf.meaning_class == "leaf":
            assert leaf.taxonomy_refs  # must reference actual concepts
```

### Existing test non-regression

All existing tests must pass:
- `pytest -q tests/test_house_books.py tests/test_house_builder.py tests/test_meaning_star.py tests/test_mesh_opcodes.py`
- GPU math: `test_math_first_twenty_problems_stay_green_on_gpu_path`

---

## Success Criteria

1. All 5 book `galaxy_ref` targets have populated content Galaxies
2. Physics book references 46+ Reality Galaxy entries across 4 chapters
3. Biology book has cross-domain reference (ecology → thermodynamics)
4. Tool manual references 47 Tool Galaxy entries across 3 chapters
5. Knowledge tree has 5 branches + 15 leaves with constructable visual_rpn
6. All existing tests pass, GPU math non-regression holds

---

## Files Changed/Created

| File | Action |
|------|--------|
| `knowledge3d/knowledgeverse/book_content_physics.py` | **NEW** — Physics Handbook entries |
| `knowledge3d/knowledgeverse/book_content_biology.py` | **NEW** — Biology Atlas entries |
| `knowledge3d/knowledgeverse/book_content_tools.py` | **NEW** — Tool Manual entries |
| `knowledge3d/knowledgeverse/house_knowledge_tree.py` | **NEW** — Tree branches + leaves |
| `knowledge3d/knowledgeverse/house_furniture.py` | Update knowledge_tree component_refs |
| `knowledge3d/knowledgeverse/house_builder.py` | Register tree branches, generate meshes |
| `knowledge3d/knowledgeverse/foundational_galaxy_bootstrap.py` | Populate 3 new book galaxies |
| `knowledge3d/knowledgeverse/__init__.py` | Exports |
| `tests/test_book_content_all.py` | **NEW** — content galaxy tests |
| `tests/test_knowledge_tree.py` | **NEW** — tree structure tests |
