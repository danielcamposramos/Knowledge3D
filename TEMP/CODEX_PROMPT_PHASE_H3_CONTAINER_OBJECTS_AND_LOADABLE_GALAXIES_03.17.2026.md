# Phase H3: Container Objects and Loadable Galaxies

**Spec Author:** Claude (Architecture Partner)
**Date:** March 17, 2026
**Depends on:** Phase H2 (GPU mesh kernels + House templates) COMPLETE
**Sovereignty:** Ingestion path (flexible). Galaxy loading on hot path must use GalaxyManager sovereign patterns.

---

## Critical Paradigm (from Daniel, March 17 2026)

**A book is NOT a star. A book's CONTENTS are a Galaxy.**

- The **star** for "book" is the CONCEPT: what a book IS — its shapes, sizes, definition, symlink words.
- A **specific book** (e.g., "Mathematics Primer") in the House is a **3D asset** — a physical object with covers, pages, spine. When the TRM "opens" it, the book's contents load as a **Galaxy** into the Knowledgeverse.
- Furniture holds 3D asset containers. Bookshelves hold book-objects. Each book-object is a Galaxy loader.
- Same pattern for any knowledge container: journal, encyclopedia, course, scroll = each is a Galaxy when loaded.
- Galaxies gravitate by semantic gravity but remain separated while being composed from lower layers.

---

## Deliverables Overview

### Track A: Container Object Schema (extend MeaningCentricStar)

Add `galaxy_ref: str` field to MeaningCentricStar — a reference to a loadable Galaxy name. When the TRM interacts with this object, GalaxyManager loads the referenced Galaxy.

### Track B: Book 3D Asset Construction

Create book-shaped 3D objects with `visual_rpn` programs (cover + spine + pages). These are House objects — physical forms placed on shelves.

### Track C: Content Galaxies (initial population)

Create 2-3 sample content Galaxies that book-objects reference. These are the actual knowledge that lives INSIDE the book, loaded on demand.

### Track D: H2 Cleanup

Fix dead code and duplicated helpers from Phase H2.

---

## Track A: Container Object Schema

### A1. Add `galaxy_ref` to MeaningCentricStar

**File:** `knowledge3d/knowledgeverse/meaning_star.py`

Add a new optional field to the MeaningCentricStar dataclass:

```python
galaxy_ref: str = ""  # Galaxy name to load when this object is opened/activated
```

Place it near `house_room` (the other House-related fields). Default empty string = not a container.

Update `to_dict()` and `from_dict()` to serialize/deserialize this field. Only include in dict if non-empty (same pattern as other optional fields).

Update `compute_star_id()` to include `galaxy_ref` in the hash if non-empty (it changes the identity of the star — a book pointing to different content is a different star).

### A2. Add `load_galaxy_on_demand()` to GalaxyManager

**File:** `knowledge3d/knowledgeverse/galaxy_manager.py`

Add a method that, given a MeaningCentricStar with a non-empty `galaxy_ref`, ensures the referenced Galaxy is loaded:

```python
def load_galaxy_on_demand(self, star: MeaningCentricStar) -> Galaxy | None:
    """Load the Galaxy referenced by a container object, if any."""
    if not star.galaxy_ref:
        return None
    return self.get_galaxy(star.galaxy_ref)
```

This is intentionally thin — `get_galaxy()` already handles lazy loading. The method exists to make the pattern explicit and discoverable.

### A3. Update `__init__.py` exports

**File:** `knowledge3d/knowledgeverse/__init__.py`

No new exports needed (galaxy_ref is a field, not a new class). Just ensure tests pass with the new field.

---

## Track B: Book 3D Asset Construction

### B1. Create `house_books.py`

**File:** `knowledge3d/knowledgeverse/house_books.py`

Define book-objects as MeaningCentricStars with:
- `meaning_class = "book"` (physical container type)
- `visual_rpn` = construction program for a book shape (cover + spine + pages block)
- `galaxy_ref` = name of the content Galaxy to load
- `house_room = "House/Library"` (placed on shelves)
- Multilingual `surface_forms` (en/pt/ja)

**Book shape construction pattern** (RPN):
A book is a thin rectangular solid (pages) sandwiched between two slightly larger covers, with a spine:

```
# Pages block (inner)
1.0 GEN_CUBE 0.18 0.26 0.03 MAT4_SCALE MAT4_APPLY
# Front cover (thin slab)
1.0 GEN_CUBE 0.19 0.27 0.003 MAT4_SCALE MAT4_APPLY 0.0 0.0 0.017 MAT4_TRANSLATE MAT4_APPLY CSG_UNION
# Back cover
1.0 GEN_CUBE 0.19 0.27 0.003 MAT4_SCALE MAT4_APPLY 0.0 0.0 -0.017 MAT4_TRANSLATE MAT4_APPLY CSG_UNION
# Spine
1.0 GEN_CUBE 0.003 0.27 0.036 MAT4_SCALE MAT4_APPLY -0.095 0.0 0.0 MAT4_TRANSLATE MAT4_APPLY CSG_UNION
```

Create **5 book-objects** (one per knowledge domain anchored in Library):

| star_id | Title (en) | galaxy_ref | Domain Focus |
|---------|-----------|------------|-------------|
| `book_mathematics_primer` | Mathematics Primer | `Book/MathematicsPrimer` | Numbers, operations, patterns |
| `book_language_foundations` | Language Foundations | `Book/LanguageFoundations` | Characters, words, grammar |
| `book_physics_handbook` | Physics Handbook | `Book/PhysicsHandbook` | Mechanics, E&M, thermo |
| `book_biology_atlas` | Biology Atlas | `Book/BiologyAtlas` | Cells, organisms, ecology |
| `book_tool_manual` | Tool Manual | `Book/ToolManual` | Construction, tools, methods |

Each book has slightly different dimensions (MAT4_SCALE variations) so they're visually distinct on the shelf. The `visual_rpn` is parameterized by width/height/depth.

### B2. Add books to Library room's `component_refs`

**File:** `knowledge3d/knowledgeverse/house_rooms.py`

Update the Library room star's `component_refs` to include the 5 book star_ids:

```python
component_refs=["furniture_bookshelf", "furniture_desk", "furniture_chair",
                "book_mathematics_primer", "book_language_foundations",
                "book_physics_handbook", "book_biology_atlas", "book_tool_manual"],
```

### B3. Register books in `house_builder.py`

**File:** `knowledge3d/knowledgeverse/house_builder.py`

Import `HOUSE_BOOKS` from `house_books.py`. Store them in the House Galaxy alongside rooms and furniture. Execute their `visual_rpn` to generate meshes. Add to the summary dict.

---

## Track C: Content Galaxies (2 initial samples)

### C1. Create `book_content_mathematics.py`

**File:** `knowledge3d/knowledgeverse/book_content_mathematics.py`

This module defines the **contents** of the Mathematics Primer book — a list of MeaningCentricStar entries that form the `Book/MathematicsPrimer` Galaxy when loaded.

These are NOT new concepts — they are **references** (symlinks) to existing Math Galaxy, Number Galaxy, and Grammar Galaxy entries, composed into a book-like reading order:

```python
MATHEMATICS_PRIMER_ENTRIES: list[MeaningCentricStar] = [
    # Chapter 1: Numbers (references to existing Number Galaxy entries)
    MeaningCentricStar(
        star_id="mathbook_ch1_intro",
        meaning_class="chapter",
        meaning_rpn="CHAPTER NUMBERS INTRODUCTION SEQUENCE",
        domain="Book/MathematicsPrimer",
        behavior_rpn="OPEN NAVIGATE SECTION",
        surface_forms=...,  # "Chapter 1: Numbers" / "Capítulo 1: Números" / "第1章：数"
        taxonomy_refs=["num_0", "num_1", "num_2", ...],  # existing Number Galaxy entries
        component_refs=["mathbook_sec1_counting", "mathbook_sec1_operations"],
        confidence=1,
        polarity=1,
    ),
    # Section 1.1: Counting
    MeaningCentricStar(
        star_id="mathbook_sec1_counting",
        meaning_class="section",
        meaning_rpn="SECTION COUNTING ORDINAL CARDINAL",
        domain="Book/MathematicsPrimer",
        taxonomy_refs=["num_0", "num_1", "num_2", "num_3", "num_4",
                       "num_5", "num_6", "num_7", "num_8", "num_9"],
        ...
    ),
    # Section 1.2: Basic Operations
    MeaningCentricStar(
        star_id="mathbook_sec1_operations",
        meaning_class="section",
        meaning_rpn="SECTION ADDITION SUBTRACTION MULTIPLICATION DIVISION",
        domain="Book/MathematicsPrimer",
        taxonomy_refs=["sym_plus", "sym_minus", "sym_times", "sym_divide"],
        grammar_refs=["rule_addition_commutative", "rule_multiplication_commutative"],
        ...
    ),
]
```

**Important:** Book content stars reference EXISTING Galaxy entries via `taxonomy_refs` and `grammar_refs`. They do NOT duplicate knowledge. They organize it into a readable sequence.

Target: ~15-20 entries (3-4 chapters with 3-5 sections each). Enough to demonstrate the pattern without being exhaustive.

### C2. Create `book_content_language.py`

**File:** `knowledge3d/knowledgeverse/book_content_language.py`

Same pattern for the Language Foundations book. References existing Character Galaxy, Word Galaxy, and Grammar Galaxy entries:

- Chapter 1: Characters (refs to char entries)
- Chapter 2: Words (refs to word entries)
- Chapter 3: Grammar (refs to grammar rules)

Target: ~15-20 entries.

### C3. Register content Galaxies in bootstrap

**File:** `knowledge3d/knowledgeverse/foundational_galaxy_bootstrap.py`

Add a function to register book content as named Galaxies:

```python
def populate_book_galaxies(manager: GalaxyManager) -> None:
    from .book_content_mathematics import MATHEMATICS_PRIMER_ENTRIES
    from .book_content_language import LANGUAGE_FOUNDATIONS_ENTRIES

    with manager.bulk_disk_sync():
        for star in MATHEMATICS_PRIMER_ENTRIES:
            manager.store_meaning_star("Book/MathematicsPrimer", star)
        for star in LANGUAGE_FOUNDATIONS_ENTRIES:
            manager.store_meaning_star("Book/LanguageFoundations", star)
```

This is called during bootstrap AFTER foundational galaxies are populated (the content references existing entries).

---

## Track D: H2 Cleanup

### D1. Remove dead code in `generate_cone`

**File:** `knowledge3d/cranium/bridges/sovereign_mesh_bridge.py`

Lines 506-512 build a MeshBuffer from uninitialized host buffers (memcpy hasn't happened yet), then the result is discarded. Delete these lines. The correct mesh is built at line 517 after the memcpy.

Before (current):
```python
            triangles: list[tuple[int, int, int]] = []
            for seg in range(segments):
                triangles.append((apex, seg, seg + 1))
                nxt = (seg + 1) % segments
                triangles.append((center, base_ring + nxt, base_ring + seg))
            mesh = self._mesh_from_buffers(      # ← DELETE: uses uninitialized host buffers
                vertex_count=vertex_count,         # ← DELETE
                vertices=host_vertices,            # ← DELETE
                normals=host_normals,              # ← DELETE
                uvs=host_uvs,                      # ← DELETE
                triangles=triangles,               # ← DELETE
                metadata={"primitive": "cone", "backend": "gpu"},  # ← DELETE
            )                                      # ← DELETE
            loader.memcpy_dtoh(...)
```

After (fixed):
```python
            triangles: list[tuple[int, int, int]] = []
            for seg in range(segments):
                triangles.append((apex, seg, seg + 1))
                nxt = (seg + 1) % segments
                triangles.append((center, base_ring + nxt, base_ring + seg))
            loader.memcpy_dtoh(...)
```

### D2. Extract shared `_char_refs` / `_surface_forms` helpers

**File:** Create `knowledge3d/knowledgeverse/_house_utils.py`

Both `house_rooms.py` and `house_furniture.py` define identical `_char_refs()` and `_surface_forms()` functions. Extract to a shared module:

```python
"""Shared helpers for House template construction."""
from .meaning_star import SurfaceForm

def char_refs(text: str, language: str) -> list[str]: ...
def surface_forms(en: str, pt: str, ja: str) -> dict[str, SurfaceForm]: ...
```

Update `house_rooms.py`, `house_furniture.py`, and the new `house_books.py` to import from `_house_utils.py`.

---

## Tests

### `tests/test_house_books.py`

```python
def test_book_objects_have_galaxy_refs():
    """Every book-object must reference a loadable Galaxy."""
    for book in HOUSE_BOOKS:
        assert book.galaxy_ref, f"{book.star_id} missing galaxy_ref"
        assert book.visual_rpn, f"{book.star_id} missing visual_rpn"
        assert book.meaning_class == "book"

def test_book_visual_rpn_produces_mesh():
    """Book 3D shapes must be constructable."""
    bridge = MeshBridge()
    for book in HOUSE_BOOKS:
        result = bridge.execute_rpn_program(book.visual_rpn)
        assert result.mesh.vertices
        assert result.mesh.triangles

def test_book_galaxy_ref_loads_content():
    """Opening a book loads its content Galaxy."""
    manager = GalaxyManager(storage_root=tmp_path / "galaxies")
    populate_book_galaxies(manager)
    build_house(manager)  # registers book-objects
    book = manager.load_meaning_star("House", "book_mathematics_primer")
    galaxy = manager.load_galaxy_on_demand(book)
    assert galaxy is not None
    assert len(galaxy.entries) > 0

def test_book_content_references_existing_entries():
    """Book content uses taxonomy_refs to existing Galaxy entries, not duplicates."""
    for star in MATHEMATICS_PRIMER_ENTRIES:
        assert star.domain.startswith("Book/")
        # Content stars should reference, not define, foundational knowledge
        assert star.meaning_class in ("chapter", "section", "page")

def test_galaxy_ref_in_star_serialization():
    """galaxy_ref roundtrips through to_dict/from_dict."""
    star = MeaningCentricStar(
        meaning_class="book",
        meaning_rpn="BOOK TEST",
        domain="Test",
        galaxy_ref="Book/Test",
    )
    d = star.to_dict()
    assert d["galaxy_ref"] == "Book/Test"
    restored = MeaningCentricStar.from_dict(d)
    assert restored.galaxy_ref == "Book/Test"

def test_star_id_changes_with_galaxy_ref():
    """Different galaxy_ref = different star identity."""
    base = dict(meaning_class="book", meaning_rpn="BOOK", domain="Test")
    s1 = MeaningCentricStar(**base, galaxy_ref="Book/A")
    s2 = MeaningCentricStar(**base, galaxy_ref="Book/B")
    assert s1.star_id != s2.star_id
```

### Update existing tests

- `tests/test_meaning_star.py`: Add `galaxy_ref` roundtrip test
- `tests/test_house_builder.py`: Verify books are stored and their meshes generated
- Ensure ALL existing tests still pass (benchmark non-regression)

---

## Success Criteria

1. `MeaningCentricStar.galaxy_ref` field exists, serializes, and affects star_id
2. `GalaxyManager.load_galaxy_on_demand()` loads the referenced Galaxy
3. 5 book-objects exist with constructable `visual_rpn` (covers + spine + pages)
4. 2 content Galaxies exist (`Book/MathematicsPrimer`, `Book/LanguageFoundations`) with ~15-20 entries each referencing existing Galaxy entries
5. H2 dead code removed, helpers deduplicated
6. All existing tests pass: `pytest -q tests/` green
7. GPU math benchmark non-regression: Math 20/20, GSM8K 10/10, ARC 10/10

---

## Files Changed/Created

| File | Action |
|------|--------|
| `knowledge3d/knowledgeverse/meaning_star.py` | Add `galaxy_ref` field |
| `knowledge3d/knowledgeverse/galaxy_manager.py` | Add `load_galaxy_on_demand()` |
| `knowledge3d/knowledgeverse/house_books.py` | **NEW** — 5 book-objects |
| `knowledge3d/knowledgeverse/book_content_mathematics.py` | **NEW** — Math Primer Galaxy entries |
| `knowledge3d/knowledgeverse/book_content_language.py` | **NEW** — Language Foundations Galaxy entries |
| `knowledge3d/knowledgeverse/_house_utils.py` | **NEW** — shared char_refs/surface_forms |
| `knowledge3d/knowledgeverse/house_rooms.py` | Update Library component_refs, use _house_utils |
| `knowledge3d/knowledgeverse/house_furniture.py` | Use _house_utils |
| `knowledge3d/knowledgeverse/house_builder.py` | Register books, generate book meshes |
| `knowledge3d/knowledgeverse/foundational_galaxy_bootstrap.py` | Add `populate_book_galaxies()` |
| `knowledge3d/knowledgeverse/__init__.py` | Export HOUSE_BOOKS, populate_book_galaxies |
| `knowledge3d/cranium/bridges/sovereign_mesh_bridge.py` | Remove dead code in generate_cone |
| `tests/test_house_books.py` | **NEW** — book + content Galaxy tests |
| `tests/test_meaning_star.py` | Add galaxy_ref roundtrip test |
| `tests/test_house_builder.py` | Verify books in build_house |
