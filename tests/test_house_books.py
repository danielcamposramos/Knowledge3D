from __future__ import annotations

from pathlib import Path

from knowledge3d.cranium.bridges.mesh_bridge import MeshBridge
from knowledge3d.knowledgeverse.book_content_language import LANGUAGE_FOUNDATIONS_ENTRIES
from knowledge3d.knowledgeverse.book_content_mathematics import MATHEMATICS_PRIMER_ENTRIES
from knowledge3d.knowledgeverse.foundational_galaxy_bootstrap import populate_book_galaxies
from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager
from knowledge3d.knowledgeverse.house_books import HOUSE_BOOKS
from knowledge3d.knowledgeverse.house_builder import build_house
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


def test_book_objects_have_galaxy_refs() -> None:
    for book in HOUSE_BOOKS:
        assert book.galaxy_ref
        assert book.visual_rpn
        assert book.meaning_class == "book"


def test_book_visual_rpn_produces_mesh() -> None:
    bridge = MeshBridge()
    for book in HOUSE_BOOKS:
        result = bridge.execute_rpn_program(book.visual_rpn or "")
        assert result.mesh.vertices
        assert result.mesh.triangles
        assert result.mesh.normals


def test_book_galaxy_ref_loads_content(tmp_path: Path) -> None:
    manager = GalaxyManager(storage_root=tmp_path / "galaxies")
    populate_book_galaxies(manager)
    build_house(manager)
    book = manager.load_meaning_star("House", "book_mathematics_primer")
    assert book is not None
    galaxy = manager.load_galaxy_on_demand(book)
    assert galaxy is not None
    assert len(galaxy.entries) > 0


def test_book_content_references_existing_entries() -> None:
    for star in MATHEMATICS_PRIMER_ENTRIES + LANGUAGE_FOUNDATIONS_ENTRIES:
        assert star.domain.startswith("Book/")
        assert star.meaning_class in {"chapter", "section", "page"}


def test_galaxy_ref_in_star_serialization() -> None:
    star = MeaningCentricStar(
        meaning_class="book",
        meaning_rpn="BOOK TEST",
        domain="Test",
        galaxy_ref="Book/Test",
    )
    payload = star.to_dict()
    restored = MeaningCentricStar.from_dict(payload)
    entry = star.to_galaxy_entry(galaxy_name="House")
    recovered = MeaningCentricStar.from_galaxy_entry(entry)

    assert payload["galaxy_ref"] == "Book/Test"
    assert restored.galaxy_ref == "Book/Test"
    assert recovered.galaxy_ref == "Book/Test"


def test_star_id_changes_with_galaxy_ref() -> None:
    base = {"meaning_class": "book", "meaning_rpn": "BOOK", "domain": "Test"}
    left = MeaningCentricStar(**base, galaxy_ref="Book/A")
    right = MeaningCentricStar(**base, galaxy_ref="Book/B")
    assert left.star_id != right.star_id
