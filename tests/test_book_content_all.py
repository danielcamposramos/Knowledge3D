from __future__ import annotations

from pathlib import Path

from knowledge3d.knowledgeverse.book_content_biology import BIOLOGY_ATLAS_ENTRIES
from knowledge3d.knowledgeverse.book_content_physics import PHYSICS_HANDBOOK_ENTRIES
from knowledge3d.knowledgeverse.book_content_tools import TOOL_MANUAL_ENTRIES
from knowledge3d.knowledgeverse.foundational_galaxy_bootstrap import populate_book_galaxies
from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager
from knowledge3d.knowledgeverse.house_books import HOUSE_BOOKS
from knowledge3d.knowledgeverse.house_builder import build_house


def test_physics_handbook_entries_reference_reality_galaxy() -> None:
    for star in PHYSICS_HANDBOOK_ENTRIES:
        assert star.domain == "Book/PhysicsHandbook"
        assert star.meaning_class in {"chapter", "section", "page"}
        refs = list(star.taxonomy_refs) + list(star.grammar_refs)
        assert any(ref.startswith("reality_") or ref.startswith("concept_") for ref in refs)


def test_biology_atlas_entries_reference_biology() -> None:
    for star in BIOLOGY_ATLAS_ENTRIES:
        assert star.domain == "Book/BiologyAtlas"
        assert star.meaning_class in {"chapter", "section", "page"}


def test_tool_manual_entries_reference_tools() -> None:
    for star in TOOL_MANUAL_ENTRIES:
        assert star.domain == "Book/ToolManual"
        assert star.meaning_class in {"chapter", "section", "page"}
        assert any(ref.startswith("tool_") or ref.startswith("concept_") for ref in star.taxonomy_refs)


def test_all_book_galaxies_load_on_demand(tmp_path: Path) -> None:
    manager = GalaxyManager(storage_root=tmp_path / "galaxies")
    populate_book_galaxies(manager)
    build_house(manager)
    for book in HOUSE_BOOKS:
        galaxy = manager.load_galaxy_on_demand(book)
        assert galaxy is not None, f"{book.star_id} galaxy not loaded"
        assert len(galaxy.entries) > 0, f"{book.star_id} galaxy is empty"


def test_cross_domain_references_in_biology() -> None:
    ecology_sections = [star for star in BIOLOGY_ATLAS_ENTRIES if "energy" in star.star_id.lower()]
    assert any(
        "reality_thermo" in ref
        for star in ecology_sections
        for ref in star.taxonomy_refs
    )
