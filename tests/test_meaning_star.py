from __future__ import annotations

from knowledge3d.knowledgeverse.foundational_operations_bootstrap import populate_foundational_operations
from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar, SurfaceForm
from knowledge3d.knowledgeverse.seed_stars import SEED_STARS
from knowledge3d.knowledgeverse.semantic_gravity import gravity_tick, meaning_mass, ternary_semantic_force


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    dx = float(left[0]) - float(right[0])
    dy = float(left[1]) - float(right[1])
    dz = float(left[2]) - float(right[2])
    return ((dx * dx) + (dy * dy) + (dz * dz)) ** 0.5


def test_star_creation_and_id_determinism() -> None:
    left = MeaningCentricStar(
        meaning_class="concept",
        meaning_rpn="CAT FELINE COMPANION",
        domain="Library/Biology/Mammalia",
    )
    right = MeaningCentricStar(
        meaning_class="concept",
        meaning_rpn="CAT FELINE COMPANION",
        domain="Library/Biology/Mammalia",
    )

    assert left.star_id
    assert left.star_id == right.star_id


def test_surface_forms_are_references(tmp_path) -> None:
    manager = GalaxyManager(storage_root=tmp_path / "galaxies")
    populate_foundational_operations(manager)

    number_entries = manager.get_galaxy("Number").entries
    word_entries = manager.get_galaxy("Word").entries
    word_ids = {str(entry.get("id", "")) for entry in word_entries if isinstance(entry, dict)}
    num_five = next(entry for entry in number_entries if entry.get("id") == "num_5")
    star = MeaningCentricStar.from_galaxy_entry(num_five)

    assert star.surface_forms["en"].word_ref in word_ids
    assert star.surface_forms["pt"].word_ref in word_ids
    assert star.surface_forms["ja"].word_ref in word_ids
    assert star.surface_forms["en"].word_ref != "five"
    assert star.surface_forms["pt"].word_ref != "cinco"
    assert star.surface_forms["ja"].word_ref != "五"


def test_meaning_mass_computation() -> None:
    minimal = MeaningCentricStar(
        meaning_class="concept",
        meaning_rpn="INTEGER",
        domain="Library/Mathematics",
    )
    enriched = MeaningCentricStar(
        meaning_class="concept",
        meaning_rpn="INTEGER COUNTABLE SYMBOL",
        domain="Library/Mathematics",
        surface_forms={
            "en": SurfaceForm(word_ref="word_integer", char_refs=["char_i"]),
            "pt": SurfaceForm(word_ref="word_pt_inteiro", char_refs=["char_i"]),
        },
        visual_rpn="GLYPH DIGIT STACK",
        visual_refs=["draw_digit"],
        audio_rpn="AUDIO INTEGER",
        audio_refs=["audio_integer"],
        pronunciations={"en": "audio_integer_en"},
        behavior_rpn="COUNT SUCCESSOR",
        reality_refs=["reality_counting"],
        grammar_refs=["grammar_numeric_literal"],
        meta_refs=["meta_stable"],
        component_refs=["num_1"],
        composite_of=["concept_mathematics"],
    )

    assert meaning_mass(enriched) > meaning_mass(minimal)


def test_ternary_force_attract_repel_neutral() -> None:
    cat = MeaningCentricStar(
        star_id="concept_cat",
        meaning_class="concept",
        meaning_rpn="CAT FELINE",
        domain="Library/Biology",
        taxonomy_refs=["concept_mammal"],
        embedding_128=(1.0, 0.0),
    )
    mammal = MeaningCentricStar(
        star_id="concept_mammal",
        meaning_class="concept",
        meaning_rpn="MAMMAL ANIMAL",
        domain="Library/Biology",
        embedding_128=(0.9, 0.1),
    )
    alive = MeaningCentricStar(
        star_id="concept_alive",
        meaning_class="property",
        meaning_rpn="ALIVE LIFE",
        domain="Library/Biology",
        polarity=1,
        embedding_128=(1.0, 0.0),
    )
    dead = MeaningCentricStar(
        star_id="concept_dead",
        meaning_class="property",
        meaning_rpn="DEAD LIFE_END",
        domain="Library/Biology",
        polarity=-1,
        embedding_128=(-1.0, 0.0),
    )
    integer = MeaningCentricStar(
        star_id="concept_integer",
        meaning_class="concept",
        meaning_rpn="INTEGER COUNTABLE",
        domain="Library/Mathematics",
        embedding_128=(0.0, 1.0),
    )

    assert ternary_semantic_force(cat, mammal) == 1
    assert ternary_semantic_force(alive, dead) == -1
    assert ternary_semantic_force(cat, integer) == 0


def test_gravity_tick_clusters_related() -> None:
    cat = MeaningCentricStar(
        star_id="concept_cat",
        meaning_class="concept",
        meaning_rpn="CAT FELINE",
        domain="Library/Biology",
        taxonomy_refs=["concept_mammal"],
        embedding_128=(1.0, 0.0),
        house_position=(0.0, 0.0, 0.0),
    )
    mammal = MeaningCentricStar(
        star_id="concept_mammal",
        meaning_class="concept",
        meaning_rpn="MAMMAL ANIMAL",
        domain="Library/Biology",
        embedding_128=(0.9, 0.1),
        house_position=(15.0, 0.0, 0.0),
    )
    integer = MeaningCentricStar(
        star_id="concept_integer",
        meaning_class="concept",
        meaning_rpn="INTEGER COUNTABLE",
        domain="Library/Mathematics",
        embedding_128=(0.0, 1.0),
        house_position=(0.0, 15.0, 0.0),
    )

    positions = None
    velocities = None
    for _ in range(80):
        positions, velocities = gravity_tick(
            [cat, mammal, integer],
            dt=0.02,
            damping=0.92,
            positions=positions,
            velocities=velocities,
        )

    assert positions is not None
    assert _distance(positions[cat.star_id], positions[mammal.star_id]) < _distance(
        positions[cat.star_id],
        positions[integer.star_id],
    )


def test_seed_stars_have_required_fields() -> None:
    assert len(SEED_STARS) == 10
    for star in SEED_STARS:
        assert star.star_id
        assert star.meaning_rpn
        assert star.visual_rpn
        assert {"en", "pt", "ja"}.issubset(set(star.surface_forms.keys()))
        assert meaning_mass(star) >= 7.0


def test_existing_galaxy_compatibility(tmp_path) -> None:
    manager = GalaxyManager(storage_root=tmp_path / "galaxies")
    star = MeaningCentricStar(
        meaning_class="concept",
        meaning_rpn="MAMMAL ANIMAL WARM_BLOODED",
        domain="Library/Biology",
        surface_forms={"en": SurfaceForm(word_ref="word_mammal", char_refs=["char_m"])},
    )

    status = manager.store_meaning_star("Meaning", star)
    loaded = manager.load_meaning_star("Meaning", star.star_id)

    assert status == "inserted"
    assert loaded is not None
    assert loaded.star_id == star.star_id
    assert loaded.meaning_rpn == star.meaning_rpn


def test_galaxy_ref_roundtrip_changes_identity() -> None:
    left = MeaningCentricStar(
        meaning_class="book",
        meaning_rpn="BOOK TEST",
        domain="House/Library/Books",
        galaxy_ref="Book/A",
    )
    right = MeaningCentricStar(
        meaning_class="book",
        meaning_rpn="BOOK TEST",
        domain="House/Library/Books",
        galaxy_ref="Book/B",
    )

    payload = left.to_dict()
    restored = MeaningCentricStar.from_dict(payload)
    entry = left.to_galaxy_entry(galaxy_name="House")
    from_entry = MeaningCentricStar.from_galaxy_entry(entry)

    assert payload["galaxy_ref"] == "Book/A"
    assert restored.galaxy_ref == "Book/A"
    assert from_entry.galaxy_ref == "Book/A"
    assert left.star_id != right.star_id
