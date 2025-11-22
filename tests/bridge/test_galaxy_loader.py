from knowledge3d.bridge.galaxy_loader import GalaxyUniverseLoader
from knowledge3d.bridge.memory_tablet import MemoryTablet


def test_defaults_include_phrase_and_user():
    loader = GalaxyUniverseLoader(enable_sublex=False)
    assert "phrase_meaning" in loader.loaded
    assert "user_phrase" in loader.loaded


def test_sublex_optional():
    loader = GalaxyUniverseLoader(enable_sublex=True)
    assert "syllables_latin" in loader.loaded
    assert "morphemes_latin" in loader.loaded


def test_memory_tablet_loads_context():
    tablet = MemoryTablet(enable_sublex=True)
    loaded = tablet.list_loaded()
    assert "word_meaning" in loaded and "phrase_meaning" in loaded
