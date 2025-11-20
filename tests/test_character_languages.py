import pytest

from knowledge3d.cranium.specialists.character_languages import (
    get_character_languages,
    get_character_stats,
)


class TestCharacterLanguages:
    def test_basic_latin(self):
        langs = get_character_languages('A')
        assert 'en' in langs
        assert 'pt' in langs
        assert len(langs) == 33

    def test_extended_latin(self):
        langs = get_character_languages('ç')
        assert set(langs) == {'pt', 'fr', 'ca', 'tr', 'sq'}

    def test_basic_cyrillic(self):
        langs = get_character_languages('Ж')
        assert 'ru' in langs and 'bg' in langs
        assert len(langs) == len(set(langs))
        assert len(langs) >= 25

    def test_extended_cyrillic(self):
        assert set(get_character_languages('ё')) == {'ru', 'be'}
        assert get_character_languages('є') == ['uk']

    def test_universal_symbols(self):
        assert get_character_languages('+') == ['universal']
        assert get_character_languages('3') == ['universal']
        assert get_character_languages('!') == ['universal']

    def test_additional_scripts(self):
        assert 'el' in get_character_languages('Ω')
        assert 'he' in get_character_languages('ש')
        assert 'ar' in get_character_languages('م')
        assert 'hi' in get_character_languages('क')
        assert 'bn' in get_character_languages('ড')
        assert 'pa' in get_character_languages('ਘ')
        assert 'ta' in get_character_languages('ழ')
        assert 'ja' in get_character_languages('あ')
        assert 'ko' in get_character_languages('한')
        assert 'zh' in get_character_languages('漢')

    def test_stats(self):
        stats = get_character_stats()
        assert stats['total_chars'] > 450
        assert stats['cyrillic_chars'] > 100
        assert stats['avg_languages_per_char'] > 2
