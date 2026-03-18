"""Unicode writing-system registry for universal foundational knowledge."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WritingSystem:
    key: str
    name: str
    unicode_start: int
    unicode_end: int
    approx_chars: int
    region: str
    direction: str
    era: str
    priority: int
    description: str

    @property
    def unicode_range(self) -> tuple[int, int]:
        return (self.unicode_start, self.unicode_end)

    def contains(self, codepoint: int) -> bool:
        return self.unicode_start <= int(codepoint) <= self.unicode_end


WRITING_SYSTEMS: dict[str, WritingSystem] = {
    "latin_extended": WritingSystem("latin_extended", "Latin Extended", 0x0080, 0x024F, 336, "Global", "LTR", "living", 1, "Extended Latin letters used across many modern languages."),
    "cyrillic": WritingSystem("cyrillic", "Cyrillic", 0x0400, 0x04FF, 256, "Eastern Europe", "LTR", "living", 1, "Cyrillic scripts for Slavic and related languages."),
    "greek": WritingSystem("greek", "Greek", 0x0370, 0x03FF, 144, "Greece", "LTR", "living", 1, "Greek script used for language, mathematics, and science."),
    "arabic": WritingSystem("arabic", "Arabic", 0x0600, 0x06FF, 256, "Middle East", "RTL", "living", 1, "Arabic script family."),
    "devanagari": WritingSystem("devanagari", "Devanagari", 0x0900, 0x097F, 128, "South Asia", "LTR", "living", 1, "Indic abugida for Hindi, Sanskrit, Marathi, and more."),
    "cjk_unified": WritingSystem("cjk_unified", "CJK Unified Ideographs", 0x4E00, 0x9FFF, 20992, "East Asia", "LTR", "living", 1, "Unified Han ideographs used in Chinese, Japanese, and Korean contexts."),
    "hangul": WritingSystem("hangul", "Hangul Syllables", 0xAC00, 0xD7AF, 11172, "Korea", "LTR", "living", 1, "Modern Korean Hangul syllables."),
    "hiragana_katakana": WritingSystem("hiragana_katakana", "Hiragana and Katakana", 0x3040, 0x30FF, 192, "Japan", "LTR", "living", 1, "Japanese syllabaries."),
    "hebrew": WritingSystem("hebrew", "Hebrew", 0x0590, 0x05FF, 112, "Israel", "RTL", "living", 1, "Hebrew script."),
    "thai": WritingSystem("thai", "Thai", 0x0E00, 0x0E7F, 128, "Thailand", "LTR", "living", 1, "Thai abugida."),
    "bengali": WritingSystem("bengali", "Bengali", 0x0980, 0x09FF, 128, "Bangladesh/India", "LTR", "living", 1, "Bengali-Assamese script."),
    "tamil": WritingSystem("tamil", "Tamil", 0x0B80, 0x0BFF, 128, "South India", "LTR", "living", 1, "Tamil script."),
    "egyptian_hieroglyphs": WritingSystem("egyptian_hieroglyphs", "Egyptian Hieroglyphs", 0x13000, 0x1342F, 1072, "Ancient Egypt", "LTR", "historical", 2, "Ancient Egyptian monumental writing."),
    "cuneiform": WritingSystem("cuneiform", "Cuneiform", 0x12000, 0x123FF, 1024, "Ancient Mesopotamia", "LTR", "historical", 2, "Mesopotamian wedge-based writing."),
    "linear_b": WritingSystem("linear_b", "Linear B", 0x10000, 0x1007F, 128, "Aegean Bronze Age", "LTR", "historical", 2, "Mycenaean Greek syllabary."),
    "phoenician": WritingSystem("phoenician", "Phoenician", 0x10900, 0x1091F, 32, "Levant", "RTL", "historical", 2, "Early consonantal alphabet."),
    "old_persian": WritingSystem("old_persian", "Old Persian", 0x103A0, 0x103DF, 64, "Achaemenid Persia", "LTR", "historical", 2, "Old Persian cuneiform."),
    "runic": WritingSystem("runic", "Runic", 0x16A0, 0x16FF, 96, "Northern Europe", "LTR", "historical", 2, "Runic alphabets."),
    "gothic": WritingSystem("gothic", "Gothic", 0x10330, 0x1034F, 32, "Late Antiquity", "LTR", "historical", 2, "Gothic alphabet."),
    "coptic": WritingSystem("coptic", "Coptic", 0x2C80, 0x2CFF, 128, "Egypt", "LTR", "historical", 2, "Coptic script."),
    "brahmi": WritingSystem("brahmi", "Brahmi", 0x11000, 0x1107F, 128, "Ancient India", "LTR", "historical", 2, "Ancestor to many Indic scripts."),
    "braille": WritingSystem("braille", "Braille Patterns", 0x2800, 0x28FF, 256, "Global", "LTR", "specialized", 3, "Braille accessibility patterns."),
    "musical_symbols": WritingSystem("musical_symbols", "Musical Symbols", 0x1D100, 0x1D1FF, 256, "Global", "LTR", "specialized", 3, "Music notation symbols."),
    "alchemical_symbols": WritingSystem("alchemical_symbols", "Alchemical Symbols", 0x1F700, 0x1F77F, 128, "Historical Science", "LTR", "specialized", 3, "Historical chemistry/alchemy notation."),
    "emoji": WritingSystem("emoji", "Emoji", 0x1F600, 0x1F64F, 80, "Global", "LTR", "specialized", 3, "Modern pictographic symbols."),
}


def iter_writing_systems(priority: int | None = None) -> list[WritingSystem]:
    items = sorted(WRITING_SYSTEMS.values(), key=lambda row: (row.priority, row.unicode_start, row.name))
    if priority is None:
        return items
    return [item for item in items if item.priority == int(priority)]


def get_writing_system(key: str) -> WritingSystem:
    lookup = str(key or "").strip().lower()
    if lookup not in WRITING_SYSTEMS:
        raise KeyError(f"Unknown writing system: {key}")
    return WRITING_SYSTEMS[lookup]


__all__ = ["WRITING_SYSTEMS", "WritingSystem", "get_writing_system", "iter_writing_systems"]
