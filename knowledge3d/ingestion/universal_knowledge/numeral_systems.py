"""Numeral-system registry and round-trip helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NumeralSystem:
    key: str
    name: str
    base: int | None
    positional: bool
    digits: tuple[str, ...] = ()
    symbols: dict[str, int] | None = None
    rules_rpn: str = ""
    description: str = ""
    surface_forms: dict[str, str] | None = None
    supports_roundtrip: bool = False


NUMERAL_SYSTEMS: dict[str, NumeralSystem] = {
    "arabic_western": NumeralSystem("arabic_western", "Western Arabic", 10, True, tuple("0123456789"), rules_rpn="POSITIONAL_BASE10_PARSE", description="Global Hindu-Arabic digits.", surface_forms={"en": "Western Arabic numerals"}, supports_roundtrip=True),
    "arabic_eastern": NumeralSystem("arabic_eastern", "Eastern Arabic", 10, True, tuple("٠١٢٣٤٥٦٧٨٩"), rules_rpn="POSITIONAL_BASE10_PARSE", description="Arabic-Indic digits.", surface_forms={"ar": "أرقام عربية مشرقية", "en": "Eastern Arabic numerals"}, supports_roundtrip=True),
    "roman": NumeralSystem("roman", "Roman Numerals", None, False, symbols={"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}, rules_rpn="ROMAN_PARSE SUBTRACTIVE_RULE", description="Additive/subtractive Roman numerals.", surface_forms={"en": "Roman numerals"}, supports_roundtrip=True),
    "chinese_traditional": NumeralSystem("chinese_traditional", "Chinese Traditional", 10, False, symbols={"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100, "千": 1000, "萬": 10000}, rules_rpn="CHINESE_MULTIPLICATIVE_PARSE", description="Traditional Chinese numerals.", surface_forms={"zh": "中文数字", "en": "Chinese traditional numerals"}),
    "chinese_financial": NumeralSystem("chinese_financial", "Chinese Financial", 10, False, symbols={"零": 0, "壹": 1, "貳": 2, "參": 3, "肆": 4, "伍": 5, "陸": 6, "柒": 7, "捌": 8, "玖": 9, "拾": 10}, rules_rpn="CHINESE_FINANCIAL_PARSE", description="Anti-fraud Chinese financial numerals.", surface_forms={"zh": "中文大写数字", "en": "Chinese financial numerals"}),
    "devanagari": NumeralSystem("devanagari", "Devanagari", 10, True, tuple("०१२३४५६७८९"), rules_rpn="POSITIONAL_BASE10_PARSE", description="Indic Devanagari digits.", surface_forms={"hi": "देवनागरी अंक", "en": "Devanagari numerals"}, supports_roundtrip=True),
    "thai": NumeralSystem("thai", "Thai", 10, True, tuple("๐๑๒๓๔๕๖๗๘๙"), rules_rpn="POSITIONAL_BASE10_PARSE", description="Thai digits.", surface_forms={"th": "เลขไทย", "en": "Thai numerals"}, supports_roundtrip=True),
    "bengali": NumeralSystem("bengali", "Bengali", 10, True, tuple("০১২৩৪৫৬৭৮৯"), rules_rpn="POSITIONAL_BASE10_PARSE", description="Bengali digits.", surface_forms={"bn": "বাংলা সংখ্যা", "en": "Bengali numerals"}, supports_roundtrip=True),
    "mayan": NumeralSystem("mayan", "Maya Vigesimal", 20, True, tuple(str(value) for value in range(20)), rules_rpn="POSITIONAL_BASE20_PARSE", description="Vigesimal positional system.", surface_forms={"en": "Mayan numerals"}, supports_roundtrip=True),
    "babylonian": NumeralSystem("babylonian", "Babylonian Sexagesimal", 60, True, tuple(str(value) for value in range(60)), rules_rpn="POSITIONAL_BASE60_PARSE", description="Base-60 place-value notation.", surface_forms={"en": "Babylonian numerals"}, supports_roundtrip=True),
    "egyptian": NumeralSystem("egyptian", "Egyptian Hieroglyphic", 10, False, symbols={"𓏺": 1, "𓎆": 10, "𓍢": 100, "𓆼": 1000, "𓂭": 10000, "𓆐": 100000, "𓁨": 1000000}, rules_rpn="EGYPTIAN_ADDITIVE_PARSE", description="Ancient Egyptian additive numerals.", surface_forms={"en": "Egyptian numerals"}),
    "greek_alphabetic": NumeralSystem("greek_alphabetic", "Greek Alphabetic", None, False, description="Ionic alphabetic numerals.", surface_forms={"en": "Greek alphabetic numerals"}),
    "hebrew_gematria": NumeralSystem("hebrew_gematria", "Hebrew Gematria", None, False, description="Hebrew letter-number system.", surface_forms={"en": "Hebrew gematria"}),
    "tally": NumeralSystem("tally", "Tally Marks", 5, False, rules_rpn="TALLY_GROUP_PARSE", description="Grouped tally marks.", surface_forms={"en": "Tally marks"}, supports_roundtrip=True),
    "finger_counting": NumeralSystem("finger_counting", "Finger Counting", 10, False, description="Cultural hand-counting systems.", surface_forms={"en": "Finger counting"}),
    "binary": NumeralSystem("binary", "Binary", 2, True, tuple("01"), rules_rpn="POSITIONAL_BASE2_PARSE", description="Base-2 numerals.", surface_forms={"en": "Binary"}, supports_roundtrip=True),
    "hexadecimal": NumeralSystem("hexadecimal", "Hexadecimal", 16, True, tuple("0123456789ABCDEF"), rules_rpn="POSITIONAL_BASE16_PARSE", description="Base-16 numerals.", surface_forms={"en": "Hexadecimal"}, supports_roundtrip=True),
    "octal": NumeralSystem("octal", "Octal", 8, True, tuple("01234567"), rules_rpn="POSITIONAL_BASE8_PARSE", description="Base-8 numerals.", surface_forms={"en": "Octal"}, supports_roundtrip=True),
}


_ROMAN_VALUES = (("M", 1000), ("CM", 900), ("D", 500), ("CD", 400), ("C", 100), ("XC", 90), ("L", 50), ("XL", 40), ("X", 10), ("IX", 9), ("V", 5), ("IV", 4), ("I", 1))


def iter_numeral_systems() -> list[NumeralSystem]:
    return [NUMERAL_SYSTEMS[key] for key in sorted(NUMERAL_SYSTEMS.keys())]


def _encode_positional(value: int, base: int, digits: tuple[str, ...], *, separator: str = "") -> str:
    if value == 0:
        return digits[0]
    out: list[str] = []
    current = int(value)
    while current > 0:
        current, remainder = divmod(current, base)
        out.append(digits[remainder])
    encoded = list(reversed(out))
    return separator.join(encoded)


def _decode_positional(text: str, base: int, digits: tuple[str, ...], *, separator: str = "") -> int:
    if separator:
        parts = [part for part in str(text).split(separator) if part != ""]
    else:
        parts = list(str(text))
    lookup = {glyph: index for index, glyph in enumerate(digits)}
    value = 0
    for token in parts:
        value = (value * base) + lookup[token]
    return value


def _encode_roman(value: int) -> str:
    current = int(value)
    if current <= 0:
        raise ValueError("Roman numerals require positive integers")
    out: list[str] = []
    for symbol, number in _ROMAN_VALUES:
        while current >= number:
            out.append(symbol)
            current -= number
    return "".join(out)


def _decode_roman(text: str) -> int:
    current = str(text).upper().strip()
    total = 0
    index = 0
    while index < len(current):
        for symbol, value in _ROMAN_VALUES:
            if current.startswith(symbol, index):
                total += value
                index += len(symbol)
                break
        else:  # pragma: no cover - malformed text guard
            raise ValueError(f"Invalid Roman numeral: {text}")
    return total


def _encode_tally(value: int) -> str:
    if value < 0:
        raise ValueError("Tally marks require non-negative integers")
    groups, remainder = divmod(int(value), 5)
    parts = ["||||/" for _ in range(groups)]
    if remainder:
        parts.append("|" * remainder)
    return " ".join(parts) if parts else ""


def _decode_tally(text: str) -> int:
    total = 0
    for token in str(text).split():
        if token == "||||/":
            total += 5
        else:
            total += token.count("|")
    return total


def encode_number(system_name: str, value: int) -> str:
    system = NUMERAL_SYSTEMS[str(system_name).strip().lower()]
    numeric_value = int(value)
    if system.key == "roman":
        return _encode_roman(numeric_value)
    if system.key == "tally":
        return _encode_tally(numeric_value)
    if system.key == "mayan":
        return _encode_positional(numeric_value, 20, system.digits, separator=".")
    if system.key == "babylonian":
        return _encode_positional(numeric_value, 60, system.digits, separator=":")
    if system.positional and system.base and system.digits:
        return _encode_positional(numeric_value, system.base, system.digits)
    raise ValueError(f"System does not support encoding: {system_name}")


def decode_number(system_name: str, text: str) -> int:
    system = NUMERAL_SYSTEMS[str(system_name).strip().lower()]
    if system.key == "roman":
        return _decode_roman(text)
    if system.key == "tally":
        return _decode_tally(text)
    if system.key == "mayan":
        return _decode_positional(text, 20, system.digits, separator=".")
    if system.key == "babylonian":
        return _decode_positional(text, 60, system.digits, separator=":")
    if system.positional and system.base and system.digits:
        return _decode_positional(text, system.base, system.digits)
    raise ValueError(f"System does not support decoding: {system_name}")


__all__ = ["NUMERAL_SYSTEMS", "NumeralSystem", "decode_number", "encode_number", "iter_numeral_systems"]
