"""
Number word normalization for template matching.
"""

from __future__ import annotations

import re

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
    "million": 1000000,
    # Specials (counts)
    "dozen": 12,
    "score": 20,
    "pair": 2,
    "couple": 2,
    "few": 3,
    "several": 5,
}

_FRACTION_DENOMS: dict[str, int] = {
    "half": 2,
    "third": 3,
    "fourth": 4,
    "quarter": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "twelfth": 12,
}


def normalize_number_words(text: str) -> str:
    """Replace number words with digits for template matching."""
    result = text.lower()

    # Normalize common word-fractions into "n/d" while preserving standalone
    # fraction keywords like "half" for downstream pattern matching.
    # Examples: "one-third" -> "1/3", "three-fourths" -> "3/4", "a quarter" -> "1/4".
    ones_only = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    def _frac_replace(m: re.Match[str]) -> str:
        num_raw = str(m.group("num")).lower()
        den_raw = str(m.group("den")).lower()
        den_key = den_raw.rstrip("s")
        if den_key.endswith("ths"):
            den_key = den_key[: -len("ths")]
        den = _FRACTION_DENOMS.get(den_key)
        if not den:
            return m.group(0)
        if num_raw.isdigit():
            num = int(num_raw)
        else:
            num = int(ones_only.get(num_raw, 0))
        if num <= 0:
            return m.group(0)
        return f"{num}/{den}"

    def _a_frac_replace(m: re.Match[str]) -> str:
        den_raw = str(m.group("den")).lower()
        den_key = den_raw.rstrip("s")
        if den_key.endswith("ths"):
            den_key = den_key[: -len("ths")]
        den = _FRACTION_DENOMS.get(den_key)
        if not den:
            return m.group(0)
        return f"1/{den}"

    # Order matters: run fraction normalization before basic word->digit replacement.
    result = re.sub(
        r"\b(?P<num>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s*(?:-| )\s*(?P<den>halves|half|thirds|third|fourths|fourth|quarters|quarter|fifths|fifth|sixths|sixth|sevenths|seventh|eighths|eighth|ninths|ninth|tenths|tenth|twelfths|twelfth)\b",
        _frac_replace,
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\b(?:a|an)\s+(?P<den>halves|half|thirds|third|fourths|fourth|quarters|quarter|fifths|fifth|sixths|sixth|sevenths|seventh|eighths|eighth|ninths|ninth|tenths|tenth|twelfths|twelfth)\b",
        _a_frac_replace,
        result,
        flags=re.IGNORECASE,
    )

    # Compound numbers like twenty-five
    tens_map = [
        ("twenty", 20),
        ("thirty", 30),
        ("forty", 40),
        ("fifty", 50),
        ("sixty", 60),
        ("seventy", 70),
        ("eighty", 80),
        ("ninety", 90),
    ]
    ones_map = [
        ("one", 1),
        ("two", 2),
        ("three", 3),
        ("four", 4),
        ("five", 5),
        ("six", 6),
        ("seven", 7),
        ("eight", 8),
        ("nine", 9),
    ]
    for tens_word, tens_val in tens_map:
        for ones_word, ones_val in ones_map:
            compound = f"{tens_word}[-\\s]?{ones_word}"
            result = re.sub(compound, str(tens_val + ones_val), result)

    # Simple words
    for word, val in NUMBER_WORDS.items():
        result = re.sub(rf"\b{word}\b", str(val), result)

    return result


__all__ = ["normalize_number_words"]
