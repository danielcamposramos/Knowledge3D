"""
Character Language Mappings for Multilingual Atomic Units

Maps characters to the languages that use them, using ISO 639-1/BCP 47 codes.

This enables:
- Proper language-aware character composition
- Pronunciation encoding per language
- Cross-linguistic knowledge representation

Standards Used:
- ISO 639-1: Two-letter language codes (en, pt, es, fr, ...)
- BCP 47: Extended tags for regional variants (pt-BR, pt-PT, en-US, en-GB)

Author: K3D Adaptive Swarm
Date: 2025-11-19
"""

from typing import Dict, List, Set

# Basic Latin alphabet (A-Z, a-z) - used across many Western languages
LATIN_BASIC_LANGUAGES = [
    'en',  # English
    'pt',  # Portuguese
    'es',  # Spanish
    'fr',  # French
    'de',  # German
    'it',  # Italian
    'nl',  # Dutch
    'sv',  # Swedish
    'no',  # Norwegian
    'da',  # Danish
    'fi',  # Finnish
    'pl',  # Polish
    'cs',  # Czech
    'sk',  # Slovak
    'ro',  # Romanian
    'hu',  # Hungarian
    'tr',  # Turkish
    'id',  # Indonesian
    'ms',  # Malay
    'tl',  # Tagalog
    'sw',  # Swahili
    'zu',  # Zulu
    'af',  # Afrikaans
    'sq',  # Albanian
    'eu',  # Basque
    'ca',  # Catalan
    'gl',  # Galician
    'cy',  # Welsh
    'ga',  # Irish
    'gd',  # Scottish Gaelic
    'is',  # Icelandic
    'lb',  # Luxembourgish
    'mt',  # Maltese
]

def _unique_list(values: List[str]) -> List[str]:
    """Return sorted unique copy of language codes."""
    return sorted(dict.fromkeys(values))


# Extended Latin characters with diacritics
EXTENDED_LATIN_LANGUAGES: Dict[str, List[str]] = {
    # Portuguese-specific
    'ç': ['pt', 'fr', 'ca', 'tr', 'sq'],
    'Ç': ['pt', 'fr', 'ca', 'tr', 'sq'],
    'ã': ['pt'],
    'Ã': ['pt'],
    'õ': ['pt', 'et'],
    'Õ': ['pt', 'et'],

    # French-specific
    'é': ['fr', 'pt', 'es', 'ca', 'it', 'cs', 'sk', 'is'],
    'É': ['fr', 'pt', 'es', 'ca', 'it', 'cs', 'sk', 'is'],
    'è': ['fr', 'it', 'ca'],
    'È': ['fr', 'it', 'ca'],
    'ê': ['fr', 'pt'],
    'Ê': ['fr', 'pt'],
    'ë': ['fr', 'nl', 'sq'],
    'Ë': ['fr', 'nl', 'sq'],
    'à': ['fr', 'it', 'pt', 'ca'],
    'À': ['fr', 'it', 'pt', 'ca'],
    'â': ['fr', 'pt', 'ro'],
    'Â': ['fr', 'pt', 'ro'],

    # German-specific
    'ä': ['de', 'fi', 'sv', 'et'],
    'Ä': ['de', 'fi', 'sv', 'et'],
    'ö': ['de', 'fi', 'sv', 'tr', 'hu', 'et'],
    'Ö': ['de', 'fi', 'sv', 'tr', 'hu', 'et'],
    'ü': ['de', 'tr', 'hu', 'et', 'es', 'ca'],
    'Ü': ['de', 'tr', 'hu', 'et', 'es', 'ca'],
    'ß': ['de'],

    # Spanish-specific
    'ñ': ['es', 'gl', 'eu', 'qu', 'ay', 'gn'],
    'Ñ': ['es', 'gl', 'eu', 'qu', 'ay', 'gn'],
    'á': ['es', 'pt', 'hu', 'cs', 'sk', 'is', 'ga'],
    'Á': ['es', 'pt', 'hu', 'cs', 'sk', 'is', 'ga'],
    'í': ['es', 'pt', 'is', 'ga'],
    'Í': ['es', 'pt', 'is', 'ga'],
    'ó': ['es', 'pt', 'hu', 'pl', 'cs', 'sk', 'is', 'ga'],
    'Ó': ['es', 'pt', 'hu', 'pl', 'cs', 'sk', 'is', 'ga'],
    'ú': ['es', 'pt', 'hu', 'cs', 'sk', 'ga'],
    'Ú': ['es', 'pt', 'hu', 'cs', 'sk', 'ga'],

    # Nordic languages
    'å': ['sv', 'no', 'da'],
    'Å': ['sv', 'no', 'da'],
    'æ': ['no', 'da', 'is'],
    'Æ': ['no', 'da', 'is'],
    'ø': ['no', 'da'],
    'Ø': ['no', 'da'],
    'þ': ['is'],
    'Þ': ['is'],
    'ð': ['is'],
    'Ð': ['is'],

    # Polish-specific
    'ą': ['pl', 'lt'],
    'Ą': ['pl', 'lt'],
    'ć': ['pl'],
    'Ć': ['pl'],
    'ę': ['pl', 'lt'],
    'Ę': ['pl', 'lt'],
    'ł': ['pl'],
    'Ł': ['pl'],
    'ń': ['pl'],
    'Ń': ['pl'],
    'ś': ['pl'],
    'Ś': ['pl'],
    'ź': ['pl'],
    'Ź': ['pl'],
    'ż': ['pl'],
    'Ż': ['pl'],

    # Czech/Slovak-specific
    'č': ['cs', 'sk', 'sl', 'hr', 'bs', 'sh'],
    'Č': ['cs', 'sk', 'sl', 'hr', 'bs', 'sh'],
    'ď': ['cs', 'sk'],
    'Ď': ['cs', 'sk'],
    'ě': ['cs'],
    'Ě': ['cs'],
    'ň': ['cs', 'sk'],
    'Ň': ['cs', 'sk'],
    'ř': ['cs'],
    'Ř': ['cs'],
    'š': ['cs', 'sk', 'sl', 'hr', 'bs', 'sh', 'lt', 'lv', 'et'],
    'Š': ['cs', 'sk', 'sl', 'hr', 'bs', 'sh', 'lt', 'lv', 'et'],
    'ť': ['cs', 'sk'],
    'Ť': ['cs', 'sk'],
    'ů': ['cs'],
    'Ů': ['cs'],
    'ž': ['cs', 'sk', 'sl', 'hr', 'bs', 'sh', 'lt', 'lv', 'et'],
    'Ž': ['cs', 'sk', 'sl', 'hr', 'bs', 'sh', 'lt', 'lv', 'et'],

    # Romanian-specific
    'ă': ['ro'],
    'Ă': ['ro'],
    'î': ['ro', 'fr'],
    'Î': ['ro', 'fr'],
    'ș': ['ro'],
    'Ș': ['ro'],
    'ț': ['ro'],
    'Ț': ['ro'],

    # Turkish-specific
    'ğ': ['tr'],
    'Ğ': ['tr'],
    'ı': ['tr'],
    'İ': ['tr'],
    'ş': ['tr', 'ro'],
    'Ş': ['tr', 'ro'],

    # Hungarian-specific
    'ő': ['hu'],
    'Ő': ['hu'],
    'ű': ['hu'],
    'Ű': ['hu'],
}

# Cyrillic basic alphabet coverage (А-Я, а-я)
CYRILLIC_BASIC_LANGUAGES: List[str] = [
    'ru',  # Russian
    'uk',  # Ukrainian
    'be',  # Belarusian
    'bg',  # Bulgarian
    'sr',  # Serbian (Cyrillic)
    'mk',  # Macedonian
    'kk',  # Kazakh
    'ky',  # Kyrgyz
    'tg',  # Tajik
    'mn',  # Mongolian (Cyrillic)
    'uz',  # Uzbek (Cyrillic)
    'ba',  # Bashkir
    'ce',  # Chechen
    'cv',  # Chuvash
    'kv',  # Komi
    'os',  # Ossetian
    'tt',  # Tatar
    'tyv', # Tuvan
    'udm', # Udmurt
    'sah', # Sakha (Yakut)
    'ab',  # Abkhaz
    'ady', # Adyghe
    'av',  # Avar
    'kbd', # Kabardian
    'krc', # Karachay-Balkar
    'lbe', # Lak
    'lez', # Lezgian
    'tab', # Tabasaran
    'cv',  # Chuvash
    'mo',  # Moldovan (Transnistria)
    'srp', # Serbian standard code
    'bs',  # Bosnian (Cyrillic usage)
    'sh',  # Serbo-Croatian (historic)
]

# Extended Cyrillic characters (language-specific additions)
EXTENDED_CYRILLIC_LANGUAGES: Dict[str, List[str]] = {
    # Russian / Belarusian
    'ё': ['ru', 'be'],
    'Ё': ['ru', 'be'],
    # Ukrainian-specific
    'є': ['uk'],
    'Є': ['uk'],
    'ї': ['uk'],
    'Ї': ['uk'],
    'і': ['uk', 'be', 'kk'],
    'І': ['uk', 'be', 'kk'],
    'ґ': ['uk'],
    'Ґ': ['uk'],
    # Belarusian-specific
    'ў': ['be'],
    'Ў': ['be'],
    # Serbian / Macedonian
    'ђ': ['sr', 'bs'],
    'Ђ': ['sr', 'bs'],
    'ј': ['sr', 'mk'],
    'Ј': ['sr', 'mk'],
    'љ': ['sr', 'mk'],
    'Љ': ['sr', 'mk'],
    'њ': ['sr', 'mk'],
    'Њ': ['sr', 'mk'],
    'ћ': ['sr', 'bs'],
    'Ћ': ['sr', 'bs'],
    'џ': ['sr', 'mk'],
    'Џ': ['sr', 'mk'],
    'ѓ': ['mk'],
    'Ѓ': ['mk'],
    'ќ': ['mk'],
    'Ќ': ['mk'],
    'ѕ': ['mk'],
    'Ѕ': ['mk'],
    # Bulgarian
    'ъ': ['bg'],
    'Ъ': ['bg'],
    'ь': ['bg', 'ru'],
    'Ь': ['bg', 'ru'],
    'ю': ['bg', 'ru', 'uk'],
    'Ю': ['bg', 'ru', 'uk'],
    'я': ['bg', 'ru', 'uk'],
    'Я': ['bg', 'ru', 'uk'],
    # Kazakh
    'ә': ['kk'],
    'Ә': ['kk'],
    'ғ': ['kk'],
    'Ғ': ['kk'],
    'қ': ['kk'],
    'Қ': ['kk'],
    'ң': ['kk'],
    'Ң': ['kk'],
    'ө': ['kk'],
    'Ө': ['kk'],
    'ұ': ['kk'],
    'Ұ': ['kk'],
    'ү': ['kk'],
    'Ү': ['kk'],
    'һ': ['kk'],
    'Һ': ['kk'],
    # Kyrgyz
    'Ң': ['ky'],
    'Ӯ': ['tg'],
    'ӯ': ['tg'],
    # Tajik
    'ҷ': ['tg'],
    'Ҷ': ['tg'],
    'ҳ': ['tg'],
    'Ҳ': ['tg'],
    # Tatar / Bashkir
    'ә': ['tt', 'ba'],
    'җ': ['tt', 'ba'],
    'Җ': ['tt', 'ba'],
    'ң': ['tt', 'ba'],
    'Ң': ['tt', 'ba'],
    'ө': ['tt', 'ba'],
    'Ө': ['tt', 'ba'],
    'ү': ['tt', 'ba'],
    'Ү': ['tt', 'ba'],
}

# Math symbols - language agnostic (universal across languages)
MATH_SYMBOLS_UNIVERSAL = [
    '+', '-', '×', '÷', '=', '≠', '≈', '≡',
    '<', '>', '≤', '≥',
    '∞', '∫', '∂', '∑', '∏',
    '√', '∛', '∜',
    '°', '%', '‰',
    'π', 'e', 'φ', 'θ', 'α', 'β', 'γ', 'δ', 'ε', 'λ', 'μ', 'σ', 'ω',
]

# Digits - universal across languages (Arabic numerals)
DIGITS_UNIVERSAL = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Punctuation - language agnostic
PUNCTUATION_UNIVERSAL = [
    '.', ',', ';', ':', '!', '?',
    '(', ')', '[', ']', '{', '}',
    '"', "'", '`', '´',
    '-', '—', '–',
    '/', '\\', '|',
    '*', '&', '@', '#', '$',
]


def get_character_languages(char: str) -> List[str]:
    """
    Get list of language codes for a given character.

    Args:
        char: Single character (e.g., 'a', 'ç', '+', '5')

    Returns:
        List of ISO 639-1 language codes that use this character
        Empty list for unknown characters

    Examples:
        >>> get_character_languages('a')
        ['en', 'pt', 'es', 'fr', 'de', ...]  # Many languages

        >>> get_character_languages('ç')
        ['pt', 'fr', 'ca', 'tr', 'sq']  # Fewer languages

        >>> get_character_languages('+')
        ['universal']  # Math symbols are language-agnostic
    """
    if not char or len(char) != 1:
        return []

    # Math symbols - universal
    if char in MATH_SYMBOLS_UNIVERSAL:
        return ['universal']

    # Digits - universal
    if char in DIGITS_UNIVERSAL:
        return ['universal']

    # Punctuation - universal
    if char in PUNCTUATION_UNIVERSAL:
        return ['universal']

    # Extended Cyrillic forms
    if char in EXTENDED_CYRILLIC_LANGUAGES:
        return _unique_list(EXTENDED_CYRILLIC_LANGUAGES[char])

    # Extended Latin with diacritics
    if char in EXTENDED_LATIN_LANGUAGES:
        return _unique_list(EXTENDED_LATIN_LANGUAGES[char])

    # Basic Cyrillic alphabet (А-Я, а-я including additional range)
    if '\u0400' <= char <= '\u04FF':
        return _unique_list(CYRILLIC_BASIC_LANGUAGES)

    # Basic Latin alphabet (A-Z, a-z)
    if char.isascii() and char.isalpha():
        return _unique_list(LATIN_BASIC_LANGUAGES)

    # Unknown character
    return []


def get_language_name(code: str) -> str:
    """
    Get human-readable language name from ISO 639-1 code.

    Args:
        code: Two-letter language code (e.g., 'en', 'pt', 'es')

    Returns:
        Language name in English
    """
    language_names = {
        'en': 'English',
        'pt': 'Portuguese',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'nl': 'Dutch',
        'sv': 'Swedish',
        'no': 'Norwegian',
        'da': 'Danish',
        'fi': 'Finnish',
        'pl': 'Polish',
        'cs': 'Czech',
        'sk': 'Slovak',
        'ro': 'Romanian',
        'hu': 'Hungarian',
        'tr': 'Turkish',
        'id': 'Indonesian',
        'ms': 'Malay',
        'tl': 'Tagalog',
        'sw': 'Swahili',
        'zu': 'Zulu',
        'af': 'Afrikaans',
        'sq': 'Albanian',
        'eu': 'Basque',
        'ca': 'Catalan',
        'gl': 'Galician',
        'cy': 'Welsh',
        'ga': 'Irish',
        'gd': 'Scottish Gaelic',
        'is': 'Icelandic',
        'lb': 'Luxembourgish',
        'mt': 'Maltese',
        'et': 'Estonian',
        'lv': 'Latvian',
        'lt': 'Lithuanian',
        'sl': 'Slovenian',
        'hr': 'Croatian',
        'bs': 'Bosnian',
        'sh': 'Serbo-Croatian',
        'qu': 'Quechua',
        'ay': 'Aymara',
        'gn': 'Guarani',
        'ru': 'Russian',
        'uk': 'Ukrainian',
        'be': 'Belarusian',
        'bg': 'Bulgarian',
        'sr': 'Serbian',
        'mk': 'Macedonian',
        'kk': 'Kazakh',
        'ky': 'Kyrgyz',
        'tg': 'Tajik',
        'mn': 'Mongolian',
        'uz': 'Uzbek',
        'ba': 'Bashkir',
        'ce': 'Chechen',
        'cv': 'Chuvash',
        'os': 'Ossetian',
        'tt': 'Tatar',
        'tyv': 'Tuvan',
        'udm': 'Udmurt',
        'sah': 'Sakha',
        'ab': 'Abkhaz',
        'ady': 'Adyghe',
        'av': 'Avar',
        'kbd': 'Kabardian',
        'krc': 'Karachay-Balkar',
        'lbe': 'Lak',
        'lez': 'Lezgian',
        'tab': 'Tabasaran',
        'universal': 'Universal (language-agnostic)',
    }

    return language_names.get(code, f'Unknown ({code})')


def get_character_stats() -> Dict[str, float]:
    """
    Get statistics about character-language mappings.

    Returns:
        Dictionary summarizing counts and averages across scripts.
    """
    unique_chars: Set[str] = set()
    total_language_links = 0

    # Basic Latin (upper + lower)
    for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz':
        unique_chars.add(c)
        total_language_links += len(LATIN_BASIC_LANGUAGES)

    # Extended Latin
    for char, langs in EXTENDED_LATIN_LANGUAGES.items():
        unique_chars.add(char)
        total_language_links += len(langs)

    # Basic Cyrillic U+0400–U+04FF
    for codepoint in range(0x0400, 0x0500):
        char = chr(codepoint)
        unique_chars.add(char)
        total_language_links += len(CYRILLIC_BASIC_LANGUAGES)

    # Extended Cyrillic overrides
    for char, langs in EXTENDED_CYRILLIC_LANGUAGES.items():
        unique_chars.add(char)
        total_language_links += len(langs)

    universal_count = (
        len(MATH_SYMBOLS_UNIVERSAL)
        + len(DIGITS_UNIVERSAL)
        + len(PUNCTUATION_UNIVERSAL)
    )

    total_chars = len(unique_chars) + universal_count
    avg_languages = total_language_links / max(len(unique_chars), 1)

    return {
        'total_chars': float(total_chars),
        'universal_chars': float(universal_count),
        'latin_chars': float(52 + len(EXTENDED_LATIN_LANGUAGES)),
        'cyrillic_chars': float((0x0500 - 0x0400) + len(EXTENDED_CYRILLIC_LANGUAGES)),
        'avg_languages_per_char': float(avg_languages),
    }


if __name__ == '__main__':
    # Demo usage
    print("="*80)
    print("Character Language Mappings - Examples")
    print("="*80)

    # Test basic Latin
    print("\n[Basic Latin]")
    for char in ['a', 'A', 'z', 'Z']:
        langs = get_character_languages(char)
        print(f"  '{char}': {len(langs)} languages ({langs[:5]}...)")

    # Test extended Latin
    print("\n[Extended Latin]")
    for char in ['ç', 'ñ', 'ä', 'ø', 'ł']:
        langs = get_character_languages(char)
        lang_names = [get_language_name(l) for l in langs[:3]]
        print(f"  '{char}': {len(langs)} languages ({', '.join(lang_names)}...)")

    # Test math symbols
    print("\n[Math Symbols]")
    for char in ['+', 'π', '∫', '√']:
        langs = get_character_languages(char)
        print(f"  '{char}': {langs}")

    # Statistics
    print("\n[Statistics]")
    stats = get_character_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
