"""
Lightweight sublexical segmenters (syllables/morphemes) for Latin-script languages.

Heuristics (imperfect but non-stub):
- Syllables: split by CV patterns, preserving digraphs and plausible onsets (plosive+liquid,
  pt/es digraphs ch/nh/lh/rr/qu/gu, common en clusters, etc.).
- Morphemes: greedy longest-match using curated prefix/suffix lists per language.

Supports: pt, es, en, fr, it, de (Latin scripts). Others return empty.
"""

from __future__ import annotations

from typing import List, Dict


def _detect_script(ch: str) -> str:
    """
    Tiny script detector used for cheap fallbacks in segmenters.

    This keeps the ingestion pipeline dependency-light while still allowing
    reasonable defaults for Arabic and CJK text.
    """
    if not ch:
        return "Unknown"
    c = ord(ch[0])
    # Arabic
    if (
        0x0600 <= c <= 0x06FF
        or 0x0750 <= c <= 0x077F
        or 0x08A0 <= c <= 0x08FF
        or 0xFB50 <= c <= 0xFDFF
        or 0xFE70 <= c <= 0xFEFF
    ):
        return "Arabic"
    # CJK Unified / Extensions
    if (
        0x3400 <= c <= 0x4DBF
        or 0x4E00 <= c <= 0x9FFF
        or 0x20000 <= c <= 0x2A6DF
        or 0x2A700 <= c <= 0x2B73F
        or 0x2B740 <= c <= 0x2B81F
        or 0x2B820 <= c <= 0x2CEAF
        or 0x2CEB0 <= c <= 0x2EBEF
    ):
        return "CJK"
    # Hiragana, Katakana, Hangul
    if (0x3040 <= c <= 0x30FF) or (0xAC00 <= c <= 0xD7AF):
        return "CJK"
    return "Latin"


VOWELS_PT = set("aeiouáéíóúâêôãõàüAEIOUÁÉÍÓÚÂÊÔÃÕÀÜ")
VOWELS_ES = set("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")
VOWELS_EN = set("aeiouAEIOU")
VOWELS_FR = set("aeiouyàâæçéèêëîïôœùûüÿAEIOUYÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ")
VOWELS_IT = set("aeiouàèéìíîòóùúAEIOUÀÈÉÌÍÎÒÓÙÚ")
VOWELS_DE = set("aeiouäöüßAEIOUÄÖÜ")
VOWELS_AR = set("اويىءؤئأإآ")  # rough vowel set for Arabic scripts

PREFIXES = {
    "pt": ["re", "des", "in", "im", "ir", "inter", "anti", "sub", "super", "pre", "pos", "co", "contra"],
    "es": ["re", "des", "in", "im", "ir", "inter", "anti", "sub", "super", "pre", "pos", "co", "contra"],
    "en": ["re", "un", "im", "in", "il", "ir", "non", "pre", "post", "anti", "sub", "super", "inter", "trans"],
    "fr": ["re", "in", "im", "ir", "il", "dé", "des", "pré", "post", "anti", "inter", "trans"],
    "it": ["re", "ri", "in", "im", "ir", "il", "pre", "post", "anti", "inter", "trans"],
    "de": ["be", "ver", "zer", "ent", "er", "ge", "miss", "un", "ur"],
}

SUFFIXES = {
    "pt": ["mente", "ção", "são", "dade", "eiro", "eira", "zinho", "zinha", "íssimo", "íssima", "mente"],
    "es": ["mente", "ción", "sión", "dad", "ero", "era", "ito", "ita", "ísimo", "ísima"],
    "en": ["ing", "ed", "er", "est", "ly", "ness", "able", "ible", "ment", "tion", "sion", "ful", "less"],
    "fr": ["ment", "tion", "sion", "able", "ible", "eur", "euse", "age", "ure"],
    "it": ["mente", "zione", "sione", "tore", "trice", "abile", "ibile", "issimo", "issima"],
    "de": ["ung", "keit", "heit", "ung", "bar", "lich", "los", "chen"],
}


def _vowel_set(lang: str):
    if lang == "pt":
        return VOWELS_PT
    if lang == "es":
        return VOWELS_ES
    if lang == "en":
        return VOWELS_EN
    if lang == "fr":
        return VOWELS_FR
    if lang == "it":
        return VOWELS_IT
    if lang == "de":
        return VOWELS_DE
    if lang in {"ar", "fa", "ur"}:
        return VOWELS_AR
    return VOWELS_EN


DIGRAPHS_PT = {"ch", "lh", "nh", "rr", "ss", "qu", "gu"}
DIGRAPHS_ES = {"ch", "ll", "rr", "qu", "gu"}
DIGRAPHS_FR = {"ch", "gn", "qu", "gu"}
DIGRAPHS_IT = {"ch", "gh", "gl", "gn", "qu"}
DIGRAPHS_EN = {"ch", "sh", "th", "ph", "wh", "qu"}
DIGRAPHS_DE = {"ch", "sch", "qu", "sp", "st"}
DIGRAPHS_AR = {"لا"}  # simple lam-alif ligature handling
DIGRAPHS_JA = set()  # Japanese handled via optional tinysegmenter or per-char

ONSET_CLUSTERS = {
    "default": {"pr", "pl", "br", "bl", "tr", "dr", "cr", "cl", "gr", "gl", "fr", "fl"},
    "en": {"pr", "pl", "br", "bl", "tr", "dr", "cr", "cl", "gr", "gl", "fr", "fl", "st", "sp", "sk", "sm", "sn", "sl", "sw"},
    "de": {"pr", "pl", "br", "bl", "tr", "dr", "kr", "kl", "gr", "gl", "fr", "fl", "sp", "st", "sch"},
}


def _digraphs(lang: str):
    if lang == "pt":
        return DIGRAPHS_PT
    if lang == "es":
        return DIGRAPHS_ES
    if lang == "fr":
        return DIGRAPHS_FR
    if lang == "it":
        return DIGRAPHS_IT
    if lang == "de":
        return DIGRAPHS_DE
    if lang == "en":
        return DIGRAPHS_EN
    return set()


def _onsets(lang: str):
    return ONSET_CLUSTERS.get(lang, ONSET_CLUSTERS["default"])


def syllabify(word: str, lang: str) -> List[Dict]:
    """
    Heuristic syllabifier: preserves digraphs and plausible onsets; splits before a vowel
    that starts a new nucleus; keeps consonant+liquid clusters together when reasonable.
    """
    if not word:
        return []
    # If pyphen is available, compute an additional hyphenation-based split
    hyph_parts: List[Dict] = []
    try:
        import pyphen  # type: ignore

        if lang in {"en", "fr", "de", "es", "pt", "it"}:
            dic = pyphen.Pyphen(lang=lang)
            hyph = dic.inserted(word)
            start = 0
            for chunk in hyph.split("-"):
                end = start + len(chunk)
                hyph_parts.append(
                    {"start": start, "end": end, "syllable": word[start:end], "pattern": "HYPH", "method": "hyphenation"}
                )
                start = end
    except ImportError:
        pass
    # Arabic script: naive split on vowels; keep consonant clusters before vowel as onset; treat lam-alif as unit
    if lang in {"ar", "fa", "ur"} or _detect_script(word[0]) in {"Arabic"}:
        vowels = VOWELS_AR
        parts = []
        start = 0
        i = 0
        while i < len(word):
            # lam-alif ligature
            if i + 1 < len(word) and word[i : i + 2] in DIGRAPHS_AR:
                i += 2
                continue
            ch = word[i]
            if ch in vowels and i > start:
                parts.append({"start": start, "end": i, "syllable": word[start:i], "pattern": "CONS"})
                start = i
            i += 1
        parts.append({"start": start, "end": len(word), "syllable": word[start:], "pattern": "MIX"})
        return parts
    # CJK: use jieba for zh if available, else per character; for ja use tinysegmenter if available else per char
    if _detect_script(word[0]) == "CJK" or lang in {"zh", "ja", "ko"}:
        tokens: List[Dict] = []
        if lang == "zh":
            try:
                import jieba  # type: ignore

                offset = 0
                for tok in jieba.cut(word):
                    tokens.append({"start": offset, "end": offset + len(tok), "syllable": tok, "pattern": "CJK", "method": "jieba"})
                    offset += len(tok)
                return tokens
            except ImportError:
                pass
        if lang == "ja":
            try:
                from tinysegmenter import TinySegmenter  # type: ignore

                seg = TinySegmenter()
                offset = 0
                for tok in seg.tokenize(word):
                    tokens.append({"start": offset, "end": offset + len(tok), "syllable": tok, "pattern": "JA", "method": "tinysegmenter"})
                    offset += len(tok)
                return tokens
            except ImportError:
                pass
        return [{"start": i, "end": i + 1, "syllable": ch, "pattern": "CJK", "method": "per_char"} for i, ch in enumerate(word)]
    vowels = _vowel_set(lang)
    digs = _digraphs(lang)
    onsets = _onsets(lang)
    i = 0
    parts: List[Dict] = []
    while i < len(word):
        start = i
        # collect onset (one or two chars if digraph or onset cluster)
        if i + 2 <= len(word) and word[i:i+2].lower() in digs:
            i += 2
        elif i + 3 <= len(word) and word[i:i+3].lower() in digs:
            i += 3
        else:
            if word[i] not in vowels:
                i += 1
                if i + 1 <= len(word) and (word[i:i+1].lower() in onsets):
                    i += 1
        # nucleus
        while i < len(word) and word[i] in vowels:
            i += 1
        # coda (optional single consonant; push to next onset if cluster)
        if i < len(word) and word[i] not in vowels:
            if i + 1 < len(word) and word[i+1] in vowels:
                pass
            else:
                i += 1
        end = i
        syl = word[start:end]
        pattern = "".join("V" if ch in vowels else "C" for ch in syl)
        parts.append({"start": start, "end": end, "syllable": syl, "pattern": pattern, "method": "heuristic"})
    # combine both heuristic and hyphenation splits (if hyph_parts present)
    return hyph_parts + parts


def morph_segment(word: str, lang: str) -> List[Dict]:
    """
    Greedy longest prefix/suffix match using curated lists. Returns spans.
    """
    # Arabic clitic-aware split (very rough)
    if lang in {"ar", "fa", "ur"} or _detect_script(word[:1]) == "Arabic":
        proclitics = ["ال", "و", "ف", "ب", "ك", "ل", "س"]  # al-, wa-, fa-, bi-, ka-, li-, sa-
        enclitics = ["ه", "هم", "هن", "كما", "كما", "كم", "كن", "نا", "ي", "ها"]
        segs: List[Dict] = []
        start = 0
        for p in sorted(proclitics, key=len, reverse=True):
            if word.startswith(p):
                segs.append({"start": 0, "end": len(p), "morpheme_id": f"CLITIC_prefix_{p}_{lang}"})
                start = len(p)
                break
        end_cut = len(word)
        for s in sorted(enclitics, key=len, reverse=True):
            if word.endswith(s) and len(word) > len(s) + start:
                end_cut = len(word) - len(s)
                segs.append({"start": end_cut, "end": len(word), "morpheme_id": f"CLITIC_suffix_{s}_{lang}"})
                break
        if start > 0 or end_cut < len(word):
            segs.append({"start": start, "end": end_cut, "morpheme_id": f"STEM_{lang}"})
            return segs
    # Latin heuristics
    word_lower = word.lower()
    prefs = sorted(PREFIXES.get(lang, []), key=len, reverse=True)
    sufs = sorted(SUFFIXES.get(lang, []), key=len, reverse=True)
    segments: List[Dict] = []

    # longest prefix
    for p in prefs:
        if word_lower.startswith(p):
            segments.append({"start": 0, "end": len(p), "morpheme_id": f"MORPHEME_prefix_{p}_{lang}"})
            break
    # longest suffix
    for s in sufs:
        if word_lower.endswith(s) and len(word_lower) > len(s):
            segments.append({"start": len(word_lower) - len(s), "end": len(word_lower), "morpheme_id": f"MORPHEME_suffix_{s}_{lang}"})
            break
    return segments
