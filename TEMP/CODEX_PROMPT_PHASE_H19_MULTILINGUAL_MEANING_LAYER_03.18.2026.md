# Phase H19 — Multilingual Meaning Layer (OMW Synset → MeaningCentricStar)

**Depends on:** H17 (Universal Knowledge Foundation), existing word_stars_ud.jsonl (161 languages)
**Creates:** `knowledge3d/ingestion/universal_knowledge/multilingual_meanings.py`, `tests/test_multilingual_meanings.py`
**Modifies:** `knowledge3d/ingestion/universal_knowledge/__init__.py`
**Data Source:** Open Multilingual Wordnet (OMW) at `/K3D/K3D_llama_cpp/datasets/omw-data/omw-data-main/wns/`
**Goal:** Build the MEANING LAYER — one MeaningCentricStar per WordNet synset, with surface_forms from ALL available languages as symlinks. This is the foundational proof that K3D's semantic gravity works: words cluster by MEANING, not by language.

---

## The Problem

We have 1,645,760 word entries across 161 languages (`word_stars_ud.jsonl`) — but they are **islands**. "water" (en), "água" (pt), "eau" (fr), "水" (zh), "вода" (ru) are separate entries with NO connection.

We also have Open Multilingual Wordnet with 117,659 unique synsets across 31 languages — where `00001740-a` means "able/capable" in EVERY language, with aligned lemmas.

**The missing piece:** A meaning star per synset, where `surface_forms` symlinks to the word in each language, and `char_refs` symlinks to the glyph entries from Layer 1.

---

## Critical Language Rule

**English is ALWAYS the primary language (W3C standard).**

- `meaning_rpn`: ALWAYS in English
- `definitions`: ALWAYS in English
- Other languages appear ONLY as `surface_forms` symlink references (word_ref + char_refs)
- **Exception:** If a concept exists ONLY in another language and has NO English equivalent (e.g., Portuguese "saudade", Japanese "木漏れ日" komorebi, German "Schadenfreude"), THEN the meaning text is in that source language. These are rare — most WordNet synsets have English definitions.
- The star's semantic content (rpn, definition, taxonomy) is English. The star's surface_forms are multilingual symlinks.

---

## Architecture

```
OMW Tab Files (31 languages, 117K synsets)
    ↓
multilingual_meanings.py
    ├─ Parse all wn-data-{lang}.tab files
    ├─ Group lemmas by synset ID across languages
    ├─ For each synset:
    │   ├─ star_id = "synset_{synset_id}" (e.g., "synset_00001740_a")
    │   ├─ meaning_class = "n" → "noun", "v" → "verb", "a" → "adjective", "r" → "adverb"
    │   ├─ meaning_rpn = ALWAYS from English definition/lemma
    │   │   (if no English def exists but lemmas exist in other languages ONLY,
    │   │    use that language's definition — this marks a culture-specific concept)
    │   ├─ surface_forms = {en: SurfaceForm(word_ref, char_refs), pt: ..., fr: ..., ja: ...}
    │   ├─ taxonomy_refs = ["concept_language", "wordnet_synset", POS category]
    │   └─ domain = from House routing (Language by default)
    │   ↓
    └─ MeaningCentricStar
        ↓
    build_meaning_layer_stars() → list[MeaningCentricStar]
        ↓
    write_stars_jsonl() or galaxy_manager.store_meaning_star()
```

---

## Data Format

OMW tab files have this format:
```
# Comment line
{synset_id}\tlemma\t{word}           ← word entry
{synset_id}\t{lang}:lemma\t{word}    ← word entry (some files use lang prefix)
{synset_id}\t{lang}:def\t{n}\t{text} ← definition
{synset_id}\t{lang}:exe\t{n}\t{text} ← example sentence
```

Synset IDs are globally shared: `00001740-a` in `eng/wn-data-eng.tab` is the SAME concept as `00001740-a` in `por/wn-data-por.tab`.

The suffix after `-` indicates part of speech: `n` = noun, `v` = verb, `a` = adjective, `r` = adverb.

---

## Available Languages (31)

```
als arb bul cldr cow cwn dan ell eng fas fin fra heb hrv isl ita iwn jpn mcr msa nld nor pol por ron slk slv swe tha wikt
```

Key coverage:
- **eng**: 206,978 lemmas (base)
- **fin**: 189,226 lemmas
- **jpn**: 158,080 lemmas
- **fra**: 102,647 lemmas
- **por**: 74,012 lemmas
- **arb**: 54,966 lemmas

---

## New File: `knowledge3d/ingestion/universal_knowledge/multilingual_meanings.py`

```python
"""Build meaning-centric stars from Open Multilingual Wordnet synsets.

Each WordNet synset becomes ONE MeaningCentricStar with surface_forms
from all available languages — proving that meaning is language-agnostic.

The synset IS the meaning. The lemmas are surface_form symlinks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from knowledge3d.knowledgeverse._house_utils import char_refs
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar, SurfaceForm


# ---------------------------------------------------------------------------
# OMW data directory (default location)
# ---------------------------------------------------------------------------

OMW_DEFAULT_PATH = Path("/K3D/K3D_llama_cpp/datasets/omw-data/omw-data-main/wns")

# Map OMW directory names to ISO 639-1 codes for surface_forms keys
OMW_LANG_MAP: dict[str, str] = {
    "eng": "en", "por": "pt", "fra": "fr", "jpn": "ja", "arb": "ar",
    "ita": "it", "dan": "da", "ell": "el", "fin": "fi", "heb": "he",
    "hrv": "hr", "isl": "is", "nld": "nl", "nor": "no", "pol": "pl",
    "ron": "ro", "slk": "sk", "slv": "sl", "swe": "sv", "tha": "th",
    "bul": "bg", "fas": "fa", "msa": "ms",
    # Special wordnets
    "als": "sq",   # Albanian
    "cwn": "zh",   # Chinese Wordnet
    "iwn": "id",   # Indonesian Wordnet
    "mcr": "es",   # Spanish MCR
    "cow": "zh",   # Chinese Open Wordnet
    "wikt": "mul", # Wiktionary (multilingual, skip or map)
    "cldr": "mul", # CLDR (multilingual, skip)
}

POS_MAP: dict[str, str] = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "r": "adverb",
}


# ---------------------------------------------------------------------------
# Parsed synset structure
# ---------------------------------------------------------------------------

@dataclass
class SynsetEntry:
    """One synset with lemmas from all languages."""

    synset_id: str                              # e.g., "00001740-a"
    pos: str = ""                               # n, v, a, r
    lemmas: dict[str, list[str]] = field(default_factory=dict)  # lang_code → [lemma, ...]
    definitions: dict[str, str] = field(default_factory=dict)   # lang_code → definition
    examples: dict[str, list[str]] = field(default_factory=dict)  # lang_code → [example, ...]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_omw_tab(filepath: Path, lang_code: str) -> dict[str, SynsetEntry]:
    """Parse one wn-data-{lang}.tab file into synset entries."""
    synsets: dict[str, SynsetEntry] = {}

    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            synset_id = parts[0].strip()
            if not re.match(r"^\d{8}-[nvar]$", synset_id):
                continue

            field_type = parts[1].strip()
            value = parts[2].strip() if len(parts) > 2 else ""

            # Normalize field_type: some files use "fra:lemma", others just "lemma"
            if ":" in field_type:
                _, field_type = field_type.rsplit(":", 1)
            field_type = field_type.strip()

            entry = synsets.setdefault(synset_id, SynsetEntry(
                synset_id=synset_id,
                pos=synset_id.split("-")[-1],
            ))

            if field_type == "lemma":
                entry.lemmas.setdefault(lang_code, []).append(value)
            elif field_type == "def":
                # parts[2] is the def index, parts[3] is the text
                if len(parts) > 3:
                    entry.definitions[lang_code] = parts[3].strip()
            elif field_type == "exe":
                if len(parts) > 3:
                    entry.examples.setdefault(lang_code, []).append(parts[3].strip())

    return synsets


def load_all_omw(omw_path: Path | None = None) -> dict[str, SynsetEntry]:
    """Load all OMW tab files and merge into a unified synset dictionary.

    Returns: dict mapping synset_id → SynsetEntry with lemmas from all languages.
    """
    omw_path = omw_path or OMW_DEFAULT_PATH
    merged: dict[str, SynsetEntry] = {}

    for lang_dir in sorted(omw_path.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang_key = lang_dir.name
        lang_code = OMW_LANG_MAP.get(lang_key)
        if not lang_code or lang_code == "mul":
            continue  # Skip multilingual/ambiguous sources

        tab_file = lang_dir / f"wn-data-{lang_key}.tab"
        if not tab_file.exists():
            continue

        parsed = parse_omw_tab(tab_file, lang_code)

        for synset_id, entry in parsed.items():
            if synset_id not in merged:
                merged[synset_id] = SynsetEntry(
                    synset_id=synset_id,
                    pos=entry.pos,
                )
            target = merged[synset_id]
            for lang, lemmas in entry.lemmas.items():
                existing = target.lemmas.setdefault(lang, [])
                for lemma in lemmas:
                    if lemma not in existing:
                        existing.append(lemma)
            for lang, defn in entry.definitions.items():
                if lang not in target.definitions:
                    target.definitions[lang] = defn
            for lang, exes in entry.examples.items():
                existing = target.examples.setdefault(lang, [])
                for ex in exes:
                    if ex not in existing:
                        existing.append(ex)

    return merged


# ---------------------------------------------------------------------------
# Star builder
# ---------------------------------------------------------------------------

def synset_to_star(entry: SynsetEntry) -> MeaningCentricStar:
    """Convert one synset into a MeaningCentricStar.

    The synset IS the meaning. Each language's lemma is a surface_form symlink.
    char_refs in each surface_form point to Layer 1 character entries.
    """
    # Star ID: synset_{id} with dash replaced by underscore
    star_id = f"synset_{entry.synset_id.replace('-', '_')}"

    # Meaning class from POS
    meaning_class = POS_MAP.get(entry.pos, "concept")

    # Meaning RPN: ALWAYS in English (W3C primary language)
    # Exception: if no English definition AND no English lemma exists,
    # this is a culture-specific concept — use the first available language
    en_def = entry.definitions.get("en", "")
    en_lemmas = entry.lemmas.get("en", [])
    first_en = en_lemmas[0] if en_lemmas else ""

    if first_en and en_def:
        # Standard case: English definition available
        meaning_rpn = f"SYNSET {entry.pos.upper()} {first_en.upper().replace(' ', '_')} DEF {en_def[:80]}"
    elif first_en:
        # English lemma but no definition
        meaning_rpn = f"SYNSET {entry.pos.upper()} {first_en.upper().replace(' ', '_')}"
    else:
        # No English at all — culture-specific concept
        # Find first available language with a lemma
        fallback_lang = ""
        fallback_lemma = ""
        fallback_def = ""
        for lang in sorted(entry.lemmas):
            if entry.lemmas[lang]:
                fallback_lang = lang
                fallback_lemma = entry.lemmas[lang][0]
                fallback_def = entry.definitions.get(lang, "")
                break
        if fallback_def:
            meaning_rpn = f"SYNSET {entry.pos.upper()} LANG_{fallback_lang.upper()} {fallback_lemma.upper().replace(' ', '_')} DEF {fallback_def[:80]}"
        elif fallback_lemma:
            meaning_rpn = f"SYNSET {entry.pos.upper()} LANG_{fallback_lang.upper()} {fallback_lemma.upper().replace(' ', '_')}"
        else:
            meaning_rpn = f"SYNSET {entry.pos.upper()} {entry.synset_id}"

    # Surface forms: one per language, using FIRST lemma (most common)
    # Additional lemmas stored as synonyms in meta_refs
    surface_forms: dict[str, SurfaceForm] = {}
    synonym_refs: list[str] = []

    for lang, lemmas in entry.lemmas.items():
        if not lemmas:
            continue
        primary = lemmas[0]
        word_ref = f"{lang}_{primary.lower().replace(' ', '_')}"
        surface_forms[lang] = SurfaceForm(
            word_ref=word_ref,
            char_refs=char_refs(primary, lang),
        )
        # Track additional lemmas as synonyms
        for extra in lemmas[1:]:
            synonym_refs.append(f"synonym:{lang}:{extra}")

    # Taxonomy refs
    taxonomy = ["concept_language", "wordnet_synset"]
    pos_concept = {
        "n": "concept_noun", "v": "concept_verb",
        "a": "concept_adjective", "r": "concept_adverb",
    }
    if entry.pos in pos_concept:
        taxonomy.append(pos_concept[entry.pos])

    # Meta refs: language count, synonym count, definition languages
    meta = [
        f"wordnet:{entry.synset_id}",
        f"languages:{len(entry.lemmas)}",
    ]
    meta.extend(synonym_refs[:20])  # Cap synonyms to keep manageable

    # Domain: Language by default (these are word meanings)
    domain = "Foundation/Language"

    return MeaningCentricStar(
        star_id=star_id,
        meaning_class=meaning_class,
        meaning_rpn=meaning_rpn,
        domain=domain,
        taxonomy_refs=taxonomy,
        surface_forms=surface_forms,
        meta_refs=meta,
        house_room="House/Library",
        confidence=1,
        polarity=1,
    )


def iter_meaning_stars(
    omw_path: Path | None = None,
    *,
    min_languages: int = 2,
    pos_filter: set[str] | None = None,
    limit: int | None = None,
) -> Iterator[MeaningCentricStar]:
    """Iterate over meaning stars built from OMW synsets.

    Args:
        omw_path: Path to OMW wns/ directory.
        min_languages: Only emit synsets with at least N language surface_forms.
        pos_filter: If set, only include synsets matching these POS tags (n, v, a, r).
        limit: Maximum number of stars to yield.
    """
    synsets = load_all_omw(omw_path)
    count = 0

    for synset_id in sorted(synsets):
        if limit is not None and count >= limit:
            return

        entry = synsets[synset_id]

        # Filter by language coverage
        if len(entry.lemmas) < min_languages:
            continue

        # Filter by POS
        if pos_filter and entry.pos not in pos_filter:
            continue

        yield synset_to_star(entry)
        count += 1


def build_meaning_layer_stars(
    omw_path: Path | None = None,
    *,
    min_languages: int = 3,
    limit: int | None = None,
) -> list[MeaningCentricStar]:
    """Build the complete meaning layer from OMW.

    Default: synsets with 3+ languages, proving multilingual alignment.
    """
    return list(iter_meaning_stars(
        omw_path,
        min_languages=min_languages,
        limit=limit,
    ))


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def meaning_layer_stats(stars: list[MeaningCentricStar]) -> dict[str, Any]:
    """Compute statistics about the meaning layer."""
    if not stars:
        return {"total": 0}

    lang_counts: dict[str, int] = {}
    pos_counts: dict[str, int] = {}
    total_surface_forms = 0

    for star in stars:
        total_surface_forms += len(star.surface_forms)
        for lang in star.surface_forms:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        pos_counts[star.meaning_class] = pos_counts.get(star.meaning_class, 0) + 1

    avg_langs = total_surface_forms / len(stars)

    return {
        "total_stars": len(stars),
        "total_surface_forms": total_surface_forms,
        "avg_languages_per_star": round(avg_langs, 1),
        "languages_covered": len(lang_counts),
        "top_languages": dict(sorted(lang_counts.items(), key=lambda x: -x[1])[:10]),
        "pos_distribution": pos_counts,
    }


__all__ = [
    "OMW_DEFAULT_PATH",
    "OMW_LANG_MAP",
    "POS_MAP",
    "SynsetEntry",
    "build_meaning_layer_stars",
    "iter_meaning_stars",
    "load_all_omw",
    "meaning_layer_stats",
    "parse_omw_tab",
    "synset_to_star",
]
```

---

## Update: `knowledge3d/ingestion/universal_knowledge/__init__.py`

Add imports and re-exports:

```python
from .multilingual_meanings import (
    OMW_DEFAULT_PATH,
    OMW_LANG_MAP,
    SynsetEntry,
    build_meaning_layer_stars,
    iter_meaning_stars,
    load_all_omw,
    meaning_layer_stats,
    parse_omw_tab,
    synset_to_star,
)
```

Add to `__all__`:
```python
"OMW_DEFAULT_PATH",
"OMW_LANG_MAP",
"SynsetEntry",
"build_meaning_layer_stars",
"iter_meaning_stars",
"load_all_omw",
"meaning_layer_stats",
"parse_omw_tab",
"synset_to_star",
```

---

## Expected Star Output

For synset `00001740-a` (able/capable):

```python
MeaningCentricStar(
    star_id="synset_00001740_a",
    meaning_class="adjective",
    meaning_rpn="SYNSET A ABLE DEF having the necessary means or skill or know-how or authority to do",
    domain="Foundation/Language",
    taxonomy_refs=["concept_language", "wordnet_synset", "concept_adjective"],
    surface_forms={
        "en": SurfaceForm(word_ref="en_able", char_refs=["char_a", "char_b", "char_l", "char_e"]),
        "pt": SurfaceForm(word_ref="pt_capaz", char_refs=["char_c", "char_a", "char_p", "char_a", "char_z"]),
        "fr": SurfaceForm(word_ref="fr_comptable", char_refs=[...]),
        "ja": SurfaceForm(word_ref="ja_53ef_80fd", char_refs=["char_ja_u53ef", "char_ja_u80fd"]),
        "ar": SurfaceForm(word_ref="ar_قادر", char_refs=["char_ar_u0642", ...]),
        "el": SurfaceForm(word_ref="el_ικανός", char_refs=[...]),
        "fi": SurfaceForm(word_ref="fi_kykenevä", char_refs=[...]),
        "it": SurfaceForm(word_ref="it_abile", char_refs=[...]),
    },
    meta_refs=[
        "wordnet:00001740-a",
        "languages:8",
        "synonym:fi:pystyvä",
        "synonym:fi:taitava",
        "synonym:it:intelligente",
        "synonym:it:valente",
        "synonym:it:capace",
    ],
    house_room="House/Library",
    confidence=1,
    polarity=1,
)
```

**This is the proof:** ONE star, EIGHT languages, all pointing to the SAME meaning. The char_refs in each surface_form trace all the way down to Layer 1 glyphs. The word_ref symlinks to the Word Galaxy entry. The star IS the meaning — language-agnostic.

---

## Tests

### test_multilingual_meanings.py

```python
"""Tests for multilingual meaning layer construction from OMW."""

from pathlib import Path

import pytest

OMW_PATH = Path("/K3D/K3D_llama_cpp/datasets/omw-data/omw-data-main/wns")
HAS_OMW = OMW_PATH.exists()


@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_parse_omw_tab_english():
    """Parse English tab file."""
    from knowledge3d.ingestion.universal_knowledge.multilingual_meanings import parse_omw_tab
    synsets = parse_omw_tab(OMW_PATH / "eng" / "wn-data-eng.tab", "en")
    assert len(synsets) > 1000
    # Check first synset
    entry = synsets.get("00001740-a")
    assert entry is not None
    assert "en" in entry.lemmas
    assert "able" in entry.lemmas["en"]


@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_parse_omw_tab_portuguese():
    """Parse Portuguese tab file."""
    from knowledge3d.ingestion.universal_knowledge.multilingual_meanings import parse_omw_tab
    synsets = parse_omw_tab(OMW_PATH / "por" / "wn-data-por.tab", "pt")
    assert len(synsets) > 1000
    entry = synsets.get("00001740-a")
    assert entry is not None
    assert "pt" in entry.lemmas
    assert "capaz" in entry.lemmas["pt"]


@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_load_all_omw_merges_languages():
    """Loading all OMW tab files merges lemmas by synset."""
    from knowledge3d.ingestion.universal_knowledge.multilingual_meanings import load_all_omw
    synsets = load_all_omw(OMW_PATH)
    assert len(synsets) > 10000
    # The "able" synset should have multiple languages
    entry = synsets.get("00001740-a")
    assert entry is not None
    assert len(entry.lemmas) >= 3  # At least en, pt, fr
    assert "en" in entry.lemmas
    assert "pt" in entry.lemmas


@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_synset_to_star_structure():
    """Synset converts to a valid MeaningCentricStar."""
    from knowledge3d.ingestion.universal_knowledge.multilingual_meanings import (
        load_all_omw, synset_to_star,
    )
    synsets = load_all_omw(OMW_PATH)
    entry = synsets["00001740-a"]
    star = synset_to_star(entry)

    assert star.star_id == "synset_00001740_a"
    assert star.meaning_class == "adjective"
    assert "en" in star.surface_forms
    assert star.surface_forms["en"].word_ref == "en_able"
    assert "char_a" in star.surface_forms["en"].char_refs
    assert "wordnet:00001740-a" in star.meta_refs
    assert "concept_language" in star.taxonomy_refs
    assert star.house_room == "House/Library"
    assert star.domain == "Foundation/Language"


@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_synset_star_multilingual_surface_forms():
    """Star has surface_forms in multiple languages with char_refs."""
    from knowledge3d.ingestion.universal_knowledge.multilingual_meanings import (
        load_all_omw, synset_to_star,
    )
    synsets = load_all_omw(OMW_PATH)
    entry = synsets["00001740-a"]
    star = synset_to_star(entry)

    # Must have at least en + pt + one more
    assert len(star.surface_forms) >= 3
    # Portuguese should be "capaz"
    if "pt" in star.surface_forms:
        assert star.surface_forms["pt"].word_ref == "pt_capaz"
    # Each surface_form must have char_refs (Layer 1 symlinks)
    for lang, sf in star.surface_forms.items():
        assert len(sf.char_refs) > 0, f"Missing char_refs for {lang}"


@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_iter_meaning_stars_min_languages():
    """Filtering by min_languages works."""
    from knowledge3d.ingestion.universal_knowledge.multilingual_meanings import iter_meaning_stars
    stars = list(iter_meaning_stars(OMW_PATH, min_languages=5, limit=20))
    assert len(stars) > 0
    for star in stars:
        assert len(star.surface_forms) >= 5


@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_iter_meaning_stars_pos_filter():
    """POS filtering returns only requested types."""
    from knowledge3d.ingestion.universal_knowledge.multilingual_meanings import iter_meaning_stars
    nouns = list(iter_meaning_stars(OMW_PATH, min_languages=2, pos_filter={"n"}, limit=20))
    assert all(s.meaning_class == "noun" for s in nouns)

    verbs = list(iter_meaning_stars(OMW_PATH, min_languages=2, pos_filter={"v"}, limit=20))
    assert all(s.meaning_class == "verb" for s in verbs)


@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_build_meaning_layer_stars():
    """Build complete meaning layer (limited)."""
    from knowledge3d.ingestion.universal_knowledge.multilingual_meanings import (
        build_meaning_layer_stars, meaning_layer_stats,
    )
    stars = build_meaning_layer_stars(OMW_PATH, min_languages=3, limit=100)
    assert len(stars) == 100

    stats = meaning_layer_stats(stars)
    assert stats["total_stars"] == 100
    assert stats["avg_languages_per_star"] >= 3.0
    assert stats["languages_covered"] >= 3


def test_pos_map():
    from knowledge3d.ingestion.universal_knowledge.multilingual_meanings import POS_MAP
    assert POS_MAP["n"] == "noun"
    assert POS_MAP["v"] == "verb"
    assert POS_MAP["a"] == "adjective"
    assert POS_MAP["r"] == "adverb"


def test_omw_lang_map_has_key_languages():
    from knowledge3d.ingestion.universal_knowledge.multilingual_meanings import OMW_LANG_MAP
    assert OMW_LANG_MAP["eng"] == "en"
    assert OMW_LANG_MAP["por"] == "pt"
    assert OMW_LANG_MAP["fra"] == "fr"
    assert OMW_LANG_MAP["jpn"] == "ja"
    assert OMW_LANG_MAP["arb"] == "ar"
```

---

## Expected Scale

| Filter | Stars | Avg Languages | Notes |
|--------|-------|---------------|-------|
| min_languages=2 | ~80,000-100,000 | ~4 | Most synsets shared by 2+ languages |
| min_languages=3 | ~50,000-70,000 | ~5 | Solid multilingual proof |
| min_languages=5 | ~20,000-30,000 | ~7 | Core vocabulary (most useful) |
| min_languages=10 | ~5,000-10,000 | ~12 | Universal concepts |

**Primary target:** `min_languages=3` for the initial meaning layer. This gives us ~50K+ meaning stars where we can PROVE that "water" = "água" = "eau" = "水" = "вода" in one unified star.

---

## File Changes Summary

| File | Action |
|------|--------|
| `knowledge3d/ingestion/universal_knowledge/multilingual_meanings.py` | **NEW** — OMW parser, synset→star converter, batch builder |
| `knowledge3d/ingestion/universal_knowledge/__init__.py` | **MODIFY** — Add multilingual imports and __all__ entries |
| `tests/test_multilingual_meanings.py` | **NEW** — 10 tests |

---

## Success Criteria

1. `parse_omw_tab()` correctly parses English tab file (206K+ lemmas)
2. `load_all_omw()` merges 31 language sources into unified synset dict
3. `synset_to_star()` produces valid MeaningCentricStar with multi-language surface_forms
4. Each surface_form has `char_refs` pointing to Layer 1 glyphs
5. `build_meaning_layer_stars(min_languages=3, limit=100)` produces 100 stars with 3+ languages each
6. Stats show proper language distribution
7. All 10 tests pass
8. Non-regression on existing tests

---

## What This Proves

**Semantic Gravity Cohered by Meaning:** In Galaxy working memory, these stars attract each other by MEANING, not by language. "water" (en) and "água" (pt) orbit the SAME meaning star — they share a gravitational center. This is the ternary force `F = T(s₁,s₂) × M(s₁) × M(s₂) / d²` in action.

**English-Primary, Language-Agnostic:** The meaning is defined in English (W3C standard). Other languages are surface_form symlinks — they don't carry semantic weight, they're access paths. The TRM reasons on the English meaning_rpn; it renders the Portuguese surface_form when talking to a Portuguese speaker. Same meaning, different rendering.

**Culture-Specific Exception:** When a concept has NO English word (saudade, schadenfreude, 木漏れ日), the meaning text stays in the source language. This is rare but important — it means the Galaxy can represent concepts that English cannot, without forcing lossy translation.

**Symlink Architecture:** No duplicated content. The word "able" is stored ONCE in Word Galaxy (en). The word "capaz" is stored ONCE in Word Galaxy (pt). The meaning star `synset_00001740_a` doesn't contain either word — it REFERENCES them through surface_form symlinks. Change the word in one language, the meaning star still works.

**Layer Stack:** Layer 1 (chars) → Layer 2 (words via symlinks) → Layer 2 (meaning stars referencing words) → Layer 3 (grammar rules that operate on meaning stars). This is the bottom-up architecture others lack.

---

## After H19: Ollama Enrichment (Phase B3 revisited)

Once the meaning layer exists, THEN we use Ollama to ENRICH stars:
1. Generate better `meaning_rpn` from definitions (currently basic string concatenation)
2. Add `grammar_rules` extracted from example sentences
3. Fill in missing language surface_forms (Ollama can translate)
4. Link synsets to domain-specific stars (element_h, constant_gravitational, etc.)

But the FOUNDATION comes first — no LLM needed to prove that synset = meaning = multilingual.
