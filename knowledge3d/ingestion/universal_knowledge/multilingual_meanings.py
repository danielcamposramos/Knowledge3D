"""Build meaning-centric stars from Open Multilingual Wordnet synsets.

Each star represents ONE meaning (synset) with surface_forms from all available
languages. In Galaxy working memory, semantic gravity operates BETWEEN different
meaning stars based on proximity of their meanings, for example "water" and
"liquid" attract each other. Language is irrelevant to the force; only meaning
distance matters. Surface forms within a star, such as "water"(en) and
"agua"(pt), are not separate gravitational bodies. They are symlink references
inside one multilingual star.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
from typing import Iterator

from knowledge3d.ingestion.canonical_lookup import canonical_char_star_id, canonical_word_star_id
from knowledge3d.ingestion.symlink_helpers import link
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar, SurfaceForm


OMW_DEFAULT_PATH = Path("/K3D/K3D_llama_cpp/datasets/omw-data/omw-data-main/wns")

OMW_LANG_MAP: dict[str, str] = {
    "eng": "en",
    "por": "pt",
    "fra": "fr",
    "jpn": "ja",
    "arb": "ar",
    "ita": "it",
    "dan": "da",
    "ell": "el",
    "fin": "fi",
    "heb": "he",
    "hrv": "hr",
    "isl": "is",
    "nld": "nl",
    "nor": "no",
    "pol": "pl",
    "ron": "ro",
    "slk": "sk",
    "slv": "sl",
    "swe": "sv",
    "tha": "th",
    "bul": "bg",
    "fas": "fa",
    "msa": "ms",
    "als": "sq",
    "cwn": "zh",
    "iwn": "id",
    "mcr": "es",
    "cow": "zh",
    "wikt": "mul",
    "cldr": "mul",
}

POS_MAP: dict[str, str] = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "r": "adverb",
}

_SYNSET_RE = re.compile(r"^\d{8}-[nvar]$")


@dataclass
class SynsetEntry:
    """One synset with lemmas merged from all languages."""

    synset_id: str
    pos: str = ""
    lemmas: dict[str, list[str]] = field(default_factory=dict)
    definitions: dict[str, str] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)


def _normalize_field_language(field_name: str, fallback_lang: str) -> tuple[str, str]:
    raw = str(field_name or "").strip()
    if ":" not in raw:
        return fallback_lang, raw
    lang_key, field_type = raw.rsplit(":", 1)
    mapped = OMW_LANG_MAP.get(lang_key.strip(), lang_key.strip().lower())
    if mapped == "mul":
        mapped = fallback_lang
    return mapped, field_type.strip()


def _append_unique(target: dict[str, list[str]], key: str, value: str) -> None:
    cleaned = str(value or "").strip()
    if not cleaned:
        return
    bucket = target.setdefault(key, [])
    if cleaned not in bucket:
        bucket.append(cleaned)


def _lemma_word_ref(language: str, lemma: str) -> str:
    return canonical_word_star_id(language, lemma)


def _char_refs(text: str) -> list[str]:
    refs: list[str] = []
    for char in str(text or ""):
        if char.isspace():
            continue
        refs.append(canonical_char_star_id(char))
    return refs


def _word_meaning_rpn(language: str, lemma: str, sense_anchor: str) -> str:
    token = _safe_rpn_token(lemma)
    return f"WORD LANG_{str(language).upper()} LEMMA {token} STORE_surface RECALL {sense_anchor}"


def _char_meaning_rpn(char: str) -> str:
    return f"CHAR U{ord(char):04X} GLYPH"


def _word_star(language: str, lemma: str, meaning_star_id: str) -> MeaningCentricStar:
    word_id = canonical_word_star_id(language, lemma)
    refs = _char_refs(lemma)
    return MeaningCentricStar(
        star_id=word_id,
        meaning_class="concept",
        meaning_rpn=_word_meaning_rpn(language, lemma, meaning_star_id),
        domain=f"Word/{language}",
        surface_forms={
            language: SurfaceForm(word_ref=word_id, char_refs=refs),
        },
        lod_class="lod_summary",
        confidence=1,
        polarity=1,
    )


def _char_star(char: str) -> MeaningCentricStar:
    return MeaningCentricStar(
        star_id=canonical_char_star_id(char),
        meaning_class="form",
        meaning_rpn=_char_meaning_rpn(char),
        domain="Character/Unicode",
        lod_class="lod_icon",
        confidence=1,
        polarity=1,
    )


def _dedupe_stars(stars: list[MeaningCentricStar]) -> list[MeaningCentricStar]:
    merged: dict[str, MeaningCentricStar] = {}
    for star in stars:
        existing = merged.get(star.star_id)
        if existing is None:
            merged[star.star_id] = star
            continue
        for ref in star.taxonomy_refs:
            if ref not in existing.taxonomy_refs:
                existing.taxonomy_refs.append(ref)
        for ref in star.meta_refs:
            if ref not in existing.meta_refs:
                existing.meta_refs.append(ref)
        for ref in star.component_refs:
            if ref not in existing.component_refs:
                existing.component_refs.append(ref)
        for ref in star.composite_of:
            if ref not in existing.composite_of:
                existing.composite_of.append(ref)
    return list(merged.values())


def parse_omw_tab(filepath: Path, lang_code: str) -> dict[str, SynsetEntry]:
    """Parse one Wordnet tab file into synset entries."""
    synsets: dict[str, SynsetEntry] = {}
    with Path(filepath).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            synset_id = parts[0].strip()
            if not _SYNSET_RE.fullmatch(synset_id):
                continue
            field_lang, field_type = _normalize_field_language(parts[1], lang_code)
            value = parts[2].strip() if len(parts) > 2 else ""
            entry = synsets.setdefault(
                synset_id,
                SynsetEntry(
                    synset_id=synset_id,
                    pos=synset_id.split("-")[-1],
                ),
            )
            if field_type == "lemma":
                _append_unique(entry.lemmas, field_lang, value)
            elif field_type == "def":
                definition = parts[3].strip() if len(parts) > 3 else value
                if definition and field_lang not in entry.definitions:
                    entry.definitions[field_lang] = definition
            elif field_type == "exe":
                example = parts[3].strip() if len(parts) > 3 else value
                _append_unique(entry.examples, field_lang, example)
    return synsets


@lru_cache(maxsize=4)
def _load_all_omw_cached(omw_path_str: str) -> dict[str, SynsetEntry]:
    omw_path = Path(omw_path_str)
    merged: dict[str, SynsetEntry] = {}
    if not omw_path.exists():
        return merged

    for lang_dir in sorted(omw_path.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang_key = lang_dir.name.strip()
        lang_code = OMW_LANG_MAP.get(lang_key)
        if not lang_code or lang_code == "mul":
            continue
        tab_file = lang_dir / f"wn-data-{lang_key}.tab"
        if not tab_file.exists():
            continue
        parsed = parse_omw_tab(tab_file, lang_code)
        for synset_id, entry in parsed.items():
            target = merged.setdefault(
                synset_id,
                SynsetEntry(
                    synset_id=synset_id,
                    pos=entry.pos,
                ),
            )
            for language, lemmas in entry.lemmas.items():
                for lemma in lemmas:
                    _append_unique(target.lemmas, language, lemma)
            for language, definition in entry.definitions.items():
                if definition and language not in target.definitions:
                    target.definitions[language] = definition
            for language, examples in entry.examples.items():
                for example in examples:
                    _append_unique(target.examples, language, example)
    return merged


def load_all_omw(omw_path: Path | None = None) -> dict[str, SynsetEntry]:
    """Load and merge all available OMW language sources."""
    return _load_all_omw_cached(str((omw_path or OMW_DEFAULT_PATH).resolve()))


def _safe_rpn_token(text: str) -> str:
    cleaned = re.sub(r"\s+", "_", str(text or "").strip().upper())
    return cleaned[:80] if cleaned else "UNKNOWN"


def synset_to_star(entry: SynsetEntry) -> MeaningCentricStar:
    """Convert one synset into a multilingual meaning-centric star.

    One synset = one meaning = one star. All languages are surface_forms inside
    the star. Semantic gravity clusters different but related meaning stars
    together in Galaxy working memory, for example "water" near "liquid" near
    "ice".
    """
    star_id = f"synset_{entry.synset_id.replace('-', '_')}"
    meaning_class = POS_MAP.get(entry.pos, "concept")

    english_lemmas = list(entry.lemmas.get("en", []))
    english_definition = str(entry.definitions.get("en", "")).strip()
    english_primary = english_lemmas[0] if english_lemmas else ""
    if english_primary and english_definition:
        meaning_rpn = f"SYNSET {entry.pos.upper()} {_safe_rpn_token(english_primary)} DEF {english_definition[:80]}"
    elif english_primary:
        meaning_rpn = f"SYNSET {entry.pos.upper()} {_safe_rpn_token(english_primary)}"
    else:
        fallback_lang = ""
        fallback_lemma = ""
        fallback_definition = ""
        for language in sorted(entry.lemmas.keys()):
            lemmas = entry.lemmas.get(language, [])
            if lemmas:
                fallback_lang = language
                fallback_lemma = lemmas[0]
                fallback_definition = str(entry.definitions.get(language, "")).strip()
                break
        if fallback_lemma and fallback_definition:
            meaning_rpn = (
                f"SYNSET {entry.pos.upper()} LANG_{fallback_lang.upper()} "
                f"{_safe_rpn_token(fallback_lemma)} DEF {fallback_definition[:80]}"
            )
        elif fallback_lemma:
            meaning_rpn = f"SYNSET {entry.pos.upper()} LANG_{fallback_lang.upper()} {_safe_rpn_token(fallback_lemma)}"
        else:
            meaning_rpn = f"SYNSET {entry.pos.upper()} {entry.synset_id}"

    surface_forms: dict[str, SurfaceForm] = {}
    synonym_refs: list[str] = []
    for language in sorted(entry.lemmas.keys()):
        lemmas = list(entry.lemmas.get(language, []))
        if not lemmas:
            continue
        primary = lemmas[0]
        surface_forms[language] = SurfaceForm(
            word_ref=_lemma_word_ref(language, primary),
            char_refs=_char_refs(primary),
        )
        for synonym in lemmas[1:]:
            synonym_refs.append(f"synonym:{language}:{synonym}")

    taxonomy_refs = ["concept_language", "wordnet_synset", f"concept_{meaning_class}"]
    meta_refs = [
        f"wordnet:{entry.synset_id}",
        f"languages:{len(surface_forms)}",
    ]
    meta_refs.extend(synonym_refs[:20])

    return MeaningCentricStar(
        star_id=star_id,
        meaning_class=meaning_class,
        meaning_rpn=meaning_rpn,
        domain="Foundation/Language",
        taxonomy_refs=taxonomy_refs,
        surface_forms=surface_forms,
        meta_refs=meta_refs,
        house_room="House/Library",
        lod_class="lod_summary",
        confidence=1,
        polarity=1,
    )


def synset_to_star_bundle(entry: SynsetEntry) -> list[MeaningCentricStar]:
    meaning_star = synset_to_star(entry)
    stars: list[MeaningCentricStar] = [meaning_star]
    chars_by_id: dict[str, MeaningCentricStar] = {}
    for language in sorted(entry.lemmas.keys()):
        lemmas = list(entry.lemmas.get(language, []))
        if not lemmas:
            continue
        primary = lemmas[0]
        word_star = _word_star(language, primary, meaning_star.star_id)
        link(meaning_star, word_star, f"surface_forms.{language}.word_ref", "taxonomy_refs")
        for char in str(primary):
            if char.isspace():
                continue
            char_star = chars_by_id.get(canonical_char_star_id(char))
            if char_star is None:
                char_star = _char_star(char)
                chars_by_id[char_star.star_id] = char_star
                stars.append(char_star)
            link(word_star, char_star, "component_refs", "composite_of")
        stars.append(word_star)
    return _dedupe_stars(stars)


def iter_meaning_stars(
    omw_path: Path | None = None,
    *,
    min_languages: int = 2,
    pos_filter: set[str] | None = None,
    limit: int | None = None,
) -> Iterator[MeaningCentricStar]:
    """Yield deterministic meaning stars filtered by language coverage and POS."""
    allowed_pos = {str(pos).strip().lower() for pos in set(pos_filter or set()) if str(pos).strip()} or None
    emitted = 0
    synsets = load_all_omw(omw_path)
    for synset_id in sorted(synsets.keys()):
        entry = synsets[synset_id]
        if len(entry.lemmas) < int(min_languages):
            continue
        if allowed_pos is not None and entry.pos.lower() not in allowed_pos:
            continue
        yield synset_to_star(entry)
        emitted += 1
        if limit is not None and emitted >= int(limit):
            break


def build_meaning_layer_stars(
    omw_path: Path | None = None,
    *,
    min_languages: int = 3,
    limit: int | None = None,
) -> list[MeaningCentricStar]:
    """Return a list of multilingual meaning stars.

    These stars form the meaning layer where semantic gravity operates between
    concepts by meaning proximity, not by language. Each star is already
    multilingual, so gravity acts between stars, not within them.
    """
    return list(iter_meaning_stars(omw_path, min_languages=min_languages, limit=limit))


def build_meaning_layer_bundle(
    omw_path: Path | None = None,
    *,
    min_languages: int = 3,
    pos_filter: set[str] | None = None,
    limit: int | None = None,
) -> list[MeaningCentricStar]:
    allowed_pos = {str(pos).strip().lower() for pos in set(pos_filter or set()) if str(pos).strip()} or None
    bundled: list[MeaningCentricStar] = []
    meaning_count = 0
    synsets = load_all_omw(omw_path)
    for synset_id in sorted(synsets.keys()):
        entry = synsets[synset_id]
        if len(entry.lemmas) < int(min_languages):
            continue
        if allowed_pos is not None and entry.pos.lower() not in allowed_pos:
            continue
        bundled.extend(synset_to_star_bundle(entry))
        meaning_count += 1
        if limit is not None and meaning_count >= int(limit):
            break
    return _dedupe_stars(bundled)


def meaning_layer_stats(stars: list[MeaningCentricStar]) -> dict[str, object]:
    """Summarize the produced meaning layer."""
    total_stars = len(stars)
    total_surface_forms = sum(len(star.surface_forms) for star in stars)
    language_counter: Counter[str] = Counter()
    pos_counter: Counter[str] = Counter()
    for star in stars:
        language_counter.update(star.surface_forms.keys())
        pos_counter.update([star.meaning_class])
    avg_languages = (total_surface_forms / total_stars) if total_stars else 0.0
    return {
        "total_stars": total_stars,
        "total_surface_forms": total_surface_forms,
        "avg_languages_per_star": round(avg_languages, 3),
        "languages_covered": sorted(language_counter.keys()),
        "top_languages": language_counter.most_common(10),
        "pos_distribution": dict(sorted(pos_counter.items())),
    }


__all__ = [
    "OMW_DEFAULT_PATH",
    "OMW_LANG_MAP",
    "POS_MAP",
    "SynsetEntry",
    "build_meaning_layer_stars",
    "build_meaning_layer_bundle",
    "iter_meaning_stars",
    "load_all_omw",
    "meaning_layer_stats",
    "parse_omw_tab",
    "synset_to_star",
    "synset_to_star_bundle",
]
