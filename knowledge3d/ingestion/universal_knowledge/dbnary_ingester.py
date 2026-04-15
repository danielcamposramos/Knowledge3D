"""DBnary OntoLex ingestion for the Phase 7 Word Galaxy slice.

DBnary is an ingestion source only. This module turns one OntoLex TTL dump at a
time into canonical word stars, then merges those records against the OMW synset
index so OMW meanings remain the primary shelf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import bz2
import re
from typing import Iterable, Iterator, Mapping

from knowledge3d.ingestion.canonical_lookup import canonical_char_star_id, canonical_word_star_id
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar, SurfaceForm

from .multilingual_meanings import SynsetEntry


DBNARY_DEFAULT_PATH = Path("/K3D/K3D_llama_cpp/datasets/dbnary")
POS_NORMALIZATION = {
    "noun": "noun",
    "proper_noun": "noun",
    "propernoun": "noun",
    "verb": "verb",
    "adjective": "adjective",
    "adj": "adjective",
    "adverb": "adverb",
    "adv": "adverb",
}

_LABEL_RE = re.compile(r'rdfs:label\s+"((?:[^"\\]|\\.)*)"@([A-Za-z0-9_-]+)')
_WRITTEN_RE = re.compile(r'ontolex:writtenRep\s+"((?:[^"\\]|\\.)*)"@([A-Za-z0-9_-]+)')
_LANG_RE = re.compile(r'lime:language\s+"([^"]+)"')
_POS_LITERAL_RE = re.compile(r'dbnary:partOfSpeech\s+"([^"]+)"')
_POS_LEXINFO_RE = re.compile(r"lexinfo:partOfSpeech\s+lexinfo:([A-Za-z_]+)")
_PHONETIC_RE = re.compile(r'ontolex:phoneticRep\s+"((?:[^"\\]|\\.)*)"@')
_DEFINITION_RE = re.compile(r'rdf:value\s+"((?:[^"\\]|\\.)*)"@([A-Za-z0-9_-]+)')


@dataclass(frozen=True)
class LexicalRecord:
    """One external lexical entry normalized into K3D canonical surfaces."""

    source: str
    language: str
    lemma: str
    pos: str = ""
    definitions: tuple[str, ...] = ()
    pronunciations: tuple[str, ...] = ()
    etymology: str = ""


@dataclass
class WordMergeResult:
    """Summary of external lexical records merged against OMW."""

    processed_count: int = 0
    merged_count: int = 0
    new_word_stars: dict[str, MeaningCentricStar] = field(default_factory=dict)
    merged_synset_ids: dict[str, list[str]] = field(default_factory=dict)
    skipped_count: int = 0


def _clean_literal(value: str) -> str:
    raw = str(value)
    if "\\" not in raw:
        return raw.strip()
    return (
        raw.replace(r"\"", '"')
        .replace(r"\\", "\\")
        .replace(r"\n", "\n")
        .replace(r"\t", "\t")
        .strip()
    )


def _normalize_language(language: str) -> str:
    raw = str(language or "").strip().lower()
    if not raw:
        return ""
    return raw.split("-", 1)[0]


def normalize_lemma(lemma: str) -> str:
    return " ".join(str(lemma or "").strip().lower().split())


def normalize_pos(pos: str) -> str:
    raw = str(pos or "").strip().lower().replace(" ", "_")
    return POS_NORMALIZATION.get(raw, raw)


def _open_text(path: Path):
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _iter_ttl_blocks(path: Path) -> Iterator[str]:
    buffer: list[str] = []
    with _open_text(Path(path)) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line.strip():
                if buffer:
                    yield "\n".join(buffer)
                    buffer = []
                continue
            if line.startswith("@prefix"):
                continue
            buffer.append(line)
            if line.rstrip().endswith(" ."):
                yield "\n".join(buffer)
                buffer = []
    if buffer:
        yield "\n".join(buffer)


def _first_match(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    if not match:
        return ""
    return _clean_literal(match.group(1))


def _record_from_block(block: str, default_language: str) -> LexicalRecord | None:
    if "ontolex:LexicalEntry" not in block:
        return None
    lemma = _first_match(_LABEL_RE, block) or _first_match(_WRITTEN_RE, block)
    if not lemma:
        return None
    language = _normalize_language(_first_match(_LANG_RE, block) or default_language)
    if not language:
        language_match = _LABEL_RE.search(block) or _WRITTEN_RE.search(block)
        if language_match:
            language = _normalize_language(language_match.group(2))
    if not language:
        return None
    pos = normalize_pos(_first_match(_POS_LITERAL_RE, block) or _first_match(_POS_LEXINFO_RE, block))
    pronunciations = tuple(dict.fromkeys(_clean_literal(match.group(1)) for match in _PHONETIC_RE.finditer(block)))
    definitions = tuple(dict.fromkeys(_clean_literal(match.group(1)) for match in _DEFINITION_RE.finditer(block)))
    return LexicalRecord(
        source="dbnary",
        language=language,
        lemma=lemma,
        pos=pos,
        definitions=definitions,
        pronunciations=pronunciations,
    )


def iter_dbnary_records(
    path: str | Path,
    *,
    language: str | None = None,
    limit: int | None = None,
) -> Iterator[LexicalRecord]:
    """Yield lexical records from one DBnary OntoLex TTL dump."""

    source_path = Path(path)
    default_language = _normalize_language(language or source_path.name.split("_", 1)[0])
    emitted = 0
    seen: set[tuple[str, str, str]] = set()
    for block in _iter_ttl_blocks(source_path):
        record = _record_from_block(block, default_language)
        if record is None:
            continue
        key = (record.language, normalize_lemma(record.lemma), record.pos)
        if key in seen:
            continue
        seen.add(key)
        yield record
        emitted += 1
        if limit is not None and emitted >= int(limit):
            break


def build_omw_lemma_index(synsets: Mapping[str, SynsetEntry]) -> dict[tuple[str, str], list[str]]:
    """Index OMW synsets by canonical language + lemma surface."""

    index: dict[tuple[str, str], list[str]] = {}
    for synset_id, entry in synsets.items():
        for language, lemmas in entry.lemmas.items():
            for lemma in lemmas:
                key = (_normalize_language(language), normalize_lemma(lemma))
                if not key[0] or not key[1]:
                    continue
                bucket = index.setdefault(key, [])
                if synset_id not in bucket:
                    bucket.append(synset_id)
    return index


def _word_char_refs(lemma: str) -> list[str]:
    refs: list[str] = []
    for char in str(lemma or ""):
        if char.isspace():
            continue
        refs.append(canonical_char_star_id(char))
    return refs


def lexical_record_to_word_star(record: LexicalRecord) -> MeaningCentricStar:
    """Create a canonical standalone word star for a lemma absent from OMW."""

    word_id = canonical_word_star_id(record.language, record.lemma)
    char_refs = _word_char_refs(record.lemma)
    meta_refs = [f"source:{record.source}"]
    if record.pos:
        meta_refs.append(f"pos:{record.pos}")
    if record.etymology:
        meta_refs.append(f"etymology:{record.etymology[:160]}")
    return MeaningCentricStar(
        star_id=word_id,
        meaning_class="concept",
        meaning_rpn=f"WORD LANG_{record.language.upper()} LEMMA {normalize_lemma(record.lemma).upper().replace(' ', '_')} STORE",
        domain=f"Word/{record.language}",
        surface_forms={
            record.language: SurfaceForm(word_ref=word_id, char_refs=char_refs),
        },
        pronunciations={record.language: record.pronunciations[0]} if record.pronunciations else {},
        meta_refs=meta_refs,
        component_refs=char_refs,
        lod_class="LOD_SUMMARY",
        confidence=1,
        polarity=1,
    )


def merge_lexical_records_into_omw(
    synsets: Mapping[str, SynsetEntry],
    records: Iterable[LexicalRecord],
) -> WordMergeResult:
    """Merge external records into OMW by lemma; create words only for misses."""

    index = build_omw_lemma_index(synsets)
    result = WordMergeResult()
    for record in records:
        result.processed_count += 1
        lemma_key = (_normalize_language(record.language), normalize_lemma(record.lemma))
        if not lemma_key[0] or not lemma_key[1]:
            result.skipped_count += 1
            continue
        matches = index.get(lemma_key, [])
        if matches:
            result.merged_count += 1
            result.merged_synset_ids[canonical_word_star_id(record.language, record.lemma)] = list(matches)
            for synset_id in matches:
                entry = synsets.get(synset_id)
                if entry and record.definitions and record.language not in entry.definitions:
                    entry.definitions[record.language] = record.definitions[0]
            continue
        word_star = lexical_record_to_word_star(record)
        existing = result.new_word_stars.get(word_star.star_id)
        if existing is None:
            result.new_word_stars[word_star.star_id] = word_star
            continue
        for ref in word_star.component_refs:
            if ref not in existing.component_refs:
                existing.component_refs.append(ref)
    return result


__all__ = [
    "DBNARY_DEFAULT_PATH",
    "LexicalRecord",
    "WordMergeResult",
    "build_omw_lemma_index",
    "iter_dbnary_records",
    "lexical_record_to_word_star",
    "merge_lexical_records_into_omw",
    "normalize_lemma",
    "normalize_pos",
]
