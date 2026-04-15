"""Kaikki.org JSONL ingestion for tertiary Word Galaxy enrichment."""

from __future__ import annotations

from pathlib import Path
import gzip
import json
from typing import Iterator

from .dbnary_ingester import LexicalRecord, normalize_pos


KAIKKI_DEFAULT_PATHS = (
    Path("/K3D/K3D_llama_cpp/datasets/lexicons/portuguese_br/kaikki.org-dictionary-Portuguese.jsonl.gz"),
    Path("/K3D/K3D_llama_cpp/datasets/lexicons/spanish/kaikki.org-dictionary-Spanish.jsonl"),
)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _sense_glosses(raw_senses: object) -> tuple[str, ...]:
    glosses: list[str] = []
    if not isinstance(raw_senses, list):
        return ()
    for sense in raw_senses:
        if not isinstance(sense, dict):
            continue
        for gloss in sense.get("glosses") or []:
            cleaned = str(gloss or "").strip()
            if cleaned and cleaned not in glosses:
                glosses.append(cleaned)
    return tuple(glosses)


def _ipa_values(raw_sounds: object) -> tuple[str, ...]:
    values: list[str] = []
    if not isinstance(raw_sounds, list):
        return ()
    for sound in raw_sounds:
        if not isinstance(sound, dict):
            continue
        ipa = str(sound.get("ipa") or "").strip()
        if ipa and ipa not in values:
            values.append(ipa)
    return tuple(values)


def iter_kaikki_records(
    path: str | Path,
    *,
    language: str | None = None,
    limit: int | None = None,
) -> Iterator[LexicalRecord]:
    """Yield lexical records from one Kaikki JSONL or JSONL.GZ file."""

    source_path = Path(path)
    emitted = 0
    seen: set[tuple[str, str, str]] = set()
    with _open_text(source_path) as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            lemma = str(payload.get("word") or "").strip()
            lang = str(language or payload.get("lang_code") or "").strip().lower()
            if not lemma or not lang:
                continue
            pos = normalize_pos(str(payload.get("pos") or ""))
            key = (lang, lemma.casefold(), pos)
            if key in seen:
                continue
            seen.add(key)
            yield LexicalRecord(
                source="kaikki",
                language=lang,
                lemma=lemma,
                pos=pos,
                definitions=_sense_glosses(payload.get("senses")),
                pronunciations=_ipa_values(payload.get("sounds")),
                etymology=str(payload.get("etymology_text") or "").strip(),
            )
            emitted += 1
            if limit is not None and emitted >= int(limit):
                break


__all__ = ["KAIKKI_DEFAULT_PATHS", "iter_kaikki_records"]
