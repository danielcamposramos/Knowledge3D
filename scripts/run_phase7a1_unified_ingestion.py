#!/usr/bin/env python3
"""Run the Phase 7.A.1 unified canonical knowledge ingestion pass.

Bulk artifacts are written under Knowledge3D.local. The repository only owns the
runner and reusable ingestion modules.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Iterable, Mapping

from knowledge3d.ingestion.canonical_lookup import canonical_word_star_id
from knowledge3d.ingestion.fonts.glyph_to_rpn import MANIFEST_PATH
from knowledge3d.ingestion.grammar.ud_grammar_builder import LOCAL_GRAMMAR_PATH
from knowledge3d.ingestion.letter_galaxy_builder import build_letter_galaxy, canonical_letter_entry
from knowledge3d.ingestion.math_symbol_builder import build_math_symbol_galaxy, canonical_math_symbol_entry
from knowledge3d.ingestion.symlink_helpers import link
from knowledge3d.ingestion.universal_knowledge.dbnary_ingester import (
    DBNARY_DEFAULT_PATH,
    LexicalRecord,
    WordMergeResult,
    iter_dbnary_records,
    merge_lexical_records_into_omw,
)
from knowledge3d.ingestion.universal_knowledge.kaikki_ingester import (
    KAIKKI_DEFAULT_PATHS,
    iter_kaikki_records,
)
from knowledge3d.ingestion.universal_knowledge.multilingual_meanings import (
    OMW_DEFAULT_PATH,
    SynsetEntry,
    load_all_omw,
    synset_to_star_bundle,
)
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


LOCAL_PHASE7A1_DIR = Path("/K3D/Knowledge3D.local/assets/phase7a1")
LOCAL_UNIFIED_STAR_PATH = LOCAL_PHASE7A1_DIR / "PHASE7A1_UNIFIED_STARS.jsonl"
LOCAL_UNIFIED_CANONICAL_PATH = LOCAL_PHASE7A1_DIR / "PHASE7A1_UNIFIED_CANONICAL.json"
LOCAL_UNIFIED_DANGLING_PATH = LOCAL_PHASE7A1_DIR / "PHASE7A1_DANGLING_REFS.json"

STAR_REF_FIELDS = (
    "taxonomy_refs",
    "visual_refs",
    "audio_refs",
    "reality_refs",
    "grammar_refs",
    "char_refs",
    "component_refs",
    "composite_of",
)


@dataclass
class UnifiedIngestionResult:
    stars: dict[str, dict] = field(default_factory=dict)
    canonical_entries: list[dict] = field(default_factory=list)
    dangling_refs: list[dict] = field(default_factory=list)
    source_counts: dict[str, int] = field(default_factory=dict)
    merge_result: WordMergeResult = field(default_factory=WordMergeResult)


def _progress(**counts: object) -> None:
    payload = " ".join(f"{key}={value}" for key, value in counts.items())
    print(f"[phase7a1] {payload}", flush=True)


def _read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict]:
    source_path = Path(path)
    if not source_path.exists():
        return []
    rows: list[dict] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                rows.append(json.loads(raw_line))
    return rows


def _star_from_row(row: Mapping[str, object]) -> MeaningCentricStar:
    return MeaningCentricStar.from_dict(row)


def _merge_row_links(row: dict, star: MeaningCentricStar) -> dict:
    updated = dict(row)
    star_payload = star.to_dict()
    for key in (
        "taxonomy_refs",
        "surface_forms",
        "visual_refs",
        "audio_refs",
        "reality_refs",
        "grammar_refs",
        "meta_refs",
        "char_refs",
        "component_refs",
        "composite_of",
    ):
        updated[key] = star_payload.get(key)
    return updated


def _add_rows(rows: dict[str, dict], incoming: Iterable[Mapping[str, object]]) -> None:
    for raw_row in incoming:
        row = dict(raw_row)
        star_id = str(row.get("star_id") or "").strip()
        if not star_id:
            continue
        existing = rows.get(star_id)
        if existing is None:
            rows[star_id] = row
            continue
        existing_star = _star_from_row(existing)
        incoming_star = _star_from_row(row)
        for field_name in STAR_REF_FIELDS + ("meta_refs",):
            merged = list(dict.fromkeys(getattr(existing_star, field_name) + getattr(incoming_star, field_name)))
            setattr(existing_star, field_name, merged)
        rows[star_id] = _merge_row_links(existing, existing_star)


def _chunked(items: Iterable[Mapping[str, object]], chunk_size: int) -> Iterable[list[Mapping[str, object]]]:
    chunk: list[Mapping[str, object]] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _canonical_word_entry(row: Mapping[str, object]) -> dict | None:
    domain = str(row.get("domain") or "")
    if not domain.startswith("Word/"):
        return None
    language = domain.split("/", 1)[1].strip().lower()
    surface_forms = dict(row.get("surface_forms") or {})
    lemma = ""
    if language in surface_forms:
        word_ref = str(dict(surface_forms[language]).get("word_ref") or "")
        prefix = f"word_{language}_"
        if word_ref.startswith(prefix):
            lemma = word_ref.removeprefix(prefix)
    key = f"{language}:{lemma or row.get('star_id')}"
    return {
        "kind": "word_lemma",
        "key": key,
        "star_id": str(row["star_id"]),
        "metadata": {
            "language": language,
            "char_count": len(row.get("component_refs") or []),
        },
    }


def _canonical_entries_from_rows(rows: Mapping[str, dict], grammar_canonical_rows: Iterable[dict]) -> list[dict]:
    entries: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows.values():
        entry: dict | None = None
        if str(row.get("domain") or "").startswith("Character/"):
            codepoint = row.get("codepoint")
            if codepoint is not None:
                entry = canonical_letter_entry(int(codepoint), font_count=len(row.get("font_glyphs") or []))
        elif str(row.get("domain") or "").startswith("Math/") and row.get("codepoint") is not None:
            entry = canonical_math_symbol_entry(row)
        elif str(row.get("domain") or "").startswith("Word/"):
            entry = _canonical_word_entry(row)
        if entry:
            key = (str(entry["kind"]), str(entry["key"]), str(entry["star_id"]))
            if key not in seen:
                seen.add(key)
                entries.append(entry)
    for entry in grammar_canonical_rows:
        key = (str(entry.get("kind")), str(entry.get("key")), str(entry.get("star_id")))
        if key not in seen:
            seen.add(key)
            entries.append(dict(entry))
    return sorted(entries, key=lambda item: (str(item.get("kind")), str(item.get("key"))))


def _supporting_word_star(word_ref: str, language: str) -> dict:
    star = MeaningCentricStar(
        star_id=word_ref,
        meaning_class="form",
        meaning_rpn=f"SURFACE_FORM LANG_{language.upper()} {word_ref.upper()} STORE",
        domain=f"Word/{language}",
        lod_class="LOD_ICON",
    )
    return star.to_dict()


def _supporting_taxonomy_star(ref_id: str) -> dict:
    star = MeaningCentricStar(
        star_id=ref_id,
        meaning_class="concept",
        meaning_rpn=f"TAXONOMY {ref_id.upper()} STORE",
        domain="Foundation/Taxonomy",
        lod_class="LOD_SUMMARY",
    )
    return star.to_dict()


def _supporting_font_glyph_star(glyph: Mapping[str, object]) -> dict:
    star_id = str(glyph.get("font_glyph_star_id") or "").strip()
    star = MeaningCentricStar(
        star_id=star_id,
        meaning_class="form",
        meaning_rpn=f"FONT_GLYPH {star_id.upper()}",
        domain="FontGlyph/Text",
        lod_class="LOD_ICON",
    )
    payload = star.to_dict()
    payload.update(
        {
            "family": glyph.get("family"),
            "style": glyph.get("style"),
            "font_file": glyph.get("font_file"),
            "font_index": glyph.get("font_index"),
            "font_glyph_key": glyph.get("font_glyph_key"),
        }
    )
    return payload


def _add_supporting_rows(rows: dict[str, dict]) -> None:
    additions: dict[str, dict] = {}
    for row in list(rows.values()):
        for glyph in row.get("font_glyphs") or []:
            glyph_id = str(dict(glyph).get("font_glyph_star_id") or "").strip()
            if glyph_id and glyph_id not in rows and glyph_id not in additions:
                additions[glyph_id] = _supporting_font_glyph_star(glyph)
        for field_name, ref_id in _iter_star_refs(row):
            if ref_id in rows or ref_id in additions:
                continue
            if field_name.startswith("surface_forms.") and ".word_ref" in field_name:
                language = field_name.split(".", 2)[1]
                additions[ref_id] = _supporting_word_star(ref_id, language)
            elif field_name == "taxonomy_refs" and ":" not in ref_id:
                additions[ref_id] = _supporting_taxonomy_star(ref_id)
    _add_rows(rows, additions.values())


def _apply_link_sweep(rows: dict[str, dict]) -> dict[str, dict]:
    stars = {star_id: _star_from_row(row) for star_id, row in rows.items()}
    for star_id, star in list(stars.items()):
        if star.domain.startswith("Word/"):
            for char_id in list(star.component_refs):
                target = stars.get(char_id)
                if target is not None:
                    link(star, target, "component_refs", "composite_of")
            continue
        if star.domain.startswith("Math/"):
            for char_id in list(star.char_refs):
                target = stars.get(char_id)
                if target is not None:
                    link(star, target, "char_refs", "mathematical_role")
            continue
        for language, surface in star.surface_forms.items():
            word_id = surface.word_ref
            target = stars.get(word_id)
            if target is not None:
                link(star, target, f"surface_forms.{language}.word_ref", "taxonomy_refs")
    return {star_id: _merge_row_links(rows[star_id], star) for star_id, star in stars.items()}


def _iter_star_refs(row: Mapping[str, object]):
    for field_name in STAR_REF_FIELDS:
        for ref_id in row.get(field_name) or []:
            yield field_name, str(ref_id)
    for language, form in dict(row.get("surface_forms") or {}).items():
        payload = dict(form or {})
        word_ref = str(payload.get("word_ref") or "").strip()
        if word_ref:
            yield f"surface_forms.{language}.word_ref", word_ref
        for ref_id in payload.get("char_refs") or []:
            yield f"surface_forms.{language}.char_refs", str(ref_id)


def find_dangling_refs(rows: Mapping[str, Mapping[str, object]]) -> list[dict]:
    star_ids = set(rows.keys())
    dangling: list[dict] = []
    for star_id, row in sorted(rows.items()):
        for field_name, ref_id in _iter_star_refs(row):
            if ref_id and ref_id not in star_ids:
                dangling.append({"source": star_id, "field": field_name, "target": ref_id})
    return dangling


def _iter_external_records(dbnary_paths: Iterable[Path], kaikki_paths: Iterable[Path]) -> Iterable[LexicalRecord]:
    dbnary_done = 0
    kaikki_done = 0
    for path in dbnary_paths:
        if Path(path).exists():
            for record in iter_dbnary_records(path):
                dbnary_done += 1
                if dbnary_done % 10000 == 0:
                    _progress(dbnary_records_done=dbnary_done, total_external_records=dbnary_done + kaikki_done)
                yield record
    for path in kaikki_paths:
        if Path(path).exists():
            for record in iter_kaikki_records(path):
                kaikki_done += 1
                if kaikki_done % 10000 == 0:
                    _progress(kaikki_records_done=kaikki_done, total_external_records=dbnary_done + kaikki_done)
                yield record
    _progress(
        dbnary_records_done=dbnary_done,
        kaikki_records_done=kaikki_done,
        total_external_records=dbnary_done + kaikki_done,
    )


def _default_dbnary_paths() -> list[Path]:
    if not DBNARY_DEFAULT_PATH.exists():
        return []
    return sorted(DBNARY_DEFAULT_PATH.glob("*_dbnary_ontolex.ttl.bz2"))


def build_unified_ingestion(
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    grammar_star_path: str | Path = LOCAL_GRAMMAR_PATH,
    omw_path: str | Path = OMW_DEFAULT_PATH,
    dbnary_paths: Iterable[str | Path] | None = None,
    kaikki_paths: Iterable[str | Path] | None = None,
    use_default_kaikki: bool = False,
    letter_workers: int | None = None,
) -> UnifiedIngestionResult:
    start_time = time.perf_counter()
    rows: dict[str, dict] = {}
    result = UnifiedIngestionResult()

    _progress(stage="letters_start")
    manifest = _read_json(manifest_path)
    resolved_letter_workers = int(letter_workers or max(1, min(12, (os.cpu_count() or 1))))
    letter_build = build_letter_galaxy(
        manifest,
        include_rpn=False,
        workers=resolved_letter_workers,
        progress=lambda data: _progress(stage="letters_progress", **data),
    )
    _add_rows(rows, letter_build.stars.values())
    result.source_counts["letters"] = len(letter_build.stars)
    result.source_counts["fonts_done"] = len(manifest.get("fonts", []) or [])
    result.source_counts["letter_workers"] = resolved_letter_workers
    _progress(
        stage="letters_done",
        letters_done=len(letter_build.stars),
        fonts_done=result.source_counts["fonts_done"],
        letter_workers=resolved_letter_workers,
    )

    letter_stars = {star_id: _star_from_row(row) for star_id, row in rows.items()}
    _progress(stage="math_symbols_start")
    math_build = build_math_symbol_galaxy(existing_char_stars=letter_stars)
    _add_rows(rows, math_build.stars.values())
    _add_rows(rows, math_build.target_updates.values())
    result.source_counts["math_symbols"] = len(math_build.stars)
    _progress(stage="math_symbols_done", math_symbols_done=len(math_build.stars))

    _progress(stage="grammar_start")
    grammar_rows = _read_jsonl(grammar_star_path)
    _add_rows(rows, grammar_rows)
    result.source_counts["grammar"] = len(grammar_rows)
    _progress(stage="grammar_done", grammar_done=len(grammar_rows))
    grammar_canonical_path = Path(grammar_star_path).with_suffix(Path(grammar_star_path).suffix + ".canonical.json")
    grammar_canonical_rows = _read_json(grammar_canonical_path) if grammar_canonical_path.exists() else []

    _progress(stage="omw_load_start")
    synsets: dict[str, SynsetEntry] = load_all_omw(Path(omw_path))
    _progress(stage="omw_load_done", synsets_done=len(synsets))
    _progress(stage="external_records_start")
    resolved_dbnary_paths = [Path(path) for path in (dbnary_paths if dbnary_paths is not None else _default_dbnary_paths())]
    resolved_kaikki_paths = [Path(path) for path in (kaikki_paths if kaikki_paths is not None else (KAIKKI_DEFAULT_PATHS if use_default_kaikki else []))]
    result.merge_result = merge_lexical_records_into_omw(
        synsets,
        _iter_external_records(resolved_dbnary_paths, resolved_kaikki_paths),
    )
    _progress(stage="external_records_done", external_records_done=result.merge_result.processed_count)
    _progress(
        stage="external_merge_done",
        external_records_done=result.merge_result.processed_count,
        external_merged=result.merge_result.merged_count,
        external_new_words=len(result.merge_result.new_word_stars),
    )
    _progress(stage="word_rows_start")
    word_row_count = 0
    for index, synset_id in enumerate(sorted(synsets), start=1):
        synset_rows = [star.to_dict() for star in synset_to_star_bundle(synsets[synset_id])]
        _add_rows(rows, synset_rows)
        word_row_count += len(synset_rows)
        if index % 5000 == 0:
            _progress(stage="word_rows_progress", synsets_done=index, words_done=word_row_count)

    _progress(
        stage="external_new_words_start",
        external_new_words=len(result.merge_result.new_word_stars),
    )
    external_done = 0
    external_total = len(result.merge_result.new_word_stars)
    external_rows = (star.to_dict() for star in result.merge_result.new_word_stars.values())
    for chunk in _chunked(external_rows, 10000):
        _add_rows(rows, chunk)
        external_done += len(chunk)
        word_row_count += len(chunk)
        _progress(
            stage="external_new_words_progress",
            external_new_words_done=external_done,
            external_new_words_total=external_total,
            words_done=word_row_count,
        )

    result.source_counts["word_and_meaning"] = word_row_count
    result.source_counts["external_lexical_records"] = result.merge_result.processed_count
    result.source_counts["external_merged"] = result.merge_result.merged_count
    result.source_counts["external_new_words"] = len(result.merge_result.new_word_stars)
    _progress(stage="words_done", words_done=word_row_count)

    _progress(stage="supporting_rows_start")
    _add_supporting_rows(rows)
    _progress(stage="supporting_rows_done", total_stars=len(rows))
    _progress(stage="link_sweep_start")
    rows = _apply_link_sweep(rows)
    link_count = 0
    for row in rows.values():
        for field_name in STAR_REF_FIELDS:
            link_count += len(row.get(field_name) or [])
        for form in dict(row.get("surface_forms") or {}).values():
            payload = dict(form or {})
            if payload.get("word_ref"):
                link_count += 1
            link_count += len(payload.get("char_refs") or [])
    _progress(stage="link_sweep_done", links_wired=link_count)
    result.stars = rows
    result.canonical_entries = _canonical_entries_from_rows(rows, grammar_canonical_rows)
    result.dangling_refs = find_dangling_refs(rows)
    result.source_counts["links_wired"] = link_count
    result.source_counts["wall_clock_seconds"] = round(time.perf_counter() - start_time, 3)
    _progress(
        stage="validation_done",
        dangling_refs=len(result.dangling_refs),
        canonical_entries=len(result.canonical_entries),
        wall_clock_seconds=result.source_counts["wall_clock_seconds"],
    )
    return result


def write_unified_ingestion(result: UnifiedIngestionResult, output_path: str | Path = LOCAL_UNIFIED_STAR_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path = output_path.with_name(output_path.stem + "_CANONICAL.json")
    dangling_path = output_path.with_name(output_path.stem + "_DANGLING_REFS.json")
    with output_path.open("w", encoding="utf-8") as handle:
        for star_id in sorted(result.stars):
            handle.write(json.dumps(result.stars[star_id], ensure_ascii=False) + "\n")
    canonical_path.write_text(
        json.dumps(result.canonical_entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    dangling_path.write_text(
        json.dumps(result.dangling_refs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "star_count": len(result.stars),
                "canonical_entry_count": len(result.canonical_entries),
                "dangling_ref_count": len(result.dangling_refs),
                "source_counts": result.source_counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if result.dangling_refs:
        raise RuntimeError(f"phase7a1_dangling_refs:{len(result.dangling_refs)}:{dangling_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 7.A.1 unified knowledge ingestion.")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--grammar-stars", default=str(LOCAL_GRAMMAR_PATH))
    parser.add_argument("--omw-path", default=str(OMW_DEFAULT_PATH))
    parser.add_argument("--dbnary-path", action="append", default=None)
    parser.add_argument("--kaikki-path", action="append", default=None)
    parser.add_argument("--use-default-kaikki", action="store_true")
    parser.add_argument("--out", default=str(LOCAL_UNIFIED_STAR_PATH))
    parser.add_argument("--letter-workers", type=int, default=max(1, min(12, (os.cpu_count() or 1))))
    args = parser.parse_args(argv)

    result = build_unified_ingestion(
        manifest_path=args.manifest,
        grammar_star_path=args.grammar_stars,
        omw_path=args.omw_path,
        dbnary_paths=args.dbnary_path,
        kaikki_paths=args.kaikki_path,
        use_default_kaikki=args.use_default_kaikki,
        letter_workers=args.letter_workers,
    )
    output_path = write_unified_ingestion(result, args.out)
    wall_clock = result.source_counts.get("wall_clock_seconds", 0.0)
    print(
        f"phase7a1_unified_stars={len(result.stars)} "
        f"letters_done={result.source_counts.get('letters', 0)} "
        f"fonts_done={result.source_counts.get('fonts_done', 0)} "
        f"math_symbols_done={result.source_counts.get('math_symbols', 0)} "
        f"grammar_done={result.source_counts.get('grammar', 0)} "
        f"words_done={result.source_counts.get('word_and_meaning', 0)} "
        f"links_wired={result.source_counts.get('links_wired', 0)} "
        f"canonical_entries={len(result.canonical_entries)} "
        f"dangling_refs={len(result.dangling_refs)} "
        f"wall_clock_seconds={wall_clock} "
        f"out={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
