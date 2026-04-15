"""Letter Galaxy builder for Phase 7.A.1.

Ingestion-path only. It turns the local system font manifest into one
meaning-centric character star per Unicode letter/digit codepoint, with all
matching text-font glyph drawings embedded as Drawing Galaxy RPN.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Callable, Iterable, Mapping
import unicodedata

from knowledge3d.ingestion.canonical_lookup import (
    canonical_char_star_id,
    canonical_entry_id,
)
from knowledge3d.ingestion.fonts.glyph_to_rpn import (
    MANIFEST_PATH,
    extract_glyph_rpn,
    font_glyph_metadata,
    glyph_key,
    glyph_star_id,
    script_for_codepoint,
    update_manifest_unreadable_codepoints,
)
from knowledge3d.ingestion.symlink_helpers import link
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


LOCAL_LETTER_STAR_PATH = Path("/K3D/Knowledge3D.local/assets/fonts/text/LETTER_STARS.jsonl")

DIGIT_TARGETS = {
    ord("0"): "concept_digit_zero",
    ord("1"): "concept_digit_one",
    ord("2"): "concept_digit_two",
    ord("3"): "concept_digit_three",
    ord("4"): "concept_digit_four",
    ord("5"): "concept_digit_five",
    ord("6"): "concept_digit_six",
    ord("7"): "concept_digit_seven",
    ord("8"): "concept_digit_eight",
    ord("9"): "concept_digit_nine",
}

GREEK_MATH_TARGETS = {
    ord("α"): "concept_math_alpha",
    ord("β"): "concept_math_beta",
    ord("γ"): "concept_math_gamma",
    ord("δ"): "concept_math_delta",
    ord("θ"): "concept_math_theta",
    ord("λ"): "concept_math_lambda",
    ord("μ"): "concept_math_mu",
    ord("π"): "concept_math_pi",
    ord("σ"): "concept_math_sigma",
    ord("Σ"): "concept_math_summation",
    ord("Π"): "concept_math_product",
    ord("Δ"): "concept_math_difference",
}

LATIN_VARIABLE_ROLE_TARGETS = {
    ord("x"): "concept_variable_role_unknown_x",
    ord("y"): "concept_variable_role_unknown_y",
    ord("z"): "concept_variable_role_unknown_z",
    ord("i"): "concept_variable_role_index_i",
    ord("j"): "concept_variable_role_index_j",
    ord("k"): "concept_variable_role_index_k",
    ord("n"): "concept_variable_role_count_n",
    ord("m"): "concept_variable_role_count_m",
}

SCRIPT_LANGUAGES = {
    "latn": ["en", "pt", "es", "fr", "de", "it"],
    "grek": ["el"],
    "cyrl": ["ru"],
    "arab": ["ar"],
    "deva": ["hi", "sa"],
    "hani": ["zh", "ja"],
    "hira": ["ja"],
    "kana": ["ja"],
    "hang": ["ko"],
    "thai": ["th"],
    "taml": ["ta"],
    "gujr": ["gu"],
    "mlym": ["ml"],
    "tibt": ["bo"],
    "geor": ["ka"],
    "hebr": ["he"],
    "beng": ["bn"],
    "guru": ["pa"],
    "ethi": ["am"],
}


@dataclass
class LetterGalaxyBuild:
    stars: dict[str, dict]
    target_updates: dict[str, dict]
    skipped_links: list[dict[str, str]]
    glyph_failures: list[dict[str, str]]


def _is_letter_or_digit(codepoint: int) -> bool:
    try:
        category = unicodedata.category(chr(codepoint))
    except ValueError:
        return False
    return category.startswith("L") or category == "Nd"


def _range_to_ints(start: str, end: str) -> range:
    left = int(start.removeprefix("U+"), 16)
    right = int(end.removeprefix("U+"), 16)
    return range(left, right + 1)


def _entry_covers_codepoint(entry: Mapping[str, object], codepoint: int) -> bool:
    if f"U+{codepoint:04X}" in {str(value).upper() for value in entry.get("unreadable_codepoints", []) or []}:
        return False
    for start, end in entry.get("codepoint_ranges", []) or []:
        if codepoint in _range_to_ints(str(start), str(end)):
            return True
    return False


def _iter_entry_codepoints(entry: Mapping[str, object]) -> Iterable[int]:
    unreadable = {str(value).upper() for value in entry.get("unreadable_codepoints", []) or []}
    seen: set[int] = set()
    for start, end in entry.get("codepoint_ranges", []) or []:
        for codepoint in _range_to_ints(str(start), str(end)):
            if codepoint in seen:
                continue
            if f"U+{codepoint:04X}" in unreadable:
                continue
            if not _is_letter_or_digit(codepoint):
                continue
            seen.add(codepoint)
            yield codepoint


def _glyph_refs_for_font_entry(entry: Mapping[str, object]) -> list[tuple[int, dict]]:
    family = str(entry.get("family") or "").strip()
    style = str(entry.get("style") or "").strip()
    path = str(entry.get("path") or "").strip()
    font_index = int(entry.get("font_index") or 0)
    if not family or not style or not path:
        return []
    rows: list[tuple[int, dict]] = []
    for codepoint in _iter_entry_codepoints(entry):
        rows.append(
            (
                codepoint,
                {
                    "family": family,
                    "style": style,
                    "font_glyph_star_id": glyph_star_id(family, style, codepoint),
                    "font_glyph_key": glyph_key(family, style, codepoint),
                    "font_file": path,
                    "font_index": font_index,
                },
            )
        )
    return rows


def _chunked_entries(entries: list[Mapping[str, object]], chunk_size: int) -> list[list[Mapping[str, object]]]:
    if chunk_size <= 0:
        chunk_size = 1
    return [entries[index : index + chunk_size] for index in range(0, len(entries), chunk_size)]


def _glyph_refs_for_entry_chunk(chunk: list[Mapping[str, object]]) -> list[tuple[int, dict]]:
    rows: list[tuple[int, dict]] = []
    for entry in chunk:
        rows.extend(_glyph_refs_for_font_entry(entry))
    return rows


def _iter_manifest_codepoints(manifest: Mapping[str, object]) -> Iterable[int]:
    seen: set[int] = set()
    for entry in manifest.get("fonts", []) or []:
        for codepoint in _iter_entry_codepoints(entry):
            if codepoint in seen:
                continue
            seen.add(codepoint)
            yield codepoint


def _glyph_entries_for_codepoint(
    manifest: Mapping[str, object],
    codepoint: int,
    *,
    include_rpn: bool,
) -> tuple[list[dict], list[dict[str, str]]]:
    glyphs: list[dict] = []
    failures: list[dict[str, str]] = []
    for entry in manifest.get("fonts", []) or []:
        if not _entry_covers_codepoint(entry, codepoint):
            continue
        family = str(entry.get("family") or "").strip()
        style = str(entry.get("style") or "").strip()
        path = str(entry.get("path") or "").strip()
        font_index = int(entry.get("font_index") or 0)
        if not family or not style or not path:
            continue
        glyph_row = {
            "family": family,
            "style": style,
            "font_glyph_star_id": glyph_star_id(family, style, codepoint),
            "font_glyph_key": glyph_key(family, style, codepoint),
            "font_file": path,
            "font_index": font_index,
        }
        if include_rpn:
            try:
                glyph = extract_glyph_rpn(path, codepoint, font_index=font_index)
            except Exception as exc:
                failures.append(
                    {
                        "codepoint": f"U+{codepoint:04X}",
                        "font": path,
                        "font_index": str(font_index),
                        "error": str(exc),
                    }
                )
                continue
            glyph_row.update(
                {
                    "rpn_program": glyph.rpn_program,
                    "rpn_bytes_hex": glyph.rpn_bytes.hex(),
                    "rpn_program_ref": "",
                    "metadata": font_glyph_metadata(family, style, codepoint, glyph),
                }
            )
        glyphs.append(glyph_row)
    return glyphs, failures


def _build_glyph_index_refs_only(
    manifest: Mapping[str, object],
    *,
    workers: int = 1,
    progress: Callable[[dict[str, int]], None] | None = None,
) -> dict[int, list[dict]]:
    entries = [dict(entry) for entry in manifest.get("fonts", []) or []]
    codepoint_to_glyphs: dict[int, list[dict]] = defaultdict(list)
    if not entries:
        return codepoint_to_glyphs
    if workers <= 1:
        for index, entry in enumerate(entries, start=1):
            for codepoint, glyph_row in _glyph_refs_for_font_entry(entry):
                codepoint_to_glyphs[codepoint].append(glyph_row)
            if progress is not None and (index == len(entries) or index % 64 == 0):
                progress({"fonts_done": index, "fonts_total": len(entries), "letters_done": len(codepoint_to_glyphs)})
        return codepoint_to_glyphs

    chunk_size = max(8, min(64, len(entries) // max(1, workers * 4) or 8))
    chunks = _chunked_entries(entries, chunk_size)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_glyph_refs_for_entry_chunk, chunk) for chunk in chunks]
        completed = 0
        for future in as_completed(futures):
            for codepoint, glyph_row in future.result():
                codepoint_to_glyphs[codepoint].append(glyph_row)
            completed += 1
            if progress is not None:
                done_fonts = min(len(entries), completed * chunk_size)
                progress({"fonts_done": done_fonts, "fonts_total": len(entries), "letters_done": len(codepoint_to_glyphs)})
    return codepoint_to_glyphs


def _char_domain(codepoint: int) -> str:
    return f"Character/{script_for_codepoint(codepoint)}"


def _languages_for_codepoint(codepoint: int) -> list[str]:
    return list(SCRIPT_LANGUAGES.get(script_for_codepoint(codepoint), []))


def _math_targets_for_codepoint(codepoint: int) -> list[str]:
    targets: list[str] = []
    for table in (DIGIT_TARGETS, GREEK_MATH_TARGETS, LATIN_VARIABLE_ROLE_TARGETS):
        target = table.get(codepoint)
        if target:
            targets.append(target)
    return targets


def make_letter_star(codepoint: int, font_glyphs: list[dict]) -> MeaningCentricStar:
    char = chr(codepoint)
    category = unicodedata.category(char)
    name = unicodedata.name(char, f"U+{codepoint:04X}")
    script = script_for_codepoint(codepoint)
    meaning_rpn = f"CHAR U+{codepoint:04X} GLYPH {script} {category}"
    return MeaningCentricStar(
        star_id=canonical_char_star_id(char),
        meaning_class="form",
        domain=_char_domain(codepoint),
        meaning_rpn=meaning_rpn,
        visual_refs=[row["font_glyph_star_id"] for row in font_glyphs],
        lod_class="LOD_ICON",
    )


def _star_payload(star: MeaningCentricStar, codepoint: int, font_glyphs: list[dict]) -> dict:
    char = chr(codepoint)
    payload = star.to_dict()
    payload.update(
        {
            "name": f"letter '{char}'" if unicodedata.category(char).startswith("L") else f"digit '{char}'",
            "codepoint": codepoint,
            "unicode_char": char,
            "script": script_for_codepoint(codepoint),
            "unicode_category": unicodedata.category(char),
            "unicode_name": unicodedata.name(char, ""),
            "languages": _languages_for_codepoint(codepoint),
            "font_glyphs": font_glyphs,
            "answer_eligible": False,
            "selection_role": "unknown",
        }
    )
    return payload


def build_letter_galaxy(
    manifest: Mapping[str, object],
    *,
    existing_targets: Mapping[str, MeaningCentricStar] | None = None,
    include_rpn: bool = True,
    codepoints: Iterable[int] | None = None,
    workers: int = 1,
    progress: Callable[[dict[str, int]], None] | None = None,
) -> LetterGalaxyBuild:
    """Build letter/digit stars and apply bidirectional links to existing targets."""

    targets = dict(existing_targets or {})
    star_rows: dict[str, dict] = {}
    target_updates: dict[str, dict] = {}
    skipped_links: list[dict[str, str]] = []
    glyph_failures: list[dict[str, str]] = []
    points = sorted(set(codepoints)) if codepoints is not None else None
    if not include_rpn:
        glyph_index = _build_glyph_index_refs_only(manifest, workers=workers, progress=progress)
        selected_points = points if points is not None else sorted(glyph_index.keys())
        total_points = len(selected_points)
        for index, codepoint in enumerate(selected_points, start=1):
            font_glyphs = list(glyph_index.get(codepoint, []))
            if not font_glyphs:
                continue
            star = make_letter_star(codepoint, font_glyphs)
            for target_id in _math_targets_for_codepoint(codepoint):
                target = targets.get(target_id)
                if target is None:
                    skipped_links.append({"source": star.star_id, "target": target_id, "reason": "target_missing"})
                    continue
                link(star, target, "taxonomy_refs", "component_refs")
                target_updates[target_id] = target.to_dict()
            star_rows[star.star_id] = _star_payload(star, codepoint, font_glyphs)
            if progress is not None and (index == total_points or index % 4096 == 0):
                progress({"letters_done": index, "letters_total": total_points, "stars_done": len(star_rows)})
    else:
        selected_points = points if points is not None else list(_iter_manifest_codepoints(manifest))
        total_points = len(selected_points)
        for index, codepoint in enumerate(sorted(set(selected_points)), start=1):
            category = unicodedata.category(chr(codepoint))
            if not (category.startswith("L") or category == "Nd"):
                continue
            font_glyphs, failures = _glyph_entries_for_codepoint(manifest, codepoint, include_rpn=include_rpn)
            glyph_failures.extend(failures)
            if not font_glyphs:
                continue
            star = make_letter_star(codepoint, font_glyphs)
            for target_id in _math_targets_for_codepoint(codepoint):
                target = targets.get(target_id)
                if target is None:
                    skipped_links.append({"source": star.star_id, "target": target_id, "reason": "target_missing"})
                    continue
                link(star, target, "taxonomy_refs", "component_refs")
                target_updates[target_id] = target.to_dict()
            star_rows[star.star_id] = _star_payload(star, codepoint, font_glyphs)
            if progress is not None and (index == total_points or index % 256 == 0):
                progress({"letters_done": index, "letters_total": total_points, "stars_done": len(star_rows)})
    return LetterGalaxyBuild(
        stars=star_rows,
        target_updates=target_updates,
        skipped_links=skipped_links,
        glyph_failures=glyph_failures,
    )


def write_letter_galaxy_build(build: LetterGalaxyBuild, output_path: str | Path = LOCAL_LETTER_STAR_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for star_id in sorted(build.stars):
            handle.write(json.dumps(build.stars[star_id], ensure_ascii=False) + "\n")
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "star_count": len(build.stars),
                "target_update_count": len(build.target_updates),
                "skipped_link_count": len(build.skipped_links),
                "glyph_failure_count": len(build.glyph_failures),
                "skipped_links": build.skipped_links[:500],
                "glyph_failures": build.glyph_failures[:500],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def canonical_letter_entry(codepoint: int, font_count: int) -> dict[str, object]:
    char = chr(codepoint)
    return {
        "id": canonical_entry_id("letter_star", f"U+{codepoint:04X}"),
        "kind": "letter_star",
        "key": f"U+{codepoint:04X}",
        "star_id": canonical_char_star_id(char),
        "metadata": {
            "script": script_for_codepoint(codepoint),
            "languages": _languages_for_codepoint(codepoint),
            "font_count": int(font_count),
        },
    }


def register_mathematical_role_symlink_kind(lookup) -> str:
    return lookup.register(
        kind="symlink_kind",
        key="mathematical_role",
        star_id="mathematical_role",
        metadata={"forward_field": "taxonomy_refs", "backward_field": "component_refs"},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build local Letter Galaxy stars from the system font manifest.")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--out", default=str(LOCAL_LETTER_STAR_PATH))
    parser.add_argument(
        "--refs-only",
        action="store_true",
        help="Write font_glyph references without extracting embedded RPN programs.",
    )
    parser.add_argument(
        "--mark-unreadable",
        action="store_true",
        help="Write glyph extraction failures back into MANIFEST.json unreadable_codepoints.",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 1))))
    args = parser.parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    build = build_letter_galaxy(manifest, include_rpn=not args.refs_only, workers=max(1, int(args.workers)))
    if args.mark_unreadable and build.glyph_failures:
        update_manifest_unreadable_codepoints(args.manifest, build.glyph_failures)
    output_path = write_letter_galaxy_build(build, args.out)
    print(
        f"letter_stars={len(build.stars)} "
        f"target_updates={len(build.target_updates)} "
        f"skipped_links={len(build.skipped_links)} "
        f"glyph_failures={len(build.glyph_failures)} "
        f"out={output_path}"
    )
    return 0


__all__ = [
    "DIGIT_TARGETS",
    "GREEK_MATH_TARGETS",
    "LATIN_VARIABLE_ROLE_TARGETS",
    "LOCAL_LETTER_STAR_PATH",
    "LetterGalaxyBuild",
    "build_letter_galaxy",
    "canonical_letter_entry",
    "make_letter_star",
    "register_mathematical_role_symlink_kind",
    "write_letter_galaxy_build",
]


if __name__ == "__main__":
    raise SystemExit(main())
