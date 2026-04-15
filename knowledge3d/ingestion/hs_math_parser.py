"""Batch 8 HS math catalogue parser."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from knowledge3d.ingestion.canonical_lookup import canonical_slug
from knowledge3d.ingestion.math_semantic_aliases import UNICODE_TO_NAME


CANONICAL_LANGUAGES: tuple[str, ...] = ("en", "pt", "es", "fr", "de", "it", "ja", "zh", "ru")


class MathSymlinkNormaliseError(ValueError):
    """Raised when a symlink ref cannot be normalised safely."""


@dataclass(frozen=True)
class MathMeaningStarRow:
    canonical_id_raw: str
    category: str
    slug: str
    is_a: tuple[str, ...]
    rpn_sketch_raw: str
    symlink_refs_raw: tuple[str, ...]
    symlink_refs_norm: tuple[str, ...]
    surface_forms: dict[str, str]
    saudades: bool
    source_file: str
    source_line: int


@dataclass(frozen=True)
class ParseDiagnostics:
    rows: tuple[MathMeaningStarRow, ...]
    skipped_lines: tuple[int, ...]


def _strip_ticks(text: str) -> str:
    return str(text or "").strip().strip("`").strip()


def _trim_known_summary_tail(text: str) -> str:
    raw = str(text)
    markers = (
        "\n---\n\n**Star count:**",
        "\n---\n\n**Required canonical alias seeds",
    )
    cut = len(raw)
    for marker in markers:
        idx = raw.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return raw[:cut]


def normalise_symlink_ref(raw: str, *, source_file: str = "", source_line: int = 0) -> str:
    cleaned = _strip_ticks(raw)
    if not cleaned:
        raise MathSymlinkNormaliseError(f"empty_ref:{source_file}:{source_line}")
    prefix_map = (
        ("star.letter.", "letter::"),
        ("letter::", "letter::"),
        ("letter_", "letter::"),
        ("star.symbol.", "symbol::"),
        ("symbol::", "symbol::"),
        ("symbol_", "symbol::"),
        ("star.constant.", "constant::"),
        ("constant::", "constant::"),
        ("constant_", "constant::"),
        ("star.concept.", "concept::"),
        ("concept::", "concept::"),
        ("concept_", "concept::"),
    )
    target_prefix = ""
    tail = cleaned
    for candidate, normalized in prefix_map:
        if cleaned.startswith(candidate):
            target_prefix = normalized
            tail = cleaned[len(candidate) :]
            break
    if not target_prefix:
        raise MathSymlinkNormaliseError(f"unsupported_namespace:{cleaned}:{source_file}:{source_line}")
    tail = tail.strip()
    if not tail:
        raise MathSymlinkNormaliseError(f"empty_tail:{cleaned}:{source_file}:{source_line}")
    if len(tail) == 1 and ord(tail) > 127 and target_prefix in {"symbol::", "constant::"}:
        if tail not in UNICODE_TO_NAME:
            raise MathSymlinkNormaliseError(f"unknown_glyph:{tail}:{source_file}:{source_line}")
        tail = UNICODE_TO_NAME[tail]
    tail = canonical_slug(tail)
    if not tail:
        raise MathSymlinkNormaliseError(f"empty_slug:{cleaned}:{source_file}:{source_line}")
    return f"{target_prefix}{tail}"


def _parse_bool(text: str) -> bool:
    cleaned = _strip_ticks(text).lower()
    if not cleaned:
        return False
    return cleaned.startswith("true")


def _parse_list_cell(text: str) -> tuple[str, ...]:
    cleaned = _strip_ticks(text)
    if not cleaned:
        return ()
    return tuple(_strip_ticks(part) for part in cleaned.split(",") if _strip_ticks(part))


def _finalize_block(block: dict[str, object], source_file: str, skipped_lines: list[int]) -> MathMeaningStarRow | None:
    canonical_id_raw = str(block.get("canonical_id_raw") or "").strip()
    header_name = str(block.get("header_name") or "").strip()
    source_line = int(block.get("source_line") or 0)
    if not canonical_id_raw:
        if header_name:
            raise ValueError(f"missing_canonical_id:{source_file}:{source_line}")
        return None
    surface_forms = dict(block.get("surface_forms") or {})
    if not surface_forms:
        skipped_lines.append(source_line)
        return None
    category = canonical_id_raw.replace("::", "_").split("_", 1)[0].strip().lower()
    slug = canonical_slug(canonical_id_raw.replace("::", "_").split("_", 1)[1] if "_" in canonical_id_raw.replace("::", "_") else "")
    raw_refs = tuple(block.get("symlink_refs_raw") or ())
    norm_refs = tuple(block.get("symlink_refs_norm") or ())
    return MathMeaningStarRow(
        canonical_id_raw=canonical_id_raw,
        category=category,
        slug=slug,
        is_a=tuple(block.get("is_a") or ()),
        rpn_sketch_raw=str(block.get("rpn_sketch_raw") or "").strip(),
        symlink_refs_raw=raw_refs,
        symlink_refs_norm=norm_refs,
        surface_forms=surface_forms,
        saudades=bool(block.get("saudades") or False),
        source_file=source_file,
        source_line=source_line,
    )


def parse_cluster1_bullets_with_diagnostics(text: str, *, source_file: str) -> ParseDiagnostics:
    rows: list[MathMeaningStarRow] = []
    skipped_lines: list[int] = []
    current: dict[str, object] | None = None
    surface_mode = False
    for line_number, raw_line in enumerate(_trim_known_summary_tail(text).splitlines(), start=1):
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if stripped.startswith("#### "):
            if current is not None:
                resolved = _finalize_block(current, source_file, skipped_lines)
                if resolved is not None:
                    rows.append(resolved)
            current = {
                "header_name": stripped[5:].strip(),
                "source_line": line_number,
                "surface_forms": {},
            }
            surface_mode = False
            continue
        if current is None:
            continue
        if surface_mode and stripped.startswith("- **") and "**:" in stripped:
            surface_mode = False
        if surface_mode and stripped.startswith("- ") and ":" in stripped:
            language, value = stripped[2:].split(":", 1)
            language = language.strip().lower()
            if language not in CANONICAL_LANGUAGES:
                raise ValueError(f"unknown_surface_form_language:{language}:{source_file}:{line_number}")
            current_surface = dict(current.get("surface_forms") or {})
            current_surface[language] = _strip_ticks(value).strip('"')
            current["surface_forms"] = current_surface
            continue
        if stripped.startswith("- **surface_forms**"):
            surface_mode = True
            continue
        if stripped.startswith("- **") and "**:" in stripped:
            surface_mode = False
            field_name, value = stripped[3:].split("**:", 1)
            field_name = field_name.strip("* ").lower()
            value = value.strip()
            if field_name == "canonical_id":
                current["canonical_id_raw"] = _strip_ticks(value)
            elif field_name == "is_a":
                current["is_a"] = _parse_list_cell(value)
            elif field_name == "rpn_sketch":
                current["rpn_sketch_raw"] = _strip_ticks(value)
            elif field_name == "symlinks":
                raw_refs = _parse_list_cell(value)
                current["symlink_refs_raw"] = raw_refs
                current["symlink_refs_norm"] = tuple(
                    normalise_symlink_ref(ref, source_file=source_file, source_line=line_number)
                    for ref in raw_refs
                )
            elif field_name == "saudades":
                current["saudades"] = _parse_bool(value)
            continue
    if current is not None:
        resolved = _finalize_block(current, source_file, skipped_lines)
        if resolved is not None:
            rows.append(resolved)
    return ParseDiagnostics(rows=tuple(rows), skipped_lines=tuple(skipped_lines))


def parse_cluster1_bullets(text: str, *, source_file: str) -> list[MathMeaningStarRow]:
    return list(parse_cluster1_bullets_with_diagnostics(text, source_file=source_file).rows)


def parse_cluster2_json(text: str, *, source_file: str) -> list[MathMeaningStarRow]:
    return parse_cluster1_bullets(text, source_file=source_file)


def parse_cluster3_hybrid(text: str, *, source_file: str) -> list[MathMeaningStarRow]:
    return parse_cluster1_bullets(text, source_file=source_file)


def parse_hs_math_file(path: Path, *, shape: str | None = None) -> list[MathMeaningStarRow]:
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8")
    decided = shape
    if decided is None:
        upper = resolved.name.upper()
        if "CLUSTER1" in upper:
            decided = "cluster1"
        elif "CLUSTER2" in upper:
            decided = "cluster2"
        elif "CLUSTER3" in upper:
            decided = "cluster3"
        else:
            raise ValueError(f"unknown_hs_math_shape:{resolved}")
    if decided == "cluster1":
        return parse_cluster1_bullets(text, source_file=resolved.name)
    if decided == "cluster2":
        return parse_cluster1_bullets(text, source_file=resolved.name)
    if decided == "cluster3":
        return parse_cluster1_bullets(text, source_file=resolved.name)
    raise ValueError(f"unsupported_hs_math_shape:{decided}")


__all__ = [
    "CANONICAL_LANGUAGES",
    "MathMeaningStarRow",
    "MathSymlinkNormaliseError",
    "ParseDiagnostics",
    "normalise_symlink_ref",
    "parse_cluster1_bullets",
    "parse_cluster1_bullets_with_diagnostics",
    "parse_cluster2_json",
    "parse_cluster3_hybrid",
    "parse_hs_math_file",
]
