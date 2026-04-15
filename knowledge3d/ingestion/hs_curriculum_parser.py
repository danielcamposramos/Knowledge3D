"""Generic structured parser for remaining HS curriculum TEMP files.

This parser is intentionally conservative. It only extracts stars from source
shapes that are already structured enough to ingest safely:

- fenced JSON blocks containing a single star or a ``stars`` array
- fenced YAML blocks with ``canonical_id`` records
- inline JSONL objects with ``canonical_id``
- markdown exemplar blocks keyed by ``**canonical_id:**``

Narrative-only prose is ignored on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from knowledge3d.ingestion.canonical_lookup import canonical_char_star_id, canonical_slug
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar, SurfaceForm


FENCED_BLOCK_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_-]*)\n(?P<body>.*?)```", re.S)
MARKDOWN_CANONICAL_RE = re.compile(
    r"^\s*(?:-\s*)?(?:\*\*canonical_id:?\*\*|canonical_id)\s*:?\s*(?P<value>.+?)\s*$",
    re.I | re.M,
)
INLINE_JSON_RE = re.compile(r"^\s*\{.*\"canonical_id\".*\}\s*$")
LANG_KEYS = ("en", "pt", "es", "fr", "de", "it", "ja", "zh", "ru")


@dataclass(frozen=True)
class CurriculumRow:
    canonical_id: str
    source_file: str
    source_line: int
    is_a: tuple[str, ...]
    rpn_sketch: str
    surface_forms: dict[str, str]
    symlinks: tuple[str, ...]
    saudades: str
    domain: str
    context_id: int = 0
    ethical_trit: int = 0


@dataclass(frozen=True)
class CurriculumPayload:
    rows: tuple[CurriculumRow, ...]
    skipped: tuple[str, ...]


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and ((text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'"))):
        return text[1:-1].strip()
    return text


def _strip_ticks(value: str) -> str:
    text = _clean_text(value)
    if text.startswith("`") and text.endswith("`") and len(text) >= 2:
        return text[1:-1].strip()
    return text


def _line_of_offset(text: str, offset: int) -> int:
    return int(text.count("\n", 0, max(0, offset)) + 1)


def _flatten_is_a(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip().strip("`")
        if not text:
            return ()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        parts = re.split(r"[,\n]", text)
        return tuple(part.strip().strip("`").strip() for part in parts if part.strip().strip("`").strip())
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            text = _clean_text(item).strip("`")
            if text:
                out.append(text)
        return tuple(out)
    return ()


def _flatten_symlinks(value: Any) -> tuple[str, ...]:
    out: list[str] = []

    def _push(raw: Any) -> None:
        text = _clean_text(raw)
        if not text:
            return
        text = text.replace("<->", "").replace("->", "").replace("bidirectional_to:", "").replace("to:", "")
        text = text.replace("bidirectional:", "").strip()
        text = text.strip("`")
        if text and text not in out:
            out.append(text)

    if isinstance(value, str):
        for part in re.split(r"[,|\n]", value):
            _push(part)
    elif isinstance(value, dict):
        for nested in value.values():
            if isinstance(nested, (list, tuple)):
                for item in nested:
                    _push(item)
            else:
                _push(nested)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, dict):
                for nested in item.values():
                    _push(nested)
            else:
                _push(item)
    return tuple(out)


def _parse_surface_forms(value: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(value, dict):
        for language in LANG_KEYS:
            text = _clean_text(value.get(language))
            if text:
                out[language] = text
        return out
    if isinstance(value, str):
        text = value.strip()
        for segment in re.split(r"[;\n]", text):
            if ":" not in segment:
                continue
            key, raw = segment.split(":", 1)
            language = key.strip().lstrip("-").strip().lower()
            if language in LANG_KEYS:
                cleaned = _clean_text(raw)
                if cleaned:
                    out[language] = cleaned
    return out


def _parse_compact_lang_map(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for segment in re.split(r"[;\n]", str(text or "")):
        if ":" not in segment:
            continue
        key, raw = segment.split(":", 1)
        language = key.strip().lower()
        if language not in LANG_KEYS:
            continue
        cleaned = _clean_text(raw)
        if cleaned:
            out[language] = cleaned
    return out


def _infer_domain(source_file: str, canonical_id: str) -> str:
    lowered = source_file.lower()
    cid = canonical_id.lower()
    if "math" in lowered:
        return "mathematics"
    if "arc_reasoning" in lowered or cid.startswith(("concept_moore_", "concept_von_neumann_", "rule_manhattan_", "rule_chebyshev_", "method_connected_", "method_flood_", "method_bounding_", "concept_object_")):
        return "arc"
    if any(token in lowered for token in ("natural_sciences", "earth_space", "history_geography")):
        return "reality"
    if any(token in lowered for token in ("languages_linguistics", "crosscultural", "humanities")):
        return "language"
    if "applied_cs" in lowered:
        if any(token in cid for token in ("system_", "skill_", "technique_", "concept_binary", "concept_media", "concept_algorithm")):
            return "tools"
        if any(token in cid for token in ("disorder_", "nutrient_", "exercise_", "concept_bmi", "concept_bac")):
            return "reality"
        return "language"
    return "language"


def _coerce_row(payload: Mapping[str, Any], *, source_file: str, source_line: int) -> CurriculumRow | None:
    canonical_id = _strip_ticks(_clean_text(payload.get("canonical_id") or payload.get("star_id")))
    if not canonical_id:
        return None
    is_a = _flatten_is_a(payload.get("is_a"))
    rpn_sketch = _clean_text(payload.get("rpn_sketch") or payload.get("RPN") or payload.get("rpn"))
    surface_forms = _parse_surface_forms(payload.get("surface_forms"))
    if not rpn_sketch or not surface_forms:
        return None
    symlinks = _flatten_symlinks(payload.get("symlinks"))
    saudades = _clean_text(payload.get("saudades") or payload.get("saudades_flag"))
    context_id = int(payload.get("context_id", 0) or 0)
    ethical_trit = int(payload.get("ethical_trit", 0) or 0)
    return CurriculumRow(
        canonical_id=canonical_id,
        source_file=source_file,
        source_line=int(source_line),
        is_a=is_a,
        rpn_sketch=rpn_sketch,
        surface_forms=surface_forms,
        symlinks=symlinks,
        saudades=saudades,
        domain=_infer_domain(source_file, canonical_id),
        context_id=context_id,
        ethical_trit=max(-1, min(1, ethical_trit)),
    )


def _parse_loose_key_value_block(body: str) -> dict[str, Any]:
    lines = body.splitlines()
    payload: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        if ":" not in stripped:
            index += 1
            continue
        key, value = stripped.split(":", 1)
        key = key.strip().strip('"').strip("'")
        value = value.strip()
        if key in {"rpn_sketch", "RPN", "rpn"} and value == "|":
            block: list[str] = []
            index += 1
            while index < len(lines):
                candidate = lines[index]
                if candidate.startswith("  ") or candidate.startswith("\t"):
                    block.append(candidate.strip())
                    index += 1
                    continue
                if not candidate.strip():
                    index += 1
                    continue
                break
            payload[key] = "\n".join(block).strip()
            continue
        if key == "surface_forms":
            block: list[str] = []
            index += 1
            while index < len(lines):
                candidate = lines[index]
                stripped_candidate = candidate.strip()
                if candidate.startswith("  ") or candidate.startswith("\t"):
                    block.append(stripped_candidate)
                    index += 1
                    continue
                if stripped_candidate and re.match(rf"^({'|'.join(LANG_KEYS)}):", stripped_candidate):
                    block.append(stripped_candidate)
                    index += 1
                    continue
                if not stripped_candidate:
                    index += 1
                    continue
                break
            payload[key] = _parse_compact_lang_map("\n".join(block))
            continue
        if key == "symlinks":
            items: list[str] = []
            index += 1
            while index < len(lines):
                candidate = lines[index].strip()
                if candidate.startswith("-"):
                    items.append(candidate[1:].strip())
                    index += 1
                    continue
                if not candidate:
                    index += 1
                    continue
                break
            payload[key] = items
            continue
        payload[key] = value.strip().strip('"').strip("'")
        index += 1
    return payload


def _extract_json_object_slices(body: str) -> list[str]:
    slices: list[str] = []
    start = 0
    while True:
        marker = body.find('"canonical_id"', start)
        if marker < 0:
            break
        left = body.rfind("{", 0, marker)
        if left < 0:
            break
        depth = 0
        in_string = False
        escaped = False
        right = left
        while right < len(body):
            char = body[right]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        slices.append(body[left : right + 1])
                        start = right + 1
                        break
            right += 1
        else:
            break
    return slices


def _iter_fenced_candidates(text: str, *, source_file: str) -> Iterable[CurriculumRow]:
    for match in FENCED_BLOCK_RE.finditer(text):
        body = match.group("body")
        lang = str(match.group("lang") or "").strip().lower()
        offset_line = _line_of_offset(text, match.start())
        candidates: list[dict[str, Any]] = []
        parsed: Any = None
        if lang in {"json", ""}:
            try:
                parsed = json.loads(body)
            except Exception:
                parsed = None
        if parsed is None and lang in {"yaml", "yml", ""}:
            try:
                parsed = yaml.safe_load(body)
            except Exception:
                parsed = None
        if isinstance(parsed, dict):
            if "stars" in parsed and isinstance(parsed.get("stars"), list):
                candidates.extend(item for item in parsed.get("stars") or [] if isinstance(item, dict))
            elif "canonical_id" in parsed:
                candidates.append(parsed)
        elif isinstance(parsed, list):
            candidates.extend(item for item in parsed if isinstance(item, dict))
        if not candidates and "canonical_id:" in body:
            loose = _parse_loose_key_value_block(body)
            if loose:
                candidates.append(loose)
        if not candidates and '"canonical_id"' in body:
            for obj_text in _extract_json_object_slices(body):
                parsed_obj = None
                try:
                    parsed_obj = json.loads(obj_text)
                except Exception:
                    parsed_obj = None
                if isinstance(parsed_obj, dict):
                    candidates.append(parsed_obj)
        for payload in candidates:
            row = _coerce_row(payload, source_file=source_file, source_line=offset_line)
            if row is not None:
                yield row


def _iter_inline_json_candidates(text: str, *, source_file: str) -> Iterable[CurriculumRow]:
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or not INLINE_JSON_RE.match(line):
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        row = _coerce_row(payload, source_file=source_file, source_line=line_no)
        if row is not None:
            yield row


def _parse_markdown_surface_forms(block: str) -> dict[str, str]:
    return _parse_surface_forms(_parse_markdown_field(block, "surface_forms"))


def _parse_markdown_field(block: str, label: str) -> str:
    pattern = re.compile(
        rf"^\s*(?:-\s*)?(?:\*\*{re.escape(label)}:?\*\*|{re.escape(label)})\s*:?\s*(?P<body>.*?)(?=^\s*(?:-\s*)?(?:\*\*[A-Za-z0-9_ /-]+:?\*\*|[A-Za-z0-9_ /-]+)\s*:|\Z)",
        re.I | re.M | re.S,
    )
    match = pattern.search(block)
    if match is None:
        return ""
    body = match.group("body").strip()
    fenced = re.search(r"```.*?\n(.*?)```", body, re.S)
    if fenced is not None:
        return fenced.group(1).strip()
    return body


def _iter_markdown_candidates(text: str, *, source_file: str) -> Iterable[CurriculumRow]:
    matches = list(MARKDOWN_CANONICAL_RE.finditer(text))
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end]
        canonical_id = _strip_ticks(match.group("value"))
        is_a = _flatten_is_a(_parse_markdown_field(block, "is_a"))
        rpn_sketch = (
            _parse_markdown_field(block, "rpn_sketch")
            or _parse_markdown_field(block, "RPN Sketch")
            or _parse_markdown_field(block, "RPN")
        )
        surface_forms = _parse_markdown_surface_forms(block)
        if not canonical_id or not rpn_sketch or not surface_forms:
            continue
        symlinks = _flatten_symlinks(_parse_markdown_field(block, "symlinks"))
        saudades = _clean_text(_parse_markdown_field(block, "saudades"))
        row = CurriculumRow(
            canonical_id=canonical_id,
            source_file=source_file,
            source_line=_line_of_offset(text, start),
            is_a=is_a,
            rpn_sketch=rpn_sketch,
            surface_forms=surface_forms,
            symlinks=symlinks,
            saudades=saudades,
            domain=_infer_domain(source_file, canonical_id),
        )
        yield row


def parse_curriculum_file(path: str | Path) -> CurriculumPayload:
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8")
    rows: list[CurriculumRow] = []
    seen: set[str] = set()
    for iterator in (
        _iter_fenced_candidates(text, source_file=resolved.name),
        _iter_inline_json_candidates(text, source_file=resolved.name),
        _iter_markdown_candidates(text, source_file=resolved.name),
    ):
        for row in iterator:
            if row.canonical_id in seen:
                continue
            seen.add(row.canonical_id)
            rows.append(row)
    rows.sort(key=lambda item: (item.source_line, item.canonical_id))
    skipped: list[str] = []
    if not rows:
        skipped.append(f"no_structured_rows:{resolved.name}")
    return CurriculumPayload(rows=tuple(rows), skipped=tuple(skipped))


def build_meaning_star(row: CurriculumRow) -> MeaningCentricStar:
    surface_forms = {
        language: SurfaceForm(
            word_ref=text,
            char_refs=[
                canonical_char_star_id(char)
                for char in text
                if char.strip()
            ],
        )
        for language, text in sorted(row.surface_forms.items())
        if _clean_text(text)
    }
    return MeaningCentricStar(
        star_id=row.canonical_id,
        meaning_class=(row.is_a[0] if row.is_a else canonical_slug(row.canonical_id.split("_", 1)[0] or "concept")),
        meaning_rpn=row.rpn_sketch,
        domain=row.domain,
        taxonomy_refs=list(row.is_a),
        surface_forms=surface_forms,
        component_refs=list(row.symlinks),
        untranslatable_languages=(["global"] if row.saudades and row.saudades.lower() not in {"false", "no", "0"} else []),
        context_id=row.context_id,
        ethical_trit=row.ethical_trit,
    )


__all__ = [
    "CurriculumPayload",
    "CurriculumRow",
    "build_meaning_star",
    "parse_curriculum_file",
]
