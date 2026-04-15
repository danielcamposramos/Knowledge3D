"""Batch 7 reasoning-taxonomy markdown parser.

This module is ingestion-only. It parses the KIMI reasoning catalogue markdown
files into immutable dataclasses without touching Qdrant or runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


CANONICAL_LANGUAGES: tuple[str, ...] = ("en", "pt", "es", "fr", "de", "it", "ja", "zh", "ru")
_EMPTY_MARKERS = frozenset({"", "-", "—", "`—`", "[]", "`[]`", "n/a", "none", "null"})
_LOGIC_SYMBOL_TO_STAR_ID = {
    "∀": "logic_forall",
    "∃": "logic_exists",
    "→": "logic_implies",
    "∧": "logic_and",
    "∨": "logic_or",
    "¬": "logic_not",
    "≡": "logic_iff",
    "⊢": "logic_turnstile",
    "⊨": "logic_entails",
    "λ": "concept_lambda_calculus",
}
_HEADER_ALIASES = {
    "#": "ordinal",
    "star_id": "star_id",
    "star_id_": "star_id",
    "starid": "star_id",
    "star_ids": "star_id",
    "star_id__": "star_id",
    "star_id___": "star_id",
    "starid_": "star_id",
    "star_id____": "star_id",
    "star_id_____": "star_id",
    "starid__": "star_id",
    "star_id______": "star_id",
    "star_id_______": "star_id",
    "star_id________": "star_id",
    "star_id_________": "star_id",
    "star_id__________": "star_id",
    "star_id___________": "star_id",
    "star_id____________": "star_id",
    "star_id_____________": "star_id",
    "star_id______________": "star_id",
    "star_id_______________": "star_id",
    "star_id________________": "star_id",
    "star_id_________________": "star_id",
    "star_id__________________": "star_id",
    "star_id___________________": "star_id",
    "star_id____________________": "star_id",
    "star_id_____________________": "star_id",
    "star_id______________________": "star_id",
    "star_id_______________________": "star_id",
    "star_id________________________": "star_id",
    "star_id_________________________": "star_id",
    "star_id__________________________": "star_id",
    "star_id___________________________": "star_id",
    "star_id____________________________": "star_id",
    "star_id_____________________________": "star_id",
    "star_id______________________________": "star_id",
    "star_id_______________________________": "star_id",
    "star_id________________________________": "star_id",
    "star_id_________________________________": "star_id",
    "star_id__________________________________": "star_id",
    "star_id___________________________________": "star_id",
    "star_id____________________________________": "star_id",
    "class": "meaning_class",
    "meaning_class": "meaning_class",
    "domain": "domain",
    "domain_path": "domain",
    "meaning_rpn_sketch": "meaning_rpn_sketch",
    "meaning_rpn_sketch_": "meaning_rpn_sketch",
    "meaning_rpn_sketch__": "meaning_rpn_sketch",
    "meaning_rpn_sketch___": "meaning_rpn_sketch",
    "meaning_rpn_sketch____": "meaning_rpn_sketch",
    "meaning_rpn_sketch_____": "meaning_rpn_sketch",
    "meaning_rpn_sketch______": "meaning_rpn_sketch",
    "meaning_rpn_sketch_______": "meaning_rpn_sketch",
    "meaning_rpn_sketch________": "meaning_rpn_sketch",
    "meaning_rpn_sketch_________": "meaning_rpn_sketch",
    "meaning_rpn": "meaning_rpn_sketch",
    "rpn_sketch": "meaning_rpn_sketch",
    "key_grammar_refs": "grammar_refs",
    "grammar_refs": "grammar_refs",
    "taxonomy_refs": "taxonomy_refs",
    "is_a": "taxonomy_refs",
    "meta_refs": "meta_refs",
    "component_refs": "component_refs",
    "phase_7_a_1_symlinks": "component_refs",
    "surface_forms": "surface_forms",
    "surface_forms_excerpt": "surface_forms",
    "surface_forms_9_lang": "surface_forms",
    "saudades": "saudades",
    "dangling_ref_risk": "dangling_ref_risk",
    "periphrastic_notes": "periphrastic_notes",
}


@dataclass(frozen=True)
class CanonicalStarRow:
    star_id: str
    meaning_class: str
    domain: str
    meaning_rpn_sketch: str
    grammar_refs: tuple[str, ...]
    taxonomy_refs: tuple[str, ...]
    meta_refs: tuple[str, ...]
    component_refs: tuple[str, ...] = ()
    surface_forms: tuple[tuple[str, str], ...] = ()
    saudades: tuple[str, ...] = ()
    context_id: int = 0
    ethical_trit: int = 0
    source_file: str = ""
    source_line: int = 0


@dataclass(frozen=True)
class LogicOperatorCrossLink:
    symbol: str
    star_id: str
    related_star_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PeriphrasticTemplate:
    star_id: str
    language: str
    template_text: str


@dataclass(frozen=True)
class CataloguePayload:
    source_file: str
    stars: tuple[CanonicalStarRow, ...] = ()
    logic_operators: tuple[LogicOperatorCrossLink, ...] = ()
    periphrastic_templates: tuple[PeriphrasticTemplate, ...] = ()
    dangling_risks: tuple[str, ...] = ()


def _strip_markdown(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = cleaned.replace("`", "")
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    return cleaned.strip()


def _normalized_header(text: str) -> str:
    cleaned = _strip_markdown(text).lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    return _HEADER_ALIASES.get(cleaned, cleaned)


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        raise ValueError(f"not_markdown_row:{line!r}")
    cells = stripped.strip("|").split("|")
    return [cell.strip() for cell in cells]


def _is_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False
    for cell in cells:
        token = cell.replace("-", "").replace(":", "").strip()
        if token:
            return False
    return True


def _iter_markdown_tables(lines: list[str]) -> tuple[tuple[int, list[str], list[list[str]]], ...]:
    tables: list[tuple[int, list[str], list[list[str]]]] = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        start = index
        block: list[str] = []
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            block.append(lines[index].rstrip("\n"))
            index += 1
        if len(block) < 2:
            continue
        header = _split_markdown_row(block[0])
        separator = _split_markdown_row(block[1])
        if not _is_separator_row(separator):
            continue
        rows = [_split_markdown_row(line) for line in block[2:]]
        tables.append((start + 1, header, rows))
    return tuple(tables)


def _clean_cell(text: str) -> str:
    cleaned = _strip_markdown(text)
    if cleaned.lower() in _EMPTY_MARKERS:
        return ""
    return cleaned


def _split_ref_tokens(cell: str) -> tuple[str, ...]:
    cleaned = _clean_cell(cell)
    if not cleaned:
        return ()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1].strip()
    if not cleaned:
        return ()
    fragments = re.split(r"[;,]", cleaned)
    refs: list[str] = []
    for fragment in fragments:
        token = _strip_markdown(fragment).strip()
        if not token or token.lower() in _EMPTY_MARKERS:
            continue
        if token in _LOGIC_SYMBOL_TO_STAR_ID:
            refs.append(_LOGIC_SYMBOL_TO_STAR_ID[token])
            continue
        if " " in token and all(part in _LOGIC_SYMBOL_TO_STAR_ID for part in token.split()):
            refs.extend(_LOGIC_SYMBOL_TO_STAR_ID[part] for part in token.split())
            continue
        token = token.strip("[]()")
        if token:
            refs.append(token)
    deduped: list[str] = []
    for ref in refs:
        if ref not in deduped:
            deduped.append(ref)
    return tuple(deduped)


def _parse_saudades(cell: str) -> tuple[str, ...]:
    cleaned = _clean_cell(cell)
    if not cleaned:
        return ()
    codes = re.findall(r"\b([a-z]{2})\b", cleaned.lower())
    if "[saudades:" in cleaned.lower():
        codes.extend(re.findall(r"\[saudades:([a-z]{2})\]", cleaned.lower()))
    deduped = [code for code in CANONICAL_LANGUAGES if code in codes]
    return tuple(deduped)


def _parse_surface_forms(cell: str) -> tuple[tuple[str, str], ...]:
    cleaned = _clean_cell(cell)
    if not cleaned:
        return ()
    matches = re.findall(
        r"(?:\*\*)?([a-z]{2})(?:\*\*)?\s*:\s*(.+?)(?=(?:,\s*(?:\*\*)?[a-z]{2}(?:\*\*)?\s*:)|(?:;\s*(?:\*\*)?[a-z]{2}(?:\*\*)?\s*:)|$)",
        cell,
        flags=re.IGNORECASE,
    )
    forms: list[tuple[str, str]] = []
    for language, text in matches:
        lang = str(language).strip().lower()
        if lang not in CANONICAL_LANGUAGES:
            continue
        value = _strip_markdown(text)
        value = re.sub(r"\s+", " ", value).strip()
        if not value:
            continue
        forms.append((lang, value))
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in forms:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return tuple(deduped)


def _infer_meaning_class(star_id: str, explicit: str) -> str:
    cleaned = _clean_cell(explicit).lower()
    if cleaned:
        return cleaned
    key = str(star_id).strip().lower()
    if key.startswith("logic_"):
        return "form"
    if key.startswith("star_op_"):
        return "action"
    if key.startswith("meta_"):
        return "meta"
    if key.startswith("relation_"):
        return "relation"
    if key.startswith("property_"):
        return "property"
    return "concept"


def _infer_domain(path: Path, star_id: str, explicit: str) -> str:
    cleaned = _clean_cell(explicit)
    if cleaned:
        return cleaned
    name = path.name.lower()
    key = str(star_id).strip().lower()
    if key.startswith("aml_"):
        return "Math/Optimization/AML"
    if "solver" in key:
        return "Math/Optimization/Solver"
    if key.startswith("logic_"):
        return "Math/Logic/Symbols"
    if "automated_reasoning" in name:
        return "Logic/AutomatedReasoning"
    if "heuristics" in name:
        return "Math/Heuristics"
    return "Logic/ReasoningExtension"


def _ethical_trit_for_row(star_id: str, domain: str, text: str) -> int:
    joined = f"{star_id} {domain} {text}".lower()
    if "defeasible" not in joined:
        return 0
    domain_key = str(domain or "").lower()
    if "argumentation" in domain_key or "knowledgesemantics" in domain_key or "knowledge" in domain_key:
        return 1
    return 0


def _extract_dangling_risks(lines: list[str]) -> tuple[str, ...]:
    risks: list[str] = []
    active = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and "dangling reference risk" in stripped.lower():
            active = True
            continue
        if active and stripped.startswith("#"):
            break
        if not active:
            continue
        if not stripped:
            continue
        if stripped.startswith(("-", "*")) or re.match(r"^\d+\.", stripped):
            risks.append(stripped.lstrip("-* ").strip())
    return tuple(risks)


def _looks_like_star_table(headers: tuple[str, ...]) -> bool:
    return "star_id" in headers and (
        "meaning_rpn_sketch" in headers or "surface_forms" in headers or "taxonomy_refs" in headers
    )


def _row_to_star(path: Path, header_map: dict[str, int], row: list[str], source_line: int) -> CanonicalStarRow | None:
    def cell(name: str) -> str:
        index = header_map.get(name)
        if index is None or index >= len(row):
            return ""
        return row[index]

    ordinal = _clean_cell(cell("ordinal"))
    star_id = _clean_cell(cell("star_id"))
    if ordinal and not ordinal.isdigit():
        return None
    if not star_id:
        return None
    if star_id.startswith("**") or " " in star_id and not star_id.startswith(("concept_", "logic_", "aml_", "solver_", "theorem_", "star_op_")):
        return None
    meaning_rpn = _clean_cell(cell("meaning_rpn_sketch"))
    meaning_class = _infer_meaning_class(star_id, cell("meaning_class"))
    domain = _infer_domain(path, star_id, cell("domain"))
    grammar_refs = _split_ref_tokens(cell("grammar_refs"))
    taxonomy_refs = _split_ref_tokens(cell("taxonomy_refs"))
    meta_refs = _split_ref_tokens(cell("meta_refs"))
    component_refs = _split_ref_tokens(cell("component_refs"))
    surface_forms = _parse_surface_forms(cell("surface_forms"))
    saudades = _parse_saudades(cell("saudades"))
    joined_text = " ".join(
        value
        for value in (
            meaning_rpn,
            cell("surface_forms"),
            cell("saudades"),
            cell("periphrastic_notes"),
            cell("dangling_ref_risk"),
        )
        if value
    )
    ethical_trit = _ethical_trit_for_row(star_id, domain, joined_text)
    return CanonicalStarRow(
        star_id=star_id,
        meaning_class=meaning_class,
        domain=domain,
        meaning_rpn_sketch=meaning_rpn,
        grammar_refs=grammar_refs,
        taxonomy_refs=taxonomy_refs,
        meta_refs=meta_refs,
        component_refs=component_refs,
        surface_forms=surface_forms,
        saudades=saudades,
        context_id=0,
        ethical_trit=ethical_trit,
        source_file=str(path),
        source_line=source_line,
    )


def _collect_logic_cross_links(rows: tuple[CanonicalStarRow, ...]) -> tuple[LogicOperatorCrossLink, ...]:
    related_by_symbol: dict[str, list[str]] = {}
    for row in rows:
        for grammar_ref in row.grammar_refs:
            for symbol, star_id in _LOGIC_SYMBOL_TO_STAR_ID.items():
                if grammar_ref != star_id:
                    continue
                related_by_symbol.setdefault(symbol, []).append(row.star_id)
    links: list[LogicOperatorCrossLink] = []
    for symbol, star_id in _LOGIC_SYMBOL_TO_STAR_ID.items():
        related = tuple(sorted(set(related_by_symbol.get(symbol, ()))))
        if not related:
            continue
        links.append(LogicOperatorCrossLink(symbol=symbol, star_id=star_id, related_star_ids=related))
    return tuple(links)


def _collect_periphrastic_templates(rows: tuple[CanonicalStarRow, ...]) -> tuple[PeriphrasticTemplate, ...]:
    templates: list[PeriphrasticTemplate] = []
    for row in rows:
        surface_map = dict(row.surface_forms)
        saudades = set(row.saudades)
        for language, text in row.surface_forms:
            if language not in CANONICAL_LANGUAGES:
                continue
            if language in saudades or " " in text or "-" in text or "/" in text or language in {"ja", "zh"}:
                templates.append(
                    PeriphrasticTemplate(
                        star_id=row.star_id,
                        language=language,
                        template_text=text,
                    )
                )
            elif surface_map and len(text) > 24:
                templates.append(
                    PeriphrasticTemplate(
                        star_id=row.star_id,
                        language=language,
                        template_text=text,
                    )
                )
    deduped: list[PeriphrasticTemplate] = []
    seen: set[tuple[str, str, str]] = set()
    for item in templates:
        key = (item.star_id, item.language, item.template_text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return tuple(deduped)


def parse_catalogue(path: Path) -> CataloguePayload:
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8")
    lines = text.splitlines()
    stars: list[CanonicalStarRow] = []
    for start_line, header_cells, row_cells in _iter_markdown_tables(lines):
        headers = tuple(_normalized_header(cell) for cell in header_cells)
        if "star_id" in headers and not _looks_like_star_table(headers):
            raise ValueError(f"reasoning_taxonomy_missing_required_columns:{resolved}:{start_line}")
        if not _looks_like_star_table(headers):
            continue
        header_map = {name: index for index, name in enumerate(headers)}
        if "star_id" not in header_map:
            raise ValueError(f"reasoning_taxonomy_missing_star_id:{resolved}:{start_line}")
        if "meaning_rpn_sketch" not in header_map and "surface_forms" not in header_map:
            raise ValueError(f"reasoning_taxonomy_missing_rpn_or_surface:{resolved}:{start_line}")
        for offset, row in enumerate(row_cells):
            star = _row_to_star(resolved, header_map, row, start_line + 2 + offset)
            if star is None:
                continue
            stars.append(star)
    seen_star_ids: set[str] = set()
    for star in stars:
        if star.star_id in seen_star_ids:
            raise ValueError(f"reasoning_taxonomy_duplicate_star_id:{star.star_id}:{star.source_file}:{star.source_line}")
        seen_star_ids.add(star.star_id)
    star_tuple = tuple(stars)
    return CataloguePayload(
        source_file=str(resolved),
        stars=star_tuple,
        logic_operators=_collect_logic_cross_links(star_tuple),
        periphrastic_templates=_collect_periphrastic_templates(star_tuple),
        dangling_risks=_extract_dangling_risks(lines),
    )


__all__ = [
    "CANONICAL_LANGUAGES",
    "CanonicalStarRow",
    "CataloguePayload",
    "LogicOperatorCrossLink",
    "PeriphrasticTemplate",
    "parse_catalogue",
]
