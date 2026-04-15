#!/usr/bin/env python3
"""Ingest the Batch 7 reasoning taxonomy into the canonical registry."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.ingestion.canonical_lookup import (  # noqa: E402
    CanonicalLookup,
    canonical_entry_id,
    canonical_grammar_template_id,
    canonical_slug,
)
from knowledge3d.ingestion.reasoning_taxonomy_parser import (  # noqa: E402
    CANONICAL_LANGUAGES,
    CataloguePayload,
    CanonicalStarRow,
    LogicOperatorCrossLink,
    PeriphrasticTemplate,
    parse_catalogue,
)
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar  # noqa: E402


CATALOGUE_ORDER = (
    "TEMP/KIMI_KNOWLEDGE_AUTOMATED_REASONING_2026-04-13.md",
    "TEMP/KIMI_KNOWLEDGE_AML_AND_SOLVERS_2026-04-13.md",
    "TEMP/KIMI_KNOWLEDGE_HEURISTICS_AND_METAHEURISTICS_2026-04-13.md",
    "TEMP/KIMI_KNOWLEDGE_EXTENSION_AML_HEURISTICS_REASONING_2026-04-13.md",
)
ALLOWLIST_PATH = REPO_ROOT / "knowledge3d" / "ingestion" / "reasoning_taxonomy_allowlist.txt"


def load_allowlist(path: Path = ALLOWLIST_PATH) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    entries: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.split("#", 1)[0].strip()
        if cleaned:
            entries.append(cleaned)
    return frozenset(entries)


def parse_all_catalogues(paths: Iterable[str | Path] = CATALOGUE_ORDER) -> tuple[CataloguePayload, ...]:
    return tuple(parse_catalogue(REPO_ROOT / Path(path)) for path in paths)


def _collect_local_star_ids(payloads: Iterable[CataloguePayload]) -> set[str]:
    star_ids: set[str] = set()
    for payload in payloads:
        for star in payload.stars:
            if star.star_id in star_ids:
                raise ValueError(f"reasoning_taxonomy_duplicate_star_id:{star.star_id}:{star.source_file}:{star.source_line}")
            star_ids.add(star.star_id)
    return star_ids


def _iter_refs(row: CanonicalStarRow) -> tuple[str, ...]:
    refs: list[str] = []
    refs.extend(row.grammar_refs)
    refs.extend(row.taxonomy_refs)
    refs.extend(row.meta_refs)
    refs.extend(row.component_refs)
    deduped: list[str] = []
    for ref in refs:
        cleaned = str(ref or "").strip()
        if not cleaned or cleaned == row.star_id:
            continue
        if cleaned not in deduped:
            deduped.append(cleaned)
    return tuple(deduped)


def find_dangling_references(
    payloads: Iterable[CataloguePayload],
    lookup: CanonicalLookup,
    allowlist: frozenset[str],
) -> tuple[str, ...]:
    local_star_ids = _collect_local_star_ids(payloads)
    missing: list[str] = []
    for payload in payloads:
        for row in payload.stars:
            for ref in _iter_refs(row):
                if ref in local_star_ids or ref in allowlist:
                    continue
                if lookup.star_id_exists(ref):
                    continue
                if ref not in missing:
                    missing.append(ref)
    return tuple(sorted(missing))


def enforce_integrity(
    payloads: Iterable[CataloguePayload],
    lookup: CanonicalLookup,
    allowlist: frozenset[str],
) -> None:
    missing = find_dangling_references(payloads, lookup, allowlist)
    if missing:
        raise ValueError(f"reasoning_taxonomy_dangling_refs:{','.join(missing)}")


def _meaning_star_from_row(row: CanonicalStarRow) -> MeaningCentricStar:
    return MeaningCentricStar(
        star_id=row.star_id,
        meaning_class=row.meaning_class,
        meaning_rpn=row.meaning_rpn_sketch,
        domain=row.domain,
        taxonomy_refs=list(row.taxonomy_refs),
        grammar_refs=list(row.grammar_refs),
        meta_refs=list(row.meta_refs),
        component_refs=list(row.component_refs),
        untranslatable_languages=list(row.saudades),
        context_id=row.context_id,
        ethical_trit=row.ethical_trit,
    )


def _register_star(lookup: CanonicalLookup, row: CanonicalStarRow) -> None:
    meaning_star = _meaning_star_from_row(row)
    lookup.register(
        kind="meaning_star",
        key=row.star_id,
        star_id=row.star_id,
        metadata={
            "source_file": row.source_file,
            "source_line": row.source_line,
            "context_id": row.context_id,
            "ethical_trit": row.ethical_trit,
            "surface_forms_raw": dict(row.surface_forms),
            "meaning_star": meaning_star.to_dict(),
            "meaning_star_id": row.star_id,
            "grammar_refs": list(row.grammar_refs),
            "taxonomy_refs": list(row.taxonomy_refs),
            "meta_refs": list(row.meta_refs),
            "component_refs": list(row.component_refs),
            "saudades": list(row.saudades),
        },
    )


def ingest_catalogues(payloads: Iterable[CataloguePayload], lookup: CanonicalLookup) -> int:
    count = 0
    for payload in payloads:
        for row in payload.stars:
            _register_star(lookup, row)
            count += 1
    return count


def _register_logic_cross_links(payloads: Iterable[CataloguePayload], lookup: CanonicalLookup) -> int:
    count = 0
    for payload in payloads:
        for cross_link in payload.logic_operators:
            count += _register_logic_cross_link(cross_link, lookup)
    return count


def _register_logic_cross_link(cross_link: LogicOperatorCrossLink, lookup: CanonicalLookup) -> int:
    count = 0
    for related_star_id in cross_link.related_star_ids:
        left_key = f"{cross_link.star_id}->{related_star_id}:grammar_refs"
        right_key = f"{related_star_id}->{cross_link.star_id}:grammar_refs"
        for key, left, right in (
            (left_key, cross_link.star_id, related_star_id),
            (right_key, related_star_id, cross_link.star_id),
        ):
            lookup.register(
                kind="reasoning_taxonomy_symlink",
                key=key,
                star_id=f"reasoning_taxonomy_symlink_{canonical_slug(left)}_{canonical_slug(right)}",
                metadata={
                    "left_star_id": left,
                    "right_star_id": right,
                    "symbol": cross_link.symbol,
                    "symlink_field": "grammar_refs",
                    "context_id": 0,
                    "ethical_trit": 0,
                },
            )
            count += 1
    return count


def _register_templates(payloads: Iterable[CataloguePayload], lookup: CanonicalLookup) -> int:
    count = 0
    for payload in payloads:
        for template in payload.periphrastic_templates:
            if template.language not in CANONICAL_LANGUAGES:
                continue
            template_name = f"reasoning_taxonomy_{template.star_id}"
            lookup.register(
                kind="grammar_template",
                key=f"{template.language}:{template_name}",
                star_id=canonical_grammar_template_id(template.language, template_name),
                metadata={
                    "language": template.language,
                    "source_star_id": template.star_id,
                    "template_text": template.template_text,
                    "context_id": 0,
                    "ethical_trit": 0,
                },
            )
            count += 1
    return count


def ingest_reasoning_taxonomy(
    lookup: CanonicalLookup | None = None,
    *,
    allowlist_path: Path = ALLOWLIST_PATH,
    paths: Iterable[str | Path] = CATALOGUE_ORDER,
) -> dict[str, object]:
    registry = lookup or CanonicalLookup()
    allowlist = load_allowlist(allowlist_path)
    payloads = parse_all_catalogues(paths)
    enforce_integrity(payloads, registry, allowlist)
    star_count = ingest_catalogues(payloads, registry)
    symlink_count = _register_logic_cross_links(payloads, registry)
    template_count = _register_templates(payloads, registry)
    return {
        "catalogues": [payload.source_file for payload in payloads],
        "stars": star_count,
        "symlinks": symlink_count,
        "templates": template_count,
        "allowlist_entries": len(allowlist),
    }


def main() -> None:
    summary = ingest_reasoning_taxonomy()
    print(summary)


if __name__ == "__main__":
    main()
