#!/usr/bin/env python3
"""Seed Batch 8 canonical math aliases into k3d_canonical."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.ingestion.canonical_lookup import CanonicalLookup  # noqa: E402
from knowledge3d.ingestion.math_semantic_aliases import (  # noqa: E402
    CHAR_ALIASES,
    CONCEPT_MEANING_STARS,
    CONSTANT_ALIASES,
    LETTER_ALIASES,
    SYMBOL_ALIASES,
)


def _dispatch_kind(star_id: str) -> str:
    target = str(star_id or "").strip()
    if target.startswith("math_symbol_"):
        return "math_symbol"
    if target.startswith("char_"):
        return "char"
    return "concept"


def iter_alias_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    rows.extend((alias_name, star_id) for alias_name, star_id in CHAR_ALIASES.items())
    rows.extend((alias_name, star_id) for alias_name, star_id in LETTER_ALIASES.items())
    rows.extend((alias_name, star_id) for alias_name, star_id in SYMBOL_ALIASES.items())
    rows.extend((alias_name, star_id) for alias_name, star_id in CONSTANT_ALIASES.items())
    return rows


def iter_concept_alias_rows() -> list[tuple[str, str]]:
    return [(alias_name, star_id) for alias_name, star_id in CONCEPT_MEANING_STARS.items()]


def iter_concept_seed_rows() -> list[tuple[str, str]]:
    return [(star_id, star_id) for star_id in CONCEPT_MEANING_STARS.values()]


def seed(lookup: CanonicalLookup | None = None) -> dict[str, int]:
    registry = lookup or CanonicalLookup()
    registry.ensure_collection()
    counts = {"math_symbol": 0, "char": 0, "concept": 0, "meaning_star": 0}
    for alias_name, star_id in iter_alias_rows():
        kind = _dispatch_kind(star_id)
        registry.register(
            kind=kind,
            key=alias_name,
            star_id=star_id,
            metadata={"context_id": 0, "ethical_trit": 0, "source": "batch8_seed"},
        )
        counts[kind] += 1
    for alias_name, star_id in iter_concept_alias_rows():
        registry.register(
            kind="concept",
            key=alias_name,
            star_id=star_id,
            metadata={"context_id": 0, "ethical_trit": 0, "source": "batch8_seed"},
        )
        counts["concept"] += 1
    for key, star_id in iter_concept_seed_rows():
        registry.register(
            kind="meaning_star",
            key=key,
            star_id=star_id,
            metadata={"context_id": 0, "ethical_trit": 0, "source": "batch8_seed"},
        )
        counts["meaning_star"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Batch 8 canonical math aliases into k3d_canonical.")
    parser.add_argument("--url", default="http://localhost:6333")
    args = parser.parse_args()
    counts = seed(CanonicalLookup(url=args.url))
    total = sum(counts.values())
    print(
        f"seeded: total={total} "
        f"math_symbol={counts['math_symbol']} "
        f"char={counts['char']} "
        f"concept={counts['concept']} "
        f"meaning_star={counts['meaning_star']}"
    )


if __name__ == "__main__":
    main()
