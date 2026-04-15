#!/usr/bin/env python3
"""Audit Phase 7.A.1 seed coverage for Batch 8 math aliases."""

from __future__ import annotations

import json
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


REPORT_PATH = Path("/K3D/Knowledge3D.local/reports/batch8_phase7a1_audit.json")


def run_audit(lookup: CanonicalLookup) -> dict[str, object]:
    rows = [("char", alias, target) for alias, target in sorted(CHAR_ALIASES.items())]
    rows.extend(("letter", alias, target) for alias, target in sorted(LETTER_ALIASES.items()))
    rows.extend(("symbol", alias, target) for alias, target in sorted(SYMBOL_ALIASES.items()))
    rows.extend(("constant", alias, target) for alias, target in sorted(CONSTANT_ALIASES.items()))
    rows.extend(("concept", alias, star_id) for alias, star_id in sorted(CONCEPT_MEANING_STARS.items()))
    rows.extend(("meaning_star", key, star_id) for key, star_id in sorted(CONCEPT_MEANING_STARS.items()))
    missing: list[dict[str, str]] = []
    present = 0
    for kind, alias, expected in rows:
        if lookup.star_id_exists(expected):
            present += 1
            continue
        missing.append({"kind": kind, "alias": alias, "expected": expected, "reason": "not_in_canonical"})
    payload = {
        "checked": len(rows),
        "present": present,
        "missing": missing,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    payload = run_audit(CanonicalLookup())
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["missing"]:
        print("python scripts/seed_batch8_canonical_math_aliases.py")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
