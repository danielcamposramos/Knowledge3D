#!/usr/bin/env python3
"""Ingest the remaining crafted HS curriculum TEMP files into k3d_canonical."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.ingestion.canonical_lookup import CanonicalLookup  # noqa: E402
from knowledge3d.ingestion.hs_curriculum_parser import build_meaning_star, parse_curriculum_file  # noqa: E402


CURRICULUM_SOURCES: tuple[tuple[str, str], ...] = (
    ("TEMP/KIMI_HS_NATURAL_SCIENCES_PHYS_CHEM_BIO_2026-04-13.md", "hs_natural_sciences"),
    ("TEMP/KIMI_HS_EARTH_SPACE_ENVIRONMENTAL_2026-04-13.md", "hs_earth_space_environmental"),
    ("TEMP/KIMI_HS_HISTORY_GEOGRAPHY_CIVICS_ECONOMICS_2026-04-13.md", "hs_history_geography_civics_economics"),
    ("TEMP/KIMI_HS_HUMANITIES_LIT_PHIL_RELIGION_ARTS_2026-04-13.md", "hs_humanities_lit_phil_religion_arts"),
    ("TEMP/KIMI_HS_LANGUAGES_LINGUISTICS_2026-04-13.md", "hs_languages_linguistics"),
    ("TEMP/KIMI_HS_APPLIED_CS_HEALTH_PSYCH_SOCIOLOGY_2026-04-13.md", "hs_applied_cs_health_psych_sociology"),
    ("TEMP/KIMI_HS_CROSSCULTURAL_SAUDADES_CALENDAR_EXAMS_PROVERBS_2026-04-13.md", "hs_crosscultural_glue"),
    ("TEMP/KIMI_ARC_REASONING_PRIMITIVES_CLUSTER_2026-04-14.md", "arc_reasoning_primitives"),
)

SUMMARY_PATH = Path("/K3D/Knowledge3D.local/reports/hs_curriculum_remaining_ingest.json")


def _write_summary(payload: dict[str, object]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def ingest_remaining_curriculum(
    lookup: CanonicalLookup,
    *,
    write: bool = False,
) -> dict[str, object]:
    parsed: list[tuple[Path, str, object]] = []
    summary_files: dict[str, dict[str, object]] = {}
    total_rows = 0
    for relative_path, subkind in CURRICULUM_SOURCES:
        source = REPO_ROOT / relative_path
        payload = parse_curriculum_file(source)
        parsed.append((source, subkind, payload))
        total_rows += len(payload.rows)
        summary_files[source.name] = {
            "subkind": subkind,
            "rows": len(payload.rows),
            "skipped": list(payload.skipped),
        }

    if not write:
        summary = {
            "phase": "dry_run",
            "files": summary_files,
            "total_rows": total_rows,
        }
        _write_summary(summary)
        return summary

    written = 0
    by_subkind: dict[str, int] = {}
    for source, subkind, payload in parsed:
        for row in payload.rows:
            star = build_meaning_star(row)
            lookup.register(
                kind="meaning_star",
                key=row.canonical_id,
                star_id=star.star_id,
                metadata={
                    "context_id": int(row.context_id),
                    "ethical_trit": int(row.ethical_trit),
                    "subkind": subkind,
                    "source_file": row.source_file,
                    "source_line": int(row.source_line),
                    "is_a": list(row.is_a),
                    "rpn_sketch": row.rpn_sketch,
                    "surface_forms_raw": dict(row.surface_forms),
                    "symlink_refs": list(row.symlinks),
                    "saudades": row.saudades,
                    "meaning_star": star.to_dict(),
                    "meaning_star_id": star.star_id,
                    "domain": row.domain,
                },
            )
            written += 1
            by_subkind[subkind] = int(by_subkind.get(subkind, 0)) + 1

    confirmed = 0
    misses: list[str] = []
    for _, _, payload in parsed:
        for row in payload.rows:
            if lookup.star_id_exists(row.canonical_id):
                confirmed += 1
            else:
                misses.append(row.canonical_id)

    summary = {
        "phase": "write",
        "files": summary_files,
        "total_rows": total_rows,
        "written": written,
        "confirmed": confirmed,
        "misses": misses,
        "by_subkind": by_subkind,
    }
    _write_summary(summary)
    if misses:
        raise RuntimeError(f"hs_curriculum_remaining_confirmation_failed:{','.join(misses[:10])}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    write = "--write" in args
    summary = ingest_remaining_curriculum(CanonicalLookup(), write=write)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
