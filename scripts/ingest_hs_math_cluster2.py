#!/usr/bin/env python3
"""Batch 9 HS Math Cluster 2 ingestion driver."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.ingestion.canonical_lookup import CanonicalLookup, canonical_entry_id  # noqa: E402
from knowledge3d.ingestion.hs_math_parser import parse_cluster1_bullets_with_diagnostics  # noqa: E402
from knowledge3d.ingestion.math_canonical_id import normalise_canonical_id  # noqa: E402
from knowledge3d.ingestion.math_symlink_resolver import MathSymlinkResolveError, MathSymlinkResolver  # noqa: E402
from knowledge3d.ingestion.rpn_sketch_lexer import write_coverage_report  # noqa: E402


DEFAULT_SOURCE = REPO_ROOT / "TEMP" / "KIMI_MATH_HS_CLUSTER2_GEOMETRY_TRIG_2026-04-13.md"
DEFAULT_ALLOWLIST = REPO_ROOT / "knowledge3d" / "ingestion" / "math_symlink_allowlist.txt"
SUMMARY_PATH = Path("/K3D/Knowledge3D.local/reports/batch9_cluster2_ingest.json")


@dataclass(frozen=True)
class CachedMathRow:
    canonical_key: str
    star_id: str
    category: str
    row: object


class Batch9Cluster2DriverError(RuntimeError):
    def __init__(self, code: int, message: str, payload: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = int(code)
        self.payload = dict(payload or {})


def _load_allowlist(path: Path) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.split("#", 1)[0].strip()
        if cleaned:
            rows.append(cleaned)
    return frozenset(rows)


def _write_summary(payload: dict[str, object]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _edge_payload(lookup: object, kind: str, key: str) -> dict[str, object]:
    if hasattr(lookup, "_scroll_exact"):
        return dict(getattr(lookup, "_scroll_exact")(kind=kind, key=key) or {})
    if hasattr(lookup, "records"):
        return dict(getattr(lookup, "records").get((kind, key), {}))
    return {}


def run_cluster2_ingestion(
    lookup: CanonicalLookup,
    *,
    source: Path = DEFAULT_SOURCE,
    allowlist_path: Path = DEFAULT_ALLOWLIST,
    write: bool = False,
) -> dict[str, object]:
    diagnostics = parse_cluster1_bullets_with_diagnostics(source.read_text(encoding="utf-8"), source_file=source.name)
    allowlist = _load_allowlist(allowlist_path)
    cached_rows: list[CachedMathRow] = []
    category_counts: dict[str, int] = {}
    concept_keys: set[str] = set()
    for row in diagnostics.rows:
        category, canonical_key = normalise_canonical_id(row.canonical_id_raw)
        cached_rows.append(CachedMathRow(canonical_key=canonical_key, star_id=f"math_{canonical_key}", category=category, row=row))
        category_counts[category] = category_counts.get(category, 0) + 1
        if canonical_key.startswith("concept_"):
            concept_keys.add(canonical_key)

    resolver = MathSymlinkResolver(lookup, allowlist_path=None)
    hard_misses: list[dict[str, object]] = []
    allowlisted: list[str] = []
    forward_refs: list[str] = []
    for cached in cached_rows:
        for raw_ref, norm_ref in zip(cached.row.symlink_refs_raw, cached.row.symlink_refs_norm):
            if raw_ref in allowlist:
                allowlisted.append(raw_ref)
                continue
            try:
                resolver.resolve(norm_ref)
            except MathSymlinkResolveError as exc:
                if norm_ref.startswith("concept::") and f"concept_{norm_ref.split('::', 1)[1]}" in concept_keys:
                    forward_refs.append(norm_ref)
                    continue
                hard_misses.append({"source_star_id": cached.star_id, "raw_ref": raw_ref, "norm_ref": norm_ref, "error": str(exc)})
    if hard_misses:
        payload = {
            "phase": "pass0",
            "hard_misses": hard_misses,
            "skipped_blocks": list(diagnostics.skipped_lines),
        }
        _write_summary(payload)
        raise Batch9Cluster2DriverError(1, "batch9_cluster2_pass0_hard_miss", payload)

    if not write:
        coverage = write_coverage_report([cached.row.rpn_sketch_raw for cached in cached_rows])
        payload = {
            "phase": "dry_run",
            "rows": len(cached_rows),
            "category_counts": category_counts,
            "allowlisted_symlinks": allowlisted,
            "forward_refs": forward_refs,
            "skipped_blocks": list(diagnostics.skipped_lines),
            "coverage": coverage,
        }
        _write_summary(payload)
        return payload

    written_stars: list[tuple[str, str]] = []
    for cached in cached_rows:
        lookup.register(
            kind="meaning_star",
            key=cached.canonical_key,
            star_id=cached.star_id,
            metadata={
                "context_id": 0,
                "ethical_trit": 0,
                "subkind": "math_hs_cluster2",
                "category": cached.category,
                "is_a": list(cached.row.is_a),
                "rpn_sketch": cached.row.rpn_sketch_raw,
                "surface_forms": dict(cached.row.surface_forms),
                "saudades": bool(cached.row.saudades),
                "source_file": cached.row.source_file,
                "source_line": cached.row.source_line,
                "symlink_refs_raw": list(cached.row.symlink_refs_raw),
                "symlink_refs_norm": list(cached.row.symlink_refs_norm),
            },
        )
        written_stars.append((cached.canonical_key, cached.star_id))

    written_edges: list[tuple[str, str]] = []
    for cached in cached_rows:
        for raw_ref, norm_ref in zip(cached.row.symlink_refs_raw, cached.row.symlink_refs_norm):
            if raw_ref in allowlist:
                continue
            resolved_star_id = resolver.resolve(norm_ref)
            if resolved_star_id is None:
                continue
            edge_key = f"{cached.star_id}::{resolved_star_id}"
            lookup.register(
                kind="math_symlink",
                key=edge_key,
                star_id=f"math_symlink_{canonical_entry_id('math_symlink', edge_key)}",
                metadata={
                    "source_star_id": cached.star_id,
                    "target_star_id": resolved_star_id,
                    "raw_ref": raw_ref,
                    "norm_ref": norm_ref,
                    "bidirectional": True,
                },
            )
            written_edges.append((edge_key, resolved_star_id))

    meaning_star_confirmed = 0
    math_symlink_confirmed = 0
    target_star_id_confirmed = 0
    confirmation_misses: list[dict[str, object]] = []
    for _, star_id in written_stars:
        if lookup.star_id_exists(star_id):
            meaning_star_confirmed += 1
        else:
            confirmation_misses.append({"kind": "meaning_star", "star_id": star_id})
    for edge_key, expected_target in written_edges:
        if lookup.exists(kind="math_symlink", key=edge_key):
            math_symlink_confirmed += 1
            payload = _edge_payload(lookup, "math_symlink", edge_key)
            metadata = dict(payload.get("metadata") or {})
            target_star_id = str(metadata.get("target_star_id") or expected_target)
            if lookup.star_id_exists(target_star_id):
                target_star_id_confirmed += 1
            else:
                confirmation_misses.append({"kind": "target_star_id", "edge_key": edge_key, "target_star_id": target_star_id})
        else:
            confirmation_misses.append({"kind": "math_symlink", "edge_key": edge_key})

    coverage = write_coverage_report([cached.row.rpn_sketch_raw for cached in cached_rows])
    payload = {
        "phase": "write",
        "rows": len(cached_rows),
        "category_counts": category_counts,
        "allowlisted_symlinks": allowlisted,
        "forward_refs": forward_refs,
        "skipped_blocks": list(diagnostics.skipped_lines),
        "meaning_star_written": len(written_stars),
        "math_symlink_written": len(written_edges),
        "confirmation": {
            "meaning_star_confirmed": meaning_star_confirmed,
            "math_symlink_confirmed": math_symlink_confirmed,
            "target_star_id_confirmed": target_star_id_confirmed,
            "misses": confirmation_misses,
        },
        "coverage": coverage,
    }
    _write_summary(payload)
    if confirmation_misses:
        raise Batch9Cluster2DriverError(3, "batch9_cluster2_pass3_confirmation_failed", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    source = DEFAULT_SOURCE
    write = "--write" in args
    if not write and "--dry-run" not in args:
        args.append("--dry-run")
    if "--source" in args:
        index = args.index("--source")
        source = (REPO_ROOT / args[index + 1]).resolve() if not Path(args[index + 1]).is_absolute() else Path(args[index + 1])
    if write and os.environ.get("K3D_QDRANT_INTEGRATION") != "1":
        raise SystemExit(2)
    try:
        summary = run_cluster2_ingestion(CanonicalLookup(), source=source, write=write)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Batch9Cluster2DriverError as exc:
        print(json.dumps(exc.payload, indent=2, sort_keys=True))
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
