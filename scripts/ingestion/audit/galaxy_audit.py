#!/usr/bin/env python3
"""D1 live-galaxy audit for proceduralization and normalization.

Dependencies:
- Python 3 standard library only.

This script audits the live JSONL galaxy root, writes deterministic JSONL
artifacts, and can regenerate the Markdown report from those staged outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STORAGE_ROOT = Path("/K3D/Knowledge3D.local/galaxies")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts" / "ingestion" / "staging" / "D1_audit"
DEFAULT_REPORT_PATH = REPO_ROOT / "TEMP" / "CODEX_D1_AUDIT_REPORT_04.18.2026.md"
REPRO_COMMAND = "bash scripts/ingestion/audit/run.sh"

PROCEDURAL_FIELDS = (
    "rpn_program",
    "meaning_rpn",
    "visual_rpn",
    "audio_rpn",
    "behavior_rpn",
)
CANONICAL_ID_PREFIXES = (
    "word_",
    "char_",
    "grammar_template_",
    "drawing_primitive_",
    "synset_",
)
K3D_CANONICAL_RE = re.compile(r"^k3d-[a-z0-9_]+/[0-9a-f]{16}$")
CANONICAL_ALIAS_PATHS = (
    ("star_id",),
    ("canonical_id",),
    ("canonical_star_id",),
    ("metadata", "meaning_star_id"),
    ("metadata", "meaning_star", "star_id"),
)
GENERIC_SYMMETRIC_FIELDS = (
    "taxonomy_refs",
    "grammar_refs",
    "meta_refs",
    "reality_refs",
    "visual_refs",
    "audio_refs",
)
REF_LIST_FIELDS = GENERIC_SYMMETRIC_FIELDS + (
    "component_refs",
    "composite_of",
)
VOLATILE_HASH_KEYS = {
    "context_id",
    "created_at",
    "ethical_trit",
    "event_id",
    "id",
    "ingested",
    "last_accessed",
    "line_no",
    "meaning_star_id",
    "parent_event_id",
    "path",
    "processed_at",
    "star_id",
    "temporal",
    "timestamp",
    "updated",
    "updated_at",
    "vector_clock",
}
NULL_TEXT_VALUES = {"", "none", "null", "n/a"}
MATRYOSHKA_FIELD_NAMES = {"embedding_64", "embedding_128", "embedding_512", "embedding_2048"}


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _looks_canonical_id(value: str) -> bool:
    text = str(value or "").strip()
    return bool(text) and (
        any(text.startswith(prefix) for prefix in CANONICAL_ID_PREFIXES)
        or bool(K3D_CANONICAL_RE.match(text))
    )


def _deep_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _canonical_alias_for_row(row: dict[str, Any]) -> str:
    for path in CANONICAL_ALIAS_PATHS:
        candidate = _deep_get(row, path)
        if isinstance(candidate, str) and _looks_canonical_id(candidate):
            return candidate.strip()
    aliases = row.get("aliases")
    if isinstance(aliases, list):
        for candidate in aliases:
            if isinstance(candidate, str) and _looks_canonical_id(candidate):
                return candidate.strip()
    return ""


def _classify_row_id(row_id: str, row: dict[str, Any]) -> tuple[str, str]:
    text = str(row_id or "").strip()
    if not text:
        return "missing", ""
    if _looks_canonical_id(text):
        return "canonical", text
    alias = _canonical_alias_for_row(row)
    if alias:
        return "canonical", alias
    return "ad_hoc", ""


def _resolve_row_id(row: dict[str, Any], galaxy: str, line_no: int) -> tuple[str, bool]:
    for key in ("id", "star_id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), False
    nested_star_id = _deep_get(row, ("metadata", "meaning_star", "star_id"))
    if isinstance(nested_star_id, str) and nested_star_id.strip():
        return nested_star_id.strip(), False
    return f"__missing_id__:{galaxy}:{line_no}", True


def _non_placeholder_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return ""
    text = str(value).strip()
    if text.lower() in NULL_TEXT_VALUES:
        return ""
    return text


def _collect_procedural_payload(row: dict[str, Any]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for field in PROCEDURAL_FIELDS:
        text = _non_placeholder_text(row.get(field))
        if text:
            payload[field] = text
    meaning_star = _deep_get(row, ("metadata", "meaning_star"))
    if isinstance(meaning_star, dict):
        for field in PROCEDURAL_FIELDS:
            text = _non_placeholder_text(meaning_star.get(field))
            if text:
                payload[f"metadata.meaning_star.{field}"] = text
    return payload


def _normalize_for_hash(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key in sorted(value):
            if key in VOLATILE_HASH_KEYS:
                continue
            child = _normalize_for_hash(value[key])
            normalized[key] = child
        return normalized
    if isinstance(value, list):
        return [_normalize_for_hash(item) for item in value]
    return value


def _content_hash(row: dict[str, Any], procedural_payload: dict[str, str]) -> str:
    if procedural_payload:
        basis: Any = {"procedural_payload": procedural_payload}
    else:
        meaning_star = _deep_get(row, ("metadata", "meaning_star"))
        if isinstance(meaning_star, dict):
            basis = {"meaning_star": _normalize_for_hash(meaning_star)}
        else:
            basis = {"row": _normalize_for_hash(row)}
    return hashlib.sha256(_stable_json(basis).encode("utf-8")).hexdigest()


def _is_non_null_matryoshka_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(_non_placeholder_text(value))
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _has_matryoshka_payload(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in MATRYOSHKA_FIELD_NAMES or "matryoshka" in lowered:
                if _is_non_null_matryoshka_value(child):
                    return True
            if _has_matryoshka_payload(child):
                return True
        return False
    if isinstance(value, list):
        return any(_has_matryoshka_payload(item) for item in value)
    return False


def _normalize_surface_ref_kind(field_path: str) -> str:
    if ".word_ref" in field_path:
        return "surface_forms.*.word_ref"
    return "surface_forms.*.char_refs"


def _extract_refs_from_object(obj: dict[str, Any], *, path_prefix: str, fallback_source_id: str) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    source_id = str(obj.get("star_id") or obj.get("id") or fallback_source_id).strip()
    for field in REF_LIST_FIELDS:
        values = obj.get(field)
        if not isinstance(values, list):
            continue
        for target in values:
            target_id = str(target or "").strip()
            if not target_id:
                continue
            refs.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "field_path": f"{path_prefix}{field}",
                    "field_kind": field,
                }
            )
    surface_forms = obj.get("surface_forms")
    if isinstance(surface_forms, dict):
        for language in sorted(surface_forms):
            payload = surface_forms.get(language)
            if not isinstance(payload, dict):
                continue
            word_ref = str(payload.get("word_ref") or "").strip()
            if word_ref:
                refs.append(
                    {
                        "source_id": source_id,
                        "target_id": word_ref,
                        "field_path": f"{path_prefix}surface_forms.{language}.word_ref",
                        "field_kind": "surface_forms.*.word_ref",
                    }
                )
            char_refs = payload.get("char_refs")
            if isinstance(char_refs, list):
                for target in char_refs:
                    target_id = str(target or "").strip()
                    if not target_id:
                        continue
                    refs.append(
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "field_path": f"{path_prefix}surface_forms.{language}.char_refs",
                            "field_kind": "surface_forms.*.char_refs",
                        }
                    )
    return refs


def _expected_inverse_kind(field_kind: str) -> str:
    if field_kind == "component_refs":
        return "composite_of"
    if field_kind == "composite_of":
        return "component_refs"
    if field_kind == "surface_forms.*.word_ref":
        return "taxonomy_refs"
    if field_kind == "surface_forms.*.char_refs":
        return "composite_of"
    return field_kind


def _iter_source_paths(storage_root: Path) -> list[Path]:
    return sorted(path for path in storage_root.glob("*.jsonl") if path.is_file())


def _new_census_row(path: Path) -> dict[str, Any]:
    return {
        "galaxy": path.stem,
        "path": str(path),
        "entry_count": 0,
        "missing_id_count": 0,
        "canonical_id_count": 0,
        "ad_hoc_id_count": 0,
        "procedural_rpn_count": 0,
        "raw_payload_count": 0,
        "matryoshka_present_count": 0,
        "matryoshka_missing_count": 0,
        "duplicate_content_groups": 0,
        "duplicate_row_count": 0,
        "symlink_edge_count": 0,
        "unidirectional_site_count": 0,
    }


def _append_violation(
    violations: list[tuple[str, str, str, str, str, str, str, str]],
    *,
    violation_kind: str,
    galaxy: str,
    path: str,
    row_id: str,
    target_id: str,
    field_path: str,
    evidence: str,
    content_hash: str,
) -> None:
    violations.append(
        (
            violation_kind,
            galaxy,
            path,
            row_id,
            target_id,
            field_path,
            evidence,
            content_hash,
        )
    )


def _connect_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-200000")
    connection.execute(
        """
        CREATE TABLE real_ids (
            id TEXT PRIMARY KEY
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE refs (
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            field_kind TEXT NOT NULL,
            inverse_kind TEXT NOT NULL,
            field_path TEXT NOT NULL,
            galaxy TEXT NOT NULL,
            path TEXT NOT NULL,
            row_id TEXT NOT NULL,
            content_hash TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE hash_members (
            content_hash TEXT NOT NULL,
            galaxy TEXT NOT NULL,
            path TEXT NOT NULL,
            row_id TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE violations (
            violation_kind TEXT NOT NULL,
            galaxy TEXT NOT NULL,
            path TEXT NOT NULL,
            row_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            field_path TEXT NOT NULL,
            evidence TEXT NOT NULL,
            content_hash TEXT NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX idx_refs_forward ON refs(source_id, target_id, field_kind)")
    connection.execute("CREATE INDEX idx_hash_members_hash ON hash_members(content_hash)")
    connection.execute("CREATE INDEX idx_violations_sort ON violations(violation_kind, galaxy, path, row_id, field_path, target_id, content_hash)")
    return connection


def _flush_batches(
    connection: sqlite3.Connection,
    *,
    real_ids: list[tuple[str]],
    refs: list[tuple[str, str, str, str, str, str, str, str, str]],
    hash_members: list[tuple[str, str, str, str]],
    violations: list[tuple[str, str, str, str, str, str, str, str]],
) -> None:
    if real_ids:
        connection.executemany("INSERT OR IGNORE INTO real_ids(id) VALUES (?)", real_ids)
        real_ids.clear()
    if refs:
        connection.executemany(
            """
            INSERT INTO refs(
                source_id,
                target_id,
                field_kind,
                inverse_kind,
                field_path,
                galaxy,
                path,
                row_id,
                content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            refs,
        )
        refs.clear()
    if hash_members:
        connection.executemany(
            "INSERT INTO hash_members(content_hash, galaxy, path, row_id) VALUES (?, ?, ?, ?)",
            hash_members,
        )
        hash_members.clear()
    if violations:
        connection.executemany(
            """
            INSERT INTO violations(
                violation_kind,
                galaxy,
                path,
                row_id,
                target_id,
                field_path,
                evidence,
                content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            violations,
        )
        violations.clear()
    connection.commit()


def _write_violations_from_db(connection: sqlite3.Connection, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        cursor = connection.execute(
            """
            SELECT
                violation_kind,
                galaxy,
                path,
                row_id,
                target_id,
                field_path,
                evidence,
                content_hash
            FROM violations
            ORDER BY
                violation_kind,
                galaxy,
                path,
                row_id,
                field_path,
                target_id,
                content_hash,
                evidence
            """
        )
        for row in cursor:
            payload = {
                "violation_kind": row[0],
                "galaxy": row[1],
                "path": row[2],
                "row_id": row[3],
                "target_id": row[4],
                "field_path": row[5],
                "evidence": row[6],
                "content_hash": row[7],
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def run_scan(storage_root: Path, output_dir: Path) -> dict[str, Path]:
    source_paths = _iter_source_paths(storage_root)
    census_rows: list[dict[str, Any]] = []
    census_by_path: dict[str, dict[str, Any]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / ".galaxy_audit.sqlite3"
    if db_path.exists():
        db_path.unlink()

    connection = _connect_db(db_path)
    pending_real_ids: list[tuple[str]] = []
    pending_refs: list[tuple[str, str, str, str, str, str, str, str, str]] = []
    pending_hash_members: list[tuple[str, str, str, str]] = []
    pending_violations: list[tuple[str, str, str, str, str, str, str, str]] = []
    batch_size = 5000

    try:
        for source_path in source_paths:
            census = _new_census_row(source_path)
            census_by_path[str(source_path)] = census
            galaxy = source_path.stem

            with source_path.open("r", encoding="utf-8") as handle:
                for line_no, raw_line in enumerate(handle, start=1):
                    stripped = raw_line.strip()
                    if not stripped:
                        continue

                    census["entry_count"] += 1
                    try:
                        row = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        synthetic_id = f"__invalid_json__:{galaxy}:{line_no}"
                        census["missing_id_count"] += 1
                        _append_violation(
                            pending_violations,
                            violation_kind="invalid_json",
                            galaxy=galaxy,
                            path=str(source_path),
                            row_id=synthetic_id,
                            target_id="",
                            field_path="",
                            evidence=f"invalid JSON at line {line_no}: {exc.msg}",
                            content_hash="",
                        )
                        continue

                    if not isinstance(row, dict):
                        synthetic_id = f"__invalid_row__:{galaxy}:{line_no}"
                        census["missing_id_count"] += 1
                        _append_violation(
                            pending_violations,
                            violation_kind="invalid_row",
                            galaxy=galaxy,
                            path=str(source_path),
                            row_id=synthetic_id,
                            target_id="",
                            field_path="",
                            evidence=f"non-object JSON row at line {line_no}",
                            content_hash="",
                        )
                        continue

                    row_id, used_synthetic_id = _resolve_row_id(row, galaxy, line_no)
                    if not used_synthetic_id:
                        pending_real_ids.append((row_id,))

                    if used_synthetic_id:
                        id_classification, alias = "missing", ""
                    else:
                        id_classification, alias = _classify_row_id(row_id, row)
                    if id_classification == "missing":
                        census["missing_id_count"] += 1
                        _append_violation(
                            pending_violations,
                            violation_kind="missing_id",
                            galaxy=galaxy,
                            path=str(source_path),
                            row_id=row_id,
                            target_id="",
                            field_path="id",
                            evidence=f"row at line {line_no} has no id, star_id, or nested meaning-star id",
                            content_hash="",
                        )
                    elif id_classification == "canonical":
                        census["canonical_id_count"] += 1
                    else:
                        census["ad_hoc_id_count"] += 1
                        alias_note = f"; canonical alias {alias}" if alias else ""
                        _append_violation(
                            pending_violations,
                            violation_kind="ad_hoc_id",
                            galaxy=galaxy,
                            path=str(source_path),
                            row_id=row_id,
                            target_id=alias,
                            field_path="id",
                            evidence=f"id '{row_id}' is not in canonical families{alias_note}",
                            content_hash="",
                        )

                    procedural_payload = _collect_procedural_payload(row)
                    content_hash = _content_hash(row, procedural_payload)
                    if procedural_payload:
                        census["procedural_rpn_count"] += 1
                    else:
                        census["raw_payload_count"] += 1
                        _append_violation(
                            pending_violations,
                            violation_kind="raw_payload",
                            galaxy=galaxy,
                            path=str(source_path),
                            row_id=row_id,
                            target_id="",
                            field_path="",
                            evidence="no non-placeholder procedural field found in row or metadata.meaning_star",
                            content_hash=content_hash,
                        )

                    if galaxy in {"Word", "Character"}:
                        if _has_matryoshka_payload(row):
                            census["matryoshka_present_count"] += 1
                        else:
                            census["matryoshka_missing_count"] += 1
                            _append_violation(
                                pending_violations,
                                violation_kind="missing_matryoshka",
                                galaxy=galaxy,
                                path=str(source_path),
                                row_id=row_id,
                                target_id="",
                                field_path="embeddings",
                                evidence="row has no non-null Matryoshka tier field",
                                content_hash=content_hash,
                            )

                    pending_hash_members.append((content_hash, galaxy, str(source_path), row_id))

                    row_refs = _extract_refs_from_object(row, path_prefix="", fallback_source_id=row_id)
                    meaning_star = _deep_get(row, ("metadata", "meaning_star"))
                    if isinstance(meaning_star, dict):
                        row_refs.extend(
                            _extract_refs_from_object(
                                meaning_star,
                                path_prefix="metadata.meaning_star.",
                                fallback_source_id=row_id,
                            )
                        )
                    census["symlink_edge_count"] += len(row_refs)
                    for ref in row_refs:
                        pending_refs.append(
                            (
                                ref["source_id"],
                                ref["target_id"],
                                ref["field_kind"],
                                _expected_inverse_kind(ref["field_kind"]),
                                ref["field_path"],
                                galaxy,
                                str(source_path),
                                row_id,
                                content_hash,
                            )
                        )

                    if (
                        len(pending_real_ids) >= batch_size
                        or len(pending_refs) >= batch_size
                        or len(pending_hash_members) >= batch_size
                        or len(pending_violations) >= batch_size
                    ):
                        _flush_batches(
                            connection,
                            real_ids=pending_real_ids,
                            refs=pending_refs,
                            hash_members=pending_hash_members,
                            violations=pending_violations,
                        )

            census_rows.append(census)

        _flush_batches(
            connection,
            real_ids=pending_real_ids,
            refs=pending_refs,
            hash_members=pending_hash_members,
            violations=pending_violations,
        )

        duplicate_hashes = connection.execute(
            """
            SELECT content_hash
            FROM hash_members
            GROUP BY content_hash
            HAVING COUNT(*) > 1
            ORDER BY content_hash
            """
        )
        for (content_hash,) in duplicate_hashes:
            members = list(
                connection.execute(
                    """
                    SELECT galaxy, path, row_id
                    FROM hash_members
                    WHERE content_hash = ?
                    ORDER BY galaxy, row_id, path
                    """,
                    (content_hash,),
                )
            )
            files_in_group = Counter(member[1] for member in members)
            for path, count in files_in_group.items():
                census = census_by_path[path]
                census["duplicate_content_groups"] += 1
                census["duplicate_row_count"] += count
            peers = [member[2] for member in members]
            duplicate_violations: list[tuple[str, str, str, str, str, str, str, str]] = []
            for galaxy, path, row_id in members:
                peer_ids = [peer for peer in peers if peer != row_id]
                _append_violation(
                    duplicate_violations,
                    violation_kind="duplicate_content",
                    galaxy=galaxy,
                    path=path,
                    row_id=row_id,
                    target_id="",
                    field_path="",
                    evidence=f"content hash shared by {len(members)} rows; peers={','.join(peer_ids[:10])}",
                    content_hash=content_hash,
                )
            _flush_batches(
                connection,
                real_ids=pending_real_ids,
                refs=pending_refs,
                hash_members=pending_hash_members,
                violations=duplicate_violations,
            )

        ref_cursor = connection.execute(
            """
            SELECT
                r.galaxy,
                r.path,
                r.row_id,
                r.target_id,
                r.field_path,
                r.content_hash,
                EXISTS(SELECT 1 FROM real_ids ids WHERE ids.id = r.target_id) AS target_exists,
                EXISTS(
                    SELECT 1
                    FROM refs inv
                    WHERE inv.source_id = r.target_id
                      AND inv.target_id = r.source_id
                      AND inv.field_kind = r.inverse_kind
                ) AS inverse_exists,
                r.inverse_kind
            FROM refs r
            ORDER BY
                r.galaxy,
                r.row_id,
                r.field_path,
                r.target_id,
                r.path
            """
        )
        edge_violations: list[tuple[str, str, str, str, str, str, str, str]] = []
        for galaxy, path, row_id, target_id, field_path, content_hash, target_exists, inverse_exists, inverse_kind in ref_cursor:
            if not target_exists:
                _append_violation(
                    edge_violations,
                    violation_kind="missing_target",
                    galaxy=galaxy,
                    path=path,
                    row_id=row_id,
                    target_id=target_id,
                    field_path=field_path,
                    evidence=f"target '{target_id}' is not present in the scanned live root",
                    content_hash=content_hash,
                )
            if not inverse_exists:
                census_by_path[path]["unidirectional_site_count"] += 1
                _append_violation(
                    edge_violations,
                    violation_kind="unidirectional_ref",
                    galaxy=galaxy,
                    path=path,
                    row_id=row_id,
                    target_id=target_id,
                    field_path=field_path,
                    evidence=f"expected inverse field '{inverse_kind}' on target '{target_id}'",
                    content_hash=content_hash,
                )
            if len(edge_violations) >= batch_size:
                _flush_batches(
                    connection,
                    real_ids=pending_real_ids,
                    refs=pending_refs,
                    hash_members=pending_hash_members,
                    violations=edge_violations,
                )

        _flush_batches(
            connection,
            real_ids=pending_real_ids,
            refs=pending_refs,
            hash_members=pending_hash_members,
            violations=edge_violations,
        )

        census_rows = sorted(census_rows, key=lambda row: (row["galaxy"], row["path"]))
        census_path = output_dir / "galaxy_census.jsonl"
        violations_path = output_dir / "violations.jsonl"
        _write_jsonl(census_path, census_rows)
        _write_violations_from_db(connection, violations_path)
        return {"census": census_path, "violations": violations_path}
    finally:
        connection.close()
        if db_path.exists():
            db_path.unlink()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def render_report(storage_root: Path, output_dir: Path, report_path: Path) -> None:
    census_path = output_dir / "galaxy_census.jsonl"
    violations_path = output_dir / "violations.jsonl"
    census_rows = _read_jsonl(census_path)

    totals = Counter()
    for row in census_rows:
        for key in (
            "entry_count",
            "missing_id_count",
            "canonical_id_count",
            "ad_hoc_id_count",
            "procedural_rpn_count",
            "raw_payload_count",
            "matryoshka_present_count",
            "matryoshka_missing_count",
            "duplicate_content_groups",
            "duplicate_row_count",
            "symlink_edge_count",
            "unidirectional_site_count",
        ):
            totals[key] += int(row.get(key, 0) or 0)

    violation_counts = Counter()
    for row in _iter_jsonl(violations_path):
        violation_counts[row["violation_kind"]] += 1
    most_unidirectional = sorted(
        census_rows,
        key=lambda row: (-int(row["unidirectional_site_count"]), row["galaxy"]),
    )[:5]
    most_ad_hoc = sorted(
        census_rows,
        key=lambda row: (-int(row["ad_hoc_id_count"]), row["galaxy"]),
    )[:5]

    lines: list[str] = [
        "# CODEX D1 Audit Report — 2026-04-18",
        "",
        f"Reproduction command: `{REPRO_COMMAND}`",
        "",
        "## Scope",
        "",
        f"- Live storage root: `{storage_root}`",
        f"- Census artifact: `{census_path}`",
        f"- Violations artifact: `{violations_path}`",
        f"- Files audited: {len(census_rows)}",
        f"- Total rows audited: {totals['entry_count']}",
        "",
        "## Headline Counts",
        "",
        f"- Canonical IDs: {totals['canonical_id_count']}",
        f"- Missing IDs: {totals['missing_id_count']}",
        f"- Ad-hoc IDs: {totals['ad_hoc_id_count']}",
        f"- Procedural rows: {totals['procedural_rpn_count']}",
        f"- Raw/non-procedural rows: {totals['raw_payload_count']}",
        f"- Matryoshka-present Word/Character rows: {totals['matryoshka_present_count']}",
        f"- Matryoshka-missing Word/Character rows: {totals['matryoshka_missing_count']}",
        f"- Duplicate content groups: {totals['duplicate_content_groups']}",
        f"- Rows participating in duplicate groups: {totals['duplicate_row_count']}",
        f"- Symlink edges scanned: {totals['symlink_edge_count']}",
        f"- Unidirectional symlink sites: {totals['unidirectional_site_count']}",
        "",
        "## Violation Counts",
        "",
    ]

    for violation_kind, count in sorted(violation_counts.items()):
        lines.append(f"- `{violation_kind}`: {count}")

    lines.extend(
        [
            "",
            "## Galaxy Census",
            "",
            "| Galaxy | Entries | Canonical | Missing ID | Ad-hoc ID | Procedural | Raw | Matryoshka Present | Matryoshka Missing | Duplicate Groups | Duplicate Rows | Symlink Edges | Unidirectional |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in census_rows:
        lines.append(
            "| {galaxy} | {entry_count} | {canonical_id_count} | {missing_id_count} | {ad_hoc_id_count} | {procedural_rpn_count} | {raw_payload_count} | {matryoshka_present_count} | {matryoshka_missing_count} | {duplicate_content_groups} | {duplicate_row_count} | {symlink_edge_count} | {unidirectional_site_count} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Highest Unidirectional Counts",
            "",
        ]
    )
    for row in most_unidirectional:
        lines.append(f"- `{row['galaxy']}`: {row['unidirectional_site_count']} unidirectional sites")

    lines.extend(
        [
            "",
            "## Highest Ad-hoc ID Counts",
            "",
        ]
    )
    for row in most_ad_hoc:
        lines.append(f"- `{row['galaxy']}`: {row['ad_hoc_id_count']} ad-hoc ids")

    lines.extend(
        [
            "",
            "## Evidence Commands",
            "",
            f"- Rebuild artifacts: `{REPRO_COMMAND}`",
            f"- Confirm raw source line counts: `wc -l {storage_root}/*.jsonl`",
            f"- Inspect staged violation counts: `python3 - <<'PY'\\nimport json\\nfrom collections import Counter\\nfrom pathlib import Path\\npath = Path('{violations_path}')\\ncounts = Counter(json.loads(line)['violation_kind'] for line in path.read_text(encoding='utf-8').splitlines() if line.strip())\\nprint(dict(sorted(counts.items())))\\nPY`",
            "",
            "## Notes",
            "",
            "- D1 is audit-only. No live JSONL row was rewritten.",
            "- Counts come from the live resident root, not the older 38,144-entry claim in the handoff text.",
        ]
    )

    _write_text(report_path, "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the live K3D galaxy JSONL root for D1.")
    parser.add_argument(
        "--mode",
        choices=("scan", "report"),
        default="scan",
        help="Scan source JSONL into staged artifacts or regenerate the Markdown report from staged outputs.",
    )
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=DEFAULT_STORAGE_ROOT,
        help="Directory containing live galaxy JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where deterministic D1 JSONL artifacts are staged.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown report path for the D1 narrative summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    storage_root = Path(args.storage_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report_path).resolve()
    if args.mode == "scan":
        paths = run_scan(storage_root, output_dir)
        print(
            json.dumps(
                {
                    "mode": "scan",
                    "storage_root": str(storage_root),
                    "output_dir": str(output_dir),
                    "census_path": str(paths["census"]),
                    "violations_path": str(paths["violations"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    render_report(storage_root, output_dir, report_path)
    print(
        json.dumps(
            {
                "mode": "report",
                "storage_root": str(storage_root),
                "output_dir": str(output_dir),
                "report_path": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
