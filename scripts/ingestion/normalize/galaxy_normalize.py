#!/usr/bin/env python3
"""D2 staging-only Galaxy normalization pipeline.

This script performs the D2 normalization passes against the live Galaxy root
without mutating the source tree. It writes deterministic staged artifacts to
`scripts/ingestion/staging/D2_normalize/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STORAGE_ROOT = Path("/K3D/Knowledge3D.local/galaxies")
DEFAULT_STAGE_DIR = REPO_ROOT / "scripts" / "ingestion" / "staging" / "D2_normalize"
DEFAULT_TEMP_REPORT_PATH = REPO_ROOT / "TEMP" / "CODEX_D2_NORMALIZATION_REPORT_04.18.2026.md"
DEFAULT_D1_CENSUS_PATH = REPO_ROOT / "scripts" / "ingestion" / "staging" / "D1_audit" / "galaxy_census.jsonl"
REPRO_COMMAND = "bash scripts/ingestion/normalize/run.sh"

CANONICAL_ID_PREFIXES = (
    "word_",
    "char_",
    "grammar_template_",
    "drawing_primitive_",
    "synset_",
)
K3D_CANONICAL_RE = re.compile(r"^k3d-[a-z0-9_]+/[0-9a-f]{16}$")
NULL_TEXT_VALUES = {"", "none", "null", "n/a"}
PROCEDURAL_FIELDS = (
    "rpn_program",
    "meaning_rpn",
    "visual_rpn",
    "audio_rpn",
    "behavior_rpn",
)
VOLATILE_HASH_KEYS = {
    "created_at",
    "event_id",
    "id",
    "ingested",
    "line_no",
    "path",
    "processed_at",
    "star_id",
    "timestamp",
    "updated",
    "updated_at",
}
KNOWN_REF_LIST_FIELDS = {
    "anti_pattern_refs",
    "audio_refs",
    "char_refs",
    "component_refs",
    "composite_of",
    "drawing_refs",
    "executor_refs",
    "form_refs",
    "grammar_refs",
    "math_refs",
    "meaning_refs",
    "meta_refs",
    "object_refs",
    "reality_refs",
    "router_refs",
    "rule_refs",
    "symbol_refs",
    "symlinks",
    "taxonomy_refs",
    "tool_refs",
    "validator_refs",
    "visual_refs",
    "word_refs",
}
KNOWN_REF_SCALAR_FIELDS = {
    "char_ref",
    "formalizes_ref",
    "meaning_layer_id",
    "meaning_ref",
    "meaning_star_id",
    "number_ref",
    "parent_id",
    "see_also",
    "symlink_target",
    "symlink_to",
    "template_ref",
    "word_id",
    "word_ref",
}
EXCLUDED_REF_FIELDS = {
    "arc_task_id",
    "context_id",
    "galaxy_ref",
    "layer_id",
    "meta_id",
    "pattern_id",
    "pool_id",
    "problem_id",
    "question_id",
    "route_contract_schema_version",
    "source_galaxy",
    "symlink",
    "symlink_galaxy",
    "target_galaxy",
    "task_id",
}
SYMLINK_LIST_KINDS = {
    "anti_pattern_refs",
    "audio_refs",
    "char_refs",
    "component_refs",
    "composite_of",
    "drawing_refs",
    "executor_refs",
    "grammar_refs",
    "math_refs",
    "meaning_refs",
    "meta_refs",
    "object_refs",
    "reality_refs",
    "router_refs",
    "rule_refs",
    "symlinks",
    "taxonomy_refs",
    "tool_refs",
    "validator_refs",
    "visual_refs",
    "word_refs",
}
MATRYOSHKA_DIMENSIONS = (64, 128, 256, 512)


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


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _blake16(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _slug(text: str) -> str:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return "unknown"
    parts: list[str] = []
    pending_sep = False
    for char in lowered:
        if char.isascii() and char.isalnum():
            if pending_sep and parts and parts[-1] != "_":
                parts.append("_")
            parts.append(char)
            pending_sep = False
            continue
        pending_sep = True
    slug = "".join(parts).strip("_")
    return slug or "unknown"


def _non_placeholder_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return ""
    text = str(value).strip()
    if text.lower() in NULL_TEXT_VALUES:
        return ""
    return text


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


def _row_identity_values(row: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("id", "star_id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("meaning_star_id", "word_id", "meta_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
        meaning_star = metadata.get("meaning_star")
        if isinstance(meaning_star, dict):
            value = meaning_star.get("star_id")
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _resolve_primary_row_id(row: dict[str, Any], galaxy: str, line_no: int) -> tuple[str, bool]:
    for candidate in _row_identity_values(row):
        return candidate, False
    return f"__missing_id__:{galaxy}:{line_no}", True


def _canonical_candidates(row: dict[str, Any]) -> list[str]:
    return [candidate for candidate in _row_identity_values(row) if _looks_canonical_id(candidate)]


def _is_ref_field(path: tuple[str, ...], key: str, value: Any) -> bool:
    if key in EXCLUDED_REF_FIELDS:
        return False
    if len(path) >= 1 and path[-1] == "surface_forms" and key not in {"word_ref", "char_refs"}:
        return False
    if key in KNOWN_REF_LIST_FIELDS or key in KNOWN_REF_SCALAR_FIELDS:
        return True
    if key == "symlinks" and isinstance(value, list):
        return True
    if key.endswith("_refs") or key.endswith("_ids"):
        return True
    if key.endswith("_ref") or key.endswith("_id"):
        return True
    return False


def _field_kind(path: tuple[str, ...], key: str) -> str:
    if len(path) >= 1 and path[-1] == "surface_forms" and key == "word_ref":
        return "surface_forms.*.word_ref"
    if len(path) >= 1 and path[-1] == "surface_forms" and key == "char_refs":
        return "surface_forms.*.char_refs"
    return key


def _is_ref_scalar_path(path: tuple[str, ...]) -> bool:
    if not path:
        return False
    kind = _field_kind(path[:-1], path[-1])
    return kind in KNOWN_REF_SCALAR_FIELDS or kind in {"surface_forms.*.word_ref"}


def _is_ref_list_path(path: tuple[str, ...]) -> bool:
    if not path:
        return False
    kind = _field_kind(path[:-1], path[-1])
    return kind in KNOWN_REF_LIST_FIELDS or kind == "surface_forms.*.char_refs"


def _sort_ref_list(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered = sorted(str(value) for value in values if str(value).strip())
    result: list[str] = []
    for value in ordered:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _normalize_for_hash(value: Any, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key in sorted(value):
            if key in VOLATILE_HASH_KEYS:
                continue
            child = value[key]
            if _is_ref_field(path, key, child):
                continue
            normalized[key] = _normalize_for_hash(child, path + (key,))
        return normalized
    if isinstance(value, list):
        return [_normalize_for_hash(item, path + ("[]",)) for item in value]
    return value


def _content_hash(row: dict[str, Any]) -> str:
    normalized = _normalize_for_hash(row)
    return _blake16(_stable_json(normalized))


def _sync_row_ids(row: dict[str, Any], canonical_id: str) -> dict[str, Any]:
    synced = json.loads(json.dumps(row, ensure_ascii=False))
    synced["id"] = canonical_id
    if "star_id" in synced:
        synced["star_id"] = canonical_id
    metadata = synced.get("metadata")
    if isinstance(metadata, dict):
        if "meaning_star_id" in metadata:
            metadata["meaning_star_id"] = canonical_id
        meaning_star = metadata.get("meaning_star")
        if isinstance(meaning_star, dict):
            meaning_star["star_id"] = canonical_id
    return synced


def _rewrite_ref_scalar(value: Any, rewrite_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return rewrite_map.get(value, value)
    return value


def _rewrite_refs(value: Any, rewrite_map: dict[str, str], path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        rewritten: dict[str, Any] = {}
        for key in sorted(value):
            child = value[key]
            child_path = path + (key,)
            if _is_ref_field(path, key, child):
                if isinstance(child, list):
                    items = [_rewrite_ref_scalar(item, rewrite_map) for item in child]
                    rewritten[key] = _sort_ref_list(items)
                elif isinstance(child, str):
                    rewritten[key] = _rewrite_ref_scalar(child, rewrite_map)
                else:
                    rewritten[key] = child
                continue
            rewritten[key] = _rewrite_refs(child, rewrite_map, child_path)
        return rewritten
    if isinstance(value, list):
        return [_rewrite_refs(item, rewrite_map, path + ("[]",)) for item in value]
    return value


def _extract_ref_edges(value: Any, row_id: str, *, path: tuple[str, ...] = ()) -> list[tuple[str, str, str, str]]:
    edges: list[tuple[str, str, str, str]] = []
    if isinstance(value, dict):
        for key in sorted(value):
            child = value[key]
            child_path = path + (key,)
            if _is_ref_field(path, key, child):
                kind = _field_kind(path, key)
                field_path = ".".join(child_path)
                if isinstance(child, list):
                    for item in child:
                        if isinstance(item, str) and item.strip():
                            edges.append((row_id, item.strip(), field_path, kind))
                elif isinstance(child, str) and child.strip():
                    edges.append((row_id, child.strip(), field_path, kind))
                continue
            edges.extend(_extract_ref_edges(child, row_id, path=child_path))
        return edges
    if isinstance(value, list):
        for item in value:
            edges.extend(_extract_ref_edges(item, row_id, path=path + ("[]",)))
    return edges


def _expected_inverse_kind(kind: str) -> str | None:
    if kind == "component_refs":
        return "composite_of"
    if kind == "composite_of":
        return "component_refs"
    if kind == "surface_forms.*.word_ref":
        return "taxonomy_refs"
    if kind == "surface_forms.*.char_refs":
        return "composite_of"
    if kind in SYMLINK_LIST_KINDS:
        return kind
    return None


def _inverse_field_path(source_field_path: str, source_kind: str) -> str:
    parts = source_field_path.split(".")
    if source_kind.startswith("surface_forms.*."):
        if len(parts) >= 4 and parts[-3] == "surface_forms":
            prefix = parts[:-3]
        else:
            prefix = parts[:-1]
        inverse_kind = _expected_inverse_kind(source_kind)
        if not inverse_kind:
            return source_field_path
        return ".".join(prefix + [inverse_kind]) if prefix else inverse_kind
    inverse_kind = _expected_inverse_kind(source_kind) or source_kind
    parts[-1] = inverse_kind
    return ".".join(parts)


def _row_has_meaning_star(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata")
    return isinstance(metadata, dict) and isinstance(metadata.get("meaning_star"), dict)


def _has_matryoshka_payload(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in {"embedding_64", "embedding_128", "embedding_512", "embedding_2048"} or "matryoshka" in lowered:
                if child is None:
                    continue
                if isinstance(child, str) and not _non_placeholder_text(child):
                    continue
                if isinstance(child, (list, tuple, dict)) and not child:
                    continue
                return True
            if _has_matryoshka_payload(child):
                return True
        return False
    if isinstance(value, list):
        return any(_has_matryoshka_payload(item) for item in value)
    return False


def _collect_procedural_payload(row: dict[str, Any]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for field in PROCEDURAL_FIELDS:
        text = _non_placeholder_text(row.get(field))
        if text:
            payload[field] = text
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        meaning_star = metadata.get("meaning_star")
        if isinstance(meaning_star, dict):
            for field in PROCEDURAL_FIELDS:
                text = _non_placeholder_text(meaning_star.get(field))
                if text:
                    payload[f"metadata.meaning_star.{field}"] = text
    return payload


def _extract_meaning_source_text(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        meaning_star = metadata.get("meaning_star")
        if isinstance(meaning_star, dict):
            for field in ("meaning_rpn", "visual_rpn", "audio_rpn", "behavior_rpn"):
                text = _non_placeholder_text(meaning_star.get(field))
                if text:
                    return text
    for field in ("meaning_rpn", "rpn_program", "content", "description", "summary", "name"):
        text = _non_placeholder_text(row.get(field))
        if text:
            return text
    return ""


def _generate_matryoshka_vectors(seed_text: str) -> dict[int, list[float]]:
    try:
        from knowledge3d.ingestion.enrichment_pipeline import EnrichmentPipeline

        pipeline = EnrichmentPipeline(use_local_models=False)
        embeddings = pipeline.generate_matryoshka_embedding(seed_text)
        return {
            64: [float(value) for value in embeddings[64].tolist()],
            128: [float(value) for value in embeddings[128].tolist()],
            256: [float(value) for value in embeddings[512][:256].tolist()],
            512: [float(value) for value in embeddings[512].tolist()],
        }
    except Exception:
        seed = hashlib.sha256(seed_text.encode("utf-8")).digest()
        values: list[float] = []
        block = seed
        while len(values) < 512:
            block = hashlib.sha256(block).digest()
            for index in range(0, len(block), 4):
                chunk = block[index : index + 4]
                if len(chunk) < 4:
                    continue
                raw = int.from_bytes(chunk, "little", signed=False)
                values.append(((raw / 0xFFFFFFFF) * 2.0) - 1.0)
                if len(values) == 512:
                    break
        return {
            64: values[:64],
            128: values[:128],
            256: values[:256],
            512: values[:512],
        }


def _set_path_value(payload: dict[str, Any], field_path: str, value: Any) -> None:
    parts = field_path.split(".")
    current: Any = payload
    for key in parts[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    current[parts[-1]] = value


def _get_path_value(payload: dict[str, Any], field_path: str) -> Any:
    current: Any = payload
    for key in field_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _remove_target_from_path(payload: dict[str, Any], field_path: str, target_id: str) -> None:
    current = _get_path_value(payload, field_path)
    if isinstance(current, list):
        current[:] = [item for item in current if item != target_id]
    elif isinstance(current, str) and current == target_id:
        _set_path_value(payload, field_path, None)


def _append_target_to_path(payload: dict[str, Any], field_path: str, ref_id: str) -> None:
    current = _get_path_value(payload, field_path)
    if isinstance(current, list):
        items = current + [ref_id]
        _set_path_value(payload, field_path, _sort_ref_list(items))
        return
    if current is None:
        _set_path_value(payload, field_path, [ref_id])
        return
    if isinstance(current, str):
        _set_path_value(payload, field_path, ref_id)


class D2Normalizer:
    """Implements the D2 normalization pipeline."""

    def __init__(self, *, storage_root: Path, stage_dir: Path, temp_report_path: Path) -> None:
        self.storage_root = storage_root
        self.stage_dir = stage_dir
        self.temp_report_path = temp_report_path
        self.normalized_dir = self.stage_dir / "normalized"
        self.refs_rewrite_map_path = self.stage_dir / "refs_rewrite_map.jsonl"
        self.bidirectional_edges_path = self.stage_dir / "bidirectional_edges.jsonl"
        self.orphan_targets_path = self.stage_dir / "orphan_targets.jsonl"
        self.matryoshka_fills_path = self.stage_dir / "matryoshka_fills.jsonl"
        self.procedural_upgrades_path = self.stage_dir / "procedural_upgrades.jsonl"
        self.stage_report_path = self.stage_dir / "D2_NORMALIZATION_REPORT.md"
        self.hashes_path = self.stage_dir / "hashes.txt"
        self.reaudit_dir = self.stage_dir / "re_audit"
        self.reaudit_census_path = self.reaudit_dir / "galaxy_census.jsonl"
        self.reaudit_violations_path = self.reaudit_dir / "violations.jsonl"
        self.reaudit_report_path = self.reaudit_dir / "RE_AUDIT_REPORT.md"
        self.db_path = self.stage_dir / ".normalize.sqlite3"
        self.rewrite_map: dict[str, str] = {}

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path))
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=MEMORY")
        return connection

    def _reset_stage_dir(self) -> None:
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        for path in (
            self.normalized_dir,
            self.reaudit_dir,
        ):
            if path.exists():
                shutil.rmtree(path)
        for path in (
            self.refs_rewrite_map_path,
            self.bidirectional_edges_path,
            self.orphan_targets_path,
            self.matryoshka_fills_path,
            self.procedural_upgrades_path,
            self.stage_report_path,
            self.hashes_path,
        ):
            if path.exists():
                path.unlink()
        if self.db_path.exists():
            self.db_path.unlink()

    def _init_db(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE source_rows (
                row_key INTEGER PRIMARY KEY AUTOINCREMENT,
                galaxy TEXT NOT NULL,
                file_name TEXT NOT NULL,
                line_no INTEGER NOT NULL,
                source_id TEXT NOT NULL,
                missing_id INTEGER NOT NULL,
                canonical_candidates TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                row_json TEXT NOT NULL
            );
            CREATE TABLE normalized_rows (
                row_id TEXT PRIMARY KEY,
                galaxy TEXT NOT NULL,
                file_name TEXT NOT NULL,
                row_json TEXT NOT NULL
            );
            CREATE TABLE rewrite_map (
                old_id TEXT PRIMARY KEY,
                canonical_id TEXT NOT NULL,
                galaxy TEXT NOT NULL,
                content_hash TEXT NOT NULL
            );
            CREATE TABLE ref_edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                field_path TEXT NOT NULL,
                field_kind TEXT NOT NULL
            );
            CREATE TABLE orphan_actions (
                source_id TEXT NOT NULL,
                field_path TEXT NOT NULL,
                target_id TEXT NOT NULL,
                galaxy TEXT NOT NULL,
                reason TEXT NOT NULL
            );
            CREATE TABLE edge_additions (
                target_id TEXT NOT NULL,
                field_path TEXT NOT NULL,
                ref_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                field_kind TEXT NOT NULL,
                PRIMARY KEY (target_id, field_path, ref_id)
            );
            """
        )
        connection.execute("CREATE INDEX idx_source_bucket ON source_rows (galaxy, content_hash)")
        connection.execute("CREATE INDEX idx_ref_target ON ref_edges (target_id)")
        connection.execute("CREATE INDEX idx_ref_source ON ref_edges (source_id)")
        connection.execute("CREATE INDEX idx_orphan_source ON orphan_actions (source_id)")
        connection.execute("CREATE INDEX idx_additions_target ON edge_additions (target_id)")
        connection.commit()

    def _load_source_rows(self, connection: sqlite3.Connection) -> None:
        source_paths = sorted(path for path in self.storage_root.glob("*.jsonl") if path.is_file())
        for path in source_paths:
            galaxy = path.stem
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    row = json.loads(text)
                    source_id, missing_id = _resolve_primary_row_id(row, galaxy, line_no)
                    candidates = _canonical_candidates(row)
                    content_hash = _content_hash(row)
                    connection.execute(
                        """
                        INSERT INTO source_rows (
                            galaxy, file_name, line_no, source_id, missing_id,
                            canonical_candidates, content_hash, row_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            galaxy,
                            path.name,
                            line_no,
                            source_id,
                            1 if missing_id else 0,
                            json.dumps(candidates, ensure_ascii=False, sort_keys=True),
                            content_hash,
                            json.dumps(row, ensure_ascii=False, sort_keys=True),
                        ),
                    )
        connection.commit()

    def _upsert_rewrite(self, connection: sqlite3.Connection, old_id: str, canonical_id: str, galaxy: str, content_hash: str) -> None:
        if not old_id or old_id == canonical_id:
            return
        connection.execute(
            """
            INSERT INTO rewrite_map (old_id, canonical_id, galaxy, content_hash)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(old_id) DO UPDATE SET
                canonical_id=CASE
                    WHEN rewrite_map.canonical_id = excluded.canonical_id THEN rewrite_map.canonical_id
                    WHEN rewrite_map.canonical_id < excluded.canonical_id THEN rewrite_map.canonical_id
                    ELSE excluded.canonical_id
                END,
                galaxy=CASE
                    WHEN rewrite_map.canonical_id = excluded.canonical_id THEN rewrite_map.galaxy
                    WHEN rewrite_map.canonical_id < excluded.canonical_id THEN rewrite_map.galaxy
                    ELSE excluded.galaxy
                END,
                content_hash=CASE
                    WHEN rewrite_map.canonical_id = excluded.canonical_id THEN rewrite_map.content_hash
                    WHEN rewrite_map.canonical_id < excluded.canonical_id THEN rewrite_map.content_hash
                    ELSE excluded.content_hash
                END
            """,
            (old_id, canonical_id, galaxy, content_hash),
        )

    def _pass_a_assign_ids(self, connection: sqlite3.Connection) -> None:
        cursor = connection.execute(
            """
            SELECT row_key, galaxy, file_name, line_no, source_id, missing_id,
                   canonical_candidates, content_hash, row_json
            FROM source_rows
            ORDER BY galaxy, content_hash, line_no, source_id
            """
        )
        current_bucket: tuple[str, str] | None = None
        bucket_rows: list[dict[str, Any]] = []

        def flush_bucket() -> None:
            if not bucket_rows:
                return
            galaxy = bucket_rows[0]["galaxy"]
            content_hash = bucket_rows[0]["content_hash"]
            candidates = sorted(
                {
                    candidate
                    for item in bucket_rows
                    for candidate in item["canonical_candidates"]
                    if _looks_canonical_id(candidate)
                }
            )
            canonical_id = candidates[0] if candidates else f"k3d-{_slug(galaxy)}/{content_hash[:16]}"
            chosen_row = None
            for item in bucket_rows:
                if item["source_id"] == canonical_id or canonical_id in item["canonical_candidates"]:
                    chosen_row = item
                    break
            if chosen_row is None:
                chosen_row = bucket_rows[0]
            synced_row = _sync_row_ids(chosen_row["row"], canonical_id)
            row_json_text = json.dumps(synced_row, ensure_ascii=False, sort_keys=True)
            existing = connection.execute(
                "SELECT file_name, row_json FROM normalized_rows WHERE row_id = ?",
                (canonical_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO normalized_rows (row_id, galaxy, file_name, row_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        canonical_id,
                        galaxy,
                        chosen_row["file_name"],
                        row_json_text,
                    ),
                )
            else:
                existing_file_name, existing_row_json = existing
                if (row_json_text, chosen_row["file_name"]) < (existing_row_json, existing_file_name):
                    connection.execute(
                        """
                        UPDATE normalized_rows
                        SET galaxy = ?, file_name = ?, row_json = ?
                        WHERE row_id = ?
                        """,
                        (
                            galaxy,
                            chosen_row["file_name"],
                            row_json_text,
                            canonical_id,
                        ),
                    )
            for item in bucket_rows:
                for identity in item["identity_values"]:
                    self._upsert_rewrite(connection, identity, canonical_id, galaxy, content_hash)
                if item is chosen_row and item["source_id"] == canonical_id and not item["missing_id"]:
                    continue
                if item is chosen_row and item["source_id"] != canonical_id:
                    self._upsert_rewrite(connection, item["source_id"], canonical_id, galaxy, content_hash)
            bucket_rows.clear()

        for row_key, galaxy, file_name, line_no, source_id, missing_id, candidates_json, content_hash, row_json in cursor:
            bucket = (galaxy, content_hash)
            if current_bucket != bucket:
                flush_bucket()
                current_bucket = bucket
            row = json.loads(row_json)
            bucket_rows.append(
                {
                    "row_key": row_key,
                    "galaxy": galaxy,
                    "file_name": file_name,
                    "line_no": line_no,
                    "source_id": source_id,
                    "missing_id": bool(missing_id),
                    "canonical_candidates": json.loads(candidates_json),
                    "content_hash": content_hash,
                    "row": row,
                    "identity_values": _row_identity_values(row),
                }
            )
        flush_bucket()
        connection.commit()

        rewrite_rows = [
            {
                "old_id": old_id,
                "canonical_id": canonical_id,
                "galaxy": galaxy,
                "content_hash": content_hash,
            }
            for old_id, canonical_id, galaxy, content_hash in connection.execute(
                "SELECT old_id, canonical_id, galaxy, content_hash FROM rewrite_map ORDER BY old_id"
            )
        ]
        _write_jsonl(self.refs_rewrite_map_path, rewrite_rows)
        self.rewrite_map = {row["old_id"]: row["canonical_id"] for row in rewrite_rows}

    def _pass_b_rewrite_refs(self, connection: sqlite3.Connection) -> None:
        for row_id, row_json in connection.execute(
            "SELECT row_id, row_json FROM normalized_rows ORDER BY file_name, row_id"
        ):
            row = json.loads(row_json)
            rewritten = _rewrite_refs(row, self.rewrite_map)
            if rewritten.get("id") != row_id:
                rewritten["id"] = row_id
            connection.execute(
                "UPDATE normalized_rows SET row_json = ? WHERE row_id = ?",
                (json.dumps(rewritten, ensure_ascii=False, sort_keys=True), row_id),
            )
        connection.commit()

    def _extract_edges(self, connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM ref_edges")
        for row_id, row_json in connection.execute(
            "SELECT row_id, row_json FROM normalized_rows ORDER BY file_name, row_id"
        ):
            row = json.loads(row_json)
            edges = _extract_ref_edges(row, row_id)
            if not edges:
                continue
            connection.executemany(
                "INSERT INTO ref_edges (source_id, target_id, field_path, field_kind) VALUES (?, ?, ?, ?)",
                edges,
            )
        connection.commit()

    def _pass_cd_symlinks_and_orphans(self, connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM orphan_actions")
        connection.execute("DELETE FROM edge_additions")
        existing_ids = {
            row_id
            for (row_id,) in connection.execute("SELECT row_id FROM normalized_rows")
        }
        row_to_galaxy = {
            row_id: galaxy
            for row_id, galaxy in connection.execute("SELECT row_id, galaxy FROM normalized_rows")
        }
        for source_id, target_id, field_path, field_kind in connection.execute(
            "SELECT source_id, target_id, field_path, field_kind FROM ref_edges ORDER BY source_id, field_path, target_id"
        ):
            if target_id not in existing_ids:
                connection.execute(
                    """
                    INSERT INTO orphan_actions (source_id, field_path, target_id, galaxy, reason)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        source_id,
                        field_path,
                        target_id,
                        row_to_galaxy.get(source_id, ""),
                        "missing_target",
                    ),
                )
                continue
            inverse_kind = _expected_inverse_kind(field_kind)
            if not inverse_kind:
                continue
            inverse_exists = connection.execute(
                """
                SELECT 1
                FROM ref_edges
                WHERE source_id = ? AND target_id = ? AND field_kind = ?
                LIMIT 1
                """,
                (target_id, source_id, inverse_kind),
            ).fetchone()
            if inverse_exists:
                continue
            inverse_path = _inverse_field_path(field_path, field_kind)
            connection.execute(
                """
                INSERT OR IGNORE INTO edge_additions (target_id, field_path, ref_id, source_id, field_kind)
                VALUES (?, ?, ?, ?, ?)
                """,
                (target_id, inverse_path, source_id, source_id, inverse_kind),
            )
        connection.commit()

        orphan_rows = [
            {
                "source_row_id": source_id,
                "field": field_path,
                "target_id": target_id,
                "galaxy": galaxy,
                "reason": reason,
            }
            for source_id, field_path, target_id, galaxy, reason in connection.execute(
                """
                SELECT source_id, field_path, target_id, galaxy, reason
                FROM orphan_actions
                ORDER BY source_id, field_path, target_id
                """
            )
        ]
        _write_jsonl(self.orphan_targets_path, orphan_rows)

        edge_rows = [
            {
                "target_row_id": target_id,
                "field_path": field_path,
                "ref_id": ref_id,
                "source_row_id": source_id,
                "field_kind": field_kind,
            }
            for target_id, field_path, ref_id, source_id, field_kind in connection.execute(
                """
                SELECT target_id, field_path, ref_id, source_id, field_kind
                FROM edge_additions
                ORDER BY target_id, field_path, ref_id
                """
            )
        ]
        _write_jsonl(self.bidirectional_edges_path, edge_rows)

        orphan_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for source_id, field_path, target_id in connection.execute(
            "SELECT source_id, field_path, target_id FROM orphan_actions ORDER BY source_id, field_path, target_id"
        ):
            orphan_map[source_id].append((field_path, target_id))
        addition_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for target_id, field_path, ref_id in connection.execute(
            "SELECT target_id, field_path, ref_id FROM edge_additions ORDER BY target_id, field_path, ref_id"
        ):
            addition_map[target_id].append((field_path, ref_id))

        for row_id, row_json in connection.execute(
            "SELECT row_id, row_json FROM normalized_rows ORDER BY file_name, row_id"
        ):
            row = json.loads(row_json)
            changed = False
            for field_path, target_id in orphan_map.get(row_id, []):
                _remove_target_from_path(row, field_path, target_id)
                changed = True
            for field_path, ref_id in addition_map.get(row_id, []):
                _append_target_to_path(row, field_path, ref_id)
                changed = True
            if changed:
                connection.execute(
                    "UPDATE normalized_rows SET row_json = ? WHERE row_id = ?",
                    (json.dumps(row, ensure_ascii=False, sort_keys=True), row_id),
                )
        connection.commit()

    def _pass_e_fill_matryoshka(self, connection: sqlite3.Connection) -> None:
        self._extract_edges(connection)
        meaning_ref_lookup: dict[str, list[str]] = defaultdict(list)
        for source_id, target_id, field_kind in connection.execute(
            """
            SELECT source_id, target_id, field_kind
            FROM ref_edges
            WHERE field_kind IN ('surface_forms.*.word_ref', 'surface_forms.*.char_refs')
            ORDER BY target_id, source_id
            """
        ):
            meaning_ref_lookup[target_id].append(source_id)

        row_cache: dict[str, dict[str, Any]] = {}
        for row_id, row_json in connection.execute("SELECT row_id, row_json FROM normalized_rows"):
            row_cache[row_id] = json.loads(row_json)

        fill_rows: list[dict[str, Any]] = []
        orphan_rows: list[dict[str, Any]] = []
        for row_id, galaxy, row_json in connection.execute(
            """
            SELECT row_id, galaxy, row_json
            FROM normalized_rows
            WHERE galaxy IN ('Word', 'Character')
            ORDER BY galaxy, row_id
            """
        ):
            row = json.loads(row_json)
            if _has_matryoshka_payload(row):
                continue
            source_candidates: list[str] = []
            for field_path in (
                "meaning_layer_id",
                "metadata.meaning_layer_id",
                "metadata.meaning_star_id",
                "meaning_ref",
                "metadata.meaning_ref",
            ):
                value = _get_path_value(row, field_path)
                if isinstance(value, str) and value.strip():
                    source_candidates.append(value.strip())
            source_candidates.extend(meaning_ref_lookup.get(row_id, []))
            if _row_has_meaning_star(row):
                source_candidates.insert(0, row_id)
            source_id = ""
            seed_text = ""
            for candidate in source_candidates:
                source_row = row_cache.get(candidate)
                if not source_row:
                    continue
                seed_text = _extract_meaning_source_text(source_row)
                if seed_text:
                    source_id = candidate
                    break
            if not seed_text:
                orphan_rows.append(
                    {
                        "source_row_id": row_id,
                        "field": "matryoshka",
                        "target_id": source_candidates[0] if source_candidates else "",
                        "galaxy": galaxy,
                        "reason": "missing_matryoshka_source",
                    }
                )
                continue
            vectors = _generate_matryoshka_vectors(seed_text)
            row["matryoshka"] = {f"dim_{dim}": values for dim, values in sorted(vectors.items())}
            metadata = row.get("metadata")
            if isinstance(metadata, dict):
                meaning_star = metadata.get("meaning_star")
                if isinstance(meaning_star, dict):
                    meaning_star["embedding_64"] = list(vectors[64])
                    meaning_star["embedding_128"] = list(vectors[128])
                    meaning_star["embedding_512"] = list(vectors[512])
                    meaning_star["embedding_2048"] = list(vectors[512]) + list(vectors[512]) + list(vectors[512]) + list(vectors[512])
            connection.execute(
                "UPDATE normalized_rows SET row_json = ? WHERE row_id = ?",
                (json.dumps(row, ensure_ascii=False, sort_keys=True), row_id),
            )
            fill_rows.append(
                {
                    "row_id": row_id,
                    "galaxy": galaxy,
                    "source_row_id": source_id,
                    "dims": list(MATRYOSHKA_DIMENSIONS),
                    "content_hash": _blake16(seed_text),
                }
            )
        connection.commit()

        existing_orphans = []
        if self.orphan_targets_path.exists():
            with self.orphan_targets_path.open("r", encoding="utf-8") as handle:
                existing_orphans = [json.loads(line) for line in handle if line.strip()]
        existing_orphans.extend(orphan_rows)
        existing_orphans.sort(key=lambda row: (row.get("source_row_id", ""), row.get("field", ""), row.get("target_id", "")))
        _write_jsonl(self.orphan_targets_path, existing_orphans)
        _write_jsonl(self.matryoshka_fills_path, sorted(fill_rows, key=lambda row: (row["galaxy"], row["row_id"])))

    def _copy_procedural_from_source(self, row: dict[str, Any], source_row: dict[str, Any]) -> bool:
        source_payload = _collect_procedural_payload(source_row)
        if not source_payload:
            return False
        if "meaning_rpn" in source_row and _non_placeholder_text(source_row.get("meaning_rpn")):
            row["meaning_rpn"] = _non_placeholder_text(source_row.get("meaning_rpn"))
            return True
        metadata = source_row.get("metadata")
        if isinstance(metadata, dict):
            meaning_star = metadata.get("meaning_star")
            if isinstance(meaning_star, dict):
                for field in PROCEDURAL_FIELDS:
                    text = _non_placeholder_text(meaning_star.get(field))
                    if text:
                        row[field] = text
                        return True
        for field in PROCEDURAL_FIELDS:
            text = _non_placeholder_text(source_row.get(field))
            if text:
                row[field] = text
                return True
        return False

    def _pass_f_raw_to_procedural(self, connection: sqlite3.Connection) -> None:
        self._extract_edges(connection)
        meaning_ref_lookup: dict[str, list[str]] = defaultdict(list)
        for source_id, target_id, field_kind in connection.execute(
            """
            SELECT source_id, target_id, field_kind
            FROM ref_edges
            WHERE field_kind IN ('surface_forms.*.word_ref', 'surface_forms.*.char_refs')
            ORDER BY target_id, source_id
            """
        ):
            meaning_ref_lookup[target_id].append(source_id)
        row_cache: dict[str, dict[str, Any]] = {}
        for row_id, row_json in connection.execute("SELECT row_id, row_json FROM normalized_rows"):
            row_cache[row_id] = json.loads(row_json)
        upgrade_rows: list[dict[str, Any]] = []
        for row_id, galaxy, row_json in connection.execute(
            "SELECT row_id, galaxy, row_json FROM normalized_rows ORDER BY galaxy, row_id"
        ):
            row = json.loads(row_json)
            if _collect_procedural_payload(row):
                continue
            source_candidates: list[str] = []
            for field_path in ("metadata.meaning_star_id", "meaning_ref", "metadata.meaning_ref"):
                value = _get_path_value(row, field_path)
                if isinstance(value, str) and value.strip():
                    source_candidates.append(value.strip())
            source_candidates.extend(meaning_ref_lookup.get(row_id, []))
            upgraded = False
            source_id = ""
            for candidate in source_candidates:
                source_row = row_cache.get(candidate)
                if not source_row or candidate == row_id:
                    continue
                if self._copy_procedural_from_source(row, source_row):
                    upgraded = True
                    source_id = candidate
                    break
            if upgraded:
                connection.execute(
                    "UPDATE normalized_rows SET row_json = ? WHERE row_id = ?",
                    (json.dumps(row, ensure_ascii=False, sort_keys=True), row_id),
                )
            upgrade_rows.append(
                {
                    "row_id": row_id,
                    "galaxy": galaxy,
                    "source_row_id": source_id,
                    "status": "upgraded" if upgraded else "deferred",
                }
            )
        connection.commit()
        _write_jsonl(self.procedural_upgrades_path, upgrade_rows)

    def _write_normalized_files(self, connection: sqlite3.Connection) -> None:
        if self.normalized_dir.exists():
            shutil.rmtree(self.normalized_dir)
        self.normalized_dir.mkdir(parents=True, exist_ok=True)
        rows_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for file_name, row_json in connection.execute(
            "SELECT file_name, row_json FROM normalized_rows ORDER BY file_name, row_id"
        ):
            rows_by_file[file_name].append(json.loads(row_json))
        for file_name, rows in sorted(rows_by_file.items()):
            _write_jsonl(self.normalized_dir / file_name, rows)

    def run_normalize(self) -> None:
        self._reset_stage_dir()
        connection = self._connect()
        try:
            self._init_db(connection)
            self._load_source_rows(connection)
            self._pass_a_assign_ids(connection)
            self._pass_b_rewrite_refs(connection)
            self._extract_edges(connection)
            self._pass_cd_symlinks_and_orphans(connection)
            self._pass_e_fill_matryoshka(connection)
            self._pass_f_raw_to_procedural(connection)
            self._write_normalized_files(connection)
        finally:
            connection.close()
            if self.db_path.exists():
                self.db_path.unlink()

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _totals_from_census(self, rows: list[dict[str, Any]]) -> dict[str, int]:
        totals = Counter()
        for row in rows:
            for key, value in row.items():
                if key in {"galaxy", "path"}:
                    continue
                totals[key] += int(value)
        return dict(totals)

    def _data_hashes(self) -> list[tuple[str, str]]:
        hashed_files: list[Path] = []
        for path in sorted(self.normalized_dir.glob("*.jsonl")):
            hashed_files.append(path)
        for path in (
            self.refs_rewrite_map_path,
            self.bidirectional_edges_path,
            self.orphan_targets_path,
            self.matryoshka_fills_path,
            self.procedural_upgrades_path,
            self.reaudit_census_path,
            self.reaudit_violations_path,
        ):
            if path.exists():
                hashed_files.append(path)
        rows: list[tuple[str, str]] = []
        for path in hashed_files:
            rows.append((_sha256_path(path), _display_path(path)))
        return rows

    def _write_hashes_file(self, data_hashes: list[tuple[str, str]]) -> None:
        rows = list(data_hashes)
        if self.stage_report_path.exists():
            rows.append((_sha256_path(self.stage_report_path), _display_path(self.stage_report_path)))
        if self.temp_report_path.exists():
            rows.append((_sha256_path(self.temp_report_path), _display_path(self.temp_report_path)))
        text = "\n".join(f"{digest}  {relpath}" for digest, relpath in rows) + ("\n" if rows else "")
        _write_text(self.hashes_path, text)

    def run_report(self) -> None:
        d1_rows = self._read_jsonl(DEFAULT_D1_CENSUS_PATH)
        d2_rows = self._read_jsonl(self.reaudit_census_path)
        d1_totals = self._totals_from_census(d1_rows)
        d2_totals = self._totals_from_census(d2_rows)
        rewrite_rows = self._read_jsonl(self.refs_rewrite_map_path)
        bidirectional_rows = self._read_jsonl(self.bidirectional_edges_path)
        orphan_rows = self._read_jsonl(self.orphan_targets_path)
        matryoshka_rows = self._read_jsonl(self.matryoshka_fills_path)
        procedural_rows = self._read_jsonl(self.procedural_upgrades_path)
        hash_rows = self._data_hashes()

        d1_by_file = {Path(row["path"]).name: row for row in d1_rows}
        d2_by_file = {Path(row["path"]).name: row for row in d2_rows}
        lines: list[str] = [
            f"Reproduction command: `{REPRO_COMMAND}`",
            "",
            "# D2 Normalization Report",
            "",
            f"- Live source root: `{self.storage_root}`",
            f"- Normalized staging root: `{self.normalized_dir}`",
            "",
            "## Pass Artifacts",
            "",
            f"- `normalized/*.jsonl`: {len(list(self.normalized_dir.glob('*.jsonl')))} files",
            f"- `refs_rewrite_map.jsonl`: {len(rewrite_rows)} rows",
            f"- `bidirectional_edges.jsonl`: {len(bidirectional_rows)} rows",
            f"- `orphan_targets.jsonl`: {len(orphan_rows)} rows",
            f"- `matryoshka_fills.jsonl`: {len(matryoshka_rows)} rows",
            f"- `procedural_upgrades.jsonl`: {len(procedural_rows)} rows",
            "",
            "## D1 vs D2 Totals",
            "",
            "| Metric | D1 | D2 | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
        total_pairs = [
            ("Rows", d1_totals.get("entry_count", 0), d2_totals.get("entry_count", 0)),
            ("Missing IDs", d1_totals.get("missing_id_count", 0), d2_totals.get("missing_id_count", 0)),
            ("Ad-hoc IDs", d1_totals.get("ad_hoc_id_count", 0), d2_totals.get("ad_hoc_id_count", 0)),
            ("Duplicate Rows", d1_totals.get("duplicate_row_count", 0), d2_totals.get("duplicate_row_count", 0)),
            ("Missing Matryoshka", d1_totals.get("matryoshka_missing_count", 0), d2_totals.get("matryoshka_missing_count", 0)),
            ("Raw Payload", d1_totals.get("raw_payload_count", 0), d2_totals.get("raw_payload_count", 0)),
            ("Unidirectional Sites", d1_totals.get("unidirectional_site_count", 0), d2_totals.get("unidirectional_site_count", 0)),
        ]
        for label, before, after in total_pairs:
            lines.append(f"| {label} | {before} | {after} | {after - before} |")

        lines.extend(
            [
                "",
                "## Per-Galaxy Delta",
                "",
                "| Galaxy File | D1 Rows | D2 Rows | D1 Raw | D2 Raw | D1 Missing Matryoshka | D2 Missing Matryoshka | D1 Ad-hoc | D2 Ad-hoc |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        file_names = sorted(set(d1_by_file) | set(d2_by_file))
        for file_name in file_names:
            before = d1_by_file.get(file_name, {})
            after = d2_by_file.get(file_name, {})
            lines.append(
                "| {name} | {d1_rows} | {d2_rows} | {d1_raw} | {d2_raw} | {d1_mat} | {d2_mat} | {d1_adhoc} | {d2_adhoc} |".format(
                    name=file_name,
                    d1_rows=before.get("entry_count", 0),
                    d2_rows=after.get("entry_count", 0),
                    d1_raw=before.get("raw_payload_count", 0),
                    d2_raw=after.get("raw_payload_count", 0),
                    d1_mat=before.get("matryoshka_missing_count", 0),
                    d2_mat=after.get("matryoshka_missing_count", 0),
                    d1_adhoc=before.get("ad_hoc_id_count", 0),
                    d2_adhoc=after.get("ad_hoc_id_count", 0),
                )
            )

        upgraded = sum(1 for row in procedural_rows if row.get("status") == "upgraded")
        deferred = sum(1 for row in procedural_rows if row.get("status") == "deferred")
        lines.extend(
            [
                "",
                "## Procedural Upgrade Outcome",
                "",
                f"- Upgraded rows: {upgraded}",
                f"- Deferred rows: {deferred}",
                "",
                "## Hashes",
                "",
            ]
        )
        for digest, relpath in hash_rows:
            lines.append(f"- `{relpath}` `{digest}`")

        report_text = "\n".join(lines) + "\n"
        _write_text(self.stage_report_path, report_text)
        _write_text(self.temp_report_path, report_text)
        self._write_hashes_file(hash_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D2 Galaxy normalization pipeline")
    parser.add_argument("--mode", choices=("normalize", "report"), default="normalize")
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--temp-report-path", type=Path, default=DEFAULT_TEMP_REPORT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    normalizer = D2Normalizer(
        storage_root=args.storage_root,
        stage_dir=args.stage_dir,
        temp_report_path=args.temp_report_path,
    )
    if args.mode == "normalize":
        normalizer.run_normalize()
        return 0
    normalizer.run_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
