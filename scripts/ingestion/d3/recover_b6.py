#!/usr/bin/env python3
"""D3 recovery pass for B6.1 duplicate diagnostics and B6 edge reconciliation."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ingestion.audit.galaxy_audit import (  # type: ignore[attr-defined]
    _content_hash,
    _looks_canonical_id,
    _non_placeholder_text,
    render_report as audit_render_report,
    run_scan as audit_run_scan,
)
from scripts.ingestion.d3.galaxy_d3 import (  # type: ignore[attr-defined]
    _iter_jsonl,
    _meaning_group_key,
    _read_jsonl,
    _sha256_path,
    _stable_json,
    _write_jsonl,
    _write_text,
)
from scripts.ingestion.normalize.galaxy_normalize import (  # type: ignore[attr-defined]
    _append_target_to_path,
    _expected_inverse_kind,
    _extract_ref_edges,
    _get_path_value,
    _inverse_field_path,
    _rewrite_refs,
)


DEFAULT_STAGE_DIR = REPO_ROOT / "scripts" / "ingestion" / "staging" / "D3_dedup"
DEFAULT_D2_STAGE_DIR = REPO_ROOT / "scripts" / "ingestion" / "staging" / "D2_normalize"
DEFAULT_TEMP_REPORT_PATH = REPO_ROOT / "TEMP" / "CODEX_D3_FINAL_REPORT_04.19.2026.md"
DEFAULT_DIGEST_PATH = REPO_ROOT / "TEMP" / "D3_duplicate_cluster_digest_04.19.2026.jsonl"
REPRO_COMMAND = "bash scripts/ingestion/d3/run.sh --recover-only"


def _display_run_relative(path: Path, run_dir: Path) -> str:
    return str(path.relative_to(run_dir))


class B6Recovery:
    def __init__(
        self,
        *,
        stage_dir: Path,
        d2_stage_dir: Path,
        temp_report_path: Path,
        digest_path: Path,
    ) -> None:
        self.stage_dir = stage_dir
        self.d2_stage_dir = d2_stage_dir
        self.temp_report_path = temp_report_path
        self.digest_path = digest_path
        self.source_rows_dir = self.stage_dir / "merged_by_galaxy"
        self.source_join_map_path = self.stage_dir / "dedup_join_map.jsonl"
        self.temp_run_dir = self.stage_dir.parent / ".D3_b6_recover_run"
        self._carry_forward_excludes = {
            "merged_by_galaxy",
            "merged_stars.jsonl",
            "re_audit_d3",
            "D3_FINAL_REPORT.md",
            "hashes.txt",
            "duplicate_cluster_digest.jsonl",
            "B6_edge_rewrites.jsonl",
            "B6_edge_reconciliation.jsonl",
        }

    def _reset_dir(self, path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    def _connect(self, db_path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(str(db_path))
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=MEMORY")
        return connection

    def _init_db(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE rows (
                row_id TEXT PRIMARY KEY,
                galaxy TEXT NOT NULL,
                file_name TEXT NOT NULL,
                row_json TEXT NOT NULL
            );
            CREATE TABLE aliases (
                alias_id TEXT PRIMARY KEY,
                canonical_id TEXT NOT NULL
            );
            CREATE TABLE edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                field_path TEXT NOT NULL,
                field_kind TEXT NOT NULL,
                inverse_kind TEXT NOT NULL
            );
            CREATE TABLE additions (
                target_id TEXT NOT NULL,
                inverse_field_path TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_field_path TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                PRIMARY KEY (target_id, inverse_field_path, source_id)
            );
            """
        )
        connection.execute("CREATE INDEX idx_rows_galaxy ON rows (galaxy, row_id)")
        connection.execute("CREATE INDEX idx_edges_source ON edges (source_id)")
        connection.execute("CREATE INDEX idx_edges_target ON edges (target_id)")
        connection.execute("CREATE INDEX idx_edges_inverse ON edges (source_id, target_id, field_kind)")
        connection.commit()

    def _load_rows(self, connection: sqlite3.Connection) -> int:
        count = 0
        for path in sorted(self.source_rows_dir.glob("*.jsonl")):
            galaxy = path.stem
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    row = json.loads(text)
                    row_id = str(row.get("id") or "").strip()
                    connection.execute(
                        "INSERT INTO rows (row_id, galaxy, file_name, row_json) VALUES (?, ?, ?, ?)",
                        (row_id, galaxy, path.name, json.dumps(row, ensure_ascii=False, sort_keys=True)),
                    )
                    count += 1
        connection.commit()
        return count

    def _load_alias_map(self, connection: sqlite3.Connection) -> dict[str, str]:
        alias_map: dict[str, str] = {}
        for row in _iter_jsonl(self.source_join_map_path):
            canonical_id = row["merged_row_id"]
            for alias in row.get("aliases", []):
                alias_map[alias] = canonical_id
                connection.execute(
                    "INSERT OR REPLACE INTO aliases (alias_id, canonical_id) VALUES (?, ?)",
                    (alias, canonical_id),
                )
        connection.commit()
        changed = True
        while changed:
            changed = False
            for alias, canonical in list(alias_map.items()):
                resolved = alias_map.get(canonical, canonical)
                if resolved != canonical:
                    alias_map[alias] = resolved
                    changed = True
        return alias_map

    def _carry_forward_stage_artifacts(self, run_dir: Path) -> None:
        for source in sorted(self.stage_dir.iterdir(), key=lambda path: path.name):
            if source.name.startswith(".") or source.name in self._carry_forward_excludes:
                continue
            destination = run_dir / source.name
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

    def _rewrite_row_refs(self, connection: sqlite3.Connection, alias_map: dict[str, str], output_path: Path) -> dict[str, int]:
        rows = connection.execute(
            "SELECT row_id, row_json FROM rows ORDER BY galaxy, row_id"
        )
        rewrite_rows: list[dict[str, Any]] = []
        rewritten_rows = 0
        rewritten_targets = 0
        for row_id, row_json in rows:
            row = json.loads(row_json)
            before_edges = _extract_ref_edges(row, row_id)
            rewritten = _rewrite_refs(row, alias_map)
            after_edges = _extract_ref_edges(rewritten, row_id)
            edge_pairs = zip(before_edges, after_edges, strict=False)
            local_changes = 0
            for before, after in edge_pairs:
                if before[1] == after[1]:
                    continue
                local_changes += 1
                rewrite_rows.append(
                    {
                        "reproduction_command": REPRO_COMMAND,
                        "operation": "rewrite_target",
                        "row_id": row_id,
                        "field_path": before[2],
                        "source_kind": before[3],
                        "from_target_id": before[1],
                        "to_target_id": after[1],
                    }
                )
            if local_changes:
                rewritten_rows += 1
                rewritten_targets += local_changes
                connection.execute(
                    "UPDATE rows SET row_json = ? WHERE row_id = ?",
                    (json.dumps(rewritten, ensure_ascii=False, sort_keys=True), row_id),
                )
        connection.commit()
        rewrite_rows.sort(
            key=lambda row: (
                row["row_id"],
                row["field_path"],
                row["from_target_id"],
                row["to_target_id"],
            )
        )
        _write_jsonl(output_path, rewrite_rows)
        return {
            "rewritten_rows": rewritten_rows,
            "rewritten_targets": rewritten_targets,
            "rewrite_log_rows": len(rewrite_rows),
        }

    def _rebuild_edges(self, connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM edges")
        batch: list[tuple[str, str, str, str, str]] = []
        for row_id, row_json in connection.execute("SELECT row_id, row_json FROM rows ORDER BY galaxy, row_id"):
            row = json.loads(row_json)
            for source_id, target_id, field_path, field_kind in _extract_ref_edges(row, row_id):
                inverse_kind = _expected_inverse_kind(field_kind) or ""
                batch.append((source_id, target_id, field_path, field_kind, inverse_kind))
                if len(batch) >= 10000:
                    connection.executemany(
                        """
                        INSERT INTO edges (source_id, target_id, field_path, field_kind, inverse_kind)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        batch,
                    )
                    batch.clear()
        if batch:
            connection.executemany(
                """
                INSERT INTO edges (source_id, target_id, field_path, field_kind, inverse_kind)
                VALUES (?, ?, ?, ?, ?)
                """,
                batch,
            )
        connection.commit()

    def _collect_reciprocal_additions(self, connection: sqlite3.Connection) -> list[dict[str, Any]]:
        connection.execute("DELETE FROM additions")
        additions: list[dict[str, Any]] = []
        cursor = connection.execute(
            """
            SELECT
                e.source_id,
                e.target_id,
                e.field_path,
                e.field_kind,
                e.inverse_kind,
                EXISTS(SELECT 1 FROM rows r WHERE r.row_id = e.target_id) AS target_exists,
                EXISTS(
                    SELECT 1
                    FROM edges inv
                    WHERE inv.source_id = e.target_id
                      AND inv.target_id = e.source_id
                      AND inv.field_kind = e.inverse_kind
                ) AS inverse_exists
            FROM edges e
            WHERE e.inverse_kind != ''
            ORDER BY e.source_id, e.field_path, e.target_id
            """
        )
        for source_id, target_id, field_path, field_kind, inverse_kind, target_exists, inverse_exists in cursor:
            if not target_exists or inverse_exists:
                continue
            inverse_field_path = _inverse_field_path(field_path, field_kind)
            connection.execute(
                """
                INSERT OR IGNORE INTO additions (
                    target_id,
                    inverse_field_path,
                    source_id,
                    source_field_path,
                    source_kind
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (target_id, inverse_field_path, source_id, field_path, field_kind),
            )
        connection.commit()
        for target_id, inverse_field_path, source_id, source_field_path, source_kind in connection.execute(
            """
            SELECT target_id, inverse_field_path, source_id, source_field_path, source_kind
            FROM additions
            ORDER BY target_id, inverse_field_path, source_id
            """
        ):
            additions.append(
                {
                    "reproduction_command": REPRO_COMMAND,
                    "operation": "insert_reciprocal",
                    "target_row_id": target_id,
                    "inverse_field_path": inverse_field_path,
                    "source_row_id": source_id,
                    "source_field_path": source_field_path,
                    "source_kind": source_kind,
                }
            )
        return additions

    def _apply_reciprocal_additions(self, connection: sqlite3.Connection, additions: list[dict[str, Any]], output_path: Path) -> dict[str, int]:
        for addition in additions:
            target_id = addition["target_row_id"]
            row_json = connection.execute(
                "SELECT row_json FROM rows WHERE row_id = ?",
                (target_id,),
            ).fetchone()
            if row_json is None:
                continue
            row = json.loads(row_json[0])
            current = _get_path_value(row, addition["inverse_field_path"])
            already_present = False
            if isinstance(current, list):
                already_present = addition["source_row_id"] in current
            elif isinstance(current, str):
                already_present = current == addition["source_row_id"]
            if already_present:
                continue
            _append_target_to_path(row, addition["inverse_field_path"], addition["source_row_id"])
            connection.execute(
                "UPDATE rows SET row_json = ? WHERE row_id = ?",
                (json.dumps(row, ensure_ascii=False, sort_keys=True), target_id),
            )
        connection.commit()
        _write_jsonl(output_path, additions)
        return {"reciprocal_additions": len(additions)}

    def _write_rows(self, connection: sqlite3.Connection, run_dir: Path) -> int:
        merged_dir = run_dir / "merged_by_galaxy"
        merged_dir.mkdir(parents=True, exist_ok=True)
        grouped: dict[str, list[dict[str, Any]]] = {}
        merged_rows: list[dict[str, Any]] = []
        for galaxy, file_name, row_json in connection.execute(
            "SELECT galaxy, file_name, row_json FROM rows ORDER BY galaxy, row_id"
        ):
            row = json.loads(row_json)
            grouped.setdefault(file_name, []).append(row)
            merged_rows.append(row)
        for file_name, rows in sorted(grouped.items()):
            rows.sort(key=lambda row: str(row.get("id", "")))
            _write_jsonl(merged_dir / file_name, rows)
        _write_jsonl(run_dir / "merged_stars.jsonl", merged_rows)
        return len(merged_rows)

    def _build_duplicate_digest(self, run_dir: Path, digest_stage_path: Path, digest_temp_path: Path) -> list[dict[str, Any]]:
        row_map: dict[str, dict[str, Any]] = {}
        for path in (run_dir / "merged_by_galaxy").glob("*.jsonl"):
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    row = json.loads(text)
                    row_map[str(row.get("id") or "")] = row

        clusters: dict[str, list[str]] = {}
        for violation in _iter_jsonl(run_dir / "re_audit_d3" / "violations.jsonl"):
            if violation.get("violation_kind") != "duplicate_content":
                continue
            clusters.setdefault(violation["content_hash"], []).append(violation["row_id"])

        digest_rows: list[dict[str, Any]] = []
        for audit_cluster_key in sorted(clusters):
            row_ids = sorted(set(clusters[audit_cluster_key]))
            for row_id_a, row_id_b in combinations(row_ids, 2):
                row_a = row_map.get(row_id_a, {})
                row_b = row_map.get(row_id_b, {})
                meaning_hash_a = _meaning_group_key(row_a) if row_a else ""
                meaning_hash_b = _meaning_group_key(row_b) if row_b else ""
                digest_rows.append(
                    {
                        "reproduction_command": REPRO_COMMAND,
                        "row_id_a": row_id_a,
                        "row_id_b": row_id_b,
                        "audit_cluster_key": audit_cluster_key,
                        "b2_meaning_hash_a": meaning_hash_a,
                        "b2_meaning_hash_b": meaning_hash_b,
                        "same_b2_meaning_hash": meaning_hash_a == meaning_hash_b,
                        "galaxy_a": row_a.get("galaxy", ""),
                        "galaxy_b": row_b.get("galaxy", ""),
                    }
                )
                if len(digest_rows) == 100:
                    _write_jsonl(digest_stage_path, digest_rows)
                    _write_jsonl(digest_temp_path, digest_rows)
                    return digest_rows
        _write_jsonl(digest_stage_path, digest_rows)
        _write_jsonl(digest_temp_path, digest_rows)
        return digest_rows

    def _build_hash_manifest(self, run_dir: Path) -> list[tuple[str, str]]:
        manifest_paths = [
            run_dir / "B6_edge_rewrites.jsonl",
            run_dir / "B6_edge_reconciliation.jsonl",
            run_dir / "duplicate_cluster_digest.jsonl",
            run_dir / "merged_stars.jsonl",
            run_dir / "re_audit_d3" / "galaxy_census.jsonl",
            run_dir / "re_audit_d3" / "violations.jsonl",
            run_dir / "re_audit_d3" / "RE_AUDIT_REPORT.md",
        ]
        return [(_display_run_relative(path, run_dir), _sha256_path(path)) for path in manifest_paths]

    def _raw_drift_breakdown(self) -> dict[str, int]:
        d2_raw_count = 0
        d2_raw_by_galaxy = Counter()
        for violation in _iter_jsonl(self.d2_stage_dir / "re_audit" / "violations.jsonl"):
            if violation.get("violation_kind") != "raw_payload":
                continue
            d2_raw_count += 1
            d2_raw_by_galaxy[violation["galaxy"]] += 1

        success = 0
        queued = 0
        success_by_galaxy = Counter()
        queued_by_galaxy = Counter()
        for row in _iter_jsonl(self.stage_dir / "rpn_upgrades.jsonl"):
            if row.get("conversion_status") == "success":
                success += 1
                success_by_galaxy[row["galaxy"]] += 1
            elif row.get("conversion_status") == "queued":
                queued += 1
                queued_by_galaxy[row["galaxy"]] += 1

        return {
            "d2_raw_count": d2_raw_count,
            "b3_success_count": success,
            "b3_queued_count": queued,
            "b3_processed_total": success + queued,
            "drift": (success + queued) - d2_raw_count,
        }

    def _render_report(self, run_dir: Path, stats: dict[str, int], manifest: list[tuple[str, str]]) -> str:
        totals = Counter()
        for row in _read_jsonl(run_dir / "re_audit_d3" / "galaxy_census.jsonl"):
            for key in ("duplicate_row_count", "matryoshka_missing_count", "raw_payload_count", "unidirectional_site_count"):
                totals[key] += int(row.get(key, 0) or 0)
        violation_counts = Counter()
        for row in _iter_jsonl(run_dir / "re_audit_d3" / "violations.jsonl"):
            violation_counts[row["violation_kind"]] += 1
        raw_drift = self._raw_drift_breakdown()
        lines = [
            "# D3 FINAL REPORT",
            "",
            f"Reproduction command: `{REPRO_COMMAND}`",
            "",
            "## Scope",
            "",
            f"- D3 stage root: `{self.stage_dir}`",
            f"- Duplicate digest: `{self.digest_path}`",
            "",
            "## B6.1 + B6 Results",
            "",
            f"- Alias target rewrites: {stats['rewritten_targets']}",
            f"- Rows touched by alias rewrites: {stats['rewritten_rows']}",
            f"- Reciprocal edges inserted: {stats['reciprocal_additions']}",
            f"- Merged rows after recovery: {stats['merged_row_count']}",
            f"- Duplicate digest entries: {stats['digest_rows']}",
            "",
            "## Post-B6 Gates",
            "",
            f"- `duplicate_row_count`: {totals['duplicate_row_count']}",
            f"- `missing_matryoshka`: {totals['matryoshka_missing_count']}",
            f"- `raw_payload`: {totals['raw_payload_count']}",
            f"- `unidirectional_site_count`: {totals['unidirectional_site_count']}",
            f"- `missing_target`: {violation_counts.get('missing_target', 0)}",
            "",
            "## Raw Payload Drift",
            "",
            f"- D2 `raw_payload` rows: {raw_drift['d2_raw_count']}",
            f"- B3 `success`: {raw_drift['b3_success_count']}",
            f"- B3 `queued`: {raw_drift['b3_queued_count']}",
            f"- B3 processed total: {raw_drift['b3_processed_total']}",
            f"- Drift (`processed_total - d2_raw_count`): {raw_drift['drift']}",
            "",
            "## Violation Counts",
            "",
        ]
        for kind, count in sorted(violation_counts.items()):
            lines.append(f"- `{kind}`: {count}")
        lines.extend(["", "## Artifact Hashes", ""])
        for rel_path, digest in manifest:
            lines.append(f"- `{rel_path}` `{digest}`")
        return "\n".join(lines) + "\n"

    def _execute(self, run_dir: Path) -> tuple[dict[str, int], list[tuple[str, str]]]:
        self._reset_dir(run_dir)
        db_path = run_dir / ".recover.sqlite3"
        connection = self._connect(db_path)
        try:
            self._init_db(connection)
            self._load_rows(connection)
            alias_map = self._load_alias_map(connection)
            rewrite_stats = self._rewrite_row_refs(connection, alias_map, run_dir / "B6_edge_rewrites.jsonl")
            self._rebuild_edges(connection)
            additions = self._collect_reciprocal_additions(connection)
            addition_stats = self._apply_reciprocal_additions(connection, additions, run_dir / "B6_edge_reconciliation.jsonl")
            self._rebuild_edges(connection)
            merged_row_count = self._write_rows(connection, run_dir)
        finally:
            connection.close()
            if db_path.exists():
                db_path.unlink()

        re_audit_dir = run_dir / "re_audit_d3"
        audit_run_scan(run_dir / "merged_by_galaxy", re_audit_dir)
        audit_render_report(run_dir / "merged_by_galaxy", re_audit_dir, re_audit_dir / "RE_AUDIT_REPORT.md")
        digest_rows = self._build_duplicate_digest(
            run_dir,
            run_dir / "duplicate_cluster_digest.jsonl",
            self.digest_path,
        )
        self._carry_forward_stage_artifacts(run_dir)
        manifest = self._build_hash_manifest(run_dir)
        stats = {
            **rewrite_stats,
            **addition_stats,
            "merged_row_count": merged_row_count,
            "digest_rows": len(digest_rows),
        }
        report_text = self._render_report(run_dir, stats, manifest)
        _write_text(run_dir / "D3_FINAL_REPORT.md", report_text)
        manifest.append(("D3_FINAL_REPORT.md", _sha256_path(run_dir / "D3_FINAL_REPORT.md")))
        _write_text(
            run_dir / "hashes.txt",
            "\n".join(f"{digest}  {rel_path}" for rel_path, digest in manifest) + "\n",
        )
        return stats, manifest

    def _compare_manifests(self, left: list[tuple[str, str]], right: list[tuple[str, str]]) -> None:
        if dict(left) != dict(right):
            diffs = []
            left_map = dict(left)
            right_map = dict(right)
            for key in sorted(set(left_map) | set(right_map)):
                if left_map.get(key) != right_map.get(key):
                    diffs.append(f"{key}: {left_map.get(key)} != {right_map.get(key)}")
            raise RuntimeError("B6 determinism check failed:\n" + "\n".join(diffs))

    def _publish(self, source_dir: Path, stats: dict[str, int]) -> None:
        published = self.stage_dir.parent / ".D3_b6_publish"
        if published.exists():
            shutil.rmtree(published)
        shutil.copytree(source_dir, published)
        if self.stage_dir.exists():
            shutil.rmtree(self.stage_dir)
        shutil.copytree(published, self.stage_dir)
        shutil.copyfile(self.stage_dir / "duplicate_cluster_digest.jsonl", self.digest_path)
        final_manifest = self._build_hash_manifest(self.stage_dir)
        report_text = self._render_report(self.stage_dir, stats, final_manifest)
        _write_text(self.stage_dir / "D3_FINAL_REPORT.md", report_text)
        final_manifest.append(("D3_FINAL_REPORT.md", _sha256_path(self.stage_dir / "D3_FINAL_REPORT.md")))
        _write_text(
            self.stage_dir / "hashes.txt",
            "\n".join(f"{digest}  {rel_path}" for rel_path, digest in final_manifest) + "\n",
        )
        shutil.copyfile(self.stage_dir / "D3_FINAL_REPORT.md", self.temp_report_path)
        shutil.rmtree(published)

    def run(self, *, verify_twice: bool) -> None:
        stats_a, manifest_a = self._execute(self.temp_run_dir)
        if verify_twice:
            _, manifest_b = self._execute(self.temp_run_dir)
            self._compare_manifests(manifest_a, manifest_b)
        self._publish(self.temp_run_dir, stats_a)
        if self.temp_run_dir.exists():
            shutil.rmtree(self.temp_run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover D3 with B6.1 duplicate digest and B6 edge reconciliation.")
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--d2-stage-dir", type=Path, default=DEFAULT_D2_STAGE_DIR)
    parser.add_argument("--temp-report-path", type=Path, default=DEFAULT_TEMP_REPORT_PATH)
    parser.add_argument("--digest-path", type=Path, default=DEFAULT_DIGEST_PATH)
    parser.add_argument("--no-verify-twice", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recovery = B6Recovery(
        stage_dir=args.stage_dir,
        d2_stage_dir=args.d2_stage_dir,
        temp_report_path=args.temp_report_path,
        digest_path=args.digest_path,
    )
    recovery.run(verify_twice=not args.no_verify_twice)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
