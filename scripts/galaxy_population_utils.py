from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def upsert_entries(
    path: Path,
    generated: list[dict[str, Any]],
    *,
    remove_ids: set[str] | None = None,
) -> dict[str, int]:
    remove_ids = {str(item).strip() for item in (remove_ids or set()) if str(item).strip()}
    existing = read_jsonl(path)
    existing_ids = {
        str(row.get("id", "")).strip()
        for row in existing
        if str(row.get("id", "")).strip()
    }
    kept_existing: list[dict[str, Any]] = []
    removed = 0
    generated_ids = {
        str(row.get("id", "")).strip()
        for row in generated
        if str(row.get("id", "")).strip()
    }
    for row in existing:
        row_id = str(row.get("id", "")).strip()
        if row_id in remove_ids or row_id in generated_ids:
            removed += 1
            continue
        kept_existing.append(row)

    write_jsonl(path, kept_existing + [dict(row) for row in generated])
    before = len(existing)
    after = len(kept_existing) + len(generated)
    appended = sum(
        1
        for row in generated
        if str(row.get("id", "")).strip() and str(row.get("id", "")).strip() not in existing_ids
    )
    replaced = sum(
        1
        for row in generated
        if str(row.get("id", "")).strip() and str(row.get("id", "")).strip() in existing_ids
    )
    return {
        "before": before,
        "after": after,
        "appended": appended,
        "replaced": replaced,
        "removed": removed,
    }
