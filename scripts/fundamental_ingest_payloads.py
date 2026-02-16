#!/usr/bin/env python3
"""Fundamental payload ingestion into a single persistent Knowledgeverse world.

PURPOSE:
  Ingest foundational JSONL payloads produced by augmentation pipelines
  into one world instance with deterministic upsert behavior.

WHEN TO USE:
  - Initial/foundational world construction.
  - Knowledge expansion batches from benchmark/PDF augmentation outputs.

NOT FOR:
  - PTX hot-path inference.
  - Runtime daemon specialist routing.

ARCHITECTURE:
  - Single-world contract: one Knowledgeverse instance, default galaxies loaded once.
  - Mid-term ingestion step applies symlink compression (form->meaning references).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-']{1,31}")
SYMBOL_RE = re.compile(r"(<=|>=|==|!=|\+|\-|\*|/|=|\(|\)|\[|\]|\{|\}|\^|%|\d+)")


def _iter_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _counts(kv: Knowledgeverse) -> dict[str, int]:
    return {name: len(kv.galaxy_manager.get_galaxy(name).entries) for name in kv.DEFAULT_GALAXIES}


def _word_fallback_id(token: str) -> str:
    return f"word_bench_{hashlib.sha1(token.encode('utf-8', errors='ignore')).hexdigest()[:12]}"


def _build_word_name_index(kv: Knowledgeverse) -> dict[str, str]:
    index: dict[str, str] = {}
    word_galaxy = kv.galaxy_manager.get_galaxy("Word")
    for entry in getattr(word_galaxy, "entries", []):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip().lower()
        entry_id = str(entry.get("id", "")).strip()
        if not name or not entry_id:
            continue
        index.setdefault(name, entry_id)
    return index


def _extract_text_tokens(entry: dict[str, Any]) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    name = entry.get("name")
    if isinstance(name, str) and name.strip():
        texts.append(name)
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        for key in ("summary", "embedding_text", "question_text", "problem_text", "ollama_hint"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value)

    word_tokens: list[str] = []
    symbol_tokens: list[str] = []
    for text in texts:
        word_tokens.extend(m.group(0).lower() for m in WORD_RE.finditer(text))
        symbol_tokens.extend(m.group(0) for m in SYMBOL_RE.finditer(text))
    return word_tokens, symbol_tokens


def _apply_symlink_compression(entry: dict[str, Any], *, word_index: dict[str, str]) -> dict[str, Any]:
    out = dict(entry)
    metadata = out.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)

    word_tokens, symbol_tokens = _extract_text_tokens(out)
    unique_words = list(dict.fromkeys(word_tokens))
    unique_symbols = list(dict.fromkeys(symbol_tokens))

    word_refs = [word_index.get(tok, _word_fallback_id(tok)) for tok in unique_words[:256]]
    char_refs = [f"char_u{ord(ch):04x}" for tok in unique_words[:128] for ch in tok[:24]]
    char_refs = list(dict.fromkeys(char_refs))[:512]

    if word_refs:
        metadata["word_refs"] = word_refs
    if char_refs:
        metadata["char_refs"] = char_refs
    if unique_symbols:
        metadata["symbol_refs"] = unique_symbols[:256]
    metadata["symlink_compression"] = "applied_v1"
    metadata["symlink"] = metadata.get("symlink") or "character_galaxy|word_galaxy"
    out["metadata"] = metadata
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest JSONL payload(s) using one persistent Knowledgeverse")
    parser.add_argument("--payload", type=Path, nargs="+", required=True, help="One or more payload JSONL files")
    parser.add_argument("--storage-root", type=Path, default=Path("../Knowledge3D.local"))
    parser.add_argument("--report", type=Path, default=Path("../Knowledge3D.local/results/external_ingestion_report.json"))
    parser.add_argument(
        "--disable-symlink-compression",
        action="store_true",
        help="Skip text->symlink metadata swap during ingestion (debug only).",
    )
    args = parser.parse_args()

    kv = Knowledgeverse(storage_root=args.storage_root, eager_load_default_galaxies=True)
    before = _counts(kv)
    word_index = _build_word_name_index(kv)

    seen_per_galaxy: dict[str, set[str]] = {
        name: {str(e.get("id", "")) for e in kv.galaxy_manager.get_galaxy(name).entries}
        for name in kv.DEFAULT_GALAXIES
    }

    ingest_stats: dict[str, dict[str, int]] = {}
    total_added = 0
    total_skipped = 0

    for payload_path in args.payload:
        rows = _iter_rows(payload_path)
        added = 0
        skipped = 0
        for row in rows:
            galaxy = str(row.get("galaxy", "")).strip()
            entry = row.get("entry") if isinstance(row.get("entry"), dict) else None
            if not galaxy or entry is None:
                skipped += 1
                continue

            entry_id = str(entry.get("id", "")).strip()
            if not entry_id:
                skipped += 1
                continue

            if galaxy not in seen_per_galaxy:
                seen_per_galaxy[galaxy] = {str(e.get("id", "")) for e in kv.galaxy_manager.get_galaxy(galaxy).entries}

            if entry_id in seen_per_galaxy[galaxy]:
                skipped += 1
                continue

            prepared_entry = (
                dict(entry)
                if args.disable_symlink_compression
                else _apply_symlink_compression(entry, word_index=word_index)
            )
            kv.galaxy_manager.add_entry(galaxy, prepared_entry)
            seen_per_galaxy[galaxy].add(entry_id)
            added += 1

        ingest_stats[str(payload_path)] = {"rows": len(rows), "added": added, "skipped": skipped}
        total_added += added
        total_skipped += skipped

    after = _counts(kv)

    report = {
        "storage_root": str(args.storage_root),
        "shared_instance": True,
        "instance_id": id(kv),
        "payloads": [str(p) for p in args.payload],
        "ingest_stats": ingest_stats,
        "totals": {"added": total_added, "skipped": total_skipped},
        "galaxy_counts_before": before,
        "galaxy_counts_after": after,
        "symlink_compression_enabled": not bool(args.disable_symlink_compression),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[ingest] shared_instance=true instance_id={id(kv)}")
    print(f"[ingest] added={total_added} skipped={total_skipped}")
    print(f"[ingest] report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
