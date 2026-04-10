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

from knowledge3d.local_paths import resolve_storage_root
from knowledge3d.ingestion import (
    ingest_cas_grammar,
    ingest_entity_bootstrap,
    ingest_physics_bootstrap,
    ingest_sas_bootstrap,
)
from knowledge3d.knowledgeverse.galaxy_manager import normalize_disk_entry
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-']{1,31}")
SYMBOL_RE = re.compile(r"(<=|>=|==|!=|\+|\-|\*|/|=|\(|\)|\[|\]|\{|\}|\^|%|\d+)")
BENCHMARK_SOURCE_PREFIXES = (
    "benchmark_augmentation_",
    "mmlu",
    "gsm8k",
    "lhe",
    "amc_aime",
    "omni_math",
    "imo",
    "arc",
)
BENCHMARK_NAME_TOKENS = (
    "mmlu",
    "gsm8k",
    "lhe",
    "arc",
    "amc",
    "aime",
    "omni_math",
    "imo",
)
BENCHMARK_NAME_TOKEN_SET = frozenset(BENCHMARK_NAME_TOKENS) - {"amc_aime", "omni_math"}
BENCHMARK_NAME_COMPOSITES = frozenset({("amc", "aime"), ("omni", "math")})
BENCHMARK_NAME_SPLIT_RE = re.compile(r"[^a-z0-9]+")
ROUTE_FIELD_KEYS = (
    "route_family",
    "selection_role",
    "layer_id",
    "answer_eligible",
    "route_policy",
    "executor_refs",
    "validator_refs",
    "anti_pattern_refs",
)


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
    description = entry.get("description")
    if isinstance(description, str) and description.strip():
        texts.append(description)
    tags = entry.get("tags")
    if isinstance(tags, list):
        texts.extend(str(tag) for tag in tags if str(tag).strip())
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        for key in (
            "summary",
            "embedding_text",
            "question_text",
            "problem_text",
            "ollama_hint",
            "description",
            "procedural_goal",
        ):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value)
        for key in ("inputs", "outputs", "modalities", "cross_modal", "promotion_targets"):
            value = metadata.get(key)
            if isinstance(value, list):
                texts.extend(str(item) for item in value if str(item).strip())

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


def _benchmark_source_name(entry: dict[str, Any]) -> str:
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    source = str(metadata.get("source") or "").strip().lower()
    return source


def _is_benchmark_derived_entry(entry: dict[str, Any]) -> bool:
    source = _benchmark_source_name(entry)
    return any(source.startswith(prefix) for prefix in BENCHMARK_SOURCE_PREFIXES)


def _requires_benchmark_route_contract(galaxy: str, entry: dict[str, Any]) -> bool:
    if not _is_benchmark_derived_entry(entry):
        return False
    return str(galaxy).strip() not in {"Word", "Drawing", "Character"}


def _normalize_route_contract(entry: dict[str, Any]) -> dict[str, Any]:
    row = dict(entry)
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    for key in ROUTE_FIELD_KEYS:
        if row.get(key) is None and key in metadata:
            row[key] = metadata.get(key)
    row["metadata"] = metadata
    return row


def _missing_route_fields(entry: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in ("route_family", "selection_role", "route_policy"):
        value = entry.get(key)
        if isinstance(value, dict):
            if not value:
                missing.append(key)
            continue
        if value is None or not str(value).strip():
            missing.append(key)
    for key in ("layer_id", "answer_eligible"):
        if key not in entry:
            missing.append(key)
    return missing


def _benchmark_name_leakage(entry: dict[str, Any]) -> list[str]:
    leaks: list[str] = []
    for key in ("id", "category", "name"):
        value = str(entry.get(key) or "").strip().lower()
        if not value:
            continue
        parts = [part for part in BENCHMARK_NAME_SPLIT_RE.split(value) if part]
        if any(part in BENCHMARK_NAME_TOKEN_SET for part in parts):
            leaks.append(key)
            continue
        if any((parts[idx], parts[idx + 1]) in BENCHMARK_NAME_COMPOSITES for idx in range(len(parts) - 1)):
            leaks.append(key)
    return leaks


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest JSONL payload(s) using one persistent Knowledgeverse")
    parser.add_argument("--payload", type=Path, nargs="+", required=True, help="One or more payload JSONL files")
    parser.add_argument("--storage-root", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--disable-symlink-compression",
        action="store_true",
        help="Skip text->symlink metadata swap during ingestion (debug only).",
    )
    args = parser.parse_args()
    storage_root = resolve_storage_root(args.storage_root)
    report_path = Path(args.report) if args.report is not None else (storage_root / "results" / "external_ingestion_report.json")

    kv = Knowledgeverse(
        storage_root=storage_root,
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    physics_bootstrap_ingested = ingest_physics_bootstrap(kv.galaxy_manager)
    entity_bootstrap_ingested = ingest_entity_bootstrap(kv.galaxy_manager)
    cas_grammar_ingested = ingest_cas_grammar(kv.galaxy_manager)
    _sas_values, _sas_star_ids, sas_stars = ingest_sas_bootstrap(kv.galaxy_manager)
    sas_bootstrap_ingested = len(sas_stars)
    before = _counts(kv)
    word_index = _build_word_name_index(kv)

    seen_per_galaxy: dict[str, set[str]] = {
        name: {str(e.get("id", "")) for e in kv.galaxy_manager.get_galaxy(name).entries}
        for name in kv.DEFAULT_GALAXIES
    }

    ingest_stats: dict[str, dict[str, int]] = {}
    total_added = 0
    total_skipped = 0
    rejected_missing_route_metadata = 0
    rejected_benchmark_name_leakage = 0

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

            prepared_entry = normalize_disk_entry(galaxy, _normalize_route_contract(entry))
            if _requires_benchmark_route_contract(galaxy, prepared_entry):
                leaks = _benchmark_name_leakage(prepared_entry)
                if leaks:
                    skipped += 1
                    rejected_benchmark_name_leakage += 1
                    continue
                missing = _missing_route_fields(prepared_entry)
                if missing:
                    skipped += 1
                    rejected_missing_route_metadata += 1
                    continue

            prepared_entry = (
                dict(prepared_entry)
                if args.disable_symlink_compression
                else _apply_symlink_compression(prepared_entry, word_index=word_index)
            )
            kv.galaxy_manager.add_entry(galaxy, prepared_entry)
            seen_per_galaxy[galaxy].add(entry_id)
            added += 1

        ingest_stats[str(payload_path)] = {"rows": len(rows), "added": added, "skipped": skipped}
        total_added += added
        total_skipped += skipped

    after = _counts(kv)

    checkpoint_summary = kv.save_consolidated_state()

    report = {
        "storage_root": str(storage_root),
        "shared_instance": True,
        "instance_id": id(kv),
        "payloads": [str(p) for p in args.payload],
        "physics_bootstrap_ingested": physics_bootstrap_ingested,
        "entity_bootstrap_ingested": entity_bootstrap_ingested,
        "cas_grammar_ingested": cas_grammar_ingested,
        "sas_bootstrap_ingested": sas_bootstrap_ingested,
        "ingest_stats": ingest_stats,
        "totals": {
            "added": total_added,
            "skipped": total_skipped,
            "rejected_missing_route_metadata": rejected_missing_route_metadata,
            "rejected_benchmark_name_leakage": rejected_benchmark_name_leakage,
        },
        "galaxy_counts_before": before,
        "galaxy_counts_after": after,
        "symlink_compression_enabled": not bool(args.disable_symlink_compression),
        "checkpoint": dict(checkpoint_summary or {}),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[ingest] shared_instance=true instance_id={id(kv)}")
    print(f"[ingest] added={total_added} skipped={total_skipped}")
    print(f"[ingest] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
