#!/usr/bin/env python3
"""Differentiate residual duplicate-content clusters via cloud proceduralizer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.ingestion.mcp_web_search import DEFAULT_CACHE_DIR, WebSearchUnavailable, web_search
from knowledge3d.ingestion.ollama_manager import OllamaManager
from knowledge3d.tools.knowledge_proceduralizer import differentiate_cluster_receipts
from scripts.ingestion.audit.galaxy_audit import _collect_procedural_payload, _content_hash


DEFAULT_VIOLATIONS = REPO_ROOT / "scripts" / "ingestion" / "staging" / "D3_dedup" / "re_audit_d3" / "violations.jsonl"
DEFAULT_MERGED_IN = REPO_ROOT / "scripts" / "ingestion" / "staging" / "D3_dedup" / "merged_stars.jsonl"
DEFAULT_UNRESOLVED = REPO_ROOT / "scripts" / "ingestion" / "staging" / "D3_dedup" / "differentiate_b7_unresolved.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("star_id") or "").strip()


def _cache_path_for_query(query: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(str(query).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


class CountingWebSearch:
    def __init__(self, *, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.issued = 0
        self.cache_hits = 0

    def __call__(self, query: str, max_results: int) -> list[dict[str, str]]:
        cache_path = _cache_path_for_query(query, self.cache_dir)
        if cache_path.exists():
            self.cache_hits += 1
        self.issued += 1
        return web_search(query, max_results=max_results, cache_dir=self.cache_dir)


def _cluster_map(violations: list[dict[str, Any]]) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = defaultdict(list)
    for row in violations:
        if row.get("violation_kind") != "duplicate_content":
            continue
        content_hash = str(row.get("content_hash") or "").strip()
        row_id = str(row.get("row_id") or "").strip()
        if content_hash and row_id:
            clusters[content_hash].append(row_id)
    return {key: sorted(set(value)) for key, value in clusters.items()}


def _patch_row_from_receipt(
    original_row: dict[str, Any],
    *,
    request: Any,
    receipt: Any,
) -> dict[str, Any]:
    if not getattr(receipt, "parsed_bundle", None):
        return copy.deepcopy(original_row)
    bundle = receipt.parsed_bundle
    packets = list(getattr(bundle, "knowledge_packets", []) or [])
    if not packets:
        return copy.deepcopy(original_row)
    packet = packets[0]
    patched = copy.deepcopy(original_row)
    patched["name"] = str(packet.summary or patched.get("name") or "").strip()
    patched["rpn_program"] = str(packet.meaning_rpn or patched.get("rpn_program") or "").strip()
    metadata = dict(patched.get("metadata") or {})
    existing_sources = [
        str(item).strip()
        for item in list(metadata.get("sources") or [])
        if str(item).strip()
    ]
    for source in list(packet.sources or []):
        source_text = str(source or "").strip()
        if source_text and source_text not in existing_sources:
            existing_sources.append(source_text)
    metadata.update(
        {
            "confidence": float(packet.confidence),
            "needs_review": bool(packet.needs_review),
            "surface_forms": dict(packet.surface_forms),
            "symbol_refs": list(packet.symbol_refs),
            "word_refs": list(packet.word_refs),
            "taxonomy_refs": list(packet.taxonomy_refs),
            "grammar_refs": list(packet.grammar_refs),
            "reality_refs": list(packet.reality_refs),
            "meta_refs": sorted(
                {
                    *(str(item).strip() for item in list(metadata.get("meta_refs") or []) if str(item).strip()),
                    *(str(item).strip() for item in list(packet.meta_refs or []) if str(item).strip()),
                    "b7_cloud_differentiation",
                }
            ),
            "relationships": list(packet.relationships),
            "sources": existing_sources,
            "proceduralizer_request_mode": str(request.mode or "standard"),
            "meaning_star": {
                "star_id": str(packet.star_id or packet.proposed_star_id or _row_id(patched)).strip(),
                "layer_kind": str(packet.layer_kind or "meaning").strip(),
                "meaning_class": str(packet.meaning_class or "entry").strip(),
                "meaning_rpn": str(packet.meaning_rpn or "").strip(),
                "summary": str(packet.summary or "").strip(),
                "domain": str(packet.domain or "").strip(),
                "surface_forms": dict(packet.surface_forms),
                "symbol_refs": list(packet.symbol_refs),
                "word_refs": list(packet.word_refs),
                "taxonomy_refs": list(packet.taxonomy_refs),
                "grammar_refs": list(packet.grammar_refs),
                "reality_refs": list(packet.reality_refs),
                "meta_refs": list(packet.meta_refs),
                "sources": list(packet.sources),
            },
        }
    )
    patched["metadata"] = metadata
    return patched


def _resolved_bundle(receipt: Any) -> bool:
    bundle = getattr(receipt, "parsed_bundle", None)
    if bundle is None:
        return False
    if str(getattr(bundle, "status", "") or "").strip().lower() == "unresolvable":
        return False
    if not bool(getattr(receipt, "schema_ok", False)):
        return False
    if str(getattr(receipt, "failure_code", "") or "").strip():
        return False
    return str(getattr(bundle, "ingest_action", "") or "").strip().lower() == "augment" and bool(
        list(getattr(bundle, "knowledge_packets", []) or [])
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    violations = _read_jsonl(args.violations)
    merged_rows = _read_jsonl(args.merged_in)
    row_positions = {_row_id(row): index for index, row in enumerate(merged_rows)}
    clusters = _cluster_map(violations)
    selected = [
        (content_hash, row_ids)
        for content_hash, row_ids in clusters.items()
        if int(args.min_cluster) <= len(row_ids) <= int(args.max_cluster)
    ]
    selected.sort(key=lambda item: (len(item[1]), min(row_positions.get(row_id, 10**12) for row_id in item[1]), item[0]))
    if args.limit:
        selected = selected[: int(args.limit)]

    search_client = CountingWebSearch(cache_dir=Path(args.cache_dir))
    manager = OllamaManager(default_timeout=float(args.timeout))
    unresolved_rows: list[dict[str, Any]] = []
    rows_enriched = 0
    clusters_attempted = 0
    clusters_resolved = 0

    for content_hash, row_ids in selected:
        clusters_attempted += 1
        cluster_results = differentiate_cluster_receipts(
            row_ids,
            Path(args.merged_in),
            search_client,
            manager,
            model=str(args.model).strip(),
            num_ctx=int(args.num_ctx),
            max_peer_samples=int(args.max_peer_samples),
            max_web_results=int(args.max_web_results),
            timeout=float(args.timeout),
        )
        cluster_changed = False
        for result in cluster_results:
            row_id = str(result["row_id"])
            original_row = merged_rows[row_positions[row_id]]
            if not _resolved_bundle(result["receipt"]):
                unresolved_rows.append(
                    {
                        "row_id": row_id,
                        "content_hash": content_hash,
                        "reason": str(getattr(result["receipt"].parsed_bundle, "status", "") or "unresolvable"),
                        "query": str(result["query"]),
                    }
                )
                continue
            patched = _patch_row_from_receipt(
                original_row,
                request=result["request"],
                receipt=result["receipt"],
            )
            if _content_hash(original_row, _collect_procedural_payload(original_row)) == _content_hash(
                patched,
                _collect_procedural_payload(patched),
            ):
                unresolved_rows.append(
                    {
                        "row_id": row_id,
                        "content_hash": content_hash,
                        "reason": "hash_unchanged",
                        "query": str(result["query"]),
                    }
                )
                continue
            merged_rows[row_positions[row_id]] = patched
            rows_enriched += 1
            cluster_changed = True
        if cluster_changed:
            clusters_resolved += 1

    if args.merged_out:
        _write_jsonl(Path(args.merged_out), merged_rows)
    _write_jsonl(Path(args.unresolved_out), sorted(unresolved_rows, key=lambda row: (row["content_hash"], row["row_id"], row["reason"])))

    summary = {
        "clusters_attempted": clusters_attempted,
        "clusters_resolved": clusters_resolved,
        "rows_enriched": rows_enriched,
        "rows_unresolved": len(unresolved_rows),
        "web_searches_issued": search_client.issued,
        "web_cache_hits": search_client.cache_hits,
        "dry_run": bool(args.dry_run),
        "merged_out": str(args.merged_out or ""),
        "unresolved_out": str(args.unresolved_out),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--violations", type=Path, default=DEFAULT_VIOLATIONS)
    parser.add_argument("--merged-in", type=Path, default=DEFAULT_MERGED_IN)
    parser.add_argument("--merged-out", type=Path, required=False)
    parser.add_argument("--unresolved-out", type=Path, default=DEFAULT_UNRESOLVED)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--min-cluster", type=int, default=2)
    parser.add_argument("--max-cluster", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="qwen3.5:397b-cloud")
    parser.add_argument("--num-ctx", type=int, default=65536)
    parser.add_argument("--max-peer-samples", type=int, default=3)
    parser.add_argument("--max-web-results", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.merged_out is None:
        parser.error("--merged-out is required")
    try:
        summary = run(args)
    except WebSearchUnavailable as exc:
        print(json.dumps({"status": "web_search_unavailable", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
