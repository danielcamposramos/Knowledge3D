#!/usr/bin/env python3
"""Differentiate residual duplicate-content clusters via cloud proceduralizer."""

from __future__ import annotations

import argparse
import asyncio
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import random
import socket
import sys
import threading
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.ingestion.mcp_web_search import DEFAULT_CACHE_DIR, WebSearchUnavailable, web_search
from knowledge3d.ingestion.ollama_manager import OllamaManager
from knowledge3d.ingestion.proceduralizer_contract import PROCEDURALIZER_MODEL_PROFILES, ProceduralizerRequest
from knowledge3d.ingestion.proceduralizer_wine import ProceduralizerWineBridge
from knowledge3d.tools.knowledge_proceduralizer import (
    MODEL_OPTIONS,
    _differentiation_content,
    _differentiation_query,
    _peer_sample_text,
    _row_anchor,
    _row_domain_hint,
    build_rag_context,
    load_star_rows_index,
)
from scripts.ingestion.audit.galaxy_audit import _collect_procedural_payload, _content_hash


DEFAULT_VIOLATIONS = REPO_ROOT / "scripts" / "ingestion" / "staging" / "D3_dedup" / "re_audit_d3" / "violations.jsonl"
DEFAULT_MERGED_IN = REPO_ROOT / "scripts" / "ingestion" / "staging" / "D3_dedup" / "merged_stars.jsonl"
DEFAULT_UNRESOLVED = REPO_ROOT / "scripts" / "ingestion" / "staging" / "D3_dedup" / "differentiate_b7_unresolved.jsonl"
DEFAULT_STAGE_ROOT = REPO_ROOT / "scripts" / "ingestion" / "staging" / "D3_dedup" / "differentiate_b7"
CLAIM_TTL_SECONDS = 20 * 60
STATUS_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_text() -> str:
    return _utc_now().strftime(STATUS_TIME_FORMAT)


def _parse_time(text: str) -> datetime | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield json.loads(text)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    _write_text_atomic(path, text)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("star_id") or "").strip()


def _default_shard_filename(row: dict[str, Any]) -> str:
    galaxy = str(row.get("galaxy") or "").strip()
    star_type = str(row.get("star_type") or "").strip().lower()
    row_id = _row_id(row)
    if galaxy == "meaning_layer_stars" or star_type == "meaning_concept" or row_id.startswith("meaning/"):
        return "meaning_layer_stars.jsonl"
    if galaxy:
        return f"{galaxy}.jsonl"
    return "_unknown.jsonl"


def _row_file_map(merged_by_galaxy_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not merged_by_galaxy_dir.exists():
        return mapping
    for path in sorted(merged_by_galaxy_dir.glob("*.jsonl")):
        for row in _iter_jsonl(path):
            row_id = _row_id(row)
            if row_id:
                mapping[row_id] = path.name
    return mapping


def _rewrite_merged_by_galaxy(*, merged_rows_path: Path, merged_by_galaxy_dir: Path, row_to_file_name: dict[str, str]) -> dict[str, int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_count = 0
    for row in _iter_jsonl(merged_rows_path):
        row_count += 1
        row_id = _row_id(row)
        file_name = row_to_file_name.get(row_id) or _default_shard_filename(row)
        grouped[file_name].append(row)
    if merged_by_galaxy_dir.exists():
        for path in sorted(merged_by_galaxy_dir.glob("*.jsonl")):
            path.unlink()
    merged_by_galaxy_dir.mkdir(parents=True, exist_ok=True)
    for file_name, rows in sorted(grouped.items()):
        rows.sort(key=_row_id)
        _write_jsonl_atomic(merged_by_galaxy_dir / file_name, rows)
    return {"row_count": row_count, "shard_count": len(grouped)}


def _cache_path_for_query(query: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(str(query).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


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


def _patch_row_from_receipt(original_row: dict[str, Any], *, request: Any, receipt: Any) -> dict[str, Any]:
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
    existing_sources = [str(item).strip() for item in list(metadata.get("sources") or []) if str(item).strip()]
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


class CountingWebSearch:
    def __init__(self, *, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.issued = 0
        self.cache_hits = 0
        self._lock = threading.Lock()

    def __call__(self, query: str, max_results: int) -> list[dict[str, str]]:
        cache_path = _cache_path_for_query(query, self.cache_dir)
        with self._lock:
            if cache_path.exists():
                self.cache_hits += 1
            self.issued += 1
        return web_search(query, max_results=max_results, cache_dir=self.cache_dir)


def _build_cluster_manifest(args: argparse.Namespace) -> dict[str, Any]:
    violations = _read_jsonl(args.violations)
    clusters = _cluster_map(violations)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for content_hash, row_ids in sorted(clusters.items()):
        if not (int(args.min_cluster) <= len(row_ids) <= int(args.max_cluster)):
            continue
        path = out_dir / f"{content_hash}.cluster.json"
        payload = {"content_hash": content_hash, "row_ids": row_ids}
        if path.exists():
            skipped += 1
            continue
        _write_json_atomic(path, payload)
        written += 1
    summary = {"clusters_written": written, "clusters_skipped": skipped, "out_dir": str(out_dir)}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def _stage_paths(out_root: Path) -> dict[str, Path]:
    return {
        "root": out_root,
        "clusters": out_root / "clusters",
        "claims": out_root / "claims",
        "done": out_root / "done",
        "enriched": out_root / "enriched",
        "unresolved": out_root / "unresolved",
        "workers": out_root / "workers",
    }


def _ensure_stage_dirs(out_root: Path) -> dict[str, Path]:
    paths = _stage_paths(out_root)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _worker_log(worker_log: Path, event: str, **payload: Any) -> None:
    _append_jsonl(worker_log, {"t": _utc_now_text(), "event": event, **payload})


def _done_path(paths: dict[str, Path], content_hash: str) -> Path:
    return paths["done"] / f"{content_hash}.done"


def _claim_path(paths: dict[str, Path], content_hash: str) -> Path:
    return paths["claims"] / f"{content_hash}.claim"


def _enriched_path(paths: dict[str, Path], content_hash: str) -> Path:
    return paths["enriched"] / f"{content_hash}.jsonl"


def _unresolved_cluster_path(paths: dict[str, Path], content_hash: str) -> Path:
    return paths["unresolved"] / f"{content_hash}.jsonl"


def _claim_is_stale(claim_path: Path, ttl_seconds: int) -> bool:
    if not claim_path.exists():
        return False
    try:
        payload = _read_json(claim_path)
    except Exception:
        return True
    claimed_at = _parse_time(str(payload.get("claimed_at") or ""))
    if claimed_at is None:
        return True
    age = (_utc_now() - claimed_at).total_seconds()
    return age >= int(ttl_seconds)


def _claim_cluster(paths: dict[str, Path], content_hash: str, worker_id: str, *, ttl_seconds: int) -> bool:
    done_path = _done_path(paths, content_hash)
    claim_path = _claim_path(paths, content_hash)
    if done_path.exists():
        return False
    payload = {"worker_id": worker_id, "claimed_at": _utc_now_text(), "pid": os.getpid()}
    try:
        fd = os.open(str(claim_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if done_path.exists() or not _claim_is_stale(claim_path, ttl_seconds):
            return False
        stale_path = claim_path.with_name(f"{claim_path.name}.stale.{worker_id}.{int(time.time())}")
        try:
            os.rename(claim_path, stale_path)
        except FileNotFoundError:
            return False
        except OSError:
            return False
        try:
            fd = os.open(str(claim_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _release_claim(paths: dict[str, Path], content_hash: str) -> None:
    claim_path = _claim_path(paths, content_hash)
    try:
        claim_path.unlink()
    except FileNotFoundError:
        pass


def _cluster_order(cluster_dir: Path, worker_id: str) -> list[Path]:
    cluster_paths = sorted(cluster_dir.glob("*.cluster.json"))
    if not cluster_paths:
        return []
    offset = int(hashlib.sha256(worker_id.encode("utf-8")).hexdigest(), 16) % len(cluster_paths)
    return cluster_paths[offset:] + cluster_paths[:offset]


def _cluster_payload(path: Path) -> dict[str, Any]:
    return _read_json(path)


def _done_payload(content_hash: str, enriched_rows: list[dict[str, Any]], unresolved_rows: list[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    for row in enriched_rows:
        digest.update((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    return {
        "content_hash": content_hash,
        "finished_at": _utc_now_text(),
        "resolved_count": len(enriched_rows),
        "unresolved_count": len(unresolved_rows),
        "enriched_sha256": digest.hexdigest(),
    }


def _row_sync_result(
    *,
    row_id: str,
    row: dict[str, Any],
    peer_rows: list[dict[str, Any]],
    merged_in: Path,
    search_client: CountingWebSearch,
    model: str,
    num_ctx: int,
    max_web_results: int,
    timeout: float,
) -> dict[str, Any]:
    query = _differentiation_query(row)
    evidence = list(search_client(query, int(max_web_results)))
    request = ProceduralizerRequest(
        source_kind="d3_duplicate_cluster",
        source_id=row_id,
        source_path=str(merged_in),
        domain_hint=_row_domain_hint(row),
        content=_differentiation_content(row),
        context_chunks=[
            f"cluster_size={1 + len(peer_rows)}",
            f"anchor={_row_anchor(row)}",
        ],
        existing_ref_menu=build_rag_context(_row_domain_hint(row), "", _row_anchor(row)),
        quality_profile="long_context_engineering",
        ingest_mode="augment",
        mode="differentiation",
        peer_content_sample=[_peer_sample_text(peer_row) for peer_row in peer_rows[:3]],
        web_evidence=evidence,
    )
    options = dict(MODEL_OPTIONS.get(model, {}))
    options.setdefault("temperature", 0.1)
    options.setdefault("num_predict", 3072)
    options["num_ctx"] = int(num_ctx)
    bridge = ProceduralizerWineBridge(
        provider="ollama",
        default_timeout=float(timeout),
        ollama=OllamaManager(default_timeout=float(timeout)),
    )
    receipt = bridge.submit(
        request,
        model_profile="long_context_engineering",
        model=model,
        timeout=float(timeout),
        options=options,
    )
    return {
        "row_id": row_id,
        "query": query,
        "request": request,
        "receipt": receipt,
        "row": row,
        "web_evidence": evidence,
    }


async def _process_cluster_async(
    *,
    content_hash: str,
    row_ids: list[str],
    merged_in: Path,
    row_index: dict[str, dict[str, Any]],
    search_client: CountingWebSearch,
    worker_log: Path,
    model: str,
    num_ctx: int,
    max_web_results: int,
    row_concurrency: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    semaphore = asyncio.Semaphore(max(1, int(row_concurrency)))

    async def run_row(row_id: str) -> dict[str, Any]:
        async with semaphore:
            row = row_index[row_id]
            peer_rows = [row_index[peer_id] for peer_id in row_ids if peer_id != row_id and peer_id in row_index]
            return await asyncio.to_thread(
                _row_sync_result,
                row_id=row_id,
                row=row,
                peer_rows=peer_rows,
                merged_in=merged_in,
                search_client=search_client,
                model=model,
                num_ctx=num_ctx,
                max_web_results=max_web_results,
                timeout=timeout,
            )

    results = await asyncio.gather(*(run_row(row_id) for row_id in row_ids if row_id in row_index))
    enriched_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    rate_limited = False
    for result in results:
        receipt = result["receipt"]
        row_id = str(result["row_id"])
        bundle = getattr(receipt, "parsed_bundle", None)
        status = str(getattr(bundle, "status", "") or "resolved").strip().lower()
        if str(getattr(receipt, "failure_code", "") or "").strip() == "plan_limit_consumed":
            rate_limited = True
        if not _resolved_bundle(receipt):
            unresolved_rows.append(
                {
                    "content_hash": content_hash,
                    "query": str(result["query"]),
                    "reason": str(getattr(receipt, "failure_code", "") or status or "unresolvable"),
                    "row_id": row_id,
                }
            )
            _worker_log(worker_log, "bundle", cluster=content_hash, row=row_id, status="unresolvable")
            continue
        patched = _patch_row_from_receipt(result["row"], request=result["request"], receipt=receipt)
        if _content_hash(result["row"], _collect_procedural_payload(result["row"])) == _content_hash(
            patched,
            _collect_procedural_payload(patched),
        ):
            unresolved_rows.append(
                {
                    "content_hash": content_hash,
                    "query": str(result["query"]),
                    "reason": "hash_unchanged",
                    "row_id": row_id,
                }
            )
            _worker_log(worker_log, "bundle", cluster=content_hash, row=row_id, status="unresolvable")
            continue
        enriched_rows.append(patched)
        _worker_log(worker_log, "bundle", cluster=content_hash, row=row_id, status="resolved")
    return enriched_rows, unresolved_rows, rate_limited


def _write_cluster_outputs(paths: dict[str, Path], content_hash: str, enriched_rows: list[dict[str, Any]], unresolved_rows: list[dict[str, Any]]) -> None:
    _write_jsonl_atomic(_enriched_path(paths, content_hash), enriched_rows)
    _write_jsonl_atomic(_unresolved_cluster_path(paths, content_hash), unresolved_rows)
    _write_json_atomic(_done_path(paths, content_hash), _done_payload(content_hash, enriched_rows, unresolved_rows))


def _worker_id(value: str | None) -> str:
    if value:
        return str(value)
    return f"{socket.gethostname()}-{os.getpid()}-{int(time.time())}"


def _live_claims(paths: dict[str, Path], ttl_seconds: int) -> int:
    count = 0
    for claim_path in paths["claims"].glob("*.claim"):
        if not _claim_is_stale(claim_path, ttl_seconds):
            count += 1
    return count


def _worker(args: argparse.Namespace) -> dict[str, Any]:
    paths = _ensure_stage_dirs(Path(args.out_root))
    cluster_dir = Path(args.cluster_dir)
    cluster_paths = _cluster_order(cluster_dir, _worker_id(args.worker_id))
    merged_in = Path(args.merged_in)
    row_index = load_star_rows_index(merged_in)
    worker_id = _worker_id(args.worker_id)
    worker_log = paths["workers"] / f"{worker_id}.log"
    stop_after = float(args.stop_after)
    deadline = time.monotonic() + stop_after if stop_after > 0 else None
    search_client = CountingWebSearch(cache_dir=Path(args.cache_dir))
    claimed = 0
    resolved_clusters = 0
    rate_limit_backoff = 15.0

    while True:
        if deadline is not None and time.monotonic() >= deadline:
            break
        progress = False
        for cluster_path in cluster_paths:
            if deadline is not None and time.monotonic() >= deadline:
                break
            cluster = _cluster_payload(cluster_path)
            content_hash = str(cluster.get("content_hash") or "").strip()
            row_ids = [str(item).strip() for item in list(cluster.get("row_ids") or []) if str(item).strip()]
            if not content_hash or not row_ids:
                continue
            if _done_path(paths, content_hash).exists():
                continue
            if not _claim_cluster(paths, content_hash, worker_id, ttl_seconds=int(args.claim_ttl)):
                continue
            claimed += 1
            progress = True
            _worker_log(worker_log, "claim", cluster=content_hash, size=len(row_ids))
            try:
                if _done_path(paths, content_hash).exists():
                    continue
                cache_before_hits = search_client.cache_hits
                cache_before_issued = search_client.issued
                enriched_rows, unresolved_rows, rate_limited = asyncio.run(
                    _process_cluster_async(
                        content_hash=content_hash,
                        row_ids=row_ids,
                        merged_in=merged_in,
                        row_index=row_index,
                        search_client=search_client,
                        worker_log=worker_log,
                        model=str(args.model).strip(),
                        num_ctx=int(args.num_ctx),
                        max_web_results=int(args.max_web_results),
                        row_concurrency=int(args.row_concurrency),
                        timeout=float(args.timeout),
                    )
                )
                _worker_log(
                    worker_log,
                    "web_cache",
                    cluster=content_hash,
                    hits=search_client.cache_hits - cache_before_hits,
                    misses=(search_client.issued - cache_before_issued) - (search_client.cache_hits - cache_before_hits),
                )
                _write_cluster_outputs(paths, content_hash, enriched_rows, unresolved_rows)
                if enriched_rows:
                    resolved_clusters += 1
                _worker_log(worker_log, "done", cluster=content_hash, resolved=len(enriched_rows), unresolved=len(unresolved_rows))
                if rate_limited:
                    _worker_log(worker_log, "rate_limit", cluster=content_hash, backoff_seconds=rate_limit_backoff)
                    time.sleep(rate_limit_backoff + random.uniform(0.0, 3.0))
                    rate_limit_backoff = min(rate_limit_backoff * 2.0, 300.0)
                else:
                    rate_limit_backoff = 15.0
            except Exception as exc:
                _worker_log(worker_log, "error", cluster=content_hash, exc=repr(exc))
                raise
            finally:
                _release_claim(paths, content_hash)
        if not progress:
            if len(list(cluster_dir.glob("*.cluster.json"))) == len(list(paths["done"].glob("*.done"))):
                break
            if _live_claims(paths, int(args.claim_ttl)) == 0:
                break
            time.sleep(10.0)
    summary = {
        "worker_id": worker_id,
        "claimed_clusters": claimed,
        "resolved_clusters": resolved_clusters,
        "web_searches_issued": search_client.issued,
        "web_cache_hits": search_client.cache_hits,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def _merge(args: argparse.Namespace) -> dict[str, Any]:
    enriched_dir = Path(args.enriched_dir)
    unresolved_dir = Path(args.unresolved_dir)
    done_dir = Path(args.done_dir)
    merged_in = Path(args.merged_in)
    merged_out = Path(args.merged_out)
    merged_by_galaxy_dir = merged_out.parent / "merged_by_galaxy"
    row_to_file_name = _row_file_map(merged_by_galaxy_dir)
    row_updates: dict[str, dict[str, Any]] = {}
    for path in sorted(enriched_dir.glob("*.jsonl")):
        for row in _read_jsonl(path):
            row_id = _row_id(row)
            if row_id:
                row_updates[row_id] = row
    done_files = sorted(done_dir.glob("*.done"))
    unresolved_rows: list[dict[str, Any]] = []
    for done_file in done_files:
        content_hash = done_file.stem
        enriched_path = enriched_dir / f"{content_hash}.jsonl"
        unresolved_path = unresolved_dir / f"{content_hash}.jsonl"
        if not enriched_path.exists() and not unresolved_path.exists():
            raise RuntimeError(f"done marker without outputs: {done_file}")
        if unresolved_path.exists():
            unresolved_rows.extend(_read_jsonl(unresolved_path))
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    with merged_in.open("r", encoding="utf-8") as source, merged_out.open("w", encoding="utf-8") as target:
        for line in source:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            row_id = _row_id(row)
            payload = row_updates.get(row_id, row)
            target.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    _write_jsonl_atomic(Path(args.consolidated_unresolved_out), sorted(unresolved_rows, key=lambda row: (row["content_hash"], row["row_id"], row["reason"])))
    digest = hashlib.sha256()
    with merged_out.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    shard_stats = _rewrite_merged_by_galaxy(
        merged_rows_path=merged_out,
        merged_by_galaxy_dir=merged_by_galaxy_dir,
        row_to_file_name=row_to_file_name,
    )
    summary = {
        "done_clusters": len(done_files),
        "enriched_rows": len(row_updates),
        "merged_out": str(merged_out),
        "merged_sha256": digest.hexdigest(),
        "merged_by_galaxy_row_count": shard_stats["row_count"],
        "merged_by_galaxy_shard_count": shard_stats["shard_count"],
        "unresolved_rows": len(unresolved_rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def _status(args: argparse.Namespace) -> dict[str, Any]:
    paths = _ensure_stage_dirs(Path(args.out_root))
    cluster_total = len(list(paths["clusters"].glob("*.cluster.json")))
    done_files = sorted(paths["done"].glob("*.done"))
    done_count = len(done_files)
    claims_live = _live_claims(paths, int(args.claim_ttl))
    rows_resolved = 0
    rows_unresolved = 0
    finished_times: list[datetime] = []
    for path in done_files:
        payload = _read_json(path)
        rows_resolved += int(payload.get("resolved_count", 0) or 0)
        rows_unresolved += int(payload.get("unresolved_count", 0) or 0)
        finished_at = _parse_time(str(payload.get("finished_at") or ""))
        if finished_at is not None:
            finished_times.append(finished_at)
    eta_seconds: int | None = None
    if len(finished_times) >= 2:
        duration = (max(finished_times) - min(finished_times)).total_seconds()
        if duration > 0 and done_count > 0 and cluster_total > done_count:
            throughput = done_count / duration
            if throughput > 0:
                eta_seconds = int((cluster_total - done_count) / throughput)
    summary = {
        "clusters_total": cluster_total,
        "done": done_count,
        "claims_live": claims_live,
        "unclaimed": max(cluster_total - done_count - claims_live, 0),
        "rows_resolved": rows_resolved,
        "rows_unresolved": rows_unresolved,
        "eta_seconds": eta_seconds,
    }
    print(f"clusters_total={summary['clusters_total']} done={summary['done']} claims_live={summary['claims_live']} unclaimed={summary['unclaimed']}")
    print(f"rows_resolved={summary['rows_resolved']} rows_unresolved={summary['rows_unresolved']}")
    print(f"eta_seconds={summary['eta_seconds'] if summary['eta_seconds'] is not None else 'unknown'}")
    return summary


def _run_direct(args: argparse.Namespace) -> dict[str, Any]:
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
        # Preserve the preview path for the dry-run/reporting mode.
        cluster_results = []
        for row_id in row_ids:
            from knowledge3d.tools.knowledge_proceduralizer import differentiate_cluster_receipts

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
            break
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
            patched = _patch_row_from_receipt(original_row, request=result["request"], receipt=result["receipt"])
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
        _write_jsonl_atomic(Path(args.merged_out), merged_rows)
    _write_jsonl_atomic(Path(args.unresolved_out), sorted(unresolved_rows, key=lambda row: (row["content_hash"], row["row_id"], row["reason"])))

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
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--violations", type=Path, default=DEFAULT_VIOLATIONS)
    parser.add_argument("--merged-in", type=Path, default=DEFAULT_MERGED_IN)
    parser.add_argument("--merged-out", type=Path, required=False)
    parser.add_argument("--unresolved-out", type=Path, default=DEFAULT_UNRESOLVED)
    parser.add_argument("--consolidated-unresolved-out", type=Path, default=DEFAULT_UNRESOLVED)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--min-cluster", type=int, default=2)
    parser.add_argument("--max-cluster", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default=PROCEDURALIZER_MODEL_PROFILES["long_context_engineering"])
    parser.add_argument("--num-ctx", type=int, default=65536)
    parser.add_argument("--max-peer-samples", type=int, default=3)
    parser.add_argument("--max-web-results", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_STAGE_ROOT / "clusters")
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_STAGE_ROOT / "clusters")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--enriched-dir", type=Path, default=DEFAULT_STAGE_ROOT / "enriched")
    parser.add_argument("--unresolved-dir", type=Path, default=DEFAULT_STAGE_ROOT / "unresolved")
    parser.add_argument("--done-dir", type=Path, default=DEFAULT_STAGE_ROOT / "done")
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--row-concurrency", type=int, default=4)
    parser.add_argument("--stop-after", type=float, default=0.0)
    parser.add_argument("--claim-ttl", type=int, default=CLAIM_TTL_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.build_manifest:
            _build_cluster_manifest(args)
            return 0
        if args.worker:
            _worker(args)
            return 0
        if args.merge:
            if args.merged_out is None:
                parser.error("--merged-out is required with --merge")
            _merge(args)
            return 0
        if args.status:
            _status(args)
            return 0
        if args.merged_out is None:
            parser.error("--merged-out is required")
        _run_direct(args)
        return 0
    except WebSearchUnavailable as exc:
        print(json.dumps({"status": "web_search_unavailable", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
