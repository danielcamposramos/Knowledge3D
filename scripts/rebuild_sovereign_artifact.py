#!/usr/bin/env python3
"""Canonical maintenance rebuild for the sovereign runtime artifact bundle."""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.local_paths import resolve_storage_root
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def _runtime_payload(kv: Knowledgeverse, rebuild: dict[str, Any], *, storage_root: Path) -> dict[str, Any]:
    house_summary = kv.house_state_summary()
    manifest = kv._sovereign_runtime_manifest()
    catalog_summary = kv.gpu_catalog_build_summary()
    route_audit_path = storage_root / "checkpoints" / "meaning_family_route_audit.json"
    closure_audit_path = storage_root / "checkpoints" / "meaning_route_closure_audit.json"
    route_audit = {}
    closure_audit = {}
    if route_audit_path.exists():
        try:
            route_audit = json.loads(route_audit_path.read_text(encoding="utf-8"))
        except Exception:
            route_audit = {}
    if closure_audit_path.exists():
        try:
            closure_audit = json.loads(closure_audit_path.read_text(encoding="utf-8"))
        except Exception:
            closure_audit = {}
    return {
        "storage_root": str(storage_root.resolve()),
        "mode": str(rebuild.get("mode") or ""),
        "star_count": int(rebuild.get("star_count") or manifest.get("star_count") or 0),
        "default_knowledge_signature": str(
            manifest.get("default_knowledge_signature") or house_summary.get("default_knowledge_signature") or ""
        ),
        "house_signature_base": str(
            manifest.get("house_signature_base") or house_summary.get("gpu_buffer_signature_base") or ""
        ),
        "house_state": {
            "total_persisted_entries": int(house_summary.get("total_persisted_entries") or 0),
            "default_knowledge_entries": int(house_summary.get("default_knowledge_entries") or 0),
            "default_knowledge_signature": str(house_summary.get("default_knowledge_signature") or ""),
            "gpu_buffer_signature_base": str(house_summary.get("gpu_buffer_signature_base") or ""),
        },
        "catalog_summary": dict(catalog_summary or {}),
        "rebuild": dict(rebuild or {}),
        "manifest": dict(manifest or {}),
        "meaning_family_route_audit": route_audit,
        "meaning_family_route_audit_path": str(route_audit_path),
        "meaning_route_closure_audit": closure_audit,
        "meaning_route_closure_audit_path": str(closure_audit_path),
    }


def _print_stage_timings(payload: dict[str, Any], *, verbose: bool) -> None:
    rebuild = dict(payload.get("rebuild") or {})
    catalog_summary = dict(payload.get("catalog_summary") or {})
    feed_source_compile = dict(payload.get("feed_source_compile") or {})
    feed_compile = dict(payload.get("feed_compile") or {})
    print("[artifact] stage timings", flush=True)
    print(
        f"  default galaxy load: {float(rebuild.get('default_galaxy_load_s', 0.0)):.3f}s",
        flush=True,
    )
    if feed_compile:
        print(
            "  build feed compile: "
            f"{float(feed_compile.get('elapsed_s', 0.0)):.3f}s "
            f"(stars={int(feed_compile.get('star_count') or 0)}, "
            f"forward_refs={int(feed_compile.get('forward_ref_count') or 0)})",
            flush=True,
        )
    if feed_source_compile:
        print(
            "  feed source compile: "
            f"{float(feed_source_compile.get('elapsed_s', 0.0)):.3f}s "
            f"(stars={int(feed_source_compile.get('star_count') or 0)}, "
            f"forward_refs={int(feed_source_compile.get('forward_ref_count') or 0)})",
            flush=True,
        )
    if catalog_summary:
        print(
            "  flat cache: "
            f"{str(catalog_summary.get('cache_mode') or 'unknown')} "
            f"(signature={str(catalog_summary.get('signature') or '')}, "
            f"entries={int(catalog_summary.get('catalog_entries') or 0)})",
            flush=True,
        )
        if "cache_save_s" in catalog_summary:
            print(f"  cache write: {float(catalog_summary.get('cache_save_s', 0.0)):.3f}s", flush=True)
        if verbose and catalog_summary.get("cache_write"):
            cache_write = dict(catalog_summary.get("cache_write") or {})
            print(
                "  cache prune: "
                f"removed={int(cache_write.get('removed_count') or 0)}",
                flush=True,
            )
    if "catalog_build_s" in rebuild:
        print(f"  catalog build: {float(rebuild.get('catalog_build_s', 0.0)):.3f}s", flush=True)
    if "build_feed_load_s" in rebuild:
        print(f"  load build feed: {float(rebuild.get('build_feed_load_s', 0.0)):.3f}s", flush=True)
    if "load_build_feed_s" in rebuild:
        print(f"  stream build rows: {float(rebuild.get('load_build_feed_s', 0.0)):.3f}s", flush=True)
    if "load_feed_source_s" in feed_compile:
        print(f"  load feed source: {float(feed_compile.get('load_feed_source_s', 0.0)):.3f}s", flush=True)
    if "compile_build_rows_s" in feed_compile:
        print(f"  compile build rows: {float(feed_compile.get('compile_build_rows_s', 0.0)):.3f}s", flush=True)
    if "decode_feed_source_s" in feed_compile:
        print(f"  decode feed source: {float(feed_compile.get('decode_feed_source_s', 0.0)):.3f}s", flush=True)
    if "expand_reverse_ref_hashes_s" in feed_compile:
        print(
            f"  expand reverse ref hashes: {float(feed_compile.get('expand_reverse_ref_hashes_s', 0.0)):.3f}s",
            flush=True,
        )
    if "decode_build_rows_s" in rebuild:
        print(f"  decode build rows: {float(rebuild.get('decode_build_rows_s', 0.0)):.3f}s", flush=True)
    if "boot_finalize_s" in rebuild:
        print(f"  boot finalize: {float(rebuild.get('boot_finalize_s', 0.0)):.3f}s", flush=True)
    if "star_build_s" in rebuild:
        print(f"  star build: {float(rebuild.get('star_build_s', 0.0)):.3f}s", flush=True)
    if "star_materialize_s" in rebuild:
        print(f"  star materialize: {float(rebuild.get('star_materialize_s', 0.0)):.3f}s", flush=True)
    if "build_star_hash_index_s" in rebuild:
        print(f"  build star hash index: {float(rebuild.get('build_star_hash_index_s', 0.0)):.3f}s", flush=True)
    if "resolve_ref_hashes_s" in rebuild:
        print(f"  resolve ref hashes: {float(rebuild.get('resolve_ref_hashes_s', 0.0)):.3f}s", flush=True)
    if "expand_reverse_symlinks_s" in rebuild:
        print(f"  expand reverse symlinks: {float(rebuild.get('expand_reverse_symlinks_s', 0.0)):.3f}s", flush=True)
    if "ref_csr_build_s" in rebuild:
        print(f"  csr scan/scatter: {float(rebuild.get('ref_csr_build_s', 0.0)):.3f}s", flush=True)
    if "star_table_upload_s" in rebuild:
        print(f"  star-table upload: {float(rebuild.get('star_table_upload_s', 0.0)):.3f}s", flush=True)
    if "artifact_save_s" in rebuild:
        print(f"  artifact save: {float(rebuild.get('artifact_save_s', 0.0)):.3f}s", flush=True)
    if rebuild.get("build_backend"):
        print(f"  build backend: {str(rebuild.get('build_backend'))}", flush=True)
    if rebuild.get("boot_finalize_ptx_signature"):
        print(f"  boot finalize ptx: {str(rebuild.get('boot_finalize_ptx_signature'))}", flush=True)
    if rebuild.get("materializer_ptx_signature"):
        print(f"  materializer ptx: {str(rebuild.get('materializer_ptx_signature'))}", flush=True)
    if rebuild.get("csr_builder_ptx_signature"):
        print(f"  csr builder ptx: {str(rebuild.get('csr_builder_ptx_signature'))}", flush=True)
    route_audit = dict(payload.get("meaning_family_route_audit") or {})
    if route_audit:
        print(
            "  meaning-family route audit: "
            f"passed={bool(route_audit.get('passed'))} "
            f"path={str(payload.get('meaning_family_route_audit_path') or '')}",
            flush=True,
        )
    closure_audit = dict(payload.get("meaning_route_closure_audit") or {})
    if closure_audit:
        print(
            "  meaning-route closure audit: "
            f"passed={bool(closure_audit.get('passed'))} "
            f"path={str(payload.get('meaning_route_closure_audit_path') or '')}",
            flush=True,
        )
    print(f"  total elapsed: {float(rebuild.get('total_elapsed_s', 0.0)):.3f}s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-root",
        default=None,
        help="Knowledge3D.local root containing the canonical checkpoint and sovereign artifact.",
    )
    parser.add_argument(
        "--force-default-knowledge",
        action="store_true",
        help="Force re-materialization of the default always-loaded knowledge before rebuild.",
    )
    parser.add_argument(
        "--skip-save-consolidated",
        action="store_true",
        help="Do not write a refreshed consolidated checkpoint after the sovereign artifact rebuild.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the final JSON payload in addition to the stage summary.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Bypass artifact restore and force a full device rebuild of the sovereign runtime bundle.",
    )
    parser.add_argument(
        "--refresh-feed-source",
        action="store_true",
        help="Compile the authoritative sovereign feed-source cache before the build-feed refresh.",
    )
    parser.add_argument(
        "--feed-source-workers",
        type=int,
        default=None,
        help="Override the process count for feed-source extraction workers.",
    )
    parser.add_argument(
        "--feed-source-chunk-size",
        type=int,
        default=None,
        help="Override the catalog chunk size used during feed-source extraction.",
    )
    parser.add_argument(
        "--refresh-build-feed",
        action="store_true",
        help="Compile the authoritative sovereign build feed before the rebuild.",
    )
    args = parser.parse_args()

    os.environ.setdefault("K3D_IDLE_SLEEP_SECONDS", "31536000")
    os.environ.setdefault("K3D_IDLE_SLEEP_POLL_SECONDS", "31536000")
    os.environ.setdefault("K3D_SOVEREIGN_BUILD_PROGRESS", "1")
    faulthandler.enable()

    storage_root = resolve_storage_root(args.storage_root)
    constructor_t0 = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        kv = Knowledgeverse(
            storage_root=storage_root,
            eager_load_default_galaxies=False,
            start_live_loops=False,
        )
    constructor_s = float(time.perf_counter() - constructor_t0)
    print(f"[artifact] knowledgeverse booted in {constructor_s:.3f}s", flush=True)

    try:
        print("[artifact] loading all default stars into the resident Knowledgeverse", flush=True)
        default_load_t0 = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            galaxy_counts = kv.ensure_default_galaxies_loaded(force=bool(args.force_default_knowledge))
        default_load_s = float(time.perf_counter() - default_load_t0)
        print(
            "[artifact] default galaxy load complete "
            f"({default_load_s:.3f}s, galaxies={len(galaxy_counts)})",
            flush=True,
        )

        runtime = kv._get_sovereign_hot_path()
        runtime.invalidate_loaded_state()

        feed_source_compile = {}
        feed_compile = {}
        if args.refresh_feed_source:
            print("[artifact] compiling sovereign feed source", flush=True)
            feed_source_t0 = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                feed_source_compile = runtime.refresh_feed_source(
                    worker_count=args.feed_source_workers,
                    chunk_size=args.feed_source_chunk_size,
                    force=True,
                )
            feed_source_compile = dict(feed_source_compile or {})
            feed_source_compile.setdefault("elapsed_s", float(time.perf_counter() - feed_source_t0))
            print(
                "[artifact] feed source ready "
                f"mode={str(feed_source_compile.get('mode') or 'feed_source_compile')} "
                f"signature={str(feed_source_compile.get('feed_source_signature') or '')} "
                f"stars={int(feed_source_compile.get('star_count') or 0)} "
                f"elapsed={float(feed_source_compile.get('elapsed_s', 0.0)):.3f}s",
                flush=True,
            )
        if args.refresh_build_feed:
            print("[artifact] compiling sovereign build feed", flush=True)
            feed_t0 = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                feed_compile = runtime.refresh_build_feed()
            feed_compile = dict(feed_compile or {})
            feed_compile.setdefault("elapsed_s", float(time.perf_counter() - feed_t0))
            print(
                "[artifact] build feed ready "
                f"signature={str(feed_compile.get('build_feed_signature') or '')} "
                f"stars={int(feed_compile.get('star_count') or 0)} "
                f"elapsed={float(feed_compile.get('elapsed_s', 0.0)):.3f}s",
                flush=True,
            )

        print("[artifact] rebuilding sovereign runtime artifact", flush=True)
        rebuild_t0 = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            rebuild = kv._boot_sovereign_runtime(
                force_reload=True,
                force_rebuild=bool(args.force_rebuild),
            )
        rebuild_elapsed_s = float(time.perf_counter() - rebuild_t0)
        rebuild = dict(rebuild or {})
        rebuild.setdefault("total_elapsed_s", rebuild_elapsed_s)

        consolidated = {}
        if not args.skip_save_consolidated:
            print("[artifact] saving consolidated checkpoint", flush=True)
            with contextlib.redirect_stdout(sys.stderr):
                consolidated = kv.save_consolidated_state()

        payload = _runtime_payload(kv, rebuild, storage_root=storage_root)
        payload["galaxy_counts"] = {str(name): int(count) for name, count in sorted(galaxy_counts.items())}
        payload["constructor_s"] = constructor_s
        payload["default_galaxy_load_s"] = default_load_s
        payload["consolidated"] = dict(consolidated or {})
        payload["feed_source_compile"] = dict(feed_source_compile or {})
        payload["feed_compile"] = dict(feed_compile or {})

        _print_stage_timings(payload, verbose=bool(args.verbose))
        print(
            "[artifact] result "
            f"mode={payload['mode']} "
            f"star_count={int(payload['star_count'])} "
            f"default_knowledge_signature={payload['default_knowledge_signature']} "
            f"house_signature_base={payload['house_signature_base']}",
            flush=True,
        )
        if args.verbose:
            print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        try:
            runtime = getattr(kv, "_sovereign_hot_path", None)
            if runtime is not None:
                runtime.close()
                kv._sovereign_hot_path = None
        except Exception:
            pass
        try:
            kv._trm_game_loop.stop()
        except Exception:
            pass
        try:
            kv._stop_idle_monitor()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
