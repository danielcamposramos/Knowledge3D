"""Persistent K3D daemon entrypoint (game-style runtime).

The daemon keeps one Knowledgeverse + TRM instance alive and serves JSON
commands over stdio or TCP line protocol. This avoids one-shot script
orchestration and enforces a single-world process lifecycle.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import socketserver
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC, TabletSessionTape
from knowledge3d.daemon.tick_driver import TickDriver
from knowledge3d.local_paths import default_storage_root
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
try:
    from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
except Exception:  # pragma: no cover
    ModularRPNEngine = None  # type: ignore[assignment]

try:
    from knowledge3d.cranium.sovereign.loader import get_vram_usage
except Exception:  # pragma: no cover
    get_vram_usage = None  # type: ignore[assignment]

try:
    from knowledge3d.gpu.perf_counters import gpu_utilisation
except Exception:  # pragma: no cover
    gpu_utilisation = None  # type: ignore[assignment]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_env_path(var_name: str, path_value: str) -> None:
    current = os.environ.get(var_name, "").strip()
    if not current:
        os.environ[var_name] = path_value
        return
    items = [item for item in current.split(":") if item]
    if path_value in items:
        return
    os.environ[var_name] = f"{current}:{path_value}"


def _configure_cuda_include_paths() -> dict[str, Any]:
    """
    Ensure NVRTC can resolve core CUDA headers (e.g., cuda_fp16.h).

    This is a daemon-level safeguard for sovereign PTX query/runtime paths:
    we do not enable fallbacks; we only make required CUDA include paths explicit.
    """
    include_candidates: list[Path] = [
        Path("/usr/local/cuda/include"),
        Path("/usr/include"),
    ]
    include_candidates.extend(sorted(Path("/usr/local").glob("cuda*/include")))

    selected: Path | None = None
    for inc in include_candidates:
        if not inc.exists():
            continue
        if (inc / "cuda_fp16.h").exists():
            selected = inc
            break

    configured = {"applied": False, "include_path": None, "cuda_path": None}
    if selected is None:
        return configured

    include_str = str(selected)
    _append_env_path("CPATH", include_str)
    _append_env_path("CPLUS_INCLUDE_PATH", include_str)

    # Derive CUDA_PATH from include parent when possible.
    cuda_root = selected.parent if selected.name == "include" else selected
    if cuda_root.exists() and not os.environ.get("CUDA_PATH"):
        os.environ["CUDA_PATH"] = str(cuda_root)

    configured["applied"] = True
    configured["include_path"] = include_str
    configured["cuda_path"] = os.environ.get("CUDA_PATH")
    return configured


def _mean_embedding_rows(rows: list[list[float]]) -> list[float]:
    if not rows:
        return []
    width = len(rows[0])
    if width <= 0:
        return []
    totals = [0.0] * width
    count = 0
    for row in rows:
        if len(row) != width:
            continue
        for index, value in enumerate(row):
            totals[index] += float(value)
        count += 1
    if count <= 0:
        return []
    return [value / float(count) for value in totals]


@dataclass
class DaemonConfig:
    storage_root: Path
    require_ptx_query: bool = True
    eager_load_default_galaxies: bool = True
    host: str = "127.0.0.1"
    port: int = 7777
    idle_threshold_seconds: float = 30.0
    tcp_poll_seconds: float = 0.2
    sleep_sample_size: int = 512
    warm_gpu_runtime_on_boot: bool = False


class K3DDaemon:
    """Long-lived command server for K3D runtime orchestration."""

    def __init__(
        self,
        config: DaemonConfig,
        *,
        knowledgeverse: Knowledgeverse | None = None,
    ):
        self.config = config
        self.started_at = _now_iso()
        self._shutdown_requested = False
        self._command_count = 0
        self._gpu_calls_total = 0
        self._cuda_env = _configure_cuda_include_paths()
        self._repo_root = Path(__file__).resolve().parents[2]
        self._boot_status_paths = [
            config.storage_root / "runtime" / "runtime_boot.json",
            self._repo_root / "viewer" / "public" / "runtime_boot.json",
        ]
        self._sleep_cluster_refiner = None
        self._sleep_glyph_consolidator = None
        self._boot_binding: dict[str, Any] = {}
        self._sleep_tick_count = 0
        self._sleep_tick_cursor = 0
        self._last_sleep_tick: dict[str, Any] = {}
        self._sleep_tick_history: list[dict[str, Any]] = []
        self._sleep_tick_history_max = 16
        self._pending_sleep_embedding_updates = 0
        self._idle_elapsed_seconds = 0.0
        # Lane B (Gap 2) in-memory state — BitNet-packed tile bytes + tile_format
        # sidecar mirror.  Loaded from disk on first tick, updated in place after
        # each kernel launch.  convert-on-touch: per feedback_live_inline_weight_conversion.md,
        # there is NO boot-time batch conversion; first tick handles it.
        #
        # _lane_b_n_valid (2026-04-21, tail-pad information-loss fix):
        #   Authoritative count of REAL (non-padding) f32 weights in the canonical
        #   concatenated buffer.  Persisted to ``<checkpoint>.n_valid`` sidecar so
        #   any later readback / unpack consumes only the first n_valid_weights
        #   trits and discards the Gap 2 tile-boundary zero padding.  Daniel's
        #   ruling: "Require exact fit, but do not refuse, recalculate and accept —
        #   we must not lose information."  Mis-aligned checkpoints are ACCEPTED:
        #   we record the real count and zero-pad the final tile for GPU layout.
        #   Genuine corruption (stored count ≠ derived) fails loud.
        self._lane_b_bitnet_tiles: bytes | None = None
        self._lane_b_tile_format: bytes | None = None
        self._lane_b_n_tiles: int = 0
        self._lane_b_n_valid: int = 0
        self._lane_b_sidecars_loaded: bool = False
        self._sleep_tick_order: tuple[str, ...] = (
            "cluster_refiner",
            "glyph_consolidator",
            "memory_updater",
            "graph_crystallizer",
            "lane_a_ingest",
            "lane_b_weights",
        )
        self._write_boot_status(stage="daemon_boot", progress=0.05, state="starting")

        # Sovereign invariant: ONE CUDA context, established at daemon boot (not on first query).
        # See TEMP/CLAUDE_SINGLE_CONTEXT_LIVING_AI_SPEC_04.18.2026.md §5.2.
        from knowledge3d.cranium.sovereign import loader
        loader.ensure_init()
        _vram_used, _vram_total = loader.get_vram_usage()
        if os.environ.get("K3D_RPN_DEBUG"):
            print(f"[daemon] single-context boot: vram_used={_vram_used:.1f} MB / {_vram_total:.1f} MB")

        os.environ["K3D_REQUIRE_PTX_QUERY"] = "true" if config.require_ptx_query else "false"
        os.environ.setdefault(
            "K3D_RING_TRACE_PATH",
            str(self._repo_root / "TEMP" / "validation_sweep_2026-04-17" / "ring_trace.jsonl"),
        )

        self._write_boot_status(stage="knowledgeverse_load", progress=0.2, state="loading")
        self.kv = knowledgeverse or Knowledgeverse(
            storage_root=config.storage_root,
            eager_load_default_galaxies=config.eager_load_default_galaxies,
        )
        self.trm = self.kv.trm_navigator
        self._default_counts = (
            self.kv.ensure_default_galaxies_loaded()
            if config.eager_load_default_galaxies
            else {str(name): 0 for name in getattr(self.kv, "DEFAULT_GALAXIES", ())}
        )
        self._tablet_boundary: HeadlessTabletMPC | None = None
        if self.config.warm_gpu_runtime_on_boot:
            self._write_boot_status(stage="sovereign_runtime_load", progress=0.62, state="warming")
            self._boot_binding = self._warmup_gpu_runtime_binding()
        self._write_boot_status(
            stage="knowledgeverse_ready",
            progress=0.55,
            state="loading",
            extra={
                "default_galaxy_counts": dict(self._default_counts),
                "gpu_binding": dict(self._boot_binding),
            },
        )
        self._tick_driver = TickDriver(self.kv)
        self._tick_driver.start()
        self.kv._external_tick_driver_active = True
        self._write_boot_status(
            stage="ready",
            progress=1.0,
            state="ready",
            extra={
                "gpu_binding": dict(self._boot_binding),
                "tick_driver": self._tick_driver.stats(),
            },
        )

    def _write_boot_status(
        self,
        *,
        stage: str,
        progress: float,
        state: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "status": "ok",
            "state": state,
            "stage": stage,
            "progress": max(0.0, min(1.0, float(progress))),
            "timestamp": _now_iso(),
            "pid": int(os.getpid()),
        }
        if extra:
            payload.update(extra)
        for path in self._boot_status_paths:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
            except Exception:
                continue

    def _gpu_snapshot(self) -> dict[str, Any]:
        used = 0
        total = 0
        util = 0.0
        if get_vram_usage is not None:
            try:
                used, total = get_vram_usage()
            except Exception:
                used, total = 0, 0
        if gpu_utilisation is not None:
            try:
                util = float(gpu_utilisation(default=0.0))
            except Exception:
                util = 0.0
        return {
            "vram_used_bytes": int(used),
            "vram_total_bytes": int(total),
            "gpu_utilization": float(util),
        }

    def _get_tablet_boundary(self) -> HeadlessTabletMPC:
        if self._tablet_boundary is None:
            self._tablet_boundary = HeadlessTabletMPC(
                command_handler=lambda payload: {"status": "error", "error": "daemon_tablet_route_fallback_forbidden"},
                knowledgeverse=self.kv,
                storage_root=self.config.storage_root,
            )
        return self._tablet_boundary

    def _warmup_gpu_runtime_binding(self) -> dict[str, Any]:
        try:
            runtime = self.kv._get_sovereign_hot_path()
            binding = runtime.ensure_loaded()
            return {"status": "ok", **dict(binding)}
        except Exception as exc:
            return {
                "status": "error",
                "exception_type": type(exc).__name__,
                "detail": str(exc),
            }

    def _binding_report(self) -> dict[str, Any]:
        runtime = getattr(self.kv, "_sovereign_hot_path", None)
        if runtime is not None and getattr(runtime.star_table, "star_count", 0) > 0:
            return {
                "status": "ready",
                "mode": "sovereign",
                "star_count": int(runtime.star_table.star_count),
                "manifest": dict(runtime.current_runtime_manifest()) if hasattr(runtime, "current_runtime_manifest") else {},
                "load_summary": dict(getattr(runtime, "_last_load_summary", {}) or {}),
            }
        return {"status": "unbound", "mode": "sovereign"}

    def _semantic_graph_report(self) -> dict[str, Any]:
        graph = getattr(self.kv, "_semantic_csr_graph", None)
        if graph is None:
            return {"status": "unbound"}
        node_count = 0
        edge_count = 0
        try:
            node_count = int(getattr(graph, "embeddings").shape[0])
        except Exception:
            pass
        try:
            edge_count = int(getattr(graph, "col_indices").shape[0])
        except Exception:
            pass
        return {
            "status": "ready",
            "signature": str(getattr(graph, "signature", "")),
            "node_count": node_count,
            "edge_count": edge_count,
            "knn_k": int(getattr(graph, "knn_k", 0)),
            "similarity_threshold": float(getattr(graph, "similarity_threshold", 0.0)),
        }

    def _vram_report_payload(self) -> dict[str, Any]:
        galaxy_entry_counts = {
            name: len(self.kv.galaxy_manager.get_galaxy(name).entries)
            for name in self.kv._discover_live_galaxy_names()
        }
        return {
            "status": "ok",
            "timestamp": _now_iso(),
            "gpu": self._gpu_snapshot(),
            "binding": self._binding_report(),
            "semantic_csr_graph": self._semantic_graph_report(),
            "catalog_entry_count": len(self.kv.get_gpu_galaxy_catalog()),
            "default_galaxy_counts": galaxy_entry_counts,
            "sleep": {
                "tick_count": int(self._sleep_tick_count),
                "tick_cursor": int(self._sleep_tick_cursor),
                "tick_order": list(self._sleep_tick_order),
                "last_tick": dict(self._last_sleep_tick),
                "tick_history": [dict(item) for item in self._sleep_tick_history],
                "pending_embedding_updates": int(self._pending_sleep_embedding_updates),
                "idle_elapsed_seconds": float(self._idle_elapsed_seconds),
                "idle_threshold_seconds": float(self.config.idle_threshold_seconds),
            },
        }

    def _all_default_galaxies(self) -> list[str]:
        return list(self.kv._discover_live_galaxy_names())

    def _coalesce_query(
        self,
        payload: dict[str, Any],
        task: dict[str, Any] | None,
    ) -> str:
        task_payload = dict(task or {})
        if isinstance(task_payload.get("messages"), list):
            for message in reversed(task_payload["messages"]):
                if not isinstance(message, dict):
                    continue
                if str(message.get("role", "")).strip().lower() != "user":
                    continue
                content = str(message.get("content", "")).strip()
                if content:
                    return content
        return str(
            payload.get("query", "")
            or task_payload.get("query", "")
            or task_payload.get("question", "")
            or task_payload.get("prompt", "")
        ).strip()

    def _meaning_route(
        self,
        *,
        specialist: str,
        domain_hint: str | None,
        galaxy_names: list[str],
        route_policy: str,
    ) -> dict[str, Any]:
        route: dict[str, Any] = {
            "galaxy_names": list(galaxy_names),
            "route_policy": route_policy or "all_live_galaxies",
        }
        specialist_name = str(specialist or "auto").strip().lower() or "auto"
        if specialist_name != "auto":
            route["specialist"] = specialist_name
        domain_name = str(domain_hint or "").strip()
        if domain_name:
            route["domain_hint"] = domain_name
        return route

    def _meaning_task_payload(
        self,
        *,
        task: dict[str, Any] | None,
        query: str,
        galaxies: list[str],
        route_policy: str,
    ) -> dict[str, Any]:
        task_payload = dict(task or {})
        for key in (
            "surface_kind",
            "type",
            "task_type",
            "question_mode",
            "spatial_mode",
            "math_mode",
        ):
            task_payload.pop(key, None)
        task_payload["query"] = str(query)
        if "question" in task_payload or not str(task_payload.get("prompt", "")).strip():
            task_payload["question"] = str(query)
        if "prompt" in task_payload or isinstance(task_payload.get("messages"), list):
            task_payload["prompt"] = str(query)
        if galaxies:
            task_payload["galaxies"] = list(galaxies)
        if route_policy:
            task_payload["route_policy"] = str(route_policy)
        return task_payload

    def _get_sleep_cluster_refiner(self):
        if self._sleep_cluster_refiner is False:
            return None
        if self._sleep_cluster_refiner is None:
            try:
                from knowledge3d.cranium.bridges.sovereign_bridges import SleepClusterRefiner

                self._sleep_cluster_refiner = SleepClusterRefiner()
            except Exception:
                self._sleep_cluster_refiner = False
                return None
        return self._sleep_cluster_refiner

    def _get_sleep_glyph_consolidator(self):
        if self._sleep_glyph_consolidator is False:
            return None
        if self._sleep_glyph_consolidator is None:
            try:
                from knowledge3d.cranium.bridges.sovereign_bridges import SleepGlyphConsolidator

                self._sleep_glyph_consolidator = SleepGlyphConsolidator()
            except Exception:
                self._sleep_glyph_consolidator = False
                return None
        return self._sleep_glyph_consolidator

    def _live_embedding_rows(
        self,
        *,
        galaxy_names: list[str],
        limit: int | None = None,
    ) -> list[tuple[str, dict[str, Any], list[float]]]:
        rows: list[tuple[str, dict[str, Any], list[float]]] = []
        max_rows = None if limit is None else max(0, int(limit))
        for galaxy_name in galaxy_names:
            galaxy = self.kv.galaxy_manager.get_galaxy(galaxy_name)
            for entry in getattr(galaxy, "entries", []):
                if not isinstance(entry, dict):
                    continue
                try:
                    embedding = list(self.kv._entry_embedding16(entry))
                except Exception:
                    continue
                if not embedding:
                    continue
                rows.append((galaxy_name, entry, embedding))
                if max_rows is not None and len(rows) >= max_rows:
                    return rows
        return rows

    def _apply_embedding_updates(
        self,
        rows: list[tuple[str, dict[str, Any], list[float]]],
        updated_embeddings: list[list[float]],
    ) -> int:
        updated = 0
        for (_, entry, _), embedding in zip(rows, updated_embeddings):
            normalized = [float(value) for value in embedding[:16]]
            entry["embedding16"] = list(normalized)
            metadata = entry.get("metadata")
            if isinstance(metadata, dict):
                metadata["embedding16"] = list(normalized)
            updated += 1
        self._pending_sleep_embedding_updates += updated
        return updated

    def _sleep_cluster_tick(self) -> dict[str, Any]:
        refiner = self._get_sleep_cluster_refiner()
        if refiner is None:
            return {"status": "skipped", "reason": "sleep_cluster_refiner_unavailable"}
        sample_rows = self._live_embedding_rows(
            galaxy_names=self.kv._discover_live_galaxy_names(),
            limit=min(int(self.config.sleep_sample_size), 512),
        )
        if len(sample_rows) < 2:
            return {"status": "skipped", "reason": "insufficient_embeddings", "rows": len(sample_rows)}
        matrix = [embedding for _, _, embedding in sample_rows]
        clusters = max(2, min(32, max(2, len(sample_rows) // 16)))
        result = refiner.refine_clusters(matrix, n_clusters=clusters, n_iterations=2, learning_rate=0.12)
        refined = result.get("refined_embeddings")
        updated = 0
        if refined is not None:
            try:
                updated = self._apply_embedding_updates(
                    sample_rows,
                    [[float(value) for value in row.tolist()] for row in refined],
                )
            except Exception:
                updated = 0
        return {
            "status": "ok",
            "rows": len(sample_rows),
            "clusters": int(clusters),
            "updated_embeddings": int(updated),
            "mean_silhouette": float(result.get("mean_silhouette", 0.0)),
        }

    def _sleep_glyph_tick(self) -> dict[str, Any]:
        consolidator = self._get_sleep_glyph_consolidator()
        if consolidator is None:
            return {"status": "skipped", "reason": "sleep_glyph_consolidator_unavailable"}
        sample_rows = self._live_embedding_rows(
            galaxy_names=["Character"],
            limit=min(int(self.config.sleep_sample_size), 512),
        )
        if len(sample_rows) < 2:
            return {"status": "skipped", "reason": "insufficient_glyph_embeddings", "rows": len(sample_rows)}
        matrix = [embedding for _, _, embedding in sample_rows]
        result = consolidator.consolidate_glyphs(matrix, similarity_threshold=0.92)
        return {
            "status": "ok",
            "rows": len(sample_rows),
            "group_count": int(result.get("group_count", 0)),
            "group_sizes": list(result.get("group_sizes", []))[:12],
        }

    def _sleep_memory_update_tick(self) -> dict[str, Any]:
        try:
            from knowledge3d.cranium.ptx_runtime.galaxy_memory_updater import GalaxyMemoryUpdater

            updater = GalaxyMemoryUpdater()
        except Exception:
            return {"status": "skipped", "reason": "galaxy_memory_updater_unavailable"}
        sample_rows = self._live_embedding_rows(
            galaxy_names=self.kv._discover_live_galaxy_names(),
            limit=min(int(self.config.sleep_sample_size), 256),
        )
        if len(sample_rows) < 2:
            return {"status": "skipped", "reason": "insufficient_embeddings", "rows": len(sample_rows)}
        matrix = [[float(value) for value in embedding] for _, _, embedding in sample_rows]
        teacher = _mean_embedding_rows(matrix)
        if not teacher:
            return {"status": "skipped", "reason": "invalid_embedding_rows", "rows": len(sample_rows)}
        updated_rows: list[list[float]] = []
        for row in matrix:
            blended = updater.blend(row, teacher, blend_factor=0.06)
            updated_rows.append([float(value) for value in blended.reshape(-1).tolist()])
        updated = self._apply_embedding_updates(sample_rows, updated_rows)
        return {
            "status": "ok",
            "rows": len(sample_rows),
            "updated_embeddings": int(updated),
            "blend_factor": 0.06,
        }

    def _sleep_graph_crystallization_tick(self) -> dict[str, Any]:
        crystallizer = self.kv.get_graph_crystallizer() if hasattr(self.kv, "get_graph_crystallizer") else None
        if crystallizer is None:
            return {"status": "skipped", "reason": "graph_crystallizer_unavailable"}
        graph = getattr(self.kv, "_semantic_csr_graph", None)
        catalog = self.kv.get_gpu_galaxy_catalog()
        if graph is None or not catalog:
            return {"status": "skipped", "reason": "semantic_graph_unavailable"}

        max_rows = min(int(self.config.sleep_sample_size), min(256, len(catalog)))
        sample_indexes = list(range(max_rows))
        node_rows: list[list[float]] = []
        neighbor_rows: list[list[float]] = []
        for idx in sample_indexes:
            entry = catalog[idx]
            node_rows.append([float(value) for value in entry.get("embedding16", [0.0] * 16)[:16]])
            row_start = int(graph.row_offsets[idx])
            row_end = int(graph.row_offsets[idx + 1])
            neighbors = [
                [float(value) for value in catalog[int(graph.col_indices[edge_idx])].get("embedding16", [0.0] * 16)[:16]]
                for edge_idx in range(row_start, row_end)
                if int(graph.col_indices[edge_idx]) < len(catalog)
            ]
            if neighbors:
                neighbor_rows.append(_mean_embedding_rows(neighbors))
            else:
                neighbor_rows.append(list(node_rows[-1]))
        crystallized = crystallizer.crystallize_list(node_rows, neighbor_rows, ema_rate=0.985)
        return {
            "status": "ok",
            "rows": len(sample_indexes),
            "graph_signature": str(getattr(graph, "signature", "")),
            "crystallized_rows": len(crystallized),
        }

    def _sleep_lane_a_ingest_tick(self) -> dict[str, Any]:
        """Sleeptime Lane A — GPU-based temporary-star promote / merge / discard.

        Python is launcher + I/O only (feedback_sleeptime_orchestration_is_ptx_not_python.md).
        All gravity computation and decision logic runs inside sleeptime_lane_a.cu.
        """
        try:
            from knowledge3d.knowledgeverse.sleeptime_ingest import run_lane_a_tick
        except Exception as exc:
            return {"status": "skipped", "reason": f"sleeptime_ingest_import_failed: {exc}"}

        storage_root = getattr(self.kv, "storage_root", None)
        if storage_root is None:
            return {"status": "skipped", "reason": "storage_root_unavailable"}

        # Collect existing house stars for the gravity probe.
        # This is JSONL I/O — Python reads existing galaxy entries (not reasoning).
        house_stars: list[Any] = []
        try:
            catalog = self.kv.get_gpu_galaxy_catalog() if hasattr(self.kv, "get_gpu_galaxy_catalog") else []
            if catalog:
                house_stars = list(catalog)
        except Exception:
            house_stars = []

        # Collect grammar rules for defeasibility pass.
        grammar_rules: list[Any] = []
        try:
            gm = getattr(self.kv, "galaxy_manager", None)
            if gm is not None:
                grammar_gal = gm.get_galaxy("Grammar") if hasattr(gm, "get_galaxy") else None
                if grammar_gal is not None:
                    grammar_rules = [
                        e for e in getattr(grammar_gal, "entries", [])
                        if isinstance(e, dict) and "rule_strength" in e
                    ]
        except Exception:
            grammar_rules = []

        return run_lane_a_tick(
            storage_root=Path(storage_root),
            house_stars=house_stars,
            grammar_rules=grammar_rules,
        )

    def _sleep_lane_b_weights_tick(self) -> dict[str, Any]:
        """Sleeptime Lane B — GPU-based wake-cycle weight consolidation with
        live inline f32→BitNet convert-on-touch (Gap 2, opcode 0x313).

        Folds shadow-copy delta traces from the wake cycle into the TRM
        BitNet b1.58 weight tiles, in VRAM, via sleeptime_lane_b.cu.  On the
        first-ever touch of each tile the kernel quantises its float32 source
        (from the .bin checkpoint) into packed 1.6-bit tiles and flips the
        matching tile_format byte.  Idempotent — already-packed tiles skip.

        Python is launcher + I/O only (feedback_sleeptime_orchestration_is_ptx_not_python.md):
          • read f32 bytes from kv._trm_host_weights
          • read/update the .tile_format + .bitnet sidecars
          • hand the whole batch to the PTX kernel
          • read back the updated sidecars written by the launcher's I/O tail
        No Python iteration over tiles.  No boot-time batch conversion
        (feedback_live_inline_weight_conversion.md).  No fallbacks
        (feedback_no_fallbacks_ever_including_sleeptime.md).
        """
        try:
            from knowledge3d.knowledgeverse.sleeptime_weights import run_lane_b_tick
        except Exception as exc:
            return {"status": "skipped", "reason": f"sleeptime_weights_import_failed: {exc}"}

        # ── Constants mirroring sleeptime_weights.py / sleeptime_lane_b.cu ──
        _TILE_TRITS = 20
        _TILE_BYTES = 4
        _F32_PER_TILE_BYTES = _TILE_TRITS * 4   # 80 bytes of f32 per tile
        _WEIGHT_FORMAT_F32 = 0                   # default for freshly loaded .bin tiles

        # ── Collect shadow-copy event buffer (JSONL I/O, not math) ──────────
        shadow = getattr(self.kv, "shadow_copy", None)
        if shadow is None:
            return {"status": "skipped", "reason": "shadow_copy_unavailable"}

        shadow_events: list[Any] = list(getattr(shadow, "event_buffer", []))

        # ── Collect raw f32 weight bytes from kv._trm_host_weights ─────────
        # _trm_host_weights: dict[str, bytes] — raw float32 bytes loaded from the
        # K3DTRM01 .bin checkpoint (one entry per matrix: W1..W4).  Each entry
        # is rows*cols*4 bytes of little-endian float32.  We concatenate in the
        # canonical order defined by kv.TRM_WEIGHT_SHAPES to keep tile ordering
        # stable across daemon restarts (so the .tile_format sidecar stays aligned).
        host_weights: dict[str, Any] = getattr(self.kv, "_trm_host_weights", None) or {}
        if not host_weights:
            return {"status": "skipped", "reason": "trm_host_weights_unavailable"}

        canonical_names = tuple(getattr(self.kv, "TRM_WEIGHT_SHAPES", {}).keys())
        if not canonical_names:
            return {"status": "skipped", "reason": "trm_weight_shapes_unavailable"}

        missing = [n for n in canonical_names if n not in host_weights]
        if missing:
            # No fallback: this is a loud structural error, not a silent skip.
            raise RuntimeError(
                "sleeptime_lane_b: kv._trm_host_weights missing canonical matrices "
                f"{missing}; checkpoint load invariant broken."
            )

        f32_bytes_parts = []
        for name in canonical_names:
            raw = host_weights[name]
            f32_bytes_parts.append(bytes(raw) if not isinstance(raw, (bytes, bytearray)) else raw)
        f32_concat = b"".join(f32_bytes_parts)

        if not f32_concat:
            return {"status": "skipped", "reason": "trm_host_weights_empty"}

        if len(f32_concat) % 4 != 0:
            # Loud: f32 buffer must be word-aligned — a corrupt checkpoint is a bug.
            raise RuntimeError(
                f"sleeptime_lane_b: f32 weight buffer length {len(f32_concat)} "
                "is not a multiple of 4 (float32 word size); checkpoint corrupt."
            )

        # Compute the AUTHORITATIVE count of real (non-padding) f32 weights
        # BEFORE any tile-boundary padding.  This is what the .n_valid sidecar
        # records; readback consumers honour it to avoid the information loss
        # that silent zero-padding would cause on unpack (a padded zero trit is
        # indistinguishable from a real weight rounded to zero).  See Daniel's
        # ruling in feedback (2026-04-21, tail-pad information-loss fix):
        # "Require exact fit, but do not refuse, recalculate and accept."
        total_f32_floats = len(f32_concat) // 4
        n_valid_weights = total_f32_floats  # real weights, pre-pad

        # Pad f32 buffer up to a whole number of tiles (TILE_TRITS floats each).
        # Padding is I/O-level byte filling — not math.  Padded trits quantise to
        # 0 under quantize_trit(0.0f) = 0 and contribute nothing to reasoning.
        # Mis-aligned checkpoints (total_f32_floats % _TILE_TRITS != 0) are
        # ACCEPTED here: we zero-pad the tail for the GPU layout and rely on
        # n_valid_weights (persisted to the sidecar) to bound any later readback.
        n_tiles = (total_f32_floats + _TILE_TRITS - 1) // _TILE_TRITS
        expected_f32_bytes = n_tiles * _F32_PER_TILE_BYTES
        if len(f32_concat) < expected_f32_bytes:
            f32_concat = f32_concat + b"\x00" * (expected_f32_bytes - len(f32_concat))

        # ── Resolve sidecar paths ──────────────────────────────────────────
        storage_root = getattr(self.kv, "storage_root", None)
        if storage_root is None:
            return {"status": "skipped", "reason": "storage_root_unavailable"}
        weights_dir = Path(storage_root) / "weights"
        checkpoint_name = "trm_bitnet"
        bitnet_path = weights_dir / f"{checkpoint_name}.bitnet"
        tile_format_path = weights_dir / f"{checkpoint_name}.tile_format"
        n_valid_path = weights_dir / f"{checkpoint_name}.n_valid"

        # ── Load sidecars into memory cache on first tick ──────────────────
        # After first tick, cache is authoritative during this daemon's lifetime;
        # we still re-read after each launcher call because the launcher writes
        # the updated bytes to disk as its I/O tail.
        if not self._lane_b_sidecars_loaded:
            # tile_format: missing file → all-F32 (first run, every tile needs quantising)
            if tile_format_path.exists():
                fmt_bytes = tile_format_path.read_bytes()
                if len(fmt_bytes) != n_tiles:
                    # Sidecar size mismatch → fail loud (no fallback).  Most likely
                    # cause: checkpoint shapes changed — delete sidecars and re-boot.
                    raise RuntimeError(
                        f"sleeptime_lane_b: {tile_format_path} has {len(fmt_bytes)} "
                        f"bytes, expected {n_tiles}; delete the sidecars at "
                        f"{weights_dir} and re-boot to re-quantise on next tick."
                    )
                self._lane_b_tile_format = fmt_bytes
            else:
                self._lane_b_tile_format = bytes([_WEIGHT_FORMAT_F32] * n_tiles)

            # packed tiles: missing file → zeroed buffer (kernel will overwrite on touch)
            if bitnet_path.exists():
                packed_bytes = bitnet_path.read_bytes()
                if len(packed_bytes) != n_tiles * _TILE_BYTES:
                    raise RuntimeError(
                        f"sleeptime_lane_b: {bitnet_path} has {len(packed_bytes)} "
                        f"bytes, expected {n_tiles * _TILE_BYTES}; delete the "
                        f"sidecars at {weights_dir} and re-boot."
                    )
                self._lane_b_bitnet_tiles = packed_bytes
            else:
                self._lane_b_bitnet_tiles = b"\x00" * (n_tiles * _TILE_BYTES)

            # n_valid sidecar: authoritative real-weight count (Gap 2 tail-pad fix).
            # Missing file → seed from the derived count (first write, no prior state).
            # Present file → MUST match the derived count exactly; any disagreement
            # indicates the checkpoint's f32 layout changed between boots (e.g. a
            # matrix was resized) and the stored count is now a lie.  That is
            # GENUINE corruption — fail loud, no silent recovery.
            from knowledge3d.knowledgeverse.sleeptime_weights import (
                read_n_valid_weights,
            )
            stored_n_valid = read_n_valid_weights(weights_dir, checkpoint_name)
            if stored_n_valid is None:
                self._lane_b_n_valid = n_valid_weights
            else:
                if stored_n_valid != n_valid_weights:
                    raise RuntimeError(
                        f"sleeptime_lane_b: {n_valid_path} stores "
                        f"n_valid_weights={stored_n_valid} but current f32 buffer "
                        f"derives n_valid_weights={n_valid_weights}; "
                        "checkpoint count disagrees with sidecar — genuine "
                        f"corruption.  Delete the sidecars at {weights_dir} and "
                        "re-boot to re-quantise on next tick."
                    )
                self._lane_b_n_valid = stored_n_valid

            self._lane_b_n_tiles = n_tiles
            self._lane_b_sidecars_loaded = True

        # ── Tile-count invariant ───────────────────────────────────────────
        # Once loaded, n_tiles must stay constant for the daemon's lifetime
        # (checkpoint shape is fixed).  A mismatch is a structural bug, not a
        # recoverable condition — raise loud, no fallback.
        if self._lane_b_n_tiles != n_tiles:
            raise RuntimeError(
                f"sleeptime_lane_b: tile count changed mid-flight "
                f"(cached {self._lane_b_n_tiles}, now {n_tiles}); "
                "checkpoint shape invariant broken."
            )

        # n_valid-count invariant (Gap 2 tail-pad fix).  Cached authoritative
        # count must also stay constant for the daemon's lifetime: the canonical
        # TRM_WEIGHT_SHAPES order is fixed, so total_f32_floats is fixed.  Any
        # drift is GENUINE corruption (someone mutated _trm_host_weights under
        # us) — fail loud.
        if self._lane_b_n_valid != n_valid_weights:
            raise RuntimeError(
                f"sleeptime_lane_b: n_valid_weights changed mid-flight "
                f"(cached {self._lane_b_n_valid}, now {n_valid_weights}); "
                "kv._trm_host_weights mutated under the daemon — genuine corruption."
            )

        # ── Dispatch the kernel via the PTX launcher (I/O only here) ───────
        # The launcher raises RuntimeError on any kernel failure — we let that
        # propagate (no except/pass).  Per sovereignty rules Python does not
        # iterate tiles: the whole batch goes to the GPU in one launch.
        # n_valid_weights is passed through so the launcher persists it to the
        # .n_valid sidecar; readback consumers honour it as the pre-padding
        # truth (first n_valid_weights trits = real weights; tail = padding).
        result = run_lane_b_tick(
            weight_tiles_bytes=self._lane_b_bitnet_tiles,
            n_tiles=n_tiles,
            shadow_events=shadow_events,
            weights_dir=weights_dir,
            checkpoint_name=checkpoint_name,
            weight_tiles_f32_bytes=f32_concat,
            tile_format_bytes=self._lane_b_tile_format,
            n_valid_weights=n_valid_weights,
        )

        # ── Refresh in-memory cache from the launcher's sidecar writes ─────
        # The launcher writes the updated bytes to disk as its I/O tail; we
        # re-read so subsequent ticks pass the fresh state to the kernel
        # (and already-BITNET tiles short-circuit in the idempotent path).
        if result.get("status") == "ok":
            if tile_format_path.exists():
                self._lane_b_tile_format = tile_format_path.read_bytes()
            if bitnet_path.exists():
                self._lane_b_bitnet_tiles = bitnet_path.read_bytes()

        return result

    def _run_sleep_consolidation_tick(self) -> dict[str, Any]:
        tick_name = self._sleep_tick_order[self._sleep_tick_cursor % len(self._sleep_tick_order)]
        handlers = {
            "cluster_refiner": self._sleep_cluster_tick,
            "glyph_consolidator": self._sleep_glyph_tick,
            "memory_updater": self._sleep_memory_update_tick,
            "graph_crystallizer": self._sleep_graph_crystallization_tick,
            "lane_a_ingest": self._sleep_lane_a_ingest_tick,
            "lane_b_weights": self._sleep_lane_b_weights_tick,
        }
        started = time.perf_counter()
        try:
            summary = dict(handlers[tick_name]())
        except Exception as exc:
            summary = {
                "status": "error",
                "exception_type": type(exc).__name__,
                "detail": str(exc),
            }
        summary.update(
            {
                "tick_name": tick_name,
                "tick_index": int(self._sleep_tick_count),
                "elapsed_ms": float((time.perf_counter() - started) * 1000.0),
                "timestamp": _now_iso(),
            }
        )
        self._sleep_tick_count += 1
        self._sleep_tick_cursor = (self._sleep_tick_cursor + 1) % len(self._sleep_tick_order)
        self._last_sleep_tick = summary
        self._sleep_tick_history.append(dict(summary))
        if len(self._sleep_tick_history) > int(self._sleep_tick_history_max):
            self._sleep_tick_history = self._sleep_tick_history[-int(self._sleep_tick_history_max) :]
        return dict(summary)

    def _persist_sleep_state(self) -> dict[str, Any]:
        pending = int(self._pending_sleep_embedding_updates)
        if pending <= 0:
            return {"status": "skipped", "reason": "no_pending_updates"}
        manager = getattr(self.kv, "galaxy_manager", None)
        if manager is None or not hasattr(manager, "_rewrite_galaxy_disk"):
            return {"status": "skipped", "reason": "galaxy_persistence_unavailable"}

        persisted = 0
        failures: list[str] = []
        for galaxy_name in self.kv._discover_live_galaxy_names():
            try:
                galaxy = manager.get_galaxy(galaxy_name)
                manager._rewrite_galaxy_disk(str(galaxy_name), galaxy)
                persisted += 1
            except Exception as exc:
                failures.append(f"{galaxy_name}:{type(exc).__name__}")
        self._pending_sleep_embedding_updates = 0
        return {
            "status": "ok",
            "persisted_galaxies": int(persisted),
            "pending_updates_flushed": int(pending),
            "failures": failures,
        }

    def _advance_idle_clock(self, *, had_request: bool) -> dict[str, Any] | None:
        if had_request:
            self._idle_elapsed_seconds = 0.0
            return None
        self._idle_elapsed_seconds += float(self.config.tcp_poll_seconds)
        if self._idle_elapsed_seconds + 1e-9 < float(self.config.idle_threshold_seconds):
            return None
        self._idle_elapsed_seconds = 0.0
        return self._run_sleep_consolidation_tick()

    @property
    def should_shutdown(self) -> bool:
        return self._shutdown_requested

    def _finalize_shutdown(self) -> dict[str, Any]:
        summary: dict[str, Any] = {"status": "ok"}
        tick_driver = getattr(self, "_tick_driver", None)
        if tick_driver is not None:
            try:
                tick_driver.stop()
                summary["tick_driver"] = tick_driver.stats()
            except Exception as exc:
                summary["tick_driver_error"] = str(exc)
        self.kv._external_tick_driver_active = False
        boundary = self._tablet_boundary
        if boundary is not None:
            try:
                if bool(boundary.live_session_status().get("active", False)):
                    summary["tablet_session"] = boundary.close_live_session()
            except Exception as exc:
                summary["tablet_session_error"] = str(exc)
        try:
            summary["knowledgeverse"] = self.kv.shutdown(persist=False, profile="benchmark")
        except Exception as exc:
            summary["knowledgeverse_error"] = str(exc)
        return summary

    @staticmethod
    def _implied_gpu_calls(result: dict[str, Any]) -> int:
        task_result = result.get("task_result") if isinstance(result.get("task_result"), dict) else {}
        direct_gpu = bool(result.get("gpu_execution", False))
        task_gpu = bool(task_result.get("gpu_execution", False))
        task_status = str(task_result.get("status") or "").strip().lower()
        task_has_trm = "trm_tick" in task_result or "action_buffers" in task_result
        return 1 if (direct_gpu or task_gpu or task_has_trm or task_status == "success") else 0

    def status_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "timestamp": _now_iso(),
            "daemon_started_at": self.started_at,
            "pid": int(os.getpid()),
            "require_ptx_query": bool(self.config.require_ptx_query),
            "manifest_version": str(self.kv.manifest_version),
            "default_galaxy_counts": dict(self._default_counts),
            "command_count": int(self._command_count),
            "gpu_calls_total": int(self._gpu_calls_total),
            "cuda_env": dict(self._cuda_env),
            "gpu_binding": self._binding_report(),
            "semantic_csr_graph": self._semantic_graph_report(),
            "idle_threshold_seconds": float(self.config.idle_threshold_seconds),
            "sleep_tick_count": int(self._sleep_tick_count),
            "last_sleep_tick": dict(self._last_sleep_tick),
            "sleep_tick_history": [dict(item) for item in self._sleep_tick_history],
            "boot_status_paths": [str(path) for path in self._boot_status_paths],
            "tick_driver": getattr(self, "_tick_driver", None).stats() if getattr(self, "_tick_driver", None) is not None else {},
        }

    def _gpu_call_snapshot(self) -> int:
        if ModularRPNEngine is None:
            return 0
        try:
            return int(ModularRPNEngine.get_global_gpu_call_count())
        except Exception:
            return 0

    def _dispatch_task(
        self,
        *,
        route: dict[str, Any],
        task: dict[str, Any] | None,
        query: str,
        specialist: str,
        domain_hint: str | None,
        use_enriched: bool,
        max_wall_ms: int | None = None,
    ) -> dict[str, Any]:
        if not hasattr(self.kv, "execute_task"):
            return {"status": "error", "error": "knowledgeverse_missing_execute_task"}
        route_policy = str(route.get("route_policy") or "").strip().lower()
        galaxy_names = [
            str(name)
            for name in (route.get("galaxy_names") or [])
            if str(name).strip()
        ]
        task_payload = self._meaning_task_payload(
            task=task,
            query=query,
            galaxies=galaxy_names,
            route_policy=route_policy,
        )
        response = self.kv.execute_task(
            task=task_payload,
            route=route,
            specialist=str(specialist or "auto"),
            domain_hint=domain_hint,
            use_enriched=use_enriched,
            max_wall_ms=max_wall_ms,
        )
        result = dict(response or {})
        if isinstance(task, dict) and task.get("task_id") is not None and "task_id" not in result:
            result["task_id"] = task.get("task_id")
        result.setdefault("runtime", "knowledgeverse_gpu_query")
        return result

    def _wire_activation_scratch(self) -> None:
        """Expose the composed head's final-layer activation buffer on
        kv._activation_scratch so wake-delta capture can read it without
        allocating a stub.

        Source of truth: the TRM step-fused bridge writes its final-layer
        activation vector to the `y_new` device pointer inside
        `bridge._default_tick_buffers`.  We publish that pointer as
        (device_ptr, T=1, N=TRM_DIMS) on the Knowledgeverse.  The kernel
        tile count must match n_tiles derived from _trm_host_weights;
        callers that mismatch T fail loud (no silent reshape).

        No fallback — if the TRM launcher/bridge isn't available, we leave
        kv._activation_scratch unset and capture_wake_delta will raise with
        a clear error.  Per Daniel's ruling "1 - wire it" (2026-04-21).
        """
        kv = self.kv
        if getattr(kv, "_activation_scratch", None) is not None:
            return
        launcher = getattr(kv, "_trm", None)
        if launcher is None:
            return
        bridge = getattr(launcher, "_step_fused_bridge", None)
        if bridge is None:
            return
        tick_buffers = getattr(bridge, "_default_tick_buffers", None)
        if not isinstance(tick_buffers, dict):
            return
        y_new_ptr = tick_buffers.get("y_new")
        if y_new_ptr is None:
            return
        # TRM_DIMS = 512 floats of final-layer activation written by
        # trm_step_fused.  Expose as one tile of TRM_DIMS activations;
        # the wake-delta kernel expects act_T == n_tiles so the weight
        # inventory must be a single tile when this wiring is used.
        try:
            from knowledge3d.cranium.bridges.trm_step_fused_bridge import TRM_DIMS
        except Exception:
            return
        kv._activation_scratch = (y_new_ptr, 1, int(TRM_DIMS))

    def _read_real_halting_value(self, solved: dict[str, Any]) -> float | None:
        """Read the sovereign halting scalar propagated by the TRM tick.

        Source of truth: `solved["trm_tick"]["halting_value"]` — a float
        in [0.0, 1.0] written by the swarm halting gate in PTX and
        propagated through the TRM game loop.  See
        `knowledge3d/cranium/cuda/k3d_swarm_persistent.cu` (PTX write
        site) and `TEMP/CLAUDE_HALTING_READBACK_HOOK_SPEC_04.21.2026.md`
        (full hook contract).

        Returns None only when the tick did not produce a halting scalar
        (non-GPU path, or the swarm was not invoked for this task — e.g.
        degenerate single-candidate paths).  The caller silently declines
        to emit a wake-delta in that case — an observation, not a
        reasoning fallback.
        """
        if str(solved.get("status", "")).lower() != "ok":
            return None
        if not bool(solved.get("gpu_execution", False)):
            return None
        trm_tick = solved.get("trm_tick") or {}
        if "halting_value" not in trm_tick:
            return None
        value = float(trm_tick.get("halting_value", 0.0) or 0.0)
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _maybe_emit_wake_delta(
        self,
        solved: dict[str, Any],
        halting_value: float,
    ) -> None:
        """Append a wake-cycle delta event to shadow_copy.event_buffer.

        Called from CHAT / ROUTE / SOLVE_MATH handlers after execute_task()
        returns a converged result AND the caller has already read the
        real halting scalar from the composed head.  Fires the
        WAKE_CYCLE_DELTA_CAPTURE kernel (opcode 0x320) which reads TRM
        activation scratch, computes per-tile signed-magnitude deltas,
        and packages them for Sleeptime Lane B.

        halting_value is REQUIRED (no default) — per Daniel's ruling
        "3 - No default! real halting readback." (2026-04-21).  The
        caller is responsible for sourcing it from the sovereign path
        (see _read_real_halting_value).

        Exceptions propagate — per Daniel's ruling "2 - fail loud."
        No bare except, no silent swallow.
        """
        if str(solved.get("status", "")).lower() != "ok":
            return
        if not bool(solved.get("gpu_execution", False)):
            return
        shadow = getattr(self.kv, "shadow_copy", None)
        if shadow is None:
            return
        self._wire_activation_scratch()
        from knowledge3d.knowledgeverse.wake_delta_capture import capture_wake_delta
        delta_event = capture_wake_delta(
            self,
            halting_value=float(halting_value),
            confidence=float(halting_value),
        )
        if delta_event is not None:
            shadow.record_event(
                event_type="wake_delta_capture",
                event_data=delta_event,
            )

    def handle_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._command_count += 1
        cmd = str(payload.get("command", "")).strip().upper()
        if not cmd:
            return {"status": "error", "error": "missing_command"}

        if cmd in {"PING", "STATUS"}:
            return self.status_payload()

        if cmd == "VRAM_REPORT":
            return self._vram_report_payload()

        if cmd == "TICK_STATUS":
            tick_driver = getattr(self, "_tick_driver", None)
            return {
                "status": "ok",
                "tick_driver": tick_driver.stats() if tick_driver is not None else {},
            }

        if cmd == "SHUTDOWN":
            persist_result = self._persist_sleep_state()
            self._shutdown_requested = True
            return {
                "status": "ok",
                "message": "shutdown_requested",
                "timestamp": _now_iso(),
                "sleep_persistence": persist_result,
            }

        if cmd == "TABLET_SESSION_OPEN":
            session_id = str(payload.get("session_id") or f"daemon_tablet_session_{int(time.time() * 1000)}")
            boundary = self._get_tablet_boundary()
            try:
                session = boundary.open_live_session(
                    session_id=session_id,
                    reset_runtime=bool(payload.get("reset_runtime", True)),
                    tick_hz=float(payload.get("tick_hz", 50.0) or 50.0),
                    delta_time=float(payload.get("delta_time", 0.02) or 0.02),
                    enforce_preflight=bool(payload.get("enforce_preflight", True)),
                )
            except Exception as exc:
                return {
                    "status": "error",
                    "error": "tablet_session_open_failed",
                    "detail": str(exc),
                }
            return {
                "status": "ok",
                "session": session,
            }

        if cmd == "TABLET_SESSION_STATUS":
            boundary = self._get_tablet_boundary()
            return {
                "status": "ok",
                "session": boundary.live_session_status(),
            }

        if cmd == "TABLET_SESSION_CLOSE":
            boundary = self._get_tablet_boundary()
            return {
                "status": "ok",
                "session": boundary.close_live_session(),
            }

        if cmd == "TABLET_SESSION_RUN_TAPE":
            tape_payload = payload.get("tape")
            if not isinstance(tape_payload, dict):
                return {"status": "error", "error": "tablet_session_tape_missing"}
            boundary = self._get_tablet_boundary()
            try:
                tape = TabletSessionTape.from_payload(tape_payload)
                response = boundary.run_tape_session(
                    tape,
                    tick_hz=float(payload.get("tick_hz", 50.0) or 50.0),
                    delta_time=float(payload.get("delta_time", 0.02) or 0.02),
                    frame_timeout_s=float(payload.get("frame_timeout_s", 30.0) or 30.0),
                    enforce_preflight=bool(payload.get("enforce_preflight", True)),
                    reuse_open_session=bool(boundary.live_session_status().get("active", False)),
                )
            except Exception as exc:
                return {
                    "status": "error",
                    "error": "tablet_session_run_tape_failed",
                    "detail": str(exc),
                }
            serializable_results: list[dict[str, Any]] = []
            for row in list(response.get("results") or []):
                if not isinstance(row, dict):
                    continue
                clean = dict(row)
                envelope = clean.get("envelope")
                if hasattr(envelope, "to_payload"):
                    clean["envelope"] = envelope.to_payload()
                serializable_results.append(clean)
            return {
                "status": "ok",
                "session": {
                    "session_id": response.get("session_id"),
                    "suite_name": response.get("suite_name"),
                    "surface_kind": response.get("surface_kind"),
                    "preflight": response.get("preflight", {}),
                },
                "results": serializable_results,
            }

        if cmd == "ROUTE":
            task = payload.get("task")
            if task is not None and not isinstance(task, dict):
                return {"status": "error", "error": "task_must_be_object"}
            task_obj = task if isinstance(task, dict) else None
            if not any(int(value) > 0 for value in self._default_counts.values()):
                self._default_counts = self.kv.ensure_default_galaxies_loaded()
            query = self._coalesce_query(payload, task_obj)
            if not query:
                return {"status": "error", "error": "missing_query_or_task"}
            use_enriched = bool(payload.get("use_enriched", True))
            all_galaxies = self._all_default_galaxies()
            route_policy = str(
                payload.get("route_policy") or (task_obj or {}).get("route_policy") or ""
            ).strip().lower()
            preferred_galaxies = [
                str(name)
                for name in (
                    payload.get("galaxies")
                    or (task_obj or {}).get("galaxies")
                    or []
                )
                if str(name).strip()
            ]
            route_galaxies = list(all_galaxies) if route_policy == "all_live_galaxies" else (
                preferred_galaxies or list(all_galaxies)
            )
            specialist = str(payload.get("specialist") or (task_obj or {}).get("specialist") or "auto").strip() or "auto"
            domain_hint = str(payload.get("domain_hint") or (task_obj or {}).get("domain_hint") or "").strip() or None
            route = self._meaning_route(
                specialist=specialist,
                domain_hint=domain_hint,
                galaxy_names=list(route_galaxies),
                route_policy=route_policy or "all_live_galaxies",
            )
            dispatched = dict(
                self._dispatch_task(
                    route=route,
                    task=task_obj,
                    query=query,
                    specialist=specialist,
                    domain_hint=domain_hint,
                    use_enriched=use_enriched,
                    max_wall_ms=payload.get("max_wall_ms"),
                )
                or {}
            )
            response: dict[str, Any] = {
                "status": "ok",
                "route": dict(dispatched.get("route") or route),
                "task_result": dispatched,
            }
            response["task_status"] = str(dispatched.get("status") or "ok").strip().lower() or "ok"
            real_halting = self._read_real_halting_value(dispatched)
            if real_halting is not None:
                self._maybe_emit_wake_delta(dispatched, real_halting)
            return response

        if cmd == "QUERY":
            query = str(payload.get("query", "")).strip()
            if not query:
                return {"status": "error", "error": "missing_query"}
            top_k = int(payload.get("top_k", 10))
            rows = self.trm.query(
                query=query,
                galaxy_names=payload.get("galaxies"),
                top_k=max(1, top_k),
                specialist=str(payload.get("specialist", "auto")),
                domain_hint=payload.get("domain_hint"),
            )
            return {
                "status": "ok",
                "count": len(rows),
                "results": rows,
            }

        if cmd == "SOLVE_MATH":
            question = str(payload.get("question", "") or payload.get("query", "")).strip()
            if not question:
                return {"status": "error", "error": "missing_question"}
            use_enriched = bool(payload.get("use_enriched", True))
            solved = self.kv.execute_task(
                task={
                    "query": question,
                    "question": question,
                },
                route={
                    "galaxy_names": self._all_default_galaxies(),
                    "route_policy": "all_live_galaxies",
                },
                specialist="math",
                domain_hint=str(payload.get("domain_hint") or "math"),
                use_enriched=use_enriched,
            )
            if str(solved.get("status", "")).lower() != "ok":
                return {
                    "status": "error",
                    "error": "knowledgeverse_math_query_failed",
                    "detail": solved,
                }
            real_halting = self._read_real_halting_value(solved)
            if real_halting is not None:
                self._maybe_emit_wake_delta(solved, real_halting)
            return {
                "status": "ok",
                "result": solved.get("result"),
                "program_id": solved.get("program_id"),
                "runtime": solved.get("runtime"),
                "gpu_execution": bool(solved.get("gpu_execution", False)),
            }

        if cmd == "CHAT":
            from knowledge3d.tablet.wine.chat_wine import _validate_chat_input
            raw_messages = payload.get("messages")
            if not isinstance(raw_messages, list):
                prompt = str(payload.get("prompt", "") or payload.get("query", "")).strip()
                if not prompt:
                    return {"status": "error", "error": "missing_messages_or_prompt"}
                raw_messages = [{"role": "user", "content": prompt}]
            messages = raw_messages
            context = dict(payload.get("context") or {})
            stream = bool(payload.get("stream", False))
            task_id = payload.get("task_id")

            # Input gate — pure type/length checks, not reasoning.
            validation_error = _validate_chat_input(messages, context)
            if validation_error is not None:
                return validation_error

            # Route through the WINE contract — no inline envelope construction.
            from knowledge3d.bridge.headless_tablet import TabletIngest as _TabletIngest
            envelope = _TabletIngest.chat_task(
                messages,
                context=context,
                stream=stream,
                task_id=task_id,
            )

            solved = self.kv.execute_task(
                task=dict(envelope.task),
                route={
                    "galaxy_names": self._all_default_galaxies(),
                    "route_policy": "all_live_galaxies",
                },
                specialist="chat",
                domain_hint=str(payload.get("domain_hint") or "general"),
                use_enriched=bool(payload.get("use_enriched", True)),
            )
            if str(solved.get("status", "")).lower() != "ok":
                return {"status": "error", "error": "knowledgeverse_chat_query_failed", "detail": solved}
            real_halting = self._read_real_halting_value(solved)
            if real_halting is not None:
                self._maybe_emit_wake_delta(solved, real_halting)
            return {
                "status": "ok",
                "response": solved.get("response", solved.get("answer", "")),
                "program_id": solved.get("program_id"),
                "gpu_execution": bool(solved.get("gpu_execution", False)),
                "telemetry": dict(solved.get("telemetry") or {}),
                "task_id": envelope.task_id or task_id,
                "task_result": solved,
            }

        if cmd == "INGEST":
            source_uri = payload.get("source_uri")
            mime = payload.get("mime")
            chunking = dict(payload.get("chunking") or {})
            lang_hint = payload.get("lang_hint")
            task_id = payload.get("task_id") or _make_ingest_id()

            validation_error = _validate_ingest_input(source_uri, mime, chunking)
            if validation_error is not None:
                return validation_error

            from knowledge3d.bridge.headless_tablet import TabletIngest as _TabletIngest
            envelope = _TabletIngest.ingest_task(
                task_id=task_id,
                source_uri=str(source_uri),
                mime=str(mime),
                chunking=chunking or None,
                lang_hint=lang_hint,
            )

            # Queue into temporary-star region. Synchronous receipt.
            receipt = self.kv.enqueue_ingest(envelope=envelope)

            return {
                "status": "ok",
                "result_kind": "ingest_receipt",
                "ingest_id": receipt["ingest_id"],
                "task_id": task_id,
                "queued_chunks_estimate": receipt.get("queued_chunks_estimate", 0),
                "telemetry": dict(receipt.get("telemetry") or {}),
            }

        return {"status": "error", "error": "unknown_command", "command": cmd}

    def _handle_line(self, raw_line: str) -> str:
        cmd_started = time.perf_counter()
        gpu_before = self._gpu_snapshot()
        gpu_calls_before = self._gpu_call_snapshot()
        line = raw_line.strip()
        if not line:
            response = {"status": "error", "error": "empty_command"}
            return json.dumps(response, separators=(",", ":"))
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {
                "status": "error",
                "error": "invalid_json",
                "detail": str(exc),
            }
            return json.dumps(
                {"status": "error", "error": "invalid_json", "detail": str(exc)},
                separators=(",", ":"),
            )
        if not isinstance(payload, dict):
            return json.dumps({"status": "error", "error": "command_must_be_object"}, separators=(",", ":"))
        try:
            result = self.handle_command(payload)
        except Exception as exc:
            result = {
                "status": "error",
                "error": "command_execution_failed",
                "exception_type": type(exc).__name__,
                "detail": str(exc),
            }
        gpu_after = self._gpu_snapshot()
        gpu_calls_after = self._gpu_call_snapshot()
        gpu_calls_this_command = max(
            0,
            int(gpu_calls_after - gpu_calls_before),
            int(self._implied_gpu_calls(result)),
        )
        self._gpu_calls_total += gpu_calls_this_command
        elapsed_ms = (time.perf_counter() - cmd_started) * 1000.0
        result["telemetry"] = {
            "elapsed_ms": float(elapsed_ms),
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "daemon_command_count": int(self._command_count),
            "gpu_call_counter_before": int(gpu_calls_before),
            "gpu_call_counter_after": int(gpu_calls_after),
            "gpu_calls_this_command": int(gpu_calls_this_command),
            "gpu_calls_total": int(self._gpu_calls_total),
            "fallback_triggered": False,
        }
        return json.dumps(result, separators=(",", ":"), sort_keys=True)

    def serve_stdio(self) -> int:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "message": "k3d_daemon_started",
                    "mode": "stdio",
                    "timestamp": _now_iso(),
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )
        for line in sys.stdin:
            response = self._handle_line(line)
            print(response, flush=True)
            if self._shutdown_requested:
                break
        self._finalize_shutdown()
        return 0

    def serve_tcp(self) -> int:
        daemon = self

        class ReusableTCPServer(socketserver.TCPServer):
            allow_reuse_address = True

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:  # type: ignore[override]
                raw = self.rfile.readline().decode("utf-8", errors="replace")
                if not raw:
                    return
                out = daemon._handle_line(raw) + "\n"
                self.wfile.write(out.encode("utf-8"))

        with ReusableTCPServer((self.config.host, self.config.port), Handler) as server:
            server.timeout = float(self.config.tcp_poll_seconds)
            while not self._shutdown_requested:
                commands_before = int(self._command_count)
                server.handle_request()
                had_request = int(self._command_count) != commands_before
                self._advance_idle_clock(had_request=had_request)
        tick_driver = getattr(self, "_tick_driver", None)
        if tick_driver is not None:
            try:
                tick_driver.stop()
            except Exception:
                pass
        boundary = self._tablet_boundary
        if boundary is not None:
            try:
                if bool(boundary.live_session_status().get("active", False)):
                    boundary.close_live_session()
            except Exception:
                pass
        # SHUTDOWN already persists daemon-managed sleep state in handle_command().
        # For TCP mode we fast-exit here to avoid slow Knowledgeverse teardown keeping
        # the service process alive after the client has received the shutdown ack.
        os._exit(0)


def _make_ingest_id() -> str:
    """UUIDv7-style ingest ID: timestamp prefix + random hex suffix.

    IDs sort chronologically because the millisecond timestamp is the prefix.
    Pure Python — no uuid library required.
    """
    import os as _os
    ts_ms = int(time.time() * 1000)
    rand_hex = _os.urandom(8).hex()
    return f"ingest-{ts_ms:016x}-{rand_hex}"


def _validate_ingest_input(
    source_uri: Any,
    mime: Any,
    chunking: Any,
) -> dict[str, Any] | None:
    """Pure input gate for INGEST command. Returns error dict or None."""
    from knowledge3d.tablet.wine.ingest_wine import _validate_ingest_input as _wine_validate
    return _wine_validate(source_uri, mime, chunking)


_INPROCESS_DAEMON: Optional[K3DDaemon] = None


def _get_inprocess_daemon() -> K3DDaemon:
    """Lazy singleton for in-process CLI dispatch. Boots once per process; atexit persists."""
    global _INPROCESS_DAEMON
    if _INPROCESS_DAEMON is None:
        _INPROCESS_DAEMON = K3DDaemon(
            config=DaemonConfig(
                storage_root=default_storage_root(),
                require_ptx_query=True,
                eager_load_default_galaxies=True,
            )
        )
        atexit.register(_shutdown_inprocess_daemon)
    return _INPROCESS_DAEMON


def _shutdown_inprocess_daemon() -> None:
    """atexit hook — persist Knowledgeverse state on process exit. Never silently drops state."""
    global _INPROCESS_DAEMON
    if _INPROCESS_DAEMON is None:
        return
    try:
        _INPROCESS_DAEMON.kv.shutdown(persist=True, profile="service")
    finally:
        _INPROCESS_DAEMON = None


def handle_command_inprocess(payload: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a command to the in-process daemon singleton.

    The daemon is lazily constructed on first call and reused across calls.
    Knowledgeverse state is persisted at process exit via an atexit hook
    (see `_shutdown_inprocess_daemon`).

    This is the CLI entrypoint for `knowledge3d.tablet.chat` / `.ingest`.
    """
    daemon = _get_inprocess_daemon()
    try:
        result = daemon.handle_command(payload)
    except Exception as exc:
        result = {
            "status": "error",
            "error": "command_execution_failed",
            "exception_type": type(exc).__name__,
            "detail": str(exc),
        }
    if "telemetry" not in result:
        result["telemetry"] = {}
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run persistent K3D daemon command loop.")
    parser.add_argument("--storage-root", default=str(default_storage_root()), help="Knowledgeverse storage root.")
    parser.add_argument(
        "--mode",
        choices=("stdio", "tcp"),
        default="stdio",
        help="Command transport mode.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="TCP host when --mode=tcp.")
    parser.add_argument("--port", type=int, default=7777, help="TCP port when --mode=tcp.")
    parser.add_argument(
        "--idle-threshold-seconds",
        type=float,
        default=30.0,
        help="Idle time in seconds before a single sleep consolidation tick runs in TCP mode.",
    )
    parser.add_argument(
        "--sleep-sample-size",
        type=int,
        default=512,
        help="Maximum embedding rows sampled by a sleep consolidation tick.",
    )
    parser.add_argument(
        "--warm-gpu-runtime-on-boot",
        action="store_true",
        help="Bind the default galaxy runtime during daemon boot instead of on first query.",
    )
    parser.add_argument(
        "--allow-nonsovereign-query",
        action="store_true",
        help="Allow CPU query path for diagnostics (default is strict PTX query required).",
    )
    parser.add_argument(
        "--no-eager-load-default-galaxies",
        action="store_true",
        help="Disable eager default galaxy load at startup.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = DaemonConfig(
        storage_root=Path(args.storage_root),
        require_ptx_query=not bool(args.allow_nonsovereign_query),
        eager_load_default_galaxies=not bool(args.no_eager_load_default_galaxies or args.allow_nonsovereign_query),
        host=str(args.host),
        port=int(args.port),
        idle_threshold_seconds=float(args.idle_threshold_seconds),
        tcp_poll_seconds=0.2,
        sleep_sample_size=max(16, int(args.sleep_sample_size)),
        warm_gpu_runtime_on_boot=bool(args.warm_gpu_runtime_on_boot),
    )
    daemon = K3DDaemon(config=config)
    if args.mode == "tcp":
        return daemon.serve_tcp()
    return daemon.serve_stdio()


if __name__ == "__main__":
    raise SystemExit(main())
