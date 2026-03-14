"""Persistent K3D daemon entrypoint (game-style runtime).

The daemon keeps one Knowledgeverse + TRM instance alive and serves JSON
commands over stdio or TCP line protocol. This avoids one-shot script
orchestration and enforces a single-world process lifecycle.
"""

from __future__ import annotations

import argparse
import json
import os
import socketserver
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge
from knowledge3d.cranium.bridges.procedural_geometry_bridge import ProceduralGeometryBridge
from knowledge3d.cranium.bridges.procedural_material_bridge import ProceduralMaterialBridge

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
        self._drawing_bridge: ProceduralDrawingBridge | None = None
        self._geometry_bridge: ProceduralGeometryBridge | None = None
        self._material_bridge: ProceduralMaterialBridge | None = None
        self._sleep_cluster_refiner = None
        self._sleep_glyph_consolidator = None
        self._drawing_warmup: dict[str, Any] = {}
        self._geometry_warmup: dict[str, Any] = {}
        self._material_warmup: dict[str, Any] = {}
        self._boot_binding: dict[str, Any] = {}
        self._sleep_tick_count = 0
        self._sleep_tick_cursor = 0
        self._last_sleep_tick: dict[str, Any] = {}
        self._sleep_tick_history: list[dict[str, Any]] = []
        self._sleep_tick_history_max = 16
        self._pending_sleep_embedding_updates = 0
        self._idle_elapsed_seconds = 0.0
        self._sleep_tick_order: tuple[str, ...] = (
            "cluster_refiner",
            "glyph_consolidator",
            "memory_updater",
            "graph_crystallizer",
        )
        self._write_boot_status(stage="daemon_boot", progress=0.05, state="starting")

        os.environ["K3D_REQUIRE_PTX_QUERY"] = "true" if config.require_ptx_query else "false"

        self._write_boot_status(stage="knowledgeverse_load", progress=0.2, state="loading")
        self.kv = knowledgeverse or Knowledgeverse(
            storage_root=config.storage_root,
            eager_load_default_galaxies=config.eager_load_default_galaxies,
        )
        self.trm = self.kv.trm_navigator
        self._default_counts = self.kv.ensure_default_galaxies_loaded()
        if self.config.warm_gpu_runtime_on_boot:
            self._write_boot_status(stage="gpu_runtime_bind", progress=0.62, state="warming")
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
        self._warmup_boot_runtime()
        self._write_boot_status(
            stage="ready",
            progress=1.0,
            state="ready",
            extra={
                "drawing_warmup": dict(self._drawing_warmup),
                "geometry_warmup": dict(self._geometry_warmup),
                "material_warmup": dict(self._material_warmup),
                "gpu_binding": dict(self._boot_binding),
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

    def _warmup_boot_runtime(self) -> None:
        if os.environ.get("K3D_WARMUP_DRAWING", "1") != "1":
            self._drawing_warmup = {"status": "skipped", "reason": "K3D_WARMUP_DRAWING=0"}
        else:
            self._write_boot_status(stage="drawing_runtime_warmup", progress=0.72, state="warming")
            try:
                self._drawing_bridge = ProceduralDrawingBridge(matryoshka_dim=64)
                self._drawing_warmup = self._drawing_bridge.warmup_runtime()
                self._write_boot_status(
                    stage="drawing_runtime_warm",
                    progress=0.84,
                    state="warming",
                    extra={"drawing_warmup": dict(self._drawing_warmup)},
                )
            except Exception as exc:
                self._drawing_warmup = {
                    "status": "error",
                    "exception_type": type(exc).__name__,
                    "detail": str(exc),
                }
                self._write_boot_status(
                    stage="drawing_runtime_warmup_failed",
                    progress=0.84,
                    state="warning",
                    extra={"drawing_warmup": dict(self._drawing_warmup)},
                )

        if os.environ.get("K3D_WARMUP_GEOMETRY", "1") != "1":
            self._geometry_warmup = {"status": "skipped", "reason": "K3D_WARMUP_GEOMETRY=0"}
        else:
            self._write_boot_status(
                stage="geometry_runtime_warmup",
                progress=0.9,
                state="warming",
                extra={"drawing_warmup": dict(self._drawing_warmup)},
            )
            try:
                self._geometry_bridge = ProceduralGeometryBridge()
                self._geometry_warmup = self._geometry_bridge.warmup_runtime()
                self._write_boot_status(
                    stage="geometry_runtime_warm",
                    progress=0.96,
                    state="warming",
                    extra={
                        "drawing_warmup": dict(self._drawing_warmup),
                        "geometry_warmup": dict(self._geometry_warmup),
                    },
                )
            except Exception as exc:
                self._geometry_warmup = {
                    "status": "error",
                    "exception_type": type(exc).__name__,
                    "detail": str(exc),
                }
                self._write_boot_status(
                    stage="geometry_runtime_warmup_failed",
                    progress=0.96,
                    state="warning",
                    extra={
                        "drawing_warmup": dict(self._drawing_warmup),
                        "geometry_warmup": dict(self._geometry_warmup),
                    },
                )

        if os.environ.get("K3D_WARMUP_MATERIAL", "1") != "1":
            self._material_warmup = {"status": "skipped", "reason": "K3D_WARMUP_MATERIAL=0"}
            return

        self._write_boot_status(
            stage="material_runtime_warmup",
            progress=0.985,
            state="warming",
            extra={
                "drawing_warmup": dict(self._drawing_warmup),
                "geometry_warmup": dict(self._geometry_warmup),
            },
        )
        try:
            self._material_bridge = ProceduralMaterialBridge()
            self._material_warmup = self._material_bridge.warmup_runtime()
            self._write_boot_status(
                stage="material_runtime_warm",
                progress=0.995,
                state="warming",
                extra={
                    "drawing_warmup": dict(self._drawing_warmup),
                    "geometry_warmup": dict(self._geometry_warmup),
                    "material_warmup": dict(self._material_warmup),
                },
            )
        except Exception as exc:
            self._material_warmup = {
                "status": "error",
                "exception_type": type(exc).__name__,
                "detail": str(exc),
            }
            self._write_boot_status(
                stage="material_runtime_warmup_failed",
                progress=0.995,
                state="warning",
                extra={
                    "drawing_warmup": dict(self._drawing_warmup),
                    "geometry_warmup": dict(self._geometry_warmup),
                    "material_warmup": dict(self._material_warmup),
                },
            )

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

    def _warmup_gpu_runtime_binding(self) -> dict[str, Any]:
        try:
            binding = self.kv.bind_gpu_galaxy_runtime(galaxy_names=list(self.kv.DEFAULT_GALAXIES))
            return {
                "status": "ok",
                "entry_count": int(binding.get("entry_count", 0)),
                "buffer_bytes": int(binding.get("buffer_bytes", 0)),
                "galaxies": list(binding.get("galaxies", [])),
            }
        except Exception as exc:
            return {
                "status": "error",
                "exception_type": type(exc).__name__,
                "detail": str(exc),
            }

    def _binding_report(self) -> dict[str, Any]:
        binding = getattr(self.kv, "_gpu_galaxy_binding", None)
        if not isinstance(binding, dict):
            return {"status": "unbound"}
        return {
            "status": "ready",
            "entry_count": int(binding.get("entry_count", 0)),
            "buffer_bytes": int(binding.get("buffer_bytes", 0)),
            "galaxies": list(binding.get("galaxies", [])),
            "runtime_artifact_entries": int(binding.get("runtime_artifact_entries", 0)),
        }

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
            for name in self.kv.DEFAULT_GALAXIES
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
        return [str(name) for name in self.kv.DEFAULT_GALAXIES]

    def _looks_like_math_prompt(self, text: str) -> bool:
        prompt = str(text).strip().lower()
        if not prompt:
            return False
        has_digit = any(ch.isdigit() for ch in prompt)
        if has_digit and any(ch in prompt for ch in "+-*/=^"):
            return True
        math_markers = (
            "solve ",
            "calculate ",
            "compute ",
            "evaluate ",
            "factorial",
            "binomial",
            "derivative",
            "integral",
            "quadratic",
            "equation",
        )
        return has_digit and any(marker in prompt for marker in math_markers)

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
            galaxy_names=list(self.kv.DEFAULT_GALAXIES),
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
            galaxy_names=list(self.kv.DEFAULT_GALAXIES),
            limit=min(int(self.config.sleep_sample_size), 256),
        )
        if len(sample_rows) < 2:
            return {"status": "skipped", "reason": "insufficient_embeddings", "rows": len(sample_rows)}
        import numpy as np

        matrix = np.asarray([embedding for _, _, embedding in sample_rows], dtype=np.float32)
        teacher = np.mean(matrix, axis=0, keepdims=False).astype(np.float32)
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
        import numpy as np

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
                neighbor_rows.append(
                    np.mean(np.asarray(neighbors, dtype=np.float32), axis=0).astype(np.float32).tolist()
                )
            else:
                neighbor_rows.append(list(node_rows[-1]))
        crystallized = crystallizer.crystallize_list(node_rows, neighbor_rows, ema_rate=0.985)
        return {
            "status": "ok",
            "rows": len(sample_indexes),
            "graph_signature": str(getattr(graph, "signature", "")),
            "crystallized_rows": len(crystallized),
        }

    def _run_sleep_consolidation_tick(self) -> dict[str, Any]:
        tick_name = self._sleep_tick_order[self._sleep_tick_cursor % len(self._sleep_tick_order)]
        handlers = {
            "cluster_refiner": self._sleep_cluster_tick,
            "glyph_consolidator": self._sleep_glyph_tick,
            "memory_updater": self._sleep_memory_update_tick,
            "graph_crystallizer": self._sleep_graph_crystallization_tick,
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
        for galaxy_name in self.kv.DEFAULT_GALAXIES:
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
            "drawing_warmup": dict(self._drawing_warmup),
            "geometry_warmup": dict(self._geometry_warmup),
            "material_warmup": dict(self._material_warmup),
            "gpu_binding": self._binding_report(),
            "semantic_csr_graph": self._semantic_graph_report(),
            "idle_threshold_seconds": float(self.config.idle_threshold_seconds),
            "sleep_tick_count": int(self._sleep_tick_count),
            "last_sleep_tick": dict(self._last_sleep_tick),
            "sleep_tick_history": [dict(item) for item in self._sleep_tick_history],
            "boot_status_paths": [str(path) for path in self._boot_status_paths],
        }

    def _gpu_call_snapshot(self) -> int:
        if ModularRPNEngine is None:
            return 0
        try:
            return int(ModularRPNEngine.get_global_gpu_call_count())
        except Exception:
            return 0

    def _collect_parse_bundle(
        self,
        query: str,
        *,
        specialist: str,
        galaxy_names: list[str],
        domain_hint: str | None = None,
    ) -> dict[str, Any]:
        navigator = getattr(self.kv, "navigator_specialist", None)
        if navigator is None:
            navigator = getattr(self.trm, "navigator_specialist", None)
        if navigator is None:
            return {}
        try:
            routes = navigator.plan_routes(
                query=query,
                specialist=specialist,
                galaxy_names=galaxy_names,
                domain_hint=domain_hint,
                use_forward_backward=True,
            )
        except Exception:
            return {}
        bundle: dict[str, Any] = {"route_plan": routes}
        for key in ("forward_parse", "backward_parse", "fusion_parse"):
            for route in routes:
                if not isinstance(route, dict):
                    continue
                value = route.get(key)
                if isinstance(value, dict):
                        bundle[key] = value
                        break
        return bundle

    def _dispatch_lhe_task(self, *, route: dict[str, Any], task: dict[str, Any], use_enriched: bool) -> dict[str, Any]:
        response = self.kv.execute_task(
            task=task,
            route=route,
            specialist=str(route.get("specialist", "auto")),
            domain_hint=task.get("domain_hint"),
            use_enriched=use_enriched,
        )
        if isinstance(response, dict):
            response.setdefault("runtime", "knowledgeverse_gpu_query")
        return response

    def _dispatch_task(self, *, route: dict[str, Any], task: dict[str, Any], use_enriched: bool) -> dict[str, Any]:
        specialist = str(route.get("specialist", "grammar")).lower()
        task_type = str(task.get("type", "")).upper()
        all_galaxies = self._all_default_galaxies()

        if task_type == "LHE_TASK":
            return self._dispatch_lhe_task(route=route, task=task, use_enriched=use_enriched)

        if specialist == "visual":
            if task_type != "ARC_TASK":
                return {"status": "not_implemented", "reason": "visual_specialist_expected_arc_task"}
            if not hasattr(self.kv, "execute_task"):
                return {"status": "error", "error": "knowledgeverse_missing_execute_task"}
            arc_route = {
                "specialist": "visual",
                "domain_hint": str(route.get("domain") or route.get("domain_hint") or "visual"),
                "galaxy_names": list(all_galaxies),
            }
            solved = self.kv.execute_task(
                task=task,
                route=arc_route,
                specialist="visual",
                domain_hint="visual",
                use_enriched=use_enriched,
            )
            output_grid = solved.get("output_grid")
            response = {
                "status": "ok" if str(solved.get("status", "")).lower() == "ok" and output_grid is not None else "error",
                "task_type": "ARC_TASK",
                "task_id": task.get("task_id"),
                "program_type": str(solved.get("program_type") or "knowledgeverse_gpu_query"),
                "output_grid": output_grid,
                "reasoning_trace": list(solved.get("reasoning_trace", solved.get("thinking_trace", []))),
                "thinking_trace": list(solved.get("thinking_trace", [])),
                "thinking_xml": solved.get("thinking_xml"),
                "solver": solved.get("solver", "knowledgeverse_gpu_query"),
                "patterns_used": int(solved.get("patterns_used", 1 if output_grid is not None else 0)),
                "generated_pattern_count": int(solved.get("generated_pattern_count", 0)),
                "score": float(solved.get("score", 1.0 if output_grid is not None else 0.0)),
                "fuzzy_score": float(solved.get("fuzzy_score", 1.0 if output_grid is not None else 0.0)),
                "exact_match": bool(output_grid == task.get("expected_output")) if task.get("expected_output") is not None else False,
                "gpu_execution": bool(solved.get("gpu_execution", False)),
                "runtime": solved.get("runtime", "knowledgeverse_gpu_query"),
                "program_id": solved.get("program_id"),
                "route": solved.get("route", arc_route),
            }
            return response

        if specialist == "math":
            question = str(task.get("question", "") or task.get("query", "")).strip()
            if not question:
                return {"status": "error", "error": "math_task_missing_question"}
            math_route = {
                "specialist": "math",
                "domain_hint": str(route.get("domain") or route.get("domain_hint") or "math"),
                "galaxy_names": list(all_galaxies),
            }
            solved = self.kv.execute_task(
                task={
                    **dict(task),
                    "type": task_type or "MATH_TASK",
                    "query": question,
                    "question": question,
                },
                route=math_route,
                specialist="math",
                domain_hint="math",
                use_enriched=use_enriched,
            )
            response = {
                **solved,
                "task_type": task_type or "MATH_TASK",
                "task_id": task.get("task_id"),
            }
            response["status"] = "success" if str(solved.get("status", "")).lower() == "ok" else "error"
            return response

        if specialist in {"chat", "grammar", "any"} or task_type == "MMLU_TASK":
            messages = task.get("messages")
            if not isinstance(messages, list):
                prompt = str(task.get("prompt", "") or task.get("query", "")).strip()
                if not prompt:
                    return {"status": "error", "error": "chat_task_missing_prompt"}
                messages = [{"role": "user", "content": prompt}]
            chat_prompt = str(task.get("prompt", "") or task.get("query", "")).strip()
            if not chat_prompt:
                for message in reversed(messages):
                    if not isinstance(message, dict):
                        continue
                    if str(message.get("role", "")).strip().lower() != "user":
                        continue
                    chat_prompt = str(message.get("content", "")).strip()
                    if chat_prompt:
                        break
            if not chat_prompt:
                return {"status": "error", "error": "chat_task_missing_prompt"}
            if task_type != "MMLU_TASK" and self._looks_like_math_prompt(chat_prompt):
                math_route = {
                    "specialist": "math",
                    "domain_hint": "math",
                    "galaxy_names": list(all_galaxies),
                }
                solved = self.kv.execute_task(
                    task={
                        **dict(task),
                        "type": "MATH_TASK",
                        "query": chat_prompt,
                        "question": chat_prompt,
                    },
                    route=math_route,
                    specialist="math",
                    domain_hint="math",
                    use_enriched=use_enriched,
                )
                return {
                    **solved,
                    "task_type": "MATH_TASK",
                    "task_id": task.get("task_id"),
                }
            chat_route = {
                "specialist": "chat",
                "domain_hint": str(route.get("domain") or route.get("domain_hint") or "general"),
                "galaxy_names": list(all_galaxies),
            }
            solved = self.kv.execute_task(
                task={
                    **dict(task),
                "type": task_type or "CHAT_TASK",
                "prompt": chat_prompt,
                "query": chat_prompt,
                "messages": list(messages),
            },
                route=chat_route,
                specialist="chat",
                domain_hint=str(route.get("domain") or route.get("domain_hint") or "general"),
                use_enriched=use_enriched,
            )
            return {
                **solved,
                "task_type": task_type or "CHAT_TASK",
                "task_id": task.get("task_id"),
            }

        return {
            "status": "not_implemented",
            "reason": f"specialist_dispatch_not_implemented:{specialist}",
            "task_type": task_type,
        }

    def handle_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._command_count += 1
        cmd = str(payload.get("command", "")).strip().upper()
        if not cmd:
            return {"status": "error", "error": "missing_command"}

        if cmd in {"PING", "STATUS"}:
            return self.status_payload()

        if cmd == "VRAM_REPORT":
            return self._vram_report_payload()

        if cmd == "SHUTDOWN":
            persist_result = self._persist_sleep_state()
            self._shutdown_requested = True
            return {
                "status": "ok",
                "message": "shutdown_requested",
                "timestamp": _now_iso(),
                "sleep_persistence": persist_result,
            }

        if cmd == "ROUTE":
            task = payload.get("task")
            if task is not None and not isinstance(task, dict):
                return {"status": "error", "error": "task_must_be_object"}
            task_obj = task if isinstance(task, dict) else None
            task_type = str((task_obj or {}).get("type", "")).upper()
            query = str(
                payload.get("query", "")
                or (task_obj or {}).get("query", "")
                or (task_obj or {}).get("question", "")
                or (task_obj or {}).get("prompt", "")
                or (task_obj or {}).get("type", "")
            ).strip()
            if not query:
                return {"status": "error", "error": "missing_query_or_task"}
            use_enriched = bool(payload.get("use_enriched", True))
            all_galaxies = self._all_default_galaxies()
            if task_type == "ARC_TASK":
                route = {
                    "specialist": "visual",
                    "domain": str(payload.get("domain_hint") or (task_obj or {}).get("domain_hint") or "visual"),
                    "reason": "knowledgeverse_gpu_query",
                    "galaxy_names": list(all_galaxies),
                }
            elif task_type == "LHE_TASK":
                route = {
                    "specialist": str(payload.get("specialist", "auto") or "auto"),
                    "domain": str(payload.get("domain_hint") or (task_obj or {}).get("domain_hint") or ""),
                    "reason": "knowledgeverse_gpu_query",
                    "galaxy_names": list(all_galaxies),
                }
            elif task_type == "MATH_TASK":
                route = {
                    "specialist": "math",
                    "domain": str(payload.get("domain_hint") or (task_obj or {}).get("domain_hint") or "math"),
                    "reason": "knowledgeverse_gpu_query",
                    "galaxy_names": list(all_galaxies),
                }
            elif task_type in {"CHAT_TASK", "GENERAL_TASK", "GRAMMAR_TASK"}:
                specialist = "math" if self._looks_like_math_prompt(query) else "chat"
                domain = "math" if specialist == "math" else str(
                    payload.get("domain_hint") or (task_obj or {}).get("domain_hint") or "general"
                )
                route = {
                    "specialist": specialist,
                    "domain": domain,
                    "reason": "knowledgeverse_gpu_query",
                    "galaxy_names": list(all_galaxies),
                }
            elif task_type == "MMLU_TASK":
                route = {
                    "specialist": "chat",
                    "domain": str(payload.get("domain_hint") or (task_obj or {}).get("domain_hint") or "general"),
                    "reason": "knowledgeverse_gpu_query",
                    "galaxy_names": list(all_galaxies),
                }
            else:
                route = self.trm.route(
                    query=query,
                    specialist=str(payload.get("specialist", "auto")),
                    domain_hint=payload.get("domain_hint") or (task_obj or {}).get("domain_hint"),
                    galaxy_names=payload.get("galaxies") or (task_obj or {}).get("galaxies"),
                )
            response: dict[str, Any] = {"status": "ok", "route": route}
            if task_obj is not None:
                response["task_result"] = self._dispatch_task(
                    route=route,
                    task=task_obj,
                    use_enriched=use_enriched,
                )
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
                    "type": "MATH_TASK",
                    "query": question,
                    "question": question,
                },
                route={
                    "specialist": "math",
                    "domain_hint": str(payload.get("domain_hint") or "math"),
                    "galaxy_names": self._all_default_galaxies(),
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
            return {
                "status": "ok",
                "result": solved.get("result"),
                "program_id": solved.get("program_id"),
                "runtime": solved.get("runtime"),
                "gpu_execution": bool(solved.get("gpu_execution", False)),
            }

        if cmd == "CHAT":
            messages = payload.get("messages")
            if not isinstance(messages, list):
                prompt = str(payload.get("prompt", "") or payload.get("query", "")).strip()
                if not prompt:
                    return {"status": "error", "error": "missing_messages_or_prompt"}
                messages = [{"role": "user", "content": prompt}]
            prompt = str(payload.get("prompt", "") or payload.get("query", "")).strip()
            if not prompt:
                for message in reversed(messages):
                    if not isinstance(message, dict):
                        continue
                    if str(message.get("role", "")).strip().lower() != "user":
                        continue
                    prompt = str(message.get("content", "")).strip()
                    if prompt:
                        break
            if not prompt:
                return {"status": "error", "error": "missing_messages_or_prompt"}
            if self._looks_like_math_prompt(prompt):
                solved = self.kv.execute_task(
                    task={
                        "type": "MATH_TASK",
                        "question": prompt,
                        "query": prompt,
                    },
                    route={
                        "specialist": "math",
                        "domain_hint": "math",
                        "galaxy_names": self._all_default_galaxies(),
                    },
                    specialist="math",
                    domain_hint="math",
                    use_enriched=bool(payload.get("use_enriched", True)),
                )
                if str(solved.get("status", "")).lower() != "ok":
                    return {"status": "error", "error": "knowledgeverse_math_query_failed", "detail": solved}
                return {
                    "status": "ok",
                    "response": solved.get("response", solved.get("result", solved.get("answer", ""))),
                    "runtime": solved.get("runtime"),
                    "gpu_execution": bool(solved.get("gpu_execution", False)),
                    "program_id": solved.get("program_id"),
                    "task_result": solved,
                }
            solved = self.kv.execute_task(
                task={
                    "type": "CHAT_TASK",
                    "prompt": prompt,
                    "query": prompt,
                    "messages": list(messages),
                },
                route={
                    "specialist": "chat",
                    "domain_hint": str(payload.get("domain_hint") or "general"),
                    "galaxy_names": self._all_default_galaxies(),
                },
                specialist="chat",
                domain_hint=str(payload.get("domain_hint") or "general"),
                use_enriched=bool(payload.get("use_enriched", True)),
            )
            if str(solved.get("status", "")).lower() != "ok":
                return {"status": "error", "error": "knowledgeverse_chat_query_failed", "detail": solved}
            return {
                "status": "ok",
                "response": solved.get("response", solved.get("answer", "")),
                "runtime": solved.get("runtime"),
                "gpu_execution": bool(solved.get("gpu_execution", False)),
                "program_id": solved.get("program_id"),
                "task_result": solved,
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
        gpu_calls_this_command = max(0, int(gpu_calls_after - gpu_calls_before))
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
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run persistent K3D daemon command loop.")
    parser.add_argument("--storage-root", default="../Knowledge3D.local", help="Knowledgeverse storage root.")
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
        eager_load_default_galaxies=not bool(args.no_eager_load_default_galaxies),
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
