"""Thin runtime bridge into the sovereign GPU hot path."""

from __future__ import annotations

import concurrent.futures
import ctypes
import json
import multiprocessing as mp
import os
import pickle
import re
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge3d.cranium.sovereign import loader

from . import route_contract
from .galaxy_vram_table import (
    GalaxyVRAMTable,
    ROLE_ANTI_PATTERN,
    ROLE_ANSWER,
    ROLE_EXECUTOR,
    ROLE_ROUTER,
    ROLE_VALIDATOR,
    STAR_FLAG_ACTIVE,
    STAR_FLAG_LEARNABLE,
    _fnv1a32,
    encode_route_policy,
)
from .gpu_task_dispatch import GPUTaskDispatch
from .knowledge_gap_inventory import (
    curated_math_question_coverage_packets,
    curated_math_question_knowledge_gap_inventory,
)
from .lesson_vram_ring import SemanticLessonGPU, VRAMLessonRing
from .persistent_brain import PersistentBrainState
from .semantic_gravity_gpu import SemanticGravityGPU
from .star_materializer_bridge import (
    BUILD_REF_HASH_BYTES,
    CATALOG_INPUT_ENTRY_BYTES,
    FEED_SOURCE_REF_BYTES,
    FINALIZED_CATALOG_INPUT_ENTRY_BYTES,
    REF_TUPLE_BYTES,
    RAW_CATALOG_INPUT_ENTRY_BYTES,
    StarMaterializerBridge,
)
from .vram_task_buffer import VRAMTaskBuffer

RAW_CATALOG_INPUT_STRUCT = struct.Struct("<16f6Ii5f7IQ2f20x")
REF_TUPLE_STRUCT = struct.Struct("<4I")
BUILD_REF_HASH_STRUCT = struct.Struct("<IIQ")
FEED_SOURCE_REF_STRUCT = struct.Struct("<IIQI4x")
GPU_BUILD_CHUNK_SIZE = 4096
GPU_BUILD_PROGRESS_EVERY = 8
BUILD_BACKEND = "gpu_build_feed_v2"
BUILD_FEED_VERSION = 3
FEED_SOURCE_VERSION = 2
ROLE_TYPE_BY_KEY = {
    "router_refs": 0,
    "executor_refs": 1,
    "validator_refs": 2,
    "anti_pattern_refs": 3,
}
ROLE_ID_BY_NAME = {
    "router": ROLE_ROUTER,
    "executor": ROLE_EXECUTOR,
    "validator": ROLE_VALIDATOR,
    "answer": ROLE_ANSWER,
    "anti_pattern": ROLE_ANTI_PATTERN,
}
MEANING_FAMILY_ROUTE_MINIMA = {
    "GRAMMAR": {"routers": 1, "executors": 4, "validators": 2, "anti_patterns": 2},
    "GENERAL": {"routers": 1, "executors": 3, "validators": 2, "anti_patterns": 2},
    "CHAT": {"routers": 1, "executors": 2, "validators": 2, "anti_patterns": 2},
    "QUESTION": {"routers": 1, "executors": 2, "validators": 2, "anti_patterns": 2},
    "MATH": {"routers": 1, "executors": 5, "validators": 3, "anti_patterns": 2},
    "GAME_2D": {"routers": 1, "executors": 4, "validators": 2, "anti_patterns": 3},
}
MEANING_ROUTE_CLOSURE_MINIMA = {
    "GAME_2D": {"surface_bridges": 1, "routers": 1, "executors": 4, "materializers": 1, "validators": 2, "anti_patterns": 3},
    "MATH": {"surface_bridges": 1, "routers": 1, "executors": 6, "materializers": 1, "validators": 3, "anti_patterns": 3},
    "QUESTION": {"surface_bridges": 1, "routers": 1, "executors": 4, "materializers": 1, "validators": 3, "anti_patterns": 3},
    "GENERAL": {"surface_bridges": 1, "routers": 1, "executors": 4, "materializers": 1, "validators": 3, "anti_patterns": 3},
    "GRAMMAR": {"surface_bridges": 1, "routers": 1, "executors": 5, "materializers": 1, "validators": 2, "anti_patterns": 2},
}

EXPLICIT_POLARITY_MASK = 0x01
EXPLICIT_FOCUS_MASK = 0x02
EXPLICIT_MASS_MASK = 0x04
EXPLICIT_ATTRACTIVE_MASK = 0x08
EXPLICIT_REPULSIVE_MASK = 0x10
ROUTE_FAMILY_FLAG_SHIFT = 8
ROUTE_FAMILY_FLAG_MASK = 0xFF << ROUTE_FAMILY_FLAG_SHIFT
DEFAULT_FEED_SOURCE_CHUNK_SIZE = 4096
DEFAULT_FEED_SOURCE_MAX_WORKERS = 8
_FEED_SOURCE_PARALLEL_STATE: dict[str, Any] = {}


def _fnv1a64(text: str) -> int:
    value = 14695981039346656037
    for byte in str(text or "").encode("utf-8"):
        value ^= int(byte)
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(value)


def _pad32(values: list[float] | tuple[float, ...] | Any) -> list[float]:
    row = [float(value) for value in list(values or [])[:32]]
    if len(row) < 32:
        row.extend([0.0] * (32 - len(row)))
    return row[:32]


def _stable_hash(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return _fnv1a64(value)
    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        payload = str(value)
    return _fnv1a64(payload)


def _route_family_id(route_family: Any) -> int:
    token = str(route_family or "").strip()
    if not token:
        return 0
    return int(VRAMTaskBuffer.task_type_id(token))


def _encode_runtime_flags(flags: int, route_family: Any) -> int:
    base_flags = int(flags or 0) & ~ROUTE_FAMILY_FLAG_MASK
    family_id = _route_family_id(route_family)
    return int(base_flags | ((family_id & 0xFF) << ROUTE_FAMILY_FLAG_SHIFT))


def _looks_like_route_label(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    canonical = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if canonical.startswith("anti_pattern_"):
        return True
    if any(
        canonical == token or canonical.endswith(f"_{token}")
        for token in ("router", "executor", "validator", "materializer", "anti_pattern")
    ):
        return True
    return text in {"router", "executor", "validator", "materializer", "anti_pattern"}


def _is_surface_bridge_star(star: dict[str, Any]) -> bool:
    route_policy = dict(star.get("route_policy") or {})
    return bool(route_policy.get("surface_bridge"))


def _is_materializer_star(star: dict[str, Any]) -> bool:
    route_policy = dict(star.get("route_policy") or {})
    return any(
        bool(route_policy.get(key))
        for key in ("materialize_answer", "materialize_choice", "materialize_action", "materialize_grid")
    )


def _materialize_answer_text(
    *,
    options: list[Any],
    answer_index: int,
    winner_star: dict[str, Any] | None,
) -> str:
    if options and 0 <= int(answer_index) < len(options):
        return str(options[int(answer_index)])
    if not isinstance(winner_star, dict):
        return ""
    metadata = winner_star.get("metadata") if isinstance(winner_star.get("metadata"), dict) else {}
    for key in ("answer_text", "answer", "response", "resolved_answer", "boxed_answer"):
        value = winner_star.get(key)
        if value is None and isinstance(metadata, dict):
            value = metadata.get(key)
        text = str(value or "").strip()
        if text and not _looks_like_route_label(text):
            return text
    for key in ("id", "name"):
        text = str(winner_star.get(key) or "").strip()
        if text and not _looks_like_route_label(text):
            return text
    return ""


def _trace_role_names(role_ids: list[int]) -> list[str]:
    names = {
        ROLE_ROUTER: "router",
        ROLE_EXECUTOR: "executor",
        ROLE_VALIDATOR: "validator",
        ROLE_ANSWER: "answer",
        ROLE_ANTI_PATTERN: "anti_pattern",
    }
    values: list[str] = []
    for role_id in list(role_ids or []):
        values.append(str(names.get(int(role_id), "unknown")))
    return values


def _write_pickle_file(path: Path, payload: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _feed_source_parallel_chunk(task: tuple[int, int]) -> dict[str, Any]:
    runtime = _FEED_SOURCE_PARALLEL_STATE.get("runtime")
    catalog = _FEED_SOURCE_PARALLEL_STATE.get("catalog")
    if runtime is None or not isinstance(catalog, list):
        raise RuntimeError("sovereign_feed_source_parallel_state_missing")
    start, count = task
    return runtime._compile_feed_source_chunk(
        catalog,
        start=int(start),
        count=int(count),
    )


@dataclass
class SovereignRouteTrace:
    router_index: int = -1
    executor_index: int = -1
    validator_index: int = -1
    winner_index: int = -1
    winner_role_id: int = 0
    route_depth: int = 0
    anti_pattern_signal: int = 0
    route_budget_used: int = 0
    route_budget_min: int = 0
    recursion_depth_used: int = 0
    route_trace_star_indices: list[int] | None = None
    route_trace_role_ids: list[int] | None = None


class SovereignHotPath:
    """Own the device table, persistent brain, lesson ring, and dispatch kernel."""

    ARTIFACT_VERSION = 6

    def __init__(self, knowledgeverse: Any) -> None:
        self.knowledgeverse = knowledgeverse
        self.task_buffer = VRAMTaskBuffer(max_tasks=2048)
        self.brain = PersistentBrainState()
        self.star_table = GalaxyVRAMTable(max_stars=350_000)
        self.dispatch = GPUTaskDispatch()
        self.materializer = StarMaterializerBridge()
        self.lesson_ring = VRAMLessonRing(capacity=262_144)
        self.lesson_gpu = SemanticLessonGPU()
        self.gravity_gpu = SemanticGravityGPU()
        self._host_stars: list[dict[str, Any]] = []
        self._catalog_signature = ""
        self._build_feed_signature = ""
        self._feed_source_signature = ""
        self._last_build_feed_manifest: dict[str, Any] = {}
        jarvis_state = getattr(self.knowledgeverse, "_jarvis_state", {})
        if isinstance(jarvis_state, dict):
            self.lesson_ring.load_stats(jarvis_state.get("sovereign_learning"))
        self._last_load_summary: dict[str, Any] = {}
        self._last_runtime_manifest: dict[str, Any] = self._read_runtime_manifest()

    def close(self, *, profile: str = "service") -> dict[str, Any]:
        normalized_profile = str(profile or "service").strip().lower() or "service"
        if normalized_profile == "benchmark":
            return {
                "status": "fast_exit",
                "profile": normalized_profile,
                "closed": False,
            }
        self.task_buffer.close()
        self.brain.close()
        self.star_table.close()
        self.lesson_ring.close()
        return {
            "status": "completed",
            "profile": normalized_profile,
            "closed": True,
        }

    def _current_signature(self, catalog: list[dict[str, Any]]) -> str:
        sample = [
            f"{entry.get('galaxy','')}|{entry.get('id','')}|{entry.get('index','')}"
            for entry in catalog[:8192]
        ]
        payload = "|".join(sample) + f"|count={len(catalog)}"
        return f"{len(catalog)}:{zlib.crc32(payload.encode('utf-8')) & 0xFFFFFFFF:08x}"

    def _artifact_bundle_path(self) -> Path:
        return self.knowledgeverse.storage_root / "checkpoints" / "sovereign_runtime_bundle.pkl"

    def _artifact_manifest_path(self) -> Path:
        return self.knowledgeverse.storage_root / "checkpoints" / "sovereign_runtime_manifest.json"

    def _meaning_family_route_audit_path(self) -> Path:
        return self.knowledgeverse.storage_root / "checkpoints" / "meaning_family_route_audit.json"

    def _meaning_route_closure_audit_path(self) -> Path:
        return self.knowledgeverse.storage_root / "checkpoints" / "meaning_route_closure_audit.json"

    def _meaning_knowledge_coverage_audit_path(self) -> Path:
        return self.knowledgeverse.storage_root / "checkpoints" / "meaning_knowledge_coverage_audit.json"

    def _gpu_cache_dir(self) -> Path:
        return self.knowledgeverse.storage_root / "gpu_cache"

    def _build_feed_paths(self, signature: str) -> dict[str, Path]:
        cache_dir = self._gpu_cache_dir()
        return {
            "rows": cache_dir / f"build_rows_{signature}.bin",
            "ref_hashes": cache_dir / f"build_ref_hashes_{signature}.bin",
            "host_stars": cache_dir / f"build_host_stars_{signature}.pkl",
            "manifest": cache_dir / f"build_manifest_{signature}.json",
        }

    def _feed_source_paths(self, signature: str) -> dict[str, Path]:
        cache_dir = self._gpu_cache_dir()
        return {
            "rows": cache_dir / f"feed_source_rows_{signature}.bin",
            "refs": cache_dir / f"feed_source_refs_{signature}.bin",
            "host_stars": cache_dir / f"feed_source_host_stars_{signature}.pkl",
            "manifest": cache_dir / f"feed_source_manifest_{signature}.json",
        }

    def _expected_build_feed_signature(self) -> tuple[str, list[str]]:
        galaxy_names = list(self.knowledgeverse._discover_live_galaxy_names())
        signature = str(self.knowledgeverse._gpu_flat_cache_signature(galaxy_names))
        return signature, galaxy_names

    @staticmethod
    def _finalize_component_refs(stars: list[dict[str, Any]]) -> list[dict[str, Any]]:
        finalized: list[dict[str, Any]] = []
        for star in stars:
            row = dict(star)
            component_refs: list[int] = []
            for key in ROLE_TYPE_BY_KEY:
                refs = [int(value) for value in list(row.get(key) or []) if int(value) >= 0]
                row[key] = refs
                for ref_index in refs:
                    if ref_index not in component_refs:
                        component_refs.append(ref_index)
            row["component_refs"] = component_refs
            finalized.append(row)
        return finalized

    def _prune_stale_build_feed_cache(self, active_signature: str) -> dict[str, list[str] | int]:
        cache_dir = self._gpu_cache_dir()
        removed: list[str] = []
        if not cache_dir.exists():
            return {
                "removed": removed,
                "removed_count": 0,
            }
        prefixes = (
            "build_rows_",
            "build_ref_hashes_",
            "build_host_stars_",
            "build_manifest_",
        )
        for path in cache_dir.iterdir():
            if not path.is_file():
                continue
            if not any(path.name.startswith(prefix) for prefix in prefixes):
                continue
            if active_signature in path.name:
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            removed.append(str(path))
        return {
            "removed": removed,
            "removed_count": int(len(removed)),
        }

    def _prune_stale_feed_source_cache(self, active_signature: str) -> dict[str, list[str] | int]:
        cache_dir = self._gpu_cache_dir()
        removed: list[str] = []
        if not cache_dir.exists():
            return {
                "removed": removed,
                "removed_count": 0,
            }
        prefixes = (
            "feed_source_rows_",
            "feed_source_refs_",
            "feed_source_host_stars_",
            "feed_source_manifest_",
        )
        for path in cache_dir.iterdir():
            if not path.is_file():
                continue
            if not any(path.name.startswith(prefix) for prefix in prefixes):
                continue
            if active_signature in path.name:
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            removed.append(str(path))
        return {
            "removed": removed,
            "removed_count": int(len(removed)),
        }

    def _cached_feed_source_summary(self, signature: str) -> dict[str, Any] | None:
        paths = self._feed_source_paths(signature)
        manifest_path = paths["manifest"]
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(manifest, dict):
            return None
        if int(manifest.get("feed_source_version") or 0) != int(FEED_SOURCE_VERSION):
            return None
        if str(manifest.get("feed_source_signature") or "").strip() != str(signature):
            return None
        expected_house = self._expected_house_signature_base()
        expected_default = str(
            self.knowledgeverse._house_state_summary.get("default_knowledge_signature") or ""
        ).strip()
        if expected_house and str(manifest.get("house_signature_base") or "").strip() != expected_house:
            return None
        if expected_default and str(manifest.get("default_knowledge_signature") or "").strip() != expected_default:
            return None
        for key, path in paths.items():
            if key == "manifest":
                continue
            if not path.exists():
                return None
        summary = {
            "status": "ready",
            "mode": "feed_source_cached",
            "build_backend": BUILD_BACKEND,
            "feed_source_signature": str(signature),
            "catalog_signature": str(manifest.get("catalog_signature") or ""),
            "star_count": int(manifest.get("star_count") or 0),
            "forward_ref_count": int(manifest.get("forward_ref_count") or 0),
            "elapsed_s": 0.0,
            "feed_source_extract_s": float(manifest.get("feed_source_extract_s") or 0.0),
            "feed_source_audit_s": float(manifest.get("feed_source_audit_s") or 0.0),
            "feed_source_write_s": float(manifest.get("feed_source_write_s") or 0.0),
            "feed_source_parallel": bool(manifest.get("feed_source_parallel", False)),
            "feed_source_worker_count": int(manifest.get("feed_source_worker_count") or 0),
            "feed_source_chunk_size": int(manifest.get("feed_source_chunk_size") or 0),
            "prune_summary": {"removed": [], "removed_count": 0},
        }
        for key in (
            "route_valid_count",
            "route_neutral_count",
            "route_broken_count",
            "route_family_health",
        ):
            if key in manifest:
                summary[key] = manifest[key]
        for key, value in self.materializer.ptx_signatures().items():
            summary[key] = manifest.get(key, value)
        return summary

    def _load_feed_source(self, signature: str) -> dict[str, Any]:
        paths = self._feed_source_paths(signature)
        manifest_path = paths["manifest"]
        if not manifest_path.exists():
            raise RuntimeError(
                "sovereign_feed_source_missing:"
                "run scripts/rebuild_sovereign_artifact.py --refresh-feed-source"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"sovereign_feed_source_manifest_invalid:{type(exc).__name__}:{exc}") from exc
        if not isinstance(manifest, dict):
            raise RuntimeError("sovereign_feed_source_manifest_invalid:payload")
        if int(manifest.get("feed_source_version") or 0) != int(FEED_SOURCE_VERSION):
            raise RuntimeError(
                "sovereign_feed_source_stale:"
                f"version={int(manifest.get('feed_source_version') or 0)} expected={int(FEED_SOURCE_VERSION)}"
            )
        if str(manifest.get("feed_source_signature") or "").strip() != str(signature):
            raise RuntimeError(
                "sovereign_feed_source_signature_mismatch:"
                f"expected={signature} got={str(manifest.get('feed_source_signature') or '')}"
            )
        expected_house = self._expected_house_signature_base()
        expected_default = str(
            self.knowledgeverse._house_state_summary.get("default_knowledge_signature") or ""
        ).strip()
        if expected_house and str(manifest.get("house_signature_base") or "").strip() != expected_house:
            raise RuntimeError(
                "sovereign_feed_source_house_signature_mismatch:"
                f"expected={expected_house} got={str(manifest.get('house_signature_base') or '')}"
            )
        if expected_default and str(manifest.get("default_knowledge_signature") or "").strip() != expected_default:
            raise RuntimeError(
                "sovereign_feed_source_default_signature_mismatch:"
                f"expected={expected_default} got={str(manifest.get('default_knowledge_signature') or '')}"
            )
        for key, path in paths.items():
            if key == "manifest":
                continue
            if not path.exists():
                raise RuntimeError(f"sovereign_feed_source_missing_component:{key}:{path}")
        try:
            with paths["host_stars"].open("rb") as handle:
                host_stars = pickle.load(handle)
        except Exception as exc:
            raise RuntimeError(f"sovereign_feed_source_host_stars_invalid:{type(exc).__name__}:{exc}") from exc
        if not isinstance(host_stars, list):
            raise RuntimeError("sovereign_feed_source_host_stars_invalid:payload")
        return {
            "signature": str(signature),
            "manifest": dict(manifest),
            "paths": paths,
            "host_stars": [dict(star) for star in host_stars if isinstance(star, dict)],
        }

    def _load_build_feed(self, signature: str) -> dict[str, Any]:
        paths = self._build_feed_paths(signature)
        manifest_path = paths["manifest"]
        if not manifest_path.exists():
            raise RuntimeError(
                "sovereign_build_feed_missing:"
                "run scripts/rebuild_sovereign_artifact.py --refresh-build-feed --force-rebuild"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"sovereign_build_feed_manifest_invalid:{type(exc).__name__}:{exc}") from exc
        if not isinstance(manifest, dict):
            raise RuntimeError("sovereign_build_feed_manifest_invalid:payload")
        if int(manifest.get("build_feed_version") or 0) != int(BUILD_FEED_VERSION):
            raise RuntimeError(
                "sovereign_build_feed_stale:"
                f"version={int(manifest.get('build_feed_version') or 0)} expected={int(BUILD_FEED_VERSION)}"
            )
        if str(manifest.get("build_feed_signature") or "").strip() != str(signature):
            raise RuntimeError(
                "sovereign_build_feed_signature_mismatch:"
                f"expected={signature} got={str(manifest.get('build_feed_signature') or '')}"
            )
        expected_house = self._expected_house_signature_base()
        expected_default = str(
            self.knowledgeverse._house_state_summary.get("default_knowledge_signature") or ""
        ).strip()
        if expected_house and str(manifest.get("house_signature_base") or "").strip() != expected_house:
            raise RuntimeError(
                "sovereign_build_feed_house_signature_mismatch:"
                f"expected={expected_house} got={str(manifest.get('house_signature_base') or '')}"
            )
        if expected_default and str(manifest.get("default_knowledge_signature") or "").strip() != expected_default:
            raise RuntimeError(
                "sovereign_build_feed_default_signature_mismatch:"
                f"expected={expected_default} got={str(manifest.get('default_knowledge_signature') or '')}"
            )
        for key, path in paths.items():
            if key == "manifest":
                continue
            if not path.exists():
                raise RuntimeError(f"sovereign_build_feed_missing_component:{key}:{path}")
        try:
            with paths["host_stars"].open("rb") as handle:
                host_stars = pickle.load(handle)
        except Exception as exc:
            raise RuntimeError(f"sovereign_build_feed_host_stars_invalid:{type(exc).__name__}:{exc}") from exc
        if not isinstance(host_stars, list):
            raise RuntimeError("sovereign_build_feed_host_stars_invalid:payload")
        return {
            "signature": str(signature),
            "manifest": dict(manifest),
            "paths": paths,
            "host_stars": [dict(star) for star in host_stars if isinstance(star, dict)],
        }

    def _expected_house_signature_base(self) -> str:
        return str(self.knowledgeverse._house_state_summary.get("gpu_buffer_signature_base") or "").strip()

    def _read_runtime_manifest(self) -> dict[str, Any]:
        path = self._artifact_manifest_path()
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def current_runtime_manifest(self) -> dict[str, Any]:
        if isinstance(self._last_runtime_manifest, dict) and self._last_runtime_manifest:
            return dict(self._last_runtime_manifest)
        return self._read_runtime_manifest()

    def invalidate_loaded_state(self) -> None:
        self._host_stars = []
        self._catalog_signature = ""
        self._build_feed_signature = ""
        self._feed_source_signature = ""
        self._last_build_feed_manifest = {}
        self.star_table.star_count = 0
        self._last_load_summary = {}

    def _load_runtime_artifacts(self) -> bool:
        total_t0 = time.perf_counter()
        bundle_path = self._artifact_bundle_path()
        manifest_t0 = time.perf_counter()
        manifest = self._read_runtime_manifest()
        summary: dict[str, Any] = {
            "status": "pending",
            "mode": "artifact",
            "artifact_bundle_path": str(bundle_path),
            "artifact_manifest_path": str(self._artifact_manifest_path()),
            "manifest_read_s": float(time.perf_counter() - manifest_t0),
        }
        if not bundle_path.exists():
            summary.update(
                {
                    "status": "missing",
                    "reason": "artifact_bundle_missing",
                    "total_elapsed_s": float(time.perf_counter() - total_t0),
                }
            )
            self._last_load_summary = summary
            return False
        if not manifest:
            summary.update(
                {
                    "status": "missing",
                    "reason": "artifact_manifest_missing",
                    "total_elapsed_s": float(time.perf_counter() - total_t0),
                }
            )
            self._last_load_summary = summary
            return False
        if int(manifest.get("version") or 0) != int(self.ARTIFACT_VERSION):
            summary.update(
                {
                    "status": "stale",
                    "reason": "artifact_version_mismatch",
                    "artifact_version": int(manifest.get("version") or 0),
                    "expected_version": int(self.ARTIFACT_VERSION),
                    "total_elapsed_s": float(time.perf_counter() - total_t0),
                }
            )
            self._last_load_summary = summary
            return False
        expected_signature_base = self._expected_house_signature_base()
        manifest_signature_base = str(manifest.get("house_signature_base") or "").strip()
        expected_default_signature = str(
            self.knowledgeverse._house_state_summary.get("default_knowledge_signature") or ""
        ).strip()
        manifest_default_signature = str(manifest.get("default_knowledge_signature") or "").strip()
        if expected_signature_base and manifest_signature_base and expected_signature_base != manifest_signature_base:
            summary.update(
                {
                    "status": "stale",
                    "reason": "house_signature_base_mismatch",
                    "expected_house_signature_base": expected_signature_base,
                    "artifact_house_signature_base": manifest_signature_base,
                    "total_elapsed_s": float(time.perf_counter() - total_t0),
                }
            )
            self._last_load_summary = summary
            return False
        if expected_default_signature and manifest_default_signature != expected_default_signature:
            summary.update(
                {
                    "status": "stale",
                    "reason": "default_knowledge_signature_mismatch",
                    "expected_default_knowledge_signature": expected_default_signature,
                    "artifact_default_knowledge_signature": manifest_default_signature,
                    "total_elapsed_s": float(time.perf_counter() - total_t0),
                }
            )
            self._last_load_summary = summary
            return False
        deserialize_t0 = time.perf_counter()
        try:
            with bundle_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            summary.update(
                {
                    "status": "error",
                    "reason": "artifact_bundle_deserialize_failed",
                    "exception_type": type(exc).__name__,
                    "detail": str(exc),
                    "artifact_bundle_deserialize_s": float(time.perf_counter() - deserialize_t0),
                    "total_elapsed_s": float(time.perf_counter() - total_t0),
                }
            )
            self._last_load_summary = summary
            return False
        if not isinstance(payload, dict):
            summary.update(
                {
                    "status": "error",
                    "reason": "artifact_bundle_invalid",
                    "artifact_bundle_deserialize_s": float(time.perf_counter() - deserialize_t0),
                    "total_elapsed_s": float(time.perf_counter() - total_t0),
                }
            )
            self._last_load_summary = summary
            return False
        table_bundle = payload.get("table_bundle")
        if not isinstance(table_bundle, dict):
            summary.update(
                {
                    "status": "error",
                    "reason": "artifact_table_bundle_missing",
                    "artifact_bundle_deserialize_s": float(time.perf_counter() - deserialize_t0),
                    "total_elapsed_s": float(time.perf_counter() - total_t0),
                }
            )
            self._last_load_summary = summary
            return False
        host_stars = list(payload.get("host_stars") or [])
        table_load_t0 = time.perf_counter()
        try:
            self.star_table.load_artifact_bundle(table_bundle, host_stars=host_stars)
        except Exception as exc:
            summary.update(
                {
                    "status": "error",
                    "reason": "artifact_table_upload_failed",
                    "exception_type": type(exc).__name__,
                    "detail": str(exc),
                    "artifact_bundle_deserialize_s": float(time.perf_counter() - deserialize_t0),
                    "star_table_upload_s": float(time.perf_counter() - table_load_t0),
                    "total_elapsed_s": float(time.perf_counter() - total_t0),
                }
            )
            self._last_load_summary = summary
            return False
        self._host_stars = [dict(star) for star in host_stars[: self.star_table.star_count]]
        self._catalog_signature = str(payload.get("catalog_signature") or "")
        self._build_feed_signature = str(payload.get("build_feed_signature") or manifest.get("build_feed_signature") or "")
        self._feed_source_signature = str(payload.get("feed_source_signature") or manifest.get("feed_source_signature") or "")
        learning_state = payload.get("learning_state")
        learning_restore_t0 = time.perf_counter()
        if isinstance(learning_state, dict):
            self.lesson_ring.load_stats(learning_state)
        self._last_runtime_manifest = dict(manifest)
        self._last_load_summary = {
            "status": "ready",
            "mode": "artifact",
            "star_count": int(self.star_table.star_count),
            "catalog_signature": self._catalog_signature,
            "build_feed_signature": self._build_feed_signature,
            "feed_source_signature": self._feed_source_signature,
            "house_signature_base": manifest_signature_base,
            "default_knowledge_signature": manifest_default_signature,
            "artifact_bundle_deserialize_s": float(time.perf_counter() - deserialize_t0),
            "star_table_upload_s": float(time.perf_counter() - table_load_t0),
            "learning_restore_s": float(time.perf_counter() - learning_restore_t0),
            "total_elapsed_s": float(time.perf_counter() - total_t0),
        }
        return True

    def save_runtime_artifacts(self) -> dict[str, Any]:
        if self.star_table.star_count <= 0:
            raise RuntimeError("sovereign_runtime_not_loaded")
        bundle_path = self._artifact_bundle_path()
        manifest_path = self._artifact_manifest_path()
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        table_bundle = self.star_table.export_artifact_bundle()
        symlink_edges = 0
        reciprocal_edges = 0
        for star_index, star in enumerate(self._host_stars):
            for key in ("router_refs", "executor_refs", "validator_refs", "anti_pattern_refs"):
                for target_index in list(star.get(key) or []):
                    symlink_edges += 1
                    if 0 <= int(target_index) < len(self._host_stars):
                        target = self._host_stars[int(target_index)]
                        target_refs = list(target.get("router_refs") or []) + list(target.get("executor_refs") or []) + list(target.get("validator_refs") or []) + list(target.get("anti_pattern_refs") or [])
                        if star_index in target_refs:
                            reciprocal_edges += 1
        manifest = {
            "version": int(self.ARTIFACT_VERSION),
            "saved_at": time.time(),
            "star_count": int(self.star_table.star_count),
            "build_backend": BUILD_BACKEND,
            "build_feed_version": int(BUILD_FEED_VERSION),
            "build_feed_signature": str(self._build_feed_signature),
            "feed_source_version": int(self._last_build_feed_manifest.get("feed_source_version") or FEED_SOURCE_VERSION),
            "feed_source_signature": str(
                self._last_build_feed_manifest.get("feed_source_signature") or self._feed_source_signature
            ),
            "catalog_signature": str(self._catalog_signature),
            "house_signature_base": self._expected_house_signature_base(),
            "default_knowledge_signature": str(
                self.knowledgeverse._house_state_summary.get("default_knowledge_signature") or ""
            ).strip(),
            "role_csr_artifact_signature": f"refs:{len(table_bundle.get('ref_indices') or [])}",
            "gravity_prior_artifact_signature": f"gravity:{int(self.star_table.star_count)}",
            "lesson_ring_schema_version": 1,
            "source_checkpoint_version": int(self.knowledgeverse.HOUSE_STATE_VERSION),
            "symlink_edges": int(symlink_edges),
            "reciprocal_edges": int(reciprocal_edges),
            "meaning_family_route_audit_path": str(
                self._last_build_feed_manifest.get("meaning_family_route_audit_path") or self._meaning_family_route_audit_path()
            ),
            "meaning_family_route_audit_passed": bool(
                self._last_build_feed_manifest.get("meaning_family_route_audit_passed")
            ),
            "meaning_route_closure_audit_path": str(
                self._last_build_feed_manifest.get("meaning_route_closure_audit_path") or self._meaning_route_closure_audit_path()
            ),
            "meaning_route_closure_audit_passed": bool(
                self._last_build_feed_manifest.get("meaning_route_closure_audit_passed")
            ),
            "meaning_knowledge_coverage_audit_path": str(
                self._last_build_feed_manifest.get("meaning_knowledge_coverage_audit_path")
                or self._meaning_knowledge_coverage_audit_path()
            ),
            "meaning_knowledge_coverage_audit_passed": bool(
                self._last_build_feed_manifest.get("meaning_knowledge_coverage_audit_passed")
            ),
        }
        manifest.update(self.materializer.ptx_signatures())
        payload = {
            "version": int(self.ARTIFACT_VERSION),
            "build_backend": BUILD_BACKEND,
            "build_feed_version": int(BUILD_FEED_VERSION),
            "build_feed_signature": str(self._build_feed_signature),
            "feed_source_version": int(self._last_build_feed_manifest.get("feed_source_version") or FEED_SOURCE_VERSION),
            "feed_source_signature": str(
                self._last_build_feed_manifest.get("feed_source_signature") or self._feed_source_signature
            ),
            "catalog_signature": str(self._catalog_signature),
            "host_stars": [dict(star) for star in self._host_stars],
            "table_bundle": table_bundle,
            "learning_state": self.current_learning_state(),
        }
        with bundle_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        self._last_runtime_manifest = dict(manifest)
        return dict(manifest)

    def ensure_loaded(self, *, force_rebuild: bool = False) -> dict[str, Any]:
        total_t0 = time.perf_counter()
        default_load_t0 = time.perf_counter()
        default_load_skipped = bool(getattr(self.knowledgeverse, "_default_galaxies_loaded", False))
        if not default_load_skipped:
            self.knowledgeverse.ensure_default_galaxies_loaded()
        default_galaxy_load_s = float(time.perf_counter() - default_load_t0)
        if (not force_rebuild) and self.star_table.star_count > 0 and self._catalog_signature:
            summary = {
                "status": "ready",
                "mode": "resident",
                "star_count": int(self.star_table.star_count),
                "catalog_signature": self._catalog_signature,
                "default_galaxy_load_s": default_galaxy_load_s,
                "default_galaxy_load_skipped": bool(default_load_skipped),
                "total_elapsed_s": float(time.perf_counter() - total_t0),
            }
            self._last_load_summary = dict(summary)
            return summary
        if (not force_rebuild) and self._load_runtime_artifacts():
            summary = dict(self._last_load_summary)
            summary.setdefault("default_galaxy_load_s", default_galaxy_load_s)
            summary.setdefault("default_galaxy_load_skipped", bool(default_load_skipped))
            summary["total_elapsed_s"] = float(time.perf_counter() - total_t0)
            self._last_load_summary = dict(summary)
            return summary
        artifact_fallback = (
            {"status": "skipped", "reason": "force_rebuild"}
            if force_rebuild
            else dict(self._last_load_summary)
        )
        build_feed_signature, _resolved_names = self._expected_build_feed_signature()
        build_feed_t0 = time.perf_counter()
        build_feed = self._load_build_feed(build_feed_signature)
        build_feed_load_s = float(time.perf_counter() - build_feed_t0)
        star_build_t0 = time.perf_counter()
        build_summary = self._build_stars_from_build_feed(build_feed)
        star_build_s = float(time.perf_counter() - star_build_t0)
        self._build_feed_signature = str(build_feed_signature)
        self._catalog_signature = str(build_feed.get("manifest", {}).get("catalog_signature") or build_feed_signature)
        artifact_save_t0 = time.perf_counter()
        manifest = self.save_runtime_artifacts()
        artifact_save_s = float(time.perf_counter() - artifact_save_t0)
        summary = {
            "status": "ready",
            "mode": "rebuilt",
            "star_count": int(self.star_table.star_count),
            "build_backend": BUILD_BACKEND,
            "catalog_signature": self._catalog_signature,
            "house_signature_base": str(manifest.get("house_signature_base") or ""),
            "default_knowledge_signature": str(manifest.get("default_knowledge_signature") or ""),
            "default_galaxy_load_s": default_galaxy_load_s,
            "default_galaxy_load_skipped": bool(default_load_skipped),
            "build_feed_load_s": build_feed_load_s,
            "build_feed_manifest": dict(build_feed.get("manifest") or {}),
            "star_build_s": star_build_s,
            "artifact_save_s": artifact_save_s,
            "artifact_restore_fallback": artifact_fallback,
            "total_elapsed_s": float(time.perf_counter() - total_t0),
        }
        summary.update(build_summary)
        summary.update(self.materializer.ptx_signatures())
        self._last_load_summary = dict(summary)
        return summary

    def _entry_metadata(self, entry: dict[str, Any]) -> dict[str, Any]:
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        expanded = dict(metadata or {})
        for key in ("meaning_star", "grammar_star", "meta_star", "answer_star"):
            nested = metadata.get(key)
            if isinstance(nested, dict):
                for nested_key, nested_value in nested.items():
                    expanded.setdefault(str(nested_key), nested_value)
        for key in ("meaning_star", "grammar_star", "meta_star", "answer_star"):
            nested = entry.get(key)
            if isinstance(nested, dict):
                for nested_key, nested_value in nested.items():
                    expanded.setdefault(str(nested_key), nested_value)
        return expanded

    @staticmethod
    def _finite_boot_float(value: Any, default: float = 0.0) -> float:
        try:
            numeric = float(value)
        except Exception:
            return float(default)
        if numeric != numeric or numeric in {float("inf"), float("-inf")}:
            return float(default)
        return float(numeric)

    def _infer_layer_id(self, galaxy_name: str, metadata: dict[str, Any], source: dict[str, Any]) -> int:
        raw = metadata.get(
            "layer_id",
            source.get("layer_id", metadata.get("layer", source.get("layer", 0))),
        )
        try:
            numeric = int(raw)
            if numeric > 0:
                return numeric
        except Exception:
            pass
        name = str(raw or "").strip().lower()
        if name in {"form", "layer1", "1"}:
            return 1
        if name in {"meaning", "layer2", "2"}:
            return 2
        if name in {"rules", "rule", "layer3", "3"}:
            return 3
        if name in {"meta", "meta_rule", "meta-rules", "layer4", "4"}:
            return 4
        return 0

    def _infer_selection_role(self, galaxy_name: str, metadata: dict[str, Any], source: dict[str, Any], layer_id: int) -> str:
        explicit = str(metadata.get("selection_role") or source.get("selection_role") or "").strip().lower()
        if explicit in {"router", "executor", "validator", "answer", "anti_pattern"}:
            return explicit
        return "unknown"

    def _infer_answer_eligible(self, role: str, metadata: dict[str, Any], source: dict[str, Any]) -> bool:
        raw = metadata.get("answer_eligible", source.get("answer_eligible"))
        if raw is None:
            return False
        return bool(raw)

    def _infer_route_policy(self, star_id: str, selection_role: str, metadata: dict[str, Any], source: dict[str, Any]) -> int:
        explicit = metadata.get("route_policy")
        if not isinstance(explicit, dict):
            explicit = source.get("route_policy") if isinstance(source.get("route_policy"), dict) else {}
        decompose_on_fail = bool(explicit.get("decompose_on_fail"))
        requires_executor = bool(explicit.get("requires_executor"))
        requires_validator = bool(explicit.get("requires_validator"))
        answer_gate = bool(explicit.get("answer_gate"))
        branch_topk = int(explicit.get("branch_topk", 0) or 0)
        return encode_route_policy(
            decompose_on_fail=decompose_on_fail,
            requires_executor=requires_executor,
            requires_validator=requires_validator,
            answer_gate=answer_gate,
            branch_topk=branch_topk,
        )

    @staticmethod
    def _top_level_or_metadata_list(source: dict[str, Any], metadata: dict[str, Any], *keys: str) -> list[Any]:
        values: list[Any] = []
        for key in keys:
            for container in (source, metadata):
                raw = container.get(key)
                if isinstance(raw, list):
                    for item in raw:
                        if item is None:
                            continue
                        if isinstance(item, int):
                            values.append(int(item))
                            continue
                        text = str(item).strip()
                        if text:
                            values.append(text)
        return values

    @staticmethod
    def _default_feed_source_worker_count() -> int:
        cpu_count = os.cpu_count() or 2
        return int(min(DEFAULT_FEED_SOURCE_MAX_WORKERS, max(2, int(cpu_count) - 2)))

    @staticmethod
    def _resolve_ref_index_from_hash_or_index(
        ref_value: Any,
        hash_to_index: dict[int, int],
        star_count: int,
    ) -> int | None:
        if isinstance(ref_value, int):
            candidate = int(ref_value)
            if 0 <= candidate < int(star_count):
                return candidate
            return None
        ref_id = str(ref_value or "").strip()
        if not ref_id:
            return None
        ref_index = hash_to_index.get(_fnv1a64(ref_id))
        if ref_index is not None:
            return int(ref_index)
        if ref_id.isdigit():
            candidate = int(ref_id)
            if 0 <= candidate < int(star_count):
                return candidate
        return None

    def _resolve_ref_index(self, ref_value: Any, id_to_index: dict[str, int], star_count: int) -> int | None:
        if isinstance(ref_value, int):
            candidate = int(ref_value)
            if 0 <= candidate < int(star_count):
                return candidate
            return None
        ref_id = str(ref_value or "").strip()
        if not ref_id:
            return None
        ref_index = id_to_index.get(ref_id)
        if ref_index is not None:
            return int(ref_index)
        if ref_id.isdigit():
            candidate = int(ref_id)
            if 0 <= candidate < int(star_count):
                return candidate
        return None

    def _translate_catalog_row(
        self,
        catalog_row: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, list[Any]], list[str]]:
        metadata_errors: list[str] = []
        source = dict(self.knowledgeverse._catalog_source_entry(catalog_row) or catalog_row)
        metadata = self._entry_metadata(source)
        catalog_metadata = dict(catalog_row.get("metadata") or {})
        if catalog_metadata:
            for key, value in catalog_metadata.items():
                if isinstance(value, list):
                    current = metadata.get(key)
                    if isinstance(current, list):
                        merged = list(current)
                        for item in value:
                            if item not in merged:
                                merged.append(item)
                        metadata[key] = merged
                    else:
                        metadata[key] = list(value)
                    continue
                if isinstance(value, dict):
                    current = metadata.get(key)
                    merged = dict(current) if isinstance(current, dict) else {}
                    for nested_key, nested_value in value.items():
                        if (
                            nested_key not in merged
                            or merged.get(nested_key) in (None, "", [], {})
                        ):
                            merged[nested_key] = nested_value
                    metadata[key] = merged
                    continue
                if key not in metadata or metadata.get(key) in (None, ""):
                    metadata[key] = value
        galaxy_name = str(catalog_row.get("galaxy") or source.get("galaxy") or source.get("domain") or "")
        star_id = str(source.get("id") or source.get("rule_id") or catalog_row.get("id") or "").strip()
        if not star_id:
            star_id = f"{galaxy_name}:{catalog_row.get('entry_idx', 0)}"
        if not star_id:
            return None, {}, metadata_errors
        layer_id = self._infer_layer_id(galaxy_name, metadata, source)
        selection_role = self._infer_selection_role(galaxy_name, metadata, source, layer_id)
        answer_eligible = self._infer_answer_eligible(selection_role, metadata, source)
        embedding16 = [float(value) for value in list(catalog_row.get("embedding16") or source.get("embedding16") or [])[:16]]
        if len(embedding16) < 16:
            embedding16.extend([0.0] * (16 - len(embedding16)))
        if not any(abs(float(value)) > 0.0 for value in embedding16):
            metadata_errors.append(f"{star_id}:missing_precomputed_embedding16")
            return None, {}, metadata_errors
        route_policy = metadata.get("route_policy")
        if not isinstance(route_policy, dict):
            route_policy = source.get("route_policy") if isinstance(source.get("route_policy"), dict) else {}
        explicit_role_refs_present = any(
            isinstance(container.get(key), list) and bool(container.get(key))
            for container in (source, metadata)
            for key in ("router_refs", "executor_refs", "validator_refs", "anti_pattern_refs")
        )
        sovereign_route_exempt = bool(
            metadata.get(
                "sovereign_route_exempt",
                source.get("sovereign_route_exempt", catalog_row.get("sovereign_route_exempt", False)),
            )
        )
        routing_refs = {
            "router_refs": self._top_level_or_metadata_list(
                source,
                metadata,
                "router_refs",
                "component_refs",
                "composite_of",
                "reality_refs",
                "taxonomy_refs",
            ),
            "executor_refs": self._top_level_or_metadata_list(
                source,
                metadata,
                "grammar_refs",
                "executor_refs",
                "behavior_refs",
                "visual_refs",
            ),
            "validator_refs": self._top_level_or_metadata_list(
                source,
                metadata,
                "meta_refs",
                "validator_refs",
                "validation_refs",
            ),
            "anti_pattern_refs": self._top_level_or_metadata_list(
                source,
                metadata,
                "anti_pattern_refs",
                "negative_refs",
                "contrastive_refs",
                "superior_to",
            ),
        }
        if sovereign_route_exempt:
            layer_id = 0
            selection_role = "unknown"
            answer_eligible = False
            route_policy = {}
            explicit_role_refs_present = False
            routing_refs = {
                "router_refs": [],
                "executor_refs": [],
                "validator_refs": [],
                "anti_pattern_refs": [],
            }
        route_active = (not sovereign_route_exempt) and (
            selection_role != "unknown" or explicit_role_refs_present or bool(route_policy)
        )
        if route_active:
            if selection_role == "unknown":
                metadata_errors.append(f"{star_id}:missing_selection_role")
            if layer_id <= 0:
                metadata_errors.append(f"{star_id}:missing_layer_id")
            if "answer_eligible" not in metadata and "answer_eligible" not in source:
                metadata_errors.append(f"{star_id}:missing_answer_eligible")
            if selection_role in {"router", "executor"}:
                if not route_policy:
                    metadata_errors.append(f"{star_id}:missing_route_policy")
                else:
                    required_keys = {"requires_validator", "branch_topk"}
                    if selection_role == "router":
                        required_keys = required_keys | {"requires_executor", "answer_gate"}
                    missing_keys = [key for key in sorted(required_keys) if key not in route_policy]
                    if missing_keys:
                        metadata_errors.append(f"{star_id}:missing_route_policy_fields={','.join(missing_keys)}")
        explicit_mask = 0
        if "semantic_polarity" in metadata or "semantic_polarity" in source:
            explicit_mask |= EXPLICIT_POLARITY_MASK
        if "semantic_focus" in metadata or "semantic_focus" in source:
            explicit_mask |= EXPLICIT_FOCUS_MASK
        if "semantic_mass" in metadata or "semantic_mass" in source:
            explicit_mask |= EXPLICIT_MASS_MASK
        if "attractive_prior" in metadata or "attractive_prior" in source:
            explicit_mask |= EXPLICIT_ATTRACTIVE_MASK
        if "repulsive_prior" in metadata or "repulsive_prior" in source:
            explicit_mask |= EXPLICIT_REPULSIVE_MASK
        route_policy_flags = 0
        if bool(route_policy.get("decompose_on_fail")):
            route_policy_flags |= 0x01
        if bool(route_policy.get("requires_executor")):
            route_policy_flags |= 0x02
        if bool(route_policy.get("requires_validator")):
            route_policy_flags |= 0x04
        if bool(route_policy.get("answer_gate")):
            route_policy_flags |= 0x08
        confidence = self._finite_boot_float(
            catalog_row.get("confidence", metadata.get("trust_weight", source.get("confidence", 0.5))),
            0.5,
        )
        domain_hash = self._finite_boot_float(catalog_row.get("domain_hash", 0.0), 0.0)
        subject_hash = self._finite_boot_float(catalog_row.get("subject_hash", 0.0), 0.0)
        galaxy_id = str(galaxy_name or catalog_row.get("galaxy") or "reality")
        route_family = str(
            source.get("route_family")
            or metadata.get("route_family")
            or catalog_row.get("route_family")
            or ""
        ).strip()
        if sovereign_route_exempt:
            route_family = ""
        elif route_family:
            route_family = str(VRAMTaskBuffer.normalize_task_type(route_family))
        elif galaxy_id.strip().upper() in {"GAME_2D", "MATH", "QUESTION", "CHAT", "GENERAL", "GRAMMAR", "INTERACTION"}:
            route_family = str(VRAMTaskBuffer.normalize_task_type(galaxy_id))
        star = {
            "id": star_id,
            "name": str(source.get("name") or catalog_row.get("name") or star_id),
            "galaxy_id": galaxy_id,
            "galaxy_id_u32": _fnv1a32(galaxy_id),
            "route_family": route_family,
            "star_type": int(catalog_row.get("gpu_source_class", 0) or 0),
            "selection_role": selection_role,
            "selection_role_id": int(ROLE_ID_BY_NAME.get(selection_role, 0)),
            "layer_id": int(layer_id),
            "flags": STAR_FLAG_ACTIVE | STAR_FLAG_LEARNABLE,
            "answer_eligible": bool(answer_eligible),
            "sovereign_route_exempt": bool(sovereign_route_exempt),
            "semantic_polarity_raw": int(metadata.get("semantic_polarity", source.get("semantic_polarity", 0)) or 0),
            "semantic_focus_raw": self._finite_boot_float(metadata.get("semantic_focus", source.get("semantic_focus", 0.0)), 0.0),
            "semantic_mass_raw": self._finite_boot_float(metadata.get("semantic_mass", source.get("semantic_mass", 0.0)), 0.0),
            "attractive_prior_raw": self._finite_boot_float(metadata.get("attractive_prior", source.get("attractive_prior", 0.0)), 0.0),
            "repulsive_prior_raw": self._finite_boot_float(metadata.get("repulsive_prior", source.get("repulsive_prior", 0.0)), 0.0),
            "confidence": confidence,
            "route_policy_flags": int(route_policy_flags),
            "route_policy_branch_topk": int(route_policy.get("branch_topk", 0) or 0),
            "explicit_mask": int(explicit_mask),
            "star_hash": _fnv1a64(star_id),
            "embedding16": list(embedding16),
            "domain_hash": float(domain_hash),
            "subject_hash": float(subject_hash),
            "route_policy": dict(route_policy or {}),
            "router_refs": [],
            "executor_refs": [],
            "validator_refs": [],
            "anti_pattern_refs": [],
        }
        return star, routing_refs, metadata_errors

    def _validate_route_link_coverage(self, stars: list[dict[str, Any]]) -> None:
        route_errors: list[str] = []
        for star in stars:
            role = str(star.get("selection_role") or "unknown")
            if role == "router":
                if not list(star.get("executor_refs") or []):
                    route_errors.append(f"{star.get('id')}:router_missing_executor_refs")
                if not list(star.get("validator_refs") or []):
                    route_errors.append(f"{star.get('id')}:router_missing_validator_refs")
            elif role == "executor":
                route_policy = dict(star.get("route_policy") or {})
                if bool(route_policy.get("requires_validator")) and not list(star.get("validator_refs") or []):
                    route_errors.append(f"{star.get('id')}:executor_missing_validator_refs")
        if route_errors:
            sample = ", ".join(route_errors[:12])
            if len(route_errors) > 12:
                sample += f", ... (+{len(route_errors) - 12} more)"
            raise ValueError(f"sovereign_build_route_invalid:{sample}")

    def _compile_feed_source_chunk(
        self,
        catalog: list[dict[str, Any]],
        *,
        start: int,
        count: int,
    ) -> dict[str, Any]:
        chunk_stars: list[dict[str, Any]] = []
        pending_refs: list[tuple[int, int, int, Any]] = []
        metadata_errors: list[str] = []
        row_buffer = bytearray(max(1, int(count) * RAW_CATALOG_INPUT_ENTRY_BYTES))
        for local_index in range(int(count)):
            catalog_row = catalog[int(start) + local_index]
            star, routing_refs, row_errors = self._translate_catalog_row(catalog_row)
            metadata_errors.extend(row_errors)
            if star is None:
                continue
            source_index = len(chunk_stars)
            chunk_stars.append(star)
            self._pack_catalog_input_row(
                star,
                target=memoryview(row_buffer),
                local_index=source_index,
            )
            for key, role_type in ROLE_TYPE_BY_KEY.items():
                reverse_role_type = self._reverse_role_type_for(role_type)
                for ref_value in list(routing_refs.get(key) or []):
                    pending_refs.append((int(source_index), int(role_type), int(reverse_role_type), ref_value))
        row_bytes = bytes(row_buffer[: len(chunk_stars) * RAW_CATALOG_INPUT_ENTRY_BYTES])
        return {
            "start": int(start),
            "count": int(count),
            "stars": chunk_stars,
            "pending_refs": pending_refs,
            "row_bytes": row_bytes,
            "metadata_errors": metadata_errors,
        }

    def _translate_catalog_entries(
        self,
        catalog: list[dict[str, Any]],
        *,
        apply_bidirectional_symlinkage: bool = True,
        progress_phase: str = "translate-catalog",
    ) -> tuple[list[dict[str, Any]], list[tuple[int, int, int, int]]]:
        stars: list[dict[str, Any]] = []
        id_to_index: dict[str, int] = {}
        pending_refs: list[dict[str, list[Any]]] = []
        metadata_errors: list[str] = []
        total_catalog = len(catalog)
        progress_enabled = os.environ.get("K3D_SOVEREIGN_BUILD_PROGRESS", "0") == "1"
        for catalog_index, catalog_row in enumerate(catalog, start=1):
            star, routing_refs, row_errors = self._translate_catalog_row(catalog_row)
            metadata_errors.extend(row_errors)
            if star is None:
                continue
            stars.append(star)
            id_to_index[str(star.get("id") or "")] = len(stars) - 1
            pending_refs.append(routing_refs)
            if progress_enabled and ((catalog_index % 25000) == 0 or catalog_index == total_catalog):
                self._emit_rebuild_progress(
                    f"{progress_phase} {catalog_index}/{total_catalog} stars={len(stars)}"
                )
        if metadata_errors:
            sample = ", ".join(metadata_errors[:12])
            if len(metadata_errors) > 12:
                sample += f", ... (+{len(metadata_errors) - 12} more)"
            raise ValueError(f"sovereign_build_metadata_invalid:{sample}")
        for index, ref_payload in enumerate(pending_refs):
            star = stars[index]
            for key, ref_ids in ref_payload.items():
                resolved: list[int] = []
                for ref_id in ref_ids:
                    ref_index = self._resolve_ref_index(ref_id, id_to_index, len(stars))
                    if ref_index is None or ref_index in resolved:
                        continue
                    resolved.append(int(ref_index))
                star[key] = resolved
            if progress_enabled and (((index + 1) % 25000) == 0 or (index + 1) == len(pending_refs)):
                self._emit_rebuild_progress(
                    f"resolve-refs {index + 1}/{len(pending_refs)}"
                )
        self._validate_route_link_coverage(stars)
        if apply_bidirectional_symlinkage:
            self._ensure_bidirectional_symlinkage(stars)
        ref_tuples: list[tuple[int, int, int, int]] = []
        finalized_stars = self._finalize_component_refs(stars)
        for star_index, star in enumerate(finalized_stars):
            for key, role_type in ROLE_TYPE_BY_KEY.items():
                refs = [int(value) for value in list(star.get(key) or []) if int(value) >= 0]
                for slot, ref_index in enumerate(refs):
                    ref_tuples.append((int(star_index), int(role_type), int(ref_index), int(slot)))
            if progress_enabled and (((star_index + 1) % 25000) == 0 or (star_index + 1) == len(stars)):
                self._emit_rebuild_progress(
                    f"emit-tuples {star_index + 1}/{len(stars)} tuples={len(ref_tuples)}"
                )
        return finalized_stars, ref_tuples

    @staticmethod
    def _reverse_role_type_for(role_type: int) -> int:
        reverse_role_type = {
            0: 0,
            1: 0,
            2: 1,
            3: 3,
        }.get(int(role_type), -1)
        return int(reverse_role_type)

    @staticmethod
    def _strip_role_refs(star: dict[str, Any]) -> dict[str, Any]:
        row = dict(star)
        for key in ROLE_TYPE_BY_KEY:
            row[key] = []
        row["component_refs"] = []
        return row

    @staticmethod
    def _read_i32_device(ptr, count: int) -> list[int]:
        total = max(0, int(count))
        if ptr is None or total <= 0:
            return []
        payload = bytearray(total * 4)
        loader.memcpy_dtoh(
            ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(payload))),
            ptr,
            len(payload),
        )
        return list(struct.unpack_from(f"<{total}i", payload, 0))

    def _audit_route_capability_rows(
        self,
        rows: bytes,
        host_stars: list[dict[str, Any]],
    ) -> dict[str, Any]:
        star_count = int(len(host_stars))
        if star_count <= 0:
            return {
                "route_valid_count": 0,
                "route_neutral_count": 0,
                "route_broken_count": 0,
                "route_family_health": {},
            }
        row_bytes = star_count * RAW_CATALOG_INPUT_ENTRY_BYTES
        if len(rows) != row_bytes:
            raise RuntimeError(
                f"sovereign_feed_source_rows_size_mismatch:{len(rows)}:{row_bytes}"
            )
        host_buffer = loader.PinnedHostBuffer(max(1, row_bytes))
        raw_device = loader.gpu_malloc(max(1, row_bytes))
        trit_device = loader.gpu_malloc(max(1, star_count * 4))
        counts_device = loader.gpu_malloc(12)
        try:
            view = host_buffer.view().cast("B")[:row_bytes]
            view[:] = rows
            zero_counts = bytearray(12)
            zero_trits = bytearray(max(1, star_count * 4))
            loader.memcpy_htod(raw_device, host_buffer.ptr, row_bytes)
            loader.memcpy_htod(
                trit_device,
                ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(zero_trits))),
                len(zero_trits),
            )
            loader.memcpy_htod(
                counts_device,
                ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(zero_counts))),
                len(zero_counts),
            )
            audit_stream = loader.create_stream()
            try:
                self.materializer.audit_route_capability(
                    raw_input_ptr=raw_device,
                    star_count=star_count,
                    route_trits_ptr=trit_device,
                    audit_counts_ptr=counts_device,
                    stream=audit_stream,
                )
                loader.stream_synchronize(audit_stream)
            finally:
                loader.destroy_stream(audit_stream)
            trits = self._read_i32_device(trit_device, star_count)
            counts = self._read_u32_device(counts_device, 3)
        finally:
            host_buffer.close()
            try:
                loader.gpu_free(raw_device)
            except Exception:
                pass
            try:
                loader.gpu_free(trit_device)
            except Exception:
                pass
            try:
                loader.gpu_free(counts_device)
            except Exception:
                pass
        family_health = self._summarize_route_family_health(host_stars, trits)
        return {
            "route_valid_count": int(counts[0]) if len(counts) > 0 else 0,
            "route_neutral_count": int(counts[1]) if len(counts) > 1 else 0,
            "route_broken_count": int(counts[2]) if len(counts) > 2 else 0,
            "route_family_health": family_health,
        }

    @staticmethod
    def _host_star_route_trit(star: dict[str, Any]) -> int:
        role = str(star.get("selection_role") or "").strip().lower()
        layer_id = int(star.get("layer_id") or 0)
        answer_eligible = bool(star.get("answer_eligible"))
        route_policy = dict(star.get("route_policy") or {})
        executor_refs = len(list(star.get("executor_refs") or []))
        validator_refs = len(list(star.get("validator_refs") or []))
        if role == "router":
            valid = (
                layer_id > 0
                and (not answer_eligible)
                and bool(route_policy.get("requires_executor"))
                and bool(route_policy.get("requires_validator"))
                and bool(route_policy.get("answer_gate"))
                and executor_refs > 0
                and validator_refs > 0
            )
            return 1 if valid else -1
        if role == "executor":
            valid = layer_id > 0 and (
                (not bool(route_policy.get("requires_validator")))
                or validator_refs > 0
            )
            return 1 if valid else -1
        if role in {"validator", "answer", "anti_pattern"}:
            return 1 if layer_id > 0 else -1
        return 0

    def _audit_host_star_route_capability(
        self,
        host_stars: list[dict[str, Any]],
    ) -> dict[str, Any]:
        trits = [self._host_star_route_trit(star) for star in list(host_stars or [])]
        valid_count = sum(1 for value in trits if int(value) > 0)
        broken_count = sum(1 for value in trits if int(value) < 0)
        neutral_count = max(0, len(trits) - valid_count - broken_count)
        family_health = self._summarize_route_family_health(host_stars, trits)
        return {
            "route_valid_count": int(valid_count),
            "route_neutral_count": int(neutral_count),
            "route_broken_count": int(broken_count),
            "route_family_health": family_health,
        }

    def _route_family_name(self, star: dict[str, Any]) -> str:
        explicit = str(star.get("route_family") or "").strip()
        if explicit:
            normalized = VRAMTaskBuffer.normalize_task_type(explicit)
            return str(normalized or explicit.upper())
        galaxy_id = star.get("galaxy_id")
        if isinstance(galaxy_id, str):
            token = galaxy_id.strip()
            if token:
                normalized = VRAMTaskBuffer.normalize_task_type(token)
                if normalized != "GENERAL" or token.strip().upper() == "GENERAL":
                    return str(normalized)
                return token
        return "unknown"

    def _summarize_route_family_health(
        self,
        host_stars: list[dict[str, Any]],
        trits: list[int],
    ) -> dict[str, dict[str, int]]:
        family_health: dict[str, dict[str, int]] = {}
        reverse_key = {
            "executor_refs": "router_refs",
            "validator_refs": "executor_refs",
            "anti_pattern_refs": "anti_pattern_refs",
        }

        def _bucket(family: str) -> dict[str, int]:
            return family_health.setdefault(
                family,
                {
                    "valid": 0,
                    "neutral": 0,
                    "broken": 0,
                    "surface_bridges": 0,
                    "routers": 0,
                    "executors": 0,
                    "materializers": 0,
                    "validators": 0,
                    "anti_patterns": 0,
                    "missing_reciprocal_links": 0,
                    "incomplete_validator_coverage": 0,
                    "missing_materializer_paths": 0,
                },
            )

        for star, trit in zip(host_stars, trits):
            family = self._route_family_name(star)
            bucket = _bucket(family)
            if int(trit) > 0:
                bucket["valid"] += 1
            elif int(trit) < 0:
                bucket["broken"] += 1
            else:
                bucket["neutral"] += 1
            role = str(star.get("selection_role") or "unknown").strip().lower()
            if role == "router":
                bucket["routers"] += 1
            elif role == "executor":
                bucket["executors"] += 1
                if _is_materializer_star(star):
                    bucket["materializers"] += 1
            elif role == "validator":
                bucket["validators"] += 1
            elif role == "anti_pattern":
                bucket["anti_patterns"] += 1
            route_policy = dict(star.get("route_policy") or {})
            if _is_surface_bridge_star(star):
                bucket["surface_bridges"] += 1
            if (
                role in {"router", "executor"}
                and bool(route_policy.get("requires_validator"))
                and not list(star.get("validator_refs") or [])
            ):
                bucket["incomplete_validator_coverage"] += 1

        materializer_indices = {
            index
            for index, star in enumerate(host_stars)
            if _is_materializer_star(star)
        }
        materializer_path_cache: dict[int, bool] = {}

        def _reaches_materializer(source_index: int, seen: set[int] | None = None) -> bool:
            if source_index in materializer_path_cache:
                return materializer_path_cache[source_index]
            if source_index in materializer_indices:
                materializer_path_cache[source_index] = True
                return True
            visited = set(seen or ())
            if source_index in visited:
                materializer_path_cache[source_index] = False
                return False
            visited.add(source_index)
            star = host_stars[int(source_index)]
            for target_index in list(star.get("executor_refs") or []):
                if not isinstance(target_index, int):
                    continue
                if target_index < 0 or target_index >= len(host_stars):
                    continue
                if _reaches_materializer(int(target_index), visited):
                    materializer_path_cache[source_index] = True
                    return True
            materializer_path_cache[source_index] = False
            return False

        for source_index, star in enumerate(host_stars):
            family = self._route_family_name(star)
            bucket = _bucket(family)
            role = str(star.get("selection_role") or "unknown").strip().lower()
            route_policy = dict(star.get("route_policy") or {})
            if (
                role in {"router", "executor"}
                and bool(route_policy.get("answer_gate"))
                and not _is_materializer_star(star)
                and not _reaches_materializer(source_index)
            ):
                bucket["missing_materializer_paths"] += 1

        for source_index, star in enumerate(host_stars):
            source_family = self._route_family_name(star)
            source_bucket = _bucket(source_family)
            for key, target_key in reverse_key.items():
                for target_index in list(star.get(key) or []):
                    if not isinstance(target_index, int):
                        continue
                    if target_index < 0 or target_index >= len(host_stars):
                        continue
                    target_refs = list(host_stars[target_index].get(target_key) or [])
                    if source_index not in target_refs:
                        source_bucket["missing_reciprocal_links"] += 1
        return family_health

    def _build_meaning_route_closure_audit(
        self,
        *,
        host_stars: list[dict[str, Any]],
        audit_summary: dict[str, Any],
        build_feed_signature: str,
        feed_source_signature: str,
    ) -> dict[str, Any]:
        family_health = {
            str(family): {
                key: int(value)
                for key, value in dict(metrics or {}).items()
            }
            for family, metrics in dict(audit_summary.get("route_family_health") or {}).items()
        }
        families: dict[str, Any] = {}
        minima_passed = True
        total_missing_materializer_paths = 0
        for family, minima in MEANING_ROUTE_CLOSURE_MINIMA.items():
            actual = dict(family_health.get(family) or {})
            meets_minima = {
                key: int(actual.get(key) or 0) >= int(expected)
                for key, expected in minima.items()
            }
            family_missing_materializer_paths = int(actual.get("missing_materializer_paths") or 0)
            if not all(meets_minima.values()) or family_missing_materializer_paths != 0:
                minima_passed = False
            total_missing_materializer_paths += family_missing_materializer_paths
            families[family] = {
                "actual": {key: int(actual.get(key) or 0) for key in minima},
                "minimum": {key: int(expected) for key, expected in minima.items()},
                "meets_minima": dict(meets_minima),
                "missing_materializer_paths": family_missing_materializer_paths,
                "missing_reciprocal_links": int(actual.get("missing_reciprocal_links") or 0),
                "incomplete_validator_coverage": int(actual.get("incomplete_validator_coverage") or 0),
            }
        governed_families = set(MEANING_ROUTE_CLOSURE_MINIMA)
        total_missing_reciprocal_links = sum(
            int((family_health.get(family) or {}).get("missing_reciprocal_links") or 0)
            for family in governed_families
        )
        total_incomplete_validator_coverage = sum(
            int((family_health.get(family) or {}).get("incomplete_validator_coverage") or 0)
            for family in governed_families
        )
        passed = bool(
            minima_passed
            and total_missing_materializer_paths == 0
            and total_missing_reciprocal_links == 0
            and total_incomplete_validator_coverage == 0
        )
        return {
            "generated_at": time.time(),
            "build_backend": BUILD_BACKEND,
            "build_feed_signature": str(build_feed_signature),
            "feed_source_signature": str(feed_source_signature),
            "route_valid_count": int(audit_summary.get("route_valid_count") or 0),
            "route_neutral_count": int(audit_summary.get("route_neutral_count") or 0),
            "route_broken_count": int(audit_summary.get("route_broken_count") or 0),
            "total_missing_materializer_paths": int(total_missing_materializer_paths),
            "total_missing_reciprocal_links": int(total_missing_reciprocal_links),
            "total_incomplete_validator_coverage": int(total_incomplete_validator_coverage),
            "families": families,
            "route_family_health": family_health,
            "passed": passed,
        }

    def _build_meaning_knowledge_coverage_audit(
        self,
        *,
        host_stars: list[dict[str, Any]],
        build_feed_signature: str,
        feed_source_signature: str,
    ) -> dict[str, Any]:
        inventory = curated_math_question_knowledge_gap_inventory()
        packets = curated_math_question_coverage_packets()
        stars_by_id = {
            str(star.get("id") or "").strip(): dict(star)
            for star in list(host_stars or [])
            if str(star.get("id") or "").strip()
        }
        packet_results: dict[str, Any] = {}
        total_missing_required_ids = 0
        total_failed_packets = 0
        total_failed_route_packets = 0
        open_text_materializer_present = 0
        targeted_anti_pattern_present = 0

        for packet_name, packet in packets.items():
            packet_kind = str(packet.get("kind") or "knowledge_packet").strip().lower()
            family = str(packet.get("family") or "").strip().upper()
            required_ids = [
                str(entry_id).strip()
                for entry_id in list(packet.get("required_ids") or [])
                if str(entry_id).strip()
            ]
            materializer_ids = [
                str(entry_id).strip()
                for entry_id in list(packet.get("materializer_ids") or [])
                if str(entry_id).strip()
            ]
            anti_pattern_ids = [
                str(entry_id).strip()
                for entry_id in list(packet.get("anti_pattern_ids") or [])
                if str(entry_id).strip()
            ]
            present_ids = [entry_id for entry_id in required_ids if entry_id in stars_by_id]
            missing_ids = [entry_id for entry_id in required_ids if entry_id not in stars_by_id]
            total_missing_required_ids += len(missing_ids)
            role_counts = {
                "routers": 0,
                "executors": 0,
                "materializers": 0,
                "validators": 0,
                "anti_patterns": 0,
            }
            for entry_id in present_ids:
                star = dict(stars_by_id.get(entry_id) or {})
                role = str(star.get("selection_role") or "").strip().lower()
                if role == "router":
                    role_counts["routers"] += 1
                elif role == "executor":
                    role_counts["executors"] += 1
                    if _is_materializer_star(star):
                        role_counts["materializers"] += 1
                elif role == "validator":
                    role_counts["validators"] += 1
                elif role == "anti_pattern":
                    role_counts["anti_patterns"] += 1
            has_materializers = all(entry_id in stars_by_id for entry_id in materializer_ids) if materializer_ids else True
            has_anti_patterns = all(entry_id in stars_by_id for entry_id in anti_pattern_ids) if anti_pattern_ids else True
            if any(
                entry_id in stars_by_id
                for entry_id in materializer_ids
                if entry_id in {"question_open_text_materializer", "general_open_text_materializer"}
            ):
                open_text_materializer_present += 1
            if any(entry_id in stars_by_id for entry_id in anti_pattern_ids):
                targeted_anti_pattern_present += 1
            route_chain_complete = True
            if packet_kind == "route_packet":
                route_chain_complete = bool(
                    role_counts["executors"] > 0
                    and role_counts["validators"] > 0
                    and has_materializers
                    and (role_counts["routers"] > 0 or family in {"MATH", "QUESTION", "GENERAL"})
                )
            packet_passed = bool(not missing_ids and has_materializers and has_anti_patterns and route_chain_complete)
            if not packet_passed:
                total_failed_packets += 1
                if packet_kind == "route_packet":
                    total_failed_route_packets += 1
            packet_results[packet_name] = {
                "kind": packet_kind,
                "family": family,
                "required_ids": required_ids,
                "present_ids": present_ids,
                "missing_ids": missing_ids,
                "role_counts": role_counts,
                "materializer_ids": materializer_ids,
                "anti_pattern_ids": anti_pattern_ids,
                "has_materializers": bool(has_materializers),
                "has_anti_patterns": bool(has_anti_patterns),
                "route_chain_complete": bool(route_chain_complete),
                "passed": bool(packet_passed),
            }

        passed = bool(total_missing_required_ids == 0 and total_failed_packets == 0)
        return {
            "generated_at": time.time(),
            "build_backend": BUILD_BACKEND,
            "build_feed_signature": str(build_feed_signature),
            "feed_source_signature": str(feed_source_signature),
            "wave_id": str(inventory.get("wave_id") or ""),
            "artifact_root": str(inventory.get("artifact_root") or ""),
            "focus_families": list(inventory.get("focus_families") or []),
            "failure_clusters": dict(inventory.get("failure_clusters") or {}),
            "packet_count": int(len(packet_results)),
            "total_missing_required_ids": int(total_missing_required_ids),
            "total_failed_packets": int(total_failed_packets),
            "total_failed_route_packets": int(total_failed_route_packets),
            "open_text_materializer_packet_hits": int(open_text_materializer_present),
            "targeted_anti_pattern_packet_hits": int(targeted_anti_pattern_present),
            "packets": packet_results,
            "passed": passed,
        }

    def _build_meaning_family_route_audit(
        self,
        *,
        host_stars: list[dict[str, Any]],
        audit_summary: dict[str, Any],
        build_feed_signature: str,
        feed_source_signature: str,
    ) -> dict[str, Any]:
        family_health = {
            str(family): {
                key: int(value)
                for key, value in dict(metrics or {}).items()
            }
            for family, metrics in dict(audit_summary.get("route_family_health") or {}).items()
        }
        route_capable_roles = {"router", "executor", "validator", "anti_pattern"}
        missing_explicit_route_family = sum(
            1
            for star in list(host_stars or [])
            if str(star.get("selection_role") or "").strip().lower() in route_capable_roles
            and not str(star.get("route_family") or "").strip()
        )
        total_missing_reciprocal_links = sum(
            int(metrics.get("missing_reciprocal_links") or 0)
            for metrics in family_health.values()
        )
        total_incomplete_validator_coverage = sum(
            int(metrics.get("incomplete_validator_coverage") or 0)
            for metrics in family_health.values()
        )
        families: dict[str, Any] = {}
        minima_passed = True
        for family, minima in MEANING_FAMILY_ROUTE_MINIMA.items():
            actual = dict(family_health.get(family) or {})
            meets_minima = {
                key: int(actual.get(key) or 0) >= int(expected)
                for key, expected in minima.items()
            }
            if not all(meets_minima.values()):
                minima_passed = False
            families[family] = {
                "actual": {key: int(actual.get(key) or 0) for key in minima},
                "minimum": {key: int(expected) for key, expected in minima.items()},
                "meets_minima": dict(meets_minima),
                "missing_reciprocal_links": int(actual.get("missing_reciprocal_links") or 0),
                "incomplete_validator_coverage": int(actual.get("incomplete_validator_coverage") or 0),
            }
        passed = bool(
            minima_passed
            and missing_explicit_route_family == 0
            and total_missing_reciprocal_links == 0
            and total_incomplete_validator_coverage == 0
        )
        return {
            "generated_at": time.time(),
            "build_backend": BUILD_BACKEND,
            "build_feed_signature": str(build_feed_signature),
            "feed_source_signature": str(feed_source_signature),
            "route_valid_count": int(audit_summary.get("route_valid_count") or 0),
            "route_neutral_count": int(audit_summary.get("route_neutral_count") or 0),
            "route_broken_count": int(audit_summary.get("route_broken_count") or 0),
            "missing_explicit_route_family": int(missing_explicit_route_family),
            "total_missing_reciprocal_links": int(total_missing_reciprocal_links),
            "total_incomplete_validator_coverage": int(total_incomplete_validator_coverage),
            "families": families,
            "route_family_health": family_health,
            "passed": passed,
        }

    def _compile_build_ref_hashes_from_feed_source(
        self,
        *,
        rows: bytes,
        ref_rows: bytes,
        star_count: int,
        forward_ref_count: int,
    ) -> tuple[bytes, dict[str, Any]]:
        row_bytes = int(star_count) * RAW_CATALOG_INPUT_ENTRY_BYTES
        decode_s = 0.0
        finalize_s = 0.0
        materialize_s = 0.0
        hash_index_s = 0.0
        reverse_ref_hash_s = 0.0
        temp_table = GalaxyVRAMTable(max_stars=max(int(star_count), 1))
        temp_table.prepare_gpu_build(star_count=int(star_count), ref_capacity=1)
        chunk_count = (int(star_count) + GPU_BUILD_CHUNK_SIZE - 1) // GPU_BUILD_CHUNK_SIZE
        input_buffers = [loader.gpu_malloc(max(1, GPU_BUILD_CHUNK_SIZE * RAW_CATALOG_INPUT_ENTRY_BYTES)) for _ in range(2)]
        raw_buffers = [loader.gpu_malloc(max(1, GPU_BUILD_CHUNK_SIZE * RAW_CATALOG_INPUT_ENTRY_BYTES)) for _ in range(2)]
        finalized_buffers = [loader.gpu_malloc(max(1, GPU_BUILD_CHUNK_SIZE * FINALIZED_CATALOG_INPUT_ENTRY_BYTES)) for _ in range(2)]
        pinned_buffers = [loader.PinnedHostBuffer(max(1, GPU_BUILD_CHUNK_SIZE * RAW_CATALOG_INPUT_ENTRY_BYTES)) for _ in range(2)]
        streams = [loader.create_stream(), loader.create_stream()]
        hash_keys_ptr = None
        hash_values_ptr = None
        collision_flag_ptr = None
        build_ref_hash_ptr = None
        feed_source_ref_ptr = None
        unresolved_source_ptr = None
        unresolved_target_hash_ptr = None
        unresolved_count_ptr = None
        try:
            for chunk_index in range(chunk_count):
                buffer_index = chunk_index % 2
                if chunk_index >= 2:
                    loader.stream_synchronize(streams[buffer_index])
                chunk_start = chunk_index * GPU_BUILD_CHUNK_SIZE
                chunk_size = min(GPU_BUILD_CHUNK_SIZE, int(star_count) - chunk_start)
                bytes_to_copy = int(chunk_size) * RAW_CATALOG_INPUT_ENTRY_BYTES
                chunk = rows[chunk_start * RAW_CATALOG_INPUT_ENTRY_BYTES : (chunk_start * RAW_CATALOG_INPUT_ENTRY_BYTES) + bytes_to_copy]
                target_view = pinned_buffers[buffer_index].view().cast("B")[:bytes_to_copy]
                target_view[:] = chunk
                if getattr(pinned_buffers[buffer_index], "pinned", False):
                    loader.memcpy_htod_async(
                        input_buffers[buffer_index],
                        pinned_buffers[buffer_index].ptr,
                        bytes_to_copy,
                        stream=streams[buffer_index],
                    )
                else:
                    loader.memcpy_htod(input_buffers[buffer_index], pinned_buffers[buffer_index].ptr, bytes_to_copy)
                decode_t0 = time.perf_counter()
                self.materializer.decode_build_rows(
                    build_rows_ptr=input_buffers[buffer_index],
                    raw_input_ptr=raw_buffers[buffer_index],
                    entry_count=chunk_size,
                    stream=streams[buffer_index],
                )
                decode_s += float(time.perf_counter() - decode_t0)
                finalize_t0 = time.perf_counter()
                self.materializer.finalize_chunk(
                    raw_input_ptr=raw_buffers[buffer_index],
                    finalized_input_ptr=finalized_buffers[buffer_index],
                    entry_count=chunk_size,
                    stream=streams[buffer_index],
                )
                finalize_s += float(time.perf_counter() - finalize_t0)
                materialize_t0 = time.perf_counter()
                self.materializer.materialize_chunk(
                    galaxy_table_ptr=temp_table.gpu_ptr,
                    input_ptr=finalized_buffers[buffer_index],
                    entry_count=chunk_size,
                    star_offset=chunk_start,
                    router_offsets_ptr=temp_table.router_offsets_ptr,
                    router_counts_ptr=temp_table.router_counts_ptr,
                    executor_offsets_ptr=temp_table.executor_offsets_ptr,
                    executor_counts_ptr=temp_table.executor_counts_ptr,
                    validator_offsets_ptr=temp_table.validator_offsets_ptr,
                    validator_counts_ptr=temp_table.validator_counts_ptr,
                    anti_pattern_offsets_ptr=temp_table.anti_pattern_offsets_ptr,
                    anti_pattern_counts_ptr=temp_table.anti_pattern_counts_ptr,
                    stream=streams[buffer_index],
                )
                materialize_s += float(time.perf_counter() - materialize_t0)
                if (
                    chunk_index == 0
                    or chunk_index + 1 == chunk_count
                    or ((chunk_index + 1) % GPU_BUILD_PROGRESS_EVERY) == 0
                ):
                    self._emit_rebuild_progress(
                        f"decode-feed-source {chunk_index + 1}/{max(chunk_count, 1)} "
                        f"(stars={chunk_start + chunk_size}/{star_count})"
                    )
            for stream in streams:
                loader.stream_synchronize(stream)

            hash_capacity = self._hash_table_capacity(int(star_count))
            hash_keys_ptr = loader.gpu_malloc(max(1, hash_capacity * 8))
            hash_values_ptr = loader.gpu_malloc(max(1, hash_capacity * 4))
            collision_flag_ptr = loader.gpu_malloc(8)
            zero_hash_keys = bytearray(hash_capacity * 8)
            zero_hash_values = bytearray(hash_capacity * 4)
            zero_collision = bytearray(8)
            loader.memcpy_htod(
                hash_keys_ptr,
                ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(zero_hash_keys))),
                len(zero_hash_keys),
            )
            loader.memcpy_htod(
                hash_values_ptr,
                ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(zero_hash_values))),
                len(zero_hash_values),
            )
            loader.memcpy_htod(
                collision_flag_ptr,
                ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(zero_collision))),
                len(zero_collision),
            )
            hash_stream = loader.create_stream()
            try:
                hash_t0 = time.perf_counter()
                self.materializer.build_star_hash_index(
                    galaxy_table_ptr=temp_table.gpu_ptr,
                    star_count=int(star_count),
                    hash_keys_ptr=hash_keys_ptr,
                    hash_values_ptr=hash_values_ptr,
                    table_capacity=hash_capacity,
                    collision_flag_ptr=collision_flag_ptr,
                    stream=hash_stream,
                )
                loader.stream_synchronize(hash_stream)
                hash_index_s = float(time.perf_counter() - hash_t0)
            finally:
                loader.destroy_stream(hash_stream)
            collision_values = self._read_u64_device(collision_flag_ptr, 1)
            collision_hash = int(collision_values[0]) if collision_values else 0
            if collision_hash:
                raise ValueError(f"sovereign_feed_source_hash_collision:{collision_hash:016x}")

            if int(forward_ref_count) <= 0:
                return b"", {
                    "decode_feed_source_s": decode_s,
                    "boot_finalize_s": finalize_s,
                    "star_materialize_s": materialize_s,
                    "build_star_hash_index_s": hash_index_s,
                    "expand_reverse_ref_hashes_s": reverse_ref_hash_s,
                }
            ref_bytes = int(forward_ref_count) * FEED_SOURCE_REF_BYTES
            if len(ref_rows) != ref_bytes:
                raise RuntimeError(
                    f"sovereign_feed_source_ref_size_mismatch:{len(ref_rows)}:{ref_bytes}"
                )
            ref_host = loader.PinnedHostBuffer(max(1, ref_bytes))
            build_ref_hash_ptr = loader.gpu_malloc(max(1, int(forward_ref_count) * 2 * BUILD_REF_HASH_BYTES))
            feed_source_ref_ptr = loader.gpu_malloc(max(1, ref_bytes))
            unresolved_source_ptr = loader.gpu_malloc(max(1, int(forward_ref_count) * 4))
            unresolved_target_hash_ptr = loader.gpu_malloc(max(1, int(forward_ref_count) * 8))
            unresolved_count_ptr = loader.gpu_malloc(4)
            try:
                ref_view = ref_host.view().cast("B")[:ref_bytes]
                ref_view[:] = ref_rows
                zero_unresolved = bytearray(4)
                loader.memcpy_htod(feed_source_ref_ptr, ref_host.ptr, ref_bytes)
                loader.memcpy_htod(
                    unresolved_count_ptr,
                    ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(zero_unresolved))),
                    4,
                )
                ref_stream = loader.create_stream()
                try:
                    reverse_t0 = time.perf_counter()
                    self.materializer.expand_reverse_ref_hashes(
                        feed_source_ref_rows_ptr=feed_source_ref_ptr,
                        ref_count=int(forward_ref_count),
                        star_count=int(star_count),
                        hash_keys_ptr=hash_keys_ptr,
                        hash_values_ptr=hash_values_ptr,
                        table_capacity=hash_capacity,
                        galaxy_table_ptr=temp_table.gpu_ptr,
                        build_ref_hash_rows_ptr=build_ref_hash_ptr,
                        unresolved_source_ptr=unresolved_source_ptr,
                        unresolved_target_hash_ptr=unresolved_target_hash_ptr,
                        unresolved_count_ptr=unresolved_count_ptr,
                        stream=ref_stream,
                    )
                    loader.stream_synchronize(ref_stream)
                    reverse_ref_hash_s = float(time.perf_counter() - reverse_t0)
                finally:
                    loader.destroy_stream(ref_stream)
                unresolved_count_values = self._read_u32_device(unresolved_count_ptr, 1)
                unresolved_count = int(unresolved_count_values[0]) if unresolved_count_values else 0
                if unresolved_count > 0:
                    source_rows = self._read_u32_device(unresolved_source_ptr, unresolved_count)
                    target_hashes = self._read_u64_device(unresolved_target_hash_ptr, unresolved_count)
                    errors = [
                        f"source_index={int(source_index)}:target_hash={int(target_hash):016x}"
                        for source_index, target_hash in zip(source_rows, target_hashes)
                    ]
                    sample = ", ".join(errors[:12])
                    if len(errors) > 12:
                        sample += f", ... (+{len(errors) - 12} more)"
                    raise ValueError(f"sovereign_feed_source_unresolved_refs:{sample}")
                build_ref_hash_bytes = bytearray(int(forward_ref_count) * 2 * BUILD_REF_HASH_BYTES)
                loader.memcpy_dtoh(
                    ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(build_ref_hash_bytes))),
                    build_ref_hash_ptr,
                    len(build_ref_hash_bytes),
                )
            finally:
                ref_host.close()
                if feed_source_ref_ptr is not None:
                    loader.gpu_free(feed_source_ref_ptr)
                if build_ref_hash_ptr is not None:
                    loader.gpu_free(build_ref_hash_ptr)
                if unresolved_source_ptr is not None:
                    loader.gpu_free(unresolved_source_ptr)
                if unresolved_target_hash_ptr is not None:
                    loader.gpu_free(unresolved_target_hash_ptr)
                if unresolved_count_ptr is not None:
                    loader.gpu_free(unresolved_count_ptr)
            return bytes(build_ref_hash_bytes), {
                "decode_feed_source_s": decode_s,
                "boot_finalize_s": finalize_s,
                "star_materialize_s": materialize_s,
                "build_star_hash_index_s": hash_index_s,
                "expand_reverse_ref_hashes_s": reverse_ref_hash_s,
            }
        finally:
            for stream in streams:
                try:
                    loader.destroy_stream(stream)
                except Exception:
                    pass
            for pinned in pinned_buffers:
                pinned.close()
            for device_buffer in input_buffers + raw_buffers + finalized_buffers:
                try:
                    loader.gpu_free(device_buffer)
                except Exception:
                    pass
            if hash_keys_ptr is not None:
                try:
                    loader.gpu_free(hash_keys_ptr)
                except Exception:
                    pass
            if hash_values_ptr is not None:
                try:
                    loader.gpu_free(hash_values_ptr)
                except Exception:
                    pass
            if collision_flag_ptr is not None:
                try:
                    loader.gpu_free(collision_flag_ptr)
                except Exception:
                    pass
            temp_table.close()

    def _host_stars_from_build_ref_hashes(
        self,
        base_host_stars: list[dict[str, Any]],
        build_ref_hash_rows: bytes,
    ) -> list[dict[str, Any]]:
        host_stars = [self._strip_role_refs(star) for star in list(base_host_stars or [])]
        hash_to_index: dict[int, int] = {}
        for index, star in enumerate(host_stars):
            star_hash = int(star.get("star_hash") or 0)
            if star_hash in hash_to_index and hash_to_index[star_hash] != index:
                first_index = int(hash_to_index[star_hash])
                first_id = str(host_stars[first_index].get("id") or first_index)
                current_id = str(star.get("id") or index)
                if first_id == current_id:
                    raise ValueError(
                        "sovereign_build_feed_duplicate_id:"
                        f"{current_id}:first_index={first_index}:second_index={index}"
                    )
                raise ValueError(
                    "sovereign_build_feed_hash_collision:"
                    f"{star_hash:016x}:first_id={first_id}:second_id={current_id}:"
                    f"first_index={first_index}:second_index={index}"
                )
            hash_to_index[star_hash] = index
        role_key_by_type = {value: key for key, value in ROLE_TYPE_BY_KEY.items()}
        for offset in range(0, len(build_ref_hash_rows), BUILD_REF_HASH_BYTES):
            source_index, role_type, target_hash = BUILD_REF_HASH_STRUCT.unpack_from(build_ref_hash_rows, offset)
            if int(source_index) < 0 or int(source_index) >= len(host_stars):
                raise ValueError(f"sovereign_build_feed_invalid_source_index:{int(source_index)}")
            target_index = hash_to_index.get(int(target_hash))
            if target_index is None:
                source_id = str(host_stars[int(source_index)].get("id") or source_index)
                raise ValueError(
                    f"sovereign_build_feed_host_ref_invalid:{source_id}:target_hash={int(target_hash):016x}"
                )
            key = role_key_by_type.get(int(role_type))
            if not key:
                raise ValueError(f"sovereign_build_feed_invalid_role_type:{int(role_type)}")
            refs = list(host_stars[int(source_index)].get(key) or [])
            if int(target_index) not in refs:
                refs.append(int(target_index))
            host_stars[int(source_index)][key] = refs
        reverse_key = {
            "router_refs": "router_refs",
            "executor_refs": "router_refs",
            "validator_refs": "executor_refs",
            "anti_pattern_refs": "anti_pattern_refs",
        }
        for source_index, star in enumerate(host_stars):
            for key, target_key in reverse_key.items():
                for target_index in list(star.get(key) or []):
                    if int(target_index) < 0 or int(target_index) >= len(host_stars):
                        continue
                    target_refs = list(host_stars[int(target_index)].get(target_key) or [])
                    if int(source_index) not in target_refs:
                        target_refs.append(int(source_index))
                        host_stars[int(target_index)][target_key] = target_refs
        return self._finalize_component_refs(host_stars)

    def _pack_build_ref_hash_rows(self, host_stars: list[dict[str, Any]]) -> bytes:
        final_ref_count = sum(
            len(list(star.get(key) or []))
            for star in host_stars
            for key in ROLE_TYPE_BY_KEY
        )
        if final_ref_count <= 0:
            return b""
        build_ref_rows = bytearray(final_ref_count * BUILD_REF_HASH_BYTES)
        ref_index = 0
        for source_index, star in enumerate(host_stars):
            for key, role_type in ROLE_TYPE_BY_KEY.items():
                for target_index in list(star.get(key) or []):
                    target_star = host_stars[int(target_index)]
                    BUILD_REF_HASH_STRUCT.pack_into(
                        build_ref_rows,
                        ref_index * BUILD_REF_HASH_BYTES,
                        int(source_index),
                        int(role_type),
                        int(target_star["star_hash"]),
                    )
                    ref_index += 1
        return bytes(build_ref_rows)

    def refresh_feed_source(
        self,
        *,
        galaxy_names: list[str] | None = None,
        catalog: list[dict[str, Any]] | None = None,
        worker_count: int | None = None,
        chunk_size: int | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        signature, resolved_names = self._expected_build_feed_signature()
        if galaxy_names is not None:
            resolved_names = list(galaxy_names)
            signature = str(self.knowledgeverse._gpu_flat_cache_signature(resolved_names))
        if catalog is None and not force:
            cached_summary = self._cached_feed_source_summary(signature)
            if cached_summary is not None:
                self._feed_source_signature = str(signature)
                return cached_summary
        if catalog is None:
            catalog = list(self.knowledgeverse.build_gpu_catalog_only(galaxy_names=resolved_names))
        compile_t0 = time.perf_counter()
        catalog = list(catalog)
        catalog_signature = self._current_signature(catalog)
        resolved_chunk_size = max(1, int(chunk_size or DEFAULT_FEED_SOURCE_CHUNK_SIZE))
        requested_workers = max(1, int(worker_count or self._default_feed_source_worker_count()))
        tasks = [
            (start, min(resolved_chunk_size, len(catalog) - start))
            for start in range(0, len(catalog), resolved_chunk_size)
        ]
        can_parallelize = bool(
            os.name == "posix"
            and len(tasks) > 1
            and requested_workers > 1
        )
        actual_worker_count = int(requested_workers if can_parallelize else 1)
        self._emit_rebuild_progress(
            f"feed-source-extract start chunks={len(tasks)} workers={actual_worker_count} chunk_size={resolved_chunk_size}"
        )
        extract_t0 = time.perf_counter()
        source_host_stars: list[dict[str, Any]] = []
        rows = bytearray()
        pending_refs: list[tuple[int, int, int, Any]] = []
        metadata_errors: list[str] = []
        try:
            if can_parallelize:
                _FEED_SOURCE_PARALLEL_STATE["runtime"] = self
                _FEED_SOURCE_PARALLEL_STATE["catalog"] = catalog
                ctx = mp.get_context("fork")
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=actual_worker_count,
                    mp_context=ctx,
                ) as executor:
                    for task_index, result in enumerate(executor.map(_feed_source_parallel_chunk, tasks), start=1):
                        chunk_stars = [dict(star) for star in list(result.get("stars") or []) if isinstance(star, dict)]
                        chunk_bytes = bytes(result.get("row_bytes") or b"")
                        chunk_pending = list(result.get("pending_refs") or [])
                        metadata_errors.extend(str(value) for value in list(result.get("metadata_errors") or []))
                        base_index = len(source_host_stars)
                        source_host_stars.extend(chunk_stars)
                        rows.extend(chunk_bytes)
                        for local_source_index, role_type, reverse_role_type, ref_value in chunk_pending:
                            pending_refs.append(
                                (
                                    base_index + int(local_source_index),
                                    int(role_type),
                                    int(reverse_role_type),
                                    ref_value,
                                )
                            )
                        self._emit_rebuild_progress(
                            f"feed-source-extract chunk {task_index}/{len(tasks)} stars={len(source_host_stars)}"
                        )
            else:
                for task_index, (start, count) in enumerate(tasks, start=1):
                    result = self._compile_feed_source_chunk(catalog, start=start, count=count)
                    chunk_stars = [dict(star) for star in list(result.get("stars") or []) if isinstance(star, dict)]
                    chunk_bytes = bytes(result.get("row_bytes") or b"")
                    chunk_pending = list(result.get("pending_refs") or [])
                    metadata_errors.extend(str(value) for value in list(result.get("metadata_errors") or []))
                    base_index = len(source_host_stars)
                    source_host_stars.extend(chunk_stars)
                    rows.extend(chunk_bytes)
                    for local_source_index, role_type, reverse_role_type, ref_value in chunk_pending:
                        pending_refs.append(
                            (
                                base_index + int(local_source_index),
                                int(role_type),
                                int(reverse_role_type),
                                ref_value,
                            )
                        )
                    self._emit_rebuild_progress(
                        f"feed-source-extract chunk {task_index}/{len(tasks)} stars={len(source_host_stars)}"
                    )
        finally:
            _FEED_SOURCE_PARALLEL_STATE.clear()
        if metadata_errors:
            sample = ", ".join(metadata_errors[:12])
            if len(metadata_errors) > 12:
                sample += f", ... (+{len(metadata_errors) - 12} more)"
            raise ValueError(f"sovereign_build_metadata_invalid:{sample}")
        hash_to_index: dict[int, int] = {}
        role_key_by_type = {value: key for key, value in ROLE_TYPE_BY_KEY.items()}
        star_hashes: list[int] = []
        for index, star in enumerate(source_host_stars):
            star_hash = int(star.get("star_hash") or 0)
            if star_hash in hash_to_index and hash_to_index[star_hash] != index:
                first_index = int(hash_to_index[star_hash])
                first_id = str(source_host_stars[first_index].get("id") or first_index)
                current_id = str(star.get("id") or index)
                if first_id == current_id:
                    raise ValueError(
                        "sovereign_feed_source_duplicate_id:"
                        f"{current_id}:first_index={first_index}:second_index={index}"
                    )
                raise ValueError(
                    "sovereign_feed_source_hash_collision:"
                    f"{star_hash:016x}:first_id={first_id}:second_id={current_id}:"
                    f"first_index={first_index}:second_index={index}"
                )
            hash_to_index[star_hash] = index
            star_hashes.append(star_hash)
        ref_rows = bytearray(max(1, len(pending_refs) * FEED_SOURCE_REF_BYTES))
        ref_index = 0
        for pending_index, (source_index, role_type, reverse_role_type, ref_value) in enumerate(pending_refs, start=1):
            target_index = self._resolve_ref_index_from_hash_or_index(
                ref_value,
                hash_to_index,
                len(source_host_stars),
            )
            if target_index is None:
                continue
            key = role_key_by_type.get(int(role_type))
            if not key:
                raise RuntimeError(f"sovereign_feed_source_invalid_role_type:{int(role_type)}")
            refs = list(source_host_stars[int(source_index)].get(key) or [])
            if int(target_index) in refs:
                continue
            refs.append(int(target_index))
            source_host_stars[int(source_index)][key] = refs
            FEED_SOURCE_REF_STRUCT.pack_into(
                ref_rows,
                ref_index * FEED_SOURCE_REF_BYTES,
                int(source_index),
                int(role_type),
                int(star_hashes[int(target_index)]),
                int(reverse_role_type),
            )
            ref_index += 1
            if (pending_index % 25000) == 0 or pending_index == len(pending_refs):
                self._emit_rebuild_progress(
                    f"feed-source-extract refs {pending_index}/{len(pending_refs)} packed={ref_index}"
                )
        self._validate_route_link_coverage(source_host_stars)
        source_host_stars = self._finalize_component_refs(source_host_stars)
        feed_source_extract_s = float(time.perf_counter() - extract_t0)
        self._emit_rebuild_progress(
            f"feed-source-audit stars={len(source_host_stars)} refs={ref_index}"
        )
        audit_t0 = time.perf_counter()
        audit_summary = self._audit_route_capability_rows(
            bytes(rows),
            source_host_stars,
        )
        feed_source_audit_s = float(time.perf_counter() - audit_t0)
        paths = self._feed_source_paths(signature)
        self._gpu_cache_dir().mkdir(parents=True, exist_ok=True)
        self._emit_rebuild_progress(
            f"feed-source-write stars={len(source_host_stars)} refs={ref_index}"
        )
        write_t0 = time.perf_counter()
        rows_payload = bytes(rows)
        refs_payload = bytes(ref_rows[: ref_index * FEED_SOURCE_REF_BYTES])
        manifest = {
            "feed_source_version": int(FEED_SOURCE_VERSION),
            "build_feed_version": int(BUILD_FEED_VERSION),
            "build_backend": BUILD_BACKEND,
            "route_contract_schema_version": int(route_contract.ROUTE_CONTRACT_SCHEMA_VERSION),
            "feed_source_signature": str(signature),
            "catalog_signature": str(catalog_signature),
            "house_signature_base": self._expected_house_signature_base(),
            "default_knowledge_signature": str(
                self.knowledgeverse._house_state_summary.get("default_knowledge_signature") or ""
            ).strip(),
            "star_count": int(len(source_host_stars)),
            "forward_ref_count": int(ref_index),
            "feed_source_extract_s": float(feed_source_extract_s),
            "feed_source_audit_s": float(feed_source_audit_s),
            "feed_source_parallel": bool(can_parallelize),
            "feed_source_worker_count": int(actual_worker_count),
            "feed_source_chunk_size": int(resolved_chunk_size),
            "saved_at": time.time(),
        }
        manifest.update(audit_summary)
        manifest.update(self.materializer.ptx_signatures())
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as write_executor:
            host_future = write_executor.submit(_write_pickle_file, paths["host_stars"], source_host_stars)
            manifest_future = write_executor.submit(_write_json_file, paths["manifest"], manifest)
            paths["rows"].write_bytes(rows_payload)
            paths["refs"].write_bytes(refs_payload)
            host_future.result()
            manifest_future.result()
        feed_source_write_s = float(time.perf_counter() - write_t0)
        manifest["feed_source_write_s"] = float(feed_source_write_s)
        _write_json_file(paths["manifest"], manifest)
        prune_summary = self._prune_stale_feed_source_cache(signature)
        self._feed_source_signature = str(signature)
        summary = {
            "status": "ready",
            "mode": "feed_source_compile",
            "build_backend": BUILD_BACKEND,
            "feed_source_signature": str(signature),
            "catalog_signature": str(catalog_signature),
            "star_count": int(len(source_host_stars)),
            "forward_ref_count": int(ref_index),
            "elapsed_s": float(time.perf_counter() - compile_t0),
            "feed_source_extract_s": float(feed_source_extract_s),
            "feed_source_audit_s": float(feed_source_audit_s),
            "feed_source_write_s": float(feed_source_write_s),
            "feed_source_parallel": bool(can_parallelize),
            "feed_source_worker_count": int(actual_worker_count),
            "feed_source_chunk_size": int(resolved_chunk_size),
            "prune_summary": prune_summary,
        }
        summary.update(audit_summary)
        summary.update(self.materializer.ptx_signatures())
        return summary

    def refresh_build_feed(
        self,
        *,
        galaxy_names: list[str] | None = None,
        catalog: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        del catalog
        signature, resolved_names = self._expected_build_feed_signature()
        if galaxy_names is not None:
            resolved_names = list(galaxy_names)
            signature = str(self.knowledgeverse._gpu_flat_cache_signature(resolved_names))
        compile_t0 = time.perf_counter()
        feed_source_t0 = time.perf_counter()
        feed_source = self._load_feed_source(signature)
        load_feed_source_s = float(time.perf_counter() - feed_source_t0)
        manifest = dict(feed_source.get("manifest") or {})
        source_paths = dict(feed_source.get("paths") or {})
        source_host_stars = [dict(star) for star in list(feed_source.get("host_stars") or [])]
        star_count = int(manifest.get("star_count") or len(source_host_stars) or 0)
        forward_ref_count = int(manifest.get("forward_ref_count") or 0)
        if star_count <= 0:
            raise RuntimeError("sovereign_feed_source_invalid:star_count")
        self._emit_rebuild_progress(
            f"load-feed-source stars={star_count} forward_refs={forward_ref_count}"
        )
        rows = Path(source_paths["rows"]).read_bytes()
        ref_rows = Path(source_paths["refs"]).read_bytes()
        build_rows_t0 = time.perf_counter()
        build_rows = bytes(rows)
        compile_build_rows_s = float(time.perf_counter() - build_rows_t0)
        build_ref_hash_rows, compiler_summary = self._compile_build_ref_hashes_from_feed_source(
            rows=build_rows,
            ref_rows=ref_rows,
            star_count=star_count,
            forward_ref_count=forward_ref_count,
        )
        host_stars = self._host_stars_from_build_ref_hashes(source_host_stars, build_ref_hash_rows)
        build_ref_hash_rows = self._pack_build_ref_hash_rows(host_stars)
        final_ref_count = len(build_ref_hash_rows) // BUILD_REF_HASH_BYTES
        audit_summary = self._audit_host_star_route_capability(host_stars)
        route_audit = self._build_meaning_family_route_audit(
            host_stars=host_stars,
            audit_summary=audit_summary,
            build_feed_signature=str(signature),
            feed_source_signature=str(manifest.get("feed_source_signature") or signature),
        )
        closure_audit = self._build_meaning_route_closure_audit(
            host_stars=host_stars,
            audit_summary=audit_summary,
            build_feed_signature=str(signature),
            feed_source_signature=str(manifest.get("feed_source_signature") or signature),
        )
        coverage_audit = self._build_meaning_knowledge_coverage_audit(
            host_stars=host_stars,
            build_feed_signature=str(signature),
            feed_source_signature=str(manifest.get("feed_source_signature") or signature),
        )
        route_audit_path = self._meaning_family_route_audit_path()
        closure_audit_path = self._meaning_route_closure_audit_path()
        coverage_audit_path = self._meaning_knowledge_coverage_audit_path()
        route_audit_path.parent.mkdir(parents=True, exist_ok=True)
        route_audit_path.write_text(
            json.dumps(route_audit, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        closure_audit_path.write_text(
            json.dumps(closure_audit, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        coverage_audit_path.write_text(
            json.dumps(coverage_audit, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        catalog_signature = str(manifest.get("catalog_signature") or "")
        paths = self._build_feed_paths(signature)
        self._gpu_cache_dir().mkdir(parents=True, exist_ok=True)
        paths["rows"].write_bytes(build_rows)
        paths["ref_hashes"].write_bytes(build_ref_hash_rows)
        with paths["host_stars"].open("wb") as handle:
            pickle.dump(host_stars, handle, protocol=pickle.HIGHEST_PROTOCOL)
        build_manifest = {
            "build_feed_version": int(BUILD_FEED_VERSION),
            "feed_source_version": int(manifest.get("feed_source_version") or FEED_SOURCE_VERSION),
            "build_backend": BUILD_BACKEND,
            "route_contract_schema_version": int(route_contract.ROUTE_CONTRACT_SCHEMA_VERSION),
            "build_feed_signature": str(signature),
            "feed_source_signature": str(manifest.get("feed_source_signature") or signature),
            "catalog_signature": str(catalog_signature),
            "house_signature_base": self._expected_house_signature_base(),
            "default_knowledge_signature": str(
                self.knowledgeverse._house_state_summary.get("default_knowledge_signature") or ""
            ).strip(),
            "star_count": int(star_count),
            "forward_ref_count": int(forward_ref_count),
            "final_ref_count": int(final_ref_count),
            "reverse_symlinks_compiled": True,
            "meaning_family_route_audit_path": str(route_audit_path),
            "meaning_family_route_audit_passed": bool(route_audit.get("passed")),
            "meaning_route_closure_audit_path": str(closure_audit_path),
            "meaning_route_closure_audit_passed": bool(closure_audit.get("passed")),
            "meaning_knowledge_coverage_audit_path": str(coverage_audit_path),
            "meaning_knowledge_coverage_audit_passed": bool(coverage_audit.get("passed")),
            "saved_at": time.time(),
        }
        build_manifest.update(audit_summary)
        build_manifest.update(compiler_summary)
        build_manifest.update(self.materializer.ptx_signatures())
        paths["manifest"].write_text(
            json.dumps(build_manifest, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        prune_summary = self._prune_stale_build_feed_cache(signature)
        self._build_feed_signature = str(signature)
        self._feed_source_signature = str(manifest.get("feed_source_signature") or signature)
        self._last_build_feed_manifest = dict(build_manifest)
        summary = {
            "status": "ready",
            "mode": "build_feed_compile",
            "build_backend": BUILD_BACKEND,
            "build_feed_signature": str(signature),
            "feed_source_signature": str(manifest.get("feed_source_signature") or signature),
            "catalog_signature": str(catalog_signature),
            "star_count": int(star_count),
            "forward_ref_count": int(forward_ref_count),
            "final_ref_count": int(final_ref_count),
            "load_feed_source_s": load_feed_source_s,
            "compile_build_rows_s": compile_build_rows_s,
            "elapsed_s": float(time.perf_counter() - compile_t0),
            "prune_summary": prune_summary,
            "meaning_family_route_audit_path": str(route_audit_path),
            "meaning_family_route_audit_passed": bool(route_audit.get("passed")),
            "meaning_route_closure_audit_path": str(closure_audit_path),
            "meaning_route_closure_audit_passed": bool(closure_audit.get("passed")),
            "meaning_knowledge_coverage_audit_path": str(coverage_audit_path),
            "meaning_knowledge_coverage_audit_passed": bool(coverage_audit.get("passed")),
        }
        summary.update(audit_summary)
        summary["meaning_family_route_audit"] = route_audit
        summary["meaning_route_closure_audit"] = closure_audit
        summary["meaning_knowledge_coverage_audit"] = coverage_audit
        summary.update(compiler_summary)
        summary.update(self.materializer.ptx_signatures())
        return summary

    def _build_stars_from_catalog(self, catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._build_stars_sovereign(catalog)
        return [dict(star) for star in self._host_stars]

    @staticmethod
    def _read_u32_device(ptr, count: int) -> list[int]:
        total = max(0, int(count))
        if ptr is None or total <= 0:
            return []
        payload = bytearray(total * 4)
        loader.memcpy_dtoh(ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(payload))), ptr, len(payload))
        return list(struct.unpack_from(f"<{total}I", payload, 0))

    @staticmethod
    def _read_u64_device(ptr, count: int) -> list[int]:
        total = max(0, int(count))
        if ptr is None or total <= 0:
            return []
        payload = bytearray(total * 8)
        loader.memcpy_dtoh(ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(payload))), ptr, len(payload))
        return list(struct.unpack_from(f"<{total}Q", payload, 0))

    @staticmethod
    def _hash_table_capacity(star_count: int) -> int:
        target = max(8, int(star_count) * 2)
        capacity = 1
        while capacity < target:
            capacity <<= 1
        return int(capacity)

    def _build_stars_from_build_feed(self, build_feed: dict[str, Any]) -> dict[str, Any]:
        manifest = dict(build_feed.get("manifest") or {})
        paths = dict(build_feed.get("paths") or {})
        host_stars = [dict(star) for star in list(build_feed.get("host_stars") or [])]
        star_count = int(manifest.get("star_count") or len(host_stars) or 0)
        forward_ref_count = int(manifest.get("forward_ref_count") or 0)
        final_ref_count = int(manifest.get("final_ref_count") or 0)
        reverse_symlinks_compiled = bool(manifest.get("reverse_symlinks_compiled"))
        if star_count <= 0:
            raise RuntimeError("sovereign_build_feed_invalid:star_count")
        self._emit_rebuild_progress(
            f"load-build-feed stars={star_count} forward_refs={forward_ref_count} final_refs={final_ref_count}"
        )

        self.star_table.prepare_gpu_build(
            star_count=star_count,
            ref_capacity=max(1, final_ref_count),
        )

        row_chunk_bytes = max(1, GPU_BUILD_CHUNK_SIZE * RAW_CATALOG_INPUT_ENTRY_BYTES)
        pinned_buffers = [loader.PinnedHostBuffer(row_chunk_bytes), loader.PinnedHostBuffer(row_chunk_bytes)]
        input_buffers = [loader.gpu_malloc(row_chunk_bytes), loader.gpu_malloc(row_chunk_bytes)]
        finalized_buffers = [loader.gpu_malloc(max(1, GPU_BUILD_CHUNK_SIZE * FINALIZED_CATALOG_INPUT_ENTRY_BYTES)), loader.gpu_malloc(max(1, GPU_BUILD_CHUNK_SIZE * FINALIZED_CATALOG_INPUT_ENTRY_BYTES))]
        streams = [loader.create_stream(), loader.create_stream()]
        decode_s = 0.0
        finalize_s = 0.0
        materialize_s = 0.0
        build_rows_t0 = time.perf_counter()
        chunk_count = (star_count + GPU_BUILD_CHUNK_SIZE - 1) // GPU_BUILD_CHUNK_SIZE
        try:
            with Path(paths["rows"]).open("rb") as row_handle:
                for chunk_index in range(chunk_count):
                    buffer_index = chunk_index % 2
                    if chunk_index >= 2:
                        loader.stream_synchronize(streams[buffer_index])
                    chunk_start = chunk_index * GPU_BUILD_CHUNK_SIZE
                    chunk_size = min(GPU_BUILD_CHUNK_SIZE, star_count - chunk_start)
                    bytes_to_copy = int(chunk_size) * RAW_CATALOG_INPUT_ENTRY_BYTES
                    target_view = pinned_buffers[buffer_index].view()[:bytes_to_copy]
                    read_bytes = row_handle.readinto(target_view)
                    if int(read_bytes or 0) != bytes_to_copy:
                        raise RuntimeError(
                            f"sovereign_build_feed_rows_short_read:{chunk_index}:{int(read_bytes or 0)}:{bytes_to_copy}"
                        )
                    if getattr(pinned_buffers[buffer_index], "pinned", False):
                        loader.memcpy_htod_async(
                            input_buffers[buffer_index],
                            pinned_buffers[buffer_index].ptr,
                            bytes_to_copy,
                            stream=streams[buffer_index],
                        )
                    else:
                        loader.memcpy_htod(
                            input_buffers[buffer_index],
                            pinned_buffers[buffer_index].ptr,
                            bytes_to_copy,
                        )
                    decode_t0 = time.perf_counter()
                    self.materializer.decode_build_rows(
                        build_rows_ptr=input_buffers[buffer_index],
                        raw_input_ptr=input_buffers[buffer_index],
                        entry_count=chunk_size,
                        stream=streams[buffer_index],
                    )
                    decode_s += float(time.perf_counter() - decode_t0)
                    finalize_t0 = time.perf_counter()
                    self.materializer.finalize_chunk(
                        raw_input_ptr=input_buffers[buffer_index],
                        finalized_input_ptr=finalized_buffers[buffer_index],
                        entry_count=chunk_size,
                        stream=streams[buffer_index],
                    )
                    finalize_s += float(time.perf_counter() - finalize_t0)
                    materialize_t0 = time.perf_counter()
                    self.materializer.materialize_chunk(
                        galaxy_table_ptr=self.star_table.gpu_ptr,
                        input_ptr=finalized_buffers[buffer_index],
                        entry_count=chunk_size,
                        star_offset=chunk_start,
                        router_offsets_ptr=self.star_table.router_offsets_ptr,
                        router_counts_ptr=self.star_table.router_counts_ptr,
                        executor_offsets_ptr=self.star_table.executor_offsets_ptr,
                        executor_counts_ptr=self.star_table.executor_counts_ptr,
                        validator_offsets_ptr=self.star_table.validator_offsets_ptr,
                        validator_counts_ptr=self.star_table.validator_counts_ptr,
                        anti_pattern_offsets_ptr=self.star_table.anti_pattern_offsets_ptr,
                        anti_pattern_counts_ptr=self.star_table.anti_pattern_counts_ptr,
                        stream=streams[buffer_index],
                    )
                    materialize_s += float(time.perf_counter() - materialize_t0)
                    if (
                        chunk_index == 0
                        or chunk_index + 1 == chunk_count
                        or ((chunk_index + 1) % GPU_BUILD_PROGRESS_EVERY) == 0
                    ):
                        self._emit_rebuild_progress(
                            f"star-materialize {chunk_index + 1}/{max(chunk_count, 1)} "
                            f"(stars={chunk_start + chunk_size}/{star_count})"
                        )
            for stream in streams:
                loader.stream_synchronize(stream)
        finally:
            for stream in streams:
                try:
                    loader.destroy_stream(stream)
                except Exception:
                    pass
            for pinned in pinned_buffers:
                pinned.close()
            for device_buffer in input_buffers:
                try:
                    loader.gpu_free(device_buffer)
                except Exception:
                    pass
            for finalized_buffer in finalized_buffers:
                try:
                    loader.gpu_free(finalized_buffer)
                except Exception:
                    pass
        build_rows_s = float(time.perf_counter() - build_rows_t0)

        hash_capacity = self._hash_table_capacity(star_count)
        hash_keys_ptr = loader.gpu_malloc(max(1, hash_capacity * 8))
        hash_values_ptr = loader.gpu_malloc(max(1, hash_capacity * 4))
        collision_flag_ptr = loader.gpu_malloc(8)
        zero_hash_keys = bytearray(hash_capacity * 8)
        zero_hash_values = bytearray(hash_capacity * 4)
        zero_collision = bytearray(8)
        loader.memcpy_htod(
            hash_keys_ptr,
            ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(zero_hash_keys))),
            len(zero_hash_keys),
        )
        loader.memcpy_htod(
            hash_values_ptr,
            ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(zero_hash_values))),
            len(zero_hash_values),
        )
        loader.memcpy_htod(
            collision_flag_ptr,
            ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(zero_collision))),
            len(zero_collision),
        )
        hash_index_s = 0.0
        hash_stream = None
        try:
            hash_stream = loader.create_stream()
            hash_t0 = time.perf_counter()
            self.materializer.build_star_hash_index(
                galaxy_table_ptr=self.star_table.gpu_ptr,
                star_count=star_count,
                hash_keys_ptr=hash_keys_ptr,
                hash_values_ptr=hash_values_ptr,
                table_capacity=hash_capacity,
                collision_flag_ptr=collision_flag_ptr,
                stream=hash_stream,
            )
            loader.stream_synchronize(hash_stream)
            hash_index_s = float(time.perf_counter() - hash_t0)
        finally:
            try:
                loader.destroy_stream(hash_stream)
            except Exception:
                pass
        collision_values = self._read_u64_device(collision_flag_ptr, 1)
        collision_hash = int(collision_values[0]) if collision_values else 0
        if collision_hash:
            duplicates: list[str] = []
            by_hash: dict[int, list[str]] = {}
            for star in host_stars:
                star_hash = int(star.get("star_hash") or 0)
                by_hash.setdefault(star_hash, []).append(str(star.get("id") or star.get("name") or star_hash))
            for ids in by_hash.values():
                if len(ids) > 1:
                    duplicates.extend(ids)
            sample = ",".join(duplicates[:12]) if duplicates else hex(collision_hash)
            raise ValueError(f"sovereign_build_feed_hash_collision:{sample}")

        resolve_s = 0.0
        reverse_s = 0.0
        scan_scatter_s = 0.0
        stored_ref_hash_count = int(final_ref_count if reverse_symlinks_compiled else forward_ref_count)
        if stored_ref_hash_count > 0:
            ref_hash_bytes = int(stored_ref_hash_count) * BUILD_REF_HASH_BYTES
            ref_hash_host = loader.PinnedHostBuffer(max(1, ref_hash_bytes))
            ref_hash_device = loader.gpu_malloc(max(1, ref_hash_bytes))
            resolved_ref_device = loader.gpu_malloc(max(1, stored_ref_hash_count * REF_TUPLE_BYTES))
            expanded_ref_device = loader.gpu_malloc(max(1, stored_ref_hash_count * 2 * REF_TUPLE_BYTES))
            unresolved_source_device = loader.gpu_malloc(max(1, stored_ref_hash_count * 4))
            unresolved_target_hash_device = loader.gpu_malloc(max(1, stored_ref_hash_count * 8))
            unresolved_count_device = loader.gpu_malloc(4)
            try:
                zero_unresolved_count = bytearray(4)
                loader.memcpy_htod(
                    unresolved_count_device,
                    ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(zero_unresolved_count))),
                    4,
                )
                with Path(paths["ref_hashes"]).open("rb") as handle:
                    view = ref_hash_host.view()[:ref_hash_bytes]
                    read_bytes = handle.readinto(view)
                if int(read_bytes or 0) != ref_hash_bytes:
                    raise RuntimeError(
                        f"sovereign_build_feed_ref_hashes_short_read:{int(read_bytes or 0)}:{ref_hash_bytes}"
                    )
                ref_stream = None
                ref_stream = loader.create_stream()
                try:
                    if getattr(ref_hash_host, "pinned", False):
                        loader.memcpy_htod_async(
                            ref_hash_device,
                            ref_hash_host.ptr,
                            ref_hash_bytes,
                            stream=ref_stream,
                        )
                    else:
                        loader.memcpy_htod(ref_hash_device, ref_hash_host.ptr, ref_hash_bytes)
                    resolve_t0 = time.perf_counter()
                    self.materializer.resolve_ref_hashes(
                        ref_hash_rows_ptr=ref_hash_device,
                        ref_count=stored_ref_hash_count,
                        star_count=star_count,
                        hash_keys_ptr=hash_keys_ptr,
                        hash_values_ptr=hash_values_ptr,
                        table_capacity=hash_capacity,
                        router_counts_ptr=self.star_table.router_counts_ptr,
                        executor_counts_ptr=self.star_table.executor_counts_ptr,
                        validator_counts_ptr=self.star_table.validator_counts_ptr,
                        anti_pattern_counts_ptr=self.star_table.anti_pattern_counts_ptr,
                        resolved_ref_tuples_ptr=resolved_ref_device,
                        unresolved_source_ptr=unresolved_source_device,
                        unresolved_target_hash_ptr=unresolved_target_hash_device,
                        unresolved_count_ptr=unresolved_count_device,
                        stream=ref_stream,
                    )
                    resolve_s = float(time.perf_counter() - resolve_t0)
                    tuple_ptr = resolved_ref_device
                    tuple_count = stored_ref_hash_count
                    if not reverse_symlinks_compiled:
                        reverse_t0 = time.perf_counter()
                        self.materializer.expand_reverse_symlinks(
                            forward_ref_tuples_ptr=resolved_ref_device,
                            ref_count=stored_ref_hash_count,
                            star_count=star_count,
                            router_counts_ptr=self.star_table.router_counts_ptr,
                            executor_counts_ptr=self.star_table.executor_counts_ptr,
                            validator_counts_ptr=self.star_table.validator_counts_ptr,
                            anti_pattern_counts_ptr=self.star_table.anti_pattern_counts_ptr,
                            expanded_ref_tuples_ptr=expanded_ref_device,
                            stream=ref_stream,
                        )
                        reverse_s = float(time.perf_counter() - reverse_t0)
                        tuple_ptr = expanded_ref_device
                        tuple_count = stored_ref_hash_count * 2
                    loader.stream_synchronize(ref_stream)
                finally:
                    try:
                        loader.destroy_stream(ref_stream)
                    except Exception:
                        pass
                unresolved_count_values = self._read_u32_device(unresolved_count_device, 1)
                unresolved_count = int(unresolved_count_values[0]) if unresolved_count_values else 0
                if unresolved_count > 0:
                    source_rows = self._read_u32_device(unresolved_source_device, unresolved_count)
                    target_hashes = self._read_u64_device(unresolved_target_hash_device, unresolved_count)
                    errors: list[str] = []
                    for source_index, target_hash in zip(source_rows, target_hashes):
                        if int(source_index) == 0xFFFFFFFF:
                            continue
                        source_id = (
                            str(host_stars[int(source_index)].get("id") or host_stars[int(source_index)].get("name") or source_index)
                            if 0 <= int(source_index) < len(host_stars)
                            else f"source_index:{int(source_index)}"
                        )
                        errors.append(f"{source_id}:target_hash={int(target_hash):016x}")
                    sample = ", ".join(errors[:12])
                    if len(errors) > 12:
                        sample += f", ... (+{len(errors) - 12} more)"
                    raise ValueError(f"sovereign_build_feed_unresolved_refs:{sample}")
                scan_t0 = time.perf_counter()
                self.materializer.scan_offsets(
                    star_count=star_count,
                    router_counts_ptr=self.star_table.router_counts_ptr,
                    router_offsets_ptr=self.star_table.router_offsets_ptr,
                    executor_counts_ptr=self.star_table.executor_counts_ptr,
                    executor_offsets_ptr=self.star_table.executor_offsets_ptr,
                    validator_counts_ptr=self.star_table.validator_counts_ptr,
                    validator_offsets_ptr=self.star_table.validator_offsets_ptr,
                    anti_pattern_counts_ptr=self.star_table.anti_pattern_counts_ptr,
                    anti_pattern_offsets_ptr=self.star_table.anti_pattern_offsets_ptr,
                )
                self.materializer.scatter_refs(
                    galaxy_table_ptr=self.star_table.gpu_ptr,
                    ref_indices_ptr=self.star_table.ref_indices_ptr,
                    ref_tuples_ptr=tuple_ptr,
                    ref_count=tuple_count,
                    star_count=star_count,
                    router_offsets_ptr=self.star_table.router_offsets_ptr,
                    executor_offsets_ptr=self.star_table.executor_offsets_ptr,
                    validator_offsets_ptr=self.star_table.validator_offsets_ptr,
                    anti_pattern_offsets_ptr=self.star_table.anti_pattern_offsets_ptr,
                )
                scan_scatter_s = float(time.perf_counter() - scan_t0)
            finally:
                try:
                    loader.gpu_free(ref_hash_device)
                except Exception:
                    pass
                try:
                    loader.gpu_free(resolved_ref_device)
                except Exception:
                    pass
                try:
                    loader.gpu_free(expanded_ref_device)
                except Exception:
                    pass
                try:
                    loader.gpu_free(unresolved_source_device)
                except Exception:
                    pass
                try:
                    loader.gpu_free(unresolved_target_hash_device)
                except Exception:
                    pass
                try:
                    loader.gpu_free(unresolved_count_device)
                except Exception:
                    pass
                ref_hash_host.close()
        self.star_table.star_count = int(star_count)
        self.star_table._host_stars = [dict(star) for star in host_stars]
        self._host_stars = [dict(star) for star in host_stars]
        self._build_feed_signature = str(manifest.get("build_feed_signature") or "")
        self._feed_source_signature = str(manifest.get("feed_source_signature") or "")
        self._last_build_feed_manifest = dict(manifest)
        self._catalog_signature = str(manifest.get("catalog_signature") or self._build_feed_signature)
        validation_summary = self._validate_build_summary_with_math_core(host_stars)
        summary = {
            "build_backend": BUILD_BACKEND,
            "build_feed_signature": str(manifest.get("build_feed_signature") or ""),
            "feed_source_signature": str(manifest.get("feed_source_signature") or ""),
            "build_feed_version": int(manifest.get("build_feed_version") or 0),
            "translated_rows": int(star_count),
            "ref_tuple_count": int(final_ref_count),
            "chunk_count": int(chunk_count),
            "load_build_feed_s": build_rows_s,
            "decode_build_rows_s": decode_s,
            "boot_finalize_s": finalize_s,
            "star_materialize_s": materialize_s,
            "build_star_hash_index_s": hash_index_s,
            "resolve_ref_hashes_s": resolve_s,
            "expand_reverse_symlinks_s": reverse_s,
            "ref_csr_build_s": scan_scatter_s,
            "star_table_upload_s": materialize_s,
        }
        summary.update(validation_summary)
        summary.update(self.materializer.ptx_signatures())
        return summary

    @staticmethod
    def _emit_rebuild_progress(message: str) -> None:
        print(f"[sovereign-build] {message}", flush=True)

    @staticmethod
    def _pack_catalog_input_row(
        row: dict[str, Any],
        *,
        target: memoryview,
        local_index: int,
    ) -> None:
        embedding16 = [float(value) for value in list(row.get("embedding16") or [])[:16]]
        if len(embedding16) < 16:
            embedding16.extend([0.0] * (16 - len(embedding16)))
        RAW_CATALOG_INPUT_STRUCT.pack_into(
            target,
            int(local_index) * RAW_CATALOG_INPUT_ENTRY_BYTES,
            *embedding16,
            int(row.get("galaxy_id_u32") or _fnv1a32(str(row.get("galaxy_id") or "reality"))),
            int(row.get("star_type", 0) or 0),
            int(row.get("selection_role_id") or ROLE_ID_BY_NAME.get(str(row.get("selection_role") or ""), 0)),
            int(row.get("layer_id", 0) or 0),
            _encode_runtime_flags(
                int(row.get("flags", STAR_FLAG_ACTIVE) or STAR_FLAG_ACTIVE),
                row.get("route_family"),
            ),
            1 if row.get("answer_eligible") else 0,
            int(row.get("semantic_polarity_raw", 0) or 0),
            float(row.get("semantic_focus_raw", 0.0) or 0.0),
            float(row.get("semantic_mass_raw", 0.0) or 0.0),
            float(row.get("attractive_prior_raw", 0.0) or 0.0),
            float(row.get("repulsive_prior_raw", 0.0) or 0.0),
            float(row.get("confidence", 0.0) or 0.0),
            int(row.get("route_policy_flags", 0) or 0),
            int(row.get("route_policy_branch_topk", 0) or 0),
            int(row.get("explicit_mask", 0) or 0),
            len(list(row.get("router_refs") or [])),
            len(list(row.get("executor_refs") or [])),
            len(list(row.get("validator_refs") or [])),
            len(list(row.get("anti_pattern_refs") or [])),
            int(row.get("star_hash", 0) or 0),
            float(row.get("domain_hash", 0.0) or 0.0),
            float(row.get("subject_hash", 0.0) or 0.0),
        )

    def _pack_catalog_input_chunk(
        self,
        rows: list[dict[str, Any]],
        *,
        start: int,
        count: int,
        target: memoryview,
    ) -> None:
        for local_index in range(int(count)):
            row = rows[int(start) + local_index]
            self._pack_catalog_input_row(
                row,
                target=target,
                local_index=local_index,
            )

    def _build_stars_sovereign(self, catalog: list[dict[str, Any]]) -> dict[str, Any]:
        parse_t0 = time.perf_counter()
        host_stars, ref_tuples = self._translate_catalog_entries(catalog)
        parse_s = float(time.perf_counter() - parse_t0)
        star_count = len(host_stars)
        ref_count = len(ref_tuples)
        self._emit_rebuild_progress(f"translated {star_count} stars and {ref_count} ref tuples")
        self.star_table.prepare_gpu_build(star_count=star_count, ref_capacity=max(1, ref_count))

        raw_input_buffer_bytes = max(1, GPU_BUILD_CHUNK_SIZE * RAW_CATALOG_INPUT_ENTRY_BYTES)
        finalized_input_buffer_bytes = max(1, GPU_BUILD_CHUNK_SIZE * FINALIZED_CATALOG_INPUT_ENTRY_BYTES)
        pinned_buffers = [loader.PinnedHostBuffer(raw_input_buffer_bytes), loader.PinnedHostBuffer(raw_input_buffer_bytes)]
        input_buffers = [loader.gpu_malloc(raw_input_buffer_bytes), loader.gpu_malloc(raw_input_buffer_bytes)]
        finalized_buffers = [loader.gpu_malloc(finalized_input_buffer_bytes), loader.gpu_malloc(finalized_input_buffer_bytes)]
        streams = [loader.create_stream(), loader.create_stream()]
        chunk_pack_s = 0.0
        boot_finalize_t0 = time.perf_counter()
        finalize_launch_s = 0.0
        star_materialize_t0 = time.perf_counter()
        chunk_count = (star_count + GPU_BUILD_CHUNK_SIZE - 1) // GPU_BUILD_CHUNK_SIZE if star_count > 0 else 0
        try:
            for chunk_index in range(chunk_count):
                buffer_index = chunk_index % 2
                if chunk_index >= 2:
                    loader.stream_synchronize(streams[buffer_index])
                chunk_start = chunk_index * GPU_BUILD_CHUNK_SIZE
                chunk_size = min(GPU_BUILD_CHUNK_SIZE, star_count - chunk_start)
                pack_t0 = time.perf_counter()
                target_view = pinned_buffers[buffer_index].view()
                self._pack_catalog_input_chunk(
                    host_stars,
                    start=chunk_start,
                    count=chunk_size,
                    target=target_view,
                )
                chunk_pack_s += float(time.perf_counter() - pack_t0)
                bytes_to_copy = int(chunk_size) * RAW_CATALOG_INPUT_ENTRY_BYTES
                if getattr(pinned_buffers[buffer_index], "pinned", False):
                    loader.memcpy_htod_async(
                        input_buffers[buffer_index],
                        pinned_buffers[buffer_index].ptr,
                        bytes_to_copy,
                        stream=streams[buffer_index],
                    )
                else:
                    loader.memcpy_htod(
                        input_buffers[buffer_index],
                        pinned_buffers[buffer_index].ptr,
                        bytes_to_copy,
                    )
                finalize_t0 = time.perf_counter()
                self.materializer.finalize_chunk(
                    raw_input_ptr=input_buffers[buffer_index],
                    finalized_input_ptr=finalized_buffers[buffer_index],
                    entry_count=chunk_size,
                    stream=streams[buffer_index],
                )
                finalize_launch_s += float(time.perf_counter() - finalize_t0)
                self.materializer.materialize_chunk(
                    galaxy_table_ptr=self.star_table.gpu_ptr,
                    input_ptr=finalized_buffers[buffer_index],
                    entry_count=chunk_size,
                    star_offset=chunk_start,
                    router_offsets_ptr=self.star_table.router_offsets_ptr,
                    router_counts_ptr=self.star_table.router_counts_ptr,
                    executor_offsets_ptr=self.star_table.executor_offsets_ptr,
                    executor_counts_ptr=self.star_table.executor_counts_ptr,
                    validator_offsets_ptr=self.star_table.validator_offsets_ptr,
                    validator_counts_ptr=self.star_table.validator_counts_ptr,
                    anti_pattern_offsets_ptr=self.star_table.anti_pattern_offsets_ptr,
                    anti_pattern_counts_ptr=self.star_table.anti_pattern_counts_ptr,
                    stream=streams[buffer_index],
                )
                if (
                    chunk_index == 0
                    or chunk_index + 1 == chunk_count
                    or ((chunk_index + 1) % GPU_BUILD_PROGRESS_EVERY) == 0
                ):
                    self._emit_rebuild_progress(
                        f"materialized chunk {chunk_index + 1}/{max(chunk_count, 1)} "
                        f"(stars={chunk_start + chunk_size}/{star_count})"
                    )
            for stream in streams:
                loader.stream_synchronize(stream)
        finally:
            for stream in streams:
                try:
                    loader.destroy_stream(stream)
                except Exception:
                    pass
        boot_finalize_s = float(time.perf_counter() - boot_finalize_t0)
        star_materialize_s = float(time.perf_counter() - star_materialize_t0)

        ref_csr_t0 = time.perf_counter()
        ref_tuple_bytes = max(1, ref_count * REF_TUPLE_BYTES)
        ref_stream = loader.create_stream()
        ref_host = loader.PinnedHostBuffer(ref_tuple_bytes)
        ref_device = loader.gpu_malloc(ref_tuple_bytes)
        try:
            ref_view = ref_host.view()
            for index, ref_tuple in enumerate(ref_tuples):
                REF_TUPLE_STRUCT.pack_into(ref_view, index * REF_TUPLE_BYTES, *ref_tuple)
            if ref_count > 0:
                if getattr(ref_host, "pinned", False):
                    loader.memcpy_htod_async(
                        ref_device,
                        ref_host.ptr,
                        ref_count * REF_TUPLE_BYTES,
                        stream=ref_stream,
                    )
                else:
                    loader.memcpy_htod(
                        ref_device,
                        ref_host.ptr,
                        ref_count * REF_TUPLE_BYTES,
                    )
                self.materializer.count_refs(
                    ref_tuples_ptr=ref_device,
                    ref_count=ref_count,
                    star_count=star_count,
                    router_counts_ptr=self.star_table.router_counts_ptr,
                    executor_counts_ptr=self.star_table.executor_counts_ptr,
                    validator_counts_ptr=self.star_table.validator_counts_ptr,
                    anti_pattern_counts_ptr=self.star_table.anti_pattern_counts_ptr,
                    stream=ref_stream,
                )
                self.materializer.scan_offsets(
                    star_count=star_count,
                    router_counts_ptr=self.star_table.router_counts_ptr,
                    router_offsets_ptr=self.star_table.router_offsets_ptr,
                    executor_counts_ptr=self.star_table.executor_counts_ptr,
                    executor_offsets_ptr=self.star_table.executor_offsets_ptr,
                    validator_counts_ptr=self.star_table.validator_counts_ptr,
                    validator_offsets_ptr=self.star_table.validator_offsets_ptr,
                    anti_pattern_counts_ptr=self.star_table.anti_pattern_counts_ptr,
                    anti_pattern_offsets_ptr=self.star_table.anti_pattern_offsets_ptr,
                    stream=ref_stream,
                )
                self.materializer.scatter_refs(
                    galaxy_table_ptr=self.star_table.gpu_ptr,
                    ref_indices_ptr=self.star_table.ref_indices_ptr,
                    ref_tuples_ptr=ref_device,
                    ref_count=ref_count,
                    star_count=star_count,
                    router_offsets_ptr=self.star_table.router_offsets_ptr,
                    executor_offsets_ptr=self.star_table.executor_offsets_ptr,
                    validator_offsets_ptr=self.star_table.validator_offsets_ptr,
                    anti_pattern_offsets_ptr=self.star_table.anti_pattern_offsets_ptr,
                    stream=ref_stream,
                )
            loader.stream_synchronize(ref_stream)
        finally:
            try:
                loader.destroy_stream(ref_stream)
            except Exception:
                pass
            try:
                loader.gpu_free(ref_device)
            except Exception:
                pass
            ref_host.close()
            for pinned in pinned_buffers:
                pinned.close()
            for device_buffer in input_buffers:
                try:
                    loader.gpu_free(device_buffer)
                except Exception:
                    pass
            for finalized_buffer in finalized_buffers:
                try:
                    loader.gpu_free(finalized_buffer)
                except Exception:
                    pass
        ref_csr_build_s = float(time.perf_counter() - ref_csr_t0)
        self.star_table.star_count = int(star_count)
        self.star_table._host_stars = []
        device_stars = self.star_table.read_stars(star_count)
        readback_stars: list[dict[str, Any]] = []
        for parsed_star, device_star in zip(host_stars, device_stars):
            merged = dict(device_star)
            for key in ("id", "name", "galaxy_id", "route_family", "domain_hash", "subject_hash", "confidence"):
                if key in parsed_star:
                    merged[key] = parsed_star.get(key)
            readback_stars.append(merged)
        self.star_table._host_stars = [dict(star) for star in readback_stars]
        self._host_stars = [dict(star) for star in readback_stars]
        validation_summary = self._validate_build_summary_with_math_core(readback_stars)
        self._emit_rebuild_progress(
            f"device build complete stars={star_count} refs={ref_count} "
            f"finalize={boot_finalize_s:.3f}s materialize={star_materialize_s:.3f}s csr={ref_csr_build_s:.3f}s"
        )
        summary = {
            "build_backend": BUILD_BACKEND,
            "translated_rows": int(star_count),
            "ref_tuple_count": int(ref_count),
            "chunk_count": int(chunk_count),
            "parse_s": parse_s,
            "chunk_pack_s": chunk_pack_s,
            "boot_finalize_s": boot_finalize_s,
            "boot_finalize_launch_s": finalize_launch_s,
            "star_materialize_s": star_materialize_s,
            "ref_csr_build_s": ref_csr_build_s,
            "star_table_upload_s": star_materialize_s,
        }
        summary.update(validation_summary)
        summary.update(self.materializer.ptx_signatures())
        return summary

    def _validate_build_summary_with_math_core(self, stars: list[dict[str, Any]]) -> dict[str, Any]:
        route_ready = 0
        route_linked = 0
        for star in stars:
            role = str(star.get("selection_role") or "unknown")
            if role not in {"router", "executor", "validator", "answer", "anti_pattern"}:
                continue
            route_ready += 1
            if role == "router":
                if list(star.get("executor_refs") or []) and list(star.get("validator_refs") or []):
                    route_linked += 1
            elif role == "executor":
                route_policy = dict(star.get("route_policy") or {})
                if not bool(route_policy.get("requires_validator")) or list(star.get("validator_refs") or []):
                    route_linked += 1
            else:
                route_linked += 1
        if route_ready <= 0:
            return {
                "validation_route_ready_count": 0,
                "validation_route_linked_count": 0,
                "validation_route_ratio": 0.0,
                "validation_route_trit": 0,
            }
        route_ratio = float(route_linked) / float(max(route_ready, 1))
        try:
            from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

            engine = ModularRPNEngine()
            route_trit = int(
                round(
                    float(
                        engine.evaluate(f"{route_ratio:.6f} 0.950000 tcomp")
                    )
                )
            )
        except Exception:
            route_trit = 0
        return {
            "validation_route_ready_count": int(route_ready),
            "validation_route_linked_count": int(route_linked),
            "validation_route_ratio": float(route_ratio),
            "validation_route_trit": int(route_trit),
        }

    def _ensure_bidirectional_symlinkage(self, stars: list[dict[str, Any]]) -> None:
        reverse_key = {
            "router_refs": "router_refs",
            "executor_refs": "router_refs",
            "validator_refs": "executor_refs",
            "anti_pattern_refs": "anti_pattern_refs",
        }
        for source_index, star in enumerate(stars):
            for key, target_key in reverse_key.items():
                for target_index in list(star.get(key) or []):
                    if target_index < 0 or target_index >= len(stars):
                        continue
                    target_refs = list(stars[target_index].get(target_key) or [])
                    if source_index not in target_refs:
                        target_refs.append(source_index)
                        stars[target_index][target_key] = target_refs

    def _host_star(self, star_index: int) -> dict[str, Any] | None:
        if 0 <= int(star_index) < len(self._host_stars):
            return dict(self._host_stars[int(star_index)])
        return None

    def _task_payload(self, task: dict[str, Any]) -> dict[str, Any]:
        query_text = str(task.get("query") or task.get("prompt") or task.get("question") or "").strip()
        family = VRAMTaskBuffer.normalize_task_type(task.get("surface_kind") or task.get("type") or "")
        raw_options = list(task.get("options") or [])
        if not raw_options and family == "GAME_2D":
            raw_options = list(task.get("action_options") or [])
        options = [str(option) for option in raw_options if str(option).strip()]
        option_embeddings = [
            list(self.knowledgeverse._embed_query_gpu(option))
            for option in options[:7]
        ]
        option_hashes = [_stable_hash(option) for option in options[:7]]
        expected_answer = task.get("expected_answer")
        expected_hash = int(task.get("expected_hash") or _stable_hash(expected_answer if expected_answer is not None else task.get("expected_output")))
        expected_index = -1
        if expected_answer is not None and options:
            expected_text = str(expected_answer).strip()
            for index, option in enumerate(options):
                if str(option).strip() == expected_text:
                    expected_index = index
                    break
        return {
            "surface_kind": family,
            "type": family,
            "query_embedding": list(task.get("query_embedding") or self.knowledgeverse._embed_query_gpu(query_text, task=task)),
            "option_embeddings": option_embeddings,
            "option_hashes": option_hashes,
            "options": options,
            "subject": str(task.get("subject") or ""),
            "domain_hint": str(task.get("domain_hint") or task.get("domain") or ""),
            "thinking_budget": int(task.get("thinking_budget", 10) or 10),
            "action_history": list(task.get("action_history") or [])[:7],
            "ternary_signal": int(task.get("ternary_signal", 0) or 0),
            "goal_embedding": list(task.get("goal_embedding") or []),
            "expected_hash": expected_hash,
            "expected_index": expected_index,
            "task_id": str(task.get("task_id") or ""),
        }

    def dispatch_task(self, task: dict[str, Any]) -> dict[str, Any]:
        self.ensure_loaded()
        payload = self._task_payload(task)
        self.task_buffer.bulk_load([payload])
        self.dispatch.launch(
            self.task_buffer,
            1,
            brain_ptr=self.brain.gpu_ptr,
            star_table=self.star_table,
            lesson_ring=self.lesson_ring,
            trm_weight_buffers=getattr(self.knowledgeverse, "_trm_weight_buffers", None),
        )
        row = dict(self.task_buffer.read_results(1)[0])
        trace = SovereignRouteTrace(
            router_index=int(row.get("router_star_index", -1)),
            executor_index=int(row.get("executor_star_index", -1)),
            validator_index=int(row.get("validator_star_index", -1)),
            winner_index=int(row.get("winner_star_index", -1)),
            winner_role_id=int(row.get("winner_role_id", 0)),
            route_depth=int(row.get("route_depth", 0)),
            anti_pattern_signal=int(row.get("anti_pattern_signal", 0)),
            route_budget_used=int(row.get("route_budget_used", 0)),
            route_budget_min=int(row.get("route_budget_min", 0)),
            recursion_depth_used=int(row.get("recursion_depth_used", 0)),
            route_trace_star_indices=[int(value) for value in list(row.get("route_trace_star_indices") or [])],
            route_trace_role_ids=[int(value) for value in list(row.get("route_trace_role_ids") or [])],
        )
        router_star = self._host_star(trace.router_index)
        executor_star = self._host_star(trace.executor_index)
        validator_star = self._host_star(trace.validator_index)
        winner_star = self._host_star(trace.winner_index)
        options = list(task.get("options") or [])
        answer_index = int(row.get("answer_index", 0) or 0)
        answer_text = _materialize_answer_text(
            options=options,
            answer_index=answer_index,
            winner_star=winner_star,
        )
        resolved_route_family = str(
            (winner_star or {}).get("route_family")
            or (validator_star or {}).get("route_family")
            or (executor_star or {}).get("route_family")
            or (router_star or {}).get("route_family")
            or payload["surface_kind"]
        )
        trace_star_ids = [
            str((self._host_star(index) or {}).get("id", ""))
            for index in list(trace.route_trace_star_indices or [])
            if 0 <= int(index) < len(self._host_stars)
        ]
        trace_role_ids = [int(value) for value in list(trace.route_trace_role_ids or [])]
        runtime_packet = self.knowledgeverse.materialize_runtime_result(
            task=task,
            route_family=resolved_route_family,
            answer_kind=str(
                (
                    (winner_star or {}).get("route_policy", {})
                    if isinstance((winner_star or {}).get("route_policy"), dict)
                    else {}
                ).get("answer_kind")
                or (
                    (validator_star or {}).get("route_policy", {})
                    if isinstance((validator_star or {}).get("route_policy"), dict)
                    else {}
                ).get("answer_kind")
                or (
                    (executor_star or {}).get("route_policy", {})
                    if isinstance((executor_star or {}).get("route_policy"), dict)
                    else {}
                ).get("answer_kind")
                or ""
            ),
            answer_index=answer_index,
            stars=[
                star
                for star in (winner_star, validator_star, executor_star, router_star)
                if isinstance(star, dict)
            ],
        )
        if not runtime_packet.get("answer_text") and answer_text and not _looks_like_route_label(answer_text):
            runtime_packet["answer_text"] = answer_text
            if runtime_packet.get("numeric_answer") is None:
                try:
                    runtime_packet["numeric_answer"] = float(str(answer_text).replace(",", "").replace("$", ""))
                except Exception:
                    pass
        runtime_packet["answer_materialized"] = bool(runtime_packet.get("answer_materialized"))
        failure_code = str(runtime_packet.get("failure_code") or "").strip()
        task_status = "ok" if (failure_code == "" or runtime_packet["answer_materialized"]) else "error"
        anti_pattern_ids: list[str] = []
        if int(trace.anti_pattern_signal) > 0:
            anti_star = self._host_star(trace.winner_index)
            if isinstance(anti_star, dict):
                anti_id = str(anti_star.get("id", "")).strip()
                if anti_id.startswith("anti_pattern_"):
                    anti_pattern_ids.append(anti_id)
        return {
            "status": "ok" if task_status == "ok" else "error",
            "query_type": payload["surface_kind"],
            "task_result": {
                "status": task_status,
                "answer_index": answer_index,
                "answer": str(runtime_packet.get("answer_text") or ""),
                "response": str(runtime_packet.get("answer_text") or ""),
                "confidence": float(row.get("confidence", 0.0) or 0.0),
                "convergence_signal": int(row.get("convergence_signal", 0) or 0),
                "iterations_used": int(row.get("iterations_used", 0) or 0),
                "answer_text_hash": int(row.get("answer_text_hash", 0) or 0),
                "goal_progress": float(row.get("goal_progress", 0.0) or 0.0),
                "winner_star_id": str((winner_star or {}).get("id", "")),
                "winner_role": str((winner_star or {}).get("selection_role", "")),
                "route_family": resolved_route_family,
                "route_depth": int(trace.route_depth),
                "anti_pattern_signal": int(trace.anti_pattern_signal),
                "route_budget_used": int(trace.route_budget_used),
                "route_budget_min": int(trace.route_budget_min),
                "recursion_depth_used": int(trace.recursion_depth_used),
                "answer_kind": str(runtime_packet.get("answer_kind") or "none"),
                "answer_text": str(runtime_packet.get("answer_text") or ""),
                "numeric_answer": runtime_packet.get("numeric_answer"),
                "answer_choice": str(runtime_packet.get("answer_choice") or ""),
                "output_grid": runtime_packet.get("output_grid"),
                "action_index": runtime_packet.get("action_index"),
                "action_name": str(runtime_packet.get("action_name") or ""),
                "answer_materialized": bool(runtime_packet.get("answer_materialized")),
                "failure_code": failure_code,
                "trace_star_ids": list(trace_star_ids),
                "trace_roles": _trace_role_names(trace_role_ids),
                "anti_pattern_ids": anti_pattern_ids,
            },
            "route": {
                "surface_kind": payload["surface_kind"],
                "router_star": str((router_star or {}).get("id", "")),
                "executor_star": str((executor_star or {}).get("id", "")),
                "validator_star": str((validator_star or {}).get("id", "")),
                "winner_star": str((winner_star or {}).get("id", "")),
                "route_family": resolved_route_family,
                "trace_star_ids": list(trace_star_ids),
                "trace_role_ids": list(trace_role_ids),
                "trace_roles": _trace_role_names(trace_role_ids),
                "anti_pattern_ids": anti_pattern_ids,
            },
            "program_type": "gpu_task_dispatch_sovereign",
            "trm_dispatch": {
                "task_type": payload["surface_kind"],
                "router_star_id": str((router_star or {}).get("id", "")),
                "executor_star_id": str((executor_star or {}).get("id", "")),
                "validator_star_id": str((validator_star or {}).get("id", "")),
                "winner_star_id": str((winner_star or {}).get("id", "")),
                "winner_role_id": int(trace.winner_role_id),
                "route_depth": int(trace.route_depth),
                "anti_pattern_signal": int(trace.anti_pattern_signal),
                "route_budget_used": int(trace.route_budget_used),
                "route_budget_min": int(trace.route_budget_min),
                "recursion_depth_used": int(trace.recursion_depth_used),
            },
        }

    def sleep_flush_tick(self, *, profile: str = "shutdown") -> dict[str, Any]:
        self.ensure_loaded()
        applied_lessons = self.lesson_gpu.apply(
            self.star_table,
            self.lesson_ring,
        )
        stats = self.lesson_ring.read_stats()
        return {
            "profile": str(profile),
            "applied_lessons": int(applied_lessons),
            "gravity": {
                "skipped": True,
                "profile": str(profile),
            },
            **stats,
        }

    def sleep_tick(self) -> dict[str, Any]:
        summary = dict(self.sleep_flush_tick(profile="service"))
        gravity_summary = self.gravity_gpu.evolve_table(
            self.star_table,
        )
        summary["gravity"] = dict(gravity_summary)
        return summary

    def current_learning_state(self) -> dict[str, Any]:
        return dict(self.lesson_ring.read_stats())


__all__ = ["SovereignHotPath"]
