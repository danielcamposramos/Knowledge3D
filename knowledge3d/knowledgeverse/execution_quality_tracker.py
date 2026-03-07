"""Persistent quality tracker over Tool execution events.

This is the first consumer of the execution-event stream. It keeps per-tool
quality stats and lightweight specialist centroids for ternary contrastive
routing updates.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _ternary(value: float, low: float = -0.05, high: float = 0.05) -> int:
    if value <= low:
        return -1
    if value >= high:
        return 1
    return 0


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-8:
        return [0.0 for _ in vec]
    return [float(v / norm) for v in vec]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def _quality_level(record: Mapping[str, Any] | None) -> float:
    if not isinstance(record, Mapping):
        return 0.5
    bayesian = float(record.get("bayesian_quality", 0.5))
    recent = [
        float(v)
        for v in record.get("recent_quality", [])
        if isinstance(v, (int, float))
    ]
    recent_mean = (sum(recent) / float(len(recent))) if recent else float(record.get("last_quality_signal", bayesian))
    return _clamp((0.3 * bayesian) + (0.7 * recent_mean), 0.0, 1.0)


class ExecutionQualityTracker:
    def __init__(
        self,
        *,
        state_path: str | Path,
        gap_log_path: str | Path,
        dims: int = 16,
        spawn_threshold: float = 0.3,
        save_every: int = 64,
        save_interval_s: float = 2.0,
    ):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.gap_log_path = Path(gap_log_path)
        self.gap_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.dims = max(8, int(dims))
        self.spawn_threshold = float(spawn_threshold)
        self.save_every = max(1, int(save_every))
        self.save_interval_s = max(0.0, float(save_interval_s))
        self._dirty_observations = 0
        self._last_save_monotonic = time.monotonic()
        self._gap_buffer: list[str] = []
        self._gap_buffer_size = max(1, int(save_every))
        self._last_gap_flush_monotonic = time.monotonic()
        self._state_path_ready = self.state_path.exists()
        self._gap_log_ready = self.gap_log_path.exists()
        self._token_cache: dict[str, list[str]] = {}
        self._embed_cache: dict[str, list[float]] = {}
        self._state: dict[str, Any] = {
            "tools": {},
            "specialists": {},
            "tool_sources": {},
            "route_sources": {},
        }
        self._load()
        atexit.register(self.flush)

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        self._state["tools"] = dict(payload.get("tools", {}) or {})
        self._state["specialists"] = dict(payload.get("specialists", {}) or {})
        self._state["tool_sources"] = dict(payload.get("tool_sources", {}) or {})
        self._state["route_sources"] = dict(payload.get("route_sources", {}) or {})

    def _save(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force:
            if self._dirty_observations <= 0:
                return
            if (
                self._dirty_observations < self.save_every
                and (now - self._last_save_monotonic) < self.save_interval_s
            ):
                return
        self.state_path.write_text(json.dumps(self._state, indent=2, sort_keys=True), encoding="utf-8")
        self._dirty_observations = 0
        self._last_save_monotonic = now
        self._state_path_ready = True

    def flush(self) -> None:
        self._save(force=True)
        self._flush_gap_buffer(force=True)

    def _append_gap(self, payload: Mapping[str, Any]) -> None:
        self._gap_buffer.append(json.dumps(dict(payload), separators=(",", ":"), sort_keys=True))
        if not self._gap_log_ready:
            self._flush_gap_buffer(force=True)
            return
        now = time.monotonic()
        if (
            len(self._gap_buffer) >= self._gap_buffer_size
            or (now - self._last_gap_flush_monotonic) >= self.save_interval_s
        ):
            self._flush_gap_buffer()

    def _flush_gap_buffer(self, *, force: bool = False) -> None:
        if not self._gap_buffer:
            return
        if not force:
            now = time.monotonic()
            if (
                len(self._gap_buffer) < self._gap_buffer_size
                and (now - self._last_gap_flush_monotonic) < self.save_interval_s
            ):
                return
        with self.gap_log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(self._gap_buffer) + "\n")
        self._gap_buffer.clear()
        self._last_gap_flush_monotonic = time.monotonic()
        self._gap_log_ready = True

    def _tokenize(self, text: str) -> list[str]:
        token = str(text or "")
        cached = self._token_cache.get(token)
        if cached is not None:
            return cached
        expanded = re.sub(r"([a-z0-9])([A-Z])", r"\\1 \\2", token)
        rows = [
            tok
            for tok in "".join(ch.lower() if ch.isalnum() else " " for ch in expanded).split()
            if tok
        ]
        if len(self._token_cache) >= 2048:
            self._token_cache.clear()
        self._token_cache[token] = rows
        return rows

    def _embed_text(self, text: str) -> list[float]:
        cache_key = str(text or "")
        cached = self._embed_cache.get(cache_key)
        if cached is not None:
            return cached
        tokens = self._tokenize(text)
        if not tokens:
            vec = [0.0] * self.dims
            self._embed_cache[cache_key] = vec
            return vec
        vec = [0.0] * self.dims
        inv = 1.0 / float(len(tokens))
        for token_text in tokens:
            digest = hashlib.sha1(token_text.encode("utf-8")).digest()
            for i in range(0, min(len(digest), self.dims)):
                sign = 1.0 if (digest[i] & 1) else -1.0
                vec[i] += sign * inv
        embedded = _normalize(vec)
        if len(self._embed_cache) >= 2048:
            self._embed_cache.clear()
        self._embed_cache[cache_key] = embedded
        return embedded

    def _ensure_specialist(self, specialist_id: str) -> dict[str, Any]:
        token = str(specialist_id or "").strip() or "unknown"
        specialists = self._state.setdefault("specialists", {})
        record = specialists.get(token)
        if isinstance(record, dict):
            return record
        centroid = self._embed_text(token)
        record = {
            "specialist_id": token,
            "centroid": centroid,
            "update_count": 0,
            "exploration_count": 0,
            "success_updates": 0,
            "failure_updates": 0,
            "last_relevance": 0.0,
        }
        specialists[token] = record
        return record

    def get_tool_record(self, tool_id: str) -> dict[str, Any] | None:
        return self._state.get("tools", {}).get(str(tool_id or "").strip())

    @staticmethod
    def classify_route_source(runtime_status: str, tool_kind: str | None = None) -> str:
        normalized = str(runtime_status or "").strip().lower()
        if normalized == "ptx_rpn_available":
            return "kernel"
        if normalized == "ptx_bridge_available":
            return "bridge"
        if normalized == "ptx_runtime_available" or normalized.startswith("ptx_"):
            return "bridge"
        if "recipe" in str(tool_kind or "").strip().lower():
            return "recipe"
        return "recipe"

    def get_tool_source_record(
        self,
        tool_id: str,
        *,
        runtime_status: str = "",
        tool_kind: str | None = None,
    ) -> dict[str, Any] | None:
        route_source = self.classify_route_source(runtime_status, tool_kind)
        key = f"{str(tool_id or '').strip()}::{route_source}"
        return self._state.get("tool_sources", {}).get(key)

    def get_route_source_record(self, runtime_status: str, tool_kind: str | None = None) -> dict[str, Any] | None:
        route_source = self.classify_route_source(runtime_status, tool_kind)
        return self._state.get("route_sources", {}).get(route_source)

    def tool_quality_bonus(self, tool_id: str) -> float:
        record = self.get_tool_record(tool_id)
        if not isinstance(record, dict):
            return 0.0
        bayesian = float(record.get("bayesian_quality", 0.5))
        trend = int(record.get("ternary_trend", 0))
        return (0.3 * bayesian) + (0.08 * trend)

    def source_quality_bonus(
        self,
        tool_id: str,
        *,
        runtime_status: str = "",
        tool_kind: str | None = None,
    ) -> float:
        tool_source = self.get_tool_source_record(
            tool_id,
            runtime_status=runtime_status,
            tool_kind=tool_kind,
        )
        route_source = self.get_route_source_record(runtime_status, tool_kind)
        tool_source_quality = _quality_level(tool_source)
        route_quality = _quality_level(route_source)
        tool_source_trend = int((tool_source or {}).get("ternary_trend", 0))
        route_trend = int((route_source or {}).get("ternary_trend", 0))
        combined_quality = (0.65 * tool_source_quality) + (0.35 * route_quality)
        combined_trend = _ternary(float(tool_source_trend + route_trend), low=-0.5, high=0.5)
        return (0.22 * combined_quality) + (0.06 * combined_trend)

    def routing_gate(
        self,
        tool_id: str,
        *,
        runtime_status: str = "",
        tool_kind: str | None = None,
    ) -> dict[str, Any]:
        route_source = self.classify_route_source(runtime_status, tool_kind)
        tool_record = self.get_tool_record(tool_id) or {}
        tool_source_record = self.get_tool_source_record(
            tool_id,
            runtime_status=runtime_status,
            tool_kind=tool_kind,
        ) or {}
        route_source_record = self.get_route_source_record(runtime_status, tool_kind) or {}

        tool_quality = _quality_level(tool_record)
        source_quality = (
            0.65 * _quality_level(tool_source_record)
            + 0.35 * _quality_level(route_source_record)
        )
        tool_trend = int(tool_record.get("ternary_trend", 0))
        source_trend = _ternary(
            float(int(tool_source_record.get("ternary_trend", 0)) + int(route_source_record.get("ternary_trend", 0))),
            low=-0.5,
            high=0.5,
        )
        combined_quality = (0.6 * tool_quality) + (0.4 * source_quality)
        combined_trend = _ternary(float(tool_trend + source_trend), low=-0.5, high=0.5)

        preferred_route = "bridge"
        ternary_gate = 0
        if combined_quality >= 0.78 and combined_trend >= 0:
            preferred_route = "kernel"
            ternary_gate = 1
        elif combined_quality <= 0.42 and combined_trend <= 0:
            preferred_route = "recipe"
            ternary_gate = -1

        return {
            "tool_id": str(tool_id or "").strip(),
            "route_source": route_source,
            "preferred_route": preferred_route,
            "ternary_gate": int(ternary_gate),
            "combined_quality": float(combined_quality),
            "combined_trend": int(combined_trend),
            "tool_bayesian_quality": float(tool_quality),
            "source_bayesian_quality": float(source_quality),
        }

    @staticmethod
    def routing_alignment_bonus(gate: Mapping[str, Any]) -> float:
        route_source = str(gate.get("route_source", "")).strip()
        preferred_route = str(gate.get("preferred_route", "")).strip()
        if not route_source or not preferred_route:
            return 0.0
        if route_source == preferred_route:
            return 2.5
        if preferred_route == "kernel" and route_source == "bridge":
            return -1.0
        if preferred_route == "kernel" and route_source == "recipe":
            return -2.0
        if preferred_route == "bridge" and route_source == "kernel":
            return -1.5
        if preferred_route == "bridge" and route_source == "recipe":
            return -1.0
        if preferred_route == "recipe" and route_source == "bridge":
            return -2.0
        if preferred_route == "recipe" and route_source == "kernel":
            return -3.0
        return -1.0

    def specialist_relevance(self, specialist_id: str, query_context: str) -> float:
        record = self._ensure_specialist(specialist_id)
        centroid = [float(v) for v in record.get("centroid", [])]
        task_vec = self._embed_text(query_context)
        cosine = max(0.0, _cosine(_normalize(centroid), task_vec))
        specialist_tokens = set(self._tokenize(specialist_id))
        query_tokens = set(self._tokenize(query_context))
        if not specialist_tokens or not query_tokens:
            overlap_ratio = 0.0
        else:
            overlap_ratio = float(len(specialist_tokens & query_tokens)) / float(len(specialist_tokens | query_tokens))
        return float((0.5 * cosine) + (0.5 * overlap_ratio))

    def max_specialist_relevance(self, specialist_ids: Sequence[str], query_context: str) -> tuple[str | None, float]:
        best_id: str | None = None
        best_score = -1.0
        for specialist_id in specialist_ids:
            token = str(specialist_id).strip()
            if not token:
                continue
            score = self.specialist_relevance(token, query_context)
            if score > best_score:
                best_id = token
                best_score = score
        return best_id, max(0.0, float(best_score))

    def observe_event(
        self,
        event: Mapping[str, Any],
        *,
        specialist_catalog: Sequence[str] | None = None,
        update_specialist: bool = True,
        update_gap_detection: bool = True,
    ) -> dict[str, Any]:
        tool_id = str(event.get("tool_id", "")).strip()
        if not tool_id:
            return {}
        quality = _clamp(float(event.get("quality_signal", 0.0) or 0.0), 0.0, 1.0)
        execution_us = max(0, int(event.get("execution_us", 0) or 0))
        outcome = int(max(-1, min(1, int(event.get("outcome", 0) or 0))))
        specialist_id = str(event.get("specialist_id", "") or "").strip()
        query_context = str(event.get("query_context", "") or "")
        runtime_status = str(event.get("runtime_status", "") or "").strip()
        tool_kind = str(event.get("tool_kind", "") or "").strip()
        route_source = self.classify_route_source(runtime_status, tool_kind)

        tools = self._state.setdefault("tools", {})
        record = dict(tools.get(tool_id, {}) or {})
        previous_total = int(record.get("total_executions", 0))
        previous_bayesian = float(record.get("bayesian_quality", 0.5))
        recent_quality = [float(v) for v in record.get("recent_quality", []) if isinstance(v, (int, float))][-7:]
        recent_quality.append(quality)

        total = previous_total + 1
        success_count = int(record.get("success_count", 0)) + (1 if outcome > 0 else 0)
        failure_count = int(record.get("failure_count", 0)) + (1 if outcome < 0 else 0)
        uncertain_count = int(record.get("uncertain_count", 0)) + (1 if outcome == 0 else 0)
        avg_execution_us = (
            (float(record.get("avg_execution_us", 0.0)) * previous_total + float(execution_us)) / float(total)
            if total > 0 else float(execution_us)
        )
        bayesian_quality = float(success_count + 1) / float(total + 2)
        previous_recent = recent_quality[:-1]
        previous_mean = (sum(previous_recent) / float(len(previous_recent))) if previous_recent else previous_bayesian
        current_mean = sum(recent_quality) / float(len(recent_quality))
        ternary_trend = _ternary(current_mean - previous_mean)
        record.update(
            {
                "tool_id": tool_id,
                "total_executions": total,
                "success_count": success_count,
                "failure_count": failure_count,
                "uncertain_count": uncertain_count,
                "avg_execution_us": float(avg_execution_us),
                "bayesian_quality": float(bayesian_quality),
                "ternary_trend": int(ternary_trend),
                "last_quality_signal": float(quality),
                "recent_quality": recent_quality,
                "last_timestamp_us": int(event.get("timestamp_us", 0) or 0),
            }
        )
        tools[tool_id] = record

        def _update_quality_record(store: dict[str, Any], key: str) -> dict[str, Any]:
            source_record = dict(store.get(key, {}) or {})
            previous_total = int(source_record.get("total_executions", 0))
            recent = [float(v) for v in source_record.get("recent_quality", []) if isinstance(v, (int, float))][-7:]
            recent.append(quality)
            total = previous_total + 1
            success = int(source_record.get("success_count", 0)) + (1 if outcome > 0 else 0)
            failure = int(source_record.get("failure_count", 0)) + (1 if outcome < 0 else 0)
            uncertain = int(source_record.get("uncertain_count", 0)) + (1 if outcome == 0 else 0)
            avg_us = (
                (float(source_record.get("avg_execution_us", 0.0)) * previous_total + float(execution_us)) / float(total)
                if total > 0 else float(execution_us)
            )
            previous_recent = recent[:-1]
            previous_mean = (
                sum(previous_recent) / float(len(previous_recent))
                if previous_recent else float(source_record.get("bayesian_quality", 0.5))
            )
            current_mean = sum(recent) / float(len(recent))
            source_record.update(
                {
                    "total_executions": total,
                    "success_count": success,
                    "failure_count": failure,
                    "uncertain_count": uncertain,
                    "avg_execution_us": float(avg_us),
                    "bayesian_quality": float(success + 1) / float(total + 2),
                    "ternary_trend": int(_ternary(current_mean - previous_mean)),
                    "recent_quality": recent,
                    "last_quality_signal": float(quality),
                    "last_timestamp_us": int(event.get("timestamp_us", 0) or 0),
                }
            )
            store[key] = source_record
            return source_record

        route_sources = self._state.setdefault("route_sources", {})
        route_source_record = _update_quality_record(route_sources, route_source)
        route_source_record.update({"route_source": route_source})
        tool_sources = self._state.setdefault("tool_sources", {})
        tool_source_key = f"{tool_id}::{route_source}"
        tool_source_record = _update_quality_record(tool_sources, tool_source_key)
        tool_source_record.update(
            {
                "tool_id": tool_id,
                "route_source": route_source,
                "runtime_status": runtime_status,
                "tool_kind": tool_kind,
            }
        )

        gap_logged = False
        best_specialist_id: str | None = None
        best_relevance = 0.0
        if update_specialist and specialist_id:
            specialist = self._ensure_specialist(specialist_id)
            task_vec = self._embed_text(query_context)
            current_centroid = [float(v) for v in specialist.get("centroid", [])]
            current_relevance = _cosine(_normalize(current_centroid), task_vec)
            specialist["last_relevance"] = float(current_relevance)
            if outcome > 0:
                lr = 0.10
                updated = [((1.0 - lr) * c) + (lr * t) for c, t in zip(current_centroid, task_vec)]
                specialist["centroid"] = _normalize(updated)
                specialist["success_updates"] = int(specialist.get("success_updates", 0)) + 1
                specialist["update_count"] = int(specialist.get("update_count", 0)) + 1
            elif outcome < 0:
                lr = 0.05
                updated = [((1.0 - lr) * c) - (lr * t) for c, t in zip(current_centroid, task_vec)]
                specialist["centroid"] = _normalize(updated)
                specialist["failure_updates"] = int(specialist.get("failure_updates", 0)) + 1
                specialist["update_count"] = int(specialist.get("update_count", 0)) + 1
            else:
                specialist["exploration_count"] = int(specialist.get("exploration_count", 0)) + 1

        catalog = [str(item).strip() for item in (specialist_catalog or []) if str(item).strip()]
        if update_gap_detection and query_context and catalog:
            best_specialist_id, best_relevance = self.max_specialist_relevance(catalog, query_context)
            if best_relevance < self.spawn_threshold:
                gap_payload = {
                    "timestamp_us": int(event.get("timestamp_us", 0) or 0),
                    "query_context": query_context,
                    "specialist_id": specialist_id or None,
                    "best_specialist_id": best_specialist_id,
                    "best_relevance": float(best_relevance),
                    "spawn_threshold": float(self.spawn_threshold),
                }
                self._append_gap(gap_payload)
                gap_logged = True

        self._dirty_observations += 1
        self._save(force=not self._state_path_ready)
        routing_gate = self.routing_gate(
            tool_id,
            runtime_status=runtime_status,
            tool_kind=tool_kind,
        )
        return {
            "tool_id": tool_id,
            "bayesian_quality": float(bayesian_quality),
            "ternary_trend": int(ternary_trend),
            "best_specialist_id": best_specialist_id,
            "best_relevance": float(best_relevance),
            "gap_logged": bool(gap_logged),
            "route_source": route_source,
            "source_bayesian_quality": float(tool_source_record.get("bayesian_quality", 0.5)),
            "source_trend": int(tool_source_record.get("ternary_trend", 0)),
            "routing_gate": routing_gate,
        }


__all__ = ["ExecutionQualityTracker"]
