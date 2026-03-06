"""Execution-event recording for Tool routes.

This module is intentionally orchestration-side only. It records a compact,
ternary execution journal that both scene quality and specialist learning can
consume without touching the sovereign PTX hot path.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass, fields, is_dataclass, replace
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ExecutionEvent:
    tool_id: str
    query_context: str
    specialist_id: str | None
    math_core_tier: int
    execution_us: int
    outcome: int
    quality_signal: float
    ternary_quality: int
    timestamp_us: int
    chain_depth: int
    promotion_pressure: bool
    domain_hint: str | None = None
    execution_mode: str = "tool_entrypoint_chain"
    runtime_status: str = ""
    chain_tool_ids: tuple[str, ...] = ()
    chain_runtime_statuses: tuple[str, ...] = ()
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_id": str(self.tool_id),
            "query_context": str(self.query_context),
            "specialist_id": self.specialist_id,
            "math_core_tier": int(self.math_core_tier),
            "execution_us": int(self.execution_us),
            "outcome": int(self.outcome),
            "quality_signal": float(self.quality_signal),
            "ternary_quality": int(self.ternary_quality),
            "timestamp_us": int(self.timestamp_us),
            "chain_depth": int(self.chain_depth),
            "promotion_pressure": bool(self.promotion_pressure),
            "domain_hint": self.domain_hint,
            "execution_mode": str(self.execution_mode),
            "runtime_status": str(self.runtime_status),
            "chain_tool_ids": [str(value) for value in self.chain_tool_ids],
            "chain_runtime_statuses": [str(value) for value in self.chain_runtime_statuses],
            "error": self.error,
        }


class ExecutionEventRecorder:
    def __init__(
        self,
        *,
        storage_root: str | Path,
        buffer_size: int = 64,
        flush_interval_s: float = 1.0,
    ):
        self.storage_root = Path(storage_root)
        self.logs_dir = self.storage_root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.event_path = self.logs_dir / "execution_events.jsonl"
        self.buffer_size = max(1, int(buffer_size))
        self.flush_interval_s = max(0.0, float(flush_interval_s))
        self._buffer: list[str] = []
        self._last_flush_monotonic = time.monotonic()
        atexit.register(self.flush)

    def append(self, event: ExecutionEvent, *, knowledgeverse: Any | None = None) -> None:
        payload = event.as_dict()
        self._buffer.append(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        is_chain_step = event.execution_mode == "tool_chain_step"
        force_flush = (
            len(self._buffer) >= self.buffer_size
            or (time.monotonic() - self._last_flush_monotonic) >= self.flush_interval_s
        )
        if force_flush:
            self.flush()

        if (
            not is_chain_step
            and knowledgeverse is not None
            and hasattr(knowledgeverse, "shadow_copy")
        ):
            try:
                knowledgeverse.shadow_copy.record_event(
                    event_type="tool_execution",
                    event_data={
                        **payload,
                        "timestamp": float(payload["timestamp_us"]) / 1_000_000.0,
                        "confidence": float(payload["quality_signal"]),
                        "specialist": str(payload.get("specialist_id") or "unknown"),
                        "galaxy": "Tool",
                        "verification": "execution_event",
                    },
                )
            except Exception:
                pass

    def flush(self) -> None:
        if not self._buffer:
            return
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(self._buffer) + "\n")
        self._buffer.clear()
        self._last_flush_monotonic = time.monotonic()


def timestamp_us() -> int:
    return int(time.time_ns() // 1_000)


def ternary_quantize_quality(value: float, *, low: float = 0.40, high: float = 0.70) -> int:
    signal = max(0.0, min(1.0, float(value)))
    if signal >= high:
        return 1
    if signal <= low:
        return -1
    return 0


def normalize_material_score(value: Any) -> float | None:
    try:
        score = float(value)
    except Exception:
        return None
    if -1.0 <= score <= 1.0:
        return max(0.0, min((score + 1.0) * 0.5, 1.0))
    return max(0.0, min(score, 1.0))


def extract_math_core_tier(execution_plan: Mapping[str, Any] | None, result: Any | None) -> int:
    if isinstance(execution_plan, Mapping):
        plan = execution_plan.get("math_core_plan")
        if isinstance(plan, Mapping):
            try:
                tier = int(plan.get("preferred_tier", 0) or 0)
            except Exception:
                tier = 0
            if tier > 0:
                return tier

    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, Mapping):
        tier = _search_math_core_tier(metadata)
        if tier > 0:
            return tier
    return 1


def _search_math_core_tier(metadata: Mapping[str, Any]) -> int:
    best = 0
    for key, value in metadata.items():
        if key == "math_core_plan" and isinstance(value, Mapping):
            try:
                best = max(best, int(value.get("preferred_tier", 0) or 0))
            except Exception:
                pass
        elif key.endswith("math_core_plan") and isinstance(value, Mapping):
            try:
                best = max(best, int(value.get("preferred_tier", 0) or 0))
            except Exception:
                pass
    return best


def extract_quality_signal(result: Any | None) -> float:
    if result is None:
        return 0.0

    signals: list[float] = []
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, Mapping):
        coherence = metadata.get("overall_coherence")
        if coherence is not None:
            try:
                signals.append(max(0.0, min(float(coherence), 1.0)))
            except Exception:
                pass
        variance = metadata.get("coherence_variance")
        if variance is not None:
            try:
                signals.append(max(0.0, min(1.0 - float(variance), 1.0)))
            except Exception:
                pass
        score_table = metadata.get("material_score_table")
        if isinstance(score_table, list) and score_table:
            best_score = normalize_material_score((score_table[0] or {}).get("score"))
            if best_score is not None:
                signals.append(best_score)
        signal_summary = metadata.get("signal_projection_summary")
        if isinstance(signal_summary, Mapping):
            try:
                positive = float(signal_summary.get("positive_ratio", 0.0) or 0.0)
                negative = float(signal_summary.get("negative_ratio", 0.0) or 0.0)
                neutral = float(signal_summary.get("neutral_ratio", 0.0) or 0.0)
                signals.append(max(0.0, min(0.50 + 0.25 * (positive + negative) - 0.10 * neutral, 1.0)))
            except Exception:
                pass
        if bool(metadata.get("codec_enabled", False)):
            signals.append(0.75)
        if "projection_strategy" in metadata:
            signals.append(0.70)

    coherence_scores = getattr(result, "coherence_scores", None)
    if coherence_scores is not None:
        try:
            arr = np.asarray(coherence_scores, dtype=np.float32)
            if arr.size:
                signals.append(float(np.clip(np.mean(arr), 0.0, 1.0)))
        except Exception:
            pass

    if hasattr(result, "frames"):
        signals.append(0.70)
    if hasattr(result, "vertex_rgba"):
        signals.append(0.70)
    if hasattr(result, "selected_material"):
        signals.append(0.65)

    if not signals:
        return 0.65
    quality = float(np.clip(np.mean(np.asarray(signals, dtype=np.float32)), 0.0, 1.0))
    return quality


def build_execution_event(
    *,
    execution_plan: Mapping[str, Any] | None,
    tool_id: str,
    query_context: str,
    specialist_id: str | None,
    domain_hint: str | None,
    chain_depth: int,
    runtime_status: str,
    execution_us: int,
    result: Any | None = None,
    error: BaseException | None = None,
    chain_tool_ids: list[str] | tuple[str, ...] | None = None,
    chain_runtime_statuses: list[str] | tuple[str, ...] | None = None,
    execution_mode: str | None = None,
) -> ExecutionEvent:
    quality_signal = 0.0 if error is not None else extract_quality_signal(result)
    ternary_quality = -1 if error is not None else ternary_quantize_quality(quality_signal)
    outcome = -1 if error is not None else ternary_quality
    promotion_targets = []
    if isinstance(execution_plan, Mapping):
        promotion_targets = [
            str(value).strip()
            for value in execution_plan.get("promotion_targets", [])
            if str(value).strip()
        ]
    if chain_tool_ids is None and isinstance(execution_plan, Mapping):
        chain_tool_ids = [
            str(row.get("tool_id", "")).strip()
            for row in execution_plan.get("execution_chain", [])
            if isinstance(row, Mapping) and str(row.get("tool_id", "")).strip()
        ]
    if chain_runtime_statuses is None and isinstance(execution_plan, Mapping):
        chain_runtime_statuses = [
            str(row.get("runtime_status", "")).strip()
            for row in execution_plan.get("execution_chain", [])
            if isinstance(row, Mapping) and str(row.get("tool_id", "")).strip()
        ]
    normalized_chain_tool_ids = tuple(
        str(value).strip()
        for value in (chain_tool_ids or [])
        if str(value).strip()
    )
    normalized_chain_statuses = tuple(
        str(value).strip()
        for value in (chain_runtime_statuses or [])
        if str(value).strip()
    )
    promotion_pressure = bool(promotion_targets or int(chain_depth) > 1)
    return ExecutionEvent(
        tool_id=str(tool_id),
        query_context=str(query_context),
        specialist_id=(None if specialist_id is None else str(specialist_id)),
        math_core_tier=extract_math_core_tier(execution_plan, result),
        execution_us=max(0, int(execution_us)),
        outcome=int(outcome),
        quality_signal=float(quality_signal),
        ternary_quality=int(ternary_quality),
        timestamp_us=timestamp_us(),
        chain_depth=max(1, int(chain_depth)),
        promotion_pressure=bool(promotion_pressure),
        domain_hint=(None if domain_hint is None else str(domain_hint)),
        execution_mode=str(execution_mode or (execution_plan or {}).get("mode", "tool_entrypoint_chain")),
        runtime_status=str(runtime_status),
        chain_tool_ids=normalized_chain_tool_ids,
        chain_runtime_statuses=normalized_chain_statuses,
        error=(None if error is None else f"{type(error).__name__}: {error}"),
    )


def attach_execution_event(result: Any, event: ExecutionEvent) -> Any:
    metadata_update = {
        "execution_event": event.as_dict(),
        "quality_signal": float(event.quality_signal),
        "ternary_quality": int(event.ternary_quality),
        "execution_outcome": int(event.outcome),
        "execution_us": int(event.execution_us),
        "promotion_pressure": bool(event.promotion_pressure),
        "chain_tool_ids": [str(value) for value in event.chain_tool_ids],
        "chain_runtime_statuses": [str(value) for value in event.chain_runtime_statuses],
    }
    if result is None:
        return result
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        merged = dict(metadata)
        merged.update(metadata_update)
        if is_dataclass(result):
            names = {field.name for field in fields(result)}
            if "metadata" in names:
                return replace(result, metadata=merged)
        try:
            setattr(result, "metadata", merged)
            return result
        except Exception:
            pass
    if isinstance(result, Mapping):
        updated = dict(result)
        updated.update(metadata_update)
        return updated
    return result


__all__ = [
    "ExecutionEvent",
    "ExecutionEventRecorder",
    "attach_execution_event",
    "build_execution_event",
    "extract_quality_signal",
    "ternary_quantize_quality",
    "timestamp_us",
]
