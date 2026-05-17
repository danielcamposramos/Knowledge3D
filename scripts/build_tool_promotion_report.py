#!/usr/bin/env python3
"""Aggregate Tool promotion pressure and execution telemetry into a ranked JSON report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import re
from typing import Any


_ALREADY_PROMOTED_TARGETS = {
    "TRIPLANAR_MAP",
    "RESHAPE_TO_BLOCKS",
    "BLOCKS_TO_GRID",
    "DCT8X8_FORWARD",
    "IDCT8X8_INVERSE",
    "SIGNAL_SURFACE_VERTICES",
    "SIGNAL_SURFACE_NORMALS",
    "TEMPORAL_FRAME_SYNTHESIS",
    "TEMPORAL_PRESET_APPLY",
}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                row = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                row["_line_number"] = line_number
                rows.append(row)
    return rows


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _rank(counter: Counter[str], *, limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"name": name, "count": int(count)}
        for name, count in counter.most_common(limit)
    ]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def _normalize(value: float, upper: float) -> float:
    if upper <= 1e-12:
        return 0.0
    return max(0.0, min(float(value) / float(upper), 1.0))


def _tokenize(text: str) -> list[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text or ""))
    return [token for token in re.split(r"[^a-z0-9]+", expanded.lower()) if token]


def _aggregate_event_tools(event_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    per_tool: dict[str, dict[str, Any]] = {}
    for row in event_rows:
        tool_id = str(row.get("tool_id", "")).strip()
        if not tool_id:
            continue
        record = per_tool.setdefault(
            tool_id,
            {
                "tool_id": tool_id,
                "execution_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "uncertain_count": 0,
                "promotion_pressure_count": 0,
                "execution_us_sum": 0,
                "quality_signal_sum": 0.0,
                "chain_depth_sum": 0,
                "runtime_statuses": Counter(),
                "execution_modes": Counter(),
            },
        )
        record["execution_count"] += 1
        outcome = int(row.get("outcome", 0) or 0)
        if outcome > 0:
            record["success_count"] += 1
        elif outcome < 0:
            record["failure_count"] += 1
        else:
            record["uncertain_count"] += 1
        if bool(row.get("promotion_pressure", False)):
            record["promotion_pressure_count"] += 1
        record["execution_us_sum"] += max(0, int(row.get("execution_us", 0) or 0))
        record["quality_signal_sum"] += max(0.0, min(_safe_float(row.get("quality_signal", 0.0)), 1.0))
        record["chain_depth_sum"] += max(1, int(row.get("chain_depth", 1) or 1))
        runtime_status = str(row.get("runtime_status", "")).strip()
        if runtime_status:
            record["runtime_statuses"][runtime_status] += 1
        execution_mode = str(row.get("execution_mode", "")).strip()
        if execution_mode:
            record["execution_modes"][execution_mode] += 1

    for record in per_tool.values():
        total = max(1, int(record["execution_count"]))
        record["avg_execution_us"] = float(record["execution_us_sum"]) / float(total)
        record["avg_quality_signal"] = float(record["quality_signal_sum"]) / float(total)
        record["success_rate"] = float(record["success_count"]) / float(total)
        record["avg_chain_depth"] = float(record["chain_depth_sum"]) / float(total)
        record["runtime_statuses"] = _rank(record["runtime_statuses"])
        record["execution_modes"] = _rank(record["execution_modes"])
    return per_tool


def _aggregate_grammar_support(grammar_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    per_tool: dict[str, dict[str, Any]] = {}
    for row in grammar_rows:
        sequence = [
            str(value).strip()
            for value in row.get("sequence", []) or []
            if str(value).strip()
        ]
        if not sequence:
            continue
        occurrence_count = max(1, int(row.get("count", 0) or 0))
        avg_quality_signal = max(0.0, min(_safe_float(row.get("avg_quality_signal", 0.0)), 1.0))
        for tool_id in sequence:
            record = per_tool.setdefault(
                tool_id,
                {
                    "tool_id": tool_id,
                    "pattern_count": 0,
                    "occurrence_count": 0,
                    "quality_sum": 0.0,
                },
            )
            record["pattern_count"] += 1
            record["occurrence_count"] += occurrence_count
            record["quality_sum"] += avg_quality_signal * float(occurrence_count)

    for record in per_tool.values():
        occurrences = max(1, int(record["occurrence_count"]))
        record["avg_quality_signal"] = float(record["quality_sum"]) / float(occurrences)
    return per_tool


def _aggregate_multimodal_grammar_support(grammar_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in grammar_rows:
        event_name = str(row.get("event", "")).strip()
        if event_name not in {
            "execution_multimodal_grammar_promoted",
            "execution_multimodal_antipattern_promoted",
        }:
            continue
        polarity = "positive" if "antipattern" not in event_name else "negative"
        query_tokens = [
            str(value).strip().lower()
            for value in row.get("query_tokens", []) or []
            if str(value).strip()
        ]
        family_tokens = [
            str(value).strip().lower()
            for value in row.get("tool_family_tokens", []) or []
            if str(value).strip()
        ]
        modalities = [
            str(value).strip().lower()
            for value in row.get("modalities", []) or []
            if str(value).strip()
        ]
        route_sources = [
            str(value).strip().lower()
            for value in row.get("route_sources", []) or []
            if str(value).strip()
        ]
        rows.append(
            {
                "rule_id": str(row.get("rule_id", "")).strip(),
                "polarity": polarity,
                "count": max(1, int(row.get("count", 0) or 0)),
                "avg_quality_signal": max(0.0, min(_safe_float(row.get("avg_quality_signal", 0.0)), 1.0)),
                "query_tokens": query_tokens,
                "tool_family_tokens": family_tokens,
                "modalities": modalities,
                "route_sources": route_sources,
            }
        )
    return rows


def _aggregate_chain_patterns(event_rows: list[dict[str, Any]]) -> Counter[str]:
    sequences: Counter[str] = Counter()
    for row in event_rows:
        if str(row.get("execution_mode", "")).strip() == "tool_chain_step":
            continue
        chain = [
            str(value).strip()
            for value in row.get("chain_tool_ids", []) or []
            if str(value).strip()
        ]
        if len(chain) >= 2:
            sequences[" -> ".join(chain)] += 1
    return sequences


def _quality_level(record: dict[str, Any] | None) -> float:
    if not isinstance(record, dict):
        return 0.5
    bayesian = _safe_float(record.get("bayesian_quality", 0.5), 0.5)
    recent = [
        _safe_float(v, 0.0)
        for v in record.get("recent_quality", []) or []
        if isinstance(v, (int, float))
    ]
    recent_mean = _mean(recent) if recent else _safe_float(record.get("last_quality_signal", bayesian), bayesian)
    return max(0.0, min((0.3 * bayesian) + (0.7 * recent_mean), 1.0))


def _build_quality_state_rankings(quality_state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    route_sources = dict(quality_state.get("route_sources", {}) or {})
    tool_sources = dict(quality_state.get("tool_sources", {}) or {})

    route_rows: list[dict[str, Any]] = []
    for route_source, record in route_sources.items():
        if not isinstance(record, dict):
            continue
        route_rows.append(
            {
                "name": str(route_source),
                "total_executions": int(record.get("total_executions", 0) or 0),
                "bayesian_quality": round(_safe_float(record.get("bayesian_quality", 0.5), 0.5), 6),
                "quality_level": round(_quality_level(record), 6),
                "ternary_trend": int(record.get("ternary_trend", 0) or 0),
                "avg_execution_us": round(_safe_float(record.get("avg_execution_us", 0.0), 0.0), 3),
            }
        )
    route_rows.sort(
        key=lambda row: (
            -float(row["quality_level"]),
            -int(row["total_executions"]),
            row["name"],
        )
    )

    tool_source_rows: list[dict[str, Any]] = []
    for key, record in tool_sources.items():
        if not isinstance(record, dict):
            continue
        tool_source_rows.append(
            {
                "name": str(key),
                "tool_id": str(record.get("tool_id", "")).strip(),
                "route_source": str(record.get("route_source", "")).strip(),
                "total_executions": int(record.get("total_executions", 0) or 0),
                "bayesian_quality": round(_safe_float(record.get("bayesian_quality", 0.5), 0.5), 6),
                "quality_level": round(_quality_level(record), 6),
                "ternary_trend": int(record.get("ternary_trend", 0) or 0),
                "avg_execution_us": round(_safe_float(record.get("avg_execution_us", 0.0), 0.0), 3),
            }
        )
    tool_source_rows.sort(
        key=lambda row: (
            -float(row["quality_level"]),
            -int(row["total_executions"]),
            row["name"],
        )
    )
    return {
        "route_source_quality": route_rows[:20],
        "tool_source_quality": tool_source_rows[:20],
    }


def _build_candidate_rankings(
    pressure_rows: list[dict[str, Any]],
    *,
    event_tool_stats: dict[str, dict[str, Any]],
    grammar_tool_stats: dict[str, dict[str, Any]],
    multimodal_grammar_rows: list[dict[str, Any]] | None = None,
    quality_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    quality_state = dict(quality_state or {})
    multimodal_grammar_rows = list(multimodal_grammar_rows or [])
    tool_sources = {
        str(key): value
        for key, value in dict(quality_state.get("tool_sources", {}) or {}).items()
        if isinstance(value, dict)
    }
    route_sources = {
        str(key): value
        for key, value in dict(quality_state.get("route_sources", {}) or {}).items()
        if isinstance(value, dict)
    }
    candidates: dict[str, dict[str, Any]] = {}
    for row in pressure_rows:
        primary_tool = str(row.get("primary_tool_id", "")).strip()
        tool_ids = {
            str(value).strip()
            for value in row.get("tool_ids", []) or []
            if str(value).strip()
        }
        if primary_tool:
            tool_ids.add(primary_tool)
        query = str(row.get("query", "")).strip()
        entrypoints = {
            str(value).strip()
            for value in row.get("entrypoints", []) or []
            if str(value).strip()
        }
        for target in row.get("promotion_targets", []) or []:
            token = str(target).strip()
            if not token:
                continue
            query_tokens = {item for item in _tokenize(query) if item}
            tool_kind_tokens = {
                item
                for kind in row.get("tool_kinds", []) or []
                for item in _tokenize(str(kind))
                if item
            }
            tool_id_tokens = {
                item
                for tool_id in tool_ids
                for item in _tokenize(str(tool_id))
                if item
            }
            modality_tokens = {
                item
                for item in row.get("modalities", []) or []
                if str(item).strip()
            }
            record = candidates.setdefault(
                token,
                {
                    "name": token,
                    "already_promoted": token in _ALREADY_PROMOTED_TARGETS,
                    "pressure_count": 0,
                    "primary_tools": set(),
                    "tool_ids": set(),
                    "queries": set(),
                    "entrypoints": set(),
                    "route_sources": Counter(),
                    "query_tokens": set(),
                    "tool_kind_tokens": set(),
                    "tool_id_tokens": set(),
                    "modalities": set(),
                },
            )
            record["pressure_count"] += 1
            if primary_tool:
                record["primary_tools"].add(primary_tool)
            record["tool_ids"].update(tool_ids)
            if query:
                record["queries"].add(query)
            record["entrypoints"].update(entrypoints)
            record["query_tokens"].update(query_tokens)
            record["tool_kind_tokens"].update(tool_kind_tokens)
            record["tool_id_tokens"].update(tool_id_tokens)
            record["modalities"].update(str(item).strip().lower() for item in modality_tokens)
            for runtime_status in row.get("runtime_statuses", []) or []:
                normalized = str(runtime_status).strip().lower()
                if normalized == "ptx_rpn_available":
                    record["route_sources"]["kernel"] += 1
                elif normalized == "ptx_bridge_available" or normalized.startswith("ptx_"):
                    record["route_sources"]["bridge"] += 1
                else:
                    record["route_sources"]["recipe"] += 1

    if not candidates:
        return []

    max_pressure = max(record["pressure_count"] for record in candidates.values())
    max_latency = 0.0
    max_grammar_occurrences = 0
    for record in candidates.values():
        associated_tools = [event_tool_stats[tool] for tool in record["tool_ids"] if tool in event_tool_stats]
        executions = sum(int(tool.get("execution_count", 0) or 0) for tool in associated_tools)
        execution_us_sum = sum(float(tool.get("avg_execution_us", 0.0)) * int(tool.get("execution_count", 0) or 0) for tool in associated_tools)
        quality_sum = sum(float(tool.get("avg_quality_signal", 0.0)) * int(tool.get("execution_count", 0) or 0) for tool in associated_tools)
        success_sum = sum(float(tool.get("success_count", 0) or 0) for tool in associated_tools)
        failure_sum = sum(float(tool.get("failure_count", 0) or 0) for tool in associated_tools)
        uncertain_sum = sum(float(tool.get("uncertain_count", 0) or 0) for tool in associated_tools)
        source_quality_values = [
            _quality_level(tool_sources[f"{tool}::{route_source}"])
            for tool in record["tool_ids"]
            for route_source in ("recipe", "bridge", "kernel")
            if f"{tool}::{route_source}" in tool_sources
        ]
        route_quality_values = [
            _quality_level(route_sources[route_source])
            for route_source in record["route_sources"]
            if route_source in route_sources
        ]
        grammar_occurrences = sum(
            int(grammar_tool_stats[tool].get("occurrence_count", 0) or 0)
            for tool in record["tool_ids"]
            if tool in grammar_tool_stats
        )
        record["event_execution_count"] = int(executions)
        record["avg_execution_us"] = (execution_us_sum / float(executions)) if executions else 0.0
        record["avg_quality_signal"] = (quality_sum / float(executions)) if executions else 0.0
        record["success_rate"] = (success_sum / float(executions)) if executions else 0.0
        record["failure_rate"] = (failure_sum / float(executions)) if executions else 0.0
        record["uncertain_rate"] = (uncertain_sum / float(executions)) if executions else 0.0
        record["source_quality_level"] = _mean(source_quality_values)
        record["route_quality_level"] = _mean(route_quality_values)
        record["grammar_occurrence_count"] = int(grammar_occurrences)
        record["grammar_pattern_count"] = sum(
            int(grammar_tool_stats[tool].get("pattern_count", 0) or 0)
            for tool in record["tool_ids"]
            if tool in grammar_tool_stats
        )
        multimodal_positive_support = 0
        multimodal_negative_support = 0
        multimodal_positive_patterns = 0
        multimodal_negative_patterns = 0
        for grammar_row in multimodal_grammar_rows:
            grammar_token_pool = set(grammar_row.get("query_tokens", [])) | set(grammar_row.get("tool_family_tokens", []))
            grammar_modalities = set(grammar_row.get("modalities", []))
            grammar_routes = set(grammar_row.get("route_sources", []))
            candidate_token_pool = set(record["query_tokens"]) | set(record["tool_kind_tokens"]) | set(record["tool_id_tokens"])
            token_overlap = candidate_token_pool & grammar_token_pool
            modality_overlap = set(record["modalities"]) & grammar_modalities
            route_overlap = set(record["route_sources"]) & grammar_routes
            if len(token_overlap) < 2 and not (modality_overlap and route_overlap):
                continue
            support = int(grammar_row.get("count", 0) or 0)
            if str(grammar_row.get("polarity", "positive")) == "negative":
                multimodal_negative_support += support
                multimodal_negative_patterns += 1
            else:
                multimodal_positive_support += support
                multimodal_positive_patterns += 1
        record["multimodal_positive_support"] = int(multimodal_positive_support)
        record["multimodal_negative_support"] = int(multimodal_negative_support)
        record["multimodal_positive_patterns"] = int(multimodal_positive_patterns)
        record["multimodal_negative_patterns"] = int(multimodal_negative_patterns)
        max_latency = max(max_latency, float(record["avg_execution_us"]))
        max_grammar_occurrences = max(max_grammar_occurrences, int(grammar_occurrences))

    ranked: list[dict[str, Any]] = []
    max_multimodal_positive = max(
        [int(record.get("multimodal_positive_support", 0) or 0) for record in candidates.values()] or [0]
    )
    max_multimodal_negative = max(
        [int(record.get("multimodal_negative_support", 0) or 0) for record in candidates.values()] or [0]
    )
    for record in candidates.values():
        frequency_norm = _normalize(float(record["pressure_count"]), float(max_pressure))
        latency_norm = _normalize(float(record["avg_execution_us"]), float(max_latency))
        quality_gap = max(0.0, min(1.0 - float(record["avg_quality_signal"]), 1.0))
        source_gap = max(0.0, min(1.0 - float(record["source_quality_level"]), 1.0))
        grammar_norm = _normalize(float(record["grammar_occurrence_count"]), float(max_grammar_occurrences))
        multimodal_positive_norm = _normalize(float(record["multimodal_positive_support"]), float(max_multimodal_positive))
        multimodal_negative_norm = _normalize(float(record["multimodal_negative_support"]), float(max_multimodal_negative))
        support_norm = _normalize(float(len(record["primary_tools"])), float(max(1, max(len(r["primary_tools"]) for r in candidates.values()))))
        already_promoted = bool(record.get("already_promoted", False))
        priority_score = (
            0.28 * frequency_norm
            + 0.20 * latency_norm
            + 0.16 * quality_gap
            + 0.10 * source_gap
            + 0.10 * grammar_norm
            + 0.09 * multimodal_negative_norm
            + 0.04 * multimodal_positive_norm
            + 0.05 * support_norm
        )
        readiness_score = (
            0.28 * frequency_norm
            + 0.23 * (1.0 - quality_gap)
            + 0.10 * (1.0 - source_gap)
            + 0.20 * grammar_norm
            + 0.09 * multimodal_positive_norm
            + 0.05 * (1.0 - multimodal_negative_norm)
            + 0.05 * (1.0 - float(record["failure_rate"]))
        )
        if already_promoted:
            priority_score *= 0.15
            readiness_score *= 0.35
        dominant_route_source = ""
        if record["route_sources"]:
            dominant_route_source = record["route_sources"].most_common(1)[0][0]
        ranked.append(
            {
                "name": str(record["name"]),
                "promotion_status": "materialized" if already_promoted else "candidate",
                "already_promoted": bool(already_promoted),
                "pressure_count": int(record["pressure_count"]),
                "primary_tool_count": int(len(record["primary_tools"])),
                "event_execution_count": int(record["event_execution_count"]),
                "avg_execution_us": round(float(record["avg_execution_us"]), 3),
                "avg_quality_signal": round(float(record["avg_quality_signal"]), 6),
                "quality_gap": round(float(quality_gap), 6),
                "source_quality_level": round(float(record["source_quality_level"]), 6),
                "source_quality_gap": round(float(source_gap), 6),
                "route_quality_level": round(float(record["route_quality_level"]), 6),
                "success_rate": round(float(record["success_rate"]), 6),
                "failure_rate": round(float(record["failure_rate"]), 6),
                "uncertain_rate": round(float(record["uncertain_rate"]), 6),
                "grammar_occurrence_count": int(record["grammar_occurrence_count"]),
                "grammar_pattern_count": int(record["grammar_pattern_count"]),
                "multimodal_positive_support": int(record["multimodal_positive_support"]),
                "multimodal_negative_support": int(record["multimodal_negative_support"]),
                "multimodal_positive_patterns": int(record["multimodal_positive_patterns"]),
                "multimodal_negative_patterns": int(record["multimodal_negative_patterns"]),
                "promotion_priority_score": round(float(priority_score), 6),
                "promotion_readiness_score": round(float(readiness_score), 6),
                "dominant_route_source": dominant_route_source,
                "route_source_counts": [
                    {"name": name, "count": int(count)}
                    for name, count in record["route_sources"].most_common()
                ],
                "primary_tools": sorted(record["primary_tools"]),
                "tool_ids": sorted(record["tool_ids"]),
                "entrypoints": sorted(record["entrypoints"]),
                "sample_queries": sorted(record["queries"])[:5],
                "sample_query_tokens": sorted(record["query_tokens"])[:8],
                "sample_tool_kind_tokens": sorted(record["tool_kind_tokens"])[:8],
            }
        )
    ranked.sort(
        key=lambda row: (
            bool(row.get("already_promoted", False)),
            -float(row["promotion_priority_score"]),
            -int(row["pressure_count"]),
            -float(row["avg_execution_us"]),
            row["name"],
        )
    )
    return ranked


def build_report(
    rows: list[dict[str, Any]],
    *,
    source_path: Path,
    event_rows: list[dict[str, Any]] | None = None,
    event_source_path: Path | None = None,
    grammar_rows: list[dict[str, Any]] | None = None,
    grammar_source_path: Path | None = None,
    quality_state: dict[str, Any] | None = None,
    quality_state_path: Path | None = None,
) -> dict[str, Any]:
    primary_tool_counter: Counter[str] = Counter()
    tool_counter: Counter[str] = Counter()
    tool_kind_counter: Counter[str] = Counter()
    runtime_counter: Counter[str] = Counter()
    promotion_counter: Counter[str] = Counter()
    entrypoint_counter: Counter[str] = Counter()
    codec_counter: Counter[str] = Counter()
    specialist_counter: Counter[str] = Counter()
    source_target_counter: Counter[str] = Counter()
    primary_tool_target_counter: Counter[str] = Counter()
    math_core_tier_counter: Counter[str] = Counter()
    math_core_role_counter: Counter[str] = Counter()
    math_core_cascade_counter: Counter[str] = Counter()

    recent_queries: list[dict[str, Any]] = []
    event_rows = list(event_rows or [])
    grammar_rows = list(grammar_rows or [])
    quality_state = dict(quality_state or {})

    for row in rows:
        primary_tool = str(row.get("primary_tool_id", "")).strip()
        if primary_tool:
            primary_tool_counter[primary_tool] += 1
        for item in row.get("tool_ids", []) or []:
            token = str(item).strip()
            if token:
                tool_counter[token] += 1
        for item in row.get("tool_kinds", []) or []:
            token = str(item).strip()
            if token:
                tool_kind_counter[token] += 1
        for item in row.get("runtime_statuses", []) or []:
            token = str(item).strip()
            if token:
                runtime_counter[token] += 1
        for item in row.get("promotion_targets", []) or []:
            token = str(item).strip()
            if token:
                promotion_counter[token] += 1
                if primary_tool:
                    primary_tool_target_counter[f"{primary_tool}::{token}"] += 1
        for item in row.get("entrypoints", []) or []:
            token = str(item).strip()
            if token:
                entrypoint_counter[token] += 1
        for item in row.get("codec_ops", []) or []:
            token = str(item).strip()
            if token:
                codec_counter[token] += 1
        for item in row.get("math_core_tiers", []) or []:
            token = str(item).strip()
            if token:
                math_core_tier_counter[token] += 1
        for item in row.get("math_core_roles", []) or []:
            token = str(item).strip()
            if token:
                math_core_role_counter[token] += 1
        for item in row.get("math_core_cascades", []) or []:
            token = str(item).strip()
            if token:
                math_core_cascade_counter[token] += 1
        specialist = str(row.get("specialist", "")).strip()
        if specialist:
            specialist_counter[specialist] += 1
        source_galaxy = str(row.get("source_galaxy", "")).strip()
        target_galaxy = str(row.get("target_galaxy", "")).strip()
        if source_galaxy or target_galaxy:
            source_target_counter[f"{source_galaxy}->{target_galaxy}"] += 1

        if len(recent_queries) < 10:
            recent_queries.append(
                {
                    "timestamp": str(row.get("timestamp", "")).strip(),
                    "query": str(row.get("query", "")).strip(),
                    "primary_tool_id": primary_tool,
                    "promotion_targets": [str(item).strip() for item in row.get("promotion_targets", []) if str(item).strip()],
                }
            )

    event_tool_stats = _aggregate_event_tools(event_rows)
    grammar_tool_stats = _aggregate_grammar_support(grammar_rows)
    multimodal_grammar_stats = _aggregate_multimodal_grammar_support(grammar_rows)
    chain_counter = _aggregate_chain_patterns(event_rows)
    quality_rankings = _build_quality_state_rankings(quality_state)
    candidate_rankings = _build_candidate_rankings(
        rows,
        event_tool_stats=event_tool_stats,
        grammar_tool_stats=grammar_tool_stats,
        multimodal_grammar_rows=multimodal_grammar_stats,
        quality_state=quality_state,
    )
    top_candidate = candidate_rankings[0] if candidate_rankings else None

    return {
        "source_path": str(source_path),
        "source_exists": bool(source_path.exists()),
        "rows": int(len(rows)),
        "event_rows": int(len(event_rows)),
        "grammar_rows": int(len(grammar_rows)),
        "source_paths": {
            "pressure": str(source_path),
            "events": str(event_source_path or ""),
            "grammar": str(grammar_source_path or ""),
            "quality_state": str(quality_state_path or ""),
        },
        "source_exists_map": {
            "pressure": bool(source_path.exists()),
            "events": bool(event_source_path.exists()) if event_source_path is not None else False,
            "grammar": bool(grammar_source_path.exists()) if grammar_source_path is not None else False,
            "quality_state": bool(quality_state_path.exists()) if quality_state_path is not None else False,
        },
        "stats": {
            "distinct_primary_tools": int(len(primary_tool_counter)),
            "distinct_tools": int(len(tool_counter)),
            "distinct_tool_kinds": int(len(tool_kind_counter)),
            "distinct_promotion_targets": int(len(promotion_counter)),
            "distinct_entrypoints": int(len(entrypoint_counter)),
            "distinct_event_tools": int(len(event_tool_stats)),
            "distinct_event_chains": int(len(chain_counter)),
            "distinct_grammar_supported_tools": int(len(grammar_tool_stats)),
            "distinct_multimodal_grammar_patterns": int(len(multimodal_grammar_stats)),
            "distinct_route_sources": int(len(dict(quality_state.get("route_sources", {}) or {}))),
            "distinct_tool_sources": int(len(dict(quality_state.get("tool_sources", {}) or {}))),
        },
        "rankings": {
            "primary_tools": _rank(primary_tool_counter),
            "tools": _rank(tool_counter),
            "tool_kinds": _rank(tool_kind_counter),
            "runtime_statuses": _rank(runtime_counter),
            "promotion_targets": _rank(promotion_counter),
            "entrypoints": _rank(entrypoint_counter),
            "codec_ops": _rank(codec_counter),
            "math_core_tiers": _rank(math_core_tier_counter),
            "math_core_roles": _rank(math_core_role_counter),
            "math_core_cascades": _rank(math_core_cascade_counter),
            "specialists": _rank(specialist_counter),
            "source_target_routes": _rank(source_target_counter),
            "primary_tool_targets": _rank(primary_tool_target_counter),
            "event_tools": sorted(
                (
                    {
                        "name": tool_id,
                        "execution_count": int(record["execution_count"]),
                        "success_rate": round(float(record["success_rate"]), 6),
                        "avg_execution_us": round(float(record["avg_execution_us"]), 3),
                        "avg_quality_signal": round(float(record["avg_quality_signal"]), 6),
                        "promotion_pressure_count": int(record["promotion_pressure_count"]),
                    }
                    for tool_id, record in event_tool_stats.items()
                ),
                key=lambda row: (-int(row["execution_count"]), -float(row["avg_quality_signal"]), row["name"]),
            )[:20],
            "event_chains": _rank(chain_counter),
            "promotion_candidates": candidate_rankings[:20],
            **quality_rankings,
        },
        "candidate_summary": {
            "top_candidate": top_candidate,
            "candidate_count": int(len(candidate_rankings)),
        },
        "recent_queries": recent_queries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/K3D/Knowledge3D.local/logs/tool_promotion_pressure.jsonl",
        help="Path to tool_promotion_pressure.jsonl",
    )
    parser.add_argument(
        "--events-input",
        default="/K3D/Knowledge3D.local/logs/execution_events.jsonl",
        help="Path to execution_events.jsonl",
    )
    parser.add_argument(
        "--grammar-input",
        default="/K3D/Knowledge3D.local/logs/execution_grammar_patterns.jsonl",
        help="Path to execution_grammar_patterns.jsonl",
    )
    parser.add_argument(
        "--quality-state-input",
        default="/K3D/Knowledge3D.local/checkpoints/execution_quality_tracker.json",
        help="Path to execution_quality_tracker.json",
    )
    parser.add_argument(
        "--output",
        default="/K3D/Knowledge3D.local/results/tool_promotion_report.json",
        help="Path to write aggregated JSON report",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    event_input_path = Path(args.events_input)
    grammar_input_path = Path(args.grammar_input)
    quality_state_input_path = Path(args.quality_state_input)
    output_path = Path(args.output)
    rows = _load_rows(input_path)
    event_rows = _load_rows(event_input_path)
    grammar_rows = _load_rows(grammar_input_path)
    quality_state = _load_json_object(quality_state_input_path)
    report = build_report(
        rows,
        source_path=input_path,
        event_rows=event_rows,
        event_source_path=event_input_path,
        grammar_rows=grammar_rows,
        grammar_source_path=grammar_input_path,
        quality_state=quality_state,
        quality_state_path=quality_state_input_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "[tool-promotion-report] "
        f"pressure_rows={len(rows)} event_rows={len(event_rows)} grammar_rows={len(grammar_rows)} "
        f"output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
