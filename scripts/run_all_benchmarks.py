#!/usr/bin/env python3
"""Run all Week 14 benchmarks and write a unified report."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure repo root is importable when script is executed via "python scripts/..."
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.arc_agi_2 import ARCAGI2Benchmark
from benchmarks.last_humanity_exam import LastHumanityExamBenchmark
from benchmarks.math_competitions import MathCompetitionBenchmark
from benchmarks.mmlu import MMLUBenchmark
from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC
from knowledge3d.cranium.sovereign.loader import get_vram_usage
from knowledge3d.gpu.perf_counters import gpu_utilisation
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


BENCHMARK_CONFIG: dict[str, dict[str, str]] = {
    "arc_agi_2": {
        "metric": "accuracy",
        "label": "ARC-AGI 2",
        "specialist": "visual",
        "galaxy": "Drawing",
    },
    "math_competitions": {
        "metric": "overall_accuracy",
        "label": "Math",
        "specialist": "math",
        "galaxy": "Math",
    },
    "last_humanity_exam": {
        "metric": "accuracy",
        "label": "LHE",
        "specialist": "grammar",
        "galaxy": "Grammar",
    },
    "mmlu": {
        "metric": "accuracy",
        "label": "MMLU",
        "specialist": "chat",
        "galaxy": "Multi",
    },
}


class SovereigntyViolation(RuntimeError):
    """Raised when solved benchmark tasks do not show sovereign GPU evidence."""


def _build_tablet_boundary(kv: Knowledgeverse) -> HeadlessTabletMPC:
    return HeadlessTabletMPC(knowledgeverse=kv)


def _safe_gpu_snapshot() -> dict[str, Any]:
    used = 0
    total = 0
    util = 0.0
    try:
        used, total = get_vram_usage()
    except Exception:
        pass
    try:
        util = float(gpu_utilisation(default=0.0))
    except Exception:
        util = 0.0
    return {
        "vram_used_bytes": int(used),
        "vram_total_bytes": int(total),
        "gpu_utilization": float(util),
    }


def _rss_bytes() -> int:
    try:
        # On Linux ru_maxrss is KiB.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return 0


def _collect_default_galaxy_counts(kv: Knowledgeverse) -> dict[str, int]:
    names = tuple(getattr(kv, "DEFAULT_GALAXIES", ()))
    if not names:
        return {}
    counts: dict[str, int] = {}
    for name in names:
        try:
            galaxy = kv.galaxy_manager.get_galaxy(name)
            counts[str(name)] = int(len(galaxy.entries))
        except Exception:
            counts[str(name)] = -1
    return counts


def _run_with_metrics(label: str, fn: Any) -> tuple[Any, dict[str, Any]]:
    start_ts = datetime.now(tz=timezone.utc).isoformat()
    start = time.perf_counter()
    gpu_before = _safe_gpu_snapshot()
    rss_before = _rss_bytes()
    result = fn()
    elapsed = time.perf_counter() - start
    gpu_after = _safe_gpu_snapshot()
    rss_after = _rss_bytes()
    return result, {
        "label": label,
        "started_at": start_ts,
        "elapsed_sec": float(elapsed),
        "pid": int(os.getpid()),
        "rss_before_bytes": int(rss_before),
        "rss_after_bytes": int(rss_after),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
    }


def _skip_metrics(label: str) -> dict[str, Any]:
    ts = datetime.now(tz=timezone.utc).isoformat()
    gpu = _safe_gpu_snapshot()
    rss = _rss_bytes()
    return {
        "label": label,
        "started_at": ts,
        "elapsed_sec": 0.0,
        "pid": int(os.getpid()),
        "rss_before_bytes": int(rss),
        "rss_after_bytes": int(rss),
        "gpu_before": gpu,
        "gpu_after": gpu,
        "skipped": True,
    }


def _is_enabled_limit(limit: int | None) -> bool:
    # Safety semantics: <=0 means "skip this benchmark".
    if limit is None:
        return True
    try:
        return int(limit) > 0
    except Exception:
        return False


def _arc_skipped_result(*, dataset_path: str | None, dataset_version: str, use_enriched: bool) -> dict[str, Any]:
    return {
        "benchmark": "ARC-AGI 2/3",
        "dataset_path": dataset_path or "skipped",
        "dataset_version": dataset_version,
        "use_enriched": bool(use_enriched),
        "total_tasks": 0,
        "correct": 0,
        "accuracy": 0.0,
        "generated_pattern_total": 0,
        "tasks_with_generated_patterns": 0,
        "pattern_source_accuracy": {},
        "oracle_diagnostics": {},
        "results": [],
    }


def _math_skipped_result(*, dataset_path: str | None, use_enriched: bool) -> dict[str, Any]:
    return {
        "benchmark": "Math Competitions",
        "dataset_path": dataset_path or "skipped",
        "use_enriched": bool(use_enriched),
        "results_by_competition": {},
        "overall_accuracy": 0.0,
        "total": 0,
        "correct": 0,
    }


def _lhe_skipped_result(*, dataset_path: str | None, use_enriched: bool) -> dict[str, Any]:
    return {
        "benchmark": "Last Humanity Exam",
        "dataset_path": dataset_path or "skipped",
        "dataset_source": "skipped",
        "dataset_file": None,
        "synthetic_fallback": False,
        "use_enriched": bool(use_enriched),
        "total_questions": 0,
        "correct": 0,
        "accuracy": 0.0,
        "results": [],
    }


def _mmlu_skipped_result(*, dataset_path: str | None, use_enriched: bool) -> dict[str, Any]:
    return {
        "benchmark": "MMLU",
        "dataset_path": dataset_path or "skipped",
        "dataset_source": "skipped",
        "dataset_file": None,
        "synthetic_fallback": False,
        "subjects_tested": 0,
        "use_enriched": bool(use_enriched),
        "total_questions": 0,
        "correct": 0,
        "accuracy": 0.0,
        "domain_breakdown": {},
        "subject_breakdown": {},
        "results": [],
    }


def _append_usage_metrics(log_path: Path, payload: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")


def _assert_arc_solver_contract(arc_result: dict[str, Any], *, required_solver: str) -> None:
    results = arc_result.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError("ARC result payload missing per-task results for solver contract check.")
    bad = [row.get("task_id") for row in results if str(row.get("solver")) != required_solver]
    if bad:
        sample = ", ".join(str(tid) for tid in bad[:5])
        raise RuntimeError(
            f"ARC solver contract violated: expected '{required_solver}' for all tasks; "
            f"found {len(bad)} mismatches (sample: {sample})."
        )


def _assert_tablet_boundary_contract(benchmark_name: str, payload: dict[str, Any]) -> None:
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"{benchmark_name}_tablet_contract_missing_results")
    bad_indexes = [
        index
        for index, row in enumerate(rows)
        if not isinstance(row, dict) or not bool(row.get("tablet_contract"))
    ]
    if bad_indexes:
        sample = ", ".join(str(index) for index in bad_indexes[:5])
        raise RuntimeError(
            f"{benchmark_name}_tablet_contract_violated: "
            f"{len(bad_indexes)} rows missing tablet boundary evidence (sample indexes: {sample})"
        )


def _extract_enriched_score(summary: dict[str, Any], benchmark_name: str) -> float:
    bench = summary.get("benchmarks", {}).get(benchmark_name, {})
    enriched = bench.get("enriched", {})
    metric_key = BENCHMARK_CONFIG[benchmark_name]["metric"]
    return float(enriched.get(metric_key, 0.0))


def _delta_to_ternary(delta: float, epsilon: float = 0.01) -> int:
    if delta > epsilon:
        return 1
    if delta < -epsilon:
        return -1
    return 0


def _score_to_ternary(score: float) -> int:
    if score < 0.33:
        return -1
    if score < 0.66:
        return 0
    return 1


def _status_from_delta(delta: float, epsilon: float = 0.01) -> str:
    if abs(delta) < epsilon:
        return "MAINTAINED"
    return "IMPROVEMENT" if delta > 0 else "REGRESSION"


def _extract_task_gpu_used(benchmark_name: str, row: dict[str, Any]) -> bool | None:
    telemetry = row.get("telemetry")
    if isinstance(telemetry, dict):
        if "gpu_calls_this_command" in telemetry:
            try:
                return int(telemetry.get("gpu_calls_this_command", 0)) > 0
            except Exception:
                return False
    if "gpu_calls_this_command" in row:
        try:
            return int(row.get("gpu_calls_this_command", 0)) > 0
        except Exception:
            return False
    if benchmark_name == "arc_agi_2":
        arc_keys = ("ptx_full_used", "ptx_ranking_used", "ptx_oracle_used")
        if any(key in row for key in arc_keys):
            return any(bool(row.get(key, False)) for key in arc_keys)
    return None


def _extract_task_fallback_triggered(row: dict[str, Any]) -> bool:
    telemetry = row.get("telemetry")
    if isinstance(telemetry, dict):
        return bool(telemetry.get("fallback_triggered", False))
    return bool(row.get("fallback_triggered", False)) or bool(row.get("ptx_ranking_error", False))


def _summarize_benchmark_sovereignty(
    benchmark_name: str,
    phase: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        rows = []
    solved_rows = [row for row in rows if bool(row.get("correct", False))]
    solved_count = len(solved_rows)
    gpu_used = 0
    gpu_missing = 0
    gpu_unknown = 0
    fallback_count = 0
    for row in solved_rows:
        if _extract_task_fallback_triggered(row):
            fallback_count += 1
        gpu_flag = _extract_task_gpu_used(benchmark_name, row)
        if gpu_flag is True:
            gpu_used += 1
        elif gpu_flag is False:
            gpu_missing += 1
        else:
            gpu_unknown += 1
    compliance = (gpu_used / solved_count) if solved_count else 1.0
    violations: list[str] = []
    if fallback_count > 0:
        violations.append(f"fallback_triggered={fallback_count}")
    if gpu_missing > 0:
        violations.append(f"solved_without_gpu={gpu_missing}")
    if gpu_unknown > 0:
        violations.append(f"gpu_telemetry_missing_for_solved={gpu_unknown}")
    return {
        "benchmark": benchmark_name,
        "phase": phase,
        "total_tasks": int(payload.get("total_tasks", payload.get("total", payload.get("total_questions", 0)) or 0)),
        "solved_tasks": int(solved_count),
        "tasks_using_gpu": int(gpu_used),
        "tasks_without_gpu": int(gpu_missing),
        "tasks_without_gpu_telemetry": int(gpu_unknown),
        "fallback_triggered_count": int(fallback_count),
        "sovereignty_compliance": float(compliance),
        "violations": violations,
    }


def _load_previous_history_entry(history_path: Path) -> dict[str, Any] | None:
    if not history_path.exists():
        return None
    lines = [line.strip() for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _load_recent_history_entries(history_path: Path, limit: int = 3) -> list[dict[str, Any]]:
    if not history_path.exists():
        return []
    lines = [line.strip() for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    entries: list[dict[str, Any]] = []
    for raw in lines[-max(0, int(limit)) :]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def _extract_arc_generation_failure(summary_payload: dict[str, Any]) -> float | None:
    try:
        value = (
            summary_payload.get("benchmarks", {})
            .get("arc_agi_2", {})
            .get("diagnostics", {})
            .get("enriched", {})
            .get("generation_failure_rate")
        )
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _compute_adaptive_penalties_from_history(
    history_path: Path,
    *,
    lookback: int = 3,
    target_sum: float = 5.0,
    floor: float = 0.5,
) -> dict[str, float] | None:
    entries = _load_recent_history_entries(history_path, limit=max(1, int(lookback)))
    failure_totals = {"family": 0.0, "shape": 0.0, "palette": 0.0, "object_count": 0.0}
    for entry in entries:
        summary_path_raw = entry.get("summary_path")
        if not summary_path_raw:
            continue
        summary_path = Path(str(summary_path_raw))
        if not summary_path.exists():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        mode_counts = (
            payload.get("benchmarks", {})
            .get("arc_agi_2", {})
            .get("diagnostics", {})
            .get("enriched", {})
            .get("oracle_failure_mode_counts", {})
        )
        if not isinstance(mode_counts, dict):
            continue
        for key in failure_totals:
            try:
                failure_totals[key] += float(mode_counts.get(key, 0.0))
            except Exception:
                continue
    total = sum(failure_totals.values())
    if total <= 0.0:
        return None
    target = max(float(target_sum), float(floor) * len(failure_totals))
    raw = {
        key: max(float(floor), (value / total) * target)
        for key, value in failure_totals.items()
    }
    raw_sum = sum(raw.values())
    if raw_sum > 0:
        scale = target / raw_sum
        scaled = {key: max(float(floor), value * scale) for key, value in raw.items()}
    else:
        scaled = raw
    return {
        "family_penalty_weight": float(round(scaled["family"], 4)),
        "shape_penalty_weight": float(round(scaled["shape"], 4)),
        "palette_penalty_weight": float(round(scaled["palette"], 4)),
        "object_penalty_weight": float(round(scaled["object_count"], 4)),
    }


def _build_historical_comparison(
    previous: dict[str, Any] | None,
    current_summary: dict[str, Any],
) -> dict[str, Any] | None:
    if previous is None:
        return None
    prev_scores = previous.get("scores", {})
    if not isinstance(prev_scores, dict):
        return None

    out: dict[str, Any] = {}
    for bench_name, cfg in BENCHMARK_CONFIG.items():
        prev_score = float(prev_scores.get(bench_name, 0.0))
        curr_score = _extract_enriched_score(current_summary, bench_name)
        delta = curr_score - prev_score
        out[bench_name] = {
            "label": cfg["label"],
            "previous": prev_score,
            "current": curr_score,
            "delta": delta,
            "delta_ternary": _delta_to_ternary(delta),
            "current_score_ternary": _score_to_ternary(curr_score),
            "status": _status_from_delta(delta),
        }
    return out


def _append_history_entry(
    history_path: Path,
    *,
    summary_path: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": summary.get("timestamp"),
        "summary_path": str(summary_path),
        "scores": {
            bench_name: _extract_enriched_score(summary, bench_name)
            for bench_name in BENCHMARK_CONFIG
        },
    }
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, separators=(",", ":"), sort_keys=True) + "\n")
    return entry


def _persist_benchmark_memory(
    *,
    knowledgeverse: Knowledgeverse,
    summary: dict[str, Any],
    historical_comparison: dict[str, Any] | None,
) -> None:
    timestamp = str(summary.get("timestamp", "")).replace("-", "").replace(":", "").replace(".", "")
    for bench_name, cfg in BENCHMARK_CONFIG.items():
        hist = (historical_comparison or {}).get(bench_name, {})
        curr_score = _extract_enriched_score(summary, bench_name)
        delta = float(hist.get("delta", 0.0))
        status = str(hist.get("status", "INITIALIZED"))
        delta_ternary = int(hist.get("delta_ternary", 0))
        previous_score = float(hist.get("previous", curr_score))

        event_data = {
            "benchmark": bench_name,
            "label": cfg["label"],
            "previous_score": previous_score,
            "current_score": curr_score,
            "delta": delta,
            "delta_ternary": delta_ternary,
            "current_score_ternary": _score_to_ternary(curr_score),
            "status": status,
            "specialist": cfg["specialist"],
            "galaxy": cfg["galaxy"],
            "query": f"benchmark outcome {bench_name}",
            "confidence": curr_score,
            "verification": "historical_benchmark_tracking",
        }
        knowledgeverse.log_event(event_type="benchmark_outcome", event_data=event_data)

        grammar_entry = {
            "id": f"benchmark_memory_{bench_name}_{timestamp}",
            "name": f"Benchmark Memory {cfg['label']}",
            "domain": "grammar",
            "category": "benchmark_memory",
            "rpn_program": "PREV CURR SUB SIGN_TERNARY",
            "metadata": {
                "benchmark": bench_name,
                "label": cfg["label"],
                "previous_score": previous_score,
                "current_score": curr_score,
                "delta": delta,
                "delta_ternary": delta_ternary,
                "status": status,
                "generated": False,
                "source": "scripts/run_all_benchmarks.py",
            },
        }
        knowledgeverse.galaxy_manager.add_entry("Grammar", grammar_entry)


def _compute_arc_stage_gate(summary: dict[str, Any]) -> dict[str, Any]:
    """
    Hard gate for ARC stage progression (quality-aware, not accuracy-only).
    """
    total_tasks = int(
        summary.get("benchmarks", {})
        .get("arc_agi_2", {})
        .get("enriched", {})
        .get("total_tasks", 0)
        or 0
    )
    if total_tasks <= 0:
        return {
            "passed": True,
            "skipped": True,
            "thresholds": {},
            "actuals": {},
            "checks": {},
            "reason": "ARC skipped (max-arc-tasks <= 0)",
        }
    diagnostics = (
        summary.get("benchmarks", {})
        .get("arc_agi_2", {})
        .get("diagnostics", {})
        .get("enriched", {})
    )
    coverage_gate = (
        summary.get("runtime_usage", {})
        .get("curriculum_coverage", {})
        .get("coverage_gate", {})
    )
    enriched = summary.get("benchmarks", {}).get("arc_agi_2", {}).get("enriched", {})
    thresholds = {
        "accuracy_min": 0.10,
        "oracle_at_all_min": 0.10,
        "fuzzy_oracle_at_all_min": 0.20,
        "generation_failure_rate_max": 0.50,
    }
    min_query_galaxies = int(coverage_gate.get("min_required", 0) or 0)
    min_cross_rate = float(coverage_gate.get("min_cross_galaxy_navigation_rate", 0.0) or 0.0)
    if min_query_galaxies > 0:
        thresholds["query_galaxies_min"] = float(min_query_galaxies)
    if min_cross_rate > 0.0:
        thresholds["cross_galaxy_navigation_rate_min"] = float(min_cross_rate)
    actuals = {
        "accuracy": float(enriched.get("accuracy", 0.0)),
        "oracle_at_all": float(diagnostics.get("oracle_at_all", 0.0)),
        "fuzzy_oracle_at_all": float(diagnostics.get("fuzzy_oracle_at_all", 0.0)),
        "generation_failure_rate": float(diagnostics.get("generation_failure_rate", 1.0)),
    }
    if min_query_galaxies > 0:
        actuals["query_galaxies"] = float(coverage_gate.get("enriched_query_count", 0) or 0)
    if min_cross_rate > 0.0:
        actuals["cross_galaxy_navigation_rate"] = float(
            coverage_gate.get("enriched_cross_galaxy_navigation_rate", 0.0) or 0.0
        )
    checks = {
        "accuracy": actuals["accuracy"] >= thresholds["accuracy_min"],
        "oracle_at_all": actuals["oracle_at_all"] >= thresholds["oracle_at_all_min"],
        "fuzzy_oracle_at_all": actuals["fuzzy_oracle_at_all"] >= thresholds["fuzzy_oracle_at_all_min"],
        "generation_failure_rate": actuals["generation_failure_rate"] <= thresholds["generation_failure_rate_max"],
    }
    if min_query_galaxies > 0:
        checks["query_galaxies"] = actuals["query_galaxies"] >= thresholds["query_galaxies_min"]
    if min_cross_rate > 0.0:
        checks["cross_galaxy_navigation_rate"] = (
            actuals["cross_galaxy_navigation_rate"] >= thresholds["cross_galaxy_navigation_rate_min"]
        )
    reasons: list[str] = []
    if not checks["accuracy"]:
        reasons.append(
            f"accuracy {actuals['accuracy']:.3f} < {thresholds['accuracy_min']:.3f}"
        )
    if not checks["oracle_at_all"]:
        reasons.append(
            f"oracle_at_all {actuals['oracle_at_all']:.3f} < {thresholds['oracle_at_all_min']:.3f}"
        )
    if not checks["fuzzy_oracle_at_all"]:
        reasons.append(
            f"fuzzy_oracle_at_all {actuals['fuzzy_oracle_at_all']:.3f} < {thresholds['fuzzy_oracle_at_all_min']:.3f}"
        )
    if not checks["generation_failure_rate"]:
        reasons.append(
            "generation_failure_rate "
            f"{actuals['generation_failure_rate']:.3f} > {thresholds['generation_failure_rate_max']:.3f}"
        )
    if min_query_galaxies > 0 and not checks.get("query_galaxies", True):
        reasons.append(
            "query_galaxies "
            f"{actuals['query_galaxies']:.0f} < {thresholds['query_galaxies_min']:.0f}"
        )
    if min_cross_rate > 0.0 and not checks.get("cross_galaxy_navigation_rate", True):
        reasons.append(
            "cross_galaxy_navigation_rate "
            f"{actuals['cross_galaxy_navigation_rate']:.3f} < {thresholds['cross_galaxy_navigation_rate_min']:.3f}"
        )
    passed = all(checks.values())
    return {
        "passed": passed,
        "thresholds": thresholds,
        "actuals": actuals,
        "checks": checks,
        "reason": "Stage promotion approved" if passed else "; ".join(reasons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-dataset-path", default=None, help="ARC dataset directory")
    parser.add_argument(
        "--arc-dataset-version",
        default="arc_agi_2",
        choices=["arc_agi_2", "arc_agi_3"],
        help="ARC benchmark dataset version",
    )
    parser.add_argument("--math-dataset-path", default=None, help="Math dataset directory")
    parser.add_argument("--lhe-dataset-path", default=None, help="LHE dataset directory")
    parser.add_argument("--mmlu-dataset-path", default=None, help="MMLU dataset directory")
    parser.add_argument("--max-arc-tasks", type=int, default=100, help="ARC task limit")
    parser.add_argument(
        "--arc-query-scope-galaxies",
        default="Drawing,Grammar,3DObjects,Math,Reality",
        help="Comma-separated galaxy scope for ARC queries; empty disables scoping.",
    )
    parser.add_argument(
        "--arc-enable-contrastive-learning",
        action="store_true",
        help="Enable contrastive forward/backward/fusion ARC pattern discovery",
    )
    parser.add_argument(
        "--arc-enable-validity-gates",
        action="store_true",
        help="Enable train-derived validity filtering before ARC candidate selection",
    )
    parser.add_argument(
        "--arc-enable-fuzzy-oracle",
        action="store_true",
        help="Enable fuzzy oracle diagnostics for ARC near-miss capture",
    )
    parser.add_argument(
        "--arc-enable-ptx-ranking",
        action="store_true",
        help="Enable PTX/GPU-backed ARC candidate scoring and top-1 selection.",
    )
    parser.add_argument(
        "--arc-enable-full-ptx",
        action="store_true",
        help="Enable full PTX path for ARC discovery/validity/oracle (not only ranking).",
    )
    parser.add_argument(
        "--arc-ptx-validity-strictness",
        default="medium",
        choices=["strict", "medium", "relaxed"],
        help="Strictness profile for PTX validity gates when full PTX is enabled.",
    )
    parser.add_argument(
        "--arc-constraint-mode",
        default="reject",
        choices=["reject", "penalty"],
        help="ARC constraint handling mode: hard reject or score penalty.",
    )
    parser.add_argument(
        "--arc-enable-figure-ground-reversal",
        action="store_true",
        help="Augment ARC discovery with positive/negative (figure-ground) train-pair reversals.",
    )
    parser.add_argument(
        "--arc-enable-negative-forms",
        action="store_true",
        help="Alias for --arc-enable-figure-ground-reversal (positive/negative form duality).",
    )
    parser.add_argument(
        "--arc-enable-object-aware-generation",
        action="store_true",
        help="Enable object-count-aware ARC candidate generation variants before ranking.",
    )
    parser.add_argument(
        "--arc-enable-forced-navigation-curriculum",
        action="store_true",
        help=(
            "Inject curriculum patterns that explicitly query underused galaxies "
            "(e.g., Math/Reality) before ARC ranking."
        ),
    )
    parser.add_argument(
        "--arc-forced-navigation-ratio",
        type=float,
        default=0.0,
        help="Fraction of discovery pattern budget reserved for forced navigation injection (0.0-1.0).",
    )
    parser.add_argument(
        "--arc-forced-navigation-required-galaxies",
        default="Math,Reality",
        help="Comma-separated galaxies to force into ARC curriculum navigation injection.",
    )
    parser.add_argument(
        "--arc-curriculum-stage",
        default="none",
        choices=["none", "week22_1a", "week22_1b", "week22_1c"],
        help="Apply stage defaults for Week 22.1 ARC navigation curriculum.",
    )
    parser.add_argument(
        "--arc-enable-rescue-lane",
        action="store_true",
        help="Enable top-k rescue lane selection (exact-first, fuzzy fallback).",
    )
    parser.add_argument(
        "--arc-rescue-lane-size",
        type=int,
        default=16,
        help="Top-k size for rescue lane selection when enabled.",
    )
    parser.add_argument(
        "--arc-oracle-search-lane-size",
        type=int,
        default=32,
        help="Top-k oracle search lane size used for diagnostics/learning signal (separate from prediction lane).",
    )
    parser.add_argument(
        "--arc-enable-oracle-rejected-rescue",
        action="store_true",
        help=(
            "Enable oracle-only rejected-candidate rescue lane (bounded top-k from rejected pool). "
            "This does not affect top-1 prediction selection."
        ),
    )
    parser.add_argument(
        "--arc-oracle-rejected-rescue-size",
        type=int,
        default=16,
        help="Top-k rejected candidates to include in oracle-only rescue diagnostics.",
    )
    parser.add_argument(
        "--arc-enable-dual-track-oracle",
        action="store_true",
        help="Use dual-track oracle weighting (exact=full reinforcement, fuzzy=partial).",
    )
    parser.add_argument(
        "--arc-family-penalty-weight",
        type=float,
        default=1.0,
        help="Exponent weight for family consistency in ARC penalty scoring.",
    )
    parser.add_argument(
        "--arc-shape-penalty-weight",
        type=float,
        default=1.0,
        help="Exponent weight for shape consistency in ARC penalty scoring.",
    )
    parser.add_argument(
        "--arc-palette-penalty-weight",
        type=float,
        default=2.0,
        help="Exponent weight for palette consistency in ARC penalty scoring.",
    )
    parser.add_argument(
        "--arc-object-penalty-weight",
        type=float,
        default=1.0,
        help="Exponent weight for object-count consistency in ARC penalty scoring.",
    )
    parser.add_argument(
        "--arc-enable-adaptive-penalties",
        action="store_true",
        help="Auto-tune ARC penalty weights from recent oracle failure distributions in benchmark history.",
    )
    parser.add_argument(
        "--arc-adaptive-penalty-lookback",
        type=int,
        default=3,
        help="Number of recent benchmark summaries used to compute adaptive penalties.",
    )
    parser.add_argument(
        "--arc-adaptive-penalty-target-sum",
        type=float,
        default=5.0,
        help="Target sum for adaptive ARC penalty weights.",
    )
    parser.add_argument(
        "--arc-embedding-lazy-mode",
        default="skip",
        choices=["compute", "skip", "fail"],
        help=(
            "Policy for missing ARC embeddings in hot path: "
            "'skip' (default, no lazy compute), 'fail' (strict fail-fast), 'compute' (legacy)."
        ),
    )
    parser.add_argument(
        "--arc-fuzzy-oracle-threshold",
        type=float,
        default=0.95,
        help="Fuzzy match threshold in [0.5, 0.99] for ARC oracle diagnostics",
    )
    parser.add_argument("--max-math-problems", type=int, default=100, help="Math problem limit")
    parser.add_argument(
        "--math-query-scope-galaxies",
        default="Math,Grammar,Reality",
        help="Comma-separated galaxy scope for Math benchmark queries; empty disables scoping.",
    )
    parser.add_argument("--max-lhe-questions", type=int, default=100, help="LHE question limit")
    parser.add_argument(
        "--lhe-query-scope-galaxies",
        default="Grammar,Word,Math,Reality,Drawing",
        help="Comma-separated galaxy scope for LHE benchmark queries; empty disables scoping.",
    )
    parser.add_argument(
        "--lhe-min-questions",
        type=int,
        default=1,
        help=(
            "Minimum acceptable evaluated LHE questions for integrity checks. "
            "Use higher values (e.g. 100/1000) for paper-grade validation."
        ),
    )
    parser.add_argument(
        "--lhe-require-real-dataset",
        action="store_true",
        help=(
            "Fail-fast if LHE benchmark falls back to synthetic questions or evaluates "
            "fewer than --lhe-min-questions."
        ),
    )
    parser.add_argument("--max-mmlu-questions", type=int, default=1000, help="MMLU question limit")
    parser.add_argument(
        "--mmlu-query-scope-galaxies",
        default="Grammar,Word,Math,Reality",
        help="Comma-separated galaxy scope for MMLU benchmark queries; empty disables scoping.",
    )
    parser.add_argument(
        "--mmlu-subjects",
        type=str,
        default="all",
        help=(
            "MMLU subjects to test (comma-separated) or 'all' for all 57 subjects. "
            "Examples: 'all', 'abstract_algebra,college_mathematics,college_physics'"
        ),
    )
    parser.add_argument(
        "--mmlu-min-questions",
        type=int,
        default=100,
        help=(
            "Minimum acceptable evaluated MMLU questions for integrity checks. "
            "Use higher values (e.g. 1000+) for paper-grade validation."
        ),
    )
    parser.add_argument(
        "--mmlu-require-real-dataset",
        action="store_true",
        help=(
            "Fail-fast if MMLU benchmark falls back to synthetic questions or evaluates "
            "fewer than --mmlu-min-questions."
        ),
    )
    parser.add_argument(
        "--benchmark-runtime-seeding",
        action="store_true",
        help="Allow benchmark classes to inject seed entries during task loops (disabled by default).",
    )
    parser.add_argument(
        "--track-curriculum-coverage",
        action="store_true",
        help="Emit curriculum coverage telemetry (galaxies touched, route depth, ternary deltas).",
    )
    parser.add_argument(
        "--require-min-galaxies-per-block",
        type=int,
        default=0,
        help=(
            "Soft gate target for minimum unique galaxies touched per evaluation block. "
            "Currently logs pass/fail in summary without aborting."
        ),
    )
    parser.add_argument(
        "--require-min-cross-galaxy-navigation-rate",
        type=float,
        default=0.0,
        help=(
            "Soft gate target for minimum ARC cross-galaxy navigation rate per block "
            "(0.0-1.0). Logs pass/fail in summary."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="/K3D/Knowledge3D.local/results/week14",
        help="Directory for benchmark outputs",
    )
    parser.add_argument(
        "--storage-root",
        default="/K3D/Knowledge3D.local",
        help="Base storage root used to derive benchmark roots",
    )
    parser.add_argument(
        "--model-persistence-mode",
        default="unified",
        choices=["unified", "dual"],
        help=(
            "Persistence strategy. "
            "'unified' reuses one evolving Knowledgeverse instance across the full run; "
            "'dual' keeps separate empty/enriched worlds for diagnostic comparisons."
        ),
    )
    parser.add_argument(
        "--enforce-sovereignty",
        dest="enforce_sovereignty",
        action="store_true",
        default=True,
        help=(
            "Fail-fast when solved tasks do not include sovereign GPU evidence "
            "(gpu_calls > 0 and no fallback)."
        ),
    )
    parser.add_argument(
        "--no-enforce-sovereignty",
        dest="enforce_sovereignty",
        action="store_false",
        help="Disable sovereignty fail-fast checks for debugging only.",
    )
    parser.add_argument(
        "--empty-storage-root",
        default=None,
        help="Optional explicit storage root for empty-mind runs",
    )
    parser.add_argument(
        "--enriched-storage-root",
        default=None,
        help="Optional explicit storage root for enriched runs",
    )
    parser.add_argument(
        "--unified-storage-root",
        default=None,
        help="Optional explicit storage root for unified mode (defaults to enriched root).",
    )
    args = parser.parse_args()
    arc_enable_negative_forms = bool(
        args.arc_enable_figure_ground_reversal or args.arc_enable_negative_forms
    )
    stage_defaults = {
        "week22_1a": {
            "forced_navigation_ratio": 0.20,
            "required_galaxies": "Math",
            "min_galaxies": 4,
            "min_cross_rate": 0.40,
            "oracle_lane_size": 64,
        },
        "week22_1b": {
            "forced_navigation_ratio": 0.33,
            "required_galaxies": "Math,Reality",
            "min_galaxies": 5,
            "min_cross_rate": 0.50,
            "oracle_lane_size": 48,
        },
        "week22_1c": {
            "forced_navigation_ratio": 0.33,
            "required_galaxies": "Math,Reality,3DObjects",
            "min_galaxies": 5,
            "min_cross_rate": 0.60,
            "oracle_lane_size": 32,
        },
    }
    if args.arc_curriculum_stage in stage_defaults:
        defaults = stage_defaults[args.arc_curriculum_stage]
        args.arc_enable_forced_navigation_curriculum = True
        if float(args.arc_forced_navigation_ratio) <= 0.0:
            args.arc_forced_navigation_ratio = float(defaults["forced_navigation_ratio"])
        if str(args.arc_forced_navigation_required_galaxies).strip() in {"", "Math,Reality"}:
            args.arc_forced_navigation_required_galaxies = str(defaults["required_galaxies"])
        if int(args.require_min_galaxies_per_block) <= 0:
            args.require_min_galaxies_per_block = int(defaults["min_galaxies"])
        if float(args.require_min_cross_galaxy_navigation_rate) <= 0.0:
            args.require_min_cross_galaxy_navigation_rate = float(defaults["min_cross_rate"])
        if int(args.arc_oracle_search_lane_size) == 32:
            args.arc_oracle_search_lane_size = int(defaults["oracle_lane_size"])
    if args.arc_enable_full_ptx and not args.arc_enable_ptx_ranking:
        # Full PTX mode implies PTX ranking. Keep CLI ergonomic and avoid
        # accidental CPU fallback when users only pass --arc-enable-full-ptx.
        args.arc_enable_ptx_ranking = True
    os.environ["K3D_ARC_EMBEDDING_LAZY_MODE"] = str(args.arc_embedding_lazy_mode)
    # Enforce PTX-only ARC path by default for Week 21+ architecture.
    os.environ.setdefault("K3D_REQUIRE_PTX_ARC_PIPELINE", "true")
    if args.arc_enable_full_ptx:
        os.environ.setdefault("K3D_ALLOW_LEGACY_ARC_PIPELINE", "false")

    run_arc = _is_enabled_limit(args.max_arc_tasks)
    run_math = _is_enabled_limit(args.max_math_problems)
    run_lhe = _is_enabled_limit(args.max_lhe_questions)
    run_mmlu = _is_enabled_limit(args.max_mmlu_questions)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_storage_root = Path(args.storage_root)
    empty_storage_root = (
        Path(args.empty_storage_root) if args.empty_storage_root else (base_storage_root / "galaxies_empty_mind")
    )
    enriched_storage_root = (
        Path(args.enriched_storage_root) if args.enriched_storage_root else (base_storage_root / "galaxies_enriched")
    )
    unified_storage_root = (
        Path(args.unified_storage_root) if args.unified_storage_root else enriched_storage_root
    )
    empty_storage_root.mkdir(parents=True, exist_ok=True)
    enriched_storage_root.mkdir(parents=True, exist_ok=True)
    unified_storage_root.mkdir(parents=True, exist_ok=True)
    usage_log_path = base_storage_root / "logs" / "benchmark_usage_metrics.jsonl"
    history_path = base_storage_root / "benchmarks" / "run_all_benchmarks_history.jsonl"
    adaptive_penalties: dict[str, Any] = {
        "enabled": bool(args.arc_enable_adaptive_penalties),
        "applied": False,
        "lookback": int(args.arc_adaptive_penalty_lookback),
        "target_sum": float(args.arc_adaptive_penalty_target_sum),
        "weights": None,
    }
    if args.arc_enable_adaptive_penalties:
        computed = _compute_adaptive_penalties_from_history(
            history_path,
            lookback=int(args.arc_adaptive_penalty_lookback),
            target_sum=float(args.arc_adaptive_penalty_target_sum),
        )
        if computed:
            args.arc_family_penalty_weight = float(computed["family_penalty_weight"])
            args.arc_shape_penalty_weight = float(computed["shape_penalty_weight"])
            args.arc_palette_penalty_weight = float(computed["palette_penalty_weight"])
            args.arc_object_penalty_weight = float(computed["object_penalty_weight"])
            adaptive_penalties["weights"] = computed
            adaptive_penalties["applied"] = True
    persistence_mode = str(args.model_persistence_mode)
    continuity: dict[str, Any] = {
        "mode": persistence_mode,
        "shared_instance": False,
        "instance_ids": {},
    }

    if persistence_mode == "unified":
        shared_kv = Knowledgeverse(storage_root=unified_storage_root)
        shared_tablet = _build_tablet_boundary(shared_kv)
        continuity["shared_instance"] = True
        continuity["instance_ids"] = {
            "empty_mind": int(id(shared_kv)),
            "enriched": int(id(shared_kv)),
        }

        shared_galaxy_counts_start = _collect_default_galaxy_counts(shared_kv)
        if run_arc:
            arc_empty, arc_empty_metrics = _run_with_metrics(
                "arc_empty_mind",
                lambda: ARCAGI2Benchmark(
                    knowledgeverse=shared_kv,
                    dataset_path=args.arc_dataset_path,
                    max_tasks=args.max_arc_tasks,
                    dataset_version=args.arc_dataset_version,
                    enable_contrastive_learning=args.arc_enable_contrastive_learning,
                    enable_validity_gates=args.arc_enable_validity_gates,
                    enable_fuzzy_oracle=args.arc_enable_fuzzy_oracle,
                    fuzzy_oracle_threshold=args.arc_fuzzy_oracle_threshold,
                    enable_ptx_ranking=args.arc_enable_ptx_ranking,
                    enable_full_ptx=args.arc_enable_full_ptx,
                    ptx_validity_strictness=args.arc_ptx_validity_strictness,
                    constraint_mode=args.arc_constraint_mode,
                    enable_figure_ground_reversal=arc_enable_negative_forms,
                    enable_object_aware_generation=args.arc_enable_object_aware_generation,
                    enable_forced_navigation_curriculum=args.arc_enable_forced_navigation_curriculum,
                    forced_navigation_ratio=args.arc_forced_navigation_ratio,
                    forced_navigation_required_galaxies=args.arc_forced_navigation_required_galaxies,
                    enable_rescue_lane=args.arc_enable_rescue_lane,
                    rescue_lane_size=args.arc_rescue_lane_size,
                    oracle_search_lane_size=args.arc_oracle_search_lane_size,
                    enable_oracle_rejected_rescue=args.arc_enable_oracle_rejected_rescue,
                    oracle_rejected_rescue_size=args.arc_oracle_rejected_rescue_size,
                    enable_dual_track_oracle=args.arc_enable_dual_track_oracle,
                    family_penalty_weight=args.arc_family_penalty_weight,
                    shape_penalty_weight=args.arc_shape_penalty_weight,
                    palette_penalty_weight=args.arc_palette_penalty_weight,
                    object_penalty_weight=args.arc_object_penalty_weight,
                    query_scope_galaxies=args.arc_query_scope_galaxies,
                    runtime_seed_knowledge=args.benchmark_runtime_seeding,
                    tablet_boundary=shared_tablet,
                ).run_benchmark(use_enriched=False),
            )
        else:
            arc_empty = _arc_skipped_result(
                dataset_path=args.arc_dataset_path,
                dataset_version=args.arc_dataset_version,
                use_enriched=False,
            )
            arc_empty_metrics = _skip_metrics("arc_empty_mind")

        if run_math:
            math_empty, math_empty_metrics = _run_with_metrics(
                "math_empty_mind",
                lambda: MathCompetitionBenchmark(
                    knowledgeverse=shared_kv,
                    dataset_path=args.math_dataset_path,
                    max_problems=args.max_math_problems,
                    query_scope_galaxies=args.math_query_scope_galaxies,
                    runtime_seed_knowledge=args.benchmark_runtime_seeding,
                    tablet_boundary=shared_tablet,
                ).run_benchmark(use_enriched=False),
            )
        else:
            math_empty = _math_skipped_result(dataset_path=args.math_dataset_path, use_enriched=False)
            math_empty_metrics = _skip_metrics("math_empty_mind")

        if run_lhe:
            lhe_empty, lhe_empty_metrics = _run_with_metrics(
                "lhe_empty_mind",
                lambda: LastHumanityExamBenchmark(
                    knowledgeverse=shared_kv,
                    dataset_path=args.lhe_dataset_path,
                    max_questions=args.max_lhe_questions,
                    query_scope_galaxies=args.lhe_query_scope_galaxies,
                    runtime_seed_knowledge=args.benchmark_runtime_seeding,
                    tablet_boundary=shared_tablet,
                ).run_benchmark(use_enriched=False),
            )
        else:
            lhe_empty = _lhe_skipped_result(dataset_path=args.lhe_dataset_path, use_enriched=False)
            lhe_empty_metrics = _skip_metrics("lhe_empty_mind")

        if run_mmlu:
            mmlu_empty, mmlu_empty_metrics = _run_with_metrics(
                "mmlu_empty_mind",
                lambda: MMLUBenchmark(
                    knowledgeverse=shared_kv,
                    dataset_path=args.mmlu_dataset_path,
                    max_questions=args.max_mmlu_questions,
                    query_scope_galaxies=args.mmlu_query_scope_galaxies,
                    subjects=args.mmlu_subjects,
                    runtime_seed_knowledge=args.benchmark_runtime_seeding,
                    tablet_boundary=shared_tablet,
                ).run_benchmark(use_enriched=False),
            )
        else:
            mmlu_empty = _mmlu_skipped_result(dataset_path=args.mmlu_dataset_path, use_enriched=False)
            mmlu_empty_metrics = _skip_metrics("mmlu_empty_mind")
        shared_after_empty = _collect_default_galaxy_counts(shared_kv)

        if run_arc:
            arc_enriched, arc_enriched_metrics = _run_with_metrics(
                "arc_enriched",
                lambda: ARCAGI2Benchmark(
                    knowledgeverse=shared_kv,
                    dataset_path=args.arc_dataset_path,
                    max_tasks=args.max_arc_tasks,
                    dataset_version=args.arc_dataset_version,
                    enable_contrastive_learning=args.arc_enable_contrastive_learning,
                    enable_validity_gates=args.arc_enable_validity_gates,
                    enable_fuzzy_oracle=args.arc_enable_fuzzy_oracle,
                    fuzzy_oracle_threshold=args.arc_fuzzy_oracle_threshold,
                    enable_ptx_ranking=args.arc_enable_ptx_ranking,
                    enable_full_ptx=args.arc_enable_full_ptx,
                    ptx_validity_strictness=args.arc_ptx_validity_strictness,
                    constraint_mode=args.arc_constraint_mode,
                    enable_figure_ground_reversal=arc_enable_negative_forms,
                    enable_object_aware_generation=args.arc_enable_object_aware_generation,
                    enable_forced_navigation_curriculum=args.arc_enable_forced_navigation_curriculum,
                    forced_navigation_ratio=args.arc_forced_navigation_ratio,
                    forced_navigation_required_galaxies=args.arc_forced_navigation_required_galaxies,
                    enable_rescue_lane=args.arc_enable_rescue_lane,
                    rescue_lane_size=args.arc_rescue_lane_size,
                    oracle_search_lane_size=args.arc_oracle_search_lane_size,
                    enable_oracle_rejected_rescue=args.arc_enable_oracle_rejected_rescue,
                    oracle_rejected_rescue_size=args.arc_oracle_rejected_rescue_size,
                    enable_dual_track_oracle=args.arc_enable_dual_track_oracle,
                    family_penalty_weight=args.arc_family_penalty_weight,
                    shape_penalty_weight=args.arc_shape_penalty_weight,
                    palette_penalty_weight=args.arc_palette_penalty_weight,
                    object_penalty_weight=args.arc_object_penalty_weight,
                    query_scope_galaxies=args.arc_query_scope_galaxies,
                    runtime_seed_knowledge=args.benchmark_runtime_seeding,
                    tablet_boundary=shared_tablet,
                ).run_benchmark(use_enriched=True),
            )
        else:
            arc_enriched = _arc_skipped_result(
                dataset_path=args.arc_dataset_path,
                dataset_version=args.arc_dataset_version,
                use_enriched=True,
            )
            arc_enriched_metrics = _skip_metrics("arc_enriched")

        if run_math:
            math_enriched, math_enriched_metrics = _run_with_metrics(
                "math_enriched",
                lambda: MathCompetitionBenchmark(
                    knowledgeverse=shared_kv,
                    dataset_path=args.math_dataset_path,
                    max_problems=args.max_math_problems,
                    query_scope_galaxies=args.math_query_scope_galaxies,
                    runtime_seed_knowledge=args.benchmark_runtime_seeding,
                    tablet_boundary=shared_tablet,
                ).run_benchmark(use_enriched=True),
            )
        else:
            math_enriched = _math_skipped_result(dataset_path=args.math_dataset_path, use_enriched=True)
            math_enriched_metrics = _skip_metrics("math_enriched")

        if run_lhe:
            lhe_enriched, lhe_enriched_metrics = _run_with_metrics(
                "lhe_enriched",
                lambda: LastHumanityExamBenchmark(
                    knowledgeverse=shared_kv,
                    dataset_path=args.lhe_dataset_path,
                    max_questions=args.max_lhe_questions,
                    query_scope_galaxies=args.lhe_query_scope_galaxies,
                    runtime_seed_knowledge=args.benchmark_runtime_seeding,
                    tablet_boundary=shared_tablet,
                ).run_benchmark(use_enriched=True),
            )
        else:
            lhe_enriched = _lhe_skipped_result(dataset_path=args.lhe_dataset_path, use_enriched=True)
            lhe_enriched_metrics = _skip_metrics("lhe_enriched")

        if run_mmlu:
            mmlu_enriched, mmlu_enriched_metrics = _run_with_metrics(
                "mmlu_enriched",
                lambda: MMLUBenchmark(
                    knowledgeverse=shared_kv,
                    dataset_path=args.mmlu_dataset_path,
                    max_questions=args.max_mmlu_questions,
                    query_scope_galaxies=args.mmlu_query_scope_galaxies,
                    subjects=args.mmlu_subjects,
                    runtime_seed_knowledge=args.benchmark_runtime_seeding,
                    tablet_boundary=shared_tablet,
                ).run_benchmark(use_enriched=True),
            )
        else:
            mmlu_enriched = _mmlu_skipped_result(dataset_path=args.mmlu_dataset_path, use_enriched=True)
            mmlu_enriched_metrics = _skip_metrics("mmlu_enriched")
        shared_after_enriched = _collect_default_galaxy_counts(shared_kv)

        empty_kv = shared_kv
        enriched_kv = shared_kv
        empty_storage_root = unified_storage_root
        enriched_storage_root = unified_storage_root
        empty_galaxy_counts_start = shared_galaxy_counts_start
        empty_galaxy_counts_end = shared_after_empty
        enriched_galaxy_counts_start = shared_after_empty
        enriched_galaxy_counts_end = shared_after_enriched
    else:
        empty_kv = Knowledgeverse(storage_root=empty_storage_root)
        empty_tablet = _build_tablet_boundary(empty_kv)
        continuity["instance_ids"]["empty_mind"] = int(id(empty_kv))
        empty_galaxy_counts_start = _collect_default_galaxy_counts(empty_kv)
        if run_arc:
            arc_empty, arc_empty_metrics = _run_with_metrics(
                "arc_empty_mind",
                lambda: ARCAGI2Benchmark(
                    knowledgeverse=empty_kv,
                    dataset_path=args.arc_dataset_path,
                    max_tasks=args.max_arc_tasks,
                    dataset_version=args.arc_dataset_version,
                    enable_contrastive_learning=args.arc_enable_contrastive_learning,
                    enable_validity_gates=args.arc_enable_validity_gates,
                    enable_fuzzy_oracle=args.arc_enable_fuzzy_oracle,
                    fuzzy_oracle_threshold=args.arc_fuzzy_oracle_threshold,
                    enable_ptx_ranking=args.arc_enable_ptx_ranking,
                    enable_full_ptx=args.arc_enable_full_ptx,
                    ptx_validity_strictness=args.arc_ptx_validity_strictness,
                    constraint_mode=args.arc_constraint_mode,
                    enable_figure_ground_reversal=arc_enable_negative_forms,
                    enable_object_aware_generation=args.arc_enable_object_aware_generation,
                    enable_forced_navigation_curriculum=args.arc_enable_forced_navigation_curriculum,
                    forced_navigation_ratio=args.arc_forced_navigation_ratio,
                    forced_navigation_required_galaxies=args.arc_forced_navigation_required_galaxies,
                    enable_rescue_lane=args.arc_enable_rescue_lane,
                    rescue_lane_size=args.arc_rescue_lane_size,
                    oracle_search_lane_size=args.arc_oracle_search_lane_size,
                    enable_oracle_rejected_rescue=args.arc_enable_oracle_rejected_rescue,
                    oracle_rejected_rescue_size=args.arc_oracle_rejected_rescue_size,
                    enable_dual_track_oracle=args.arc_enable_dual_track_oracle,
                    family_penalty_weight=args.arc_family_penalty_weight,
                    shape_penalty_weight=args.arc_shape_penalty_weight,
                    palette_penalty_weight=args.arc_palette_penalty_weight,
                    object_penalty_weight=args.arc_object_penalty_weight,
                    query_scope_galaxies=args.arc_query_scope_galaxies,
                    runtime_seed_knowledge=args.benchmark_runtime_seeding,
                    tablet_boundary=empty_tablet,
                ).run_benchmark(use_enriched=False),
            )
        else:
            arc_empty = _arc_skipped_result(
                dataset_path=args.arc_dataset_path,
                dataset_version=args.arc_dataset_version,
                use_enriched=False,
            )
            arc_empty_metrics = _skip_metrics("arc_empty_mind")
        if run_math:
            math_empty, math_empty_metrics = _run_with_metrics(
                "math_empty_mind",
                lambda: MathCompetitionBenchmark(
                    knowledgeverse=empty_kv,
                    dataset_path=args.math_dataset_path,
                    max_problems=args.max_math_problems,
                    query_scope_galaxies=args.math_query_scope_galaxies,
                    runtime_seed_knowledge=args.benchmark_runtime_seeding,
                    tablet_boundary=empty_tablet,
                ).run_benchmark(use_enriched=False),
            )
        else:
            math_empty = _math_skipped_result(dataset_path=args.math_dataset_path, use_enriched=False)
            math_empty_metrics = _skip_metrics("math_empty_mind")
        if run_lhe:
            lhe_empty, lhe_empty_metrics = _run_with_metrics(
                "lhe_empty_mind",
                lambda: LastHumanityExamBenchmark(
                    knowledgeverse=empty_kv,
                    dataset_path=args.lhe_dataset_path,
                    max_questions=args.max_lhe_questions,
                    query_scope_galaxies=args.lhe_query_scope_galaxies,
                    runtime_seed_knowledge=args.benchmark_runtime_seeding,
                    tablet_boundary=empty_tablet,
                ).run_benchmark(use_enriched=False),
            )
        else:
            lhe_empty = _lhe_skipped_result(dataset_path=args.lhe_dataset_path, use_enriched=False)
            lhe_empty_metrics = _skip_metrics("lhe_empty_mind")
        empty_galaxy_counts_end = _collect_default_galaxy_counts(empty_kv)

        enriched_kv = Knowledgeverse(storage_root=enriched_storage_root)
        enriched_tablet = _build_tablet_boundary(enriched_kv)
        continuity["instance_ids"]["enriched"] = int(id(enriched_kv))
        continuity["shared_instance"] = continuity["instance_ids"]["empty_mind"] == continuity["instance_ids"]["enriched"]
        enriched_galaxy_counts_start = _collect_default_galaxy_counts(enriched_kv)
        if run_arc:
            arc_enriched, arc_enriched_metrics = _run_with_metrics(
                "arc_enriched",
                lambda: ARCAGI2Benchmark(
                    knowledgeverse=enriched_kv,
                    dataset_path=args.arc_dataset_path,
                    max_tasks=args.max_arc_tasks,
                    dataset_version=args.arc_dataset_version,
                    enable_contrastive_learning=args.arc_enable_contrastive_learning,
                    enable_validity_gates=args.arc_enable_validity_gates,
                    enable_fuzzy_oracle=args.arc_enable_fuzzy_oracle,
                    fuzzy_oracle_threshold=args.arc_fuzzy_oracle_threshold,
                    enable_ptx_ranking=args.arc_enable_ptx_ranking,
                    enable_full_ptx=args.arc_enable_full_ptx,
                    ptx_validity_strictness=args.arc_ptx_validity_strictness,
                    constraint_mode=args.arc_constraint_mode,
                    enable_figure_ground_reversal=arc_enable_negative_forms,
                    enable_object_aware_generation=args.arc_enable_object_aware_generation,
                    enable_forced_navigation_curriculum=args.arc_enable_forced_navigation_curriculum,
                    forced_navigation_ratio=args.arc_forced_navigation_ratio,
                    forced_navigation_required_galaxies=args.arc_forced_navigation_required_galaxies,
                    enable_rescue_lane=args.arc_enable_rescue_lane,
                    rescue_lane_size=args.arc_rescue_lane_size,
                    oracle_search_lane_size=args.arc_oracle_search_lane_size,
                    enable_oracle_rejected_rescue=args.arc_enable_oracle_rejected_rescue,
                    oracle_rejected_rescue_size=args.arc_oracle_rejected_rescue_size,
                    enable_dual_track_oracle=args.arc_enable_dual_track_oracle,
                    family_penalty_weight=args.arc_family_penalty_weight,
                    shape_penalty_weight=args.arc_shape_penalty_weight,
                    palette_penalty_weight=args.arc_palette_penalty_weight,
                    object_penalty_weight=args.arc_object_penalty_weight,
                    query_scope_galaxies=args.arc_query_scope_galaxies,
                    runtime_seed_knowledge=args.benchmark_runtime_seeding,
                    tablet_boundary=enriched_tablet,
                ).run_benchmark(use_enriched=True),
            )
        else:
            arc_enriched = _arc_skipped_result(
                dataset_path=args.arc_dataset_path,
                dataset_version=args.arc_dataset_version,
                use_enriched=True,
            )
            arc_enriched_metrics = _skip_metrics("arc_enriched")
        if run_math:
            math_enriched, math_enriched_metrics = _run_with_metrics(
                "math_enriched",
                lambda: MathCompetitionBenchmark(
                    knowledgeverse=enriched_kv,
                    dataset_path=args.math_dataset_path,
                    max_problems=args.max_math_problems,
                    query_scope_galaxies=args.math_query_scope_galaxies,
                    runtime_seed_knowledge=args.benchmark_runtime_seeding,
                    tablet_boundary=enriched_tablet,
                ).run_benchmark(use_enriched=True),
            )
        else:
            math_enriched = _math_skipped_result(dataset_path=args.math_dataset_path, use_enriched=True)
            math_enriched_metrics = _skip_metrics("math_enriched")
        if run_lhe:
            lhe_enriched, lhe_enriched_metrics = _run_with_metrics(
                "lhe_enriched",
                lambda: LastHumanityExamBenchmark(
                    knowledgeverse=enriched_kv,
                    dataset_path=args.lhe_dataset_path,
                    max_questions=args.max_lhe_questions,
                    query_scope_galaxies=args.lhe_query_scope_galaxies,
                    runtime_seed_knowledge=args.benchmark_runtime_seeding,
                    tablet_boundary=enriched_tablet,
                ).run_benchmark(use_enriched=True),
            )
        else:
            lhe_enriched = _lhe_skipped_result(dataset_path=args.lhe_dataset_path, use_enriched=True)
            lhe_enriched_metrics = _skip_metrics("lhe_enriched")
        if run_mmlu:
            mmlu_empty, mmlu_empty_metrics = _run_with_metrics(
                "mmlu_empty_mind",
                lambda: MMLUBenchmark(
                    knowledgeverse=empty_kv,
                    dataset_path=args.mmlu_dataset_path,
                    max_questions=args.max_mmlu_questions,
                    query_scope_galaxies=args.mmlu_query_scope_galaxies,
                    subjects=args.mmlu_subjects,
                    runtime_seed_knowledge=args.benchmark_runtime_seeding,
                    tablet_boundary=empty_tablet,
                ).run_benchmark(use_enriched=False),
            )
            mmlu_enriched, mmlu_enriched_metrics = _run_with_metrics(
                "mmlu_enriched",
                lambda: MMLUBenchmark(
                    knowledgeverse=enriched_kv,
                    dataset_path=args.mmlu_dataset_path,
                    max_questions=args.max_mmlu_questions,
                    query_scope_galaxies=args.mmlu_query_scope_galaxies,
                    subjects=args.mmlu_subjects,
                    runtime_seed_knowledge=args.benchmark_runtime_seeding,
                    tablet_boundary=enriched_tablet,
                ).run_benchmark(use_enriched=True),
            )
        else:
            mmlu_empty = _mmlu_skipped_result(dataset_path=args.mmlu_dataset_path, use_enriched=False)
            mmlu_empty_metrics = _skip_metrics("mmlu_empty_mind")
            mmlu_enriched = _mmlu_skipped_result(dataset_path=args.mmlu_dataset_path, use_enriched=True)
            mmlu_enriched_metrics = _skip_metrics("mmlu_enriched")
        enriched_galaxy_counts_end = _collect_default_galaxy_counts(enriched_kv)

    if args.arc_enable_full_ptx and run_arc:
        _assert_arc_solver_contract(arc_empty, required_solver="arc_ptx_ops")
        _assert_arc_solver_contract(arc_enriched, required_solver="arc_ptx_ops")

    if run_arc:
        _assert_tablet_boundary_contract("arc_empty", arc_empty)
        _assert_tablet_boundary_contract("arc_enriched", arc_enriched)
    if run_math:
        _assert_tablet_boundary_contract("math_empty", math_empty)
        _assert_tablet_boundary_contract("math_enriched", math_enriched)
    if run_lhe:
        _assert_tablet_boundary_contract("lhe_empty", lhe_empty)
        _assert_tablet_boundary_contract("lhe_enriched", lhe_enriched)
    if run_mmlu:
        _assert_tablet_boundary_contract("mmlu_empty", mmlu_empty)
        _assert_tablet_boundary_contract("mmlu_enriched", mmlu_enriched)

    summary = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "storage_roots": {
            "mode": persistence_mode,
            "empty_mind": str(empty_storage_root),
            "enriched": str(enriched_storage_root),
            "unified": str(unified_storage_root) if persistence_mode == "unified" else None,
        },
        "benchmarks": {
            "arc_agi_2": {
                "empty_mind": arc_empty,
                "enriched": arc_enriched,
                "diagnostics": {
                    "empty_mind": arc_empty.get("oracle_diagnostics", {}),
                    "enriched": arc_enriched.get("oracle_diagnostics", {}),
                },
                "pattern_source_accuracy": {
                    "empty_mind": arc_empty.get("pattern_source_accuracy", {}),
                    "enriched": arc_enriched.get("pattern_source_accuracy", {}),
                },
                "improvement": arc_enriched["accuracy"] - arc_empty["accuracy"],
                "target": 0.55,
            },
            "math_competitions": {
                "empty_mind": math_empty,
                "enriched": math_enriched,
                "improvement": math_enriched["overall_accuracy"] - math_empty["overall_accuracy"],
                "target": 0.30,
            },
            "last_humanity_exam": {
                "empty_mind": lhe_empty,
                "enriched": lhe_enriched,
                "improvement": lhe_enriched["accuracy"] - lhe_empty["accuracy"],
                "target": 0.40,
            },
            "mmlu": {
                "empty_mind": mmlu_empty,
                "enriched": mmlu_enriched,
                "improvement": mmlu_enriched["accuracy"] - mmlu_empty["accuracy"],
                "target": 0.70,
            },
        },
        "runtime_usage": {
            "runtime_seed_knowledge": bool(args.benchmark_runtime_seeding),
            "arc_enable_ptx_ranking": bool(args.arc_enable_ptx_ranking),
            "arc_enable_full_ptx": bool(args.arc_enable_full_ptx),
            "arc_query_scope_galaxies": str(args.arc_query_scope_galaxies),
            "math_query_scope_galaxies": str(args.math_query_scope_galaxies),
            "lhe_query_scope_galaxies": str(args.lhe_query_scope_galaxies),
            "mmlu_query_scope_galaxies": str(args.mmlu_query_scope_galaxies),
            "arc_ptx_validity_strictness": str(args.arc_ptx_validity_strictness),
            "arc_constraint_mode": str(args.arc_constraint_mode),
            "arc_enable_figure_ground_reversal": bool(arc_enable_negative_forms),
            "arc_enable_object_aware_generation": bool(args.arc_enable_object_aware_generation),
            "arc_enable_forced_navigation_curriculum": bool(args.arc_enable_forced_navigation_curriculum),
            "arc_forced_navigation_ratio": float(args.arc_forced_navigation_ratio),
            "arc_forced_navigation_required_galaxies": str(args.arc_forced_navigation_required_galaxies),
            "arc_curriculum_stage": str(args.arc_curriculum_stage),
            "arc_enable_rescue_lane": bool(args.arc_enable_rescue_lane),
            "arc_rescue_lane_size": int(args.arc_rescue_lane_size),
            "arc_oracle_search_lane_size": int(args.arc_oracle_search_lane_size),
            "arc_enable_oracle_rejected_rescue": bool(args.arc_enable_oracle_rejected_rescue),
            "arc_oracle_rejected_rescue_size": int(args.arc_oracle_rejected_rescue_size),
            "arc_enable_dual_track_oracle": bool(args.arc_enable_dual_track_oracle),
            "arc_family_penalty_weight": float(args.arc_family_penalty_weight),
            "arc_shape_penalty_weight": float(args.arc_shape_penalty_weight),
            "arc_palette_penalty_weight": float(args.arc_palette_penalty_weight),
            "arc_object_penalty_weight": float(args.arc_object_penalty_weight),
            "arc_adaptive_penalties": adaptive_penalties,
            "arc_embedding_lazy_mode": str(args.arc_embedding_lazy_mode),
            "persistence": continuity,
            "runs": [
                arc_empty_metrics,
                math_empty_metrics,
                lhe_empty_metrics,
                mmlu_empty_metrics,
                arc_enriched_metrics,
                math_enriched_metrics,
                lhe_enriched_metrics,
                mmlu_enriched_metrics,
            ],
            "galaxy_counts": {
                "empty_mind_start": empty_galaxy_counts_start,
                "empty_mind_end": empty_galaxy_counts_end,
                "enriched_start": enriched_galaxy_counts_start,
                "enriched_end": enriched_galaxy_counts_end,
            },
        },
    }
    sovereignty_checks: dict[str, Any] = {}
    for benchmark_name in ("arc_agi_2", "math_competitions", "last_humanity_exam", "mmlu"):
        benchmark_payload = summary.get("benchmarks", {}).get(benchmark_name, {})
        for phase in ("empty_mind", "enriched"):
            payload = benchmark_payload.get(phase, {})
            if isinstance(payload, dict):
                key = f"{benchmark_name}.{phase}"
                sovereignty_checks[key] = _summarize_benchmark_sovereignty(
                    benchmark_name=benchmark_name,
                    phase=phase,
                    payload=payload,
                )
    solved_total = sum(int(check.get("solved_tasks", 0)) for check in sovereignty_checks.values())
    gpu_total = sum(int(check.get("tasks_using_gpu", 0)) for check in sovereignty_checks.values())
    fallback_total = sum(int(check.get("fallback_triggered_count", 0)) for check in sovereignty_checks.values())
    missing_gpu_total = sum(int(check.get("tasks_without_gpu", 0)) for check in sovereignty_checks.values())
    missing_telemetry_total = sum(
        int(check.get("tasks_without_gpu_telemetry", 0)) for check in sovereignty_checks.values()
    )
    compliance_total = (gpu_total / solved_total) if solved_total else 1.0
    sovereignty_violations = {
        key: value.get("violations", [])
        for key, value in sovereignty_checks.items()
        if value.get("violations")
    }
    summary["runtime_usage"]["sovereignty"] = {
        "enforced": bool(args.enforce_sovereignty),
        "checks": sovereignty_checks,
        "totals": {
            "solved_tasks": int(solved_total),
            "tasks_using_gpu": int(gpu_total),
            "tasks_without_gpu": int(missing_gpu_total),
            "tasks_without_gpu_telemetry": int(missing_telemetry_total),
            "fallback_triggered_count": int(fallback_total),
            "sovereignty_compliance": float(compliance_total),
        },
        "violations": sovereignty_violations,
    }
    if args.enforce_sovereignty and sovereignty_violations:
        samples: list[str] = []
        for key, reasons in sovereignty_violations.items():
            reason_text = ", ".join(str(item) for item in reasons)
            samples.append(f"{key}: {reason_text}")
            if len(samples) >= 5:
                break
        raise SovereigntyViolation(
            "Sovereignty violation detected in benchmark runner. "
            f"compliance={compliance_total:.1%}; samples={'; '.join(samples)}"
        )

    lhe_requested = int(args.max_lhe_questions)
    lhe_skipped = not run_lhe
    lhe_enriched_total = int(lhe_enriched.get("total_questions", 0) or 0)
    lhe_enriched_source = str(lhe_enriched.get("dataset_source") or "")
    lhe_synthetic_fallback = bool(
        lhe_enriched.get("synthetic_fallback")
        or lhe_enriched_source.startswith("synthetic_fallback")
    )
    lhe_min_required = 0 if lhe_skipped else int(max(1, args.lhe_min_questions))
    lhe_integrity = {
        "dataset_path": lhe_enriched.get("dataset_path"),
        "dataset_source": lhe_enriched.get("dataset_source"),
        "dataset_file": lhe_enriched.get("dataset_file"),
        "skipped": bool(lhe_skipped),
        "synthetic_fallback": lhe_synthetic_fallback,
        "evaluated_questions": lhe_enriched_total,
        "requested_max_questions": lhe_requested,
        "min_required_questions": lhe_min_required,
        "question_count_ok": True if lhe_skipped else (lhe_enriched_total >= lhe_min_required),
        "real_dataset_ok": True if lhe_skipped else (not lhe_synthetic_fallback),
    }
    summary.setdefault("integrity", {})["last_humanity_exam"] = lhe_integrity
    summary["runtime_usage"]["lhe_integrity"] = lhe_integrity
    if args.lhe_require_real_dataset and not lhe_skipped and (
        not bool(lhe_integrity["real_dataset_ok"]) or not bool(lhe_integrity["question_count_ok"])
    ):
        raise RuntimeError(
            "LHE integrity check failed: "
            f"synthetic_fallback={lhe_integrity['synthetic_fallback']} "
            f"evaluated_questions={lhe_integrity['evaluated_questions']} "
            f"min_required={lhe_integrity['min_required_questions']} "
            f"dataset_source={lhe_integrity['dataset_source']!r}. "
            "Provide a real dataset and rerun."
        )

    mmlu_requested = int(args.max_mmlu_questions)
    mmlu_skipped = not run_mmlu
    mmlu_enriched_total = int(mmlu_enriched.get("total_questions", 0) or 0)
    mmlu_enriched_source = str(mmlu_enriched.get("dataset_source") or "")
    mmlu_synthetic_fallback = bool(
        mmlu_enriched.get("synthetic_fallback")
        or mmlu_enriched_source.startswith("synthetic_fallback")
    )
    mmlu_min_required = 0 if mmlu_skipped else int(max(1, args.mmlu_min_questions))
    mmlu_integrity = {
        "dataset_path": mmlu_enriched.get("dataset_path"),
        "dataset_source": mmlu_enriched.get("dataset_source"),
        "dataset_file": mmlu_enriched.get("dataset_file"),
        "skipped": bool(mmlu_skipped),
        "synthetic_fallback": mmlu_synthetic_fallback,
        "evaluated_questions": mmlu_enriched_total,
        "requested_max_questions": mmlu_requested,
        "min_required_questions": mmlu_min_required,
        "question_count_ok": True if mmlu_skipped else (mmlu_enriched_total >= mmlu_min_required),
        "real_dataset_ok": True if mmlu_skipped else (not mmlu_synthetic_fallback),
        "subjects_tested": int(mmlu_enriched.get("subjects_tested", 0) or 0),
    }
    summary.setdefault("integrity", {})["mmlu"] = mmlu_integrity
    summary["runtime_usage"]["mmlu_integrity"] = mmlu_integrity
    if args.mmlu_require_real_dataset and not mmlu_skipped and (
        not bool(mmlu_integrity["real_dataset_ok"]) or not bool(mmlu_integrity["question_count_ok"])
    ):
        raise RuntimeError(
            "MMLU integrity check failed: "
            f"synthetic_fallback={mmlu_integrity['synthetic_fallback']} "
            f"evaluated_questions={mmlu_integrity['evaluated_questions']} "
            f"min_required={mmlu_integrity['min_required_questions']} "
            f"dataset_source={mmlu_integrity['dataset_source']!r}. "
            "Provide a real dataset and rerun."
        )

    if args.track_curriculum_coverage:
        def _touched(start: dict[str, int], end: dict[str, int]) -> list[str]:
            keys = sorted(set(start.keys()) | set(end.keys()))
            touched: list[str] = []
            for key in keys:
                if int(start.get(key, 0)) != int(end.get(key, 0)):
                    touched.append(key)
            return touched

        def _query_participation(arc_payload: dict[str, Any]) -> dict[str, Any]:
            rows = arc_payload.get("results", [])
            if not isinstance(rows, list):
                rows = []
            galaxy_task_counts: dict[str, int] = {}
            total_queries = 0
            task_query_counts: list[int] = []
            cross_nav = 0
            for row in rows:
                queried = row.get("queried_galaxies", [])
                if not isinstance(queried, list):
                    queried = []
                cleaned = sorted({str(item) for item in queried if str(item).strip()})
                task_query_counts.append(len(cleaned))
                if len(cleaned) >= 3:
                    cross_nav += 1
                total_queries += len(cleaned)
                for name in cleaned:
                    galaxy_task_counts[name] = int(galaxy_task_counts.get(name, 0)) + 1
            task_count = len(rows)
            return {
                "unique_galaxies": sorted(galaxy_task_counts.keys()),
                "galaxy_task_counts": galaxy_task_counts,
                "avg_queried_galaxies_per_task": (sum(task_query_counts) / task_count) if task_count else 0.0,
                "cross_galaxy_navigation_rate": (cross_nav / task_count) if task_count else 0.0,
                "total_queries": int(total_queries),
                "task_count": int(task_count),
            }

        empty_touched = _touched(empty_galaxy_counts_start, empty_galaxy_counts_end)
        enriched_touched = _touched(enriched_galaxy_counts_start, enriched_galaxy_counts_end)
        union_touched = sorted(set(empty_touched) | set(enriched_touched))
        empty_query = _query_participation(arc_empty)
        enriched_query = _query_participation(arc_enriched)
        query_union = sorted(
            set(empty_query.get("unique_galaxies", [])) | set(enriched_query.get("unique_galaxies", []))
        )

        deltas = {
            bench_name: float(summary["benchmarks"][bench_name]["improvement"])
            for bench_name in BENCHMARK_CONFIG
        }
        ternary_quality_delta = {
            "positive": sum(1 for d in deltas.values() if d > 0.0),
            "neutral": sum(1 for d in deltas.values() if d == 0.0),
            "negative": sum(1 for d in deltas.values() if d < 0.0),
        }
        specialist_routes = {
            bench_name: {
                "specialist": cfg["specialist"],
                "depth": 1,
                "primary_galaxy": cfg["galaxy"],
            }
            for bench_name, cfg in BENCHMARK_CONFIG.items()
        }

        min_required = max(0, int(args.require_min_galaxies_per_block))
        min_cross_rate = max(0.0, min(1.0, float(args.require_min_cross_galaxy_navigation_rate)))
        empty_cross_rate = float(empty_query.get("cross_galaxy_navigation_rate", 0.0))
        enriched_cross_rate = float(enriched_query.get("cross_galaxy_navigation_rate", 0.0))
        gate = {
            "min_required": min_required,
            "min_cross_galaxy_navigation_rate": min_cross_rate,
            "empty_block_count": len(empty_touched),
            "enriched_block_count": len(enriched_touched),
            "empty_block_passed": True,
            "enriched_block_passed": True,
            "empty_query_count": len(empty_query.get("unique_galaxies", [])),
            "enriched_query_count": len(enriched_query.get("unique_galaxies", [])),
            "empty_query_passed": (len(empty_query.get("unique_galaxies", [])) >= min_required) if min_required else True,
            "enriched_query_passed": (len(enriched_query.get("unique_galaxies", [])) >= min_required) if min_required else True,
            "empty_cross_galaxy_navigation_rate": empty_cross_rate,
            "enriched_cross_galaxy_navigation_rate": enriched_cross_rate,
            "empty_cross_rate_passed": (empty_cross_rate >= min_cross_rate) if min_cross_rate > 0 else True,
            "enriched_cross_rate_passed": (enriched_cross_rate >= min_cross_rate) if min_cross_rate > 0 else True,
            "storage_growth_informational_only": True,
            "soft_gate_only": True,
        }

        summary["runtime_usage"]["curriculum_coverage"] = {
            "storage_growth_touched": {
                "empty_block": empty_touched,
                "enriched_block": enriched_touched,
                "union": union_touched,
            },
            "query_participation": {
                "empty_block": empty_query,
                "enriched_block": enriched_query,
                "union": query_union,
            },
            "specialist_routes": specialist_routes,
            "ternary_quality_delta": ternary_quality_delta,
            "coverage_gate": gate,
        }
    summary["runtime_usage"]["arc_stage_gate"] = _compute_arc_stage_gate(summary)
    recent_entries = _load_recent_history_entries(history_path, limit=3)
    trend_points: list[dict[str, Any]] = []
    for entry in recent_entries:
        summary_path_raw = entry.get("summary_path")
        if not summary_path_raw:
            continue
        path = Path(str(summary_path_raw))
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rate = _extract_arc_generation_failure(payload)
        if rate is None:
            continue
        trend_points.append(
            {
                "timestamp": payload.get("timestamp"),
                "generation_failure_rate": float(rate),
            }
        )
    current_failure = _extract_arc_generation_failure(summary)
    if current_failure is not None:
        trend_points.append(
            {
                "timestamp": summary.get("timestamp"),
                "generation_failure_rate": float(current_failure),
            }
        )
    trend_values = [float(point["generation_failure_rate"]) for point in trend_points]
    trend_delta_latest = (
        (trend_values[-1] - trend_values[-2]) if len(trend_values) >= 2 else None
    )
    trend_flat_last3 = (
        (max(trend_values[-3:]) - min(trend_values[-3:]) <= 0.005)
        if len(trend_values) >= 3
        else False
    )
    summary["runtime_usage"]["arc_generation_failure_trend"] = {
        "points": trend_points,
        "delta_latest": trend_delta_latest,
        "flat_last_3_runs": bool(trend_flat_last3),
    }
    previous_entry = _load_previous_history_entry(history_path)
    historical_comparison = _build_historical_comparison(previous_entry, summary)
    summary["historical_comparison"] = historical_comparison
    summary["history"] = {
        "path": str(history_path),
        "previous_timestamp": (previous_entry or {}).get("timestamp"),
    }

    _persist_benchmark_memory(
        knowledgeverse=enriched_kv,
        summary=summary,
        historical_comparison=historical_comparison,
    )

    (output_dir / "arc_agi_2_empty_mind.json").write_text(
        json.dumps(arc_empty, indent=2),
        encoding="utf-8",
    )
    (output_dir / "arc_agi_2_enriched.json").write_text(
        json.dumps(arc_enriched, indent=2),
        encoding="utf-8",
    )
    (output_dir / "math_competitions_empty_mind.json").write_text(
        json.dumps(math_empty, indent=2),
        encoding="utf-8",
    )
    (output_dir / "math_competitions_enriched.json").write_text(
        json.dumps(math_enriched, indent=2),
        encoding="utf-8",
    )
    (output_dir / "last_humanity_exam_empty_mind.json").write_text(
        json.dumps(lhe_empty, indent=2),
        encoding="utf-8",
    )
    (output_dir / "last_humanity_exam_enriched.json").write_text(
        json.dumps(lhe_enriched, indent=2),
        encoding="utf-8",
    )
    (output_dir / "mmlu_empty_mind.json").write_text(
        json.dumps(mmlu_empty, indent=2),
        encoding="utf-8",
    )
    (output_dir / "mmlu_enriched.json").write_text(
        json.dumps(mmlu_enriched, indent=2),
        encoding="utf-8",
    )
    summary_path = output_dir / "week14_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _append_history_entry(history_path, summary_path=summary_path, summary=summary)
    _append_usage_metrics(
        usage_log_path,
        {
            "timestamp": summary["timestamp"],
            "summary_path": str(summary_path),
            "runtime_usage": summary["runtime_usage"],
            "storage_roots": summary["storage_roots"],
        },
    )

    print("Knowledge3D Week 14 benchmark suite")
    print(f"  Persistence mode: {persistence_mode} (shared_instance={continuity.get('shared_instance')})")
    if historical_comparison:
        print("  Previous -> Current enriched comparison")
        for bench_name, cfg in BENCHMARK_CONFIG.items():
            entry = historical_comparison[bench_name]
            status = entry["status"]
            marker = "✅" if status == "MAINTAINED" else ("🎉" if status == "IMPROVEMENT" else "⚠️")
            print(
                f"  {cfg['label']}: {entry['previous']:.2%} -> {entry['current']:.2%} "
                f"({entry['delta']:+.2%}) {marker} {status} [ternary={entry['delta_ternary']}]"
            )
    else:
        print("  Previous -> Current enriched comparison: first run, no historical baseline yet")
    print("  Empty mind -> Enriched (same-run diagnostic)")
    print(
        f"  ARC-AGI 2: {arc_empty['accuracy']:.2%} -> {arc_enriched['accuracy']:.2%} "
        f"({arc_enriched['accuracy'] - arc_empty['accuracy']:+.2%})"
    )
    print(
        "  Math:      "
        f"{math_empty['overall_accuracy']:.2%} -> {math_enriched['overall_accuracy']:.2%} "
        f"({math_enriched['overall_accuracy'] - math_empty['overall_accuracy']:+.2%})"
    )
    print(
        f"  LHE:       {lhe_empty['accuracy']:.2%} -> {lhe_enriched['accuracy']:.2%} "
        f"({lhe_enriched['accuracy'] - lhe_empty['accuracy']:+.2%})"
    )
    print(
        f"  MMLU:      {mmlu_empty['accuracy']:.2%} -> {mmlu_enriched['accuracy']:.2%} "
        f"({mmlu_enriched['accuracy'] - mmlu_empty['accuracy']:+.2%})"
    )
    if bool(lhe_integrity.get("skipped", False)):
        print("  LHE integrity: skipped (max-lhe-questions <= 0)")
    else:
        print(
            "  LHE integrity: "
            f"source={lhe_integrity.get('dataset_source')} "
            f"fallback={bool(lhe_integrity.get('synthetic_fallback'))} "
            f"questions={int(lhe_integrity.get('evaluated_questions', 0))}/"
            f"{int(args.max_lhe_questions)} "
            f"min_ok={bool(lhe_integrity.get('question_count_ok'))}"
        )
    if bool(mmlu_integrity.get("skipped", False)):
        print("  MMLU integrity: skipped (max-mmlu-questions <= 0)")
    else:
        print(
            "  MMLU integrity: "
            f"source={mmlu_integrity.get('dataset_source')} "
            f"fallback={bool(mmlu_integrity.get('synthetic_fallback'))} "
            f"questions={int(mmlu_integrity.get('evaluated_questions', 0))}/"
            f"{int(args.max_mmlu_questions)} "
            f"min_ok={bool(mmlu_integrity.get('question_count_ok'))}"
        )
    print(f"  Empty storage root:    {empty_storage_root}")
    print(f"  Enriched storage root: {enriched_storage_root}")
    print(f"  ARC full PTX enabled: {bool(args.arc_enable_full_ptx)}")
    print(f"  ARC query scope: {args.arc_query_scope_galaxies}")
    print(f"  Math query scope: {args.math_query_scope_galaxies}")
    print(f"  LHE query scope: {args.lhe_query_scope_galaxies}")
    print(f"  MMLU query scope: {args.mmlu_query_scope_galaxies}")
    print(f"  ARC PTX validity strictness: {args.arc_ptx_validity_strictness}")
    print(f"  ARC constraint mode: {args.arc_constraint_mode}")
    print(f"  ARC figure-ground reversal: {bool(arc_enable_negative_forms)}")
    print(f"  ARC object-aware generation: {bool(args.arc_enable_object_aware_generation)}")
    print(
        "  ARC forced-navigation curriculum: "
        f"enabled={bool(args.arc_enable_forced_navigation_curriculum)} "
        f"ratio={float(args.arc_forced_navigation_ratio):.2f} "
        f"galaxies={args.arc_forced_navigation_required_galaxies}"
    )
    print(f"  ARC curriculum stage: {args.arc_curriculum_stage}")
    print(
        "  ARC rescue lane: "
        f"enabled={bool(args.arc_enable_rescue_lane)} "
        f"size={int(args.arc_rescue_lane_size)}"
    )
    print(f"  ARC oracle search lane size: {int(args.arc_oracle_search_lane_size)}")
    print(
        "  ARC oracle rejected rescue: "
        f"enabled={bool(args.arc_enable_oracle_rejected_rescue)} "
        f"size={int(args.arc_oracle_rejected_rescue_size)}"
    )
    print(f"  ARC dual-track oracle: {bool(args.arc_enable_dual_track_oracle)}")
    print(
        "  ARC penalty weights:"
        f" family={float(args.arc_family_penalty_weight):.2f}"
        f" shape={float(args.arc_shape_penalty_weight):.2f}"
        f" palette={float(args.arc_palette_penalty_weight):.2f}"
        f" object={float(args.arc_object_penalty_weight):.2f}"
    )
    print(
        "  ARC adaptive penalties: "
        f"enabled={bool(adaptive_penalties.get('enabled', False))} "
        f"applied={bool(adaptive_penalties.get('applied', False))}"
    )
    print(f"  ARC embedding lazy mode: {args.arc_embedding_lazy_mode}")
    sovereignty = summary["runtime_usage"].get("sovereignty", {})
    sovereignty_totals = sovereignty.get("totals", {})
    print(
        "  Sovereignty summary: "
        f"enforced={bool(sovereignty.get('enforced', False))} "
        f"gpu_tasks={int(sovereignty_totals.get('tasks_using_gpu', 0))}/"
        f"{int(sovereignty_totals.get('solved_tasks', 0))} "
        f"compliance={float(sovereignty_totals.get('sovereignty_compliance', 1.0)):.1%} "
        f"fallbacks={int(sovereignty_totals.get('fallback_triggered_count', 0))}"
    )
    if sovereignty.get("violations"):
        for key, reasons in sovereignty["violations"].items():
            print(f"    ⚠️ {key}: {', '.join(str(item) for item in reasons)}")
    arc_stage_gate = summary["runtime_usage"].get("arc_stage_gate", {})
    print(
        "  ARC stage gate: "
        f"passed={bool(arc_stage_gate.get('passed', False))} "
        f"reason={arc_stage_gate.get('reason', 'n/a')}"
    )
    gen_trend = summary["runtime_usage"].get("arc_generation_failure_trend", {})
    if gen_trend:
        print(
            "  ARC generation-failure trend: "
            f"delta_latest={gen_trend.get('delta_latest')} "
            f"flat_last_3_runs={bool(gen_trend.get('flat_last_3_runs', False))}"
        )
    if args.track_curriculum_coverage:
        coverage = summary["runtime_usage"].get("curriculum_coverage", {})
        gate = coverage.get("coverage_gate", {})
        print(
            "  Curriculum coverage gate (soft): "
            f"empty={gate.get('empty_block_count', 0)} "
            f"enriched={gate.get('enriched_block_count', 0)} "
            f"empty_query={gate.get('empty_query_count', 0)} "
            f"enriched_query={gate.get('enriched_query_count', 0)} "
            f"min={gate.get('min_required', 0)} "
            f"min_cross={gate.get('min_cross_galaxy_navigation_rate', 0.0):.2f} "
            f"pass_empty={gate.get('empty_block_passed', True)} "
            f"pass_enriched={gate.get('enriched_block_passed', True)} "
            f"pass_empty_query={gate.get('empty_query_passed', True)} "
            f"pass_enriched_query={gate.get('enriched_query_passed', True)} "
            f"pass_empty_cross={gate.get('empty_cross_rate_passed', True)} "
            f"pass_enriched_cross={gate.get('enriched_cross_rate_passed', True)}"
        )
    print(f"  Usage metrics log: {usage_log_path}")
    print(f"  History log: {history_path}")
    print(f"  Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
