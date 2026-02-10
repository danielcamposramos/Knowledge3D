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
}


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
    diagnostics = (
        summary.get("benchmarks", {})
        .get("arc_agi_2", {})
        .get("diagnostics", {})
        .get("enriched", {})
    )
    enriched = summary.get("benchmarks", {}).get("arc_agi_2", {}).get("enriched", {})
    thresholds = {
        "accuracy_min": 0.10,
        "oracle_at_all_min": 0.10,
        "fuzzy_oracle_at_all_min": 0.20,
        "generation_failure_rate_max": 0.50,
    }
    actuals = {
        "accuracy": float(enriched.get("accuracy", 0.0)),
        "oracle_at_all": float(diagnostics.get("oracle_at_all", 0.0)),
        "fuzzy_oracle_at_all": float(diagnostics.get("fuzzy_oracle_at_all", 0.0)),
        "generation_failure_rate": float(diagnostics.get("generation_failure_rate", 1.0)),
    }
    checks = {
        "accuracy": actuals["accuracy"] >= thresholds["accuracy_min"],
        "oracle_at_all": actuals["oracle_at_all"] >= thresholds["oracle_at_all_min"],
        "fuzzy_oracle_at_all": actuals["fuzzy_oracle_at_all"] >= thresholds["fuzzy_oracle_at_all_min"],
        "generation_failure_rate": actuals["generation_failure_rate"] <= thresholds["generation_failure_rate_max"],
    }
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
    parser.add_argument("--max-arc-tasks", type=int, default=100, help="ARC task limit")
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
    parser.add_argument("--max-lhe-questions", type=int, default=100, help="LHE question limit")
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
        "--output-dir",
        default="../Knowledge3D.local/results/week14",
        help="Directory for benchmark outputs",
    )
    parser.add_argument(
        "--storage-root",
        default="../Knowledge3D.local",
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
    if args.arc_enable_full_ptx and not args.arc_enable_ptx_ranking:
        # Full PTX mode implies PTX ranking. Keep CLI ergonomic and avoid
        # accidental CPU fallback when users only pass --arc-enable-full-ptx.
        args.arc_enable_ptx_ranking = True
    os.environ["K3D_ARC_EMBEDDING_LAZY_MODE"] = str(args.arc_embedding_lazy_mode)
    # Enforce PTX-only ARC path by default for Week 21+ architecture.
    os.environ.setdefault("K3D_REQUIRE_PTX_ARC_PIPELINE", "true")
    if args.arc_enable_full_ptx:
        os.environ.setdefault("K3D_ALLOW_LEGACY_ARC_PIPELINE", "false")

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
    persistence_mode = str(args.model_persistence_mode)
    continuity: dict[str, Any] = {
        "mode": persistence_mode,
        "shared_instance": False,
        "instance_ids": {},
    }

    if persistence_mode == "unified":
        shared_kv = Knowledgeverse(storage_root=unified_storage_root)
        continuity["shared_instance"] = True
        continuity["instance_ids"] = {
            "empty_mind": int(id(shared_kv)),
            "enriched": int(id(shared_kv)),
        }

        shared_galaxy_counts_start = _collect_default_galaxy_counts(shared_kv)
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
                enable_rescue_lane=args.arc_enable_rescue_lane,
                rescue_lane_size=args.arc_rescue_lane_size,
                enable_dual_track_oracle=args.arc_enable_dual_track_oracle,
                family_penalty_weight=args.arc_family_penalty_weight,
                shape_penalty_weight=args.arc_shape_penalty_weight,
                palette_penalty_weight=args.arc_palette_penalty_weight,
                object_penalty_weight=args.arc_object_penalty_weight,
                runtime_seed_knowledge=args.benchmark_runtime_seeding,
            ).run_benchmark(use_enriched=False),
        )
        math_empty, math_empty_metrics = _run_with_metrics(
            "math_empty_mind",
            lambda: MathCompetitionBenchmark(
                knowledgeverse=shared_kv,
                dataset_path=args.math_dataset_path,
                max_problems=args.max_math_problems,
                runtime_seed_knowledge=args.benchmark_runtime_seeding,
            ).run_benchmark(use_enriched=False),
        )
        lhe_empty, lhe_empty_metrics = _run_with_metrics(
            "lhe_empty_mind",
            lambda: LastHumanityExamBenchmark(
                knowledgeverse=shared_kv,
                dataset_path=args.lhe_dataset_path,
                max_questions=args.max_lhe_questions,
                runtime_seed_knowledge=args.benchmark_runtime_seeding,
            ).run_benchmark(use_enriched=False),
        )
        shared_after_empty = _collect_default_galaxy_counts(shared_kv)

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
                enable_rescue_lane=args.arc_enable_rescue_lane,
                rescue_lane_size=args.arc_rescue_lane_size,
                enable_dual_track_oracle=args.arc_enable_dual_track_oracle,
                family_penalty_weight=args.arc_family_penalty_weight,
                shape_penalty_weight=args.arc_shape_penalty_weight,
                palette_penalty_weight=args.arc_palette_penalty_weight,
                object_penalty_weight=args.arc_object_penalty_weight,
                runtime_seed_knowledge=args.benchmark_runtime_seeding,
            ).run_benchmark(use_enriched=True),
        )
        math_enriched, math_enriched_metrics = _run_with_metrics(
            "math_enriched",
            lambda: MathCompetitionBenchmark(
                knowledgeverse=shared_kv,
                dataset_path=args.math_dataset_path,
                max_problems=args.max_math_problems,
                runtime_seed_knowledge=args.benchmark_runtime_seeding,
            ).run_benchmark(use_enriched=True),
        )
        lhe_enriched, lhe_enriched_metrics = _run_with_metrics(
            "lhe_enriched",
            lambda: LastHumanityExamBenchmark(
                knowledgeverse=shared_kv,
                dataset_path=args.lhe_dataset_path,
                max_questions=args.max_lhe_questions,
                runtime_seed_knowledge=args.benchmark_runtime_seeding,
            ).run_benchmark(use_enriched=True),
        )
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
        continuity["instance_ids"]["empty_mind"] = int(id(empty_kv))
        empty_galaxy_counts_start = _collect_default_galaxy_counts(empty_kv)
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
                enable_rescue_lane=args.arc_enable_rescue_lane,
                rescue_lane_size=args.arc_rescue_lane_size,
                enable_dual_track_oracle=args.arc_enable_dual_track_oracle,
                family_penalty_weight=args.arc_family_penalty_weight,
                shape_penalty_weight=args.arc_shape_penalty_weight,
                palette_penalty_weight=args.arc_palette_penalty_weight,
                object_penalty_weight=args.arc_object_penalty_weight,
                runtime_seed_knowledge=args.benchmark_runtime_seeding,
            ).run_benchmark(use_enriched=False),
        )
        math_empty, math_empty_metrics = _run_with_metrics(
            "math_empty_mind",
            lambda: MathCompetitionBenchmark(
                knowledgeverse=empty_kv,
                dataset_path=args.math_dataset_path,
                max_problems=args.max_math_problems,
                runtime_seed_knowledge=args.benchmark_runtime_seeding,
            ).run_benchmark(use_enriched=False),
        )
        lhe_empty, lhe_empty_metrics = _run_with_metrics(
            "lhe_empty_mind",
            lambda: LastHumanityExamBenchmark(
                knowledgeverse=empty_kv,
                dataset_path=args.lhe_dataset_path,
                max_questions=args.max_lhe_questions,
                runtime_seed_knowledge=args.benchmark_runtime_seeding,
            ).run_benchmark(use_enriched=False),
        )
        empty_galaxy_counts_end = _collect_default_galaxy_counts(empty_kv)

        enriched_kv = Knowledgeverse(storage_root=enriched_storage_root)
        continuity["instance_ids"]["enriched"] = int(id(enriched_kv))
        continuity["shared_instance"] = continuity["instance_ids"]["empty_mind"] == continuity["instance_ids"]["enriched"]
        enriched_galaxy_counts_start = _collect_default_galaxy_counts(enriched_kv)
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
                enable_rescue_lane=args.arc_enable_rescue_lane,
                rescue_lane_size=args.arc_rescue_lane_size,
                enable_dual_track_oracle=args.arc_enable_dual_track_oracle,
                family_penalty_weight=args.arc_family_penalty_weight,
                shape_penalty_weight=args.arc_shape_penalty_weight,
                palette_penalty_weight=args.arc_palette_penalty_weight,
                object_penalty_weight=args.arc_object_penalty_weight,
                runtime_seed_knowledge=args.benchmark_runtime_seeding,
            ).run_benchmark(use_enriched=True),
        )
        math_enriched, math_enriched_metrics = _run_with_metrics(
            "math_enriched",
            lambda: MathCompetitionBenchmark(
                knowledgeverse=enriched_kv,
                dataset_path=args.math_dataset_path,
                max_problems=args.max_math_problems,
                runtime_seed_knowledge=args.benchmark_runtime_seeding,
            ).run_benchmark(use_enriched=True),
        )
        lhe_enriched, lhe_enriched_metrics = _run_with_metrics(
            "lhe_enriched",
            lambda: LastHumanityExamBenchmark(
                knowledgeverse=enriched_kv,
                dataset_path=args.lhe_dataset_path,
                max_questions=args.max_lhe_questions,
                runtime_seed_knowledge=args.benchmark_runtime_seeding,
            ).run_benchmark(use_enriched=True),
        )
        enriched_galaxy_counts_end = _collect_default_galaxy_counts(enriched_kv)

    if args.arc_enable_full_ptx:
        _assert_arc_solver_contract(arc_empty, required_solver="arc_ptx_ops")
        _assert_arc_solver_contract(arc_enriched, required_solver="arc_ptx_ops")

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
        },
        "runtime_usage": {
            "runtime_seed_knowledge": bool(args.benchmark_runtime_seeding),
            "arc_enable_ptx_ranking": bool(args.arc_enable_ptx_ranking),
            "arc_enable_full_ptx": bool(args.arc_enable_full_ptx),
            "arc_ptx_validity_strictness": str(args.arc_ptx_validity_strictness),
            "arc_constraint_mode": str(args.arc_constraint_mode),
            "arc_enable_figure_ground_reversal": bool(arc_enable_negative_forms),
            "arc_enable_object_aware_generation": bool(args.arc_enable_object_aware_generation),
            "arc_enable_rescue_lane": bool(args.arc_enable_rescue_lane),
            "arc_rescue_lane_size": int(args.arc_rescue_lane_size),
            "arc_enable_dual_track_oracle": bool(args.arc_enable_dual_track_oracle),
            "arc_family_penalty_weight": float(args.arc_family_penalty_weight),
            "arc_shape_penalty_weight": float(args.arc_shape_penalty_weight),
            "arc_palette_penalty_weight": float(args.arc_palette_penalty_weight),
            "arc_object_penalty_weight": float(args.arc_object_penalty_weight),
            "arc_embedding_lazy_mode": str(args.arc_embedding_lazy_mode),
            "persistence": continuity,
            "runs": [
                arc_empty_metrics,
                math_empty_metrics,
                lhe_empty_metrics,
                arc_enriched_metrics,
                math_enriched_metrics,
                lhe_enriched_metrics,
            ],
            "galaxy_counts": {
                "empty_mind_start": empty_galaxy_counts_start,
                "empty_mind_end": empty_galaxy_counts_end,
                "enriched_start": enriched_galaxy_counts_start,
                "enriched_end": enriched_galaxy_counts_end,
            },
        },
    }

    if args.track_curriculum_coverage:
        def _touched(start: dict[str, int], end: dict[str, int]) -> list[str]:
            keys = sorted(set(start.keys()) | set(end.keys()))
            touched: list[str] = []
            for key in keys:
                if int(start.get(key, 0)) != int(end.get(key, 0)):
                    touched.append(key)
            return touched

        empty_touched = _touched(empty_galaxy_counts_start, empty_galaxy_counts_end)
        enriched_touched = _touched(enriched_galaxy_counts_start, enriched_galaxy_counts_end)
        union_touched = sorted(set(empty_touched) | set(enriched_touched))

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
        gate = {
            "min_required": min_required,
            "empty_block_count": len(empty_touched),
            "enriched_block_count": len(enriched_touched),
            "empty_block_passed": (len(empty_touched) >= min_required) if min_required else True,
            "enriched_block_passed": (len(enriched_touched) >= min_required) if min_required else True,
            "soft_gate_only": True,
        }

        summary["runtime_usage"]["curriculum_coverage"] = {
            "galaxies_touched": {
                "empty_block": empty_touched,
                "enriched_block": enriched_touched,
                "union": union_touched,
            },
            "specialist_routes": specialist_routes,
            "ternary_quality_delta": ternary_quality_delta,
            "coverage_gate": gate,
        }
    summary["runtime_usage"]["arc_stage_gate"] = _compute_arc_stage_gate(summary)
    history_path = base_storage_root / "benchmarks" / "run_all_benchmarks_history.jsonl"
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
    print(f"  Empty storage root:    {empty_storage_root}")
    print(f"  Enriched storage root: {enriched_storage_root}")
    print(f"  ARC full PTX enabled: {bool(args.arc_enable_full_ptx)}")
    print(f"  ARC PTX validity strictness: {args.arc_ptx_validity_strictness}")
    print(f"  ARC constraint mode: {args.arc_constraint_mode}")
    print(f"  ARC figure-ground reversal: {bool(arc_enable_negative_forms)}")
    print(f"  ARC object-aware generation: {bool(args.arc_enable_object_aware_generation)}")
    print(
        "  ARC rescue lane: "
        f"enabled={bool(args.arc_enable_rescue_lane)} "
        f"size={int(args.arc_rescue_lane_size)}"
    )
    print(f"  ARC dual-track oracle: {bool(args.arc_enable_dual_track_oracle)}")
    print(
        "  ARC penalty weights:"
        f" family={float(args.arc_family_penalty_weight):.2f}"
        f" shape={float(args.arc_shape_penalty_weight):.2f}"
        f" palette={float(args.arc_palette_penalty_weight):.2f}"
        f" object={float(args.arc_object_penalty_weight):.2f}"
    )
    print(f"  ARC embedding lazy mode: {args.arc_embedding_lazy_mode}")
    arc_stage_gate = summary["runtime_usage"].get("arc_stage_gate", {})
    print(
        "  ARC stage gate: "
        f"passed={bool(arc_stage_gate.get('passed', False))} "
        f"reason={arc_stage_gate.get('reason', 'n/a')}"
    )
    if args.track_curriculum_coverage:
        coverage = summary["runtime_usage"].get("curriculum_coverage", {})
        gate = coverage.get("coverage_gate", {})
        print(
            "  Curriculum coverage gate (soft): "
            f"empty={gate.get('empty_block_count', 0)} "
            f"enriched={gate.get('enriched_block_count', 0)} "
            f"min={gate.get('min_required', 0)} "
            f"pass_empty={gate.get('empty_block_passed', True)} "
            f"pass_enriched={gate.get('enriched_block_passed', True)}"
        )
    print(f"  Usage metrics log: {usage_log_path}")
    print(f"  History log: {history_path}")
    print(f"  Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
