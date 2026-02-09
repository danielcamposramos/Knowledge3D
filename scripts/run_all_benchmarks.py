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
        "--output-dir",
        default="../Knowledge3D.local/results/week14",
        help="Directory for benchmark outputs",
    )
    parser.add_argument(
        "--storage-root",
        default="../Knowledge3D.local",
        help="Base storage root used to derive isolated benchmark roots",
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
    args = parser.parse_args()
    os.environ["K3D_ARC_EMBEDDING_LAZY_MODE"] = str(args.arc_embedding_lazy_mode)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_storage_root = Path(args.storage_root)
    empty_storage_root = (
        Path(args.empty_storage_root) if args.empty_storage_root else (base_storage_root / "galaxies_empty_mind")
    )
    enriched_storage_root = (
        Path(args.enriched_storage_root) if args.enriched_storage_root else (base_storage_root / "galaxies_enriched")
    )
    empty_storage_root.mkdir(parents=True, exist_ok=True)
    enriched_storage_root.mkdir(parents=True, exist_ok=True)
    usage_log_path = base_storage_root / "logs" / "benchmark_usage_metrics.jsonl"

    # Empty mind runs
    empty_kv = Knowledgeverse(storage_root=empty_storage_root)
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

    # Enriched runs
    enriched_kv = Knowledgeverse(storage_root=enriched_storage_root)
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

    summary = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "storage_roots": {
            "empty_mind": str(empty_storage_root),
            "enriched": str(enriched_storage_root),
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
            "arc_embedding_lazy_mode": str(args.arc_embedding_lazy_mode),
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
    print(f"  ARC embedding lazy mode: {args.arc_embedding_lazy_mode}")
    print(f"  Usage metrics log: {usage_log_path}")
    print(f"  History log: {history_path}")
    print(f"  Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
