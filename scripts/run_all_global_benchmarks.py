#!/usr/bin/env python3
"""Run integrated K3D benchmarks and inventory global benchmark universe assets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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
from knowledge3d.knowledgeverse.trm_navigator import TRMNavigator

GLOBAL_BENCHMARK_CONFIG: dict[str, dict[str, str]] = {
    "arc_agi_2": {
        "label": "ARC-AGI 2",
        "metric": "accuracy",
        "specialist": "visual",
        "galaxy": "Drawing",
    },
    "math_competitions": {
        "label": "Math",
        "metric": "overall_accuracy",
        "specialist": "math",
        "galaxy": "Math",
    },
    "last_humanity_exam": {
        "label": "LHE",
        "metric": "accuracy",
        "specialist": "grammar",
        "galaxy": "Grammar",
    },
    "gsm8k_proxy": {
        "label": "GSM8K",
        "metric": "accuracy",
        "specialist": "math",
        "galaxy": "Math",
    },
    "mmlu_proxy": {
        "label": "MMLU",
        "metric": "accuracy",
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
    started_at = datetime.now(tz=timezone.utc).isoformat()
    start = time.perf_counter()
    gpu_before = _safe_gpu_snapshot()
    rss_before = _rss_bytes()
    result = fn()
    elapsed = time.perf_counter() - start
    gpu_after = _safe_gpu_snapshot()
    rss_after = _rss_bytes()
    return result, {
        "label": label,
        "started_at": started_at,
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


def _extract_score(summary: dict[str, Any], benchmark_name: str) -> float:
    if benchmark_name in {"arc_agi_2", "math_competitions", "last_humanity_exam"}:
        integrated = summary.get("integrated_results", {}).get(benchmark_name, {})
        metric = GLOBAL_BENCHMARK_CONFIG[benchmark_name]["metric"]
        enriched = integrated.get("enriched", {})
        return float(enriched.get(metric, 0.0))
    proxy = summary.get("proxy_results", {}).get(benchmark_name, {})
    return float(proxy.get("accuracy", 0.0))


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
    for benchmark_name, cfg in GLOBAL_BENCHMARK_CONFIG.items():
        prev_score = float(prev_scores.get(benchmark_name, 0.0))
        curr_score = _extract_score(current_summary, benchmark_name)
        delta = curr_score - prev_score
        out[benchmark_name] = {
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
) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": summary.get("timestamp"),
        "summary_path": str(summary_path),
        "scores": {
            benchmark_name: _extract_score(summary, benchmark_name)
            for benchmark_name in GLOBAL_BENCHMARK_CONFIG
        },
    }
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, separators=(",", ":"), sort_keys=True) + "\n")


def _persist_global_benchmark_memory(
    *,
    knowledgeverse: Knowledgeverse,
    summary: dict[str, Any],
    historical_comparison: dict[str, Any] | None,
) -> None:
    timestamp = str(summary.get("timestamp", "")).replace("-", "").replace(":", "").replace(".", "")
    for benchmark_name, cfg in GLOBAL_BENCHMARK_CONFIG.items():
        hist = (historical_comparison or {}).get(benchmark_name, {})
        curr_score = _extract_score(summary, benchmark_name)
        prev_score = float(hist.get("previous", curr_score))
        delta = float(hist.get("delta", 0.0))
        status = str(hist.get("status", "INITIALIZED"))
        delta_ternary = int(hist.get("delta_ternary", 0))

        event_data = {
            "benchmark": benchmark_name,
            "label": cfg["label"],
            "previous_score": prev_score,
            "current_score": curr_score,
            "delta": delta,
            "delta_ternary": delta_ternary,
            "current_score_ternary": _score_to_ternary(curr_score),
            "status": status,
            "specialist": cfg["specialist"],
            "galaxy": cfg["galaxy"],
            "query": f"global benchmark outcome {benchmark_name}",
            "confidence": curr_score,
            "verification": "historical_global_benchmark_tracking",
        }
        knowledgeverse.log_event(event_type="global_benchmark_outcome", event_data=event_data)

        grammar_entry = {
            "id": f"global_benchmark_memory_{benchmark_name}_{timestamp}",
            "name": f"Global Benchmark Memory {cfg['label']}",
            "domain": "grammar",
            "category": "benchmark_memory_global",
            "rpn_program": "PREV CURR SUB SIGN_TERNARY",
            "metadata": {
                "benchmark": benchmark_name,
                "label": cfg["label"],
                "previous_score": prev_score,
                "current_score": curr_score,
                "delta": delta,
                "delta_ternary": delta_ternary,
                "status": status,
                "generated": False,
                "source": "scripts/run_all_global_benchmarks.py",
            },
        }
        knowledgeverse.galaxy_manager.add_entry("Grammar", grammar_entry)


def _dataset_inventory(dataset_root: Path) -> dict[str, Any]:
    inventory: dict[str, Any] = {}

    def _entry(name: str, base: Path, patterns: list[str]) -> None:
        files: list[Path] = []
        for pattern in patterns:
            files.extend(base.glob(pattern))
        inventory[name] = {
            "path": str(base),
            "present": base.exists(),
            "file_count": len(files),
            "sample_files": [str(p) for p in sorted(files)[:5]],
        }

    _entry("gpqa", dataset_root / "gpqa", ["**/*.json", "**/*.jsonl", "**/*.csv"])
    _entry("mmlu", dataset_root / "mmlu", ["**/*.csv"])
    _entry("gsm8k", dataset_root / "gsm8k", ["**/*.jsonl"])
    _entry("humaneval", dataset_root / "humaneval", ["**/*.jsonl", "**/*.json"])
    _entry("hellaswag", dataset_root / "hellaswag", ["**/*.jsonl", "**/*.json"])
    _entry("truthfulqa", dataset_root / "truthfulqa", ["**/*.csv", "**/*.json"])
    _entry("big_bench", dataset_root / "big_bench", ["**/*.json", "**/*.jsonl"])
    _entry("alphageometry", dataset_root / "alphageometry", ["**/*.json", "**/*.txt"])
    _entry("theoremqa", dataset_root / "theoremqa", ["**/*.json", "**/*.jsonl"])
    _entry("bbh", dataset_root / "bbh", ["**/*.json", "**/*.jsonl"])
    _entry("drop", dataset_root / "drop", ["**/*.json", "**/*.jsonl"])
    _entry("piqa", dataset_root / "piqa", ["**/*.jsonl", "**/*.json", "**/*.lst"])
    _entry("math", dataset_root / "math", ["**/*.jsonl", "**/*.parquet"])
    _entry("arc_agi_2", dataset_root / "arc_agi_2", ["**/*.json"])
    _entry("imo_grand_challenge", dataset_root / "imo_grand_challenge", ["**/*"])
    return inventory


def _extract_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value)
    direct = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not direct:
        return None
    try:
        return float(direct[-1])
    except ValueError:
        return None


def _run_gsm8k_proxy(
    *,
    kv: Knowledgeverse,
    dataset_root: Path,
    max_questions: int,
) -> dict[str, Any]:
    # Supports OpenAI grade-school-math repo layout.
    candidates = [
        dataset_root / "gsm8k/repo/grade_school_math/data/test.jsonl",
        dataset_root / "gsm8k/repo/grade_school_math/data/train.jsonl",
        dataset_root / "gsm8k/test.jsonl",
    ]
    data_path = next((p for p in candidates if p.exists()), None)
    if data_path is None:
        return {"available": False, "reason": "dataset_not_found"}

    navigator = TRMNavigator(knowledgeverse=kv)
    total = 0
    correct = 0
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if total >= max_questions:
                break
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = str(payload.get("question", "")).strip()
            answer = payload.get("answer")
            if not question or answer is None:
                continue
            composed = navigator.navigate_and_compose(
                query=question,
                specialist="auto",
                domain_hint="math",
                use_enriched=True,
                use_forward_backward=True,
            )
            predicted = navigator.execute(composed)
            expected_num = _extract_number(answer)
            predicted_num = _extract_number(predicted)
            if expected_num is not None and predicted_num is not None:
                is_correct = abs(expected_num - predicted_num) <= 1e-3
            else:
                is_correct = str(predicted).strip().lower() == str(answer).strip().lower()
            total += 1
            if is_correct:
                correct += 1

    return {
        "available": True,
        "dataset_path": str(data_path),
        "total_questions": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
    }


def _run_mmlu_proxy(
    *,
    kv: Knowledgeverse,
    dataset_root: Path,
    max_questions: int,
) -> dict[str, Any]:
    # Supports Hendrycks MMLU tar layout.
    candidates = [
        dataset_root / "mmlu/data/test",
        dataset_root / "mmlu/repo/data/test",
    ]
    test_dir = next((p for p in candidates if p.exists()), None)
    if test_dir is None:
        return {"available": False, "reason": "dataset_not_found"}

    csv_files = sorted(test_dir.glob("*.csv"))
    if not csv_files:
        return {"available": False, "reason": "no_csv_files", "dataset_path": str(test_dir)}

    navigator = TRMNavigator(knowledgeverse=kv)
    total = 0
    correct = 0
    for csv_path in csv_files:
        if total >= max_questions:
            break
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if total >= max_questions:
                    break
                if len(row) < 6:
                    continue
                question, a, b, c, d, answer = row[:6]
                options = [a, b, c, d]
                composed = navigator.navigate_and_compose(
                    query=question,
                    specialist="auto",
                    domain_hint="logic",
                    use_enriched=True,
                    use_forward_backward=True,
                )
                reasoning = navigator.execute(composed)
                predicted = navigator.select_answer(reasoning=reasoning, options=options)

                # MMLU answer labels are usually A/B/C/D
                letter = str(answer).strip().upper()
                expected = ""
                if letter in {"A", "B", "C", "D"}:
                    expected = options[ord(letter) - ord("A")]
                else:
                    expected = str(answer).strip()
                is_correct = str(predicted).strip() == expected
                total += 1
                if is_correct:
                    correct += 1

    return {
        "available": True,
        "dataset_path": str(test_dir),
        "total_questions": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
    }


def _run_integrated_suite(
    *,
    output_dir: Path,
    storage_root: Path,
    max_arc_tasks: int,
    max_math_problems: int,
    max_lhe_questions: int,
    arc_enable_contrastive_learning: bool = False,
    arc_enable_validity_gates: bool = False,
    arc_enable_fuzzy_oracle: bool = False,
    arc_fuzzy_oracle_threshold: float = 0.95,
    arc_enable_ptx_ranking: bool = False,
    benchmark_runtime_seeding: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], Knowledgeverse, Knowledgeverse]:
    empty_root = storage_root / "galaxies_empty_mind"
    enriched_root = storage_root / "galaxies_enriched"
    empty_root.mkdir(parents=True, exist_ok=True)
    enriched_root.mkdir(parents=True, exist_ok=True)

    empty_kv = Knowledgeverse(storage_root=empty_root)
    enriched_kv = Knowledgeverse(storage_root=enriched_root)

    empty_galaxy_counts_start = _collect_default_galaxy_counts(empty_kv)
    arc_empty, arc_empty_metrics = _run_with_metrics(
        "global_arc_empty_mind",
        lambda: ARCAGI2Benchmark(
            knowledgeverse=empty_kv,
            max_tasks=max_arc_tasks,
            enable_contrastive_learning=arc_enable_contrastive_learning,
            enable_validity_gates=arc_enable_validity_gates,
            enable_fuzzy_oracle=arc_enable_fuzzy_oracle,
            fuzzy_oracle_threshold=arc_fuzzy_oracle_threshold,
            enable_ptx_ranking=arc_enable_ptx_ranking,
            runtime_seed_knowledge=benchmark_runtime_seeding,
        ).run_benchmark(use_enriched=False),
    )
    math_empty, math_empty_metrics = _run_with_metrics(
        "global_math_empty_mind",
        lambda: MathCompetitionBenchmark(
            knowledgeverse=empty_kv,
            max_problems=max_math_problems,
            runtime_seed_knowledge=benchmark_runtime_seeding,
        ).run_benchmark(use_enriched=False),
    )
    lhe_empty, lhe_empty_metrics = _run_with_metrics(
        "global_lhe_empty_mind",
        lambda: LastHumanityExamBenchmark(
            knowledgeverse=empty_kv,
            max_questions=max_lhe_questions,
            runtime_seed_knowledge=benchmark_runtime_seeding,
        ).run_benchmark(use_enriched=False),
    )
    empty_galaxy_counts_end = _collect_default_galaxy_counts(empty_kv)

    enriched_galaxy_counts_start = _collect_default_galaxy_counts(enriched_kv)
    arc_enriched, arc_enriched_metrics = _run_with_metrics(
        "global_arc_enriched",
        lambda: ARCAGI2Benchmark(
            knowledgeverse=enriched_kv,
            max_tasks=max_arc_tasks,
            enable_contrastive_learning=arc_enable_contrastive_learning,
            enable_validity_gates=arc_enable_validity_gates,
            enable_fuzzy_oracle=arc_enable_fuzzy_oracle,
            fuzzy_oracle_threshold=arc_fuzzy_oracle_threshold,
            enable_ptx_ranking=arc_enable_ptx_ranking,
            runtime_seed_knowledge=benchmark_runtime_seeding,
        ).run_benchmark(use_enriched=True),
    )
    math_enriched, math_enriched_metrics = _run_with_metrics(
        "global_math_enriched",
        lambda: MathCompetitionBenchmark(
            knowledgeverse=enriched_kv,
            max_problems=max_math_problems,
            runtime_seed_knowledge=benchmark_runtime_seeding,
        ).run_benchmark(use_enriched=True),
    )
    lhe_enriched, lhe_enriched_metrics = _run_with_metrics(
        "global_lhe_enriched",
        lambda: LastHumanityExamBenchmark(
            knowledgeverse=enriched_kv,
            max_questions=max_lhe_questions,
            runtime_seed_knowledge=benchmark_runtime_seeding,
        ).run_benchmark(use_enriched=True),
    )
    enriched_galaxy_counts_end = _collect_default_galaxy_counts(enriched_kv)

    integrated = {
        "arc_agi_2": {
            "empty_mind": arc_empty,
            "enriched": arc_enriched,
            "delta": arc_enriched["accuracy"] - arc_empty["accuracy"],
        },
        "math_competitions": {
            "empty_mind": math_empty,
            "enriched": math_enriched,
            "delta": math_enriched["overall_accuracy"] - math_empty["overall_accuracy"],
        },
        "last_humanity_exam": {
            "empty_mind": lhe_empty,
            "enriched": lhe_enriched,
            "delta": lhe_enriched["accuracy"] - lhe_empty["accuracy"],
        },
    }

    (output_dir / "integrated_arc_empty.json").write_text(json.dumps(arc_empty, indent=2), encoding="utf-8")
    (output_dir / "integrated_arc_enriched.json").write_text(json.dumps(arc_enriched, indent=2), encoding="utf-8")
    (output_dir / "integrated_math_empty.json").write_text(json.dumps(math_empty, indent=2), encoding="utf-8")
    (output_dir / "integrated_math_enriched.json").write_text(json.dumps(math_enriched, indent=2), encoding="utf-8")
    (output_dir / "integrated_lhe_empty.json").write_text(json.dumps(lhe_empty, indent=2), encoding="utf-8")
    (output_dir / "integrated_lhe_enriched.json").write_text(json.dumps(lhe_enriched, indent=2), encoding="utf-8")

    runtime_usage = {
        "runtime_seed_knowledge": bool(benchmark_runtime_seeding),
        "arc_enable_ptx_ranking": bool(arc_enable_ptx_ranking),
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
    }

    return integrated, runtime_usage, empty_kv, enriched_kv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../Knowledge3D.local/results/global_benchmarks"),
        help="Output folder for reports.",
    )
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=Path("../Knowledge3D.local"),
        help="Knowledgeverse storage root.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../Knowledge3D.local/datasets/global_benchmarks"),
        help="Downloaded benchmark universe root.",
    )
    parser.add_argument("--max-arc-tasks", type=int, default=100)
    parser.add_argument("--max-math-problems", type=int, default=100)
    parser.add_argument("--max-lhe-questions", type=int, default=50)
    parser.add_argument(
        "--arc-enable-contrastive-learning",
        action="store_true",
        help="Enable ARC contrastive anti-pattern generation.",
    )
    parser.add_argument(
        "--arc-enable-validity-gates",
        action="store_true",
        help="Enable ARC validity gates before candidate selection.",
    )
    parser.add_argument(
        "--arc-enable-fuzzy-oracle",
        action="store_true",
        help="Enable fuzzy oracle diagnostics for ARC benchmark.",
    )
    parser.add_argument(
        "--arc-fuzzy-oracle-threshold",
        type=float,
        default=0.95,
        help="Fuzzy oracle threshold in [0,1].",
    )
    parser.add_argument(
        "--arc-enable-ptx-ranking",
        action="store_true",
        help="Enable PTX/GPU-backed ARC candidate scoring and top-1 selection.",
    )
    parser.add_argument(
        "--run-proxy",
        action="store_true",
        help="Run lightweight proxy evaluations for MMLU/GSM8K when datasets exist.",
    )
    parser.add_argument(
        "--benchmark-runtime-seeding",
        action="store_true",
        help="Allow benchmark classes to inject seed entries during task loops (disabled by default).",
    )
    parser.add_argument("--max-proxy-questions", type=int, default=50)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = _dataset_inventory(args.dataset_root)
    integrated, runtime_usage, _empty_kv, enriched_kv = _run_integrated_suite(
        output_dir=args.output_dir,
        storage_root=args.storage_root,
        max_arc_tasks=args.max_arc_tasks,
        max_math_problems=args.max_math_problems,
        max_lhe_questions=args.max_lhe_questions,
        arc_enable_contrastive_learning=args.arc_enable_contrastive_learning,
        arc_enable_validity_gates=args.arc_enable_validity_gates,
        arc_enable_fuzzy_oracle=args.arc_enable_fuzzy_oracle,
        arc_fuzzy_oracle_threshold=args.arc_fuzzy_oracle_threshold,
        arc_enable_ptx_ranking=args.arc_enable_ptx_ranking,
        benchmark_runtime_seeding=args.benchmark_runtime_seeding,
    )

    proxy_results: dict[str, Any] = {}
    proxy_metrics: list[dict[str, Any]] = []
    if args.run_proxy:
        # Reuse the same enriched world for proxy tasks to preserve single-universe continuity.
        proxy_results["gsm8k_proxy"], gsm8k_proxy_metrics = _run_with_metrics(
            "global_gsm8k_proxy",
            lambda: _run_gsm8k_proxy(
                kv=enriched_kv,
                dataset_root=args.dataset_root,
                max_questions=args.max_proxy_questions,
            ),
        )
        proxy_results["mmlu_proxy"], mmlu_proxy_metrics = _run_with_metrics(
            "global_mmlu_proxy",
            lambda: _run_mmlu_proxy(
                kv=enriched_kv,
                dataset_root=args.dataset_root,
                max_questions=args.max_proxy_questions,
            ),
        )
        proxy_metrics.extend([gsm8k_proxy_metrics, mmlu_proxy_metrics])

    runtime_usage["proxy_runs"] = proxy_metrics
    runtime_usage["proxy_reuses_enriched_world"] = bool(args.run_proxy)

    history_path = args.storage_root / "benchmarks" / "run_all_global_benchmarks_history.jsonl"
    usage_log_path = args.storage_root / "logs" / "global_benchmark_usage_metrics.jsonl"
    summary = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root.resolve()),
        "storage_root": str(args.storage_root.resolve()),
        "integrated_results": integrated,
        "global_inventory": inventory,
        "proxy_results": proxy_results,
        "runtime_usage": runtime_usage,
    }
    previous_entry = _load_previous_history_entry(history_path)
    historical_comparison = _build_historical_comparison(previous_entry, summary)
    summary["historical_comparison"] = historical_comparison
    summary["history"] = {
        "path": str(history_path),
        "previous_timestamp": (previous_entry or {}).get("timestamp"),
    }

    _persist_global_benchmark_memory(
        knowledgeverse=enriched_kv,
        summary=summary,
        historical_comparison=historical_comparison,
    )

    summary_path = args.output_dir / "global_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _append_history_entry(history_path, summary_path=summary_path, summary=summary)
    _append_usage_metrics(
        usage_log_path,
        {
            "timestamp": summary["timestamp"],
            "summary_path": str(summary_path),
            "runtime_usage": summary["runtime_usage"],
            "storage_root": summary["storage_root"],
        },
    )

    arc = integrated["arc_agi_2"]
    math = integrated["math_competitions"]
    lhe = integrated["last_humanity_exam"]
    print("Knowledge3D global benchmark suite")
    if historical_comparison:
        print("  Previous -> Current enriched/proxy comparison")
        for benchmark_name, cfg in GLOBAL_BENCHMARK_CONFIG.items():
            entry = historical_comparison[benchmark_name]
            status = entry["status"]
            marker = "✅" if status == "MAINTAINED" else ("🎉" if status == "IMPROVEMENT" else "⚠️")
            print(
                f"  {cfg['label']}: {entry['previous']:.2%} -> {entry['current']:.2%} "
                f"({entry['delta']:+.2%}) {marker} {status} [ternary={entry['delta_ternary']}]"
            )
    else:
        print("  Previous -> Current comparison: first run, no historical baseline yet")
    print("  Empty mind -> Enriched (same-run integrated diagnostic)")
    print(
        f"  ARC-AGI 2: {arc['empty_mind']['accuracy']:.2%} -> {arc['enriched']['accuracy']:.2%} "
        f"({arc['delta']:+.2%})"
    )
    print(
        f"  Math:      {math['empty_mind']['overall_accuracy']:.2%} -> "
        f"{math['enriched']['overall_accuracy']:.2%} ({math['delta']:+.2%})"
    )
    print(
        f"  LHE:       {lhe['empty_mind']['accuracy']:.2%} -> {lhe['enriched']['accuracy']:.2%} "
        f"({lhe['delta']:+.2%})"
    )
    if proxy_results:
        for name, result in proxy_results.items():
            if result.get("available"):
                print(f"  {name}: {result.get('accuracy', 0.0):.2%} ({result.get('correct', 0)}/{result.get('total_questions', 0)})")
            else:
                print(f"  {name}: unavailable ({result.get('reason', 'n/a')})")
    print(f"  Inventory root: {args.dataset_root}")
    print(f"  Usage metrics log: {usage_log_path}")
    print(f"  History log: {history_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
