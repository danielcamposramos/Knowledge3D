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


class SovereigntyViolation(RuntimeError):
    """Raised when solved benchmark tasks have no sovereign GPU evidence."""


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
    if limit is None:
        return True
    try:
        return int(limit) > 0
    except Exception:
        return False


def _arc_skipped_result(*, use_enriched: bool) -> dict[str, Any]:
    return {
        "benchmark": "ARC-AGI 2/3",
        "dataset_path": "skipped",
        "dataset_version": "arc_agi_2",
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


def _math_skipped_result(*, use_enriched: bool) -> dict[str, Any]:
    return {
        "benchmark": "Math Competitions",
        "dataset_path": "skipped",
        "use_enriched": bool(use_enriched),
        "results_by_competition": {},
        "overall_accuracy": 0.0,
        "total": 0,
        "correct": 0,
        "results": [],
    }


def _lhe_skipped_result(*, use_enriched: bool) -> dict[str, Any]:
    return {
        "benchmark": "Last Humanity Exam",
        "dataset_path": "skipped",
        "dataset_source": "skipped",
        "dataset_file": None,
        "synthetic_fallback": False,
        "use_enriched": bool(use_enriched),
        "total_questions": 0,
        "correct": 0,
        "accuracy": 0.0,
        "results": [],
        "results_by_domain": {},
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


def _summarize_payload_sovereignty(
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
    arc_enable_full_ptx: bool = False,
    arc_ptx_validity_strictness: str = "medium",
    benchmark_runtime_seeding: bool = False,
    model_persistence_mode: str = "unified",
    unified_storage_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Knowledgeverse, Knowledgeverse]:
    empty_root = storage_root / "galaxies_empty_mind"
    enriched_root = storage_root / "galaxies_enriched"
    shared_root = unified_storage_root or enriched_root
    empty_root.mkdir(parents=True, exist_ok=True)
    enriched_root.mkdir(parents=True, exist_ok=True)
    shared_root.mkdir(parents=True, exist_ok=True)
    run_arc = _is_enabled_limit(max_arc_tasks)
    run_math = _is_enabled_limit(max_math_problems)
    run_lhe = _is_enabled_limit(max_lhe_questions)

    continuity: dict[str, Any] = {
        "mode": str(model_persistence_mode),
        "shared_instance": False,
        "instance_ids": {},
    }

    if str(model_persistence_mode) == "unified":
        shared_kv = Knowledgeverse(storage_root=shared_root)
        continuity["shared_instance"] = True
        continuity["instance_ids"] = {
            "empty_mind": int(id(shared_kv)),
            "enriched": int(id(shared_kv)),
        }

        shared_start = _collect_default_galaxy_counts(shared_kv)
        if run_arc:
            arc_empty, arc_empty_metrics = _run_with_metrics(
                "global_arc_empty_mind",
                lambda: ARCAGI2Benchmark(
                    knowledgeverse=shared_kv,
                    max_tasks=max_arc_tasks,
                    enable_contrastive_learning=arc_enable_contrastive_learning,
                    enable_validity_gates=arc_enable_validity_gates,
                    enable_fuzzy_oracle=arc_enable_fuzzy_oracle,
                    fuzzy_oracle_threshold=arc_fuzzy_oracle_threshold,
                    enable_ptx_ranking=arc_enable_ptx_ranking,
                    enable_full_ptx=arc_enable_full_ptx,
                    ptx_validity_strictness=arc_ptx_validity_strictness,
                    runtime_seed_knowledge=benchmark_runtime_seeding,
                ).run_benchmark(use_enriched=False),
            )
        else:
            arc_empty = _arc_skipped_result(use_enriched=False)
            arc_empty_metrics = _skip_metrics("global_arc_empty_mind")
        if run_math:
            math_empty, math_empty_metrics = _run_with_metrics(
                "global_math_empty_mind",
                lambda: MathCompetitionBenchmark(
                    knowledgeverse=shared_kv,
                    max_problems=max_math_problems,
                    runtime_seed_knowledge=benchmark_runtime_seeding,
                ).run_benchmark(use_enriched=False),
            )
        else:
            math_empty = _math_skipped_result(use_enriched=False)
            math_empty_metrics = _skip_metrics("global_math_empty_mind")
        if run_lhe:
            lhe_empty, lhe_empty_metrics = _run_with_metrics(
                "global_lhe_empty_mind",
                lambda: LastHumanityExamBenchmark(
                    knowledgeverse=shared_kv,
                    max_questions=max_lhe_questions,
                    runtime_seed_knowledge=benchmark_runtime_seeding,
                ).run_benchmark(use_enriched=False),
            )
        else:
            lhe_empty = _lhe_skipped_result(use_enriched=False)
            lhe_empty_metrics = _skip_metrics("global_lhe_empty_mind")
        shared_after_empty = _collect_default_galaxy_counts(shared_kv)

        if run_arc:
            arc_enriched, arc_enriched_metrics = _run_with_metrics(
                "global_arc_enriched",
                lambda: ARCAGI2Benchmark(
                    knowledgeverse=shared_kv,
                    max_tasks=max_arc_tasks,
                    enable_contrastive_learning=arc_enable_contrastive_learning,
                    enable_validity_gates=arc_enable_validity_gates,
                    enable_fuzzy_oracle=arc_enable_fuzzy_oracle,
                    fuzzy_oracle_threshold=arc_fuzzy_oracle_threshold,
                    enable_ptx_ranking=arc_enable_ptx_ranking,
                    enable_full_ptx=arc_enable_full_ptx,
                    ptx_validity_strictness=arc_ptx_validity_strictness,
                    runtime_seed_knowledge=benchmark_runtime_seeding,
                ).run_benchmark(use_enriched=True),
            )
        else:
            arc_enriched = _arc_skipped_result(use_enriched=True)
            arc_enriched_metrics = _skip_metrics("global_arc_enriched")
        if run_math:
            math_enriched, math_enriched_metrics = _run_with_metrics(
                "global_math_enriched",
                lambda: MathCompetitionBenchmark(
                    knowledgeverse=shared_kv,
                    max_problems=max_math_problems,
                    runtime_seed_knowledge=benchmark_runtime_seeding,
                ).run_benchmark(use_enriched=True),
            )
        else:
            math_enriched = _math_skipped_result(use_enriched=True)
            math_enriched_metrics = _skip_metrics("global_math_enriched")
        if run_lhe:
            lhe_enriched, lhe_enriched_metrics = _run_with_metrics(
                "global_lhe_enriched",
                lambda: LastHumanityExamBenchmark(
                    knowledgeverse=shared_kv,
                    max_questions=max_lhe_questions,
                    runtime_seed_knowledge=benchmark_runtime_seeding,
                ).run_benchmark(use_enriched=True),
            )
        else:
            lhe_enriched = _lhe_skipped_result(use_enriched=True)
            lhe_enriched_metrics = _skip_metrics("global_lhe_enriched")
        shared_after_enriched = _collect_default_galaxy_counts(shared_kv)

        empty_kv = shared_kv
        enriched_kv = shared_kv
        empty_galaxy_counts_start = shared_start
        empty_galaxy_counts_end = shared_after_empty
        enriched_galaxy_counts_start = shared_after_empty
        enriched_galaxy_counts_end = shared_after_enriched
    else:
        empty_kv = Knowledgeverse(storage_root=empty_root)
        enriched_kv = Knowledgeverse(storage_root=enriched_root)
        continuity["instance_ids"] = {
            "empty_mind": int(id(empty_kv)),
            "enriched": int(id(enriched_kv)),
        }
        continuity["shared_instance"] = continuity["instance_ids"]["empty_mind"] == continuity["instance_ids"]["enriched"]

        empty_galaxy_counts_start = _collect_default_galaxy_counts(empty_kv)
        if run_arc:
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
                    enable_full_ptx=arc_enable_full_ptx,
                    ptx_validity_strictness=arc_ptx_validity_strictness,
                    runtime_seed_knowledge=benchmark_runtime_seeding,
                ).run_benchmark(use_enriched=False),
            )
        else:
            arc_empty = _arc_skipped_result(use_enriched=False)
            arc_empty_metrics = _skip_metrics("global_arc_empty_mind")
        if run_math:
            math_empty, math_empty_metrics = _run_with_metrics(
                "global_math_empty_mind",
                lambda: MathCompetitionBenchmark(
                    knowledgeverse=empty_kv,
                    max_problems=max_math_problems,
                    runtime_seed_knowledge=benchmark_runtime_seeding,
                ).run_benchmark(use_enriched=False),
            )
        else:
            math_empty = _math_skipped_result(use_enriched=False)
            math_empty_metrics = _skip_metrics("global_math_empty_mind")
        if run_lhe:
            lhe_empty, lhe_empty_metrics = _run_with_metrics(
                "global_lhe_empty_mind",
                lambda: LastHumanityExamBenchmark(
                    knowledgeverse=empty_kv,
                    max_questions=max_lhe_questions,
                    runtime_seed_knowledge=benchmark_runtime_seeding,
                ).run_benchmark(use_enriched=False),
            )
        else:
            lhe_empty = _lhe_skipped_result(use_enriched=False)
            lhe_empty_metrics = _skip_metrics("global_lhe_empty_mind")
        empty_galaxy_counts_end = _collect_default_galaxy_counts(empty_kv)

        enriched_galaxy_counts_start = _collect_default_galaxy_counts(enriched_kv)
        if run_arc:
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
                    enable_full_ptx=arc_enable_full_ptx,
                    ptx_validity_strictness=arc_ptx_validity_strictness,
                    runtime_seed_knowledge=benchmark_runtime_seeding,
                ).run_benchmark(use_enriched=True),
            )
        else:
            arc_enriched = _arc_skipped_result(use_enriched=True)
            arc_enriched_metrics = _skip_metrics("global_arc_enriched")
        if run_math:
            math_enriched, math_enriched_metrics = _run_with_metrics(
                "global_math_enriched",
                lambda: MathCompetitionBenchmark(
                    knowledgeverse=enriched_kv,
                    max_problems=max_math_problems,
                    runtime_seed_knowledge=benchmark_runtime_seeding,
                ).run_benchmark(use_enriched=True),
            )
        else:
            math_enriched = _math_skipped_result(use_enriched=True)
            math_enriched_metrics = _skip_metrics("global_math_enriched")
        if run_lhe:
            lhe_enriched, lhe_enriched_metrics = _run_with_metrics(
                "global_lhe_enriched",
                lambda: LastHumanityExamBenchmark(
                    knowledgeverse=enriched_kv,
                    max_questions=max_lhe_questions,
                    runtime_seed_knowledge=benchmark_runtime_seeding,
                ).run_benchmark(use_enriched=True),
            )
        else:
            lhe_enriched = _lhe_skipped_result(use_enriched=True)
            lhe_enriched_metrics = _skip_metrics("global_lhe_enriched")
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
        "model_persistence_mode": str(model_persistence_mode),
        "persistence": continuity,
        "runtime_seed_knowledge": bool(benchmark_runtime_seeding),
        "arc_enable_ptx_ranking": bool(arc_enable_ptx_ranking),
        "arc_enable_full_ptx": bool(arc_enable_full_ptx),
        "arc_ptx_validity_strictness": str(arc_ptx_validity_strictness),
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
        "--model-persistence-mode",
        default="unified",
        choices=["unified", "dual"],
        help=(
            "Persistence strategy. "
            "'unified' reuses one evolving Knowledgeverse instance across the full run; "
            "'dual' keeps separate empty/enriched worlds for diagnostics."
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
        "--unified-storage-root",
        type=Path,
        default=None,
        help="Optional explicit storage root for unified mode (defaults to galaxies_enriched).",
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
        "--arc-embedding-lazy-mode",
        default="skip",
        choices=["compute", "skip", "fail"],
        help=(
            "Policy for missing ARC embeddings in hot path: "
            "'skip' (default, no lazy compute), 'fail' (strict fail-fast), 'compute' (legacy)."
        ),
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
    if args.arc_enable_full_ptx and not args.arc_enable_ptx_ranking:
        args.arc_enable_ptx_ranking = True
    os.environ["K3D_ARC_EMBEDDING_LAZY_MODE"] = str(args.arc_embedding_lazy_mode)
    os.environ.setdefault("K3D_REQUIRE_PTX_ARC_PIPELINE", "true")
    if args.arc_enable_full_ptx:
        os.environ.setdefault("K3D_ALLOW_LEGACY_ARC_PIPELINE", "false")

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
        arc_enable_full_ptx=args.arc_enable_full_ptx,
        arc_ptx_validity_strictness=args.arc_ptx_validity_strictness,
        benchmark_runtime_seeding=args.benchmark_runtime_seeding,
        model_persistence_mode=args.model_persistence_mode,
        unified_storage_root=args.unified_storage_root,
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
    runtime_usage["arc_embedding_lazy_mode"] = str(args.arc_embedding_lazy_mode)

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
    sovereignty_checks: dict[str, Any] = {}
    for benchmark_name in ("arc_agi_2", "math_competitions", "last_humanity_exam"):
        phases = summary.get("integrated_results", {}).get(benchmark_name, {})
        for phase in ("empty_mind", "enriched"):
            payload = phases.get(phase, {})
            if isinstance(payload, dict):
                key = f"{benchmark_name}.{phase}"
                sovereignty_checks[key] = _summarize_payload_sovereignty(
                    benchmark_name=benchmark_name,
                    phase=phase,
                    payload=payload,
                )
    for proxy_name in ("gsm8k_proxy", "mmlu_proxy"):
        payload = summary.get("proxy_results", {}).get(proxy_name)
        if isinstance(payload, dict) and payload.get("available"):
            key = f"{proxy_name}.proxy"
            sovereignty_checks[key] = _summarize_payload_sovereignty(
                benchmark_name=proxy_name,
                phase="proxy",
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
    summary.setdefault("runtime_usage", {})["sovereignty"] = {
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
            "Sovereignty violation detected in global benchmark runner. "
            f"compliance={compliance_total:.1%}; samples={'; '.join(samples)}"
        )
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
    print(
        "  Persistence mode: "
        f"{runtime_usage.get('model_persistence_mode')} "
        f"(shared_instance={runtime_usage.get('persistence', {}).get('shared_instance')})"
    )
    sovereignty = summary.get("runtime_usage", {}).get("sovereignty", {})
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
    print(f"  ARC embedding lazy mode: {args.arc_embedding_lazy_mode}")
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
