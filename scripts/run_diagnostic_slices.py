#!/usr/bin/env python3
"""Run sovereign diagnostic slices across GSM8K, MMLU, ARC-1, LHE, and optionally ARC-2."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.arc_agi_2 import ARCAGI2Benchmark
from benchmarks.gsm8k import GSM8KBenchmark
from benchmarks.last_humanity_exam import LastHumanityExamBenchmark
from benchmarks.mmlu import MMLUBenchmark
from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC
from knowledge3d.daemon.main import DaemonConfig, K3DDaemon
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def _log(message: str) -> None:
    print(f"[DIAGNOSTIC] {message}", file=sys.stderr, flush=True)


def _failure_id(result: dict[str, Any]) -> str:
    for key in ("task_id", "question_id", "problem_id", "id"):
        value = str(result.get(key, "")).strip()
        if value:
            return value
    return "unknown"


def _failure_type(result: dict[str, Any]) -> str:
    for key in ("subject", "domain", "competition", "question_type"):
        value = str(result.get(key, "")).strip()
        if value:
            return value
    return "unknown"


def _slice_summary(name: str, results: list[dict[str, Any]], accuracy: float) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for row in results if bool(row.get("correct")))
    gpu_true = sum(1 for row in results if bool(row.get("gpu_execution")))
    failure_types = Counter(_failure_type(row) for row in results if not bool(row.get("correct")))
    failure_programs = Counter(str(row.get("program_id", "")).strip() or "unknown" for row in results if not bool(row.get("correct")))
    failure_examples: list[dict[str, Any]] = []
    for row in results:
        if bool(row.get("correct")):
            continue
        failure_examples.append(
            {
                "id": _failure_id(row),
                "type": _failure_type(row),
                "predicted": row.get("predicted_answer", row.get("predicted")),
                "expected": row.get("correct_answer", row.get("expected_answer", row.get("expected"))),
                "program_id": row.get("program_id"),
                "route": row.get("route"),
            }
        )
        if len(failure_examples) >= 5:
            break
    return {
        "benchmark": name,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "gpu_execution_true": gpu_true,
        "gpu_execution_all": gpu_true == total if total else False,
        "failure_type_counts": dict(failure_types.most_common()),
        "failure_program_counts": dict(failure_programs.most_common()),
        "failure_examples": failure_examples,
    }


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    storage_root = Path(args.storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)
    kv = Knowledgeverse(storage_root=storage_root)
    daemon = K3DDaemon(DaemonConfig(storage_root=storage_root), knowledgeverse=kv)
    tablet = HeadlessTabletMPC(command_handler=daemon.handle_command, storage_root=storage_root)

    _log("starting GSM8K slice")
    gsm8k = GSM8KBenchmark(
        knowledgeverse=kv,
        dataset_path=args.gsm8k_dataset_path,
        max_questions=args.max_math_questions,
    ).run_benchmark(use_enriched=True)
    _log("completed GSM8K slice")
    kv.reset_query_session()

    _log("starting MMLU slice")
    mmlu = MMLUBenchmark(
        knowledgeverse=kv,
        dataset_path=args.mmlu_dataset_path,
        max_questions=args.max_mmlu_questions if args.max_mmlu_questions > 0 else None,
        subjects=args.mmlu_subjects,
    ).run_benchmark(use_enriched=True)
    _log("completed MMLU slice")
    kv.reset_query_session()

    _log("starting ARC-1 slice")
    arc1 = ARCAGI2Benchmark(
        knowledgeverse=kv,
        dataset_path=args.arc1_dataset_path,
        max_tasks=args.max_arc1_tasks,
    ).run_benchmark(use_enriched=True)
    _log("completed ARC-1 slice")
    kv.reset_query_session()

    _log("starting LHE slice")
    lhe = LastHumanityExamBenchmark(
        knowledgeverse=kv,
        dataset_path=args.lhe_dataset_path,
        max_questions=args.max_lhe_questions,
        tablet_boundary=tablet,
    ).run_benchmark(use_enriched=True)
    _log("completed LHE slice")
    kv.reset_query_session()

    arc2_summary: dict[str, Any] | None = None
    if args.max_arc2_tasks > 0:
        _log("starting ARC-2 slice")
        arc2 = ARCAGI2Benchmark(
            knowledgeverse=kv,
            dataset_path=args.arc2_dataset_path,
            dataset_version="arc_agi_2_main",
            max_tasks=args.max_arc2_tasks,
        ).run_benchmark(use_enriched=True)
        _log("completed ARC-2 slice")
        arc2_summary = _slice_summary("ARC-2", arc2.get("results", []), float(arc2.get("accuracy", 0.0)))
        arc2_summary["dataset_path"] = arc2.get("dataset_path")

    summary = {
        "storage_root": str(storage_root),
        "gsm8k": _slice_summary("GSM8K", gsm8k.get("results", []), float(gsm8k.get("accuracy", 0.0))),
        "mmlu": _slice_summary("MMLU", mmlu.get("results", []), float(mmlu.get("accuracy", 0.0))),
        "arc_1": _slice_summary("ARC-1", arc1.get("results", []), float(arc1.get("accuracy", 0.0))),
        "lhe": _slice_summary("LHE", lhe.get("results", []), float(lhe.get("accuracy", 0.0))),
        "raw": {
            "gsm8k": gsm8k,
            "mmlu": mmlu,
            "arc_1": arc1,
            "lhe": lhe,
        },
    }
    if arc2_summary is not None:
        summary["arc_2"] = arc2_summary
        summary["raw"]["arc_2"] = arc2
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", default="/K3D/Knowledge3D.local/logs/diagnostic_slices")
    parser.add_argument("--gsm8k-dataset-path", default=None)
    parser.add_argument("--mmlu-dataset-path", default=None)
    parser.add_argument("--arc1-dataset-path", default=None)
    parser.add_argument("--arc2-dataset-path", default=None)
    parser.add_argument("--lhe-dataset-path", default=None)
    parser.add_argument("--max-gsm8k-questions", type=int, default=50)
    parser.add_argument("--max-mmlu-questions", type=int, default=0)
    parser.add_argument("--mmlu-subjects", default="elementary_mathematics,college_physics")
    parser.add_argument("--max-arc1-tasks", type=int, default=50)
    parser.add_argument("--max-arc2-tasks", type=int, default=0)
    parser.add_argument("--max-lhe-questions", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    summary = run_diagnostics(args)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
