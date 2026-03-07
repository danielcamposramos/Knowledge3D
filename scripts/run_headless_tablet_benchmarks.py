#!/usr/bin/env python3
"""Run benchmark smoke/eval passes through the headless Tablet boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.arc_agi_2 import ARCAGI2Benchmark
from benchmarks.last_humanity_exam import LastHumanityExamBenchmark
from benchmarks.math_competitions import MathCompetitionBenchmark
from knowledge3d.bridge.headless_tablet import CommandHandler, HeadlessTabletMPC
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def _skip_summary(reason: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": reason,
    }


def run_tablet_benchmark_suite(
    args: argparse.Namespace,
    *,
    command_handler: CommandHandler | None = None,
) -> dict[str, Any]:
    storage_root = Path(args.storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)

    kv = Knowledgeverse(storage_root=storage_root)
    tablet = HeadlessTabletMPC(
        knowledgeverse=kv,
        storage_root=storage_root,
        command_handler=command_handler,
    )

    if int(args.max_arc_tasks) > 0:
        arc = ARCAGI2Benchmark(
            knowledgeverse=kv,
            dataset_path=args.arc_dataset_path,
            max_tasks=args.max_arc_tasks,
            tablet_boundary=tablet,
        ).run_benchmark(use_enriched=bool(args.use_enriched))
    else:
        arc = _skip_summary("max_arc_tasks<=0")

    if int(args.max_math_problems) > 0:
        math = MathCompetitionBenchmark(
            knowledgeverse=kv,
            dataset_path=args.math_dataset_path,
            max_problems=args.max_math_problems,
            tablet_boundary=tablet,
        ).run_benchmark(use_enriched=bool(args.use_enriched))
    else:
        math = _skip_summary("max_math_problems<=0")

    if int(args.max_lhe_questions) > 0:
        lhe = LastHumanityExamBenchmark(
            knowledgeverse=kv,
            dataset_path=args.lhe_dataset_path,
            max_questions=args.max_lhe_questions,
            tablet_boundary=tablet,
        ).run_benchmark(use_enriched=bool(args.use_enriched))
    else:
        lhe = _skip_summary("max_lhe_questions<=0")

    return {
        "mode": "headless_tablet_boundary",
        "storage_root": str(storage_root),
        "use_enriched": bool(args.use_enriched),
        "benchmarks": {
            "arc_agi_2": arc,
            "math_competitions": math,
            "last_humanity_exam": lhe,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", default="../Knowledge3D.local/tablet_benchmark_runs")
    parser.add_argument("--arc-dataset-path", default=None)
    parser.add_argument("--math-dataset-path", default=None)
    parser.add_argument("--lhe-dataset-path", default=None)
    parser.add_argument("--max-arc-tasks", type=int, default=1)
    parser.add_argument("--max-math-problems", type=int, default=1)
    parser.add_argument("--max-lhe-questions", type=int, default=1)
    parser.add_argument("--use-enriched", action="store_true", default=False)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    summary = run_tablet_benchmark_suite(args)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
