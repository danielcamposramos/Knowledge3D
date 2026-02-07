#!/usr/bin/env python3
"""Run all Week 14 benchmarks and write a unified report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.arc_agi_2 import ARCAGI2Benchmark
from benchmarks.last_humanity_exam import LastHumanityExamBenchmark
from benchmarks.math_competitions import MathCompetitionBenchmark
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


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
    parser.add_argument("--max-math-problems", type=int, default=100, help="Math problem limit")
    parser.add_argument("--max-lhe-questions", type=int, default=100, help="LHE question limit")
    parser.add_argument(
        "--output-dir",
        default="../Knowledge3D.local/results/week14",
        help="Directory for benchmark outputs",
    )
    parser.add_argument(
        "--storage-root",
        default="../Knowledge3D.local",
        help="Knowledgeverse storage root",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Empty mind runs
    arc_empty = ARCAGI2Benchmark(
        knowledgeverse=Knowledgeverse(storage_root=args.storage_root),
        dataset_path=args.arc_dataset_path,
        max_tasks=args.max_arc_tasks,
        dataset_version=args.arc_dataset_version,
    ).run_benchmark(use_enriched=False)
    math_empty = MathCompetitionBenchmark(
        knowledgeverse=Knowledgeverse(storage_root=args.storage_root),
        dataset_path=args.math_dataset_path,
        max_problems=args.max_math_problems,
    ).run_benchmark(use_enriched=False)
    lhe_empty = LastHumanityExamBenchmark(
        knowledgeverse=Knowledgeverse(storage_root=args.storage_root),
        dataset_path=args.lhe_dataset_path,
        max_questions=args.max_lhe_questions,
    ).run_benchmark(use_enriched=False)

    # Enriched runs
    arc_enriched = ARCAGI2Benchmark(
        knowledgeverse=Knowledgeverse(storage_root=args.storage_root),
        dataset_path=args.arc_dataset_path,
        max_tasks=args.max_arc_tasks,
        dataset_version=args.arc_dataset_version,
    ).run_benchmark(use_enriched=True)
    math_enriched = MathCompetitionBenchmark(
        knowledgeverse=Knowledgeverse(storage_root=args.storage_root),
        dataset_path=args.math_dataset_path,
        max_problems=args.max_math_problems,
    ).run_benchmark(use_enriched=True)
    lhe_enriched = LastHumanityExamBenchmark(
        knowledgeverse=Knowledgeverse(storage_root=args.storage_root),
        dataset_path=args.lhe_dataset_path,
        max_questions=args.max_lhe_questions,
    ).run_benchmark(use_enriched=True)

    summary = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "benchmarks": {
            "arc_agi_2": {
                "empty_mind": arc_empty,
                "enriched": arc_enriched,
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
    }
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

    print("Knowledge3D Week 14 benchmark suite")
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
    print(f"  Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
