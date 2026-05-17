#!/usr/bin/env python3
"""Run math benchmark comparison: empty mind vs enriched."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.math_competitions import MathCompetitionBenchmark
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=None, help="Math competition dataset directory")
    parser.add_argument("--max-problems", type=int, default=100, help="Max problems")
    parser.add_argument(
        "--output-dir",
        default="/K3D/Knowledge3D.local/results/week14",
        help="Directory to write result JSON files",
    )
    parser.add_argument(
        "--storage-root",
        default="/K3D/Knowledge3D.local",
        help="Knowledgeverse storage root",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    empty_kv = Knowledgeverse(storage_root=args.storage_root)
    empty_bench = MathCompetitionBenchmark(
        knowledgeverse=empty_kv,
        dataset_path=args.dataset_path,
        max_problems=args.max_problems,
    )
    empty_result = empty_bench.run_benchmark(use_enriched=False)
    (output_dir / "math_competitions_empty_mind.json").write_text(
        json.dumps(empty_result, indent=2),
        encoding="utf-8",
    )

    enriched_kv = Knowledgeverse(storage_root=args.storage_root)
    enriched_bench = MathCompetitionBenchmark(
        knowledgeverse=enriched_kv,
        dataset_path=args.dataset_path,
        max_problems=args.max_problems,
    )
    enriched_result = enriched_bench.run_benchmark(use_enriched=True)
    (output_dir / "math_competitions_enriched.json").write_text(
        json.dumps(enriched_result, indent=2),
        encoding="utf-8",
    )

    improvement = enriched_result["overall_accuracy"] - empty_result["overall_accuracy"]
    print("Math competitions comparison")
    print(
        f"  Empty mind: {empty_result['overall_accuracy']:.2%} "
        f"({empty_result['correct']}/{empty_result['total']})"
    )
    print(
        f"  Enriched:   {enriched_result['overall_accuracy']:.2%} "
        f"({enriched_result['correct']}/{enriched_result['total']})"
    )
    print(f"  Improvement: {improvement:+.2%}")

    for comp in sorted(enriched_result["results_by_competition"].keys()):
        empty_bucket = empty_result["results_by_competition"].get(comp, {"accuracy": 0.0})
        enriched_bucket = enriched_result["results_by_competition"][comp]
        print(
            f"  {comp}: {empty_bucket['accuracy']:.2%} -> "
            f"{enriched_bucket['accuracy']:.2%}"
        )


if __name__ == "__main__":
    main()
