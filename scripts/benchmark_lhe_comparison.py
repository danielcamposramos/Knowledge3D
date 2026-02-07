#!/usr/bin/env python3
"""Run Last Humanity Exam comparison: empty mind vs enriched."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.last_humanity_exam import LastHumanityExamBenchmark
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=None, help="LHE dataset directory")
    parser.add_argument("--max-questions", type=int, default=100, help="Max questions")
    parser.add_argument(
        "--output-dir",
        default="../Knowledge3D.local/results/week14",
        help="Directory to write result JSON files",
    )
    parser.add_argument(
        "--storage-root",
        default="../Knowledge3D.local",
        help="Knowledgeverse storage root",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    empty_kv = Knowledgeverse(storage_root=args.storage_root)
    empty_bench = LastHumanityExamBenchmark(
        knowledgeverse=empty_kv,
        dataset_path=args.dataset_path,
        max_questions=args.max_questions,
    )
    empty_result = empty_bench.run_benchmark(use_enriched=False)
    (output_dir / "last_humanity_exam_empty_mind.json").write_text(
        json.dumps(empty_result, indent=2),
        encoding="utf-8",
    )

    enriched_kv = Knowledgeverse(storage_root=args.storage_root)
    enriched_bench = LastHumanityExamBenchmark(
        knowledgeverse=enriched_kv,
        dataset_path=args.dataset_path,
        max_questions=args.max_questions,
    )
    enriched_result = enriched_bench.run_benchmark(use_enriched=True)
    (output_dir / "last_humanity_exam_enriched.json").write_text(
        json.dumps(enriched_result, indent=2),
        encoding="utf-8",
    )

    improvement = enriched_result["accuracy"] - empty_result["accuracy"]
    print("Last Humanity Exam comparison")
    print(
        f"  Empty mind: {empty_result['accuracy']:.2%} "
        f"({empty_result['correct']}/{empty_result['total_questions']})"
    )
    print(
        f"  Enriched:   {enriched_result['accuracy']:.2%} "
        f"({enriched_result['correct']}/{enriched_result['total_questions']})"
    )
    print(f"  Improvement: {improvement:+.2%}")

    for domain in sorted(enriched_result["results_by_domain"].keys()):
        empty_bucket = empty_result["results_by_domain"].get(domain, {"accuracy": 0.0})
        enriched_bucket = enriched_result["results_by_domain"][domain]
        print(
            f"  {domain}: {empty_bucket['accuracy']:.2%} -> "
            f"{enriched_bucket['accuracy']:.2%}"
        )


if __name__ == "__main__":
    main()
