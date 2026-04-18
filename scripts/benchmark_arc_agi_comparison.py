#!/usr/bin/env python3
"""Run ARC-AGI benchmark comparison: empty mind vs enriched."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.arc_agi_2 import ARCAGI2Benchmark
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=None, help="ARC evaluation directory")
    parser.add_argument(
        "--dataset-version",
        default="arc_agi_2",
        choices=["arc_agi_2", "arc_agi_3"],
        help="ARC benchmark dataset version to target",
    )
    parser.add_argument("--max-tasks", type=int, default=100, help="Max ARC tasks to evaluate")
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
    empty_bench = ARCAGI2Benchmark(
        knowledgeverse=empty_kv,
        dataset_path=args.dataset_path,
        max_tasks=args.max_tasks,
        dataset_version=args.dataset_version,
    )
    empty_result = empty_bench.run_benchmark(use_enriched=False)
    (output_dir / "arc_agi_2_empty_mind.json").write_text(
        json.dumps(empty_result, indent=2),
        encoding="utf-8",
    )

    enriched_kv = Knowledgeverse(storage_root=args.storage_root)
    enriched_bench = ARCAGI2Benchmark(
        knowledgeverse=enriched_kv,
        dataset_path=args.dataset_path,
        max_tasks=args.max_tasks,
        dataset_version=args.dataset_version,
    )
    enriched_result = enriched_bench.run_benchmark(use_enriched=True)
    (output_dir / "arc_agi_2_enriched.json").write_text(
        json.dumps(enriched_result, indent=2),
        encoding="utf-8",
    )

    improvement = enriched_result["accuracy"] - empty_result["accuracy"]
    print("ARC-AGI 2 comparison")
    print(f"  Empty mind: {empty_result['accuracy']:.2%} ({empty_result['correct']}/{empty_result['total_tasks']})")
    print(f"  Enriched:   {enriched_result['accuracy']:.2%} ({enriched_result['correct']}/{enriched_result['total_tasks']})")
    print(f"  Improvement: {improvement:+.2%}")


if __name__ == "__main__":
    main()
