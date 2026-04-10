#!/usr/bin/env python3
"""Generate ARC-AGI-2 submission artifacts through the canonical ARC local runner."""

from __future__ import annotations

import argparse

from benchmarks.arc2_local_runner import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        default="/K3D/K3D_llama_cpp/datasets/ARC-AGI-master/data/evaluation",
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--submission-output", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_evaluation(
        args.dataset_path,
        max_tasks=int(args.max_tasks) if args.max_tasks is not None else 1000000,
        submission_output=args.submission_output,
        summary_output=args.summary_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
