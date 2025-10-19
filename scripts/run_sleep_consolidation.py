#!/usr/bin/env python3
"""
Run sleep-time consolidation for RPN embeddings.

Usage
-----
PYTHONPATH=. ./scripts/run_sleep_consolidation.py \
    --embeddings /K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl \
    --output /K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl \
    [--metrics /K3D/Knowledge3D.local/logs/sleep_metrics.jsonl]

Notes
-----
Run inside the GPU-oriented `k3d-cranium` environment as described in
`envs/k3d-cranium.yml` so CUDA libraries are available for downstream phases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.sleep_time_consolidator import SleepTimeConsolidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sleep-time RPN consolidation")
    parser.add_argument(
        "--embeddings",
        required=True,
        type=Path,
        help="Path to existing RPN embedding pickle",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination path for consolidated embeddings (defaults to --embeddings)",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Optional JSONL file for logging consolidation metrics",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=256,
        help="Number of clusters for refinement (default: 256)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.2,
        help="Learning rate toward cluster centroids (default: 0.2)",
    )
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=0.95,
        help="Cosine similarity for redundancy pruning (default: 0.95)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.embeddings

    engine = RPNEmbeddingEngine()
    engine.load_embeddings(args.embeddings)

    consolidator = SleepTimeConsolidator(
        engine,
        cluster_count=args.clusters,
        learning_rate=args.learning_rate,
        redundancy_threshold=args.redundancy_threshold,
        metrics_path=args.metrics,
    )

    result = consolidator.consolidate()
    engine.save_embeddings(output_path)

    print(json.dumps(result, indent=2, default=float))
    print(f"[SLEEP] Consolidated embeddings saved to {output_path}")


if __name__ == "__main__":
    main()
