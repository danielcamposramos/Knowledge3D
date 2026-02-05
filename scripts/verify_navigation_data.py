#!/usr/bin/env python3
"""
Smoke test for NavigationDataset binary layout and rule registry decoding.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

from knowledge3d.training.math_benchmarks.navigation_dataset import NavigationDataset


def _decode_rules(rule_ids: List[int], registry: List[str]) -> List[str]:
    decoded = []
    for rule_id in rule_ids:
        if 0 <= rule_id < len(registry):
            decoded.append(registry[rule_id])
        else:
            decoded.append(f"unknown_{rule_id}")
    return decoded


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify NavigationDataset binary layout.")
    parser.add_argument(
        "--bin",
        type=str,
        default="/tmp/log_galaxy_microbench.bin",
        help="Path to the Log Galaxy binary file.",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="/tmp/log_galaxy_microbench.json",
        help="Path to the Log Galaxy metadata JSON.",
    )
    parser.add_argument("--samples", type=int, default=3, help="Random samples to print.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = parser.parse_args()

    meta_path = Path(args.meta)
    if not meta_path.exists():
        raise SystemExit(f"Metadata file not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    registry = meta.get("rule_registry", [])
    embedding_dim = int(meta["counts"]["embedding_dim"])

    dataset = NavigationDataset(bin_path=args.bin, meta_path=args.meta)
    try:
        total = len(dataset)
        if total == 0:
            raise SystemExit("NavigationDataset is empty.")

        rng = random.Random(args.seed)
        indices = [rng.randrange(0, total) for _ in range(min(args.samples, total))]
        print(f"[Verify] samples={len(indices)} total={total} embedding_dim={embedding_dim}")

        for idx in indices:
            embed, rule_ids = dataset[idx]
            if int(embed.numel()) != embedding_dim:
                raise SystemExit(
                    f"Embedding dim mismatch at idx={idx}: "
                    f"got {embed.numel()} expected {embedding_dim}"
                )
            rule_id_list = [int(v) for v in rule_ids.tolist()]
            decoded = _decode_rules(rule_id_list, registry)
            print(f"\n[Sample {idx}]")
            print(f"  rule_ids: {rule_id_list}")
            print(f"  rules: {decoded}")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
