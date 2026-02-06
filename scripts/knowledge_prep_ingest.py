#!/usr/bin/env python3
"""Run Knowledge Preparation ingestion batches."""

from __future__ import annotations

import argparse
import asyncio
import json

from knowledge3d.ingestion.batch_orchestrator import BatchOrchestrator
from knowledge3d.ingestion.corpus_manifest import CorpusManifest, CorpusTier
from knowledge3d.knowledgeverse.stargate import IngestionStargate


async def _run(tier: int | None, max_parallel: int, manifest_defaults: bool) -> dict:
    manifest = CorpusManifest(load_defaults=manifest_defaults)
    stargate = IngestionStargate(manifest_version="kv-2026-02-06")
    orchestrator = BatchOrchestrator(manifest=manifest, stargate=stargate)
    if tier is None:
        return await orchestrator.ingest_all(max_parallel=max_parallel)
    return {
        "stats_before": manifest.get_stats(),
        "results": {
            f"tier_{tier}": await orchestrator.ingest_tier(
                tier=CorpusTier(int(tier)),
                max_parallel=max_parallel,
            )
        },
        "stats_after": manifest.get_stats(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge preparation ingestion runner")
    parser.add_argument(
        "--tier",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help="Run only one tier (1/2/3). If omitted, runs all tiers.",
    )
    parser.add_argument("--max-parallel", type=int, default=4, help="Parallel ingestion workers")
    parser.add_argument(
        "--no-default-manifest",
        action="store_true",
        help="Skip default corpus entries (for custom/testing harnesses)",
    )
    args = parser.parse_args()

    payload = asyncio.run(
        _run(
            tier=args.tier,
            max_parallel=max(1, args.max_parallel),
            manifest_defaults=not args.no_default_manifest,
        )
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
