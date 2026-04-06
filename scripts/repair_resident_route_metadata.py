#!/usr/bin/env python3
"""Repair resident route metadata and persist a fresh consolidated checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge3d.local_paths import resolve_storage_root
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.resident_route_metadata import (
    repair_knowledgeverse_resident_route_metadata,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair resident sovereign route metadata.")
    parser.add_argument("--storage-root", type=Path, default=None)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    storage_root = resolve_storage_root(args.storage_root)
    report_path = Path(args.report) if args.report is not None else (storage_root / "results" / "route_metadata_repair_report.json")

    kv = Knowledgeverse(
        storage_root=storage_root,
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    repair_summary = repair_knowledgeverse_resident_route_metadata(kv, persist_to_disk=True)
    checkpoint_summary = kv.save_consolidated_state()
    payload = {
        "status": "ok",
        "repair": repair_summary,
        "checkpoint": checkpoint_summary,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
