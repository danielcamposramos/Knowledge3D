#!/usr/bin/env python3
"""
Build drawing grammar JSONL (and optional GLB) using drawing_grammar_builder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from knowledge3d.ingestion.atomic.drawing_grammar_builder import main as build_main
from knowledge3d.ingestion.upsert_bridge import GalaxyUpsertBridge
from knowledge3d.bridge.glb_ctypes_loader import GLTF2  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build drawing grammar JSONL/GLB.")
    ap.add_argument("--output", type=Path, default=Path("/K3D/Knowledge3D.local/galaxy_pending/drawing_grammar.jsonl"))
    ap.add_argument("--export-gltf", action="store_true", help="Also export GLB.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # reuse builder main to produce JSONL
    build_main()
    output = args.output
    # bridge export if needed
    if args.export_gltf:
        # load from JSONL produced by builder
        stars = []
        with output.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                import json
                stars.append(json.loads(line))
        bridge = GalaxyUpsertBridge(output_dir=output.parent)
        bridge.export_gltf("drawing_grammar", stars)
        print(f"Exported drawing grammar GLB at {output.parent/'drawing_grammar.glb'}")


if __name__ == "__main__":
    main()
