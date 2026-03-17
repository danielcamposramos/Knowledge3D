#!/usr/bin/env python3
"""Batch GLB -> meaning-star proceduralization for House and Workshop assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge3d.tools.training_pipelines.glb_decomposer import decompose_glb_to_stars


ASSET_SOURCES = [
    (Path("/K3D/Knowledge3D.local/datasets/gltf_samples/"), "Workshop/Assets"),
    (Path("/K3D/Knowledge3D.local/datasets/gltf_curated/library/"), "House/Library/Furniture"),
    (Path("/K3D/Knowledge3D.local/datasets/gltf_curated/office/"), "House/Workshop/Furniture"),
    (Path("/K3D/Knowledge3D.local/datasets/gltf_curated/workshop/"), "House/Workshop/Tools"),
    (Path("/K3D/Knowledge3D.local/datasets/gltf_house/"), "House/Furniture"),
]


def proceduralize_assets(*, output: Path, language: str = "en") -> int:
    stars = []
    for source_root, domain in ASSET_SOURCES:
        if not source_root.exists():
            continue
        for glb_path in sorted(source_root.rglob("*.glb")):
            stars.extend(decompose_glb_to_stars(glb_path, domain=domain, language=language))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for star in stars:
            handle.write(json.dumps(star.to_dict(), ensure_ascii=True, separators=(",", ":")) + "\n")
    return len(stars)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch proceduralize GLB assets into meaning-centric stars.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/galaxies/House_Assets.jsonl"),
        help="Output JSONL path for emitted meaning stars.",
    )
    parser.add_argument("--language", default="en", help="Surface-form language tag for emitted stars.")
    args = parser.parse_args(argv)
    count = proceduralize_assets(output=args.output, language=args.language)
    print(f"proceduralized {count} stars -> {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
