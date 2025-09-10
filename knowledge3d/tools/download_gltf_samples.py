from __future__ import annotations

"""
Download a curated set of glTF Sample Models (Khronos) into a local folder.

This provides a quick, open set of GLB assets to bootstrap 3D ingestion tests.

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.tools.download_gltf_samples \
    --out ../Knowledge3D.local/datasets/gltf_samples
"""

import argparse
import os
from pathlib import Path
from typing import List

SAMPLES: List[str] = [
    "Box",
    "BoxTextured",
    "Duck",
    "DamagedHelmet",
    "Avocado",
    "BoomBox",
    "CesiumMan",
    "CesiumMilkTruck",
    "WaterBottle",
    "BrainStem",
    "Buggy",
    "2CylinderEngine",
    "MetalRoughSpheres",
]


def url_for(name: str) -> str:
    base = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0"
    return f"{base}/{name}/glTF-Binary/{name}.glb"


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Download glTF sample GLBs")
    ap.add_argument("--out", default="../Knowledge3D.local/datasets/gltf_samples")
    args = ap.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    for name in SAMPLES:
        u = url_for(name)
        out = root / f"{name}.glb"
        if out.exists() and out.stat().st_size > 0:
            print(f"exists: {out}")
            continue
        print(f"downloading: {name} -> {out}")
        code = os.system(f"curl -fsSL '{u}' -o '{out}'")
        if code != 0 or (not out.exists()) or out.stat().st_size == 0:
            print(f"failed: {name} ({u})")
            continue
    print(str(root))


if __name__ == "__main__":
    main()
