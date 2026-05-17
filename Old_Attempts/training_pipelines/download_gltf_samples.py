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
import json
from urllib.request import urlopen
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


def _download(url: str, out: Path) -> bool:
    code = os.system(f"curl -fsSL '{url}' -o '{out}'")
    return code == 0 and out.exists() and out.stat().st_size > 0


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Download glTF sample GLBs")
    ap.add_argument("--out", default="../Knowledge3D.local/datasets/gltf_samples")
    ap.add_argument("--all", action="store_true", help="Download all models from model-index.json")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of models when using --all (0 = no limit)")
    args = ap.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    if args.all:
        try:
            idx_url = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/model-index.json"
            with urlopen(idx_url, timeout=60) as fh:
                data = json.load(fh)
            models = [m.get("name") for m in data if isinstance(m, dict) and m.get("name")]
            count = 0
            for name in models:
                out = root / f"{name}.glb"
                if out.exists() and out.stat().st_size > 0:
                    print(f"exists: {out}")
                    continue
                u = url_for(name)
                print(f"downloading: {name} -> {out}")
                ok = _download(u, out)
                if not ok:
                    print(f"failed: {name} ({u})")
                count += 1
                if args.limit and count >= args.limit:
                    break
        except Exception as e:
            raise SystemExit(f"Failed to download model-index: {e}")
    else:
        for name in SAMPLES:
            u = url_for(name)
            out = root / f"{name}.glb"
            if out.exists() and out.stat().st_size > 0:
                print(f"exists: {out}")
                continue
            print(f"downloading: {name} -> {out}")
            ok = _download(u, out)
            if not ok:
                print(f"failed: {name} ({u})")
    print(str(root))


if __name__ == "__main__":
    main()
