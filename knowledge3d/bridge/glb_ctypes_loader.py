"""
Lightweight GLB loader/saver for K3D stars using pygltflib.

This is a bridge to move from JSONL/ProceduralGalaxy fallback toward a real glTF
pipeline. Geometry is omitted (procedural-first); nodes carry extras.k3d.

If pygltflib is not installed, functions will raise ImportError with guidance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

try:
    from pygltflib import GLTF2, Scene, Node
except ImportError:  # pragma: no cover - dependency optional
    GLTF2 = None  # type: ignore
    Scene = None  # type: ignore
    Node = None  # type: ignore


def _require_pygltflib():
    if GLTF2 is None:
        raise ImportError("pygltflib is required for GLB export/import. Install via pip.")


def save_stars_to_glb(stars: List[Dict], output: Path) -> Path:
    """
    Save a list of star dicts into a .glb with extras.k3d only (no geometry).
    """
    _require_pygltflib()
    gltf = GLTF2()
    gltf.scenes = [Scene(nodes=list(range(len(stars))))]
    gltf.scene = 0
    gltf.nodes = []
    for idx, star in enumerate(stars):
        name = star.get("id") or star.get("letter_concept") or star.get("meaning_id") or star.get("symbol_concept") or star.get("phrase_id") or f"star_{idx}"
        node = Node(name=name, extras={"k3d": star})
        gltf.nodes.append(node)
    output.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(output.as_posix())
    return output


def load_stars_from_glb(path: Path) -> List[Dict]:
    """
    Load stars from a .glb produced by save_stars_to_glb.
    """
    _require_pygltflib()
    gltf = GLTF2().load(path.as_posix())
    stars: List[Dict] = []
    for node in gltf.nodes or []:
        if node.extras and "k3d" in node.extras:
            stars.append(node.extras["k3d"])
    return stars
