from __future__ import annotations

"""
Register a K3D Galaxy GLB with the live server by sending a dataset_graph
event (ids, neighbors, labels, positions) and optional dataset_snippets with
label->text pairs for RAG.

Usage:
  python -m knowledge3d.tools.register_galaxy --gltf viewer/public/galaxy.cross.glb \
    --url ws://127.0.0.1:8787

Notes:
  - Expects the GLB to embed K3D extras in meshes[0].primitives[0].extras.k3d
  - Positions are read from bufferView 0 as float32 XYZ triples
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pygltflib import GLTF2  # type: ignore


def _load_from_glb(path: Path) -> Tuple[Dict[str, Any], List[Tuple[float, float, float]]]:
    g = GLTF2().load(str(path))
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    positions: List[Tuple[float, float, float]] = []
    try:
        import numpy as np  # type: ignore
        bv = g.bufferViews[0]
        blob = g.binary_blob()
        start = bv.byteOffset or 0
        end = start + bv.byteLength
        arr = np.frombuffer(blob[start:end], dtype=np.float32)
        pos = arr.reshape((-1, 3))
        positions = [(float(x), float(y), float(z)) for x, y, z in pos]
    except Exception:
        positions = []
    return k3d, positions


async def _send(url: str, msgs: List[Dict[str, Any]]) -> None:
    import websockets  # type: ignore

    # Allow a longer handshake window for slower server startups
    async with websockets.connect(url, open_timeout=30) as ws:
        # drain initial server messages (welcome/system)
        async def recv_once():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                _ = json.loads(raw)
            except Exception:
                pass

        await recv_once()
        for m in msgs:
            await ws.send(json.dumps(m))
            await recv_once()


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Register a K3D Galaxy GLB with the live server")
    ap.add_argument("--gltf", required=True, help="Path to GLB")
    ap.add_argument("--url", default="ws://127.0.0.1:8765", help="Live server ws URL")
    args = ap.parse_args()
    glb = Path(args.gltf)
    k3d, positions = _load_from_glb(glb)
    ids: List[str] = list(k3d.get("ids", []))
    neighbors: List[List[str]] = [list(r) for r in k3d.get("neighbors", [])]
    meta = [m if isinstance(m, dict) else {} for m in k3d.get("metadata", [])]
    labels: List[str] = [ (m.get("label") if isinstance(m, dict) else None) or ids[i] for i, m in enumerate(meta) ]
    snippets: List[Tuple[str, str]] = []
    for i, m in enumerate(meta):
        if not isinstance(m, dict):
            continue
        lab = str(m.get("label") or labels[i] or ids[i])
        txt = str(m.get("text") or "")
        if lab and txt:
            snippets.append((lab, txt))
    ds_evt = {
        "type": "event",
        "event": {"kind": "dataset_graph", "ids": ids, "neighbors": neighbors, "labels": labels, "positions": positions},
    }
    sn_evt = {
        "type": "event",
        "event": {"kind": "dataset_snippets", "pairs": snippets[: min(50000, len(snippets))]},  # cap to keep message reasonable
    }
    asyncio.run(_send(str(args.url), [ds_evt, sn_evt]))
    print(f"Registered graph ({len(ids)} nodes) and {len(snippets)} snippets → {args.url}")


if __name__ == "__main__":
    main()
