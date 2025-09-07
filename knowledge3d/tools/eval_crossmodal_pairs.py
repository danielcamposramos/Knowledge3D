from __future__ import annotations

"""
Evaluate cross‑modal pair ranking inside Galaxy.

Given a Galaxy GLB and a matches.jsonl of {audio_id, video_id, score},
report how often a matched partner appears in each other’s top‑k neighbors
across modalities.

Usage
  python -m knowledge3d.tools.eval_crossmodal_pairs \
    --gltf viewer/public/galaxy.glb \
    --pairs ../Knowledge3D.local/datasets/matched/matches.jsonl \
    --k 10 \
    --out docs/reports/status/galaxy_crossmodal@10.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pygltflib import GLTF2  # type: ignore


def load_embeddings(glb: Path) -> Tuple[List[str], np.ndarray, List[dict]]:
    g = GLTF2().load(str(glb))
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    ids: List[str] = list(k3d.get("ids", []) or [])
    meta: List[dict] = [m if isinstance(m, dict) else {} for m in (k3d.get("metadata", []) or [])]
    ev = int(k3d.get("embeddingsView", 1))
    dims = int(k3d.get("embeddingDims", 384))
    prec = str(k3d.get("embeddingPrecision", "f32")).lower()
    bv = g.bufferViews[ev]
    blob = g.binary_blob()
    start = (bv.byteOffset or 0)
    end = start + bv.byteLength
    raw = blob[start:end]
    if prec == "f16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        arr = np.frombuffer(raw, dtype=np.float32)
    n = arr.size // dims
    emb = arr.reshape((n, dims)).copy()
    return ids, emb, meta


def load_pairs(path: Path) -> List[Tuple[str, str, float]]:
    out: List[Tuple[str, str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                out.append((str(j.get("audio_id")), str(j.get("video_id")), float(j.get("score") or 0.0)))
            except Exception:
                continue
    return out


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Evaluate cross‑modal pair ranking inside Galaxy")
    ap.add_argument("--gltf", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids, emb, meta = load_embeddings(Path(args.gltf))
    idx = {ids[i]: i for i in range(len(ids))}
    types = [ (m.get("type") if isinstance(m, dict) else None) for m in meta ]
    # L2 distances
    def topk(i: int, pool: List[int], k: int) -> List[int]:
        v = emb[i]
        U = emb[pool]
        d = np.sum((U - v) ** 2, axis=1)
        order = np.argsort(d)[:k]
        return [pool[j] for j in order]

    pairs = load_pairs(Path(args.pairs))
    # Build pools by type
    pool_by_type: Dict[str, List[int]] = {}
    for i, t in enumerate(types):
        if t:
            pool_by_type.setdefault(str(t), []).append(i)
    ok_both = 0
    ok_either = 0
    total = 0
    k = int(args.k)
    for a, v, _ in pairs[:5000]:  # cap for speed
        # IDs were prefixed like 'audio:' in galaxy build
        ga = idx.get(f"audio:{a}") or idx.get(a)
        gv = idx.get(f"video:{v}") or idx.get(v)
        if ga is None or gv is None:
            continue
        total += 1
        # neighbors in the opposite type pool
        t_a = 'video'; t_v = 'audio'
        pool_v = pool_by_type.get(t_a, [])  # from audio, search videos
        pool_a = pool_by_type.get(t_v, [])  # from video, search audios
        ta = topk(ga, pool_v, k) if pool_v else []
        tv = topk(gv, pool_a, k) if pool_a else []
        hit_a = gv in ta
        hit_v = ga in tv
        if hit_a and hit_v:
            ok_both += 1
        if hit_a or hit_v:
            ok_either += 1
    res = {
        "pairs": total,
        "k": k,
        "hit_either_topk": ok_either,
        "hit_both_topk": ok_both,
        "rate_either": (ok_either / total) if total else 0.0,
        "rate_both": (ok_both / total) if total else 0.0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

