from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pygltflib import GLTF2


def load_k3d_from_gltf(path: Path) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """Return mapping id->record with keys: id,label,embedding, and embedding dim."""
    g = GLTF2().load(path.as_posix())
    mesh = g.meshes[0]
    prim = mesh.primitives[0]
    ext = prim.extras.get("k3d", {}) if prim.extras else {}
    ids: List[str] = ext.get("ids", [])
    # metadata
    meta = ext.get("metadata", [{} for _ in ids])
    # embeddings
    recs: Dict[str, Dict[str, Any]] = {}
    dims = int(ext.get("embeddingDims") or (len(ext.get("embeddings", [])[0]) if ext.get("embeddings") else 0))
    if "embeddingsView" in ext:
        view_idx = int(ext["embeddingsView"])  # type: ignore
        buf = g.get_data_from_buffer_uri(g.buffers[g.bufferViews[view_idx].buffer].uri)
        bv = g.bufferViews[view_idx]
        raw = buf[bv.byteOffset : bv.byteOffset + bv.byteLength]
        # precision: default f32; optionally f16
        prec = ext.get("embeddingPrecision", "f32")
        total = len(ids) * int(dims)
        if prec == "f16":
            # struct 'e' = IEEE 754 half-precision float (Python 3.6+)
            flat = list(struct.unpack(f"<{total}e", raw[: total * 2]))
        else:
            flat = list(struct.unpack(f"<{total}f", raw[: total * 4]))
        mat = [flat[i * dims : (i + 1) * dims] for i in range(len(ids))]
    else:
        mat = ext.get("embeddings", [])

    for i, id_ in enumerate(ids):
        recs[id_] = {
            "id": id_,
            "label": (meta[i] or {}).get("label", id_),
            "embedding": mat[i] if i < len(mat) else [],
        }
    return recs, dims


def iter_logs(logs_dir: Path) -> Iterable[Dict[str, Any]]:
    for p in sorted(logs_dir.glob("session-*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def extract_edges_from_explain(text: str) -> Optional[Tuple[str, str]]:
    # Expect patterns like: "Step: A → B (neighbor hop, sim=0.123)."
    if "Step:" in text and "→" in text:
        try:
            core = text.split("Step:", 1)[1].strip()
            left, rest = core.split("→", 1)
            right = rest.split("(", 1)[0]
            a = left.strip()
            b = right.strip().strip(".")
            return a, b
        except Exception:
            return None
    return None


def build_samples(logs_dir: Path, asset: Optional[Path]) -> List[Dict[str, Any]]:
    idmap: Dict[str, Dict[str, Any]] = {}
    dims = 0
    if asset:
        try:
            idmap, dims = load_k3d_from_gltf(asset)
        except Exception:
            idmap, dims = {}, 0

    samples: List[Dict[str, Any]] = []
    # We’ll accumulate edges from explain events and create IL tuples
    for evt in iter_logs(logs_dir):
        if evt.get("type") == "event" and evt.get("kind") == "explain":
            pair = extract_edges_from_explain(str(evt.get("text", "")))
            if not pair:
                continue
            src, dst = pair
            # build sample; enrich using idmap when possible by matching label
            def record_for(label: str) -> Dict[str, Any]:
                # try direct label match
                for rec in idmap.values():
                    if rec.get("label") == label:
                        return rec
                # fallback id
                if label in idmap:
                    return idmap[label]
                return {"label": label, "embedding": []}

            a = record_for(src)
            b = record_for(dst)
            samples.append(
                {
                    "input": {
                        "from_label": a.get("label"),
                        "from_embedding": a.get("embedding"),
                        "context": {"explain": evt.get("text")},
                    },
                    "target": {
                        "next_label": b.get("label"),
                        "next_embedding": b.get("embedding"),
                    },
                    "meta": {"ts": evt.get("ts"), "dims": dims},
                }
            )
    return samples


def main() -> None:
    ap = argparse.ArgumentParser(description="Build IL replay samples from live logs")
    ap.add_argument("--logs_dir", default="../Knowledge3D.local/logs", help="Directory with session-*.jsonl")
    ap.add_argument("--gltf", help="Optional GLTF/GLB to enrich labels/embeddings")
    ap.add_argument("--out", default="../Knowledge3D.local/datasets/replay.jsonl", help="Output JSONL path")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir).resolve()
    asset = Path(args.gltf).resolve() if args.gltf else None
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    samples = build_samples(logs_dir, asset)
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"wrote {len(samples)} samples to {out}")


if __name__ == "__main__":  # pragma: no cover
    main()

