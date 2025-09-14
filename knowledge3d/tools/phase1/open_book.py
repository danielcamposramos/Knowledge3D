import argparse
import json
import struct
from pathlib import Path
from typing import Optional

import numpy as np
from pygltflib import GLTF2


def decode_embedding_to_text(emb: np.ndarray, limit: int = 1000) -> str:
    text = []
    for i in range(0, min(len(emb), 128), 4):
        code = int(abs(float(emb[i])) * 255) % 128
        if 32 <= code <= 126:
            text.append(chr(code))
    return ("".join(text))[:limit]


def find_book_by_title(glb_path: Path, title: str) -> Optional[tuple[int, dict]]:
    m = GLTF2().load_binary(str(glb_path))
    for mi, mesh in enumerate(m.meshes or []):
        for prim in mesh.primitives or []:
            k3d = (prim.extras or {}).get("k3d") if prim.extras else None
            if isinstance(k3d, dict) and k3d.get("object", {}).get("kind") == "book":
                if str(k3d.get("object", {}).get("title", "")) == title:
                    return mi, k3d
    return None


def project_text(glb_path: Path, title: str, out_dir: Path) -> str:
    found = find_book_by_title(glb_path, title)
    if not found:
        return f"Book '{title}' not found."
    _, k3d = found
    emb_view = int(k3d.get("embeddingsView"))
    m = GLTF2().load_binary(str(glb_path))
    blob = m.binary_blob()
    bv = m.bufferViews[emb_view]
    data = blob[(bv.byteOffset or 0) : (bv.byteOffset or 0) + bv.byteLength]
    emb = np.array(struct.unpack("<" + "f" * (bv.byteLength // 4), data), dtype=np.float32)
    txt = decode_embedding_to_text(emb)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join([c for c in title if c.isalnum() or c in ("_","-")])
    out_file = out_dir / f"{safe}.txt"
    out_file.write_text(txt or "(empty)", encoding="utf-8")
    return f"Projected '{title}' -> {out_file.name} ({len(txt)} chars)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", default="viewer/public/library_room.glb")
    ap.add_argument("--title", required=True)
    ap.add_argument("--outdir", default="viewer/public/projections")
    args = ap.parse_args()
    msg = project_text(Path(args.glb), args.title, Path(args.outdir))
    print(msg)


if __name__ == "__main__":
    main()

