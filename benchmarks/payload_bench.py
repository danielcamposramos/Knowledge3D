from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from pygltflib import (
    GLTF2,
    Accessor,
    Asset,
    Buffer,
    BufferView,
    Mesh,
    Node,
    Primitive,
    Scene,
)

ARRAY_BUFFER = 34962
FLOAT = 5126


def synthesize(n: int, d: int, seed: int = 0) -> Tuple[List[str], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = [str(i) for i in range(n)]
    embeddings = rng.normal(size=(n, d)).astype(np.float32)
    # simple 3D projection via first 3 components (not important for size)
    if d >= 3:
        points = embeddings[:, :3].copy()
    else:
        points = np.zeros((n, 3), dtype=np.float32)
        points[:, :d] = embeddings
    return ids, embeddings, points


def write_json_extras_gltf(path: Path, ids: List[str], points: np.ndarray, embeddings: np.ndarray) -> None:
    # base buffer only for POSITION attribute
    positions = points.astype(np.float32)
    data_bytes = positions.tobytes()
    uri = "data:application/octet-stream;base64," + base64.b64encode(data_bytes).decode("ascii")
    buffer = Buffer(byteLength=len(data_bytes), uri=uri)
    view = BufferView(buffer=0, byteOffset=0, byteLength=len(data_bytes), target=ARRAY_BUFFER)
    accessor = Accessor(
        bufferView=0,
        byteOffset=0,
        componentType=FLOAT,
        count=len(points),
        type="VEC3",
        max=positions.max(axis=0).tolist(),
        min=positions.min(axis=0).tolist(),
    )
    prim = Primitive(attributes={"POSITION": 0}, mode=0, extras={
        "k3dIds": ids,
        "k3d": {
            "ids": ids,
            "vectors": points.tolist(),
            "embeddings": embeddings.astype(float).tolist(),
            "metadata": [{"label": i} for i in ids],
            "neighbors": [[] for _ in ids],
        }
    })
    gltf = GLTF2(
        asset=Asset(generator="payload_bench"),
        buffers=[buffer], bufferViews=[view], accessors=[accessor],
        meshes=[Mesh(primitives=[prim])], nodes=[Node(mesh=0)], scenes=[Scene(nodes=[0])], scene=0
    )
    gltf.save(path.as_posix())


def write_glb_bufferviews(path: Path, ids: List[str], points: np.ndarray, embeddings: np.ndarray) -> None:
    pos = points.astype(np.float32).tobytes()
    emb = embeddings.astype(np.float32).tobytes()
    blob = pos + emb
    buffer = Buffer(byteLength=len(blob))
    view_pos = BufferView(buffer=0, byteOffset=0, byteLength=len(pos), target=ARRAY_BUFFER)
    view_emb = BufferView(buffer=0, byteOffset=len(pos), byteLength=len(emb))
    accessor = Accessor(bufferView=0, byteOffset=0, componentType=FLOAT, count=len(points), type="VEC3",
                        max=points.max(axis=0).tolist(), min=points.min(axis=0).tolist())
    prim = Primitive(attributes={"POSITION": 0}, mode=0, extras={
        "k3dIds": ids,
        "k3d": {
            "ids": ids,
            "vectorsView": 0,
            "embeddingsView": 1,
            "embeddingDims": int(embeddings.shape[1]),
            "metadata": [{"label": i} for i in ids],
            "neighbors": [[] for _ in ids],
        }
    })
    gltf = GLTF2(
        asset=Asset(generator="payload_bench"),
        buffers=[buffer], bufferViews=[view_pos, view_emb], accessors=[accessor],
        meshes=[Mesh(primitives=[prim])], nodes=[Node(mesh=0)], scenes=[Scene(nodes=[0])], scene=0
    )
    gltf.set_binary_blob(blob)
    gltf.save(path.as_posix())


@dataclass
class Result:
    n: int
    d: int
    json_size: int
    glb_size: int
    json_parse_ms: float
    glb_parse_ms: float


def run_case(tmp: Path, n: int, d: int) -> Result:
    ids, emb, pts = synthesize(n, d)
    json_path = tmp / f"case_{n}_{d}.gltf"
    glb_path = tmp / f"case_{n}_{d}.glb"
    write_json_extras_gltf(json_path, ids, pts, emb)
    write_glb_bufferviews(glb_path, ids, pts, emb)

    json_size = json_path.stat().st_size
    glb_size = glb_path.stat().st_size

    # parse timing with pygltflib
    t0 = time.perf_counter()
    GLTF2().load(json_path.as_posix())
    t1 = time.perf_counter()
    GLTF2().load(glb_path.as_posix())
    t2 = time.perf_counter()

    return Result(n, d, json_size, glb_size, (t1 - t0) * 1000, (t2 - t1) * 1000)


def main():
    # Prefer sibling .local, fallback to repo docs/benchmarks when permissions fail
    try:
        out_dir = Path("../Knowledge3D.local/benchmarks").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        try:
            out_dir = Path("docs/benchmarks").resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            out_dir = Path("/tmp/k3d_benchmarks").resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "payload"
    tmp.mkdir(parents=True, exist_ok=True)

    cases = [(200, 64), (2000, 256), (10000, 256)]
    rows = []
    for n, d in cases:
        r = run_case(tmp, n, d)
        rows.append(r)
        print(f"n={r.n} d={r.d} json={r.json_size/1e6:.2f}MB glb={r.glb_size/1e6:.2f}MB parse_ms json={r.json_parse_ms:.1f} glb={r.glb_parse_ms:.1f}")

    # write summary markdown
    md = out_dir / "payload_results.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Payload Benchmark (JSON extras vs GLB bufferViews)\n\n")
        f.write("| n | d | JSON size (MB) | GLB size (MB) | JSON parse (ms) | GLB parse (ms) |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r.n} | {r.d} | {r.json_size/1e6:.2f} | {r.glb_size/1e6:.2f} | {r.json_parse_ms:.1f} | {r.glb_parse_ms:.1f} |\n")


if __name__ == "__main__":  # pragma: no cover
    main()
