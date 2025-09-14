import argparse
import struct
import numpy as np


def cube_vertices(width: float, height: float, depth: float) -> np.ndarray:
    hx, hy, hz = width * 0.5, height * 0.5, depth * 0.5
    return np.array([
        -hx, -hy, -hz,  # 0 (-x)
         hx, -hy, -hz,  # 1 (+x)
         hx,  hy, -hz,  # 2
        -hx,  hy, -hz,  # 3
        -hx, -hy,  hz,  # 4
         hx, -hy,  hz,  # 5
         hx,  hy,  hz,  # 6
        -hx,  hy,  hz,  # 7
    ], dtype=np.float32)


def displace_by_embedding(verts: np.ndarray, emb: np.ndarray, scale_spine=0.1, scale_cover=0.05) -> np.ndarray:
    out = verts.copy()
    # Spine vertices (left/-X face): 0,3,7,4 → use dims [0:12]
    spine_idx = [0, 3, 7, 4]
    for i, vi in enumerate(spine_idx):
        base = i * 3
        if base + 2 < emb.shape[0]:
            out[vi*3+0] += float(emb[base+0]) * scale_spine
            out[vi*3+1] += float(emb[base+1]) * scale_spine
            out[vi*3+2] += float(emb[base+2]) * scale_spine
    # Cover vertices (right/+X face): 1,2,6,5 → dims [12:24]
    cover_idx = [1, 2, 6, 5]
    for i, vi in enumerate(cover_idx):
        base = 12 + i * 3
        if base + 2 < emb.shape[0]:
            out[vi*3+0] += float(emb[base+0]) * scale_cover
            out[vi*3+1] += float(emb[base+1]) * scale_cover
            out[vi*3+2] += float(emb[base+2]) * scale_cover
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", default="embedding.npy")
    ap.add_argument("--out", default="book_mesh.bin")
    ap.add_argument("--dims", type=float, nargs=3, default=[0.05, 0.30, 0.20], help="width height depth")
    args = ap.parse_args()

    emb = np.load(args.embedding).astype(np.float32)
    w, h, d = args.dims
    v = cube_vertices(w, h, d)
    v = displace_by_embedding(v, emb)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<24f", *v.tolist()))
    print(f"Wrote {args.out} (24 floats for 8 vertices)")


if __name__ == "__main__":
    main()

