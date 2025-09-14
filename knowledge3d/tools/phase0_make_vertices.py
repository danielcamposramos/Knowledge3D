import argparse
import numpy as np
import struct


BASE_VERTICES = np.array([
    0.0,    1.0,   0.0,
   -0.866, -0.5,   0.0,
    0.866, -0.5,   0.0,
    0.0,    0.0,   1.633,
], dtype=np.float32)


def make_vertices(embedding: np.ndarray, scale: float = 0.2) -> np.ndarray:
    out = BASE_VERTICES.copy()
    count = min(12, embedding.shape[0])
    d = np.clip(embedding[:count] * scale, -scale, scale)
    out[:count] += d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", default="embedding.npy")
    ap.add_argument("--bin", default="vertex_buffer.bin")
    ap.add_argument("--txt", default=None)
    ap.add_argument("--scale", type=float, default=0.2)
    args = ap.parse_args()

    emb = np.load(args.embedding).astype(np.float32)
    verts = make_vertices(emb, args.scale)
    with open(args.bin, "wb") as f:
        f.write(struct.pack("<12f", *verts.tolist()))
    if args.txt:
        with open(args.txt, "w") as f:
            f.write("\n".join(str(x) for x in verts.tolist()))
    print(f"Wrote {args.bin} (12 floats)")


if __name__ == "__main__":
    main()

