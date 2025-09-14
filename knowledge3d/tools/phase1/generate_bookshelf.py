import argparse
import struct
import numpy as np


def fractal_positions(embs: np.ndarray) -> np.ndarray:
    # CPU version of the simple PTX logic
    n = embs.shape[0]
    pos = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        e0 = float(embs[i, 0]) if embs.shape[1] > 0 else 0.0
        e1 = float(embs[i, 1]) if embs.shape[1] > 1 else 0.0
        e2 = float(embs[i, 2]) if embs.shape[1] > 2 else 0.0
        level = int(abs(e0) * 10.0)
        branch = e1 * 5.0
        x = np.sin(e0) * float(level)
        y = np.cos(e1) * float(branch)
        z = e2 * 3.0
        pos[i] = (x, y, z)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="embeddings.npy", help="N x D .npy stack")
    ap.add_argument("--out", default="bookshelf_positions.bin")
    args = ap.parse_args()

    embs = np.load(args.embeddings).astype(np.float32)
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    pos = fractal_positions(embs)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<" + "f" * (pos.size), *pos.reshape(-1).tolist()))
    print(f"Wrote {args.out} ({len(pos)} positions)")


if __name__ == "__main__":
    main()

