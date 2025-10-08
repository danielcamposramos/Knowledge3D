# DEPRECATED: legacy pre-PTX script; kept for reference. Outputs belong in Knowledge3D.local/old_attempts.
import argparse
import json
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_rays(embedding: np.ndarray, vertices: np.ndarray) -> list:
    rays = []
    if embedding.shape[0] < 73:
        return rays
    start = vertices[:3].astype(float)
    end = start + embedding[64:67].astype(float) * 2.0
    thickness = float(abs(embedding[67]) * 0.1)
    color = sigmoid(embedding[68:71].astype(float))
    tv = float(embedding[71])
    if tv > 0.7:
        rtype = 0
    elif tv > 0.3:
        rtype = 1
    else:
        rtype = 2
    honesty = float(embedding[72])
    rays.append({
        "start": start.tolist(),
        "end": end.tolist(),
        "thickness": thickness,
        "color": color.tolist(),
        "type": rtype,
        "honesty": honesty,
    })
    return rays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", default="embedding.npy")
    ap.add_argument("--vertices", default="vertex_buffer.bin")
    ap.add_argument("--out", default="rays.json")
    args = ap.parse_args()

    emb = np.load(args.embedding).astype(np.float32)
    verts = np.fromfile(args.vertices, dtype=np.float32, count=12)
    rays = generate_rays(emb, verts)
    with open(args.out, "w") as f:
        json.dump(rays, f, indent=2)
    print(f"Wrote {args.out} ({len(rays)} rays)")


if __name__ == "__main__":
    main()

