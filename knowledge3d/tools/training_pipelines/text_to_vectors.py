"""
Convert a UTF-8 text corpus (one record per line) into a dense numeric
vector CSV suitable for k3dgen.

Why: Enables large-scale generation (e.g., 80k) without depending on
sentence-transformers. Uses scikit-learn's HashingVectorizer for speed
and zero external downloads. Vectors are deterministic per input text.

Outputs
- CSV with header: id,v0,v1,...,v{dims-1}

Usage
  python3 -m knowledge3d.tools.text_to_vectors \
    --text data/ai_compendium_80k.txt \
    --out data/ai_compendium_80k_vectors.csv \
    --dims 512

Notes
- HashingVectorizer produces row-normalized TF features; we L2-normalize
  the vectors to stabilize neighbor search.
- For semantic quality, sentence-transformers is preferred, but this
  path is reliable and fast for proving scale and pipeline integrity.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore
from sklearn.preprocessing import normalize  # type: ignore


def iter_lines(p: Path) -> Iterable[str]:
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if s:
                yield s


def to_vectors(text_path: Path, dims: int) -> np.ndarray:
    vec = HashingVectorizer(n_features=dims, alternate_sign=False, norm="l2")
    X = vec.transform(list(iter_lines(text_path)))  # sparse
    # Convert to dense float32 for GLB embedding
    dense = X.astype(np.float32).toarray()
    # Extra L2 normalize to be safe
    dense = normalize(dense, norm="l2")
    return dense


def write_csv(out_path: Path, vectors: np.ndarray) -> None:
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["id"] + [f"v{i}" for i in range(vectors.shape[1])]
        w.writerow(header)
        for i in range(vectors.shape[0]):
            row = [str(i)] + [f"{float(x):.7f}" for x in vectors[i]]
            w.writerow(row)


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Text -> numeric vectors (CSV)")
    ap.add_argument("--text", required=True, help="Input text file (one record per line)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--dims", type=int, default=512, help="Embedding dimensions")
    args = ap.parse_args()
    text_path = Path(args.text)
    out_path = Path(args.out)
    vectors = to_vectors(text_path, args.dims)
    write_csv(out_path, vectors)
    print(f"Wrote vectors: shape={vectors.shape} -> {out_path}")


if __name__ == "__main__":
    main()

