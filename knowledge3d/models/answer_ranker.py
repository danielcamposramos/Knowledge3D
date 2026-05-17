from __future__ import annotations

"""
Answer Ranker — learns to rank contexts for K3D compose based on RLWHF rewards.

Model: simple linear regression over ST cosine features between query and
each context snippet. Trained from RLWHF dataset JSONL.

Usage:
  # Train
  scripts/k3d_env.sh run python -m knowledge3d.models.answer_ranker \
    --dataset docs/reports/training/rlwhf_dataset.jsonl \
    --out /K3D/Knowledge3D.local/models/answer_ranker.pkl

  # Score (debug)
  scripts/k3d_env.sh run python -m knowledge3d.models.answer_ranker \
    --score --model /K3D/Knowledge3D.local/models/answer_ranker.pkl \
    --query "What is UMAP?" --context "UMAP is a dimensionality reduction."
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


@dataclass
class Ranker:
    st_model: object
    coef: np.ndarray
    bias: float

    def score(self, query: str, context: str) -> float:
        import numpy as _np  # type: ignore
        qv = self.st_model.encode([query], convert_to_numpy=True)[0]
        cv = self.st_model.encode([context], convert_to_numpy=True)[0]
        sim = float(np.dot(qv, cv) / (np.linalg.norm(qv) * np.linalg.norm(cv) + 1e-9))
        x = _np.asarray([sim], dtype=_np.float32)
        return float(self.bias + self.coef[0] * x[0])


def train(dataset: Path) -> Tuple[Ranker, dict]:
    # Load ST on GPU if available
    from sentence_transformers import SentenceTransformer  # type: ignore
    try:
        import os
        import torch  # type: ignore
        strict = os.getenv("K3D_STRICT_GPU", "0").strip() != "0"
        if strict and not torch.cuda.is_available():
            raise SystemExit("GPU required (K3D_STRICT_GPU=1) but CUDA is not available")
        dev = {"device": "cuda"} if torch.cuda.is_available() else {}
    except Exception as e:
        raise SystemExit(f"GPU setup error: {e}")
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", **dev)
    X: List[float] = []
    y: List[float] = []
    n_rows = 0
    for rec in iter_jsonl(dataset):
        q = str(rec.get("query") or "").strip()
        rw = float(rec.get("reward") or 0.0)
        ctxs = rec.get("contexts") or []
        # If no contexts captured, skip
        if not q or not ctxs:
            continue
        n_rows += 1
        # Use the first few contexts as training points
        for c in ctxs[:6]:
            c = str(c or "").strip()
            if not c:
                continue
            qv = st.encode([q], convert_to_numpy=True)[0]
            cv = st.encode([c], convert_to_numpy=True)[0]
            sim = float(np.dot(qv, cv) / (np.linalg.norm(qv) * np.linalg.norm(cv) + 1e-9))
            X.append(sim)
            y.append(rw)
    if not X:
        raise SystemExit("No training samples in RLWHF dataset (need chat_response entries with contexts)")
    Xn = np.asarray(X, dtype=np.float32).reshape(-1, 1)
    yn = np.asarray(y, dtype=np.float32)
    # Linear regression (closed form)
    Xb = np.hstack([np.ones((Xn.shape[0], 1), dtype=np.float32), Xn])
    theta = np.linalg.pinv(Xb).dot(yn)
    bias = float(theta[0]); coef = np.asarray(theta[1:], dtype=np.float32)
    mdl = Ranker(st_model=st, coef=coef, bias=bias)
    # Simple metric
    yhat = Xb.dot(np.hstack([bias, coef]))
    mse = float(np.mean((yhat - yn) ** 2))
    return mdl, {"samples": int(Xn.shape[0]), "rows": n_rows, "mse": mse, "coef": coef.tolist(), "bias": bias}


def save(model: Ranker, out: Path) -> None:
    import pickle
    out.parent.mkdir(parents=True, exist_ok=True)
    # Save only coef/bias; ST loads at runtime by path
    with out.open("wb") as f:
        pickle.dump({"coef": model.coef, "bias": model.bias}, f)


def load(model_path: Path) -> Ranker:
    import pickle
    from sentence_transformers import SentenceTransformer  # type: ignore
    try:
        import os
        import torch  # type: ignore
        strict = os.getenv("K3D_STRICT_GPU", "0").strip() != "0"
        if strict and not torch.cuda.is_available():
            raise SystemExit("GPU required (K3D_STRICT_GPU=1) but CUDA is not available")
        dev = {"device": "cuda"} if torch.cuda.is_available() else {}
    except Exception as e:
        raise SystemExit(f"GPU setup error: {e}")
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", **dev)
    with model_path.open("rb") as f:
        d = pickle.load(f)
    coef = np.asarray(d.get("coef"), dtype=np.float32)
    bias = float(d.get("bias"))
    return Ranker(st_model=st, coef=coef, bias=bias)


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Train/score Answer Ranker from RLWHF dataset")
    ap.add_argument("--dataset")
    ap.add_argument("--out")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--query")
    ap.add_argument("--context")
    args = ap.parse_args()
    if args.score:
        mdl = load(Path(args.model))
        if not args.query or not args.context:
            raise SystemExit("--query and --context required for --score")
        print(mdl.score(args.query, args.context))
        return
    mdl, info = train(Path(args.dataset))
    save(mdl, Path(args.out))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
