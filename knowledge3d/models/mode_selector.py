from __future__ import annotations

"""
Mode Selector — chooses between compose (retrieval+stitching) and
compose_generate (grounded generative) for a given query+contexts.

Baseline: logistic regression over lightweight features:
- n_ctx: number of contexts
- avg_ctx_len: average context text length
- sum_ctx_len: total context length
- q_len: query length
- media_flags: fraction of contexts that look like media (jpg/png/wav/mp4)

Training labels (heuristic bootstrapping):
- If contexts[] is empty → label=compose (0)
- If contexts[] non-empty → label=compose_generate (1)

Usage:
  Train:
    scripts/k3d_env.sh run python -m knowledge3d.models.mode_selector \
      --dataset docs/reports/training/rlwhf_dataset_unified_v3.jsonl \
      --out ../Knowledge3D.local/models/mode_selector.pkl

  Predict (debug):
    scripts/k3d_env.sh run python -m knowledge3d.models.mode_selector \
      --predict --model ../Knowledge3D.local/models/mode_selector.pkl \
      --query "What is UMAP?" --contexts "label1::This is text" "label2::Other text"
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


def _features(query: str, contexts: List[str]) -> np.ndarray:
    q = (query or "").strip()
    ctx = [str(c or "") for c in contexts]
    n_ctx = float(len(ctx))
    lens = [len(c) for c in ctx] or [0]
    avg_ctx_len = float(sum(lens) / len(lens)) if lens else 0.0
    sum_ctx_len = float(sum(lens))
    q_len = float(len(q))
    media_hits = 0
    for c in ctx:
        if any(s in c.lower() for s in [".jpg", ".png", ".jpeg", ".wav", ".mp4", "audio:", "video:"]):
            media_hits += 1
    media_frac = float(media_hits) / max(1.0, n_ctx)
    return np.asarray([n_ctx, avg_ctx_len, sum_ctx_len, q_len, media_frac], dtype=np.float32)


@dataclass
class Selector:
    pipe: object

    def predict_proba(self, query: str, contexts: List[str]) -> float:
        X = _features(query, contexts).reshape(1, -1)
        return float(self.pipe.predict_proba(X)[0][1])

    def predict(self, query: str, contexts: List[str]) -> int:
        X = _features(query, contexts).reshape(1, -1)
        return int(self.pipe.predict(X)[0])


def train(dataset: Path) -> Tuple[Selector, dict]:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    n = 0
    for rec in iter_jsonl(dataset):
        q = str(rec.get("query") or "").strip()
        contexts = [str(c or "") for c in (rec.get("contexts") or [])]
        lbl = 1 if contexts else 0
        X_rows.append(_features(q, contexts))
        y_rows.append(lbl)
        n += 1
    if not X_rows:
        raise SystemExit("No samples in dataset for mode selector")
    X = np.vstack(X_rows)
    y = np.asarray(y_rows, dtype=np.int64)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    sel = Selector(pipe=clf)
    acc = float((clf.predict(X) == y).mean())
    return sel, {"rows": int(n), "acc_train": acc}


def save(model: Selector, out: Path) -> None:
    import joblib
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.pipe, out)


def load(model_path: Path) -> Selector:
    import joblib
    pipe = joblib.load(model_path)
    return Selector(pipe=pipe)


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Train/Predict mode selector (compose vs compose_generate)")
    ap.add_argument("--dataset")
    ap.add_argument("--out")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--query")
    ap.add_argument("--contexts", nargs="*")
    args = ap.parse_args()
    if args.predict:
        sel = load(Path(args.model))
        ctxs: List[str] = []
        for c in (args.contexts or []):
            # allow "label::text" or just text
            parts = c.split("::", 1)
            ctxs.append(parts[1] if len(parts) == 2 else c)
        y = sel.predict(args.query or "", ctxs)
        print(json.dumps({"mode": ("compose_generate" if y == 1 else "compose"), "proba_generate": sel.predict_proba(args.query or "", ctxs)}, indent=2))
        return
    sel, info = train(Path(args.dataset))
    save(sel, Path(args.out))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

