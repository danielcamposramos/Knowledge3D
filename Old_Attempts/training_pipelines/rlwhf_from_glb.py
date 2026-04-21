from __future__ import annotations

"""
Build RLWHF dataset directly from a GLB by sampling labels, composing K3D answers,
and computing similarity-based rewards (no external LLM calls).

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.tools.rlwhf_from_glb \
    --gltf viewer/public/galaxy.cross.glb --out docs/reports/training/rlwhf_dataset_glb.jsonl --queries 200
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pygltflib import GLTF2  # type: ignore


def load_k3d(glb_path: Path):
    g = GLTF2().load(str(glb_path))
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    ids: List[str] = list(k3d.get("ids", []))
    meta = [m if isinstance(m, dict) else {} for m in k3d.get("metadata", [])]
    labels: List[str] = [ (m.get("label") if isinstance(m, dict) else None) or ids[i] for i, m in enumerate(meta) ]
    snip: Dict[str, str] = {}
    for i, m in enumerate(meta):
        if not isinstance(m, dict):
            continue
        lab = str(m.get("label") or labels[i] or ids[i])
        txt = str(m.get("text") or "")
        if lab and txt:
            snip[lab] = txt
    return ids, labels, snip


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Build RLWHF dataset from GLB using K3D compose")
    ap.add_argument("--gltf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--queries", type=int, default=200)
    args = ap.parse_args()
    ids, labels, snip = load_k3d(Path(args.gltf))
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    vec = TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 2))
    corpus = [(f"{lab} — {snip.get(lab, '')}" if snip.get(lab) else lab) for lab in labels]
    X = vec.fit_transform(corpus)
    # Compose answers via K3D
    from knowledge3d.skills.spatial_text import compose_answer  # type: ignore
    # ST encoder for rewards
    from sentence_transformers import SentenceTransformer  # type: ignore
    try:
        import torch  # type: ignore
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=("cuda" if torch.cuda.is_available() else "cpu"))
    except Exception as e:
        raise SystemExit(f"sentence-transformers required: {e}")
    rng = np.random.default_rng(42)
    idxs = [i for i, lab in enumerate(labels) if lab and len(lab) >= 8]
    chosen = list(rng.choice(idxs, size=min(int(args.queries), len(idxs)), replace=False))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, idx in enumerate(chosen):
            label = labels[idx]
            # Build contexts
            q = f"Explain {label} in simple terms."
            qv = vec.transform([q])
            scores = (X @ qv.T).toarray().ravel()
            top = np.argsort(-scores)[:6]
            ctx_labels = [labels[int(j)] for j in top]
            ctx_texts = [snip.get(l, "") for l in ctx_labels]
            ctx_pairs = list(zip(ctx_labels, ctx_texts))
            # Compose and score
            ans = compose_answer(q, ctx_pairs)
            blob = "\n".join(ctx_texts)
            e1 = st.encode([ans], convert_to_numpy=True)[0]
            e2 = st.encode([blob], convert_to_numpy=True)[0]
            num = float(np.dot(e1, e2)); den = float(np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9)
            sim = num / den
            reward = 1.0 if sim >= 0.70 else (0.5 if sim >= 0.40 else -0.25)
            f.write(json.dumps({"query": q, "answer": ans, "contexts": ctx_texts, "reward": reward}, ensure_ascii=False) + "\n")
    print(str(out))


if __name__ == "__main__":
    main()

