from __future__ import annotations

"""
Build RLWHF dataset JSONL from offline benchmark results + GLB.

Reconstructs contexts via TF-IDF over labels+snippets, computes similarity-based
rewards, and writes RLWHF rows: {query, answer, contexts[], reward}.

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.tools.rlwhf_from_offline_benchmark \
    --gltf viewer/public/galaxy.cross.glb \
    --bench docs/reports/status/chat_benchmark_offline.json \
    --out docs/reports/training/rlwhf_dataset_offline.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from pygltflib import GLTF2  # type: ignore


def load_k3d(glb_path: Path):
    g = GLTF2().load(str(glb_path))
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    ids: List[str] = list(k3d.get("ids", []))
    meta = [m if isinstance(m, dict) else {} for m in k3d.get("metadata", [])]
    labels: List[str] = [ (m.get("label") if isinstance(m, dict) else None) or ids[i] for i, m in enumerate(meta) ]
    snip = {}
    for i, m in enumerate(meta):
        if isinstance(m, dict):
            lab = str(m.get("label") or labels[i] or ids[i])
            txt = str(m.get("text") or "")
            if lab and txt:
                snip[lab] = txt
    return labels, snip


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Convert offline benchmark JSON to RLWHF dataset")
    ap.add_argument("--gltf", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    labels, snip = load_k3d(Path(args.gltf))
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    vec = TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 2))
    corpus = [(f"{lab} — {snip.get(lab, '')}" if snip.get(lab) else lab) for lab in labels]
    X = vec.fit_transform(corpus)
    # ST for similarity
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import torch  # type: ignore
        dev = {"device": "cuda"} if torch.cuda.is_available() else {}
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", **dev)
    except Exception as e:
        raise SystemExit(f"sentence-transformers required: {e}")
    bench = json.loads(Path(args.bench).read_text(encoding="utf-8"))
    rows = bench.get("rows", [])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            if r.get("mode") != "k3d":
                continue
            q = str(r.get("query") or "").strip()
            a = ""  # we can reconstruct from contexts via compose if needed, but not required
            if not q:
                continue
            # Rebuild contexts via TF-IDF
            qv = vec.transform([q])
            scores = (X @ qv.T).toarray().ravel()
            top = np.argsort(-scores)[:6]
            ctx_labels = [labels[int(j)] for j in top]
            ctx_texts = [snip.get(l, "") for l in ctx_labels]
            # Use offline sim from row if available, else compute with ST after generating a with K3D
            ans = str(r.get("answer") or "")
            if not ans:
                from knowledge3d.skills.spatial_text import compose_answer  # type: ignore
                ans = compose_answer(q, list(zip(ctx_labels, ctx_texts)))
            ans_v = st.encode([ans], convert_to_numpy=True)[0]
            ctx_v = st.encode(["\n".join(ctx_texts)], convert_to_numpy=True)[0]
            sim = float(np.dot(ans_v, ctx_v) / (np.linalg.norm(ans_v) * np.linalg.norm(ctx_v) + 1e-9))
            reward = 1.0 if sim >= 0.70 else (0.5 if sim >= 0.40 else -0.25)
            f.write(json.dumps({
                "query": q,
                "answer": ans,
                "contexts": ctx_texts,
                "reward": reward,
            }, ensure_ascii=False) + "\n")
    print(str(out))


if __name__ == "__main__":
    main()

