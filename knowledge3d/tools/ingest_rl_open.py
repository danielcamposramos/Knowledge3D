from __future__ import annotations

"""
Ingest open RL prompts and build a grounded RLWHF dataset using K3D memory.

Prompts come from public RLHF datasets (e.g., Anthropic HH-RLHF). For each prompt,
we retrieve contexts from the House GLB, generate a grounded answer (compose or
compose_generate), and compute a similarity-based reward. Outputs JSONL rows:
  {query, answer, contexts[], reward}

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.tools.ingest_rl_open \
    --gltf viewer/public/galaxy.cross.glb \
    --out docs/reports/training/rlwhf_dataset_open_1000.jsonl \
    --n 1000 --dataset anthropic --mode compose

Options:
  --dataset: anthropic|oasst (default: anthropic)
  --mode: compose|generate (default: compose)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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
        if isinstance(m, dict):
            lab = str(m.get("label") or labels[i] or ids[i])
            txt = str(m.get("text") or "")
            if lab and txt:
                snip[lab] = txt
    return ids, labels, snip


def load_prompts(kind: str, n: int) -> List[str]:
    from datasets import load_dataset  # type: ignore
    out: List[str] = []
    if kind == "anthropic":
        ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")  # smaller, general prompts
        for r in ds["train"]:
            t = (r.get("chosen") or r.get("prompt") or "")
            # Many rows have two variants; prefer the prompt if present
            if not t and isinstance(r.get("chosen"), str):
                t = r["chosen"].split("\n\n")[:1][0]
            if t and len(t) >= 16:
                out.append(str(t).strip())
    elif kind == "oasst":
        ds = load_dataset("OpenAssistant/oasst1")
        for r in ds["train"]:
            role = str(r.get("role") or r.get("role_name") or "").lower()
            # OASST commonly uses 'prompter' for user turns
            if role in {"user", "prompter"}:
                t = str(r.get("text") or r.get("message") or "").strip()
                if len(t) >= 16:
                    out.append(t)
    else:
        raise SystemExit(f"Unsupported dataset kind: {kind}")
    # Deduplicate and cap
    seen = set(); uniq: List[str] = []
    for q in out:
        if q in seen:
            continue
        seen.add(q); uniq.append(q)
        if len(uniq) >= n:
            break
    return uniq


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Build grounded RLWHF dataset from open RL prompts")
    ap.add_argument("--gltf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--dataset", default="anthropic", choices=["anthropic", "oasst"])
    ap.add_argument("--mode", default="compose", choices=["compose", "generate"], help="compose uses retrieval+stitching; generate uses internal LLM with grounded policy")
    args = ap.parse_args()

    ids, labels, snip = load_k3d(Path(args.gltf))
    # TF-IDF index over labels+snippets
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    vec = TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 2))
    corpus = [(f"{lab} — {snip.get(lab, '')}" if snip.get(lab) else lab) for lab in labels]
    X = vec.fit_transform(corpus)

    # ST encoder for similarity-based rewards
    from sentence_transformers import SentenceTransformer  # type: ignore
    try:
        import torch  # type: ignore
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=("cuda" if torch.cuda.is_available() else "cpu"))
    except Exception as e:
        raise SystemExit(f"sentence-transformers required: {e}")

    # Answer composers
    from knowledge3d.skills.spatial_text import compose_answer, compose_generate  # type: ignore

    prompts = load_prompts(args.dataset, int(args.n))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    with out.open("w", encoding="utf-8") as f:
        for i, q in enumerate(prompts):
            # Retrieve contexts via TF-IDF top-k
            qv = vec.transform([q])
            scores = (X @ qv.T).toarray().ravel()
            top = np.argsort(-scores)[:6]
            ctx_labels = [labels[int(j)] for j in top]
            ctx_texts = [snip.get(l, "") for l in ctx_labels]
            ctx_pairs = list(zip(ctx_labels, ctx_texts))
            # Compose answer
            if args.mode == "generate":
                ans = compose_generate(q, ctx_pairs, max_tokens=256)
            else:
                ans = compose_answer(q, ctx_pairs)
            # Reward (similarity to context blob)
            blob = "\n".join(ctx_texts)
            e1 = st.encode([ans], convert_to_numpy=True)[0]
            e2 = st.encode([blob], convert_to_numpy=True)[0]
            sim = cosine(e1, e2)
            reward = 1.0 if sim >= 0.70 else (0.5 if sim >= 0.40 else -0.25)
            f.write(json.dumps({"query": q, "answer": ans, "contexts": ctx_texts, "reward": reward}, ensure_ascii=False) + "\n")
            if (i+1) % 100 == 0:
                print(f"{i+1}/{len(prompts)}")
    print(str(out))


if __name__ == "__main__":
    main()
