from __future__ import annotations

"""
Generate RLWHF records grounded in a K3D Galaxy using a local Ollama model
(e.g., exaone3.5). Each record contains a user query, an answer grounded only
in the provided contexts, and an implicit reward computed via ST cosine
similarity to the context blob. The prompt enforces honesty ("I don't know")
when insufficient evidence exists and asks for a short feedback section
explaining mistakes and corrections.

Usage
  python -m knowledge3d.tools.gen_rlwhf_exaone \
    --gltf viewer/public/galaxy.cross.glb \
    --out docs/reports/training/rlwhf_exaone_v1.jsonl \
    --n 1000 \
    --ollama http://192.168.0.4:11434 \
    --model exaone3.5:latest
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import requests  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore


def load_k3d(glb_path: Path) -> Tuple[List[str], List[str], List[str]]:
    from pygltflib import GLTF2  # type: ignore
    g = GLTF2().load(str(glb_path))
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    meta = [m if isinstance(m, dict) else {} for m in (k3d.get("metadata", []) or [])]
    labels: List[str] = []
    texts: List[str] = []
    ids = [str(x) for x in (k3d.get("ids") or [])]
    for i, m in enumerate(meta):
        lab = str(m.get("label") or (ids[i] if i < len(ids) else f"{i}"))
        labels.append(lab)
        texts.append(str(m.get("text") or ""))
    return ids, labels, texts


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def build_query_pool(labels: List[str], texts: List[str], n: int) -> List[str]:
    # Use labels + text snippets to craft short queries; deduplicate and cap
    qs: List[str] = []
    for lab, txt in zip(labels, texts):
        lab = lab.strip()
        if lab:
            qs.append(f"What is {lab}?")
        if txt and len(txt) > 24:
            qs.append(f"Explain: {txt[:80]}")
        if len(qs) >= n * 4:
            break
    seen = set(); out: List[str] = []
    for q in qs:
        if q in seen:
            continue
        seen.add(q); out.append(q)
        if len(out) >= n:
            break
    return out


def ask_exaone(ollama: str, model: str, question: str, ctx_pairs: List[Tuple[str, str]]) -> str:
    ctx_text = "\n".join([f"- {lab}: {txt}" for lab, txt in ctx_pairs if txt])
    prompt = (
        "You are an assistant that must answer ONLY using the provided context.\n"
        "If the context is insufficient, respond with `I don't know` and a one-sentence reason.\n"
        "Then add a short section `Feedback:` describing what was uncertain or wrong, and how to correct it.\n\n"
        f"Context:\n{ctx_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    url = ollama.rstrip("/") + "/api/generate"
    data = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}}
    r = requests.post(url, json=data, timeout=240)
    r.raise_for_status()
    out = r.json()
    return str(out.get("response") or out.get("data") or "").strip()


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Generate RLWHF records using exaone over a K3D Galaxy")
    ap.add_argument("--gltf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="exaone3.5:latest")
    args = ap.parse_args()

    ids, labels, texts = load_k3d(Path(args.gltf))
    # Build TF-IDF index for retrieval of contexts
    vec = TfidfVectorizer(lowercase=True, analyzer='word', ngram_range=(1, 2))
    corpus = [(f"{lab} — {txt}" if txt else lab) for lab, txt in zip(labels, texts)]
    X = vec.fit_transform(corpus)

    # ST encoder for reward
    from sentence_transformers import SentenceTransformer  # type: ignore
    import torch  # type: ignore
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

    queries = build_query_pool(labels, texts, int(args.n))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    with out.open('w', encoding='utf-8') as f:
        for i, q in enumerate(queries):
            qv = vec.transform([q])
            scores = (X @ qv.T).toarray().ravel()
            top = np.argsort(-scores)[:6]
            ctx_pairs = [(labels[int(j)], texts[int(j)]) for j in top]
            ans = ask_exaone(str(args.ollama), str(args.model), q, ctx_pairs)
            blob = "\n".join([t for _, t in ctx_pairs if t]) or ""
            e1 = st.encode([ans], convert_to_numpy=True)[0]
            e2 = st.encode([blob], convert_to_numpy=True)[0] if blob else np.zeros_like(e1)
            sim = cosine(e1, e2) if blob else 0.0
            reward = 1.0 if sim >= 0.70 else (0.5 if sim >= 0.40 else (-0.25 if blob else 0.0))
            rec = {"query": q, "answer": ans, "contexts": [t for _, t in ctx_pairs], "reward": reward}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"{i+1}/{len(queries)}")
    print(str(out))


if __name__ == '__main__':
    main()

