from __future__ import annotations

"""
Offline benchmark of K3D-native composition vs. LLM baselines on a Galaxy GLB.

Measures per-query latency and semantic alignment to retrieved contexts.
Generates a JSON report and a concise Markdown summary.

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.tools.benchmark_offline \
    --gltf viewer/public/galaxy.cross.glb \
    --queries 20 \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --out-json docs/reports/status/chat_benchmark_offline.json \
    --out-md docs/reports/status/chat_benchmark_offline.md
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pygltflib import GLTF2  # type: ignore


def load_k3d(glb_path: Path) -> Tuple[List[str], List[str], Dict[str, str]]:
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


@dataclass
class Row:
    query: str
    mode: str
    latency_ms: float
    answer_chars: int
    context_sim: Optional[float]


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Offline chat benchmark: K3D vs LLM (transformers)")
    ap.add_argument("--gltf", required=True)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    glb = Path(args.gltf)
    ids, labels, snip = load_k3d(glb)

    # Context selection (TF-IDF) for each query
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    vec = TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 2))
    corpus = [(f"{lab} — {snip.get(lab, '')}" if snip.get(lab) else lab) for lab in labels]
    X = vec.fit_transform(corpus)

    # ST encoder for semantic similarity on GPU when available
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import torch  # type: ignore
        dev = {"device": "cuda"} if torch.cuda.is_available() else {}
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", **dev)
    except Exception:
        st = None

    # K3D-native composition
    try:
        from knowledge3d.skills.spatial_text import compose_answer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Missing spatial_text skill: {e}")

    # LLM baseline (transformers)
    from knowledge3d.skills.llm import LLMSkill, LLMConfig  # type: ignore
    llm = LLMSkill(LLMConfig(backend="transformers", model=str(args.model)))

    # Build queries
    rng = np.random.default_rng(42)
    idxs = [i for i, lab in enumerate(labels) if lab and len(lab) >= 8]
    chosen = list(rng.choice(idxs, size=min(int(args.queries), len(idxs)), replace=False))
    templates = [
        "What is {}?",
        "Explain {} in simple terms.",
        "Give a brief overview of {}.",
        "Summarize {}.",
        "Tell me about {}.",
    ]

    rows: List[Dict[str, Any]] = []
    for i, idx in enumerate(chosen):
        label = labels[idx]
        q = templates[i % len(templates)].format(label)
        qv = vec.transform([q])
        scores = (X @ qv.T).toarray().ravel()
        top = np.argsort(-scores)[:6]
        ctx_labels = [labels[int(j)] for j in top]
        ctx_texts = [snip.get(l, "") for l in ctx_labels]
        ctx_pairs = list(zip(ctx_labels, ctx_texts))
        ctx_blob = "\n".join([f"{l}: {t}" for l, t in ctx_pairs])

        def sim(ans: str) -> Optional[float]:
            if st is None or not ans:
                return None
            try:
                e1 = st.encode([ans], convert_to_numpy=True)
                e2 = st.encode([ctx_blob], convert_to_numpy=True)
                num = float(np.dot(e1[0], e2[0])); den = float(np.linalg.norm(e1[0]) * np.linalg.norm(e2[0]) + 1e-9)
                return num / den
            except Exception:
                return None

        # K3D
        t0 = time.perf_counter()
        a_k3d = compose_answer(q, ctx_pairs)
        dt_k3d = (time.perf_counter() - t0) * 1000.0
        rows.append({
            "query": q,
            "mode": "k3d",
            "latency_ms": float(dt_k3d),
            "answer_chars": len(a_k3d),
            "context_sim": sim(a_k3d),
        })

        # LLM (no RAG)
        t1 = time.perf_counter()
        a_llm = llm.generate(q, system=None, max_tokens=256)
        dt_llm = (time.perf_counter() - t1) * 1000.0
        rows.append({
            "query": q,
            "mode": "llm",
            "latency_ms": float(dt_llm),
            "answer_chars": len(a_llm or ""),
            "context_sim": sim(a_llm or ""),
        })

        # LLM+RAG
        t2 = time.perf_counter()
        a_rag = llm.answer_with_rag(q, ctx_pairs, max_tokens=256)
        dt_rag = (time.perf_counter() - t2) * 1000.0
        rows.append({
            "query": q,
            "mode": "llm_rag",
            "latency_ms": float(dt_rag),
            "answer_chars": len(a_rag or ""),
            "context_sim": sim(a_rag or ""),
        })

    def agg(mode: str) -> Dict[str, Any]:
        sel = [r for r in rows if r["mode"] == mode]
        lat = [r["latency_ms"] for r in sel]
        simv = [r["context_sim"] for r in sel if r["context_sim"] is not None]
        return {
            "count": len(sel),
            "latency_ms_avg": (sum(lat) / len(lat)) if lat else None,
            "latency_ms_p50": (sorted(lat)[len(lat)//2] if lat else None),
            "context_sim_avg": (sum(simv) / len(simv)) if simv else None,
        }

    report = {
        "n_labels": len(labels),
        "n_queries": len(chosen),
        "model": str(args.model),
        "modes": {
            "k3d": agg("k3d"),
            "llm": agg("llm"),
            "llm_rag": agg("llm_rag"),
        },
        "rows": rows,
    }

    out_json = Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write concise markdown summary
    md = Path(args.out_md)
    md.parent.mkdir(parents=True, exist_ok=True)
    def fmt(m: str) -> str:
        d = report["modes"][m]
        la = d["latency_ms_avg"]; lp = d["latency_ms_p50"]; cs = d["context_sim_avg"]
        return f"avg_latency={la:.1f}ms p50={lp:.1f}ms sim={cs:.3f}" if (la and lp and cs) else "n/a"
    md.write_text(
        (
            f"# Chat Benchmark (Offline)\n\n"
            f"- labels: {report['n_labels']}  queries: {report['n_queries']}  model: {report['model']}\n"
            f"- K3D: {fmt('k3d')}\n"
            f"- LLM: {fmt('llm')}\n"
            f"- LLM+RAG: {fmt('llm_rag')}\n"
        ),
        encoding="utf-8",
    )
    print(json.dumps(report["modes"], indent=2))


if __name__ == "__main__":
    main()

