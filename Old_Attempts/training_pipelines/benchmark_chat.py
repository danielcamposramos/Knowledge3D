from __future__ import annotations

"""
Benchmark K3D chat (/ask) vs. baseline LLM (/llm ask and /llm rag) on a live server
loaded with a Galaxy GLB. Measures latency and simple semantic alignment metrics.

Outputs JSON with per-query metrics and aggregates.

Usage:
  python -m knowledge3d.tools.benchmark_chat \
    --gltf viewer/public/galaxy.cross.glb \
    --url ws://127.0.0.1:8787 \
    --queries 20 \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --out docs/reports/status/chat_benchmark.json
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pygltflib import GLTF2  # type: ignore


def _load_k3d(glb_path: Path) -> Tuple[List[str], List[List[str]], List[str], List[Tuple[float, float, float]], Dict[str, str]]:
    g = GLTF2().load(str(glb_path))
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    ids: List[str] = list(k3d.get("ids", []))
    neighbors: List[List[str]] = [list(r) for r in k3d.get("neighbors", [])]
    meta = [m if isinstance(m, dict) else {} for m in k3d.get("metadata", [])]
    labels: List[str] = [ (m.get("label") if isinstance(m, dict) else None) or ids[i] for i, m in enumerate(meta) ]
    # positions in bufferView 0
    try:
        bv = g.bufferViews[0]
        blob = g.binary_blob()
        start = bv.byteOffset or 0
        end = start + bv.byteLength
        arr = np.frombuffer(blob[start:end], dtype=np.float32)
        pos = arr.reshape((-1, 3))
        positions = [(float(x), float(y), float(z)) for x, y, z in pos]
    except Exception:
        positions = []
    # snippets from metadata
    snip: Dict[str, str] = {}
    for i, m in enumerate(meta):
        if not isinstance(m, dict):
            continue
        lab = str(m.get("label") or labels[i] or ids[i])
        txt = str(m.get("text") or "")
        if lab and txt:
            snip[lab] = txt
    return ids, neighbors, labels, positions, snip


@dataclass
class QueryResult:
    query: str
    mode: str  # k3d|llm|llm_rag
    latency_ms: float
    answer: str
    context_sim: Optional[float]
    answer_chars: int


async def _chat_roundtrip(ws, text: str, want_sender: str = "agent", timeout_s: float = 20.0) -> Tuple[str, float]:
    # Send chat and wait for first agent reply
    t0 = time.perf_counter()
    await ws.send(json.dumps({"type": "chat", "from": "bench", "text": text}))
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        if msg.get("type") == "chat" and str(msg.get("from")) == want_sender:
            dt = (time.perf_counter() - t0) * 1000.0
            return str(msg.get("text") or ""), dt


async def _send_event(ws, event: Dict[str, Any]) -> None:
    await ws.send(json.dumps({"type": "event", "event": event}))


async def run_benchmark(url: str, glb_path: Path, queries: int, model: Optional[str]) -> Dict[str, Any]:
    ids, neighbors, labels, positions, snip = _load_k3d(glb_path)
    # Build simple TF-IDF on snippets locally for fair context selection
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    vec = TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 2))
    corpus = [(f"{lab} — {snip.get(lab, '')}" if snip.get(lab) else lab) for lab in labels]
    X = vec.fit_transform(corpus)

    # ST encoder for semantic similarity (GPU preferred via torch cuda)
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import torch  # type: ignore
        dev = {"device": "cuda"} if torch.cuda.is_available() else {}
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", **dev)
    except Exception:
        st = None  # similarity metric becomes None

    async with __import__("websockets").connect(url) as ws:  # type: ignore
        # Drain welcomes
        try:
            await asyncio.wait_for(ws.recv(), timeout=1.0)
        except Exception:
            pass
        # Register dataset graph + snippets
        await _send_event(ws, {"kind": "dataset_graph", "ids": ids, "neighbors": neighbors, "labels": labels, "positions": positions})
        # Chunk snippets to avoid huge messages (send first 20k pairs)
        labs = list(snip.keys())
        pairs_all = [(lab, snip[lab]) for lab in labs]
        chunk = pairs_all[: min(20000, len(pairs_all))]
        await _send_event(ws, {"kind": "dataset_snippets", "pairs": chunk})

        # Change LLM backend model if provided
        if model:
            await _chat_roundtrip(ws, f"/llm backend transformers {model}", want_sender="system")

        # Generate queries by sampling labels with readable text
        idxs = [i for i, lab in enumerate(labels) if lab and len(lab) >= 8]
        rng = np.random.default_rng(42)
        chosen = list(rng.choice(idxs, size=min(queries, len(idxs)), replace=False))
        # Prompt templates to diversify phrasing
        templates = [
            "What is {}?",
            "Explain {} in simple terms.",
            "Give a brief overview of {}.",
            "Summarize {}.",
            "Tell me about {}.",
        ]

        out: List[Dict[str, Any]] = []
        for i, idx in enumerate(chosen):
            label = labels[idx]
            q = templates[i % len(templates)].format(label)
            # Compute contexts (top 6 by TF-IDF cosine)
            qv = vec.transform([q])
            scores = (X @ qv.T).toarray().ravel()
            top = np.argsort(-scores)[:6]
            ctx_labels = [labels[int(j)] for j in top]
            ctx_texts = [snip.get(l, "") for l in ctx_labels]
            ctx_blob = "\n".join([f"{l}: {t}" for l, t in zip(ctx_labels, ctx_texts)])

            # 1) K3D native (/ask)
            ans_k3d, t_k3d = await _chat_roundtrip(ws, f"/ask {q}")
            sim_k3d = None
            if st is not None and ans_k3d:
                try:
                    e1 = st.encode([ans_k3d], convert_to_numpy=True)
                    e2 = st.encode([ctx_blob], convert_to_numpy=True)
                    num = float(np.dot(e1[0], e2[0]))
                    den = float(np.linalg.norm(e1[0]) * np.linalg.norm(e2[0]) + 1e-9)
                    sim_k3d = num / den
                except Exception:
                    sim_k3d = None
            out.append({
                "query": q,
                "mode": "k3d",
                "latency_ms": float(t_k3d),
                "answer_chars": len(ans_k3d),
                "context_sim": (float(sim_k3d) if sim_k3d is not None else None),
            })

            # 2) LLM baseline (no RAG)
            ans_llm, t_llm = await _chat_roundtrip(ws, f"/llm ask {q}")
            sim_llm = None
            if st is not None and ans_llm:
                try:
                    e1 = st.encode([ans_llm], convert_to_numpy=True)
                    e2 = st.encode([ctx_blob], convert_to_numpy=True)
                    num = float(np.dot(e1[0], e2[0])); den = float(np.linalg.norm(e1[0]) * np.linalg.norm(e2[0]) + 1e-9)
                    sim_llm = num / den
                except Exception:
                    sim_llm = None
            out.append({
                "query": q,
                "mode": "llm",
                "latency_ms": float(t_llm),
                "answer_chars": len(ans_llm),
                "context_sim": (float(sim_llm) if sim_llm is not None else None),
            })

            # 3) LLM+RAG baseline
            ans_rag, t_rag = await _chat_roundtrip(ws, f"/llm rag {q} 6")
            sim_rag = None
            if st is not None and ans_rag:
                try:
                    e1 = st.encode([ans_rag], convert_to_numpy=True)
                    e2 = st.encode([ctx_blob], convert_to_numpy=True)
                    num = float(np.dot(e1[0], e2[0])); den = float(np.linalg.norm(e1[0]) * np.linalg.norm(e2[0]) + 1e-9)
                    sim_rag = num / den
                except Exception:
                    sim_rag = None
            out.append({
                "query": q,
                "mode": "llm_rag",
                "latency_ms": float(t_rag),
                "answer_chars": len(ans_rag),
                "context_sim": (float(sim_rag) if sim_rag is not None else None),
            })

        # Aggregate
        def agg(mode: str) -> Dict[str, Any]:
            rows = [r for r in out if r["mode"] == mode]
            lat = [r["latency_ms"] for r in rows]
            sim = [r["context_sim"] for r in rows if r["context_sim"] is not None]
            return {
                "count": len(rows),
                "latency_ms_avg": (sum(lat) / len(lat)) if lat else None,
                "latency_ms_p50": (sorted(lat)[len(lat)//2] if lat else None),
                "context_sim_avg": (sum(sim) / len(sim)) if sim else None,
            }

        return {
            "n_labels": len(labels),
            "n_queries": int(len(chosen)),
            "model": model or "default",
            "modes": {
                "k3d": agg("k3d"),
                "llm": agg("llm"),
                "llm_rag": agg("llm_rag"),
            },
            "rows": out,
        }


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Benchmark K3D chat vs LLM baselines")
    ap.add_argument("--gltf", required=True)
    ap.add_argument("--url", default="ws://127.0.0.1:8787")
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--model", help="HF model id for transformers backend")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = asyncio.run(run_benchmark(args.url, Path(args.gltf), int(args.queries), args.model))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

