from __future__ import annotations

"""
Build an algorithmic-thinking RLWHF dataset (with K3D hints) by generating
simple algorithmic tasks tied to House labels/snippets, computing answers
programmatically, and assigning rewards.

Tasks include sorting labels, grouping by prefix, counting words in a snippet,
and reversing lists. Contexts are drawn from the House GLB text snippets so the
policy still learns to ground answers in K3D memory while practicing
algorithmic structure.

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.tools.build_algorithmic_thinking \
    --out docs/reports/training/rl_dataset_algo_rlwhf_2000.jsonl \
    --n 2000 --mode rlwhf --gltf viewer/public/galaxy.cross.glb
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pygltflib import GLTF2  # type: ignore


def load_k3d(glb_path: Path) -> Tuple[List[str], List[str], Dict[str, str]]:
    g = GLTF2().load(str(glb_path))
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    ids: List[str] = list(k3d.get("ids", []))
    meta = [m if isinstance(m, dict) else {} for m in k3d.get("metadata", [])]
    labels: List[str] = [(m.get("label") if isinstance(m, dict) else None) or ids[i] for i, m in enumerate(meta)]
    snip: Dict[str, str] = {}
    for i, m in enumerate(meta):
        if not isinstance(m, dict):
            continue
        lab = str(m.get("label") or labels[i] or ids[i])
        txt = str(m.get("text") or "")
        if lab and txt:
            snip[lab] = txt
    return ids, labels, snip


def build_tasks(labels: List[str], snip: Dict[str, str], n: int) -> List[Tuple[str, List[str], str]]:
    rng = random.Random(42)
    clean = [l for l in labels if l and isinstance(l, str) and len(l) >= 3]
    out: List[Tuple[str, List[str], str]] = []
    if not clean:
        return out
    for _ in range(max(1, int(n))):
        kind = rng.choice(["sort_asc", "sort_desc", "group_prefix", "count_words", "reverse"])
        picks = rng.sample(clean, k=min(5, len(clean)))
        ctxs = [snip.get(p, "") for p in picks]
        if kind == "sort_asc":
            q = f"Sort these topics alphabetically: {', '.join(picks)}. Output as a comma-separated list."
            a = ", ".join(sorted(picks, key=lambda s: s.lower()))
        elif kind == "sort_desc":
            q = f"Sort these topics in reverse alphabetical order: {', '.join(picks)}. Output as a comma-separated list."
            a = ", ".join(sorted(picks, key=lambda s: s.lower(), reverse=True))
        elif kind == "group_prefix":
            buckets: Dict[str, List[str]] = {}
            for p in picks:
                key = p[:1].upper()
                buckets.setdefault(key, []).append(p)
            q = f"Group these topics by first letter: {', '.join(picks)}. Format 'A: x, y; B: z'."
            parts = []
            for k in sorted(buckets.keys()):
                parts.append(f"{k}: {', '.join(sorted(buckets[k], key=lambda s: s.lower()))}")
            a = "; ".join(parts)
        elif kind == "count_words":
            target = rng.choice(picks)
            t = (snip.get(target, "") or "").strip()
            words = [w for w in t.split() if w]
            q = f"How many words are in the description for '{target}'? Answer with an integer only."
            a = str(len(words))
        else:  # reverse
            q = f"Reverse this list of topics: {', '.join(picks)}. Output as a comma-separated list."
            a = ", ".join(list(reversed(picks)))
        out.append((q, ctxs, a))
    return out


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Build Algorithmic Thinking RLWHF dataset (K3D hints)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--mode", default="rlwhf")
    ap.add_argument("--gltf", required=True)
    args = ap.parse_args()
    ids, labels, snip = load_k3d(Path(args.gltf))
    tasks = build_tasks(labels, snip, int(args.n))
    # Optional ST reward based on answer-context similarity
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import torch  # type: ignore
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=("cuda" if torch.cuda.is_available() else "cpu"))
        def cosine(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    except Exception:
        st = None
        cosine = lambda a,b: 0.0  # type: ignore
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for q, ctxs, a in tasks:
            if st is not None and ctxs:
                try:
                    e1 = st.encode([a], convert_to_numpy=True)[0]
                    blob = "\n".join([c for c in ctxs if c][:4])
                    e2 = st.encode([blob], convert_to_numpy=True)[0]
                    sim = cosine(e1, e2)
                    reward = 1.0 if sim >= 0.70 else (0.5 if sim >= 0.40 else 0.2)
                except Exception:
                    reward = 0.6
            else:
                reward = 0.6  # neutral positive weight for algorithmic correctness
            f.write(json.dumps({"query": q, "answer": a, "contexts": ctxs, "reward": reward}, ensure_ascii=False) + "\n")
    print(str(outp))


if __name__ == "__main__":
    main()

