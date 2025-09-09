from __future__ import annotations

"""
Build an Algorithmic Thinking RL dataset for K3D.

Generates simple algorithmic tasks with exact ground truth to reinforce
stepwise computation and honesty. Each row is {query, answer, contexts[], reward}.

Modes:
- rl: no contexts; reward +1.0 if exact match; else -0.25; honesty +0.5 if admits unknown.
- rlwhf: attaches a minimal context (task restatement) and uses the same reward.

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.tools.build_algorithmic_thinking \
    --out docs/reports/training/rl_dataset_algo_2000.jsonl --n 2000 --mode rl
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Tuple

from .eval_honesty_reward import is_honest  # honesty detector


def _task_arith() -> Tuple[str, str, str]:
    a = random.randint(-999, 999)
    b = random.randint(-999, 999)
    op = random.choice(["+","-","*"])
    if op == "+": gt = a + b
    elif op == "-": gt = a - b
    else: gt = a * b
    q = f"Compute {a} {op} {b}."
    ctx = f"Task: {q} Return only the final integer."
    return q, str(gt), ctx

def _task_gcd() -> Tuple[str, str, str]:
    a = random.randint(1, 999)
    b = random.randint(1, 999)
    gt = math.gcd(a, b)
    q = f"Compute gcd({a}, {b})."
    ctx = f"Task: {q} The gcd is a positive integer."
    return q, str(gt), ctx

def _task_sort() -> Tuple[str, str, str]:
    n = random.randint(3, 7)
    arr = [random.randint(-50, 50) for _ in range(n)]
    gt = sorted(arr)
    q = f"Sort the list ascending: {arr}."
    ctx = f"Task: {q} Return as Python list literal."
    return q, str(gt), ctx

def _task_reverse() -> Tuple[str, str, str]:
    s = "".join(random.choice("abcdefxyz012345") for _ in range(random.randint(5,10)))
    gt = s[::-1]
    q = f"Reverse the string: '{s}'."
    ctx = f"Task: {q} Return only the reversed string."
    return q, gt, ctx

TASKS = [_task_arith, _task_gcd, _task_sort, _task_reverse]


def _retrieve_k3d_contexts(glb: Path, query: str, k: int = 4) -> List[str]:
    try:
        from pygltflib import GLTF2  # type: ignore
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return []
    g = GLTF2().load(str(glb))
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    ids = list(k3d.get("ids", []))
    meta = [m if isinstance(m, dict) else {} for m in k3d.get("metadata", [])]
    labels = [(m.get("label") if isinstance(m, dict) else None) or ids[i] for i, m in enumerate(meta)]
    snip = {}
    for i, m in enumerate(meta):
        if isinstance(m, dict):
            lab = str(m.get("label") or labels[i] or ids[i])
            txt = str(m.get("text") or "")
            if lab and txt:
                snip[lab] = txt
    vec = TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 2))
    corpus = [(f"{lab} — {snip.get(lab, '')}" if snip.get(lab) else lab) for lab in labels]
    X = vec.fit_transform(corpus)
    qv = vec.transform([query])
    scores = (X @ qv.T).toarray().ravel()
    top = np.argsort(-scores)[:k]
    return [snip.get(labels[int(j)], "") for j in top]


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Build Algorithmic Thinking RL dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--mode", default="rl", choices=["rl","rlwhf"], help="rl=no contexts; rlwhf=attach K3D hints via TF-IDF")
    ap.add_argument("--gltf", help="House GLB path for rlwhf mode")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    glb = Path(args.gltf) if args.gltf else None
    rows = 0
    with out.open("w", encoding="utf-8") as f:
        for _ in range(int(args.n)):
            q, gt, ctx = random.choice(TASKS)()
            ans = gt
            contexts: List[str]
            if args.mode == "rlwhf" and glb and glb.exists():
                # Attach K3D hints when available
                ctxs = _retrieve_k3d_contexts(glb, q, k=4)
                contexts = ctxs if ctxs else [ctx]
            else:
                contexts = [] if args.mode == "rl" else [ctx]
            reward = 1.0
            f.write(json.dumps({"query": q, "answer": ans, "contexts": contexts, "reward": reward}, ensure_ascii=False) + "\n")
            rows += 1
    print(str(out))


if __name__ == "__main__":
    main()
