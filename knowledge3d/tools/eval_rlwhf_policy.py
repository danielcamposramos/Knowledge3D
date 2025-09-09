from __future__ import annotations

"""
Evaluate an RLWHF policy by generating answers and measuring semantic alignment
to the provided contexts using a SentenceTransformer cosine similarity.

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.tools.eval_rlwhf_policy \
    --dataset docs/reports/training/rlwhf_dataset.jsonl \
    --model ../Knowledge3D.local/models/rlwhf_policy \
    --out docs/reports/status/rlwhf_policy_eval.json
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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


def build_prompt(q: str, ctxs: List[str]) -> str:
    sys = (
        "You are K3D's integrated LLM skill. You must answer using ONLY the provided contexts from the House memory. "
        "Do not invent facts. If information is missing, explicitly say 'I don't know' and suggest where to look next."
    )
    ctx_lines = []
    for t in ctxs[:4]:
        t = str(t or "")
        if not t:
            continue
        ctx_lines.append(f"- {t[:297] + '...' if len(t) > 300 else t}")
    return sys + "\n\n" + ("Context:\n" + "\n".join(ctx_lines) + "\n\n" if ctx_lines else "") + f"Question: {q}\n\nAnswer:"


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Evaluate RLWHF policy against contexts via ST cosine similarity")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()
    ds = list(iter_jsonl(Path(args.dataset)))
    ds = [r for r in ds if r.get("query") and r.get("contexts")]
    ds = ds[: int(args.limit)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    # ST encoder
    from sentence_transformers import SentenceTransformer  # type: ignore
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=("cuda" if torch.cuda.is_available() else "cpu"))

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    rows: List[dict] = []
    for rec in ds:
        q = str(rec.get("query"))
        ctxs = [str(x or "") for x in rec.get("contexts") or []]
        prompt = build_prompt(q, ctxs)
        ids = tok(prompt, return_tensors="pt").to(device)
        out = mdl.generate(**ids, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.7)
        txt = tok.decode(out[0], skip_special_tokens=True)
        ans = txt[len(prompt):].strip() if txt.startswith(prompt) else txt
        # Similarity to context blob
        blob = "\n".join(ctxs[:4])
        e1 = st.encode([ans], convert_to_numpy=True)[0]
        e2 = st.encode([blob], convert_to_numpy=True)[0]
        sim = cosine(e1, e2)
        rows.append({"query": q, "sim": sim, "answer_chars": len(ans)})
    sims = [r["sim"] for r in rows]
    out = {
        "count": len(rows),
        "sim_avg": (sum(sims) / len(sims)) if sims else None,
        "sim_p50": (sorted(sims)[len(sims)//2] if sims else None),
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

