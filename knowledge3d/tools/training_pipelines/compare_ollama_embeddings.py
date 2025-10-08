from __future__ import annotations

"""
Compare embeddings from multiple Ollama embedding models on the same prompts.

Writes a markdown report with a simple table: prompt (truncated), model, dim,
L2 norm, and the first 6 coefficients.

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.tools.compare_ollama_embeddings \
    --ollama http://192.168.0.4:11434 \
    --models qwen3-embedding:4b,embeddinggemma,snowflake-arctic-embed2 \
    --prompts "energia; conhecimento; sistemas; probability theory; computer vision" \
    --out docs/reports/status/embedding_comparison_2025-10-01.md
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import subprocess
import math


def call_embeddings(url: str, model: str, text: str, timeout: int = 120) -> List[float]:
    payload = {"model": model, "prompt": text}
    r = subprocess.run(
        ["curl", "-s", f"{url.rstrip('/')}/api/embeddings", "-d", json.dumps(payload)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if r.returncode != 0:
        return []
    try:
        obj = json.loads(r.stdout)
        v = obj.get("embedding") or []
        return [float(x) for x in v]
    except Exception:
        return []


def trunc(s: str, n: int = 36) -> str:
    s = s.strip().replace("\n", " ")
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Compare Ollama embeddings across models")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--models", required=True, help="Comma-separated models")
    ap.add_argument("--prompts", required=True, help="Semicolon-separated prompts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    url = str(args.ollama)
    models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    prompts = [p.strip() for p in str(args.prompts).split(";") if p.strip()]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Embedding Model Comparison\n")
    lines.append("Prompt | Model | Dim | L2 Norm | First 6 Coeffs")
    lines.append("---|---|---:|---:|---")
    for p in prompts:
        for m in models:
            v = call_embeddings(url, m, p)
            dim = len(v)
            if dim == 0:
                lines.append(f"{trunc(p)} | {m} | 0 | 0.0 | (n/a)")
                continue
            norm = math.sqrt(sum(x * x for x in v))
            preview = ", ".join(f"{x:.3f}" for x in v[:6])
            lines.append(f"{trunc(p)} | {m} | {dim} | {norm:.3f} | [{preview}]")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()

