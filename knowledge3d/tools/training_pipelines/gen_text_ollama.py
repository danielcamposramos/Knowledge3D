from __future__ import annotations

"""
Generate topic-coherent text via local Ollama models.

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.tools.gen_text_ollama \
    --ollama http://192.168.0.4:11434 \
    --models exaone3.5:latest,granite3.3:8b,gemma3:12b,gemma3n \
    --topics physics,biology,engineering,ethics,ai,systems,mathematics,economics,history,art \
    --per-topic 50 \
    --out ../Knowledge3D.local/datasets/text_gen_ollama_v1.txt
"""

import argparse
import json
import os
from pathlib import Path
from typing import List
import time
import subprocess


SYSTEM_PROMPT = (
    "You are a concise domain writer. Generate short, factual, self-contained lines (max 24 words). "
    "Use clear language and avoid lists in a single output; produce one line per generation."
)


def ollama_generate(url: str, model: str, prompt: str, timeout: int = 180) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "keep_alive": "10m",
    }
    try:
        r = subprocess.run(
            ["curl", "-s", f"{url.rstrip('/')}/api/generate", "-d", json.dumps(payload)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if r.returncode != 0:
            return ""
        obj = json.loads(r.stdout)
        return (obj.get("response") or "").strip()
    except Exception:
        return ""


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Generate topic-coherent text via Ollama")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--models", required=True, help="Comma-separated models")
    ap.add_argument(
        "--topics",
        default="physics,biology,engineering,ethics,ai,systems,mathematics,economics,history,art",
    )
    ap.add_argument("--per-topic", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    url = str(args.ollama)
    models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    topics = [t.strip() for t in str(args.topics).split(",") if t.strip()]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for model in models:
            # warmup tag list
            try:
                subprocess.run(["curl", "-s", f"{url.rstrip('/')}/api/tags"], timeout=10)
            except Exception:
                pass
            for topic in topics:
                for i in range(int(args.per_topic)):
                    prompt = f"Topic: {topic}. Write one compact, factual line covering a specific sub-point."
                    resp = ollama_generate(url, model, prompt)
                    if not resp:
                        # brief backoff and retry once
                        time.sleep(2.0)
                        resp = ollama_generate(url, model, prompt, timeout=240)
                    if not resp:
                        continue
                    f.write(f"[{model}][{topic}] {resp}\n")
                f.flush()
    print(str(out))


if __name__ == "__main__":
    main()

