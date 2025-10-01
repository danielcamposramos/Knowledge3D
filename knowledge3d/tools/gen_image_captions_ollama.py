from __future__ import annotations

"""
Generate image captions via local Ollama vision models (one model at a time).

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.tools.gen_image_captions_ollama \
    --ollama http://192.168.0.4:11434 --model qwen2.5vl:7b-q8_0 \
    --images-root viewer/public/house/materialized_objects/docs --limit 200 \
    --out ../Knowledge3D.local/datasets/image_captions_qwen25vl.jsonl
"""

import argparse
import base64
import json
from pathlib import Path
from typing import Iterable, List
import subprocess


def iter_images(root: Path, limit: int) -> Iterable[Path]:
    n = 0
    for p in root.rglob("*.png"):
        if p.is_file():
            yield p
            n += 1
            if limit and n >= limit:
                return


def encode_b64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def ollama_generate_vision(url: str, model: str, prompt: str, b64: str, timeout: int = 600) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64],
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
    ap = argparse.ArgumentParser(description="Generate captions for images via Ollama vision models")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--model", required=True, help="Vision model (e.g., llama3.2-vision, qwen2.5vl:7b-q8_0)")
    ap.add_argument("--images-root", default="viewer/public/house/materialized_objects/docs")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    url = str(args.ollama)
    model = str(args.model)
    root = Path(args.images_root)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    # Warmup: ensure the model tag exists and keep it alive
    try:
        subprocess.run(["curl", "-s", f"{url.rstrip('/')}/api/tags"], timeout=10)
        subprocess.run(["curl", "-s", f"{url.rstrip('/')}/api/generate", "-d", json.dumps({
            "model": model, "prompt": "warmup", "stream": False, "keep_alive": "30m"
        })], timeout=min(max(int(args.timeout), 60), 600))
    except Exception:
        pass

    with out.open("w", encoding="utf-8") as f:
        for p in iter_images(root, int(args.limit)):
            b64 = encode_b64(p)
            resp = ollama_generate_vision(url, model, "Provide a short, factual caption.", b64, timeout=int(args.timeout))
            if not resp:
                continue
            f.write(json.dumps({"image": str(p), "caption": resp}, ensure_ascii=False) + "\n")
    print(str(out))


if __name__ == "__main__":
    main()
