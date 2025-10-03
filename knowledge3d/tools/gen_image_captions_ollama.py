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

import requests  # type: ignore


def iter_images(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*.png")):
        if p.is_file():
            yield p


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
        resp = requests.post(
            f"{url.rstrip('/')}/api/generate",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        obj = resp.json()
        return str(obj.get("response", "")).strip()
    except Exception:
        return ""


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Generate captions for images via Ollama vision models")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--model", required=True, help="Vision model (e.g., llama3.2-vision, qwen2.5vl:7b-q8_0)")
    ap.add_argument("--images-root", default="viewer/public/house/materialized_objects/docs")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cycle", type=int, default=20, help="Unload model every N images to clear context/memory")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    url = str(args.ollama)
    model = str(args.model)
    root = Path(args.images_root)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if out.exists():
        try:
            for line in out.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                img = obj.get("image")
                if isinstance(img, str):
                    seen.add(img)
        except Exception:
            pass
    # Warmup: ensure the model tag exists and keep it alive
    try:
        requests.get(f"{url.rstrip('/')}/api/tags", timeout=10)
        requests.post(
            f"{url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": "warmup", "stream": False, "keep_alive": "30m"},
            timeout=min(max(int(args.timeout), 60), 600),
        )
    except Exception:
        pass

    with out.open("a", encoding="utf-8") as f:
        written = 0
        for p in iter_images(root):
            if str(p) in seen:
                continue
            b64 = encode_b64(p)
            resp = ollama_generate_vision(url, model, "Provide a short, factual caption.", b64, timeout=int(args.timeout))
            if not resp:
                continue
            f.write(json.dumps({"image": str(p), "caption": resp}, ensure_ascii=False) + "\n")
            seen.add(str(p))
            written += 1
            # Periodically unload to respect context/memory limits
            if args.cycle and written % int(args.cycle) == 0:
                try:
                    requests.post(
                        f"{url.rstrip('/')}/api/generate",
                        json={"model": model, "prompt": "unload", "stream": False, "keep_alive": "0s"},
                        timeout=30,
                    )
                except Exception:
                    pass
            if args.limit and written >= int(args.limit):
                break
    print(str(out))


if __name__ == "__main__":
    main()
