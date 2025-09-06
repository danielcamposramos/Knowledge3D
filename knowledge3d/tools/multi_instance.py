"""
Spawn two concurrent chat agents (scout + gardener) to explore a house and
generate training logs in the live server. This accelerates data collection by
running multiple instances in the same space.

Usage
  python3 -m knowledge3d.tools.multi_instance \
    --gltf viewer/public/k3d_foundation.6k.umap.glb \
    --url ws://127.0.0.1:8765 --count 100 --delay 0.5

Notes
- Extracts labels from primitive.extras.k3d.metadata[].label when available.
- Falls back to simple synthetic topics if labels are missing.
- Logs are written by the live server as chat + chat_response pairs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import struct
from pathlib import Path
from typing import Iterable, List, Tuple, Dict
import contextlib

try:
    import websockets  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Install websockets: python3 -m pip install --user --break-system-packages websockets"
    ) from exc


def _read_json_from_glb(path: Path) -> dict:
    with path.open("rb") as f:
        header = f.read(12)
        if len(header) < 12:
            raise RuntimeError("Invalid GLB header")
        magic, version, length = struct.unpack("<III", header)
        if magic != 0x46546C67:  # b'glTF'
            raise RuntimeError("Not a GLB file")
        # First chunk should be JSON
        chunk_header = f.read(8)
        if len(chunk_header) < 8:
            raise RuntimeError("Invalid GLB chunk header")
        chunk_length, chunk_type = struct.unpack("<II", chunk_header)
        if chunk_type != 0x4E4F534A:  # b'JSON'
            raise RuntimeError("First GLB chunk is not JSON")
        json_bytes = f.read(chunk_length)
        text = json_bytes.decode("utf-8")
        return json.loads(text)


def _load_gltf_json(path: Path) -> dict:
    if path.suffix.lower() == ".glb":
        return _read_json_from_glb(path)
    # glTF JSON
    return json.loads(path.read_text(encoding="utf-8"))


def extract_labels(path: Path, max_n: int = 256) -> List[str]:
    try:
        obj = _load_gltf_json(path)
    except Exception:
        return []
    labels: List[str] = []
    for m in (obj.get("meshes") or []):
        for p in (m.get("primitives") or []):
            k3d = (p.get("extras") or {}).get("k3d")
            if not k3d:
                continue
            md = k3d.get("metadata") or []
            if isinstance(md, list):
                for it in md:
                    lab = (it or {}).get("label")
                    if isinstance(lab, str) and lab.strip():
                        s = lab.strip()
                        # Trim long labels at em-dash / en-dash for cleaner commands
                        s = s.split(" — ")[0].split(" – ")[0]
                        labels.append(s)
            # only first embedded block is needed
            break
        if labels:
            break
    # Unique and sample
    seen = set()
    uniq = []
    for s in labels:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    if not uniq:
        # fallback topics
        uniq = [
            "embeddings", "neural networks", "graphs", "physics", "math",
            "quantum", "clustering", "routing", "agents", "ethics",
        ]
    random.shuffle(uniq)
    return uniq[:max_n]


def make_phrases(labels: List[str], langs: List[str]) -> Tuple[List[str], List[str]]:
    """Return (scout_msgs, gardener_msgs) with multilingual variants."""
    scout: List[str] = []
    gardener: List[str] = []

    # Phrase banks per language
    nav_phrases: Dict[str, List[str]] = {
        "en": ["go to {x}", "please take me to {x}", "navigate to {x}"],
        "pt": ["ir para {x}", "por favor, leve-me para {x}", "navegar até {x}"],
        "es": ["ir a {x}", "por favor, llévame a {x}", "navegar a {x}"],
    }
    move_dirs: Dict[str, List[str]] = {
        "en": ["move left {d}", "move right {d}", "move forward {d}", "move back {d}"],
        "pt": ["mova para a esquerda {d}", "mova para a direita {d}", "mova para frente {d}", "mova para trás {d}"],
        "es": ["muévete a la izquierda {d}", "muévete a la derecha {d}", "avanza {d}", "retrocede {d}"],
    }
    show_phrases: Dict[str, List[str]] = {
        "en": ["show me {x}", "please highlight {x}", "expand {x}", "find related to {x}"],
        "pt": ["mostre {x}", "por favor, destaque {x}", "expandir {x}", "encontrar relacionado a {x}"],
        "es": ["mostrar {x}", "por favor, resalta {x}", "expandir {x}", "encontrar relacionado con {x}"],
    }

    for lab in labels:
        ln = random.choice(langs)
        # Scout
        scout.append(random.choice(nav_phrases[ln]).format(x=lab))
        scout.append(random.choice(move_dirs[ln]).format(d=random.choice([1, 2, 3])))
        # Gardener
        gardener.append(random.choice(show_phrases[ln]).format(x=lab))
        gardener.append(random.choice(show_phrases[ln]).format(x=lab))
    return scout, gardener


async def _agent(url: str, nick: str, lines: Iterable[str], delay: float):
    async with websockets.connect(url) as ws:
        # Set nickname
        await ws.send(json.dumps({"type": "chat", "from": "human", "text": f"/nick {nick}"}))
        # Basic recv loop in background
        async def recv_loop():
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    print(raw)
                    continue
                t = msg.get("type")
                if t == "chat":
                    ch = msg.get("channel") or "#general"
                    who = msg.get("from")
                    txt = msg.get("text")
                    print(f"[{ch}] {nick} recv <- {who}: {txt}")
                elif t == "command":
                    print(f"[{nick}] cmd <- {msg.get('command')} {msg.get('target')}")
        task = asyncio.create_task(recv_loop())
        # Send scripted lines
        for line in lines:
            payload = json.dumps({"type": "chat", "from": nick, "text": line})
            print(f"[{nick}] send -> {line}")
            await ws.send(payload)
            await asyncio.sleep(delay)
        await asyncio.sleep(1.0)
        task.cancel()
        with contextlib.suppress(Exception):
            await task


async def main_async(url: str, gltf: Path, count: int, delay: float, langs: List[str]) -> None:
    labels = extract_labels(gltf, max_n=count)
    scout_msgs, gardener_msgs = make_phrases(labels, langs)
    # Interleave limited slices
    scout_msgs = scout_msgs[:count]
    gardener_msgs = gardener_msgs[:count]
    # Run both agents concurrently
    await asyncio.gather(
        _agent(url, "scout", scout_msgs, delay),
        _agent(url, "gardener", gardener_msgs, delay),
    )


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Run two concurrent chat agents to generate training logs")
    ap.add_argument("--url", default="ws://127.0.0.1:8765")
    ap.add_argument("--gltf", default="viewer/public/k3d_foundation.6k.umap.glb")
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--langs", default="en,pt,es", help="Comma-separated languages to sample: en,pt,es")
    args = ap.parse_args()
    langs = [s.strip() for s in str(args.langs).split(',') if s.strip() in ("en","pt","es")]
    if not langs:
        langs = ["en"]
    asyncio.run(main_async(args.url, Path(args.gltf), args.count, args.delay, langs))


if __name__ == "__main__":
    main()
