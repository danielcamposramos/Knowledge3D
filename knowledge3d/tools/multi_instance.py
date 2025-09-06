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


def _extract_graph(path: Path) -> Tuple[List[str], List[List[str]], List[str], List[Dict[str,str]]]:
    """Return (ids, neighbors, labels, doors) from a K3D GLTF/GLB.

    doors: list of {label, address} if metadata marks type=='door'. Address falls
    back to k3d:// label form.
    """
    try:
        obj = _load_gltf_json(path)
    except Exception:
        return [], [], [], []
    ids: List[str] = []
    neighbors: List[List[str]] = []
    labels: List[str] = []
    doors: List[Dict[str, str]] = []
    for m in (obj.get("meshes") or []):
        for p in (m.get("primitives") or []):
            ex = (p.get("extras") or {})
            k3d = ex.get("k3d") or {}
            kid = ex.get("k3dIds") or []
            if k3d and kid:
                ids = list(kid)
                neighbors = list(k3d.get("neighbors") or [[] for _ in range(len(ids))])
                md = k3d.get("metadata") or []
                if isinstance(md, list) and md:
                    for i in range(len(ids)):
                        lab = None
                        try:
                            lab = (md[i] or {}).get("label")
                        except Exception:
                            lab = None
                        labels.append(lab if isinstance(lab, str) and lab.strip() else ids[i])
                    # doors
                    for i in range(min(len(md), len(labels))):
                        try:
                            t = (md[i] or {}).get("type")
                            if t == "door":
                                lab = labels[i]
                                doors.append({"label": lab, "address": f"k3d://@?label={lab}"})
                        except Exception:
                            continue
                else:
                    labels = ids[:]
                break
        if ids:
            break
    return ids, neighbors, labels, doors


def extract_labels(path: Path, max_n: int = 256) -> List[str]:
    ids, _neighbors, labels, _doors = _extract_graph(path)
    if labels:
        # Trim long labels at em-/en-dash for cleaner commands
        labels = [ (s.split(" — ")[0].split(" – ")[0] if isinstance(s, str) else str(s)) for s in labels ]
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
        if ln not in nav_phrases:
            ln = 'en'
        # Scout
        scout.append(random.choice(nav_phrases[ln]).format(x=lab))
        scout.append(random.choice(move_dirs[ln]).format(d=random.choice([1, 2, 3])))
        # Gardener
        gardener.append(random.choice(show_phrases[ln]).format(x=lab))
        gardener.append(random.choice(show_phrases[ln]).format(x=lab))
    return scout, gardener


async def _agent(url: str, nick: str, lines: Iterable[str], delay: float, jitter: float = 0.0):
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
        import random as _rnd
        for line in lines:
            payload = json.dumps({"type": "chat", "from": nick, "text": line})
            print(f"[{nick}] send -> {line}")
            await ws.send(payload)
            dt = delay + ( _rnd.uniform(0.0, jitter) if jitter > 0 else 0.0 )
            await asyncio.sleep(max(0.0, dt))
        await asyncio.sleep(1.0)
        task.cancel()
        with contextlib.suppress(Exception):
            await task


async def _maintenance_agent(url: str, interval: float) -> None:
    if interval <= 0:
        return
    try:
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({"type": "chat", "from": "human", "text": "/nick janitor"}))
            while True:
                # ask status then trigger compression sweep
                await ws.send(json.dumps({"type": "chat", "from": "janitor", "text": "/logs status"}))
                await ws.send(json.dumps({"type": "chat", "from": "janitor", "text": "/logs compress"}))
                await asyncio.sleep(interval)
    except Exception:
        # Run silent if maintenance cannot attach
        await asyncio.sleep(interval)


async def main_async(url: str, gltf: Path, count: int, delay: float, langs: List[str], workers: int = 1, jitter: float = 0.0, rounds: int = 1, maint_interval: float = 0.0) -> None:
    # Push dataset graph to server once before spawning agents
    ids, neigh, labels_full, doors = _extract_graph(gltf)
    if ids and neigh:
        try:
            async with websockets.connect(url) as ws:
                await ws.send(json.dumps({"type": "chat", "from": "human", "text": "/nick loader"}))
                await ws.send(json.dumps({"type": "event", "event": {"kind": "dataset_graph", "ids": ids, "neighbors": neigh, "labels": labels_full}}))
                if doors:
                    await ws.send(json.dumps({"type": "event", "event": {"kind": "doors", "items": doors}}))
        except Exception:
            pass
    labels = extract_labels(gltf, max_n=count)
    scout_msgs, gardener_msgs = make_phrases(labels, langs)
    # Interleave limited slices
    scout_msgs = scout_msgs[:count]
    gardener_msgs = gardener_msgs[:count]
    tasks: list[asyncio.Task] = []
    # Maintenance task
    if maint_interval and maint_interval > 0:
        tasks.append(asyncio.create_task(_maintenance_agent(url, maint_interval)))
    # Worker groups (each spawns two agents)
    for r in range(max(1, rounds)):
        for w in range(max(1, workers)):
            s_nick = f"scout{r+1}-{w+1}" if workers > 1 or rounds > 1 else "scout"
            g_nick = f"gardener{r+1}-{w+1}" if workers > 1 or rounds > 1 else "gardener"
            tasks.append(asyncio.create_task(_agent(url, s_nick, scout_msgs, delay, jitter)))
            tasks.append(asyncio.create_task(_agent(url, g_nick, gardener_msgs, delay, jitter)))
    await asyncio.gather(*tasks)


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Run two concurrent chat agents to generate training logs")
    ap.add_argument("--url", default="ws://127.0.0.1:8765")
    ap.add_argument("--gltf", default="viewer/public/k3d_foundation.6k.umap.glb")
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--langs", default="en,pt,es", help="Comma-separated languages to sample: en,pt,es")
    ap.add_argument("--workers", type=int, default=1, help="Number of agent pairs to run concurrently")
    ap.add_argument("--rounds", type=int, default=1, help="Repeat the message set this many times")
    ap.add_argument("--jitter", type=float, default=0.0, help="Random jitter added to delay (seconds)")
    ap.add_argument("--maint-interval", type=float, default=0.0, help="Seconds between /logs compress sweeps (0=off)")
    args = ap.parse_args()
    langs = [s.strip() for s in str(args.langs).split(',') if s.strip() in ("en","pt","es")]
    if not langs:
        langs = ["en"]
    asyncio.run(main_async(args.url, Path(args.gltf), args.count, args.delay, langs, workers=int(args.workers), jitter=float(args.jitter), rounds=int(args.rounds), maint_interval=float(args.maint_interval)))


if __name__ == "__main__":
    main()
