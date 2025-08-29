from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Any, Optional


try:
    import websockets
except Exception as exc:  # pragma: no cover
    raise SystemExit("Install websockets to use CLI client: pip install websockets") from exc


@dataclass
class Options:
    url: str = "ws://127.0.0.1:8765"
    nick: Optional[str] = None
    auto: bool = False
    model_path: Optional[str] = None


async def run_cli(opts: Options) -> None:
    model = None
    predict = None
    if opts.auto and opts.model_path:
        try:
            from ..models.intent_classifier import load_model, predict_action  # type: ignore
            model = load_model(__import__('pathlib').Path(opts.model_path))
            predict = predict_action
            print(f"[cli] loaded model from {opts.model_path}")
        except Exception as e:  # pragma: no cover
            print(f"[cli] failed to load model: {e}")
            model = None
            predict = None
    async with websockets.connect(opts.url) as ws:
        print(f"[cli] connected to {opts.url}")
        # basic recv loop task
        async def recv_loop():
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    print(raw)
                    continue
                t = msg.get("type")
                if t == "chat":
                    who = msg.get("from")
                    txt = msg.get("text")
                    ch = msg.get("channel") or "#general"
                    print(f"[{ch}] {who}: {txt}")
                    if opts.auto and predict and model and who != "agent":
                        action, conf = predict(model, str(txt))
                        if conf >= 0.7:
                            reply = f"[model:{conf:.2f}] intent={action}"
                        else:
                            reply = f"[model:{conf:.2f}] unsure; try /help"
                        await ws.send(json.dumps({"type": "chat", "from": "agent", "text": reply}))
                elif t == "command":
                    cmd = msg.get("command")
                    target = msg.get("target")
                    ch = msg.get("channel") or "#general"
                    print(f"[{ch}] <command {cmd}> {target}")
                else:
                    print(json.dumps(msg))

        recv_task = asyncio.create_task(recv_loop())
        # stdin loop
        for line in sys.stdin:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("/"):
                # raw command; pass through as chat to server which will parse slash
                await ws.send(json.dumps({"type": "chat", "from": opts.nick or "human", "text": line}))
            else:
                await ws.send(json.dumps({"type": "chat", "from": opts.nick or "human", "text": line}))
        await recv_task


def main():  # pragma: no cover
    import argparse
    p = argparse.ArgumentParser(description="Headless CLI chat client for K3D live server")
    p.add_argument("--url", default="ws://127.0.0.1:8765", help="WebSocket URL")
    p.add_argument("--nick", help="Nickname to send in messages")
    p.add_argument("--auto", action="store_true", help="Enable model auto-replies")
    p.add_argument("--model", dest="model_path", help="Path to trained model for auto mode")
    a = p.parse_args()
    asyncio.run(run_cli(Options(url=a.url, nick=a.nick, auto=a.auto, model_path=a.model_path)))


if __name__ == "__main__":
    main()

