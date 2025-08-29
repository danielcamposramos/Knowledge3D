import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set

try:
    import websockets
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: websockets. Install with: python3 -m pip install --user --break-system-packages websockets"
    ) from exc


@dataclass(eq=True, frozen=True)
class _ClientKey:
    ident: int


@dataclass
class Client:
    ws: websockets.WebSocketServerProtocol
    key: _ClientKey
    nick: str = field(default_factory=lambda: "user")
    channel: str = field(default_factory=lambda: "#general")


class LiveServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self.clients: Set[Client] = set()
        self.channels: Dict[str, Set[_ClientKey]] = {"#general": set()}
        self.by_key: Dict[_ClientKey, Client] = {}
        self.by_nick: Dict[str, _ClientKey] = {}
        # Logging setup: write JSONL to ../<repo>.local/logs/session-TS.jsonl
        repo_root = Path(__file__).resolve().parents[2]
        local_root = repo_root.parent / f"{repo_root.name}.local"
        self.log_dir = local_root / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self.session_file = self.log_dir / f"session-{ts}.jsonl"
        self._log_lock = asyncio.Lock()
        # Enhanced chat state
        try:
            from .enhanced_chat_processor import EnhancedChatProcessor, ConversationContext  # type: ignore
        except Exception:  # pragma: no cover
            EnhancedChatProcessor = None  # type: ignore
            ConversationContext = None  # type: ignore
        self._processor = EnhancedChatProcessor() if EnhancedChatProcessor else None
        self._ctx_by_nick: Dict[str, Any] = {}
        # OSI/Spatial state per channel
        self._graphs: Dict[str, Dict[str, Any]] = {}
        self._current_label: Dict[str, str] = {}
        self._doors: Dict[str, Dict[str, str]] = {}
        # Lazy imports for spatial runtime
        try:
            from ..spatial.address import SpatialAddress  # type: ignore
            from ..spatial.osi import Network3D  # type: ignore
            self._SpatialAddress = SpatialAddress
            self._Network3D = Network3D
        except Exception:  # pragma: no cover
            self._SpatialAddress = None
            self._Network3D = None

    async def handler(self, ws):
        key = _ClientKey(id(ws))
        nick = f"user{str(id(ws))[-4:]}"
        client = Client(ws=ws, key=key, nick=nick)
        self.clients.add(client)
        self.by_key[key] = client
        self.by_nick[nick] = key
        self.channels.setdefault(client.channel, set()).add(key)
        try:
            await self.send_system(client.channel, f"{client.nick} joined {client.channel}")
            await self.send_chat(sender="system", text="Welcome to K3D live mode.", channel=client.channel)
            await self.log({"type": "presence", "event": "join", "nick": client.nick, "channel": client.channel})
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                await self.route(msg, client)
        finally:
            self.clients.discard(client)
            self.channels.get(client.channel, set()).discard(client.key)
            self.by_key.pop(client.key, None)
            self.by_nick.pop(client.nick, None)
            await self.send_system(client.channel, f"{client.nick} left {client.channel}")
            await self.log({"type": "presence", "event": "leave", "nick": client.nick, "channel": client.channel})

    async def route(self, msg, client: Client):
        t = msg.get("type")
        if t == "chat":
            text = str(msg.get("text", "")).strip()
            if not text:
                return
            # mIRC-like slash commands
            if text.startswith("/"):
                await self.handle_command(text, client)
                return

            await self.send_chat(sender=client.nick, text=text, channel=client.channel)
            await self.log({"type": "chat", "from": client.nick, "channel": client.channel, "text": text})
            # Enhanced processing (fallbacks to naive path if processor missing)
            if self._processor is not None:
                ctx = self._ctx_by_nick.setdefault(client.nick, ConversationContext())
                resp = self._processor.process_message(text, ctx)
                ctx.update(text, resp)
                await self.send_json(
                    {
                        "type": "chat_response",
                        "response": resp,
                        "context": ctx.get_relevant_context(text),
                        "suggestions": resp.get("suggestions"),
                    },
                    channel=client.channel,
                )
                if resp.get("type") in ("navigation", "exploration", "interaction"):
                    action = resp.get("action") or "action"
                    # payload as JSON string in target for compatibility
                    payload = json.dumps({k: v for k, v in resp.items() if k not in {"type", "message"}})
                    await self.send_command(action, payload, channel=client.channel)
                    if msg_text := resp.get("message"):
                        await self.send_chat(sender="agent", text=msg_text, channel=client.channel)
            else:
                # naive intent: goto <label>
                if text.lower().startswith("goto "):
                    target = text[5:].strip()
                    await self.send_command("goto", target)
                    await self.send_chat(sender="agent", text=f"On my way to {target}.", channel=client.channel)
                    await self.log({"type": "command", "command": "goto", "target": target, "source": "naive_from_chat", "channel": client.channel})
        elif t == "command":
            cmd = msg.get("command")
            if cmd == "goto":
                target = str(msg.get("target", "")).strip()
                if target:
                    await self.send_command("goto", target)
                    await self.send_chat(sender="agent", text=f"Navigating to {target}", channel=client.channel)
                    await self.log({"type": "command", "command": "goto", "target": target, "source": "ws", "channel": client.channel})
        elif t == "event":
            ev = msg.get("event", {})
            kind = ev.get("kind")
            # Capture dataset graph for routing
            if kind == "dataset_graph":
                ids = ev.get("ids") or []
                neighbors = ev.get("neighbors") or []
                labels = ev.get("labels") or []
                if isinstance(ids, list) and isinstance(neighbors, list):
                    self._graphs[client.channel] = {"ids": ids, "neighbors": neighbors, "labels": labels}
            # Track agent arrival for current label baseline
            if kind == "explain":
                txt = str(ev.get("text") or "")
                if txt.startswith("Arrived at "):
                    # Arrived at <label> (addr=...)
                    try:
                        head = txt.split(" (addr=", 1)[0]
                        label = head.replace("Arrived at ", "", 1)
                        if label:
                            self._current_label[client.channel] = label
                    except Exception:
                        pass
            await self.log({"type": "event", **ev, "nick": client.nick, "channel": client.channel})

    async def handle_command(self, text: str, client: Client):
        parts = text.split(maxsplit=2)
        cmd = parts[0].lower()
        if cmd in ("/join", "/j") and len(parts) >= 2:
            await self.join_channel(client, parts[1])
            return
        if cmd in ("/nick", "/n") and len(parts) >= 2:
            await self.change_nick(client, parts[1])
            return
        if cmd == "/me" and len(parts) >= 2:
            await self.send_chat(sender=client.nick, text=parts[1], channel=client.channel, action=True)
            return
        if cmd == "/msg" and len(parts) >= 3:
            target_nick, msg = parts[1], parts[2]
            await self.private_msg(client, target_nick, msg)
            return
        if cmd in ("/open", "/door") and len(parts) >= 2:
            await self._handle_open(parts[1], client)
            return
        await self.send_system(client.channel, f"Unknown command: {cmd}")

    async def _handle_open(self, target: str, client: Client):
        # Resolve label from k3d URI or raw label text
        if self._SpatialAddress is None or self._Network3D is None:
            await self.send_system(client.channel, "Spatial runtime unavailable.")
            return
        label = None
        address = None
        if target.startswith("k3d://"):
            try:
                addr = self._SpatialAddress.from_uri(target)
                label = addr.label
                address = target
            except Exception:
                pass
        if not label:
            label = target
        graph = self._graphs.get(client.channel)
        if not graph:
            await self.send_system(client.channel, "No dataset graph registered yet.")
            return
        # If doors are registered for this channel, restrict to known labels
        doors = self._doors.get(client.channel) or {}
        if doors:
            if not label or label not in doors:
                await self.send_system(client.channel, f"Unknown door: {label}. Use one of: {', '.join(sorted(doors.keys()))}")
                return
        ids = graph.get("ids") or []
        neighbors = graph.get("neighbors") or []
        labels = graph.get("labels") or []
        # pick start by current label if known
        start_label = self._current_label.get(client.channel)
        try:
            start_idx = labels.index(start_label) if start_label in labels else 0
        except Exception:
            start_idx = 0
        try:
            target_idx = labels.index(label) if label in labels else -1
        except Exception:
            target_idx = -1
        if target_idx < 0:
            await self.send_system(client.channel, f"Unknown label: {label}")
            return
        start_id = ids[start_idx]
        target_id = ids[target_idx]
        path_ids = self._Network3D.route(ids, neighbors, start_id, target_id)
        # compose payload and broadcast as a command
        resolved_addr = address or doors.get(label) or f"k3d://@?label={label}"
        payload = {"label": label, "address": resolved_addr, "path": path_ids or []}
        await self.send_command("open", json.dumps(payload), channel=client.channel)
        hops = max(0, (len(path_ids or []) - 1))
        await self.send_chat(sender="system", text=f"Opening door to {label} via {hops} hops", channel=client.channel)

    async def change_nick(self, client: Client, new_nick: str):
        new_nick = new_nick.strip()
        if not new_nick or new_nick in self.by_nick:
            await self.send_system(client.channel, f"Nick unavailable: {new_nick}")
            return
        old = client.nick
        # update dataclass field
        client.nick = new_nick  # type: ignore
        self.by_nick.pop(old, None)
        self.by_nick[new_nick] = client.key
        await self.send_system(client.channel, f"{old} is now known as {new_nick}")

    async def join_channel(self, client: Client, channel: str):
        if not channel.startswith("#"):
            channel = f"#{channel}"
        old = client.channel
        if old == channel:
            return
        self.channels.setdefault(channel, set())
        self.channels[old].discard(client.key)
        self.channels[channel].add(client.key)
        client.channel = channel  # type: ignore
        await self.send_system(old, f"{client.nick} left {old}")
        await self.send_system(channel, f"{client.nick} joined {channel}")

    async def private_msg(self, sender: Client, target_nick: str, text: str):
        key = self.by_nick.get(target_nick)
        if not key:
            await self.send_system(sender.channel, f"No such nick: {target_nick}")
            return
        target = self.by_key.get(key)
        if not target:
            return
        payload = json.dumps({"type": "chat", "from": sender.nick, "to": target_nick, "text": text})
        await target.ws.send(payload)
        await sender.ws.send(payload)
        await self.log({"type": "pm", "from": sender.nick, "to": target_nick, "text": text})

    async def send_chat(self, sender: str, text: str, channel: Optional[str] = None, action: bool = False):
        payload = json.dumps({"type": "chat", "from": sender, "text": text, "channel": channel, "action": action})
        if channel and channel in self.channels:
            targets = [self.by_key[k].ws for k in self.channels[channel] if k in self.by_key]
            await asyncio.gather(*[ws.send(payload) for ws in targets])
        else:
            await asyncio.gather(*[c.ws.send(payload) for c in list(self.clients)])

    async def send_command(self, command: str, target: str, channel: Optional[str] = None):
        payload = json.dumps({"type": "command", "command": command, "target": target, "channel": channel})
        if channel and channel in self.channels:
            targets = [self.by_key[k].ws for k in self.channels[channel] if k in self.by_key]
            await asyncio.gather(*[ws.send(payload) for ws in targets])
        else:
            await asyncio.gather(*[c.ws.send(payload) for c in list(self.clients)])

    async def send_system(self, channel: str, text: str):
        await self.send_chat(sender="system", text=text, channel=channel)

    async def log(self, record: Dict):
        # attach timestamp (UTC, ISO8601)
        rec = {"ts": datetime.utcnow().isoformat() + "Z", **record}
        async with self._log_lock:
            with self.session_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    async def send_json(self, obj: Dict[str, Any], channel: Optional[str] = None):
        payload = json.dumps(obj)
        if channel and channel in self.channels:
            targets = [self.by_key[k].ws for k in self.channels[channel] if k in self.by_key]
            await asyncio.gather(*[ws.send(payload) for ws in targets])
        else:
            await asyncio.gather(*[c.ws.send(payload) for c in list(self.clients)])

    async def run(self):
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"K3D live server listening on ws://{self.host}:{self.port}")
            await asyncio.Future()


            # Capture door registry
            if kind == "doors":
                items = ev.get("items") or []
                if isinstance(items, list):
                    door_map: Dict[str, str] = {}
                    for it in items:
                        try:
                            lab = str(it.get("label") or "").strip()
                            addr = str(it.get("address") or "").strip()
                            if lab:
                                door_map[lab] = addr
                        except Exception:
                            continue
                    if door_map:
                        self._doors[client.channel] = door_map
def main():  # pragma: no cover
    srv = LiveServer()
    asyncio.run(srv.run())


if __name__ == "__main__":  # pragma: no cover
    main()
