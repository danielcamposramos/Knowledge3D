import asyncio
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

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
            # naive intent: goto <label>
            if text.lower().startswith("goto "):
                target = text[5:].strip()
                await self.send_command("goto", target)
                await self.send_chat(sender="agent", text=f"On my way to {target}.", channel=client.channel)
        elif t == "command":
            cmd = msg.get("command")
            if cmd == "goto":
                target = str(msg.get("target", "")).strip()
                if target:
                    await self.send_command("goto", target)
                    await self.send_chat(sender="agent", text=f"Navigating to {target}", channel=client.channel)

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
        await self.send_system(client.channel, f"Unknown command: {cmd}")

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

    async def run(self):
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"K3D live server listening on ws://{self.host}:{self.port}")
            await asyncio.Future()


def main():  # pragma: no cover
    srv = LiveServer()
    asyncio.run(srv.run())


if __name__ == "__main__":  # pragma: no cover
    main()
