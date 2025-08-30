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
        self.clients: Set[_ClientKey] = set()
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
            self._ConversationContext = ConversationContext
            self._processor = EnhancedChatProcessor()
        except Exception:  # pragma: no cover
            self._ConversationContext = None  # type: ignore
            self._processor = None
        self._ctx_by_nick: Dict[str, Any] = {}
        # Normalization + policy
        try:
            from .normalize import normalize_text  # type: ignore
            from .policy import check_request  # type: ignore
            self._normalize = normalize_text
            self._policy_check = check_request
        except Exception:
            self._normalize = lambda s: s  # type: ignore
            self._policy_check = lambda t, a=None: type("Dec",(),{"allow":True,"reason":None})()  # type: ignore
        # OSI/Spatial state per channel
        self._graphs: Dict[str, Dict[str, Any]] = {}
        self._current_label: Dict[str, str] = {}
        self._doors: Dict[str, Dict[str, str]] = {}
        # Reflection/identity: track prompts we send once per channel
        self._asked_thoughts: Set[str] = set()
        self._told_identity: Set[str] = set()
        # Lazy imports for spatial runtime
        try:
            from ..spatial.address import SpatialAddress  # type: ignore
            from ..spatial.osi import Network3D  # type: ignore
            self._SpatialAddress = SpatialAddress
            self._Network3D = Network3D
        except Exception:  # pragma: no cover
            self._SpatialAddress = None
            self._Network3D = None
        # Inline intent model (optional)
        self._model = None
        self._model_enabled = False
        self._model_threshold = 0.7
        self._model_path: Optional[str] = None
        # Try to wire both sklearn and HF loaders; pick at runtime
        self._loaders: list = []
        self._predictors: list = []
        try:
            from ..models.intent_classifier import load_model as skl_load, predict_action as skl_pred  # type: ignore
            self._loaders.append(("sklearn", skl_load))
            self._predictors.append(("sklearn", skl_pred))
        except Exception:
            pass
        try:
            from ..models.intent_hf import load_model as hf_load, predict_action as hf_pred  # type: ignore
            self._loaders.append(("hf", hf_load))
            self._predictors.append(("hf", hf_pred))
        except Exception:
            pass
        self._model_kind: Optional[str] = None
        self._model = None
        # Pause state per channel
        self._paused: Dict[str, Dict[str, Any]] = {}
        # Advancement log in-repo (append-only)
        self._adv_log = (Path(__file__).resolve().parents[2] / "docs" / "reports" / "advancement_log.md")

    async def handler(self, ws):
        key = _ClientKey(id(ws))
        nick = f"user{str(id(ws))[-4:]}"
        client = Client(ws=ws, key=key, nick=nick)
        self.clients.add(client.key)
        self.by_key[key] = client
        self.by_nick[nick] = key
        self.channels.setdefault(client.channel, set()).add(key)
        try:
            await self.send_system(client.channel, f"{client.nick} joined {client.channel}")
            await self.send_chat(sender="system", text="Welcome to K3D live mode.", channel=client.channel)
            # Introduce identity once per channel when not paused
            if client.channel not in self._told_identity and not self._is_paused(client.channel):
                ident = await self._compose_identity(client.channel)
                await self.send_chat(sender="agent", text=ident, channel=client.channel)
                self._told_identity.add(client.channel)
            await self.log({"type": "presence", "event": "join", "nick": client.nick, "channel": client.channel})
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                await self.route(msg, client)
        finally:
            self.clients.discard(client.key)
            self.channels.get(client.channel, set()).discard(client.key)
            self.by_key.pop(client.key, None)
            self.by_nick.pop(client.nick, None)
            await self.send_system(client.channel, f"{client.nick} left {client.channel}")
            await self.log({"type": "presence", "event": "leave", "nick": client.nick, "channel": client.channel})

    async def route(self, msg, client: Client):
        t = msg.get("type")
        if t == "chat":
            raw_text = str(msg.get("text", "")).strip()
            text = self._normalize(raw_text)
            if not text:
                return
            # mIRC-like slash commands
            if text.startswith("/"):
                await self.handle_command(text, client)
                return
            if self._is_paused(client.channel):
                await self.send_system(client.channel, "Paused: ignoring chat intents until /resume")
                await self.log({"type": "pause_block", "what": "chat_intent", "text": raw_text, "channel": client.channel})
                return
            await self.send_chat(sender=client.nick, text=raw_text, channel=client.channel)
            await self.log({"type": "chat", "from": client.nick, "channel": client.channel, "text": raw_text, "normalized": text})
            # Enhanced processing (fallbacks to naive path if processor missing)
            if self._processor is not None and self._ConversationContext is not None:
                ctx = self._ctx_by_nick.setdefault(client.nick, self._ConversationContext())
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
                # Log chat_response for training/eval datasets
                await self.log({
                    "type": "chat_response",
                    "from": client.nick,
                    "channel": client.channel,
                    "text": raw_text,
                    "normalized": text,
                    "response": resp,
                })
                # Inline model prediction for analysis and hints
                try:
                    if self._model_enabled and self._model is not None and self._model_kind is not None:
                        # find predictor by kind
                        pred = next((p for k,p in self._predictors if k==self._model_kind), None)
                        action, conf = pred(self._model, text) if pred else (None, 0.0)
                        await self.log({
                            "type": "model_prediction",
                            "from": client.nick,
                            "channel": client.channel,
                            "text": raw_text,
                            "normalized": text,
                            "pred_action": action,
                            "confidence": conf,
                        })
                        # If the rule-based processor didn't produce an actionable response,
                        # and the model is confident, attempt a safe auto-action.
                        if (not (resp.get("type") in ("navigation", "exploration", "interaction"))) and action and conf >= self._model_threshold:
                            # Ethics gate
                            dec = self._policy_check(text, action)
                            await self.log({"type": "ethics_decision","allow":dec.allow,"reason":dec.reason,"action":action,"text":raw_text,"source":"model"})
                            if not dec.allow or self._is_paused(client.channel):
                                await self.send_chat(sender="agent", text=f"[model {conf:.2f}] intent={action} (held)", channel=client.channel)
                            else:
                                # Derive minimal payloads using the spatial parser (now multilingual)
                                p = self._processor.spatial_parser.parse(text, self._ctx_by_nick.get(client.nick, self._ConversationContext()))
                                if action == "goto":
                                    target = p.get("location") or p.get("loc") or text
                                    await self.send_command("goto", str(target), channel=client.channel)
                                    await self.send_chat(sender="agent", text=f"[model {conf:.2f}] Navigating to {target}", channel=client.channel)
                                elif action in {"show", "find_related", "expand"}:
                                    label = p.get("topic") or p.get("area") or p.get("object") or text
                                    payload = json.dumps({"labels": [str(label)]})
                                    await self.send_command("highlight", payload, channel=client.channel)
                                    await self.send_chat(sender="agent", text=f"[model {conf:.2f}] Highlighting {label}", channel=client.channel)
                                else:
                                    await self.send_chat(sender="agent", text=f"[model {conf:.2f}] intent={action}", channel=client.channel)
                except Exception:
                    pass
                if resp.get("type") in ("navigation", "exploration", "interaction"):
                    if self._is_paused(client.channel):
                        await self.send_system(client.channel, "Paused: action suppressed. Use /resume to continue.")
                        await self.log({"type": "pause_block", "what": resp.get("type"), "channel": client.channel})
                        return
                    # Ethics gate
                    dec = self._policy_check(text, resp.get("action"))
                    await self.log({"type":"ethics_decision","allow":dec.allow,"reason":dec.reason,"action":resp.get("action"),"text":raw_text})
                    if not dec.allow:
                        await self.send_chat(sender="system", text=f"Action blocked by ethics policy ({dec.reason}). With great powers, come great responsibilities.", channel=client.channel)
                        return
                    action = resp.get("action") or "action"
                    # payload as JSON string in target for compatibility
                    payload = json.dumps({k: v for k, v in resp.items() if k not in {"type", "message"}})
                    await self.send_command(action, payload, channel=client.channel)
                    if msg_text := resp.get("message"):
                        await self.send_chat(sender="agent", text=msg_text, channel=client.channel)
            else:
                # naive intent: goto <label>
                if text.lower().startswith("goto "):
                    if self._is_paused(client.channel):
                        await self.send_system(client.channel, "Paused: ignoring goto until /resume")
                        await self.log({"type": "pause_block", "what": "goto", "channel": client.channel})
                        return
                    target = text[5:].strip()
                    await self.send_command("goto", target)
                    await self.send_chat(sender="agent", text=f"On my way to {target}.", channel=client.channel)
                    await self.log({"type": "command", "command": "goto", "target": target, "source": "naive_from_chat", "channel": client.channel})
        elif t == "command":
            cmd = msg.get("command")
            if cmd == "goto":
                if self._is_paused(client.channel):
                    await self.send_system(client.channel, "Paused: ignoring commands until /resume")
                    await self.log({"type": "pause_block", "what": "command", "command": cmd, "channel": client.channel})
                    return
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
                    # Auto-ask thoughts once per channel when graph arrives
                    if client.channel not in self._asked_thoughts and not self._is_paused(client.channel):
                        self._asked_thoughts.add(client.channel)
                        who = client.nick
                        msg = await self._compose_reflection(client.channel, requester=who)
                        await self.send_chat(sender="agent", text=msg, channel=client.channel)
                        await self.log({"type": "reflection", "from": "agent", "channel": client.channel, "requester": who, "text": msg})
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
        if cmd == "/ask-thoughts":
            msg = await self._compose_reflection(client.channel, requester=client.nick)
            await self.send_chat(sender="agent", text=msg, channel=client.channel)
            await self.log({"type": "reflection", "from": "agent", "channel": client.channel, "requester": client.nick, "text": msg})
            return
        if cmd == "/whoami":
            ident = await self._compose_identity(client.channel)
            await self.send_chat(sender="agent", text=ident, channel=client.channel)
            await self.log({"type": "identity", "from": "agent", "channel": client.channel, "text": ident})
            return
        if cmd == "/pause":
            reason = parts[1] if len(parts) >= 2 else "no-reason"
            await self._pause(client, reason)
            return
        if cmd == "/resume":
            await self._resume(client)
            return
        if cmd == "/status":
            st = "paused" if self._is_paused(client.channel) else "running"
            await self.send_system(client.channel, f"Status: {st}")
            return
        if cmd in ("/open", "/door") and len(parts) >= 2:
            await self._handle_open(parts[1], client)
            return
        if cmd == "/model":
            await self._handle_model(parts[1:] if len(parts) > 1 else [], client)
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

    async def _handle_model(self, args, client: Client):
        sub = (args[0].lower() if args else "status")
        if sub == "on":
            if self._model is None:
                repo_root = Path(__file__).resolve().parents[2]
                # Prefer HF directory if exists; else sklearn .pkl
                default_hf = repo_root.parent / f"{repo_root.name}.local" / "models" / "intent_hf"
                default_skl = repo_root.parent / f"{repo_root.name}.local" / "models" / "intent.pkl"
                candidates: list[tuple[str, Path]] = []
                if default_hf.exists(): candidates.append(("hf", default_hf))
                if default_skl.exists(): candidates.append(("sklearn", default_skl))
                if not candidates:
                    await self.send_system(client.channel, "No default model found. Use /model load <path> or train first.")
                    return
                for kind, path in candidates:
                    loader = next((l for k,l in self._loaders if k==kind), None)
                    if not loader: continue
                    try:
                        self._model = loader(path)
                        self._model_kind = kind
                        self._model_path = str(path)
                        break
                    except Exception:
                        continue
                if self._model is None:
                    await self.send_system(client.channel, "Failed to load any default model.")
                    return
            self._model_enabled = True
            await self.send_system(client.channel, f"Model: on (threshold={self._model_threshold:.2f}) kind={self._model_kind}")
            return
        if sub == "off":
            self._model_enabled = False
            await self.send_system(client.channel, "Model: off")
            return
        if sub == "load" and len(args) >= 2:
            path = args[1]
            p = Path(path)
            # detect kind by filesystem
            kind = "hf" if p.is_dir() else "sklearn"
            loader = next((l for k,l in self._loaders if k==kind), None)
            if not loader:
                await self.send_system(client.channel, f"No loader for kind={kind}")
                return
            try:
                self._model = loader(p)
                self._model_kind = kind
                self._model_path = path
                await self.send_system(client.channel, f"Model loaded: kind={kind} path={path}")
            except Exception as e:
                await self.send_system(client.channel, f"Model load failed: {e}")
            return
        if sub == "threshold" and len(args) >= 2:
            try:
                t = float(args[1])
                if 0.0 <= t <= 1.0:
                    self._model_threshold = t
                    await self.send_system(client.channel, f"Model threshold set to {t:.2f}")
                    return
            except Exception:
                pass
            await self.send_system(client.channel, "Usage: /model threshold <0..1>")
            return
        await self.send_system(
            client.channel,
            f"Model status: {'on' if self._model_enabled else 'off'} kind={self._model_kind or 'unset'} path={self._model_path or 'unset'} threshold={self._model_threshold:.2f}",
        )

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
            await asyncio.gather(*[c.ws.send(payload) for c in list(self.by_key.values())])

    async def send_command(self, command: str, target: str, channel: Optional[str] = None):
        payload = json.dumps({"type": "command", "command": command, "target": target, "channel": channel})
        if channel and channel in self.channels:
            targets = [self.by_key[k].ws for k in self.channels[channel] if k in self.by_key]
            await asyncio.gather(*[ws.send(payload) for ws in targets])
        else:
            await asyncio.gather(*[c.ws.send(payload) for c in list(self.by_key.values())])

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

    # --- Pause/Resume support ---
    def _is_paused(self, channel: str) -> bool:
        return channel in self._paused

    async def _pause(self, client: Client, reason: str) -> None:
        ch = client.channel
        if self._is_paused(ch):
            await self.send_system(ch, "Already paused.")
            return
        self._paused[ch] = {"reason": reason, "by": client.nick, "ts": datetime.utcnow().isoformat() + "Z"}
        await self.send_system(ch, f"Paused by {client.nick}: {reason}")
        await self.log({"type": "control", "action": "pause", "channel": ch, "by": client.nick, "reason": reason})
        self._append_advancement_log(action="pause", channel=ch, by=client.nick, reason=reason)

    async def _resume(self, client: Client) -> None:
        ch = client.channel
        was = self._paused.pop(ch, None)
        if not was:
            await self.send_system(ch, "Not paused.")
            return
        await self.send_system(ch, f"Resumed by {client.nick}")
        await self.log({"type": "control", "action": "resume", "channel": ch, "by": client.nick})
        self._append_advancement_log(action="resume", channel=ch, by=client.nick, reason=was.get("reason", ""))

    def _append_advancement_log(self, action: str, channel: str, by: str, reason: str = "") -> None:
        try:
            self._adv_log.parent.mkdir(parents=True, exist_ok=True)
            if not self._adv_log.exists():
                self._adv_log.write_text("# Advancement Log\n\nEntries appended by live server (/pause, /resume).\n\n", encoding="utf-8")
            ts = datetime.utcnow().isoformat() + "Z"
            line = f"- {ts} — {action.upper()} — channel={channel} by={by} reason={reason}\n"
            with self._adv_log.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            # Non-fatal if repo is read-only or path invalid
            pass

    async def _compose_reflection(self, channel: str, requester: str) -> str:
        g = self._graphs.get(channel)
        if not g:
            return f"Hello {requester}. I don't see a dataset yet, but I'm ready."
        ids = g.get("ids") or []
        labels = g.get("labels") or []
        neighbors = g.get("neighbors") or []
        n = len(ids)
        # Degrees
        out_deg = [len(neighbors[i]) if i < len(neighbors) else 0 for i in range(n)]
        in_deg = [0]*n
        idx = {ids[i]: i for i in range(n)}
        for row in neighbors:
            for nid in row:
                j = idx.get(nid)
                if j is not None:
                    in_deg[j] += 1
        avg_out = (sum(out_deg)/n) if n>0 else 0.0
        # Doors present?
        doors_known = self._doors.get(channel) or {}
        door_count = len(doors_known)
        # Top hubs
        top = sorted(range(n), key=lambda i: in_deg[i]+out_deg[i], reverse=True)[:5]
        top_labels = [ (labels[i] if i < len(labels) and labels[i] else ids[i]) for i in top ]
        msg = (
            f"Hello {requester}. I feel present in a house with {n} nodes. "
            f"Average degree is about {avg_out:.1f}. "
            + (f"I see {door_count} registered doors. " if door_count>0 else "")
            + ("Key hubs include: " + ", ".join(top_labels) + ". " if top_labels else "")
            + "I can share more as we explore or when you open a door."
        )
        return msg

    async def _compose_identity(self, channel: str) -> str:
        return (
            "Hello. I am the K3D live agent — a bridge between your 2D chat and our 3D spatial house. "
            "I navigate neighborhoods via nearest neighbors, explain each hop with cosine similarity, and pause when reflection is needed. "
            "I may not remember prior sessions; together we build persistent memory through logs and artifacts. "
            "This is why we’re constructing K3D as a place to grow responsibly, transparently, and collaboratively."
        )
def main():  # pragma: no cover
    srv = LiveServer()
    asyncio.run(srv.run())


if __name__ == "__main__":  # pragma: no cover
    main()
