import asyncio
import json
import os
import socket
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set, List

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
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, fast_start: Optional[bool] = None) -> None:
        self.host = host
        self.port = int(port)
        # Fast-start mode: delay heavy optional imports and model loads until after the server is listening
        try:
            self._fast_start = (str(os.getenv("K3D_LIVE_FAST", "1")).strip() != "0") if fast_start is None else bool(fast_start)
        except Exception:
            self._fast_start = True
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
        self._session_ts = ts
        self._session_idx = 0
        self.session_file = self.log_dir / f"session-{ts}.jsonl"
        self._log_lock = asyncio.Lock()
        # Log maintenance policy (env-configurable)
        try:
            self._log_rotate_bytes = int(os.getenv("K3D_LOG_ROTATE_BYTES", str(1024 * 1024 * 1024)))
        except Exception:
            self._log_rotate_bytes = 1024 * 1024 * 1024
        try:
            self._log_compress_age_hours = int(os.getenv("K3D_LOG_COMPRESS_AGE_HOURS", "24"))
        except Exception:
            self._log_compress_age_hours = 24
        try:
            self._log_maint_period = int(os.getenv("K3D_LOG_MAINT_PERIOD", "60"))
        except Exception:
            self._log_maint_period = 60
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
        # Search caches per channel
        self._label_to_id: Dict[str, Dict[str, str]] = {}
        self._goto_cache: Dict[str, Dict[str, str]] = {}
        self._search_index: Dict[str, Dict[str, Any]] = {}
        # Reflection/identity: track prompts we send once per channel
        self._asked_thoughts: Set[str] = set()
        self._told_identity: Set[str] = set()
        # Chat to memory linkage (track last message id per channel)
        self._last_chat_msg: Dict[str, str] = {}
        # IRC-inspired bits: topics, flood control, message caps
        self._topics: Dict[str, str] = {}
        self._max_msg_len = 512
        self._flood_last: Dict[str, float] = {}
        self._flood_min_interval = 0.6  # seconds between messages per nick
        # Lazy imports for spatial runtime
        try:
            from ..spatial.address import SpatialAddress  # type: ignore
            from ..spatial.osi import Network3D  # type: ignore
            self._SpatialAddress = SpatialAddress
            self._Network3D = Network3D
        except Exception:  # pragma: no cover
            self._SpatialAddress = None
            self._Network3D = None
        # Optional TF-IDF embedding for open-vocab goto
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            import numpy as _np  # type: ignore
            self._TFIDF = TfidfVectorizer
            self._NP = _np
        except Exception:
            self._TFIDF = None  # type: ignore
            self._NP = None  # type: ignore
        # LLM skill (optional, with RAG) — transformers only
        try:
            from ..skills.llm import LLMSkill, LLMConfig  # type: ignore
            self._llm = LLMSkill(LLMConfig())
        except Exception:
            self._llm = None
        # Spatial text skill (memory-native composition)
        try:
            from ..skills.spatial_text import compose_answer  # type: ignore
            self._sp_compose = compose_answer
        except Exception:
            self._sp_compose = None
        # Unified cranium (multimodal core, optional)
        try:
            from ..cranium.core import CraniumCore  # type: ignore
            self._cranium = CraniumCore()
        except Exception:
            self._cranium = None
        # Gazetteer/canonicalizer (optional, tiny)
        try:
            from .gazetteer import build_gazetteer, match_gazetteer, canonicalize  # type: ignore
            self._build_gazetteer = build_gazetteer
            self._match_gazetteer = match_gazetteer
            self._canonicalize = canonicalize
        except Exception:
            self._build_gazetteer = None  # type: ignore
            self._match_gazetteer = None  # type: ignore
            self._canonicalize = lambda s: s  # type: ignore

        # Inline intent model (optional, ensemble-ready)
        self._model_enabled = False
        self._model_threshold = 0.7
        self._models: Dict[str, Any] = {}
        self._model_paths: Dict[str, str] = {}
        self._active_kinds: Set[str] = set()
        # Try to wire both sklearn and HF loaders/predictors
        self._loaders: list = []
        self._predictors: list = []
        try:
            from ..models.intent_classifier import load_model as skl_load, predict_action as skl_pred  # type: ignore
            self._loaders.append(("sklearn", skl_load))
            self._predictors.append(("sklearn", skl_pred))
        except Exception:
            pass
        # Heavy HF intent loader import and auto-loading: delay when fast_start enabled
        if not self._fast_start:
            try:
                repo_root = Path(__file__).resolve().parents[2]
                local_root = repo_root.parent / f"{repo_root.name}.local"
                env_path = os.getenv("K3D_MODEL")
                auto_on = True if str(os.getenv("K3D_MODEL_AUTO", "1")).strip() != "0" else False
                want_ensemble = True if str(os.getenv("K3D_MODEL_ENSEMBLE", "0")).strip() != "0" else False
                # Load HF loader (may import transformers)
                try:
                    from ..models.intent_hf import load_model as hf_load, predict_action as hf_pred  # type: ignore
                    self._loaders.append(("hf", hf_load))
                    self._predictors.append(("hf", hf_pred))
                except Exception:
                    pass
                candidates: list[tuple[str, Path]] = []
                if env_path:
                    p = Path(env_path)
                    kind = "hf" if p.is_dir() else "sklearn"
                    candidates.append((kind, p))
                else:
                    hf_dir = local_root / "models" / "intent_hf"
                    pkl = local_root / "models" / "intent.pkl"
                    if hf_dir.exists() and (hf_dir / "config.json").exists():
                        candidates.append(("hf", hf_dir))
                    if pkl.exists():
                        candidates.append(("sklearn", pkl))
                for kind, path in candidates:
                    loader = next((l for k,l in self._loaders if k==kind), None)
                    if not loader:
                        continue
                    try:
                        mdl = loader(path)
                        self._models[kind] = mdl
                        self._model_paths[kind] = str(path)
                        if want_ensemble:
                            self._active_kinds.add(kind)
                    except Exception:
                        continue
                if not self._active_kinds and self._models:
                    # default to hf if present, else any
                    self._active_kinds.add("hf" if "hf" in self._models else next(iter(self._models.keys())))
                self._model_enabled = auto_on and bool(self._active_kinds)
            except Exception:
                pass
        else:
            # fast start: postpone HF loader import and any model auto-load to warmup
            pass
        # Backward-compat shim removed: use _models/_active_kinds
        # Pause state per channel
        self._paused: Dict[str, Dict[str, Any]] = {}
        # Advancement log in-repo (append-only)
        self._adv_log = (Path(__file__).resolve().parents[2] / "docs" / "reports" / "advancement_log.md")
        # Autonomy state
        self._last_activity: Dict[str, float] = {}
        try:
            import time as _time
            self._now = _time.time  # type: ignore
        except Exception:  # pragma: no cover
            self._now = lambda: 0.0  # type: ignore
        import os as _os
        self._autonomy_enabled = (_os.getenv("K3D_AUTONOMY", "0").strip() != "0")
        # Idle threshold before autonomous action (seconds); default ~phi minute (≈ 37s)
        self._autonomy_idle = float(_os.getenv("K3D_AUTONOMY_IDLE_SEC", "37"))
        # Periodic check (seconds); default ~pi (≈ 9s)
        self._autonomy_period = float(_os.getenv("K3D_AUTONOMY_PERIOD_SEC", "9"))
        # Diary auto-writing
        self._last_diary_time: Dict[str, float] = {}
        self._diary_auto_enabled = (_os.getenv("K3D_DIARY_AUTO", "0").strip() != "0")
        self._diary_period = float(_os.getenv("K3D_DIARY_PERIOD_SEC", "600"))  # 10 minutes default
        self._diary_book = _os.getenv("K3D_DIARY_BOOK", "AI Diary")
        # Port scan state
        self._port_scan: Dict[str, Any] = {"tried": [], "chosen": None}

    def _is_port_in_use(self, host: str, port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.15)
                return s.connect_ex((host, int(port))) == 0
        except Exception:
            return False

    def _candidate_ports(self) -> list[int]:
        """Build candidate ports list: env K3D_LIVE_PORTS or near defaults.

        Order: [self.port, 8765, 8787, 8788, self.port+1..self.port+9]
        """
        cand: list[int] = []
        try:
            raw = os.getenv("K3D_LIVE_PORTS")
            if raw:
                xs = [int(x.strip()) for x in raw.split(",") if x.strip()]
                for x in xs:
                    if x not in cand:
                        cand.append(x)
        except Exception:
            pass
        base = [int(self.port), 8765, 8787, 8788]
        for x in base:
            if x not in cand:
                cand.append(x)
        for d in range(1, 10):
            x = int(self.port) + d
            if x not in cand:
                cand.append(x)
        return cand

    def _choose_free_port(self) -> int:
        tried: list[dict] = []
        chosen = None
        for p in self._candidate_ports():
            used = self._is_port_in_use(self.host, p)
            tried.append({"port": int(p), "used": bool(used)})
            if not used and chosen is None:
                chosen = int(p)
        if chosen is None:
            # fallback to original even if used; serve() will raise
            chosen = int(self.port)
        self._port_scan = {"tried": tried, "chosen": chosen}
        return chosen

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
            self._last_activity[client.channel] = self._now()
            # Introduce identity once per channel when not paused
            if client.channel not in self._told_identity and not self._is_paused(client.channel):
                ident = await self._compose_identity(client.channel)
                await self.send_chat(sender="agent", text=ident, channel=client.channel)
                self._told_identity.add(client.channel)
            await self.log({"type": "presence", "event": "join", "nick": client.nick, "channel": client.channel})
            # Inform clients about chosen port mapping (on first join)
            try:
                if self._port_scan.get("tried"):
                    m = ", ".join([f"{t['port']}={'used' if t['used'] else 'free'}" for t in self._port_scan["tried"][:6]])
                    await self.send_system(client.channel, f"LiveServer ports: {m} (chosen={self._port_scan.get('chosen')})")
            except Exception:
                pass
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
            # Flood control (per-nick)
            try:
                now = self._now()
                last = self._flood_last.get(client.nick, 0.0)
                if now - last < self._flood_min_interval:
                    await self.send_system(client.channel, "Slow down (flood control)")
                    return
                self._flood_last[client.nick] = now
            except Exception:
                pass
            # Message length cap (IRC-like 512 chars)
            if len(raw_text) > self._max_msg_len:
                raw_text = raw_text[: self._max_msg_len]
                text = text[: self._max_msg_len]
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
            self._last_activity[client.channel] = self._now()
            # Feed short‑term memory for unified cranium (if available)
            try:
                if self._cranium is not None:
                    self._cranium.observe_text(raw_text, label=client.nick)
            except Exception:
                pass
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
                # Inline model prediction (ensemble-capable)
                try:
                    if self._model_enabled and self._active_kinds:
                        preds: list[dict] = []
                        for kind in list(self._active_kinds):
                            mdl = self._models.get(kind)
                            pred = next((p for k,p in self._predictors if k==kind), None)
                            if mdl is None or pred is None:
                                continue
                            try:
                                action, conf = pred(mdl, text)
                                preds.append({"kind": kind, "action": action, "confidence": float(conf)})
                            except Exception:
                                continue
                        if preds:
                            chosen = max(preds, key=lambda r: (r.get("confidence") or 0.0, 1 if r.get("kind")=="hf" else 0))
                            await self.log({
                                "type": "model_prediction",
                                "from": client.nick,
                                "channel": client.channel,
                                "text": raw_text,
                                "normalized": text,
                                "predictions": preds,
                                "chosen": chosen,
                                "source": "ensemble" if len(preds) > 1 else chosen.get("kind"),
                            })
                            if (not (resp.get("type") in ("navigation", "exploration", "interaction"))):
                                action = chosen.get("action")
                                conf = float(chosen.get("confidence") or 0.0)
                                if action and conf >= self._model_threshold:
                                    dec = self._policy_check(text, action)
                                    await self.log({"type": "ethics_decision","allow":dec.allow,"reason":dec.reason,"action":action,"text":raw_text,"source":"model_ensemble"})
                                    scores = " ".join([f"{r['kind']}:{(r.get('confidence') or 0.0):.2f}" for r in preds])
                                    if not dec.allow or self._is_paused(client.channel):
                                        await self.send_chat(sender="agent", text=f"[models {scores}] intent={action} (held)", channel=client.channel)
                                    else:
                                        p = self._processor.spatial_parser.parse(text, self._ctx_by_nick.get(client.nick, self._ConversationContext()))
                                        if action == "goto":
                                            target = p.get("location") or p.get("loc") or text
                                            await self._dispatch_goto(client.channel, str(target), source="model", confidence=conf)
                                        elif action in {"show", "find_related", "expand"}:
                                            label = p.get("topic") or p.get("area") or p.get("object") or text
                                            payload = json.dumps({"labels": [str(label)]})
                                            await self.send_command("highlight", payload, channel=client.channel)
                                            await self.send_chat(sender="agent", text=f"[models {scores}] Highlighting {label}", channel=client.channel)
                                        else:
                                            await self.send_chat(sender="agent", text=f"[models {scores}] intent={action}", channel=client.channel)
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
                    if action == "goto":
                        target = str(resp.get("target") or resp.get("location") or "").strip()
                        await self._dispatch_goto(client.channel, target, source="rule")
                    else:
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
                    await self._dispatch_goto(client.channel, target, source="naive")
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
                    self._last_activity[client.channel] = self._now()
        elif t == "event":
            ev = msg.get("event", {})
            kind = ev.get("kind")
            self._last_activity[client.channel] = self._now()
            # Lightweight health check for readiness probes
            if kind == "healthz":
                await self.send_json({"type": "system", "text": "ok", "ts": datetime.utcnow().isoformat() + "Z"}, channel=client.channel)
                return
            # Capture dataset graph for routing
            if kind == "dataset_graph":
                ids = ev.get("ids") or []
                neighbors = ev.get("neighbors") or []
                labels = ev.get("labels") or []
                positions = ev.get("positions") or None
                if isinstance(ids, list) and isinstance(neighbors, list):
                    g: Dict[str, Any] = {"ids": ids, "neighbors": neighbors, "labels": labels}
                    # Optional positions for A* routing
                    if isinstance(positions, list):
                        try:
                            # validate shape minimally (list of triples)
                            if positions and isinstance(positions[0], (list, tuple)) and len(positions[0]) == 3:
                                g["positions"] = positions
                        except Exception:
                            pass
                    self._graphs[client.channel] = g
                    # build caches, gazetteer, and optional TF-IDF search index
                    try:
                        self._label_to_id[client.channel] = {str(labels[i] if i < len(labels) and labels[i] else ids[i]): ids[i] for i in range(len(ids))}
                        self._goto_cache.setdefault(client.channel, {})
                        if getattr(self, "_build_gazetteer", None) is not None:
                            self._search_index.setdefault(client.channel, {})
                            self._search_index[client.channel]["gazetteer"] = self._build_gazetteer(self._label_to_id[client.channel].keys())
                        if getattr(self, "_TFIDF", None) is not None:
                            vec = self._TFIDF(lowercase=True, analyzer='word', ngram_range=(1,2))
                            texts = [str(labels[i] if labels and i < len(labels) and labels[i] else ids[i]) for i in range(len(ids))]
                            X = vec.fit_transform(texts)
                            self._search_index.setdefault(client.channel, {})
                            self._search_index[client.channel].update({"vec": vec, "X": X, "labels": texts})
                    except Exception:
                        pass
            # Auto-ask thoughts once per channel when graph arrives
            if client.channel not in self._asked_thoughts and not self._is_paused(client.channel):
                self._asked_thoughts.add(client.channel)
                who = client.nick
                msg = await self._compose_reflection(client.channel, requester=who)
                await self.send_chat(sender="agent", text=msg, channel=client.channel)
                await self.log({"type": "reflection", "from": "agent", "channel": client.channel, "requester": who, "text": msg})
                # Event-based diary page after initial reflection (policy-gated)
                try:
                    if self._cranium is not None and getattr(self._cranium, "_stm", None) is not None:
                        vec = self._cranium._stm.snapshot_vector()
                        from ..tools.house_memory import MemoryHouse  # type: ignore
                        from ..cranium.diary import DiaryPolicy  # type: ignore
                        h = MemoryHouse(); pages = h.list_diary_pages(self._diary_book)
                        last = None
                        if pages:
                            e32 = (pages[-1].extra or {}).get("embedding32") if isinstance(pages[-1].extra, dict) else None
                            if isinstance(e32, list):
                                last = e32
                        pol = DiaryPolicy()
                        if pol.should_write(vec, last, event="reflect_init", meta={"by": who}):
                            pid = h.add_diary_page_embedding(self._diary_book, vec, meta={"event": "reflect_init", "by": who})
                            out = (Path(__file__).resolve().parents[2] / "viewer" / "public" / "houses" / (os.getenv("K3D_HOUSE_ID","default")) / "memory_house.gltf")
                            h.export_gltf(out)
                except Exception:
                    pass
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
            # Optional: aliases to enrich gazetteer (alias -> label)
            if kind == "aliases":
                try:
                    items = ev.get("items") or []
                    idx = self._search_index.get(client.channel)
                    if not idx:
                        self._search_index[client.channel] = {}
                        idx = self._search_index[client.channel]
                    gaz = idx.get("gazetteer") or {}
                    if getattr(self, "_canonicalize", None) is not None:
                        for it in items:
                            alias = str(it.get("alias") or "").strip()
                            label = str(it.get("label") or "").strip()
                            if not alias or not label:
                                continue
                            can = self._canonicalize(alias)
                            if not can:
                                continue
                            arr = gaz.setdefault(can, [])
                            if label not in arr:
                                arr.append(label)
                        self._search_index[client.channel]["gazetteer"] = gaz
                except Exception:
                    pass
            # Optional: dataset snippets (label, text) for RAG
            if kind == "dataset_snippets":
                try:
                    pairs = ev.get("pairs") or []
                    if isinstance(pairs, list) and pairs:
                        idx = self._search_index.setdefault(client.channel, {})
                        snip: Dict[str, str] = {}
                        for it in pairs:
                            try:
                                lab = str(it[0]); txt = str(it[1])
                                if lab:
                                    snip[lab] = txt
                            except Exception:
                                continue
                        idx["snip"] = snip
                        # Rebuild TF-IDF corpus using snippets when available
                        if self._TFIDF is not None:
                            labels = (self._graphs.get(client.channel) or {}).get("labels") or []
                            vec = self._TFIDF(lowercase=True, analyzer='word', ngram_range=(1,2))
                            corpus = []
                            for i in range(len(labels)):
                                lab = str(labels[i])
                                txt = snip.get(lab, "")
                                doc = f"{lab} — {txt}" if txt else lab
                                corpus.append(doc)
                            X = vec.fit_transform(corpus)
                            idx["vec"], idx["X"], idx["labels"] = vec, X, corpus
                except Exception:
                    pass
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
            # Diary entries: write to human-readable diary file in GMT-3
            if kind == "diary_entry":
                try:
                    from datetime import datetime, timezone, timedelta
                    tz = timezone(timedelta(hours=-3))
                    now = datetime.now(tz)
                    date = now.strftime('%Y-%m-%d')
                    clock = now.strftime('%H:%M')
                    text = str(ev.get('text') or '').strip()
                    if text:
                        diary_dir = Path(__file__).resolve().parents[2] / 'docs' / 'reports' / 'diary'
                        diary_dir.mkdir(parents=True, exist_ok=True)
                        diary_file = diary_dir / f'diary-{date}.md'
                        if not diary_file.exists():
                            diary_file.write_text(f"# Diary — {date} (GMT-3)\n\n", encoding='utf-8')
                        with diary_file.open('a', encoding='utf-8') as f:
                            f.write(f"- [{clock} -03:00] {client.nick}: {text}\n")
                        await self.send_chat(sender='agent', text=f"Diary updated for {date} ({clock} -03:00)", channel=client.channel)
                except Exception:
                    pass

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
            # Event-based diary page (AI-only, policy-gated)
            try:
                if self._cranium is not None and getattr(self._cranium, "_stm", None) is not None:
                    vec = self._cranium._stm.snapshot_vector()
                    from ..tools.house_memory import MemoryHouse  # type: ignore
                    from ..cranium.diary import DiaryPolicy  # type: ignore
                    h = MemoryHouse()
                    pages = h.list_diary_pages(self._diary_book)
                    last = None
                    if pages:
                        e32 = (pages[-1].extra or {}).get("embedding32") if isinstance(pages[-1].extra, dict) else None
                        if isinstance(e32, list):
                            last = e32
                    pol = DiaryPolicy()
                    meta = {"event": "reflect", "by": client.nick}
                    if pol.should_write(vec, last, event="reflect", meta=meta):
                        h.add_diary_page_embedding(self._diary_book, vec, meta=meta)
                        out = (Path(__file__).resolve().parents[2] / "viewer" / "public" / "houses" / (os.getenv("K3D_HOUSE_ID","default")) / "memory_house.gltf")
                        h.export_gltf(out)
            except Exception:
                pass
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
        if cmd == "/ai":
            # Simple chat interface: /ai <text>
            q = parts[1] if len(parts) > 1 else ""
            if not q:
                await self.send_system(client.channel, "Usage: /ai <text>")
                return
            out = f"AI processed: {q[:120]}..." if len(q) > 120 else f"AI processed: {q}"
            await self.send_chat(sender="agent", text=f"🤖 {out}", channel=client.channel)
            await self.log({"type": "chat", "mode": "ai_overlay", "text": q, "out": out})
            return
        if cmd == "/topic":
            # /topic -> show; /topic set <text> to change
            sub = parts[1] if len(parts) > 1 else "show"
            if sub == "show":
                topic = self._topics.get(client.channel) or "(no topic)"
                await self.send_system(client.channel, f"Topic for {client.channel}: {topic}")
            elif sub == "set" and len(parts) >= 3:
                topic = parts[2]
                self._topics[client.channel] = topic
                await self.send_system(client.channel, f"Topic set for {client.channel}: {topic}")
                await self.log({"type": "topic", "channel": client.channel, "topic": topic})
            else:
                await self.send_system(client.channel, "Usage: /topic [show|set <text>]")
            return
        if cmd == "/living":
            # /living render
            sub = parts[1] if len(parts) > 1 else "help"
            if sub == "render":
                try:
                    from ..tools.phase6.export_living_room import build_living_room  # type: ignore
                    out = (Path(__file__).resolve().parents[2] / 'viewer' / 'public' / 'living_room.glb')
                    build_living_room(20.0, 8.0, 20.0, str(out))
                    await self.send_chat(sender='agent', text=f"🛋️ Living room rendered: {out}", channel=client.channel)
                except Exception as e:
                    await self.send_system(client.channel, f"Living room render failed: {e}")
                return
            await self.send_system(client.channel, "Usage: /living render")
            return
        if cmd == "/names":
            # list nicks in channel
            members = sorted([self.by_key[k].nick for k in self.channels.get(client.channel, set()) if k in self.by_key])
            await self.send_system(client.channel, f"Names in {client.channel}: {', '.join(members) if members else '(none)'}")
            return
        if cmd == "/who":
            # alias to /names with simple output
            members = sorted([self.by_key[k].nick for k in self.channels.get(client.channel, set()) if k in self.by_key])
            await self.send_system(client.channel, f"{client.channel}: {len(members)} users — {', '.join(members)}")
            return
        if cmd == "/history":
            # emit last N messages from Chat Book to the requester as system lines
            try:
                N = int(parts[1]) if (len(parts) > 1 and parts[1].isdigit()) else 50
            except Exception:
                N = 50
            try:
                from ..tools.house_memory import MemoryHouse  # type: ignore
                h = MemoryHouse()
                book = f"Chat {client.channel}"
                # extract messages with ts and nick
                objs = [o for o in h.objects if o.kind == 'chat_message' and (o.extra or {}).get('parent') == h.ensure_chat_book(book)]
                # sort by ts
                def _ts(o):
                    try:
                        import datetime as _dt
                        return _dt.datetime.fromisoformat((o.extra or {}).get('ts','').replace('Z','')).timestamp()
                    except Exception:
                        return 0.0
                objs.sort(key=_ts)
                tail = objs[-max(1,N):]
                for o in tail:
                    nick = str((o.extra or {}).get('nick') or o.label.split(':',1)[0] or 'user')
                    msg = o.text or ''
                    await self.send_system(client.channel, f"[history] {nick}: {msg}")
            except Exception:
                await self.send_system(client.channel, "History unavailable.")
            return
        if cmd in ("/open", "/door") and len(parts) >= 2:
            # Special-case: /open book <title>
            try:
                if parts[1].lower().startswith("book") and len(parts) >= 3:
                    await self._handle_open_book(parts[2], client)
                    return
            except Exception:
                pass
            await self._handle_open(parts[1], client)
            return
        if cmd == "/grow" and len(parts) >= 3:
            # /grow tree <domain>
            sub = parts[1].lower()
            if sub == "tree":
                domain = parts[2]
                try:
                    from ..tools.phase2.grow_tree import GrowTreeCommand  # type: ignore
                    from ..tools.phase2.garden_renderer import render_garden  # type: ignore
                    cmdr = GrowTreeCommand()
                    msg = cmdr.execute(domain)
                    path = render_garden()
                    rel = "/" + str(Path(path).resolve().relative_to(Path(__file__).resolve().parents[2] / 'viewer' / 'public')).replace('\\','/')
                    await self.send_chat(sender="agent", text=msg + f" (asset: {rel})", channel=client.channel)
                except Exception:
                    await self.send_system(client.channel, "Grow tree failed.")
                return
        if cmd == "/repair" and len(parts) >= 3:
            # /repair tree <id>
            sub = parts[1].lower()
            if sub == "tree":
                tree_id = parts[2]
                try:
                    from ..tools.phase2.tree_repair import TreeRepair  # type: ignore
                    rep = TreeRepair()
                    ok = rep.auto_repair(tree_id)
                    await self.send_chat(sender='agent', text=(f"Repaired {tree_id}." if ok else f"Repair failed for {tree_id}."), channel=client.channel)
                except Exception:
                    await self.send_system(client.channel, "Repair failed.")
                return
        if cmd == "/move" and len(parts) >= 3:
            # /move tree <id> <angle> [distance]
            sub = parts[1].lower()
            if sub == "tree":
                try:
                    tid = parts[2]
                    ang = float(parts[3]) if len(parts) > 3 else 0.0
                    dist = float(parts[4]) if len(parts) > 4 else None
                except Exception:
                    await self.send_system(client.channel, "Usage: /move tree <id> <angle> [distance]")
                    return
                try:
                    from ..tools.phase2.registry import move_tree  # type: ignore
                    from ..tools.phase2.garden_renderer import render_garden  # type: ignore
                    ok = move_tree(tid, ang, dist)
                    if ok:
                        path = render_garden()
                        rel = "/" + str(Path(path).resolve().relative_to(Path(__file__).resolve().parents[2] / 'viewer' / 'public')).replace('\\','/')
                        await self.send_chat(sender='agent', text=f"🌳 Tree {tid} moved to {ang:.1f}° ({'d='+str(dist) if dist is not None else 'keep r'}) — asset: {rel}", channel=client.channel)
                    else:
                        await self.send_system(client.channel, "Move failed: tree not found")
                except Exception:
                    await self.send_system(client.channel, "Move failed.")
                return
        if cmd == "/model":
            await self._handle_model(parts[1:] if len(parts) > 1 else [], client)
            return
        if cmd == "/map":
            try:
                from ..tools.phase2.map_blueprint import main as _map_cli  # type: ignore
                # Generate to viewer/public/garden_map.svg
                from pathlib import Path as _P
                out = _P(__file__).resolve().parents[2] / 'viewer' / 'public' / 'garden_map.svg'
                # Call via small shim
                import sys as _sys
                _sys.argv = ['map_blueprint', '--garden', str(_P(__file__).resolve().parents[2] / 'viewer' / 'public' / 'knowledge_garden.glb'), '--out', str(out)]
                _map_cli()
                await self.send_chat(sender='agent', text='Blueprint written to /garden_map.svg', channel=client.channel)
            except Exception:
                await self.send_system(client.channel, 'Map generation failed.')
            return
        if cmd == "/logs":
            await self._handle_logs(parts[1:] if len(parts) > 1 else [], client)
            return
        if cmd == "/morph":
            # /morph <star_id> <dim> <value>
            try:
                star_id = parts[1]
                dim = int(parts[2])
                val = float(parts[3])
            except Exception:
                await self.send_system(client.channel, "Usage: /morph <star_id> <dim> <value>")
                return
            try:
                import os, json
                def _load_reg(p: str) -> dict:
                    try:
                        with open(p, 'r', encoding='utf-8') as f: return json.load(f)
                    except Exception: return {"objects": []}
                def _save_reg(reg: dict, p: str):
                    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
                    with open(p, 'w', encoding='utf-8') as f: json.dump(reg, f, indent=2)
                wpath = str((Path(__file__).resolve().parents[2] / 'viewer' / 'public' / 'workshop' / 'workshop_registry.json'))
                reg = _load_reg(wpath)
                star = next((s for s in reg.get('objects', []) if str(s.get('id')) == star_id), None)
                if not star:
                    await self.send_system(client.channel, f"Star not found: {star_id}")
                    return
                emb = star.get('embedding')
                if not isinstance(emb, list) or dim < 0 or dim >= len(emb):
                    await self.send_system(client.channel, f"Invalid dim {dim} (size={len(emb) if isinstance(emb, list) else 0})")
                    return
                emb[dim] = float(val)
                _save_reg(reg, wpath)
                await self.send_chat(sender='agent', text=f"✨ {star_id} morphed dim[{dim}]={val:.4f}", channel=client.channel)
            except Exception:
                await self.send_system(client.channel, "Morph failed.")
            return
        if cmd == "/export":
            # /export library <star_id> <title...>
            # /export garden <star_id> <domain>
            # /export bathtub <star_id>
            if len(parts) < 3:
                await self.send_system(client.channel, "Usage: /export library|garden|bathtub <star_id> [arg]")
                return
            kind = parts[1].lower()
            star_id = parts[2]
            tail = parts[3] if len(parts) > 3 else None
            try:
                import os, json
                from ..tools.phase4.export_to_house import HouseExporter  # type: ignore
                wpath = str((Path(__file__).resolve().parents[2] / 'viewer' / 'public' / 'workshop' / 'workshop_registry.json'))
                with open(wpath, 'r', encoding='utf-8') as f:
                    reg = json.load(f)
                star = next((s for s in reg.get('objects', []) if str(s.get('id')) == star_id), None)
                if not star:
                    await self.send_system(client.channel, f"Star not found: {star_id}")
                    return
                hx = HouseExporter()
                ok = False
                if kind == 'library':
                    title = tail or star_id
                    ok = hx.export_to_library(star, title)
                elif kind == 'garden':
                    domain = tail or 'Unknown'
                    ok = hx.export_to_garden(star, domain)
                elif kind == 'bathtub':
                    ok = hx.export_to_bathtub(star)
                else:
                    await self.send_system(client.channel, "Usage: /export library|garden|bathtub <star_id> [arg]")
                    return
                await self.send_chat(sender='agent', text=(f"Exported {star_id} → {kind}" if ok else f"Export failed for {star_id}"), channel=client.channel)
            except Exception:
                await self.send_system(client.channel, "Export failed.")
            return
        if cmd == "/install_app":
            # /install_app <name>
            name = parts[1] if len(parts) > 1 else ""
            if not name:
                await self.send_system(client.channel, "Usage: /install_app <name>")
                return
            try:
                from ..tools.phase6.avatar_tablet import AvatarTablet  # type: ignore
                if not hasattr(self, '_tablet') or self._tablet is None:
                    self._tablet = AvatarTablet()
                ok = self._tablet.install_app(name)
                if ok:
                    await self.send_chat(sender='agent', text=f"📱 Installed app '{name}' on avatar tablet.", channel=client.channel)
                else:
                    await self.send_system(client.channel, f"Install failed for '{name}'.")
            except Exception as e:
                await self.send_system(client.channel, f"Install failed: {e}")
            return
        if cmd == "/cast_to_screen":
            # /cast_to_screen <name> <screen_id>
            if len(parts) < 3:
                await self.send_system(client.channel, "Usage: /cast_to_screen <app_name> <screen_id>")
                return
            name, screen = parts[1], parts[2]
            try:
                from ..tools.phase6.avatar_tablet import AvatarTablet  # type: ignore
                if not hasattr(self, '_tablet') or self._tablet is None:
                    self._tablet = AvatarTablet()
                path = self._tablet.cast_to_screen(name, screen)
                if path:
                    # produce relative path for the viewer/public root
                    pub = Path(__file__).resolve().parents[2] / 'viewer' / 'public'
                    rel = "/" + str(Path(path).resolve().relative_to(pub)).replace('\\','/')
                    await self.send_chat(sender='agent', text=f"📺 Cast '{name}' to screen '{screen}' (projection {rel})", channel=client.channel)
                else:
                    await self.send_system(client.channel, f"Cast failed (app '{name}' not installed?).")
            except Exception as e:
                await self.send_system(client.channel, f"Cast failed: {e}")
            return
        if cmd == "/voice":
            # /voice say <text>
            sub = parts[1] if len(parts) > 1 else "";
            if sub != "say" or len(parts) < 3:
                await self.send_system(client.channel, "Usage: /voice say <text>")
                return
            text = parts[2]
            try:
                from ..tools.phase6.voice_chat import VoiceChat  # type: ignore
                # derive embedding from text (deterministic)
                import hashlib
                def _hash_vec(s: str, d: int = 32) -> list[float]:
                    h = hashlib.sha256(s.encode('utf-8')).digest(); v=[]; i=0
                    while len(v) < d:
                        b = h[i % len(h)]; v.append((b/255.0)-0.5); i+=1
                    return v
                emb = _hash_vec(text, 32)
                vc = VoiceChat()
                out_dir = Path(__file__).resolve().parents[2] / 'viewer' / 'public' / 'voice'
                path = vc.speak_to_file(text, emb, out_dir)
                pub = Path(__file__).resolve().parents[2] / 'viewer' / 'public'
                rel = "/" + str(path.resolve().relative_to(pub)).replace('\\','/')
                await self.send_chat(sender='agent', text=f"🔊 Spoke: '{text[:64]}' (audio {rel})", channel=client.channel)
                await self.log({"type":"voice","mode":"say","text":text,"audio":rel})
            except Exception as e:
                await self.send_system(client.channel, f"Voice synthesis failed: {e}")
            return
        if cmd == "/sleep":
            await self._handle_sleep(parts[1:] if len(parts) > 1 else [], client)
            return
        if cmd == "/upgrade":
            # /upgrade <furniture_id>
            if len(parts) < 2:
                await self.send_system(client.channel, "Usage: /upgrade <furniture_id>")
                return
            furn_id = parts[1].strip()
            repo_root = Path(__file__).resolve().parents[2]
            reg_path = repo_root / 'viewer' / 'public' / 'bathtub_registry.json'
            try:
                import json as _json
                if not reg_path.exists():
                    await self.send_system(client.channel, "Bathtub registry not found.")
                    return
                reg = _json.loads(reg_path.read_text(encoding='utf-8'))
                furn = None
                for f in list(reg.get('furniture', []) or []):
                    if str(f.get('id')) == furn_id:
                        furn = f
                        break
                if not furn:
                    await self.send_system(client.channel, f"Furniture {furn_id} not found in bathtub registry.")
                    return
                from ..tools.phase5.asset_upgrade_engine import AssetUpgradeEngine  # type: ignore
                eng = AssetUpgradeEngine()
                up = eng.upgrade_asset(furn)
                if up is furn or up == furn:
                    await self.send_system(client.channel, f"Furniture {furn_id} not eligible for upgrade (check honesty/dims).")
                    return
                # Replace in registry and save
                reg['furniture'] = [f if str(f.get('id')) != furn_id else up for f in reg.get('furniture', [])]
                reg_path.write_text(_json.dumps(reg, ensure_ascii=False, indent=2), encoding='utf-8')
                # Re-render scene
                from ..tools.phase5.bathtub_renderer import BathtubRenderer  # type: ignore
                scene_out = repo_root / 'viewer' / 'public' / 'bathtub_scene.glb'
                r = BathtubRenderer(str(reg_path), str(scene_out))
                out_path = r.render()
                gtype = ((up.get('geometry') or {}).get('type') or 'enhanced')
                await self.send_chat(sender='agent', text=f"✨ Upgraded {furn_id} → {gtype}. Scene: {out_path}", channel=client.channel)
                await self.log({'type': 'upgrade', 'furniture_id': furn_id, 'geometry_type': gtype})
            except Exception as e:
                await self.send_system(client.channel, f"Upgrade failed: {e}")
            return
        if cmd == "/llm":
            await self._handle_llm(parts[1:] if len(parts) > 1 else [], client)
            return
        if cmd == "/ask":
            await self._handle_k3d_ask(parts[1:] if len(parts) > 1 else [], client)
            return
        if cmd == "/workshop" and len(parts) >= 2:
            sub = parts[1].lower()
            if sub == "render":
                try:
                    from ..tools.phase4.workshop_renderer import WorkshopRenderer  # type: ignore
                    reg = str((Path(__file__).resolve().parents[2] / 'viewer' / 'public' / 'workshop' / 'workshop_registry.json'))
                    out = str((Path(__file__).resolve().parents[2] / 'viewer' / 'public' / 'workshop' / 'workshop_scene.glb'))
                    r = WorkshopRenderer(reg, out)
                    path = r.render()
                    rel = "/" + str(Path(path).resolve().relative_to(Path(__file__).resolve().parents[2] / 'viewer' / 'public')).replace('\\','/')
                    await self.send_chat(sender='agent', text=f"🛠️ Workshop scene rendered: {rel}", channel=client.channel)
                except Exception:
                    await self.send_system(client.channel, "Workshop render failed.")
                return
        if cmd == "/diary":
            await self._handle_diary(parts[1:] if len(parts) > 1 else [], client)
            return
        if cmd in ("/fb", "/feedback"):
            # RLWHF: /fb good|partial|bad [gold_or_notes]
            try:
                rating = (parts[1] if len(parts) > 1 else "").strip().lower()
                notes = (parts[2] if len(parts) > 2 else "").strip()
            except Exception:
                rating, notes = "", ""
            if rating not in ("good", "partial", "bad"):
                await self.send_system(client.channel, "Usage: /fb good|partial|bad [gold_or_notes]")
                return
            rec = {
                "type": "feedback",
                "channel": client.channel,
                "from": client.nick,
                "rating": rating,
                "gold": notes,
            }
            await self.log(rec)
            await self.send_chat(sender="agent", text=f"[thanks] feedback noted: {rating}", channel=client.channel)
            return
        if cmd == "/brain":
            # /brain reflect | /brain sleep [out]
            sub = parts[1].lower() if len(parts) > 1 else "status"
            if self._cranium is None:
                await self.send_system(client.channel, "Cranium not available.")
                return
            if sub == "reflect":
                msg = self._cranium.reflect()
                await self.send_chat(sender="agent", text=msg, channel=client.channel)
                await self.log({"type":"brain","action":"reflect","text":msg})
                # Event-based diary page (policy-gated)
                try:
                    if getattr(self._cranium, "_stm", None) is not None:
                        vec = self._cranium._stm.snapshot_vector()
                        from ..tools.house_memory import MemoryHouse  # type: ignore
                        from ..cranium.diary import DiaryPolicy  # type: ignore
                        h = MemoryHouse(); pages = h.list_diary_pages(self._diary_book)
                        last = None
                        if pages:
                            e32 = (pages[-1].extra or {}).get("embedding32") if isinstance(pages[-1].extra, dict) else None
                            if isinstance(e32, list):
                                last = e32
                        pol = DiaryPolicy()
                        meta = {"event": "brain_reflect"}
                        if pol.should_write(vec, last, event="brain_reflect", meta=meta):
                            h.add_diary_page_embedding(self._diary_book, vec, meta=meta)
                            out = (Path(__file__).resolve().parents[2] / "viewer" / "public" / "houses" / (os.getenv("K3D_HOUSE_ID","default")) / "memory_house.gltf")
                            h.export_gltf(out)
                except Exception:
                    pass
                return
            if sub == "sleep":
                out = None
                if len(parts) > 2:
                    out = parts[2].strip() or None
                status = self._cranium.sleep_consolidate(out_gltf=out)
                await self.send_chat(sender="agent", text=status, channel=client.channel)
                await self.log({"type":"brain","action":"sleep","status":status})
                return
            await self.send_system(client.channel, "Usage: /brain reflect | /brain sleep [viewer/public/memory_house.gltf]")
            return
        if cmd == "/mem" and len(parts) >= 2:
            await self._handle_mem(parts[1:], client)
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
        # Prefer A* when positions are available (default behavior)
        positions = graph.get("positions") if isinstance(graph, dict) else None
        if positions is not None:
            try:
                # Dynamic LOD: adjust neighbor fanout by proximity (enabled by default)
                use_lod = True if str(os.getenv("K3D_LOD_DYNAMIC", "1")).strip() != "0" else False
                if use_lod:
                    path_ids = self._Network3D.route_astar_lod(ids, neighbors, positions, start_id, target_id)
                else:
                    path_ids = self._Network3D.route_astar_ex(ids, neighbors, positions, start_id, target_id)
            except Exception:
                path_ids = self._Network3D.route(ids, neighbors, start_id, target_id)
        else:
            path_ids = self._Network3D.route(ids, neighbors, start_id, target_id)
        # compose payload and broadcast as a command
        resolved_addr = address or doors.get(label) or f"k3d://@?label={label}"
        payload = {"label": label, "address": resolved_addr, "path": path_ids or []}
        await self.send_command("open", json.dumps(payload), channel=client.channel)
        hops = max(0, (len(path_ids or []) - 1))
        await self.send_chat(sender="system", text=f"Opening door to {label} via {hops} hops", channel=client.channel)

    async def _handle_mem(self, args, client: Client):
        """Manage House Memory: rooms + objects; export GLTF to viewer/public/memory_house.gltf.

        Commands:
          /mem room <name> [desc]
          /mem add <room>|<label>|<text>
          /mem export
        """
        try:
            from ..tools.house_memory import MemoryHouse  # type: ignore
        except Exception:
            await self.send_system(client.channel, "Memory tool unavailable.")
            return
        sub = args[0].lower() if args else "help"
        h = MemoryHouse()
        if sub == "room" and len(args) >= 2:
            name = args[1]
            desc = "".join(args[2:]) if len(args) > 2 else ""
            h.add_room(name, desc)
            await self.send_chat(sender="agent", text=f"Memory: added room '{name}'.", channel=client.channel)
            return
        if sub == "add" and len(args) >= 2:
            try:
                # Expect room|label|text
                raw = " ".join(args[1:])
                room, label, text = [s.strip() for s in raw.split("|")]
                # Prevent human writes into the AI Diary room
                if room.lower() == "diary":
                    await self.send_system(client.channel, "Writes to 'Diary' are AI-only.")
                    return
                h.add_object(room, label, text)
                await self.send_chat(sender="agent", text=f"Memory: saved '{label}' in room '{room}'.", channel=client.channel)
            except Exception:
                await self.send_system(client.channel, "Usage: /mem add <room>|<label>|<text>")
            return
        if sub == "furniture" and len(args) >= 2:
            try:
                # Expect room|kind|label
                raw = " ".join(args[1:])
                room, kind, label = [s.strip() for s in raw.split("|")]
                h.add_furniture(room, kind, label)
                await self.send_chat(sender="agent", text=f"Memory: placed {kind} '{label}' in '{room}'.", channel=client.channel)
            except Exception:
                await self.send_system(client.channel, "Usage: /mem furniture <room>|<kind>|<label>")
            return
        if sub == "door" and len(args) >= 2:
            try:
                # Expect label|address
                raw = " ".join(args[1:])
                label, address = [s.strip() for s in raw.split("|")]
                h.add_door(label, address)
                await self.send_chat(sender="agent", text=f"Memory: added door '{label}' -> {address}", channel=client.channel)
            except Exception:
                await self.send_system(client.channel, "Usage: /mem door <label>|<address>")
            return
        if sub == "bootstrap" and len(args) >= 2:
            kind = args[1].lower()
            if kind == "defaults":
                h.bootstrap_defaults()
                await self.send_chat(sender="agent", text="Memory: bootstrapped defaults (rooms, furniture, doors)", channel=client.channel)
                return
            if kind == "books":
                n = h.bootstrap_books(24)
                await self.send_chat(sender="agent", text=f"Memory: bootstrapped {n} books", channel=client.channel)
                return
            if kind == "reflections":
                n = h.bootstrap_reflections(50)
                await self.send_chat(sender="agent", text=f"Memory: bootstrapped {n} reflections", channel=client.channel)
                return
            if kind == "training":
                n = h.bootstrap_training(50)
                await self.send_chat(sender="agent", text=f"Memory: bootstrapped {n} training artifacts", channel=client.channel)
                return
            await self.send_system(client.channel, "Usage: /mem bootstrap defaults|books|reflections|training")
            return
        if sub == "export":
            out = (Path(__file__).resolve().parents[2] / "viewer" / "public" / "memory_house.gltf")
            h.export_gltf(out)
            await self.send_chat(sender="agent", text=f"Memory: exported to {out}", channel=client.channel)
            return
        await self.send_system(client.channel, "Usage: /mem room <name> [desc] | /mem add <room>|<label>|<text> | /mem furniture <room>|<kind>|<label> | /mem door <label>|<address> | /mem bootstrap <kind> | /mem export")

    async def _handle_sleep(self, args, client: Client):
        """Sleep mode: pause channel and run consolidation.

        Modes:
          /sleep                         -> pause only
          /sleep consolidate              -> pause + consolidate (diary/reflections/training) + export memory
          /sleep <star_id> [shape_type]   -> consolidate a star into bathtub furniture, update registry + render scene
        """
        # Always pause channel first
        if not self._is_paused(client.channel):
            await self._pause(client, reason="sleep")

        # Case 1: Consolidate memory house
        if len(args) >= 1 and args[0].lower().startswith("consol"):
            try:
                from ..tools.house_memory import MemoryHouse  # type: ignore
                h = MemoryHouse()
                n_ref = h.bootstrap_reflections(100)
                n_tr = h.bootstrap_training(100)
                n_diary = h.bootstrap_diary()
                # Sleep-time re-embed of chat messages (GPU only; skipped if unavailable)
                try:
                    n_chat = h.reembed_chat_messages()
                except Exception:
                    n_chat = 0
                h.bootstrap_defaults()  # ensure furniture/doors exist
                out = (Path(__file__).resolve().parents[2] / "viewer" / "public" / "memory_house.gltf")
                h.export_gltf(out)
                # Grow Knowledge Garden from current Galaxy (best-effort)
                try:
                    galaxy = (Path(__file__).resolve().parents[2] / "viewer" / "public" / "galaxy.glb")
                    garden_out = (Path(__file__).resolve().parents[2] / "viewer" / "public" / "knowledge_garden.glb")
                    if galaxy.exists():
                        from ..tools import gardens as _gard
                        _gard.main  # import ok
                        import sys as _sys
                        saved = list(_sys.argv)
                        try:
                            _sys.argv = ["gardens.py", "--from-galaxy", str(galaxy), "--gltf", str(garden_out)]
                            _gard.main()
                        finally:
                            _sys.argv = saved
                except Exception:
                    pass
                await self.send_chat(sender="agent", text=f"Sleep: consolidated memory (reflections+{n_ref}, training+{n_tr}, diary+{n_diary}, chat_reembed+{n_chat}). Exported memory_house.gltf.", channel=client.channel)
                await self.log({"type":"sleep","action":"consolidate","reflections":n_ref,"training":n_tr,"diary":n_diary,"chat_reembed":n_chat})
            except Exception as e:
                await self.send_system(client.channel, f"Sleep consolidation error: {e}")
            return

        # Case 2: Consolidate a single star into bathtub furniture
        if len(args) >= 1 and args[0].strip():
            star_arg = args[0].strip()
            shape_type = (args[1] if len(args) >= 2 else 'tetrahedron')
            repo_root = Path(__file__).resolve().parents[2]
            try:
                # Resolve embedding for star_arg via simple heuristics
                emb: List[float] = []
                sid = star_arg
                import numpy as _np  # type: ignore
                def _load_npy(path: Path) -> List[float]:
                    arr = _np.load(str(path))
                    if hasattr(arr, 'shape') and len(arr.shape) == 2:
                        # take first row
                        return [float(x) for x in arr[0].tolist()]
                    return [float(x) for x in arr.reshape(-1).tolist()]
                # file:<path>
                if star_arg.startswith('file:'):
                    p = Path(star_arg[5:])
                    if p.exists():
                        if p.suffix.lower() == '.npy':
                            emb = _load_npy(p)
                        elif p.suffix.lower() in {'.json', '.txt'}:
                            try:
                                emb = list(json.loads(p.read_text(encoding='utf-8')))
                            except Exception:
                                pass
                # index:<i>
                if not emb and star_arg.startswith('index:'):
                    try:
                        i = int(star_arg.split(':', 1)[1])
                        epath = repo_root / 'embeddings.npy'
                        if epath.exists():
                            mat = _np.load(str(epath))
                            if 0 <= i < len(mat):
                                emb = [float(x) for x in mat[i].reshape(-1).tolist()]
                                sid = f"idx:{i}"
                    except Exception:
                        pass
                # plain int → embeddings.npy row
                if not emb:
                    try:
                        i = int(star_arg)
                        epath = repo_root / 'embeddings.npy'
                        if epath.exists():
                            mat = _np.load(str(epath))
                            if 0 <= i < len(mat):
                                emb = [float(x) for x in mat[i].reshape(-1).tolist()]
                                sid = f"idx:{i}"
                    except Exception:
                        pass
                # fallback to single embedding.npy
                if not emb:
                    e1 = repo_root / 'embedding.npy'
                    if e1.exists():
                        emb = _load_npy(e1)
                        sid = 'file:embedding.npy'
                # as a last resort, best-effort from galaxy.glb (may be huge)
                if not emb:
                    try:
                        from pygltflib import GLTF2  # type: ignore
                        gpath = repo_root / 'viewer' / 'public' / 'galaxy.v7.glb'
                        if not gpath.exists():
                            gpath = repo_root / 'viewer' / 'public' / 'galaxy.glb'
                        if gpath.exists():
                            g = GLTF2().load_binary(str(gpath))
                            prim = g.meshes[0].primitives[0]
                            k3d = prim.extras.get('k3d', {}) if prim.extras else {}
                            ids = list(k3d.get('ids', []) or [])
                            embs = k3d.get('embeddings') or []
                            if ids and embs and star_arg in ids:
                                idx = ids.index(star_arg)
                                vec = embs[idx]
                                if isinstance(vec, list):
                                    emb = [float(x) for x in vec]
                    except Exception:
                        pass

                if not emb:
                    await self.send_system(client.channel, f"/sleep: could not resolve embedding for '{star_arg}' (try 'index:<i>' or 'file:<path.npy>')")
                    return

                # Run ConsolidationEngine
                from ..tools.phase5.ConsolidationEngine import ConsolidationEngine  # type: ignore
                eng = ConsolidationEngine(keep_ratio=0.5)
                res = eng.consolidate_star({'id': sid, 'embedding': emb, 'shape_type': shape_type})
                item = eng.to_registry_item(res)

                # Update bathtub registry
                reg_path = repo_root / 'viewer' / 'public' / 'bathtub_registry.json'
                try:
                    if reg_path.exists():
                        reg = json.loads(reg_path.read_text(encoding='utf-8'))
                    else:
                        reg = {'bathtub_version': '1.0', 'room_path': 'viewer/public/bathtub_room.glb', 'furniture': []}
                except Exception:
                    reg = {'bathtub_version': '1.0', 'room_path': 'viewer/public/bathtub_room.glb', 'furniture': []}
                furn = list(reg.get('furniture', []) or [])
                # replace existing by id
                furn = [f for f in furn if str(f.get('id')) != item['id']]
                furn.append(item)
                reg['furniture'] = furn
                reg_path.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding='utf-8')

                # Render bathtub scene
                from ..tools.phase5.bathtub_renderer import BathtubRenderer  # type: ignore
                scene_out = repo_root / 'viewer' / 'public' / 'bathtub_scene.glb'
                r = BathtubRenderer(str(reg_path), str(scene_out))
                out_path = r.render()

                # Best-effort: also export to House Memory (Workshop room)
                try:
                    from ..tools.house_memory import MemoryHouse  # type: ignore
                    h2 = MemoryHouse()
                    h2.bootstrap_defaults()
                    label = f"Crystallized {item['star_id']}"
                    h2.add_furniture('Workshop', item['furniture_kind'], label)
                    h2.export_gltf(repo_root / 'viewer' / 'public' / 'memory_house.gltf')
                except Exception:
                    pass

                msg = (
                    f"Sleep: consolidated star '{sid}' → {item['furniture_kind']} "
                    f"(honesty={item['honesty_score']:.2f}, crystallized={'yes' if item['is_crystallized'] else 'no'}). "
                    f"Updated registry and wrote {out_path}."
                )
                await self.send_chat(sender='agent', text=msg, channel=client.channel)
                await self.log({
                    'type': 'sleep', 'action': 'consolidate_star', 'star_id': sid,
                    'furniture_kind': item['furniture_kind'], 'honesty': item['honesty_score'],
                    'crystallized': item['is_crystallized']
                })
            except Exception as e:
                await self.send_system(client.channel, f"Sleep star consolidation error: {e}")
            return

        # Default: just pause
        await self.send_chat(sender="agent", text="Entering sleep mode (paused). Use /sleep consolidate to restructure memory or /resume to wake.", channel=client.channel)
        return

    async def _handle_model(self, args, client: Client):
        sub = (args[0].lower() if args else "status")
        if sub == "on":
            if not self._models:
                repo_root = Path(__file__).resolve().parents[2]
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
                        self._models[kind] = loader(path)
                        self._model_paths[kind] = str(path)
                        self._active_kinds.add(kind)
                    except Exception:
                        continue
            self._model_enabled = True
            await self.send_system(client.channel, f"Model: on (threshold={self._model_threshold:.2f}) active={','.join(sorted(self._active_kinds)) or 'none'}")
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
                self._models[kind] = loader(p)
                self._model_paths[kind] = str(p)
                self._active_kinds.add(kind)
                await self.send_system(client.channel, f"Model loaded: kind={kind} path={path}")
            except Exception as e:
                await self.send_system(client.channel, f"Model load failed: {e}")
            return
        if sub == "use" and len(args) >= 2:
            mode = args[1].lower()
            if mode in ("both", "all"):
                self._active_kinds = set(self._models.keys())
            elif mode in ("hf", "sklearn"):
                self._active_kinds = {mode} if mode in self._models else set()
            elif mode in ("auto",):
                self._active_kinds = {"hf"} if "hf" in self._models else (set([next(iter(self._models.keys()))]) if self._models else set())
            else:
                await self.send_system(client.channel, "Usage: /model use both|hf|sklearn|auto")
                return
            await self.send_system(client.channel, f"Active models: {','.join(sorted(self._active_kinds)) or 'none'}")
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
        if sub == "list":
            loaded = ", ".join([f"{k}:{self._model_paths.get(k,'?')}" for k in sorted(self._models.keys())]) or "(none)"
            active = ",".join(sorted(self._active_kinds)) or "(none)"
            await self.send_system(client.channel, f"Models loaded: {loaded}. Active: {active}. Enabled: {self._model_enabled} threshold={self._model_threshold:.2f}")
            return
        if sub == "clear":
            self._models.clear(); self._model_paths.clear(); self._active_kinds.clear()
            await self.send_system(client.channel, "Models cleared.")
            return
        await self.send_system(
            client.channel,
            f"Model status: {'on' if self._model_enabled else 'off'} active={','.join(sorted(self._active_kinds)) or 'none'} threshold={self._model_threshold:.2f}",
        )

    async def _handle_open_book(self, title: str, client: Client):
        """Project book text to a pseudo-screen by writing a projection file.

        Demo implementation: searches viewer/public/library_room.glb for a book
        with extras.k3d.object.title == <title>, decodes a short text from its
        embedding buffer, and writes it to viewer/public/projections/<title>.txt.
        """
        try:
            from pathlib import Path as _P
            from pygltflib import GLTF2 as _G
            import numpy as _np
            import struct as _st
        except Exception:
            await self.send_system(client.channel, "OpenBook: missing deps (pygltflib/numpy)")
            return
        glb = (_P(__file__).resolve().parents[2] / "viewer" / "public" / "library_room.glb")
        if not glb.exists():
            await self.send_system(client.channel, "OpenBook: library_room.glb not found")
            return
        try:
            m = _G().load_binary(str(glb))
            found = None
            for mesh in m.meshes or []:
                for prim in mesh.primitives or []:
                    k3d = (prim.extras or {}).get("k3d") if prim.extras else None
                    if isinstance(k3d, dict) and k3d.get("object", {}).get("kind") == "book":
                        if str(k3d.get("object", {}).get("title", "")) == title:
                            found = k3d
                            break
                if found:
                    break
            if not found:
                await self.send_system(client.channel, f"OpenBook: '{title}' not found")
                return
            bv = m.bufferViews[int(found["embeddingsView"])]
            blob = m.binary_blob()
            data = blob[(bv.byteOffset or 0): (bv.byteOffset or 0) + bv.byteLength]
            emb = _np.array(_st.unpack('<' + 'f' * (bv.byteLength // 4), data), dtype=_np.float32)
            # Lightweight demo decoder
            out = []
            for i in range(0, min(len(emb), 128), 4):
                code = int(abs(float(emb[i])) * 255) % 128
                if 32 <= code <= 126:
                    out.append(chr(code))
            text = (''.join(out))[:1000] or "(empty)"
            proj_dir = glb.parent / "projections"
            proj_dir.mkdir(parents=True, exist_ok=True)
            safe = ''.join([c for c in title if c.isalnum() or c in ('_','-')])
            (proj_dir / f"{safe}.txt").write_text(text, encoding='utf-8')
            await self.send_chat(sender="agent", text=f"Projected '{title}' to screen (projections/{safe}.txt)", channel=client.channel)
        except Exception:
            await self.send_system(client.channel, "OpenBook: failed to project")
        return

    async def _handle_logs(self, args, client: Client):
        sub = (args[0].lower() if args else "status")
        if sub == "status":
            try:
                size = self.session_file.stat().st_size
            except Exception:
                size = 0
            await self.send_system(client.channel, f"Log: {self.session_file.name} size={size} rotate_bytes={self._log_rotate_bytes} compress_age(h)={self._log_compress_age_hours}")
            return
        if sub == "rotate":
            self._session_idx += 1
            self.session_file = self.log_dir / f"session-{self._session_ts}-{self._session_idx}.jsonl"
            await self.send_system(client.channel, f"Log rotated → {self.session_file.name}")
            return
        if sub == "compress":
            try:
                # run a one-shot maintenance pass
                cutoff = datetime.utcnow().timestamp() - (self._log_compress_age_hours * 3600)
                for p in self.log_dir.glob("session-*.jsonl"):
                    try:
                        if p.resolve() == self.session_file.resolve():
                            continue
                        if p.stat().st_mtime < cutoff:
                            try:
                                import zstandard as zstd  # type: ignore
                                out = p.with_suffix(p.suffix + ".zst")
                                cctx = zstd.ZstdCompressor(level=19)
                                with p.open("rb") as fin, out.open("wb") as fout:
                                    cctx.copy_stream(fin, fout)
                                p.unlink(missing_ok=True)
                            except Exception:
                                import gzip
                                out = p.with_suffix(p.suffix + ".gz")
                                with p.open("rb") as fin, gzip.open(out, "wb", compresslevel=9) as fout:
                                    while True:
                                        chunk = fin.read(1024 * 1024)
                                        if not chunk:
                                            break
                                        fout.write(chunk)
                                p.unlink(missing_ok=True)
                    except Exception:
                        continue
            except Exception:
                pass
            await self.send_system(client.channel, "Compression sweep complete.")
            return
        if sub == "set" and len(args) >= 3:
            key = args[1].lower(); val = args[2]
            if key in ("rotate", "rotate_bytes"):
                try:
                    self._log_rotate_bytes = int(val)
                    await self.send_system(client.channel, f"rotate_bytes={self._log_rotate_bytes}")
                except Exception:
                    await self.send_system(client.channel, "Invalid rotate_bytes value")
                return
            if key in ("compress_age", "compress_hours"):
                try:
                    self._log_compress_age_hours = int(val)
                    await self.send_system(client.channel, f"compress_age(h)={self._log_compress_age_hours}")
                except Exception:
                    await self.send_system(client.channel, "Invalid compress_age value")
                return
        await self.send_system(client.channel, "Usage: /logs status|rotate|compress|set rotate_bytes <N>|set compress_age <H>")

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
        # Persist chat to House memory (Chat Book) except for system messages
        try:
            if channel and sender and sender != "system" and text:
                from ..tools.house_memory import MemoryHouse  # type: ignore
                role = "agent" if sender == "agent" else "human"
                book_label = f"Chat {channel}"
                h = MemoryHouse()
                prev = self._last_chat_msg.get(channel)
                mid = h.add_chat_message(book_label=book_label, nick=sender, text=str(text), role=role, room="Diary", prev_id=prev)
                self._last_chat_msg[channel] = mid
        except Exception:
            pass

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
            # rotate if needed
            try:
                if self._log_rotate_bytes and self._log_rotate_bytes > 0:
                    size = self.session_file.stat().st_size
                    if size >= self._log_rotate_bytes:
                        self._session_idx += 1
                        self.session_file = self.log_dir / f"session-{self._session_ts}-{self._session_idx}.jsonl"
            except Exception:
                pass

    async def send_json(self, obj: Dict[str, Any], channel: Optional[str] = None):
        payload = json.dumps(obj)
        if channel and channel in self.channels:
            targets = [self.by_key[k].ws for k in self.channels[channel] if k in self.by_key]
            await asyncio.gather(*[ws.send(payload) for ws in targets])
        else:
            await asyncio.gather(*[c.ws.send(payload) for c in list(self.clients)])

    async def _warmup_heavy(self):
        if not self._fast_start:
            return
        loop = asyncio.get_running_loop()
        # Import HF loader in thread to avoid blocking event loop
        def _import_hf():
            try:
                from ..models.intent_hf import load_model as hf_load, predict_action as hf_pred  # type: ignore
                return hf_load, hf_pred
            except Exception:
                return None
        res = await loop.run_in_executor(None, _import_hf)
        if res:
            hf_load, hf_pred = res
            self._loaders.append(("hf", hf_load))
            self._predictors.append(("hf", hf_pred))
        # Auto-load default models (best effort)
        def _auto_load():
            try:
                repo_root = Path(__file__).resolve().parents[2]
                local_root = repo_root.parent / f"{repo_root.name}.local"
                env_path = os.getenv("K3D_MODEL")
                auto_on = True if str(os.getenv("K3D_MODEL_AUTO", "1")).strip() != "0" else False
                want_ensemble = True if str(os.getenv("K3D_MODEL_ENSEMBLE", "0")).strip() != "0" else False
                candidates: list[tuple[str, Path]] = []
                if env_path:
                    p = Path(env_path)
                    kind = "hf" if p.is_dir() else "sklearn"
                    candidates.append((kind, p))
                else:
                    hf_dir = local_root / "models" / "intent_hf"
                    pkl = local_root / "models" / "intent.pkl"
                    if hf_dir.exists() and (hf_dir / "config.json").exists():
                        candidates.append(("hf", hf_dir))
                    if pkl.exists():
                        candidates.append(("sklearn", pkl))
                for kind, path in candidates:
                    loader = next((l for k,l in self._loaders if k==kind), None)
                    if not loader:
                        continue
                    try:
                        mdl = loader(path)
                        self._models[kind] = mdl
                        self._model_paths[kind] = str(path)
                        if want_ensemble:
                            self._active_kinds.add(kind)
                    except Exception:
                        continue
                if not self._active_kinds and self._models:
                    self._active_kinds.add("hf" if "hf" in self._models else next(iter(self._models.keys())))
                self._model_enabled = auto_on and bool(self._active_kinds)
            except Exception:
                pass
        await loop.run_in_executor(None, _auto_load)

    async def run(self):
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"K3D live server listening on ws://{self.host}:{self.port}")
            # Start background tasks early so handshake can complete
            try:
                asyncio.create_task(self._log_maintenance_loop())
            except Exception:
                pass
            try:
                asyncio.create_task(self._warmup_heavy())
            except Exception:
                pass
            try:
                if self._autonomy_enabled:
                    asyncio.create_task(self._autonomy_loop())
            except Exception:
                pass
            await asyncio.Future()

    async def _log_maintenance_loop(self) -> None:
        while True:
            try:
                # compress session-*.jsonl older than threshold (except current)
                cutoff = datetime.utcnow().timestamp() - (self._log_compress_age_hours * 3600)
                for p in self.log_dir.glob("session-*.jsonl"):
                    try:
                        if p.resolve() == self.session_file.resolve():
                            continue
                    except Exception:
                        continue
                    try:
                        if p.stat().st_mtime < cutoff:
                            # prefer zstd, fallback gzip
                            try:
                                import zstandard as zstd  # type: ignore
                                out = p.with_suffix(p.suffix + ".zst")
                                cctx = zstd.ZstdCompressor(level=19)
                                with p.open("rb") as fin, out.open("wb") as fout:
                                    cctx.copy_stream(fin, fout)
                                p.unlink(missing_ok=True)
                            except Exception:
                                import gzip
                                out = p.with_suffix(p.suffix + ".gz")
                                with p.open("rb") as fin, gzip.open(out, "wb", compresslevel=9) as fout:
                                    while True:
                                        chunk = fin.read(1024 * 1024)
                                        if not chunk:
                                            break
                                        fout.write(chunk)
                                p.unlink(missing_ok=True)
                    except Exception:
                        continue
            except Exception:
                pass
            # Yield to the event loop to avoid starvation and control cadence
            try:
                await asyncio.sleep(max(1, int(self._log_maint_period)))
            except Exception:
                # As a last resort, yield briefly
                await asyncio.sleep(1)

    async def _autonomy_loop(self) -> None:
        """Autonomous behavior when channels are idle.

        Behavior (every ~pi seconds):
        - If a channel is idle > threshold and not paused, reflect and make a small movement
        - Prefer moving to a hub or a neighbor-of-neighbor
        - Propose link suggestions via highlight and a short chat
        """
        while True:
            try:
                await asyncio.sleep(self._autonomy_period)
                now = self._now()
                for ch, keys in list(self.channels.items()):
                    if self._is_paused(ch):
                        continue
                    last = self._last_activity.get(ch, 0.0)
                    if now - last < self._autonomy_idle:
                        continue
                    # Reflect
                    try:
                        if self._cranium is not None:
                            msg = self._cranium.reflect()
                            await self.send_chat(sender="agent", text=f"[reflect] {msg}", channel=ch)
                    except Exception:
                        pass
                    # Navigate to an interesting node (hub or unseen)
                    g = self._graphs.get(ch)
                    labels = (g or {}).get("labels") or []
                    neighbors = (g or {}).get("neighbors") or []
                    ids = (g or {}).get("ids") or []
                    if labels and neighbors and ids:
                        deg = [(i, len(neighbors[i]) if i < len(neighbors) else 0) for i in range(len(ids))]
                        target_idx = deg[0][0] if not deg else 0
                        try:
                            target_idx = sorted(deg, key=lambda t: t[1], reverse=True)[0][0]
                        except Exception:
                            target_idx = 0
                        target_label = labels[target_idx] if target_idx < len(labels) else None
                        if target_label:
                            await self._dispatch_goto(ch, target_label, source="autonomy")
                    # Suggest a link between two close labels using TF‑IDF
                    idx = self._search_index.get(ch) or {}
                    vec = idx.get("vec"); X = idx.get("X"); labs = idx.get("labels") or []
                    if vec is not None and X is not None and labs:
                        try:
                            import numpy as _np  # type: ignore
                            # pick two top mutually similar docs
                            S = (X @ X.T).toarray()
                            _np.fill_diagonal(S, -1)
                            i, j = _np.unravel_index(_np.argmax(S), S.shape)
                            if i != j and 0 <= i < len(labs) and 0 <= j < len(labs):
                                a, b = labs[int(i)], labs[int(j)]
                                score = float(S[i, j])
                                payload = json.dumps({"labels": [str(a), str(b)]})
                                await self.send_command("highlight", payload, channel=ch)
                                await self.send_chat(sender="agent", text=f"[suggest] Link {a} ↔ {b} (score≈{score:.2f})", channel=ch)
                        except Exception:
                            pass
                    # Mark the time to avoid spamming
                    self._last_activity[ch] = now
                    # Autonote: write a diary page periodically
                    try:
                        if self._diary_auto_enabled and self._cranium is not None and getattr(self._cranium, "_stm", None) is not None:
                            lastp = self._last_diary_time.get(ch, 0.0)
                            if now - lastp >= self._diary_period:
                                vec = self._cranium._stm.snapshot_vector()
                                from ..tools.house_memory import MemoryHouse  # type: ignore
                                from ..cranium.diary import DiaryPolicy  # type: ignore
                                h = MemoryHouse(); pages = h.list_diary_pages(self._diary_book)
                                last = None
                                if pages:
                                    e32 = (pages[-1].extra or {}).get("embedding32") if isinstance(pages[-1].extra, dict) else None
                                    if isinstance(e32, list):
                                        last = e32
                                pol = DiaryPolicy()
                                if pol.should_write(vec, last, event="auto", meta={"by":"autonomy"}):
                                    pid = h.add_diary_page_embedding(self._diary_book, vec, meta={"event":"auto"})
                                    out = (Path(__file__).resolve().parents[2] / "viewer" / "public" / "houses" / (os.getenv("K3D_HOUSE_ID","default")) / "memory_house.gltf")
                                    h.export_gltf(out)
                                    await self.send_chat(sender="agent", text=f"[diary] wrote page in '{self._diary_book}' (id={pid})", channel=ch)
                                    await self.log({"type":"diary","action":"auto","book":self._diary_book,"page_id":pid,"channel":ch})
                                    self._last_diary_time[ch] = now
                    except Exception:
                        pass
            except Exception:
                # Keep the loop alive regardless of errors
                continue
            await asyncio.sleep(max(5, int(self._log_maint_period or 60)))

    # --- Open-vocab goto resolution ---
    async def _dispatch_goto(self, channel: str, query: str, source: str = "rule", confidence: Optional[float] = None) -> None:
        q = (query or "").strip()
        resolved: Optional[str] = None
        score: Optional[float] = None
        if not q:
            return
        # Gazetteer pass (exact/prefix/substring on canonical forms)
        idx = self._search_index.get(channel) if hasattr(self, "_search_index") else None
        if idx and idx.get("gazetteer") and getattr(self, "_match_gazetteer", None) is not None:
            lab, sc = self._match_gazetteer(q, idx["gazetteer"])  # type: ignore[arg-type]
            if lab:
                resolved = lab
                score = sc
        # Use cache next
        cache = self._goto_cache.setdefault(channel, {})
        if not resolved:
            resolved = cache.get(q)
        # fallback TF-IDF + string heuristics
        if not resolved:
            # Exact/normalized match first
            labels_map = self._label_to_id.get(channel) or {}
            if q in labels_map:
                resolved = q
                score = 1.0
            else:
                # simple case-insensitive normalization
                lower_map = {k.lower(): k for k in labels_map.keys()}
                if q.lower() in lower_map:
                    resolved = lower_map[q.lower()]
                    score = 1.0
                else:
                    # TF-IDF cosine over labels if available
                    if idx and getattr(self, "_NP", None) is not None:
                        try:
                            vec, X, labs = idx["vec"], idx["X"], idx["labels"]
                            qv = vec.transform([q])
                            import numpy as np  # type: ignore
                            qnorm = float(np.sqrt((qv.multiply(qv)).sum())) or 1.0
                            row_norms = idx.get("row_norms")
                            if row_norms is None:
                                row_norms = np.sqrt((X.multiply(X)).sum(axis=1)).A1
                                idx["row_norms"] = row_norms
                            sims = (X @ qv.T).toarray().ravel()
                            denom = row_norms * qnorm
                            denom = np.where(denom == 0, 1.0, denom)
                            sims = sims / denom
                            best = int(sims.argmax())
                            resolved = labs[best]
                            score = float(sims[best])
                        except Exception:
                            resolved = None
                    # fallback: prefix/substring over labels
                    if not resolved and labels_map:
                        cand = next((k for k in labels_map.keys() if k.lower().startswith(q.lower())), None)
                        resolved = cand or next((k for k in labels_map.keys() if q.lower() in k.lower()), None)
            if resolved:
                cache[q] = resolved
        # Send command using resolved if available; else use raw query
        target = resolved or q
        await self.send_command("goto", target, channel=channel)
        # Human-friendly note with resolution
        note = f"Navigating to {target}" if target == q else f"Navigating to {target} (from '{q}'{', sim='+str(round(score,3)) if score is not None else ''})"
        if confidence is not None:
            note = f"[model {confidence:.2f}] " + note
        await self.send_chat(sender="agent", text=note, channel=channel)
        try:
            await self.log({
                "type": "goto_resolution",
                "channel": channel,
                "query": q,
                "resolved": target,
                "score": score,
                "source": source,
                "model_confidence": confidence,
            })
        except Exception:
            pass
        # Event-based diary entry after navigation (AI-only, policy-gated)
        try:
            if self._cranium is not None and getattr(self._cranium, "_stm", None) is not None:
                vec = self._cranium._stm.snapshot_vector()
                from ..tools.house_memory import MemoryHouse  # type: ignore
                from ..cranium.diary import DiaryPolicy  # type: ignore
                h = MemoryHouse(); pages = h.list_diary_pages(self._diary_book)
                last = None
                if pages:
                    e32 = (pages[-1].extra or {}).get("embedding32") if isinstance(pages[-1].extra, dict) else None
                    if isinstance(e32, list):
                        last = e32
                pol = DiaryPolicy()
                meta = {"event": "navigate", "target": target, "query": q, "source": source, "score": score if score is not None else 0.0}
                if pol.should_write(vec, last, event="navigate", meta=meta):
                    h.add_diary_page_embedding(self._diary_book, vec, meta=meta)
                    out = (Path(__file__).resolve().parents[2] / "viewer" / "public" / "houses" / (os.getenv("K3D_HOUSE_ID","default")) / "memory_house.gltf")
                    h.export_gltf(out)
        except Exception:
            pass
        # If we already hold a dataset graph, pre-compute and share a concise route trace
        try:
            g = self._graphs.get(channel) or {}
            ids = g.get("ids") or []
            labels = g.get("labels") or []
            neighbors = g.get("neighbors") or []
            positions = g.get("positions") if isinstance(g, dict) else None
            if ids and neighbors and labels:
                # Derive start/target from current label + target
                start_label = self._current_label.get(channel)
                si = labels.index(start_label) if start_label in labels else None
                ti = labels.index(target) if target in labels else None
                if si is not None and ti is not None:
                    start_id = ids[si]; target_id = ids[ti]
                    # Prefer A* when positions present
                    if positions is not None:
                        path = self._Network3D.route_astar_ex(ids, neighbors, positions, start_id, target_id)
                    else:
                        path = self._Network3D.route(ids, neighbors, start_id, target_id)
                    if path and len(path) > 1:
                        # Build label path
                        id_to_idx = {ids[i]: i for i in range(len(ids))}
                        path_labels = [labels[id_to_idx[p]] if id_to_idx.get(p) is not None else p for p in path]
                        # TF-IDF cosine over labels for per-hop sim
                        tfidf_sims = []
                        try:
                            idx = self._search_index.get(channel)
                            if idx:
                                vec, X, labs = idx.get("vec"), idx.get("X"), idx.get("labels")
                                import numpy as np  # type: ignore
                                # map label -> row index
                                lab_to_row = {labs[i]: i for i in range(len(labs))}
                                row_norms = idx.get("row_norms")
                                if row_norms is None:
                                    row_norms = np.sqrt((X.multiply(X)).sum(axis=1)).A1
                                    idx["row_norms"] = row_norms
                                for a, b in zip(path_labels[:-1], path_labels[1:]):
                                    ia = lab_to_row.get(a); ib = lab_to_row.get(b)
                                    if ia is None or ib is None:
                                        tfidf_sims.append(None)
                                    else:
                                        num = (X[ia] @ X[ib].T).toarray().ravel()[0]
                                        den = float(row_norms[ia] * row_norms[ib]) or 1.0
                                        tfidf_sims.append(float(num / den))
                        except Exception:
                            tfidf_sims = []
                        # Geometric distances
                        geo = []
                        if positions is not None:
                            def d3(i, j):
                                pi = positions[i]; pj = positions[j]
                                import math
                                return float(math.sqrt((pi[0]-pj[0])**2 + (pi[1]-pj[1])**2 + (pi[2]-pj[2])**2))
                            for a, b in zip(path[:-1], path[1:]):
                                ia = id_to_idx.get(a); ib = id_to_idx.get(b)
                                if ia is None or ib is None:
                                    geo.append(None)
                                else:
                                    geo.append(d3(ia, ib))
                        # Compose concise trace
                        hop_lines = []
                        for i in range(len(path_labels) - 1):
                            la = path_labels[i]; lb = path_labels[i+1]
                            s = tfidf_sims[i] if i < len(tfidf_sims) else None
                            gdist = geo[i] if i < len(geo) else None
                            parts = [f"{la} -> {lb}"]
                            if s is not None:
                                parts.append(f"sim={s:.2f}")
                            if gdist is not None:
                                parts.append(f"dist={gdist:.2f}")
                            hop_lines.append(" (" + ", ".join(parts[1:]) + ")" if len(parts)>1 else "")
                        summary = "Route: " + " -> ".join(path_labels)
                        if hop_lines:
                            # Align minimal: keep compact inline annotations per hop
                            annotated = []
                            for i, seg in enumerate(zip(path_labels[:-1], path_labels[1:])):
                                la, lb = seg
                                ann = hop_lines[i]
                                annotated.append(f"{la}->{lb}{ann}")
                            summary = "Trace: " + "; ".join(annotated)
                        await self.send_chat(sender="agent", text=summary, channel=channel)
                        await self.log({
                            "type": "route_trace",
                            "channel": channel,
                            "labels": path_labels,
                            "tfidf_sim": tfidf_sims,
                            "geo_dist": geo,
                            "source": source,
                        })
        except Exception:
            pass

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

    async def _handle_llm(self, args, client: Client):
        if self._llm is None:
            await self.send_system(client.channel, "LLM skill unavailable. Set K3D_LLM_BACKEND=ollama|transformers.")
            return
        if not args:
            await self.send_system(client.channel, "Usage: /llm ask <text> | /llm rag <text> [k] | /llm backend ollama|transformers [model] [url]")
            return
        sub = args[0].lower()
        if sub == "backend":
            # Usage:
            # /llm backend transformers <hf-model>
            # /llm backend llama_cpp <path.gguf> [n_gpu_layers] [n_ctx]
            kind = args[1].lower() if len(args) > 1 else "transformers"
            if kind == "llama_cpp":
                if len(args) < 3:
                    await self.send_system(client.channel, "Usage: /llm backend llama_cpp <model.gguf> [n_gpu_layers] [n_ctx]")
                    return
                model = args[2]
                try:
                    self._llm.set_backend("llama_cpp", model=model, url=None)
                    # Optional GPU layers and ctx size
                    n_gpu_layers = int(args[3]) if len(args) > 3 else None
                    n_ctx = int(args[4]) if len(args) > 4 else None
                    self._llm.configure_llama_cpp(n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
                    await self.send_system(client.channel, f"LLM backend set to llama_cpp model={model} gpu_layers={n_gpu_layers or '(default)'} ctx={n_ctx or '(default)'}")
                except Exception as e:
                    await self.send_system(client.channel, f"LLM backend error: {e}")
                return
            else:
                model = args[2] if len(args) > 2 else (args[1] if len(args) > 1 else None)
                try:
                    self._llm.set_backend("transformers", model=model, url=None)
                    await self.send_system(client.channel, f"LLM backend set to transformers model={model or '(unchanged)'}")
                except Exception as e:
                    await self.send_system(client.channel, f"LLM backend error: {e}")
                return
        if sub == "ask":
            text = args[1] if len(args) > 1 else ""
            if not text:
                await self.send_system(client.channel, "Usage: /llm ask <text>")
                return
            out = self._llm.generate(text)
            await self.send_chat(sender="agent", text=out, channel=client.channel)
            await self.log({"type": "llm", "mode": "ask", "text": text, "out": out})
            return
        if sub == "rag":
            q = args[1] if len(args) > 1 else ""
            try:
                k = int(args[2]) if len(args) > 2 else 5
            except Exception:
                k = 5
            if not q:
                await self.send_system(client.channel, "Usage: /llm rag <text> [k]")
                return
            idx = self._search_index.get(client.channel) or {}
            vec = idx.get("vec")
            X = idx.get("X")
            labels = idx.get("labels") or []
            contexts: list[tuple[str,str]] = []
            if vec is not None and X is not None and labels:
                try:
                    qv = vec.transform([q])
                    scores = (X @ qv.T).toarray().ravel()
                    import numpy as _np  # type: ignore
                    top = _np.argsort(-scores)[: max(1, k)]
                    for i in top:
                        lab = labels[int(i)]
                        contexts.append((lab, ""))
                except Exception:
                    pass
            out = self._llm.answer_with_rag(q, contexts)
            await self.send_chat(sender="agent", text=out, channel=client.channel)
            await self.log({"type": "llm", "mode": "rag", "text": q, "contexts": contexts, "out": out})
            return
        await self.send_system(client.channel, "Usage: /llm ask <text> | /llm rag <text> [k] | /llm backend transformers <model> | /llm backend llama_cpp <model.gguf> [n_gpu_layers] [n_ctx]")

    async def _handle_k3d_ask(self, args, client: Client):
        # Try unified cranium first
        q = args[0] if args else ""
        if not q:
            await self.send_system(client.channel, "Usage: /ask <text>")
            return
        # Build contexts from snippets
        idx = self._search_index.get(client.channel) or {}
        vec = idx.get("vec")
        X = idx.get("X")
        labels = idx.get("labels") or []
        snip = idx.get("snip") or {}
        contexts: list[tuple[str,str]] = []
        if vec is not None and X is not None and labels:
            try:
                qv = vec.transform([q])
                scores = (X @ qv.T).toarray().ravel()
                import numpy as _np  # type: ignore
                top = _np.argsort(-scores)[: max(1, 8)]
                for i in top:
                    lab = labels[int(i)]
                    # labels[] contains doc string ("label — text") when snippets were provided
                    # Extract original label key for snip lookup if possible
                    key = lab.split(" — ", 1)[0]
                    txt = snip.get(key, "")
                    contexts.append((key, txt or lab))
            except Exception:
                pass
        # Optional: prefer compose_auto (for outcome logging/training) when enabled
        try:
            import os as _os
            if (_os.getenv("K3D_USE_COMPOSE_AUTO", "0").strip() != "0"):
                from ..skills.spatial_text import compose_auto  # type: ignore
                mode, text = compose_auto(q, contexts)
                await self.send_chat(sender="agent", text=text, channel=client.channel)
                await self.log({"type":"ask","mode":"compose_auto","route":mode,"text":q,"contexts":contexts,"out":text})
                try:
                    await self._emit_reasoning_overlay(client.channel, q, contexts, mode_hint=mode, composed_text=text)
                except Exception:
                    pass
                return
        except Exception:
            pass
        if self._cranium is not None:
            try:
                resp = self._cranium.act(q, contexts=contexts)
                if resp.get("type") in ("navigation", "exploration", "interaction"):
                    # Dispatch navigation/action payloads
                    if self._is_paused(client.channel):
                        await self.send_system(client.channel, "Paused: action suppressed. Use /resume to continue.")
                        await self.log({"type": "pause_block", "what": resp.get("type"), "channel": client.channel})
                        return
                    dec = self._policy_check(q, resp.get("action"))
                    if not dec.allow:
                        await self.send_chat(sender="system", text=f"Action blocked by ethics policy ({dec.reason}).", channel=client.channel)
                        return
                    action = resp.get("action") or "action"
                    if action == "goto":
                        target = str(resp.get("target") or resp.get("location") or "").strip()
                        await self._dispatch_goto(client.channel, target, source="brain")
                    else:
                        payload = json.dumps({k: v for k, v in resp.items() if k not in {"type", "message"}})
                        await self.send_command(action, payload, channel=client.channel)
                        if msg_text := resp.get("message"):
                            await self.send_chat(sender="agent", text=msg_text, channel=client.channel)
                    await self.log({"type":"ask","mode":"brain","text":q,"contexts":contexts,"resp":resp})
                    return
                # Chat response
                out = resp.get("message") or ""
                if out:
                    await self.send_chat(sender="agent", text=out, channel=client.channel)
                    await self.log({"type":"ask","mode":"brain","text":q,"contexts":contexts,"out":out})
                    # Emit Spatial CoT overlay even when using unified cranium path
                    try:
                        await self._emit_reasoning_overlay(client.channel, q, contexts, mode_hint="compose", composed_text=out)
                    except Exception:
                        pass
                    return
            except Exception:
                pass
        # Fallback to spatial text skill
        if self._sp_compose is None:
            await self.send_system(client.channel, "Spatial text skill unavailable.")
            return
        out = self._sp_compose(q, contexts)
        await self.send_chat(sender="agent", text=out, channel=client.channel)
        await self.log({"type": "ask", "mode":"spatial_text", "text": q, "contexts": contexts, "out": out})
        # Emit Spatial CoT overlay (Phase 1) when available
        try:
            await self._emit_reasoning_overlay(client.channel, q, contexts, mode_hint="compose", composed_text=out)
        except Exception:
            pass

    async def _handle_diary(self, args, client: Client):
        sub = args[0].lower() if args else "help"
        try:
            from ..tools.house_memory import MemoryHouse  # type: ignore
        except Exception:
            await self.send_system(client.channel, "Diary unavailable (memory tools missing).")
            return
        # Writing is AI-only (no user command). Humans can only read.
        if sub == "read":
            label = args[1] if len(args) > 1 else "AI Diary"
            page = args[2] if len(args) > 2 else None
            try:
                h = MemoryHouse()
                pages = h.list_diary_pages(label)
                if not pages:
                    await self.send_system(client.channel, f"No pages in '{label}'.")
                    return
                target = None
                if page:
                    target = next((o for o in pages if o.id == page or o.label == page), None)
                if target is None:
                    target = pages[-1]
                e32 = (target.extra or {}).get("embedding32") if isinstance(target.extra, dict) else None
                if not isinstance(e32, list):
                    await self.send_system(client.channel, "Selected page missing embedding.")
                    return
                ctxs = h.nearest_contexts_for_embedding(e32, k=6)
                # Compose for humans
                try:
                    from ..skills.spatial_text import compose_answer  # type: ignore
                    txt = compose_answer(f"What is the gist of page '{target.label}'?", ctxs)
                except Exception:
                    lines = [f"- {lab}: {txt}" for lab, txt in ctxs[:5]]
                    txt = "\n".join([f"Page {target.label} (AI diary)", *lines])
                await self.send_chat(sender="agent", text=txt, channel=client.channel)
                await self.log({"type":"diary","action":"read","book":label,"page_id":target.id,"ctxs":ctxs})
            except Exception as e:
                await self.send_system(client.channel, f"Diary read error: {e}")
            return
        await self.send_system(client.channel, "Usage: /diary read [book_label] [page_id|label]")

    async def _emit_reasoning_overlay(self, channel: str, question: str, contexts: list[tuple[str, str]], mode_hint: str = "compose", composed_text: Optional[str] = None) -> None:
        """Build and send a reasoning path overlay to the viewer when supported.

        - Uses knowledge3d.skills.spatial_cot.compose_with_cot if available.
        - Resolves step labels to 3D waypoints when positions are present from dataset_graph.
        - Sends as a command: { type: 'command', command: 'reasoning_path', target: JSON }.
        """
        try:
            from ..skills.spatial_cot import compose_with_cot  # type: ignore
        except Exception:
            return
        # Avoid recomputing the answer when we already have it
        composer = (lambda q, ctxs: composed_text) if composed_text else None
        _, path = compose_with_cot(question, contexts, mode=mode_hint, composer=composer)
        payload = path.to_payload()
        # Resolve waypoints via dataset_graph if positions are present
        g = self._graphs.get(channel) or {}
        labels: list[str] = g.get("labels") or []
        positions: Optional[list] = g.get("positions") if isinstance(g, dict) else None
        if isinstance(positions, list) and positions and labels:
            # Build list of unique labels encountered in natural step order (retrieve/synthesize/verify dominate path)
            order: list[str] = []
            for s in payload.get("steps", []):
                lab = str(s.get("label") or "").strip()
                op = str(s.get("op") or "")
                # For compare, skip adding combined label as a waypoint
                if not lab or op == "compare":
                    continue
                # For synthesize labels like "a, b, c", split and take first
                if op == "synthesize" and "," in lab:
                    lab = lab.split(",", 1)[0].strip()
                if lab not in order:
                    order.append(lab)
            idx_map = {str(labels[i]): i for i in range(len(labels))}
            waypoints: list = []
            for lab in order:
                i = idx_map.get(lab)
                if i is None:
                    continue
                try:
                    pos = positions[i]
                    # Validate [x,y,z]
                    if isinstance(pos, (list, tuple)) and len(pos) == 3:
                        waypoints.append([float(pos[0]), float(pos[1]), float(pos[2])])
                except Exception:
                    continue
            if waypoints:
                payload["waypoints"] = waypoints
        # Compute per-hop similarity when 3D positions available and emit
        try:
            links: list[dict] = []
            order_labels = [s.get("label") for s in payload.get("steps", []) if s.get("op") in {"retrieve","synthesize","verify"}]
            # Condense synthesize combined labels
            order: list[str] = []
            for lab in order_labels:
                lab = str(lab or "").strip()
                if not lab:
                    continue
                if "," in lab:
                    lab = lab.split(",", 1)[0].strip()
                if lab not in order:
                    order.append(lab)
            if isinstance(positions, list) and labels and order and len(order) >= 2:
                idx_map = {str(labels[i]): i for i in range(len(labels))}
                def _cos(a, b):
                    import math
                    ax, ay, az = a; bx, by, bz = b
                    dot = ax*bx + ay*by + az*bz
                    na = math.sqrt(ax*ax + ay*ay + az*az)
                    nb = math.sqrt(bx*bx + by*by + bz*bz)
                    return float(dot / (max(1e-9, na*nb)))
                for i in range(len(order)-1):
                    a = order[i]; b = order[i+1]
                    ia = idx_map.get(a); ib = idx_map.get(b)
                    if ia is None or ib is None:
                        continue
                    try:
                        va = positions[int(ia)]; vb = positions[int(ib)]
                        if isinstance(va, (list, tuple)) and isinstance(vb, (list, tuple)) and len(va) == 3 and len(vb) == 3:
                            sim3d = _cos(va, vb)
                            links.append({"from": a, "to": b, "sim3d": round(sim3d, 6)})
                            await self.send_chat(sender="system", text=f"CoT hop: {a} → {b} (sim3d={sim3d:.3f})", channel=channel)
                    except Exception:
                        continue
            if links:
                payload["links"] = links
        except Exception:
            pass
        # Send as a command for viewer overlay rendering
        try:
            await self.send_command("reasoning_path", json.dumps(payload), channel=channel)
            await self.log({"type": "reasoning_path", "question": question, "payload": payload})
        except Exception:
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
    import argparse
    _host = os.getenv("K3D_LIVE_HOST", "127.0.0.1")
    try:
        _port = int(os.getenv("K3D_LIVE_PORT", "8765"))
    except Exception:
        _port = 8765
    ap = argparse.ArgumentParser(description="K3D Live Server")
    ap.add_argument("--host", default=_host)
    ap.add_argument("--port", type=int, default=_port)
    ap.add_argument("--fast-start", action="store_true", help="Delay heavy imports/model loads until after the server starts listening")
    ap.add_argument("--auto-port", action="store_true", help="Auto-select the nearest free port if requested port is busy")
    args = ap.parse_args()
    srv = LiveServer(host=args.host, port=args.port, fast_start=(True if args.fast_start else None))
    # Choose free port when requested or if env K3D_LIVE_AUTO is set
    want_auto = bool(args.auto_port or (os.getenv("K3D_LIVE_AUTO", "0").strip() != "0"))
    try:
        chosen = srv._choose_free_port() if (want_auto or srv._is_port_in_use(args.host, args.port)) else int(args.port)
        srv.port = int(chosen)
        # Write port scan summary
        repo_root = Path(__file__).resolve().parents[2]
        status = repo_root / "docs" / "reports" / "status"
        status.mkdir(parents=True, exist_ok=True)
        (status / "live_server_ports.json").write_text(json.dumps({"ts": datetime.utcnow().isoformat()+"Z", **srv._port_scan}, indent=2), encoding="utf-8")
        print(f"[LiveServer] binding {srv.host}:{srv.port} (auto={want_auto})")
    except Exception:
        pass
    asyncio.run(srv.run())


if __name__ == "__main__":  # pragma: no cover
    main()
