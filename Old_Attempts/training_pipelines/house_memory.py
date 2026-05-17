"""
House Memory builder: rooms + objects as long-term memory in a K3D GLTF.

- Stores a simple JSON state at data/memory_house.json
- Exports a GLTF with primitive.extras.k3d embedding (vectors/embeddings/metadata/neighbors)
  consumable by the K3D viewer.

Usage
  # add a room and an object, then export
  python3 -m knowledge3d.tools.house_memory --add-room "Books" --desc "Long-term knowledge books"
  python3 -m knowledge3d.tools.house_memory --add-object "Books" "Fine art" "Canonical article"
  python3 -m knowledge3d.tools.house_memory --export viewer/public/memory_house.glb

  # bootstrap from AI books (first N titles)
  python3 -m knowledge3d.tools.house_memory --bootstrap-books 24 --export viewer/public/memory_house.glb
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np  # type: ignore
from pygltflib import (  # type: ignore
    ARRAY_BUFFER,
    Accessor,
    Asset,
    Buffer,
    BufferView,
    GLTF2,
    Mesh,
    Node,
    Primitive,
    Scene,
)

ROOT = Path(__file__).resolve().parents[2]
def _default_state_path() -> Path:
    import os
    hid = (os.getenv("K3D_HOUSE_ID", "").strip() or "default")
    # Keep houses segregated by ID under data/houses/<id>
    base = ROOT / "data" / "houses" / hid
    return base / "memory_house.json"

STATE = _default_state_path()


def _hash_vec(text: str, dims: int = 32) -> List[float]:
    # Deterministic pseudo-embedding based on SHA256; repeat as needed
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vals: List[float] = []
    i = 0
    while len(vals) < dims:
        b = h[i % len(h)]
        # map byte [0..255] to [-0.5 .. 0.5]
        vals.append((b / 255.0) - 0.5)
        i += 1
    return vals


@dataclass
class Room:
    name: str
    desc: str = ""
    id: str = field(default_factory=lambda: f"room:{hashlib.md5(Path().name.encode()).hexdigest()[:8]}")


@dataclass
class Obj:
    room: str
    label: str
    text: str = ""
    kind: str = "object"  # object | furniture | door
    extra: Dict[str, str] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"obj:{hashlib.md5(Path().name.encode()).hexdigest()[:8]}")


class MemoryHouse:
    def __init__(self, state_path: Path = STATE):
        self.state_path = state_path
        self.rooms: Dict[str, Room] = {}
        self.objects: List[Obj] = []
        self._load()

    def _load(self):
        if not self.state_path.exists():
            return
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        for r in data.get("rooms", []):
            rm = Room(**r)
            self.rooms[rm.name] = rm
        for o in data.get("objects", []):
            self.objects.append(Obj(**o))

    def _save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "rooms": [vars(r) for r in self.rooms.values()],
            "objects": [vars(o) for o in self.objects],
        }
        self.state_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_room(self, name: str, desc: str = ""):
        if name in self.rooms:
            self.rooms[name].desc = desc
        else:
            self.rooms[name] = Room(name=name, desc=desc, id=f"room:{hashlib.md5(name.encode()).hexdigest()[:8]}")
        self._save()

    def add_object(self, room: str, label: str, text: str = "", kind: str = "object", extra: Optional[Dict[str, str]] = None):
        if room not in self.rooms:
            self.add_room(room)
        oid = f"obj:{hashlib.md5((room+'|'+label).encode()).hexdigest()[:8]}"
        self.objects = [o for o in self.objects if o.id != oid]
        self.objects.append(Obj(room=room, label=label, text=text, kind=kind, extra=extra or {}, id=oid))
        self._save()

    def add_furniture(self, room: str, kind: str, label: str, text: str = ""):
        self.add_object(room, label, text=text, kind="furniture", extra={"furniture": kind})

    def add_door(self, label: str, address: str, room: str = "Network"):
        self.add_object(room, label, text=address, kind="door", extra={"address": address})

    # --- Diary (AI personal notes as embeddings) ---
    def ensure_diary_book(self, label: str = "AI Diary", room: str = "Diary") -> str:
        """Ensure a diary book object exists; return its object id."""
        if room not in self.rooms:
            self.add_room(room, "Daily notes and reflections (AI diary)")
        oid = f"diary:{hashlib.md5((room+'|'+label).encode()).hexdigest()[:8]}"
        # If missing, add the book as a furniture-like object of kind 'diary_book'
        if not any(o.id == oid for o in self.objects):
            self.objects.append(Obj(room=room, label=label, text="", kind="diary_book", extra={}, id=oid))
            self._save()
        return oid

    def add_diary_page_embedding(self, book_label: str, vec32: List[float], ts: Optional[float] = None, room: str = "Diary", meta: Optional[Dict[str, object]] = None) -> str:
        """Append a diary page as a node with an explicit 32-d embedding.

        Stores metadata: parent (book id), ts (ISO8601), and e32 length. Returns page id.
        """
        import time
        # Normalize length
        v = list(vec32 or [])
        if len(v) != 32:
            # pad or trim to 32
            if len(v) < 32:
                v = v + [0.0] * (32 - len(v))
            else:
                v = v[:32]
        book_id = self.ensure_diary_book(book_label, room=room)
        t = ts or time.time()
        iso = datetime.utcfromtimestamp(t).isoformat() + "Z"
        # Stable page id
        pid = f"page:{hashlib.md5((book_id+'|'+iso).encode()).hexdigest()[:8]}"
        label = f"{book_label} — {iso}"
        extra = {"parent": book_id, "ts": iso, "embedding32": v}
        if meta:
            extra["meta"] = meta
        # Insert or replace
        self.objects = [o for o in self.objects if o.id != pid]
        self.objects.append(Obj(room=room, label=label, text="", kind="diary_page", extra=extra, id=pid))
        self._save()
        return pid

    def list_diary_pages(self, book_label: str, room: str = "Diary") -> List[Obj]:
        book_id = self.ensure_diary_book(book_label, room=room)
        pages = [o for o in self.objects if o.kind == "diary_page" and (o.extra or {}).get("parent") == book_id]
        # sort by ts
        def _ts(o: Obj) -> float:
            try:
                s = (o.extra or {}).get("ts") or ""
                return datetime.fromisoformat(s.replace("Z", "")).timestamp()
            except Exception:
                return 0.0
        return sorted(pages, key=_ts)

    # --- Chat Memory (human + agent turns) ---
    def ensure_chat_book(self, label: str = "Chat Book", room: str = "Diary") -> str:
        """Ensure a chat book object exists; return its id.

        Chat books group chat_message nodes by session/channel or topic.
        """
        if room not in self.rooms:
            self.add_room(room, "Daily notes and conversations")
        oid = f"chat:{hashlib.md5((room+'|'+label).encode()).hexdigest()[:8]}"
        if not any(o.id == oid for o in self.objects):
            self.objects.append(Obj(room=room, label=label, text="", kind="chat_book", extra={}, id=oid))
            self._save()
        return oid

    def add_chat_message(
        self,
        book_label: str,
        nick: str,
        text: str,
        role: str = "human",
        room: str = "Diary",
        prev_id: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> str:
        """Append a chat message node under a chat book.

        - Stores metadata: parent (book id), ts (ISO8601), nick, role (human|agent), prev (previous message id)
        - Embeds a deterministic 32‑d vector in extra.embedding32 (hash of nick|text)
        Returns message id.
        """
        book_id = self.ensure_chat_book(book_label, room=room)
        e32 = _hash_vec((nick or "") + "|" + (text or ""), 32)
        import time as _time
        t = ts or _time.time()
        iso = datetime.utcfromtimestamp(t).isoformat() + "Z"
        mid = f"msg:{hashlib.md5((book_id+'|'+iso+'|'+(nick or '')).encode()).hexdigest()[:8]}"
        label = f"{nick}: {text[:48]}" if text else f"{nick}"
        extra: Dict[str, object] = {"parent": book_id, "ts": iso, "nick": nick, "role": role, "embedding32": e32}
        if prev_id:
            extra["prev"] = prev_id
        self.objects = [o for o in self.objects if o.id != mid]
        self.objects.append(Obj(room=room, label=label, text=text, kind="chat_message", extra=extra, id=mid))
        self._save()
        return mid

    def nearest_contexts_for_embedding(self, vec32: List[float], k: int = 6) -> List[Tuple[str, str]]:
        """Return (label, text) from existing objects most similar to the given vector.

        Uses cosine on 32-d embeddings, where non-diary objects fall back to hashed
        embeddings derived from label+text.
        """
        import math
        v = list(vec32 or [])
        if len(v) != 32:
            if len(v) < 32:
                v = v + [0.0] * (32 - len(v))
            else:
                v = v[:32]
        def _cos(a: List[float], b: List[float]) -> float:
            dot = sum(x*y for x, y in zip(a, b))
            na = math.sqrt(sum(x*x for x in a)) + 1e-9
            nb = math.sqrt(sum(y*y for y in b)) + 1e-9
            return dot / (na * nb)
        scored: List[Tuple[float, str, str]] = []
        for o in self.objects:
            md = dict(o.extra or {})
            e = md.get("embedding32")
            if not isinstance(e, list):
                e = _hash_vec(o.label + "|" + o.text, 32)
            c = _cos(v, e)
            # Prefer objects with some text (e.g., files) for human contexts
            txt = o.text
            if not txt and o.kind in {"door", "diary_page", "diary_book"}:
                txt = o.label
            scored.append((c, o.label, txt))
        scored.sort(reverse=True, key=lambda t: t[0])
        out: List[Tuple[str, str]] = []
        for s, lab, txt in scored:
            if len(out) >= max(1, k):
                break
            out.append((lab, txt))
        return out

    def reembed_chat_messages(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", dims_out: int = 32) -> int:
        """Re-embed chat_message nodes using a GPU text encoder and update extra.embedding32.

        - Uses Sentence-Transformers on CUDA; if CUDA is unavailable, returns 0 (no CPU fallback).
        - Compresses to dims_out by taking the first dims_out components after L2 normalization.
        Returns number of updated messages.
        """
        # Collect chat messages
        msgs: List[Obj] = [o for o in self.objects if o.kind == "chat_message" and isinstance(o.extra, dict)]
        if not msgs:
            return 0
        # Prefer CUDA
        try:
            import torch  # type: ignore
            if not torch.cuda.is_available():
                return 0
        except Exception:
            return 0
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            mdl = SentenceTransformer(model_name, device='cuda')
        except Exception:
            return 0
        texts = [o.text or o.label for o in msgs]
        import numpy as _np  # type: ignore
        emb = _np.asarray(mdl.encode(texts, convert_to_numpy=True, device='cuda'), dtype=float)
        # L2-normalize
        norms = _np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        emb = emb / norms
        # Reduce to dims_out by slicing
        d = min(dims_out, emb.shape[1])
        emb32 = emb[:, :d]
        # If smaller than dims_out, pad with zeros
        if d < dims_out:
            pad = _np.zeros((emb32.shape[0], dims_out - d), dtype=float)
            emb32 = _np.concatenate([emb32, pad], axis=1)
        # Write back
        n = 0
        for i, o in enumerate(msgs):
            vec = [float(x) for x in emb32[i].tolist()]
            o.extra["embedding32"] = vec
            n += 1
        self._save()
        return n

    def bootstrap_books(self, limit: int = 24):
        p = ROOT / "data" / "ai_books_basic.json"
        if not p.exists():
            return 0
        data = json.loads(p.read_text(encoding="utf-8"))
        entries = data.get("entries", [])[:limit]
        self.add_room("Books", "Long-term knowledge books")
        for e in entries:
            title = e.get("title") or e.get("id")
            if not title:
                continue
            self.add_object("Books", str(title), "book entry")
        return len(entries)

    def bootstrap_defaults(self) -> None:
        # Core rooms from the Cognitive House prompt
        self.add_room("Living Area", "Human-AI interface zone")
        self.add_room("Study", "Avatar & knowledge interaction")
        self.add_room("Workshop", "Cognitive Fabrication Lab")
        self.add_room("Dream", "Internal galaxy projection / sleep")
        self.add_room("Knowledge Garden", "Ontology greenhouse — trees of knowledge")
        self.add_room("Network", "OSI doors and ports")

        # Furniture per room
        self.add_furniture("Living Area", "sofa", "Sofa")
        self.add_furniture("Living Area", "screen", "Floating Screen")

        self.add_furniture("Study", "bookshelf", "Bookshelf")
        self.add_furniture("Study", "book", "Active Book")

        self.add_furniture("Workshop", "bench", "Workbench")
        self.add_furniture("Workshop", "server_rack", "Server Rack")

        self.add_furniture("Dream", "tub", "Dream Chamber")
        self.add_furniture("Dream", "projector", "Galaxy Projector")
        # Garden fixtures
        self.add_furniture("Knowledge Garden", "greenhouse", "Indoor Greenhouse")
        self.add_furniture("Knowledge Garden", "bench", "Reading Bench")

        # Doors in Network
        self.add_door("Internet Door", "k3d://net?label=internet")
        self.add_door("Local Network Door", "k3d://lan?label=local")
        self.add_door("Sensor Door", "k3d://sensors?label=sensors")
        # Garden scene door
        self.add_door("Knowledge Garden", "/knowledge_garden.glb", room="Network")

    def bootstrap_reflections(self, limit: int = 50) -> int:
        """Add objects from docs/reports/reflections/*.md into 'Reflections' room."""
        ref_dir = ROOT / "docs" / "reports" / "reflections"
        files = sorted([p for p in ref_dir.glob("*.md")])[:limit]
        if not files:
            return 0
        self.add_room("Reflections", "Agent and human reflections")
        n = 0
        for p in files:
            try:
                text = p.read_text(encoding="utf-8")
                title = text.splitlines()[0].strip("# ") if text.splitlines() else p.stem
                label = f"{title}"[:80]
                self.add_object("Reflections", label, f"file:{p.as_posix()}")
                n += 1
            except Exception:
                continue
        return n

    def bootstrap_training(self, limit: int = 50) -> int:
        """Add objects from docs/reports/training/session-*.md and tasks-*.json into 'Experiments' room."""
        tr_dir = ROOT / "docs" / "reports" / "training"
        sessions = sorted([p for p in tr_dir.glob("session-*.md")])[:limit]
        tasks = sorted([p for p in tr_dir.glob("tasks-*.json")])[:limit]
        count = 0
        self.add_room("Experiments", "Training sessions and tasks")
        for p in sessions:
            try:
                lines = p.read_text(encoding="utf-8").splitlines()
                head = lines[0] if lines else p.stem
                label = head.replace("#", "").strip()[:80]
                self.add_object("Experiments", label, f"file:{p.as_posix()}")
                count += 1
            except Exception:
                continue
        for p in tasks:
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
                label = f"Tasks — {p.stem}"[:80]
                self.add_object("Experiments", label, f"file:{p.as_posix()}")
                count += 1
            except Exception:
                continue
        return count

    def bootstrap_standard(self) -> int:
        """Seed rooms and objects from core K3D standard docs/specs."""
        mapping = {
            "Standards & Specs": [
                ROOT / "spec" / "glTF_K3D_extension.md",
                ROOT / "spec" / "k3d_node_schema.json",
                ROOT / "spec" / "k3d_agent_protocol.md",
                ROOT / "spec" / "AI_RPN_standard.md",
            ],
            "Architecture": [
                ROOT / "docs" / "ARCHITECTURE.md",
                ROOT / "docs" / "K3D_Arch-From_Training_Base_Model_to_Web4.0.md",
            ],
            "Ethics & Care": [
                ROOT / "docs" / "ETHICS.md",
                ROOT / "docs" / "CARE_PROTOCOL.md",
            ],
            "Audio & Voice": [
                ROOT / "docs" / "AUDIO_ARCH.md",
            ],
            "HR/MR": [
                ROOT / "docs" / "HR_MR_STANDARD.md",
                ROOT / "docs" / "DUAL_CODE.md",
            ],
            "Philosophy": [
                ROOT / "docs" / "PHILOSOPHY.md",
            ],
            "Concepts": [
                ROOT / "docs" / "CONCEPTS.md",
            ],
            "AI Avatar": [
                ROOT / "docs" / "images" / "cognitive_house_prompt.md",
            ],
        }
        total = 0
        for room, files in mapping.items():
            self.add_room(room, room)
            for p in files:
                try:
                    if not p.exists():
                        continue
                    if p.suffix.lower() in {".md", ".json", ".html", ".txt"}:
                        title = p.stem.replace('_', ' ')
                        label = title[:80]
                        self.add_object(room, label, f"file:{p.as_posix()}")
                        total += 1
                except Exception:
                    continue
        return total

    def bootstrap_door_map(self) -> int:
        """Add inter-house doors to the Network room for quick navigation."""
        self.add_room("Network", "OSI doors and ports")
        doors = [
            ("Memory House", "/memory_house.glb"),
            ("Knowledge Garden", "/knowledge_garden.glb"),
            ("AI Compendium 1k", "/ai_compendium.1k.umap.doors.glb"),
            ("AI Compendium 4k", "/ai_compendium.4k.umap.doors.glb"),
            ("AI Books 4k", "/ai_books_basic.4k.umap.doors.glb"),
            ("AI Books Full", "/ai_books_basic.full.umap.doors.glb"),
            ("AI Care Multilang", "/ai_care_multilang.umap.glb"),
            ("AI Care Ancient", "/ai_care_ancient.umap.glb"),
        ]
        n = 0
        for label, addr in doors:
            self.add_door(label, addr, room="Network")
            n += 1
        return n

    def bootstrap_diary(self) -> int:
        """Add diary entries as objects in a 'Diary' room from docs/reports/diary."""
        ddir = ROOT / 'docs' / 'reports' / 'diary'
        if not ddir.exists():
            return 0
        self.add_room('Diary', 'Daily notes and learning')
        n=0
        for p in sorted(ddir.glob('diary-*.md')):
            try:
                lines = p.read_text(encoding='utf-8').splitlines()
                for ln in lines:
                    if ln.startswith('- ['):
                        # format: - [HH:MM -03:00] nick: text
                        label = ln[2:].strip()
                        self.add_object('Diary', label[:80], f'file:{p.as_posix()}')
                        n+=1
            except Exception:
                continue
        return n

    def export_gltf(self, out_path: Path):
        rooms = list(self.rooms.values())
        objs = self.objects
        ids: List[str] = []
        vectors: List[List[float]] = []
        embeddings: List[List[float]] = []
        metadata: List[Dict] = []
        neighbors: List[List[str]] = []

        # layout rooms on a circle
        R = 6.0
        n = max(1, len(rooms))
        room_pos: Dict[str, Tuple[float, float, float]] = {}
        for i, r in enumerate(rooms):
            a = (i / n) * math.tau
            x, y, z = math.cos(a) * R, math.sin(a) * R, 0.0
            room_pos[r.name] = (x, y, z)
            ids.append(r.id)
            vectors.append([x, y, z])
            embeddings.append(_hash_vec(r.name, 32))
            metadata.append({"label": r.name, "type": "room", "desc": r.desc, "layer": "rooms"})
            neighbors.append([])

        # group objects by room
        obj_by_room: Dict[str, List[Obj]] = {}
        for o in objs:
            obj_by_room.setdefault(o.room, []).append(o)

        # place objects around their room
        for room_name, group in obj_by_room.items():
            base = room_pos.get(room_name, (0.0, 0.0, 0.0))
            m = max(1, len(group))
            r = 2.2
            for j, o in enumerate(group):
                a = (j / m) * math.tau
                x, y, z = base[0] + math.cos(a) * r, base[1] + math.sin(a) * r, 0.0
                ids.append(o.id)
                vectors.append([x, y, z])
                # Prefer explicit embedding for diary pages
                e32 = (o.extra or {}).get("embedding32") if isinstance(o.extra, dict) else None
                if isinstance(e32, list) and len(e32) == 32:
                    embeddings.append(list(map(float, e32)))
                else:
                    embeddings.append(_hash_vec(o.label + "|" + o.text, 32))
                md = {"label": o.label, "type": o.kind, "room": room_name, "layer": room_name}
                md.update(o.extra or {})
                metadata.append(md)
                neighbors.append([])

        # connect rooms to their objects
        id_index = {ids[i]: i for i in range(len(ids))}
        for o in objs:
            ri = id_index.get(self.rooms[o.room].id)
            oi = id_index.get(o.id)
            if ri is not None and oi is not None:
                neighbors[ri].append(o.id)
                neighbors[oi].append(self.rooms[o.room].id)
            # Connect diary pages to their book
            if isinstance(o.extra, dict) and o.kind == "diary_page":
                parent_id = o.extra.get("parent")
                pi = id_index.get(parent_id) if parent_id else None
                if pi is not None and oi is not None:
                    neighbors[pi].append(o.id)
                    neighbors[oi].append(parent_id)
            # Connect chat messages to their chat book and previous message
            if isinstance(o.extra, dict) and o.kind == "chat_message":
                parent_id = o.extra.get("parent")
                pi = id_index.get(parent_id) if parent_id else None
                if pi is not None and oi is not None:
                    neighbors[pi].append(o.id)
                    neighbors[oi].append(parent_id)
                prev_id = o.extra.get("prev")
                pj = id_index.get(prev_id) if isinstance(prev_id, str) else None
                if pj is not None and oi is not None:
                    neighbors[oi].append(prev_id)  # link to previous
                    neighbors[pj].append(o.id)     # link back

        # Optional consolidation: add KNN links among non-room nodes (objects/doors), exclude furniture for clarity
        try:
            import numpy as np  # type: ignore
            K = 3
            obj_idx = [i for i, md in enumerate(metadata) if md.get('type') in {'object', 'door'}]
            if len(obj_idx) > K:
                emb = np.array([embeddings[i] for i in obj_idx], dtype=float)
                # cosine similarity
                norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
                X = emb / norms
                S = X @ X.T
                for ii, gi in enumerate(obj_idx):
                    sims = S[ii]
                    order = np.argsort(-sims)
                    added = 0
                    for jj in order:
                        if jj == ii: continue
                        gj = obj_idx[int(jj)]
                        # link ids[gi] <-> ids[gj]
                        if ids[gj] not in neighbors[gi]:
                            neighbors[gi].append(ids[gj])
                        if ids[gi] not in neighbors[gj]:
                            neighbors[gj].append(ids[gi])
                        added += 1
                        if added >= K: break
        except Exception:
            pass

        vectors_np = np.asarray(vectors, dtype=np.float32)
        embeddings_np = np.asarray(embeddings, dtype=np.float32)

        pos_bytes = vectors_np.tobytes()
        emb_bytes = embeddings_np.tobytes()
        blob = pos_bytes + emb_bytes

        buffer = Buffer(byteLength=len(blob))
        view_pos = BufferView(buffer=0, byteOffset=0, byteLength=len(pos_bytes), target=ARRAY_BUFFER)
        view_emb = BufferView(buffer=0, byteOffset=len(pos_bytes), byteLength=len(emb_bytes))
        accessor = Accessor(
            bufferView=0,
            byteOffset=0,
            componentType=5126,
            count=vectors_np.shape[0],
            type="VEC3",
            max=vectors_np.max(axis=0).tolist(),
            min=vectors_np.min(axis=0).tolist(),
        )

        k3d_payload = {
            "ids": ids,
            "metadata": metadata,
            "neighbors": neighbors,
            "vectorsView": 0,
            "embeddingsView": 1,
            "embeddingPrecision": "f32",
            "embeddingDims": int(embeddings_np.shape[1]),
            "ai_interaction_protocol": "direct_vector_manipulation",
            "ai_state_flags": {"is_active": True, "is_traversable": True},
        }

        primitive = Primitive(attributes={"POSITION": 0}, mode=0, extras={"k3d": k3d_payload})
        mesh = Mesh(primitives=[primitive])
        node = Node(mesh=0, name="k3d-house-memory")
        scene = Scene(nodes=[0])

        gltf = GLTF2(
            asset=Asset(generator="knowledge3d.tools.house_memory"),
            buffers=[buffer],
            bufferViews=[view_pos, view_emb],
            accessors=[accessor],
            meshes=[mesh],
            nodes=[node],
            scenes=[scene],
            scene=0,
        )
        gltf.set_binary_blob(blob)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gltf.save(out_path.as_posix())


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Manage long-term House Memory and export K3D GLTF")
    p.add_argument("--add-room", nargs=2, metavar=("NAME", "DESC"))
    p.add_argument("--add-object", nargs=3, metavar=("ROOM", "LABEL", "TEXT"))
    p.add_argument("--add-furniture", nargs=3, metavar=("ROOM", "KIND", "LABEL"))
    p.add_argument("--add-door", nargs=2, metavar=("LABEL", "ADDRESS"))
    p.add_argument("--bootstrap-books", type=int)
    p.add_argument("--bootstrap-defaults", action="store_true")
    p.add_argument("--bootstrap-reflections", type=int)
    p.add_argument("--bootstrap-training", type=int)
    p.add_argument("--bootstrap-standard", action="store_true")
    p.add_argument("--bootstrap-doors", action="store_true")
    p.add_argument("--bootstrap-diary", action="store_true")
    p.add_argument("--export", help="Output GLTF path", default=str(ROOT / "viewer" / "public" / "memory_house.glb"))
    args = p.parse_args()
    h = MemoryHouse()
    if args.add_room:
        name, desc = args.add_room
        h.add_room(name, desc)
    if args.add_object:
        room, label, text = args.add_object
        h.add_object(room, label, text)
    if args.add_furniture:
        room, kind, label = args.add_furniture
        h.add_furniture(room, kind, label)
    if args.add_door:
        label, address = args.add_door
        h.add_door(label, address)
    if args.bootstrap_books:
        n = h.bootstrap_books(args.bootstrap_books)
        print(f"Bootstrapped {n} books into room 'Books'")
    if args.bootstrap_defaults:
        h.bootstrap_defaults()
        print("Bootstrapped default rooms + furniture + doors")
    if args.bootstrap_reflections:
        n = h.bootstrap_reflections(args.bootstrap_reflections)
        print(f"Bootstrapped {n} reflections into room 'Reflections'")
    if args.bootstrap_training:
        n = h.bootstrap_training(args.bootstrap_training)
        print(f"Bootstrapped {n} training artifacts into room 'Experiments'")
    if args.bootstrap_standard:
        n = h.bootstrap_standard()
        print(f"Bootstrapped {n} standard/spec artifacts")
    if args.bootstrap_doors:
        n = h.bootstrap_door_map()
        print(f"Bootstrapped {n} doors into 'Network'")
    if args.bootstrap_diary:
        n = h.bootstrap_diary()
        print(f"Bootstrapped {n} diary entries into 'Diary'")
    out = Path(args.export)
    h.export_gltf(out)
    print(f"Exported House Memory to {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
