"""
House Memory builder: rooms + objects as long-term memory in a K3D GLTF.

- Stores a simple JSON state at data/memory_house.json
- Exports a GLTF with primitive.extras.k3d embedding (vectors/embeddings/metadata/neighbors)
  consumable by the K3D viewer.

Usage
  # add a room and an object, then export
  python3 -m knowledge3d.tools.house_memory --add-room "Books" --desc "Long-term knowledge books"
  python3 -m knowledge3d.tools.house_memory --add-object "Books" "Fine art" "Canonical article"
  python3 -m knowledge3d.tools.house_memory --export viewer/public/memory_house.gltf

  # bootstrap from AI books (first N titles)
  python3 -m knowledge3d.tools.house_memory --bootstrap-books 24 --export viewer/public/memory_house.gltf
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "data" / "memory_house.json"


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

        # Doors in Network
        self.add_door("Internet Door", "k3d://net?label=internet")
        self.add_door("Local Network Door", "k3d://lan?label=local")
        self.add_door("Sensor Door", "k3d://sensors?label=sensors")

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

        # pack GLTF json with embedded arrays
        payload = {
            "ids": ids,
            "vectors": vectors,
            "embeddings": embeddings,
            "embeddingPrecision": "f32",
            "embeddingDims": 32,
            "metadata": metadata,
            "neighbors": neighbors,
            "ai_interaction_protocol": "direct_vector_manipulation",
            "ai_state_flags": {"is_active": True, "is_traversable": True},
        }
        gltf = {
            "asset": {"version": "2.0"},
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": "k3d-house-memory"}],
            "meshes": [{"primitives": [{"mode": 0, "extras": {"k3d": payload}}]}],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(gltf, ensure_ascii=False, indent=2), encoding="utf-8")


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
    p.add_argument("--export", help="Output GLTF path", default=str(ROOT / "viewer" / "public" / "memory_house.gltf"))
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
    out = Path(args.export)
    h.export_gltf(out)
    print(f"Exported House Memory to {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
