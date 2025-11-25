"""
Drawing Galaxy container: atomic visual primitives → strokes → shapes → scenes.

This mirrors the builder in `knowledge3d/ingestion/atomic/drawing_grammar_builder.py`
and keeps an in-memory registry that can grow as new primitives/shapes are
discovered during ARC-AGI evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional

from knowledge3d.ingestion.atomic.drawing_grammar_builder import (
    build_primitives,
    build_strokes,
    build_shapes,
    build_scenes,
    build_collections,
)


@dataclass
class DrawingItem:
    item_id: str
    item_type: str
    payload: Dict


class DrawingGalaxy:
    """In-memory registry for drawing primitives and derived items."""

    def __init__(self) -> None:
        self.primitives: Dict[str, DrawingItem] = {}
        self.strokes: Dict[str, DrawingItem] = {}
        self.shapes: Dict[str, DrawingItem] = {}
        self.scenes: Dict[str, DrawingItem] = {}
        self.collections: Dict[str, DrawingItem] = {}
        self._bootstrap_defaults()

    # ------------------------------------------------------------------ #
    # Bootstrap
    # ------------------------------------------------------------------ #
    def _bootstrap_defaults(self) -> None:
        primitives = build_primitives()
        strokes = build_strokes(primitives)
        shapes = build_shapes(strokes)
        scenes = build_scenes(shapes)
        collections = build_collections(scenes)

        for prim in primitives:
            self.primitives[prim["id"]] = DrawingItem(prim["id"], "primitive", prim)
        for stroke in strokes:
            self.strokes[stroke["id"]] = DrawingItem(stroke["id"], "stroke", stroke)
        for shape in shapes:
            self.shapes[shape["id"]] = DrawingItem(shape["id"], "shape", shape)
        for scene in scenes:
            self.scenes[scene["id"]] = DrawingItem(scene["id"], "scene", scene)
        for col in collections:
            self.collections[col["id"]] = DrawingItem(col["id"], "collection", col)

    # ------------------------------------------------------------------ #
    # Discovery APIs
    # ------------------------------------------------------------------ #
    def add_shape(self, shape_id: str, rpn_program: str, source: Optional[Dict] = None) -> None:
        payload = {
            "id": shape_id,
            "type": "shape",
            "procedural_programs": {"composition": rpn_program},
        }
        if source:
            payload["discovered_from"] = source
        self.shapes[shape_id] = DrawingItem(shape_id, "shape", payload)

    def add_scene(self, scene_id: str, shape_refs: List[str], layout: str = "GRID", source: Optional[Dict] = None) -> None:
        payload = {"id": scene_id, "type": "scene", "shape_refs": list(shape_refs), "layout": layout}
        if source:
            payload["discovered_from"] = source
        self.scenes[scene_id] = DrawingItem(scene_id, "scene", payload)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, int]:
        return {
            "primitives": len(self.primitives),
            "strokes": len(self.strokes),
            "shapes": len(self.shapes),
            "scenes": len(self.scenes),
            "collections": len(self.collections),
        }

    def list_shapes(self, limit: Optional[int] = None) -> List[DrawingItem]:
        shapes = list(self.shapes.values())
        return shapes if limit is None else shapes[:limit]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        state = {
            "primitives": {k: v.payload for k, v in self.primitives.items()},
            "strokes": {k: v.payload for k, v in self.strokes.items()},
            "shapes": {k: v.payload for k, v in self.shapes.items()},
            "scenes": {k: v.payload for k, v in self.scenes.items()},
            "collections": {k: v.payload for k, v in self.collections.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        print(f"[DrawingGalaxy] Saved to {path} (shapes={len(self.shapes)})")

    def load(self, path: Path) -> None:
        if not path.exists():
            print(f"[DrawingGalaxy] No checkpoint at {path}, using bootstrap")
            return
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        def _load_map(data: Dict[str, Dict], kind: str) -> Dict[str, DrawingItem]:
            out: Dict[str, DrawingItem] = {}
            for k, payload in data.items():
                out[k] = DrawingItem(k, payload.get("type", kind), payload)
            return out

        self.primitives = _load_map(state.get("primitives", {}), "primitive") or self.primitives
        self.strokes = _load_map(state.get("strokes", {}), "stroke") or self.strokes
        self.shapes = _load_map(state.get("shapes", {}), "shape") or self.shapes
        self.scenes = _load_map(state.get("scenes", {}), "scene") or self.scenes
        self.collections = _load_map(state.get("collections", {}), "collection") or self.collections

        print(f"[DrawingGalaxy] Loaded {len(self.shapes)} shapes from {path}")

    def add_discovered_shape(self, shape: Dict) -> None:
        """Add discovered shape (recorded by shadow copy)."""
        shape_id = shape.get("id") or f"DISCOVERED_SHAPE_{len(self.shapes)}"
        payload = dict(shape)
        payload["id"] = shape_id
        self.shapes[shape_id] = DrawingItem(shape_id, payload.get("type", "shape"), payload)


__all__ = ["DrawingGalaxy", "DrawingItem"]
