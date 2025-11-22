"""
Drawing Grammar builder: emits procedural drawing grammar stars across layers.

Layers emitted:
- primitives: line, arc, quad_bezier, cubic_bezier, circle, rectangle, triangle
- strokes: stroke variants referencing primitives with styling attrs
- shapes/icons: simple motifs (arrow, box, circle_fill) referencing strokes
- scenes: small layouts referencing shapes
- collections: book/gallery referencing scenes

Outputs JSONL stars with procedural_programs placeholders (visual_rpn refs, transforms)
and hierarchical refs (primitives→strokes→shapes→scenes→collections).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict


def build_primitives() -> List[Dict]:
    primitives = [
        {"id": "PRIM_LINE", "type": "primitive", "visual_rpn": "LINE x0 y0 x1 y1"},
        {"id": "PRIM_ARC", "type": "primitive", "visual_rpn": "ARC cx cy r theta0 theta1"},
        {"id": "PRIM_QUAD", "type": "primitive", "visual_rpn": "QUAD x0 y0 x1 y1 x2 y2"},
        {"id": "PRIM_CUBIC", "type": "primitive", "visual_rpn": "CUBIC x0 y0 x1 y1 x2 y2 x3 y3"},
        {"id": "PRIM_CIRCLE", "type": "primitive", "visual_rpn": "CIRCLE cx cy r"},
        {"id": "PRIM_RECT", "type": "primitive", "visual_rpn": "RECT x y w h"},
        {"id": "PRIM_TRI", "type": "primitive", "visual_rpn": "TRI x0 y0 x1 y1 x2 y2"},
    ]
    return primitives


def build_strokes(primitives: List[Dict]) -> List[Dict]:
    strokes = []
    for prim in primitives:
        strokes.append(
            {
                "id": f"STROKE_{prim['id']}",
                "type": "stroke",
                "primitive_ref": prim["id"],
                "style": {"width": 1.0, "color": "#ffffff", "join": "miter", "cap": "butt"},
                "procedural_programs": {"visual_rpn": prim["visual_rpn"], "transform": "IDENTITY"},
            }
        )
    return strokes


def build_shapes(strokes: List[Dict]) -> List[Dict]:
    shapes = [
        {
            "id": "SHAPE_ARROW_RIGHT",
            "type": "shape",
            "stroke_refs": [s["id"] for s in strokes if s["primitive_ref"] in {"PRIM_LINE", "PRIM_TRI"}],
            "procedural_programs": {"composition": "ARROW_RIGHT"},
        },
        {
            "id": "SHAPE_BOX",
            "type": "shape",
            "stroke_refs": [s["id"] for s in strokes if s["primitive_ref"] == "PRIM_RECT"],
            "procedural_programs": {"composition": "BOX"},
        },
        {
            "id": "SHAPE_CIRCLE_FILL",
            "type": "shape",
            "stroke_refs": [s["id"] for s in strokes if s["primitive_ref"] == "PRIM_CIRCLE"],
            "procedural_programs": {"composition": "CIRCLE_FILL"},
        },
    ]
    return shapes


def build_scenes(shapes: List[Dict]) -> List[Dict]:
    scenes = [
        {
            "id": "SCENE_ARROW_SEQUENCE",
            "type": "scene",
            "shape_refs": ["SHAPE_ARROW_RIGHT"] * 3,
            "layout": "HORIZONTAL",
        },
        {
            "id": "SCENE_ICON_GRID",
            "type": "scene",
            "shape_refs": [s["id"] for s in shapes],
            "layout": "GRID",
        },
    ]
    return scenes


def build_collections(scenes: List[Dict]) -> List[Dict]:
    collections = [
        {
            "id": "COLLECTION_DRAWING_BOOK",
            "type": "collection",
            "scene_refs": [s["id"] for s in scenes],
            "metadata": {"title": "Drawing Grammar Book"},
        }
    ]
    return collections


def write_jsonl(items: List[Dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build drawing grammar JSONL (primitives→strokes→shapes→scenes→collections).")
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    primitives = build_primitives()
    strokes = build_strokes(primitives)
    shapes = build_shapes(strokes)
    scenes = build_scenes(shapes)
    collections = build_collections(scenes)
    all_items = primitives + strokes + shapes + scenes + collections
    write_jsonl(all_items, args.output)
    print(f"Wrote {len(all_items)} drawing grammar stars -> {args.output}")


if __name__ == "__main__":
    main()
