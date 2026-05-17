from __future__ import annotations

"""
Register text GLB assets under viewer/public/text into the House manifest.

Each GLB becomes a manifest entry with fields: {path, name, prompt, type}.

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.tools.register_text_glbs
"""

import json
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "viewer" / "public"
TEXT_DIR = PUBLIC / "text"
MANIFEST = PUBLIC / "house" / "materialized_objects" / "manifest.json"


def main() -> None:  # pragma: no cover
    if not TEXT_DIR.exists():
        print("no text dir; nothing to register")
        return
    try:
        obj = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"shapes": [], "rays": []}
    except Exception:
        obj = {"shapes": [], "rays": []}
    shapes = obj.get("shapes") if isinstance(obj, dict) else []
    if not isinstance(shapes, list):
        shapes = []
    existing = {s.get("path") for s in shapes if isinstance(s, dict)}
    added: List[str] = []
    for glb in sorted(TEXT_DIR.glob("*.glb")):
        rel = "/" + str(glb.relative_to(PUBLIC)).replace("\\", "/")
        if rel in existing:
            continue
        entry = {
            "path": rel,
            "name": glb.stem,
            "prompt": f"text {glb.stem}",
            "type": "external_text",
        }
        shapes.append(entry)
        added.append(rel)
    obj["shapes"] = shapes
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    if added:
        print("registered:")
        for a in added:
            print(a)
    else:
        print("no new text GLBs to register")


if __name__ == "__main__":
    main()

