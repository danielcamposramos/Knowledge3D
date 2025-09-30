"""Train the fused head's shape classifier using House materialized shapes.

Reads viewer/public/house/materialized_objects/manifest.json, extracts shape
entries with 'prompt' and 'shape_type', and trains the fused head's shape_head
to classify the type from the prompt. Packs weights into the House GLB as
appliance 'fused_shape'.

Usage:
  PYTHONPATH=. K3D_PTX_STRICT=1 K3D_FORCE_PTX_FUSE=1 \
  python -m knowledge3d.tools.phase25.shapes_trainer --epochs 10 --limit 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from knowledge3d.cranium.fused_head import AdaptedFusedHead  # type: ignore
from knowledge3d.tools.weights_in_glb import pack_pt_into_glb  # type: ignore


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "viewer/public/house/materialized_objects/manifest.json"
HOUSE_GLB = ROOT / "viewer/public/houses/default/memory_house.glb"


def load_shape_pairs(limit: Optional[int] = None) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not MANIFEST.exists():
        return pairs
    try:
        obj = json.loads(MANIFEST.read_text(encoding="utf-8"))
        shapes = obj.get("shapes") if isinstance(obj, dict) else None
        if not isinstance(shapes, list):
            return pairs
        for it in shapes:
            if not isinstance(it, dict):
                continue
            prompt = it.get("prompt") or it.get("name")
            stype = it.get("shape_type")
            if isinstance(prompt, str) and isinstance(stype, str) and prompt.strip() and stype.strip():
                pairs.append((prompt.strip(), stype.strip()))
                if limit and len(pairs) >= int(limit):
                    break
    except Exception:
        return pairs
    return pairs


def run(epochs: int, limit: int) -> None:
    fh = AdaptedFusedHead()
    # Create shape type index map from fused head
    shapes = fh._shapes  # ['tetrahedron', 'cube', ...]
    s2i: Dict[str, int] = {s: i for i, s in enumerate(shapes)}
    ds = load_shape_pairs(limit)
    if not ds:
        print("⚠️  No shape samples found in manifest; generate some shapes first.")
        return
    # Filter to known shape types
    samples: List[Tuple[str, int]] = []
    for prompt, stype in ds:
        if stype in s2i:
            samples.append((prompt, s2i[stype]))
    if not samples:
        print("⚠️  No samples with known shape types.")
        return
    print(f"📦 Shape samples: {len(samples)}")
    for ep in range(1, int(max(1, epochs)) + 1):
        losses: List[float] = []
        for prompt, label in samples:
            losses.append(fh.shape_train_step(prompt, label))
        avg = sum(losses) / max(1, len(losses))
        print(f"🧩 Shapes Epoch {ep}: avg_loss={avg:.4f} ({len(losses)} samples)")
        fh._save_shape_head()
    # Pack into GLB as fused_shape
    ckpt = ROOT / 'viewer/public/house/house_shape_head.pt'
    if HOUSE_GLB.exists() and ckpt.exists():
        try:
            pack_pt_into_glb(HOUSE_GLB, ckpt, 'fused_shape')
            print(f"📦 Packed fused_shape into {HOUSE_GLB}")
        except Exception as e:
            print(f"⚠️  Failed to pack fused_shape: {e}")
    print("✅ Shapes training complete.")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Train fused head shape classifier from House manifest")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()
    run(int(args.epochs), int(args.limit))


if __name__ == "__main__":
    main()

