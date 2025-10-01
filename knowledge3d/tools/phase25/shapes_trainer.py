"""Shapes trainer — consistency-style alignment for general shapes (PTX-only).

Reads viewer/public/house/materialized_objects/manifest.json, ensures or creates
image previews for GLB shapes, and trains the projection via normalized MSE to
align to PTX image features (consistency loss). Re-initializes shape_head for
future supervised tasks, and packs fused_shape once training is complete.

Usage:
  PYTHONPATH=. K3D_PTX_STRICT=1 K3D_FORCE_PTX_FUSE=1 \
  python -m knowledge3d.tools.phase25.shapes_trainer --epochs 50 --limit 2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from knowledge3d.cranium.fused_head import AdaptedFusedHead  # type: ignore
from knowledge3d.cranium.ptx.ptx_ops import PTX_OPS  # type: ignore
from knowledge3d.tools.weights_in_glb import pack_pt_into_glb  # type: ignore


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "viewer/public/house/materialized_objects/manifest.json"
HOUSE_GLB = ROOT / "viewer/public/houses/default/memory_house.glb"


def load_shape_pairs(limit: Optional[int] = None) -> List[Tuple[str, str, Optional[str]]]:
    pairs: List[Tuple[str, str, Optional[str]]] = []
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
            preview = it.get("preview")
            if isinstance(prompt, str) and prompt.strip():
                pairs.append((prompt.strip(), str(stype) if stype else "", str(preview) if preview else None))
                if limit and len(pairs) >= int(limit):
                    break
    except Exception:
        return pairs
    return pairs


def _ensure_preview(glb_relative: str) -> Optional[Path]:
    """Generate a simple 2D preview (PNG) by projecting vertices to XY plane."""
    try:
        from pygltflib import GLTF2  # type: ignore
        from PIL import Image, ImageDraw  # type: ignore
    except Exception:
        return None
    rel = glb_relative[1:] if glb_relative.startswith('/') else glb_relative
    glb_path = ROOT / "viewer/public" / rel
    if not glb_path.exists():
        return None
    png_path = glb_path.with_suffix('.png')
    try:
        g = GLTF2().load(glb_path.as_posix())
        blob = g.binary_blob()
        if blob is None:
            return None
        # Assume first mesh positions in first bufferView
        # Find POSITION accessor 0
        acc = g.accessors[0]
        bv = g.bufferViews[acc.bufferView]
        off = int(getattr(bv, 'byteOffset', 0) or 0)
        ln = int(getattr(bv, 'byteLength', 0) or 0)
        data = np.frombuffer(memoryview(blob)[off:off+ln], dtype=np.float32)
        pts = data.reshape(-1, 3)
        # Normalize to [0,1] and draw
        if pts.size == 0:
            return None
        xy = pts[:, :2]
        mn = xy.min(axis=0); mx = xy.max(axis=0)
        span = np.maximum(mx - mn, 1e-6)
        uv = (xy - mn) / span
        size = 256
        im = Image.new('L', (size, size), 0)
        dr = ImageDraw.Draw(im)
        for u, v in uv:
            x = int(u * (size - 1)); y = int((1.0 - v) * (size - 1))
            dr.point((x, y), fill=255)
        im = im.filter(__import__('PIL', fromlist=['ImageFilter']).ImageFilter.GaussianBlur(0.8))
        im.save(png_path.as_posix())
        return png_path
    except Exception:
        return None


def _attach_preview_to_manifest() -> None:
    try:
        obj = json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {"shapes": [], "rays": []}
    except Exception:
        return
    changed = False
    for it in obj.get('shapes', []):
        if not isinstance(it, dict):
            continue
        prev = it.get('preview')
        path = it.get('path')
        if not prev and isinstance(path, str):
            png = _ensure_preview(path)
            if png is not None:
                rel = "/" + str(png.relative_to(ROOT / 'viewer' / 'public')).replace(os.sep, '/')
                it['preview'] = rel
                changed = True
    if changed:
        MANIFEST.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def run(epochs: int, limit: int) -> None:
    fh = AdaptedFusedHead()
    # Re-init shape head (new classes may be different); ensure previews
    _attach_preview_to_manifest()
    ds = load_shape_pairs(limit)
    if not ds:
        print("⚠️  No shape samples found in manifest; generate some shapes first.")
        return
    # Consistency-style alignment: align projection to PTX image features from previews
    samples: List[Tuple[str, Path]] = []
    for prompt, _stype, preview in ds:
        if preview:
            rel = preview[1:] if preview.startswith('/') else preview
            p = ROOT / 'viewer/public' / rel
            if p.exists():
                samples.append((prompt, p))
    if not samples:
        print("⚠️  No previews available for shapes; generate previews first.")
        return
    print(f"📦 Shape previews: {len(samples)}")
    # Use normalized MSE loss similar to consistency trainer
    for ep in range(1, int(max(1, epochs)) + 1):
        losses: List[float] = []
        samples_used = 0
        skipped_nonfinite = 0
        first_loss: Optional[float] = None
        for prompt, p in samples:
            try:
                info = PTX_OPS.image_modality(p.as_posix())
                feats = info.get('features') if isinstance(info, dict) else None
                if not isinstance(feats, list) or not feats:
                    continue
                target = np.asarray(feats, dtype=np.float32)
            except Exception:
                continue
            x_vec = fh._build_ptx_fused_embedding(prompt)
            x = torch.tensor(x_vec, dtype=torch.float32, device=fh.device).unsqueeze(0)
            if x.shape[1] < 2048:
                x = torch.cat([x, torch.zeros((1, 2048 - x.shape[1]), device=fh.device)], dim=1)
            elif x.shape[1] > 2048:
                x = x[:, :2048]
            fh.projection.train()
            h = fh.projection(x)
            t = torch.tensor(target[:512], dtype=torch.float32, device=fh.device).unsqueeze(0)
            h_n = torch.nn.functional.normalize(h, dim=-1, eps=1e-6)
            t_n = torch.nn.functional.normalize(t, dim=-1, eps=1e-6)
            loss = torch.mean((h_n - t_n) ** 2)
            if not torch.isfinite(loss):
                fh._opt.zero_grad(set_to_none=True); skipped_nonfinite += 1; continue
            fh._opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(fh.projection.parameters()), 1.0)
            fh._opt.step()
            val = float(loss.detach().item())
            if first_loss is None:
                first_loss = val
            losses.append(val)
            samples_used += 1
        avg = sum(losses) / max(1, len(losses))
        print(f"🧩 Shapes Consistency Epoch {ep}: avg_loss={avg:.4f} (samples_used={samples_used}, skipped_nonfinite={skipped_nonfinite}) first_loss={first_loss}")
        if samples_used == 0:
            print("⚠️  No finite samples used in shapes epoch — check PTX features and projection path.")
        fh._save_core_heads()
        # Progress log
        try:
            import json, time
            out = ROOT / 'docs/benchmarks/progress_log.json'
            out.parent.mkdir(parents=True, exist_ok=True)
            log = []
            if out.exists():
                try:
                    log = json.loads(out.read_text(encoding='utf-8'))
                except Exception:
                    log = []
            log.append({'trainer':'shapes','epoch':ep,'avg_loss':float(avg),'ts':time.time()})
            out.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass
    # Pack into GLB as fused_shape
    ckpt = fh._save_core_heads()
    if HOUSE_GLB.exists() and ckpt.exists():
        try:
            pack_pt_into_glb(HOUSE_GLB, ckpt, 'fused_core')
            print(f"📦 Packed fused_core into {HOUSE_GLB}")
        except Exception as e:
            print(f"⚠️  Failed to pack fused_core: {e}")
    print("✅ Shapes training complete.")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Train fused head shape classifier from House manifest")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()
    run(int(args.epochs), int(args.limit))


if __name__ == "__main__":
    main()
