"""Multi-modal consistency trainer (PTX-only).

Aligns the fused projection to PTX modality features using House assets.
Currently uses image assets from the House materialized manifest; easy to
extend to audio/video if present.

Usage:
  PYTHONPATH=. K3D_PTX_STRICT=1 K3D_FORCE_PTX_FUSE=1 \
  python -m knowledge3d.tools.phase25.consistency_trainer --epochs 5 --limit 1000 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from knowledge3d.cranium.fused_head import AdaptedFusedHead  # type: ignore
from knowledge3d.cranium.ptx.ptx_ops import PTX_OPS  # type: ignore
from knowledge3d.tools.weights_in_glb import pack_pt_into_glb  # type: ignore


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "viewer/public/house/materialized_objects/manifest.json"
HOUSE_GLB = ROOT / "viewer/public/houses/default/memory_house.glb"


def _expand_to_dim(features: List[float], dim: int) -> np.ndarray:
    arr = np.asarray([float(x) for x in features], dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros(dim, dtype=np.float32)
    if arr.size == dim:
        return arr
    if arr.size > dim:
        return arr[:dim]
    reps = int(np.ceil(dim / float(arr.size)))
    return np.tile(arr, reps)[:dim].astype(np.float32)


def load_image_pairs(limit: Optional[int] = None) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    if not MANIFEST.exists():
        return out
    try:
        obj = json.loads(MANIFEST.read_text(encoding="utf-8"))
        shapes = obj.get("shapes") if isinstance(obj, dict) else None
        if not isinstance(shapes, list):
            return out
        for it in shapes:
            if not isinstance(it, dict):
                continue
            prompt = it.get("prompt") or it.get("name")
            p = it.get("path")
            if not (isinstance(prompt, str) and isinstance(p, str)):
                continue
            rel = p[1:] if p.startswith("/") else p
            candidate = ROOT / "viewer/public" / rel
            if candidate.exists():
                out.append((prompt.strip(), candidate))
                if limit and len(out) >= int(limit):
                    break
    except Exception:
        return out
    return out


def run(epochs: int, limit: int, lr: float) -> None:
    fh = AdaptedFusedHead()
    # Adjust LR for projection group
    for g in fh._opt.param_groups:
        if any(p is next(fh.projection.parameters()) for p in g.get('params', [])):
            g['lr'] = float(lr)

    pairs = load_image_pairs(limit)
    if not pairs:
        print("⚠️  No image pairs found in manifest; generate shapes first.")
        return
    print(f"📦 Consistency image pairs: {len(pairs)}")

    for ep in range(1, int(max(1, epochs)) + 1):
        losses: List[float] = []
        for prompt, path in pairs:
            # Target PTX image features
            try:
                info = PTX_OPS.image_modality(path.as_posix())
                feats = info.get("features") if isinstance(info, dict) else None
                if not isinstance(feats, list):
                    continue
                target = _expand_to_dim(feats, 512)
            except Exception:
                continue
            # Project fused embedding of the prompt
            x_vec = fh._build_ptx_fused_embedding(prompt)
            x = torch.tensor(x_vec, dtype=torch.float32, device=fh.device).unsqueeze(0)
            if x.shape[1] < 2048:
                x = torch.cat([x, torch.zeros((1, 2048 - x.shape[1]), device=fh.device)], dim=1)
            elif x.shape[1] > 2048:
                x = x[:, :2048]
            fh.projection.train()
            h = fh.projection(x)
            t = torch.tensor(target, dtype=torch.float32, device=fh.device).unsqueeze(0)
            # Normalize
            h_n = torch.nn.functional.normalize(h, dim=-1)
            t_n = torch.nn.functional.normalize(t, dim=-1)
            loss = torch.mean((h_n - t_n) ** 2)
            if not torch.isfinite(loss):
                fh._opt.zero_grad(set_to_none=True)
                continue
            fh._opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(fh.projection.parameters()), 1.0)
            fh._opt.step()
            losses.append(float(loss.detach().item()))
        avg = sum(losses) / max(1, len(losses))
        print(f"🧭 Consistency Epoch {ep}: avg_loss={avg:.4f} ({len(losses)} samples)")
        fh._save_core_heads()

    # Pack updated core heads into GLB
    core_pt = fh._save_core_heads()
    if HOUSE_GLB.exists() and core_pt.exists():
        try:
            pack_pt_into_glb(HOUSE_GLB, core_pt, "fused_core")
            print(f"📦 Packed fused_core into {HOUSE_GLB}")
        except Exception as e:
            print(f"⚠️  Failed to pack fused_core: {e}")
    print("✅ Consistency training complete.")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Train fused projection for multi-modal consistency (PTX-only)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    run(int(args.epochs), int(args.limit), float(args.lr))


if __name__ == "__main__":
    main()

