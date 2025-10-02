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
import os
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


def load_image_pairs(limit: Optional[int] = None) -> List[Tuple[str, Path, Optional[str]]]:
    out: List[Tuple[str, Path, Optional[str]]] = []
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
            # Accept either 'path' (shapes) or 'preview' (documents) as image source
            p = it.get("path") or it.get("preview")
            ocr_text = it.get("ocr_text") if isinstance(it.get("ocr_text"), str) else None
            if not (isinstance(prompt, str) and isinstance(p, str)):
                continue
            rel = p[1:] if p.startswith("/") else p
            candidate = ROOT / "viewer/public" / rel
            if candidate.exists():
                out.append((prompt.strip(), candidate, ocr_text))
                if limit and len(out) >= int(limit):
                    break
    except Exception:
        return out
    return out


def _discover_media(roots: List[Path], exts: Tuple[str, ...], limit: int) -> List[Path]:
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob('*'):
            try:
                if p.is_file() and p.suffix.lower() in exts:
                    out.append(p)
                    if len(out) >= limit:
                        return out
            except Exception:
                continue
    return out


def run(epochs: int, limit: int, lr: float) -> None:
    fh = AdaptedFusedHead()
    # Adjust LR for projection group (robustly match any projection parameter)
    proj_params = list(fh.projection.parameters())
    proj_ids = {id(p) for p in proj_params}
    for g in fh._opt.param_groups:
        params = g.get('params', [])
        if any(id(p) in proj_ids for p in params):
            g['lr'] = float(lr)

    pairs = load_image_pairs(limit)
    print(f"📦 Consistency image pairs: {len(pairs)}")

    # Discover external audio/video assets
    roots = [
        Path('/home/daniel/K3D_llama_cpp/datasets'),
        Path('/K3D/Knowledge3D.local/datasets'),
    ]
    audio_exts = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
    video_exts = ('.mp4', '.mkv', '.webm', '.mov', '.avi')
    audio_files = _discover_media(roots, audio_exts, max(1, limit))
    video_files = _discover_media(roots, video_exts, max(1, limit))
    print(f"🔊 Audio files: {len(audio_files)} | 🎥 Video files: {len(video_files)}")

    for ep in range(1, int(max(1, epochs)) + 1):
        losses: List[float] = []
        samples_used = 0
        skipped_no_target = 0
        skipped_nonfinite = 0
        first_loss: Optional[float] = None
        # Image: manifest previews
        for prompt, path, ocr_text in pairs:
            # Target PTX image features (fallback: text modality features)
            target = None
            try:
                info = PTX_OPS.image_modality(path.as_posix())
                feats = info.get("features") if isinstance(info, dict) else None
                if isinstance(feats, list) and feats:
                    target = _expand_to_dim(feats, 512)
            except Exception:
                target = None
            use_text_fb = str(os.environ.get('K3D_CONSISTENCY_FALLBACK_TEXT','0')).lower() in {'1','true','yes'}
            # If image target missing or disabled, allow OCR/text features as target
            if target is None and use_text_fb:
                try:
                    tinput = (ocr_text or prompt)
                    tinfo = PTX_OPS.text_modality(tinput)
                    tfeats = tinfo.get("features") if isinstance(tinfo, dict) else None
                    if isinstance(tfeats, list) and tfeats:
                        target = _expand_to_dim(tfeats, 512)
                except Exception:
                    target = None
            if target is None:
                continue
            # Project fused embedding of prompt or OCR text when text-fallback mode
            embed_text = (ocr_text or prompt) if use_text_fb else prompt
            x_vec = fh._build_ptx_fused_embedding(embed_text)
            x = torch.tensor(x_vec, dtype=torch.float32, device=fh.device).unsqueeze(0)
            if x.shape[1] < 2048:
                x = torch.cat([x, torch.zeros((1, 2048 - x.shape[1]), device=fh.device)], dim=1)
            elif x.shape[1] > 2048:
                x = x[:, :2048]
            fh.projection.train()
            h = fh.projection(x)
            t = torch.tensor(target, dtype=torch.float32, device=fh.device).unsqueeze(0)
            # Normalize
            h_n = torch.nn.functional.normalize(h, dim=-1, eps=1e-6)
            t_n = torch.nn.functional.normalize(t, dim=-1, eps=1e-6)
            loss = torch.mean((h_n - t_n) ** 2)
            if not torch.isfinite(loss):
                fh._opt.zero_grad(set_to_none=True)
                skipped_nonfinite += 1
                continue
            fh._opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(fh.projection.parameters()), 1.0)
            fh._opt.step()
            val = float(loss.detach().item())
            if first_loss is None:
                first_loss = val
            losses.append(val)
            samples_used += 1
        # Audio alignment
        for af in audio_files:
            try:
                info = PTX_OPS.audio_modality(af.as_posix())
                feats = info.get('features') if isinstance(info, dict) else None
                if not isinstance(feats, list) or not feats:
                    continue
                target = _expand_to_dim(feats, 512)
            except Exception:
                continue
            # Use filename as pseudo-prompt
            prompt = af.stem
            x_vec = fh._build_ptx_fused_embedding(prompt)
            x = torch.tensor(x_vec, dtype=torch.float32, device=fh.device).unsqueeze(0)
            if x.shape[1] < 2048:
                x = torch.cat([x, torch.zeros((1, 2048 - x.shape[1]), device=fh.device)], dim=1)
            elif x.shape[1] > 2048:
                x = x[:, :2048]
            fh.projection.train()
            h = fh.projection(x)
            t = torch.tensor(target, dtype=torch.float32, device=fh.device).unsqueeze(0)
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

        # Video alignment
        for vf in video_files:
            try:
                info = PTX_OPS.video_modality(vf.as_posix())
                feats = info.get('features') if isinstance(info, dict) else None
                if not isinstance(feats, list) or not feats:
                    continue
                target = _expand_to_dim(feats, 512)
            except Exception:
                continue
            prompt = vf.stem
            x_vec = fh._build_ptx_fused_embedding(prompt)
            x = torch.tensor(x_vec, dtype=torch.float32, device=fh.device).unsqueeze(0)
            if x.shape[1] < 2048:
                x = torch.cat([x, torch.zeros((1, 2048 - x.shape[1]), device=fh.device)], dim=1)
            elif x.shape[1] > 2048:
                x = x[:, :2048]
            fh.projection.train()
            h = fh.projection(x)
            t = torch.tensor(target, dtype=torch.float32, device=fh.device).unsqueeze(0)
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
        print(f"🧭 Consistency Epoch {ep}: avg_loss={avg:.4f} (samples_used={samples_used}, skipped_nonfinite={skipped_nonfinite}) first_loss={first_loss}")
        if samples_used == 0:
            print("⚠️  No finite samples used in consistency epoch — check PTX features and projection path.")
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
            log.append({'trainer':'consistency','epoch':ep,'avg_loss':float(avg),'ts':time.time()})
            out.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

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
