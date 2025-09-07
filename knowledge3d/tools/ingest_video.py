"""
Ingest videos: sample frames, compute OpenCLIP embeddings, and emit
CSV + metadata JSON for K3D generation. Stores a thumbnail per video.

Usage
  python3 -m knowledge3d.tools.ingest_video \
    --videos /k3dlocal/datasets/videos/*.mp4 \
    --out-csv /k3dlocal/datasets/video.sample.clip.csv \
    --out-meta /k3dlocal/datasets/video.sample.meta.json \
    --thumbs-dir /k3dlocal/datasets/video/thumbs \
    --base-url http://127.0.0.1:8766/video/thumbs \
    --fps 0.5 --max 10000
"""
from __future__ import annotations

import argparse
import glob
import os
from hashlib import md5
from pathlib import Path
from typing import List, Optional, Tuple


def _load_openclip():
    import open_clip  # type: ignore
    import torch  # type: ignore
    return open_clip, torch


def _encode_frames(frames, preprocess, model, device) -> List[List[float]]:
    import torch  # type: ignore
    outs: List[List[float]] = []
    batch: List = []
    for frm in frames:
        batch.append(preprocess(frm).unsqueeze(0))
        if len(batch) >= 16:
            x = torch.cat(batch, dim=0).to(device)
            with torch.no_grad():
                f = model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
            for i in range(f.shape[0]):
                outs.append(f[i].detach().cpu().numpy().astype(float).tolist())
            batch = []
    if batch:
        x = __import__("torch").cat(batch, dim=0).to(device)  # type: ignore
        with __import__("torch").no_grad():  # type: ignore
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
        for i in range(f.shape[0]):
            outs.append(f[i].detach().cpu().numpy().astype(float).tolist())
    return outs


def _sample_frames(path: Path, fps: float, max_frames: int) -> Tuple[List, Optional[object]]:
    # Returns (PIL frames, first_frame)
    from PIL import Image  # type: ignore
    import av  # type: ignore
    frames = []
    first = None
    try:
        container = av.open(str(path))
        stream = container.streams.video[0]
        base = max(1, int(stream.average_rate) if stream.average_rate else 30)
        step = max(1, int(base / max(0.1, fps)))
        i = 0
        for frame in container.decode(stream):
            if i % step == 0:
                img = frame.to_image()
                if first is None:
                    first = img.copy()
                frames.append(img)
                if len(frames) >= max_frames:
                    break
            i += 1
        container.close()
    except Exception:
        return [], None
    return frames, first


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Ingest video files into CLIP embeddings + metadata")
    ap.add_argument("--videos", nargs="+", help="Glob(s) of video files or explicit paths")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-meta", required=True)
    ap.add_argument("--thumbs-dir", required=True)
    ap.add_argument("--base-url", help="Base URL for serving thumbnails")
    ap.add_argument("--fps", type=float, default=0.5, help="Sampling fps")
    ap.add_argument("--max", type=int, default=100000, help="Max videos to process")
    args = ap.parse_args()

    open_clip, torch = _load_openclip()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    model.eval()

    vids: List[str] = []
    for pat in args.videos:
        vids.extend(glob.glob(pat))
    vids = [v for v in vids if os.path.isfile(v)]
    vids = vids[: args.max]

    # Prepare writers
    import csv, json
    out_csv = Path(args.out_csv)
    out_meta = Path(args.out_meta)
    thumbs_dir = Path(args.thumbs_dir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    ids: List[str] = []
    vecs: List[List[float]] = []
    metas: List[dict] = []
    for p in vids:
        path = Path(p)
        vid_id = md5(path.as_posix().encode("utf-8")).hexdigest()[:16]
        frames, thumb = _sample_frames(path, fps=max(0.1, args.fps), max_frames=24)
        if not frames:
            continue
        emb = _encode_frames(frames, preprocess, model, device)
        # aggregate
        import numpy as np
        arr = np.asarray(emb, dtype=float)
        mean = arr.mean(axis=0)
        ids.append(vid_id)
        vecs.append(mean.astype(float).tolist())
        img_url = None
        if thumb is not None:
            ext = ".jpg"
            thumb_path = thumbs_dir / f"{vid_id}{ext}"
            thumb.save(str(thumb_path), format="JPEG", quality=85)
            if args.base_url:
                img_url = f"{args.base_url}/{thumb_path.name}"
        metas.append({
            "label": path.name,
            "text": f"video: {path.name}",
            "image": img_url,
            "video": path.as_posix(),
        })
        if len(ids) % 200 == 0:
            print(f"processed videos: {len(ids)}")

    if not ids:
        print("no videos processed")
        return
    # write CSV
    dims = len(vecs[0])
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id"] + [f"v{i}" for i in range(dims)])
        for vid, v in zip(ids, vecs):
            w.writerow([vid] + [f"{float(x):.7f}" for x in v])
    out_meta.write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(ids)} videos -> {out_csv}, {out_meta}")


if __name__ == "__main__":
    main()

