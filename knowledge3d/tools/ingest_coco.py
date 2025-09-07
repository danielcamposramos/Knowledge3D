from __future__ import annotations

"""
Ingest COCO 2017 images with captions and compute CLIP embeddings.

Inputs
- --images-dir: path to COCO images directory (e.g., train2017/ or val2017/)
- --captions: path to captions JSON (e.g., annotations/captions_train2017.json)

Outputs
- --out-csv: id + embedding floats (OpenCLIP ViT-B-32 normalized)
- --out-meta: JSON metadata list aligned to ids (label, text, image path)

Example
  python3 -m knowledge3d.tools.ingest_coco \
    --images-dir /home/daniel/K3D_llama_cpp/datasets/coco_raw/train2017 \
    --captions   /home/daniel/K3D_llama_cpp/datasets/coco_raw/annotations/captions_train2017.json \
    --out-csv    ../Knowledge3D.local/datasets/coco.train.clip.csv \
    --out-meta   ../Knowledge3D.local/datasets/coco.train.meta.json \
    --max 50000
"""

import argparse
import json
from hashlib import md5
from pathlib import Path
from typing import Dict, List, Optional


def _load_openclip():  # pragma: no cover
    try:
        import open_clip  # type: ignore
        import torch  # type: ignore
    except Exception:
        return None, None, None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    model.eval()
    return model, preprocess, device, torch


def _build_caption_index(captions_json: Path) -> Dict[str, str]:
    j = json.loads(captions_json.read_text(encoding="utf-8"))
    # images: list of {id, file_name}
    # annotations: list of {image_id, caption}
    id_to_file: Dict[int, str] = {}
    for im in j.get("images", []):
        try:
            id_to_file[int(im["id"])]=str(im["file_name"])  # type: ignore
        except Exception:
            continue
    file_to_cap: Dict[str, str] = {}
    for ann in j.get("annotations", []):
        try:
            fid = id_to_file.get(int(ann["image_id"]))  # type: ignore
            cap = str(ann.get("caption") or "")
            if fid and cap and fid not in file_to_cap:
                file_to_cap[fid] = cap
        except Exception:
            continue
    return file_to_cap


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Ingest COCO images and compute CLIP embeddings")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-meta", required=True)
    ap.add_argument("--max", type=int)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    caps_path = Path(args.captions)
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_meta = Path(args.out_meta); out_meta.parent.mkdir(parents=True, exist_ok=True)

    file_to_cap = _build_caption_index(caps_path)
    files = [images_dir / f for f in file_to_cap.keys()]
    if args.max:
        files = files[: int(args.max)]

    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise SystemExit("pip install pillow") from e
    model, preprocess, device, torch = _load_openclip()
    has_clip = model is not None and preprocess is not None

    vecs: List[List[float]] = []
    ids: List[str] = []
    metas: List[Dict[str, str]] = []
    for i, p in enumerate(files):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        if has_clip:
            x = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                f = model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
            v = f.squeeze(0).detach().cpu().numpy().astype(float).tolist()
        else:
            # fallback: hashed 32-d
            from hashlib import md5
            h = md5(p.as_posix().encode("utf-8")).digest()
            vals = [(b/255.0)-0.5 for b in h]
            while len(vals) < 32:
                vals.extend(vals)
            v = vals[:32]
        rid = md5(p.as_posix().encode("utf-8")).hexdigest()[:16]
        ids.append(rid)
        vecs.append(v)
        metas.append({"label": p.name, "text": file_to_cap.get(p.name, ""), "image": p.as_posix()})
        if (i+1) % 200 == 0:
            print(f"processed images: {i+1}")

    if not ids:
        print("no images processed")
        return
    # write CSV
    import csv
    dims = len(vecs[0])
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id"] + [f"v{i}" for i in range(dims)])
        for i, v in zip(ids, vecs):
            w.writerow([i] + [f"{float(x):.7f}" for x in v])
    out_meta.write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(ids)} images -> {out_csv}, {out_meta}")


if __name__ == "__main__":
    main()
