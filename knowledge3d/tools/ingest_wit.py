"""
Ingest WIT (Wikipedia-based Image-Text) TSV files and produce:
1) A text corpus (title + description lines)
2) Optional image embeddings via OpenCLIP (GPU preferred), with metadata JSON

Inputs
- One or more TSV/TSV.GZ files from https://github.com/google-research-datasets/wit
  Typical columns include:
    page_title, image_url, image_alt_text_description,
    caption_reference_description, context_page_description, language, ...

Outputs
- Text: UTF-8 lines combining title + description (for quick text-only GLB)
- CSV: id + CLIP embedding floats (for image GLB)
- JSON: per-node metadata aligned to ids, including image URL/path and text

Quick usage (text-only)
  python3 -m knowledge3d.tools.ingest_wit \
    --tsv /k3dlocal/datasets/wit/wit_v1.train.sample.tsv.gz \
    --out /k3dlocal/datasets/wit.sample.txt --max 200000

Quick usage (images + embeddings -> GLB)
  python3 -m knowledge3d.tools.ingest_wit \
    --tsv /k3dlocal/datasets/wit/wit_v1.train.sample.tsv.gz \
    --out /k3dlocal/datasets/wit.sample.txt \
    --out-csv /k3dlocal/datasets/wit.sample.clip.csv \
    --out-meta /k3dlocal/datasets/wit.sample.meta.json \
    --images-dir /k3dlocal/datasets/wit/images \
    --base-url http://127.0.0.1:8766/wit/images \
    --max 100000

  K3D_STRICT_GPU=1 K3D_ACCEL=gpu K3D_FAISS_DEVICE=gpu \
  python -m k3dgen /k3dlocal/datasets/wit.sample.clip.csv \
    --gltf /k3dlocal/datasets/wit.sample.glb --k 10 --reducer umap \
    --metadata /k3dlocal/datasets/wit.sample.meta.json --emb-precision f16

Notes
- Viewer can display image thumbnails if metadata.image is a reachable URL.
- For very large downloads, consider running multiple shards.
"""
from __future__ import annotations

import argparse
import csv
import gzip
from hashlib import md5
from pathlib import Path
from typing import Iterable, List, TextIO, Optional, Tuple

def _try_import_openclip():
    try:
        import open_clip  # type: ignore
        import torch  # type: ignore
        return open_clip, torch
    except Exception as e:
        raise RuntimeError("open_clip_torch and torch are required for --out-csv") from e


def _open_tsv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


PREFERRED_FIELDS = [
    "image_alt_text_description",
    "caption_reference_description",
    "context_page_description",
    "caption_attribution_description",
    "caption_title_and_reference_description",
]


def compose_line(row: dict) -> str | None:
    title = (row.get("page_title") or "").strip()
    text = ""
    for f in PREFERRED_FIELDS:
        s = (row.get(f) or "").strip()
        if s:
            text = s
            break
    if not text:
        return None
    lang = (row.get("language") or row.get("language_code") or "").strip()
    # Build a concise line; prefix with language when available
    if lang:
        head = f"[{lang}] {title}" if title else f"[{lang}]"
    else:
        head = title or "WIT"
    # Limit overly long lines
    body = text.replace("\t", " ").replace("\n", " ")
    if len(body) > 512:
        body = body[:509] + "..."
    return f"{head} — {body}" if head else body


def ingest_text(paths: List[Path], out_path: Path, limit: int | None = None) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for p in paths:
            with _open_tsv(p) as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    line = compose_line(row)
                    if not line:
                        continue
                    out.write(line + "\n")
                    n += 1
                    if limit and n >= limit:
                        return n
    return n


def _download_image(url: str, dest: Path) -> bool:
    try:
        import urllib.request as ur
        dest.parent.mkdir(parents=True, exist_ok=True)
        with ur.urlopen(url, timeout=15) as r, open(dest, "wb") as f:
            f.write(r.read())
        return True
    except Exception:
        return False


def _row_text(row: dict) -> str:
    for f in PREFERRED_FIELDS:
        s = (row.get(f) or "").strip()
        if s:
            return s
    return ""


def ingest_images_and_embeddings(
    paths: List[Path],
    out_csv: Path,
    out_meta: Path,
    images_dir: Optional[Path],
    base_url: Optional[str],
    max_rows: Optional[int],
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "laion2b_s34b_b79k",
) -> Tuple[int, int]:
    open_clip, torch = _try_import_openclip()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=clip_pretrained, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(clip_model)
    # Prepare writers
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    ids: List[str] = []
    vectors: List[List[float]] = []
    meta: List[dict] = []
    emit = 0
    hits = 0
    for p in paths:
        with _open_tsv(p) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if max_rows and emit >= max_rows:
                    break
                img_url = (row.get("image_url") or "").strip()
                if not img_url:
                    continue
                title = (row.get("page_title") or "").strip()
                text = _row_text(row)
                # Stable id from url
                rid = md5(img_url.encode("utf-8")).hexdigest()[:16]
                rel_path = None
                disk_path = None
                if images_dir is not None:
                    # derive extension
                    ext = ".jpg"
                    low = img_url.lower()
                    for e in (".jpg", ".jpeg", ".png", ".webp"):
                        if low.endswith(e):
                            ext = e; break
                    rel_path = f"wit/images/{rid}{ext}"
                    disk_path = images_dir / f"{rid}{ext}"
                    ok = _download_image(img_url, disk_path)
                    if not ok:
                        continue
                    hits += 1
                # Compute image embedding
                try:
                    from PIL import Image  # type: ignore
                    img = Image.open(disk_path if disk_path else ur.urlopen(img_url)).convert("RGB")  # type: ignore
                except Exception:
                    continue
                image = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_feat = model.encode_image(image)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                vec = img_feat.squeeze(0).detach().cpu().numpy().astype(float).tolist()
                ids.append(rid)
                vectors.append(vec)
                meta.append({
                    "label": title or rid,
                    "text": text,
                    "image": (f"{base_url}/{rel_path.split('/',1)[1]}" if base_url and rel_path else (rel_path or img_url)),
                })
                emit += 1
                if emit % 1000 == 0:
                    print(f"processed={emit} downloaded={hits}")
        if max_rows and emit >= max_rows:
            break
    # Write CSV
    if not vectors:
        return (0, 0)
    dims = len(vectors[0])
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["id"] + [f"v{i}" for i in range(dims)])
        for rid, vec in zip(ids, vectors):
            w.writerow([rid] + [f"{float(x):.7f}" for x in vec])
    # Write metadata
    import json as _json
    out_meta.write_text(_json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return (emit, hits)


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Ingest WIT TSV(.gz) into text and optional image embeddings")
    ap.add_argument("--tsv", nargs="+", required=True, help="Paths to WIT TSV or TSV.GZ files")
    ap.add_argument("--out", required=True, help="Output text file (title + description)")
    ap.add_argument("--max", type=int, help="Maximum number of rows to process")
    # Images + embeddings
    ap.add_argument("--out-csv", help="Output CSV (id + CLIP embedding). Triggers image pipeline when set")
    ap.add_argument("--out-meta", help="Output JSON metadata aligned to ids (label,text,image)")
    ap.add_argument("--images-dir", help="Directory to store downloaded images (e.g., /k3dlocal/datasets/wit/images)")
    ap.add_argument("--base-url", help="Base URL for images as served by datasets server (e.g., http://127.0.0.1:8766/wit/images)")
    ap.add_argument("--clip-model", default="ViT-B-32", help="OpenCLIP model (default ViT-B-32)")
    ap.add_argument("--clip-pretrained", default="laion2b_s34b_b79k", help="OpenCLIP pretrained tag")
    args = ap.parse_args()
    paths = [Path(s) for s in args.tsv]
    out = Path(args.out)
    n = ingest_text(paths, out, args.max)
    print(f"[text] Wrote {n} lines -> {out}")
    if args.out_csv and args.out_meta:
        images_dir = Path(args.images_dir) if args.images_dir else None
        out_csv = Path(args.out_csv)
        out_meta = Path(args.out_meta)
        processed, downloaded = ingest_images_and_embeddings(
            paths,
            out_csv,
            out_meta,
            images_dir,
            args.base_url,
            args.max,
            args.clip_model,
            args.clip_pretrained,
        )
        print(f"[images] processed={processed} downloaded={downloaded} csv={out_csv} meta={out_meta}")


if __name__ == "__main__":
    main()
