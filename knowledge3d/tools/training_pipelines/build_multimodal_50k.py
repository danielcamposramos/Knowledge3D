from __future__ import annotations

"""
Build small (≤50k) multimodal K3D GLBs per modality using existing ingesters.

This orchestrates the following optional steps:
- Text: use a .txt file with one line per entry → GLB
- Image+Text (WIT): TSV(.gz) → text + CLIP embeddings CSV + metadata → GLB
- Audio: glob files → CLAP embeddings CSV + metadata → GLB
- Video: glob files → CLIP embeddings CSV + metadata → GLB

Examples
  python -m knowledge3d.tools.build_multimodal_50k \
    --text data/ai_books_basic.txt --text-out viewer/public/text_50k.glb \
    --wit-tsv /k3dlocal/wit/wit_v1.train.sample.tsv.gz --wit-out viewer/public/wit_50k.glb \
    --audio "/k3dlocal/audio/*.wav" --audio-out viewer/public/audio_50k.glb \
    --video "/k3dlocal/video/*.mp4" --video-out viewer/public/video_50k.glb

Notes
- Each modality is optional; the tool builds whichever outputs are requested.
- For WIT/image and video, OpenCLIP is required; for audio, LAION-CLAP is required.
- Outputs are small GLBs suitable for local testing and viewer demos.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _build_text(text_path: Path, out_glb: Path, max_rows: int = 50000) -> None:
    from k3dgen.__main__ import create_gltf_file  # type: ignore
    # If the text file is longer than max_rows, write a temp trimmed copy
    src = text_path
    lines = src.read_text(encoding="utf-8").splitlines()
    if len(lines) > max_rows:
        tmp = out_glb.with_suffix(".tmp.txt")
        tmp.write_text("\n".join(lines[:max_rows]), encoding="utf-8")
        src = tmp
    _ensure_parent(out_glb)
    create_gltf_file(
        gltf_path=str(out_glb),
        vectors_csv=None,
        text_path=str(src),
        k=8,
        reducer="umap",
        fmt="glb",
    )


def _build_wit(tsv_path: Path, out_glb: Path, work_dir: Path, max_rows: int = 50000) -> None:
    from .ingest_wit import ingest_text, ingest_images_and_embeddings  # type: ignore
    from k3dgen.__main__ import create_gltf_file  # type: ignore
    text_out = work_dir / "wit.text.txt"
    csv_out = work_dir / "wit.clip.csv"
    meta_out = work_dir / "wit.meta.json"
    imgs_dir = work_dir / "images"
    base_url = None
    n = ingest_text([tsv_path], text_out, limit=max_rows)
    print(f"[wit] text={n}")
    processed, downloaded = ingest_images_and_embeddings(
        [tsv_path], csv_out, meta_out, images_dir=imgs_dir, base_url=base_url, max_rows=max_rows
    )
    print(f"[wit] images processed={processed} downloaded={downloaded}")
    _ensure_parent(out_glb)
    create_gltf_file(
        gltf_path=str(out_glb),
        vectors_csv=str(csv_out),
        text_path=None,
        metadata_path=str(meta_out),
        k=10,
        reducer="umap",
        fmt="glb",
        emb_precision="f16",
    )


def _build_audio(glob_pat: str, out_glb: Path) -> None:
    from .ingest_audio import main as audio_ingest_main  # type: ignore
    from k3dgen.__main__ import create_gltf_file  # type: ignore
    work_dir = out_glb.parent / "_audio_tmp"
    csv_out = work_dir / "audio.clap.csv"
    meta_out = work_dir / "audio.meta.json"
    _ensure_parent(csv_out)
    import sys
    # Call the ingest module as a function via argv shim
    argv = [
        "--audio",
        glob_pat,
        "--out-csv",
        str(csv_out),
        "--out-meta",
        str(meta_out),
    ]
    sys_argv = list(os.sys.argv)
    try:
        os.sys.argv = ["ingest_audio.py"] + argv
        audio_ingest_main()
    finally:
        os.sys.argv = sys_argv
    _ensure_parent(out_glb)
    create_gltf_file(
        gltf_path=str(out_glb),
        vectors_csv=str(csv_out),
        text_path=None,
        metadata_path=str(meta_out),
        k=8,
        reducer="umap",
        fmt="glb",
        emb_precision="f16",
    )


def _build_video(glob_pat: str, out_glb: Path) -> None:
    from .ingest_video import main as video_ingest_main  # type: ignore
    from k3dgen.__main__ import create_gltf_file  # type: ignore
    work_dir = out_glb.parent / "_video_tmp"
    thumbs_dir = work_dir / "thumbs"
    csv_out = work_dir / "video.clip.csv"
    meta_out = work_dir / "video.meta.json"
    _ensure_parent(csv_out)
    import sys
    argv = [
        "--videos",
        glob_pat,
        "--out-csv",
        str(csv_out),
        "--out-meta",
        str(meta_out),
        "--thumbs-dir",
        str(thumbs_dir),
        "--fps",
        "0.5",
        "--max",
        "100000",
    ]
    sys_argv = list(os.sys.argv)
    try:
        os.sys.argv = ["ingest_video.py"] + argv
        video_ingest_main()
    finally:
        os.sys.argv = sys_argv
    _ensure_parent(out_glb)
    create_gltf_file(
        gltf_path=str(out_glb),
        vectors_csv=str(csv_out),
        text_path=None,
        metadata_path=str(meta_out),
        k=10,
        reducer="umap",
        fmt="glb",
        emb_precision="f16",
    )


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Build ≤50k multimodal K3D GLBs per modality")
    # Text
    ap.add_argument("--text", help="Path to text lines (one per row)")
    ap.add_argument("--text-out", help="Output GLB for text")
    # WIT images
    ap.add_argument("--wit-tsv", help="WIT TSV(.gz) file")
    ap.add_argument("--wit-out", help="Output GLB for WIT images")
    # Audio
    ap.add_argument("--audio", help="Glob for audio files")
    ap.add_argument("--audio-out", help="Output GLB for audio")
    # Video
    ap.add_argument("--video", help="Glob for video files")
    ap.add_argument("--video-out", help="Output GLB for video")
    # Work dir
    ap.add_argument("--work-dir", default="../Knowledge3D.local/datasets/_build50k", help="Working directory")
    ap.add_argument("--max", type=int, default=50000, help="Max rows per modality where applicable")
    args = ap.parse_args()

    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    if args.text and args.text_out:
        _build_text(Path(args.text), Path(args.text_out), max_rows=int(args.max))
    if args.wit_tsv and args.wit_out:
        _build_wit(Path(args.wit_tsv), Path(args.wit_out), work / "wit", max_rows=int(args.max))
    if args.audio and args.audio_out:
        _build_audio(args.audio, Path(args.audio_out))
    if args.video and args.video_out:
        _build_video(args.video, Path(args.video_out))


if __name__ == "__main__":
    main()

