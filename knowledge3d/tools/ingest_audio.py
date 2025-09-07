"""
Ingest audio files and compute LAION-CLAP embeddings.

Usage
  python3 -m knowledge3d.tools.ingest_audio \
    --audio /k3dlocal/datasets/audio/*.wav \
    --out-csv /k3dlocal/datasets/audio.sample.clap.csv \
    --out-meta /k3dlocal/datasets/audio.sample.meta.json
"""
from __future__ import annotations

import argparse
import glob
import os
from hashlib import md5
from pathlib import Path
from typing import List


def _load_clap():
    import laion_clap  # type: ignore
    import torch  # type: ignore
    return laion_clap, torch


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Ingest audio using LAION-CLAP embeddings")
    ap.add_argument("--audio", nargs="+", help="Glob(s) of audio files (wav/mp3/flac)")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-meta", required=True)
    args = ap.parse_args()

    laion_clap, torch = _load_clap()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.eval()
    model.to(device)
    # gather files
    files: List[str] = []
    for pat in args.audio:
        files.extend(glob.glob(pat))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print("no audio files")
        return
    # encode
    ids: List[str] = []
    vecs: List[List[float]] = []
    metas: List[dict] = []
    for p in files:
        try:
            emb = model.get_audio_embedding_from_filelist(x=[p], use_tensor=True, device=device)
            v = emb[0].detach().cpu().numpy().astype(float).tolist()
        except Exception:
            continue
        aid = md5(p.encode("utf-8")).hexdigest()[:16]
        ids.append(aid)
        vecs.append(v)
        metas.append({"label": Path(p).name, "text": f"audio: {Path(p).name}"})
        if len(ids) % 200 == 0:
            print(f"processed audio: {len(ids)}")
    # write outputs
    import csv, json
    dims = len(vecs[0])
    out_csv = Path(args.out_csv)
    out_meta = Path(args.out_meta)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id"] + [f"v{i}" for i in range(dims)])
        for i, v in zip(ids, vecs):
            w.writerow([i] + [f"{float(x):.7f}" for x in v])
    out_meta.write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(ids)} audio -> {out_csv}, {out_meta}")


if __name__ == "__main__":
    main()

