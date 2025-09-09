from __future__ import annotations

"""
GPU-only sharded text embedding to CSV (+optional metadata).

Reads a UTF-8 textlines file and writes an embeddings CSV with header:
  id,v0,v1,...,v{d-1}
and optionally a JSON metadata array aligned to ids with {label,text}.

Why
- Avoids loading entire corpora in memory. Streams lines in shards and
  encodes with Sentence-Transformers on CUDA.

Usage
  # 768-d default model (MiniLM) with 8k batch, 1M lines cap per run
  python -m knowledge3d.tools.embed_text_sharded \
    --in ../Knowledge3D.local/datasets/wikipedia.en.txt \
    --out-csv ../Knowledge3D.local/datasets/wikipedia.en.embed.csv \
    --out-meta ../Knowledge3D.local/datasets/wikipedia.en.embed.meta.json \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --batch 8192 --start 0 --limit 1000000

Then use in Galaxy build as a vector modality:
  kind:text ../.../wikipedia.en.embed.csv:../.../wikipedia.en.embed.meta.json
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np


def iter_lines(path: Path, start: int = 0, limit: int | None = None) -> Iterable[tuple[int, str]]:
    n = 0
    i = 0
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                i += 1
                continue
            if i < start:
                i += 1
                continue
            yield (i, s)
            i += 1
            n += 1
            if limit and n >= limit:
                break


def write_header_once(csv_path: Path, dims: int) -> None:
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("id," + ",".join([f"v{j}" for j in range(dims)]) + "\n")


def append_rows(csv_path: Path, ids: List[str], emb: np.ndarray) -> None:
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        for rid, row in zip(ids, emb):
            vals = ",".join(f"{float(x):.7f}" for x in row.tolist())
            f.write(f"{rid},{vals}\n")


def append_meta(meta_path: Path | None, labels: List[str], texts: List[str]) -> None:
    if not meta_path:
        return
    arr = [{"label": labels[i], "text": texts[i]} for i in range(len(labels))]
    if meta_path.exists() and meta_path.stat().st_size > 0:
        # extend existing JSON array
        cur = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(cur, list):
            cur = []
        cur.extend(arr)
        meta_path.write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")
    else:
        meta_path.write_text(json.dumps(arr, ensure_ascii=False), encoding="utf-8")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Embed textlines to CSV (GPU sharded)")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-meta")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    inp = Path(args.inp)
    out_csv = Path(args.out_csv)
    out_meta = Path(args.out_meta) if args.out_meta else None
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_meta:
        out_meta.parent.mkdir(parents=True, exist_ok=True)

    # GPU-only Sentence-Transformers
    from knowledge3d.accel import st_device_kwargs  # type: ignore
    dev = st_device_kwargs()
    if dev.get("device") != "cuda":
        raise SystemExit("CUDA is required for embedding (strict GPU policy)")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise SystemExit("sentence-transformers not installed in env") from e
    try:
        model = SentenceTransformer(args.model, **dev)
    except TypeError:
        model = SentenceTransformer(args.model)

    # iterate shards
    buf_ids: List[str] = []
    buf_txt: List[str] = []
    total = 0
    dims_written = False
    for idx, s in iter_lines(inp, start=int(args.start), limit=args.limit):
        rid = f"w:{idx}"
        buf_ids.append(rid)
        buf_txt.append(s)
        if len(buf_txt) >= int(args.batch):
            emb = np.asarray(model.encode(buf_txt, convert_to_numpy=True, **dev), dtype=float)
            if not dims_written:
                write_header_once(out_csv, emb.shape[1])
                dims_written = True
            append_rows(out_csv, buf_ids, emb)
            append_meta(out_meta, [t[:48] for t in buf_txt], buf_txt)
            total += len(buf_txt)
            print(f"[embed] wrote {total} rows @dim={emb.shape[1]}")
            buf_ids.clear(); buf_txt.clear()
    if buf_txt:
        emb = np.asarray(model.encode(buf_txt, convert_to_numpy=True, **dev), dtype=float)
        if not dims_written:
            write_header_once(out_csv, emb.shape[1])
            dims_written = True
        append_rows(out_csv, buf_ids, emb)
        append_meta(out_meta, [t[:48] for t in buf_txt], buf_txt)
        total += len(buf_txt)
        print(f"[embed] wrote {total} rows @dim={emb.shape[1]}")

    print(f"Done. Total rows: {total}. CSV: {out_csv}")


if __name__ == "__main__":
    main()

