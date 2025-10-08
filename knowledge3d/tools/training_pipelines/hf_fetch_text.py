from __future__ import annotations

"""
Fetch text from a HuggingFace dataset and write a simple textlines file.

Examples
  # Wikipedia (en) small sample
  python -m knowledge3d.tools.hf_fetch_text \
    --dataset wikipedia --config 20220301.en --split train \
    --text-field text --limit 50000 \
    --out ../Knowledge3D.local/datasets/wikipedia.en.sample.txt

  # StackExchange (StackOverflow) sample
  python -m knowledge3d.tools.hf_fetch_text \
    --dataset stackexchange --config stackoverflow \
    --split train --text-field text --limit 50000 \
    --out ../Knowledge3D.local/datasets/stackoverflow.sample.txt
"""

import argparse
from pathlib import Path
from typing import Iterable


def iter_text(ds, field: str) -> Iterable[str]:
    for ex in ds:
        txt = ex.get(field)
        if not txt:
            continue
        s = str(txt).strip()
        if s:
            yield s.replace("\n", " ")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Fetch text from HF dataset and write lines")
    ap.add_argument("--dataset", required=True, help="HF dataset name (e.g., wikipedia)")
    ap.add_argument("--config", help="Dataset config (e.g., 20220301.en)")
    ap.add_argument("--split", default="train", help="Split (train/validation/test)")
    ap.add_argument("--text-field", default="text", help="Field with text content")
    ap.add_argument("--limit", type=int, help="Max number of lines")
    ap.add_argument("--out", required=True, help="Output textlines path (.txt)")
    ap.add_argument("--streaming", type=int, default=0, help="Use HF streaming mode (1=yes) for very large datasets")
    ap.add_argument("--append", type=int, default=0, help="Append to existing file (1) and skip existing lines")
    args = ap.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise SystemExit("pip install datasets first (in your env)") from e

    # HF datasets >=3 may require trust_remote_code for some community datasets.
    # If streaming + hub path given (org/name), fall back to parquet loader from hub files.
    load_kwargs = {"trust_remote_code": True}
    if int(args.streaming or 0):
        if "/" in (args.dataset or "") and args.config:
            # List parquet files from the Hub and stream them
            try:
                from huggingface_hub import list_repo_files  # type: ignore
                prefix = f"{args.config}/data/"
                files = [p for p in list_repo_files(repo_id=args.dataset, repo_type="dataset") if p.startswith(prefix) and p.endswith(".parquet")]
                if not files:
                    # fallback to any parquet under config root
                    files = [p for p in list_repo_files(repo_id=args.dataset, repo_type="dataset") if p.startswith(f"{args.config}/") and p.endswith(".parquet")]
                hf_paths = [f"hf://datasets/{args.dataset}/{p}" for p in files]
                if not hf_paths:
                    raise RuntimeError("No parquet files found on Hub for the specified dataset/config")
                ds = load_dataset("parquet", data_files={"train": hf_paths}, split="train", streaming=True)  # type: ignore
            except Exception as e:
                raise SystemExit(f"Failed to list/load parquet files from Hub: {e}")
        else:
            ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True, **load_kwargs)  # type: ignore
    else:
        ds = load_dataset(args.dataset, args.config, split=args.split, **load_kwargs)  # type: ignore
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Determine skip count when appending
    skip = 0
    mode = "w"
    if int(args.append or 0) and out.exists():
        try:
            import subprocess, shlex
            wc_out = subprocess.check_output(shlex.split(f"wc -l {out}"), text=True).strip()
            skip = int(wc_out.split()[0]) if wc_out else 0
        except Exception:
            skip = 0
        mode = "a"
    n = 0
    seen = 0
    with out.open(mode, encoding="utf-8") as f:
        for line in iter_text(ds, args.text_field):
            if skip and seen < skip:
                seen += 1
                continue
            f.write(line + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"Wrote {n} lines to {out} (skipped {seen})")


if __name__ == "__main__":
    main()
