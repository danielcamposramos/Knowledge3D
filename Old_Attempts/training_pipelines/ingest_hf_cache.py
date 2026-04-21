"""Harvest Hugging Face cached datasets into Knowledge3D corpora."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore

from datasets import Audio, Dataset, DatasetDict, load_dataset  # type: ignore

DEFAULT_CACHE_DIR = Path("~/.cache/huggingface/datasets").expanduser()
DEFAULT_OUTPUT = Path("viewer/public/galaxy/working/hf_cache_corpus.jsonl")
DEFAULT_MANIFEST = Path("viewer/public/galaxy/working/hf_cache_manifest.json")
DEFAULT_MAX_PER_SPLIT = 200
MAX_TEXT_LENGTH = 600
MAX_LIST_ITEMS = 5
MAX_LOGGED_ERRORS = 5


@dataclass
class DatasetEntry:
    dataset_id: str  # cached directory name namespace___name
    hf_id: str       # huggingface identifier namespace/name
    config: Optional[str]
    info_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Hugging Face cached datasets into Knowledge3D corpora")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-per-split", type=int, default=DEFAULT_MAX_PER_SPLIT)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional list of dataset roots (e.g. namespace___name) to include. If omitted, process all.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output instead of overwriting.",
    )
    return parser.parse_args()


def discover_entries(cache_dir: Path, whitelist: Optional[Iterable[str]] = None) -> List[DatasetEntry]:
    selected: Optional[set[str]] = set(whitelist) if whitelist else None
    entries: Dict[Tuple[str, Optional[str]], DatasetEntry] = {}
    for info_path in cache_dir.glob("*/*/*/*/dataset_info.json"):
        hashed_dir = info_path.parent
        version_dir = hashed_dir.parent
        config_dir = version_dir.parent
        dataset_root = config_dir.parent
        if not dataset_root.is_dir():
            continue
        dataset_id = dataset_root.name
        if selected and dataset_id not in selected:
            continue
        hf_id = dataset_id.replace("___", "/")
        with info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        config_name = info.get("config_name") or (config_dir.name if config_dir.name != "default" else None)
        key = (hf_id, config_name)
        # Keep the most recent info file (based on mtime)
        existing = entries.get(key)
        if existing is None or info_path.stat().st_mtime > existing.info_path.stat().st_mtime:
            entries[key] = DatasetEntry(dataset_id=dataset_id, hf_id=hf_id, config=config_name, info_path=info_path)
    return sorted(entries.values(), key=lambda e: (e.hf_id, e.config or ""))


def dataset_to_iterable(ds_obj) -> Iterator[Tuple[str, Dataset]]:
    if isinstance(ds_obj, DatasetDict):
        for split_name, split_ds in ds_obj.items():
            yield split_name, split_ds
    elif isinstance(ds_obj, Dataset):
        yield "default", ds_obj
    else:  # pragma: no cover - defensive
        raise TypeError(f"Unexpected dataset type: {type(ds_obj)!r}")


def summarize_value(value, key: str) -> Tuple[str, List[str]]:
    tags: List[str] = []

    if value is None:
        return "null", tags

    if isinstance(value, str):
        text = value.strip()
        tags.extend(extract_keywords(text))
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH] + "…"
        return text, tags

    if isinstance(value, (int, float, bool)):
        return str(value), tags

    if np is not None and isinstance(value, np.ndarray):  # type: ignore[arg-type]
        return f"array(shape={value.shape}, dtype={value.dtype})", tags

    if isinstance(value, list):
        rendered_items = []
        for item in value[:MAX_LIST_ITEMS]:
            rendered, item_tags = summarize_value(item, key)
            rendered_items.append(rendered)
            tags.extend(item_tags)
        if len(value) > MAX_LIST_ITEMS:
            rendered_items.append("…")
        return "[" + ", ".join(rendered_items) + "]", tags

    if isinstance(value, dict):
        # Heuristics for media-like fields
        if "sampling_rate" in value and "array" in value:
            sr = value.get("sampling_rate") or 0
            duration = ""
            arr = value.get("array")
            arr_len = len(arr) if hasattr(arr, "__len__") else None
            if arr_len and sr:
                seconds = arr_len / float(sr)
                duration = f", duration≈{seconds:.2f}s"
            path = value.get("path")
            desc = f"Audio sample{duration}"
            if path:
                desc += f" (path={path})"
            tags.append("audio")
            return desc, tags
        if "path" in value and ("width" in value or "height" in value):
            width = value.get("width")
            height = value.get("height")
            desc = f"Image ({width}x{height})"
            path = value.get("path")
            if path:
                desc += f" at {path}"
            tags.append("image")
            return desc, tags
        if "vertices" in value and "faces" in value:
            desc = "Mesh(vertices={len}, faces={faces})".format(
                len=len(value.get("vertices", [])),
                faces=len(value.get("faces", [])),
            )
            tags.append("mesh")
            return desc, tags

        entries = []
        for sub_key, sub_value in list(value.items())[:MAX_LIST_ITEMS]:
            rendered, sub_tags = summarize_value(sub_value, sub_key)
            entries.append(f"{sub_key}={rendered}")
            tags.extend(sub_tags)
        if len(value) > MAX_LIST_ITEMS:
            entries.append("…")
        return "{" + ", ".join(entries) + "}", tags

    return repr(value), tags


def summarize_row(row: Dict[str, object]) -> Tuple[str, List[str]]:
    parts: List[str] = []
    tags: List[str] = []
    for key in sorted(row.keys()):
        value = row[key]
        rendered, key_tags = summarize_value(value, key)
        if rendered:
            parts.append(f"{key}: {rendered}")
        tags.extend(key_tags)
    summary = "; ".join(parts)
    return summary, list(dict.fromkeys(tags))[:8]


def extract_keywords(text: str) -> List[str]:
    if not text:
        return []
    cleaned = text.lower()
    tokens = [token for token in re_tokenize(cleaned) if len(token) >= 4]
    seen: List[str] = []
    for tok in tokens:
        if tok not in seen:
            seen.append(tok)
        if len(seen) >= 5:
            break
    return seen


def re_tokenize(text: str) -> List[str]:
    import re

    return re.findall(r"[a-z0-9]+", text)


def process_dataset(
    entry: DatasetEntry,
    cache_dir: Path,
    max_per_split: int,
    output_handle,
    stats: Dict[str, Dict[str, int]],
) -> None:
    load_kwargs = {
        "cache_dir": str(cache_dir),
        "download_mode": "reuse_cache_if_exists",
    }
    if entry.config:
        ds_obj = load_dataset(entry.hf_id, entry.config, **load_kwargs)
    else:
        ds_obj = load_dataset(entry.hf_id, **load_kwargs)

    for split, split_dataset in dataset_to_iterable(ds_obj):
        # Avoid decoding heavy media columns; keep file metadata only.
        if hasattr(split_dataset, "features"):
            for column, feature in split_dataset.features.items():
                if isinstance(feature, Audio):
                    split_dataset = split_dataset.cast_column(column, Audio(decode=False))
        try:
            length = len(split_dataset)
        except TypeError:
            length = None
        limit = min(max_per_split, length) if (length is not None and max_per_split > 0) else max_per_split
        if limit == 0:
            continue
        taken = 0
        for index in range(limit):
            if length is not None and index >= length:
                break
            row = split_dataset[index]
            summary, tags = summarize_row(row)
            if not summary.strip():
                continue
            entry_json = {
                "question": (
                    f"Dataset sample from {entry.hf_id}"
                    f" (config={entry.config or 'default'}, split={split}, index={index})."
                ),
                "answer": summary,
                "source": {
                    "dataset": entry.hf_id,
                    "config": entry.config or "default",
                    "split": split,
                    "index": index,
                    "tags": tags,
                },
                "prompt": (
                    f"Dataset sample from {entry.hf_id}"
                    f" (config={entry.config or 'default'}, split={split}, index={index})."
                ),
                "true_answer": summary,
                "predicted": summary,
                "score": 1.0,
                "quick_feedback": {
                    "score": 1.0,
                    "explanation": f"Ingested from {entry.hf_id} ({split}).",
                },
                "deep_feedback": {
                    "score": 1.0,
                    "explanation": f"Auto-ingested Hugging Face cache sample ({entry.hf_id}).",
                },
            }
            output_handle.write(json.dumps(entry_json, ensure_ascii=False) + "\n")
            taken += 1
        stats[entry.hf_id][split] += taken


def write_manifest(manifest_path: Path, stats: Dict[str, Dict[str, int]], cache_dir: Path) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir),
        "datasets": [],
    }
    for dataset_id, split_counts in sorted(stats.items()):
        manifest["datasets"].append({
            "dataset": dataset_id,
            "splits": split_counts,
        })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    entries = discover_entries(args.cache_dir, args.datasets)
    if not entries:
        print(f"⚠️  No datasets found under {args.cache_dir} matching selection.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    with args.output.open(mode, encoding="utf-8") as output_handle:
        for entry in entries:
            print(
                f"📚 Processing dataset {entry.hf_id} (config={entry.config or 'default'})"
                f" using cache {entry.info_path.parent}"
            )
            try:
                process_dataset(entry, args.cache_dir, args.max_per_split, output_handle, stats)
            except Exception as exc:  # pragma: no cover - robust logging
                print(f"⚠️  Failed to process {entry.hf_id}/{entry.config or 'default'}: {exc}")

    write_manifest(args.manifest, stats, args.cache_dir)
    total_samples = sum(sum(split.values()) for split in stats.values())
    print(f"✅ Harvest complete — {total_samples} samples from {len(stats)} datasets → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
