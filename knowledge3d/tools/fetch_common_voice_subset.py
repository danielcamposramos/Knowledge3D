from __future__ import annotations

"""Download a small Common Voice subset and emit a manifest for audio builders."""

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional

from datasets import load_dataset  # type: ignore
import soundfile as sf  # type: ignore
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_lang(language: str) -> str:
    return language.replace("-", "_").lower()


def build_manifest_row(
    *,
    rel_path: str,
    sentence: str,
    client_id: Optional[str],
    duration: float,
    language: str,
) -> Dict[str, str]:
    return {
        "path": rel_path,
        "sentence": sentence,
        "client_id": client_id or "",
        "duration": f"{duration:.4f}",
        "language": language,
    }


def write_manifest(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    fieldnames = ["path", "sentence", "client_id", "duration", "language"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fetch_subset(args: argparse.Namespace) -> None:
    clean_lang = sanitize_lang(args.language)
    split = args.split
    if args.count:
        split = f"{split}[:{args.count}]"

    if args.dataset == "common_voice":
        dataset = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            args.language,
            split=split,
            use_auth_token=None,
        )
        transcript_field = "sentence"
        speaker_field = "client_id"
    else:
        dataset = load_dataset(
            "PolyAI/minds14",
            name=args.language,
            split=split,
        )
        transcript_field = "transcription"
        speaker_field = "intent_class"

    out_dir = args.out_dir
    ensure_dir(out_dir)
    rows = []
    max_items = len(dataset) if args.count is None else min(len(dataset), args.count)

    audio_root = out_dir
    for idx, record in enumerate(dataset):
        if args.count is not None and idx >= args.count:
            break
        audio = record.get("audio")
        if not audio:
            continue
        if args.dataset == "common_voice":
            array = audio.get("array") if isinstance(audio, dict) else None
            sampling_rate = (audio.get("sampling_rate") if isinstance(audio, dict) else None) or 48_000
            if array is None:
                continue
            duration = len(array) / float(sampling_rate)
        else:
            try:
                samples = audio.get_all_samples()
            except AttributeError:
                continue
            tensor = samples.data
            if hasattr(tensor, "detach"):
                tensor = tensor.detach()
            array = np.asarray(tensor.squeeze().cpu().numpy(), dtype=np.float32)
            sampling_rate = int(getattr(samples, "sample_rate", 16_000))
            duration = float(getattr(samples, "duration_seconds", len(array) / sampling_rate))
        file_name = f"{args.dataset}_{idx:05d}.wav"
        rel_dir = Path(clean_lang) / args.dataset
        file_rel_path = rel_dir / file_name
        file_path = audio_root / file_rel_path
        ensure_dir(file_path.parent)
        sf.write(file_path, array, sampling_rate)
        sentence = (record.get(transcript_field) or "").strip()
        rows.append(
            build_manifest_row(
                rel_path=str(file_rel_path),
                sentence=sentence,
                client_id=(record.get(speaker_field) or ""),
                duration=duration,
                language=args.language,
            )
        )
        if idx + 1 >= max_items:
            break

    manifest_path = args.manifest
    ensure_dir(manifest_path.parent)
    write_manifest(manifest_path, rows)
    print(f"🗂️  Wrote {len(rows)} rows → {manifest_path}")

    if args.mirror_root:
        mirror_base = args.mirror_root / clean_lang
        ensure_dir(mirror_base)
        source_base = audio_root / clean_lang
        for child in source_base.glob("**/*"):
            dest = mirror_base / child.relative_to(source_base)
            if child.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, dest)
        print(f"🔁 Mirrored assets to {mirror_base}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch speech datasets for lexicon audio")
    parser.add_argument("--dataset", choices=["common_voice", "minds14"], default="common_voice",
                        help="Dataset to download (default: common_voice)")
    parser.add_argument("--language", required=True,
                        help="Language identifier (Common Voice code or PolyAI/minds14 config, e.g., en, pt, es, zh-CN, en-US)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Base directory for primary dataset storage")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest CSV to write")
    parser.add_argument("--mirror-root", type=Path, help="Optional mirror root for Knowledge3D.local copy")
    parser.add_argument("--split", default="train", help="Dataset split name (default: train)")
    parser.add_argument("--count", type=int, default=200, help="Number of clips to download (default: 200)")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:  # pragma: no cover
    args = parse_args(argv)
    fetch_subset(args)


if __name__ == "__main__":  # pragma: no cover
    main()
