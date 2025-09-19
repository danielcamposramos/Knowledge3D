from __future__ import annotations

"""Build pronunciation audio stars from metadata manifests."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional

from knowledge3d.tools.lexicon.common import build_star

SUPPORTED_EXT = {".csv", ".tsv", ".jsonl", ".ndjson"}


def load_rows(metadata_path: Path) -> Iterator[Mapping[str, str]]:
    suffix = metadata_path.suffix.lower()
    if suffix not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported metadata extension: {metadata_path.suffix}")
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with metadata_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                yield {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
    else:
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    yield {k: str(v) for k, v in payload.items()}


def iter_stars(args: argparse.Namespace) -> Iterator[MutableMapping[str, object]]:
    count = 0
    for row in load_rows(args.metadata):
        rel_path = row.get(args.path_field)
        transcript = row.get(args.text_field)
        if not rel_path or not transcript:
            continue
        ipa = row.get(args.ipa_field) if args.ipa_field else None
        lemma = row.get(args.lemma_field) if args.lemma_field else None
        if not lemma:
            lemma = transcript
        lemma = lemma.strip()
        if not lemma:
            continue
        rel_path = rel_path.strip()
        audio_path = (args.audio_root / rel_path).resolve() if args.audio_root else Path(rel_path).resolve()
        speaker = row.get(args.speaker_field) if args.speaker_field else None
        duration = row.get(args.duration_field) if args.duration_field else None
        meta = {
            key: value
            for key, value in row.items()
            if key not in {args.path_field, args.text_field, args.ipa_field, args.lemma_field, args.speaker_field, args.duration_field}
        }
        extra: Dict[str, object] = {
            "lexicon_audio": {
                "language": args.language,
                "transcript": transcript,
                "ipa": ipa,
                "audio_path": str(audio_path),
                "speaker": speaker,
                "duration": duration,
                "metadata": meta,
                "source": {
                    "dataset": args.source,
                },
            }
        }
        embedding_parts = [args.language, lemma, transcript]
        star = build_star(
            language=args.language,
            source=args.source,
            lemma=lemma,
            pos=None,
            sense_ref=rel_path,
            definition=transcript,
            embedding_parts=embedding_parts,
            relations={},
            extra=extra,
            zone="Zone 7 (Mirror Room)",
            tags=["modality:audio"],
            modalities=["audio", "text"],
        )
        yield star
        count += 1
        if args.limit and count >= args.limit:
            return


def build(args: argparse.Namespace) -> None:
    from knowledge3d.tools.lexicon.common import write_jsonl

    records = iter_stars(args)
    write_jsonl(args.out, records)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pronunciation audio stars")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata manifest (csv/tsv/jsonl)")
    parser.add_argument("--audio-root", type=Path, default=Path("."), help="Base directory for audio files")
    parser.add_argument("--language", type=str, required=True, help="ISO language code")
    parser.add_argument("--source", type=str, required=True, help="Dataset source label")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--path-field", type=str, default="path", help="Field containing audio relative path")
    parser.add_argument("--text-field", type=str, default="sentence", help="Field containing transcript")
    parser.add_argument("--ipa-field", type=str, default=None, help="Optional field containing IPA string")
    parser.add_argument("--lemma-field", type=str, default=None, help="Optional field containing lemma token")
    parser.add_argument("--speaker-field", type=str, default="client_id", help="Optional field for speaker id")
    parser.add_argument("--duration-field", type=str, default="duration", help="Optional field for duration value")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for generated stars")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:  # pragma: no cover
    args = parse_args(argv)
    build(args)


if __name__ == "__main__":  # pragma: no cover
    main()
