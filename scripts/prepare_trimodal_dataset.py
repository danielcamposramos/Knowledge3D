#!/usr/bin/env python3
"""
Phase G.0 - Prepare Tri-Modal Dataset

Combines RLWHF (text), image caption (text+image), and audio caption datasets
into a single JSONL file that enumerates the available modalities per sample.

The resulting file is used by Phase G multi-modal training scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


# Default dataset roots (override via CLI arguments when needed)
DEFAULT_RLWHF_PATH = Path("/K3D/Knowledge3D.local/datasets/rlwhf/teacher_evaluations.jsonl")
DEFAULT_IMAGE_CAPTIONS = [
    Path("/K3D/Knowledge3D.local/datasets/image_captions_llama32vision.jsonl"),
    Path("/K3D/Knowledge3D.local/datasets/image_captions_qwen25vl.jsonl"),
]
DEFAULT_AUDIO_ROOT = Path("/K3D/K3D_llama_cpp/datasets/audio")
DEFAULT_AUDIOCAPS_ROOT = Path("/K3D/K3D_llama_cpp/datasets/audiocaps_raw")
DEFAULT_CLOTHO_ROOT = Path("/K3D/K3D_llama_cpp/datasets/clotho_raw")
DEFAULT_OUTPUT_PATH = Path("/K3D/Knowledge3D.local/datasets/trimodal_phase_g.jsonl")

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".flac", ".m4a", ".mp3", ".ogg", ".webm")


@dataclass
class TextPayload:
    content: str
    metadata: Optional[dict] = None


@dataclass
class ImagePayload:
    path: str
    caption: Optional[str] = None


@dataclass
class AudioPayload:
    path: str
    transcript: Optional[str] = None
    language: Optional[str] = None
    source: Optional[str] = None


@dataclass
class TriModalSample:
    id: str
    source: str
    modalities: List[str]
    text: Optional[TextPayload] = None
    image: Optional[ImagePayload] = None
    audio: Optional[AudioPayload] = None
    extra: Optional[dict] = None

    def to_json(self) -> str:
        payload = {
            "id": self.id,
            "source": self.source,
            "modalities": self.modalities,
            "text": asdict(self.text) if self.text else None,
            "image": asdict(self.image) if self.image else None,
            "audio": asdict(self.audio) if self.audio else None,
            "extra": self.extra or None,
        }
        return json.dumps(payload, ensure_ascii=False)


def load_rlwhf(
    dataset_path: Path,
    start_idx: int,
    end_idx: int,
) -> List[TriModalSample]:
    samples: List[TriModalSample] = []

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number < start_idx:
                continue
            if line_number > end_idx:
                break

            record = json.loads(line)
            teacher = record.get("teacher_evaluation", {})
            student = record.get("student_attempt", {})

            text_content = "\n\n".join(
                piece for piece in [
                    f"Question: {record.get('question', '').strip()}",
                    f"Context: {record.get('context', '').strip()}",
                    f"Answer: {record.get('answer', '').strip()}",
                ]
                if piece
            )

            metadata = {
                "pdf_name": record.get("pdf_name"),
                "page_num": record.get("page_num"),
                "source": record.get("source"),
                "difficulty": record.get("difficulty"),
                "teacher_rating": teacher.get("rating_score"),
                "teacher_honesty": teacher.get("honesty_score"),
                "student_confidence": student.get("confidence"),
                "question": record.get("question"),
                "context": record.get("context"),
                "answer": record.get("answer"),
            }

            sample = TriModalSample(
                id=f"rlwhf_{line_number:06d}",
                source="rlwhf",
                modalities=["text"],
                text=TextPayload(content=text_content, metadata=metadata),
                extra={
                    "question": record.get("question"),
                    "context": record.get("context"),
                    "answer": record.get("answer"),
                    "teacher_evaluation": teacher,
                    "student_attempt": {
                        key: student.get(key)
                        for key in ["converged", "confidence", "output_norm"]
                    },
                },
            )
            samples.append(sample)

    return samples


def load_image_caption_file(path: Path, source_name: str) -> Iterable[TriModalSample]:
    if not path.exists():
        print(f"[Image] WARN Skipping missing caption file: {path}", file=sys.stderr)
        return

    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            record = json.loads(line)
            image_path = record.get("image")
            caption = record.get("caption", "").strip()

            modalities = ["image"]
            if caption:
                modalities.append("text")

            yield TriModalSample(
                id=f"{source_name}_{index:06d}",
                source=source_name,
                modalities=modalities,
                text=TextPayload(content=caption) if caption else None,
                image=ImagePayload(path=image_path or "", caption=caption or None),
            )


def load_image_captions(paths: Sequence[Path]) -> List[TriModalSample]:
    samples: List[TriModalSample] = []
    for path in paths:
        source_name = path.stem
        samples.extend(load_image_caption_file(path, source_name))
    return samples


def _resolve_audio_path(relative_path: str, manifest_dir: Path, root: Path) -> Optional[Path]:
    """Resolve audio file paths referenced inside a manifest."""
    candidates = [
        root / relative_path,
        manifest_dir / relative_path,
        manifest_dir.parent / relative_path,
    ]

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return None


def load_audio_manifests(audio_root: Path) -> List[TriModalSample]:
    samples: List[TriModalSample] = []

    if not audio_root.exists():
        print(f"[Audio] WARN Audio root missing: {audio_root}", file=sys.stderr)
        return samples

    manifest_paths = sorted(audio_root.rglob("manifest.csv"))
    for manifest_path in manifest_paths:
        language = manifest_path.parent.name

        with manifest_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader, start=1):
                relative_audio = row.get("path", "").strip()
                transcript = row.get("sentence", "").strip()

                resolved = _resolve_audio_path(relative_audio, manifest_path.parent, audio_root)
                if resolved is None:
                    print(f"[Audio] WARN Missing audio file: {relative_audio}", file=sys.stderr)
                    continue

                modalities = ["audio"]
                if transcript:
                    modalities.append("text")

                samples.append(
                    TriModalSample(
                        id=f"audio_manifest_{language}_{index:06d}",
                        source=f"audio_manifest/{language}",
                        modalities=modalities,
                        text=TextPayload(content=transcript) if transcript else None,
                        audio=AudioPayload(
                            path=str(resolved),
                            transcript=transcript or None,
                            language=row.get("language"),
                            source=language,
                        ),
                    )
                )

    return samples


def load_audiocaps_dataset(root: Path) -> List[TriModalSample]:
    samples: List[TriModalSample] = []
    manifest = root / "manifest.jsonl"
    media_root = root / "media"

    if not manifest.exists():
        print(f"[AudioCaps] WARN Missing manifest: {manifest}", file=sys.stderr)
        return samples

    with manifest.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            record = json.loads(line)
            youtube_id = record.get("youtube_id")
            caption = record.get("caption", "").strip()

            audio_path = None
            if youtube_id:
                for ext in SUPPORTED_AUDIO_EXTENSIONS:
                    candidate = media_root / f"{youtube_id}{ext}"
                    if candidate.exists():
                        audio_path = candidate
                        break
                if audio_path is None:
                    matches = list(media_root.glob(f"{youtube_id}.*"))
                    if matches:
                        audio_path = matches[0]

            if audio_path is None:
                print(f"[AudioCaps] WARN Audio missing for {youtube_id}", file=sys.stderr)
                continue

            modalities = ["audio"]
            if caption:
                modalities.append("text")

            samples.append(
                TriModalSample(
                    id=f"audiocaps_{index:06d}",
                    source="audiocaps",
                    modalities=modalities,
                    text=TextPayload(content=caption) if caption else None,
                    audio=AudioPayload(
                        path=str(audio_path),
                        transcript=caption or None,
                        source="audiocaps",
                    ),
                )
            )

    return samples


def load_clotho_dataset(root: Path) -> List[TriModalSample]:
    samples: List[TriModalSample] = []

    caption_files = sorted(root.glob("clotho_captions_*.csv"))
    if not caption_files:
        print(f"[Clotho] WARN No caption CSV files found in {root}", file=sys.stderr)
        return samples

    audio_dirs = [
        path for path in root.glob("clotho_audio_*/*")
        if path.is_dir()
    ]

    audio_map = {}
    for audio_dir in audio_dirs:
        for audio_file in audio_dir.glob("*"):
            if audio_file.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                audio_map[audio_file.name] = audio_file

    for csv_path in caption_files:
        split_name = csv_path.stem.replace("clotho_captions_", "")
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader, start=1):
                file_name = row.get("file_name", "").strip()
                audio_path = audio_map.get(file_name)
                if audio_path is None:
                    print(f"[Clotho] WARN Missing audio file for {file_name}", file=sys.stderr)
                    continue

                captions = [
                    row.get(f"caption_{i}", "").strip()
                    for i in range(1, 6)
                    if row.get(f"caption_{i}")
                ]
                transcript = captions[0] if captions else ""

                modalities = ["audio"]
                if transcript:
                    modalities.append("text")

                samples.append(
                    TriModalSample(
                        id=f"clotho_{split_name}_{index:06d}",
                        source=f"clotho/{split_name}",
                        modalities=modalities,
                        text=TextPayload(content=transcript) if transcript else None,
                        audio=AudioPayload(
                            path=str(audio_path),
                            transcript=transcript or None,
                            source=f"clotho/{split_name}",
                        ),
                        extra={"all_captions": captions} if len(captions) > 1 else None,
                    )
                )

    return samples


def build_dataset(args: argparse.Namespace) -> List[TriModalSample]:
    samples: List[TriModalSample] = []

    rlwhf_samples = load_rlwhf(args.rlwhf_path, args.rlwhf_start, args.rlwhf_end)
    print(f"[RLWHF] Collected {len(rlwhf_samples)} samples")
    samples.extend(rlwhf_samples)

    image_samples = load_image_captions(args.image_captions)
    print(f"[Images] Collected {len(image_samples)} caption samples")
    samples.extend(image_samples)

    audio_manifest_samples = load_audio_manifests(args.audio_root)
    print(f"[Audio Manifest] Collected {len(audio_manifest_samples)} samples")
    samples.extend(audio_manifest_samples)

    audiocaps_samples = load_audiocaps_dataset(args.audiocaps_root)
    print(f"[AudioCaps] Collected {len(audiocaps_samples)} samples")
    samples.extend(audiocaps_samples)

    clotho_samples = load_clotho_dataset(args.clotho_root)
    print(f"[Clotho] Collected {len(clotho_samples)} samples")
    samples.extend(clotho_samples)

    print(f"[Total] Combined samples: {len(samples)}")
    return samples


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare tri-modal dataset for Phase G multi-modal training."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--rlwhf-path",
        type=Path,
        default=DEFAULT_RLWHF_PATH,
        help="Path to RLWHF teacher evaluation JSONL file",
    )
    parser.add_argument(
        "--rlwhf-start",
        type=int,
        default=8042,
        help="First RLWHF sample index to include (1-based)",
    )
    parser.add_argument(
        "--rlwhf-end",
        type=int,
        default=10000,
        help="Last RLWHF sample index to include (inclusive, 1-based)",
    )
    parser.add_argument(
        "--image-captions",
        type=Path,
        nargs="+",
        default=DEFAULT_IMAGE_CAPTIONS,
        help="JSONL caption files (text + image)",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=DEFAULT_AUDIO_ROOT,
        help="Root directory containing LibriSpeech-style manifests",
    )
    parser.add_argument(
        "--audiocaps-root",
        type=Path,
        default=DEFAULT_AUDIOCAPS_ROOT,
        help="Root directory of AudioCaps dataset",
    )
    parser.add_argument(
        "--clotho-root",
        type=Path,
        default=DEFAULT_CLOTHO_ROOT,
        help="Root directory of Clotho dataset",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    samples = build_dataset(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.to_json())
            handle.write("\n")

    print(f"[Write] Dataset saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
