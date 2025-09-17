from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np  # type: ignore
from datasets import Audio, Dataset, DatasetDict, Video, load_dataset  # type: ignore

HDD_BASE = Path("/home/daniel/K3D_llama_cpp/datasets")

# Helper types
ExampleProcessor = Callable[[Dataset, Path, int], int]

YOUTUBE_ZONE5_URLS: List[str] = [
    "https://www.youtube.com/watch?v=IaSGqQa5O-M",  # Essence of linear algebra (spatial reasoning)
    "https://www.youtube.com/watch?v=Jkz1vQZ61Eo",  # Fractal tree growth
    "https://www.youtube.com/watch?v=I0-fpNedz-A",  # Library/knowledge visuals
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_yt_dlp() -> None:
    try:
        import yt_dlp  # type: ignore  # noqa: F401
        print("✅ yt-dlp already installed.")
    except ImportError:
        print("📦 Installing yt-dlp...")
        subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"], check=True)
        print("✅ yt-dlp installed successfully.")


def _write_audio_file(audio: Dict[str, any], dest_base: Path) -> Optional[Path]:
    """Persist an audio dict (from HF datasets) to disk."""
    import soundfile as sf  # type: ignore

    if audio is None:
        return None

    array = audio.get("array")
    if array is not None:
        sampling_rate = int(audio.get("sampling_rate", 16000))
        dest = dest_base.with_suffix(".wav")
        sf.write(dest, array, sampling_rate)
        return dest

    path_str = audio.get("path")
    if path_str:
        src = Path(path_str)
        if src.exists():
            dest = dest_base.with_suffix(src.suffix or ".wav")
            shutil.copy2(src, dest)
            return dest

    bytes_data = audio.get("bytes")
    if bytes_data:
        dest = dest_base.with_suffix(".wav")
        dest.write_bytes(bytes_data)
        return dest

    return None


def _write_video_file(video: Dict[str, any], dest_base: Path, default_suffix: str = ".mp4") -> Optional[Path]:
    if video is None:
        return None
    path_str = video.get("path")
    if path_str:
        src = Path(path_str)
        if src.exists():
            dest = dest_base.with_suffix(src.suffix or default_suffix)
            shutil.copy2(src, dest)
            return dest

    bytes_data = video.get("bytes")
    if bytes_data:
        dest = dest_base.with_suffix(default_suffix)
        dest.write_bytes(bytes_data)
        return dest

    return None


def _save_image_dataset(
    ds: Dataset,
    out_dir: Path,
    image_column: str = "image",
    text_column: str | None = None,
    limit: int | None = None,
) -> int:
    from PIL import Image  # type: ignore

    _ensure_dir(out_dir)
    saved = 0
    for row in ds:
        img_value = row.get(image_column)
        if img_value is None:
            continue
        images = img_value
        if not isinstance(images, list):
            images = [images]
        for idx, img in enumerate(images):
            if hasattr(img, "convert"):
                pil = img.convert("RGB")
            else:
                pil = Image.fromarray(np.array(img))
            file_name = f"sample_{saved:05d}_{idx:02d}.png"
            pil.save(out_dir / file_name)
        if text_column and row.get(text_column):
            (out_dir / f"sample_{saved:05d}.txt").write_text(str(row[text_column]), encoding="utf-8")
        saved += 1
        if limit and saved >= limit:
            break
    return saved


def _save_audio_dataset(
    ds: Dataset,
    out_dir: Path,
    audio_column: str = "audio",
    text_column: str | None = None,
    limit: int | None = None,
) -> int:
    _ensure_dir(out_dir)
    saved = 0
    for row in ds:
        audio = row.get(audio_column)
        if audio is None:
            continue
        dest = _write_audio_file(audio, out_dir / f"sample_{saved:05d}")
        if dest is None:
            continue
        if text_column and row.get(text_column):
            (out_dir / f"sample_{saved:05d}.txt").write_text(str(row[text_column]), encoding="utf-8")
        saved += 1
        if limit and saved >= limit:
            break
    return saved


def _save_video_dataset(
    ds: Dataset,
    out_dir: Path,
    video_column: str = "video",
    text_column: str | None = None,
    limit: int | None = None,
    default_suffix: str = ".mp4",
) -> int:
    _ensure_dir(out_dir)
    saved = 0
    for row in ds:
        video = row.get(video_column)
        if video is None:
            continue
        dest = _write_video_file(video, out_dir / f"sample_{saved:05d}", default_suffix=default_suffix)
        if dest is None:
            continue
        if text_column and row.get(text_column):
            (out_dir / f"sample_{saved:05d}.txt").write_text(str(row[text_column]), encoding="utf-8")
        saved += 1
        if limit and saved >= limit:
            break
    return saved


def _filter_by_keywords(ds: Dataset, column: str, keywords: Iterable[str]) -> Dataset:
    keywords = [k.lower() for k in keywords]

    def _predicate(example: Dict[str, str]) -> bool:
        text = str(example.get(column, "")).lower()
        return any(keyword in text for keyword in keywords)

    indices = [i for i, row in enumerate(ds) if _predicate(row)]
    if not indices:
        return ds.select([])
    return ds.select(indices)


def fetch_galaxy_geometry_images(limit: int) -> int:
    raw_dir = HDD_BASE / "galaxy_geometry" / "image" / "geometry_diagrams"
    slice_str = f"train[:{max(limit * 2, 100)}]"
    ds = load_dataset("hiyouga/geometry3k", split=slice_str)
    subset = ds.shuffle(seed=42).select(range(min(limit, len(ds))))
    return _save_image_dataset(subset, raw_dir, image_column="images", text_column="problem", limit=limit)


def fetch_galaxy_geometry_audio(limit: int) -> int:
    raw_dir = HDD_BASE / "galaxy_geometry" / "audio" / "phi_recursion"
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    filtered = _filter_by_keywords(ds, "text", ["golden ratio", "phi", "geometry", "vector", "angle", "proof"])
    if len(filtered) == 0:
        filtered = ds.shuffle(seed=123).select(range(min(limit, len(ds))))
    else:
        filtered = filtered.select(range(min(limit, len(filtered))))
    return _save_audio_dataset(filtered, raw_dir, audio_column="audio", text_column="text", limit=limit)


def fetch_zone5_images(limit: int) -> int:
    raw_dir = HDD_BASE / "house_zone5" / "image" / "garden_photos"
    slice_str = f"train[:{max(limit * 2, 200)}]"
    ds = load_dataset("huggan/flowers-102-categories", split=slice_str)
    subset = ds.shuffle(seed=55).select(range(min(limit, len(ds))))
    return _save_image_dataset(subset, raw_dir, image_column="image", text_column="label", limit=limit)


def fetch_zone5_audio(limit: int) -> int:
    raw_dir = HDD_BASE / "house_zone5" / "audio" / "garden_ambience"
    slice_str = f"test[:{max(limit * 5, 200)}]"
    ds = load_dataset("DynamicSuperb/EnvironmentalSoundClassification_ESC50-NaturalSoundscapesAndWaterSounds", split=slice_str)
    ds = ds.cast_column("audio", Audio(decode=False))
    filtered = _filter_by_keywords(ds, "category", ["birds", "insects", "water", "wind", "forest"])
    if len(filtered) == 0:
        filtered = ds.shuffle(seed=77).select(range(min(limit, len(ds))))
    else:
        filtered = filtered.select(range(min(limit, len(filtered))))
    return _save_audio_dataset(filtered, raw_dir, audio_column="audio", text_column="category", limit=limit)


def fetch_zone5_videos(limit: int) -> int:
    raw_dir = HDD_BASE / "house_zone5" / "video" / "growth_sequences"
    _ensure_dir(raw_dir)
    total = 0

    # Hugging Face animation set (small sample for growth visuals)
    try:
        video_slice = f"train[:{max(limit, 20)}]"
        video_ds = load_dataset("svjack/Petio_Animation_Ball_Videos", split=video_slice)
        video_ds = video_ds.cast_column("video", Video(decode=False))
        total += _save_video_dataset(video_ds, raw_dir, video_column="video", limit=limit)
    except Exception as exc:  # pragma: no cover
        print(f"⚠️  HF video fetch failed: {exc}")

    # Optional: YouTube samples if yt_dlp is available
    remaining = max(limit - total, 0)
    if remaining > 0:
        try:
            ensure_yt_dlp()
            import yt_dlp  # type: ignore

            ydl_opts = {
                "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "outtmpl": str(raw_dir / "%(title)s.%(ext)s"),
                "quiet": True,
                "noplaylist": True,
            }
            urls = YOUTUBE_ZONE5_URLS[:remaining]
            if urls:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    for url in urls:
                        try:
                            ydl.download([url])
                            total += 1
                        except Exception as exc:
                            print(f"⚠️  yt-dlp failed for {url}: {exc}")
        except ImportError:
            print("⚠️  yt-dlp not installed. Skip YouTube downloads.")

    print(f"🎥 Zone 5 videos saved: {total} files → {raw_dir}")
    return total


def fetch_zone7_images(limit: int) -> int:
    raw_dir = HDD_BASE / "house_zone7" / "image" / "mirror_selfies"
    slice_str = f"train[:{max(limit * 2, 100)}]"
    ds = load_dataset("emilianJR/ftinder_selfies", split=slice_str)
    subset = ds.shuffle(seed=101).select(range(min(limit, len(ds))))
    return _save_image_dataset(subset, raw_dir, image_column="image", text_column="text", limit=limit)


def fetch_zone7_audio(limit: int) -> int:
    raw_dir = HDD_BASE / "house_zone7" / "audio" / "whispered_critiques"
    _ensure_dir(raw_dir)
    total = 0

    # ESC50 whispered / critique sounds
    try:
        esc_slice = f"test[:{max(limit * 10, 500)}]"
        esc = load_dataset(
            "DynamicSuperb/EnvironmentalSoundClassification_ESC50-HumanAndNonSpeechSounds",
            split=esc_slice,
        )
        esc = esc.cast_column("audio", Audio(decode=False))
        filtered = _filter_by_keywords(
            esc,
            "caption",
            ["whisper", "breath", "quiet", "critique", "soft"],
        )
        if len(filtered) == 0:
            filtered = esc.shuffle(seed=123).select(range(min(limit // 2, len(esc))))
        else:
            filtered = filtered.select(range(min(limit // 2, len(filtered))))
        total += _save_audio_dataset(
            filtered,
            raw_dir,
            audio_column="audio",
            text_column="caption",
            limit=limit // 2,
        )
    except Exception as exc:  # pragma: no cover
        print(f"⚠️  ESC50 fetch failed: {exc}")

    # LibriSpeech honesty phrases
    remaining = max(limit - total, 0)
    if remaining > 0:
        try:
            speech_slice = f"validation[:{max(remaining * 15, 600)}]"
            librispeech = load_dataset("distil-whisper/librispeech_long", split=speech_slice)
            librispeech = librispeech.cast_column("audio", Audio(decode=False))
            filtered = _filter_by_keywords(
                librispeech,
                "text",
                ["i think", "i believe", "i was wrong", "honest", "truth", "i feel"],
            )
            if len(filtered) == 0:
                filtered = librispeech.shuffle(seed=456).select(range(min(remaining, len(librispeech))))
            else:
                filtered = filtered.select(range(min(remaining, len(filtered))))
            total += _save_audio_dataset(
                filtered,
                raw_dir,
                audio_column="audio",
                text_column="text",
                limit=remaining,
            )
        except Exception as exc:  # pragma: no cover
            print(f"⚠️  LibriSpeech fetch failed: {exc}")

    print(f"🔊 Zone 7 audio saved: {total} files → {raw_dir}")
    return total


def fetch_galaxy_walkthroughs(limit: int) -> int:
    audio_dir = HDD_BASE / "galaxy_geometry" / "audio" / "spatial_walkthroughs"
    text_dir = HDD_BASE / "galaxy_geometry" / "text" / "spatial_walkthroughs"
    _ensure_dir(audio_dir)
    _ensure_dir(text_dir)
    total = 0

    # Common Voice snippets for narrated walkthroughs
    try:
        slice_str = f"train[:{max(limit * 20, 800)}]"
        cv = load_dataset("mozilla-foundation/common_voice_16_1", "en", split=slice_str)
        cv = cv.cast_column("audio", Audio(decode=False))
        filtered = _filter_by_keywords(cv, "sentence", ["step by step", "trace", "follow", "walkthrough", "geometry"])
        total += _save_audio_dataset(
            filtered.select(range(min(limit, len(filtered)))),
            audio_dir,
            audio_column="audio",
            text_column="sentence",
            limit=limit,
        )
    except Exception as exc:  # pragma: no cover
        print(f"⚠️  Common Voice fetch failed: {exc}")

    # Math abstracts as textual spatial COT prompts
    try:
        text_slice = f"train[:{max(limit * 10, 200)}]"
        abstracts = load_dataset("gfissore/arxiv-abstracts-2021", split=text_slice)
        saved = 0
        for row in abstracts:
            if saved >= limit:
                break
            abstract = str(row.get("abstract", ""))
            if any(k in abstract.lower() for k in ["spatial", "geometry", "walkthrough", "step"]):
                (text_dir / f"spatial_cot_{saved:05d}.txt").write_text(abstract, encoding="utf-8")
                saved += 1
        total += saved
    except Exception as exc:  # pragma: no cover
        print(f"⚠️  ArXiv abstracts fetch failed: {exc}")

    print(f"🚶 Galaxy walkthrough assets saved: {total} files → {audio_dir} / {text_dir}")
    return total


FETCHERS: Dict[str, List[Callable[[int], int]]] = {
    "galaxy_geometry": [
        fetch_galaxy_geometry_images,
        fetch_galaxy_geometry_audio,
        fetch_galaxy_walkthroughs,
    ],
    "house_zone5": [
        fetch_zone5_images,
        fetch_zone5_audio,
        fetch_zone5_videos,
    ],
    "house_zone7": [
        fetch_zone7_images,
        fetch_zone7_audio,
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Download meaning-tailored subsets from Hugging Face")
    ap.add_argument("theme", choices=FETCHERS.keys())
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    total = 0
    for fetcher in FETCHERS[args.theme]:
        try:
            saved = fetcher(args.limit)
            total += saved
        except Exception as exc:  # pragma: no cover
            print(f"⚠️  Fetcher {fetcher.__name__} failed: {exc}")
    print(f"✅ Downloaded ~{total} items for theme '{args.theme}'")


if __name__ == "__main__":
    main()
