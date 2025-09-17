from __future__ import annotations

"""Meaning-tailored dataset fetch helpers.

The goal is to assemble small, curated slices of existing open data that
match the Galaxy/House themes without downloading every possible sample.

Raw assets live on the HDD (`/home/daniel/K3D_llama_cpp/datasets`)
and curated subsets (symlinked) live on the SSD
(`/K3D/Knowledge3D.local/datasets`).
"""

import shutil
from pathlib import Path
import glob
from typing import Dict, Iterable, List

from knowledge3d.tools.data.utils import ensure_symlink, hash_copy

HDD_BASE = Path("/home/daniel/K3D_llama_cpp/datasets")
SSD_BASE = Path("/K3D/Knowledge3D.local/datasets")

THEME_MAP: Dict[str, Dict[str, List[Dict[str, Iterable[str]]]]] = {
    "galaxy_geometry": {
        "text": [
            {
                "name": "ptx_specs",
                "sources": [
                    "docs/CRANIUM_CORE.md",
                    "docs/CRANIUM.md",
                    "docs/RPN_RUNTIME.md",
                    "docs/HOUSE_MEMORY.md",
                    "docs/CRANIUM_SKILLS.md",
                ],
            }
        ],
        "image": [
            {
                "name": "ray_diagrams",
                "sources": [
                    "docs/images/*.png",
                    "viewer/public/house/library/*.png",
                    "/home/daniel/K3D_llama_cpp/datasets/galaxy_geometry/image/geometry_diagrams/*.png",
                ],
            }
        ],
        "audio": [
            {
                "name": "phi_recursion",
                "sources": [
                    "docs/spatial_web_k3d_discussion.mpga",
                    "/home/daniel/K3D_llama_cpp/datasets/galaxy_geometry/audio/phi_recursion/*",
                ],
            }
        ],
        "video": [
            {
                "name": "spatial_walkthroughs",
                "sources": ["/home/daniel/K3D_llama_cpp/datasets/galaxy_geometry/video/*.mp4"],
            }
        ],
        "3d": [
            {
                "name": "core_solids",
                "sources": [
                    "viewer/public/house/workshop/*.glb",
                    "viewer/public/house/garden/*.glb",
                ],
            }
        ],
    },
    "house_zone5": {
        "text": [
            {
                "name": "garden_logs",
                "sources": [
                    "docs/KNOWLEDGE_GARDENS.md",
                    "viewer/public/house/materialized_objects/diary*garden*.json",
                ],
            }
        ],
        "image": [
            {
                "name": "garden_photos",
                "sources": [
                    "/home/daniel/K3D_llama_cpp/datasets/garden_images/*",
                    "viewer/public/house/garden/*.png",
                    "/home/daniel/K3D_llama_cpp/datasets/house_zone5/image/garden_photos/*.png",
                ],
            }
        ],
        "audio": [
            {
                "name": "garden_audio",
                "sources": [
                    "/home/daniel/K3D_llama_cpp/datasets/garden_audio/*",
                    "/home/daniel/K3D_llama_cpp/datasets/house_zone5/audio/garden_ambience/*",
                ],
            }
        ],
        "video": [
            {
                "name": "garden_timelapse",
                "sources": ["/home/daniel/K3D_llama_cpp/datasets/garden_video/*"],
            }
        ],
        "3d": [
            {
                "name": "garden_glbs",
                "sources": ["viewer/public/house/garden/*.glb"],
            }
        ],
    },
    "house_zone7": {
        "text": [
            {
                "name": "mirror_diaries",
                "sources": [
                    "docs/DIARY.md",
                    "viewer/public/house/materialized_objects/diary_*.json",
                ],
            }
        ],
        "image": [
            {
                "name": "mirror_images",
                "sources": [
                    "/home/daniel/K3D_llama_cpp/datasets/mirror_images/*",
                    "/home/daniel/K3D_llama_cpp/datasets/house_zone7/image/mirror_selfies/*.png",
                ],
            }
        ],
        "audio": [
            {
                "name": "mirror_audio",
                "sources": [
                    "/home/daniel/K3D_llama_cpp/datasets/mirror_audio/*",
                    "/home/daniel/K3D_llama_cpp/datasets/house_zone7/audio/whispered_critiques/*",
                ],
            }
        ],
        "video": [
            {
                "name": "mirror_video",
                "sources": ["/home/daniel/K3D_llama_cpp/datasets/mirror_video/*"],
            }
        ],
        "3d": [
            {
                "name": "mirror_glbs",
                "sources": [
                    "viewer/public/house/library/*.glb",
                    "viewer/public/house/workshop/mirror_*.glb",
                ],
            }
        ],
    },
}


def fetch_theme_data(theme: str, max_samples: int = 100) -> None:
    """Curate data for a meaning theme.

    Existing local files are symlinked into HDD/SSD theme folders.
    If a source glob does not match anything, we simply log it so the
    operator can populate the dataset later.
    """

    if theme not in THEME_MAP:
        raise ValueError(f"Unknown theme: {theme}")

    theme_cfg = THEME_MAP[theme]
    for modality, datasets in theme_cfg.items():
        for dataset in datasets:
            name = dataset["name"]
            target_raw = HDD_BASE / theme / modality / name
            target_raw.mkdir(parents=True, exist_ok=True)
            collected = 0
            for pattern in dataset.get("sources", []):
                matches = _expand(pattern)
                for src in matches:
                    if collected >= max_samples:
                        break
                    try:
                        dest = target_raw / src.name
                        if dest.exists():
                            collected += 1
                            continue
                        if src.is_file():
                            if src.suffix.lower() in {".md", ".txt", ".json"}:
                                hash_copy(src, dest)
                            else:
                                if src.is_symlink():
                                    dest.symlink_to(src.resolve())
                                else:
                                    dest.symlink_to(src)
                        else:
                            ensure_symlink(src, dest)
                        collected += 1
                    except Exception as exc:
                        print(f"⚠️  Failed to collect {src}: {exc}")
                if collected >= max_samples:
                    break
            curated_path = SSD_BASE / theme / modality / name
            curated_path.parent.mkdir(parents=True, exist_ok=True)
            ensure_symlink(target_raw, curated_path)
            print(f"✅ Curated {collected} samples for {theme}/{modality}/{name} → {curated_path}")


def _expand(pattern: str) -> List[Path]:
    path = Path(pattern)
    if path.exists():
        return [path]
    # Glob relative to repo root
    matches = list(Path('.').glob(pattern)) if not pattern.startswith('/') else []
    if matches:
        return matches
    # Absolute glob via glob module
    if pattern.startswith('/'):
        matches = [Path(p) for p in glob.glob(pattern)]
        if matches:
            return matches
    # Fallback: glob relative to repo using glob module (handles "docs/*.md")
    matches = [Path(p) for p in glob.glob(pattern)]
    if matches:
        return matches
    print(f"⚠️  No matches for pattern: {pattern}")
    return []


def symlink_all_themes(max_samples: int = 100) -> None:
    for theme in THEME_MAP.keys():
        fetch_theme_data(theme, max_samples=max_samples)


__all__ = ["fetch_theme_data", "symlink_all_themes"]
