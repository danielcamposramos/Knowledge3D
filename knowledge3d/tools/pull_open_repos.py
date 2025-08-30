"""
Clone a curated set of open-source AI repos locally for corpus extraction.

Repos are cloned into ../Knowledge3D.local/repos (sibling to the repo root).

Usage
  python3 -m knowledge3d.tools.pull_open_repos
  python3 -m knowledge3d.tools.pull_open_repos --add https://github.com/facebookresearch/segment-anything
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_REPOS = [
    "https://github.com/facebookresearch/segment-anything",
    "https://github.com/facebookresearch/dinov2",
    "https://github.com/facebookresearch/mae",
    "https://github.com/IDEA-Research/GroundingDINO",
    "https://github.com/openai/CLIP",
    "https://github.com/huggingface/transformers",
    "https://github.com/facebookresearch/faiss",
    "https://github.com/nmslib/hnswlib",
    "https://github.com/spotify/annoy",
    "https://github.com/milvus-io/milvus",
    "https://github.com/qdrant/qdrant",
    "https://github.com/danijar/dreamer",
    "https://github.com/werner-duvaud/muzero-general",
    "https://github.com/google-research/vision_transformer",
    "https://github.com/isl-org/MiDaS",
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=False, cwd=str(cwd) if cwd else None)


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Clone curated AI repos into Knowledge3D.local/repos")
    p.add_argument("--add", action="append", help="Additional repo URL to clone")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    local_root = repo_root.parent / f"{repo_root.name}.local"
    target = local_root / "repos"
    target.mkdir(parents=True, exist_ok=True)

    urls = list(DEFAULT_REPOS)
    if args.add:
        urls.extend(args.add)

    for url in urls:
        name = url.rstrip("/").split("/")[-1]
        dest = target / name
        if dest.exists():
            # try pull
            run(["git", "-C", str(dest), "pull", "--ff-only"])
        else:
            run(["git", "clone", "--depth", "1", url, str(dest)])
        print("synced:", url, "->", dest)


if __name__ == "__main__":  # pragma: no cover
    main()

