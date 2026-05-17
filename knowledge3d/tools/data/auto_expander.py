from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

THEMES = ["galaxy_geometry", "house_zone5", "house_zone7"]


def _run_python(code: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".")
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def auto_expand_and_build(limit: int = 200) -> None:
    print("🚀 AUTO-EXPAND + AUTO-BUILD + AUTO-RESUME INITIATED")

    # 1. Expand datasets
    print("📥 Expanding datasets...")
    _run_python(
        "from knowledge3d.tools.data.hf_theme_fetcher import fetch_zone7_audio, fetch_zone5_videos, fetch_galaxy_walkthroughs; "
        f"fetch_zone7_audio({limit}); fetch_zone5_videos({limit}); fetch_galaxy_walkthroughs({limit})"
    )

    # 2. Symlink raw → curated
    print("🔗 Symlinking HDD → SSD...")
    _run_python(
        f"from knowledge3d.tools.data.fetcher import symlink_all_themes; symlink_all_themes({limit})"
    )

    # 3. Build GLBs
    print("🌀 Building GLBs...")
    for theme in THEMES:
        _run_python(
            "from knowledge3d.tools.data.builder import build_theme_glbs; "
            f"build_theme_glbs('{theme}', max_files={limit})"
        )

    # 4. train_rlwhf_policy moved to Old_Attempts/training_pipelines/ (drift — LLM training paradigm).
    # PTX sleeptime Lane A (0x300) + Lane B replaces this path. Do not re-add.

    print("✅ AUTO-EXPAND + AUTO-BUILD + AUTO-RESUME COMPLETE")


def main() -> None:
    auto_expand_and_build()


if __name__ == "__main__":
    main()
