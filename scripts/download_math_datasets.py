#!/usr/bin/env python3
"""
Download math benchmark datasets (alternate sources without auth friction).

Usage:
    PYTHONPATH=. python scripts/download_math_datasets.py
"""

from __future__ import annotations

import subprocess
import tarfile
import json
from pathlib import Path


def download_math() -> bool:
    """Download MATH dataset from HuggingFace (alternative parquet source)."""
    data_dir = Path("/K3D/K3D_llama_cpp/datasets/math/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    parquet_url = (
        "https://huggingface.co/datasets/qwedsacf/competition_math/resolve/main/"
        "data/train-00000-of-00001-7320a6f3aba8ebd2.parquet"
    )
    parquet_file = data_dir / "train.parquet"
    jsonl_file = data_dir / "train.jsonl"

    if jsonl_file.exists():
        print(f"MATH dataset already exists at {jsonl_file}")
        return True

    print("Downloading MATH dataset...")
    result = subprocess.run(["wget", "-q", parquet_url, "-O", str(parquet_file)])
    if result.returncode != 0:
        print("Download failed")
        return False

    # Convert parquet to JSONL
    try:
        import pandas as pd
    except ImportError:
        print("Install pandas to convert parquet: pip install pandas pyarrow")
        return False

    df = pd.read_parquet(parquet_file)
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")

    print(f"Saved {len(df)} problems to {jsonl_file}")
    parquet_file.unlink(missing_ok=True)
    return True


def download_mmlu() -> bool:
    """Download MMLU dataset (math subset)."""
    base_dir = Path("/K3D/K3D_llama_cpp/datasets/MMLU")
    data_dir = base_dir / "data"
    target_tar = base_dir / "data.tar"

    if data_dir.exists():
        print(f"MMLU data already present at {data_dir}")
        return True

    url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
    base_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading MMLU dataset (full) ...")
    result = subprocess.run(["wget", "-q", url, "-O", str(target_tar)])
    if result.returncode != 0:
        print("MMLU download failed")
        return False

    print("Extracting MMLU data.tar ...")
    with tarfile.open(target_tar, "r") as tar:
        tar.extractall(base_dir)
    target_tar.unlink(missing_ok=True)
    print(f"MMLU extracted to {data_dir}")
    return True


def download_amc_aime() -> bool:
    """Download AMC/AIME datasets from public mirrors."""
    base_dir = Path("/K3D/K3D_llama_cpp/datasets/AMC-AIME/data")
    base_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        (
            "AIME 2024",
            "https://huggingface.co/datasets/Maxwell-Jia/AIME_2024/resolve/main/aime_2024.jsonl",
            base_dir / "aime_2024.jsonl",
        ),
        (
            "AI-MO train",
            "https://huggingface.co/datasets/AI-MO/aimo-validation-math-level-5/resolve/main/data/train-00000-of-00002-5c014dca34dda4c1.parquet",
            base_dir / "aimo_train.parquet",
        ),
        (
            "AI-MO test",
            "https://huggingface.co/datasets/AI-MO/aimo-validation-math-level-5/resolve/main/data/test-00000-of-00002-2fe084d3de78ad96.parquet",
            base_dir / "aimo_test.parquet",
        ),
    ]

    def download_file(url: str, dest: Path) -> bool:
        if dest.exists():
            print(f"Already present: {dest}")
            return True
        res = subprocess.run(["wget", "-q", url, "-O", str(dest)])
        if res.returncode != 0:
            print(f"Download failed: {url}")
            return False
        print(f"Downloaded {dest}")
        return True

    ok = True
    for name, url, dest in sources:
        if not download_file(url, dest):
            ok = False

    # Convert AI-MO parquet files to JSONL for loader compatibility
    try:
        import pandas as pd
    except ImportError:
        print("Install pandas to convert AI-MO parquet: pip install pandas pyarrow")
        return False

    for parquet_path, jsonl_name in [
        (base_dir / "aimo_train.parquet", base_dir / "aimo_train.jsonl"),
        (base_dir / "aimo_test.parquet", base_dir / "aimo_test.jsonl"),
    ]:
        if parquet_path.exists() and not jsonl_name.exists():
            df = pd.read_parquet(parquet_path)
            with open(jsonl_name, "w", encoding="utf-8") as f:
                for _, row in df.iterrows():
                    f.write(json.dumps(row.to_dict()) + "\n")
            print(f"Converted {parquet_path.name} -> {jsonl_name.name} ({len(df)} rows)")
            parquet_path.unlink(missing_ok=True)

    return ok


if __name__ == "__main__":
    ok = download_math()
    # Optional: MMLU math subset
    try:
        download_mmlu()
    except NameError:
        # Function defined below
        pass
    # Optional: AMC/AIME datasets
    try:
        download_amc_aime()
    except NameError:
        pass
