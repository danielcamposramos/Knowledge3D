from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_download_all_benchmarks_list() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/download_all_benchmarks.py", "--list"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "arc_agi_2" in proc.stdout
    assert "mmlu" in proc.stdout


def test_download_all_benchmarks_dry_run(tmp_path: Path) -> None:
    output_root = tmp_path / "global_benchmarks"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/download_all_benchmarks.py",
            "--root",
            str(output_root),
            "--benchmarks",
            "gpqa,mmlu",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    manifest = output_root / "download_manifest.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert "gpqa" in payload["benchmarks"]
    assert "mmlu" in payload["benchmarks"]


def test_run_all_global_benchmarks_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    storage_root = tmp_path / "storage"
    dataset_root = tmp_path / "datasets"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_all_global_benchmarks.py",
            "--output-dir",
            str(output_dir),
            "--storage-root",
            str(storage_root),
            "--dataset-root",
            str(dataset_root),
            "--max-arc-tasks",
            "2",
            "--max-math-problems",
            "2",
            "--max-lhe-questions",
            "2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    summary = output_dir / "global_benchmark_summary.json"
    assert summary.exists()
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert "integrated_results" in payload
    assert "global_inventory" in payload
    assert "arc_agi_2" in payload["integrated_results"]
