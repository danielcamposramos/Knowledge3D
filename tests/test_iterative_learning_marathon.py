from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_iterative_learning_marathon_single_iteration(tmp_path: Path) -> None:
    output_root = tmp_path / "iterative_learning"
    storage_root = tmp_path / "storage"
    dataset_root = tmp_path / "datasets"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/iterative_learning_marathon.py",
            "--iterations",
            "1",
            "--output-root",
            str(output_root),
            "--storage-root",
            str(storage_root),
            "--dataset-root",
            str(dataset_root),
            "--max-arc-tasks",
            "1",
            "--max-math-problems",
            "1",
            "--max-lhe-questions",
            "1",
            "--pause-seconds",
            "0",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0

    analysis_path = output_root / "marathon_analysis.json"
    assert analysis_path.exists()
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert payload["iterations"] == 1
    assert len(payload["iteration_results"]) == 1
    assert "progression_analysis" in payload
