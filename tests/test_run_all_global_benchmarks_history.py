from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_once(output_dir: Path, storage_root: Path, dataset_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
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
            "1",
            "--max-math-problems",
            "1",
            "--max-lhe-questions",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_global_benchmark_historical_tracking_and_memory(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    storage_root = tmp_path / "storage"
    dataset_root = tmp_path / "datasets"

    first = _run_once(output_dir=output_dir, storage_root=storage_root, dataset_root=dataset_root)
    assert first.returncode == 0
    summary_path = output_dir / "global_benchmark_summary.json"
    first_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert first_summary["historical_comparison"] is None

    second = _run_once(output_dir=output_dir, storage_root=storage_root, dataset_root=dataset_root)
    assert second.returncode == 0
    second_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    historical = second_summary["historical_comparison"]
    assert isinstance(historical, dict)
    assert set(historical.keys()) == {
        "arc_agi_2",
        "math_competitions",
        "last_humanity_exam",
        "gsm8k_proxy",
        "mmlu_proxy",
    }
    assert "delta_ternary" in historical["arc_agi_2"]

    history_path = storage_root / "benchmarks" / "run_all_global_benchmarks_history.jsonl"
    assert history_path.exists()
    lines = [line for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 2

    grammar_path = storage_root / "galaxies_enriched" / "galaxies" / "Grammar.jsonl"
    entries = [
        json.loads(line)
        for line in grammar_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    memory_entries = [entry for entry in entries if entry.get("category") == "benchmark_memory_global"]
    assert len(memory_entries) >= 5
