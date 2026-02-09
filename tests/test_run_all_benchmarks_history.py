from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_once(output_dir: Path, storage_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/run_all_benchmarks.py",
            "--max-arc-tasks",
            "1",
            "--max-math-problems",
            "1",
            "--max-lhe-questions",
            "1",
            "--output-dir",
            str(output_dir),
            "--storage-root",
            str(storage_root),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_run_all_benchmarks_historical_tracking_and_galaxy_memory(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    storage_root = tmp_path / "storage"

    first = _run_once(output_dir=output_dir, storage_root=storage_root)
    assert first.returncode == 0
    summary_path = output_dir / "week14_benchmark_summary.json"
    assert summary_path.exists()
    first_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert first_summary["historical_comparison"] is None

    second = _run_once(output_dir=output_dir, storage_root=storage_root)
    assert second.returncode == 0
    second_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    historical = second_summary["historical_comparison"]
    assert isinstance(historical, dict)
    assert set(historical.keys()) == {"arc_agi_2", "math_competitions", "last_humanity_exam"}
    for bench_name in historical:
        assert "previous" in historical[bench_name]
        assert "current" in historical[bench_name]
        assert "delta" in historical[bench_name]
        assert "delta_ternary" in historical[bench_name]
        assert historical[bench_name]["status"] in {"MAINTAINED", "IMPROVEMENT", "REGRESSION"}

    history_path = storage_root / "benchmarks" / "run_all_benchmarks_history.jsonl"
    assert history_path.exists()
    history_lines = [line for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(history_lines) >= 2

    grammar_path = storage_root / "galaxies_enriched" / "galaxies" / "Grammar.jsonl"
    assert grammar_path.exists()
    entries = [
        json.loads(line)
        for line in grammar_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    benchmark_entries = [e for e in entries if e.get("category") == "benchmark_memory"]
    assert len(benchmark_entries) >= 3
