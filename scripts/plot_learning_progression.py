#!/usr/bin/env python3
"""Plot iterative learning progression and galaxy growth from marathon analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_analysis(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_score_progression(data: dict, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        (output_dir / "plot_warning.txt").write_text(
            "matplotlib is not available; no charts were generated.\n",
            encoding="utf-8",
        )
        return

    prog = data.get("progression_analysis", {})
    benchmarks = [
        ("arc_agi_2", "ARC-AGI 2"),
        ("math_competitions", "Math Competitions"),
        ("last_humanity_exam", "Last Humanity Exam"),
        ("gsm8k_proxy", "GSM8K Proxy"),
        ("mmlu_proxy", "MMLU Proxy"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = False
    for key, label in benchmarks:
        scores = prog.get(key, {}).get("scores", [])
        if not scores:
            continue
        x = list(range(1, len(scores) + 1))
        y = [float(v) * 100.0 for v in scores]
        ax.plot(x, y, marker="o", linewidth=2, label=label)
        plotted = True
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Iterative Learning Progression")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "progression_chart.png", dpi=200)
    plt.close(fig)


def _plot_galaxy_growth(data: dict, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    history = data.get("galaxy_growth_history", [])
    galaxies = ["Drawing", "Grammar", "Math", "Reality", "3DObjects", "Audio"]
    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = False
    x = list(range(1, len(history) + 1))
    for galaxy in galaxies:
        y = [int(item.get(galaxy, {}).get("total", 0)) for item in history]
        if any(y):
            ax.plot(x, y, marker="o", linewidth=2, label=galaxy)
            plotted = True
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Entries")
    ax.set_title("Galaxy Growth Across Iterations")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "galaxy_growth_chart.png", dpi=200)
    plt.close(fig)


def _plot_specialist_growth(data: dict, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    history = data.get("specialist_tree_history", [])
    counts = [int(item.get("specialist_count", 0)) for item in history]
    x = list(range(1, len(counts) + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, counts, marker="o", linewidth=2, color="#1f77b4")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Specialist Count")
    ax.set_title("Matryoshka Specialist Growth")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "specialist_growth_chart.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-file",
        type=Path,
        default=Path("../Knowledge3D.local/results/iterative_learning/marathon_analysis.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../Knowledge3D.local/results/iterative_learning"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_analysis(args.analysis_file)

    _plot_score_progression(data, args.output_dir)
    _plot_galaxy_growth(data, args.output_dir)
    _plot_specialist_growth(data, args.output_dir)
    print(f"Charts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
