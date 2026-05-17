#!/usr/bin/env python3
"""Run iterative benchmark cycles and analyze continuous-learning progression."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure repo root is importable when executed as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def _extract_score(summary: dict[str, Any], benchmark_name: str) -> float:
    integrated = summary.get("integrated_results", {})
    if benchmark_name == "arc_agi_2":
        return float(integrated.get("arc_agi_2", {}).get("enriched", {}).get("accuracy", 0.0))
    if benchmark_name == "math_competitions":
        return float(integrated.get("math_competitions", {}).get("enriched", {}).get("overall_accuracy", 0.0))
    if benchmark_name == "last_humanity_exam":
        return float(integrated.get("last_humanity_exam", {}).get("enriched", {}).get("accuracy", 0.0))
    return float(summary.get("proxy_results", {}).get(benchmark_name, {}).get("accuracy", 0.0))


def _run_iteration(
    *,
    iteration_num: int,
    output_root: Path,
    storage_root: Path,
    dataset_root: Path,
    max_arc_tasks: int,
    max_math_problems: int,
    max_lhe_questions: int,
    run_proxy: bool,
    max_proxy_questions: int,
    python_exe: str,
    unified_storage_root: Path | None,
) -> dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"ITERATION {iteration_num} @ {datetime.now(tz=timezone.utc).isoformat()}")
    print("=" * 80)

    output_dir = output_root / f"iteration_{iteration_num:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe,
        "scripts/run_all_global_benchmarks.py",
        "--output-dir",
        str(output_dir),
        "--storage-root",
        str(storage_root),
        "--dataset-root",
        str(dataset_root),
        "--max-arc-tasks",
        str(max_arc_tasks),
        "--max-math-problems",
        str(max_math_problems),
        "--max-lhe-questions",
        str(max_lhe_questions),
        "--model-persistence-mode",
        "unified",
    ]
    if unified_storage_root is not None:
        cmd.extend(["--unified-storage-root", str(unified_storage_root)])
    if run_proxy:
        cmd.extend(["--run-proxy", "--max-proxy-questions", str(max_proxy_questions)])

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started

    summary_file = output_dir / "global_benchmark_summary.json"
    if not summary_file.exists():
        return {
            "iteration": iteration_num,
            "status": "failed",
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
            "elapsed_seconds": elapsed,
            "summary_path": str(summary_file),
        }

    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    summary["iteration"] = iteration_num
    summary["elapsed_seconds"] = elapsed
    summary["returncode"] = proc.returncode
    return summary


def _monitor_galaxy_growth(storage_root: Path) -> dict[str, Any]:
    kv = Knowledgeverse(storage_root=storage_root)
    report: dict[str, Any] = {}
    for galaxy_name in ("Drawing", "Grammar", "Math", "Reality", "3DObjects", "Audio"):
        galaxy = kv.galaxy_manager.get_galaxy(galaxy_name)
        total = len(getattr(galaxy, "entries", []))
        generated = sum(
            1
            for entry in getattr(galaxy, "entries", [])
            if isinstance(entry, dict) and bool(entry.get("metadata", {}).get("generated"))
        )
        report[galaxy_name] = {
            "total": total,
            "generated": generated,
            "generation_rate": (generated / total) if total else 0.0,
        }
    return report


def _monitor_specialist_tree(storage_root: Path) -> dict[str, Any]:
    tree_path = storage_root / "checkpoints" / "trm_specialist_tree.json"
    if not tree_path.exists():
        return {"specialist_count": 0, "tree_path": str(tree_path), "present": False}
    payload = json.loads(tree_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"specialist_count": 0, "tree_path": str(tree_path), "present": False}

    def _count(node: dict[str, Any]) -> int:
        count = 1
        for child in node.get("children", []) if isinstance(node.get("children"), list) else []:
            if isinstance(child, dict):
                count += _count(child)
        return count

    return {
        "specialist_count": _count(payload),
        "tree_path": str(tree_path),
        "present": True,
    }


def _analyze_progression(iteration_results: list[dict[str, Any]]) -> dict[str, Any]:
    benchmarks = [
        "arc_agi_2",
        "math_competitions",
        "last_humanity_exam",
        "gsm8k_proxy",
        "mmlu_proxy",
    ]
    progression: dict[str, Any] = {}
    for benchmark_name in benchmarks:
        scores = [_extract_score(summary, benchmark_name) for summary in iteration_results]
        if not scores:
            continue
        initial = scores[0]
        final = scores[-1]
        delta = final - initial
        plateau_detected = False
        if len(scores) >= 3:
            last = scores[-3:]
            plateau_detected = (max(last) - min(last)) < 0.01
        progression[benchmark_name] = {
            "scores": scores,
            "initial": initial,
            "final": final,
            "total_improvement": delta,
            "plateau_detected": plateau_detected,
        }
    return progression


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/results/iterative_learning"),
    )
    parser.add_argument("--storage-root", type=Path, default=Path("/K3D/Knowledge3D.local"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/datasets/global_benchmarks"),
    )
    parser.add_argument("--max-arc-tasks", type=int, default=100)
    parser.add_argument("--max-math-problems", type=int, default=100)
    parser.add_argument("--max-lhe-questions", type=int, default=50)
    parser.add_argument("--run-proxy", action="store_true")
    parser.add_argument("--max-proxy-questions", type=int, default=20)
    parser.add_argument("--pause-seconds", type=float, default=5.0)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument(
        "--unified-storage-root",
        type=Path,
        default=None,
        help="Optional shared world path for unified persistence mode (defaults to storage_root/galaxies_enriched).",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    shared_world = args.unified_storage_root or (args.storage_root / "galaxies_enriched")

    iteration_results: list[dict[str, Any]] = []
    galaxy_growth_history: list[dict[str, Any]] = []
    specialist_tree_history: list[dict[str, Any]] = []

    for iteration in range(1, args.iterations + 1):
        summary = _run_iteration(
            iteration_num=iteration,
            output_root=args.output_root,
            storage_root=args.storage_root,
            dataset_root=args.dataset_root,
            max_arc_tasks=args.max_arc_tasks,
            max_math_problems=args.max_math_problems,
            max_lhe_questions=args.max_lhe_questions,
            run_proxy=args.run_proxy,
            max_proxy_questions=args.max_proxy_questions,
            python_exe=args.python_exe,
            unified_storage_root=shared_world,
        )
        iteration_results.append(summary)
        galaxy_growth = _monitor_galaxy_growth(shared_world)
        specialist_tree = _monitor_specialist_tree(shared_world)
        galaxy_growth_history.append(galaxy_growth)
        specialist_tree_history.append(specialist_tree)

        print(
            "Iteration summary: "
            f"ARC={_extract_score(summary, 'arc_agi_2'):.2%}, "
            f"Math={_extract_score(summary, 'math_competitions'):.2%}, "
            f"LHE={_extract_score(summary, 'last_humanity_exam'):.2%}, "
            f"Specialists={specialist_tree.get('specialist_count', 0)}"
        )
        if iteration < args.iterations and args.pause_seconds > 0:
            time.sleep(args.pause_seconds)

    progression = _analyze_progression(iteration_results)
    analysis = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "iterations": args.iterations,
        "config": {
            "max_arc_tasks": args.max_arc_tasks,
            "max_math_problems": args.max_math_problems,
            "max_lhe_questions": args.max_lhe_questions,
            "run_proxy": args.run_proxy,
            "max_proxy_questions": args.max_proxy_questions,
            "storage_root": str(args.storage_root),
            "shared_world_root": str(shared_world),
            "dataset_root": str(args.dataset_root),
        },
        "iteration_results": iteration_results,
        "galaxy_growth_history": galaxy_growth_history,
        "specialist_tree_history": specialist_tree_history,
        "progression_analysis": progression,
    }
    output_file = args.output_root / "marathon_analysis.json"
    output_file.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"Marathon analysis written to: {output_file}")


if __name__ == "__main__":
    main()
