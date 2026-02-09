#!/usr/bin/env python3
"""Download benchmark universe datasets/repositories for Knowledge3D.

This script is ingestion-path tooling. It does not affect the sovereign hot path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Artifact:
    kind: str  # git | file | archive | manual
    url: str
    target: str
    note: str = ""


@dataclass(frozen=True)
class BenchmarkSpec:
    tier: str
    description: str
    artifacts: tuple[Artifact, ...]


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "arc_agi_2": BenchmarkSpec(
        tier="prize",
        description="ARC Prize benchmark (official data access may require manual steps).",
        artifacts=(
            Artifact(
                kind="manual",
                url="https://arcprize.org/",
                target="arc_agi_2",
                note="Official ARC-AGI 2 data may require account/auth; keep local copy under this folder.",
            ),
        ),
    ),
    "gpqa": BenchmarkSpec(
        tier="prize",
        description="Graduate-level STEM QA benchmark.",
        artifacts=(
            Artifact(
                kind="git",
                url="https://github.com/idavidrein/gpqa.git",
                target="gpqa/repo",
            ),
        ),
    ),
    "math": BenchmarkSpec(
        tier="prize",
        description="Hendrycks MATH-style corpus via public mirror.",
        artifacts=(
            Artifact(
                kind="file",
                url="https://huggingface.co/datasets/qwedsacf/competition_math/resolve/main/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet",
                target="math/train.parquet",
            ),
        ),
    ),
    "imo_grand_challenge": BenchmarkSpec(
        tier="prize",
        description="IMO Grand Challenge resources.",
        artifacts=(
            Artifact(
                kind="manual",
                url="https://imo-grand-challenge.github.io/",
                target="imo_grand_challenge",
                note="Competition data access/process is managed by challenge organizers.",
            ),
        ),
    ),
    "mmlu": BenchmarkSpec(
        tier="standard",
        description="Massive Multitask Language Understanding benchmark.",
        artifacts=(
            Artifact(
                kind="archive",
                url="https://people.eecs.berkeley.edu/~hendrycks/data.tar",
                target="mmlu/data.tar",
            ),
        ),
    ),
    "gsm8k": BenchmarkSpec(
        tier="standard",
        description="Grade School Math 8K benchmark.",
        artifacts=(
            Artifact(
                kind="git",
                url="https://github.com/openai/grade-school-math.git",
                target="gsm8k/repo",
            ),
        ),
    ),
    "humaneval": BenchmarkSpec(
        tier="standard",
        description="HumanEval code generation benchmark.",
        artifacts=(
            Artifact(
                kind="git",
                url="https://github.com/openai/human-eval.git",
                target="humaneval/repo",
            ),
        ),
    ),
    "hellaswag": BenchmarkSpec(
        tier="standard",
        description="HellaSwag commonsense benchmark.",
        artifacts=(
            Artifact(
                kind="git",
                url="https://github.com/rowanz/hellaswag.git",
                target="hellaswag/repo",
            ),
        ),
    ),
    "truthfulqa": BenchmarkSpec(
        tier="standard",
        description="TruthfulQA benchmark.",
        artifacts=(
            Artifact(
                kind="git",
                url="https://github.com/sylinrl/TruthfulQA.git",
                target="truthfulqa/repo",
            ),
        ),
    ),
    "big_bench": BenchmarkSpec(
        tier="standard",
        description="BIG-bench task suite.",
        artifacts=(
            Artifact(
                kind="git",
                url="https://github.com/google/BIG-bench.git",
                target="big_bench/repo",
            ),
        ),
    ),
    "alphageometry": BenchmarkSpec(
        tier="specialized",
        description="AlphaGeometry benchmark/code.",
        artifacts=(
            Artifact(
                kind="git",
                url="https://github.com/google-deepmind/alphageometry.git",
                target="alphageometry/repo",
            ),
        ),
    ),
    "theoremqa": BenchmarkSpec(
        tier="specialized",
        description="TheoremQA benchmark.",
        artifacts=(
            Artifact(
                kind="git",
                url="https://github.com/wenhuchen/TheoremQA.git",
                target="theoremqa/repo",
            ),
        ),
    ),
    "bbh": BenchmarkSpec(
        tier="specialized",
        description="BIG-Bench Hard benchmark.",
        artifacts=(
            Artifact(
                kind="git",
                url="https://github.com/suzgunmirac/BIG-Bench-Hard.git",
                target="bbh/repo",
            ),
        ),
    ),
    "drop": BenchmarkSpec(
        tier="specialized",
        description="DROP reading comprehension benchmark.",
        artifacts=(
            Artifact(
                kind="archive",
                url="https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip",
                target="drop/drop_dataset.zip",
            ),
        ),
    ),
    "piqa": BenchmarkSpec(
        tier="specialized",
        description="PIQA physical commonsense benchmark.",
        artifacts=(
            Artifact(
                kind="manual",
                url="https://yonatanbisk.com/piqa/",
                target="piqa",
                note="PIQA hosting varies; use official page or HuggingFace dataset mirror and place files here.",
            ),
        ),
    ),
}


def _download_file(url: str, target: Path, timeout: int) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            with target.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except urllib.error.HTTPError as exc:
        return {"status": "error", "reason": f"http_{exc.code}", "url": url}
    except urllib.error.URLError as exc:
        return {"status": "error", "reason": f"url_error:{exc.reason}", "url": url}
    except Exception as exc:  # pragma: no cover - environment/network variability.
        return {"status": "error", "reason": str(exc), "url": url}
    return {"status": "ok", "path": str(target), "url": url}


def _extract_archive(path: Path) -> dict[str, Any]:
    try:
        if path.suffix == ".zip":
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(path.parent)
            return {"status": "ok", "extracted_to": str(path.parent)}
        if path.suffix == ".tar" or "".join(path.suffixes[-2:]) in {".tar.gz", ".tgz"}:
            with tarfile.open(path, "r:*") as tf:
                try:
                    tf.extractall(path.parent, filter="data")
                except TypeError:
                    tf.extractall(path.parent)
            return {"status": "ok", "extracted_to": str(path.parent)}
    except Exception as exc:  # pragma: no cover - environment/network variability.
        return {"status": "error", "reason": f"extract_failed:{exc}"}
    return {"status": "skipped", "reason": "unsupported_archive_type"}


def _clone_git(url: str, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(target)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {
            "status": "error",
            "reason": result.stderr.strip() or "git_clone_failed",
            "url": url,
        }
    return {"status": "ok", "path": str(target), "url": url}


def _should_include(name: str, spec: BenchmarkSpec, wanted: set[str], tiers: set[str]) -> bool:
    if wanted and name not in wanted:
        return False
    if tiers and spec.tier not in tiers:
        return False
    return True


def _parse_csv_set(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("../Knowledge3D.local/datasets/global_benchmarks"),
        help="Root folder for all benchmark assets",
    )
    parser.add_argument(
        "--benchmarks",
        default="",
        help="Comma-separated benchmark names to download (default: all).",
    )
    parser.add_argument(
        "--tiers",
        default="",
        help="Comma-separated tiers to include: prize,standard,specialized.",
    )
    parser.add_argument("--list", action="store_true", help="List available benchmarks and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned actions without downloading.")
    parser.add_argument("--force", action="store_true", help="Re-download even if targets already exist.")
    parser.add_argument("--timeout", type=int, default=60, help="Network timeout in seconds.")
    args = parser.parse_args()

    if args.list:
        for name, spec in sorted(BENCHMARKS.items()):
            print(f"{name:20s} tier={spec.tier:11s} {spec.description}")
        return

    wanted = _parse_csv_set(args.benchmarks)
    tiers = _parse_csv_set(args.tiers)
    args.root.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "root": str(args.root.resolve()),
        "dry_run": bool(args.dry_run),
        "benchmarks": {},
    }

    for name, spec in sorted(BENCHMARKS.items()):
        if not _should_include(name, spec, wanted, tiers):
            continue
        bench_report: dict[str, Any] = {
            "tier": spec.tier,
            "description": spec.description,
            "artifacts": [],
        }
        print(f"[benchmark] {name} ({spec.tier})")
        for artifact in spec.artifacts:
            target = args.root / artifact.target
            artifact_report: dict[str, Any] = {
                "kind": artifact.kind,
                "url": artifact.url,
                "target": str(target),
                "note": artifact.note,
            }

            if artifact.kind == "manual":
                artifact_report["status"] = "manual_required"
                print(f"  - manual: {artifact.url}")
                if artifact.note:
                    print(f"    note: {artifact.note}")
                bench_report["artifacts"].append(artifact_report)
                continue

            if target.exists() and not args.force:
                artifact_report["status"] = "already_present"
                print(f"  - skip existing: {target}")
                bench_report["artifacts"].append(artifact_report)
                continue

            if args.dry_run:
                artifact_report["status"] = "planned"
                print(f"  - plan {artifact.kind}: {artifact.url} -> {target}")
                bench_report["artifacts"].append(artifact_report)
                continue

            if artifact.kind == "git":
                outcome = _clone_git(artifact.url, target)
            elif artifact.kind in {"file", "archive"}:
                outcome = _download_file(artifact.url, target, timeout=args.timeout)
                if outcome.get("status") == "ok" and artifact.kind == "archive":
                    outcome["extract"] = _extract_archive(target)
            else:
                outcome = {"status": "error", "reason": f"unknown_kind:{artifact.kind}"}

            artifact_report.update(outcome)
            status = str(outcome.get("status", "unknown"))
            print(f"  - {artifact.kind}: {status}")
            if outcome.get("reason"):
                print(f"    reason: {outcome['reason']}")
            bench_report["artifacts"].append(artifact_report)

        report["benchmarks"][name] = bench_report

    report_path = args.root / "download_manifest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] manifest: {report_path}")


if __name__ == "__main__":
    main()
