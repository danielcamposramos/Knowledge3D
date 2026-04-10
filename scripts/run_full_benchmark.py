from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from scripts.run_headless_tablet_benchmarks import run_tablet_benchmark_suite


LOG_ROOT = Path("/K3D/Knowledge3D.local/logs")
ARC3_ARCHIVE_REASON = "arc3_local_archived_use_arc3_sdk_agent_or_headless_tablet_runner"


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _suite_summary(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: {key: value for key, value in result.items() if key != "results"}
        for name, result in results.items()
    }


def run_full_benchmark(
    *,
    mmlu_count: int = 50,
    imo_count: int = 20,
    arc3_count: int = 20,
    gsm8k_count: int = 10,
    lhe_count: int = 10,
    arc2_count: int = 10,
    storage_root: str | Path,
    log_root: str | Path = LOG_ROOT,
) -> dict[str, object]:
    timestamp = _ts()
    log_dir = Path(log_root) / f"phase_e_{timestamp}"
    payload = run_tablet_benchmark_suite(
        argparse.Namespace(
            storage_root=str(storage_root),
            log_dir=str(log_dir),
            arc2_dataset_path=None,
            mmlu_dataset_path=None,
            gsm8k_dataset_path=None,
            lhe_dataset_path=None,
            math_dataset_path=None,
            imo_dataset_path=None,
            arc2_count=int(arc2_count),
            arc3_count=0,
            mmlu_count=int(mmlu_count),
            gsm8k_count=int(gsm8k_count),
            lhe_count=int(lhe_count),
            math_count=0,
            amc_aime_count=0,
            omni_math_count=0,
            imo_count=int(imo_count),
            use_enriched=True,
            shutdown_timeout_s=30.0,
            output=None,
        )
    )
    results = dict(payload["results"])
    headless_summary = dict(payload["summary"])
    summary = {
        "timestamp": timestamp,
        "elapsed_seconds": float(headless_summary.get("elapsed_seconds", 0.0) or 0.0),
        "log_dir": str(log_dir),
        "hardware_profile": dict(headless_summary.get("hardware_profile") or {}),
        "sleep_consolidation": dict(headless_summary.get("sleep_consolidation") or {}),
        "execution_artifacts": dict(headless_summary.get("execution_artifacts") or {}),
        "orchestrator": {
            **dict(headless_summary.get("orchestrator") or {}),
            "mode": "phase_e_compat_headless_tablet",
        },
        "suites": _suite_summary(results),
        "archived_suites": {
            "arc3_local": {
                "status": "archived",
                "requested_count": int(arc3_count),
                "reason": ARC3_ARCHIVE_REASON,
            }
        },
    }
    full_payload = {
        "summary": summary,
        "results": results,
        "headless_tablet": headless_summary,
    }
    _write_json(log_dir / "summary.json", summary)
    _write_json(log_dir / "full_results.json", results)
    return full_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="K3D Phase E benchmark compatibility wrapper")
    parser.add_argument("--mmlu-count", type=int, default=50)
    parser.add_argument("--imo-count", type=int, default=20)
    parser.add_argument("--gsm8k-count", type=int, default=10)
    parser.add_argument("--lhe-count", type=int, default=10)
    parser.add_argument("--arc2-count", type=int, default=10)
    parser.add_argument("--arc3-count", type=int, default=20)
    parser.add_argument("--storage-root", default="/K3D/Knowledge3D.local")
    parser.add_argument("--log-root", default=str(LOG_ROOT))
    args = parser.parse_args()

    payload = run_full_benchmark(
        mmlu_count=args.mmlu_count,
        imo_count=args.imo_count,
        gsm8k_count=args.gsm8k_count,
        lhe_count=args.lhe_count,
        arc2_count=args.arc2_count,
        arc3_count=args.arc3_count,
        storage_root=args.storage_root,
        log_root=args.log_root,
    )
    summary = payload["summary"]
    suites = summary["suites"]
    print(f"\n{'=' * 60}")
    print(f"Phase E Benchmark — {summary['timestamp']}")
    print(f"{'=' * 60}")
    print(
        f"  MMLU:       {suites['mmlu']['correct']}/{suites['mmlu']['total']} "
        f"({suites['mmlu']['accuracy']:.1%})"
    )
    print(
        f"  IMO Bench:  {suites['imo']['correct']}/{suites['imo']['total']} "
        f"({suites['imo']['accuracy']:.1%})"
    )
    print(
        f"  GSM8K:      {suites['gsm8k']['correct']}/{suites['gsm8k']['total']} "
        f"({suites['gsm8k']['accuracy']:.1%})"
    )
    print(
        f"  LHE:        {suites['lhe']['correct']}/{suites['lhe']['total']} "
        f"({suites['lhe']['accuracy']:.1%})"
    )
    print(
        f"  ARC-AGI-2:  {suites['arc2']['correct']}/{suites['arc2']['total']} "
        f"({suites['arc2']['accuracy']:.1%})"
    )
    print(
        f"  ARC3 Local: archived requested={summary['archived_suites']['arc3_local']['requested_count']} "
        f"reason={summary['archived_suites']['arc3_local']['reason']}"
    )
    print(
        f"  Hardware:   {summary['hardware_profile']['physical_cores']}p/"
        f"{summary['hardware_profile']['logical_cores']}t"
    )
    print(f"  Elapsed:    {summary['elapsed_seconds']:.1f}s")
    print(f"  Logs:       {summary['log_dir']}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
