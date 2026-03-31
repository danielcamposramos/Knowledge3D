from __future__ import annotations

import argparse
import json
from datetime import datetime
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LOG_ROOT = Path("/K3D/Knowledge3D.local/logs")

Knowledgeverse = None
MMLUBenchmark = None
GSM8KBenchmark = None
IMOBenchmark = None
LastHumanityExamBenchmark = None
ARCAGI2Benchmark = None
run_local_arc3 = None


def _ensure_full_benchmark_runtime() -> None:
    global Knowledgeverse
    global MMLUBenchmark
    global GSM8KBenchmark
    global IMOBenchmark
    global LastHumanityExamBenchmark
    global ARCAGI2Benchmark
    global run_local_arc3

    if Knowledgeverse is None:
        from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse as _Knowledgeverse

        Knowledgeverse = _Knowledgeverse
    if MMLUBenchmark is None:
        from benchmarks.mmlu import MMLUBenchmark as _MMLUBenchmark

        MMLUBenchmark = _MMLUBenchmark
    if GSM8KBenchmark is None:
        from benchmarks.gsm8k import GSM8KBenchmark as _GSM8KBenchmark

        GSM8KBenchmark = _GSM8KBenchmark
    if IMOBenchmark is None:
        from benchmarks.imo_bench import IMOBenchmark as _IMOBenchmark

        IMOBenchmark = _IMOBenchmark
    if LastHumanityExamBenchmark is None:
        from benchmarks.last_humanity_exam import LastHumanityExamBenchmark as _LastHumanityExamBenchmark

        LastHumanityExamBenchmark = _LastHumanityExamBenchmark
    if ARCAGI2Benchmark is None:
        from benchmarks.arc_agi_2 import ARCAGI2Benchmark as _ARCAGI2Benchmark

        ARCAGI2Benchmark = _ARCAGI2Benchmark
    if run_local_arc3 is None:
        from benchmarks.arc3_local import run_local_arc3 as _run_local_arc3

        run_local_arc3 = _run_local_arc3


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _detect_hardware_profile(storage_root: str | Path) -> dict[str, Any]:
    root = Path(storage_root)
    path = root / "hardware_profile.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass

    logical_cores = int(os.cpu_count() or 1)
    physical_cores = logical_cores
    try:
        core_rows = os.popen("lscpu -p=Core,Socket 2>/dev/null").read().splitlines()
        core_pairs = {
            tuple(line.split(","))
            for line in core_rows
            if line and not line.startswith("#") and "," in line
        }
        if core_pairs:
            physical_cores = len(core_pairs)
    except Exception:
        physical_cores = logical_cores

    total_memory_bytes = None
    try:
        total_memory_bytes = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        total_memory_bytes = None

    profile = {
        "generated_at": datetime.now().isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cores": logical_cores,
        "physical_cores": physical_cores,
        "total_memory_bytes": total_memory_bytes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    return profile


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _row_log_payload(suite_name: str, source: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    payload.setdefault("suite", suite_name)
    if isinstance(source, dict):
        source_id = source.get("id") or source.get("task_id") or source.get("question_id")
        if source_id is not None:
            payload.setdefault("source_id", str(source_id))
    return payload


def _progress_log_payload(suite_name: str, progress: dict[str, Any]) -> dict[str, Any]:
    payload = dict(progress)
    payload.setdefault("suite", suite_name)
    return payload


def _normalize_native_result(suite_name: str, raw: dict[str, Any]) -> dict[str, Any]:
    total = raw.get("total")
    if total is None:
        total = raw.get("total_questions")
    if total is None:
        total = raw.get("total_tasks")
    if total is None:
        total = len(list(raw.get("results") or []))
    result = dict(raw)
    result["suite"] = suite_name
    result["total"] = int(total)
    result["correct"] = int(raw.get("correct", 0))
    result["accuracy"] = float(raw.get("accuracy", 0.0))
    return result


def _run_native_suite(
    *,
    suite_name: str,
    count: int,
    knowledgeverse: Any,
    row_log_path: Path | None = None,
    progress_log_path: Path | None = None,
) -> dict[str, Any]:
    suite_name = str(suite_name).strip().lower()
    if row_log_path is not None:
        row_log_path.parent.mkdir(parents=True, exist_ok=True)
        row_log_path.write_text("", encoding="utf-8")
    if progress_log_path is not None:
        progress_log_path.parent.mkdir(parents=True, exist_ok=True)
        progress_log_path.write_text("", encoding="utf-8")

    def _row_cb(source: dict[str, Any], result: dict[str, Any]) -> None:
        if row_log_path is None:
            return
        _append_jsonl(row_log_path, _row_log_payload(suite_name, source, result))

    def _progress_cb(progress: dict[str, Any]) -> None:
        payload = _progress_log_payload(suite_name, progress)
        if progress_log_path is not None:
            _append_jsonl(progress_log_path, payload)
        completed = int(payload.get("completed", 0))
        total = int(payload.get("total", 0))
        correct = int(payload.get("correct", 0))
        elapsed_s = float(payload.get("elapsed_s", 0.0))
        print(
            f"[E25] Progress {suite_name}: {completed}/{total} correct={correct} elapsed={elapsed_s:.1f}s",
            flush=True,
        )

    if suite_name == "mmlu":
        benchmark = MMLUBenchmark(knowledgeverse=knowledgeverse, max_questions=max(1, int(count)))
    elif suite_name == "imo":
        benchmark = IMOBenchmark(knowledgeverse=knowledgeverse, max_questions=max(1, int(count)))
    elif suite_name == "gsm8k":
        benchmark = GSM8KBenchmark(knowledgeverse=knowledgeverse, max_questions=max(1, int(count)))
    elif suite_name == "lhe":
        benchmark = LastHumanityExamBenchmark(knowledgeverse=knowledgeverse, max_questions=max(1, int(count)))
    elif suite_name == "arc2":
        benchmark = ARCAGI2Benchmark(knowledgeverse=knowledgeverse, max_tasks=max(1, int(count)))
    elif suite_name == "arc3_local":
        return _normalize_native_result(
            suite_name,
            run_local_arc3(
                count=max(1, int(count)),
                knowledgeverse=knowledgeverse,
                log_path=row_log_path,
                progress_cb=_progress_cb,
                progress_every=1,
            ),
        )
    else:
        raise ValueError(f"unsupported_native_suite:{suite_name}")
    return _normalize_native_result(
        suite_name,
        benchmark.run_benchmark(
            use_enriched=True,
            row_cb=_row_cb,
            progress_cb=_progress_cb,
            progress_every=1,
        ),
    )


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
    _ensure_full_benchmark_runtime()
    print("[E25] Benchmark modules imported", flush=True)
    timestamp = _ts()
    log_dir = Path(log_root) / f"phase_e_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    hardware_profile = _detect_hardware_profile(storage_root)
    print(f"[E25] Initializing Knowledgeverse storage_root={storage_root}", flush=True)
    kv = Knowledgeverse(storage_root=storage_root)
    print("[E25] Knowledgeverse init complete", flush=True)

    all_results: dict[str, dict[str, object]] = {}
    suite_order = [
        ("mmlu", mmlu_count),
        ("imo", imo_count),
        ("gsm8k", gsm8k_count),
        ("lhe", lhe_count),
        ("arc2", arc2_count),
        ("arc3_local", arc3_count),
    ]

    for suite_name, suite_count in suite_order:
        print(f"[E25] Starting suite: {suite_name} count={suite_count}", flush=True)
        result = _run_native_suite(
            suite_name=suite_name,
            count=suite_count,
            knowledgeverse=kv,
            row_log_path=log_dir / f"{suite_name}.jsonl",
            progress_log_path=log_dir / f"{suite_name}_progress.jsonl",
        )
        _write_jsonl(log_dir / f"{suite_name}.jsonl", list(result.get("results") or []))
        print(f"[E25] {suite_name} complete", flush=True)
        all_results[suite_name] = result
        partial_summary = {
            "timestamp": timestamp,
            "elapsed_seconds": round(time.time() - start, 2),
            "log_dir": str(log_dir),
            "hardware_profile": hardware_profile,
            "completed_suites": list(all_results.keys()),
            "suites": {
                name: {key: value for key, value in suite_result.items() if key != "results"}
                for name, suite_result in all_results.items()
            },
        }
        (log_dir / "summary.partial.json").write_text(
            json.dumps(partial_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    elapsed = round(time.time() - start, 2)
    summary = {
        "timestamp": timestamp,
        "elapsed_seconds": elapsed,
        "log_dir": str(log_dir),
        "hardware_profile": hardware_profile,
        "suites": {
            name: {key: value for key, value in result.items() if key != "results"}
            for name, result in all_results.items()
        },
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (log_dir / "full_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return {"summary": summary, "results": all_results}


def main() -> int:
    parser = argparse.ArgumentParser(description="K3D Phase E full benchmark")
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
    print(f"Phase E Full Benchmark — {summary['timestamp']}")
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
        f"  ARC3 Local: {suites['arc3_local']['correct']}/{suites['arc3_local']['total']} "
        f"({suites['arc3_local']['accuracy']:.1%}) "
        f"first_move={suites['arc3_local'].get('correct_first_moves', 0)}/{suites['arc3_local']['total']}"
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
