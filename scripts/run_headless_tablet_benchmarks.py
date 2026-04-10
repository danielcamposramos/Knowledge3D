#!/usr/bin/env python3
"""Run local benchmark suites through one live headless Tablet/WINE session."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.arc_agi_2 import ARCAGI2Benchmark
from benchmarks.gsm8k import GSM8KBenchmark
from benchmarks.imo_bench import IMOBenchmark
from benchmarks.last_humanity_exam import LastHumanityExamBenchmark
from benchmarks.math_competitions import UnifiedMathBenchmark
from benchmarks.mmlu import MMLUBenchmark
from knowledge3d.bridge.headless_tablet import CommandHandler, HeadlessTabletMPC
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _log_section(message: str) -> None:
    print(f"[HEADLESS BENCH] {message}", file=sys.stderr, flush=True)


def _skip_summary(reason: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": reason,
    }


def _archived_suite(reason: str, requested_count: int) -> dict[str, Any]:
    return {
        "status": "archived",
        "reason": str(reason),
        "requested_count": int(requested_count),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_execution_artifacts(
    *,
    log_dir: Path,
    start: float,
    hardware_profile: dict[str, Any],
    feeder_workers: int,
    all_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    execution_summary = {
        "mode": "headless_tablet_wine_session",
        "phase": "execution",
        "timestamp": log_dir.name,
        "elapsed_seconds": round(time.time() - start, 2),
        "log_dir": str(log_dir),
        "hardware_profile": hardware_profile,
        "orchestrator": {
            "session_model": "one_live_knowledgeverse",
            "feeder_workers": int(feeder_workers),
            "mid_session_unload_allowed": False,
        },
        "completed_suites": list(all_results.keys()),
        "benchmarks": {
            name: {key: value for key, value in result.items() if key != "results"}
            for name, result in all_results.items()
        },
    }
    (log_dir / "summary.execution.json").write_text(
        json.dumps(execution_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (log_dir / "full_results.execution.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return execution_summary


def _shutdown_with_timeout(kv: Knowledgeverse, timeout_s: float) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    error: BaseException | None = None

    def _run_shutdown() -> None:
        nonlocal summary, error
        try:
            try:
                summary = dict(kv.shutdown(persist=False, profile="benchmark") or {})
            except TypeError:
                try:
                    summary = dict(kv.shutdown(persist=False) or {})
                except TypeError:
                    summary = dict(kv.shutdown() or {})
        except BaseException as exc:  # pragma: no cover - defensive runtime guard
            error = exc

    worker = threading.Thread(target=_run_shutdown, name="k3d-benchmark-shutdown", daemon=True)
    worker.start()
    worker.join(None if timeout_s <= 0 else float(timeout_s))
    if worker.is_alive():
        return {
            "status": "timed_out",
            "completed": False,
            "timeout_seconds": float(timeout_s),
        }
    if error is not None:
        return {
            "status": "error",
            "completed": False,
            "exception_type": type(error).__name__,
            "detail": str(error),
            "timeout_seconds": float(timeout_s),
        }
    payload = dict(summary)
    payload.setdefault("status", "completed")
    payload.setdefault(
        "completed",
        str(payload.get("status") or "").strip().lower() in {"completed", "fast_exit", "idempotent_noop"},
    )
    payload.setdefault("timeout_seconds", float(timeout_s))
    return payload


def _row_log_payload(suite_name: str, source: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    payload.setdefault("suite", suite_name)
    if isinstance(source, dict):
        source_id = source.get("id") or source.get("task_id") or source.get("question_id") or source.get("problem_id")
        if source_id is not None:
            payload.setdefault("source_id", str(source_id))
    return payload


def _progress_log_payload(suite_name: str, progress: dict[str, Any]) -> dict[str, Any]:
    payload = dict(progress)
    payload.setdefault("suite", suite_name)
    return payload


def _normalize_route_family(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "UNKNOWN"
    if text.endswith("_TASK"):
        text = text[:-5]
    return text


def _row_route_family(row: dict[str, Any]) -> str:
    route = row.get("route", {})
    if isinstance(route, dict):
        family = str(route.get("route_family") or route.get("surface_kind") or "").strip()
        if family:
            return _normalize_route_family(family)
    task_result = row.get("task_result", {})
    if isinstance(task_result, dict):
        direct = str(task_result.get("query_type") or "").strip()
        if direct:
            return _normalize_route_family(direct)
        nested_route = task_result.get("route", {})
        if isinstance(nested_route, dict):
            family = str(nested_route.get("route_family") or nested_route.get("surface_kind") or "").strip()
            if family:
                return _normalize_route_family(family)
        trm_dispatch = task_result.get("trm_dispatch", {})
        if isinstance(trm_dispatch, dict):
            family = str(trm_dispatch.get("task_type") or "").strip()
            if family:
                return _normalize_route_family(family)
        nested_task_result = task_result.get("task_result", {})
        if isinstance(nested_task_result, dict):
            answer_kind_family = str(nested_task_result.get("answer_kind") or "").strip()
            if answer_kind_family:
                return _normalize_route_family(answer_kind_family)
    return "UNKNOWN"


def _row_trm_dispatch_type(row: dict[str, Any]) -> str:
    task_result = row.get("task_result", {})
    if isinstance(task_result, dict):
        trm_dispatch = task_result.get("trm_dispatch", {})
        if isinstance(trm_dispatch, dict):
            task_type = str(trm_dispatch.get("task_type") or "").strip()
            if task_type:
                return _normalize_route_family(task_type)
        query_type = str(task_result.get("query_type") or "").strip()
        if query_type:
            return _normalize_route_family(query_type)
    return _row_route_family(row)


def _row_elapsed_ms(row: dict[str, Any]) -> float | None:
    if "elapsed_ms" in row:
        try:
            return float(row.get("elapsed_ms") or 0.0)
        except Exception:
            return None
    if "elapsed_s" in row:
        try:
            return float(row.get("elapsed_s") or 0.0) * 1000.0
        except Exception:
            return None
    return None


def _row_gpu_execution(row: dict[str, Any]) -> bool:
    if bool(row.get("gpu_execution", False)):
        return True
    task_result = row.get("task_result", {})
    if isinstance(task_result, dict):
        return bool(task_result.get("gpu_execution", False))
    return False


def _row_answer_format(row: dict[str, Any]) -> str:
    explicit = str(row.get("answer_format") or "").strip()
    if explicit:
        return explicit
    question_type = str(row.get("question_type") or "").strip().lower()
    if question_type == "multiple_choice":
        return "multiple_choice"
    if question_type == "open_ended":
        return "open_ended"
    options = row.get("options")
    predicted = str(row.get("predicted_answer") or "").strip()
    if isinstance(options, list) and options:
        normalized_options = {str(option).strip() for option in options if str(option).strip()}
        if predicted and predicted in normalized_options:
            return "option_text_exact"
        return "multiple_choice"
    task_result = row.get("task_result", {})
    if isinstance(task_result, dict):
        nested = task_result.get("task_result", {})
        if isinstance(nested, dict):
            answer_kind = str(nested.get("answer_kind") or "").strip().lower()
            if answer_kind == "choice":
                return "multiple_choice"
    if predicted:
        try:
            float(predicted)
            return "numeric"
        except Exception:
            return "open_ended"
    return "empty"


def _augment_suite_summary(result: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in list(result.get("results") or []) if isinstance(row, dict)]
    route_counts: Counter[str] = Counter()
    trm_counts: Counter[str] = Counter()
    answer_formats: Counter[str] = Counter()
    gpu_true = 0
    elapsed_values: list[float] = []
    for row in rows:
        route_counts[_row_route_family(row)] += 1
        trm_counts[_row_trm_dispatch_type(row)] += 1
        answer_formats[_row_answer_format(row)] += 1
        if _row_gpu_execution(row):
            gpu_true += 1
        elapsed_ms = _row_elapsed_ms(row)
        if elapsed_ms is not None and elapsed_ms >= 0.0:
            elapsed_values.append(elapsed_ms)
    total = int(result.get("total", len(rows)) or 0)
    result.setdefault("route_family_distribution", {key: int(value) for key, value in sorted(route_counts.items())})
    result.setdefault(
        "trm_dispatch_task_type_distribution",
        {key: int(value) for key, value in sorted(trm_counts.items())},
    )
    result.setdefault("gpu_result_packets", f"{gpu_true} / {total}")
    avg_elapsed_ms = (sum(elapsed_values) / len(elapsed_values)) if elapsed_values else 0.0
    result.setdefault("avg_elapsed_ms", round(avg_elapsed_ms, 3))
    result.setdefault(
        "answer_format_distribution",
        {key: int(value) for key, value in sorted(answer_formats.items())},
    )
    return result


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
    result["correct"] = int(raw.get("correct", raw.get("solved", 0)))
    accuracy = raw.get("accuracy")
    if accuracy is None and "overall_accuracy" in raw:
        accuracy = raw.get("overall_accuracy")
    if accuracy is None:
        accuracy = (result["correct"] / result["total"]) if result["total"] else 0.0
    result["accuracy"] = float(accuracy)
    return _augment_suite_summary(result)


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


def _reset_query_session(kv: Knowledgeverse) -> None:
    if hasattr(kv, "reset_query_session"):
        kv.reset_query_session()


def _build_suite(
    suite_name: str,
    *,
    count: int,
    kv: Knowledgeverse,
    tablet: HeadlessTabletMPC,
    args: argparse.Namespace,
) -> Any:
    if suite_name == "arc2":
        return ARCAGI2Benchmark(
            knowledgeverse=kv,
            dataset_path=args.arc2_dataset_path,
            max_tasks=max(1, int(count)),
            tablet_boundary=tablet,
        )
    if suite_name == "mmlu":
        return MMLUBenchmark(
            knowledgeverse=kv,
            dataset_path=args.mmlu_dataset_path,
            max_questions=max(1, int(count)),
            tablet_boundary=tablet,
        )
    if suite_name == "gsm8k":
        return GSM8KBenchmark(
            knowledgeverse=kv,
            dataset_path=args.gsm8k_dataset_path,
            max_questions=max(1, int(count)),
            tablet_boundary=tablet,
        )
    if suite_name == "lhe":
        return LastHumanityExamBenchmark(
            knowledgeverse=kv,
            dataset_path=args.lhe_dataset_path,
            max_questions=max(1, int(count)),
            tablet_boundary=tablet,
        )
    if suite_name == "imo":
        return IMOBenchmark(
            knowledgeverse=kv,
            dataset_path=args.imo_dataset_path,
            max_questions=max(1, int(count)),
            tablet_boundary=tablet,
        )
    if suite_name == "math":
        return UnifiedMathBenchmark(
            knowledgeverse=kv,
            dataset_path=args.math_dataset_path,
            max_problems=max(1, int(count)),
            source_filter=["math"],
            tablet_boundary=tablet,
            dataset_mode="present",
        )
    if suite_name == "amc_aime":
        return UnifiedMathBenchmark(
            knowledgeverse=kv,
            dataset_path=args.math_dataset_path,
            max_problems=max(1, int(count)),
            source_filter=["amc_aime"],
            tablet_boundary=tablet,
            dataset_mode="present",
        )
    if suite_name == "omni_math":
        return UnifiedMathBenchmark(
            knowledgeverse=kv,
            dataset_path=args.math_dataset_path,
            max_problems=max(1, int(count)),
            source_filter=["omni_math"],
            tablet_boundary=tablet,
            dataset_mode="present",
        )
    raise ValueError(f"unsupported_suite_builder:{suite_name}")


def _run_suite(
    *,
    suite_name: str,
    count: int,
    kv: Knowledgeverse,
    built_suite: Any | None,
    log_dir: Path,
    use_enriched: bool,
) -> dict[str, Any]:
    row_log_path = log_dir / f"{suite_name}.jsonl"
    progress_log_path = log_dir / f"{suite_name}_progress.jsonl"
    row_log_path.write_text("", encoding="utf-8")
    progress_log_path.write_text("", encoding="utf-8")

    def _row_cb(source: dict[str, Any], result: dict[str, Any]) -> None:
        _append_jsonl(row_log_path, _row_log_payload(suite_name, source, result))

    def _progress_cb(progress: dict[str, Any]) -> None:
        payload = _progress_log_payload(suite_name, progress)
        _append_jsonl(progress_log_path, payload)
        completed = int(payload.get("completed", 0))
        total = int(payload.get("total", 0))
        correct = int(payload.get("correct", 0))
        elapsed_s = float(payload.get("elapsed_s", 0.0))
        _log_section(
            f"progress {suite_name}: {completed}/{total} correct={correct} elapsed={elapsed_s:.1f}s"
        )

    assert built_suite is not None
    raw = built_suite.run_benchmark(
        use_enriched=use_enriched,
        row_cb=_row_cb,
        progress_cb=_progress_cb,
        progress_every=1,
    )
    result = _normalize_native_result(suite_name, raw)
    _write_jsonl(row_log_path, list(result.get("results") or []))
    return result


def run_tablet_benchmark_suite(
    args: argparse.Namespace,
    *,
    command_handler: CommandHandler | None = None,
) -> dict[str, Any]:
    storage_root = Path(args.storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir or (storage_root / "logs" / f"headless_tablet_{_ts()}"))
    log_dir.mkdir(parents=True, exist_ok=True)

    hardware_profile = _detect_hardware_profile(storage_root)
    feeder_workers = max(1, min(int(hardware_profile.get("physical_cores", 1) or 1), 8))

    kv = Knowledgeverse(storage_root=storage_root)
    if hasattr(kv, "suspend_auto_sleep"):
        kv.suspend_auto_sleep()
    tablet = HeadlessTabletMPC(
        knowledgeverse=kv,
        storage_root=storage_root,
        command_handler=command_handler,
    )

    suite_order = [
        ("arc2", int(args.arc2_count)),
        ("mmlu", int(args.mmlu_count)),
        ("gsm8k", int(args.gsm8k_count)),
        ("lhe", int(args.lhe_count)),
        ("math", int(args.math_count)),
        ("amc_aime", int(args.amc_aime_count)),
        ("omni_math", int(args.omni_math_count)),
        ("imo", int(args.imo_count)),
    ]
    selected_builders = [(name, count) for name, count in suite_order if count > 0]

    preloaded: dict[str, Any] = {}
    if selected_builders:
        with ThreadPoolExecutor(max_workers=min(feeder_workers, len(selected_builders))) as executor:
            future_map = {
                executor.submit(_build_suite, name, count=count, kv=kv, tablet=tablet, args=args): name
                for name, count in selected_builders
            }
            for future in as_completed(future_map):
                name = future_map[future]
                preloaded[name] = future.result()
                _log_section(f"preloaded {name}")

    archived_suites: dict[str, dict[str, Any]] = {}
    if int(getattr(args, "arc3_count", 0) or 0) > 0:
        archived_suites["arc3_local"] = _archived_suite(
            "arc3_local_archived_use_arc3_sdk_agent_or_headless_tablet_runner",
            int(getattr(args, "arc3_count", 0) or 0),
        )
        _log_section(
            f"archived arc3_local requested_count={archived_suites['arc3_local']['requested_count']}"
        )

    start = time.time()
    all_results: dict[str, dict[str, Any]] = {}
    shutdown_summary: dict[str, Any] = {}
    execution_summary: dict[str, Any] = {}
    try:
        for suite_name, suite_count in suite_order:
            if suite_count <= 0:
                all_results[suite_name] = _skip_summary(f"{suite_name}_count<=0")
                continue
            _log_section(f"starting {suite_name} count={suite_count}")
            result = _run_suite(
                suite_name=suite_name,
                count=suite_count,
                kv=kv,
                built_suite=preloaded.get(suite_name),
                log_dir=log_dir,
                use_enriched=bool(args.use_enriched),
            )
            _log_section(f"completed {suite_name}")
            all_results[suite_name] = result
            partial_summary = {
                "timestamp": log_dir.name,
                "elapsed_seconds": round(time.time() - start, 2),
                "log_dir": str(log_dir),
                "hardware_profile": hardware_profile,
                "orchestrator": {
                    "mode": "single_live_session",
                    "feeder_workers": int(min(feeder_workers, len(selected_builders) or 1)),
                },
                "completed_suites": list(all_results.keys()),
                "suites": {
                    name: {key: value for key, value in suite_result.items() if key != "results"}
                    for name, suite_result in all_results.items()
                },
            }
            if archived_suites:
                partial_summary["archived_suites"] = archived_suites
            (log_dir / "summary.partial.json").write_text(
                json.dumps(partial_summary, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
    finally:
        execution_summary = _write_execution_artifacts(
            log_dir=log_dir,
            start=start,
            hardware_profile=hardware_profile,
            feeder_workers=min(feeder_workers, len(selected_builders) or 1),
            all_results=all_results,
        )
        shutdown_summary = _shutdown_with_timeout(
            kv,
            float(getattr(args, "shutdown_timeout_s", 30.0) or 30.0),
        )
        (log_dir / "sleep_consolidation.json").write_text(
            json.dumps(shutdown_summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    summary = {
        "mode": "headless_tablet_wine_session",
        "log_dir": str(log_dir),
        "storage_root": str(storage_root),
        "use_enriched": bool(args.use_enriched),
        "elapsed_seconds": round(time.time() - start, 2),
        "hardware_profile": hardware_profile,
        "orchestrator": {
            "session_model": "one_live_knowledgeverse",
            "feeder_workers": int(min(feeder_workers, len(selected_builders) or 1)),
            "mid_session_unload_allowed": False,
            "knowledgeverse_boot_count": 1,
        },
        "execution_artifacts": {
            "summary": str(log_dir / "summary.execution.json"),
            "full_results": str(log_dir / "full_results.execution.json"),
        },
        "execution_phase": execution_summary,
        "sleep_consolidation": shutdown_summary,
        "benchmarks": {
            name: {key: value for key, value in result.items() if key != "results"}
            for name, result in all_results.items()
        },
    }
    if archived_suites:
        summary["archived_suites"] = archived_suites
    (log_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (log_dir / "full_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return {"summary": summary, "results": all_results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", default="/K3D/Knowledge3D.local")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--arc2-dataset-path", default=None)
    parser.add_argument("--mmlu-dataset-path", default=None)
    parser.add_argument("--gsm8k-dataset-path", default=None)
    parser.add_argument("--lhe-dataset-path", default=None)
    parser.add_argument("--math-dataset-path", default=None)
    parser.add_argument("--imo-dataset-path", default=None)
    parser.add_argument("--arc2-count", type=int, default=1)
    parser.add_argument("--arc3-count", type=int, default=1)
    parser.add_argument("--mmlu-count", type=int, default=1)
    parser.add_argument("--gsm8k-count", type=int, default=1)
    parser.add_argument("--lhe-count", type=int, default=1)
    parser.add_argument("--math-count", type=int, default=1)
    parser.add_argument("--amc-aime-count", type=int, default=1)
    parser.add_argument("--omni-math-count", type=int, default=1)
    parser.add_argument("--imo-count", type=int, default=1)
    parser.add_argument("--use-enriched", action="store_true", default=False)
    parser.add_argument("--shutdown-timeout-s", type=float, default=30.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    payload = run_tablet_benchmark_suite(args)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
