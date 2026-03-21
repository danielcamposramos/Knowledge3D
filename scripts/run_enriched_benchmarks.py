#!/usr/bin/env python3
"""Run enriched-galaxy benchmarks and sleep-time consolidation in one session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Any
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.math_competitions import UnifiedMathBenchmark  # noqa: E402
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse  # noqa: E402
from knowledge3d.knowledgeverse.sleeptime import SleepTimeConsolidation  # noqa: E402
from knowledge3d.tools.benchmark_health_check import load_questions, run_health_check  # noqa: E402
from knowledge3d.tools.ollama_benchmark import create_ollama_query_fn  # noqa: E402
from scripts.ingest_meaning_layer import DEFAULT_COUNTS, DEFAULT_STORAGE_ROOT, ingest_enriched_galaxy  # noqa: E402
from scripts.ingest_math_rules import ingest_math_rules  # noqa: E402


TOKEN_RE = re.compile(r"[a-z0-9_]+")
FULL_BENCHMARK_COUNTS = {
    "arc": 120,
    "math": 500,
    "gsm8k": 1319,
    "lhe": 100,
    "mmlu": 14042,
}


def _question_specialist(suite: str) -> str:
    canonical = str(suite or "").strip().lower()
    if canonical in {"gsm8k", "math"}:
        return "math"
    if canonical == "mmlu":
        return "language"
    if canonical == "arc":
        return "visual"
    return "any"


def _format_match(match: dict[str, Any]) -> str:
    entry = match.get("entry") if isinstance(match.get("entry"), dict) else {}
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    star_id = str(metadata.get("meaning_star_id") or entry.get("id") or "").strip()
    content = str(entry.get("summary") or entry.get("content") or entry.get("rpn_program") or "").strip()
    if not star_id and not content:
        return ""
    return f"{star_id}: {content}".strip(": ")


def create_enriched_ollama_query_fn(knowledgeverse: Knowledgeverse, timeout: float = 120.0):
    base_query = create_ollama_query_fn(timeout=timeout)

    def _query(row: dict[str, Any]) -> dict[str, Any]:
        enriched_row = dict(row)
        question = str(enriched_row.get("question") or "").strip()
        specialist = _question_specialist(str(enriched_row.get("suite") or ""))
        matches = knowledgeverse.galaxy_manager.query(question, specialist=specialist, top_k=5)
        snippets = [_format_match(match) for match in matches if isinstance(match, dict)]
        snippets = [snippet for snippet in snippets if snippet]
        if snippets:
            enriched_row["question"] = (
                "=== GALAXY MATCHES ===\n"
                + "\n".join(f"- {snippet}" for snippet in snippets)
                + "\n\n=== QUESTION ===\n"
                + question
            )
        result = base_query(enriched_row)
        result["galaxy_matches"] = snippets
        return result

    return _query


def _effective_suite_counts(full: bool) -> dict[str, int]:
    return dict(FULL_BENCHMARK_COUNTS if full else DEFAULT_COUNTS)


def _apply_suite_overrides(base_counts: dict[str, int], args: argparse.Namespace) -> dict[str, int]:
    counts = dict(base_counts)
    override_map = {
        "arc": getattr(args, "arc_max", None),
        "math": getattr(args, "math_max", None),
        "gsm8k": getattr(args, "gsm8k_max", None),
        "lhe": getattr(args, "lhe_max", None),
        "mmlu": getattr(args, "mmlu_max", None),
    }
    for suite, override in override_map.items():
        if override is None:
            continue
        counts[suite] = max(0, int(override))
    return counts


def _run_state_path(log_path: Path, *, full: bool) -> Path:
    suffix = "full" if full else "sample"
    return log_path.with_name(f"{log_path.stem}.{suffix}.run_state.json")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _load_or_create_run_state(
    path: Path,
    *,
    suite_counts: dict[str, int],
    full: bool,
    provider: str,
) -> dict[str, Any]:
    current = _load_json(path)
    if (
        isinstance(current, dict)
        and not bool(current.get("completed"))
        and dict(current.get("suite_counts") or {}) == dict(suite_counts)
        and str(current.get("provider") or "") == str(provider)
        and bool(current.get("full")) is bool(full)
    ):
        return current
    state = {
        "session_id": f"{'full' if full else 'sample'}-{uuid4().hex[:12]}",
        "session_start": time.time(),
        "full": bool(full),
        "provider": str(provider),
        "suite_counts": dict(suite_counts),
        "completed": False,
        "completed_summaries": {},
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    _write_json(path, state)
    return state


def _iter_session_rows(
    log_path: Path,
    *,
    session_id: str,
    session_start: float,
    suite: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not log_path.exists():
        return rows
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if suite and str(row.get("suite") or "") != suite:
                continue
            row_session = str(row.get("session_id") or "").strip()
            if row_session:
                if row_session != session_id:
                    continue
            else:
                if float(row.get("timestamp") or 0.0) < float(session_start):
                    continue
            rows.append(row)
    return rows


def _suite_already_done(log_path: Path, suite: str, expected_count: int, *, session_id: str, session_start: float) -> bool:
    return len(_iter_session_rows(log_path, session_id=session_id, session_start=session_start, suite=suite)) >= int(expected_count)


def _suite_summary_from_rows(suite: str, rows: list[dict[str, Any]], *, elapsed_s: float | None = None) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if bool(row.get("correct")))
    computed_elapsed = sum(float(row.get("elapsed_s") or 0.0) for row in rows)
    elapsed_value = float(elapsed_s) if elapsed_s is not None else computed_elapsed
    per_question_s = float(elapsed_value) / total if total else 0.0
    return {
        "suite": suite,
        "total": total,
        "correct": correct,
        "score": f"{correct}/{total}",
        "pct": round((100.0 * correct / total), 2) if total else 0.0,
        "elapsed_s": round(float(elapsed_value), 3),
        "per_question_s": round(per_question_s, 4),
    }


def _combined_summary(summaries: list[dict[str, Any]], *, elapsed_s: float) -> dict[str, Any]:
    total = sum(int(summary.get("total") or 0) for summary in summaries)
    correct = sum(int(summary.get("correct") or 0) for summary in summaries)
    return {
        "total": total,
        "correct": correct,
        "score": f"{correct}/{total}",
        "pct": round((100.0 * correct / total), 2) if total else 0.0,
        "elapsed_s": round(float(elapsed_s), 3),
    }


def _mmlu_subject_breakdown(log_path: Path, *, session_id: str, session_start: float, top_n: int = 10) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, int]] = {}
    for row in _iter_session_rows(log_path, session_id=session_id, session_start=session_start, suite="mmlu"):
        subject = str(row.get("subject") or "unknown").strip().lower()
        bucket = buckets.setdefault(subject, {"subject": subject, "correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += 1 if bool(row.get("correct")) else 0
    ranked = sorted(
        buckets.values(),
        key=lambda item: (-int(item["total"]), -int(item["correct"]), str(item["subject"])),
    )
    result: list[dict[str, Any]] = []
    for item in ranked[:top_n]:
        total = int(item["total"])
        correct = int(item["correct"])
        result.append(
            {
                "subject": item["subject"],
                "correct": correct,
                "total": total,
                "pct": round((100.0 * correct / total), 2) if total else 0.0,
            }
        )
    return result


def _print_suite_progress(summary: dict[str, Any], *, running_correct: int, running_total: int) -> None:
    print(f"\n{'=' * 60}")
    print(f"COMPLETED: {summary['suite']} {summary['correct']}/{summary['total']} ({summary['pct']:.2f}%)")
    print(f"Running total: {running_correct}/{running_total}")
    print(f"Elapsed: {summary['elapsed_s']:.1f}s ({summary['per_question_s']:.3f}s/q)")
    if summary.get("skipped_as_resumed"):
        print("Resumed from existing session log.")
    print(f"{'=' * 60}\n")


def _make_live_progress_logger(suite: str, total: int):
    def _log(snapshot: dict[str, Any]) -> None:
        completed = int(snapshot.get("completed") or 0)
        correct = int(snapshot.get("correct") or 0)
        elapsed_s = float(snapshot.get("elapsed_s") or 0.0)
        pct = (100.0 * completed / total) if total else 0.0
        acc = (100.0 * correct / completed) if completed else 0.0
        marker = str(snapshot.get("subject") or snapshot.get("domain") or "").strip()
        suffix = f" current={marker}" if marker else ""
        print(
            f"[{suite}] {completed}/{total} ({pct:.2f}%) correct={correct} "
            f"running_acc={acc:.2f}% elapsed={elapsed_s:.1f}s{suffix}",
            flush=True,
        )

    return _log


def _make_unified_math_progress_logger(math_total: int, gsm8k_total: int):
    combined_total = int(math_total) + int(gsm8k_total)

    def _log(snapshot: dict[str, Any]) -> None:
        completed = int(snapshot.get("completed") or 0)
        correct = int(snapshot.get("correct") or 0)
        elapsed_s = float(snapshot.get("elapsed_s") or 0.0)
        pct = (100.0 * completed / combined_total) if combined_total else 0.0
        acc = (100.0 * correct / completed) if completed else 0.0
        source_completed = snapshot.get("source_completed") or {}
        source_correct = snapshot.get("source_correct") or {}
        math_done = int(source_completed.get("math") or 0)
        gsm_done = int(source_completed.get("gsm8k") or 0)
        math_correct = int(source_correct.get("math") or 0)
        gsm_correct = int(source_correct.get("gsm8k") or 0)
        current_source = str(snapshot.get("current_source") or "").strip()
        suffix = f" current={current_source}" if current_source else ""
        print(
            f"[unified_math] {completed}/{combined_total} ({pct:.2f}%) correct={correct} "
            f"running_acc={acc:.2f}% math={math_correct}/{math_done or math_total} "
            f"gsm8k={gsm_correct}/{gsm_done or gsm8k_total} elapsed={elapsed_s:.1f}s{suffix}",
            flush=True,
        )

    return _log


def _append_rows(log_path: Path, rows: list[dict[str, Any]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _run_unified_math_pair(
    knowledgeverse: Knowledgeverse,
    *,
    log_path: Path,
    session_id: str,
    math_count: int,
    gsm8k_count: int,
) -> dict[str, dict[str, Any]]:
    bench = UnifiedMathBenchmark(
        knowledgeverse=knowledgeverse,
        max_problems=math_count,
        max_gsm8k_questions=gsm8k_count,
        source_filter=["math", "gsm8k"],
    )
    start = time.monotonic()
    benchmark_summary = bench.run_benchmark(
        use_enriched=True,
        progress_cb=_make_unified_math_progress_logger(math_count, gsm8k_count),
        progress_every=10,
    )
    elapsed = time.monotonic() - start
    rows_by_suite: dict[str, list[dict[str, Any]]] = {"math": [], "gsm8k": []}
    ordered_rows: list[dict[str, Any]] = []
    now = time.time()
    for source, row in zip(bench.problems, bench.results):
        suite = str(source.get("suite") or row.get("suite") or "math").strip().lower()
        payload = {
            "question_id": row["problem_id"],
            "suite": suite,
            "question": source["problem_text"],
            "answer": row.get("predicted_answer"),
            "expected": source["answer"],
            "correct": bool(row.get("correct", False)),
            "elapsed_s": 0.0,
            "timestamp": now,
            "competition": source.get("competition"),
            "source_key": source.get("source_key"),
            "session_id": session_id,
        }
        ordered_rows.append(payload)
        rows_by_suite.setdefault(suite, []).append(payload)
    _append_rows(log_path, ordered_rows)
    summaries: dict[str, dict[str, Any]] = {}
    total_rows = max(1, len(ordered_rows))
    for suite, rows in rows_by_suite.items():
        suite_elapsed = elapsed * (len(rows) / total_rows)
        summary = _suite_summary_from_rows(suite, rows, elapsed_s=suite_elapsed)
        summary["shared_unified_run"] = True
        summary["source_breakdown"] = {
            key: {
                "total": int(value.get("total") or 0),
                "correct": int(value.get("correct") or 0),
                "accuracy": float(value.get("accuracy") or 0.0),
            }
            for key, value in (benchmark_summary.get("results_by_source") or {}).items()
        }
        summaries[suite] = summary
    return summaries


def run_benchmarks(
    knowledgeverse: Knowledgeverse,
    *,
    log_path: Path,
    provider: str = "sovereign",
    timeout: float = 120.0,
    suite_counts: dict[str, int] | None = None,
    full: bool = False,
) -> dict[str, Any]:
    counts = dict(suite_counts or _effective_suite_counts(full))
    log_path = Path(log_path)
    state_path = _run_state_path(log_path, full=full)
    run_state = _load_or_create_run_state(state_path, suite_counts=counts, full=full, provider=provider)
    session_id = str(run_state["session_id"])
    session_start = float(run_state["session_start"])
    query_fn = None
    if provider == "ollama":
        query_fn = create_enriched_ollama_query_fn(knowledgeverse, timeout=timeout)

    summaries: list[dict[str, Any]] = []
    unified_math_cache: dict[str, dict[str, Any]] | None = None
    overall_start = time.monotonic()
    for suite, count in counts.items():
        suite_start = time.monotonic()
        print(f"Starting {suite} benchmark ({count} questions)...", flush=True)
        existing = dict((run_state.get("completed_summaries") or {}).get(suite) or {})
        math_pair_enabled = (
            query_fn is None
            and "math" in counts
            and "gsm8k" in counts
            and suite in {"math", "gsm8k"}
        )
        if math_pair_enabled:
            math_done = _suite_already_done(log_path, "math", counts["math"], session_id=session_id, session_start=session_start)
            gsm8k_done = _suite_already_done(log_path, "gsm8k", counts["gsm8k"], session_id=session_id, session_start=session_start)
            if unified_math_cache is None and not math_done and not gsm8k_done:
                unified_math_cache = _run_unified_math_pair(
                    knowledgeverse,
                    log_path=log_path,
                    session_id=session_id,
                    math_count=counts["math"],
                    gsm8k_count=counts["gsm8k"],
                )
            if unified_math_cache is not None:
                summary = dict(unified_math_cache[suite])
                summary["skipped_as_resumed"] = False
            elif _suite_already_done(log_path, suite, count, session_id=session_id, session_start=session_start):
                rows = _iter_session_rows(log_path, session_id=session_id, session_start=session_start, suite=suite)
                summary = _suite_summary_from_rows(suite, rows, elapsed_s=existing.get("elapsed_s"))
                summary["skipped_as_resumed"] = True
            else:
                run_health_check(
                    suite,
                    count,
                    log_path,
                    query_fn=query_fn,
                    knowledgeverse=knowledgeverse,
                    session_id=session_id,
                    progress_fn=_make_live_progress_logger(suite, count),
                )
                elapsed = time.monotonic() - suite_start
                rows = _iter_session_rows(log_path, session_id=session_id, session_start=session_start, suite=suite)
                summary = _suite_summary_from_rows(suite, rows, elapsed_s=elapsed)
                summary["skipped_as_resumed"] = False
        elif _suite_already_done(log_path, suite, count, session_id=session_id, session_start=session_start):
            rows = _iter_session_rows(log_path, session_id=session_id, session_start=session_start, suite=suite)
            summary = _suite_summary_from_rows(suite, rows, elapsed_s=existing.get("elapsed_s"))
            summary["skipped_as_resumed"] = True
        else:
            run_health_check(
                suite,
                count,
                log_path,
                query_fn=query_fn,
                knowledgeverse=knowledgeverse,
                session_id=session_id,
                progress_fn=_make_live_progress_logger(suite, count),
            )
            elapsed = time.monotonic() - suite_start
            rows = _iter_session_rows(log_path, session_id=session_id, session_start=session_start, suite=suite)
            summary = _suite_summary_from_rows(suite, rows, elapsed_s=elapsed)
            summary["skipped_as_resumed"] = False
        summary["path_used"] = provider
        summaries.append(summary)
        completed_summaries = dict(run_state.get("completed_summaries") or {})
        completed_summaries[suite] = summary
        run_state["completed_summaries"] = completed_summaries
        run_state["updated_at"] = time.time()
        _write_json(state_path, run_state)
        running_correct = sum(int(item.get("correct") or 0) for item in summaries)
        running_total = sum(int(item.get("total") or 0) for item in summaries)
        _print_suite_progress(summary, running_correct=running_correct, running_total=running_total)
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    overall_elapsed = time.monotonic() - overall_start
    run_state["completed"] = True
    run_state["completed_at"] = time.time()
    run_state["updated_at"] = time.time()
    _write_json(state_path, run_state)
    return {
        "session_id": session_id,
        "session_start": session_start,
        "state_path": str(state_path),
        "suites": summaries,
        "combined": _combined_summary(summaries, elapsed_s=overall_elapsed),
        "mmlu_breakdown": _mmlu_subject_breakdown(log_path, session_id=session_id, session_start=session_start),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--log", type=Path, default=DEFAULT_STORAGE_ROOT / "logs" / "health_log.jsonl")
    parser.add_argument("--journal", type=Path, default=DEFAULT_STORAGE_ROOT / "logs" / "sleeptime_journal.jsonl")
    parser.add_argument("--provider", choices=["sovereign", "ollama"], default="sovereign")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--full-load", action="store_true")
    parser.add_argument("--min-languages", type=int, default=5)
    parser.add_argument("--arc-max", type=int, default=None)
    parser.add_argument("--math-max", type=int, default=None)
    parser.add_argument("--gsm8k-max", type=int, default=None)
    parser.add_argument("--lhe-max", type=int, default=None)
    parser.add_argument("--mmlu-max", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    suite_counts = _apply_suite_overrides(_effective_suite_counts(bool(args.full)), args)
    print("Booting Knowledgeverse...", flush=True)
    knowledgeverse = Knowledgeverse(storage_root=args.storage_root)
    print("Knowledgeverse ready.", flush=True)
    ingest_summary = ingest_enriched_galaxy(
        knowledgeverse,
        benchmark_counts=suite_counts,
        progress=lambda message: print(message, flush=True),
    )
    print(
        f"Meaning layer: {ingest_summary['meaning_stars_loaded']} stars loaded "
        f"from {ingest_summary['meaning_stars_available']} available",
        flush=True,
    )
    math_rules_summary = ingest_math_rules(knowledgeverse, progress=lambda message: print(message, flush=True))
    ingest_summary["math_rules"] = math_rules_summary
    print(f"Math rules: {math_rules_summary['total_entries']} entries ingested", flush=True)
    benchmark_run = run_benchmarks(
        knowledgeverse,
        log_path=args.log,
        provider=args.provider,
        timeout=float(args.timeout),
        suite_counts=suite_counts,
        full=bool(args.full),
    )
    print("Starting sleep-time consolidation...", flush=True)
    sleeptime = SleepTimeConsolidation(
        knowledgeverse=knowledgeverse,
        journal_path=args.journal,
        health_log_path=args.log,
        consume_health_log=False,
    )
    sleeptime_result = sleeptime.execute()
    print("Sleep-time consolidation complete.", flush=True)
    final_summary = {
        "path_used": args.provider,
        "session_id": benchmark_run["session_id"],
        "session_start": benchmark_run["session_start"],
        "state_path": benchmark_run["state_path"],
        "galaxy_state": ingest_summary,
        "benchmarks": benchmark_run["suites"],
        "combined": benchmark_run["combined"],
        "mmlu_breakdown": benchmark_run["mmlu_breakdown"],
        "sleeptime": sleeptime_result,
    }
    print(json.dumps(final_summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
