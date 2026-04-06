from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.knowledgeverse.sovereign_text_embedder import embed_text_sovereign


EMBEDDING32_DIMS = 32


def _fnv1a64(text: str) -> int:
    value = 14695981039346656037
    for byte in str(text or "").encode("utf-8"):
        value ^= int(byte)
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(value)


def _embedding32(values: list[float]) -> list[float]:
    row = [float(value) for value in list(values or [])[:EMBEDDING32_DIMS]]
    if len(row) < EMBEDDING32_DIMS:
        row.extend([0.0] * (EMBEDDING32_DIMS - len(row)))
    return row


def _resolve_knowledgeverse(kv_or_storage: Any):
    if kv_or_storage is not None and not isinstance(kv_or_storage, (str, Path)):
        return kv_or_storage
    from knowledge3d.knowledgeverse import Knowledgeverse

    return Knowledgeverse(storage_root=kv_or_storage)


def _normalize_text_answer(value: Any) -> str:
    return "".join(str(value or "").replace("\\", "").lower().split())


def _mmlu_tasks(kv_or_storage: Any, count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from benchmarks.mmlu import MMLUBenchmark

    kv = _resolve_knowledgeverse(kv_or_storage)
    bench = MMLUBenchmark(knowledgeverse=kv, max_questions=max(1, int(count)))
    tasks: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for question in bench.questions[: max(1, int(count))]:
        tasks.append(
            {
                "query_embedding": embed_text_sovereign(question["question_text"]),
                "option_embeddings": [
                    embed_text_sovereign(option)
                    for option in list(question["options"])[:7]
                ],
                "subject": question["subject"],
                "domain_hint": question["subject"],
            }
        )
        options = list(question["options"])
        metadata.append(
            {
                "id": question["id"],
                "suite": "mmlu",
                "mode": "multiple_choice",
                "options": options,
                "correct_index": options.index(question["correct_answer"]),
                "correct_answer": question["correct_answer"],
                "subject": question["subject"],
            }
        )
    return tasks, metadata


def _math_tasks(kv_or_storage: Any, count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from benchmarks.gsm8k import GSM8KBenchmark

    kv = _resolve_knowledgeverse(kv_or_storage)
    bench = GSM8KBenchmark(knowledgeverse=kv, max_questions=max(1, int(count)))
    tasks: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for question in bench.questions[: max(1, int(count))]:
        tasks.append(
            {
                "query_embedding": embed_text_sovereign(question["question_text"]),
                "option_embeddings": [],
                "subject": "gsm8k_math",
                "domain_hint": "word_problem",
            }
        )
        metadata.append(
            {
                "id": question["id"],
                "suite": "gsm8k",
                "mode": "open_ended_hash",
                "correct_answer": str(question["correct_answer"]),
            }
        )
    return tasks, metadata


def _lhe_tasks(kv_or_storage: Any, count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from benchmarks.last_humanity_exam import LastHumanityExamBenchmark

    kv = _resolve_knowledgeverse(kv_or_storage)
    bench = LastHumanityExamBenchmark(knowledgeverse=kv, max_questions=max(1, int(count)))
    tasks: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for question in bench.questions[: max(1, int(count))]:
        options = list(question.get("options") or [])
        tasks.append(
            {
                "query_embedding": embed_text_sovereign(question["question_text"]),
                "option_embeddings": [
                    embed_text_sovereign(option)
                    for option in options[:7]
                ],
                "subject": question["domain"],
                "domain_hint": question["domain"],
            }
        )
        meta: dict[str, Any] = {
            "id": question["id"],
            "suite": "lhe",
            "correct_answer": str(question["correct_answer"]),
            "question_type": str(question.get("question_type") or "multiple_choice").lower(),
            "domain": question["domain"],
        }
        if options:
            meta["mode"] = "multiple_choice"
            meta["options"] = options[:7]
            if str(question["correct_answer"]) in options[:7]:
                meta["correct_index"] = options[:7].index(str(question["correct_answer"]))
            else:
                meta["correct_index"] = 0
        else:
            meta["mode"] = "open_ended_hash"
        metadata.append(meta)
    return tasks, metadata


def _arc2_tasks(
    kv_or_storage: Any,
    count: int,
    *,
    encoder: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from benchmarks.arc_agi_2 import ARCAGI2Benchmark
    from knowledge3d.knowledgeverse.arc3_frame_encoder import ARC3FrameEncoder

    kv = _resolve_knowledgeverse(kv_or_storage)
    bench = ARCAGI2Benchmark(knowledgeverse=kv, max_tasks=max(1, int(count)))
    frame_encoder = encoder or ARC3FrameEncoder()
    tasks: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for task in bench.tasks[: max(1, int(count))]:
        test_pairs = list(task.get("test") or [])
        test_input = list(test_pairs[0].get("input") or []) if test_pairs else []
        expected_output = list(test_pairs[0].get("output") or []) if test_pairs else []
        tasks.append(
            {
                "query_embedding": _embedding32(frame_encoder.encode(test_input)),
                "option_embeddings": [],
                "subject": "arc_agi_2",
                "domain_hint": "visual_reasoning",
            }
        )
        metadata.append(
            {
                "id": task["id"],
                "suite": "arc2",
                "mode": "grid_generation_tbd",
                "input_grid": test_input,
                "expected_grid": expected_output,
            }
        )
    return tasks, metadata


def _dispatch_tasks(
    tasks: list[dict[str, Any]],
    *,
    dispatcher: Any,
    brain: Any | None = None,
    galaxy_table: Any | None = None,
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    if dispatcher is None:
        from knowledge3d.knowledgeverse.gpu_task_dispatch import GPUTaskDispatch

        dispatcher = GPUTaskDispatch()
    from knowledge3d.knowledgeverse.vram_task_buffer import VRAMTaskBuffer

    results: list[dict[str, Any]] = []
    use_single_task = brain is not None
    task_buffer = VRAMTaskBuffer(max_tasks=1 if use_single_task else max(len(tasks), 1))
    try:
        if use_single_task:
            for task in tasks:
                task_buffer.bulk_load([task])
                dispatcher.launch(
                    task_buffer,
                    1,
                    brain_ptr=brain.gpu_ptr,
                    galaxy_ptr=galaxy_table.gpu_ptr if galaxy_table is not None else None,
                    galaxy_star_count=galaxy_table.star_count if galaxy_table is not None else 0,
                )
                results.extend(task_buffer.read_results(1))
        else:
            loaded = task_buffer.bulk_load(tasks)
            dispatcher.launch(
                task_buffer,
                loaded,
                galaxy_ptr=galaxy_table.gpu_ptr if galaxy_table is not None else None,
                galaxy_star_count=galaxy_table.star_count if galaxy_table is not None else 0,
            )
            results = task_buffer.read_results(loaded)
    finally:
        task_buffer.close()
    return results


def _score_rows(
    suite_name: str,
    metadata: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, float]:
    rows: list[dict[str, Any]] = []
    correct = 0
    scorable_total = 0
    for meta, result in zip(metadata, results):
        mode = str(meta.get("mode") or "multiple_choice")
        answer_index = int(result.get("answer_index", 0))
        answer_hash = int(result.get("answer_text_hash", 0))
        row: dict[str, Any] = {
            "suite": suite_name,
            "id": meta.get("id"),
            "mode": mode,
            "answer_index": answer_index,
            "answer_text_hash": answer_hash,
            "confidence": float(result.get("confidence", 0.0)),
            "convergence_signal": int(result.get("convergence_signal", 0)),
            "iterations_used": int(result.get("iterations_used", 0)),
        }

        if mode == "multiple_choice":
            options = list(meta.get("options") or [])
            expected_index = int(meta.get("correct_index", 0))
            answer_text = str(options[answer_index]) if 0 <= answer_index < len(options) else ""
            row.update(
                {
                    "options": options,
                    "correct_index": expected_index,
                    "correct_answer": meta.get("correct_answer"),
                    "answer_text": answer_text,
                }
            )
            row["correct"] = bool(0 <= answer_index < len(options) and answer_index == expected_index)
            scorable_total += 1
        elif mode == "open_ended_hash":
            expected_answer = str(meta.get("correct_answer") or "")
            expected_hash = _fnv1a64(_normalize_text_answer(expected_answer))
            actual_hash = _fnv1a64(_normalize_text_answer(expected_answer)) if answer_hash == expected_hash else answer_hash
            row.update(
                {
                    "correct_answer": expected_answer,
                    "expected_answer_hash": expected_hash,
                    "predicted_answer": None,
                    "predicted_answer_hash": actual_hash,
                }
            )
            row["correct"] = bool(answer_hash == expected_hash and expected_hash != 0)
            scorable_total += 1
        else:
            row.update(
                {
                    "gpu_invoked": True,
                    "correct": False,
                    "scoring_tbd": True,
                    "expected_grid": meta.get("expected_grid"),
                    "input_grid": meta.get("input_grid"),
                }
            )

        correct += int(bool(row["correct"]))
        rows.append(row)
    accuracy = (correct / scorable_total) if scorable_total else 0.0
    return rows, correct, accuracy


def flush_results_to_disk(results: list[dict[str, Any]], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in results))
        handle.write("\n")


def run_gpu_benchmark(
    *,
    suite: str,
    count: int,
    storage_root: str | Path,
    log_path: str | Path | None,
    knowledgeverse: Any | None = None,
    dispatcher: Any | None = None,
    brain: Any | None = None,
    galaxy_table: Any | None = None,
) -> dict[str, Any]:
    suite_name = str(suite).strip().lower()
    task_source = knowledgeverse if knowledgeverse is not None else storage_root

    if suite_name == "mmlu":
        tasks, metadata = _mmlu_tasks(task_source, count)
    elif suite_name == "gsm8k":
        tasks, metadata = _math_tasks(task_source, count)
    elif suite_name == "lhe":
        tasks, metadata = _lhe_tasks(task_source, count)
    elif suite_name == "arc2":
        tasks, metadata = _arc2_tasks(task_source, count)
    else:
        raise ValueError(f"unsupported_gpu_benchmark_suite: {suite}")

    results = _dispatch_tasks(
        tasks,
        dispatcher=dispatcher,
        brain=brain,
        galaxy_table=galaxy_table,
    )
    rows, correct, accuracy = _score_rows(suite_name, metadata, results)

    if log_path is not None:
        flush_results_to_disk(rows, Path(log_path))

    summary: dict[str, Any] = {
        "suite": suite_name,
        "total": len(rows),
        "correct": correct,
        "accuracy": accuracy,
        "results": rows,
    }
    if suite_name == "arc2":
        summary["scoring_tbd"] = True
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Phase E GPU task-dispatch benchmark path.")
    parser.add_argument("--suite", choices=["mmlu", "gsm8k", "lhe", "arc2"], default="mmlu")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--storage-root", default="/K3D/Knowledge3D.local")
    parser.add_argument("--log-path", default="")
    args = parser.parse_args()

    summary = run_gpu_benchmark(
        suite=args.suite,
        count=args.count,
        storage_root=args.storage_root,
        log_path=args.log_path or None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
