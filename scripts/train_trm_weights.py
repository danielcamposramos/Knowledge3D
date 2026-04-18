#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.arc_agi_2 import ARCAGI2Benchmark
from benchmarks.gsm8k import GSM8KBenchmark
from benchmarks.last_humanity_exam import LastHumanityExamBenchmark
from benchmarks.math_competitions import MathCompetitionBenchmark
from benchmarks.mmlu import MMLUBenchmark
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.training.trm_galaxy_nav import (
    apply_trm_weights_to_traces,
    evaluate_decoder_on_traces,
    fit_galaxy_decoder_from_traces,
    initialize_trm_weight_matrices,
    load_trm_weight_checkpoint,
    save_galaxy_decoder_checkpoint,
    save_trm_weight_checkpoint,
    summarize_trace_target_contributions,
    summarize_trace_top1_predictions,
    train_trm_weights_from_traces,
)


def _iter_shadow_traces(payload: Any, *, benchmark_name: str, task_type: str) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        shadow = payload.get("trm_shadow")
        if isinstance(shadow, dict) and shadow.get("query_embedding_512") and shadow.get("y_new_vector_512"):
            trace = dict(shadow)
            trace.setdefault("benchmark", benchmark_name)
            trace.setdefault("task_type", str(payload.get("task_type") or payload.get("query_type") or task_type))
            if payload.get("query_text"):
                trace.setdefault("query_text", str(payload.get("query_text")))
            if payload.get("task_id"):
                trace.setdefault("task_id", str(payload.get("task_id")))
            if isinstance(payload.get("galaxy_contribution"), dict):
                trace.setdefault("galaxy_contribution", dict(payload.get("galaxy_contribution")))
            if isinstance(payload.get("teacher_route_galaxies"), list):
                trace.setdefault(
                    "teacher_route_galaxies",
                    [str(name).strip() for name in payload.get("teacher_route_galaxies", []) if str(name).strip()],
                )
            traces.append(trace)
        for value in payload.values():
            traces.extend(_iter_shadow_traces(value, benchmark_name=benchmark_name, task_type=task_type))
    elif isinstance(payload, list):
        for item in payload:
            traces.extend(_iter_shadow_traces(item, benchmark_name=benchmark_name, task_type=task_type))
    return traces


def _write_traces(path: Path, traces: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=True) + "\n")


def _read_traces(path: Path) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            traces.append(json.loads(line))
    return traces


def collect_shadow_traces(
    *,
    storage_root: Path,
    trace_output: Path,
    trace_navigation_mode: str,
    arc_tasks: int,
    math_problems: int,
    gsm8k_questions: int,
    lhe_questions: int,
    mmlu_questions: int,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    prior_shadow = os.environ.get("K3D_TRM_SHADOW")
    prior_navigate = os.environ.get("K3D_TRM_NAVIGATE")
    os.environ["K3D_TRM_SHADOW"] = "1"
    if trace_navigation_mode == "dormant":
        os.environ["K3D_TRM_NAVIGATE"] = "0"
    elif trace_navigation_mode == "live":
        os.environ["K3D_TRM_NAVIGATE"] = "1"
    kv = Knowledgeverse(storage_root=storage_root)
    active_matryoshka = None
    if getattr(kv, "_trm_matryoshka_host_weights", None) is not None:
        active_matryoshka = np.asarray(kv._trm_matryoshka_host_weights, dtype=np.float32).copy()
    benchmarks = [
        ("ARC", "SPATIAL_TASK", ARCAGI2Benchmark(knowledgeverse=kv, max_tasks=arc_tasks)),
        (
            "MATH",
            "MATH_TASK",
            MathCompetitionBenchmark(knowledgeverse=kv, dataset_path=None, max_problems=math_problems),
        ),
        ("GSM8K", "MATH_TASK", GSM8KBenchmark(knowledgeverse=kv, max_questions=gsm8k_questions)),
        ("LHE", "QUESTION_TASK", LastHumanityExamBenchmark(knowledgeverse=kv, max_questions=lhe_questions)),
        ("MMLU", "QUESTION_TASK", MMLUBenchmark(knowledgeverse=kv, max_questions=mmlu_questions)),
    ]
    traces: list[dict[str, Any]] = []
    for benchmark_name, task_type, benchmark in benchmarks:
        kv.reset_query_session()
        benchmark.run_benchmark(use_enriched=True)
        traces.extend(_iter_shadow_traces(getattr(benchmark, "results", []), benchmark_name=benchmark_name, task_type=task_type))
    _write_traces(trace_output, traces)
    if prior_shadow is None:
        os.environ.pop("K3D_TRM_SHADOW", None)
    else:
        os.environ["K3D_TRM_SHADOW"] = prior_shadow
    if trace_navigation_mode in {"dormant", "live"}:
        if prior_navigate is None:
            os.environ.pop("K3D_TRM_NAVIGATE", None)
        else:
            os.environ["K3D_TRM_NAVIGATE"] = prior_navigate
    return traces, active_matryoshka


def main() -> int:
    parser = argparse.ArgumentParser(description="Train TRM galaxy-navigation decoder from shadow traces.")
    parser.add_argument("--storage-root", type=Path, default=Path("/K3D/Knowledge3D.local"))
    parser.add_argument("--trace-output", type=Path, default=Path("build/trm_traces.jsonl"))
    parser.add_argument("--trace-navigation-mode", choices=("dormant", "live", "inherit"), default="dormant")
    parser.add_argument("--checkpoint-output", type=Path, default=None)
    parser.add_argument("--weights-output", type=Path, default=None)
    parser.add_argument("--skip-collect", action="store_true", help="Reuse an existing trace JSONL file.")
    parser.add_argument("--train-trm", action="store_true", help="Train W1-W4 before refitting the galaxy decoder.")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--raw-signal-weight", type=float, default=1.0)
    parser.add_argument("--arc-drawing-bonus-weight", type=float, default=0.0)
    parser.add_argument("--arc-drawing-margin", type=float, default=0.0)
    parser.add_argument("--target-blend-alpha", type=float, default=0.7)
    parser.add_argument("--arc-tasks", type=int, default=10)
    parser.add_argument("--math-problems", type=int, default=20)
    parser.add_argument("--gsm8k-questions", type=int, default=10)
    parser.add_argument("--lhe-questions", type=int, default=10)
    parser.add_argument("--mmlu-questions", type=int, default=50)
    args = parser.parse_args()

    checkpoint_output = args.checkpoint_output or (args.storage_root / "checkpoints" / "trm_galaxy_nav_weights.npz")
    weights_output = args.weights_output or (args.storage_root / "checkpoints" / "trm_weights.npz")
    active_matryoshka: np.ndarray | None = None
    if args.skip_collect:
        traces = _read_traces(args.trace_output)
    else:
        traces, active_matryoshka = collect_shadow_traces(
            storage_root=args.storage_root,
            trace_output=args.trace_output,
            trace_navigation_mode=str(args.trace_navigation_mode),
            arc_tasks=int(args.arc_tasks),
            math_problems=int(args.math_problems),
            gsm8k_questions=int(args.gsm8k_questions),
            lhe_questions=int(args.lhe_questions),
            mmlu_questions=int(args.mmlu_questions),
        )
    decoder = fit_galaxy_decoder_from_traces(
        traces,
        target_blend_alpha=float(args.target_blend_alpha),
    )
    metrics = evaluate_decoder_on_traces(traces, decoder)
    trm_training_summary = None
    if args.train_trm:
        if weights_output.exists():
            loaded = load_trm_weight_checkpoint(weights_output)
            initial_weights = {name: loaded[name] for name in ("W1", "W2", "W3", "W4")}
            if "matryoshka" in loaded and active_matryoshka is None:
                active_matryoshka = np.asarray(loaded["matryoshka"], dtype=np.float32).copy()
        else:
            initial_weights = initialize_trm_weight_matrices()
        trm_result = train_trm_weights_from_traces(
            traces,
            decoder,
            initial_weights=initial_weights,
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            clip_norm=float(args.clip_norm),
            raw_signal_weight=float(args.raw_signal_weight),
            arc_drawing_bonus_weight=float(args.arc_drawing_bonus_weight),
            arc_drawing_margin=float(args.arc_drawing_margin),
            target_blend_alpha=float(args.target_blend_alpha),
        )
        updated_traces = apply_trm_weights_to_traces(traces, trm_result["weights"])
        aligned_decoder = decoder
        aligned_metrics = evaluate_decoder_on_traces(updated_traces, aligned_decoder)
        aligned_predictions = summarize_trace_top1_predictions(
            updated_traces,
            aligned_decoder,
        )
        refit_decoder = fit_galaxy_decoder_from_traces(
            updated_traces,
            target_blend_alpha=float(args.target_blend_alpha),
        )
        refit_metrics = evaluate_decoder_on_traces(updated_traces, refit_decoder)
        refit_predictions = summarize_trace_top1_predictions(
            updated_traces,
            refit_decoder,
        )
        use_aligned_decoder = (
            aligned_metrics["avg_entropy"] <= refit_metrics["avg_entropy"]
            and aligned_metrics["top1_match_rate"] + 0.1 >= refit_metrics["top1_match_rate"]
        )
        decoder = aligned_decoder if use_aligned_decoder else refit_decoder
        metrics = aligned_metrics if use_aligned_decoder else refit_metrics
        selected_predictions = aligned_predictions if use_aligned_decoder else refit_predictions
        checkpoint_weights = {name: trm_result["weights"][name] for name in ("W1", "W2", "W3", "W4")}
        if active_matryoshka is not None:
            checkpoint_weights["matryoshka"] = np.asarray(active_matryoshka, dtype=np.float32).copy()
        weights_path = save_trm_weight_checkpoint(
            weights_output,
            checkpoint_weights,
            metadata={
                "trace_count": len(traces),
                "epochs": int(args.epochs),
                "learning_rate": float(args.learning_rate),
                "clip_norm": float(args.clip_norm),
                "raw_signal_weight": float(args.raw_signal_weight),
                "arc_drawing_bonus_weight": float(args.arc_drawing_bonus_weight),
                "arc_drawing_margin": float(args.arc_drawing_margin),
                "target_blend_alpha": float(args.target_blend_alpha),
                "metrics_before": trm_result.get("metrics_before", {}),
                "metrics_after": trm_result.get("metrics_after", {}),
                "aligned_decoder_metrics_after": aligned_metrics,
                "refit_decoder_metrics_after": refit_metrics,
                "selected_decoder": "aligned" if use_aligned_decoder else "refit",
                "selected_decoder_prediction_summary": selected_predictions.get("per_benchmark", {}),
                "matryoshka_persisted": active_matryoshka is not None,
            },
        )
        trm_training_summary = {
            "weights_output": str(weights_path),
            "metrics_before": trm_result.get("metrics_before", {}),
            "metrics_after": trm_result.get("metrics_after", {}),
            "galaxy_idf": trm_result.get("galaxy_idf", decoder.get("galaxy_idf", {})),
            "aligned_decoder_metrics_after": aligned_metrics,
            "aligned_decoder_prediction_summary": aligned_predictions.get("per_benchmark", {}),
            "refit_decoder_metrics_after": refit_metrics,
            "refit_decoder_prediction_summary": refit_predictions.get("per_benchmark", {}),
            "selected_decoder": "aligned" if use_aligned_decoder else "refit",
            "decoder_metrics_after": metrics,
            "decoder_prediction_summary": selected_predictions.get("per_benchmark", {}),
            "decoder_prediction_rows": selected_predictions.get("rows", []),
            "final_loss": trm_result.get("final_loss", 0.0),
            "loss_tail": trm_result.get("loss_history", []),
            "raw_signal_weight": trm_result.get("raw_signal_weight", float(args.raw_signal_weight)),
            "arc_drawing_bonus_weight": trm_result.get(
                "arc_drawing_bonus_weight",
                float(args.arc_drawing_bonus_weight),
            ),
            "arc_drawing_margin": trm_result.get("arc_drawing_margin", float(args.arc_drawing_margin)),
            "target_blend_alpha": trm_result.get("target_blend_alpha", float(args.target_blend_alpha)),
            "trace_balance_weight_stats": {
                "min": min(trm_result.get("trace_balance_weights", [0.0])),
                "max": max(trm_result.get("trace_balance_weights", [0.0])),
            },
        }
    checkpoint_path = save_galaxy_decoder_checkpoint(
        checkpoint_output,
        decoder,
        metadata={
            "trace_count": len(traces),
            "metrics_before": decoder.get("metrics_before", {}),
            "metrics_after": metrics,
            "trace_output": str(args.trace_output),
            "train_trm": bool(args.train_trm),
            "target_blend_alpha": float(args.target_blend_alpha),
            **({"trm_training": trm_training_summary} if trm_training_summary is not None else {}),
        },
    )
    summary = {
        "status": "ok",
        "trace_count": len(traces),
        "trace_output": str(args.trace_output),
        "trace_navigation_mode": str(args.trace_navigation_mode),
        "teacher_contribution_summary": summarize_trace_target_contributions(traces),
        "checkpoint_output": str(checkpoint_path),
        "metrics_before": decoder.get("metrics_before", {}),
        "metrics_after": metrics,
        **({"trm_training": trm_training_summary} if trm_training_summary is not None else {}),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
