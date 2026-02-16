#!/usr/bin/env python3
"""Run deterministic foundation curriculum iterations against a persistent Knowledgeverse."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.arc_agi_2 import ARCAGI2Benchmark
from benchmarks.deterministic_foundation import DeterministicFoundationBenchmark
from knowledge3d.augmentation.ollama_curriculum_augmenter import OllamaAugmenter
from knowledge3d.knowledgeverse.foundational_operations_bootstrap import (
    populate_foundational_operations,
)
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.sleeptime import SleepTimeError
from knowledge3d.knowledgeverse.ternary_quality_memory import TernaryQualityMemory
from knowledge3d.training.rlwhf.teacher_student_bridge import (
    RLWHFTeacherStudentBridge,
    pool_hamming_drift,
)


def _collect_galaxy_counts(kv: Knowledgeverse) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in ("Drawing", "Grammar", "Math", "Reality", "3DObjects", "Audio"):
        try:
            out[name] = len(kv.galaxy_manager.get_galaxy(name).entries)
        except Exception:
            out[name] = 0
    return out


def _history_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {}
    initial = history[0]["results"]["overall"]["accuracy"]
    final = history[-1]["results"]["overall"]["accuracy"]
    return {
        "iterations": len(history),
        "initial_accuracy": initial,
        "final_accuracy": final,
        "delta": final - initial,
        "status": "SUCCESS" if final >= 0.80 else "IN_PROGRESS",
    }


def _stage_transfer_thresholds() -> dict[str, float]:
    return {
        # Stage A is body-control bootstrap. Keep transfer gate at baseline so
        # progression is not blocked before generation-focused stages.
        "A": 0.20,
        "B": 0.30,
        "C": 0.40,
        "D": 0.50,
    }


def _stage_oracle_thresholds() -> dict[str, float]:
    return {
        "A": 0.00,
        "B": 0.00,
        "C": 0.30,
        "D": 0.50,
    }


def _stage_generated_floors() -> dict[str, int]:
    return {
        "A": 0,
        "B": 0,
        "C": 25,
        "D": 50,
    }


def _run_transfer_probe(
    *,
    kv: Knowledgeverse,
    max_arc_tasks: int,
    enable_contrastive_learning: bool = False,
) -> dict[str, Any]:
    """
    Run a small ARC transfer probe to prevent stage-gate leakage.

    If dataset is not available, this still returns a stable payload.
    """
    try:
        probe = ARCAGI2Benchmark(
            knowledgeverse=kv,
            max_tasks=max(1, int(max_arc_tasks)),
            enable_contrastive_learning=enable_contrastive_learning,
        )
        result = probe.run_benchmark(use_enriched=True)
    except Exception as exc:
        return {
            "available": False,
            "accuracy": 0.0,
            "oracle_at_all": 0.0,
            "generated_pattern_total": 0,
            "error": str(exc),
        }
    diagnostics = result.get("oracle_diagnostics", {})
    return {
        "available": True,
        "accuracy": float(result.get("accuracy", 0.0)),
        "oracle_at_all": float(diagnostics.get("oracle_at_all", 0.0)),
        "generated_pattern_total": int(result.get("generated_pattern_total", 0)),
        "task_count": int(result.get("total_tasks", 0)),
    }


def train_deterministic_foundation(
    *,
    iterations: int = 10,
    tasks_per_category: int = 100,
    seed: int = 2026,
    storage_root: str | Path = "../Knowledge3D.local",
    output_dir: str | Path = "../Knowledge3D.local/results/foundation_training",
    enable_transfer_gates: bool = False,
    transfer_probe_arc_tasks: int = 10,
    enable_ternary_quality: bool = False,
    enable_ollama_augmentation: bool = True,
    ollama_vision_model: str = "llava",
    ollama_language_model: str = "llama3.2",
    ollama_multimodal_model: str = "llava",
    include_system_literacy: bool = False,
    enable_contrastive_learning: bool = False,
) -> dict[str, Any]:
    """Train deterministic foundation via repeated benchmark+consolidation loops."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    kv = Knowledgeverse(storage_root=storage_root)
    bootstrap_summary = populate_foundational_operations(kv.galaxy_manager)
    teacher_bridge = RLWHFTeacherStudentBridge()
    if not enable_ollama_augmentation:
        print(
            "[foundation] WARNING: Ollama augmentation disabled via explicit override. "
            "This bypasses the standard enrichment path and should be used only for diagnostics."
        )
    ollama_augmenter = OllamaAugmenter(
        enabled=enable_ollama_augmentation,
        vision_model=ollama_vision_model,
        language_model=ollama_language_model,
        multimodal_model=ollama_multimodal_model,
    )
    quality_memory = None
    if enable_ternary_quality:
        quality_state = Path(kv.storage_root) / "checkpoints" / "curriculum_quality_memory.json"
        quality_memory = TernaryQualityMemory(state_path=quality_state)

    stage_sequence = ("A", "B", "C", "D")
    stage_idx = 0
    stage_thresholds = {"A": 0.95, "B": 0.85, "C": 0.75, "D": 0.65}
    transfer_thresholds = _stage_transfer_thresholds()
    oracle_thresholds = _stage_oracle_thresholds()
    generated_floors = _stage_generated_floors()
    gate_window = 3
    recent_scores: dict[str, deque[float]] = {s: deque(maxlen=gate_window) for s in stage_sequence}
    benchmark = DeterministicFoundationBenchmark(
        tasks_per_category=tasks_per_category,
        seed=seed,
        stage=stage_sequence[stage_idx],
        include_system_literacy=include_system_literacy,
        ollama_augmenter=ollama_augmenter if enable_ollama_augmentation else None,
    )

    history: list[dict[str, Any]] = []
    previous_pool_id: str | None = None
    for iteration in range(max(1, int(iterations))):
        result = benchmark.run_benchmark(kv, iteration=iteration, log_events=True)
        transfer_probe = (
            _run_transfer_probe(
                kv=kv,
                max_arc_tasks=transfer_probe_arc_tasks,
                enable_contrastive_learning=enable_contrastive_learning,
            )
            if enable_transfer_gates
            else {
                "available": False,
                "accuracy": 0.0,
                "oracle_at_all": 0.0,
                "generated_pattern_total": 0,
            }
        )

        sleep_result: dict[str, Any]
        try:
            sleep_result = kv.sleeptime.execute()
        except SleepTimeError as exc:
            sleep_result = {"success": False, "error": str(exc)}

        payload = {
            "iteration": iteration + 1,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "results": result,
            "stage": benchmark.stage,
            "sleeptime": sleep_result,
            "specialist_count": kv.trm_navigator.count_specialists(),
            "galaxy_counts": _collect_galaxy_counts(kv),
            "shadow_copy_events": len(kv.shadow_copy.event_buffer),
            "transfer_probe": transfer_probe,
        }

        current_stage = benchmark.stage
        overall = float(result.get("overall", {}).get("accuracy", 0.0))
        recent_scores[current_stage].append(overall)

        feedback = teacher_bridge.evaluate_iteration(
            stage=current_stage,
            iteration=iteration + 1,
            train_accuracy=overall,
            transfer_accuracy=float(transfer_probe.get("accuracy", 0.0)),
            oracle_at_all=float(transfer_probe.get("oracle_at_all", 0.0)),
            generated_pattern_total=int(transfer_probe.get("generated_pattern_total", 0)),
            expected_generation_floor=generated_floors.get(current_stage, 0) or 1,
        )
        teacher_bridge.persist_feedback(kv, feedback)
        payload["teacher_feedback"] = feedback.to_event_data()
        payload["pool_id"] = feedback.pool_id
        pool_drift = pool_hamming_drift(previous_pool_id, feedback.pool_id)
        payload["pool_drift"] = {
            "previous_pool_id": previous_pool_id,
            "current_pool_id": feedback.pool_id,
            "hamming_drift": pool_drift,
        }
        previous_pool_id = feedback.pool_id

        if enable_contrastive_learning:
            contrastive_feedback = teacher_bridge.evaluate_iteration_contrastive(
                forward_feedback=feedback
            )
            teacher_bridge.persist_contrastive_feedback(kv, contrastive_feedback)
            payload["contrastive_feedback"] = contrastive_feedback
        else:
            payload["contrastive_feedback"] = None

        if quality_memory is not None:
            stage_pattern_id = f"stage_{current_stage}_iter_{iteration + 1}"
            threshold = stage_thresholds.get(current_stage, 0.65)
            if overall >= threshold:
                outcome = 1
            elif overall >= max(0.0, threshold - 0.10):
                outcome = 0
            else:
                outcome = -1
            quality_record = quality_memory.update(
                pattern_id=stage_pattern_id,
                outcome=outcome,
                confidence=overall,
                transfer_signal=float(transfer_probe.get("accuracy", 0.0)),
                knowledgeverse=kv,
                specialist="grammar",
                galaxy="Grammar",
                source="train_deterministic_foundation",
            )
            payload["quality_memory"] = quality_record.as_dict() if quality_record else None
        else:
            payload["quality_memory"] = None

        # Progression gate: promote only after consecutive window satisfies thresholds.
        advanced = False
        if current_stage in stage_thresholds and current_stage != stage_sequence[-1]:
            values = list(recent_scores[current_stage])
            train_gate_ok = len(values) == gate_window and all(
                v >= stage_thresholds[current_stage] for v in values
            )
            transfer_gate_ok = True
            if enable_transfer_gates:
                transfer_gate_ok = (
                    float(transfer_probe.get("accuracy", 0.0)) >= transfer_thresholds.get(current_stage, 0.0)
                    and float(transfer_probe.get("oracle_at_all", 0.0))
                    >= oracle_thresholds.get(current_stage, 0.0)
                    and int(transfer_probe.get("generated_pattern_total", 0))
                    >= generated_floors.get(current_stage, 0)
                )

            if train_gate_ok and transfer_gate_ok:
                stage_idx = min(stage_idx + 1, len(stage_sequence) - 1)
                next_stage = stage_sequence[stage_idx]
                benchmark = DeterministicFoundationBenchmark(
                    tasks_per_category=tasks_per_category,
                    seed=seed + (iteration + 1) * 13,
                    stage=next_stage,
                    include_system_literacy=include_system_literacy,
                    ollama_augmenter=ollama_augmenter if enable_ollama_augmentation else None,
                )
                payload["stage_advanced"] = {
                    "from": current_stage,
                    "to": next_stage,
                    "trigger_scores": values,
                    "threshold": stage_thresholds[current_stage],
                    "window": gate_window,
                    "transfer_gate_enabled": enable_transfer_gates,
                    "transfer_gate_ok": transfer_gate_ok,
                    "transfer_threshold": transfer_thresholds.get(current_stage, 0.0),
                    "oracle_threshold": oracle_thresholds.get(current_stage, 0.0),
                    "generated_floor": generated_floors.get(current_stage, 0),
                }
                advanced = True
        if not advanced:
            payload["stage_advanced"] = None

        history.append(payload)
        (output_path / f"iteration_{iteration + 1:03d}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        kv.trm_navigator.save_weights()

    final_payload = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "storage_root": str(Path(storage_root)),
        "bootstrap": bootstrap_summary,
        "summary": _history_summary(history),
        "stage_thresholds": stage_thresholds,
        "transfer_thresholds": transfer_thresholds,
        "oracle_thresholds": oracle_thresholds,
        "generated_floors": generated_floors,
        "transfer_gates_enabled": enable_transfer_gates,
        "ternary_quality_enabled": enable_ternary_quality,
        "ollama_augmentation_enabled": enable_ollama_augmentation,
        "contrastive_learning_enabled": enable_contrastive_learning,
        "gate_window": gate_window,
        "final_stage": benchmark.stage if history else "A",
        "history": history,
    }
    (output_path / "training_history.json").write_text(
        json.dumps(final_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return final_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--tasks-per-category", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--storage-root", default="../Knowledge3D.local")
    parser.add_argument("--output-dir", default="../Knowledge3D.local/results/foundation_training")
    parser.add_argument("--enable-transfer-gates", action="store_true")
    parser.add_argument("--transfer-probe-arc-tasks", type=int, default=10)
    parser.add_argument("--enable-ternary-quality", action="store_true")
    parser.set_defaults(enable_ollama_augmentation=True)
    parser.add_argument(
        "--enable-ollama-augmentation",
        dest="enable_ollama_augmentation",
        action="store_true",
        help="Enable Ollama augmentation (default: enabled).",
    )
    parser.add_argument(
        "--disable-ollama-augmentation",
        dest="enable_ollama_augmentation",
        action="store_false",
        help="EMERGENCY ONLY: disable Ollama augmentation for diagnostics.",
    )
    parser.add_argument("--ollama-vision-model", default="llava")
    parser.add_argument("--ollama-language-model", default="llama3.2")
    parser.add_argument("--ollama-multimodal-model", default="llava")
    parser.add_argument("--include-system-literacy", action="store_true")
    parser.add_argument("--enable-contrastive-learning", action="store_true")
    args = parser.parse_args()

    payload = train_deterministic_foundation(
        iterations=args.iterations,
        tasks_per_category=args.tasks_per_category,
        seed=args.seed,
        storage_root=args.storage_root,
        output_dir=args.output_dir,
        enable_transfer_gates=args.enable_transfer_gates,
        transfer_probe_arc_tasks=args.transfer_probe_arc_tasks,
        enable_ternary_quality=args.enable_ternary_quality,
        enable_ollama_augmentation=args.enable_ollama_augmentation,
        ollama_vision_model=args.ollama_vision_model,
        ollama_language_model=args.ollama_language_model,
        ollama_multimodal_model=args.ollama_multimodal_model,
        include_system_literacy=args.include_system_literacy,
        enable_contrastive_learning=args.enable_contrastive_learning,
    )
    summary = payload.get("summary", {})
    print(
        "foundation_training",
        f"iterations={summary.get('iterations', 0)}",
        f"initial={summary.get('initial_accuracy', 0.0):.3f}",
        f"final={summary.get('final_accuracy', 0.0):.3f}",
        f"delta={summary.get('delta', 0.0):+.3f}",
        f"status={summary.get('status', 'unknown')}",
    )


if __name__ == "__main__":
    main()
