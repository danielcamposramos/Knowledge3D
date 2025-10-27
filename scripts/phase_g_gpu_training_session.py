#!/usr/bin/env python3
"""
Phase G Sovereign Training Session
==================================

Runs all specialist GPU trainings in-process and triggers sleep-time consolidation
after each run so Galaxy knowledge is persisted to the House before the process exits.

Default workflow (can be overridden via CLI flags):

    1. Multimodal specialist (100 epochs @ 0.002 LR, seed 123)
    2. Sleep-time consolidation
    3. Speech specialist (100 epochs @ 0.002 LR, seed 123)
    4. Sleep-time consolidation
    5. OCR specialist (100 epochs @ 0.002 LR, seed 123)
    6. Sleep-time consolidation
    7. Router specialist (200 epochs @ 0.01 LR, seed 456)
    8. Sleep-time consolidation

Each consolidation step waits for a configurable idle “cool-down” window to ensure
all async logging settles before the RPN consolidator captures Galaxy updates.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from scripts.train_specialist_gpu import run_training

from knowledge3d.cranium.adaptive_swarm import AdaptiveSwarmTRM
from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.rpn_executor import get_rpn_executor
from knowledge3d.cranium.sleep_time_consolidator import SleepTimeConsolidator


# --------------------------------------------------------------------------- #
# Data classes / configuration
# --------------------------------------------------------------------------- #
@dataclass
class SpecialistJob:
    name: str
    dataset: Path
    epochs: int
    learning_rate: float
    seed: int


DEFAULT_PLAN: Sequence[SpecialistJob] = (
    SpecialistJob(
        name="multimodal",
        dataset=Path("/K3D/Knowledge3D.local/datasets/multimodal_embeddings.jsonl"),
        epochs=100,
        learning_rate=0.002,
        seed=123,
    ),
    SpecialistJob(
        name="speech",
        dataset=Path("/K3D/Knowledge3D.local/datasets/speech_embeddings.jsonl"),
        epochs=100,
        learning_rate=0.002,
        seed=123,
    ),
    SpecialistJob(
        name="ocr",
        dataset=Path("/K3D/Knowledge3D.local/datasets/character_embeddings_trimodal.jsonl"),
        epochs=100,
        learning_rate=0.002,
        seed=123,
    ),
    SpecialistJob(
        name="router",
        dataset=Path("/K3D/Knowledge3D.local/checkpoints/phase_g/router_bootstrap_history.json"),
        epochs=200,
        learning_rate=0.01,
        seed=456,
    ),
)


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    """Timestamped console logging."""
    print(f"[{datetime.now().isoformat()}] {msg}")


def wait_with_progress(seconds: int) -> None:
    """Wait for *seconds* while emitting minute-level progress updates."""
    remaining = seconds
    while remaining > 0:
        sleep_chunk = min(60, remaining)
        time.sleep(sleep_chunk)
        remaining -= sleep_chunk
        if remaining > 0:
            log(f"  Cooldown… {remaining} seconds remaining")


def run_sleep_time_consolidation(
    embeddings_path: Path,
    output_path: Path,
    metrics_path: Optional[Path],
    clusters: int,
    learning_rate: float,
    redundancy_threshold: float,
) -> dict:
    """Execute in-process sleep-time consolidation."""
    log("Loading RPN embeddings for sleep consolidation")
    engine = RPNEmbeddingEngine()
    engine.load_embeddings(embeddings_path)

    consolidator = SleepTimeConsolidator(
        engine,
        cluster_count=clusters,
        learning_rate=learning_rate,
        redundancy_threshold=redundancy_threshold,
        metrics_path=None,  # we handle logging ourselves
    )

    log("Running SleepTimeConsolidator.consolidate()")
    result = consolidator.consolidate()
    engine.save_embeddings(output_path)
    log(f"Consolidation result: {json.dumps(result, indent=2, default=float)}")

    if metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now().isoformat(),
            "result": result,
        }
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(payload, default=float) + "\n")
        log(f"Metrics appended to {metrics_path}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase G GPU specialists with mandatory sleep consolidation.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/checkpoints/phase_g"),
        help="Checkpoint directory that holds specialist weights (default: %(default)s)",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=300,
        help="Idle wait time between training and consolidation (default: 300)",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl"),
        help="Path to active RPN embeddings (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/logs/sleep_metrics_phase_g.jsonl"),
        help="Optional JSONL metrics log for consolidation runs",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=256,
        help="Sleep-time consolidation cluster count (default: 256)",
    )
    parser.add_argument(
        "--consolidation-lr",
        type=float,
        default=0.2,
        help="Sleep-time consolidation learning rate (default: 0.2)",
    )
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=0.95,
        help="Redundancy pruning cosine threshold (default: 0.95)",
    )
    parser.add_argument(
        "--specialists",
        nargs="*",
        choices=[job.name for job in DEFAULT_PLAN],
        help="Subset of specialists to train (default: all in canonical order)",
    )
    parser.add_argument(
        "--skip-sleep",
        action="store_true",
        help="DEBUG ONLY: Skip sleep-time consolidation steps (not recommended!)",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=15,
        help="Maximum number of samples processed in parallel during specialist training",
    )
    return parser.parse_args()


def resolve_plan(selected: Optional[Iterable[str]]) -> List[SpecialistJob]:
    if not selected:
        return list(DEFAULT_PLAN)
    selected_set = set(selected)
    return [job for job in DEFAULT_PLAN if job.name in selected_set]


def main() -> None:
    args = parse_args()
    plan = resolve_plan(args.specialists)
    if not plan:
        raise SystemExit("No specialists selected for training session.")

    log("=== Phase G Sovereign Session starting ===")
    log(f"Specialists queued: {', '.join(job.name for job in plan)}")
    get_rpn_executor()  # Prime RPN PTX context

    swarm = AdaptiveSwarmTRM()
    current_checkpoint = args.checkpoint_dir / "current"
    if not current_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {current_checkpoint}")
    swarm.load_checkpoint(current_checkpoint)

    summary = []
    for job in plan:
        log(f"--- Specialist '{job.name}' training started ---")
        stats = run_training(
            specialist=job.name,
            dataset=job.dataset,
            checkpoint_dir=args.checkpoint_dir,
            epochs=job.epochs,
            learning_rate=job.learning_rate,
            shuffle=True,
            seed=job.seed,
            load_checkpoint=None,
            swarm=swarm,
            parallel_workers=args.parallel_workers,
        )
        summary.append({"specialist": job.name, **stats})
        log(f"--- Specialist '{job.name}' training completed ---")

        if args.skip_sleep:
            log("Sleep consolidation skipped by flag (DEBUG).")
            continue

        log(f"Cooldown before consolidation ({args.cooldown_seconds} seconds)")
        wait_with_progress(args.cooldown_seconds)

        log(f"Running sleep-time consolidation after '{job.name}'")
        run_sleep_time_consolidation(
            embeddings_path=args.embeddings_path,
            output_path=args.embeddings_path,
            metrics_path=args.metrics_path,
            clusters=args.clusters,
            learning_rate=args.consolidation_lr,
            redundancy_threshold=args.redundancy_threshold,
        )

    log("=== Phase G Sovereign Session complete ===")
    log("Training summary:")
    for entry in summary:
        # Convert Path objects to strings for JSON serialization
        serializable_entry = {}
        for key, value in entry.items():
            if isinstance(value, Path):
                serializable_entry[key] = str(value)
            else:
                serializable_entry[key] = value
        log(json.dumps(serializable_entry, default=float))


if __name__ == "__main__":
    main()
