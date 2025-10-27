#!/usr/bin/env python3
"""
GPU-accelerated specialist training using sovereign PTX kernels.

Example:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
        scripts/train_specialist_gpu.py \
        --specialist ocr \
        --dataset /K3D/Knowledge3D.local/datasets/character_embeddings_trimodal.jsonl \
        --checkpoint-dir /K3D/Knowledge3D.local/checkpoints/phase_g \
        --epochs 100 \
        --learning-rate 0.002
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from knowledge3d.cranium.adaptive_swarm import AdaptiveSwarmTRM
from knowledge3d.cranium.sovereign.lora_gpu_trainer import LoRAGPUEngine


def tile_to_dim(vector: np.ndarray, dim: int) -> np.ndarray:
    """Tile or truncate vector to match target dimension."""
    vec = np.asarray(vector, dtype=np.float32)
    if vec.size == dim:
        return vec
    repeats = dim // vec.size
    remainder = dim % vec.size
    if repeats > 0:
        tiled = np.tile(vec, repeats)
    else:
        tiled = np.empty(0, dtype=np.float32)
    if remainder:
        tiled = np.concatenate([tiled, vec[:remainder]])
    return tiled.astype(np.float32, copy=False)


def pad_or_concat(segments: List[np.ndarray], dim: int) -> np.ndarray:
    """Concatenate segments and pad with zeros to reach dimension."""
    vec = np.concatenate([np.asarray(seg, dtype=np.float32) for seg in segments])
    if vec.size > dim:
        return vec[:dim].copy()
    if vec.size < dim:
        padded = np.zeros(dim, dtype=np.float32)
        padded[: vec.size] = vec
        return padded
    return vec.copy()


def prepare_dataset(
    specialist: str,
    dims: int,
    dataset_path: Path,
    router_keys: Tuple[str, ...] = (),
) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset JSONL and build input/target arrays."""
    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    if specialist == "router":
        with dataset_path.open("r") as handle:
            history = json.load(handle)
        if not router_keys:
            raise ValueError("router_keys required for router dataset preparation")
        key_index = {key: idx for idx, key in enumerate(router_keys)}
        for rec in history:
            input_vec = np.asarray(rec.get("input_data", []), dtype=np.float32)
            if input_vec.size != dims:
                input_vec = tile_to_dim(input_vec, dims)
            target_vec = np.zeros(dims, dtype=np.float32)
            performance = float(rec.get("outcome_performance", 1.0))
            specialist_weights = rec.get("specialist_weights", {}) or {}
            if specialist_weights:
                best_name = max(specialist_weights.items(), key=lambda kv: kv[1])[0]
            else:
                best_name = next(iter(router_keys))
            for key, idx in key_index.items():
                target_vec[idx] = performance if key == best_name else 0.0
            inputs.append(input_vec)
            targets.append(target_vec)
    else:
        with dataset_path.open("r") as handle:
            for line in handle:
                rec = json.loads(line)
                text = np.asarray(rec.get("text_embedding", []), dtype=np.float32)
                image = np.asarray(rec.get("image_embedding", []), dtype=np.float32)
                audio = np.asarray(rec.get("audio_embedding", []), dtype=np.float32)
                fused = np.asarray(rec.get("fused_embedding", []), dtype=np.float32)

                if specialist == "ocr":
                    segments = [text, image]
                    target = tile_to_dim(fused if fused.size else text, dims)
                elif specialist == "speech":
                    segments = [text, audio]
                    target = tile_to_dim(fused if fused.size else audio, dims)
                elif specialist == "multimodal":
                    segments = [text, image, audio]
                    if fused.size:
                        segments.append(fused)
                    target = tile_to_dim(fused if fused.size else segments[0], dims)
                else:
                    raise ValueError(f"Unsupported specialist '{specialist}'")

                inp = pad_or_concat(segments, dims)
                inputs.append(inp)
                targets.append(target)

    if not inputs:
        raise RuntimeError(f"No samples parsed from {dataset_path}")

    inputs_np = np.stack(inputs).astype(np.float32)
    targets_np = np.stack(targets).astype(np.float32)
    return inputs_np, targets_np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU specialist trainer (sovereign PTX)")
    parser.add_argument("--specialist", required=True, choices=["ocr", "speech", "multimodal", "router"])
    parser.add_argument("--dataset", required=True, type=Path, help="JSONL dataset path")
    parser.add_argument("--checkpoint-dir", required=True, type=Path, help="Phase G checkpoint directory")
    parser.add_argument("--load-checkpoint", type=Path, default=None, help="Explicit checkpoint to load (defaults to checkpoint-dir/current)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel-workers", type=int, default=15, help="Max samples to process in parallel (<=15 recommended)")
    return parser.parse_args()


def update_adapter_arrays(adapter, A_new: np.ndarray, B_new: np.ndarray) -> None:
    adapter.A = A_new.astype(np.float32)
    adapter.B = B_new.astype(np.float32)
    adapter.A_shadow = adapter.A.copy()
    adapter.B_shadow = adapter.B.copy()


def run_training(
    specialist: str,
    dataset: Path,
    checkpoint_dir: Path,
    epochs: int = 100,
    learning_rate: float = 0.002,
    shuffle: bool = True,
    seed: int = 42,
    load_checkpoint: Optional[Path] = None,
    swarm: Optional[AdaptiveSwarmTRM] = None,
    parallel_workers: int = 15,
) -> Dict[str, any]:
    checkpoint_dir = checkpoint_dir
    load_path = load_checkpoint or (checkpoint_dir / "current")

    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    if swarm is None:
        swarm = AdaptiveSwarmTRM()
        swarm.load_checkpoint(load_path)
    elif load_checkpoint is not None:
        swarm.load_checkpoint(load_path)

    print("=" * 80)
    print(f"GPU Specialist Training :: {specialist}")
    print("=" * 80)
    print(f"Checkpoint: {load_path}")
    print(f"Dataset: {dataset}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print()

    if specialist not in swarm.base.specialists:
        raise ValueError(f"Specialist '{specialist}' not registered in checkpoint")

    spec_info: Dict[str, any] = swarm.base.specialists[specialist]
    adapter = spec_info["adapter"]
    dims = spec_info["dims"]
    rank = adapter.rank
    alpha = adapter.alpha

    base_matrix = swarm.base.get_base_at_dim(dims).astype(np.float32)
    if specialist == "router":
        router_keys = tuple(k for k in swarm.base.specialists.keys() if k != "router")
        if not router_keys:
            raise RuntimeError("Router training requires non-router specialists")
        inputs_np, targets_np = prepare_dataset(specialist, dims, dataset, router_keys)
    else:
        inputs_np, targets_np = prepare_dataset(specialist, dims, dataset)
    print(f"Samples: {inputs_np.shape[0]}   Dimension: {dims}   Rank: {rank}")
    print()

    engine = LoRAGPUEngine()
    batch_cap = max(1, min(15, parallel_workers, inputs_np.shape[0]))
    buffers = engine.allocate_buffers(
        base_matrix,
        adapter.A,
        adapter.B,
        inputs_np,
        targets_np,
        max_batch=batch_cap,
    )

    rng = np.random.default_rng(seed)
    dataset_size = inputs_np.shape[0]

    losses: List[float] = []
    try:
        for epoch in range(1, epochs + 1):
            if shuffle:
                perm = rng.permutation(dataset_size)
                epoch_inputs = inputs_np[perm]
                epoch_targets = targets_np[perm]
            else:
                epoch_inputs = inputs_np
                epoch_targets = targets_np

            engine.update_dataset(buffers, epoch_inputs, epoch_targets)

            epoch_loss = 0.0
            batches = 0
            order = np.arange(dataset_size, dtype=np.int32)
            for batch_start in range(0, dataset_size, batch_cap):
                batch_end = min(batch_start + batch_cap, dataset_size)
                batch_indices = order[batch_start:batch_end]
                loss = engine.train_batch(
                    buffers=buffers,
                    batch_indices=batch_indices,
                    dims=dims,
                    rank=rank,
                    alpha=alpha,
                    learning_rate=learning_rate,
                )
                epoch_loss += loss
                batches += 1

            avg_loss = epoch_loss / float(batches or 1)
            losses.append(avg_loss)
            print(f"Epoch {epoch:03d}/{epochs} - loss={avg_loss:.6f}")

        A_host, B_host = engine.fetch_weights(buffers, dims, rank)
    finally:
        engine.free_buffers(buffers)

    update_adapter_arrays(adapter, A_host, B_host)

    # Save checkpoint
    output_dir = checkpoint_dir / f"{specialist}_gpu_epoch_{epochs}"
    output_dir.mkdir(parents=True, exist_ok=True)
    swarm.save_checkpoint(output_dir)
    print(f"Checkpoint saved: {output_dir}")

    current_dir = checkpoint_dir / "current"
    if current_dir.exists():
        shutil.rmtree(current_dir)
    shutil.copytree(output_dir, current_dir)
    print(f"Updated current checkpoint -> {current_dir}")

    return {
        "output_dir": output_dir,
        "epochs": epochs,
        "losses": losses,
        "learning_rate": learning_rate,
        "samples": inputs_np.shape[0],
    }


def main() -> None:
    args = parse_args()
    run_training(
        specialist=args.specialist,
        dataset=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        shuffle=args.shuffle,
        seed=args.seed,
        load_checkpoint=args.load_checkpoint,
        parallel_workers=args.parallel_workers,
    )


if __name__ == "__main__":
    main()
