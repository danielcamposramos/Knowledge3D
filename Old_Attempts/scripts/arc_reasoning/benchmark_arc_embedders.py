"""
Benchmark ARC grid embedders: procedural vs video vs audio vs multimodal.

Usage:
    PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
        scripts/benchmark_arc_embedders.py
"""

import time
import numpy as np
from typing import List, Dict, Tuple

from knowledge3d.training.arc_agi.grid_processor import ARCGridProcessor
from knowledge3d.training.reasoning.arc_dataset import (
    ensure_arc_dataset,
    _iter_task_files,
    _load_task,
)


def load_sample_grids(n_samples: int = 10) -> List[Tuple[List[List[int]], str]]:
    """
    Load sample grids from ARC dataset.

    Returns:
        List of (grid, task_id) tuples
    """
    dataset_path = ensure_arc_dataset()
    task_files = list(_iter_task_files(dataset_path, split="training"))[:n_samples]

    grids = []
    for task_path in task_files:
        task = _load_task(task_path)
        task_id = task_path.stem

        # Get first training example input grid
        if task.get("train") and len(task["train"]) > 0:
            grid = task["train"][0]["input"]
            grids.append((grid, task_id))

    return grids


def benchmark_embedder(
    processor: ARCGridProcessor,
    grids: List[Tuple[List[List[int]], str]],
    mode_name: str,
    routing: int = 0
) -> Dict:
    """
    Benchmark embedder mode on sample grids.

    Returns:
        {
            "mode": str,
            "avg_latency_ms": float,
            "std_latency_ms": float,
            "embeddings": List[np.ndarray],
            "grid_sizes": List[Tuple[int, int]],
        }
    """
    latencies = []
    embeddings = []
    grid_sizes = []

    print(f"\n{'='*60}")
    print(f"Benchmarking: {mode_name}")
    print(f"{'='*60}")

    for i, (grid, task_id) in enumerate(grids):
        h, w = len(grid), len(grid[0]) if grid else 0
        grid_sizes.append((h, w))

        # Measure latency
        start = time.perf_counter()
        embedding = processor.grid_to_spatial_embedding(grid, routing=routing)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        embeddings.append(embedding)

        print(f"  [{i+1}/{len(grids)}] {task_id}: {h}×{w} grid → {latency_ms:.2f}ms")

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)

    print(f"\n  Average latency: {avg_latency:.2f} ± {std_latency:.2f} ms")
    print(f"  Embedding shape: {embeddings[0].shape}")

    return {
        "mode": mode_name,
        "avg_latency_ms": avg_latency,
        "std_latency_ms": std_latency,
        "embeddings": embeddings,
        "grid_sizes": grid_sizes,
    }


def compare_embedding_similarity(
    results: List[Dict],
    grid_idx: int = 0
) -> None:
    """
    Compare embedding similarity across modes for a single grid.

    Computes pairwise cosine similarity.
    """
    print(f"\n{'='*60}")
    print(f"Embedding Similarity Analysis (Grid {grid_idx})")
    print(f"{'='*60}\n")

    embeddings = {r["mode"]: r["embeddings"][grid_idx] for r in results}

    # Pairwise cosine similarity
    modes = list(embeddings.keys())
    for i, mode1 in enumerate(modes):
        for mode2 in modes[i+1:]:
            emb1 = embeddings[mode1]
            emb2 = embeddings[mode2]

            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
            )

            print(f"  {mode1:20s} <-> {mode2:20s}: {similarity:.4f}")


def analyze_ternary_routing(
    processor_multimodal: ARCGridProcessor,
    grid: List[List[int]]
) -> None:
    """
    Analyze how ternary routing affects embeddings.

    Tests routing = {-1, 0, +1}.
    """
    print(f"\n{'='*60}")
    print(f"Ternary Routing Analysis")
    print(f"{'='*60}\n")

    routings = {
        -1: "Video-heavy (spatial)",
        0: "Balanced",
        +1: "Audio-heavy (temporal)",
    }

    embeddings = {}
    for routing, description in routings.items():
        start = time.perf_counter()
        embedding = processor_multimodal.grid_to_spatial_embedding(grid, routing=routing)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        embeddings[routing] = embedding

        print(f"  Routing {routing:+2d} ({description:25s}): {latency_ms:.2f}ms")

    # Compare routing -1 vs +1
    similarity_neg_pos = np.dot(embeddings[-1], embeddings[+1]) / (
        np.linalg.norm(embeddings[-1]) * np.linalg.norm(embeddings[+1]) + 1e-8
    )
    similarity_neg_balanced = np.dot(embeddings[-1], embeddings[0]) / (
        np.linalg.norm(embeddings[-1]) * np.linalg.norm(embeddings[0]) + 1e-8
    )
    similarity_pos_balanced = np.dot(embeddings[+1], embeddings[0]) / (
        np.linalg.norm(embeddings[+1]) * np.linalg.norm(embeddings[0]) + 1e-8
    )

    print(f"\n  Similarity:")
    print(f"    Video-heavy <-> Audio-heavy: {similarity_neg_pos:.4f}")
    print(f"    Video-heavy <-> Balanced:    {similarity_neg_balanced:.4f}")
    print(f"    Audio-heavy <-> Balanced:    {similarity_pos_balanced:.4f}")


def main():
    """Main benchmarking pipeline."""
    print("="*60)
    print("ARC-AGI Grid Embedder Benchmarking Suite")
    print("="*60)

    # Step 1: Load sample grids
    print("\n[Step 1] Loading sample grids from ARC dataset...")
    grids = load_sample_grids(n_samples=10)
    print(f"✅ Loaded {len(grids)} sample grids")

    # Step 2: Initialize all 4 embedder modes
    print("\n[Step 2] Initializing embedders...")
    processor_procedural = ARCGridProcessor(
        matryoshka_dim=512,
        embedder_type="procedural"
    )
    processor_video = ARCGridProcessor(
        matryoshka_dim=512,
        embedder_type="video"
    )
    processor_audio = ARCGridProcessor(
        matryoshka_dim=512,
        embedder_type="audio"
    )
    processor_multimodal = ARCGridProcessor(
        matryoshka_dim=512,
        embedder_type="multimodal"
    )
    print("✅ All embedders initialized")

    # Step 3: Benchmark each mode
    print("\n[Step 3] Benchmarking embedders...")
    results = []

    # Procedural
    results.append(benchmark_embedder(
        processor_procedural,
        grids,
        mode_name="Procedural (Default)"
    ))

    # Video
    results.append(benchmark_embedder(
        processor_video,
        grids,
        mode_name="Video (DCT Spatial)"
    ))

    # Audio
    results.append(benchmark_embedder(
        processor_audio,
        grids,
        mode_name="Audio (MDCT Temporal)"
    ))

    # Multi-modal (balanced)
    results.append(benchmark_embedder(
        processor_multimodal,
        grids,
        mode_name="Multi-Modal (Balanced)",
        routing=0
    ))

    # Step 4: Compare embeddings
    print("\n[Step 4] Comparing embeddings across modes...")
    compare_embedding_similarity(results, grid_idx=0)

    # Step 5: Analyze ternary routing
    print("\n[Step 5] Analyzing ternary routing...")
    analyze_ternary_routing(processor_multimodal, grids[0][0])

    # Step 6: Summary
    print(f"\n{'='*60}")
    print("Benchmark Summary")
    print(f"{'='*60}\n")

    for result in results:
        print(f"  {result['mode']:30s}: {result['avg_latency_ms']:6.2f} ± {result['std_latency_ms']:5.2f} ms")

    # Check if any mode exceeds 10ms target
    print(f"\n  Target latency: <10ms per grid")
    for result in results:
        if result['avg_latency_ms'] > 10.0:
            print(f"  ⚠️  {result['mode']} exceeds target ({result['avg_latency_ms']:.2f}ms)")
        else:
            print(f"  ✅ {result['mode']} meets target ({result['avg_latency_ms']:.2f}ms)")

    print(f"\n{'='*60}")
    print("✅ Benchmarking complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
