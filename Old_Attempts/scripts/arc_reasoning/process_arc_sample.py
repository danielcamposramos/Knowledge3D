"""Process sample ARC tasks and analyze patterns."""

import numpy as np
from pathlib import Path

from knowledge3d.training.arc_agi.grid_processor import ARCGridProcessor
from knowledge3d.training.reasoning.arc_dataset import (
    ensure_arc_dataset,
    _iter_task_files,
    _load_task,
)


def process_sample_tasks(n_tasks: int = 50):
    """Process sample ARC tasks with multi-modal embedder."""

    # Initialize processor (multi-modal is BEST)
    processor = ARCGridProcessor(matryoshka_dim=512, embedder_type="multimodal")

    # Ensure dataset
    dataset_path = ensure_arc_dataset()
    task_files = list(_iter_task_files(dataset_path, split="training"))[:n_tasks]

    print(f"Processing {len(task_files)} ARC tasks...")

    results = []

    for i, task_path in enumerate(task_files):
        task = _load_task(task_path)
        task_id = task_path.stem

        task_result = {
            "task_id": task_id,
            "n_train": len(task.get("train", [])),
            "n_test": len(task.get("test", [])),
            "grid_sizes": [],
            "embeddings": [],
        }

        # Process training examples
        for example in task.get("train", []):
            input_grid = example["input"]
            h, w = len(input_grid), len(input_grid[0]) if input_grid else 0

            # Embed with balanced routing
            embedding = processor.grid_to_spatial_embedding(input_grid, routing=0)

            task_result["grid_sizes"].append((h, w))
            task_result["embeddings"].append(embedding)

        results.append(task_result)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(task_files)} tasks...")

    # Analyze
    print("\n📊 Analysis:")

    all_sizes = [size for task in results for size in task["grid_sizes"]]
    heights = [h for h, w in all_sizes]
    widths = [w for h, w in all_sizes]

    print(f"  Total grids processed: {len(all_sizes)}")
    print("  Grid size range:")
    print(f"    Height: {min(heights)} - {max(heights)} (avg: {np.mean(heights):.1f})")
    print(f"    Width:  {min(widths)} - {max(widths)} (avg: {np.mean(widths):.1f})")

    # Embedding stats
    all_embeddings = np.array([emb for task in results for emb in task["embeddings"]])
    print("\n  Embedding statistics:")
    print(f"    Shape: {all_embeddings.shape}")
    print(f"    Mean: {np.mean(all_embeddings):.4f}")
    print(f"    Std:  {np.std(all_embeddings):.4f}")
    print(f"    Min:  {np.min(all_embeddings):.4f}")
    print(f"    Max:  {np.max(all_embeddings):.4f}")

    # Save embeddings
    output_dir = Path("/K3D/Knowledge3D.local/datasets/arc_agi_embeddings/")
    output_dir.mkdir(parents=True, exist_ok=True)

    for task_result in results:
        task_id = task_result["task_id"]
        embeddings = np.array(task_result["embeddings"], dtype=np.float32)

        output_path = output_dir / f"{task_id}.npz"
        np.savez_compressed(
            output_path,
            task_id=task_id,
            embeddings=embeddings,
            grid_sizes=task_result["grid_sizes"],
        )

    print(f"\n✅ Saved {len(results)} task embeddings to {output_dir}")

    return results


if __name__ == "__main__":
    process_sample_tasks(n_tasks=50)
    print("\n✅ Sample processing complete!")
