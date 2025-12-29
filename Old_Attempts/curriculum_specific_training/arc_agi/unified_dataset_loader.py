"""
Unified ARC-AGI dataset loader — combines ARC-AGI 1 and ARC-AGI 2.

Alternates between datasets for balanced exposure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Dataset paths
ARC_AGI_1_PATH = Path("/K3D/K3D_llama_cpp/datasets/ARC-AGI-master/data")
ARC_AGI_2_PATH = Path("/K3D/Knowledge3D.local/datasets/exams/arc-src/data")
# ARC-AGI 3 optional path (Part 7)
ARC_AGI_3_PATH = Path("/K3D/K3D_llama_cpp/datasets/ARC-AGI-3-Agents/data")


class UnifiedDatasetLoader:
    """
    Loads and alternates between ARC-AGI 1 and ARC-AGI 2 datasets.
    """

    def __init__(self, arc1_path: Optional[Path] = None, arc2_path: Optional[Path] = None, split: str = "training"):
        self.arc1_path = arc1_path or ARC_AGI_1_PATH
        self.arc2_path = arc2_path or ARC_AGI_2_PATH
        self.split = split

        self.arc1_tasks = self._load_task_list(self.arc1_path / split)
        self.arc2_tasks = self._load_task_list(self.arc2_path / split)

        self.arc1_tasks = self._sort_by_difficulty(self.arc1_tasks)
        self.arc2_tasks = self._sort_by_difficulty(self.arc2_tasks)

        print(f"[DATASET] ARC-AGI 1: {len(self.arc1_tasks)} tasks from {self.arc1_path}")
        print(f"[DATASET] ARC-AGI 2: {len(self.arc2_tasks)} tasks from {self.arc2_path}")
        print(f"[DATASET] Total: {len(self.arc1_tasks) + len(self.arc2_tasks)} tasks")

    def _load_task_list(self, path: Path) -> List[Tuple[str, Path]]:
        """Load list of (task_id, file_path) from directory."""
        if not path.exists():
            print(f"[WARNING] Dataset path not found: {path}")
            return []

        tasks: List[Tuple[str, Path]] = []
        for f in sorted(path.glob("*.json")):
            tasks.append((f.stem, f))
        return tasks

    def _sort_by_difficulty(self, tasks: List[Tuple[str, Path]]) -> List[Tuple[str, Path]]:
        """
        Sort tasks by estimated difficulty using simple heuristics.
        """
        scored_tasks: List[Tuple[float, str, Path]] = []
        for task_id, path in tasks:
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                train = data.get("train", [])
                n_examples = len(train)
                avg_input_size = 0
                avg_output_size = 0
                total_colors = set()
                for ex in train:
                    inp = ex.get("input", [[]])
                    out = ex.get("output", [[]])
                    avg_input_size += len(inp) * len(inp[0]) if inp else 0
                    avg_output_size += len(out) * len(out[0]) if out else 0
                    for row in inp:
                        total_colors.update(row)
                    for row in out:
                        total_colors.update(row)
                if n_examples > 0:
                    avg_input_size /= n_examples
                    avg_output_size /= n_examples
                difficulty = (1 / (n_examples + 1)) * 10 + avg_input_size / 100 + avg_output_size / 100 + len(total_colors) * 0.5
                scored_tasks.append((difficulty, task_id, path))
            except Exception:
                scored_tasks.append((50.0, task_id, path))

        scored_tasks.sort(key=lambda x: x[0])
        return [(task_id, path) for _, task_id, path in scored_tasks]

    def __len__(self) -> int:
        return len(self.arc1_tasks) + len(self.arc2_tasks)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate tasks with alternation: ARC-1, ARC-2, ARC-1, ARC-2...
        """
        max_len = max(len(self.arc1_tasks), len(self.arc2_tasks))
        for i in range(max_len):
            if i < len(self.arc1_tasks):
                task_id, path = self.arc1_tasks[i]
                yield self._load_task(task_id, path, "arc_agi_1")
            if i < len(self.arc2_tasks):
                task_id, path = self.arc2_tasks[i]
                yield self._load_task(task_id, path, "arc_agi_2")

    def _load_task(self, task_id: str, path: Path, source: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            data = json.load(f)
        return {
            "task_id": task_id,
            "source": source,
            "path": str(path),
            "train": data.get("train", []),
            "test": data.get("test", []),
        }

    def get_arc1_tasks(self) -> List[Dict[str, Any]]:
        return [self._load_task(task_id, path, "arc_agi_1") for task_id, path in self.arc1_tasks]

    def get_arc2_tasks(self) -> List[Dict[str, Any]]:
        return [self._load_task(task_id, path, "arc_agi_2") for task_id, path in self.arc2_tasks]

    def get_interleaved_batches(self, batch_size: int = 2) -> Iterator[List[Dict[str, Any]]]:
        """
        Get batches with 1 ARC-1 + 1 ARC-2 task per batch.
        """
        if batch_size % 2 != 0:
            batch_size += 1
            print(f"[WARNING] batch_size adjusted to {batch_size} for even alternation")

        half = batch_size // 2
        max_pairs = min(len(self.arc1_tasks), len(self.arc2_tasks))

        for i in range(0, max_pairs, half):
            batch: List[Dict[str, Any]] = []
            for j in range(half):
                if i + j < len(self.arc1_tasks):
                    task_id, path = self.arc1_tasks[i + j]
                    batch.append(self._load_task(task_id, path, "arc_agi_1"))
            for j in range(half):
                if i + j < len(self.arc2_tasks):
                    task_id, path = self.arc2_tasks[i + j]
                    batch.append(self._load_task(task_id, path, "arc_agi_2"))
            if batch:
                yield batch

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "arc_agi_1": {"path": str(self.arc1_path), "task_count": len(self.arc1_tasks)},
            "arc_agi_2": {"path": str(self.arc2_path), "task_count": len(self.arc2_tasks)},
            "total_tasks": len(self),
            "split": self.split,
        }


__all__ = ["UnifiedDatasetLoader", "ARC_AGI_1_PATH", "ARC_AGI_2_PATH"]
 
# Extended loader including ARC-AGI 3 (optional)


class UnifiedDatasetLoaderV2(UnifiedDatasetLoader):
    """
    Extended loader supporting ARC-AGI 1, 2, and optionally 3.
    """

    def __init__(
        self,
        arc1_path: Optional[Path] = None,
        arc2_path: Optional[Path] = None,
        arc3_path: Optional[Path] = None,
        split: str = "training",
        include_arc3: bool = True,
    ):
        super().__init__(arc1_path=arc1_path, arc2_path=arc2_path, split=split)
        self.arc3_path = arc3_path or ARC_AGI_3_PATH
        self.arc3_tasks: List[Tuple[str, Path]] = []
        if include_arc3 and self.arc3_path.exists():
            self.arc3_tasks = self._sort_by_difficulty(self._load_task_list(self.arc3_path / split))
            print(f"[DATASET] ARC-AGI 3: {len(self.arc3_tasks)} tasks from {self.arc3_path}")
        print(f"[DATASET] Grand total: {len(self)} tasks")

    def __len__(self) -> int:
        return len(self.arc1_tasks) + len(self.arc2_tasks) + len(self.arc3_tasks)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        max_len = max(len(self.arc1_tasks), len(self.arc2_tasks), len(self.arc3_tasks))
        for i in range(max_len):
            if i < len(self.arc1_tasks):
                task_id, path = self.arc1_tasks[i]
                yield self._load_task(task_id, path, "arc_agi_1")
            if i < len(self.arc2_tasks):
                task_id, path = self.arc2_tasks[i]
                yield self._load_task(task_id, path, "arc_agi_2")
            if i < len(self.arc3_tasks):
                task_id, path = self.arc3_tasks[i]
                yield self._load_task(task_id, path, "arc_agi_3")

    def get_arc3_tasks(self) -> List[Dict[str, Any]]:
        return [self._load_task(task_id, path, "arc_agi_3") for task_id, path in self.arc3_tasks]

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats["arc_agi_3"] = {"path": str(self.arc3_path), "task_count": len(self.arc3_tasks)}
        stats["total_tasks"] = len(self)
        return stats


__all__ += ["UnifiedDatasetLoaderV2", "ARC_AGI_3_PATH"]
