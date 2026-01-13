from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from knowledge3d.training.reasoning.arc_dataset import (
    prepare_arc_reasoning_cache,
    load_arc_reasoning_cache,
)


def _make_arc_task(path: Path, samples: list[tuple[list[list[int]], list[list[int]]]]) -> None:
    payload = {
        "train": [{"input": inp, "output": out} for inp, out in samples],
        "test": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _dummy_embed(text: str) -> np.ndarray:
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.normal(size=128).astype(np.float32)


def test_prepare_and_load_arc_cache(tmp_path):
    dataset_root = tmp_path / "arc_dataset"
    training_dir = dataset_root / "ARC-AGI-master" / "data" / "training"
    training_dir.mkdir(parents=True)

    _make_arc_task(
        training_dir / "task1.json",
        samples=[
            ([[1, 2], [3, 4]], [[0, 0], [1, 1]]),
            ([[5, 6], [7, 8]], [[2, 2], [3, 3]]),
        ],
    )
    _make_arc_task(
        training_dir / "task2.json",
        samples=[
            ([[0, 1, 0]], [[1, 1, 1]]),
        ],
    )

    cache_path = dataset_root / "cache.npz"
    built_cache = prepare_arc_reasoning_cache(
        _dummy_embed,
        dataset_root=dataset_root,
        cache_path=cache_path,
        limit=2,
        rebuild=True,
        download=False,
    )

    assert built_cache == cache_path
    cache = load_arc_reasoning_cache(cache_path)

    # Limit=2 should keep only the first two samples.
    assert cache.questions.shape == (2, 512)
    assert cache.answers.shape == (2, 512)
    assert cache.example_indices.tolist() == [0, 1]
    assert list(cache.task_ids[:2]) == ["task1", "task1"]
    assert cache.metadata["pairs"] == 2
