from __future__ import annotations

import json

from benchmarks.math_competitions import UnifiedMathBenchmark
from benchmarks.sampling import stratified_sample
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def test_stratified_covers_all_thirds() -> None:
    items = list(range(300))
    sample = stratified_sample(items, 9)
    assert len(sample) == 9
    easy = [x for x in sample if x < 100]
    mid = [x for x in sample if 100 <= x < 200]
    hard = [x for x in sample if x >= 200]
    assert len(easy) == 3
    assert len(mid) == 3
    assert len(hard) == 3


def test_stratified_none_limit_returns_all() -> None:
    items = list(range(100))
    assert stratified_sample(items, None) == items


def test_stratified_limit_exceeds_size() -> None:
    items = list(range(10))
    assert stratified_sample(items, 50) == items


def test_stratified_deterministic() -> None:
    items = list(range(300))
    a = stratified_sample(items, 15)
    b = stratified_sample(items, 15)
    assert a == b


def test_stratified_small_limit_hits_each_third() -> None:
    items = list(range(300))
    sample = stratified_sample(items, 3)
    assert len(sample) == 3
    assert sample[0] < 100
    assert 100 <= sample[1] < 200
    assert sample[2] >= 200


def test_unified_math_benchmark_stratifies_present_dataset(tmp_path) -> None:
    dataset_dir = tmp_path / "datasets"
    (dataset_dir / "data").mkdir(parents=True, exist_ok=True)
    train_path = dataset_dir / "data" / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for idx in range(9):
            handle.write(
                json.dumps(
                    {
                        "problem": f"Problem {idx}",
                        "solution": rf"The answer is \boxed{{{idx}}}.",
                        "type": "Prealgebra" if idx < 3 else ("Geometry" if idx < 6 else "Precalculus"),
                    }
                )
                + "\n"
            )

    bench = UnifiedMathBenchmark(
        knowledgeverse=Knowledgeverse(storage_root=tmp_path / "kv"),
        dataset_path=dataset_dir,
        dataset_mode="present",
        max_problems=3,
        source_filter=["math"],
    )

    assert [problem["id"] for problem in bench.problems] == ["math_0", "math_3", "math_6"]
