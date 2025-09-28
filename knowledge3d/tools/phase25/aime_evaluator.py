"""Evaluate Algorithmic Thinking fused head on the AIME 2024 benchmark."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset  # type: ignore
import pandas as _pd  # type: ignore

from knowledge3d.tools.phase18.meaning_cluster_trainer import MeaningClusterTrainer  # type: ignore


_AIME_SPLIT = "train"
_DATASET_NAME = "Maxwell-Jia/AIME_2024"
_OUTPUT_PATH = Path("docs/benchmarks/aime_2024_results.json")
_CLUSTER_HINT = "algorithmic_thinking"


@dataclass
class EvalRecord:
    problem_id: str
    question: str
    expected: int
    prediction: str
    parsed: Optional[int]
    correct: bool

    def to_dict(self) -> dict:
        return {
            "id": self.problem_id,
            "question": self.question,
            "expected": self.expected,
            "prediction": self.prediction,
            "parsed": self.parsed,
            "correct": self.correct,
        }


def _extract_numeric_answer(text: str) -> Optional[int]:
    if not text:
        return None
    # Normalise LaTeX formatting artifacts such as \boxed{033} or $033$.
    boxed = re.findall(r"\\boxed\{\s*([0-9]{1,3})\s*\}", text)
    if boxed:
        try:
            return int(boxed[-1])
        except ValueError:
            pass
    # Replace Latex-style $...$ wrappers
    cleaned = re.sub(r"[$]", " ", text)
    # Replace unicode minus with ASCII to avoid splitting issues
    cleaned = cleaned.replace("−", "-")
    matches = re.findall(r"\b\d{1,3}\b", cleaned)
    for candidate in reversed(matches):
        try:
            value = int(candidate)
        except ValueError:
            continue
        if 0 <= value <= 999:
            return value
    return None


def _load_aime_records(limit: Optional[int] = None) -> List[dict]:
    """Load AIME rows either via HF datasets or a local parquet cache.

    Falls back to viewer/public/galaxy/working/aime_2024_problems.parquet to
    avoid SciPy/Numpy import issues in some environments.
    """
    # Preferred: HuggingFace datasets
    try:
        ds = load_dataset(_DATASET_NAME)[_AIME_SPLIT]
        n = len(ds) if limit is None else min(limit, len(ds))
        return [ds[i] for i in range(n)]
    except Exception:
        pass
    # Fallback: local parquet produced during training
    cache_path = Path("viewer/public/galaxy/working/aime_2024_problems.parquet")
    if not cache_path.exists():
        raise RuntimeError(
            "Unable to load AIME dataset (HF and local parquet both unavailable)."
        )
    df = _pd.read_parquet(cache_path)
    if limit is not None:
        df = df.head(int(limit))
    return df.to_dict("records")


def evaluate_aime(limit: Optional[int] = None) -> List[EvalRecord]:
    dataset = _load_aime_records(limit=limit)
    records: List[EvalRecord] = []
    trainer = MeaningClusterTrainer()
    total = len(dataset)
    for idx, row in enumerate(dataset):
        problem_id = str(row.get("ID") or f"item_{idx}")
        question = str(row.get("Problem") or "")
        answer = int(row.get("Answer"))
        embedding = trainer.generate_multi_modal_embedding(question)
        prediction = trainer.predict_from_fused_embedding(question, embedding, cluster_name=_CLUSTER_HINT)
        parsed = _extract_numeric_answer(prediction)
        correct = parsed == answer
        records.append(
            EvalRecord(
                problem_id=problem_id,
                question=question,
                expected=answer,
                prediction=prediction,
                parsed=parsed,
                correct=correct,
            )
        )
    return records


def write_report(records: List[EvalRecord], output_path: Path = _OUTPUT_PATH) -> None:
    correct = sum(1 for r in records if r.correct)
    total = len(records)
    accuracy = correct / total if total else 0.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": _DATASET_NAME,
        "split": _AIME_SPLIT,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "details": [rec.to_dict() for rec in records],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"AIME evaluation complete: {correct}/{total} correct (accuracy={accuracy:.3f})")
    print(f"→ Report written to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate AIME 2024 benchmark using the fused head only.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of problems (default: all 30)")
    args = parser.parse_args()

    results = evaluate_aime(limit=args.limit)
    write_report(results)
