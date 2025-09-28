"""GPU-only trainer for the fused head's internal math head (0..999).

Loads math-style problems from local Hugging Face cache (no external answers),
extracts short integer answers (<= 999), generates fused embeddings, and
updates the fused head via ``train_step``. We persist checkpoints to
``viewer/public/house/house_math_head.pt`` as defined in the fused head.

Usage:
  PYTHONPATH=. python -m knowledge3d.tools.phase25.math_head_trainer \
      --epochs 10 --limit 5000 --shuffle 1

Notes:
  - Prefers local HF cache (no downloads). Will skip missing datasets.
  - Accepts AIME parquet cache if present for quick bootstrapping.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import pandas as pd  # type: ignore

try:  # Optional: use local HF cache only
    from datasets import load_dataset, DownloadConfig  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore
    DownloadConfig = None  # type: ignore

from knowledge3d.tools.phase18.meaning_cluster_trainer import (  # type: ignore
    MeaningClusterTrainer,
)
from knowledge3d.cranium.phase10.rpn_calculator import RPNCalculator  # type: ignore


WORKING_DIR = Path("viewer/public/galaxy/working")
AIME_PARQUET = WORKING_DIR / "aime_2024_problems.parquet"


def _extract_int_0_999(text: str) -> Optional[int]:
    if not isinstance(text, str):
        return None
    # Prefer LaTeX boxed answers
    m = re.findall(r"\\boxed\{\s*([0-9]{1,3})\s*\}", text)
    if m:
        try:
            v = int(m[-1])
            return v if 0 <= v <= 999 else None
        except Exception:
            pass
    # Remove $...$ wrappers and unicode minus
    cleaned = re.sub(r"[$]", " ", text or "").replace("−", "-")
    # Keep last 1-3 digit token
    tokens = re.findall(r"\b\d{1,3}\b", cleaned)
    if not tokens:
        return None
    try:
        v = int(tokens[-1])
    except Exception:
        return None
    return v if 0 <= v <= 999 else None


@dataclass
class QA:
    question: str
    answer: int  # 0..999 only


def _iter_aime_parquet(limit: Optional[int]) -> Iterator[QA]:
    if not AIME_PARQUET.exists():
        return iter(())
    df = pd.read_parquet(AIME_PARQUET)
    if limit is not None:
        df = df.head(int(limit))
    for _, row in df.iterrows():
        q = str(row.get("Problem", ""))
        ans = row.get("Answer")
        try:
            a = int(ans)
        except Exception:
            continue
        if 0 <= a <= 999 and q:
            yield QA(question=q, answer=a)


def _iter_hf_dataset(repo: str, split: str, q_key: str, a_key: str, limit: Optional[int]) -> Iterator[QA]:
    if load_dataset is None or DownloadConfig is None:
        return iter(())
    try:
        ds = load_dataset(repo, split=split, download_config=DownloadConfig(local_files_only=True))
    except Exception:
        return iter(())
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        row = ds[i]
        q = str(row.get(q_key, ""))
        a_raw = row.get(a_key)
        a = None
        if isinstance(a_raw, (int, float)):
            a = int(a_raw)
        else:
            a = _extract_int_0_999(str(a_raw or ""))
        if a is None or not q:
            continue
        if 0 <= a <= 999:
            yield QA(question=q, answer=a)


def _gather_math_qas(limit: Optional[int], shuffle: bool = True) -> List[QA]:
    items: List[QA] = []
    # 1) AIME parquet (already curated for numeric answers)
    items.extend(list(_iter_aime_parquet(limit)))
    # 2) MetaMathQA (if cached locally). Often has numeric short answers embedded in solutions.
    items.extend(list(_iter_hf_dataset("meta-math/MetaMathQA", "train", "problem", "answer", limit)))
    # 3) GSM8K (answers often numeric, but may exceed 999 — filtered by extractor)
    items.extend(list(_iter_hf_dataset("openai/gsm8k", "main", "question", "answer", limit)))
    # 4) Competition math (MATH). Answers sometimes boxed; extractor handles.
    items.extend(list(_iter_hf_dataset("hendrycks/competition_math", "train", "problem", "solution", limit)))

    # Deduplicate by (question, answer)
    seen = set()
    unique: List[QA] = []
    for qa in items:
        key = (qa.question.strip(), int(qa.answer))
        if key in seen:
            continue
        seen.add(key)
        unique.append(qa)
    if shuffle:
        random.shuffle(unique)
    return unique


def _generate_synthetic_rpn(count: int = 5000, seed: int = 42) -> List[QA]:
    """Generate synthetic RPN arithmetic questions with exact numeric answers.

    Ensures answers fall in 0..999. Uses only +,-,*,^ (limited) and avoids division
    to keep integers stable. Delegates numeric ground truth to the PTX RPN engine
    when available; falls back to Python eval if necessary.
    """
    rng = random.Random(seed)
    ops = ["+", "-", "*"]
    rpn = RPNCalculator()
    out: List[QA] = []
    def _rand_num() -> int:
        return rng.randint(0, 99)
    for _ in range(int(max(0, count))):
        # Build an RPN with 3–6 numbers
        n_terms = rng.randint(3, 6)
        stack = [str(_rand_num()) for _ in range(2)]
        tokens: List[str] = stack[:]
        for _k in range(n_terms - 2):
            tokens.append(str(_rand_num()))
            op = rng.choice(ops)
            tokens.append(op)
        expr = " ".join(tokens)
        try:
            val = rpn.evaluate(expr)
        except Exception:
            # Very rare; skip malformed
            continue
        v = int(round(val))
        if 0 <= v <= 999:
            q = f"Evaluate the RPN expression '{expr}'."
            out.append(QA(question=q, answer=v))
    return out


def train_math_head(epochs: int = 5, limit: Optional[int] = 5000, shuffle: bool = True) -> None:
    trainer = MeaningClusterTrainer()
    fh = trainer.fused_head
    print(f"🚀 Training math head on device: {fh.device}")
    qas = _gather_math_qas(limit=limit, shuffle=shuffle)
    # Add synthetic RPN supervision to anchor internal precision
    synth = _generate_synthetic_rpn(count=5000)
    qas.extend(synth)
    if shuffle:
        random.shuffle(qas)
    if not qas:
        print("⚠️  No math QA items found in local cache. Aborting.")
        return
    print(f"📚 Training set size: {len(qas)} items (0..999 answers)")

    for epoch in range(1, int(max(1, epochs)) + 1):
        correct = 0
        total = 0
        for qa in qas:
            emb = trainer.generate_multi_modal_embedding(qa.question)
            # Train step on ground-truth
            fh.train_step(emb, str(int(qa.answer)))
            # Quick prediction check
            pred = fh._predict_math_numeric(emb)
            total += 1
            if pred == int(qa.answer):
                correct += 1
        acc = correct / total if total else 0.0
        print(f"🧮 Epoch {epoch}: acc={acc:.3f} ({correct}/{total})")
        fh._save_math_head()
    print("✅ Math head training complete; checkpoint saved.")


def main() -> None:  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Train fused head math classifier (0..999)")
    ap.add_argument("--epochs", type=int, default=5, help="Epochs over the assembled dataset")
    ap.add_argument("--limit", type=int, default=5000, help="Max items per dataset source")
    ap.add_argument("--shuffle", type=int, default=1, help="Shuffle examples (1=yes,0=no)")
    args = ap.parse_args()
    train_math_head(epochs=args.epochs, limit=args.limit, shuffle=bool(args.shuffle))


if __name__ == "__main__":  # pragma: no cover
    main()
