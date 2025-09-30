"""Train the in-core RPN Policy Head (tiny GRU) to generate RPN sequences.

Data sources:
- Primary: viewer/public/galaxy/working/rpn_corpus.jsonl (from rpn_corpus_builder)
- Fallback: synthetic arithmetic RPN sequences

Usage (GPU env):
  conda run -n k3d-cranium env PYTHONPATH=. python -m knowledge3d.tools.phase25.rpn_policy_trainer \
      --epochs 2 --limit 2000 --lr 1e-3 --save-every 500
"""
from __future__ import annotations

import argparse
import os
import json
import random
from pathlib import Path
from typing import Iterable, List, Optional

import torch

from knowledge3d.cranium.fused_head import AdaptedFusedHead  # type: ignore
from knowledge3d.tools.phase25 import math_bench_evaluator as mbe  # type: ignore
from knowledge3d.tools import omni_bench_evaluator as obe  # type: ignore


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = ROOT / "viewer/public/galaxy/working/rpn_corpus.jsonl"


def iter_corpus(path: Path, limit: Optional[int]) -> Iterable[List[str]]:
    if not path.exists():
        return []
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            toks: Optional[List[str]] = None
            if isinstance(obj, dict):
                if isinstance(obj.get("tokens"), list):
                    toks = [str(t) for t in obj["tokens"]]
                elif isinstance(obj.get("rpn"), str):
                    toks = obj["rpn"].split()
            if not toks:
                continue
            yield toks
            n += 1
            if limit is not None and n >= int(limit):
                break


def synth_sequences(count: int = 2000, seed: int = 42) -> List[List[str]]:
    rng = random.Random(seed)
    ops = ["+", "-", "*"]
    seqs: List[List[str]] = []
    for _ in range(int(max(1, count))):
        # Build a small arithmetic RPN sequence
        n_terms = rng.randint(2, 4)
        toks: List[str] = []
        # push two numbers first
        toks.append(str(rng.randint(-9, 9)))
        toks.append(str(rng.randint(-9, 9)))
        for _k in range(n_terms - 1):
            if rng.random() < 0.4:
                toks.append(str(rng.randint(-9, 9)))
            toks.append(rng.choice(ops))
        # ensure ends with op
        if toks[-1] not in ops:
            toks.append(rng.choice(ops))
        seqs.append(toks)
    return seqs


def evaluate_sample(fh: AdaptedFusedHead, samples: List[List[str]], k: int = 20) -> None:
    ok = 0
    tried = 0
    for toks in samples[:k]:
        ans = fh._rpn_policy_generate("", [0.0] * 2048, max_steps=max(16, len(toks) + 2))
        tried += 1
        ok += int(bool(ans))
    print(f"Eval: policy produced answers for {ok}/{tried}")


def run(epochs: int, limit: Optional[int], lr: float, save_every: int) -> None:
    fh = AdaptedFusedHead()
    # Adjust LR if requested
    for g in fh._rpn_opt.param_groups:
        g['lr'] = float(lr)
    # Load corpus or synthesize
    seq_iter = list(iter_corpus(DEFAULT_CORPUS, limit))
    if not seq_iter:
        seq_iter = synth_sequences(count=int(limit or 2000))
        print(f"⚠️  Using synthetic RPN sequences: {len(seq_iter)}")
    else:
        print(f"📚 Loaded RPN corpus sequences: {len(seq_iter)}")

    random.shuffle(seq_iter)
    total_steps = 0
    for ep in range(1, int(max(1, epochs)) + 1):
        losses: List[float] = []
        for toks in seq_iter:
            loss = fh.rpn_policy_train_step(toks)
            losses.append(loss)
            total_steps += 1
            if save_every and total_steps % int(save_every) == 0:
                fh._save_rpn_policy()
        avg = sum(losses) / max(1, len(losses))
        print(f"🧮 Epoch {ep}: avg_loss={avg:.4f} ({len(losses)} seqs)")
        fh._save_rpn_policy()
        evaluate_sample(fh, seq_iter, k=min(20, len(seq_iter)))
    print("✅ RPN policy training complete; checkpoint saved.")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Train in-core RPN Policy Head (GPU-only)")
    ap.add_argument("--epochs", type=int, default=2, help="Epoch count")
    ap.add_argument("--limit", type=int, default=2000, help="Max sequences to load")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--save-every", type=int, default=500, help="Checkpointing interval (steps)")
    ap.add_argument("--eval-only", action="store_true", help="Skip training; run quick math + omni eval in-process")
    args = ap.parse_args()
    if args.eval_only:
        # Minimal eval path to avoid heavy trainer init elsewhere
        os.environ.setdefault("K3D_EVAL_MINIMAL", "1")
        os.environ.setdefault("K3D_DISABLE_TEXT_MODALITY", "1")
        os.environ.setdefault("K3D_ENABLE_RPN_POLICY", "1")
        # Math quick
        math_report = mbe.run(repos=[
            "Maxwell-Jia/AIME_2024",
            "meta-math/MetaMathQA",
            "openai/gsm8k",
        ], limit=min(50, int(args.limit)))
        math_out = Path("docs/benchmarks/math_bench_report.json")
        math_out.parent.mkdir(parents=True, exist_ok=True)
        math_out.write_text(json.dumps(math_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Math bench report → {math_out}")
        # Omni quick
        omni_report = obe.run(Path("/home/daniel/.cache/huggingface/datasets"), repos=None, split=None, limit=min(30, int(args.limit)))
        omni_out = Path("docs/benchmarks/omni_bench_report.json")
        omni_out.parent.mkdir(parents=True, exist_ok=True)
        omni_out.write_text(json.dumps(omni_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Omni bench report → {omni_out}")
    else:
        run(epochs=args.epochs, limit=args.limit, lr=args.lr, save_every=args.save_every)


if __name__ == "__main__":  # pragma: no cover
    main()
