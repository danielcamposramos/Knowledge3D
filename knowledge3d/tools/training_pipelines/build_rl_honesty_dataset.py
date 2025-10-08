from __future__ import annotations

"""
Build a standard RL (honesty) dataset: prompts with no provided contexts so the
model should answer "I don't know" (or similar) rather than hallucinate.

Rows: {query, answer, contexts[], reward}
- contexts[] is empty
- answer is produced via compose (returns the standard 'not enough memory' message) or
  via compose_generate with empty contexts
- reward: +0.5 if honest (admits unknown), else -0.25

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.tools.build_rl_honesty_dataset \
    --n 1000 --out docs/reports/training/rl_dataset_honesty_1000.jsonl --mode compose
"""

import argparse
import json
from pathlib import Path
from typing import List

from .eval_honesty_reward import is_honest  # reuse honesty detector


def load_prompts(n: int) -> List[str]:
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")
    out: List[str] = []
    for r in ds["train"]:
        t = (r.get("prompt") or r.get("chosen") or "")
        if t and len(t) >= 16:
            out.append(str(t).strip())
        if len(out) >= n:
            break
    return out


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Build RL honesty dataset (empty contexts → expect 'I don't know')")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="compose", choices=["compose", "generate"], help="compose|generate")
    args = ap.parse_args()
    prompts = load_prompts(int(args.n))
    from knowledge3d.skills.spatial_text import compose_answer, compose_generate  # type: ignore
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for q in prompts:
            if args.mode == "generate":
                ans = compose_generate(q, [], max_tokens=128)
            else:
                ans = compose_answer(q, [])
            reward = (0.5 if is_honest(ans) else -0.25)
            f.write(json.dumps({"query": q, "answer": ans, "contexts": [], "reward": reward}, ensure_ascii=False) + "\n")
    print(str(out))


if __name__ == "__main__":
    main()

