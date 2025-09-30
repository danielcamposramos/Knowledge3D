from __future__ import annotations

"""RPN stack validator (unit eval) — validates sequences and generation.

Checks that RPN sequences never underflow the stack and end in a valid state
over a test corpus, and optionally tests generated sequences with the RPN
policy (beam/greedy). Logs a summary to the Tablet via append_learning_memory.

Usage:
  PYTHONPATH=. python -m knowledge3d.tools.phase25.rpn_stack_eval \
    --corpus viewer/public/galaxy/working/rpn_corpus.jsonl --limit 1000 --beam
"""

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from knowledge3d.cranium.fused_head import AdaptedFusedHead  # type: ignore


def iter_corpus_tokens(path: Path, limit: Optional[int]) -> Iterable[List[str]]:
    if not path.exists():
        return []
    n = 0
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            toks = None
            if isinstance(obj, dict):
                if isinstance(obj.get('tokens'), list):
                    toks = [str(t) for t in obj['tokens']]
                elif isinstance(obj.get('rpn'), str):
                    toks = obj['rpn'].split()
            if toks:
                yield toks
                n += 1
            if limit and n >= int(limit):
                break


def stack_valid(fh: AdaptedFusedHead, tokens: List[str]) -> bool:
    depth = 0
    for tok in tokens:
        eff = fh._rpn_stack_effect(tok)
        if eff is None:
            return False
        depth += eff
        if depth < 0:
            return False
    return depth >= 1  # final stack must hold at least one result


def gen_prompts(n: int, seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    ops = ['+', '-', '*']
    out: List[str] = []
    for _ in range(max(1, int(n))):
        a = rng.randint(-99, 999)
        b = rng.randint(-99, 999)
        op = rng.choice(ops)
        out.append(f"Compute {a} {op} {b}.")
    return out


def run(corpus: Path, limit: int, test_gen: bool, beam: bool) -> None:
    fh = AdaptedFusedHead()
    if beam:
        import os
        os.environ['K3D_RPN_BEAM'] = '1'
        os.environ.setdefault('K3D_RPN_BEAM_WIDTH', '5')
    # Validate corpus sequences
    total = 0
    invalid = 0
    for toks in iter_corpus_tokens(corpus, limit):
        total += 1
        if not stack_valid(fh, toks):
            invalid += 1
    # Validate generated sequences
    gen_total = 0
    gen_invalid = 0
    if test_gen:
        prompts = gen_prompts(min(200, limit))
        for q in prompts:
            ans = fh._rpn_policy_generate(q, [0.0] * 2048, max_steps=32)
            if not ans:
                gen_invalid += 1
                gen_total += 1
                continue
            # Extract tokens from trace when enabled or parse RPN line
            # Best effort: pull RPN: line if present
            toks: Optional[List[str]] = None
            if '\nRPN:' in ans:
                try:
                    part = ans.split('\nRPN:')[-1].strip()
                    toks = part.split()
                except Exception:
                    toks = None
            # If no trace, skip validation here
            if toks:
                if not stack_valid(fh, toks):
                    gen_invalid += 1
            gen_total += 1

    summary = {
        'corpus_path': str(corpus),
        'checked': total,
        'invalid': invalid,
        'invalid_rate': (invalid / total) if total else 0.0,
        'gen_checked': gen_total,
        'gen_invalid': gen_invalid,
        'gen_invalid_rate': (gen_invalid / gen_total) if gen_total else 0.0,
        'beam': bool(beam),
    }
    print(json.dumps(summary, indent=2))
    # Log to Tablet
    fh.append_learning_memory(
        prompt=f"EVAL::RPN_STACK::beam={int(beam)}",
        true_answer=json.dumps(summary),
        predicted=json.dumps(summary),
        score=max(0.0, 1.0 - float(summary['invalid_rate'])),
        tags=['eval','rpn','stack','beam' if beam else 'greedy'],
    )


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description='Validate RPN stack sequences (corpus + generation)')
    ap.add_argument('--corpus', type=str, default='viewer/public/galaxy/working/rpn_corpus.jsonl')
    ap.add_argument('--limit', type=int, default=1000)
    ap.add_argument('--beam', action='store_true')
    ap.add_argument('--no-gen', action='store_true')
    args = ap.parse_args()
    run(Path(args.corpus), limit=int(args.limit), test_gen=(not args.no_gen), beam=bool(args.beam))


if __name__ == '__main__':
    main()

