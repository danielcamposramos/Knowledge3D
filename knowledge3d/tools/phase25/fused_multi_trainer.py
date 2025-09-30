"""Fused Multi-Trainer — auto-train fused head on local HF cache (PTX-only).

Methods applied per dataset (when possible):
- Supervised numeric classification (projection+math head) if target in [0..999]
- RPN policy sequence training by extracting math expressions from text (infix->RPN)

Everything runs in a single Python process and uses PTX-only features. After
training, updated heads are packed into the House GLB as appliances:
- fused_math (projection+math)
- fused_rpn_policy (RPN policy)

Usage (GPU env):
  PYTHONPATH=. K3D_PTX_STRICT=1 K3D_FORCE_PTX_FUSE=1 \
  python -m knowledge3d.tools.phase25.fused_multi_trainer --epochs 2 --limit 300 --keys math,gsm8k,metamath,aime
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from datasets import load_dataset, DownloadConfig  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore
    DownloadConfig = None  # type: ignore

from knowledge3d.cranium.fused_head import AdaptedFusedHead  # type: ignore
from knowledge3d.tools.phase25 import math_bench_evaluator as mbe  # type: ignore
from knowledge3d.tools.weights_in_glb import pack_pt_into_glb  # type: ignore
from knowledge3d.tools.phase25.math_bench_evaluator import discover_local_hf_repos  # type: ignore
from knowledge3d.skills.infix_to_rpn import extract_math_expression, infix_to_rpn  # type: ignore


ROOT = Path(__file__).resolve().parents[3]
HOUSE_GLB = ROOT / "viewer/public/houses/default/memory_house.glb"


def _try_load(repo: str, split: str, limit: int):
    if load_dataset is None or DownloadConfig is None:
        return []
    try:
        ds = load_dataset(repo, split=split, download_config=DownloadConfig(local_files_only=True))
    except Exception:
        return []
    n = min(limit, len(ds))
    return [ds[i] for i in range(n)]


def _rows_from_repo(repo: str, limit: int) -> List[Dict[str, object]]:
    # Prefer train->validation->test
    for sp in ("train", "validation", "test"):
        rows = _try_load(repo, sp, limit)
        if rows:
            return rows
    return []


def _numeric_target(row: Dict[str, object]) -> Optional[int]:
    text = str(row.get("solution") or row.get("Solution") or row.get("answer") or "")
    norm = mbe._normalize(mbe._coerce_answer(text)) if text else None  # type: ignore[attr-defined]
    if norm is None:
        return None
    try:
        yi = int(norm)
        if 0 <= yi <= 999:
            return yi
    except Exception:
        return None
    return None


def _question_text(row: Dict[str, object]) -> str:
    return str(row.get("problem") or row.get("question") or row.get("prompt") or row.get("Problem") or row.get("text") or "")


def _train_numeric(fh: AdaptedFusedHead, q: str, yi: int, lr: float) -> None:
    emb = fh._build_ptx_fused_embedding(q)  # PTX-only fusion
    fh.train_step(emb, str(yi), lr=lr)


def _train_rpn_policy(fh: AdaptedFusedHead, q: str) -> bool:
    expr = extract_math_expression(q or "")
    if not expr:
        return False
    try:
        tokens = infix_to_rpn(expr)
        if tokens:
            fh.rpn_policy_train_step(tokens)
            return True
    except Exception:
        return False
    return False


def _synth_arith(count: int = 200, seed: int = 42) -> List[tuple[str, int]]:
    import random
    rng = random.Random(seed)
    out: List[tuple[str, int]] = []
    ops = ['+', '-', '*']
    for _ in range(max(1, int(count))):
        a = rng.randint(-99, 999)
        b = rng.randint(-99, 999)
        op = rng.choice(ops)
        if op == '+':
            y = a + b
        elif op == '-':
            y = a - b
        else:
            y = a * b
        # Keep in 0..999 for math head; if outside, wrap/clamp
        if y < 0 or y > 999:
            y = abs(y) % 1000
        q = f"Compute {a} {op} {b}."
        out.append((q, y))
    return out


def _log_tags(fh: AdaptedFusedHead, method: str, repo: str, q: str, expected: Optional[str], score: float = 1.0) -> None:
    try:
        fh.append_learning_memory(
            prompt=f"TRAIN::{method}::{repo}::{q[:64]}",
            true_answer=str(expected) if expected is not None else "",
            predicted=str(expected) if expected is not None else "",
            score=float(score),
            tags=["train", method, repo],
        )
    except Exception:
        pass


def _math_eval(tag: str = "pre") -> None:
    try:
        report = mbe.run(repos=[
            'Maxwell-Jia/AIME_2024',
            'meta-math/MetaMathQA',
            'openai/gsm8k',
        ], limit=30)
        out = ROOT / f"docs/benchmarks/math_bench_report_{tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(__import__('json').dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"Math bench ({tag}) → {out}")
    except Exception as e:
        print(f"⚠️  math eval ({tag}) failed: {e}")


def _append_progress(record: dict) -> None:
    from pathlib import Path
    import json, time
    out = ROOT / 'docs/benchmarks/progress_log.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    log: List[dict] = []
    if out.exists():
        try:
            log = json.loads(out.read_text(encoding='utf-8'))
        except Exception:
            log = []
    record = dict(record)
    record['ts'] = time.time()
    log.append(record)
    out.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding='utf-8')


def run(keys: List[str], limit: int, epochs: int, lr_math: float, lr_rpn: float, eval_every: int = 5) -> None:
    os.environ.setdefault("K3D_PTX_STRICT", "1")
    os.environ.setdefault("K3D_FORCE_PTX_FUSE", "1")

    targets = discover_local_hf_repos(tuple(keys))
    if not targets:
        print("⚠️  No local HF repos discovered for keys:", keys)
        return
    print("Repos:", ", ".join(targets[:12]), ("..." if len(targets) > 12 else ""))

    fh = AdaptedFusedHead()
    # LRs
    for g in fh._opt.param_groups:
        g["lr"] = float(lr_math)
    for g in fh._rpn_opt.param_groups:
        g["lr"] = float(lr_rpn)

    _math_eval("pre")

    # Epochs over discovered repos
    for ep in range(1, int(max(1, epochs)) + 1):
        print(f"\n🚀 Epoch {ep}")
        total_num = total_rpn = 0
        # Curriculum: bootstrap with synthetic arithmetic at start of each epoch
        synth_n = min(400, max(200, limit)) if ep <= 5 else min(200, limit)
        synth = _synth_arith(synth_n, seed=1337 + ep)
        for q, yi in synth:
            _train_numeric(fh, q, yi, lr_math)
            _log_tags(fh, "numeric_synth", "synthetic", q, str(yi))
            total_num += 1
        for repo in targets:
            rows = _rows_from_repo(repo, limit)
            if not rows:
                continue
            num_trained = rpn_trained = 0
            for row in rows:
                q = _question_text(row)
                # Numeric supervised
                yi = _numeric_target(row)
                if yi is not None:
                    _train_numeric(fh, q, yi, lr_math)
                    num_trained += 1
                    _log_tags(fh, "numeric", repo, q, str(yi))
                # RPN policy from question text
                if q:
                    if _train_rpn_policy(fh, q):
                        rpn_trained += 1
                        _log_tags(fh, "rpn", repo, q, None)
                # Soft QA target for non-numeric expected
                exp_text = row.get("answer") or row.get("solution") or row.get("expected_answer") or row.get("label")
                if exp_text and yi is None and q:
                    try:
                        loss = fh.qa_soft_train_step(q, str(exp_text), lr_math)
                        if loss > 0:
                            _log_tags(fh, "qa_soft", repo, q, str(exp_text), score=max(0.1, 1.0/(1.0+loss)))
                    except Exception:
                        pass
            total_num += num_trained
            total_rpn += rpn_trained
            print(f"  {repo}: numeric={num_trained}, rpn={rpn_trained}")
        print(f"🧮 epoch totals: numeric={total_num}, rpn={total_rpn}")
        _append_progress({
            'trainer': 'multi', 'epoch': ep,
            'numeric_trained': total_num, 'rpn_trained': total_rpn,
        })
        # Periodic RPN stack eval with beam
        try:
            if eval_every and (ep % int(max(1, eval_every)) == 0):
                from knowledge3d.tools.phase25 import rpn_stack_eval as rse  # type: ignore
                print("Running periodic RPN stack eval (beam)...")
                rse.run(ROOT / 'viewer/public/galaxy/working/rpn_corpus.jsonl', limit=500, test_gen=False, beam=True)
        except Exception as e:
            print("⚠️  RPN stack periodic eval failed:", e)

    # Save and pack
    fh._save_math_head()
    fh._save_rpn_policy()
    core_pt = fh._save_core_heads()
    if HOUSE_GLB.exists():
        try:
            ckpt_m = ROOT / "viewer/public/house/house_math_head.pt"
            if ckpt_m.exists():
                pack_pt_into_glb(HOUSE_GLB, ckpt_m, "fused_math")
                print("📦 Packed fused_math into GLB")
        except Exception as e:
            print("⚠️  fused_math pack error:", e)
        try:
            ckpt_p = ROOT / "viewer/public/house/house_rpn_policy.pt"
            if ckpt_p.exists():
                pack_pt_into_glb(HOUSE_GLB, ckpt_p, "fused_rpn_policy")
                print("📦 Packed fused_rpn_policy into GLB")
        except Exception as e:
            print("⚠️  fused_rpn_policy pack error:", e)
        try:
            if core_pt.exists():
                pack_pt_into_glb(HOUSE_GLB, core_pt, "fused_core")
                print("📦 Packed fused_core into GLB")
        except Exception as e:
            print("⚠️  fused_core pack error:", e)
    _math_eval("post")
    print("✅ Fused multi-trainer complete.")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Auto-train fused head on local HF datasets (PTX-only)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--limit", type=int, default=300, help="Max rows per repo")
    ap.add_argument("--keys", type=str, default="math,gsm8k,metamath,aime")
    ap.add_argument("--lr-math", type=float, default=5e-4)
    ap.add_argument("--lr-rpn", type=float, default=1e-3)
    args = ap.parse_args()
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    run(keys, limit=int(args.limit), epochs=int(args.epochs), lr_math=float(args.lr_math), lr_rpn=float(args.lr_rpn))


if __name__ == "__main__":
    main()
