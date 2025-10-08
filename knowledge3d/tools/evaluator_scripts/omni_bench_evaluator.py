# DEPRECATED: evaluator retained for legacy runs; scheduled for swarm refactor.
"""Omni Bench Evaluator — scan local HF cache and evaluate fused head on many datasets.

This evaluator discovers repos under a Hugging Face cache directory and, for each,
attempts to load a local split (no downloads) and evaluate a small sample by
querying the single fused head. It reports simple metrics:
- nonempty rate (did we answer?)
- strict match (when an 'answer' field exists)
- soft match (normalized text similarity >= 0.72 or JSON structural equality)

Usage:
  conda run -n k3d-cranium env PYTHONPATH=. python -m knowledge3d.tools.omni_bench_evaluator \
    --root-cache /home/daniel/.cache/huggingface/datasets --limit 50

To target specific repos:
  --repos Maxwell-Jia/AIME_2024,meta-math/MetaMathQA,openai/gsm8k

Notes:
- Uses local HF cache only (DownloadConfig(local_files_only=True)).
- For each row: extracts a query from common fields (question/instruction/prompt/text).
- Expected answers are taken from fields like answer/solution/label when present.
"""
from __future__ import annotations

import argparse
import json
import os
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from datasets import load_dataset, DownloadConfig  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore
    DownloadConfig = None  # type: ignore

from knowledge3d.tools.phase18.meaning_cluster_trainer import MeaningClusterTrainer  # type: ignore
from knowledge3d.cranium.fused_head import AdaptedFusedHead  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/benchmarks/omni_bench_report.json"


def discover_local_repos(root: Path) -> List[str]:
    repos: List[str] = []
    if not root.exists():
        return repos
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if "___" in name:
            owner, ds = name.split("___", 1)
            repos.append(f"{owner}/{ds}")
    # Deduplicate
    seen = set()
    out: List[str] = []
    for r in repos:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _normalize(s: str) -> str:
    import re as _re
    return _re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _coerce_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_query_and_expected(row: Dict[str, object]) -> Tuple[str, Optional[str]]:
    # Try common QA patterns first
    for qk in ["question", "Problem", "prompt", "instruction", "query", "text", "content"]:
        if qk in row and isinstance(row[qk], str) and row[qk].strip():
            q = row[qk].strip()
            # Include input/context if present
            ctx = None
            for ck in ["input", "context"]:
                v = row.get(ck)
                if isinstance(v, str) and v.strip():
                    ctx = v.strip()
                    break
            if ctx:
                q = f"{q}\nContext: {ctx}"
            expected = None
            for ak in ["answer", "solution", "label", "expected_answer", "target"]:
                v = row.get(ak)
                if isinstance(v, (str, int, float)):
                    expected = str(v)
                    break
            return q, expected
    # Fallback: concatenate short string fields
    parts: List[str] = []
    for k, v in row.items():
        if isinstance(v, str) and len(v) <= 1024 and v.strip():
            parts.append(f"{k}: {v.strip()}")
            if len(parts) >= 2:
                break
    q = "\n".join(parts) if parts else ""
    return q, None


def evaluate_repo(trainer: Optional[MeaningClusterTrainer], repo: str, split: Optional[str], limit: int) -> Dict[str, object]:
    if load_dataset is None or DownloadConfig is None:
        return {"name": repo, "split": split or "auto", "limit": 0, "total": 0, "correct": 0, "soft_correct": 0, "nonempty": 0, "accuracy": 0.0, "soft_accuracy": 0.0, "nonempty_rate": 0.0, "note": "datasets unavailable"}
    # Pick a split
    splits = [s for s in [split, "train", "validation", "test"] if s]
    ds = None
    used_split = None
    for sp in splits:
        try:
            ds = load_dataset(repo, split=sp, download_config=DownloadConfig(local_files_only=True))
            used_split = sp
            break
        except Exception:
            continue
    if ds is None:
        return {"name": repo, "split": split or "auto", "limit": 0, "total": 0, "correct": 0, "soft_correct": 0, "nonempty": 0, "accuracy": 0.0, "soft_accuracy": 0.0, "nonempty_rate": 0.0, "note": "dataset not in local cache"}

    minimal = str(os.environ.get("K3D_EVAL_MINIMAL", "0")).lower() in {"1","true","yes"}
    if minimal:
        fh = AdaptedFusedHead()
    else:
        assert trainer is not None
        fh = trainer.fused_head
    total = min(limit, len(ds))
    correct = 0
    soft_correct = 0
    nonempty = 0
    for i in range(total):
        row = ds[i]
        if not isinstance(row, dict):
            continue
        q, expected = _extract_query_and_expected(row)
        if not q:
            continue
        emb = ([0.0] * 2048) if minimal else trainer.generate_multi_modal_embedding(q)  # type: ignore[union-attr]
        pred_raw = fh.predict(q, emb)
        pred = str(pred_raw or "").strip()
        nonempty += int(bool(pred))
        if expected is not None:
            strict = pred.lower() == str(expected).strip().lower()
            if strict:
                correct += 1
            # Soft: JSON structural equality, else text similarity
            soft_ok = False
            pj = _coerce_json(pred)
            ej = _coerce_json(str(expected))
            if pj is not None and ej is not None:
                soft_ok = (pj == ej)
            else:
                soft_ok = SequenceMatcher(None, _normalize(pred), _normalize(expected)).ratio() >= 0.72
            if soft_ok:
                soft_correct += 1
    return {
        "name": repo,
        "split": used_split or split or "auto",
        "limit": total,
        "total": total,
        "correct": correct,
        "soft_correct": soft_correct,
        "nonempty": nonempty,
        "accuracy": (correct / total) if total else 0.0,
        "soft_accuracy": (soft_correct / total) if total else 0.0,
        "nonempty_rate": (nonempty / total) if total else 0.0,
    }


def run(root_cache: Path, repos: Optional[List[str]], split: Optional[str], limit: int) -> Dict[str, object]:
    minimal = str(os.environ.get("K3D_EVAL_MINIMAL", "0")).lower() in {"1","true","yes"}
    trainer = None if minimal else MeaningClusterTrainer()
    targets = repos if repos else discover_local_repos(root_cache)
    suites: List[Dict[str, object]] = []
    for r in targets:
        try:
            suites.append(evaluate_repo(trainer, r, split=split, limit=limit))
        except Exception as e:
            suites.append({"name": r, "split": split or "auto", "limit": 0, "total": 0, "correct": 0, "soft_correct": 0, "nonempty": 0, "accuracy": 0.0, "soft_accuracy": 0.0, "nonempty_rate": 0.0, "note": f"error: {e}"})
    return {"root": str(root_cache), "limit": limit, "suites": suites}


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Evaluate fused head across local HF cached datasets (omni)")
    ap.add_argument("--root-cache", type=str, default="/home/daniel/.cache/huggingface/datasets", help="HF cache root")
    ap.add_argument("--repos", type=str, default="", help="Comma-separated repos (owner/name). If empty, auto-discover")
    ap.add_argument("--split", type=str, default="", help="Preferred split (train/validation/test)")
    ap.add_argument("--limit", type=int, default=50, help="Max rows per dataset")
    args = ap.parse_args()
    repos = [r.strip() for r in args.repos.split(",") if r.strip()] if args.repos else None
    split = args.split or None
    report = run(Path(args.root_cache), repos=repos, split=split, limit=int(max(1, args.limit)))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Omni bench report → {REPORT}")
    for s in report["suites"]:
        print(f"  {s['name']} ({s.get('split','')}): nonempty={s['nonempty_rate']:.2f}, strict={s['accuracy']:.2f}, soft={s['soft_accuracy']:.2f}")


if __name__ == "__main__":
    main()
