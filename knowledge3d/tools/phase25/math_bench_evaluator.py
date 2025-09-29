from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Iterable

try:
    from datasets import load_dataset, DownloadConfig  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore
    DownloadConfig = None  # type: ignore

from knowledge3d.tools.phase18.meaning_cluster_trainer import MeaningClusterTrainer  # type: ignore


BOX_RE = re.compile(r"\\boxed\{\s*([^}]+?)\s*\}")
FRAC_RE = re.compile(r"\\frac\{\s*([-+]?\d+)\s*\}\{\s*([-+]?\d+)\s*\}")
INT_RE = re.compile(r"\b[-+]?\d+\b")
RAT_RE = re.compile(r"^[-+]?\d+\s*/\s*[-+]?\d+$")


def _coerce_answer(text: str) -> Optional[str]:
    if not text:
        return None
    m = BOX_RE.findall(text)
    if m:
        return m[-1].strip()
    m2 = FRAC_RE.findall(text)
    if m2:
        a, b = m2[-1]
        try:
            ai = int(a)
            bi = int(b)
            if bi != 0:
                return f"{ai}/{bi}"
        except Exception:
            pass
    ints = INT_RE.findall(text)
    if ints:
        return ints[-1]
    return None


def _normalize(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    s = ans.strip().replace(" ", "")
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        try:
            return str(int(s))
        except Exception:
            return s
    if RAT_RE.match(s):
        a, b = s.split("/")
        try:
            ai = int(a)
            bi = int(b)
            if bi < 0:
                ai, bi = -ai, -bi
            return f"{ai}/{bi}"
        except Exception:
            return s
    return s


def evaluate_dataset(repo: str, split: str = "train", limit: int = 50) -> Dict[str, object]:
    if load_dataset is None or DownloadConfig is None:
        return {"name": repo, "split": split, "limit": 0, "total": 0, "correct": 0, "accuracy": 0.0, "details": [], "note": "datasets unavailable"}
    try:
        ds = load_dataset(repo, split=split, download_config=DownloadConfig(local_files_only=True))
    except Exception:
        return {"name": repo, "split": split, "limit": 0, "total": 0, "correct": 0, "accuracy": 0.0, "details": [], "note": "dataset not in local cache"}

    os.environ.setdefault("K3D_ENABLE_MATH_HEAD", "0")
    os.environ.setdefault("K3D_RPN_ROUND_MODE", "half_even")
    os.environ.setdefault("K3D_RATIONAL_OUTPUT", "1")

    trainer = MeaningClusterTrainer()
    fh = trainer.fused_head

    total = min(limit, len(ds))
    correct = 0
    details: List[Dict[str, object]] = []

    for i in range(total):
        row = ds[i]
        q = str(row.get("problem") or row.get("question") or row.get("Problem") or "")
        sol = str(row.get("solution") or row.get("Solution") or row.get("answer") or "")
        expected = _normalize(_coerce_answer(sol) or _coerce_answer(str(row.get("answer") or "")))
        if not q:
            continue
        pred_raw = fh.predict(q, trainer.generate_multi_modal_embedding(q))
        boxed = pred_raw.split("\\boxed{")[-1].split("}")[0].strip()
        predn = _normalize(boxed)
        ok = False
        if expected is None:
            ok = False
        else:
            if RAT_RE.match(expected):
                ok = (predn == expected)
            else:
                try:
                    ok = abs(float(predn or "nan") - float(expected)) <= max(1e-6, abs(float(expected))*1e-6)
                except Exception:
                    ok = (predn == expected)
        correct += int(ok)
        details.append({
            "idx": i,
            "question": q[:2000],
            "expected": expected,
            "prediction": predn,
            "raw": pred_raw,
            "correct": bool(ok),
        })

    return {
        "name": repo,
        "split": split,
        "limit": total,
        "correct": correct,
        "accuracy": (correct / total if total else 0.0),
        "details": details,
    }

def _hf_cache_dirs() -> List[Path]:
    roots: List[Path] = []
    # Respect HF_HOME if set
    env_home = os.environ.get("HF_HOME") or os.environ.get("HF_DATASETS_CACHE")
    if env_home:
        roots.append(Path(env_home))
    # Default user cache
    roots.append(Path.home() / ".cache" / "huggingface" / "datasets")
    # Common alt cache on some systems
    roots.append(Path("/home/daniel/.cache/huggingface/datasets"))
    # Deduplicate preserving order
    seen: set = set()
    out: List[Path] = []
    for r in roots:
        if r.exists() and str(r) not in seen:
            seen.add(str(r))
            out.append(r)
    return out


def discover_local_hf_repos(keywords: Iterable[str] = ("math", "aime", "amc", "gsm8k", "metamath")) -> List[str]:
    repos: List[str] = []
    keys = tuple(str(k).lower() for k in (keywords or ()))
    for root in _hf_cache_dirs():
        try:
            for entry in root.iterdir():
                if not entry.is_dir():
                    continue
                name = entry.name
                # Expect form like "owner___datasetname"
                if "___" in name and any(k in name.lower() for k in keys):
                    owner, ds = name.split("___", 1)
                    if owner and ds:
                        repos.append(f"{owner}/{ds}")
        except Exception:
            continue
    # Deduplicate while preserving order
    seen: set = set()
    unique: List[str] = []
    for r in repos:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def run(repos: Optional[List[str]] = None, limit: int = 50) -> Dict[str, object]:
    suites: List[Dict[str, object]] = []
    if not repos:
        repos = [
            "hendrycks/competition_math",
            "meta-math/MetaMathQA",
            "openai/gsm8k",
        ]
    for repo in repos:
        suites.append(evaluate_dataset(repo, split="train", limit=limit))
    return {"suites": suites}


def main() -> None:  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate math datasets from local HF cache")
    ap.add_argument("--limit", type=int, default=50, help="Max items per dataset")
    ap.add_argument("--auto", action="store_true", help="Discover local math-like repos from HF cache")
    ap.add_argument("--list", action="store_true", help="List discovered repos and exit")
    ap.add_argument("--repos", type=str, default="", help="Comma-separated list of repos to evaluate (owner/name)")
    args = ap.parse_args()

    repos: List[str] = []
    if args.repos:
        repos.extend([r.strip() for r in args.repos.split(",") if r.strip()])
    if args.auto:
        discovered = discover_local_hf_repos()
        if args.list:
            print("Discovered HF repos (local cache):")
            for r in discovered:
                print(" -", r)
            return
        repos.extend(discovered)
    if not repos:
        repos = None  # use defaults

    report = run(repos=repos, limit=int(max(1, args.limit)))
    out = Path("docs/benchmarks/math_bench_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Math bench report → {out}")
    for suite in report["suites"]:
        print(f"  {suite['name']} ({suite.get('split','')}): {suite['correct']}/{suite['limit']} (acc={suite['accuracy']:.3f})")


if __name__ == "__main__":  # pragma: no cover
    main()
