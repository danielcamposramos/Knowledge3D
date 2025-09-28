from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

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
        return {"name": repo, "total": 0, "correct": 0, "accuracy": 0.0, "details": [], "note": "datasets unavailable"}
    try:
        ds = load_dataset(repo, split=split, download_config=DownloadConfig(local_files_only=True))
    except Exception:
        return {"name": repo, "total": 0, "correct": 0, "accuracy": 0.0, "details": [], "note": "dataset not in local cache"}

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


def run() -> Dict[str, object]:
    suites: List[Dict[str, object]] = []
    suites.append(evaluate_dataset("hendrycks/competition_math", split="train", limit=50))
    return {"suites": suites}


def main() -> None:  # pragma: no cover
    report = run()
    out = Path("docs/benchmarks/math_bench_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Math bench report → {out}")
    for suite in report["suites"]:
        print(f"  {suite['name']} ({suite.get('split','')}): {suite['correct']}/{suite['limit']} (acc={suite['accuracy']:.3f})")


if __name__ == "__main__":  # pragma: no cover
    main()

