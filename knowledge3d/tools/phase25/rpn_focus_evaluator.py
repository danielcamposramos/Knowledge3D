from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict

from knowledge3d.tools.phase18.meaning_cluster_trainer import MeaningClusterTrainer  # type: ignore


CASES_EXPRESSIONS: List[Dict[str, str]] = [
    {"q": "Compute sqrt(9) + log2(8)", "a": "6"},
    {"q": "Evaluate (2+3)*(5-1)", "a": "20"},
    {"q": "Compute floor(3.7) + ceil(2.1) + mod(10,3) + round(2.6)", "a": "10"},
    {"q": "Find gcd(48,18) + lcm(12,18) + fact(5)", "a": "162"},
    {"q": "Compute 22/7", "a": "22/7"},
    {"q": "Compute (1/3) + (2/3)", "a": "1"},
    {"q": "Compute \\lfloor 3.7 \\rfloor + \\lceil 2.1 \\rceil", "a": "6"},
    {"q": "Compute |−3| + |2-5|", "a": "6"},
    {"q": "Compute \\binom{10}{3} + C(5,2) + P(5,2)", "a": "150"},
    {"q": "Compute ⌊π⌋ + ⌈e⌉", "a": "6"},
    {"q": "Compute lg(1000) + ln(e)", "a": "4"},
    {"q": "Compute 7 % 4 + 10 % 3", "a": "4"},
    {"q": "Compute √(16) + √(9)", "a": "7"},
]


CASES_PROGRAMS: List[Dict[str, str]] = [
    {"q": "let a=3; let b=4; let c=5; a*b + c", "a": "17"},
    {"q": "let x=2+3; let y=nCr(5,2)+nPr(5,2); x*y", "a": "150"},
]


def run() -> Dict[str, object]:
    os.environ.setdefault("K3D_ENABLE_MATH_HEAD", "0")
    os.environ.setdefault("K3D_RPN_ROUND_MODE", "half_even")
    os.environ.setdefault("K3D_RATIONAL_OUTPUT", "1")
    trainer = MeaningClusterTrainer()
    fh = trainer.fused_head

    def eval_cases(name: str, cases: List[Dict[str, str]]) -> Dict[str, object]:
        total = len(cases)
        correct = 0
        details: List[Dict[str, object]] = []
        for c in cases:
            q = c["q"]
            a = c["a"].strip()
            pred = fh.predict(q, trainer.generate_multi_modal_embedding(q))
            # extract the boxed content
            boxed = pred.split("\\boxed{")[-1].split("}")[0].strip()
            ok = False
            if "/" in a:
                ok = (boxed == a)
            else:
                try:
                    ok = abs(float(boxed) - float(a)) <= max(1e-9, abs(float(a))*1e-9)
                except Exception:
                    ok = (boxed == a)
            correct += int(ok)
            details.append({"q": q, "expected": a, "prediction": boxed, "raw": pred, "correct": bool(ok)})
        return {"name": name, "total": total, "correct": correct, "accuracy": (correct/total if total else 0.0), "details": details}

    report = {
        "precision": "F64 default",
        "rounding": os.environ.get("K3D_RPN_ROUND_MODE", "unknown"),
        "rational_output": os.environ.get("K3D_RATIONAL_OUTPUT", "0"),
        "sections": [
            eval_cases("expressions", CASES_EXPRESSIONS),
            eval_cases("programs", CASES_PROGRAMS),
        ],
    }
    return report


def main() -> None:  # pragma: no cover
    report = run()
    out = Path("docs/benchmarks/rpn_focus_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RPN focus report → {out}")
    for sec in report["sections"]:
        print(f"  {sec['name']}: {sec['correct']}/{sec['total']} (acc={sec['accuracy']:.3f})")


if __name__ == "__main__":  # pragma: no cover
    main()
