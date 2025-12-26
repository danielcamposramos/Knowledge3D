#!/usr/bin/env python3
"""
Analyze DualShadowCopy exploration traces to find highest-impact wrong-computation families.

Families are grouped by:
  - template_used
  - patterns_matched (sorted unique)

We score by: count * median(abs(expected-got)).
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class FamilyKey:
    template_used: str
    patterns: Tuple[str, ...]


def _iter_explorations(state: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    items = state.get("explorations", [])
    if isinstance(items, list):
        for e in items:
            if isinstance(e, dict):
                yield e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        type=str,
        default="/K3D/Knowledge3D.local/checkpoints/math_benchmarks/shadow_copy.json",
        help="Path to shadow_copy.json",
    )
    ap.add_argument("--recent", type=int, default=0, help="Only analyze the last N exploration entries")
    ap.add_argument("--top", type=int, default=10, help="How many families to print")
    ap.add_argument("--samples", type=int, default=3, help="How many sample prompts to show per family")
    args = ap.parse_args()

    path = Path(args.path)
    state = json.loads(path.read_text())
    explorations = list(_iter_explorations(state))
    if args.recent and args.recent > 0:
        explorations = explorations[-int(args.recent) :]

    families: Dict[FamilyKey, List[Dict[str, Any]]] = {}
    for e in explorations:
        # Wrong computation = we produced a plausible result (success=True) but benchmark says incorrect.
        if not e.get("success"):
            continue
        if e.get("correct") is not False:
            continue
        expected = e.get("expected_num")
        got = e.get("got_num")
        if not isinstance(expected, (int, float)) or not isinstance(got, (int, float)):
            continue

        patterns = e.get("patterns_matched", [])
        if not isinstance(patterns, list):
            patterns = []
        pat_sig = tuple(sorted({str(p) for p in patterns if p}))
        key = FamilyKey(template_used=str(e.get("template_used") or ""), patterns=pat_sig)
        families.setdefault(key, []).append(e)

    scored: List[Tuple[float, FamilyKey, List[Dict[str, Any]]]] = []
    for key, items in families.items():
        deltas = [abs(float(it["got_num"]) - float(it["expected_num"])) for it in items]
        if not deltas:
            continue
        med = statistics.median(deltas)
        score = float(len(items)) * float(med)
        scored.append((score, key, items))

    scored.sort(key=lambda t: t[0], reverse=True)
    top_n = max(1, int(args.top))

    print(f"Explorations analyzed: {len(explorations)}")
    print(f"Wrong-computation families: {len(scored)} (success=True & correct=False)")
    print()

    for idx, (score, key, items) in enumerate(scored[:top_n], start=1):
        deltas = [abs(float(it["got_num"]) - float(it["expected_num"])) for it in items]
        avg = sum(deltas) / max(1, len(deltas))
        med = statistics.median(deltas)
        template = key.template_used or "(unknown)"
        patterns = ", ".join(key.patterns[:12]) + ("..." if len(key.patterns) > 12 else "")
        print(f"{idx}. score={score:.2f}  failures={len(items)}  avg_abs_err={avg:.2f}  med_abs_err={med:.2f}")
        print(f"   template={template}")
        print(f"   patterns=[{patterns}]")
        for sample in items[: max(0, int(args.samples))]:
            pt = str(sample.get("problem_text") or "").replace("\n", " ")
            exp = sample.get("expected_num")
            got = sample.get("got_num")
            print(f"   - expected={exp} got={got} text={pt[:140]}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

