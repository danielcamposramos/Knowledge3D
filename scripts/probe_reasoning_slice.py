#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from collections import Counter

from benchmarks.gsm8k import GSM8KBenchmark
from benchmarks.math_competitions import MathCompetitionBenchmark
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def _cat(kind: str, row: dict) -> str:
    pred, exp = str(row.get("predicted_answer")), str(row.get("expected_answer"))
    trace = " | ".join(str(x) for x in row.get("reasoning_trace", []))
    if str(row.get("correct")) == "1" or pred.strip() == exp.strip():
        return "correct"
    if kind == "gsm8k":
        if exp and f"-> {exp}" in trace and "Halting gate: continue" in trace:
            return "correct_preview_no_halt"
        if "GSM8K fission: miss" in trace:
            return "no_operation_match"
        if "GSM8K number neighborhood: miss" in trace:
            return "no_number_match"
        return "wrong_operation_or_scoring"
    if pred.rstrip("0").rstrip(".") == exp.rstrip("0").rstrip("."):
        return "format_only"
    if "math_template_" in trace:
        return "template_hit_wrong_answer"
    if "Halting gate: continue" in trace:
        return "no_convergence"
    return "no_template_match"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("math", "gsm8k"))
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()
    kv = Knowledgeverse()
    if args.kind == "math":
        bench = MathCompetitionBenchmark(knowledgeverse=kv, max_problems=None); items = [p for p in bench.problems if not str(p.get("competition", "")).upper().startswith("GSM8K")][: args.limit]; solve = lambda x: bench._solve_problem(problem=x, use_enriched=True); galaxies = ["Math", "Grammar"]
    else:
        bench = GSM8KBenchmark(knowledgeverse=kv, max_questions=args.limit); items = bench.questions[: args.limit]; solve = lambda x: bench._solve_question(question=x, use_enriched=True); galaxies = ["Math", "Grammar", "Number", "Word"]
    kv.bind_gpu_galaxy_runtime(galaxy_names=galaxies)
    rows = []
    for item in items:
        out = solve(item); task = out.get("task_result", {}); expected = out.get("expected_answer", out.get("correct_answer"))
        rows.append({"id": out.get("problem_id", out.get("question_id")), "correct": int(bool(out.get("correct"))), "predicted_answer": out.get("predicted_answer"), "expected_answer": expected, "category": _cat(args.kind, {**out, "expected_answer": expected}), "match_id": ((task.get("match") or {}).get("id")), "similarity": ((task.get("match") or {}).get("confidence", task.get("top_match_similarity"))), "program_id": task.get("program_id"), "reasoning_trace": list(out.get("reasoning_trace", []))})
        print(json.dumps(rows[-1], ensure_ascii=True), flush=True)
    print(json.dumps({"benchmark": args.kind, "total": len(rows), "correct": sum(r["correct"] for r in rows), "categories": dict(Counter(r["category"] for r in rows))}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
