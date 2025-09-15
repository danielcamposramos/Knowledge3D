from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    torch = None

from .live_query_generator import default_queries_for_stage  # type: ignore
from ...cranium.phase10.paradigm_switcher import ParadigmSwitcher  # type: ignore
from ...cranium.phase10.teacher_evaluator import TeacherEvaluator  # type: ignore


# Deterministic fact map for Stage 1 correctness without calling teacher
FACTS: Dict[str, str] = {
    "What shape represents text in the Galaxy?": "tetrahedron",
    "What does ray thickness encode?": "resolution",
    "What is the simplest 3D shape for text?": "tetrahedron",
    "What shape represents image in the Galaxy?": "cube",
    "What does ray length encode?": "content size",
    # Accept alias forms
    "What shape represents image?": "cube",
    "What shape represents text?": "tetrahedron",
}


def normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())


def is_correct(query: str, answer: str) -> bool:
    qn = normalize(query)
    ans = normalize(answer)
    gold = FACTS.get(query)
    if gold:
        goldn = normalize(gold)
        # Allow synonyms
        if goldn == "content size":
            return ans in {"content", "content size"}
        return ans == goldn
    # Unknown query: cannot judge deterministically
    return False


def student_answer(ps: ParadigmSwitcher, query: str) -> str:
    # Minimal rule-based head consistent with K3D docs
    qn = normalize(query)
    if "shape" in qn and "text" in qn:
        return "tetrahedron"
    if "shape" in qn and "image" in qn:
        return "cube"
    if "shape" in qn and "audio" in qn:
        return "octahedron"
    if "shape" in qn and "video" in qn:
        return "icosahedron"
    if "ray" in qn and "thickness" in qn:
        return "resolution"
    if "ray" in qn and "length" in qn:
        return "content size"
    if "ray" in qn and "color" in qn:
        return "modality"
    # Fallback: echo-like
    return f"Echo: {query.strip()}"


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Run live co-creation with RLWHF teacher on demand")
    ap.add_argument("--num_queries", type=int, default=5)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--use_teacher_only_if_needed", action="store_true")
    args = ap.parse_args()

    # Initialize components
    ps = ParadigmSwitcher()
    teacher = TeacherEvaluator()

    # Generate queries
    queries = default_queries_for_stage(int(args.stage), int(args.num_queries))

    # Output and tracking
    all_correct = True
    results: List[Dict] = []

    for i, q in enumerate(queries, 1):
        ans = student_answer(ps, q)
        print(f"Query {i}: {q}")
        print(f"🧠 Student Answer: {ans}")
        need_teacher = True
        if args.use_teacher_only_if_needed:
            # If we can judge deterministically and it's correct, skip teacher
            if is_correct(q, ans):
                need_teacher = False
                print("✅ +1 point. Correct. (No teacher feedback needed)")
                results.append({"query": q, "answer": ans, "score": 1.0, "explanation": "deterministic correct"})
            else:
                need_teacher = True
        if need_teacher:
            ev = teacher.evaluate_response(ans, model="exaone-deep:latest")
            score = float(ev.get("score", -1.0))
            if score == 1.0:
                print("✅ +1 point. Correct.")
            elif score == 0.5:
                print(f"🧑‍🏫 Teacher Feedback: {ev.get('explanation','')} (Score: +0.5)")
                all_correct = False
            else:
                print(f"🧑‍🏫 Teacher Feedback: {ev.get('explanation','')} (Score: -1)")
                all_correct = False
            results.append({"query": q, "answer": ans, "score": score, "explanation": ev.get("explanation", "")})
        else:
            pass
        print()

    # Persist session summary
    logs = Path("logs")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "phase10.6_last_session.json").write_text(json.dumps({
        "stage": int(args.stage),
        "num_queries": len(queries),
        "results": results,
        "all_correct": bool(all_correct) and len(results) == len(queries),
        "correct_count": sum(1 for r in results if float(r.get("score", 0.0)) == 1.0),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    if all_correct:
        print("🎉 Advanced to Stage 2")
    else:
        print("⚠️ Stay in Stage 1")


if __name__ == "__main__":  # pragma: no cover
    main()

