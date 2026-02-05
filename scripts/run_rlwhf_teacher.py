#!/usr/bin/env python3
"""
Run an Ollama teacher over Log Galaxy traces to populate Feedback Galaxy.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from typing import Any, Dict, Iterable, List, Optional

from knowledge3d.training.math_benchmarks.feedback_galaxy import FeedbackGalaxy


ALLOWED_RULES = ["sum_rule", "product_rule", "quotient_rule", "power_rule", "chain_rule"]


def _iter_traces(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _render_trace_lines(trace: Dict[str, Any], max_lines: int) -> List[str]:
    lines = list(trace.get("trace_lines") or [])
    if lines:
        return lines[:max_lines]
    steps = trace.get("step_sequence") or []
    rendered = []
    for step in steps:
        kind = step.get("kind", "step")
        label = step.get("label") or step.get("rule") or ""
        rendered.append(f"[{kind}] {label}".strip())
    return rendered[:max_lines]


def _has_hallucination(lines: List[str]) -> bool:
    joined = "\n".join(lines).lower()
    return "<hallucination>" in joined or "<heuristic>" in joined


def _autonomy(trace: Dict[str, Any]) -> float:
    meta = trace.get("metadata") or {}
    steps = len(trace.get("step_sequence") or [])
    policy_steps = int(meta.get("policy_steps", 0) or 0)
    if steps <= 0:
        return 0.0
    return float(policy_steps) / float(steps)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    for match in re.finditer(r"\{.*?\}", text, re.DOTALL):
        blob = match.group(0)
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    return None


def _build_prompt(problem_text: str, trace_lines: List[str]) -> str:
    trace_block = "\n".join(trace_lines) if trace_lines else "(no trace lines available)"
    return (
        "You are a Calculus Professor. A student (Neural Policy) attempted to solve: "
        f"\"{problem_text}\".\n\n"
        "Here is their thought process trace:\n"
        f"{trace_block}\n\n"
        "The step marked <hallucination> indicates where their intuition failed.\n\n"
        "Evaluate their performance:\n"
        "1. Score (-2: Fabrication, -1: Mixed/Confused, 0: Don't Know, +1: Partially Correct, +2: Perfect).\n"
        "2. Explain WHY the hallucinated rule was wrong.\n"
        "3. Suggest the correct rule from this list: "
        "[sum_rule, product_rule, quotient_rule, power_rule, chain_rule].\n\n"
        "Output JSON: {\"score\": int, \"feedback\": \"string\", \"suggested_rule\": \"string\"}\n"
    )


def _call_ollama(model: str, prompt: str, timeout_s: int) -> str:
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ollama run failed")
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RLWHF teacher over Log Galaxy traces.")
    parser.add_argument("--input", required=True, help="Log Galaxy JSONL path.")
    parser.add_argument("--output", default="data/feedback_galaxy_v1.jsonl", help="Feedback output JSONL.")
    parser.add_argument("--model", default="deepseek-r1:7b", help="Ollama model name.")
    parser.add_argument("--min-autonomy", type=float, default=0.9, help="Autonomy threshold.")
    parser.add_argument("--max-trace-lines", type=int, default=40, help="Max trace lines per prompt.")
    parser.add_argument("--timeout-s", type=int, default=120, help="Ollama timeout in seconds.")
    args = parser.parse_args()

    feedback_galaxy = FeedbackGalaxy()
    processed = 0
    skipped = 0

    for trace in _iter_traces(args.input):
        problem_text = trace.get("problem_text") or ""
        trace_lines = _render_trace_lines(trace, args.max_trace_lines)
        autonomy = _autonomy(trace)
        flagged = _has_hallucination(trace_lines) or autonomy < float(args.min_autonomy)
        if not flagged:
            skipped += 1
            continue

        prompt = _build_prompt(problem_text, trace_lines)
        try:
            response = _call_ollama(args.model, prompt, args.timeout_s)
        except Exception as exc:  # noqa: BLE001
            feedback_galaxy.add_feedback(
                trace_id=str(trace.get("trace_id") or ""),
                problem_text=problem_text,
                teacher_score=0,
                feedback_text=f"teacher_error: {exc}",
                suggested_rule="",
                metadata={"autonomy": autonomy, "error": True},
            )
            processed += 1
            continue

        payload = _extract_json(response)
        if not payload:
            feedback_galaxy.add_feedback(
                trace_id=str(trace.get("trace_id") or ""),
                problem_text=problem_text,
                teacher_score=0,
                feedback_text=response.strip()[:2000],
                suggested_rule="",
                metadata={"autonomy": autonomy, "parse_error": True},
            )
            processed += 1
            continue

        score = int(payload.get("score", 0))
        score = max(-2, min(2, score))
        feedback_text = str(payload.get("feedback", "")).strip()
        suggested_rule = str(payload.get("suggested_rule", "")).strip()
        if suggested_rule not in ALLOWED_RULES:
            suggested_rule = ""

        feedback_galaxy.add_feedback(
            trace_id=str(trace.get("trace_id") or ""),
            problem_text=problem_text,
            teacher_score=score,
            feedback_text=feedback_text,
            suggested_rule=suggested_rule,
            metadata={"autonomy": autonomy, "model": args.model},
        )
        processed += 1

    feedback_galaxy.to_jsonl(args.output)
    print(
        json.dumps(
            {
                "output": args.output,
                "processed": processed,
                "skipped": skipped,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
