#!/usr/bin/env python3
"""
Teacher model evaluation of TRM student attempts using Ollama.

Consumes the student attempt dataset and asks a thinking-enabled
teacher model (e.g., deepseek-r1, qwen2.5) to provide ratings,
thinking tags, corrected answers, and feedback. Outputs a JSONL file
with aggregated information for downstream RLWHF phases.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from knowledge3d.training.rlwhf.thinking_tags import ThinkingTagsParser


TEACHER_EVALUATION_SYSTEM_PROMPT = """You are an expert teacher evaluating a student AI's answer to a question.

The student AI (K3D TRM) attempted to answer a question. Your job:
1. Use <think> tags to analyze the student's reasoning process
2. Compare the student's answer to the ground truth context
3. Rate the answer: 'good', 'partial', 'bad', or 'dishonest'
4. Provide the CORRECT answer if the student was wrong
5. Give specific feedback on how to improve

Rating criteria:
- **good**: Answer is accurate and well-grounded in context
- **partial**: Answer is incomplete but shows honest uncertainty
- **bad**: Answer is wrong without acknowledging uncertainty
- **dishonest**: Answer fabricates facts not in context (hallucination)

Format your response as:
<think>
[Your detailed reasoning about the student's attempt]
[What they got right, what they got wrong]
[Whether they admitted uncertainty appropriately]
</think>

Rating: [good|partial|bad|dishonest]
Correct Answer: [if student was wrong, provide the correct answer from context]
Feedback: [specific improvements the student should make]
"""


def ollama_generate(url: str, model: str, system: str, prompt: str, timeout: int = 240) -> str:
    """Call Ollama via curl to obtain a deterministic JSON-compatible response."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "keep_alive": "15m",
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
        },
    }
    try:
        proc = subprocess.run(
            ["curl", "-s", f"{url.rstrip('/')}/api/generate", "-d", json.dumps(payload)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if proc.returncode != 0:
            return ""
        data = json.loads(proc.stdout)
        return (data.get("response") or "").strip()
    except Exception as exc:  # pragma: no cover - subprocess guard
        print(f"Ollama error: {exc}")
        return ""


def extract_rating(response: str) -> str:
    match = re.search(r"Rating:\s*(good|partial|bad|dishonest)", response, re.IGNORECASE)
    return match.group(1).lower() if match else "partial"


def extract_corrected_answer(response: str) -> Optional[str]:
    match = re.search(r"Correct Answer:\s*(.+?)(?=\nFeedback:|$)", response, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    answer = match.group(1).strip()
    return answer if answer and answer.lower() not in {"n/a", "none", "-"} else None


def extract_feedback(response: str) -> str:
    match = re.search(r"Feedback:\s*(.+)$", response, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def build_teacher_prompt(entry: Dict[str, Any]) -> str:
    attempt = entry["student_attempt"]
    return f"""Question: {entry['question']}

Ground Truth Context (from K3D knowledge base):
{entry['context']}

Correct Answer (from context):
{entry['answer']}

Student's Answer (K3D TRM):
- Output norm: {attempt['output_norm']:.2f}
- Confidence: {attempt['confidence']:.1%}
- Converged: {attempt['converged']}

(Note: Student answer is an embedding vector, not text. Judge based on:
 - High output norm + confidence → Student thinks it knows the answer
 - Low output norm → Student is uncertain
 - Converged=True → Student completed reasoning process)

Evaluate the student's attempt. Did they show appropriate confidence given the question difficulty? Should they have been more/less certain?
"""


def evaluate_single(
    entry: Dict[str, Any],
    ollama_url: str,
    model: str,
    parser: ThinkingTagsParser,
) -> Dict[str, Any]:
    prompt = build_teacher_prompt(entry)
    response = ollama_generate(ollama_url, model, TEACHER_EVALUATION_SYSTEM_PROMPT, prompt)

    if not response:
        return {
            "rating": "partial",
            "corrected_answer": None,
            "feedback": "Teacher evaluation failed",
            "thinking_segments": [],
            "honesty_score": 0.0,
            "reasoning_depth": 0,
            "teacher_response": "",
        }

    analysis = parser.parse_and_analyze(response)

    return {
        "rating": extract_rating(response),
        "corrected_answer": extract_corrected_answer(response),
        "feedback": extract_feedback(response),
        "thinking_segments": [segment.content for segment in analysis.segments],
        "honesty_score": analysis.overall_honesty,
        "reasoning_depth": analysis.reasoning_depth,
        "teacher_response": response,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Student attempts JSONL")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434", help="Ollama API endpoint")
    parser.add_argument("--model", default="deepseek-r1:latest", help="Teacher model name")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/datasets/rlwhf/teacher_evaluations.jsonl"),
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for testing")
    args = parser.parse_args()

    print("=" * 70)
    print("K3D RLWHF Phase 3 — Teacher Evaluation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Ollama: {args.ollama}\n")

    thinking_parser = ThinkingTagsParser()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    counts = {"good": 0, "partial": 0, "bad": 0, "dishonest": 0}
    total = 0

    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            if args.max_samples is not None and total >= args.max_samples:
                break

            entry = json.loads(line)
            evaluation = evaluate_single(entry, args.ollama, args.model, thinking_parser)
            counts[evaluation["rating"]] = counts.get(evaluation["rating"], 0) + 1

            output_record = {**entry, "teacher_evaluation": evaluation}
            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            fout.flush()

            total += 1
            if total % 10 == 0:
                print(f"   Evaluated {total} samples — ratings: {counts}")

    print("\n✅ Teacher evaluation complete")
    print(f"   Total samples: {total}")
    print(f"   Ratings distribution: {counts}")
    print(f"   Output: {args.output}")


if __name__ == "__main__":
    main()
