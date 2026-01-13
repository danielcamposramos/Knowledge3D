#!/usr/bin/env python3
"""
Generate synthetic router training data using a local Ollama model.

This script produces RoutingDecision-style JSON entries mapping theorem
pattern semantic tags to grammar rule selections. It is intended to
bootstrap the theorem router specialist with synthetic data only.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


import re

from knowledge3d.cranium.math_galaxy_population import populate_theorem_patterns
from knowledge3d.training.math_benchmarks.calculus_grammar_rules import get_calculus_rules
from knowledge3d.training.math_benchmarks.latex_normalizer import normalize_latex_to_natural
from knowledge3d.training.math_benchmarks.router_embedder import embed_semantic_tags


GRAMMAR_RULES = [
    "apply_power_rule",
    "apply_product_rule",
    "apply_quotient_rule",
    "apply_chain_rule",
    "apply_sum_rule",
    "apply_constant_multiple_rule",
    "apply_integration_by_parts",
    "apply_fundamental_theorem_calculus",
    "apply_pythagorean_identity",
]

NOTATION_HINTS: Dict[str, List[Tuple[str, str]]] = {
    "apply_power_rule": [
        ("natural", "derivative of x^3 at x=2"),
        ("prime", "f'(2) where f(x)=x^3"),
        ("leibniz", "\\frac{d}{dx}[x^3]|_{x=2}"),
    ],
    "apply_constant_multiple_rule": [
        ("natural", "derivative of 5*x^3 at x=2"),
        ("prime", "f'(2) where f(x)=5x^3"),
        ("leibniz", "\\frac{d}{dx}[5x^3]|_{x=2}"),
    ],
    "apply_sum_rule": [
        ("natural", "derivative of x^2 + x^3 at x=2"),
        ("prime", "f'(2) where f(x)=x^2+x^3"),
        ("leibniz", "\\frac{d}{dx}[x^2+x^3]|_{x=2}"),
    ],
    "apply_product_rule": [
        ("natural", "derivative of x^2 * x^3 at x=2"),
        ("prime", "f'(2) where f(x)=x^2*x^3"),
        ("leibniz", "\\frac{d}{dx}[x^2*x^3]|_{x=2}"),
    ],
    "apply_quotient_rule": [
        ("natural", "derivative of x^3 / x^2 at x=2"),
        ("prime", "f'(2) where f(x)=x^3/x^2"),
        ("leibniz", "\\frac{d}{dx}[x^3/x^2]|_{x=2}"),
    ],
    "apply_chain_rule": [
        ("natural", "derivative of (x^2)^3 at x=2"),
        ("prime", "f'(2) where f(x)=(x^2)^3"),
        ("leibniz", "\\frac{d}{dx}[(x^2)^3]|_{x=2}"),
    ],
    "apply_integration_by_parts": [
        ("natural", "integral of x e^x from 0 to 1"),
        ("latex", "\\int_0^1 x e^x dx"),
    ],
    "apply_fundamental_theorem_calculus": [
        ("natural", "integral of x^2 from 0 to 1"),
        ("latex", "\\int_0^1 x^2 dx"),
    ],
    "apply_pythagorean_identity": [
        ("natural", "evaluate sin^2(theta) + cos^2(theta) at theta=2"),
        ("latex", "\\sin^2(2) + \\cos^2(2)"),
    ],
}

STATIC_EXAMPLES: Dict[str, List[str]] = {
    "apply_power_rule": [
        "Find the derivative of x^4 at x=2",
        "Evaluate f'(3) where f(x) = x^5",
        "Calculate \\frac{d}{dx}[x^2] at x=10"
    ],
    "apply_product_rule": [
        "Find the derivative of x^2 * x^3 at x=1",
        "Evaluate f'(2) where f(x) = (x+1)*(x-1)",
        "Calculate \\frac{d}{dx}[x * x^2]|_{x=3}"
    ],
    "apply_quotient_rule": [
        "Given f(x) = (3x-4)/(2x+3), find f'(1)",
        "Evaluate derivative of (x^2)/(x+1) at x=2",
        "Calculate \\frac{d}{dx}[\\frac{x^2}{x}]|_{x=4}"
    ],
    "apply_chain_rule": [
        "Given f(x) = (6x - 4)^(1/3), find f'(2)",
        "Evaluate derivative of (2x+1)^5 at x=0",
        "Calculate \\frac{d}{dx}[(x+1)^2]|_{x=0}"
    ],
    "apply_sum_rule": [
        "Given f(x) = x^3 - 3x^2 + 2x - 5, find f'(2)",
        "Evaluate derivative of x^2 + 4x at x=3",
        "Calculate \\frac{d}{dx}[x^3 + x]|_{x=2}"
    ],
    "apply_constant_multiple_rule": [
        "Find the derivative of 5x^2 at x=3",
        "Evaluate f'(1) where f(x) = 10x^3",
        "Calculate \\frac{d}{dx}[4x]|_{x=5}"
    ],
    "apply_integration_by_parts": [
        "Evaluate the integral of x e^x from 0 to 1",
        "Calculate \\int_0^1 x e^x dx"
    ],
    "apply_fundamental_theorem_calculus": [
        "Evaluate the integral of x^2 from 1 to 3",
        "Calculate \\int_0^2 x^3 dx"
    ],
    "apply_pythagorean_identity": [
        "Evaluate sin^2(5) + cos^2(5)",
        "Calculate sin^2(x) + cos^2(x) at x=3"
    ]
}

PROMPT_TEMPLATE = """You are a calculus expert helping train a routing model.

Given a theorem pattern with semantic tags, identify which grammar rule should be used.

CRITICAL INSTRUCTION: Generate a NUMERIC EVALUATION problem.
The problem MUST ask for a result at a specific point (e.g., "at x=2").
Do NOT generate symbolic problems like "Find f'(x)".

When reasoning, think in LaTeX notation (e.g., \\frac{{d}}{{dx}}, f'(x), \\frac{{A}}{{B}}, \\int).
Choose the grammar rule that matches the theorem pattern name (product_rule → apply_product_rule).

Target notation: {notation_name}
Notation example: {notation_example}

EXAMPLES OF VALID PROBLEMS:
{context_examples}

Theorem Pattern: {pattern_id}
Semantic Tags: {semantic_tags}
Domain: {domain}
Description: {description}

Available Grammar Rules:
1. apply_power_rule - derivative of x^n evaluated at x=a
2. apply_product_rule - derivative of f*g evaluated at x=a
3. apply_quotient_rule - derivative of f/g evaluated at x=a
4. apply_chain_rule - derivative of f(g(x)) evaluated at x=a
5. apply_sum_rule - derivative of f+g evaluated at x=a
6. apply_constant_multiple_rule - derivative of c*f evaluated at x=a
7. apply_integration_by_parts - integral using integration by parts (definite integral)
8. apply_fundamental_theorem_calculus - definite integral evaluation
9. apply_pythagorean_identity - sin^2(theta) + cos^2(theta) = 1

Return a JSON object with fields:
  problem: "<numeric evaluation question>"
  rule: "<grammar rule name>"

Answer with ONLY the JSON object (no extra text).
"""


def _describe_pattern(pattern: Dict[str, Any]) -> str:
    precond = pattern.get("precondition", {}) or {}
    lhs = pattern.get("transformation", {}).get("lhs")
    rhs = pattern.get("transformation", {}).get("rhs")
    cues = precond.get("context_cues", []) or []
    parts = []
    if cues:
        parts.append(f"context cues: {', '.join(cues[:6])}")
    if lhs or rhs:
        parts.append(f"transform: {lhs} -> {rhs}")
    return "; ".join(parts) if parts else pattern.get("pattern_id", "theorem pattern")


def _embed_tags(tags: Iterable[str], dim: int) -> List[float]:
    return embed_semantic_tags(tags, dim=dim)


def _add_semantic_noise(tags: List[str], noise_level: float) -> List[str]:
    if not tags or noise_level <= 0.0:
        return list(tags)
    tags = list(tags)
    if random.random() < noise_level and len(tags) > 1:
        tags.pop(random.randrange(len(tags)))
    if random.random() < noise_level:
        tags.append(random.choice(tags))
    return tags


def _base_rule_id(rule_id: str) -> str:
    suffixes = {"natural", "prime", "leibniz", "latex"}
    parts = rule_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in suffixes:
        return parts[0]
    return rule_id


def _normalize_rule_name(rule_id: str) -> Optional[str]:
    if not rule_id:
        return None
    cleaned = rule_id.strip().lower().strip('"').strip("'")
    if cleaned in GRAMMAR_RULES:
        return cleaned
    base = _base_rule_id(cleaned)
    if base in GRAMMAR_RULES:
        return base
    return None


def _parse_response(response_text: str) -> Tuple[Optional[str], Optional[str]]:
    if not response_text:
        return None, None
    # Try JSON object first.
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            problem = payload.get("problem") or payload.get("question")
            rule = payload.get("rule")
            return problem, rule
        except Exception:
            pass
    # Fallback: parse "Problem:" / "Rule:" lines.
    problem_match = re.search(r"problem\s*:\s*(.+)", response_text, re.IGNORECASE)
    rule_match = re.search(r"rule\s*:\s*(.+)", response_text, re.IGNORECASE)
    problem = problem_match.group(1).strip() if problem_match else None
    rule = rule_match.group(1).strip() if rule_match else None
    return problem, rule


def _is_numeric_problem(text: str) -> bool:
    """Check if the generated text looks like a numeric evaluation problem."""
    # Must contain explicit evaluation cues or definite integral bounds.
    if re.search(r"at\s+[a-zA-Z]\s*=\s*[-+]?\d", text, re.IGNORECASE):
        return True
    if re.search(r"from\s+\d+\s+to\s+\d+", text, re.IGNORECASE):
        return True
    if re.search(r"[a-zA-Z]+\s*\(\s*[-+]?\d+\.?\d*\s*\)", text):
        return True
    if re.search(r"evaluate", text, re.IGNORECASE) and re.search(r"\d", text):
        return True
    return False


def _build_rule_groups() -> Dict[str, List[Any]]:
    groups: Dict[str, List[Any]] = {}
    for rule in get_calculus_rules():
        base = _base_rule_id(rule.rule_id)
        groups.setdefault(base, []).append(rule)
    return groups


def _match_variant(
    base_rule: str,
    *,
    raw_text: str,
    normalized_text: str,
    rule_groups: Dict[str, List[Any]],
) -> Optional[str]:
    candidates = rule_groups.get(base_rule, [])
    for rule in candidates:
        try:
            if re.search(rule.pattern, normalized_text, re.IGNORECASE | re.DOTALL):
                return rule.rule_id
            if re.search(rule.pattern, raw_text, re.IGNORECASE | re.DOTALL):
                return rule.rule_id
        except re.error:
            continue
    return None


def _query_ollama(prompt: str, model: str, fallback_model: Optional[str]) -> str:
    try:
        import ollama  # type: ignore

        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.0},
        )
        payload = str(response.get("response") or "")
        if payload:
            return payload
        if fallback_model:
            return _query_ollama_cli(prompt, fallback_model)
        return payload
    except Exception:
        if fallback_model:
            return _query_ollama_cli(prompt, fallback_model)
        return _query_ollama_cli(prompt, model)


def _query_ollama_cli(prompt: str, model: str) -> str:
    cmd = ["ollama", "run", model, prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Ollama CLI failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _select_notation_hint(base_rule: str) -> Tuple[str, str]:
    hints = NOTATION_HINTS.get(base_rule)
    if hints:
        return random.choice(hints)
    return "natural", "derivative of x^2 at x=2"


def _get_context_examples(base_rule: str) -> str:
    examples = STATIC_EXAMPLES.get(base_rule, [])
    if not examples:
        return "No examples available."
    return "\n".join(f"- {ex}" for ex in examples)


def generate_training_dataset(
    theorem_patterns: List[Dict[str, Any]],
    *,
    examples_per_pattern: int,
    ollama_model: str,
    fallback_model: Optional[str],
    noise_level: float,
    embedding_dim: int,
) -> List[Dict[str, Any]]:
    decisions: List[Dict[str, Any]] = []
    rule_groups = _build_rule_groups()
    for pattern in theorem_patterns:
        tags = list(pattern.get("semantic_tags") or [])
        if not tags:
            continue
        base_rule = str(pattern.get("grammar_rule") or "").strip().lower()
        if not base_rule:
            continue
        
        valid_count = 0
        max_attempts_per_pattern = examples_per_pattern * 5  # Allow plenty of retries
        attempts = 0

        while valid_count < examples_per_pattern and attempts < max_attempts_per_pattern:
            attempts += 1
            noisy_tags = _add_semantic_noise(tags, noise_level)
            notation_name, notation_example = _select_notation_hint(base_rule)
            context_examples = _get_context_examples(base_rule)
            
            prompt = PROMPT_TEMPLATE.format(
                pattern_id=pattern.get("pattern_id"),
                semantic_tags=", ".join(noisy_tags),
                domain=pattern.get("domain"),
                description=_describe_pattern(pattern),
                notation_name=notation_name,
                notation_example=notation_example,
                context_examples=context_examples,
            )
            response = _query_ollama(prompt, ollama_model, fallback_model)
            problem_text, rule = _parse_response(response)
            
            if not problem_text:
                print(
                    f"[Reject] {pattern.get('pattern_id')} (Attempt {attempts}): Missing problem text in JSON response."
                )
                continue
            
            normalized = normalize_latex_to_natural(problem_text)
            if not _is_numeric_problem(normalized):
                print(
                    f"[Reject] {pattern.get('pattern_id')} (Attempt {attempts}): Not numeric. Text: {problem_text[:60]}..."
                )
                continue
            matched_rule = _match_variant(
                base_rule,
                raw_text=problem_text,
                normalized_text=normalized,
                rule_groups=rule_groups,
            )
            
            if not matched_rule:
                print(
                    f"[Reject] {pattern.get('pattern_id')} (Attempt {attempts}): No grammar match. Norm: {normalized[:60]}..."
                )
                continue

            embedding = _embed_tags(noisy_tags, embedding_dim)
            decisions.append(
                {
                    "input_data": embedding,
                    "task_description": f"{pattern.get('pattern_id')} pattern (numeric)",
                    "specialist_weights": {matched_rule: 1.0},
                    "outcome_performance": 1.0,
                    "timestamp": datetime.now().isoformat(),
                    "semantic_tags": noisy_tags,
                    "pattern_id": pattern.get("pattern_id"),
                    "selected_rule": matched_rule,
                    "problem_text": problem_text,
                    "normalized_problem": normalized,
                }
            )
            print(f"[Ollama] {pattern.get('pattern_id')} {valid_count + 1}/{examples_per_pattern} -> {matched_rule}")
            valid_count += 1
            
        if valid_count < examples_per_pattern:
            print(
                f"Warning: Could not generate enough valid samples for {pattern.get('pattern_id')} (got {valid_count}/{examples_per_pattern})"
            )

    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate router training data via Ollama.")
    parser.add_argument(
        "--artifact-dirs",
        nargs="+",
        default=["/K3D/Knowledge3D.local/galaxies/books_v5_clean2"],
        help="Artifact directories to extract theorem patterns from.",
    )
    parser.add_argument("--min-examples", type=int, default=1, help="Min examples per theorem pattern.")
    parser.add_argument("--examples-per-pattern", type=int, default=40, help="Synthetic examples per pattern.")
    parser.add_argument("--model", type=str, default="qwen2.5:14b", help="Ollama model name.")
    parser.add_argument(
        "--fallback-model",
        type=str,
        default="qwen2.5:7b",
        help="Fallback Ollama model if the primary fails.",
    )
    parser.add_argument("--noise-level", type=float, default=0.1, help="Semantic tag noise level.")
    parser.add_argument("--embedding-dim", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for noise.")
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/ollama_router_training_data.json",
        help="Output JSON path for synthetic routing decisions.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    theorem_patterns = populate_theorem_patterns(
        artifact_dirs=args.artifact_dirs,
        min_examples=int(args.min_examples),
    )
    if not theorem_patterns:
        raise SystemExit("No theorem patterns found. Check artifact dirs or min-examples.")

    print(f"[Ollama] Generating data for {len(theorem_patterns)} theorem patterns")
    decisions = generate_training_dataset(
        theorem_patterns,
        examples_per_pattern=int(args.examples_per_pattern),
        ollama_model=args.model,
        fallback_model=args.fallback_model,
        noise_level=float(args.noise_level),
        embedding_dim=int(args.embedding_dim),
    )
    if not decisions:
        raise SystemExit("No synthetic routing decisions generated.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    print(f"[Ollama] Wrote {len(decisions)} synthetic decisions to {output_path}")


if __name__ == "__main__":
    main()
