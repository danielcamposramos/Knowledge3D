# Phase B2c — Ollama Benchmark with Model Routing + RAG Context

**Depends on:** B2b (Benchmark Provider Bridge), H17 (Universal Knowledge Foundation)
**Modifies:** `knowledge3d/tools/benchmark_provider_bridge.py`, `knowledge3d/ingestion/ollama_manager.py`
**Creates:** `knowledge3d/tools/ollama_benchmark.py`, `tests/test_ollama_benchmark.py`
**Goal:** Run all benchmark suites through Ollama with two-tier model routing, tailored system prompts, and H17-backed RAG context to minimize hallucination

---

## Objective

Test the Ollama pathway specifically:
1. **Two-tier model routing** — medium model for quick/easy tasks, large model for complex reasoning
2. **Tailored system prompts** — suite-specific instructions that extract clean, parseable answers
3. **RAG context injection** — pull relevant H17 knowledge (periodic table, constants, measurements) into the prompt so the model has grounding facts and doesn't hallucinate
4. **HTTP API** — switch from `ollama run` CLI to `/api/chat` for system prompt support and structured messages

---

## Model Selection

Available locally on RTX 3070 (8GB VRAM):

| Tier | Model | Size | Use For | Why |
|------|-------|------|---------|-----|
| **Medium** | `qwen3:8b` | 5.2 GB | MMLU multiple-choice, simple GSM8K | Fast, good accuracy on factual recall. Use `think: false` option to skip chain-of-thought for speed |
| **Large** | `qwen2.5:32b` | 19 GB | LHE, ARC, Math competitions, hard GSM8K | Best local reasoner. Needs the full thinking capacity for multi-hop and pattern tasks |

**Routing logic:**
```python
SUITE_MODEL_MAP = {
    "mmlu": "qwen3:8b",        # Multiple choice — fast model sufficient
    "gsm8k": "qwen2.5:32b",    # Word problems — needs reasoning
    "math": "qwen2.5:32b",     # Competition math — needs deep reasoning
    "lhe": "qwen2.5:32b",      # Multi-hop — hardest suite
    "arc": "qwen2.5:32b",      # Pattern recognition — needs spatial reasoning
}
```

---

## Architecture: Ollama HTTP API

**CRITICAL:** The current `OllamaModelManager.query()` uses `subprocess.run(["ollama", "run", ...])` which:
- Cannot send system prompts (no message separation)
- Cannot control generation parameters (temperature, think mode)
- Mixes spinner/ANSI output with model output

**Solution:** Add an HTTP API method to `OllamaModelManager` using `urllib` (stdlib, no new deps):

### Modify: `knowledge3d/ingestion/ollama_manager.py`

Add a `chat()` method alongside the existing `query()`:

```python
import urllib.request
import urllib.error

def chat(
    self,
    model: str,
    messages: list[dict[str, str]],
    *,
    timeout: float | None = None,
    temperature: float = 0.3,
    options: dict[str, Any] | None = None,
) -> OllamaQueryResult:
    """Send a structured chat request via Ollama HTTP API.

    Uses /api/chat endpoint which supports system/user/assistant messages,
    temperature control, and clean JSON output (no ANSI spinner garbage).
    """
    run_timeout = timeout if timeout is not None else self.default_timeout
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            **(options or {}),
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=run_timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data.get("message", {}).get("content", "")
            return OllamaQueryResult(
                model=model,
                output=content.strip(),
                returncode=0,
                stderr="",
            )
    except urllib.error.URLError as exc:
        return OllamaQueryResult(
            model=model, output="", returncode=1, stderr=str(exc),
        )
    except Exception as exc:
        return OllamaQueryResult(
            model=model, output="", returncode=1, stderr=str(exc),
        )
```

**NOTE:** Add `import json` to the existing imports in `ollama_manager.py`. The `Any` type is already available or add `from typing import Any`.

---

## New File: `knowledge3d/tools/ollama_benchmark.py`

This is the main orchestrator — model routing + system prompts + RAG context + answer extraction.

```python
"""Ollama-specific benchmark runner with two-tier model routing and RAG context."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from knowledge3d.ingestion.ollama_manager import OllamaModelManager


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

SUITE_MODEL_MAP: dict[str, str] = {
    "mmlu": "qwen3:8b",
    "gsm8k": "qwen2.5:32b",
    "math": "qwen2.5:32b",
    "lhe": "qwen2.5:32b",
    "arc": "qwen2.5:32b",
}

# Options per model tier
MODEL_OPTIONS: dict[str, dict[str, Any]] = {
    "qwen3:8b": {
        "temperature": 0.1,       # Low temp for factual recall
        "num_predict": 256,       # Short answers for MMLU
    },
    "qwen2.5:32b": {
        "temperature": 0.3,       # Slightly higher for reasoning
        "num_predict": 1024,      # Longer for step-by-step
    },
}


def get_model_for_suite(suite: str) -> str:
    """Return the appropriate Ollama model for a benchmark suite."""
    return SUITE_MODEL_MAP.get(suite, "qwen2.5:32b")


# ---------------------------------------------------------------------------
# RAG context builder (H17 knowledge)
# ---------------------------------------------------------------------------

def build_rag_context(row: dict[str, Any], suite: str) -> str:
    """Build grounding context from H17 universal knowledge registries.

    Pulls relevant facts based on question domain/subject so the model
    has authoritative data and doesn't hallucinate.
    """
    context_parts: list[str] = []
    subject = str(row.get("payload", {}).get("subject", "")).lower()
    question = str(row.get("question", "")).lower()

    # Chemistry / periodic table
    if _mentions_chemistry(subject, question):
        context_parts.append(_chemistry_context(question))

    # Physics / constants
    if _mentions_physics(subject, question):
        context_parts.append(_physics_context())

    # Mathematics / measurements
    if _mentions_math(subject, question):
        context_parts.append(_math_context(question))

    # General science
    if _mentions_science(subject, question):
        context_parts.append(_science_context(question))

    if not context_parts:
        return ""

    header = "=== REFERENCE FACTS (use these to ground your answer) ==="
    return header + "\n" + "\n".join(context_parts)


def _mentions_chemistry(subject: str, question: str) -> bool:
    keywords = {"chemistry", "element", "periodic", "atom", "molecule", "compound",
                "electron", "proton", "neutron", "ion", "bond", "reaction",
                "oxide", "acid", "base", "metal", "noble gas", "halogen"}
    combined = f"{subject} {question}"
    return any(kw in combined for kw in keywords)


def _mentions_physics(subject: str, question: str) -> bool:
    keywords = {"physics", "force", "energy", "velocity", "acceleration", "gravity",
                "speed of light", "planck", "boltzmann", "newton", "coulomb",
                "magnetic", "electric", "thermodynamic", "momentum", "wavelength"}
    combined = f"{subject} {question}"
    return any(kw in combined for kw in keywords)


def _mentions_math(subject: str, question: str) -> bool:
    keywords = {"mathematics", "algebra", "calculus", "geometry", "trigonometry",
                "probability", "statistics", "equation", "integral", "derivative",
                "matrix", "vector", "polynomial"}
    combined = f"{subject} {question}"
    return any(kw in combined for kw in keywords)


def _mentions_science(subject: str, question: str) -> bool:
    keywords = {"biology", "astronomy", "geology", "ecology", "genetics",
                "anatomy", "clinical", "medicine", "nutrition"}
    combined = f"{subject} {question}"
    return any(kw in combined for kw in keywords)


def _chemistry_context(question: str) -> str:
    """Pull relevant periodic table entries."""
    try:
        from knowledge3d.ingestion.universal_knowledge.periodic_table import ELEMENTS
    except ImportError:
        return ""

    # Find mentioned elements
    mentioned = []
    q_lower = question.lower()
    for element in ELEMENTS:
        if element.name.lower() in q_lower or element.symbol.lower() in q_lower:
            mentioned.append(element)

    if not mentioned:
        # Provide a compact summary of common elements
        common_symbols = ["H", "He", "C", "N", "O", "Na", "Mg", "Al", "Si",
                          "P", "S", "Cl", "K", "Ca", "Fe", "Cu", "Zn", "Ag", "Au"]
        for element in ELEMENTS:
            if element.symbol in common_symbols:
                mentioned.append(element)

    lines = ["Periodic Table Reference:"]
    for el in mentioned[:20]:  # Cap at 20 to keep context manageable
        lines.append(
            f"  {el.symbol} ({el.name}): Z={el.atomic_number}, "
            f"mass={el.atomic_mass:.4f}, group={el.group}, period={el.period}, "
            f"block={el.block}, category={el.category}"
        )
    return "\n".join(lines)


def _physics_context() -> str:
    """Pull physical constants."""
    try:
        from knowledge3d.ingestion.universal_knowledge.physical_constants import PHYSICAL_CONSTANTS
    except ImportError:
        return ""

    lines = ["Physical Constants Reference:"]
    for const in PHYSICAL_CONSTANTS:
        exact_tag = " (exact)" if const.exact else ""
        lines.append(f"  {const.name} ({const.symbol}): {const.value} {const.unit}{exact_tag}")
    return "\n".join(lines)


def _math_context(question: str) -> str:
    """Pull relevant measurement/conversion facts."""
    try:
        from knowledge3d.ingestion.universal_knowledge.measurements import MEASUREMENT_DOMAINS
    except ImportError:
        return ""

    lines = ["Measurement Reference:"]
    q_lower = question.lower()
    for domain in MEASUREMENT_DOMAINS:
        # Include domain if question mentions its units
        domain_relevant = domain.key.lower() in q_lower
        if not domain_relevant:
            for unit_name in domain.units:
                if unit_name.replace("_", " ") in q_lower:
                    domain_relevant = True
                    break
        if domain_relevant:
            lines.append(f"  Domain: {domain.key} (SI base: {domain.si_base})")
            for unit_name, unit in domain.units.items():
                lines.append(f"    {unit_name} ({unit.symbol}): to_SI = {unit.to_si_rpn}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _science_context(question: str) -> str:
    """Provide relevant scientific context snippets."""
    # Lightweight: just flag that we have foundational knowledge available
    # The system prompt will tell the model to reason carefully
    return "Note: Answer based on established scientific consensus and textbook knowledge."


# ---------------------------------------------------------------------------
# System prompts (suite-specific)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {
    "mmlu": (
        "You are a precise academic exam assistant. "
        "You will be given a multiple-choice question with options A, B, C, D. "
        "If reference facts are provided, use them to verify your answer. "
        "Think briefly, then on the LAST line write ONLY the single letter (A, B, C, or D) "
        "of the correct answer. Nothing else on the last line."
    ),
    "gsm8k": (
        "You are a math word problem solver. "
        "Read the problem carefully. Solve step by step. "
        "If reference facts are provided, use them for unit conversions or constants. "
        "On the VERY LAST line, write ONLY the final numeric answer "
        "(no units, no text, no dollar signs, just the number)."
    ),
    "math": (
        "You are a competition mathematics solver. "
        "Show your work step by step. "
        "If reference facts are provided, use them for constants or formulas. "
        "On the VERY LAST line, write ONLY the final numeric answer."
    ),
    "lhe": (
        "You are answering a difficult multi-hop reasoning question. "
        "Think carefully and cross-reference facts. "
        "If reference facts are provided, use them as authoritative sources. "
        "Give a concise answer — ideally a single word, phrase, or short sentence."
    ),
    "arc": (
        "You are solving an ARC-AGI visual pattern recognition task. "
        "You will see training examples (input grid → output grid) and a test input. "
        "Find the transformation pattern and apply it to the test input. "
        "Output ONLY the resulting grid as a JSON array of arrays (e.g., [[1,0],[0,1]]). "
        "No explanation, just the JSON grid."
    ),
}


# ---------------------------------------------------------------------------
# Answer extractors (suite-specific)
# ---------------------------------------------------------------------------

def extract_answer(raw: str, suite: str) -> str:
    """Extract the actual answer from model output.

    Suite-specific parsing to pull the evaluator-expected format.
    """
    text = str(raw or "").strip()
    if not text:
        return ""

    # Strip thinking blocks if present (qwen3 outputs <think>...</think>)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if suite == "mmlu":
        return _extract_mmlu_letter(text)
    elif suite in ("gsm8k", "math"):
        return _extract_last_number(text)
    elif suite == "arc":
        return _extract_json_grid(text)
    elif suite == "lhe":
        # First non-empty line after stripping
        lines = text.strip().splitlines()
        return lines[-1].strip() if lines else text[:200]
    else:
        return text[:200]


def _extract_mmlu_letter(text: str) -> str:
    """Extract single letter A/B/C/D from MMLU response."""
    lines = text.strip().splitlines()
    # Check last line for standalone letter
    for line in reversed(lines):
        cleaned = line.strip().rstrip(".").strip()
        if cleaned.upper() in ("A", "B", "C", "D"):
            return cleaned.upper()

    # Pattern: "The answer is X" or "Answer: X"
    match = re.search(r"(?:answer\s*(?:is|:)\s*)([A-Da-d])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # First standalone letter in response
    match = re.search(r"\b([A-Da-d])\b", text)
    if match:
        return match.group(1).upper()

    return text[:50]


def _extract_last_number(text: str) -> str:
    """Extract the last number from a step-by-step solution."""
    lines = text.strip().splitlines()
    for line in reversed(lines):
        # Find numbers (possibly negative, decimal, with commas)
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", line)
        if numbers:
            return numbers[-1].replace(",", "")

    # Fallback: any number in the response
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return text[:50]


def _extract_json_grid(text: str) -> str:
    """Extract JSON grid array from ARC response."""
    try:
        match = re.search(r"\[\s*\[.*?\]\s*\]", text, re.DOTALL)
        if match:
            grid = json.loads(match.group())
            return json.dumps(grid)
    except (json.JSONDecodeError, AttributeError):
        pass
    return text[:500]


# ---------------------------------------------------------------------------
# Main benchmark query function
# ---------------------------------------------------------------------------

def create_ollama_query_fn(
    *,
    medium_model: str = "qwen3:8b",
    large_model: str = "qwen2.5:32b",
    timeout: float = 120.0,
    use_rag: bool = True,
) -> callable:
    """Create a benchmark query function that routes through Ollama with RAG.

    Returns a function compatible with benchmark_health_check.run_health_check(query_fn=...).

    Args:
        medium_model: Fast model for easy suites (MMLU).
        large_model: Heavy model for complex suites (GSM8K, Math, LHE, ARC).
        timeout: Per-query timeout in seconds.
        use_rag: Whether to inject H17 RAG context into prompts.
    """
    ollama = OllamaModelManager(default_timeout=timeout)

    def query_fn(row: dict[str, Any]) -> dict[str, Any]:
        suite = str(row.get("suite", "mmlu")).strip()

        # If no suite field, infer from question id
        if suite == "mmlu":
            qid = str(row.get("id", ""))
            if qid.startswith("gsm8k"):
                suite = "gsm8k"
            elif qid.startswith("math"):
                suite = "math"
            elif qid.startswith("lhe"):
                suite = "lhe"
            elif qid.startswith("arc"):
                suite = "arc"

        # Select model
        model = SUITE_MODEL_MAP.get(suite, large_model)
        if model == "qwen3:8b":
            model = medium_model
        elif model == "qwen2.5:32b":
            model = large_model

        # Build system prompt
        system_prompt = SYSTEM_PROMPTS.get(suite, SYSTEM_PROMPTS["lhe"])

        # Build user message
        user_message = _build_user_message(row, suite)

        # Add RAG context
        if use_rag:
            rag_context = build_rag_context(row, suite)
            if rag_context:
                user_message = f"{rag_context}\n\n---\n\n{user_message}"

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Model-specific options
        options = dict(MODEL_OPTIONS.get(model, {}))

        # Call Ollama HTTP API
        result = ollama.chat(
            model=model,
            messages=messages,
            timeout=timeout,
            options=options,
        )

        if result.returncode != 0:
            return {
                "answer": "",
                "provider": f"ollama:{model}",
                "error": result.stderr,
            }

        answer = extract_answer(result.output, suite)
        return {
            "answer": answer,
            "provider": f"ollama:{model}",
            "raw_output": result.output[:500],
            "model": model,
            "suite": suite,
            "used_rag": use_rag and bool(build_rag_context(row, suite)),
        }

    return query_fn


def _build_user_message(row: dict[str, Any], suite: str) -> str:
    """Build the user message from a benchmark row."""
    question = str(row.get("question", "")).strip()
    payload = row.get("payload", {})

    if suite == "mmlu":
        options = payload.get("options", [])
        if options:
            option_text = "\n".join(f"  {chr(65+i)}. {opt}" for i, opt in enumerate(options))
            return f"{question}\n\n{option_text}"
        return question

    elif suite == "arc":
        train = payload.get("train", [])
        test_input = payload.get("test", [{}])[0].get("input", [])
        return (
            f"Training examples (input → output):\n{json.dumps(train, indent=2)}\n\n"
            f"Test input:\n{json.dumps(test_input)}\n\n"
            "Predict the output grid."
        )

    else:
        return question


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Run Ollama benchmarks from command line.

    Usage:
        python -m knowledge3d.tools.ollama_benchmark --suite mmlu --count 50
        python -m knowledge3d.tools.ollama_benchmark --suite gsm8k --count 10
        python -m knowledge3d.tools.ollama_benchmark --suite all --count 10
    """
    import argparse
    from pathlib import Path
    from knowledge3d.tools.benchmark_health_check import run_health_check

    parser = argparse.ArgumentParser(description="Run benchmarks through Ollama with RAG")
    parser.add_argument("--suite", required=True,
                        help="Benchmark suite: mmlu, gsm8k, math, lhe, arc, or 'all'")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of questions per suite")
    parser.add_argument("--medium-model", default="qwen3:8b",
                        help="Fast model for easy suites")
    parser.add_argument("--large-model", default="qwen2.5:32b",
                        help="Heavy model for complex suites")
    parser.add_argument("--no-rag", action="store_true",
                        help="Disable RAG context injection")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Per-query timeout in seconds")
    parser.add_argument("--log", type=Path,
                        default=Path("../Knowledge3D.local/logs/health_log_ollama.jsonl"),
                        help="JSONL log path")
    args = parser.parse_args(argv)

    query_fn = create_ollama_query_fn(
        medium_model=args.medium_model,
        large_model=args.large_model,
        timeout=args.timeout,
        use_rag=not args.no_rag,
    )

    suites = ["mmlu", "gsm8k", "math", "lhe", "arc"] if args.suite == "all" else [args.suite]
    all_results = []

    for suite in suites:
        print(f"\n{'='*60}")
        print(f"Running {suite.upper()} ({args.count} questions)")
        print(f"Model: {get_model_for_suite(suite)}")
        print(f"RAG: {'enabled' if not args.no_rag else 'disabled'}")
        print(f"{'='*60}")

        summary = run_health_check(suite, args.count, args.log, query_fn=query_fn)
        all_results.append(summary)
        print(json.dumps(summary, indent=2))

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("COMBINED RESULTS")
        print(f"{'='*60}")
        total_correct = sum(r["correct"] for r in all_results)
        total_questions = sum(r["total"] for r in all_results)
        for r in all_results:
            print(f"  {r['suite']:>6}: {r['score']}")
        print(f"  {'TOTAL':>6}: {total_correct}/{total_questions}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## Modify: `knowledge3d/tools/benchmark_health_check.py`

Add `--provider ollama` and `--ollama` shortcut:

```python
# In _parse_args(), add:
parser.add_argument(
    "--provider",
    default=None,
    help="Run via augmentation provider (auto, ollama, claude, gpt, etc.)",
)

# In main(), BEFORE run_health_check call:
query_fn = None
if args.provider:
    if args.provider.lower() == "ollama":
        from knowledge3d.tools.ollama_benchmark import create_ollama_query_fn
        query_fn = create_ollama_query_fn()
    else:
        from knowledge3d.tools.benchmark_provider_bridge import create_provider_query_fn
        query_fn = create_provider_query_fn(args.provider)
summary = run_health_check(args.suite, args.count, args.log, query_fn=query_fn)
```

---

## MMLU Answer Evaluation Fix

**IMPORTANT:** The current `evaluate_answer()` for MMLU compares the model's answer text against `expected` (which is the full answer text like "Water"). But our extractor returns a LETTER (A/B/C/D). We need to handle both:

The `load_questions()` function sets `expected` to `question["correct_answer"]` which is the full text. But the MMLU payload also has `correct_letter`. We need to pass the letter through:

```python
# In load_questions(), for mmlu section, change expected to include letter:
# The row already has payload with correct_letter
# The extract_answer returns a letter, but evaluate_answer compares strings
# Solution: add correct_letter to the row

# In load_questions mmlu block:
return [
    {
        "id": question["id"],
        "question": question["question_text"],
        "expected": question["correct_letter"],  # <-- CHANGE: use letter, not full text
        "payload": question,
        "suite": "mmlu",  # <-- ADD: so query_fn knows the suite
    }
    for question in bench.questions[:limit]
]
```

**AND** for ALL suites, add the `suite` field to `load_questions` output rows so the query_fn can route correctly:

```python
# Each suite's return should include "suite": canonical_name
# gsm8k rows: add "suite": "gsm8k"
# math rows: add "suite": "math"
# lhe rows: add "suite": "lhe"
# arc rows: add "suite": "arc"
```

---

## Health Log Entry Format (Ollama Path)

```json
{
  "question_id": "mmlu_elementary_mathematics_3",
  "suite": "mmlu",
  "question": "What is the derivative of x^2?",
  "answer": "B",
  "expected": "B",
  "correct": true,
  "elapsed_s": 1.234,
  "timestamp": 1742000000.123
}
```

The `query_fn` returns extra fields (provider, model, used_rag) which `run_health_check` doesn't currently log. That's fine — the core fields (answer, correct, elapsed) are what matter. The extra metadata is for debugging.

---

## Tests

### test_ollama_benchmark.py

```python
"""Tests for Ollama benchmark with model routing and RAG context."""

import json
from unittest.mock import MagicMock, patch

import pytest


def test_get_model_for_suite():
    from knowledge3d.tools.ollama_benchmark import get_model_for_suite
    assert get_model_for_suite("mmlu") == "qwen3:8b"
    assert get_model_for_suite("gsm8k") == "qwen2.5:32b"
    assert get_model_for_suite("math") == "qwen2.5:32b"
    assert get_model_for_suite("lhe") == "qwen2.5:32b"
    assert get_model_for_suite("arc") == "qwen2.5:32b"
    # Unknown suite defaults to large
    assert get_model_for_suite("unknown") == "qwen2.5:32b"


def test_system_prompts_exist():
    from knowledge3d.tools.ollama_benchmark import SYSTEM_PROMPTS
    assert "mmlu" in SYSTEM_PROMPTS
    assert "gsm8k" in SYSTEM_PROMPTS
    assert "math" in SYSTEM_PROMPTS
    assert "lhe" in SYSTEM_PROMPTS
    assert "arc" in SYSTEM_PROMPTS
    # MMLU prompt asks for letter
    assert "letter" in SYSTEM_PROMPTS["mmlu"].lower() or "A, B, C" in SYSTEM_PROMPTS["mmlu"]
    # GSM8K prompt asks for number
    assert "numeric" in SYSTEM_PROMPTS["gsm8k"].lower() or "number" in SYSTEM_PROMPTS["gsm8k"].lower()


def test_extract_answer_mmlu():
    from knowledge3d.tools.ollama_benchmark import extract_answer
    assert extract_answer("The answer is B.", "mmlu") == "B"
    assert extract_answer("After analysis:\nC", "mmlu") == "C"
    assert extract_answer("<think>reasoning</think>\nD", "mmlu") == "D"
    assert extract_answer("A", "mmlu") == "A"


def test_extract_answer_gsm8k():
    from knowledge3d.tools.ollama_benchmark import extract_answer
    assert extract_answer("Step 1: 3+2=5\nStep 2: 5*4=20\n20", "gsm8k") == "20"
    assert extract_answer("<think>work</think>\n42", "gsm8k") == "42"
    assert extract_answer("The answer is 7.5", "gsm8k") == "7.5"


def test_extract_answer_arc():
    from knowledge3d.tools.ollama_benchmark import extract_answer
    result = extract_answer('[[1, 0], [0, 1]]', "arc")
    assert json.loads(result) == [[1, 0], [0, 1]]


def test_extract_answer_strips_thinking():
    from knowledge3d.tools.ollama_benchmark import extract_answer
    raw = "<think>Let me think about this carefully. The options are...</think>\n\nB"
    assert extract_answer(raw, "mmlu") == "B"


def test_build_rag_context_chemistry():
    from knowledge3d.tools.ollama_benchmark import build_rag_context
    row = {
        "question": "What is the atomic number of carbon?",
        "payload": {"subject": "college_chemistry"},
    }
    ctx = build_rag_context(row, "mmlu")
    assert "Periodic Table" in ctx or "REFERENCE FACTS" in ctx


def test_build_rag_context_physics():
    from knowledge3d.tools.ollama_benchmark import build_rag_context
    row = {
        "question": "What is the speed of light in vacuum?",
        "payload": {"subject": "college_physics"},
    }
    ctx = build_rag_context(row, "mmlu")
    assert "Physical Constants" in ctx or "speed of light" in ctx.lower() or "REFERENCE" in ctx


def test_build_rag_context_no_match():
    from knowledge3d.tools.ollama_benchmark import build_rag_context
    row = {
        "question": "Who wrote Hamlet?",
        "payload": {"subject": "philosophy"},
    }
    ctx = build_rag_context(row, "mmlu")
    # No chemistry/physics/math match — should return empty or minimal
    # (philosophy doesn't trigger any RAG domain)
    assert isinstance(ctx, str)


def test_build_user_message_mmlu():
    from knowledge3d.tools.ollama_benchmark import _build_user_message
    row = {
        "question": "What is H2O?",
        "payload": {"options": ["Water", "Salt", "Sugar", "Sand"]},
    }
    msg = _build_user_message(row, "mmlu")
    assert "A. Water" in msg
    assert "D. Sand" in msg


def test_build_user_message_gsm8k():
    from knowledge3d.tools.ollama_benchmark import _build_user_message
    row = {"question": "If you have 3 apples and buy 2 more, how many?", "payload": {}}
    msg = _build_user_message(row, "gsm8k")
    assert "3 apples" in msg


def test_create_ollama_query_fn():
    from knowledge3d.tools.ollama_benchmark import create_ollama_query_fn
    fn = create_ollama_query_fn()
    assert callable(fn)


def test_ollama_chat_method():
    """Test that OllamaModelManager has a chat method after the modification."""
    from knowledge3d.ingestion.ollama_manager import OllamaModelManager
    mgr = OllamaModelManager()
    assert hasattr(mgr, "chat"), "OllamaModelManager must have a chat() method"


@patch("knowledge3d.ingestion.ollama_manager.urllib.request.urlopen")
def test_ollama_chat_success(mock_urlopen):
    """Test chat() returns parsed output on success."""
    from knowledge3d.ingestion.ollama_manager import OllamaModelManager

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "message": {"content": "B"}
    }).encode("utf-8")
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_response

    mgr = OllamaModelManager()
    result = mgr.chat("qwen3:8b", [{"role": "user", "content": "test"}])
    assert result.returncode == 0
    assert result.output == "B"


@patch("knowledge3d.ingestion.ollama_manager.urllib.request.urlopen")
def test_ollama_chat_failure(mock_urlopen):
    """Test chat() handles errors gracefully."""
    import urllib.error
    from knowledge3d.ingestion.ollama_manager import OllamaModelManager

    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

    mgr = OllamaModelManager()
    result = mgr.chat("qwen3:8b", [{"role": "user", "content": "test"}])
    assert result.returncode == 1
    assert "refused" in result.stderr.lower()
```

---

## Usage Examples

```bash
# Run MMLU (50 questions) through fast qwen3:8b with RAG
python -m knowledge3d.tools.ollama_benchmark --suite mmlu --count 50

# Run GSM8K (10 questions) through qwen2.5:32b with RAG
python -m knowledge3d.tools.ollama_benchmark --suite gsm8k --count 10

# Run ALL suites (10 each) — full sweep
python -m knowledge3d.tools.ollama_benchmark --suite all --count 10

# Run without RAG (baseline comparison)
python -m knowledge3d.tools.ollama_benchmark --suite mmlu --count 50 --no-rag

# Custom models
python -m knowledge3d.tools.ollama_benchmark \
  --suite all --count 10 \
  --medium-model gemma3:12b \
  --large-model deepseek-r1:14b

# Via the generic benchmark_health_check entry point
python -m knowledge3d.tools.benchmark_health_check \
  --suite mmlu --count 50 --provider ollama
```

---

## File Changes Summary

| File | Action |
|------|--------|
| `knowledge3d/ingestion/ollama_manager.py` | **MODIFY** — Add `chat()` HTTP API method, add `json`/`urllib` imports |
| `knowledge3d/tools/ollama_benchmark.py` | **NEW** — Model routing, system prompts, RAG context, answer extractors, CLI |
| `knowledge3d/tools/benchmark_health_check.py` | **MODIFY** — Add `--provider` flag, add `suite` to `load_questions` rows, MMLU `expected` → letter |
| `tests/test_ollama_benchmark.py` | **NEW** — 15 tests covering routing, prompts, extraction, RAG, chat API |

---

## Success Criteria

1. `--suite mmlu --count 50` runs through `qwen3:8b` and completes in < 30 minutes
2. `--suite gsm8k --count 10` runs through `qwen2.5:32b` and gets reasonable scores
3. RAG context injects periodic table for chemistry, constants for physics questions
4. System prompts produce extractable answers (letter for MMLU, number for GSM8K)
5. `<think>...</think>` blocks stripped from qwen3 output before extraction
6. All 15 tests pass
7. Health log entries written correctly to JSONL
8. Non-regression: existing benchmark tests still pass

---

## Expected Scores (Rough Estimates)

| Suite | Model | With RAG | Without RAG | Notes |
|-------|-------|----------|-------------|-------|
| MMLU (50) | qwen3:8b | ~35-40/50 | ~30-35/50 | RAG helps chemistry/physics factual recall |
| GSM8K (10) | qwen2.5:32b | ~7-9/10 | ~7-9/10 | RAG minimal impact (word problems) |
| Math (10) | qwen2.5:32b | ~5-7/10 | ~5-7/10 | Competition math is hard |
| LHE (10) | qwen2.5:32b | ~2-4/10 | ~2-4/10 | Multi-hop is extremely hard |
| ARC (10) | qwen2.5:32b | ~1-3/10 | ~1-3/10 | Visual patterns through text is limited |

**Primary value:** MMLU with RAG — this is where H17's periodic table, constants, and measurements directly ground answers that would otherwise be hallucinated.
