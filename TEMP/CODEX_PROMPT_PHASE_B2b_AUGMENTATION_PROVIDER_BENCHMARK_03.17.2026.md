# Phase B2b — Augmentation Provider Benchmark Bridge

**Depends on:** B2 (Sovereign Benchmark), H16b (Multi-Provider Augmentation)
**Modifies:** `knowledge3d/tools/benchmark_health_check.py`
**Creates:** `knowledge3d/tools/benchmark_provider_bridge.py`
**Tests:** `tests/test_benchmark_provider_bridge.py`
**Goal:** Run benchmark questions through augmentation providers (Ollama, Claude, GPT, etc.) as a quick sanity check and provider comparison tool

---

## Objective

The sovereign GPU path (B2) is the real benchmark. But we also want to:
1. **Sanity-check** benchmark questions through external LLMs
2. **Compare providers** — which augmentation provider gives the best answers?
3. **Validate the I/O pipeline** end-to-end: question → provider → answer → health_log → sleep-time
4. **Establish baselines** — know what score each provider achieves independently, so we can measure when the sovereign GPU path surpasses them

This is NOT the sovereign path. This uses external LLMs via the augmentation providers. Results log to health_log.jsonl with `source="provider"` so sleep-time knows these are external.

---

## Architecture

The key insight from the exploration: `benchmark_health_check.py` already supports a custom `query_fn` callback in `run_health_check()`. We just need to:

1. Create a provider-backed `query_fn` that wraps `AugmentationProvider.augment()`
2. Add suite-specific answer extractors (pull the actual answer from AugmentationResult)
3. Add a `--provider` CLI flag

```
Benchmark Questions (load_questions)
    ↓
run_health_check(suite, count, log_path, query_fn=provider_query_fn)
    ↓
provider_query_fn(row):
    ├─ Build prompt from row (suite-specific)
    ├─ create_provider(name) → AugmentationProvider
    ├─ provider.augment(prompt, context) → AugmentationResult
    ├─ extract_answer(result, suite) → answer string
    └─ return {"answer": answer, "provider": provider.provider_name}
    ↓
evaluate_answer(suite, answer, expected) → correct: bool
    ↓
health_log.jsonl (with source="provider", provider=name)
    ↓
sleeptime._summarize_health_log() → neutral (no ground truth for weight updates)
```

---

## New File: `knowledge3d/tools/benchmark_provider_bridge.py`

```python
"""Bridge between benchmark questions and augmentation providers."""

from __future__ import annotations

import json
import re
from typing import Any

from knowledge3d.tools.augmentation_providers import (
    AugmentationProvider,
    AugmentationResult,
    create_provider,
)


def build_benchmark_prompt(row: dict[str, Any], suite: str) -> str:
    """Build a clear prompt for the provider from a benchmark row.

    Formats the question so the LLM gives a direct, extractable answer.
    """
    question = str(row.get("question", "")).strip()
    options = row.get("payload", {}).get("options", []) or row.get("options", [])

    if suite == "mmlu" and options:
        # Multiple choice — list options, ask for letter
        option_text = "\n".join(f"  {chr(65+i)}. {opt}" for i, opt in enumerate(options))
        return (
            f"{question}\n\n{option_text}\n\n"
            "Answer with ONLY the letter (A, B, C, or D) of the correct option."
        )
    elif suite == "gsm8k":
        return (
            f"{question}\n\n"
            "Solve step by step. On the LAST line, write ONLY the final numeric answer "
            "(no units, no text, just the number)."
        )
    elif suite == "math_competitions":
        return (
            f"{question}\n\n"
            "Solve this math problem. On the LAST line, write ONLY the final numeric answer."
        )
    elif suite == "arc_agi_2":
        # ARC has grid I/O — format as JSON
        payload = row.get("payload", {})
        train = payload.get("train", [])
        test_input = payload.get("test", [{}])[0].get("input", [])
        return (
            f"This is an ARC-AGI pattern completion task.\n\n"
            f"Training examples (input → output):\n{json.dumps(train, indent=2)}\n\n"
            f"Test input:\n{json.dumps(test_input)}\n\n"
            "Predict the output grid as a JSON array of arrays (numbers only)."
        )
    elif suite == "last_humanity_exam":
        return (
            f"{question}\n\n"
            "Answer as concisely as possible — ideally a single word, phrase, or short sentence."
        )
    else:
        return f"{question}\n\nAnswer concisely."


def extract_answer(result: AugmentationResult, suite: str) -> str:
    """Extract the actual answer from an augmentation result.

    Suite-specific logic to pull the answer the evaluator expects.
    """
    raw = str(result.summary).strip()
    if not raw:
        raw = str(result.raw_response).strip()

    if suite == "mmlu":
        # Look for a single letter (A/B/C/D) answer
        # Check last line first, then scan for standalone letter
        lines = raw.strip().splitlines()
        for line in reversed(lines):
            cleaned = line.strip().rstrip(".").strip()
            if cleaned.upper() in ("A", "B", "C", "D"):
                return cleaned.upper()
        # Regex: find "answer is X" or "the answer: X" patterns
        match = re.search(r"(?:answer\s*(?:is|:)\s*)([A-Da-d])\b", raw, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # First standalone letter
        match = re.search(r"\b([A-Da-d])\b", raw)
        if match:
            return match.group(1).upper()
        return raw[:100]  # Fallback: first 100 chars

    elif suite in ("gsm8k", "math_competitions"):
        # Extract last number from response
        lines = raw.strip().splitlines()
        for line in reversed(lines):
            # Look for a number (possibly negative, decimal)
            numbers = re.findall(r"-?\d+(?:\.\d+)?(?:,\d{3})*", line.replace(",", ""))
            if numbers:
                return numbers[-1].replace(",", "")
        # Fallback: any number in the response
        numbers = re.findall(r"-?\d+(?:\.\d+)?", raw)
        if numbers:
            return numbers[-1]
        return raw[:100]

    elif suite == "arc_agi_2":
        # Try to extract JSON grid
        try:
            # Find JSON array in response
            match = re.search(r"\[\s*\[.*?\]\s*\]", raw, re.DOTALL)
            if match:
                grid = json.loads(match.group())
                return json.dumps(grid)
        except (json.JSONDecodeError, AttributeError):
            pass
        return raw[:500]

    elif suite == "last_humanity_exam":
        # Take first line or sentence
        lines = raw.strip().splitlines()
        return lines[0].strip() if lines else raw[:200]

    else:
        return raw[:200]


def create_provider_query_fn(
    provider_name: str = "auto",
    **provider_kwargs: Any,
) -> callable:
    """Create a query function that routes benchmark questions through an augmentation provider.

    Returns a function compatible with benchmark_health_check.run_health_check(query_fn=...).
    """
    provider = create_provider(provider_name, **provider_kwargs)

    def query_fn(row: dict[str, Any]) -> dict[str, Any]:
        suite = str(row.get("suite", "general")).strip()
        prompt = build_benchmark_prompt(row, suite)
        context = {
            "name": f"benchmark_{suite}_{row.get('id', 'unknown')}",
            "domain_hint": _suite_to_domain(suite),
            "source": "benchmark_provider",
        }
        result = provider.augment(prompt, context)
        answer = extract_answer(result, suite)
        return {
            "answer": answer,
            "provider": provider.provider_name,
            "confidence": result.confidence,
            "domain": result.domain,
            "raw_summary": result.summary,
        }

    return query_fn


def _suite_to_domain(suite: str) -> str:
    """Map benchmark suite to augmentation domain hint."""
    return {
        "gsm8k": "Mathematics",
        "math_competitions": "Mathematics",
        "arc_agi_2": "Visual",
        "mmlu": "General",
        "last_humanity_exam": "General",
    }.get(suite, "General")
```

---

## Extend: `benchmark_health_check.py`

Add `--provider` CLI flag to the existing argument parser:

```python
# In _parse_args():
parser.add_argument(
    "--provider",
    default=None,
    help="Run via augmentation provider instead of sovereign GPU. "
         "Values: auto, ollama, claude, gpt, deepseek, gemini, qwen, glm, grok, kimi",
)
```

In `main()`:

```python
def main(argv=None):
    args = _parse_args(argv)
    query_fn = None
    if args.provider:
        from knowledge3d.tools.benchmark_provider_bridge import create_provider_query_fn
        query_fn = create_provider_query_fn(args.provider)
    summary = run_health_check(args.suite, args.count, args.log, query_fn=query_fn)
    print(json.dumps(summary, indent=2))
```

---

## Usage Examples

```bash
# Run MMLU through Ollama (local, free)
python -m knowledge3d.tools.benchmark_health_check \
  --suite mmlu --count 50 --provider ollama \
  --log health_log_ollama.jsonl

# Run GSM8K through Claude
python -m knowledge3d.tools.benchmark_health_check \
  --suite gsm8k --count 10 --provider claude \
  --log health_log_claude.jsonl

# Run all suites through GPT
for suite in arc_agi_2 math_competitions gsm8k last_humanity_exam mmlu; do
  python -m knowledge3d.tools.benchmark_health_check \
    --suite $suite --count 10 --provider gpt \
    --log health_log_gpt_${suite}.jsonl
done

# Compare: run same questions through multiple providers
for provider in ollama claude gpt deepseek; do
  python -m knowledge3d.tools.benchmark_health_check \
    --suite mmlu --count 50 --provider $provider \
    --log health_log_${provider}_mmlu.jsonl
done
```

---

## Health Log Entry Format (Provider Path)

```json
{
  "timestamp": 1700000000.123,
  "question_id": "mmlu_42",
  "suite": "mmlu",
  "source": "benchmark_provider",
  "question": "What is the atomic number of carbon?",
  "answer": "B",
  "expected": "B",
  "correct": true,
  "elapsed_s": 2.34,
  "provider": "claude",
  "confidence": 0.92,
  "domain": "General"
}
```

**Sleep-time handling:** Entries with `source="benchmark_provider"` are treated as **informational** — they inform Galaxy content gaps but do NOT update TRM weights (that's reserved for the sovereign path).

---

## Tests

### test_benchmark_provider_bridge.py

```python
def test_build_benchmark_prompt_mmlu():
    """MMLU prompt includes options and asks for letter answer."""
    row = {"question": "What is H2O?", "payload": {"options": ["Water", "Salt", "Sugar", "Sand"]}}
    prompt = build_benchmark_prompt(row, "mmlu")
    assert "A. Water" in prompt
    assert "letter" in prompt.lower()

def test_build_benchmark_prompt_gsm8k():
    """GSM8K prompt asks for final numeric answer."""
    row = {"question": "If you have 3 apples and buy 2 more, how many do you have?"}
    prompt = build_benchmark_prompt(row, "gsm8k")
    assert "numeric answer" in prompt.lower()

def test_extract_answer_mmlu_letter():
    """Extracts single letter from MMLU response."""
    result = AugmentationResult(
        summary="The answer is B. Water is H2O.",
        entities=[], relationships=[], domain="General",
        meaning_rpn_hint="", taxonomy_refs=[], surface_forms={},
        confidence=0.9, provider="test", raw_response="",
    )
    assert extract_answer(result, "mmlu") == "B"

def test_extract_answer_gsm8k_number():
    """Extracts final number from GSM8K response."""
    result = AugmentationResult(
        summary="Step 1: 3 + 2 = 5\nStep 2: 5 * 2 = 10\n10",
        entities=[], relationships=[], domain="Mathematics",
        meaning_rpn_hint="", taxonomy_refs=[], surface_forms={},
        confidence=0.9, provider="test", raw_response="",
    )
    assert extract_answer(result, "gsm8k") == "10"

def test_extract_answer_arc_grid():
    """Extracts JSON grid from ARC response."""
    result = AugmentationResult(
        summary='The output grid is [[1, 0], [0, 1]]',
        entities=[], relationships=[], domain="Visual",
        meaning_rpn_hint="", taxonomy_refs=[], surface_forms={},
        confidence=0.9, provider="test", raw_response="",
    )
    answer = extract_answer(result, "arc_agi_2")
    assert json.loads(answer) == [[1, 0], [0, 1]]

def test_create_provider_query_fn():
    """Provider query function returns expected structure."""
    # Mock provider
    fn = create_provider_query_fn("ollama")
    # fn(row) would call ollama — test structure only
    assert callable(fn)

def test_suite_to_domain():
    assert _suite_to_domain("gsm8k") == "Mathematics"
    assert _suite_to_domain("mmlu") == "General"
    assert _suite_to_domain("arc_agi_2") == "Visual"
```

---

## File Changes Summary

| File | Action |
|------|--------|
| `knowledge3d/tools/benchmark_provider_bridge.py` | **NEW** — Provider query fn, prompt builders, answer extractors |
| `knowledge3d/tools/benchmark_health_check.py` | **MODIFY** — Add `--provider` CLI flag |
| `tests/test_benchmark_provider_bridge.py` | **NEW** — Prompt building, answer extraction, integration tests |

---

## Success Criteria

1. `--provider ollama` flag runs benchmarks through local Ollama
2. `--provider claude` flag runs through Claude API (if ANTHROPIC_API_KEY set)
3. Answer extractors correctly parse MMLU letters, GSM8K numbers, ARC grids
4. Health log entries include `source="benchmark_provider"` and `provider` field
5. Sleep-time treats provider entries as informational (no TRM weight updates)
6. All existing benchmark tests pass (non-regression)
7. Provider comparison possible by running same suite through multiple providers
