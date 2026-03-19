# Phase B3 — Benchmark Knowledge Proceduralization via Ollama

**Depends on:** H17 (Universal Knowledge Foundation), H16b (Multi-Provider Augmentation), B2c (Ollama infrastructure)
**Modifies:** `knowledge3d/tools/ollama_benchmark.py` → repurpose as `knowledge3d/tools/knowledge_proceduralizer.py`
**Creates:** `knowledge3d/tools/knowledge_proceduralizer.py`, `tests/test_knowledge_proceduralizer.py`
**Goal:** Convert benchmark TRAINING data into procedural symlink-based Galaxy stars using Ollama as the augmentation engine — NOT answer benchmarks, but INGEST THE KNOWLEDGE they contain

---

## Critical Paradigm Shift

**WRONG approach (what B2c did):** Ask Ollama "What is the answer to this MMLU question?" → Get letter → Compare
**RIGHT approach (what B3 does):** Feed TRAINING data to Ollama → Extract meaning, entities, relationships, taxonomy → Convert to MeaningCentricStar with symlink references → Store in Galaxy → Sovereign GPU path benefits from enriched knowledge

The benchmark training datasets are KNOWLEDGE REPOSITORIES:
- **MMLU auxiliary_train** (99,841 Q&A pairs) = encyclopedic knowledge across 57 subjects
- **GSM8K train** (7,473 word problems with step-by-step solutions) = arithmetic reasoning patterns
- **MMLU val** (57 subjects × ~15 questions) = curated domain knowledge

Each Q&A pair contains extractable FACTS, RELATIONSHIPS, and PATTERNS that should become Galaxy entries — not be answered and discarded.

---

## Architecture

```
Benchmark Training Data (CSV/JSONL)
    ↓
knowledge_proceduralizer.py (batch loader)
    ↓
For each entry:
    ├─ Build proceduralization prompt (system + RAG context + entry content)
    ├─ Send to Ollama via chat() HTTP API
    ├─ Parse structured JSON response → AugmentationResult
    ├─ proceduralize_content() → symlink references
    ├─ result_to_star() → MeaningCentricStar
    └─ Accumulate stars
    ↓
write_stars_jsonl() → /K3D/Knowledge3D.local/galaxies/procedualized_*.jsonl
    ↓
(Later: galaxy_manager.store_meaning_star() → VRAM)
```

---

## System Prompt: The Proceduralizer

This is the KEY innovation. Instead of the generic `AUGMENTATION_SYSTEM_PROMPT` (which asks for entities/relationships), we give Ollama a system prompt that explains the K3D procedural symlink model and asks it to CONVERT knowledge into that format.

```python
PROCEDURALIZATION_SYSTEM_PROMPT = """You are a knowledge proceduralizer for K3D, a spatial knowledge system.

Your job is to extract MEANING from educational content and convert it into structured procedural entries.

## K3D Knowledge Architecture (4 layers)

1. **Form** (Layer 1): How something LOOKS — visual primitives, glyphs, shapes
2. **Meaning** (Layer 2): What something IS — the concept itself, language-agnostic
3. **Rules** (Layer 3): How things RELATE — grammar rules, transformation patterns, formulas
4. **Meta-Rules** (Layer 4): Rules ABOUT rules — when to apply which rule, domain constraints

## Symlink Principle

NEVER duplicate information. Instead, create REFERENCES (symlinks) to existing concepts:
- A chemistry fact about "water" should reference star_ids: element_h, element_o, material_water
- A physics fact about "gravity" should reference: constant_gravitational, concept_physics
- A math formula should reference its component operations: ADD, MUL, SQRT, etc.

## Your Output Format

Return strict JSON:
```json
{
  "meaning_class": "fact|rule|formula|definition|pattern|relationship",
  "meaning_rpn": "compact RPN-like semantic description of the core meaning",
  "domain": "Mathematics|Physics|Biology|Language|Tools|Visual|Audio|General",
  "entities": [{"type": "concept|element|unit|formula|person|event", "name": "...", "star_ref": "existing_star_id_if_known"}],
  "relationships": [{"from": "entity_name", "relation": "is_a|has_property|causes|requires|equals|composed_of", "to": "entity_name"}],
  "taxonomy_refs": ["concept_X", "concept_Y"],
  "surface_forms": {"en": "english label", "pt": "portuguese label if known"},
  "grammar_rules": [{"pattern": "IF condition THEN result", "strength": 1}],
  "layer": 2,
  "confidence": 0.0-1.0
}
```

## Key Guidelines

- **meaning_rpn**: Use RPN notation. Example: "WATER ELEMENT_H 2 MUL ELEMENT_O 1 MUL COMPOSE" for H2O
- **star_ref in entities**: Reference existing stars when possible. Known prefixes:
  element_XX (chemical elements), constant_XX (physical constants), unit_XX_YY (measurement units),
  script_XX (writing systems), numeral_system_XX, material_XX, format_XX, standard_size_XX
- **layer**: 2 for pure facts/definitions, 3 for rules/formulas, 4 for meta-rules/constraints
- **grammar_rules**: Extract IF/THEN patterns. Example: "IF temperature > 100C THEN water BOILS"
- Be PRECISE and STRUCTURED. No narrative. Every field matters.
"""
```

---

## RAG Context: Existing Star References

For each entry, we inject a compact list of EXISTING Galaxy star_ids that might be relevant. This gives the model concrete references to symlink to.

```python
def build_rag_star_references(domain: str, subject: str) -> str:
    """Build a compact reference list of existing H17 stars relevant to this domain.

    This is lightweight RAG: give the model a menu of existing star_ids it can
    reference in its symlinks, so it doesn't have to guess or invent them.
    """
    from knowledge3d.ingestion.universal_knowledge import (
        iter_elements, iter_physical_constants, iter_domains,
        iter_writing_systems, iter_numeral_systems,
    )

    refs: list[str] = []
    domain_lower = domain.lower()
    subject_lower = subject.lower()

    # Always include basic taxonomy concepts
    refs.append("## Core taxonomy refs (always available):")
    refs.append("concept_mathematics, concept_physics, concept_chemistry, concept_biology")
    refs.append("concept_language, concept_tool, concept_visual, concept_audio")

    # Chemistry references
    if any(kw in f"{domain_lower} {subject_lower}" for kw in
           ["chem", "bio", "element", "molecule", "anatomy", "medicine", "nutrition"]):
        elements = list(iter_elements())
        # Show first 30 common elements
        el_refs = [f"element_{e.symbol.lower()} ({e.name_en}, Z={e.atomic_number})"
                   for e in elements[:36]]
        refs.append(f"\n## Chemical elements (star_id format: element_XX):")
        refs.extend(el_refs)

    # Physics constants
    if any(kw in f"{domain_lower} {subject_lower}" for kw in
           ["phys", "astro", "electric", "thermo", "mechanics", "optics"]):
        constants = list(iter_physical_constants())
        const_refs = [f"constant_{c.key} ({c.name} = {c.value} {c.unit})"
                      for c in constants]
        refs.append(f"\n## Physical constants (star_id format: constant_XX):")
        refs.extend(const_refs)

    # Measurement units
    if any(kw in f"{domain_lower} {subject_lower}" for kw in
           ["math", "phys", "engineer", "geo", "astro", "econ"]):
        domains = list(iter_domains())
        refs.append(f"\n## Measurement units (star_id format: unit_DOMAIN_UNIT):")
        for d in domains[:8]:
            unit_names = list(d.units.keys())[:4]
            refs.append(f"  {d.key}: {', '.join(f'unit_{d.key}_{u}' for u in unit_names)}")

    # Materials
    if any(kw in f"{domain_lower} {subject_lower}" for kw in
           ["chem", "material", "engineer", "geology"]):
        refs.append(f"\n## Materials (star_id format: material_XX):")
        refs.append("  material_water, material_steel, material_glass, material_wood, material_concrete")

    if len(refs) <= 2:
        # Generic fallback — show what's available
        refs.append(f"\n## Available star_id prefixes:")
        refs.append("  element_*, constant_*, unit_*, material_*, script_*, numeral_system_*, format_*, standard_size_*")

    return "\n".join(refs)
```

---

## Model Selection

Same two-tier approach, but PURPOSE is different:

| Tier | Model | Use For | Why |
|------|-------|---------|-----|
| **Medium** | `qwen3:8b` | Simple facts, definitions, straightforward Q&A pairs | Fast extraction, most training data is factual |
| **Large** | `qwen2.5:32b` | Complex relationships, multi-step formulas, rule extraction | Needs reasoning to decompose compound knowledge |

**Routing by source:**
```python
SOURCE_MODEL_MAP = {
    "mmlu_auxiliary_train": "qwen3:8b",    # Factual Q&A → definitions/facts
    "mmlu_val": "qwen3:8b",                # Curated Q&A → domain knowledge
    "gsm8k_train": "qwen2.5:32b",          # Word problems → arithmetic patterns + rules
}
```

---

## New File: `knowledge3d/tools/knowledge_proceduralizer.py`

```python
"""Convert benchmark training data into procedural symlink-based Galaxy stars.

This is an INGESTION tool, not a benchmark runner. It reads training/validation
data from benchmark datasets and procedurralizes the KNOWLEDGE they contain into
MeaningCentricStar entries for the Galaxy.

Pipeline: training data → Ollama (proceduralize) → AugmentationResult → MeaningCentricStar → JSONL
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Iterator

from knowledge3d.ingestion.ollama_manager import OllamaModelManager
from knowledge3d.tools.augmentation_providers import AugmentationResult
from knowledge3d.tools.content_to_stars import result_to_star, write_stars_jsonl


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

PROCEDURALIZATION_SYSTEM_PROMPT = """You are a knowledge proceduralizer for K3D, a spatial knowledge system.

Your job is to extract MEANING from educational content and convert it into structured procedural entries.

## K3D Knowledge Architecture (4 layers)

1. **Form** (Layer 1): How something LOOKS — visual primitives, glyphs, shapes
2. **Meaning** (Layer 2): What something IS — the concept itself, language-agnostic
3. **Rules** (Layer 3): How things RELATE — grammar rules, transformation patterns, formulas
4. **Meta-Rules** (Layer 4): Rules ABOUT rules — when to apply which rule, domain constraints

## Symlink Principle

NEVER duplicate information. Instead, create REFERENCES (symlinks) to existing concepts.
When you see a known concept (element, constant, unit), reference its star_id instead of restating it.

## Output Format

Return strict JSON with these keys:
- "meaning_class": one of "fact", "rule", "formula", "definition", "pattern", "relationship"
- "meaning_rpn": compact RPN-like semantic description (e.g., "WATER ELEMENT_H 2 MUL ELEMENT_O COMPOSE")
- "domain": one of "Mathematics", "Physics", "Biology", "Language", "Tools", "Visual", "Audio", "General"
- "summary": one-line factual summary of the knowledge extracted
- "entities": list of {"type": "concept|element|unit|formula", "name": "...", "star_ref": "existing_star_id_if_known"}
- "relationships": list of {"from": "name", "relation": "is_a|has_property|causes|requires|equals|composed_of", "to": "name"}
- "taxonomy_refs": list of concept references like ["concept_physics", "periodic_table"]
- "surface_forms": {"en": "english label", "pt": "portuguese if known"}
- "grammar_rules": list of {"pattern": "IF X THEN Y", "strength": 1} (optional, for rules/formulas)
- "layer": 2 for facts/definitions, 3 for rules/formulas, 4 for meta-rules
- "confidence": float 0.0-1.0

Key: meaning_rpn should use existing star_id references where possible. Be precise, structured, no narrative."""


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

SOURCE_MODEL_MAP: dict[str, str] = {
    "mmlu_train": "qwen3:8b",
    "mmlu_val": "qwen3:8b",
    "gsm8k_train": "qwen2.5:32b",
}

MODEL_OPTIONS: dict[str, dict[str, Any]] = {
    "qwen3:8b": {"temperature": 0.1, "num_predict": 4096},
    "qwen2.5:32b": {"temperature": 0.2, "num_predict": 2048},
}


# ---------------------------------------------------------------------------
# Data loaders: yield (entry_id, content_text, subject, source_type) tuples
# ---------------------------------------------------------------------------

def load_mmlu_entries(
    data_dir: Path,
    split: str = "val",
    *,
    subjects: list[str] | None = None,
    limit_per_subject: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Load MMLU Q&A pairs as knowledge entries to proceduralize.

    Each Q&A pair becomes a knowledge nugget: the question establishes context,
    the correct answer provides the fact. Together they form extractable meaning.
    """
    split_dir = data_dir / split
    if not split_dir.exists():
        return

    suffix = {"val": "_val.csv", "test": "_test.csv", "auxiliary_train": ".csv"}
    file_suffix = suffix.get(split, ".csv")

    for csv_file in sorted(split_dir.glob(f"*{file_suffix}")):
        subject = csv_file.stem.replace("_val", "").replace("_test", "").strip().lower()
        if subjects and subject not in subjects:
            continue

        count = 0
        with csv_file.open("r", encoding="utf-8") as f:
            for idx, row in enumerate(csv.reader(f)):
                if len(row) < 6:
                    continue
                if limit_per_subject is not None and count >= limit_per_subject:
                    break

                question = row[0].strip()
                options = [row[1].strip(), row[2].strip(), row[3].strip(), row[4].strip()]
                correct_letter = row[5].strip().upper()
                if correct_letter not in {"A", "B", "C", "D"}:
                    continue
                correct_answer = options[ord(correct_letter) - ord("A")]

                # Format as knowledge content — question + answer = a fact
                option_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
                content = (
                    f"Subject: {subject.replace('_', ' ').title()}\n"
                    f"Question: {question}\n"
                    f"Options:\n{option_text}\n"
                    f"Correct Answer: {correct_letter}. {correct_answer}\n"
                    f"Key Fact: The answer to \"{question}\" is \"{correct_answer}\"."
                )

                yield {
                    "entry_id": f"mmlu_{subject}_{idx}",
                    "content": content,
                    "subject": subject,
                    "domain_hint": _subject_to_domain(subject),
                    "source": f"mmlu_{split}",
                    "correct_answer": correct_answer,
                    "question": question,
                }
                count += 1


def load_gsm8k_entries(
    data_dir: Path,
    *,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Load GSM8K training problems as arithmetic pattern knowledge.

    Each word problem + solution contains extractable arithmetic PATTERNS and RULES,
    not just the answer. We want the model to extract the REASONING STRUCTURE.
    """
    train_path = data_dir / "grade_school_math" / "data" / "train.jsonl"
    if not train_path.exists():
        return

    count = 0
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and count >= limit:
                break
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = str(entry.get("question", "")).strip()
            answer_text = str(entry.get("answer", "")).strip()

            # Extract the final numeric answer (after ####)
            final_answer = ""
            if "####" in answer_text:
                final_answer = answer_text.split("####")[-1].strip()

            content = (
                f"Subject: Grade School Mathematics\n"
                f"Problem: {question}\n"
                f"Step-by-step Solution: {answer_text}\n"
                f"Final Answer: {final_answer}\n"
                f"Extract: the arithmetic PATTERN and RULES used in this solution."
            )

            yield {
                "entry_id": f"gsm8k_train_{count}",
                "content": content,
                "subject": "arithmetic",
                "domain_hint": "Mathematics",
                "source": "gsm8k_train",
                "correct_answer": final_answer,
                "question": question,
            }
            count += 1


def _subject_to_domain(subject: str) -> str:
    """Map MMLU subject to K3D domain."""
    s = subject.lower()
    if any(kw in s for kw in ["math", "algebra", "calculus", "geometry", "statistics"]):
        return "Mathematics"
    if any(kw in s for kw in ["physics", "astronomy", "electrical"]):
        return "Physics"
    if any(kw in s for kw in ["chemistry"]):
        return "Physics"  # Chemistry → Physics domain in K3D (Reality Galaxy)
    if any(kw in s for kw in ["biology", "anatomy", "medicine", "nutrition", "clinical"]):
        return "Biology"
    if any(kw in s for kw in ["computer", "machine_learning", "security"]):
        return "Tools"
    return "General"


# ---------------------------------------------------------------------------
# RAG context builder
# ---------------------------------------------------------------------------

def build_rag_star_references(domain: str, subject: str) -> str:
    """Build compact reference list of existing H17 star_ids for symlink targets."""
    from knowledge3d.ingestion.universal_knowledge import (
        iter_elements, iter_physical_constants, iter_domains,
    )

    refs: list[str] = []
    combined = f"{domain} {subject}".lower()

    refs.append("## Existing star_id references (use these for symlinks):")
    refs.append("Taxonomy: concept_mathematics, concept_physics, concept_chemistry, concept_biology, concept_language, concept_tool")

    if any(kw in combined for kw in ["chem", "bio", "element", "molecule", "anatomy", "medicine"]):
        elements = list(iter_elements())[:36]
        refs.append(f"\nElements ({len(elements)} shown): " +
                    ", ".join(f"element_{e.symbol.lower()}" for e in elements))

    if any(kw in combined for kw in ["phys", "astro", "electric", "thermo", "mechanic"]):
        constants = list(iter_physical_constants())
        refs.append(f"\nConstants: " +
                    ", ".join(f"constant_{c.key}" for c in constants))

    if any(kw in combined for kw in ["math", "phys", "engineer", "econ"]):
        domains = list(iter_domains())[:6]
        for d in domains:
            units = list(d.units.keys())[:3]
            refs.append(f"Units ({d.key}): " + ", ".join(f"unit_{d.key}_{u}" for u in units))

    if any(kw in combined for kw in ["chem", "material", "engineer"]):
        refs.append("Materials: material_water, material_steel, material_glass, material_wood, material_concrete")

    refs.append("\nPrefixes: element_*, constant_*, unit_*, material_*, script_*, numeral_system_*, format_*")
    return "\n".join(refs)


# ---------------------------------------------------------------------------
# Core proceduralization
# ---------------------------------------------------------------------------

def proceduralize_entry(
    entry: dict[str, Any],
    ollama: OllamaModelManager,
    model: str,
    options: dict[str, Any],
    timeout: float,
) -> AugmentationResult | None:
    """Send one entry through Ollama and parse the structured JSON response."""
    domain = entry.get("domain_hint", "General")
    subject = entry.get("subject", "")

    rag_context = build_rag_star_references(domain, subject)

    user_message = (
        f"{rag_context}\n\n"
        f"---\n\n"
        f"Proceduralize the following knowledge entry into the JSON format described in your instructions.\n\n"
        f"{entry['content']}"
    )

    temperature = float(options.pop("temperature", 0.1))
    result = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": PROCEDURALIZATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        timeout=timeout,
        temperature=temperature,
        options=dict(options),
    )

    if result.returncode != 0 or not result.output.strip():
        return None

    return _parse_proceduralization_response(result.output, entry)


def _parse_proceduralization_response(
    raw: str,
    entry: dict[str, Any],
) -> AugmentationResult:
    """Parse the Ollama JSON response into an AugmentationResult.

    Handles: clean JSON, JSON in markdown fences, JSON embedded in text.
    Also strips <think>...</think> blocks from qwen3 output.
    """
    # Strip thinking blocks
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Try to extract JSON
    payload = _extract_json(text)
    if not payload:
        # Fallback: treat raw text as summary
        return AugmentationResult(
            summary=text[:500],
            entities=[],
            relationships=[],
            domain=entry.get("domain_hint", "General"),
            meaning_rpn_hint=f"{entry.get('domain_hint', 'GENERAL')} ENTRY",
            taxonomy_refs=[],
            surface_forms={"en": entry.get("subject", "entry")},
            confidence=0.3,
            provider="ollama",
            raw_response=raw,
        )

    # Map parsed fields to AugmentationResult
    domain = str(payload.get("domain", entry.get("domain_hint", "General"))).strip()
    # Normalize domain
    valid_domains = {"Mathematics", "Physics", "Biology", "Language", "Tools", "Visual", "Audio", "General"}
    if domain not in valid_domains:
        domain = entry.get("domain_hint", "General")

    summary = str(payload.get("summary", "")).strip()
    if not summary:
        summary = entry.get("correct_answer", entry.get("subject", "entry"))

    meaning_rpn = str(payload.get("meaning_rpn", "")).strip()
    if not meaning_rpn:
        meaning_rpn = f"{domain.upper()} FACT"

    entities = payload.get("entities", [])
    if not isinstance(entities, list):
        entities = []

    relationships = payload.get("relationships", [])
    if not isinstance(relationships, list):
        relationships = []

    taxonomy_refs = payload.get("taxonomy_refs", [])
    if not isinstance(taxonomy_refs, list):
        taxonomy_refs = []

    surface_forms = payload.get("surface_forms", {})
    if not isinstance(surface_forms, dict):
        surface_forms = {}
    if "en" not in surface_forms:
        surface_forms["en"] = summary[:100]

    confidence = 0.5
    try:
        confidence = float(payload.get("confidence", 0.5))
    except (TypeError, ValueError):
        pass

    return AugmentationResult(
        summary=summary,
        entities=[
            {"type": str(e.get("type", "")), "name": str(e.get("name", "")), "content": str(e.get("star_ref", ""))}
            for e in entities if isinstance(e, dict)
        ],
        relationships=[
            {"from": str(r.get("from", "")), "relation": str(r.get("relation", "")), "to": str(r.get("to", ""))}
            for r in relationships if isinstance(r, dict)
        ],
        domain=domain,
        meaning_rpn_hint=meaning_rpn,
        taxonomy_refs=[str(r) for r in taxonomy_refs if str(r).strip()],
        surface_forms=surface_forms,
        confidence=max(0.0, min(1.0, confidence)),
        provider="ollama",
        raw_response=raw,
    )


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from text (handles fenced, bare, or embedded)."""
    # Try fenced markdown
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
    for block in fenced:
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue

    # Try bare JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find first { ... } object
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    return None


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def proceduralize_dataset(
    entries: Iterator[dict[str, Any]],
    *,
    model: str = "qwen3:8b",
    timeout: float = 120.0,
    output_path: Path,
    batch_size: int = 50,
) -> dict[str, Any]:
    """Proceduralize a batch of training entries into Galaxy stars.

    Returns a summary report. Stars are written to output_path as JSONL.
    """
    ollama = OllamaModelManager(default_timeout=timeout)
    options = dict(MODEL_OPTIONS.get(model, {"temperature": 0.1, "num_predict": 4096}))

    stars = []
    total = 0
    succeeded = 0
    failed = 0
    start_time = time.time()

    for entry in entries:
        if batch_size and total >= batch_size:
            break

        total += 1
        t0 = time.time()
        aug_result = proceduralize_entry(entry, ollama, model, dict(options), timeout)
        elapsed = time.time() - t0

        if aug_result is None:
            failed += 1
            print(f"  [{total}] FAIL {entry['entry_id']} ({elapsed:.1f}s)")
            continue

        star = result_to_star(
            aug_result,
            star_id=entry["entry_id"],
            meta_refs=[f"source:{entry['source']}", f"subject:{entry['subject']}"],
        )
        stars.append(star)
        succeeded += 1
        print(f"  [{total}] OK   {entry['entry_id']} → {aug_result.domain} ({elapsed:.1f}s)")

    # Write stars
    if stars:
        write_stars_jsonl(stars, output_path)

    total_time = time.time() - start_time
    return {
        "total_entries": total,
        "succeeded": succeeded,
        "failed": failed,
        "stars_written": len(stars),
        "output_path": str(output_path),
        "model": model,
        "total_time_s": round(total_time, 1),
        "avg_time_per_entry_s": round(total_time / max(total, 1), 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proceduralize benchmark training data into Galaxy stars"
    )
    parser.add_argument(
        "--source", required=True,
        choices=["mmlu_val", "mmlu_train", "gsm8k_train"],
        help="Which training dataset to proceduralize",
    )
    parser.add_argument("--count", type=int, default=50, help="Max entries to process")
    parser.add_argument(
        "--subjects", default=None,
        help="Comma-separated MMLU subjects to process (default: all)",
    )
    parser.add_argument(
        "--limit-per-subject", type=int, default=None,
        help="Max entries per MMLU subject",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override Ollama model (default: auto by source)",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/K3D/Knowledge3D.local/galaxies/proceduralized_stars.jsonl"),
    )
    parser.add_argument(
        "--mmlu-data", type=Path,
        default=Path("/K3D/K3D_llama_cpp/datasets/MMLU/data"),
    )
    parser.add_argument(
        "--gsm8k-data", type=Path,
        default=Path("/K3D/K3D_llama_cpp/datasets/GSM8K"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects else None
    model = args.model or SOURCE_MODEL_MAP.get(args.source, "qwen3:8b")

    print(f"Proceduralizing {args.source} ({args.count} entries) via {model}")
    print(f"Output: {args.output}")

    if args.source == "mmlu_val":
        entries = load_mmlu_entries(
            args.mmlu_data, "val",
            subjects=subjects,
            limit_per_subject=args.limit_per_subject,
        )
    elif args.source == "mmlu_train":
        entries = load_mmlu_entries(
            args.mmlu_data, "auxiliary_train",
            subjects=subjects,
            limit_per_subject=args.limit_per_subject,
        )
    elif args.source == "gsm8k_train":
        entries = load_gsm8k_entries(args.gsm8k_data, limit=args.count)
    else:
        print(f"Unknown source: {args.source}")
        return 1

    report = proceduralize_dataset(
        entries,
        model=model,
        timeout=args.timeout,
        output_path=args.output,
        batch_size=args.count,
    )

    print(f"\n{'='*60}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

---

## Usage Examples

```bash
# Proceduralize 20 MMLU validation entries (astronomy + physics)
python -m knowledge3d.tools.knowledge_proceduralizer \
  --source mmlu_val --count 20 \
  --subjects astronomy,college_physics \
  --model qwen3:8b \
  --output /K3D/Knowledge3D.local/galaxies/proceduralized_mmlu_physics.jsonl

# Proceduralize 50 GSM8K training problems into arithmetic patterns
python -m knowledge3d.tools.knowledge_proceduralizer \
  --source gsm8k_train --count 50 \
  --model qwen2.5:32b --timeout 300 \
  --output /K3D/Knowledge3D.local/galaxies/proceduralized_gsm8k_patterns.jsonl

# Proceduralize MMLU auxiliary training (broad knowledge)
python -m knowledge3d.tools.knowledge_proceduralizer \
  --source mmlu_train --count 100 \
  --limit-per-subject 5 \
  --output /K3D/Knowledge3D.local/galaxies/proceduralized_mmlu_broad.jsonl
```

---

## Expected Star Output Example

Input: MMLU astronomy question about blackbody radiation

```json
{
  "star_id": "mmlu_astronomy_0",
  "meaning_class": "fact",
  "meaning_rpn": "BLACKBODY TEMPERATURE 0.5 MUL → POWER 0.0625 MUL WAVELENGTH_PEAK 2.0 MUL",
  "domain": "House/Physics",
  "taxonomy_refs": ["concept_physics", "physical_constant"],
  "surface_forms": {
    "en": {"word_ref": "stefan_boltzmann_law_half_temperature", "char_refs": [...]},
    "pt": {"word_ref": "lei_stefan_boltzmann_meia_temperatura", "char_refs": [...]}
  },
  "meta_refs": ["source:mmlu_val", "subject:astronomy", "constant_stefan_boltzmann", "constant_wien_displacement"],
  "grammar_refs": ["grammar_stefan_boltzmann_t4", "grammar_wien_displacement"],
  "house_room": "House/Library",
  "confidence": 1,
  "polarity": 1
}
```

Note how:
- `meaning_rpn` encodes the RULE (halving T → P drops by 1/16, λ_peak doubles)
- `meta_refs` symlinks to existing `constant_*` star_ids from H17
- `grammar_refs` points to transformation rules (Stefan-Boltzmann T⁴ law)
- No duplicated physics content — just references

---

## Tests

### test_knowledge_proceduralizer.py

```python
"""Tests for knowledge proceduralization pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_load_mmlu_entries_val():
    """Loads MMLU val entries with correct structure."""
    from knowledge3d.tools.knowledge_proceduralizer import load_mmlu_entries
    data_dir = Path("/K3D/K3D_llama_cpp/datasets/MMLU/data")
    entries = list(load_mmlu_entries(data_dir, "val", subjects=["astronomy"], limit_per_subject=2))
    assert len(entries) >= 1
    assert "entry_id" in entries[0]
    assert "content" in entries[0]
    assert "subject" in entries[0]
    assert entries[0]["subject"] == "astronomy"
    assert "Correct Answer:" in entries[0]["content"]


def test_load_gsm8k_entries():
    """Loads GSM8K training entries with solution patterns."""
    from knowledge3d.tools.knowledge_proceduralizer import load_gsm8k_entries
    data_dir = Path("/K3D/K3D_llama_cpp/datasets/GSM8K")
    entries = list(load_gsm8k_entries(data_dir, limit=2))
    assert len(entries) >= 1
    assert "entry_id" in entries[0]
    assert "Step-by-step Solution:" in entries[0]["content"]
    assert entries[0]["domain_hint"] == "Mathematics"


def test_build_rag_star_references_physics():
    """RAG context includes physical constants for physics subjects."""
    from knowledge3d.tools.knowledge_proceduralizer import build_rag_star_references
    refs = build_rag_star_references("Physics", "astronomy")
    assert "constant_" in refs
    assert "unit_" in refs


def test_build_rag_star_references_chemistry():
    """RAG context includes elements for chemistry subjects."""
    from knowledge3d.tools.knowledge_proceduralizer import build_rag_star_references
    refs = build_rag_star_references("Physics", "college_chemistry")
    assert "element_" in refs


def test_extract_json_clean():
    """Extracts JSON from clean response."""
    from knowledge3d.tools.knowledge_proceduralizer import _extract_json
    raw = '{"meaning_class": "fact", "domain": "Physics"}'
    result = _extract_json(raw)
    assert result["meaning_class"] == "fact"


def test_extract_json_fenced():
    """Extracts JSON from markdown-fenced response."""
    from knowledge3d.tools.knowledge_proceduralizer import _extract_json
    raw = 'Here is the result:\n```json\n{"meaning_class": "rule"}\n```'
    result = _extract_json(raw)
    assert result["meaning_class"] == "rule"


def test_extract_json_with_thinking():
    """Handles qwen3 thinking blocks."""
    from knowledge3d.tools.knowledge_proceduralizer import _parse_proceduralization_response
    raw = '<think>Let me analyze...</think>\n{"meaning_class": "fact", "domain": "Physics", "summary": "test", "meaning_rpn": "TEST"}'
    entry = {"domain_hint": "Physics", "subject": "test", "correct_answer": "x"}
    result = _parse_proceduralization_response(raw, entry)
    assert result.domain == "Physics"
    assert result.summary == "test"


def test_parse_response_fallback():
    """Falls back gracefully on unparseable response."""
    from knowledge3d.tools.knowledge_proceduralizer import _parse_proceduralization_response
    raw = "Sorry, I cannot parse this."
    entry = {"domain_hint": "General", "subject": "test", "correct_answer": "x"}
    result = _parse_proceduralization_response(raw, entry)
    assert result.provider == "ollama"
    assert result.confidence == 0.3  # Low confidence fallback


def test_subject_to_domain():
    from knowledge3d.tools.knowledge_proceduralizer import _subject_to_domain
    assert _subject_to_domain("college_physics") == "Physics"
    assert _subject_to_domain("abstract_algebra") == "Mathematics"
    assert _subject_to_domain("college_biology") == "Biology"
    assert _subject_to_domain("machine_learning") == "Tools"
    assert _subject_to_domain("world_religions") == "General"


def test_proceduralization_system_prompt():
    """System prompt contains key K3D concepts."""
    from knowledge3d.tools.knowledge_proceduralizer import PROCEDURALIZATION_SYSTEM_PROMPT
    assert "symlink" in PROCEDURALIZATION_SYSTEM_PROMPT.lower()
    assert "meaning_rpn" in PROCEDURALIZATION_SYSTEM_PROMPT
    assert "star_ref" in PROCEDURALIZATION_SYSTEM_PROMPT
    assert "Layer 2" in PROCEDURALIZATION_SYSTEM_PROMPT
```

---

## File Changes Summary

| File | Action |
|------|--------|
| `knowledge3d/tools/knowledge_proceduralizer.py` | **NEW** — Core proceduralization pipeline |
| `tests/test_knowledge_proceduralizer.py` | **NEW** — 10 tests |
| `knowledge3d/tools/ollama_benchmark.py` | **KEEP** — Still useful for raw LLM baseline comparison |

---

## Success Criteria

1. `--source mmlu_val --count 10 --subjects astronomy` produces 10 MeaningCentricStar entries in JSONL
2. Stars contain `meaning_rpn` with RPN notation (not plain English)
3. Stars contain `taxonomy_refs` that symlink to H17 star_ids (`element_*`, `constant_*`, etc.)
4. `surface_forms` include at least `en` key with meaningful label
5. Stars have appropriate `domain` and `house_room` routing
6. GSM8K entries produce `meaning_class: "rule"` or `"pattern"` (not just `"fact"`)
7. JSON extraction handles thinking blocks, markdown fences, embedded JSON
8. All 10 tests pass
9. Existing tests non-regression

---

## What Happens AFTER Proceduralization (Future Phase)

1. **Galaxy ingestion**: `galaxy_manager.store_meaning_star()` loads proceduralized stars into VRAM
2. **Sovereign benchmark re-run**: Run MMLU/GSM8K through composed head with enriched Galaxy
3. **Sleep-time consolidation**: Compare sovereign answers to expected → strengthen/weaken star weights
4. **Cycle**: More training data → More stars → Better sovereign scores → Sleep-time prunes weak stars

This is the K3D knowledge flywheel: external knowledge → procedural stars → sovereign reasoning → self-improvement.
