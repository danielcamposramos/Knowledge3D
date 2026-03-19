"""Proceduralize benchmark knowledge into MeaningCentricStar entries."""

from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Iterable, Iterator

from knowledge3d.ingestion.ollama_manager import OllamaManager
from knowledge3d.ingestion.universal_knowledge import (
    build_meaning_layer_stars,
    iter_domains,
    iter_elements,
    iter_physical_constants,
    load_all_omw,
)

from .augmentation_providers import AugmentationResult
from .content_to_stars import result_to_star, write_stars_jsonl


PROCEDURALIZATION_SYSTEM_PROMPT = """You are a knowledge proceduralizer for K3D, a spatial knowledge system.

Your job: extract MEANING from educational content and convert it to structured procedural entries that REFERENCE existing knowledge, never duplicate it.

## K3D Knowledge Layers
1. Form (Layer 1): How something looks - glyphs, shapes, visual primitives
2. Meaning (Layer 2): What something IS - language-agnostic concept. One star per meaning.
3. Rules (Layer 3): How things relate - formulas, grammar rules, transformations
4. Meta-Rules (Layer 4): Rules about rules - domain constraints, when to apply which rule

## Symlink Principle (CRITICAL)
NEVER restate a fact that already has a star_id. REFERENCE it instead.
- Chemistry fact about carbon? -> Reference "element_c"
- Physics uses speed of light? -> Reference "constant_speed_of_light"
- Word meaning in the content? -> Reference "synset_XXXXXXXX_X"

## Your Output Format
Return strict JSON:
{
  "meaning_class": "fact|rule|formula|definition|pattern",
  "meaning_rpn": "compact English RPN",
  "domain": "Mathematics|Physics|Biology|Language|Tools|General",
  "summary": "one-line English factual summary",
  "star_refs": ["element_c", "synset_14845743_n"],
  "entities": [{"name": "carbon", "star_ref": "element_c"}],
  "relationships": [{"from": "entity", "relation": "is_a", "to": "entity"}],
  "taxonomy_refs": ["concept_chemistry"],
  "surface_forms": {"en": "english label"},
  "grammar_rules": [{"pattern": "IF condition THEN result", "strength": 1}],
  "layer": 2,
  "confidence": 0.0
}

Key rules:
- meaning_rpn MUST be in English, using RPN notation
- star_refs: list all existing star_ids this entry connects to
- entities: tag each entity with its star_ref if one exists
- grammar_rules: only for Layer 3+ entries
- Be precise. No narrative. Every field matters."""


SOURCE_MODEL_MAP = {
    "mmlu_train": "qwen3:8b",
    "mmlu_val": "qwen3:8b",
    "gsm8k_train": "qwen2.5:32b",
}

MODEL_OPTIONS = {
    "qwen3:8b": {"temperature": 0.1, "num_predict": 4096},
    "qwen2.5:32b": {"temperature": 0.2, "num_predict": 2048},
}

MMLU_DEFAULT_PATH = Path("/K3D/K3D_llama_cpp/datasets/MMLU/data")
GSM8K_DEFAULT_PATH = Path("/K3D/K3D_llama_cpp/datasets/GSM8K")

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_NUMBER_RE = re.compile(r"####\s*([^\n]+)")


def _hits(text: str, keywords: set[str]) -> bool:
    combined = str(text or "").lower()
    return any(keyword in combined for keyword in keywords)


def _subject_to_domain(subject: str) -> str:
    lowered = str(subject or "").lower()
    if any(keyword in lowered for keyword in ["math", "algebra", "calculus", "geometry", "statistics"]):
        return "Mathematics"
    if any(keyword in lowered for keyword in ["physics", "astronomy", "electrical"]):
        return "Physics"
    if "chemistry" in lowered:
        return "Physics"
    if any(keyword in lowered for keyword in ["biology", "anatomy", "medicine", "nutrition", "clinical"]):
        return "Biology"
    if any(keyword in lowered for keyword in ["computer", "machine_learning", "security"]):
        return "Tools"
    return "General"


@lru_cache(maxsize=1)
def _english_synset_index() -> dict[str, list[str]]:
    synsets = load_all_omw()
    index: dict[str, list[str]] = {}
    for synset_id, entry in synsets.items():
        star_id = f"synset_{synset_id.replace('-', '_')}"
        for lemma in entry.lemmas.get("en", []):
            token = str(lemma or "").strip().lower()
            if not token:
                continue
            bucket = index.setdefault(token, [])
            if star_id not in bucket:
                bucket.append(star_id)
    return index


def _meaning_star_refs(question_text: str, limit: int = 8) -> list[str]:
    tokens = re.findall(r"[a-z][a-z_'-]+", str(question_text or "").lower())
    refs: list[str] = []
    seen: set[str] = set()
    index = _english_synset_index()
    for token in tokens:
        for star_id in index.get(token, []):
            if star_id in seen:
                continue
            seen.add(star_id)
            refs.append(star_id)
            if len(refs) >= limit:
                return refs
    return refs


def build_rag_context(domain: str, subject: str, question_text: str) -> str:
    """Build a compact reference menu of existing star ids."""
    refs: list[str] = []
    refs.append("## Existing star_ids (REFERENCE these, do not restate their content):")
    refs.append("")
    refs.append("### Taxonomy")
    refs.append(
        "concept_mathematics, concept_physics, concept_chemistry, "
        "concept_biology, concept_language, concept_tool"
    )

    combined = f"{domain} {subject} {question_text}".lower()

    if _hits(combined, {"chem", "element", "atom", "molecule", "compound", "reaction", "oxide", "acid", "metal", "halogen", "periodic", "bond", "bio", "anatomy", "medicine", "organic", "cell"}):
        refs.append("")
        refs.append("### Chemical elements (star_id = element_{symbol})")
        for element in list(iter_elements())[:36]:
            refs.append(
                f"  element_{element.symbol.lower()} = {element.name_en}, "
                f"Z={element.atomic_number}, mass={element.atomic_mass}"
            )

    if _hits(combined, {"phys", "force", "energy", "velocity", "gravity", "light", "planck", "boltzmann", "electric", "magnetic", "thermo", "momentum", "wave", "frequency", "astro", "optic"}):
        refs.append("")
        refs.append("### Physical constants (star_id = constant_{key})")
        for constant in iter_physical_constants():
            refs.append(f"  constant_{constant.key} = {constant.name} = {constant.value} {constant.unit}")

    if _hits(combined, {"math", "algebra", "calculus", "geometry", "unit", "convert", "distance", "speed", "mass", "temperature", "pressure", "econ", "statistic", "probability"}):
        refs.append("")
        refs.append("### Measurement units (star_id = unit_{domain}_{unit})")
        for domain_entry in list(iter_domains())[:8]:
            unit_refs = ", ".join(f"unit_{domain_entry.key}_{name}" for name in list(domain_entry.units.keys())[:4])
            refs.append(f"  {domain_entry.key}: {unit_refs}")

    if _hits(combined, {"material", "steel", "glass", "water", "wood", "concrete", "alloy"}):
        refs.append("")
        refs.append("### Materials (star_id = material_{name})")
        refs.append("  material_water, material_steel, material_glass, material_wood, material_concrete")

    refs.append("")
    refs.append("### Word meanings (star_id = synset_{id}, one star per meaning, multilingual)")
    meaning_refs = _meaning_star_refs(question_text)
    if meaning_refs:
        for star_id in meaning_refs:
            refs.append(f"  {star_id}")
    else:
        sample = build_meaning_layer_stars(min_languages=3, limit=3)
        for star in sample:
            refs.append(f"  {star.star_id}")
    refs.append("  Reference format: synset_XXXXXXXX_X")

    refs.append("")
    refs.append("### All available star_id prefixes:")
    refs.append(
        "  element_*, constant_*, unit_*, material_*, script_*, numeral_system_*, "
        "format_*, standard_size_*, synset_*"
    )
    return "\n".join(refs)


def load_mmlu_entries(
    data_dir: Path,
    split: str = "val",
    *,
    subjects: list[str] | None = None,
    limit_per_subject: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Load MMLU entries as proceduralization records."""
    split_key = str(split or "val").strip().lower()
    split_dir = Path(data_dir) / split_key
    if not split_dir.exists():
        return
    wanted = {str(subject).strip().lower() for subject in list(subjects or []) if str(subject).strip()}
    per_subject: dict[str, int] = {}
    pattern = "*.csv" if split_key == "auxiliary_train" else f"*_{split_key}.csv"
    for csv_path in sorted(split_dir.glob(pattern)):
        if split_key == "auxiliary_train":
            subject = csv_path.stem.strip().lower()
        else:
            suffix = f"_{split_key}"
            subject = csv_path.stem[: -len(suffix)].strip().lower() if csv_path.stem.endswith(suffix) else csv_path.stem.strip().lower()
        if wanted and subject not in wanted:
            continue
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for index, row in enumerate(reader):
                if len(row) < 6:
                    continue
                if limit_per_subject is not None and per_subject.get(subject, 0) >= int(limit_per_subject):
                    break
                question, option_a, option_b, option_c, option_d, correct_letter = [str(value).strip() for value in row[:6]]
                if not question:
                    continue
                options = [option_a, option_b, option_c, option_d]
                letter = correct_letter.upper()
                if letter not in {"A", "B", "C", "D"}:
                    continue
                correct_text = options[ord(letter) - ord("A")]
                per_subject[subject] = per_subject.get(subject, 0) + 1
                yield {
                    "entry_id": f"mmlu_{split_key}_{subject}_{index}",
                    "content": (
                        f"Subject: {subject.replace('_', ' ').title()}\n"
                        f"Question: {question}\n"
                        f"Correct Answer: {letter}. {correct_text}\n"
                        f'Key Fact: The answer to "{question}" is "{correct_text}".'
                    ),
                    "subject": subject,
                    "domain_hint": _subject_to_domain(subject),
                    "source": f"mmlu_{split_key}",
                    "correct_answer": correct_text,
                    "correct_letter": letter,
                    "question": question,
                }


def load_gsm8k_entries(data_dir: Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Load GSM8K training entries as arithmetic-pattern records."""
    train_path = Path(data_dir) / "grade_school_math" / "data" / "train.jsonl"
    if not train_path.exists():
        return
    with train_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= int(limit):
                break
            raw = json.loads(line)
            question = str(raw.get("question") or "").strip()
            answer = str(raw.get("answer") or "").strip()
            match = _NUMBER_RE.search(answer)
            final_answer = match.group(1).strip() if match else answer.splitlines()[-1].strip()
            yield {
                "entry_id": f"gsm8k_train_{index}",
                "content": (
                    "Subject: Grade School Mathematics\n"
                    f"Problem: {question}\n"
                    f"Step-by-step Solution: {answer}\n"
                    f"Final Answer: {final_answer}\n"
                    "Extract the arithmetic PATTERN and RULES used in this solution."
                ),
                "subject": "arithmetic",
                "domain_hint": "Mathematics",
                "source": "gsm8k_train",
                "correct_answer": final_answer,
                "question": question,
            }


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", str(text or "")).strip()


def _extract_json(raw: str) -> dict[str, Any] | None:
    text = _strip_thinking(raw)
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    for candidate in fenced:
        try:
            parsed = json.loads(candidate.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _fallback_result(entry: dict[str, Any], raw: str, provider: str = "ollama") -> AugmentationResult:
    domain = str(entry.get("domain_hint") or "General")
    summary = str(entry.get("question") or entry.get("subject") or entry.get("entry_id") or "proceduralized entry")
    return AugmentationResult(
        summary=summary,
        entities=[],
        relationships=[],
        domain=domain,
        meaning_rpn_hint=f"{domain.upper()} CONTENT ENTRY",
        taxonomy_refs=[f"concept_{domain.lower()}"] if domain != "General" else [],
        surface_forms={"en": summary},
        confidence=0.2,
        provider=provider,
        raw_response=str(raw or ""),
    )


def _parse_response(raw: str, entry: dict[str, Any], provider: str = "ollama") -> AugmentationResult:
    payload = _extract_json(raw)
    if not isinstance(payload, dict):
        return _fallback_result(entry, raw, provider=provider)

    domain = str(payload.get("domain") or entry.get("domain_hint") or "General")
    summary = str(payload.get("summary") or entry.get("question") or entry.get("entry_id") or "proceduralized entry").strip()
    meaning_rpn = str(payload.get("meaning_rpn") or payload.get("meaning_rpn_hint") or f"{domain.upper()} CONTENT ENTRY").strip()
    taxonomy_refs = [
        str(item).strip()
        for item in list(payload.get("taxonomy_refs") or [])
        if str(item).strip()
    ]
    if not taxonomy_refs and domain != "General":
        taxonomy_refs.append(f"concept_{domain.lower()}")
    surface_forms = dict(payload.get("surface_forms") or {})
    if "en" not in surface_forms:
        surface_forms["en"] = summary

    entities = []
    for row in list(payload.get("entities") or []):
        if not isinstance(row, dict):
            continue
        entity = {
            "type": str(row.get("type") or "entity").strip(),
            "name": str(row.get("name") or "").strip(),
            "content": str(row.get("star_ref") or row.get("content") or "").strip(),
        }
        entities.append(entity)

    relationships = []
    for row in list(payload.get("relationships") or []):
        if not isinstance(row, dict):
            continue
        relationships.append(
            {
                "from": str(row.get("from") or "").strip(),
                "relation": str(row.get("relation") or "").strip(),
                "to": str(row.get("to") or "").strip(),
            }
        )

    try:
        confidence = float(payload.get("confidence", 0.35))
    except Exception:
        confidence = 0.35

    return AugmentationResult(
        summary=summary,
        entities=entities,
        relationships=relationships,
        domain=domain,
        meaning_rpn_hint=meaning_rpn,
        taxonomy_refs=taxonomy_refs,
        surface_forms={str(language).strip().lower(): str(text).strip() for language, text in surface_forms.items() if str(language).strip() and str(text).strip()},
        confidence=max(0.0, min(1.0, confidence)),
        provider=provider,
        raw_response=str(raw or ""),
    )


def proceduralize_entry(
    entry: dict[str, Any],
    ollama: OllamaManager,
    model: str,
    options: dict[str, Any] | None,
    timeout: float,
) -> AugmentationResult | None:
    """Send one entry through Ollama and parse the structured response."""
    rag = build_rag_context(
        str(entry.get("domain_hint") or "General"),
        str(entry.get("subject") or ""),
        str(entry.get("question") or ""),
    )
    user_message = f"{rag}\n\n---\n\nProceduralize this knowledge entry:\n\n{entry['content']}"
    resolved_options = dict(options or {})
    temperature = float(resolved_options.pop("temperature", 0.2))
    result = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": PROCEDURALIZATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        timeout=timeout,
        temperature=temperature,
        options=resolved_options,
    )
    if result.returncode != 0:
        return _fallback_result(entry, result.stderr or result.output, provider="ollama")
    return _parse_response(result.output, entry, provider="ollama")


def _response_meta_refs(result: AugmentationResult) -> list[str]:
    payload = _extract_json(result.raw_response)
    if not isinstance(payload, dict):
        return []
    refs: list[str] = []
    for star_ref in list(payload.get("star_refs") or []):
        text = str(star_ref).strip()
        if text:
            refs.append(text)
    grammar_rules = payload.get("grammar_rules")
    if isinstance(grammar_rules, list) and grammar_rules:
        refs.append("grammar_rules:" + json.dumps(grammar_rules, ensure_ascii=False, sort_keys=True))
    return refs


def proceduralize_dataset(
    entries: Iterable[dict[str, Any]],
    *,
    model: str,
    timeout: float,
    output_path: Path,
    batch_size: int = 50,
    ollama: OllamaManager | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run batch proceduralization and write stars as JSONL."""
    manager = ollama or OllamaManager(default_timeout=timeout)
    stars = []
    processed = 0
    created = 0
    for entry in entries:
        processed += 1
        result = proceduralize_entry(entry, manager, model, options, timeout)
        if result is None:
            continue
        meta_refs = [
            f"source:{entry['source']}",
            f"subject:{entry['subject']}",
            *_response_meta_refs(result),
        ]
        star = result_to_star(
            result,
            star_id=str(entry.get("entry_id") or "").strip() or None,
            meta_refs=meta_refs,
        )
        stars.append(star)
        created += 1
        if batch_size and processed % int(batch_size) == 0:
            print(f"[knowledge_proceduralizer] processed={processed} created={created}")
    written_path = write_stars_jsonl(stars, output_path)
    return {
        "processed": processed,
        "created": created,
        "output_path": str(written_path),
        "model": model,
    }


def _iter_limited(entries: Iterable[dict[str, Any]], count: int | None) -> Iterator[dict[str, Any]]:
    for index, entry in enumerate(entries):
        if count is not None and index >= int(count):
            break
        yield entry


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=["mmlu_val", "mmlu_train", "gsm8k_train"])
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--subjects", default="", help="Comma-separated MMLU subjects")
    parser.add_argument("--limit-per-subject", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/galaxies/proceduralized_stars.jsonl"),
    )
    parser.add_argument("--mmlu-data", type=Path, default=MMLU_DEFAULT_PATH)
    parser.add_argument("--gsm8k-data", type=Path, default=GSM8K_DEFAULT_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source = str(args.source).strip().lower()
    if source.startswith("mmlu_"):
        split = "auxiliary_train" if source == "mmlu_train" else "val"
        subject_list = [item.strip() for item in str(args.subjects).split(",") if item.strip()]
        entries = load_mmlu_entries(
            args.mmlu_data,
            split=split,
            subjects=subject_list or None,
            limit_per_subject=args.limit_per_subject,
        )
    else:
        entries = load_gsm8k_entries(args.gsm8k_data, limit=args.count)
    resolved_model = str(args.model or SOURCE_MODEL_MAP[source]).strip()
    options = dict(MODEL_OPTIONS.get(resolved_model, {}))
    summary = proceduralize_dataset(
        _iter_limited(entries, args.count),
        model=resolved_model,
        timeout=float(args.timeout),
        output_path=args.output,
        options=options,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


__all__ = [
    "GSM8K_DEFAULT_PATH",
    "MMLU_DEFAULT_PATH",
    "MODEL_OPTIONS",
    "PROCEDURALIZATION_SYSTEM_PROMPT",
    "SOURCE_MODEL_MAP",
    "_extract_json",
    "_hits",
    "_parse_response",
    "_subject_to_domain",
    "build_rag_context",
    "load_gsm8k_entries",
    "load_mmlu_entries",
    "proceduralize_dataset",
    "proceduralize_entry",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
