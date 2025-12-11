#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingest foundational PDFs into the 4-layer architecture.

Layers:
  - Layer 2: Word Galaxy (char_sequence symlinks to Layer 1)
  - Layer 3: Grammar rules (symbol_refs + word_refs symlinks)
  - Layer 4: Meta-rules (rule_refs symlinks)

This script is intentionally light and symlink-first:
  - No duplication of glyphs/strings
  - Uses placeholders for extraction logic (extend with NLP as needed)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set
import re

from knowledge3d.cranium.word_galaxy import WordDefinition, get_word_galaxy
from knowledge3d.cranium.eloquence_galaxy import MetaRule, get_eloquence_galaxy
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy, GrammarRule
from knowledge3d.cranium.math_galaxy import get_math_galaxy

# Categories and target layers (from FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md)
PDF_CATEGORIES: Dict[str, Dict] = {
    "advanced_math": {"path": "Advanced Mathematics/", "layer": [1, 3], "count": 5},
    "pedagogy": {"path": "Pedagogy & Learning/", "layer": [4], "count": 5},
    "language_grammar": {"path": "Language, Grammar & Semantics/", "layer": [2, 3], "count": 10},
    "eloquence": {"path": "Eloquence, Rhetoric & Persuasion/", "layer": [4], "count": 8},
    "self_reflection": {"path": "Self-Reflection/", "layer": [4], "count": 7},
    "storytelling": {"path": "Story Telling/", "layer": [4], "count": 7},
    "delivery": {"path": "Acting - Delivery/", "layer": [4], "count": 3},
    "context": {"path": "Context & Contextual Understanding/", "layer": [2, 3], "count": 13},
    "temporal": {"path": "Temporal Understanding/", "layer": [3], "count": 13},
    "research": {"path": "Academic Research Methods/", "layer": [4], "count": 3},
}


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text using PyMuPDF if available; otherwise return empty."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("  [WARN] PyMuPDF not installed; skipping text extraction")
        return ""

    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_words_from_text(text: str, domain: str) -> List[WordDefinition]:
    """
    Lightweight word extraction:
      - Looks for "term: definition" or "term - definition" patterns
      - Keeps ASCII sequences; non-ASCII must exist in Math Galaxy
      - Returns WordDefinition objects with char_sequence symlinks
    """
    if not text:
        return []

    math_galaxy = get_math_galaxy()
    words: List[WordDefinition] = []
    seen_ids: Set[str] = set()

    # Acceptable "term: definition" patterns (simple heuristic)
    line_pattern = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _/\\-]{2,64})\s*[:\-–]\s*(.+)$")
    for line in text.splitlines():
        match = line_pattern.match(line)
        if not match:
            continue

        term_raw = match.group(1).strip()
        definition = match.group(2).strip()
        if not definition or len(definition.split()) < 2:
            continue

        # Build a stable word_id (symlink style, no duplication)
        slug = re.sub(r"[^a-z0-9]+", "_", term_raw.lower()).strip("_")
        word_id = f"{domain}_{slug}" if slug else None
        if not word_id or word_id in seen_ids:
            continue

        char_sequence = [ord(c) for c in term_raw]
        # Related symbols are only those present in Math Galaxy
        related_symbols = []
        for cp in set(ord(c) for c in term_raw + definition):
            if cp > 127 and math_galaxy.get(cp) is not None:
                related_symbols.append(cp)

        word = WordDefinition(
            word_id=word_id,
            char_sequence=char_sequence,
            definition=definition[:512],  # keep it small
            domain=domain,
            rpn_context=None,
            related_symbols=related_symbols,
            examples=[],
        )

        try:
            if word.validate_char_sequence():
                words.append(word)
                seen_ids.add(word_id)
        except Exception:
            # Skip invalid entries silently; ingestion should never crash hot path
            continue

        if len(words) >= 50:  # safety cap per PDF
            break

    return words


def extract_rules_from_text(text: str, domain: str) -> List[GrammarRule]:
    """
    Lightweight rule extraction:
      - Finds lines containing math/logical symbols
      - Builds GrammarRule with symbol_refs pointing to Math Galaxy
      - RPN program is a minimal placeholder (procedural, tiny)
    """
    if not text:
        return []

    math_galaxy = get_math_galaxy()
    rules: List[GrammarRule] = []
    seen_ids: Set[str] = set()

    symbol_regex = re.compile(r"[∑∫∂∇∆∏√∞±≈≠≤≥⊂⊃⊆⊇∪∩∅∀∃∧∨¬⇒⇔↔→←≡⊥∥∠△□○πμΣΩ]")

    for line in text.splitlines():
        if not symbol_regex.search(line):
            continue

        symbol_refs = []
        for cp in set(ord(c) for c in line if ord(c) > 127):
            if math_galaxy.get(cp) is not None:
                symbol_refs.append(cp)

        if not symbol_refs:
            continue

        slug = re.sub(r"[^a-z0-9]+", "_", line.lower()).strip("_")
        rule_id = f"{domain}_rule_{slug[:40]}" if slug else None
        if not rule_id or rule_id in seen_ids:
            continue

        rule = GrammarRule(
            rule_id=rule_id,
            language="math",
            pattern=line.strip()[:120],
            rpn_program="1",  # Minimal valid RPN placeholder
            domain=domain,
            symbol_refs=symbol_refs,
            word_refs=[],
            examples=[{"input": line.strip(), "output": line.strip()}],
            description="Auto-extracted rule (symlinked symbols)",
            is_canonical=False,
        )

        rules.append(rule)
        seen_ids.add(rule_id)

        if len(rules) >= 40:  # safety cap per PDF
            break

    return rules


def extract_meta_rules_from_text(text: str, category: str) -> List[MetaRule]:
    """
    Lightweight meta-rule extraction:
      - Looks for guidance cues ("should", "recommend", "strategy", "reflect")
      - Produces MetaRule entries referencing no duplicated rules (symlink later)
    """
    if not text:
        return []

    meta_rules: List[MetaRule] = []
    seen_ids: Set[str] = set()

    guidance_regex = re.compile(r"\b(should|recommend|strategy|reflect|practice|guidance|lesson)\b", re.IGNORECASE)

    for line in text.splitlines():
        if not guidance_regex.search(line):
            continue

        slug = re.sub(r"[^a-z0-9]+", "_", line.lower()).strip("_")
        meta_id = f"{category}_meta_{slug[:40]}" if slug else None
        if not meta_id or meta_id in seen_ids:
            continue

        meta = MetaRule(
            meta_id=meta_id,
            category=category,
            condition="1",  # always-true placeholder; real predicates come later
            action="1",  # minimal procedural placeholder
            rule_refs=[],
            priority=0.6,
            description=line.strip()[:200],
        )

        meta_rules.append(meta)
        seen_ids.add(meta_id)

        if len(meta_rules) >= 30:  # safety cap per PDF
            break

    return meta_rules


def ingest_category(category: str, base_path: Path) -> Dict[str, int]:
    """Ingest all PDFs in a category."""
    config = PDF_CATEGORIES[category]
    pdf_dir = base_path / config["path"]

    word_galaxy = get_word_galaxy()
    grammar_galaxy = GrammarGalaxy()
    eloquence_galaxy = get_eloquence_galaxy()

    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        print(f"Processing: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)

        if 2 in config["layer"]:
            for word in extract_words_from_text(text, category):
                word_galaxy.add_word(word)

        if 3 in config["layer"]:
            for rule in extract_rules_from_text(text, category):
                grammar_galaxy.add_rule(rule)

        if 4 in config["layer"]:
            for meta in extract_meta_rules_from_text(text, category):
                eloquence_galaxy.add_meta_rule(meta)

    return {
        "words": word_galaxy.stats()["total_words"],
        "rules": getattr(grammar_galaxy, "count", lambda: len(getattr(grammar_galaxy, "rules", {})))(),
        "meta_rules": eloquence_galaxy.stats()["total_meta_rules"],
    }


def main() -> None:
    base_path = Path("/K3D/Knowledge3D.local/datasets/foundational_pdfs")
    totals = {"words": 0, "rules": 0, "meta_rules": 0}

    if not base_path.exists():
        print(f"[ERROR] Base path not found: {base_path}")
        return

    for category in PDF_CATEGORIES:
        print(f"\n=== Ingesting {category} ===")
        stats = ingest_category(category, base_path)
        totals["words"] = stats["words"]
        totals["rules"] = stats["rules"]
        totals["meta_rules"] = stats["meta_rules"]
        print(f"  [STATS] Words={stats['words']}, Rules={stats['rules']}, Meta={stats['meta_rules']}")

    print("\n=== INGESTION COMPLETE ===")
    print(f"Layer 2 (Words): {totals['words']}")
    print(f"Layer 3 (Rules): {totals['rules']}")
    print(f"Layer 4 (Meta-Rules): {totals['meta_rules']}")


if __name__ == "__main__":
    main()
