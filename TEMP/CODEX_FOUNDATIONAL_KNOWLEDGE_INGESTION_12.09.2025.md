# Codex Briefing: Foundational Knowledge Ingestion (4-Layer Architecture)

**Date**: December 9, 2025
**Priority**: Critical
**Phase**: C.4 — Knowledge Ingestion (Post Math Galaxy)
**Pre-requisite**: Math Galaxy complete (176 symbols, verified 100% RPN)

---

## Context: Math Galaxy Results

Two full training runs (108 tasks × 162 epochs, hybrid mode) with Math Galaxy show:
- **Sustained 42-51%** accuracy (average ~46%)
- **Peak 50.93%** (vs 46.7% pre-Math)
- **No degradation** on 1.8× harder task set
- **100% PTX** (zero CPU fallbacks)

**Conclusion**: Architecture is stable. Ready for knowledge amplification via 74 PDFs.

---

## Mission

Ingest foundational knowledge into K3D's 4-layer architecture, following the **symlink pattern** to avoid duplication. Store knowledge in Galaxy/House (not weights). Enable TRM to reason using procedural knowledge references.

**Critical Constraint (from CLAUDE.md lines 94-100)**:
> DON'T duplicate what exists! Use references (symlink pattern):
> - Characters already have font + language + meaning
> - Words reference character IDs (not duplicate glyphs)
> - Grammar metadata references words (not duplicate strings)
> - Discoveries reference canonical programs

---

## 4-Layer Architecture

```
Layer 4: META-RULES (Strategy/Eloquence)     → eloquence_galaxy.py
    ↓ rule_refs (references Layer 3)
Layer 3: RULES (Grammar/Transformation)       → grammar_galaxy.py (exists, expand)
    ↓ symbol_refs, word_refs (references L1/L2)
Layer 2: MEANING (Words/Semantics)            → word_galaxy.py (CREATE)
    ↓ char_sequence (references Layer 1)
Layer 1: FORM (Characters/Glyphs)             → math_galaxy.py (exists, 176 symbols)
```

---

## Phase 1: Layer 2 — Word Galaxy (CREATE)

### 1.1 Create `knowledge3d/cranium/word_galaxy.py`

```python
"""
Word Galaxy — Semantic definitions referencing Layer 1 characters.
NO visual data duplication — only character ID sequences.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import json
from pathlib import Path

@dataclass
class WordDefinition:
    word_id: str                         # Unique identifier
    char_sequence: List[int]             # Layer 1 character codepoints (SYMLINK!)
    definition: str                      # Semantic meaning
    domain: str                          # math_calculus, pedagogy, eloquence, etc.
    rpn_context: Optional[str] = None    # RPN program providing usage context
    related_symbols: List[int] = field(default_factory=list)  # Layer 1 symbol refs
    examples: List[str] = field(default_factory=list)

    def validate_char_sequence(self) -> bool:
        """Ensure all characters exist (ASCII or in Math Galaxy)."""
        # ASCII 0-127 always valid, others must be in Math Galaxy
        from knowledge3d.cranium.math_galaxy import get_math_galaxy
        math_galaxy = get_math_galaxy()
        for cp in self.char_sequence:
            if cp > 127 and math_galaxy.get(cp) is None:
                return False
        return True


class WordGalaxy:
    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or Path("/K3D/Knowledge3D.local/galaxies/words")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._words: Dict[str, WordDefinition] = {}
        self._load()

    def _load(self):
        """Load words from storage."""
        words_file = self.storage_path / "words.json"
        if words_file.exists():
            data = json.loads(words_file.read_text())
            for word_data in data:
                word = WordDefinition(**word_data)
                self._words[word.word_id] = word

    def _save(self):
        """Persist words to storage."""
        words_file = self.storage_path / "words.json"
        data = [vars(w) for w in self._words.values()]
        words_file.write_text(json.dumps(data, indent=2))

    def add_word(self, word: WordDefinition) -> bool:
        """Add word if char_sequence validates."""
        if not word.validate_char_sequence():
            raise ValueError(f"Invalid char_sequence for {word.word_id}")
        self._words[word.word_id] = word
        self._save()
        return True

    def get(self, word_id: str) -> Optional[WordDefinition]:
        return self._words.get(word_id)

    def search_by_domain(self, domain: str) -> List[WordDefinition]:
        return [w for w in self._words.values() if w.domain == domain]

    def compose_from_text(self, text: str) -> List[int]:
        """Convert text string to Layer 1 char_sequence (symlink pattern)."""
        return [ord(c) for c in text]

    def stats(self) -> Dict:
        return {
            "total_words": len(self._words),
            "domains": list(set(w.domain for w in self._words.values())),
            "avg_char_sequence_len": sum(len(w.char_sequence) for w in self._words.values()) / max(len(self._words), 1)
        }


# Singleton accessor
_word_galaxy: Optional[WordGalaxy] = None

def get_word_galaxy() -> WordGalaxy:
    global _word_galaxy
    if _word_galaxy is None:
        _word_galaxy = WordGalaxy()
    return _word_galaxy
```

### 1.2 Storage Format

```json
// /K3D/Knowledge3D.local/galaxies/words/words.json
[
  {
    "word_id": "derivative",
    "char_sequence": [100, 101, 114, 105, 118, 97, 116, 105, 118, 101],
    "definition": "Rate of change of a function with respect to a variable",
    "domain": "math_calculus",
    "rpn_context": "FUNCTION RECALL VAR RECALL SYMBOL_DIFF",
    "related_symbols": [8706],
    "examples": ["The derivative of x^2 is 2x"]
  }
]
```

---

## Phase 2: Layer 3 — Expand Grammar Galaxy

### 2.1 Enhance Existing Rules with Symlinks

The `grammar_galaxy.py` already has `symbol_refs` and `word_refs` fields. Expand from 11 → 1,000+ rules.

### 2.2 Create Math Grammar Rules

```python
# Add to knowledge3d/training/arc_agi/math_grammar_rules.py

MATH_GRAMMAR_RULES = [
    GrammarRule(
        rule_id="calc_power_rule",
        language="math",
        pattern="d/dx[x^n]",
        rpn_program="n RECALL 1 SUB x SWAP POW n MUL",
        domain="math_calculus",
        symbol_refs=[8706, 8747],  # ∂, ∫ (SYMLINKS to Layer 1!)
        word_refs=["derivative", "exponent"],  # SYMLINKS to Layer 2!
        examples=[{"input": "d/dx[x^3]", "output": "3x^2"}]
    ),
    GrammarRule(
        rule_id="calc_chain_rule",
        language="math",
        pattern="d/dx[f(g(x))]",
        rpn_program="OUTER_FUNC RECALL INNER_FUNC RECALL COMPOSE SYMBOL_DIFF INNER_FUNC SYMBOL_DIFF MUL",
        domain="math_calculus",
        symbol_refs=[8706],  # ∂
        word_refs=["derivative", "composition"],
        examples=[{"input": "d/dx[sin(x^2)]", "output": "2x*cos(x^2)"}]
    ),
    GrammarRule(
        rule_id="stats_expected_value",
        language="math",
        pattern="E[X]",
        rpn_program="X_VALUES RECALL PROBS RECALL PAIRWISE_MUL SUMMATION",
        domain="math_statistics",
        symbol_refs=[8721],  # ∑ (SYMLINK!)
        word_refs=["expected_value", "probability"],
        examples=[{"input": "E[dice]", "output": "3.5"}]
    ),
    # ... 200+ math rules from PDFs
]
```

### 2.3 Create Language Grammar Rules

```python
LANGUAGE_GRAMMAR_RULES = [
    GrammarRule(
        rule_id="pt_verb_conjugate_present",
        language="pt",
        pattern="VERB_CONJUGATE",
        rpn_program="""
            VERB_STEM RECALL PERSON RECALL
            PERSON 1 == { VERB_STEM "o" CONCAT }
            PERSON 2 == { VERB_STEM "as" CONCAT }
            PERSON 3 == { VERB_STEM "a" CONCAT }
            ifelse ifelse
        """,
        domain="language_grammar",
        symbol_refs=[],  # No math symbols
        word_refs=["verb", "conjugation", "person"],
        examples=[{"verb_stem": "am", "person": 1, "output": "amo"}]
    ),
    # ... 300+ language rules from PDFs
]
```

---

## Phase 3: Layer 4 — Eloquence Galaxy (CREATE)

### 3.1 Create `knowledge3d/cranium/eloquence_galaxy.py`

```python
"""
Eloquence Galaxy — Meta-rules for strategy, pedagogy, self-reflection.
References Layer 3 rules, not duplicates.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import json
from pathlib import Path

@dataclass
class MetaRule:
    meta_id: str
    category: str  # eloquence, pedagogy, self_reflection, storytelling, delivery
    condition: str  # RPN predicate (when to apply)
    action: str     # RPN program (what to do)
    rule_refs: List[str] = field(default_factory=list)  # SYMLINKS to Layer 3!
    priority: float = 1.0
    description: str = ""

    def validate_rule_refs(self) -> bool:
        """Ensure referenced rules exist in Grammar Galaxy."""
        from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
        grammar = GrammarGalaxy()
        return all(grammar.get_rule(r) is not None for r in self.rule_refs)


class EloquenceGalaxy:
    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or Path("/K3D/Knowledge3D.local/galaxies/eloquence")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._meta_rules: Dict[str, MetaRule] = {}
        self._load()

    def _load(self):
        """Load meta-rules from storage."""
        meta_file = self.storage_path / "meta_rules.json"
        if meta_file.exists():
            data = json.loads(meta_file.read_text())
            for meta_data in data:
                meta = MetaRule(**meta_data)
                self._meta_rules[meta.meta_id] = meta

    def _save(self):
        """Persist meta-rules to storage."""
        meta_file = self.storage_path / "meta_rules.json"
        data = [vars(m) for m in self._meta_rules.values()]
        meta_file.write_text(json.dumps(data, indent=2))

    def add_meta_rule(self, meta: MetaRule) -> bool:
        self._meta_rules[meta.meta_id] = meta
        self._save()
        return True

    def get(self, meta_id: str) -> Optional[MetaRule]:
        return self._meta_rules.get(meta_id)

    def get_by_category(self, category: str) -> List[MetaRule]:
        return [m for m in self._meta_rules.values() if m.category == category]

    def evaluate_condition(self, meta_id: str, context: Dict) -> bool:
        """Evaluate meta-rule condition against context."""
        meta = self.get(meta_id)
        if not meta:
            return False
        # Execute RPN condition
        from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine
        engine = TieredRPNEngine()
        result = engine.execute(meta.condition, context)
        return bool(result)

    def stats(self) -> Dict:
        return {
            "total_meta_rules": len(self._meta_rules),
            "categories": list(set(m.category for m in self._meta_rules.values())),
            "avg_rule_refs": sum(len(m.rule_refs) for m in self._meta_rules.values()) / max(len(self._meta_rules), 1)
        }


# Singleton accessor
_eloquence_galaxy: Optional[EloquenceGalaxy] = None

def get_eloquence_galaxy() -> EloquenceGalaxy:
    global _eloquence_galaxy
    if _eloquence_galaxy is None:
        _eloquence_galaxy = EloquenceGalaxy()
    return _eloquence_galaxy
```

### 3.2 Example Meta-Rules

```python
PEDAGOGY_META_RULES = [
    MetaRule(
        meta_id="meta_scaffold_task",
        category="pedagogy",
        condition="TASK_DIFFICULTY 0.7 >",
        action="""
            TASK DECOMPOSE
            SUB_TASKS EACH { WORKED_EXAMPLE PROVIDE }
            GUIDED_PRACTICE
        """,
        rule_refs=["decomposition_rule", "worked_example_rule"],
        priority=0.9,
        description="Scaffold complex tasks with worked examples"
    ),
]

SELF_REFLECTION_META_RULES = [
    MetaRule(
        meta_id="meta_assess_performance",
        category="self_reflection",
        condition="TASK_COMPLETE",
        action="""
            TASK_RESULT EXPECTED_RESULT ==
            { SUCCESSFUL_RULES MARK_HIGH_PRIORITY }
            { ERROR_PATTERNS EXTRACT CORRECTIVE_RULES SUGGEST }
            ifelse
        """,
        rule_refs=["consolidation_priority_rule"],
        priority=0.95,
        description="Assess task performance for sleeptime consolidation"
    ),
]

ELOQUENCE_META_RULES = [
    MetaRule(
        meta_id="meta_build_ethos",
        category="eloquence",
        condition="AUDIENCE_SKEPTICAL",
        action="""
            SPEAKER_CREDENTIALS RECALL
            "As someone who has" NARRATIVE_OPENING
            EXPERIENCE DESCRIBE
            AUDIENCE_CONCERN ACKNOWLEDGE
        """,
        rule_refs=["narrative_opening_rule", "acknowledgment_rule"],
        priority=0.8,
        description="Build credibility with skeptical audience"
    ),
]
```

---

## Phase 4: PDF Ingestion Pipeline

### 4.1 Create `scripts/ingest_foundational_pdfs.py`

```python
"""
Ingest 74 PDFs into 4-layer architecture.
Extract: words (L2), rules (L3), meta-rules (L4).
ALL use symlink references to Layer 1 (no duplication!).
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict

from knowledge3d.cranium.word_galaxy import get_word_galaxy, WordDefinition
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy, GrammarRule
from knowledge3d.cranium.eloquence_galaxy import get_eloquence_galaxy, MetaRule
from knowledge3d.cranium.math_galaxy import get_math_galaxy

# PDF categories from FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md
PDF_CATEGORIES = {
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
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_words_from_text(text: str, domain: str) -> List[WordDefinition]:
    """Extract key terms and definitions from text."""
    words = []
    # Simple extraction - enhance with NLP as needed
    # ... term extraction logic ...
    return words

def extract_rules_from_text(text: str, domain: str) -> List[GrammarRule]:
    """Extract transformation rules from text."""
    rules = []
    # ... rule extraction logic ...
    return rules

def extract_meta_rules_from_text(text: str, category: str) -> List[MetaRule]:
    """Extract meta-rules (strategy/pedagogy) from text."""
    meta_rules = []
    # ... meta-rule extraction logic ...
    return meta_rules

def ingest_category(category: str, base_path: Path):
    """Ingest all PDFs in a category."""
    config = PDF_CATEGORIES[category]
    pdf_dir = base_path / config["path"]

    word_galaxy = get_word_galaxy()
    grammar_galaxy = GrammarGalaxy()
    eloquence_galaxy = get_eloquence_galaxy()

    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f"Processing: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)

        if 2 in config["layer"]:
            words = extract_words_from_text(text, category)
            for word in words:
                word_galaxy.add_word(word)

        if 3 in config["layer"]:
            rules = extract_rules_from_text(text, category)
            for rule in rules:
                grammar_galaxy.add_rule(rule)

        if 4 in config["layer"]:
            meta_rules = extract_meta_rules_from_text(text, category)
            for meta in meta_rules:
                eloquence_galaxy.add_meta_rule(meta)

    return {
        "words_added": word_galaxy.stats()["total_words"],
        "rules_added": grammar_galaxy.count(),
        "meta_rules_added": eloquence_galaxy.stats()["total_meta_rules"]
    }

def main():
    base_path = Path("/K3D/Knowledge3D.local/datasets/foundational_pdfs")

    totals = {"words": 0, "rules": 0, "meta_rules": 0}

    for category in PDF_CATEGORIES:
        print(f"\n=== Ingesting {category} ===")
        stats = ingest_category(category, base_path)
        totals["words"] += stats["words_added"]
        totals["rules"] += stats["rules_added"]
        totals["meta_rules"] += stats["meta_rules_added"]

    print(f"\n=== INGESTION COMPLETE ===")
    print(f"Layer 2 (Words): {totals['words']}")
    print(f"Layer 3 (Rules): {totals['rules']}")
    print(f"Layer 4 (Meta-Rules): {totals['meta_rules']}")

if __name__ == "__main__":
    main()
```

---

## Phase 5: Wire Into Training Pipeline

### 5.1 Update `sovereign_pipeline.py`

```python
# Add to sovereign_pipeline.py __init__

from knowledge3d.cranium.word_galaxy import get_word_galaxy
from knowledge3d.cranium.eloquence_galaxy import get_eloquence_galaxy

class SovereignPipeline:
    def __init__(self, ...):
        # Existing
        self.math_galaxy = get_math_galaxy()
        self.grammar_galaxy = GrammarGalaxy()

        # NEW: Load additional layers
        self.word_galaxy = get_word_galaxy()
        self.eloquence_galaxy = get_eloquence_galaxy()

        # Log counts
        print(f"[MATH] Loaded {len(self.math_galaxy.canonical_symbols)} symbols")
        print(f"[WORD] Loaded {self.word_galaxy.stats()['total_words']} words")
        print(f"[GRAMMAR] Loaded {self.grammar_galaxy.count()} rules")
        print(f"[ELOQUENCE] Loaded {self.eloquence_galaxy.stats()['total_meta_rules']} meta-rules")
```

### 5.2 Enable Semantic Routing via Word Galaxy

```python
def enhance_task_embedding(self, task_embedding, task_context: Dict):
    """Enhance embedding with Word Galaxy semantic context."""
    # Find relevant words for this task
    task_domain = self._detect_domain(task_context)
    relevant_words = self.word_galaxy.search_by_domain(task_domain)

    # Add word semantic context to embedding
    for word in relevant_words[:10]:  # Top 10 relevant
        if word.rpn_context:
            context_embedding = self.rpn_engine.execute(word.rpn_context)
            task_embedding = self._merge_embeddings(task_embedding, context_embedding)

    return task_embedding
```

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `knowledge3d/cranium/word_galaxy.py` | CREATE | Layer 2 word storage |
| `knowledge3d/cranium/eloquence_galaxy.py` | CREATE | Layer 4 meta-rule storage |
| `scripts/ingest_foundational_pdfs.py` | CREATE | PDF ingestion pipeline |
| `knowledge3d/training/arc_agi/math_grammar_rules.py` | MODIFY | Add 200+ math rules |
| `knowledge3d/training/arc_agi/language_grammar_rules.py` | CREATE | Add 300+ language rules |
| `knowledge3d/training/arc_agi/sovereign_pipeline.py` | MODIFY | Wire all 4 layers |
| `tests/test_knowledge_ingestion.py` | CREATE | Validate symlink integrity |

---

## Success Criteria

### Symlink Integrity (CRITICAL)
- [ ] Layer 2 words reference Layer 1 via `char_sequence` (no glyph duplication)
- [ ] Layer 3 rules reference Layer 1 via `symbol_refs` (no visual data)
- [ ] Layer 3 rules reference Layer 2 via `word_refs` (no string duplication)
- [ ] Layer 4 meta-rules reference Layer 3 via `rule_refs` (no rule duplication)

### Compression Targets
- [ ] 666× compression on repeated symbols (symlink vs duplicate)
- [ ] <50 KB total Layer 3 metadata (1,000 rules × 4-byte refs)

### Counts
- [ ] Layer 1: 176 symbols (already complete)
- [ ] Layer 2: 15,000+ words from PDFs
- [ ] Layer 3: 1,000+ rules (200 math + 300 language + 500 other)
- [ ] Layer 4: 500+ meta-rules (pedagogy, eloquence, self-reflection, story, delivery)

### Execution
- [ ] All rules execute in ModularRPNEngine without errors
- [ ] All symlink references resolve (validate_*_refs() pass)
- [ ] Pipeline loads all 4 layers at init

### ARC-AGI Impact
- [ ] Run post-ingestion training (108 × 162 hybrid)
- [ ] Target: >50% sustained (vs current ~46%)
- [ ] Document improvement attribution

---

## Existing Assets to Leverage

- `knowledge3d/cranium/math_galaxy.py` — Layer 1 complete (176 symbols)
- `knowledge3d/training/arc_agi/grammar_galaxy.py` — Layer 3 structure (11 rules, symlink fields ready)
- `knowledge3d/cranium/discovery_layer.py` — Cross-domain connection finder
- `scripts/extract_math_symbol_glyphs.py` — Bézier → RPN extraction pattern
- `/K3D/Knowledge3D.local/galaxies/` — Storage directory structure

---

## Execution Order

1. **Create `word_galaxy.py`** — Layer 2 infrastructure
2. **Create `eloquence_galaxy.py`** — Layer 4 infrastructure
3. **Expand `grammar_galaxy.py`** — Add math/language rules with symlinks
4. **Create `ingest_foundational_pdfs.py`** — PDF extraction pipeline
5. **Wire into `sovereign_pipeline.py`** — Enable at training init
6. **Run validation tests** — Ensure symlink integrity
7. **Run ARC-AGI training** — Measure improvement

---

## References

- `docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md` — Full 4-layer spec
- `CLAUDE.md` — Save Information Principle, symlink pattern
- `BRIEFING.md` — Dual Client Reality, procedural foundation
- `TEMP/CODEX_DRAWING_GALAXY_IMPLEMENTATION_12.07.2025.md` — Related galaxy work
