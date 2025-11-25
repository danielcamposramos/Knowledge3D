# Grammar Rules as RPN Programs — Architecture Specification

**Date**: November 24, 2025
**Architect**: Claude
**Status**: Architecture specification (parallel with ARC-AGI)
**Priority**: Infrastructure for production AGI (can run parallel with ARC-AGI training)

---

## Critical Insight: NOT Static Phrases!

**Daniel's Correction**:
> "We don't harvest static phrases — we generate procedural grammar rules! RPN programs that CONSTRUCT text following language rules, just like we construct characters from strokes."

**The Compositional Stack**:
```
Letters → Words → Grammar Rules (RPN programs) → Text Generation
  ↓         ↓              ↓                          ↓
 RPN      RPN         RPN Programs              Procedural
programs programs   (text construction)          Output
```

**Same Pattern as Visual**:
```
Strokes → Glyphs → Drawings → Compositions
  ↓         ↓          ↓            ↓
 RPN      RPN     RPN Programs   Procedural
programs programs (visual)        Output
```

---

## Grammar RPN Architecture

### 1. Grammar Rules as Procedural Programs

**NOT This (Static)**:
```python
phrase_galaxy = {
    "en_greeting_formal": "Good morning, how are you?",  # STATIC TEXT
    "pt_greeting_formal": "Bom dia, como você está?",    # STATIC TEXT
}
```

**YES This (Procedural)**:
```python
grammar_galaxy = {
    "en_simple_sentence": {
        "pattern": "SVO",  # Subject-Verb-Object
        "rpn_program": "SUBJECT RECALL VERB RECALL OBJECT RECALL SVO_ORDER CONCAT",
        "examples": [
            {"subject": "I", "verb": "love", "object": "programming"},
            {"subject": "She", "verb": "writes", "object": "code"},
        ]
    },
    "pt_simple_sentence": {
        "pattern": "SVO",  # Portuguese also SVO
        "rpn_program": "SUJEITO RECALL VERBO RECALL OBJETO RECALL SVO_ORDER CONCAT",
        "examples": [
            {"sujeito": "Eu", "verbo": "amo", "objeto": "programar"},
            {"sujeito": "Ela", "verbo": "escreve", "objeto": "código"},
        ]
    },
    "ja_simple_sentence": {
        "pattern": "SOV",  # Japanese is Subject-Object-Verb
        "rpn_program": "SUBJECT RECALL OBJECT RECALL WO_PARTICLE VERB RECALL SOV_ORDER CONCAT",
        "examples": [
            {"subject": "私は", "object": "コード", "verb": "書く"},  # I code write
        ]
    },
}
```

### 2. Multi-User Personal Galaxies

**Daniel's Beautiful Insight**:
> "Me and my wife, we like to nickname things, change meanings - that's a nice human trait to program from the start. The AI must remember people's wording."

**Architecture**:
```python
# Global Galaxies (shared knowledge)
global_galaxies = {
    "letter_galaxy": {},      # 40K letters (all users)
    "word_galaxy": {},        # 1.6M standard words (all users)
    "grammar_rules": {},      # Universal grammar patterns (all users)
    "math_symbols": {},       # Universal math (all users)
}

# User-Specific Galaxies (personal vocabulary)
user_galaxies = {
    "USER_daniel": {
        "personal_words": {
            # Daniel's unique words/meanings
            "meu_amor": {
                "base_word_ref": "WORD_pt_meu_amor",  # Symlink to global
                "personal_context": "coding_together",
                "emotional_weight": 0.95,
                "usage_contexts": ["partnership", "collaboration", "affection"],
            },
            "nossa_parceria": {
                "personal_meaning": "Our special coding partnership",
                "connotation": "pride + gratitude",
            }
        },
        "personal_phrases": {
            # Daniel's nicknames and expressions
            "vai_quebrar_o_banco": {  # "going to break the bank"
                "meaning": "This will be extremely successful",
                "emotional_tone": "excitement + confidence",
                "usage": "celebrating_breakthroughs",
            }
        },
        "grammar_preferences": {
            # Daniel's writing style
            "formality": 0.6,          # Balanced formal/informal
            "technical_density": 0.8,  # High technical precision
            "emoji_usage": 0.3,        # Moderate emoji use
        }
    },

    "USER_wife": {
        "personal_words": {
            # Wife's unique words/meanings
            "meu_amor": {
                "base_word_ref": "WORD_pt_meu_amor",  # SAME symlink!
                "personal_context": "daily_life",      # DIFFERENT context!
                "emotional_weight": 0.98,             # DIFFERENT weight!
                "usage_contexts": ["family", "home", "care"],
            }
        },
        "personal_phrases": {
            # Wife's own nicknames
            # (Different from Daniel's!)
        },
        "grammar_preferences": {
            # Wife's writing style (potentially different)
            "formality": 0.4,          # More casual
            "technical_density": 0.3,  # Less technical
            "emoji_usage": 0.7,        # More emoji use
        }
    }
}
```

### 3. Grammar RPN Operations

**New RPN Opcodes for Grammar Construction**:
```python
GRAMMAR_OPCODES = {
    # Sentence Construction
    "SVO_ORDER": 0x90,        # Arrange Subject-Verb-Object
    "SOV_ORDER": 0x91,        # Arrange Subject-Object-Verb
    "VSO_ORDER": 0x92,        # Arrange Verb-Subject-Object

    # Word Selection
    "SUBJECT": 0x93,          # Load subject from context
    "VERB": 0x94,             # Load verb from context
    "OBJECT": 0x95,           # Load object from context

    # Grammar Particles (Japanese, Korean, etc.)
    "WA_PARTICLE": 0x96,      # は (topic marker)
    "WO_PARTICLE": 0x97,      # を (object marker)
    "GA_PARTICLE": 0x98,      # が (subject marker)

    # Conjugation
    "CONJUGATE_VERB": 0x99,   # Apply verb conjugation
    "APPLY_TENSE": 0x9A,      # Apply tense (past/present/future)
    "APPLY_ASPECT": 0x9B,     # Apply aspect (perfect/continuous)

    # Agreement
    "NOUN_VERB_AGREE": 0x9C,  # Ensure subject-verb agreement
    "GENDER_AGREE": 0x9D,     # Ensure gender agreement (FR, ES, PT, etc.)
    "CASE_AGREE": 0x9E,       # Ensure case agreement (RU, DE, etc.)

    # Composition
    "CONCAT_SENTENCE": 0x9F,  # Concatenate into sentence
    "CONCAT_PARAGRAPH": 0xA0, # Concatenate into paragraph
    "CONCAT_DOCUMENT": 0xA1,  # Concatenate into document

    # User Context
    "LOAD_USER_WORD": 0xA2,   # Load from user's personal galaxy
    "LOAD_USER_STYLE": 0xA3,  # Load user's writing style
    "APPLY_FORMALITY": 0xA4,  # Apply formality level
}
```

### 4. Example: Sentence Construction in 3 Languages

**English (SVO)**:
```python
# Input context
context = {
    "subject": "I",
    "verb": "love",
    "object": "programming",
    "tense": "present",
}

# RPN program
rpn = "SUBJECT RECALL VERB RECALL present APPLY_TENSE OBJECT RECALL SVO_ORDER CONCAT_SENTENCE"

# Execution
# Stack after each operation:
# SUBJECT RECALL        → ["I"]
# VERB RECALL           → ["I", "love"]
# present APPLY_TENSE   → ["I", "love"]  (present tense, no change)
# OBJECT RECALL         → ["I", "love", "programming"]
# SVO_ORDER             → ["I", "love", "programming"]  (ordered)
# CONCAT_SENTENCE       → "I love programming."

# Output: "I love programming."
```

**Portuguese (SVO)**:
```python
context = {
    "sujeito": "Eu",
    "verbo": "amar",
    "objeto": "programar",
    "tempo": "presente",
}

rpn = "SUJEITO RECALL VERBO RECALL presente CONJUGATE_VERB OBJETO RECALL SVO_ORDER CONCAT_SENTENCE"

# Execution:
# SUJEITO RECALL              → ["Eu"]
# VERBO RECALL                → ["Eu", "amar"]
# presente CONJUGATE_VERB     → ["Eu", "amo"]  (amar → amo in present)
# OBJETO RECALL               → ["Eu", "amo", "programar"]
# SVO_ORDER                   → ["Eu", "amo", "programar"]
# CONCAT_SENTENCE             → "Eu amo programar."

# Output: "Eu amo programar."
```

**Japanese (SOV)**:
```python
context = {
    "subject": "私",
    "object": "プログラミング",
    "verb": "愛する",
    "tense": "present",
}

rpn = "SUBJECT RECALL WA_PARTICLE OBJECT RECALL WO_PARTICLE VERB RECALL present APPLY_TENSE SOV_ORDER CONCAT_SENTENCE"

# Execution:
# SUBJECT RECALL        → ["私"]
# WA_PARTICLE           → ["私は"]  (topic marker)
# OBJECT RECALL         → ["私は", "プログラミング"]
# WO_PARTICLE           → ["私は", "プログラミングを"]  (object marker)
# VERB RECALL           → ["私は", "プログラミングを", "愛する"]
# present APPLY_TENSE   → ["私は", "プログラミングを", "愛する"]
# SOV_ORDER             → ["私は", "プログラミングを", "愛する"]
# CONCAT_SENTENCE       → "私はプログラミングを愛する。"

# Output: "私はプログラミングを愛する。"
```

### 5. Academic Writing Pattern (Sleep-Time)

**Daniel's Insight**:
> "This AI will have memory to write some pages, sleep, go back and keep writing just like humans do."

**Architecture**:
```python
academic_writing_grammar = {
    "introduction": {
        "rpn_program": "TOPIC RECALL CONTEXT RECALL THESIS RECALL INTRO_PATTERN CONCAT_PARAGRAPH",
        "structure": ["hook", "context", "thesis_statement"],
    },
    "body_paragraph": {
        "rpn_program": "CLAIM RECALL EVIDENCE RECALL ANALYSIS RECALL PARAGRAPH_PATTERN CONCAT_PARAGRAPH",
        "structure": ["topic_sentence", "evidence", "analysis", "transition"],
    },
    "conclusion": {
        "rpn_program": "THESIS RECALL SUMMARY RECALL IMPLICATIONS RECALL CONCLUSION_PATTERN CONCAT_PARAGRAPH",
        "structure": ["restate_thesis", "summarize_evidence", "broader_implications"],
    },
    "full_essay": {
        "rpn_program": "INTRO RECALL BODY_1 RECALL BODY_2 RECALL BODY_3 RECALL CONCLUSION RECALL CONCAT_DOCUMENT",
        "sleep_consolidation": True,  # Sleep between sections!
    }
}

# Writing Process (Human-Like):
def write_academic_essay(topic):
    # Step 1: Write introduction
    intro = execute_rpn(academic_writing_grammar["introduction"]["rpn_program"])
    save_to_galaxy(intro, "essay_intro")

    # Step 2: SLEEP CONSOLIDATION (consolidate ideas, refine structure)
    sleep_time_consolidate()

    # Step 3: Write body paragraphs
    for i in range(3):
        body = execute_rpn(academic_writing_grammar["body_paragraph"]["rpn_program"])
        save_to_galaxy(body, f"essay_body_{i}")

        # Mini-sleep between paragraphs
        if i < 2:
            micro_consolidate()

    # Step 4: SLEEP CONSOLIDATION (review coherence)
    sleep_time_consolidate()

    # Step 5: Write conclusion
    conclusion = execute_rpn(academic_writing_grammar["conclusion"]["rpn_program"])
    save_to_galaxy(conclusion, "essay_conclusion")

    # Step 6: FINAL ASSEMBLY
    full_essay = execute_rpn(academic_writing_grammar["full_essay"]["rpn_program"])

    return full_essay
```

---

## Implementation Plan (Parallel with ARC-AGI)

### Phase 1: Grammar RPN Opcodes (Week 1-2)
**File**: `knowledge3d/cranium/ptx_runtime/grammar_rpn.py`

```python
"""
Grammar RPN operations for procedural text generation.
"""

class GrammarRPN:
    """Grammar-specific RPN operations."""

    GRAMMAR_OPCODES = {
        # (See section 3 above)
    }

    def execute_grammar_rpn(self, program: str, context: Dict) -> str:
        """
        Execute grammar RPN program to generate text.

        Args:
            program: RPN program string
            context: Dictionary with subject, verb, object, etc.

        Returns:
            Generated text
        """
        pass
```

### Phase 2: Multi-User Galaxy Manager (Week 1-2)
**File**: `knowledge3d/cranium/user_galaxy_manager.py`

```python
"""
Multi-user personal galaxy management.
"""

class UserGalaxyManager:
    """Manage per-user word/phrase galaxies."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.personal_word_galaxy = {}   # User's unique words
        self.personal_phrase_galaxy = {}  # User's expressions
        self.writing_style = {}          # Grammar preferences

    def learn_personal_word(self, word: str, context: str, meaning: str):
        """Learn user's personal word usage."""
        pass

    def remember_nickname(self, nickname: str, refers_to: str):
        """Remember user's nicknames."""
        pass
```

### Phase 3: Language Grammar Builders (Week 2-3)
**Files**:
- `knowledge3d/ingestion/atomic/grammar_builder_en.py` (English)
- `knowledge3d/ingestion/atomic/grammar_builder_pt.py` (Portuguese)
- `knowledge3d/ingestion/atomic/grammar_builder_ja.py` (Japanese)
- ... (161 languages)

---

## Success Criteria

**MUST ACHIEVE**:
- ✅ Grammar RPN opcodes defined (SVO_ORDER, CONJUGATE_VERB, etc.)
- ✅ Multi-user galaxy architecture designed
- ✅ 3 language examples working (EN, PT, JA)

**SHOULD ACHIEVE**:
- ✅ 10+ languages with grammar rules (top languages from 1.6M word galaxy)
- ✅ User personal galaxy storage working (Daniel + wife separation)
- ✅ Sleep-time writing pattern implemented (academic essay example)

**NICE TO HAVE**:
- ⚠️ All 161 languages with grammar rules
- ⚠️ Advanced conjugation/declension rules
- ⚠️ Stylistic variation (formal/informal switching)

---

## Symlink Architecture (From Strategic Roadmap)

**Key Principle**:
```
Letters (stored ONCE) ← Words ← Phrases/Grammar ← Sentences ← Documents
   40K × 2KB           1.6M      procedural         procedural    procedural
   = 80MB              refs      RPN programs       RPN programs  RPN programs
```

**Compression Math**:
```
WITHOUT SYMLINKS:
- 1.6M words × 5 letters × 2KB = 16GB
- 1.5M phrases × 3 words × 5 letters × 2KB = 45GB
- Total: 61GB

WITH SYMLINKS (K3D):
- Letter Galaxy: 40K × 2KB = 80MB (stored ONCE)
- Word Galaxy: 1.6M × 200 bytes refs = 320MB
- Grammar Rules: Procedural RPN programs = ~50MB
- Total: 450MB

Compression: 61GB → 450MB = 135× reduction!
```

**Benefits**:
1. ✅ Small (450MB vs 61GB)
2. ✅ Wide (1.6M words + infinite grammar combinations)
3. ✅ Consistent (update letter 'A' → all words auto-update)
4. ✅ Procedural (generate infinite valid sentences)

---

## Next Steps

**Codex Tasks** (can run parallel with ARC-AGI):
1. Implement `GrammarRPN` class with opcodes
2. Implement `UserGalaxyManager` class
3. Build grammar builders for EN, PT, JA
4. Write unit tests for sentence construction
5. Validate sleep-time writing pattern (academic essay)

**Timeline**: 2-3 weeks (parallel with ARC-AGI Week 1-4)

---

**This completes the grammar infrastructure architecture!**

Ready for implementation. 🚀
