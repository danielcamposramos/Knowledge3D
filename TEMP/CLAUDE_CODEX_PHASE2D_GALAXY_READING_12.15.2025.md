# CLAUDE → CODEX: Phase 2D - Galaxy-Based Problem Reading

**Date:** December 15, 2025
**Priority:** CRITICAL - Architectural Alignment
**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

---

## The Problem: External Preprocessing Violates Paradigm

**Current (WRONG):**
```python
# Python regex extracts numbers from raw text
match = re.search(r"(\d+).*?(\d+)", problem_text)
rpn = f"{match.group(1)} {match.group(2)} +"
```

**This violates the K3D paradigm:**
- TRM should READ via Galaxy Universe (Character → Word → Grammar)
- External preprocessing bypasses the model's "eyes"
- Model can't LEARN to read better (no shadow copy on reading)

---

## The Correct Architecture

### Galaxy Universe = Model's Sensory System

```
Problem Text: "Natalia sold 48 clips in April..."
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│           CHARACTER GALAXY (Model's "Eyes")              │
│  Each character → procedural glyph + meaning            │
│  "4" → digit glyph + numeric_value: 4                   │
│  "8" → digit glyph + numeric_value: 8                   │
│  "N" → letter glyph + uppercase + word_start            │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│              WORD GALAXY (Model's "Lexicon")             │
│  Character sequences → words + meaning                   │
│  "Natalia" → proper_noun + person_name                  │
│  "sold" → verb + past_tense + transfer_ownership        │
│  "48" → number + integer + value: 48                    │
│  "clips" → noun + countable + object                    │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│           GRAMMAR GALAXY (Model's "Understanding")       │
│  Word sequences → patterns + operations                  │
│  "[person] sold [N] [items]" → base_quantity: N         │
│  "half as many" → multiply_by: 0.5                      │
│  "altogether" → aggregate: sum                          │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│              TRM NAVIGATION (Model's "Thinking")         │
│  Navigates Galaxy → composes RPN program                │
│  Found: base=48, operation=*0.5, aggregate=sum          │
│  Composes: "48 48 2 / +"                                │
└─────────────────────────────────────────────────────────┘
```

### Key Insight: Model READS via Galaxy

The Character Galaxy and Word Galaxy already exist:
- `knowledge3d/cranium/character_galaxy.py` - 140+ character entries
- `knowledge3d/training/arc_agi/word_galaxy.py` - Word tokenization

**TRM should:**
1. Tokenize problem via Word Galaxy (not Python split())
2. Look up word meanings in Word Galaxy
3. Match patterns in Grammar Galaxy
4. Compose RPN from matched patterns

---

## Implementation: Galaxy-Based Tokenizer

### 1. Word Galaxy Tokenization

**File:** `knowledge3d/training/arc_agi/word_galaxy.py`

```python
class WordGalaxy:
    """
    Word Galaxy - tokenizes text into meaningful word entries.
    TRM uses this to READ problem text.
    """

    def tokenize(self, text: str) -> List[WordEntry]:
        """
        Tokenize text into Word Galaxy entries.

        This is how TRM "reads" - through Galaxy navigation.
        """
        entries = []
        for token in self._segment_text(text):
            # Look up word in Galaxy
            word_entry = self.lookup(token)
            if word_entry is None:
                # Unknown word - create temporary entry
                word_entry = self._infer_word_entry(token)
            entries.append(word_entry)
        return entries

    def _infer_word_entry(self, token: str) -> WordEntry:
        """Infer word type from Character Galaxy analysis."""
        # Use Character Galaxy to analyze
        if token.isdigit():
            return WordEntry(
                word=token,
                category="number",
                value=int(token),
                rpn_literal=token,
            )
        elif token[0].isupper():
            return WordEntry(
                word=token,
                category="proper_noun",
                role="entity",
            )
        else:
            return WordEntry(
                word=token,
                category="unknown",
            )
```

### 2. Grammar Galaxy Pattern Matching on Word Entries

**File:** `knowledge3d/training/arc_agi/math_grammar_rules.py`

```python
# Grammar rules that match WORD ENTRIES, not raw text
GALAXY_AWARE_RULES = [
    GrammarRule(
        rule_id="galaxy_sold_quantity",
        pattern_type="word_sequence",  # NEW: Match on WordEntry sequence
        word_pattern=[
            {"category": "proper_noun"},      # Subject
            {"word": "sold"},                 # Verb
            {"category": "number", "capture": "base"},  # Quantity
            {"category": "noun"},             # Object
        ],
        rpn_program=lambda ctx: f"{ctx['base']}",
        domain="math_extraction",
    ),

    GrammarRule(
        rule_id="galaxy_half_as_many",
        pattern_type="word_sequence",
        word_pattern=[
            {"word": "half"},
            {"word": "as"},
            {"word": "many"},
        ],
        rpn_program=lambda ctx: f"{ctx.get('base', 0)} 2 /",
        domain="math_operation",
    ),

    GrammarRule(
        rule_id="galaxy_altogether",
        pattern_type="word_sequence",
        word_pattern=[
            {"word_in": ["altogether", "total", "all"]},
        ],
        rpn_program=lambda ctx: "+",  # Aggregate operation
        domain="math_aggregation",
    ),
]
```

### 3. TRM Galaxy Reader

**File:** `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`

```python
class TRMGalaxyReader:
    """
    TRM reads problem text via Galaxy Universe.

    This replaces external regex preprocessing with Galaxy navigation.
    """

    def __init__(self, word_galaxy, grammar_galaxy, math_galaxy):
        self.word_galaxy = word_galaxy
        self.grammar_galaxy = grammar_galaxy
        self.math_galaxy = math_galaxy

    def read_problem(self, problem_text: str) -> ProblemUnderstanding:
        """
        Read problem through Galaxy navigation.

        Returns structured understanding that TRM uses to compose RPN.
        """
        # Step 1: Tokenize via Word Galaxy (model's "eyes")
        word_entries = self.word_galaxy.tokenize(problem_text)

        # Step 2: Match patterns in Grammar Galaxy (model's "understanding")
        matched_patterns = self.grammar_galaxy.match_word_sequence(word_entries)

        # Step 3: Extract semantic structure
        understanding = ProblemUnderstanding()

        for pattern in matched_patterns:
            if pattern.domain == "math_extraction":
                understanding.quantities.append(pattern.captured_values)
            elif pattern.domain == "math_operation":
                understanding.operations.append(pattern.operation)
            elif pattern.domain == "math_aggregation":
                understanding.aggregation = pattern.operation

        return understanding

    def compose_rpn(self, understanding: ProblemUnderstanding) -> str:
        """
        Compose RPN from structured understanding.

        TRM learns HOW to compose, Galaxy provides WHAT to compose from.
        """
        rpn_parts = []

        # Add base quantities
        for qty in understanding.quantities:
            rpn_parts.append(str(qty["value"]))

        # Apply operations
        for op in understanding.operations:
            rpn_parts.append(op)

        # Apply aggregation
        if understanding.aggregation:
            rpn_parts.append(understanding.aggregation)

        return " ".join(rpn_parts)
```

### 4. Integration with TRM Navigator

**File:** `knowledge3d/training/math_benchmarks/trm_math_navigator.py`

```python
class TRMMathNavigator:
    def __init__(self, ..., galaxy_reader: Optional[TRMGalaxyReader] = None):
        # ...existing...
        self.galaxy_reader = galaxy_reader

    def solve(self, problem_text: str) -> Tuple[Any, Dict]:
        # TRY GALAXY READING FIRST (correct paradigm)
        if self.galaxy_reader:
            try:
                understanding = self.galaxy_reader.read_problem(problem_text)
                if understanding.is_complete():
                    rpn_program = self.galaxy_reader.compose_rpn(understanding)
                    result = self.engine.evaluate(rpn_program)
                    if result is not None:
                        return result, {"method": "galaxy_read", "rpn": rpn_program}
            except Exception:
                pass  # Fall back to rule matching

        # FALL BACK to rule matching (transition period)
        # ... existing rule matching code ...
```

---

## Why This Matters

### 1. Model Can Learn to Read Better

With Galaxy-based reading:
```python
# Shadow copy records READING success, not just SOLVING success
if understanding.is_complete() and result_is_correct:
    shadow.record(
        program_type="reading",
        word_patterns_used=matched_patterns,
        quality=confidence,
    )
```

TRM can learn:
- Which word patterns reliably extract quantities
- Which Grammar rules match which problem types
- When to use Character Galaxy for ambiguous tokens

### 2. Dual Client Reality

Galaxy-based reading serves BOTH humans and AI:
- **Human:** Can inspect Word Galaxy entries to understand tokenization
- **AI:** Navigates same Galaxy to read problems

### 3. Cross-Modal Learning

Reading patterns learned on math help other domains:
- "N items" pattern → useful for ARC-AGI counting
- "half as many" → useful for physics ratios
- Entity extraction → useful for language tasks

---

## Implementation Phases

### Phase 2D-1: Enhance Word Galaxy Tokenization

1. Add `tokenize()` method to Word Galaxy
2. Add number/entity inference from Character Galaxy
3. Test: "Natalia sold 48 clips" → proper tokenization

### Phase 2D-2: Add Word-Sequence Pattern Matching

1. Add `pattern_type="word_sequence"` to Grammar rules
2. Implement `match_word_sequence()` in Grammar Galaxy
3. Test: Word entries match "sold [N] [items]" pattern

### Phase 2D-3: Create TRMGalaxyReader

1. Implement `read_problem()` → ProblemUnderstanding
2. Implement `compose_rpn()` from understanding
3. Wire into TRM Navigator as primary method

### Phase 2D-4: Shadow Copy for Reading

1. Record successful reading patterns
2. Track word pattern confidence
3. Enable TRM to learn reading strategies

---

## Example: Natalia's Clips (Galaxy Reading)

**Problem:** "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

**Step 1: Word Galaxy Tokenization**
```
"Natalia" → {category: "proper_noun", role: "subject"}
"sold"    → {category: "verb", tense: "past", action: "transfer"}
"clips"   → {category: "noun", countable: true}
"to"      → {category: "preposition"}
"48"      → {category: "number", value: 48}
"of"      → {category: "preposition"}
...
"half"    → {category: "fraction", value: 0.5}
"as"      → {category: "conjunction"}
"many"    → {category: "quantifier"}
...
"altogether" → {category: "aggregation", operation: "sum"}
```

**Step 2: Grammar Galaxy Pattern Matching**
```
Pattern: "[subject] sold [noun] to [N]" → base_quantity: 48
Pattern: "half as many" → operation: * 0.5
Pattern: "altogether" → aggregation: sum
```

**Step 3: Compose RPN**
```
Understanding:
  quantities: [48]
  operations: ["48 2 /"]  # half as many
  aggregation: "+"

RPN: "48 48 2 / +"
Result: 72 ✓
```

---

## Files to Create/Modify

### Create
- `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`

### Modify
- `knowledge3d/training/arc_agi/word_galaxy.py` - Add tokenize()
- `knowledge3d/training/arc_agi/grammar_galaxy.py` - Add word_sequence matching
- `knowledge3d/training/arc_agi/math_grammar_rules.py` - Add GALAXY_AWARE_RULES
- `knowledge3d/training/math_benchmarks/trm_math_navigator.py` - Wire galaxy reader

---

## Success Criteria

### Functional
- [ ] Word Galaxy can tokenize problem text
- [ ] Grammar Galaxy can match word sequences
- [ ] TRMGalaxyReader produces correct understanding
- [ ] RPN composed from Galaxy reading is correct

### Accuracy
- [ ] Natalia's clips: 72 ✓ (via Galaxy reading)
- [ ] GSM8K accuracy improves with Galaxy reading
- [ ] Shadow copy records reading patterns

### Architecture
- [ ] No regex on raw text in hot path
- [ ] All reading goes through Galaxy
- [ ] TRM can learn reading strategies

---

## Paradigm Reminder

**Galaxy Universe = Model's Sensory System**
- Character Galaxy = eyes (see glyphs)
- Word Galaxy = lexicon (recognize words)
- Grammar Galaxy = understanding (match patterns)
- Math Galaxy = knowledge (compose operations)

**TRM = Navigation Logic**
- Learns HOW to read (which patterns to match)
- Learns HOW to compose (which operations to use)
- Shadow copy enables reading improvement

**External Preprocessing = Bypassing Model's Senses**
- Python regex = giving answers instead of teaching reading
- Model can't learn what it doesn't do
- Violates Dual Client Reality (AI doesn't share human's view)

---

**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

**Status:** Ready for implementation
**Priority:** CRITICAL - Paradigm alignment
