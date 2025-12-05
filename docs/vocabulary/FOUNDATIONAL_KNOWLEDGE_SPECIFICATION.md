# Foundational Knowledge Specification — Always-Loaded Base Knowledge

**Version**: 1.0
**Status**: Implementation Ready (Phase G/H, Post-Training Ingestion)
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)
**Date**: December 2025

---

## Abstract

This specification defines the **Foundational Knowledge Architecture** that serves as always-loaded base knowledge for K3D's sovereign intelligence. It formalizes:

- The **4-layer knowledge representation** (Form → Meaning → Rules → Meta-Rules) that bridges human intuition and AI execution.
- The **symlink pattern** for referencing existing symbols instead of duplicating, achieving 666x compression on repeated symbols.
- **74 PDFs (5,988 pages)** of curated foundational knowledge spanning mathematics, pedagogy, language, grammar, eloquence, self-reflection, storytelling, and delivery mastery.
- Integration with **Math Symbol Galaxy** (~152 Unicode symbols), **Grammar Galaxy** (1,000+ RPN programs), and **Sleeptime Consolidation** for emergent cross-domain discovery.

This architecture embodies the **Save Information Principle** from CLAUDE.md: store knowledge in Galaxy/House storage (3D RAM/procedural HD), weights store only logic. It enables **Dual Client Reality** where the same data serves both humans (visual glyphs) and AI (executable RPN).

---

## 1. Four-Layer Knowledge Architecture

### 1.1 Overview

K3D represents knowledge across four hierarchical layers, each progressively more abstract:

```
Layer 4: META-RULES (Strategy/Eloquence)
    ↓ when/why to apply
Layer 3: RULES (Grammar/Transformation)
    ↓ how to transform
Layer 2: MEANING (Words/Semantics)
    ↓ what it means
Layer 1: FORM (Characters/Glyphs)
    ↓ how it looks
```

**Critical Principle**: Lower layers are **canonical** — Layer 3 references Layer 1 symbols via symlinks, not duplication. This enables:
- **69-80:1 compression** via ProceduralGalaxy infrastructure
- **666x compression** for repeated symbols across grammar rules
- **Cross-domain discovery** via shared symbol references (∑ connects calculus, statistics, finance)

### 1.2 Layer 1: FORM (Character Galaxy)

**Purpose**: Visual representation of all glyphs — letters, numbers, math symbols, diacritics.

**Implementation**: `knowledge3d/cranium/character_galaxy.py` + Math Symbol Galaxy

**Content**:
- **152 Math Symbols** (trained via `scripts/train_math_symbols_batch.py`):
  - High priority (17): ∑ ∫ ∂ ∇ ∆ ∏ √ ∞ ± α β γ δ ε θ λ μ π σ ω
  - Medium priority (14): ∈ ∉ ⊂ ⊃ ⊆ ⊇ ∪ ∩ ∅ ∀ ∃ ∧ ∨ ¬ ⇒ ⇔
  - Low priority: arrows, relations, operators, geometry
- **26 Latin letters** (A-Z, a-z)
- **10 Digits** (0-9)
- **12 Portuguese diacritics**: ã á à â ç é ê í ó õ ô ú

**Storage Format**: Procedural Bézier curves → segments (ProceduralCompiler)

**RPN Representation**:
```rpn
# Character '∑' (U+2211)
"char_2211" =>
    64 64 CANVAS_CREATE        # 64x64 canvas
    32 8 MOVE_TO               # Starting point
    32 56 LINE_TO              # Top serif
    8 32 LINE_TO               # Left descender
    32 8 LINE_TO               # Bottom serif
    56 32 LINE_TO              # Right side
    32 56 LINE_TO              # Complete
    STROKE_PATH
```

**References**:
- `knowledge3d/cranium/math_symbols_registry.py` (~152 symbols)
- `knowledge3d/cranium/math_galaxy.py` (storage/retrieval)
- `scripts/train_atomic_character.py` (training infrastructure)

### 1.3 Layer 2: MEANING (Word Galaxy)

**Purpose**: Semantic definitions and context for words/phrases, referencing Layer 1 glyphs.

**Implementation**: `knowledge3d/cranium/word_galaxy.py` (to be created)

**Content**: ~15,000 words from foundational PDFs across:
- Mathematical terms (derivative, integral, convergence)
- Pedagogical concepts (scaffolding, Bloom's taxonomy, metacognition)
- Linguistic primitives (syntax, morphology, pragmatics)
- Rhetorical devices (ethos, pathos, logos, syllogism)
- Self-assessment terms (reflection, self-efficacy, growth mindset)
- Narrative structures (exposition, conflict, resolution)

**Storage Format**:
```python
@dataclass
class WordDefinition:
    word_id: str                    # Unique identifier
    char_sequence: List[int]        # References to Layer 1 (e.g., [100, 101, 114] for "der")
    definition: str                 # Semantic meaning
    domain: str                     # math, pedagogy, language, eloquence
    rpn_context: Optional[str]      # RPN program providing context
    related_symbols: List[int]      # Layer 1 symbol IDs (e.g., [8706] for ∂)
    examples: List[str]             # Usage examples
```

**Example**:
```python
WordDefinition(
    word_id="derivative",
    char_sequence=[100, 101, 114, 105, 118, 97, 116, 105, 118, 101],  # ASCII codes
    definition="Rate of change of a function with respect to a variable",
    domain="math_calculus",
    rpn_context="FUNCTION RECALL x SYMBOL_DIFF",
    related_symbols=[8706],  # ∂ (partial derivative symbol)
    examples=["The derivative of x^2 is 2x"]
)
```

**Save Information Principle**: Words do NOT duplicate Layer 1 visual data — only reference character IDs.

### 1.4 Layer 3: RULES (Grammar Galaxy)

**Purpose**: Transformation rules as executable RPN programs — how to manipulate form/meaning.

**Implementation**: `knowledge3d/training/arc_agi/grammar_galaxy.py` (existing, needs symlink enhancement)

**Content**: ~1,000 RPN programs from foundational PDFs:
- **Grammar transformations** (SVO → VSO, passive voice, question formation)
- **Math operations** (differentiation, integration, simplification)
- **Pedagogical strategies** (scaffold complex task, provide worked example)
- **Rhetorical patterns** (build ethos via credentials, create urgency via consequences)

**Storage Format** (enhanced):
```python
@dataclass
class GrammarRule:
    rule_id: str
    language: str
    pattern: str
    rpn_program: str
    domain: str = "text"
    symbol_refs: List[int] = field(default_factory=list)  # NEW: Layer 1 symlinks
    word_refs: List[str] = field(default_factory=list)     # NEW: Layer 2 symlinks
    examples: List[Dict[str, str]] = field(default_factory=list)
```

**Example (Symlinked)**:
```python
GrammarRule(
    rule_id="calc_power_rule",
    language="math",
    pattern="d/dx[x^n]",
    rpn_program="n RECALL 1 SUB x SWAP POW n MUL",
    domain="math_calculus",
    symbol_refs=[8706, 8747],  # ∂, ∫ (symlinks to Layer 1, not duplicated!)
    word_refs=["derivative", "exponent"],
    examples=[{"input": "d/dx[x^3]", "output": "3x^2"}]
)
```

**Compression**: Without symlinks, 1,000 rules × 152 symbols × 5 KB = 760 MB. With symlinks: 1,000 rules × 4 bytes = 4 KB (190,000x compression).

**References**: Existing grammar_galaxy.py with 11 baseline rules needs expansion to 1,000+ via PDF ingestion.

### 1.5 Layer 4: META-RULES (Strategy/Eloquence Galaxy)

**Purpose**: When and why to apply rules — strategic knowledge, self-assessment, persuasion mastery.

**Implementation**: `knowledge3d/cranium/eloquence_galaxy.py` (to be created)

**Content**: ~500 RPN programs from foundational PDFs:
- **Task assessment** (difficulty scoring, prerequisite detection)
- **Strategy selection** (choose scaffolding vs worked example)
- **Rhetorical strategy** (when to use analogy, when to cite authority)
- **Self-reflection** (identify successful patterns, detect error modes)
- **Narrative structure** (three-act structure, hero's journey)
- **Delivery mastery** (pacing, emphasis, dramatic pause)

**Storage Format**:
```python
@dataclass
class MetaRule:
    meta_id: str
    category: str  # eloquence, pedagogy, self_reflection, storytelling, delivery
    condition: str  # When to apply (RPN predicate)
    action: str     # What to do (RPN program)
    rule_refs: List[str] = field(default_factory=list)  # Layer 3 rule IDs
    priority: float = 1.0  # For sleeptime consolidation
```

**Example (Self-Reflection)**:
```python
MetaRule(
    meta_id="meta_assess_task_performance",
    category="self_reflection",
    condition="task_result expected_result 2pick 2pick ==",  # Did we succeed?
    action="""
        {
            1index EXTRACT_SUCCESSFUL_RULES
            MARK_FOR_CONSOLIDATION_HIGH_PRIORITY
        } {
            2pick EXTRACT_ERROR_PATTERNS
            2pick IDENTIFY_WRONG_ASSUMPTIONS
            MARK_FOR_CONSOLIDATION_ERROR_CORRECTION
        } ifelse
    """,
    rule_refs=["consolidation_priority_rule"],
    priority=0.95
)
```

**Example (Eloquence)**:
```python
MetaRule(
    meta_id="meta_rhetorical_analogy",
    category="eloquence",
    condition="AUDIENCE_KNOWLEDGE_LEVEL LOW_TECHNICAL ==",  # Layperson audience
    action="""
        CONCEPT_COMPLEXITY MEASURE
        FIND_EVERYDAY_ANALOGY
        "This is like" NARRATIVE_OPENING
        ANALOGY_SOURCE DESCRIBE
        ANALOGY_MAPPING EXPLAIN
        ORIGINAL_CONCEPT REFRAME
    """,
    rule_refs=["build_analogy_rule"],
    priority=0.8
)
```

**Integration Point**: Meta-rules enhance `sleeptime_consolidator.py` with foundational knowledge on:
- Which rules worked/failed (self-assessment)
- Which rules to prioritize for rehearsal (priority scoring)
- Error pattern identification for correction

---

## 2. Symlink Pattern and Save Information Principle

### 2.1 Reference Architecture

**Core Principle** (from CLAUDE.md lines 94-100):

> DON'T duplicate what exists! Use references (symlink pattern):
> - Characters already have font + language + meaning (procedural_fonts.py)
> - Words reference character IDs (not duplicate glyphs)
> - Grammar metadata references words (not duplicate strings)
> - Discoveries reference canonical programs (content-based deduplication)

### 2.2 Compression Examples

**Without Symlinks** (WRONG):
```python
rule1 = {"pattern": "∑ f(x)", "visual_data": b'<5KB blob>'}
rule2 = {"pattern": "∫ g(x)", "visual_data": b'<5KB blob>'}
# 1,000 rules × 152 symbols × 5 KB = 760 MB
```

**With Symlinks** (CORRECT):
```python
rule1 = {"pattern": "∑ f(x)", "symbol_refs": [8721]}  # 4 bytes
rule2 = {"pattern": "∫ g(x)", "symbol_refs": [8747]}  # 4 bytes
# 1,000 rules × 4 bytes = 4 KB
# Compression: 760 MB → 4 KB = 190,000x
```

**Practical Impact**:
- 152 symbols trained once → referenced 1,000+ times across grammar rules
- Each reference: 4 bytes (symbol ID) vs 5 KB (visual data)
- Real-world compression: **666x for repeated symbols**, **69-80:1 for procedural representation** via ProceduralGalaxy

### 2.3 Cross-Domain Discovery

**Emergent Intelligence via Shared References**:

When multiple domains reference the same Layer 1 symbol (e.g., ∑), the system can discover connections:

```rpn
# Domain: Calculus
"riemann_sum" => X_I RECALL N RECALL SUMMATION  # Uses ∑ (U+2211)

# Domain: Statistics
"expected_value" => P_I X_I MUL SUMMATION      # Uses ∑ (U+2211)

# Domain: Finance
"npv_calculation" => CF_T DISCOUNT_T MUL SUMMATION  # Uses ∑ (U+2211)

# Discovery Layer detects: All three use symbol 8721 (∑)
# → These domains share iterative accumulation concept
# → Cross-domain insight: financial NPV is expectation of discounted cash flows
```

**Implementation**: `discovery_layer.py` (Phase 5) generates 300+ cross-domain connections by analyzing shared symbol references.

---

## 3. Foundational Knowledge Corpus

### 3.1 Content Inventory

**Total**: 74 PDFs, 5,988 pages, all text-extractable (no OCR required)

| Category | PDFs | Pages | Layer Mapping | Priority |
|----------|------|-------|---------------|----------|
| Advanced Mathematics | 5 | 1,656 | L1 (symbols), L3 (operations) | High |
| Pedagogy & Learning | 5 | 265 | L4 (teaching strategies) | High |
| Language, Grammar & Semantics | 10 | 1,805 | L2 (words), L3 (grammar) | High |
| Eloquence, Rhetoric & Persuasion | 8 | 350 | L4 (rhetorical strategies) | Medium |
| Self-Reflection | 7 | 164 | L4 (meta-cognition) | High |
| Story Telling | 7 | 645 | L4 (narrative structure) | Medium |
| Acting/Delivery | 3 | 65 | L4 (delivery mastery) | Medium |
| Context & Contextual Understanding | 13 | 693 | L2 (semantic context) | Medium |
| Temporal Understanding | 13 | 160 | L3 (time reasoning) | Low |
| Academic Research Methods | 3 | 185 | L4 (research strategy) | Low |

**Priority Rationale**:
- **High**: Essential for core intelligence (math, language, pedagogy, self-reflection)
- **Medium**: Enhances communication and strategy (eloquence, storytelling, delivery)
- **Low**: Specialized capabilities (temporal reasoning, research methods)

### 3.2 Advanced Mathematics (Layer 1 + Layer 3)

**5 PDFs, 1,656 pages** — Math symbols and operations

**Content**:
- Calculus (derivatives, integrals, series, limits)
- Linear algebra (matrices, eigenvalues, vector spaces)
- Differential equations (ODEs, PDEs)
- Complex analysis (contour integration, residues)
- Numerical methods (interpolation, approximation)
- Financial mathematics (option pricing, risk models)

**RPN Programs Extracted** (~200 programs):
```rpn
# Derivative operator
"calc_derivative" =>
    FUNCTION RECALL VAR RECALL
    EPSILON RECALL           # Small h for numerical derivative
    2pick 1index EPSILON ADD EVAL  # f(x + h)
    2pick 1index EVAL        # f(x)
    SUB EPSILON DIV          # [f(x+h) - f(x)] / h

# Definite integral (trapezoidal rule)
"calc_integral" =>
    FUNCTION RECALL A RECALL B RECALL N RECALL
    B A SUB N DIV            # h = (b-a)/n
    # ... (trapezoidal sum logic)

# Black-Scholes option pricing
"fin_black_scholes_call" =>
    S K r T sigma            # Stock, strike, rate, time, vol
    # ... (d1/d2 calculation, CDF lookup)
    S d1 NORM_CDF MUL
    K r T MUL EXP d2 NORM_CDF MUL
    SUB
```

**Symlink References**: All programs reference Layer 1 symbols (∂, ∫, ∑, ∇, etc.) via character IDs, not visual data.

### 3.3 Language, Grammar & Semantics (Layer 2 + Layer 3)

**10 PDFs, 1,805 pages** — Words, grammar, linguistic structures

**Content**:
- Syntax and sentence structure (phrase structure, transformations)
- Morphology (word formation, inflection)
- Semantics (meaning representation, compositional semantics)
- Pragmatics (context, speech acts)
- Portuguese grammar (verb conjugation, diacritics, gender/number agreement)
- Multilingual support (English, Portuguese, Spanish, French)

**RPN Programs Extracted** (~300 programs):
```rpn
# SVO → VSO transformation
"grammar_svo_to_vso" =>
    SENTENCE RECALL
    EXTRACT_SUBJECT EXTRACT_VERB EXTRACT_OBJECT
    3roll CONCAT_SENTENCE    # Verb-Subject-Object

# Portuguese verb conjugation (present tense)
"pt_verb_conjugate_present" =>
    VERB_STEM RECALL PERSON RECALL
    PERSON 1 ==
        { VERB_STEM "o" CONCAT }   # eu amo
    PERSON 2 ==
        { VERB_STEM "as" CONCAT }  # tu amas
    PERSON 3 ==
        { VERB_STEM "a" CONCAT }   # ele/ela ama
    ifelse ifelse

# Diacritic application (Portuguese)
"pt_apply_tilde" =>
    BASE_CHAR RECALL        # 'a' or 'o'
    BASE_CHAR 'a' ==
        { 227 CHAR_REF }    # ã (U+00E3)
    BASE_CHAR 'o' ==
        { 245 CHAR_REF }    # õ (U+00F5)
    ifelse
```

**Word Galaxy Population**: ~15,000 words extracted with semantic definitions, examples, and Layer 1 character sequences.

### 3.4 Pedagogy & Learning (Layer 4)

**5 PDFs, 265 pages** — Teaching strategies, metacognition

**Content**:
- Bloom's taxonomy (knowledge, comprehension, application, analysis, synthesis, evaluation)
- Scaffolding techniques (worked examples, faded guidance)
- Metacognitive strategies (self-monitoring, planning, evaluation)
- Learning theories (constructivism, cognitive load, spaced repetition)

**RPN Programs Extracted** (~80 programs):
```rpn
# Scaffold complex task
"meta_scaffold_task" =>
    TASK RECALL DIFFICULTY_SCORE
    DIFFICULTY 0.7 >
        {
            TASK DECOMPOSE
            SUB_TASKS EACH
                { WORKED_EXAMPLE PROVIDE }
                { GUIDED_PRACTICE }
                { INDEPENDENT_PRACTICE }
        }
    DIFFICULTY 0.3 0.7 BETWEEN
        {
            HINTS_PROVIDE
            SELF_CHECK_QUESTIONS
        }
    # Low difficulty: direct practice
        { INDEPENDENT_PRACTICE }
    ifelse ifelse

# Assess prerequisite knowledge
"meta_check_prerequisites" =>
    CONCEPT RECALL
    CONCEPT_DEPENDENCIES RETRIEVE
    EACH_DEPENDENCY
        { LEARNER_KNOWLEDGE_TEST }
    ALL_PASSED
        { PROCEED_WITH_CONCEPT }
        { TEACH_MISSING_PREREQUISITES }
    ifelse
```

**Integration**: These meta-rules guide Layer 3 grammar rule application — when to scaffold, when to provide worked examples.

### 3.5 Eloquence, Rhetoric & Persuasion (Layer 4)

**8 PDFs, 350 pages** — Rhetorical strategies, persuasion mastery

**Content**:
- Classical rhetoric (ethos, pathos, logos)
- Argument structures (syllogism, enthymeme, toulmin model)
- Persuasion techniques (scarcity, authority, social proof, reciprocity)
- Stylistic devices (metaphor, analogy, antithesis, chiasmus)
- Audience analysis and adaptation

**RPN Programs Extracted** (~100 programs):
```rpn
# Build ethos (credibility)
"meta_build_ethos" =>
    SPEAKER_CREDENTIALS RECALL
    SPEAKER_EXPERIENCE RECALL
    "As someone who has" NARRATIVE_OPENING
    EXPERIENCE DESCRIBE
    "I understand" TRANSITION
    AUDIENCE_CONCERN ACKNOWLEDGE

# Create urgency via consequences
"meta_urgency_consequences" =>
    PROBLEM RECALL
    "If we don't act now" WARNING_SIGNAL
    NEGATIVE_CONSEQUENCES DESCRIBE
    TIME_CONSTRAINT EMPHASIZE
    "We must" CALL_TO_ACTION

# Syllogistic reasoning
"meta_syllogism" =>
    MAJOR_PREMISE RECALL  # All A are B
    MINOR_PREMISE RECALL  # C is A
    "Therefore" LOGICAL_CONNECTOR
    CONCLUSION DEDUCE     # C is B
```

### 3.6 Self-Reflection (Layer 4)

**7 PDFs, 164 pages** — Meta-cognition, performance assessment

**Content**:
- Self-efficacy and growth mindset
- Reflective practice (Schön, Kolb)
- Error analysis and pattern detection
- Performance attribution (internal/external, stable/unstable)
- Self-regulation strategies

**RPN Programs Extracted** (~70 programs):
```rpn
# Assess task performance (sleeptime consolidation integration)
"meta_assess_task_performance" =>
    TASK_RESULT RECALL EXPECTED_RESULT RECALL
    ATTEMPTED_RULES RECALL
    2pick 2pick ==           # Did we succeed?
    {
        # Success branch
        1index EXTRACT_SUCCESSFUL_RULES
        EACH_RULE
            { RULE_ID PRIORITY_INCREMENT }
        MARK_FOR_CONSOLIDATION_HIGH_PRIORITY
    } {
        # Failure branch
        2pick EXTRACT_ERROR_PATTERNS
        2pick IDENTIFY_WRONG_ASSUMPTIONS
        EACH_ERROR
            { ERROR_PATTERN ANALYZE }
            { CORRECTIVE_RULE SUGGEST }
        MARK_FOR_CONSOLIDATION_ERROR_CORRECTION
    } ifelse

# Identify error patterns
"meta_identify_error_pattern" =>
    FAILED_ATTEMPTS RECALL
    COMMON_FEATURES EXTRACT
    PATTERN_FREQUENCY COUNT
    PATTERN_FREQUENCY 3 >
        { SYSTEMATIC_ERROR_DETECTED }
        { RANDOM_ERROR_LIKELY }
    ifelse

# Consolidation priority scoring
"meta_consolidation_priority" =>
    RULE_ID RECALL
    RULE_SUCCESS_RATE RETRIEVE
    RULE_USAGE_FREQUENCY RETRIEVE
    RULE_NOVELTY_SCORE RETRIEVE
    # Priority = 0.5*success + 0.3*frequency + 0.2*novelty
    SUCCESS 0.5 MUL
    FREQUENCY 0.3 MUL ADD
    NOVELTY 0.2 MUL ADD
```

**Critical Integration**: These programs provide the foundational knowledge for `sleeptime_consolidator.py` to:
- Identify which Layer 3 rules succeeded/failed
- Score which rules to prioritize for rehearsal
- Detect error patterns for correction
- Guide weight updates toward successful strategies

### 3.7 Story Telling (Layer 4)

**7 PDFs, 645 pages** — Narrative structure, story mastery

**Content**:
- Three-act structure (setup, confrontation, resolution)
- Hero's journey (Campbell)
- Character development and arcs
- Plot devices (foreshadowing, flashback, twist)
- Narrative pacing and tension
- Point of view and voice

**RPN Programs Extracted** (~120 programs):
```rpn
# Three-act structure
"meta_story_three_act" =>
    CONTEXT RECALL PROBLEM RECALL SOLUTION RECALL
    # Act 1: Setup
    CONTEXT DESCRIBE
    "Once upon a time" NARRATIVE_OPENING
    SETTING ESTABLISH
    CHARACTERS_INTRODUCE
    # Act 2: Confrontation
    PROBLEM DESCRIBE
    "But then" NARRATIVE_TRANSITION
    CONFLICT_ESCALATE
    STAKES_RAISE
    # Act 3: Resolution
    SOLUTION DESCRIBE
    "And so" NARRATIVE_CONCLUSION
    RESOLUTION_PROVIDE
    LESSON_EXTRACT
    CONCAT_NARRATIVE

# Hero's journey (simplified)
"meta_story_hero_journey" =>
    HERO RECALL QUEST RECALL
    "ORDINARY_WORLD" HERO INTRODUCE
    "CALL_TO_ADVENTURE" QUEST PRESENT
    "REFUSAL_OF_CALL" HERO_DOUBT SHOW
    "MEETING_MENTOR" GUIDANCE_PROVIDE
    "CROSSING_THRESHOLD" JOURNEY_BEGIN
    "TRIALS_ALLIES_ENEMIES" CHALLENGES_FACE
    "ORDEAL" CENTRAL_CRISIS
    "REWARD" TREASURE_SEIZE
    "RETURN" HERO_TRANSFORMS
    CONCAT_NARRATIVE

# Foreshadowing
"meta_story_foreshadow" =>
    FUTURE_EVENT RECALL
    SUBTLE_HINT RECALL
    EARLY_SCENE RECALL
    HINT EARLY_SCENE EMBED
    HINT_SUBTLETY 0.7 >
        { IMPLICIT_FORESHADOW }
        { EXPLICIT_FORESHADOW }
    ifelse
```

**Application**: Enables K3D to structure explanations, tutorials, and responses as narratives — making complex information more memorable via story structure.

### 3.8 Acting/Delivery (Layer 4)

**3 PDFs, 65 pages** — Delivery mastery, performance techniques

**Content**:
- Vocal variety (pitch, volume, rate, pause)
- Emphasis and pacing
- Dramatic pause and timing
- Gesture and body language (when applicable to voice synthesis)
- Emotional authenticity

**RPN Programs Extracted** (~40 programs):
```rpn
# Apply emphasis to key phrase
"meta_delivery_emphasize" =>
    PHRASE RECALL KEY_WORDS RECALL
    PHRASE TOKENIZE
    EACH_WORD
        WORD KEY_WORDS CONTAINS
            { WORD VOLUME_INCREASE PITCH_RAISE RATE_SLOW }
            { WORD NORMAL_DELIVERY }
        ifelse
    CONCAT_WITH_PROSODY

# Dramatic pause
"meta_delivery_pause" =>
    BEFORE_PHRASE AFTER_PHRASE
    PAUSE_TYPE RECALL         # anticipation, emphasis, transition
    PAUSE_TYPE "anticipation" ==
        { 0.8 PAUSE_DURATION }  # 800ms
    PAUSE_TYPE "emphasis" ==
        { 0.5 PAUSE_DURATION }  # 500ms
    PAUSE_TYPE "transition" ==
        { 0.3 PAUSE_DURATION }  # 300ms
    ifelse ifelse
    BEFORE_PHRASE SPEAK
    PAUSE_DURATION SILENCE
    AFTER_PHRASE SPEAK

# Pacing variation
"meta_delivery_pace" =>
    CONTENT RECALL COMPLEXITY RECALL
    COMPLEXITY HIGH ==
        { SLOW_RATE 0.8 RATE_MULTIPLIER }  # 80% of normal
    COMPLEXITY MEDIUM ==
        { NORMAL_RATE 1.0 RATE_MULTIPLIER }
    COMPLEXITY LOW ==
        { FAST_RATE 1.2 RATE_MULTIPLIER }   # 120% of normal
    ifelse ifelse
    CONTENT RATE_MULTIPLIER APPLY_RATE_TO_SPEECH
```

**Integration**: These delivery meta-rules can guide future voice synthesis (Phase 4+) and text formatting for readable explanations.

### 3.9 Context & Temporal Understanding (Layer 2 + Layer 3)

**26 PDFs, 853 pages** — Contextual reasoning and time understanding

**Content (Context)**: 13 PDFs, 693 pages
- Pragmatic inference (implicature, presupposition)
- Frame semantics (Fillmore)
- Discourse coherence (topic, focus, given/new)
- Anaphora resolution (pronoun reference)

**Content (Temporal)**: 13 PDFs, 160 pages
- Temporal logic (before, after, during, overlaps)
- Tense and aspect (past, present, future, perfective, imperfective)
- Event ordering and causality
- Duration and frequency

**RPN Programs Extracted** (~100 programs):
```rpn
# Resolve pronoun reference (anaphora)
"context_resolve_pronoun" =>
    PRONOUN RECALL DISCOURSE_HISTORY RECALL
    DISCOURSE_HISTORY REVERSE_SCAN
    FIRST_NOUN_MATCHING_GENDER_NUMBER
    REPLACE_PRONOUN_WITH_REFERENT

# Temporal ordering
"time_order_events" =>
    EVENT_A EVENT_B
    EVENT_A_TIME EVENT_B_TIME
    EVENT_A_TIME EVENT_B_TIME <
        { EVENT_A "before" EVENT_B DESCRIBE }
    EVENT_A_TIME EVENT_B_TIME ==
        { EVENT_A "simultaneous" EVENT_B DESCRIBE }
    EVENT_A_TIME EVENT_B_TIME >
        { EVENT_A "after" EVENT_B DESCRIBE }
    ifelse ifelse

# Duration calculation
"time_duration" =>
    START_TIME END_TIME
    END_TIME START_TIME SUB
    DURATION_UNIT CONVERT    # Convert to seconds/minutes/hours
```

---

## 4. Implementation Plan

### 4.1 Six-Week Roadmap

**Phase 1: Train Layer 1 Symbols (Week 1)**
- Run `scripts/train_math_symbols_batch.py --priority=all`
- Train 152 math symbols + 12 Portuguese diacritics
- Store in Math Galaxy (`/K3D/Knowledge3D.local/procedural_galaxy/math/symbols/`)
- Validation: All 164 symbols achieve >90% recognition accuracy

**Phase 2: Extend Grammar Galaxy with Symlinks (Week 2)**
- Enhance `grammar_galaxy.py` with `symbol_refs`, `word_refs` fields
- Create `k3dgen` symlink generator (or extend existing k3dgen)
- Ingest Language/Grammar PDFs (10 PDFs, 1,805 pages)
- Extract ~300 grammar rules with symlink references
- Validation: 300+ grammar rules successfully reference Layer 1 symbols without duplication

**Phase 3: Math/Context/Time Ingestion (Week 3-4)**
- Ingest Advanced Mathematics (5 PDFs, 1,656 pages) → ~200 Layer 3 programs
- Ingest Context PDFs (13 PDFs, 693 pages) → ~50 Layer 2/3 programs
- Ingest Temporal PDFs (13 PDFs, 160 pages) → ~50 Layer 3 programs
- Validation: 300+ new RPN programs successfully execute in `ModularRPNEngine`

**Phase 4: Meta-Rules Ingestion (Week 5)**
- Create `eloquence_galaxy.py` for Layer 4 storage
- Ingest Pedagogy (5 PDFs, 265 pages) → ~80 meta-rules
- Ingest Eloquence (8 PDFs, 350 pages) → ~100 meta-rules
- Ingest Self-Reflection (7 PDFs, 164 pages) → ~70 meta-rules
- Ingest Story Telling (7 PDFs, 645 pages) → ~120 meta-rules
- Ingest Acting/Delivery (3 PDFs, 65 pages) → ~40 meta-rules
- Total: ~410 Layer 4 meta-rules
- Validation: All meta-rules successfully condition Layer 3 rule application

**Phase 5: Cross-Domain Discovery (Week 6, Part 1)**
- Implement `discovery_layer.py` to analyze shared symbol references
- Generate 300+ cross-domain connections (e.g., ∑ connects calculus, statistics, finance)
- Validation: Discovery layer identifies connections across at least 5 distinct domains

**Phase 6: Integration Testing (Week 6, Part 2)**
- Test symlink integrity (all references resolve correctly)
- Test RPN execution (all 1,500+ programs execute without errors)
- Test sleeptime consolidation (self-reflection meta-rules guide rule prioritization)
- Test ARC-AGI performance improvement (target: >5% improvement over baseline)
- Validation: All integration tests pass; ARC-AGI improvement documented

### 4.2 Success Criteria

**Layer 1 (Form)**:
- ✓ 164 symbols trained (152 math + 12 Portuguese) with >90% accuracy
- ✓ All symbols stored in Math Galaxy with 69-80:1 compression

**Layer 2 (Meaning)**:
- ✓ 15,000+ words extracted from PDFs
- ✓ All words reference Layer 1 character IDs (no visual duplication)

**Layer 3 (Rules)**:
- ✓ 1,000+ RPN programs ingested from PDFs
- ✓ All programs reference Layer 1 symbols via symlinks
- ✓ 666x compression achieved on repeated symbols
- ✓ All programs execute successfully in `ModularRPNEngine`

**Layer 4 (Meta-Rules)**:
- ✓ 500+ meta-rules ingested (pedagogy, eloquence, self-reflection, storytelling, delivery)
- ✓ Meta-rules successfully condition Layer 3 rule application
- ✓ Sleeptime consolidation uses self-reflection meta-rules for priority scoring

**Cross-Domain Discovery**:
- ✓ Discovery layer generates 300+ connections across domains
- ✓ Connections verified manually (sample of 30)

**ARC-AGI Validation**:
- ✓ >5% improvement over baseline on ARC-AGI evaluation dataset
- ✓ Improvement attributed to foundational knowledge (via ablation study)

---

## 5. Integration with Existing Systems

### 5.1 Math Symbol Galaxy

**Existing**: `knowledge3d/cranium/math_symbols_registry.py` (~152 symbols registered)

**Integration**:
- Layer 1 symbols trained via `train_math_symbols_batch.py`
- Stored in Math Galaxy (`math_galaxy.py`) using ProceduralGalaxy infrastructure
- Layer 3 grammar rules reference symbols via `symbol_refs: List[int]`

**Example**:
```python
# Layer 1: Symbol trained and stored
math_galaxy.store_symbol('∑', embedding)

# Layer 3: Grammar rule references symbol
GrammarRule(
    rule_id="riemann_sum",
    symbol_refs=[8721],  # U+2211 = ∑
    rpn_program="X_I RECALL N RECALL SUMMATION"
)
```

### 5.2 Grammar Galaxy

**Existing**: `knowledge3d/training/arc_agi/grammar_galaxy.py` (11 baseline rules)

**Enhancement**:
- Add `symbol_refs: List[int]` field (Layer 1 symlinks)
- Add `word_refs: List[str]` field (Layer 2 symlinks)
- Expand from 11 → 1,000+ rules via PDF ingestion

**Migration Strategy**:
```python
# Old format (11 rules)
GrammarRule(rule_id="en_simple_sentence", pattern="SVO", rpn_program="...")

# New format (1,000+ rules with symlinks)
GrammarRule(
    rule_id="en_simple_sentence",
    pattern="SVO",
    rpn_program="...",
    symbol_refs=[],       # No math symbols in this rule
    word_refs=["subject", "verb", "object"]  # Layer 2 references
)
```

### 5.3 Sleeptime Consolidation

**Existing**: `knowledge3d/cranium/sleeptime_consolidator.py` (experience replay infrastructure)

**Enhancement**:
- Layer 4 self-reflection meta-rules provide foundational knowledge on:
  - Which rules succeeded/failed (`meta_assess_task_performance`)
  - Which rules to prioritize for rehearsal (`meta_consolidation_priority`)
  - Error pattern identification (`meta_identify_error_pattern`)

**Integration**:
```python
# sleeptime_consolidator.py uses Layer 4 meta-rules
class SleeptimeConsolidator:
    def prioritize_experiences(self, experiences: List[Experience]) -> List[Experience]:
        # Use meta_consolidation_priority to score each experience
        for exp in experiences:
            exp.priority = self.meta_rules.evaluate("meta_consolidation_priority", exp)
        return sorted(experiences, key=lambda e: e.priority, reverse=True)

    def identify_error_patterns(self, failed_experiences: List[Experience]) -> List[str]:
        # Use meta_identify_error_pattern to detect systematic errors
        return self.meta_rules.evaluate("meta_identify_error_pattern", failed_experiences)
```

### 5.4 Discovery Layer

**New**: `discovery_layer.py` (to be created in Phase 5)

**Purpose**: Analyze shared symbol references across domains to discover emergent connections

**Implementation**:
```python
class DiscoveryLayer:
    def __init__(self, grammar_galaxy: GrammarGalaxy):
        self.grammar_galaxy = grammar_galaxy
        self.symbol_to_rules = self._build_reverse_index()

    def _build_reverse_index(self) -> Dict[int, List[str]]:
        """Map each symbol ID → list of rule IDs that use it."""
        symbol_to_rules = defaultdict(list)
        for rule in self.grammar_galaxy.rules:
            for symbol_id in rule.symbol_refs:
                symbol_to_rules[symbol_id].append(rule.rule_id)
        return symbol_to_rules

    def discover_cross_domain_connections(self) -> List[Connection]:
        """Find symbols shared across multiple domains."""
        connections = []
        for symbol_id, rule_ids in self.symbol_to_rules.items():
            domains = set(self.grammar_galaxy.get_rule(r).domain for r in rule_ids)
            if len(domains) >= 2:  # Symbol used in 2+ domains
                connections.append(Connection(
                    symbol_id=symbol_id,
                    domains=list(domains),
                    rule_ids=rule_ids,
                    strength=len(rule_ids) / len(domains)  # Frequency per domain
                ))
        return sorted(connections, key=lambda c: c.strength, reverse=True)
```

**Output Example**:
```
Symbol ∑ (U+2211) connects 5 domains:
  - math_calculus (30 rules)
  - math_statistics (25 rules)
  - math_finance (15 rules)
  - math_physics (10 rules)
  - math_probability (8 rules)
  Strength: 17.6 (avg rules per domain)
  Insight: Iterative accumulation is fundamental to quantitative reasoning
```

---

## 6. Dual Client Reality

### 6.1 Humans See Glyphs, AI Executes RPN

**From CLAUDE.md lines 81-93**:

> K3D serves TWO clients with the SAME data — Humans AND AI.
>
> **Character Galaxy** → Glyphs (Bézier → segments) + language/pronunciation metadata
> **Word Level** → Character sequences (references, not duplicates)
> **Grammar Galaxy** → Transformation rules (RPN) + context metadata

### 6.2 Example: Human vs AI Perspective

**Human sees** (Layer 1 visual):
```
∂f/∂x = lim[h→0] (f(x+h) - f(x)) / h
```

**AI executes** (Layer 3 RPN):
```rpn
FUNCTION RECALL VAR RECALL EPSILON RECALL
2pick 1index EPSILON ADD EVAL
2pick 1index EVAL SUB
EPSILON DIV
```

**Both representations reference the same Layer 1 symbol** (∂, U+2202) via symlink:
- Human: Procedural Bézier curve rendered as visual glyph
- AI: Character ID 8706 in `symbol_refs` field

### 6.3 Save Information Principle

**Wrong Approach** (duplication):
```python
# Store visual representation in each rule
rule = {
    "pattern": "∂f/∂x",
    "visual_data": b'<Bézier curve data for ∂>',  # 5 KB
    "rpn_program": "FUNCTION RECALL VAR RECALL SYMBOL_DIFF"
}
```

**Correct Approach** (symlink):
```python
# Reference Layer 1 canonical symbol
rule = {
    "pattern": "∂f/∂x",
    "symbol_refs": [8706],  # U+2202 = ∂ (4 bytes)
    "rpn_program": "FUNCTION RECALL VAR RECALL SYMBOL_DIFF"
}
```

**Benefit**: 1,000 rules referencing ∂ → 4 KB total (vs 5 MB duplicated)

---

## 7. Future Extensions

### 7.1 Multi-Modal Extensions

**Vision**: Extend foundational knowledge to multi-modal domains (images, audio, video)

**Implementation**:
- Layer 1: Visual primitives (shapes, colors, textures)
- Layer 2: Object concepts (car, tree, person)
- Layer 3: Visual transformations (rotate, scale, blend)
- Layer 4: Visual rhetoric (composition, balance, emphasis)

**Example**:
```rpn
# Visual emphasis via color contrast
"meta_visual_emphasize" =>
    ELEMENT RECALL BACKGROUND_COLOR RECALL
    ELEMENT_COLOR BACKGROUND_COLOR COMPLEMENT_COLOR
    ELEMENT_COLOR SATURATION_INCREASE
    RENDER_WITH_CONTRAST
```

### 7.2 Domain-Specific Knowledge

**Vision**: Extend to specialized domains (physics, chemistry, biology, law, medicine)

**Implementation**:
- Physics: Classical mechanics, E&M, thermodynamics, quantum (already in progress)
- Chemistry: Molecular structures, reactions, stoichiometry
- Biology: Cellular processes, genetics, ecosystems
- Law: Legal reasoning, case precedent, statutory interpretation
- Medicine: Diagnosis, treatment protocols, pharmacology

**Note**: All follow the same 4-layer pattern (Form → Meaning → Rules → Meta-Rules) with symlink references to canonical symbols.

### 7.3 Personalized Knowledge

**Vision**: Allow users to add personal foundational knowledge

**Implementation**:
- User-defined symbols (custom notation)
- User-defined rules (personal workflows)
- User-defined meta-rules (personal strategies)
- Private Galaxy instances (user-specific storage)

**Example**:
```python
# User adds custom symbol for "convolution"
user_galaxy.add_symbol('⊛', embedding)  # Custom operator

# User defines custom rule
user_galaxy.add_rule(GrammarRule(
    rule_id="my_convolution_rule",
    symbol_refs=[9947],  # ⊛ (custom)
    rpn_program="SIGNAL KERNEL CONVOLUTION_FFT"
))
```

---

## 8. References

### 8.1 Core Implementation Files

- `knowledge3d/cranium/math_symbols_registry.py` — 152 registered math symbols
- `knowledge3d/cranium/math_galaxy.py` — Math Symbol Galaxy storage/retrieval
- `knowledge3d/training/arc_agi/grammar_galaxy.py` — Grammar rules (needs enhancement)
- `knowledge3d/cranium/sleeptime_consolidator.py` — Experience replay infrastructure
- `scripts/train_math_symbols_batch.py` — Batch training script for Layer 1 symbols
- `scripts/train_atomic_character.py` — Character training infrastructure

### 8.2 Architecture Documents

- `CLAUDE.md` — Architecture principles (Dual Client Reality, Save Information Principle)
- `BRIEFING.md` — Central project overview
- `docs/vocabulary/MATH_CORE_SPECIFICATION.md` — 3-tier math core details
- `docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md` — Human/AI dual client contract
- `docs/vocabulary/SLEEPTIME_PROTOCOL_SPECIFICATION.md` — Consolidation protocol

### 8.3 Implementation Plans

- `TEMP/KNOWLEDGE_INGESTION_PLAN_V5_CODEX_READY.md` — Detailed implementation plan for Codex
- `docs/ROADMAP.md` — Current phase and milestones

---

## 9. Conclusion

The Foundational Knowledge Architecture provides K3D with always-loaded base knowledge spanning four hierarchical layers (Form → Meaning → Rules → Meta-Rules). By leveraging the **symlink pattern** to reference existing Layer 1 symbols instead of duplicating, we achieve:

- **666x compression** on repeated symbols across grammar rules
- **69-80:1 compression** via ProceduralGalaxy infrastructure
- **Cross-domain discovery** via shared symbol references (300+ connections)
- **Sleeptime consolidation** enhancement via self-reflection meta-rules

This architecture embodies the **Save Information Principle**: knowledge lives in Galaxy/House storage (3D RAM/procedural HD), weights store only logic. It enables **Dual Client Reality** where humans see visual glyphs and AI executes RPN — both perspectives reference the same canonical data.

**Total Corpus**: 74 PDFs (5,988 pages) → 164 symbols, 15,000 words, 1,000 rules, 500 meta-rules — ready for 6-week implementation.

---

**Version History**:
- 1.0 (December 2025): Initial specification documenting 4-layer architecture, 74 PDFs, symlink pattern, implementation plan.
