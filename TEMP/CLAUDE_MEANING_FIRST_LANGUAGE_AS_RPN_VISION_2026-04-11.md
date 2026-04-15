# CLAUDE → CODEX — Meaning-First, Language-as-RPN-Math, Layer-0 Instancing & Canonical Qdrant DB — 2026-04-11

## Status

This is an **architectural vision** spec. It supersedes [TEMP/CLAUDE_PHASE7_PROCEDURALIZATION_QUALITY_AND_MULTILINGUAL_SYMLINKS_2026-04-11.md](CLAUDE_PHASE7_PROCEDURALIZATION_QUALITY_AND_MULTILINGUAL_SYMLINKS_2026-04-11.md). Phase 7 still applies for dangling pointers + bidirectional symlinks; the parts about "embeddings as the embedding-quality goal" and "STORE/RECALL meaning_rpn templates" are updated below to fit this larger vision. Read this first, then revisit Phase 7 with the corrections in §10.

Grounded in:
- [docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md](../docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md) — meaning IS the center; surface forms are symlinks
- [docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md](../docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md) §1.6 — Form→Meaning, dual-client procedural foundation
- [docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md](../docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md) §1 — 4-layer architecture (Form→Meaning→Rules→Meta-Rules)
- [docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md](../docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md) §1 — programs before opcodes
- [docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md](../docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md) — Cranium + Galaxy + House
- Kimi swarm synthesis: [TEMP/KIMI_MEANING_FIRST_LANGUAGE_AS_RPN_2026-04-11.md](KIMI_MEANING_FIRST_LANGUAGE_AS_RPN_2026-04-11.md)

---

## 0. The principle (one sentence)

**Knowledge is procedural RPN that BOTH humans and TRM can read; TRM stores only problem-solving strategies; language is composed at runtime by grammar-driven RPN math from meaning stars; the same Layer-0 drawing primitives instance everything visible in the system; LOD + frustum culling apply to knowledge as much as to geometry; canonical IDs and opcodes live in their own Qdrant DB so the proceduralizer never invents duplicates.**

Daniel's words this session, paraphrased and pinned:

- "I want a way to represent knowledge based on meaning, but languages also need to be supported."
- "When a meaning is single language only, like 'saudades', this is the meaning that's recorded."
- "We MUST respect the dual client contract — embeddings are not the best place to store [knowledge]."
- "TRM does not store knowledge, it only stores RPN strategies, problem solving logic — one single model to all modalities, no 'arc' namings, generic and generalizable."
- "Implement language as math operation in a completely different way [from] AI today — more akin to how humans form words and texts, following meaning with grammar rules."
- "All the drawing inside the system, from basic forms to characters to interfaces, use the same drawing premisses of layer 0 — load once, instance many, plus LOD plus FOV — applied to knowledge in the Knowledgeverse and in the house as well."
- "A human gets a book in a game (K3D is the game) and reads it on its virtual hands."
- "We should and could do another qdrant database with the canonical things and RPN instructions, so the proceduralizer can leverage it."

---

## 1. The Knowledge / TRM separation (this is the load-bearing rule)

### What TRM stores (only this)

- **RPN problem-solving strategies**: how to navigate, when to pull a meaning star into Galaxy, when to halt, how to compose a grammar rule with a meaning, when to back-track, when to branch the swarm.
- **One generic model for all modalities**. No `arc_*`, no `gsm8k_*`, no `mmlu_*` weights. The strategies are domain-agnostic. A successful grid-transform strategy must compose the same way for math word problems, multi-hop reasoning, and physics simulation.
- **No knowledge**. TRM does not memorize "Paris is the capital of France." It memorizes "when the query is a relation lookup, navigate the meaning star for the entity, follow the relation symlink, return the target."

### What the Galaxy / House stores (everything else)

- **Meaning-centric stars** with `meaning_rpn` programs (procedural, dual-readable).
- **Surface-form word stars** in per-language Word Galaxies (symlinks, not duplicates).
- **Character Galaxy stars** (one star per Unicode codepoint, language-agnostic for shared scripts).
- **Grammar Galaxy stars** carrying RPN transformation programs per language and per construction.
- **Drawing Galaxy stars** carrying Layer-0 primitive RPN that instances ALL visible geometry.
- **Reality Galaxy stars** carrying physics/chem/bio behavior RPN.
- **Books, journals, papers in the House** as 3D objects whose contents are *closed Galaxies* until opened.

### Why this separation matters

If TRM stored knowledge in weights, K3D becomes another LLM with the opacity problem the dual-client contract was designed to defeat. Because TRM stores only strategies, the **Galaxy is the auditable substrate** — humans can read the same `meaning_rpn` programs the TRM executes. There is no hidden state.

> **Concrete consequence:** stop measuring "embedding quality" as the success metric for the proceduralizer. Embeddings are navigation tools — not knowledge storage. Quality is measured by the **`meaning_rpn` programs being human-readable AND machine-executable**, and by the **symlinks resolving in both directions**. Embeddings just have to be good enough to find the right star.

---

## 2. Meaning-first with the saudades rule

### 2.1 The rule

**One concept = one meaning star.** All languages that have a word for that concept symlink their per-language word stars to the same meaning star. Cross-language navigation is via the symlinks, not via translation tables.

**Exception (the saudades rule):** when a meaning exists in only one language with no translatable equivalent in others, the meaning star IS recorded with that single language as its canonical surface. Other languages have NO `word_ref` for that meaning — they must reach it via an explanatory periphrastic construction (also generated at runtime by grammar rules), or use the loanword.

### 2.2 The schema for saudades-class meanings

```
MeaningCentricStar {
    star_id:        "concept_saudades_pt"   // canonical: language tag in id only when single-language
    meaning_class:  "concept"

    // Meaning is procedural and language-agnostic in structure even when only one
    // language has a single-word surface form
    meaning_rpn:    "EMOTION longing TQUANT  TEMPORAL past TQUANT  AGENT self TQUANT
                     OBJECT absent_other STORE_target
                     IRRECOVERABLE TQUANT STORE_modality
                     INTENSITY high STORE_amplitude"

    domain:         "Library/Linguistics/Emotion/Untranslatable"
    taxonomy_refs:  [ "concept_emotion", "concept_longing", "concept_nostalgia" ]

    surface_forms: {
        "pt": { word_ref: "word_pt_saudades", char_refs: ["char_s","char_a","char_u","char_d","char_a","char_d","char_e","char_s"] }
        // No "en" key. No "ja" key. The absence is the encoded fact.
    }

    untranslatable_languages: [ "en", "ja", "es", "fr", ... ]   // NEW field — see §2.3

    // When asked for English surface, the GRAMMAR pipeline emits a periphrastic
    // construction by composing other meaning stars (longing + irrecoverable + past + intensity)
    // — see §3.4.
}
```

### 2.3 The `untranslatable_languages` field

A new optional field on meaning stars listing languages where the concept has **no single-word equivalent**. Two purposes:

1. **Honest representation**: the absence of a `word_ref` for English does not mean "we forgot to add English" — it means "no English word exists". Humans inspecting the star see this immediately.
2. **Grammar dispatch hint**: when the TRM composes a sentence in English about a saudades-class meaning, the grammar pipeline knows to expand into a periphrastic construction (or use the loanword) instead of failing.

### 2.4 What this is NOT

- This is NOT a translation table. There is no `english_equivalent: "longing"` field. The english surface, when needed, is **synthesized at runtime** by grammar rules composing other meaning stars. See §3.
- This is NOT a "primary language" architecture. Portuguese is not "primary" for saudades — it is the only language with a single-word form. The meaning is what's primary.

---

## 3. Language as RPN math (TRM "speaks" by composition)

### 3.1 The breakaway from how today's AI does language

| Today's LLM | K3D language-as-RPN-math |
|-------------|---------------------------|
| Tokens are statistical units; meaning is implicit in attention | Tokens are surface-form symlinks; meaning lives in `meaning_rpn` |
| Translation = parameter lookup over English-pivoted weights | Translation = grammar rules navigate from one word galaxy to another via the meaning |
| Untranslatables hallucinated or approximated | Untranslatables encoded honestly via `untranslatable_languages`; periphrastic synthesis on demand |
| Generation = next-token prediction | Generation = grammar program composes morphemes from meaning stars |
| Add a language = retrain | Add a language = add a Word Galaxy + grammar rules; meaning stars unchanged |
| Black-box | Every step is an inspectable RPN stack trace |

This is closer to how humans speak: meaning first ("I want to convey nostalgia for an irrecoverable past"), grammar selects the construction ("I miss…", "tenho saudades de…"), morphology adjusts agreement, surface emits.

### 3.2 The sentence-synthesis pipeline (pure RPN, no Python string formatting)

```
INPUT: query_embedding → cosine pick → meaning_star  +  target_language
                                          │
                                          ▼
              ┌─────────────────────────────────────────────┐
              │ STAGE 1: MEANING DECOMPOSE                   │
              │ Execute meaning_rpn → produce a stack of     │
              │ semantic primitives (entities, relations,    │
              │ properties, modalities)                      │
              └────────────────────┬─────────────────────────┘
                                   ▼
              ┌─────────────────────────────────────────────┐
              │ STAGE 2: GRAMMAR SELECT                      │
              │ Cosine over Grammar Galaxy {target_language} │
              │ → pick construction template (RPN program)   │
              │ Saudades-class? → choose periphrastic temp.  │
              └────────────────────┬─────────────────────────┘
                                   ▼
              ┌─────────────────────────────────────────────┐
              │ STAGE 3: WORD RESOLVE                        │
              │ For each semantic slot in the template,      │
              │ navigate meaning → surface_forms[lang].word_ │
              │ ref → word star → component_refs (chars)     │
              └────────────────────┬─────────────────────────┘
                                   ▼
              ┌─────────────────────────────────────────────┐
              │ STAGE 4: MORPHOLOGY APPLY                    │
              │ Grammar program emits agreement, conjugation │
              │ via RPN ops on word star fields              │
              └────────────────────┬─────────────────────────┘
                                   ▼
              ┌─────────────────────────────────────────────┐
              │ STAGE 5: SURFACE EMIT                        │
              │ Walk char_refs in order, write codepoints to │
              │ ActionBuffer.tablet_data → Tablet renders    │
              └────────────────────┬─────────────────────────┘
                                   ▼
                       OUTPUT: utterance bytes
```

**Every stage is RPN math on the same RPN core that runs `2+3=5`.** No Python `f"{noun} {verb} {object}"` anywhere. No translation table. The Tablet at the end is the dual-client surface: humans read the bytes, the AI knows it produced them by RPN.

### 3.3 Opcode surface — programs before opcodes

Per [RPN_DOMAIN_OPCODE_REGISTRY.md](../docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md) §1, we **do not introduce new opcodes** for language. We compose grammar from existing opcodes:

- `STORE / RECALL` for semantic-slot filling
- `TQUANT / TCOMP / TADD / TMUL` for morphology agreement (gender/number/case as ternary)
- `OP_BRANCH / OP_LOOP` for construction selection and morpheme iteration
- `GALAXY_LOOKUP` for symlink resolution
- `OP_VEC_BLEND / OP_DOT_PRODUCT` for word-star cosine within a language galaxy
- `OP_CMP / OP_PICK / OP_SWAP / OP_DUP / OP_ROT` for stack manipulation in templates

A grammar template for English SVO is then literally an RPN program stored in a Grammar Galaxy star — not a Python class, not a config file. Codex implements a small set of construction templates per language; the proceduralizer expands them by symlink composition.

If usage data later shows a particular grammar primitive (e.g., a morpheme-concatenation step) is hot enough to deserve a fused opcode, it goes through the Stage 0 → Stage 3 promotion pipeline in [RPN_DOMAIN_OPCODE_REGISTRY.md](../docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md) §6. **Not before.**

### 3.4 Saudades through the pipeline

**Query:** "what does saudades mean" (in English, target_language=en)

```
STAGE 1: meaning_rpn for concept_saudades_pt executes →
         stack: [longing, past, irrecoverable, self, absent_other, intensity_high]

STAGE 2: target_lang=en. Grammar Galaxy lookup for concept whose
         untranslatable_languages contains "en" → SELECT periphrastic_explanation_en
         template (an RPN program that walks the semantic-primitive stack and
         emits an explanatory English construction).

STAGE 3: For each semantic primitive on the stack, navigate to its English
         surface form: longing → word_en_longing, past → word_en_past, etc.
         These DO have English surfaces (they are not saudades-class).

STAGE 4: Morphology applies (no agreement needed for English nouns; tense
         marking on the explanatory verb).

STAGE 5: Surface emit. Output: "a deep longing for something irrecoverably
         lost, that you miss with intensity"

         Plus, because untranslatable_languages contained "en", the grammar
         template ALSO appends: " — Portuguese: 'saudades'"
```

**Same query, target_language=pt:**

```
STAGE 2 sees target_lang=pt is NOT in untranslatable_languages → SELECT
         direct_lookup_pt template.

STAGE 3 navigates surface_forms.pt.word_ref → word_pt_saudades.

STAGE 5 emits: "saudades"
```

**Same star, no translation table, no special-case Python.** Both outputs come from the same meaning star via different grammar paths.

---

## 4. Layer 0 instances everything (one drawing surface, all visuals)

### 4.1 The principle

Per [FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md](../docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md) §1.2, Layer 1 (FORM) is the Character Galaxy and Math Symbol Galaxy storing glyphs as procedural RPN. Per [DUAL_CLIENT_CONTRACT_SPECIFICATION.md](../docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md) §1.6.2, the Drawing Galaxy carries `LINE`, `CIRCLE`, `RECT` as procedural primitives. **These are the same primitives that draw everything else.**

The system has **one set of Layer-0 drawing programs**. They are loaded once into VRAM as Drawing Galaxy stars. Every visible thing in K3D is built by **GPU instancing** of those Layer-0 stars with per-instance transforms (position, rotation, scale, color, material).

| Visible thing | Built by instancing |
|---------------|---------------------|
| Glyph for character `a` | `LINE` + `CIRCLE` programs, glyph-specific parameters |
| Wall of a House room | `RECT` instanced N times along the wall path |
| Bookshelf | `RECT` planks instanced for each shelf |
| A book on the shelf | `RECT` for spine + `RECT` for covers (closed); add page `RECT`s on open |
| The text on a book page | `LINE`/`CIRCLE` glyph programs instanced per character |
| A hologram in the Living Room | Same `LINE`/`CIRCLE` programs at hologram-scale with emissive material |
| A UI element on the Tablet | Same `RECT`/`CIRCLE` programs, screen-space projection |
| A node in Knowledgeverse view | `CIRCLE` icon at LOD_ICON; expanded `LINE` graph at LOD_FULL |

**This is the GPU instancing argument Daniel made.** A single Drawing Galaxy star occupying ~1KB of VRAM can render 10,000 instances of itself across the House and the Knowledgeverse using only per-instance transform data. The alternative — storing 10,000 distinct `RECT` meshes — would burn VRAM for no information gain. K3D refuses that alternative as a matter of architecture.

### 4.2 Instance buffers

The composed-head pipeline already does frustum culling and LOD per [docs/briefings/ARCHITECTURE_BRIEFING.md](../docs/briefings/ARCHITECTURE_BRIEFING.md). Layer-0 instancing extends that:

- **Static instance buffer** (House geometry, furniture): populated once on House load, lives in VRAM permanently. Tens of thousands of `RECT`s for walls/floors/shelves; thousands of `LINE`/`CIRCLE`s for permanent decor.
- **Semi-static instance buffer** (books on shelves, glyphs on signage, character labels): updated when the House is reorganized or when new books are placed.
- **Dynamic instance buffer** (Tablet UI, hologram, active glyphs being typed, active Knowledgeverse view): rewritten per frame from the current ActionBuffer + TRM state.
- **Streaming instance buffer** (Knowledgeverse stars in the current FOV): driven by frustum culling over the GVRAM star table. Stars enter the buffer when LOD selects them; leave when they go out of frustum.

A single GPU draw call per primitive type (one for `LINE`, one for `CIRCLE`, one for `RECT`, …) covers the entire scene by walking these instance buffers. **This is the load-once-instance-many pattern that AAA game engines use.** K3D applies it to knowledge.

### 4.3 LOD + frustum culling apply to knowledge

The composed-head pipeline already applies LOD + frustum culling to spatial geometry. The same kernels apply to **knowledge structures in the Knowledgeverse**:

| LOD level | Knowledge representation | Visual representation |
|-----------|-------------------------|-----------------------|
| `LOD_INVISIBLE` | Star not loaded into Galaxy | Not in instance buffer |
| `LOD_ICON` | Star loaded but only `embedding` + `name` cached | Single `CIRCLE` instance, color from `meaning_class` |
| `LOD_SUMMARY` | Star loaded with `meaning_rpn` and one tier of refs | Constellation glyph: `CIRCLE` center + `LINE` to ref'd stars |
| `LOD_FULL` | Star loaded with full RPN, all symlinks resolved, all surface_forms instantiated | Full RPN expression rendered as nested `LINE`/`RECT` boxes |

The same frustum-culling kernel that drops out-of-FOV walls also drops out-of-FOV stars from the Galaxy load set. **The Galaxy is, finally, a working memory governed by what the avatar is currently looking at.** If the avatar walks into the Library and looks at the math shelf, math meaning stars stream into Galaxy at the appropriate LOD. If the avatar walks out, they decay back to LOD_ICON or LOD_INVISIBLE.

This is the second half of "no knowledge caps" (per the no-knowledge-caps memory): we never cap *how much* knowledge can be on disk in the House, we just LOD-cull what's currently active in Galaxy by what the avatar can see/think about. The Galaxy is always full of *relevant* stars at the right detail level.

---

## 5. Books as 3D Galaxies (closed → opened transition)

### 5.1 What a book is

Per the books-are-galaxies feedback memory: a book in the House is a 3D object on a shelf. The CONCEPT of "book" is a star (its shapes, sizes, defining properties, symlinks to the word "book" in many languages). The book OBJECT in the House is an instance of that concept's drawing programs, parameterized by the specific book's title, thickness, etc.

**The book's CONTENTS are a closed Galaxy on disk.** When the book is closed (the default), its Galaxy is not loaded into Knowledgeverse — its meaning stars sit in a GLB or JSONL file, not in VRAM.

### 5.2 Opening a book

When the avatar picks up a book and opens it (or the user clicks "open" via the Tablet):

1. **Animation phase** — the book mesh transitions from closed to open via Layer-0 `RECT` rotations (cover swings on spine hinge). Pure visual, no Galaxy load yet.
2. **Symlink phase** — the book's `content_galaxy_id` is dereferenced. The Galaxy file is memory-mapped (not eagerly loaded). Page faults bring in stars on demand.
3. **Symlink phase 2** — each meaning star in the book's Galaxy is registered into the **active Galaxy** in Knowledgeverse via symlink. Not copied. The Knowledgeverse grows by ref-counted symlink entries.
4. **LOD phase** — the new symlinked stars enter at LOD_ICON. As the avatar reads (camera focuses on a page), the stars at the focused position upgrade to LOD_SUMMARY → LOD_FULL.
5. **Closing phase** — when the book closes, ref counts drop. Stars whose only symlink was via this book unload from Galaxy back to LOD_INVISIBLE (still on disk in the book's Galaxy file).

**This is the "load once, instance many" pattern at the knowledge layer.** A library with 10,000 books does not put 10,000 Galaxies in VRAM. It puts the *currently-open books'* contents in VRAM, at the LOD the avatar's attention demands.

### 5.3 Reading a book

Two viewing modes (both procedural, both Layer-0):

- **In-hands mode** (default in the House): the book mesh is held by the avatar; pages render as `RECT`s with glyph instances drawn from the Character Galaxy on each page surface. The text on the page is **the same glyph stars** that label the rooms and the Tablet UI — instanced again, with different transforms.
- **Fullscreen reading mode** (Tablet pop-out, "more old way of seeing things"): the same content is projected onto a Tablet view that occupies the screen. Same glyphs, same Layer-0 primitives, different camera projection. No alternative renderer.

Both modes consume the same Galaxy. The "page" in either mode is just instance positions for glyphs. There is no "PDF renderer" or "EPUB renderer" — there is the K3D renderer that knows how to instance Layer-0 stars.

### 5.4 Why this matters for benchmarks (later)

When Daniel says "K3D is the game" and "benchmarks are natural activity," this is what he means: a math benchmark question is just a query the avatar reads from a Tablet. Solving it means TRM navigates Galaxy, picks meaning stars (digits, operators), runs `rpn_execute_device`, and answers via the Tablet output. The benchmark is a natural reading-and-answering activity inside the game. There is no separate "benchmark mode."

---

## 6. The Canonical Qdrant Database (NEW — Daniel's request)

### 6.1 What it is

A **second Qdrant collection** that sits alongside the existing `k3d_specifications` collection. It contains every CANONICAL thing the proceduralizer needs to look up at ingestion time:

- Canonical RPN opcodes (with name, hex, stack effect, semantic description, examples)
- Canonical opcode patterns / RPN program templates (e.g., "store-and-recall a property", "ternary force compute", "char sequence from word")
- Canonical star IDs that already exist in the House (so the proceduralizer never invents a duplicate for a concept that's already there)
- Canonical drawing primitives (Layer 0 stars) with their parameter shapes
- Canonical grammar construction templates per language (the RPN programs from §3.3)
- Canonical character galaxy stars (one per Unicode codepoint that K3D has materialized)
- Canonical meaning classes, domain paths, symlink kinds

**Why a separate Qdrant collection** rather than payload tags inside `k3d_specifications`:

- Specifications change slowly and chunk by document section. Canonical things change with every ingestion run and chunk by individual canonical entity. Mixing them would make incremental updates painful.
- Different lookups: the proceduralizer wants "what's the canonical opcode for store?" → expects exactly one answer. Spec lookups want "tell me about the dual-client contract" → expects a paragraph.
- Different access patterns: spec collection is read-mostly by humans + agents during conversation. Canonical collection is read-write by the proceduralizer during ingestion runs (and read by both Codex and Claude when reasoning about ingestion decisions).
- Different embedding scope: spec chunks are 2000 chars; canonical entries are 50–500 chars (an opcode definition fits in one paragraph). Same model (`fast-all-minilm-l6-v2`), different chunking.

### 6.2 Collection schema

```python
COLLECTION = "k3d_canonical"
VECTOR_NAME = "fast-all-minilm-l6-v2"   # same model as k3d_specifications, multi-user friendly
VECTOR_SIZE = 384

# Payload schema (every canonical entry has these fields)
{
    "kind":          str,    # "opcode" | "rpn_template" | "star_id" | "drawing_primitive"
                             # | "grammar_template" | "character_star" | "meaning_class"
                             # | "domain_path" | "symlink_kind"
    "canonical_id":  str,    # the ID the proceduralizer should use (e.g., "STORE", "char_a",
                             # "concept_water", "grammar_en_svo", "drawing_rect")
    "canonical_name": str,   # human label
    "summary":       str,    # 1-3 sentence description (the embedded text)
    "details":       dict,   # kind-specific structured fields (stack effect, params, etc.)
    "source_file":   str,    # where this canonical entry was extracted from
    "source_line":   int,
    "added_at":      str,    # ISO date
    "version":       int,    # bumps when details change (id stays stable)
}

# Payload indexes for fast filtering
client.create_payload_index("kind", KEYWORD)
client.create_payload_index("canonical_id", KEYWORD)
```

### 6.3 What the proceduralizer queries

| When the proceduralizer needs… | It queries `k3d_canonical` for… | And uses the result to… |
|-------------------------------|------------------------------|------------------------|
| The right opcode for "store a property on the stack" | `kind:opcode` near "store property to register" | Emit `STORE` (not invent `SAVE_PROP`) |
| Whether "concept_water" already exists | `kind:star_id canonical_id:concept_water` exact | Skip creation, just symlink |
| The canonical char_id for `é` | `kind:character_star canonical_name:é` exact | Use `char_u00e9` (not `char_pt_e_acute`) |
| The grammar template for English SVO | `kind:grammar_template` near "english subject verb object" | Bind the existing template to the new meaning (don't write a fresh one) |
| The canonical RPN program for "ternary semantic force" | `kind:rpn_template` near "ternary semantic force" | Reuse the program, parameterize for the two stars |
| The right domain path for a new chemistry concept | `kind:domain_path` near "chemistry organic" | Use `Library/Reality/Chemistry/Organic` (not invent `chem_org`) |

**The proceduralizer becomes a lookup-first, create-only-when-missing process.** Every new ingestion first queries the canonical DB; only if no canonical match exists does it create. When it creates, the new canonical entry is added back to `k3d_canonical` so future runs can find it.

### 6.4 What populates `k3d_canonical`

A new ingest script `scripts/ingest_canonical_to_qdrant.py` modeled on `ingest_specs_to_qdrant.py`. It harvests canonical entries from authoritative sources:

- **Opcodes** — extracted from `knowledge3d/cranium/rpn_opcodes.py` (or wherever the canonical opcode table lives)
- **RPN templates** — extracted from `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` example sections + `star_crafter.py` reference programs
- **Star IDs** — extracted from the live `house_stars.jsonl` (or whatever the canonical House persistence file is); only confirmed (`confidence=+1`) stars
- **Drawing primitives** — extracted from `knowledge3d/ingestion/atomic/drawing_grammar_builder.py`
- **Grammar templates** — extracted from the Grammar Galaxy population modules (when they exist; for now, what's in `multilingual_meanings.py` and `_house_utils.py`)
- **Character stars** — extracted from `knowledge3d/cranium/character_galaxy.py` and `procedural_fonts.py`
- **Meaning classes / domain paths / symlink kinds** — extracted from the meaning star schema dataclass definitions

The script runs idempotently: re-runs update entries whose `version` field bumped, skip unchanged ones.

### 6.5 How the proceduralizer reaches it

A small Python helper `knowledge3d/ingestion/canonical_lookup.py`:

```python
class CanonicalLookup:
    """Read-only client for the k3d_canonical Qdrant collection.

    Used by the proceduralizer to avoid duplicating canonical entities.
    Sovereign? No — this is INGESTION-PATH ONLY. Never imported from
    anything under knowledge3d/cranium/ or knowledge3d/knowledgeverse/.
    """

    def __init__(self, qdrant_url: str = "http://localhost:6333"): ...

    def find_opcode(self, semantic_query: str) -> Opcode | None: ...
    def find_star_id(self, canonical_id: str) -> StarRef | None: ...
    def find_character_star(self, char: str) -> CharStar | None: ...
    def find_grammar_template(self, language: str, construction: str) -> GrammarTemplate | None: ...
    def find_domain_path(self, semantic_query: str) -> str | None: ...

    def register(self, kind: str, canonical_id: str, **fields) -> None:
        """Add a NEW canonical entry. Used after the proceduralizer
        materializes a star or template that didn't exist before."""
```

**Sovereignty boundary.** This module is *ingestion-path only*. The hot path (PTX kernels, Galaxy queries, RPN execution) NEVER touches Qdrant. Daniel's no-fallbacks rule still holds. Qdrant is just the proceduralizer's lookup partner during ingestion runs — same status as numpy / sentence-transformers in the ingestion path today.

### 6.6 Bootstrap order (for Codex)

1. Land the canonical schema and the index script (no live data yet).
2. Backfill from `rpn_opcodes.py` (opcodes only) — should be a few dozen entries. Verify via `qdrant-find` from the MCP that "store" returns `STORE`.
3. Backfill from `star_crafter.py` (the 15 reference math stars) — should populate ~15 `star_id` entries and ~5 `rpn_template` entries.
4. Wire `CanonicalLookup` into `multilingual_meanings.py` first (since it's the worst offender for ID drift per Phase 7). Replace `_lemma_word_ref()` with `CanonicalLookup.find_star_id()` + create-if-missing.
5. Wire into `word_meaning_builder.py`, `_house_utils.py`, and `star_crafter.py`. Each call site stops minting fresh IDs; they query first.
6. Phase 7 dangling-pointer check (per the Phase 7 spec) becomes a Qdrant grep: every `word_ref` and `char_ref` in the build must resolve to a `star_id` entry in `k3d_canonical`.

### 6.7 What this prevents

Re-stating the five gaps from [TEMP/CLAUDE_PHASE7_PROCEDURALIZATION_QUALITY_AND_MULTILINGUAL_SYMLINKS_2026-04-11.md](CLAUDE_PHASE7_PROCEDURALIZATION_QUALITY_AND_MULTILINGUAL_SYMLINKS_2026-04-11.md), in light of the canonical DB:

- **Gap 1 — Dangling `word_ref` (ID format mismatch).** Solved by canonical lookup: there is exactly one canonical ID format per kind, registered in Qdrant. Two modules that ask "what's the star_id for the English word 'cat'" get the same answer.
- **Gap 2 — Dangling `char_refs`.** Solved: character stars are pre-registered as `kind:character_star` for every Unicode codepoint K3D has materialized. The proceduralizer can't reference a char_id that doesn't exist — `find_character_star('é')` either returns the canonical id or registers a new char star (and adds it to the canonical DB).
- **Gap 3 — Unidirectional symlinks.** Not solved by Qdrant directly — still requires the `_link()` enforcement from Phase 7.B. But the canonical DB makes it impossible to *try* to link to an id that doesn't exist.
- **Gap 4 — `meaning_rpn` is descriptive text.** Solved by `kind:rpn_template`: the proceduralizer queries for the canonical template for "store a noun's properties" and binds it, instead of inventing a `"SYNSET NOUN cat DEF small domesticated feline"` string.
- **Gap 5 — No sovereign embedding.** Solved already by 6.B.3 (sovereign Matryoshka). Canonical DB just makes the embeddings consistent because every star's text fields go through the same canonical-vocabulary normalization at proceduralization time.

---

## 7. The dual-client contract, restated for this vision

[DUAL_CLIENT_CONTRACT_SPECIFICATION.md](../docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md) §1.6 already establishes the form+meaning principle. This vision spec adds three concrete obligations:

1. **Knowledge is procedural, not embedded.** Every meaning star must carry a `meaning_rpn` program that a human can read top-to-bottom and a TRM can execute opcode-by-opcode. Embeddings are search keys ONLY. If the embedding is lost, the star is fully recoverable from `meaning_rpn` and surface forms.

2. **Language is composed, not stored.** Translations are not stored. Surface forms in language X are produced by composing meaning star + grammar template + word stars + character stars at runtime. Both the human inspecting the system and the TRM executing it can see the composition trace (the RPN stack states across stages 1-5 in §3.2). The Tablet that displays the result also displays the trace (in a debug overlay) on demand.

3. **Visual and knowledge share Layer 0.** The same Drawing Galaxy stars that draw glyphs draw rooms. The same LOD pipeline that culls walls culls knowledge stars. Humans and AI watching the same House see and navigate the same procedural scene at the same time. There is no "AI sees vectors, human sees pixels" — both see (and reason about) the procedural primitives instanced at the same positions.

---

## 8. What this vision is NOT (boundaries)

- **Not retraining TRM.** Strategies are added by sleep-time consolidation as new RPN composition patterns. The base TRM weights stay generic and modality-agnostic.
- **Not adding language opcodes.** Per §3.3, grammar is composed from existing opcodes. Promotion to a fused opcode goes through the [RPN_DOMAIN_OPCODE_REGISTRY.md](../docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md) §6 pipeline only after usage justifies it.
- **Not building a translation system.** There is no language-pair lookup. Cross-language navigation is by symlink composition through the meaning center.
- **Not deprecating embeddings.** Embeddings remain the cosine search index for navigation. They are just NOT the knowledge.
- **Not making Qdrant a hot-path dependency.** Qdrant is ingestion-path only. The composed-head pipeline never touches it.
- **Not solving the per-language Word Galaxy partitioning question.** Stars in different language Word Galaxies live in the same VRAM star table; `domain` field discriminates. No physical partitioning of VRAM by language.

---

## 9. Order of operations (replacing Phase 7 ordering)

**Phase 7.0 — Canonical Qdrant DB (NEW, prerequisite to everything else)**
- Land schema + index script + `CanonicalLookup` helper
- Backfill from rpn_opcodes.py, star_crafter.py
- Verify via MCP `qdrant-find`

**Phase 7.A — Canonical IDs everywhere (revised from prior 7.A)**
- Replace ad-hoc ID minting in `multilingual_meanings.py`, `word_meaning_builder.py`, `_house_utils.py`, `character_galaxy.py` with `CanonicalLookup` calls
- Character Galaxy backfill: every Unicode codepoint K3D has touched gets a canonical character star, language-agnostic by codepoint
- Word Galaxy instantiation: `synset_to_star()` extended to yield word stars per language alongside meaning stars

**Phase 7.B — Bidirectional symlinks (unchanged from prior spec)**
- `_link()` extracted to shared helper
- All multilingual builder writes go through `_link()`
- "water" cross-language round-trip test passes

**Phase 7.C — Grammar Galaxy seed templates (NEW, replaces prior 7.C)**
- 5 grammar construction templates per supported language (EN, PT, JA, ES, FR for v1):
  - declarative SVO/SOV/VSO
  - question-form
  - definitional ("X is Y")
  - relational ("X has Y", "X is part of Y")
  - periphrastic explanation (the saudades-class fallback template per language)
- Each template is an RPN program registered as `kind:grammar_template` in the canonical DB
- Each template references the stable opcodes from §3.3
- Codex implements one cross-language synthesis test: feed `concept_saudades_pt` with target_language=en, assert output contains "longing" / "irrecoverable" / "Portuguese: saudades" via grammar composition (no Python string formatting)

**Phase 7.D — `meaning_rpn` quality tiers (unchanged from prior 7.C, demoted)**
- Star crafter continues stamping bytecode for executable program stars
- New value-bearing stars (digits, concepts) get STORE/RECALL programs from canonical templates
- Form stars (chars, glyphs) get minimal identity programs

**Phase 7.E — Layer-0 instancing audit (NEW)**
- Inventory existing Drawing Galaxy stars
- Confirm House geometry, glyphs, Tablet UI all draw from the same Drawing Galaxy stars
- Add the four instance buffers (static / semi-static / dynamic / streaming) if not present
- LOD-cull the Knowledgeverse star load set by avatar frustum + attention

---

## 10. What changes in the prior Phase 7 spec

| Phase 7 element | Status now |
|-----------------|------------|
| `canonical_ids.py` (Python module) | **REPLACED** by canonical Qdrant DB + `CanonicalLookup` helper |
| Character Galaxy builder (62 ASCII + lazy Unicode) | **KEPT** but reframed: it populates the canonical DB with character stars |
| Word Galaxy instantiation in `synset_to_star()` | **KEPT** but uses `CanonicalLookup.find_star_id()` first |
| Bidirectional `_link()` enforcement | **KEPT, unchanged** (Phase 7.B) |
| `meaning_rpn` quality tiers | **KEPT** but demoted to 7.D, after grammar templates land |
| Sovereign Matryoshka embedding for word/char stars | **KEPT** — already done in 6.B.3, re-verified during 7.A backfill |
| "STORE/RECALL pattern for word stars" requirement | **REPLACED**: word stars carry minimal form RPN (per §3.3); the *meaning* stars they symlink to carry the procedural definitions |
| Cosine probe gates | **KEPT** as informational; no longer the success metric |

The success metric for the proceduralizer is no longer "embedding cosine ranks." It is:
1. Zero dangling refs (every `word_ref` and `char_ref` resolves)
2. All bidirectional symlinks resolve in both directions
3. Cross-language synthesis test (saudades → English periphrastic) passes
4. The canonical DB has zero duplicates for the same semantic (verified by a script that groups by Stage-3 cosine top-1 within `k3d_canonical`)

---

## 11. Quality gates for the vision (long-term, not Phase 7 alone)

- **Procedural readability** — pick 10 random meaning stars; a human (Codex acting as a fresh reader) can read each `meaning_rpn` and explain what the concept is, with no other context. Embeddings not consulted.
- **Cross-language synthesis** — given a meaning star and a target language, the grammar pipeline produces a sentence by RPN composition; the trace is inspectable; no Python `str.format()` or `f"{...}"` involvement at any stage.
- **Saudades round-trip** — "saudades" in PT → meaning → English periphrastic explanation → close to "deep longing for irrecoverable past" with the loanword appended.
- **Layer-0 instancing audit** — every visible primitive in the live House+Knowledgeverse view comes from a Drawing Galaxy star instanced via the four instance buffers. No bespoke meshes for glyphs, walls, or holograms.
- **Knowledge LOD** — walking from the Library to the Garden in the House changes the Galaxy's loaded star set per frame; math stars unload, biology stars load, in proportion to FOV + attention.
- **Canonical DB freshness** — re-running the canonical ingest script after a major proceduralizer run finds zero new entries the proceduralizer didn't already register (proves bidirectional sync).
- **Sovereignty unchanged** — composed-head pipeline still runs zero Python in the hot path; Qdrant only ever touched in ingestion runs.

---

## 12. The principle, restated

**Meaning is the center. Languages are composed at runtime by RPN math. Drawing primitives are instanced everywhere. Knowledge has LOD. Books are Galaxies. The proceduralizer asks Qdrant before it invents.**

This is not "make Phase 7 bigger." It is the architecture Phase 7 was approximating without naming. Phase 7's symlink work is still needed and still goes to Codex. But the success metric is no longer "embedding ranks" — it is "the system speaks and reads procedurally."

When this lands, Daniel's answer to "what does saudades mean in English?" comes from the same TRM that answers "what is 2+3?" — by navigating Galaxy, picking a meaning, composing via grammar templates, and emitting through the Tablet. One system, many modalities, one model, generic and generalizable.

---

## Appendix A — Canonical Qdrant DB: minimal payload examples

```json
// kind=opcode
{
  "kind": "opcode",
  "canonical_id": "STORE",
  "canonical_name": "STORE",
  "summary": "Pop a value from the stack and write it to a named register. Inverse of RECALL. Used for property storage on meaning stars and for grammar slot filling.",
  "details": {
    "hex": "0x60",
    "stack_effect": "value name -- ",
    "register_kind": "ternary_triple"
  },
  "source_file": "knowledge3d/cranium/rpn_opcodes.py",
  "source_line": 142
}

// kind=star_id
{
  "kind": "star_id",
  "canonical_id": "concept_water",
  "canonical_name": "water",
  "summary": "The substance H2O in all states; the universal solvent in chemistry; essential to biology; cultural archetype across many languages.",
  "details": {
    "meaning_class": "concept",
    "domain": "Library/Reality/Chemistry/Compounds",
    "surface_languages": ["en", "pt", "ja", "ar", "zh"]
  },
  "source_file": "house_stars.jsonl",
  "source_line": 4127
}

// kind=character_star
{
  "kind": "character_star",
  "canonical_id": "char_a",
  "canonical_name": "a",
  "summary": "Latin lowercase letter A. Codepoint U+0061. Shared across English, Portuguese, Spanish, French, Italian, German, etc.",
  "details": { "codepoint": 97, "script": "Latin", "case": "lower" },
  "source_file": "knowledge3d/cranium/character_galaxy.py",
  "source_line": 88
}

// kind=grammar_template
{
  "kind": "grammar_template",
  "canonical_id": "grammar_en_definitional",
  "canonical_name": "English definitional construction",
  "summary": "Generates 'X is Y' constructions in English given a subject meaning star and a predicate meaning star. Handles article selection (a/an/the) by phonetic check on the predicate's first character.",
  "details": {
    "language": "en",
    "rpn": "RECALL subject WORD_RESOLVE en  RECALL is_verb_form WORD_RESOLVE en  RECALL predicate WORD_RESOLVE en  ARTICLE_SELECT_EN  CONCAT_MORPHEMES",
    "slots": ["subject", "predicate"]
  },
  "source_file": "TEMP/CLAUDE_MEANING_FIRST_LANGUAGE_AS_RPN_VISION_2026-04-11.md",
  "source_line": 0
}

// kind=rpn_template
{
  "kind": "rpn_template",
  "canonical_id": "rpn_template_property_store",
  "canonical_name": "Store a numeric property on a meaning star",
  "summary": "Push a numeric value, then a property name, then STORE. Standard pattern for value-bearing concept stars (mass, count, density, temperature, etc.).",
  "details": {
    "rpn": "<value> STORE_<property_name>",
    "example": "4.0 STORE_mass_kg"
  },
  "source_file": "knowledge3d/ingestion/star_crafter.py",
  "source_line": 314
}
```

---

## Appendix B — One sentence for Codex

> Build the canonical Qdrant DB first; replace every ad-hoc ID-mint and template-string in the proceduralizer with a `CanonicalLookup` call; then write the five grammar construction templates per language as RPN programs registered in the canonical DB; then make the cross-language saudades→English synthesis test pass — without Python string formatting anywhere on the path.
