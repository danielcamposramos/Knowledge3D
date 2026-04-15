# CLAUDE → CODEX — Phase 7: Proceduralization Quality + Multilingual Symlink Integrity — 2026-04-11

> **SUPERSEDED IN PART by [CLAUDE_MEANING_FIRST_LANGUAGE_AS_RPN_VISION_2026-04-11.md](CLAUDE_MEANING_FIRST_LANGUAGE_AS_RPN_VISION_2026-04-11.md).**
> Read the vision spec first. It establishes Phase 7.0 (Canonical Qdrant DB) as prerequisite, reframes language as RPN math composition (not translation tables), pins the saudades rule (`untranslatable_languages` field), and adds Layer-0 instancing audit. The order of operations and section labels below are revised in §9 of the vision spec. This file is preserved as the audit record + slice-level detail for 7.A and 7.B.

## Context

Phase 6.C landed the first sovereign GPU math answer (`"2+3?" = "5"`). The substrate works: TRM navigates, Galaxy carries programs, RPN core runs them, Tablet presents the result. `star_crafter.py` is the reference-quality module — it uses `_link()` for bidirectional symlinks, stamps `meta_rule_addr` into the native 400-byte record, uses the sovereign Matryoshka RPN embedder, and sets `selection_role`/`answer_eligible` correctly.

Daniel's next question: does the **rest** of the proceduralization pipeline produce stars of the same quality? Specifically:
1. Do stars carry actionable RPN programs with executable bytecode?
2. Are symlinks bidirectional and resolving (not dangling strings)?
3. Do words symlink to characters (sometimes shared across languages)?
4. Does each language have its own Word Galaxy, with all multi-language stars symlinking by meaning?

## Audit findings (Claude + Kimi swarm)

Full swarm output: [TEMP/KIMI_PROCEDURALIZATION_QUALITY_AUDIT_2026-04-11.md](KIMI_PROCEDURALIZATION_QUALITY_AUDIT_2026-04-11.md)

### Module scorecard

| Module | Executable RPN | Roles set | Bidirectional `_link()` | `meta_rule_addr` | Sovereign embedder | Grade |
|--------|---|---|---|---|---|---|
| `star_crafter.py` | ✅ bytecode in program table | ✅ correct | ✅ `_link()` helper | ✅ native 400-byte field | ✅ Matryoshka singleton | **A** |
| `word_meaning_builder.py` | ❌ pass-through from JSONL | ❌ not set | ❌ manual dicts | ❌ absent | ❌ deferred (`None`) | **F** |
| `multilingual_meanings.py` | ❌ descriptive text (`SYNSET NOUN cat DEF ...`) | ❌ not set | ❌ one-sided `SurfaceForm` | ❌ absent | ❌ not used | **F** |

**Note on star_crafter.py confusion in the Kimi report:** the swarm read a truncated file and missed `extra_top_level` at line 464-468 which stamps `meta_rule_addr`, `program_flags`, `program_length`, `program_opcode_count` onto the row, and `_link()` at line 358-360 which enforces bidirectional symlinks. The crafter IS reference quality. The other two modules are not.

### Five systemic gaps

#### Gap 1 — Dangling `word_ref` pointers (ID format mismatch)

`multilingual_meanings.py` line 101-102:
```python
def _lemma_word_ref(language: str, lemma: str) -> str:
    return f"{language}_{str(lemma or '').strip().lower().replace(' ', '_')}"
```
Produces `"en_cat"`. But `word_meaning_builder.py` produces word stars with `meaning_id = "WORD_en_cat_default"`. These two modules **do not agree on the ID format** for Word Galaxy stars. Every `word_ref` in a multilingual meaning star is a dangling pointer.

#### Gap 2 — Dangling `char_refs` (Character Galaxy stars never created)

`_house_utils.py` line 8-17 produces `char_refs` like `"char_a"` (ASCII) and `"char_pt_u00e3"` (non-ASCII). `word_meaning_builder.py` line 99 produces `"LETTER_A_LATIN"` via `_letter_concept()`. **No module creates Character Galaxy stars with either ID format.** Both are dangling strings that point to nothing.

#### Gap 3 — Unidirectional symlinks (no reverse path)

`multilingual_meanings.py` writes `surface_forms[language].word_ref = "en_cat"` on the meaning star. But the word star (if it existed) has **no** `taxonomy_ref` or `meaning_star_ref` pointing back. The spec requires **all symlinks bidirectional**. Currently:
- Meaning → Word: ✅ (forward ref exists as a string)
- Word → Meaning: ❌ (no reverse field)
- Word → Character: ❌ (word_meaning_builder uses `letter_refs` dict, not `char_refs`)
- Character → Word: ❌ (characters don't reference their parent words)

#### Gap 4 — `meaning_rpn` is descriptive text, not executable RPN

`multilingual_meanings.py` line 203:
```python
meaning_rpn = f"SYNSET {entry.pos.upper()} {_safe_rpn_token(english_primary)} DEF {english_definition[:80]}"
```
This is a human-readable label, not an executable RPN program. It cannot be dispatched to the GPU RPN executor. `word_meaning_builder.py` similarly passes through `star.get("meaning_program")` from JSONL input without any compilation step.

The star_crafter solved this for math: operator stars have `meta_rule_addr → VRAM program table → bytecode`. The word/multilingual path has no equivalent.

**Important nuance:** not all stars need executable programs. Digit/concept stars are value-bearing (the star IS the answer). But any star that represents an *action* or *transformation* (e.g., grammar rules, operator words, verb meanings) should carry executable RPN. The current pipeline makes no such distinction.

#### Gap 5 — No sovereign Matryoshka embedding

`word_meaning_builder.py` line 134: `"embeddings": {"matryoshka": None, "regenerable": True}` — defers embedding. `multilingual_meanings.py` has no embedding path at all. Both modules produce stars that will get whatever fallback the sovereign build path provides, which (before 6.B.3) was either a raw trigram hash or a sentence-transformer — neither of which produces Matryoshka-stack embeddings in the same space as the crafter stars.

After 6.B.3, the sovereign path's `_entry_embedding64` is wired to the Matryoshka singleton. So if the word/multilingual stars carry enough text in the `name`/`description` fields, the sovereign build will embed them correctly at boot. **But this should be verified, not assumed.**

---

## Phase 7 spec (revised — see vision spec §9 for full ordering)

> **Revised ordering:** 7.0 (Canonical Qdrant DB) → 7.A (Canonical IDs via DB lookup) → 7.B (Bidirectional symlinks) → 7.C (Grammar Galaxy seed templates per language, NEW) → 7.D (`meaning_rpn` quality, demoted from old 7.C) → 7.E (Layer-0 instancing audit, NEW). The slice details below cover 7.A and 7.B; for 7.0, 7.C, 7.D, 7.E read the vision spec.

### 7.0 — Canonical Qdrant DB (PREREQUISITE — see vision spec §6)

Stand up sibling collection `k3d_canonical` alongside `k3d_specifications`. Holds opcodes, RPN templates, canonical star IDs, drawing primitives, grammar templates, character stars, meaning classes, domain paths, and symlink kinds. Proceduralizer uses `CanonicalLookup` (ingestion-path-only) to query before creating any new artifact. Sovereignty boundary: `qdrant_client` import allowed under `knowledge3d/ingestion/`, FORBIDDEN under `knowledge3d/cranium/` and `knowledge3d/knowledgeverse/`. Bootstrap script: `scripts/ingest_canonical_to_qdrant.py` (model: `scripts/ingest_specs_to_qdrant.py`). Land this BEFORE 7.A — the canonical-ID work below assumes the lookup helper exists.

### 7.A — Canonical ID resolution and Character/Word Galaxy instantiation

**The root problem:** meaning stars reference word and character stars that don't exist. Before fixing symlink direction, we need the targets to exist.

**Canonical ID contract — sourced from `k3d_canonical`, not a hand-rolled Python module.** The ID format functions below are the LOGIC the canonical DB encodes; at proceduralization time, callers go through `CanonicalLookup.find_star_id()` first and only fall through to deterministic generation when the lookup misses (then `register()` the result). This keeps one source of truth across all ingestion paths.

```python
# Logic encoded in k3d_canonical (kind="star_id" entries) and mirrored as a thin
# fallback in knowledge3d/ingestion/canonical_ids.py for offline ingestion runs.

def word_star_id(language: str, lemma: str) -> str:
    """Canonical Word Galaxy star ID."""
    return f"word_{language}_{lemma.strip().lower().replace(' ', '_')}"

def char_star_id(char: str, language: str) -> str:
    """Canonical Character Galaxy star ID. ASCII chars are language-agnostic."""
    if char.isascii() and char.isalnum():
        return f"char_{char.lower()}"          # shared: English "cat" and Portuguese "gato" both ref char_a
    return f"char_u{ord(char):04x}"            # Unicode: language-agnostic by codepoint
```

**Key design decision on character sharing:** per schema spec, the same character star is shared across languages. English "cat" and Portuguese "gato" both reference `char_a`. The character star for `ã` is `char_u00e3` — it is NOT `char_pt_u00e3`. Language does not belong on character IDs. Characters are Layer 1 Form — they exist once, referenced by many words in many languages.

**Character Galaxy instantiation:** New helper `build_character_galaxy_stars()` in `knowledge3d/ingestion/atomic/character_galaxy_builder.py`:
- ASCII alphanumeric: 62 stars (a-z, A-Z, 0-9) + common punctuation
- Unicode: lazily instantiated as words reference them during ingestion
- Each character star: `meaning_class="form"`, `meaning_rpn="CHAR {codepoint} GLYPH"`, `domain="Character/{script}"`, `selection_role="unknown"`, `answer_eligible=False`
- Embedding via sovereign Matryoshka embedder

**Word Galaxy instantiation:** `multilingual_meanings.py::synset_to_star()` already produces meaning stars with `surface_forms[language].word_ref`. But the word stars themselves don't exist. Two paths:

1. **Preferred:** extend `synset_to_star()` to also yield word stars alongside meaning stars. For each `(language, lemma)` in a synset, yield:
   - One meaning star (the concept, as today)
   - One word star per language/lemma pair: `meaning_class="concept"`, `domain=f"Word/{language}"`, `star_id=word_star_id(language, lemma)`, carrying `char_refs` via `canonical_ids.char_star_id()` for each character

2. **Alternative:** a separate `build_word_galaxy_stars()` that consumes the output of `synset_to_star()` and materializes the referenced word stars.

Path 1 is simpler and guarantees consistency. Use it.

**Migrate all callers** of `_lemma_word_ref()` and `_letter_concept()` and `_house_utils.char_refs()` to use `CanonicalLookup` (with `canonical_ids` as the deterministic fallback). The old functions become thin wrappers that route through the lookup, or are deleted. Every new word/character star MUST be `register()`-ed in `k3d_canonical` so the next proceduralization run finds it.

**Saudades rule (per vision spec §2):** when a meaning has no equivalent in another language, set `untranslatable_languages` on the meaning star and let the meaning self-symlink (`surface_forms.{lang}.word_ref` may equal the meaning's own `star_id` for languages where the meaning IS the surface form). Do NOT manufacture fake translations for completeness.

**Validation:** after 7.A, the sovereign build should have zero dangling `word_ref` or `char_refs` in any meaning star's `surface_forms`. Add a build-time check: for every star with `surface_forms`, assert every `word_ref` and every `char_ref` resolves to an existing star in the same build.

### 7.B — Bidirectional symlink enforcement

**Now that targets exist, wire the reverse links.**

Extend `multilingual_meanings.py` to use a `_link()` helper (import from `star_crafter.py` or extract into a shared `knowledge3d/ingestion/symlink_helpers.py`). The single `_link(a, b, fwd_kind, bwd_kind)` pattern from the star_crafter is the reference implementation.

Required bidirectional pairs:

| Forward | Backward | Example |
|---------|----------|---------|
| meaning_star → `surface_forms.{lang}.word_ref` | word_star → `taxonomy_refs` | `concept_cat.surface_forms.en.word_ref = word_en_cat` ↔ `word_en_cat.taxonomy_refs = [concept_cat]` |
| word_star → `component_refs` (char_refs) | char_star → `composite_of` | `word_en_cat.component_refs = [char_c, char_a, char_t]` ↔ `char_c.composite_of = [word_en_cat, word_en_car, ...]` |
| meaning_star → `grammar_refs` | grammar_star → `taxonomy_refs` | bidirectional as in star_crafter |
| meaning_star → `meta_refs` | program_star → `taxonomy_refs` | bidirectional as in star_crafter |

**Multi-language meaning-to-word:** the same meaning star has `surface_forms.en.word_ref = word_en_cat` AND `surface_forms.pt.word_ref = word_pt_gato` AND `surface_forms.ja.word_ref = word_ja_u732b`. All three word stars have `taxonomy_refs` pointing back to the same meaning star. That is how cross-language navigation works: from `word_pt_gato` → `taxonomy_refs` → `concept_cat` → `surface_forms.en.word_ref` → `word_en_cat`. Bidirectional symlinks are the routing mechanism, not translation tables.

**Character sharing across languages:** `word_en_cat` and `word_pt_gato` both have `component_refs` containing `char_a`. `char_a.composite_of` contains both `word_en_cat` and `word_pt_gato`. Shared by construction — the canonical character ID is language-agnostic for ASCII.

**Validation:** pick one multilingual synset (e.g., the synset for "water"). Assert:
- `concept_water.surface_forms.en.word_ref == "word_en_water"`
- `concept_water.surface_forms.pt.word_ref == "word_pt_agua"`
- `word_en_water.taxonomy_refs` contains `concept_water`
- `word_pt_agua.taxonomy_refs` contains `concept_water`
- `word_en_water.component_refs == ["char_w", "char_a", "char_t", "char_e", "char_r"]`
- `word_pt_agua.component_refs` contains `char_a` (shared with English)
- `char_a.composite_of` contains both `word_en_water` and `word_pt_agua`

### 7.C — Grammar Galaxy seed templates per language (NEW — see vision spec §3.3, §9)

Before we can demand `meaning_rpn` quality, the Grammar Galaxy needs the templates that compose meanings into surface forms via the 5-stage RPN pipeline (MEANING DECOMPOSE → GRAMMAR SELECT → WORD RESOLVE → MORPHOLOGY APPLY → SURFACE EMIT). Per language, seed:

- One `grammar_template` star per major construction (NP, VP, copula, predicate, periphrastic phrase)
- For untranslatable concepts (saudades), at least one **periphrastic template** in every other language ("a feeling of nostalgic longing for ...") so the synthesis pipeline has a fallback path that does NOT require a single-word lemma
- Templates registered in `k3d_canonical` (kind=`grammar_template`) so the proceduralizer reuses them across crafters
- Compose ONLY from existing RPN opcodes (Programs Before Opcodes — RPN_DOMAIN_OPCODE_REGISTRY §1). Do NOT add language-specific opcodes.

This slice is the bridge between meaning stars (7.A/7.B output) and `meaning_rpn` quality (7.D below). Without grammar templates, value-bearing concept stars have nowhere to dispatch their `STORE`/`RECALL` programs to produce surface text.

### 7.D — Sovereign Matryoshka embedding and `meaning_rpn` quality

**Embedding:** ensure all word and character stars pass through the sovereign Matryoshka embedder singleton at ingestion time. The `word_meaning_builder` and `multilingual_meanings` modules should either:
- Stamp the full Matryoshka tier stack on the star dict (preferred — same pattern as star_crafter)
- Or leave `embedding=[]` and rely on the sovereign build's fallback to the Matryoshka singleton (acceptable if the star carries rich enough text in `name`/`domain`/`description` — verify this)

**`meaning_rpn` quality tiers:** not all stars need executable bytecode. The spec (schema §2.2) says `meaning_rpn` is "an RPN program that COMPUTES the concept's meaning — its properties, relationships, constraints." Three tiers:

1. **Executable program stars** (operators, grammar rules, transformation rules): `meaning_rpn` = bytecode in VRAM program table, `meta_rule_addr` > 0. These are the stars the fused tick can dispatch. star_crafter covers this for arithmetic.

2. **Value-bearing concept stars** (digits, nouns, adjectives): `meaning_rpn` = a declarative RPN that stores properties. E.g., `"4.0 STORE_mass_kg 4 STORE_leg_count GALAXY_LOOKUP mammalia STORE_class"` for the cat concept. Not dispatched for execution — used by TRM navigation and semantic gravity to compute meaning mass. These should be meaningful text programs, not descriptive labels. The current `"SYNSET NOUN cat DEF small domesticated feline"` is close but should be restructured into `STORE`/`RECALL` operations per RPN_DOMAIN_OPCODE_REGISTRY §2.

3. **Form stars** (characters, word-forms): `meaning_rpn` = minimal procedural form. E.g., `"CHAR 0x0061 GLYPH LATIN LOWERCASE"` for `char_a`. Not executed — just identity.

For Phase 7.C, **do not attempt to make all 41,000+ existing stars carry executable RPN**. That is a years-long task. Instead:
- Ensure all **new** stars from the crafter/builder path carry tier-appropriate `meaning_rpn`
- Ensure the sovereign build does not reject stars with descriptive `meaning_rpn` (it shouldn't — `meaning_rpn` is not validated as bytecode anywhere today)
- File a follow-up for sleeptime consolidation to progressively upgrade descriptive `meaning_rpn` to structured RPN during idle cycles

**Validation:**
- After 7.D, the sovereign build embeds word/character stars in the same Matryoshka space as crafter stars. Cosine probe: `"cat"` → `word_en_cat` rank ≤ 20 at tier_64. `"gato"` → `word_pt_gato` rank ≤ 20.
- `meaning_rpn` for new word stars follows the `STORE`/`RECALL` pattern, not bare descriptive labels.
- **Embeddings are an INDEX, not the knowledge itself.** Per dual-client contract: human-readable `meaning_rpn` text is the canonical knowledge; embeddings are the cosine routing layer over it. If the `meaning_rpn` is empty or descriptive-only, the embedding is masking missing knowledge.

### 7.E — Layer-0 instancing audit (NEW — see vision spec §4, §9)

Audit the proceduralization pipeline for Layer-0 instance reuse. Every visible artifact (glyph, character, word surface, book cover, shelf, room wall, UI panel) should resolve to a `drawing_primitive` registered in `k3d_canonical`, instanced from one of the four buffers (static / semi-static / dynamic / streaming). Knowledge LOD applies to Galaxy stars themselves (LOD_INVISIBLE → LOD_ICON → LOD_SUMMARY → LOD_FULL) — confirm the proceduralizer stamps `lod_class` on every star. This is an audit slice, not a rewrite — file follow-ups for any module that owns its own draw path instead of reusing Layer-0.

---

## Order of operations (revised)

1. **7.0** — Stand up `k3d_canonical` Qdrant collection, write `scripts/ingest_canonical_to_qdrant.py`, ship `CanonicalLookup` ingestion-path helper, sovereignty grep gate.
2. **7.A** — Character galaxy builder, word star instantiation in `synset_to_star()`, migrate callers through `CanonicalLookup`. Dangling-pointer build check. Saudades `untranslatable_languages` field landed.
3. **7.B** — `_link()` extraction to shared helper, bidirectional symlinks in multilingual builder, cross-language resolution test for "water" + saudades self-symlink test.
4. **7.C** — Grammar Galaxy seed templates per language (NP/VP/copula/periphrastic), all composed from existing RPN opcodes, all registered in `k3d_canonical`.
5. **7.D** — Sovereign Matryoshka embedding for word/char stars, `meaning_rpn` quality tiers, cosine probe for multilingual routing.
6. **7.E** — Layer-0 instancing audit, `lod_class` stamping on every star, follow-up tickets for non-conforming modules.

## What this does NOT do

- Does not rewrite 41,000+ existing grammar/math/reality stars. Those retain their current shape. Sleeptime consolidation upgrades them progressively.
- Does not introduce per-language Word Galaxy partitioning in VRAM (one Galaxy table serves all). Language is a field on the word star, not a separate VRAM region.
- Does not wire Grammar Galaxy parsing for word-form queries (`"two plus three"`). That is Phase 7+ and requires the grammar rule stars from 7.B to be cosine-navigable.
- Does not change the sovereign execution path. Phase 6.C's HANDLING_QUERY wiring is stable.

## Quality gates

- Canonical DB: `k3d_canonical` reachable from `CanonicalLookup`; sovereignty grep finds zero `qdrant_client` imports under `cranium/` or `knowledgeverse/`
- Build-time: zero dangling `word_ref` or `char_refs` in any meaning star's `surface_forms`
- Bidirectional: every `_link()` pair resolves in both directions
- Saudades round-trip: `concept_saudades` carries `untranslatable_languages=["en","ja",...]`; en-side periphrastic grammar template synthesizes a phrase via the 5-stage pipeline (no fake single-word translation)
- Cosine: `"cat"` → `word_en_cat` rank ≤ 20 at tier_64; `"gato"` → `word_pt_gato` rank ≤ 20
- Cross-language: `word_pt_gato.taxonomy_refs` → `concept_cat` → `surface_forms.en.word_ref` → `word_en_cat` — full round-trip resolves
- Character sharing: `char_a.composite_of` contains words from ≥ 2 languages
- Layer-0: every new star carries `lod_class`; visible artifacts resolve through a `drawing_primitive` registered in `k3d_canonical`
- Regression: Phase 1–6.C suite stays green

## The principle

*One concept, one star. All languages are surface-form references pointing at that star. All references are bidirectional. Characters are shared across languages by construction. The meaning IS the center.*

This is not aspirational — it's what the spec says and what the substrate now supports. Phase 7 makes the rest of the pipeline match what `star_crafter.py` already does.
