# CLAUDE → CODEX — Phase 7.A.1: Canonical Knowledge Seed (Letters, Math, Grammar, Words) — 2026-04-11

## Context

Phase 7.0 landed (canonical Qdrant registry, 54 entries, `CanonicalLookup` live) and Phase 7.A's first slice migrated the multilingual meaning builder onto canonical IDs (28 tests green). This spec covers the next push: **document what's there, then fill the canonical shelves with real knowledge** — every letter, every math symbol, every grammar rule, every word for every language with glyphs.

Source of record: [TEMP/CLAUDE_MEANING_FIRST_LANGUAGE_AS_RPN_VISION_2026-04-11.md](CLAUDE_MEANING_FIRST_LANGUAGE_AS_RPN_VISION_2026-04-11.md). Source of prior slice detail: [TEMP/CLAUDE_PHASE7_PROCEDURALIZATION_QUALITY_AND_MULTILINGUAL_SYMLINKS_2026-04-11.md](CLAUDE_PHASE7_PROCEDURALIZATION_QUALITY_AND_MULTILINGUAL_SYMLINKS_2026-04-11.md).

Ollama specialists are currently offline, so the documentation pass falls to Codex. Write it into the repo.

---

## What Codex already landed (verified by audit)

- [knowledge3d/ingestion/canonical_lookup.py](../knowledge3d/ingestion/canonical_lookup.py) — `canonical_slug`, `canonical_char_star_id`, `canonical_word_star_id`, `canonical_grammar_template_id`, `canonical_drawing_primitive_id`, `canonical_entry_id`, `CanonicalLookup` (Qdrant overlay, no deterministic fallback — fail and fix)
- [knowledge3d/ingestion/qdrant_credentials.py](../knowledge3d/ingestion/qdrant_credentials.py) — file-backed credential resolver (`/K3D/Knowledge3D.local/secrets/qdrant_api_key.txt`)
- [knowledge3d/ingestion/symlink_helpers.py](../knowledge3d/ingestion/symlink_helpers.py) — `append_ref()` and `link()` helpers enforcing bidirectional pair updates over `taxonomy_refs`, `meta_refs`, `grammar_refs`, `component_refs`, `composite_of`, and `surface_forms.{lang}.word_ref/char_refs`
- [scripts/ingest_canonical_to_qdrant.py](../scripts/ingest_canonical_to_qdrant.py) — seeds 54 canonical entries (26 ASCII letters + 10 digits + 3 drawing primitives + 4 grammar templates + 6 meaning classes + 5 symlink kinds)
- [knowledge3d/ingestion/universal_knowledge/multilingual_meanings.py](../knowledge3d/ingestion/universal_knowledge/multilingual_meanings.py) — migrated to `canonical_char_star_id` / `canonical_word_star_id` (via `_lemma_word_ref` / `_char_refs` thin wrappers); supports `untranslatable_languages` and `lod_class` fields
- [knowledge3d/knowledgeverse/_house_utils.py](../knowledge3d/knowledgeverse/_house_utils.py), [knowledge3d/knowledgeverse/foundational_operations_bootstrap.py](../knowledge3d/knowledgeverse/foundational_operations_bootstrap.py) — consume canonical IDs
- Tests: [tests/test_phase7_multilingual_symlinks.py](../tests/test_phase7_multilingual_symlinks.py), [tests/test_math_symbol_symlinks.py](../tests/test_math_symbol_symlinks.py), [tests/test_multilingual_meanings.py](../tests/test_multilingual_meanings.py) (28 passing)

---

## Slice 0 — Documentation pass (Codex owns, lands FIRST)

The Phase 7.0 + 7.A first slice is in the tree but not described anywhere a future agent can find without re-reading the diff. Fix that.

**Deliverable:** `docs/vocabulary/CANONICAL_REGISTRY_SPECIFICATION.md` (new file, same shape as existing docs/vocabulary specs), covering:

1. **Purpose** — why `k3d_canonical` exists, the sovereignty boundary (ingestion-path only, never imported from `cranium/` or `knowledgeverse/`)
2. **Kinds** — table of every `kind` the collection accepts (`star_id`, `drawing_primitive`, `grammar_template`, `meaning_class`, `symlink_kind`, plus kinds added later in this phase: `math_symbol`, `letter_star`, `font_glyph`, `word_lemma`, `grammar_rule`) with payload schemas
3. **ID format functions** — the full contract of `canonical_slug`, `canonical_char_star_id`, `canonical_word_star_id`, `canonical_grammar_template_id`, `canonical_drawing_primitive_id`, `canonical_entry_id`, with examples for each
4. **Credential resolution** — how `qdrant_credentials.resolve_qdrant_api_key()` works, where the secret file lives, env override
5. **CanonicalLookup API** — `ensure_collection`, `find_star_id(kind=, key=)`, `register(kind=, key=, star_id=, metadata=)`, KeyError semantics ("canonical_lookup_miss:{kind}:{key}"), no-fallback principle
6. **Symlink helpers** — `append_ref`, `link`, the 10 symlink kinds they enforce, worked examples for meaning→word→char round-trip
7. **Bootstrap script** — what `scripts/ingest_canonical_to_qdrant.py` seeds, how to re-run safely
8. **Cross-references** — link to MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md, DUAL_CLIENT_CONTRACT_SPECIFICATION.md, FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md, RPN_DOMAIN_OPCODE_REGISTRY.md
9. **Registration** — add this spec to `docs/vocabulary/README.md` and re-run `python scripts/ingest_specs_to_qdrant.py` so `k3d_specifications` indexes it

Plus: a short completion report at `TEMP/CODEX_PHASE7_CANONICAL_REGISTRY_LANDED_2026-04-11.md` (≤200 lines) — what shipped, what's green, what's next. Future agents read TEMP/ for recent context.

**Do NOT** generate changelogs, tutorials, or migration guides. One authoritative spec + one completion report.

---

## Slice 1 — Font source decision and glyph RPN

**Font source: system-installed open-source fonts (Daniel's directive: use system fonts).**

The build host already has **2,415 font families** (2,918 TTF/OTF files) installed under `/usr/share/fonts/`. This includes broad script coverage out of the box:

| Script | Families (examples) |
|--------|---------------------|
| Latin | Liberation, DejaVu, Roboto, Ubuntu, Source Sans Pro, URW base35, Linux Libertine, Cantarell |
| CJK | AR PL UKai/UMing (zh), IPA Gothic/Mincho (ja) |
| Arabic | Amiri, Amiri Quran |
| Devanagari | Samyak Devanagari, Lohit Devanagari |
| Georgian | BPG family (15+ variants) |
| Greek | GFS Porson, Linux Libertine (partial) |
| Cyrillic | DejaVu, Liberation, Roboto, Ubuntu, many Uralic variants |
| Thai | Purisa, Sawasdee, Tlwg, Umpush, Waree, Garuda, Norasi |
| Tamil | Samyak Tamil |
| Gujarati | Samyak Gujarati |
| Malayalam | Samyak Malayalam |
| Tibetan | Tibetan Machine Uni |
| Math | STIX Two Math, TeX Gyre Math family |

**No download step.** The ingestion-path module scans `/usr/share/fonts/` via `fontTools.ttLib` (or `fc-list` as a fast pre-filter), reads each font file's `cmap` table to discover which codepoints it covers, and generates `/K3D/Knowledge3D.local/assets/fonts/text/MANIFEST.json` listing each file with: family, style (regular/bold/italic/…), scripts covered, codepoint ranges.

**Symbol-font exclusion list (hardcoded in the scanner):** Wingdings, Webdings, FontAwesome, PowerlineSymbols, StandardSymbolsPS, D050000L, Symbol, Symbola, POV-Ray Logo, stmary10, wasy10, rsfs10, Dingbats, any family containing "Emoji". This is Daniel's explicit constraint: letter stars carry text font drawings only. STIX Two Math is KEPT (it contains real glyph outlines for math characters, not symbol pictograms).

**Glyph extraction:** new ingestion-path module `knowledge3d/ingestion/fonts/glyph_to_rpn.py`. For each `(font_file, codepoint)` pair:

1. Load glyph outline via `fontTools.ttLib` → `glyph.getCoordinates()`
2. Decompose contours into segments: `moveTo(x,y)`, `lineTo(x,y)`, `qCurveTo(cx,cy,x,y)`, `closePath()`
3. Emit RPN bytecode using existing Drawing Galaxy opcodes. Reuse the same LINE / CIRCLE / RECT primitive family — do NOT add new opcodes (Programs Before Opcodes, per RPN_DOMAIN_OPCODE_REGISTRY §1). A glyph is a sequence of `LINE` / `CURVE` subroutines wrapped in `CONTOUR_OPEN` / `CONTOUR_CLOSE` markers composed from existing opcodes; if a subroutine the compiler needs does not already exist, register it as a `rpn_template` entry in `k3d_canonical` (kind=`rpn_template`) and reference by ID. DO NOT invent new opcode numbers.
4. Normalize coordinates to the font's em-square (2048 units standard), keep as int16 for bytecode compactness
5. Return `(rpn_bytes, glyph_metrics)` where `glyph_metrics = {advance_width, lsb, xmin, ymin, xmax, ymax}`

**Canonical registration:** every extracted glyph becomes a `font_glyph` canonical entry:
- `kind="font_glyph"`
- `key=f"{font_family}::{style}::U+{codepoint:04X}"` (e.g., `"Noto Sans::Regular::U+0061"`)
- `star_id=f"font_glyph_{canonical_slug(family)}_{canonical_slug(style)}_u{codepoint:04x}"`
- `metadata={script, em_square, advance_width, contour_count, opcode_count}`

---

## Slice 2 — Letter stars: one star per letter, ALL text fonts inside

**Daniel's directive, verbatim:** *"The star for each letter must contain also all fonts drawings of said letter (include all letter a — do not include symbol fonts here, just text) as RPN, as well as symlinkage to math where's due (numbers, operations and so on) — all bi-directional links."*

**Implementation:** new module `knowledge3d/ingestion/letter_galaxy_builder.py`.

For every codepoint the MANIFEST.json identifies as a "letter" (Unicode category `L*` — `Lu`/`Ll`/`Lt`/`Lm`/`Lo`), build one canonical letter star:

```python
{
    "star_id": canonical_char_star_id(char),         # e.g. "char_a" or "char_u00e3"
    "meaning_class": "form",
    "domain": f"Character/{script}",                  # "Character/latn", "Character/hani", ...
    "name": f"letter '{char}'",
    "meaning_rpn": f"CHAR U+{codepoint:04X} GLYPH {script} {case}",  # declarative form
    "codepoint": codepoint,
    "script": script,
    "unicode_category": category,                     # Lu/Ll/Lt/Lm/Lo
    "unicode_name": unicodedata.name(char, ""),
    "languages": [<list of languages using this script>],
    "font_glyphs": [
        {
            "family": "DejaVu Sans",
            "style": "Regular",
            "font_glyph_star_id": "font_glyph_dejavu_sans_regular_u0061",  # symlink target
            "rpn_program_ref": "<meta_rule_addr pointer>",                  # VRAM program table
        },
        {
            "family": "Liberation Serif",
            "style": "Regular",
            "font_glyph_star_id": "font_glyph_liberation_serif_regular_u0061",
            "rpn_program_ref": "<meta_rule_addr pointer>",
        },
        # ... one entry per (family, style) from system fonts covering this codepoint
    ],
    "component_refs": [],                              # letters have no smaller components
    "composite_of": [],                                # filled as words register (back-link)
    "taxonomy_refs": [],                               # filled by math/digit cross-links below
    "meta_refs": [],
    "grammar_refs": [],
    "selection_role": "unknown",
    "answer_eligible": False,
    "lod_class": "LOD_ICON",                           # letters are icon-level knowledge
    "untranslatable_languages": [],                    # letters are form, not meaning
}
```

**Bidirectional math symlinks:** for every letter that has a mathematical role, wire both sides via `symlink_helpers.link()`:

- Digits `0-9` → math concept digit stars (e.g., `char_0` ↔ `concept_digit_zero`, `char_1` ↔ `concept_digit_one`, …). Forward `taxonomy_refs` on `char_0`, backward `component_refs` on `concept_digit_zero`.
- Greek letters → math variable/constant roles where they're customary (`α` alpha, `β` beta, `π` pi, `θ` theta, `λ` lambda, `μ` mu, `σ` sigma, `Σ` sigma uppercase for sum, `Π` pi uppercase for product, `Δ` delta for difference, `∇` nabla, `∂` partial — these last three are handled in Slice 3 as math symbols, not letter stars). Link with a descriptive symlink kind (`mathematical_role`) — register it in the `symlink_kind` canonical kind first.
- Latin letters with common variable roles (`x`, `y`, `z` for unknowns; `i`, `j`, `k` for indices; `n`, `m` for counts) → link to `concept_variable_role_{name}` stars if those stars exist; otherwise defer and file a follow-up. Do NOT auto-create stars just to satisfy the link.

**Dangling-ref rule (unchanged from 7.A):** every registered letter star passes through the build-time check that every ref resolves. No string pointers to non-existent targets.

**Registration in `k3d_canonical`:** each letter star also lands a canonical entry:
- `kind="letter_star"`
- `key=f"U+{codepoint:04X}"`
- `star_id=<the canonical char id>`
- `metadata={script, languages, font_count}`

**Validation:**
- `char_a` has `font_glyphs` containing entries for every text font in MANIFEST.json covering `latn` script
- Every `font_glyph_star_id` in `char_a.font_glyphs` resolves to a registered `font_glyph` canonical entry
- `char_0.taxonomy_refs` contains `concept_digit_zero`; `concept_digit_zero.component_refs` contains `char_0` (round-trip asserted)
- No font entry under `font_glyphs` references a symbol-only font (grep MANIFEST.json at build time)

---

## Slice 3 — Math symbol stars

Daniel's ask: **all math symbols meaning + RPN**. The datasets folder has `math/train.jsonl` (LaTeX problem/solution pairs, 11 MB) — this is evidence, not a symbol table. Source the symbol inventory from Unicode directly:

**Seed list (ingestion-path script `knowledge3d/ingestion/math_symbol_builder.py`):**

1. Unicode blocks via stdlib `unicodedata`:
   - `U+2200–22FF` Mathematical Operators
   - `U+27C0–27EF` Miscellaneous Mathematical Symbols-A
   - `U+2980–29FF` Miscellaneous Mathematical Symbols-B
   - `U+2A00–2AFF` Supplemental Mathematical Operators
   - `U+1D400–1D7FF` Mathematical Alphanumeric Symbols
   - `U+2150–218F` Number Forms (fractions, Roman numerals)
   - `U+00B1` (±), `U+00D7` (×), `U+00F7` (÷), `U+221A` (√), `U+221E` (∞), `U+2260` (≠), `U+2264` (≤), `U+2265` (≥), `U+2208` (∈), `U+2209` (∉), `U+2282` (⊂), `U+2286` (⊆), `U+222B` (∫), `U+2211` (∑), `U+220F` (∏), `U+221B` (∛), `U+221C` (∜) — pin by codepoint
2. LaTeX → Unicode mapping curated from a single authoritative source (the Unicode math table at `https://www.unicode.org/Public/math/revision-15/MathClassEx-15.txt` — one-shot download into `/K3D/Knowledge3D.local/assets/math/MathClassEx-15.txt` at boot, parse offline thereafter). This gives every symbol its LaTeX command(s) (`\cos`, `\sum`, `\partial`, …), math class (N/A/B/C/D/F/G/L/O/P/R/U/V), operator spacing rules.
3. LaTeX commands that are NOT single-codepoint (e.g., `\frac{a}{b}`, `\binom{n}{k}`, `\sqrt[n]{x}`, `\sum_{i=1}^{n}`) — these are RPN TEMPLATES, not symbols. Register under `kind="rpn_template"` in `k3d_canonical` with their bytecode and argument shape; reference from math symbol stars that invoke them.

**Math symbol star shape:**

```python
{
    "star_id": f"math_symbol_{canonical_slug(unicode_name)}",  # "math_symbol_plus_sign", "math_symbol_n_ary_summation"
    "meaning_class": "concept",                                 # the symbol stands for a concept
    "domain": f"Math/{math_class}",                             # "Math/Binary", "Math/Large_Operator", ...
    "name": unicodedata.name(char),
    "meaning_rpn": <executable bytecode where possible>,
    "codepoint": codepoint,
    "unicode_char": char,
    "math_class": math_class,                                    # from MathClassEx-15
    "latex_commands": ["\\sum", "\\summation"],                  # all LaTeX surface forms
    "surface_forms": {                                            # language-invariant — math is universal
        "tex": {"word_ref": f"word_tex_{latex_primary}"},
        "unicode": {"word_ref": f"word_unicode_u{codepoint:04x}"},
    },
    "char_refs": [canonical_char_star_id(char)],                 # symlink to the letter/glyph star
    "taxonomy_refs": [],                                          # filled from parent math concept
    "meta_refs": [<rpn_template ids if template-bound>],
    "grammar_refs": [],
    "component_refs": [],
    "composite_of": [],
    "selection_role": "operator" | "operand" | "relation" | "delimiter",
    "answer_eligible": False,
    "lod_class": "LOD_SUMMARY",
    "meaning_program": <bytecode or None if value-bearing>,
}
```

**Executable `meaning_rpn` — tiering (per vision spec §3 and Phase 7 §7.D):**

- **Operator symbols** (`+`, `−`, `×`, `÷`, `∑`, `∏`, `∫`, `√`, …) — carry executable bytecode in the VRAM program table, `meta_rule_addr > 0`, dispatched by the fused tick. Reuse the arithmetic opcodes already live from Phase 6.C (the same path `math_operator_addition` uses). For `∑`, `∏`, `∫`, `√`: either register a `rpn_template` and reference, or defer to Slice 4's grammar layer. Do not block on deferred templates — mark them `lod_class="LOD_ICON"` and file follow-ups per-symbol.
- **Relational symbols** (`=`, `<`, `>`, `≤`, `≥`, `≠`, `∈`, `⊂`, …) — executable comparator bytecode using existing `TCOMP` / comparison opcodes
- **Constants** (`π`, `e`, `∞`, `ℝ`, `ℕ`, `ℂ`, …) — value-bearing, `meaning_rpn` = RPN `STORE` of the constant value or set-theoretic marker
- **Delimiters** (`(`, `)`, `[`, `]`, `{`, `}`, `⟨`, `⟩`, `⌊`, `⌋`, …) — form-only, no `meta_rule_addr`; `meaning_rpn` = `"DELIMITER OPEN"` / `"DELIMITER CLOSE"` style declarative form

**Bidirectional char ↔ math links:** every math symbol star that IS a single character (most of them) back-links to its `char_u{codepoint:04x}` letter star via `link(math_symbol_star, letter_star, "char_refs", "mathematical_role")`. Letter stars that have a mathematical role accumulate these links via the reverse edge.

**Canonical registration:** `kind="math_symbol"`, `key=f"U+{codepoint:04X}"`, `star_id=<math_symbol_star_id>`, `metadata={latex_commands, math_class, has_executable_program}`.

**Validation:**
- `math_symbol_plus_sign.char_refs` contains `char_u002b`; `char_u002b.taxonomy_refs` (mathematical_role edge) contains `math_symbol_plus_sign`
- `math_symbol_n_ary_summation.latex_commands` contains `"\\sum"`
- For every symbol with `math_class == "O" (Large Operator)`, either `meta_rule_addr > 0` OR a follow-up is filed
- Cosine probe: `"plus"` → `math_symbol_plus_sign` rank ≤ 20 at tier_64; `"\sum"` → `math_symbol_n_ary_summation` rank ≤ 20

---

## Slice 4 — Grammar Galaxy from Universal Dependencies

The datasets folder has `/K3D/K3D_llama_cpp/datasets/ud/` — Universal Dependencies v2.14, 284 treebanks, ~80 languages, CONLLU format. This is the grammar source.

**Ingestion-path module:** `knowledge3d/ingestion/grammar/ud_grammar_builder.py`.

For each treebank:

1. Parse CONLLU with stdlib (pyconll or conllu on ingestion path is fine — they're pure Python, small, no GPU dependency)
2. Extract per-sentence: lemma, UPOS (universal part of speech), XPOS, feats (morphological features), head index, deprel (dependency relation)
3. Aggregate into language-level rule templates:
   - **Constituent templates** — NP, VP, AP, PP structures derived from deprel patterns (nsubj, obj, amod, det, case, …). One `grammar_template` canonical entry per (language, construction).
   - **Morphology tables** — per-lemma inflection paradigms. Verb conjugation (tense × mood × person × number), noun declension (case × number × gender), adjective agreement. Stored as grammar RULE stars (not templates) with `meaning_rpn` dispatching to existing RPN opcodes (STORE/RECALL/OP_BRANCH) — do NOT add language-specific opcodes.
   - **Word order rules** — language-level SVO/SOV/VSO markers encoded as meta-rule stars
4. Each rule/template star gets:
   - `domain=f"Grammar/{language}"`
   - `meaning_class="rule"` (templates) or `"meta"` (word-order rules)
   - `meaning_rpn` = executable bytecode composing one of the 5 synthesis stages (MEANING DECOMPOSE / GRAMMAR SELECT / WORD RESOLVE / MORPHOLOGY APPLY / SURFACE EMIT) per vision spec §3.1
   - Bidirectional links: template star ↔ every word star it governs (`grammar_refs` forward, `taxonomy_refs` backward)
5. **Periphrastic templates for saudades rule:** for every language in the MANIFEST, register at least one `grammar_template_{lang}_periphrastic_explanation` that the multilingual synthesis pipeline falls back to for untranslatable concepts. English already seeded in Phase 7.0 (`grammar_template_en_periphrastic_explanation`, `grammar_template_en_copula`); extend Portuguese and Japanese (already seeded with copula, add periphrastic), and add the same pair for every language with a UD treebank that K3D ingests.

**Rule budget:** do NOT try to load all 284 treebanks. Start with languages K3D already has glyph coverage for (check MANIFEST.json from Slice 1): English (`en`), Portuguese (`pt`), Spanish (`es`), French (`fr`), German (`de`), Italian (`it`), Japanese (`ja`), Chinese (`zh`), Russian (`ru`). For each, pick the canonical treebank (e.g., `UD_English-EWT`, `UD_Portuguese-Bosque`, `UD_Japanese-GSD`, `UD_Chinese-GSD`). Nine languages, nine treebanks, ~10k sentences each aggregated into ~50–200 templates per language. That's tractable for a first pass.

**Canonical registration:** `kind="grammar_rule"` for inflection/word-order rules, `kind="grammar_template"` for constituent templates (already defined in Phase 7.0 schema).

**Validation:**
- For every language in MANIFEST, `CanonicalLookup.find_star_id(kind="grammar_template", key=f"{lang}:copula")` resolves
- For every language, `CanonicalLookup.find_star_id(kind="grammar_template", key=f"{lang}:periphrastic_explanation")` resolves
- Portuguese saudades round-trip: a query for the meaning in English invokes `grammar_template_en_periphrastic_explanation` and synthesizes a phrase (no fake single-word translation)
- Every grammar rule/template star has `meaning_rpn` composed from existing RPN opcodes — grep the bytecode generator for new opcode numbers, fail if any appear

---

## Slice 5 — Word Galaxy population

The datasets folder provides three word sources:

1. **OMW** (`/K3D/K3D_llama_cpp/datasets/omw-data/`) — 32 language wordnet TSVs, already consumed by `multilingual_meanings.py`. Keep using it as the primary meaning-centric synset source.
2. **DBnary** (`/K3D/K3D_llama_cpp/datasets/dbnary/`) — 28 OntolexLemon RDF dumps (TTL.bz2). Richer morphological and etymological data than OMW but bulkier. Ingestion-path module `knowledge3d/ingestion/universal_knowledge/dbnary_ingester.py` parses one language at a time, extracts lemma + pronunciation + lexical entry type, and **merges into existing meaning stars** rather than creating new ones (match on OMW synset where possible, fall back to lemma-level word star creation for lemmas OMW missed).
3. **Kaikki.org dictionaries** (Portuguese, Spanish, etc. under `/lexicons/`) and **Wiktionary-derived** — single JSONL files, usable as supplementary lemma sources.

**Rule:** every word star lands through `canonical_word_star_id(language, lemma)`. Every word star's `component_refs` resolve to existing letter stars (Slice 2 provides them). Every word star bi-directionally links to its meaning star(s) per Phase 7.B's `link()` pattern. Every word star carries `meaning_rpn` — at minimum a declarative form like `"WORD {lang} {lemma} REF {meaning_star_id}"`, ideally structured per vision spec §3.

**Pronunciation symlinks (bonus, optional for this slice):** the datasets folder has CMUdict (English phonemes), audio corpora per language, and phonetic data in dbnary. If time allows, word stars get an `audio_refs` field pointing to a pronunciation RPN program (IPA string → spectrogram template). Deferrable — file a follow-up if not done in this slice.

**Canonical registration:** `kind="word_lemma"`, `key=f"{language}:{lemma}"`, `star_id=canonical_word_star_id(language, lemma)`, `metadata={pos, synset_ids, char_count}`.

**Validation:**
- OMW ingestion through the new canonical path: `word_pt_gato.taxonomy_refs` → synset star → `surface_forms.en.word_ref` → `word_en_cat` (round-trip from Phase 7.B must still pass)
- DBnary merge adds etymology to at least 1000 existing OMW synsets per language
- For every language in MANIFEST, at least 5000 word stars registered (floor, not ceiling — no knowledge caps per Daniel)
- Every word star has component_refs resolving to letter stars; dangling-pointer gate stays clean

---

## Sovereignty rules (all slices)

- **Ingestion-path-only imports:** `fontTools`, `freetype-py`, `pyconll`/`conllu`, `unicodedata`, `rdflib` (for dbnary TTL parsing), `fastembed`, `qdrant_client`. NONE of these may appear under `knowledge3d/cranium/` or `knowledge3d/knowledgeverse/`. Grep gate at CI.
- **No new RPN opcodes.** Everything composes from existing opcodes. If a slice needs a pattern that seems like a new opcode, it's an `rpn_template` registered in `k3d_canonical` — a COMPOSITE of existing opcodes with a fixed argument shape.
- **No knowledge caps** (per Daniel, March 2026 memory). Quality filters OK (dedup, stopword skip, UTF-8 validation, Unicode category filter for letter vs symbol). Quantity caps never.
- **No Python fallbacks in hot path.** Ingestion runs once and populates the canonical registry + Galaxy; after that the sovereign runtime uses Galaxy navigation only. `CanonicalLookup` miss throws — fail and fix.
- **`MEMORY.md` cap preserved** — do not append session logs to it. Move any new pointers into existing memory topic files.

---

## Quality gates (expanded from Phase 7)

- Canonical Qdrant DB: `k3d_canonical` grew from 54 entries to ≥ (26 letters × 10 fonts + 300 math symbols + 50 grammar templates × 9 languages + word budget) = thousands of entries; exact count reported post-run
- Font coverage: every letter star with `script="latn"` has ≥ 5 `font_glyphs` entries from system-installed text fonts (e.g., DejaVu Sans, Liberation Sans, Liberation Serif, Ubuntu, Roboto)
- Symbol fonts excluded: grep MANIFEST.json for families containing `Symbola`, `Wingdings`, `Emoji`, `Dingbats`, `MathJax Symbols` — must be empty
- Bidirectional: every new link passes through `symlink_helpers.link()`; one-way string refs fail CI
- Math round-trip: `char_u002b` ↔ `math_symbol_plus_sign`; `char_u03c0` ↔ `math_symbol_greek_small_letter_pi` (where applicable)
- Grammar round-trip: Portuguese→English saudades periphrasis test (Phase 7.B gate, extended)
- Cosine probes (tier_64): `"plus"` rank ≤ 20, `"sum"` rank ≤ 20, `"cat"` rank ≤ 20, `"gato"` rank ≤ 20, `"猫"` rank ≤ 20
- Regression: Phase 1–7.B suite stays green (currently 28 phase-7 tests + prior)

---

## Order of operations

1. **Slice 0** — Codex writes `docs/vocabulary/CANONICAL_REGISTRY_SPECIFICATION.md` + completion report. Register in `README.md`. Re-ingest specs. **Land and commit before starting Slice 1.**
2. **Slice 1** — Scan system fonts at `/usr/share/fonts/` via `fontTools`; generate `MANIFEST.json` at `/K3D/Knowledge3D.local/assets/fonts/text/`; implement `glyph_to_rpn.py`; unit tests for 10 glyph extractions (Latin `a`, `ã`, `Я`, `π`, `漢`, `あ`, `ع`, `ह`, `ㄱ`, `ñ`); register as `font_glyph` canonical entries.
3. **Slice 2** — Letter Galaxy builder; every Unicode letter category codepoint the MANIFEST covers gets a star with all font_glyph symlinks; bidirectional digit ↔ concept links; unit tests for `char_a`, `char_0`, `char_u03c0`.
4. **Slice 3** — Math symbol builder from Unicode blocks + MathClassEx-15 download; operator stars carry executable `meaning_rpn`; symbol ↔ char bidirectional links; cosine probe tests.
5. **Slice 4** — UD treebank ingester for 9 languages; grammar templates + morphology rules + periphrastic templates; saudades round-trip test (extends Phase 7.B gate).
6. **Slice 5** — OMW + DBnary + Kaikki word lemma ingestion through canonical path; component_refs resolve to letter stars; per-language word budget tests.
7. **Commit gate** — after each slice, green tests, clean `git diff --check`, one commit per slice with detail in the body. Codex decides whether to squash at end.

---

## What this does NOT do

- Does not build Drawing Galaxy instance buffers (that's Phase 7.E — Layer-0 instancing audit)
- Does not migrate Phase 6.C's math operator stars to the new math_symbol schema — those already work, file a follow-up for alignment
- Does not attempt to load all 284 UD treebanks (9 languages only, named above)
- Does not train a learned projection (sleeptime territory)
- Does not wire grammar templates into the fused tick — they're read by the proceduralizer, not the hot path, until Phase 8

---

## One principle

*Every letter, every symbol, every word, every grammar rule lives as a star in Galaxy Universe with a canonical ID in `k3d_canonical`, bidirectional symlinks to everything it composes with, and `meaning_rpn` composed from existing RPN opcodes. Nothing is a string; nothing is a Python lookup table; nothing is duplicated. The canonical registry is the librarian, Galaxy Universe is the shelf, RPN bytecode is the reader.*

Report per slice. Raw canonical-entry counts, test deltas, and any dangling-pointer failures beat passing CI at this stage — the shape of the knowledge matters more than the count.
