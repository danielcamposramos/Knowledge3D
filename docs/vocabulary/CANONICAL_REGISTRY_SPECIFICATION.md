# Canonical Registry Specification

**Version**: 1.0  
**Status**: Implementation Specification (Phase 7.0 / 7.A.1)  
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)  
**Date**: April 11, 2026

---

## Abstract

This specification defines the **Canonical Registry** for K3D: the `k3d_canonical` Qdrant collection plus the deterministic ID functions and helper APIs that keep stars, glyphs, grammar templates, and symlink surfaces from drifting into duplicate naming schemes.

The registry is the **ingestion-side librarian**. It does not reason. It does not execute the hot path. It exists so that ingestion modules, proceduralizers, and build scripts can agree on one stable identifier for each canonical object before that object is placed into the House or loaded into the Galaxy.

This specification covers:

- Why `k3d_canonical` exists
- Which `kind` values the registry accepts
- The contract of each deterministic ID helper
- Credential resolution for Qdrant access
- `CanonicalLookup` API semantics
- Bidirectional symlink helper behavior
- The bootstrap seed script and its seeded rows

---

## 1. Purpose

### 1.1 Why `k3d_canonical` Exists

K3D is meaning-first, not string-first. The same concept, glyph, grammar template, or word lemma MUST NOT be recreated with competing ad-hoc IDs by different ingestion passes.

The canonical registry exists to provide:

- **Stable canonical IDs** for repeated knowledge surfaces
- **Deterministic reconstruction** across reruns and machines
- **Build-time deduplication** before knowledge enters House or Galaxy
- **Reference-preserving ingestion** so higher layers point to lower layers instead of restating them

Without `k3d_canonical`, every ingestion pass risks inventing duplicate names such as:

- `en_able` vs `word_en_able`
- `char_pt_u00e3` vs `char_u00e3`
- `copula_en` vs `grammar_template_en_copula`

The registry prevents that drift.

### 1.2 Sovereignty Boundary

The canonical registry is **ingestion-path infrastructure only**.

Allowed:

- `knowledge3d/ingestion/*`
- build scripts under `scripts/`
- offline proceduralization and content registration

Forbidden:

- imports from `knowledge3d/cranium/`
- imports inside PTX/CUDA hot-path logic
- runtime reasoning dispatch
- Python fallback routing in active inference

The sovereign runtime consumes Galaxy stars and VRAM tables. It does not query Qdrant.

---

## 2. Collection Contract

### 2.1 Collection Name

- `collection_name = "k3d_canonical"`

### 2.2 Vector Contract

- `vector_name = "fast-all-minilm-l6-v2"`
- `vector_size = 384`
- `distance = cosine`

The named vector matches the MCP/Qdrant FastEmbed expectation so both direct code and MCP tooling address the same vector slot.

### 2.3 Payload Contract

Every canonical point MUST carry:

```json
{
  "kind": "string",
  "key": "string",
  "star_id": "string",
  "text": "string",
  "document": "string",
  "metadata": {}
}
```

`text` and `document` are duplicated intentionally because downstream MCP/Qdrant tooling expects one or both.

---

## 3. Registry Kinds

The canonical registry accepts the following `kind` values.

| kind | Purpose | Key Shape | star_id Shape | Metadata Shape |
| --- | --- | --- | --- | --- |
| `star_id` | Canonical character-like or atomic star IDs | `char::<glyph>` | `char_a`, `char_0`, `char_u00e3` | optional |
| `drawing_primitive` | Stable IDs for procedural drawing primitives | `line`, `circle`, `rect` | `drawing_primitive_line` | optional |
| `grammar_template` | Stable grammar-template IDs | `{lang}:{template}` | `grammar_template_en_copula` | optional |
| `meaning_class` | Registry of allowed meaning-layer classes | `concept`, `relation`, ... | same as key | optional |
| `symlink_kind` | Registry of allowed bidirectional link categories | `taxonomy_refs`, `component_refs`, ... | same as key | optional |
| `math_symbol` | Canonical math symbol stars | `U+2211` | `math_symbol_n_ary_summation` | `latex_commands`, `math_class`, `has_executable_program` |
| `letter_star` | Canonical letter stars | `U+0061` | `char_a` | `script`, `languages`, `font_count` |
| `font_glyph` | Per-font glyph programs | `Noto Sans::Regular::U+0061` | `font_glyph_noto_sans_regular_u0061` | `script`, `em_square`, `advance_width`, `contour_count`, `opcode_count` |
| `word_lemma` | Canonical word lemma stars | `{lang}:{lemma}` | `word_en_cat` | `pos`, `synset_ids`, `char_count` |
| `grammar_rule` | Executable grammar or morphology rule stars | language-specific rule key | rule star id | construction, language, stage metadata |
| `rpn_template` | Named composite RPN templates made from existing opcodes | template key | template star id | `arg_shape`, `opcode_count` |

The first five kinds are live in Phase 7.0. The remaining kinds are reserved by this specification for Phase 7.A.1 and later slices.

---

## 4. Deterministic ID Functions

All deterministic ID helpers live in [canonical_lookup.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/ingestion/canonical_lookup.py).

### 4.1 `canonical_slug(text)`

Purpose:

- Normalize free text into a deterministic ASCII-safe slug
- Preserve non-Latin content by encoding alphanumeric Unicode codepoints as `uXXXX`

Rules:

- Trim and lowercase input
- Keep ASCII letters and digits as-is
- Convert separators (`space`, `-`, `_`, `/`, `.`, `:`) to underscore boundaries
- Convert non-ASCII alphanumeric characters to `uXXXX`
- Collapse repeated underscores
- Empty result becomes `unknown`

Examples:

- `canonical_slug("Living Room") -> "living_room"`
- `canonical_slug("Sala de Estar") -> "sala_de_estar"`
- `canonical_slug("ação") -> "au00e7u00e3o"`
- `canonical_slug("図書館") -> "u56f3u66f8u9928"`

### 4.2 `canonical_char_star_id(char)`

Purpose:

- Produce the canonical star ID for a single codepoint

Rules:

- Input MUST be exactly one codepoint
- ASCII alnum uses `char_<lower>`
- Everything else uses `char_uXXXX`

Examples:

- `canonical_char_star_id("a") -> "char_a"`
- `canonical_char_star_id("0") -> "char_0"`
- `canonical_char_star_id("ã") -> "char_u00e3"`
- `canonical_char_star_id("π") -> "char_u03c0"`

### 4.3 `canonical_word_star_id(language, lemma)`

Purpose:

- Produce the canonical word-lemma star ID

Format:

- `word_{language}_{canonical_slug(lemma)}`

Examples:

- `canonical_word_star_id("en", "cat") -> "word_en_cat"`
- `canonical_word_star_id("pt", "gato") -> "word_pt_gato"`
- `canonical_word_star_id("ja", "図書館") -> "word_ja_u56f3u66f8u9928"`

### 4.4 `canonical_grammar_template_id(language, template_name)`

Format:

- `grammar_template_{language}_{canonical_slug(template_name)}`

Examples:

- `canonical_grammar_template_id("en", "copula") -> "grammar_template_en_copula"`
- `canonical_grammar_template_id("en", "periphrastic explanation") -> "grammar_template_en_periphrastic_explanation"`

### 4.5 `canonical_drawing_primitive_id(primitive_name)`

Format:

- `drawing_primitive_{canonical_slug(primitive_name)}`

Examples:

- `canonical_drawing_primitive_id("line") -> "drawing_primitive_line"`
- `canonical_drawing_primitive_id("rounded rect") -> "drawing_primitive_rounded_rect"`

### 4.6 `canonical_entry_id(kind, key)`

Purpose:

- Produce the deterministic Qdrant point identifier for a canonical row

Format:

- `uuid5(NAMESPACE_URL, "{kind}::{key}")`

Properties:

- Deterministic across reruns
- Valid Qdrant point ID
- Different kinds can reuse the same key string safely because `kind` is part of the UUID payload

Example:

- `canonical_entry_id("star_id", "char::a") -> deterministic UUID`

---

## 5. Credential Resolution

Credential resolution lives in [qdrant_credentials.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/ingestion/qdrant_credentials.py).

### 5.1 Resolution Order

`resolve_qdrant_api_key()` resolves the Qdrant API key in this order:

1. `QDRANT_API_KEY` environment variable
2. `/K3D/Knowledge3D.local/secrets/qdrant_api_key.txt`
3. Raise `RuntimeError`

### 5.2 Secret File

- Path: `/K3D/Knowledge3D.local/secrets/qdrant_api_key.txt`
- This file is local-only and MUST NOT be committed
- Repo code points to the file; it does not embed the secret literal

### 5.3 Failure Semantics

If neither env nor file is available:

```text
RuntimeError: QDRANT_API_KEY not set and /K3D/Knowledge3D.local/secrets/qdrant_api_key.txt not found
```

This is intentional. The canonical registry is fail-and-fix infrastructure.

---

## 6. `CanonicalLookup` API

`CanonicalLookup` is the strict overlay over `k3d_canonical`.

### 6.1 `ensure_collection()`

Behavior:

- Confirms `k3d_canonical` exists in Qdrant
- Raises `RuntimeError("canonical_collection_missing:k3d_canonical")` if absent

### 6.2 `find_star_id(kind=, key=)`

Behavior:

- Scrolls the collection for the exact `kind` + `key` payload pair
- Returns the stored `star_id`
- Raises:
  - `KeyError("canonical_lookup_miss:{kind}:{key}")` when the row is absent
  - `RuntimeError("canonical_lookup_missing_star_id:{kind}:{key}")` if payload is malformed

Examples:

- `find_star_id(kind="star_id", key="char::a") -> "char_a"`
- `find_star_id(kind="grammar_template", key="en:copula") -> "grammar_template_en_copula"`

### 6.3 `register(kind=, key=, star_id=, metadata=)`

Behavior:

- Ensures collection exists
- Embeds the registration text with FastEmbed
- Upserts a deterministic point into Qdrant
- Returns the registered `star_id`

Payload text shape:

```text
{kind}
{key}
{star_id}
```

### 6.4 No-Fallback Principle

`CanonicalLookup` deliberately does **not** synthesize local fallback IDs when a lookup misses. If the registry does not know an object, ingestion MUST register it or fail.

This is the registry’s core contract:

- no silent fallback
- no duplicate side registries
- no string invention in the caller

---

## 7. Symlink Helpers

Symlink helpers live in [symlink_helpers.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/ingestion/symlink_helpers.py).

### 7.1 `append_ref(star, field_path, ref_id)`

`append_ref()` mutates a `MeaningCentricStar` in-place while enforcing deduplicated list updates or field assignment.

Supported field paths:

1. `taxonomy_refs`
2. `meta_refs`
3. `grammar_refs`
4. `reality_refs`
5. `visual_refs`
6. `audio_refs`
7. `char_refs`
8. `component_refs`
9. `composite_of`
10. `surface_forms.{lang}.word_ref`
11. `surface_forms.{lang}.char_refs`
12. `mathematical_role` (alias to `taxonomy_refs` for reverse math/form links)

Unsupported field paths raise:

```text
ValueError: unsupported_symlink_field:{field_path}
```

### 7.2 `link(left, right, forward_kind, backward_kind)`

`link()` is the only bidirectional write helper. It applies:

- `append_ref(left, forward_kind, right.star_id)`
- `append_ref(right, backward_kind, left.star_id)`

This enforces symmetry at construction time.

### 7.3 Worked Example — Meaning → Word → Character

Meaning star:

- `synset_12345678_n`

Word star:

- `word_en_water`

Character star:

- `char_a`

Construction:

```python
link(meaning_star, word_star, "surface_forms.en.word_ref", "taxonomy_refs")
link(word_star, char_star, "component_refs", "composite_of")
```

Result:

- `meaning_star.surface_forms["en"].word_ref == "word_en_water"`
- `"synset_12345678_n" in word_star.taxonomy_refs`
- `"char_a" in word_star.component_refs`
- `"word_en_water" in char_a.composite_of`

This round-trip is the reference-preservation invariant made explicit.

---

## 8. Bootstrap Seed Script

Bootstrap seeding lives in [ingest_canonical_to_qdrant.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/scripts/ingest_canonical_to_qdrant.py).

### 8.1 What It Seeds

Phase 7.0 seed rows:

- 26 ASCII letters
- 10 digits
- 3 drawing primitives
- 18 grammar templates
- 6 meaning classes
- 6 symlink kinds

Total:

- `69 canonical entries`

### 8.2 Current Seeded Keys

Letters/digits:

- `kind="star_id"`
- `key="char::a"` through `key="char::z"`
- `key="char::0"` through `key="char::9"`

Drawing primitives:

- `line`
- `circle`
- `rect`

Grammar templates:

- `{lang}:copula` for `en`, `pt`, `es`, `fr`, `de`, `it`, `ja`, `zh`, `ru`
- `{lang}:periphrastic_explanation` for `en`, `pt`, `es`, `fr`, `de`, `it`, `ja`, `zh`, `ru`

Meaning classes:

- `concept`
- `relation`
- `action`
- `property`
- `meta`
- `form`

Symlink kinds:

- `taxonomy_refs`
- `meta_refs`
- `grammar_refs`
- `component_refs`
- `composite_of`
- `mathematical_role`

`mathematical_role` is an edge label used by Phase 7.A.1 for letter and math-symbol
relationships. The concrete bidirectional fields remain `taxonomy_refs` forward
and `component_refs` backward until typed edge records land.

### 8.3 Safe Re-run

Safe rerun command:

```bash
python3 scripts/ingest_canonical_to_qdrant.py
```

Behavior:

- recreates `k3d_canonical`
- rebuilds the 69 canonical seed rows
- upserts deterministic UUID point IDs

This script is safe to rerun because IDs are deterministic and the seed set is authoritative for the current phase.

---

## 9. Cross-References

This specification builds on:

- [MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md)
- [DUAL_CLIENT_CONTRACT_SPECIFICATION.md](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md)
- [FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md)
- [RPN_DOMAIN_OPCODE_REGISTRY.md](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md)

Related implementation files:

- [canonical_lookup.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/ingestion/canonical_lookup.py)
- [qdrant_credentials.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/ingestion/qdrant_credentials.py)
- [symlink_helpers.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/ingestion/symlink_helpers.py)
- [ingest_canonical_to_qdrant.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/scripts/ingest_canonical_to_qdrant.py)

---

## 10. Registration

This specification MUST be:

1. Listed in `docs/vocabulary/README.md`
2. Ingested into `k3d_specifications` via:

```bash
python3 scripts/ingest_specs_to_qdrant.py
```

Future additions to canonical kinds MUST update:

- this specification
- the bootstrap seed script if the kind is seed-level
- the vocabulary README

The registry is authoritative only if the documentation, seed scripts, and Qdrant contents agree.
