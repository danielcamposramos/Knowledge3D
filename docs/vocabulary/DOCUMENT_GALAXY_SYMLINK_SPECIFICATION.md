---
title: Document Galaxy as Symlinks to Words
authors: Daniel Campos Ramos (founding insight), Claude (architectural draft)
date: 2026-04-20
status: normative — v1
supersedes: none
companion_specs:
  - docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md
  - docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md
  - docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md
  - docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §11 (block 0x2D0-0x2DF)
  - TEMP/CLAUDE_TEXTURE_FORGE_IMAGE_TO_3D_ARC3_SCREEN_04.20.2026.md §10 (Memory-as-Image)
---

# Document Galaxy Specification — Documents as Symlinks to Word Stars

## 1. Core Principle

**A document is not stored as bytes + metadata. A document is a star whose content is an ordered list of symlinks to Word Galaxy stars.**

Downstream of that ordering, everything is already stored exactly once elsewhere in the Galaxy Universe:

```
Document star (0x2D0 OP_DOC_STAR_NEW)
  └── ordered list of Word Galaxy symlinks (OP_DOC_WORD_REF)
        └── each Word Galaxy star = ordered list of Character Galaxy symlinks
              └── each Character Galaxy star = { Drawing primitive + Font metadata + Meaning ref + Language tag }
                    └── Meaning star (language-agnostic — shared across all languages that express the same concept)
```

Documents own: title, author, structural metadata (paragraph breaks, style spans, tables), ordering of symlinks, and a content hash.

Documents do NOT own: characters, glyphs, font curves, pronunciations, language tags, meanings, or any text bytes.

## 2. Why This Matters — Storage, Semantics, Universality

### 2.1 Storage savings

A corpus of one million documents averaging 10,000 words each = 10^10 word occurrences. In a naïve store, that is 10^10 strings with per-word metadata.

In K3D: those 10^10 occurrences resolve to ~10^5 unique Word stars (Zipf's law — English vocabulary at web scale tops out in the low hundreds of thousands). Word → Character → Meaning symlinks compress further. The per-document storage approaches O(symlinks × log(corpus)) instead of O(characters × documents).

Practical impact: the word "the" stored ONCE; the character 'a' stored ONCE; the meaning of "walk" stored ONCE across all languages that name it.

### 2.2 Semantic properties

Because meanings are language-agnostic and live at the bottom of the symlink chain, every operation that traverses the chain can stop at any level:

- **Stop at Word level** → get language-surface text (English-only retrieval).
- **Stop at Character level** → get glyph shapes (typography, rendering).
- **Stop at Meaning level** → get concept (multilingual retrieval, cross-lingual reasoning).

Semantic search over the corpus operates at the Meaning layer. TRM navigation via semantic gravity (F = T(s₁,s₂) × M(s₁) × M(s₂) / d²) happens between Meaning stars — language drops out.

### 2.3 Universality / multilingual rendering

The same document star can render in any language by swapping the terminal font + language symlinks at evaluation time. `OP_DOC_RENDER_IN_LANG` (0x2D7) walks the document's symlink list, substitutes Meaning-star terminals with Word stars in the target language, and emits a glyph stream.

English source document → Portuguese output = free. Japanese output = free. Document author wrote once; readers receive in their language.

## 3. The Four Tiers of Document-Galaxy Symlinks

Per `FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md` layers (Form → Meaning → Rules → Meta-Rules):

| Tier | Layer | Star Kind | Example | Owned Metadata |
|---|---|---|---|---|
| T0 | Form | Drawing primitive | Bézier path for 'a' | segment list (RPN) |
| T1 | Form+Meaning | Character star | 'a' in Helvetica, English | ref to T0 + font_id + language + pronunciation IPA |
| T2 | Meaning | Word star | "walk" | ordered T1 refs + Meaning ref + POS tag |
| T3 | Meaning | Meaning star (language-agnostic) | concept(walk) | ref to Reality/Grammar stars + Matryoshka embedding |
| T4 | Rules | Document star | "A Tale of Two Cities" | ordered T2/T3 refs + structure + author |

Document star at T4 is a pure composition of lower-tier symlinks + document-specific metadata. No character bytes are stored inside it.

## 4. Opcode Surface (RPN block 0x2D0-0x2DF)

| Opcode | Mnemonic | Stack Contract | Purpose |
|---|---|---|---|
| 0x2D0 | OP_DOC_STAR_NEW | title_ref author_ref → doc_star_id | Create empty document star |
| 0x2D1 | OP_DOC_WORD_REF | doc_id word_star_id → doc_id | Append Word Galaxy symlink |
| 0x2D2 | OP_DOC_CHAR_REF | doc_id char_star_id → doc_id | Append Character Galaxy symlink (rare: symbols, math) |
| 0x2D3 | OP_DOC_MEANING_REF | doc_id meaning_star_id → doc_id | Append language-agnostic Meaning symlink |
| 0x2D4 | OP_DOC_STYLE_SPAN | doc_id style_mask start end → doc_id | Mark style range (bold/italic/header) |
| 0x2D5 | OP_DOC_PARA_BREAK | doc_id → doc_id | Paragraph/section break marker |
| 0x2D6 | OP_DOC_STRUCT_EMIT | doc_id struct_kind payload → doc_id | List/table/figure structural node |
| 0x2D7 | OP_DOC_RENDER_IN_LANG | doc_id target_lang_tag → glyph_stream | Swap terminal symlinks to target language |
| 0x2D8 | OP_DOC_RENDER_DOTMAP | doc_id viewport_w viewport_h → dotmap_star | Emit DotMap artifact (memory-image bridge) |
| 0x2D9 | OP_DOC_SYMLINK_RESOLVE | symlink_id depth → resolved_payload | Walk symlink chain to requested depth |
| 0x2DA | OP_DOC_CONTENT_HASH | doc_id → hash_u64 | Deterministic hash of symlink sequence |
| 0x2DB | OP_DOC_MATRYOSHKA_EMBED | doc_id dims → embedding | Prefix-compatible embed (64/128/256) |

## 5. Storage Schema

### 5.1 Document star binary layout (on-disk, House JSONL entry, Layer 4)

```
document_star {
  id:               uint64,        // content-hash (via 0x2DA)
  kind:             "document",
  title_ref:        word_star_id,  // symlink
  author_ref:       agent_star_id, // symlink
  created_ts:       uint64,
  structure_rpn:    bytecode,      // OP_DOC_* sequence
  matryoshka_64:    int8[64],      // coarse prefix
  matryoshka_128:   int8[128],     // refines 64
  matryoshka_256:   int8[256],     // refines 128
  symlinks: [                      // ordered content
    { kind: "word",    ref: word_star_id },
    { kind: "word",    ref: word_star_id },
    { kind: "meaning", ref: meaning_star_id, rendered_lang: "en" },
    { kind: "para_break" },
    { kind: "struct", struct_kind: "list", items: [...] },
    ...
  ],
  style_spans:      [{ start, end, mask }],
  provenance:       { source_uri, ingestion_ts, version },
}
```

No character bytes. No glyph paths. No font tables. All metadata lives below in lower tiers.

### 5.2 VRAM representation (Galaxy Universe, hot path)

Documents live in the `Tool Galaxy` region (Knowledgeverse §3.4). Each open document is an active Galaxy star whose symlink list is navigable by LED-A*. The RPN execution path:

1. Query enters Doc Galaxy neighborhood via semantic gravity.
2. Nine-Chain Swarm workers traverse the symlink list.
3. Each worker dereferences symlinks to the requested depth (OP_DOC_SYMLINK_RESOLVE).
4. Attention/resonance scoring (BitNet b1.58 ternary + contrastive margin) accumulates over the walk.
5. Halting Gate returns answer trit.

No numpy, no string ops. Pure symlink traversal on GPU.

## 6. Ingestion Contract

Ingestion of a raw document (PDF, HTML, plain text) happens on the flexible Python path (`ingestion/` subtree). The path is:

1. Text tokenized → tokens matched against Word Galaxy (symlink reused) OR new Word star minted.
2. New Word stars trigger: character decomposition → existing Character stars or new mints → Font/Meaning resolution.
3. Meaning resolution: nearest-neighbor in Meaning Galaxy via Matryoshka 64D prefix; link to existing or mint.
4. Structural metadata (headings, paragraphs, tables) becomes `OP_DOC_STRUCT_EMIT` nodes.
5. Final `document_star` emitted as `OP_DOC_STAR_NEW` + content sequence.

**Sovereignty:** ingestion may use numpy/pandas/regex (ingestion is NOT hot path). The result deposited into Galaxy Universe must be a pure symlink composition — no embedded text bytes, no tokenizer state, no language-specific literals at document level.

## 7. Interaction with Memory-as-Image (Lane E)

Per companion spec `TEMP/CLAUDE_TEXTURE_FORGE_IMAGE_TO_3D_ARC3_SCREEN_04.20.2026.md §10`:

A document star, being a pure symlink composition, is trivially convertible to a DotMap (`OP_DOC_RENDER_DOTMAP`, 0x2D8). The DotMap is:

- A raster (renderable to screen for human consumption).
- An RPN program (regeneratable at any resolution / language / font).
- A Galaxy star (indexable by semantic gravity).
- A Matryoshka embedding (hierarchical recall).

Therefore a document is a **4-way-addressable memory cell**: text (via symlink walk), image (via DotMap render), program (via RPN bytecode), concept (via embedding). The same underlying data plays all four roles without duplication.

## 8. Interaction with Semantic Gravity

Per `feedback_semantic_gravity_between_stars.md` (Christoph's coinage, Daniel's formula):

Document stars participate in semantic gravity through their T3 Meaning symlinks. The gravitational force between two documents is the aggregate force between their Meaning star sets, normalized by trace length. Language drops out — two documents about the same topic in different languages attract equally.

During sleep-time consolidation (see `feedback_exploratory_grammar_deferred.md`), frequently co-orbiting documents form compressed "gist" stars — document clusters whose representative is itself a document star at a higher abstraction tier.

## 9. Dual-Client Contract Compliance

Per `DUAL_CLIENT_CONTRACT_SPECIFICATION.md` (Form + Meaning, same data for humans AND AI):

- **Human clients** open the document via viewer's Document Pane: symlinks resolve to glyphs, glyphs render via Drawing Galaxy's procedural paths → pixels on screen. Language switch triggers re-evaluation at `OP_DOC_RENDER_IN_LANG` without re-fetching data.
- **AI clients** (TRM) traverse the symlink list directly at the Meaning tier. Never decodes to glyph pixels for reasoning — reasoning operates on Meaning stars, not surface forms (per `feedback_knowledge_not_benchmark_named.md`: "reasoning operates on MEANING, not language surface").

Same RPN bytecode. Same symlink chain. Different terminal resolution depth.

## 10. Sovereignty Gates

1. `grep -rn "document.*=.*\"\|document.*=.*'" knowledge3d/cranium/` returns ZERO hits in runtime paths. Documents are star IDs, never inline string literals.
2. `grep -rn "chars\[\|text\[\|content\[" knowledge3d/cranium/codecs/` — no character arrays embedded in codec data structures.
3. Any new document ingested via ingestion path MUST produce a content-hash via `OP_DOC_CONTENT_HASH` that is deterministic from the symlink list alone. Hash collisions across re-ingestion of the same document = pass; hash mismatch after byte-identical re-ingestion = bug.
4. Round-trip test: ingest `sample.pdf` → document star S → render S in same language → compare emitted glyph stream against original PDF's glyph extraction. ≥99% glyph-exact match required.

## 11. Out of Scope

- Book-level organization (a book is a Galaxy of document stars per `feedback_book_is_galaxy_not_star.md` — covered there).
- DRM / access control (handled by House room permissions, not document stars).
- Version control (documents are content-hashed; new version = new star with `supersedes` link).
- Binary attachments inside documents (images, videos) — those are 3DObjects / Audio / Frame Galaxy stars linked via `OP_DOC_STRUCT_EMIT` with `struct_kind = "figure"` referencing external star IDs.

## 12. Implementation Status

- Opcode block 0x2D0-0x2DF reserved in `RPN_DOMAIN_OPCODE_REGISTRY.md §11` (2026-04-20).
- Opcode constants minted in `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` (2026-04-20).
- Tokens registered in `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py` OPCODES dict (2026-04-20).
- Kernel implementation pending — Codex handoff in CODEX.md § Minecraft-for-Cognition lanes.
- Ingestion pipeline integration pending — requires companion spec update for `ingestion/document_ingestor.py`.

---

*"The document is the symlink list. The metadata lives elsewhere. Save information — don't duplicate it." — Daniel Campos Ramos, 2026-04-20.*
