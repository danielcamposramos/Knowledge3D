# Claude Architecture Spec — Ingestion Path Symlink Rewire

**Date:** 2026-04-18
**Author:** Claude (Architecture Partner)
**Scope:** Ingestion-path only. Hot-path sovereignty untouched.
**Directive origin:** Daniel's 2026-04-18 request — "make the ingestion path actually work as intended, following the proper standard and embedding strategy."

---

## 1. Problem Statement

### 1.1 Observed symptom

Knowledgeverse boot loads only a modest VRAM footprint (low hundreds of MiB)
where prior builds peaked near **1.5 GB** with the full four-layer cascade
materialized. The House-on-disk looks populated, but stars above Layer 0 lack
actionable RPN and proper upward symlinks to drawing primitives / character
glyphs / canonical meaning classes.

### 1.2 Root cause (three interlocking defects)

**Defect A — Layer 0 is stranded in `galaxy_pending/`.**
`knowledge3d/ingestion/atomic/drawing_grammar_builder.py` emits 7 drawing
primitives (LINE / ARC / QUAD / CUBIC / CIRCLE / RECT / TRI) to an output
JSONL path provided by caller and is commonly invoked into
`galaxy_pending/drawing_grammar.jsonl`. It is never promoted into the live
House, so the Galaxy loader at boot has no Layer-0 anchor to load. Upper
layers therefore ingest "orphan" — the symlinks they should hang off do not
exist in live storage.

**Defect B — Non-canonical IDs at the primitive floor.**
`drawing_grammar_builder.py` emits `PRIM_LINE`, `PRIM_CIRCLE`, `PRIM_RECT`,
etc. The canonical registry spec
(`docs/vocabulary/CANONICAL_REGISTRY_SPECIFICATION.md` §3, §4.6) requires
`drawing_primitive_line`, `drawing_primitive_circle`, … produced by
`canonical_drawing_primitive_id("line")`. Any upper-layer symlink that tried
to reference a primitive by canonical ID (the correct pattern) would miss
and — under the no-fallback contract — raise `canonical_lookup_miss`. The
current behavior is silent drift: upper layers simply do not emit the
symlink at all.

**Defect C — Sparse canonical seed + no symlink backfill.**
`scripts/ingest_canonical_to_qdrant.py` seeds **3** drawing primitives
(line / circle / rect) into `k3d_canonical`, not all 7. Meaning-class and
symlink-kind vocabularies are minimal. The "canonical library of system
labels and metadata" Daniel asked for exists in code (`CanonicalLookup`,
`symlink_helpers.link()`) but the Qdrant collection has never been seeded
to the completeness the four-layer architecture requires.

### 1.3 What "working as intended" looks like

- At boot, Knowledgeverse maps a House where every Grammar rule, Word,
  Character, Reality program, Math symbol, and ARC pattern carries a
  **resolved** symlink chain upward to canonical Layer-0 primitives and
  downward to `meaning_class` anchors.
- `drawing_grammar_builder.py` emits canonical IDs registered into
  `k3d_canonical` before any upper-layer ingestor runs.
- New corpora ingest through a **two-agent parallel pipeline** with
  file-level locking (no two agents on one file, ever).
- OCR runs as a **dedicated vision-model sidecar**, not inline Python
  pytesseract. Vision model = `qwen3-vl:235b-instruct-cloud` (Ollama cloud).
- Every star carries **actionable RPN** (`meaning_rpn`, and where relevant
  `grammar_rpn`, `reality_rpn`) so the hot path can execute procedurally
  rather than pattern-match surface strings.
- Embedding regeneration from `meaning_rpn` using the Matryoshka tiers
  **{64, 128, 512, 2048}** — **not** the 384/768 defaults.

---

## 2. Four-Layer Anchoring Contract

This is the invariant the ingestion path must preserve. No exceptions.

```
Layer 0  FORM       drawing_primitive_{line|arc|quad|cubic|circle|rect|tri}
                    (canonical ids, registered in k3d_canonical)
                    │
                    ▼   composite_of / component_refs (forward+back)
Layer 1  STROKE     strokes_<content_hash>  (Bezier decomposed to primitives)
                    │
                    ▼   visual_refs  /  glyph_refs
Layer 2  MEANING    meaning_<content_hash>  (star_id = ContentHash(meaning_rpn))
                    │                        ↑
                    │                        └── char_star_<lang>_<cp>
                    │                        └── word_star_<lang>_<lemma>
                    │                        └── math_symbol_<id>
                    ▼   grammar_refs  /  reality_refs  /  audio_refs
Layer 3  RULES      grammar_template_<kind>_<lang>_<token>
                    reality_program_<domain>_<slug>
                    rpn_template_<domain>_<slug>
                    │
                    ▼   meta_refs
Layer 4  META       meta_rule_<slug>  (rules that rewrite/compose Layer 3)
```

**Every star must carry:**
- `star_id` — content-hashed from `meaning_rpn || meaning_class || domain`
- `meaning_rpn` — executable RPN for the hot path
- At least one upward symlink (`visual_refs`, `char_refs`, or
  `component_refs`) terminating at a canonical Layer-0 primitive
- Bidirectional consistency enforced by `symlink_helpers.link()`
- Matryoshka prefix embeddings at tiers {64, 128, 512, 2048} regenerable
  from `meaning_rpn`

**No star ships without resolved symlinks.** On lookup miss the ingestor
MUST raise — not synthesize a placeholder. This is the no-fallback contract
from `knowledge3d/ingestion/canonical_lookup.py`.

---

## 3. Fix Sequence (ordered, gated)

Each gate must be green before the next begins. No parallelization across
gates — Layer 0 anchors before anyone references them.

### Gate 1 — Canonical Layer-0 seed (blocks everything else)

1. Fix `knowledge3d/ingestion/atomic/drawing_grammar_builder.py` to emit
   canonical IDs via `canonical_drawing_primitive_id(name)`:
   - `PRIM_LINE`   → `drawing_primitive_line`
   - `PRIM_ARC`    → `drawing_primitive_arc`
   - `PRIM_QUAD`   → `drawing_primitive_quad`
   - `PRIM_CUBIC`  → `drawing_primitive_cubic`
   - `PRIM_CIRCLE` → `drawing_primitive_circle`
   - `PRIM_RECT`   → `drawing_primitive_rect`
   - `PRIM_TRI`    → `drawing_primitive_tri`
2. Same builder calls `CanonicalLookup.register()` for each primitive
   immediately after emission — so any upper-layer ingestor that does
   `lookup.find_star_id("drawing_primitive", "line")` finds it.
3. Promote the JSONL from `galaxy_pending/drawing_grammar.jsonl` to the
   **live House** under the canonical drawing-galaxy path used by the
   live loader (coordinate exact path with Codex — see runbook).
4. Extend `scripts/ingest_canonical_to_qdrant.py` to seed all 7 primitives
   (currently seeds 3), plus extended `meaning_class` vocabulary
   (`drawing`, `glyph`, `word`, `number`, `grammar`, `reality`, `math`,
   `audio`, `object_3d`, `tool`, `game_2d`) and all 12 supported
   `symlink_kind` field paths from `CANONICAL_REGISTRY_SPECIFICATION.md` §7.
5. **Gate-1 check:**
   ```bash
   python -c "
   from knowledge3d.ingestion.canonical_lookup import CanonicalLookup
   l = CanonicalLookup()
   for k in ['line','arc','quad','cubic','circle','rect','tri']:
       sid = l.find_star_id('drawing_primitive', k)
       assert sid == f'drawing_primitive_{k}', (k, sid)
   print('gate1_ok')
   "
   ```

### Gate 2 — Canonical registry kinds complete

Add seed rows for the kinds reserved in Phase 7.A.1 that the ingestion
agents will need: `letter_star`, `font_glyph`, `word_lemma`, `grammar_rule`,
`rpn_template`, `math_symbol`. Seed the **vocabulary**, not per-language
instances — instances are what the ingestion agents produce and register.

**Gate-2 check:** every kind listed in
`CANONICAL_REGISTRY_SPECIFICATION.md` §3 has at least one row in
`k3d_canonical` and a schema docstring committed to
`knowledge3d/ingestion/canonical_lookup.py` describing its `key` format.

### Gate 3 — Symlink backfill for existing upper-layer stars

Stars already on disk (game_mechanics 100% coverage, Grammar 64%,
Reality 24%, upper layers sparse) need their missing upward symlinks
filled in. This is a **stitching job**, not a re-ingestion. The two-agent
pipeline handles it — Agent A walks existing JSONL, Agent B reads the
agent A proposals and commits them through `symlink_helpers.link()`.

**Gate-3 check:** coverage audit report shows every star has at least one
resolved upward symlink terminating at a Layer-0 canonical primitive
(or is explicitly exempted — e.g., pure audio stars with `audio_refs`
terminating at an audio-primitive anchor).

### Gate 4 — New corpus ingestion on the parallel pipeline

All new PDFs, markdown, datasets go through the parallel agent pipeline
from §4. Not through `ingest_full_corpus_parallel.py` (legacy, uses
PyPDF2 + multiprocessing.Pool — keep for sovereign-text embedding only).

---

## 4. Two-Agent Parallel Ingestion Topology

### 4.1 Agents

- **Agent A (Extractor)** — reads one file, produces a candidate
  star-graph draft (nodes + proposed symlinks) WITHOUT writing to live
  storage. Output: JSON proposal in `data/ingest_proposals/<file_id>.json`.
- **Agent B (Stitcher+Committer)** — reads Agent A's proposal, resolves
  every symlink through `CanonicalLookup` (no-fallback), adds the
  Matryoshka embeddings at {64, 128, 512, 2048}, runs
  `symlink_helpers.link()` for each bidirectional pair, writes to the
  live House, marks the `CorpusEntry.ingested = True`.
- **Agent C (OCR sidecar)** — invoked only when Agent A encounters a
  scanned-image page. Input: page bitmap PNG. Output: text tokens with
  bbox coords. Runs on `qwen3-vl:235b-instruct-cloud` via
  `mcp__ollama-specialists__ask_cloud`.

### 4.2 File-level locking (prevents two agents on one file)

Extend `knowledge3d/ingestion/corpus_manifest.py` `CorpusEntry` dataclass:

```python
@dataclass
class CorpusEntry:
    # existing fields ...
    ingested: bool = False
    # NEW:
    locked_by: str | None = None        # agent_id holding the lock
    locked_at: str | None = None        # ISO-8601 timestamp
    lock_kind: str | None = None        # "extract" | "stitch" | "ocr"
```

New methods (Codex implements — this spec is authoritative):

- `claim_next_available(agent_id, lock_kind)` — atomic compare-and-swap:
  picks the topologically-first entry with `ingested == False` and
  `locked_by is None`, sets `locked_by = agent_id`. Uses
  `fcntl.flock(LOCK_EX)` on the manifest file during the swap.
- `release(agent_id, entry_id, success: bool)` — clears lock; if
  `success` also sets `ingested = True`.
- `expire_stale_locks(max_age_seconds=1800)` — reclaims locks held > 30 min
  (agent crash recovery).

### 4.3 Resume-aware loop (exact pseudocode)

```
while True:
    entry = manifest.claim_next_available(agent_id, "extract")
    if entry is None:
        break                    # nothing left
    try:
        if needs_ocr(entry.path):
            pages = ocr_agent.process(entry.path)     # Agent C
        else:
            pages = native_text_reader(entry.path)
        proposal = extractor.draft(pages, entry)       # Agent A logic
        proposal_path = write_proposal(proposal, entry)
        manifest.release(agent_id, entry.id, success=True)
        stitch_queue.push(proposal_path)
    except Exception as e:
        manifest.release(agent_id, entry.id, success=False)
        log_failure(entry, e)
        raise                    # no silent fallback, ever
```

Agent B runs an identical loop over `stitch_queue` with `lock_kind="stitch"`.

### 4.4 Concurrency invariant

At any moment, for any file `f`:
`count(agents working on f) <= 1 per lock_kind, <= 2 total`
(one extractor AND one stitcher may progress, but they operate on
sequential artifacts — extract output → stitch input — never on the
same live-storage region simultaneously).

### 4.5 Why two agents, not N

Daniel's directive: *"two agents, each working in one file (to avoid
problems)"*. Two is the smallest number that keeps the pipeline full
while guaranteeing one writer per file. Scaling up is a later
optimization and requires proving the lock-semantics hold under load.

---

## 5. OCR Sidecar (Agent C)

### 5.1 Why vision-model OCR, not pytesseract

Daniel: *"use ollama vision models to do OCR instead of local python
library"*. Rationale: pytesseract misses math notation, non-Latin
scripts, and diagram text; vision models handle layout + notation
natively and can emit semantic bbox hints that help Agent A assign
`visual_refs` correctly.

### 5.2 Contract

**Input:** page image (PNG or JPEG bytes), rendering DPI ≥ 200.
**Output (JSON):**
```json
{
  "page_id": "...",
  "language": "eng|por|...",
  "blocks": [
    {"kind": "paragraph|heading|caption|equation|table_cell",
     "bbox": [x0, y0, x1, y1],
     "text": "...",
     "confidence": 0.0-1.0}
  ],
  "notes": "free-form observations (e.g. 'handwritten', 'rotated 90°')"
}
```

### 5.3 Implementation hook

Call via `mcp__ollama-specialists__ask_cloud` with `model_tag="vision"`
(server-side routes to `qwen3-vl:235b-instruct-cloud`). Prompt lives in
`prompts/ingestion/SYSTEM_PROMPT_OCR_AGENT.md`.

---

## 6. Canonical Metadata Library ("Outside Qdrant Library")

This is what Daniel meant by *"an outside qdrant library with all
canonical part of the system labels and meta-data"*. It is already
scaffolded — Phase 7.0 lives in `k3d_canonical` at `localhost:6333`
with 5 kinds. The ingestion path will not function correctly until this
is filled out.

### 6.1 Seed completeness checklist

| Kind | Key format | Source | Gate |
|------|-----------|--------|------|
| `drawing_primitive` | `line|arc|quad|cubic|circle|rect|tri` | `drawing_grammar_builder.py` fixed output | Gate 1 |
| `meaning_class` | `drawing|glyph|word|number|grammar|reality|math|audio|object_3d|tool|game_2d` | extended seed | Gate 2 |
| `symlink_kind` | one of 12 supported field paths (§7 of registry spec) | extended seed | Gate 2 |
| `star_id` | `<kind>_<hash>` computed from content | live ingestion | Gate 3+ |
| `grammar_template` | `<lang>_<pattern>` | atomic grammar builders | Gate 3 |
| `letter_star` | `<lang>_<codepoint_hex>` | character ingestion | Gate 3 |
| `font_glyph` | `<font>_<codepoint_hex>_<weight>` | glyph ingestion | Gate 3 |
| `word_lemma` | `<lang>_<lemma>` | word ingestion | Gate 3 |
| `math_symbol` | `<tex_command_sans_backslash>` | math ingestion | Gate 3 |
| `rpn_template` | `<domain>_<slug>` | RPN template ingestion | Gate 3 |
| `grammar_rule` | `<lang>_<rule_slug>` | grammar ingestion | Gate 3 |

### 6.2 Where agents look it up

Agents A and B both instantiate `CanonicalLookup(qdrant_host=...,
api_key=...)` at startup. Agent A uses `find_star_id(kind, key)` while
drafting proposals — on miss, it flags the proposal as `needs_new_star`
(Agent B then mints + registers). Agent B is the only agent permitted
to call `register()`, keeping writes serialized through one process.

---

## 7. Embedding Strategy

### 7.1 Tiers

- **Canonical registry:** 384-dim `fast-all-minilm-l6-v2` on FastEmbed
  (local, CPU-fine). This is the *label lookup* channel, not the
  *meaning* channel.
- **Star Matryoshka:** prefix embeddings at **64 / 128 / 512 / 2048**
  dimensions, generated from `meaning_rpn` via the Phenom host
  (`192.168.0.60:11434`, models `qwen/nomic-embed-text-v2-moe`). The
  64-dim prefix is what the LOD system loads first; 2048-dim is the
  full-quality channel the halting gate uses for final scoring.

**NOT** 384 and **NOT** 768. Those are registry labels, not meaning
vectors. (See Daniel's correction — this was confused in prior builds.)

### 7.2 Regenerability invariant

Any star must be reproducible in full from its `meaning_rpn` string
alone: hash → `star_id`, embed → Matryoshka tiers, resolve symlinks →
live House entry. If a star exists that cannot be regenerated, it is
malformed and must be either repaired or deleted. No "legacy" exemption.

### 7.3 Why the embedders live on the Phenom host

Frees the RTX 3070 for sovereign hot-path execution + local GPU
testing. Ingestion is explicitly allowed to depend on external
hardware (see CLAUDE.md sovereignty boundary — *"Ingestion Path =
Flexible"*).

---

## 8. Hot-Path Sovereignty Unchanged

This spec is entirely ingestion-side. Zero changes to:
- `knowledge3d/cranium/ptx_runtime/*`
- PTX kernels
- TRM game loop
- Galaxy VRAM layout
- Knowledgeverse hot path

The ingestion path's job is to produce a House-on-disk that the
sovereign loader can mmap at boot with every symlink resolved. After
that, the hot path does what it always does — PTX + Galaxy + RPN, no
Python, no fallbacks, ever.

---

## 9. Deliverables & File Map

### 9.1 Architecture (Claude — this session)
- [x] `TEMP/CLAUDE_INGESTION_SYMLINK_REWIRE_04.18.2026.md` (this file)
- [x] `prompts/ingestion/SYSTEM_PROMPT_INGEST_AGENT.md`
- [x] `prompts/ingestion/SYSTEM_PROMPT_OCR_AGENT.md`
- [x] `prompts/ingestion/TASK_TEMPLATE_INGEST.md`
- [x] `TEMP/CODEX_LAYER0_SEED_AND_PARALLEL_INGEST_04.18.2026.md` (runbook)

### 9.2 Implementation (Codex — per runbook)
- Fix `knowledge3d/ingestion/atomic/drawing_grammar_builder.py`
- Extend `scripts/ingest_canonical_to_qdrant.py`
- Build `scripts/ingest_parallel_agents.py`
- Build `scripts/ocr_sidecar_service.py`
- Extend `knowledge3d/ingestion/corpus_manifest.py` (locking)
- Write tests under `tests/ingestion/`

### 9.3 Post-run artifacts (both)
- Coverage audit report (Codex runs, Claude reviews)
- VRAM-load regression baseline (should hit ≥ 1 GB once Layer 0 ships live)

---

## 10. Success Criteria

1. **VRAM load at boot ≥ 1 GB** once Gate 3 completes (restoration of the
   observed prior peak).
2. **Zero `canonical_lookup_miss` exceptions** in a clean ingestion run
   over any subset of the corpus manifest.
3. **Coverage audit:** ≥ 95% of stars carry at least one upward symlink
   to a Layer-0 canonical primitive.
4. **Parallel pipeline demonstrably processes 2 files concurrently**
   with no lock violations across a 20-file smoke corpus.
5. **OCR sidecar handles a known-scanned PDF** (e.g. an older math-competition
   problem booklet) and produces semantically-usable blocks.
6. **Benchmarks non-regress:** ARC 10/10, Math 20/20 pinned post-rewire.

---

## 11. Forbidden

- Silent synthesis of missing canonical IDs in either agent (violates
  no-fallback contract).
- Writing to live House before Agent B's symlink resolution step
  (Agent A must never touch live storage).
- Using pytesseract / tesseract / easyocr in the new path (vision model
  only).
- Bypassing `CanonicalLookup` by hard-coding primitive IDs in upper-layer
  ingestors.
- Calling the Phenom embedder from the hot path (ingestion only).
- Adding any new 384-dim or 768-dim embedding tier to stars — Matryoshka
  only.
- Re-numbering opcodes or renaming any existing canonical `key` value
  (violates expand-not-replace from
  `memory/feedback_expand_not_replace_opcodes.md`).

---

## 12. Locked decisions (Daniel, 2026-04-18)

1. **Proposal artifacts committed to git.** `data/ingest_proposals/**`
   is tracked. Daniel's reasoning: *"this is cheap and allows for
   deterministic proof of concept (current tech enabling the next
   frontier and latter on integrating as well)"*. Same applies to the
   virtual-page PoC artifacts (§13.6).
2. **Corpus source = NAS library.** Root:
   `/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/`.
   **Process locally, do NOT copy into the project repo.** The manifest
   records NAS paths + SHA-256 hashes + ingestion state. Binary content
   stays on the NAS; stars + proposals land in the repo. Non-knowledge
   material (empty strips, system files) is filtered at manifest
   construction time.
3. **OCR cache persists.** `data/ocr_cache/<pdf_sha>/<page>.json`
   per-page cache, committed to git (deterministic proof of concept,
   same logic as decision 1). Re-ingestion MUST NOT re-pay the cloud-
   vision bill.

---

## 13. Virtual Page Proceduralization (Daniel's Layer-3 Extension)

Daniel's 2026-04-18 addition: *"translate drawing elements into a
virtual page constructed using drawing premisses and glyphs (maybe
some graph grammar RPN rules to draw tables, pages and such to be
written into, with all word processing standard rules)"*.

OCR output is NOT the final artifact — it is the *intermediate*. The
final artifact per document page is a `virtual_page_<doc_sha>_<page_n>`
star carrying a **graph-grammar RPN program** that can reconstruct the
page from canonical drawing primitives + glyph stars + word stars. The
Galaxy then reasons over form+meaning, not surface text strings.

### 13.1 Graph node kinds

**Structural:** `Document`, `Page`, `Frame`, `Column`, `Block`, `Section`
**Flow:** `Paragraph`, `List`, `ListItem`, `LogicalLine`, `LayoutLine`,
`Run`, `Word`, `Glyph`
**Tabular:** `Table`, `RowGroup` (thead/tbody/tfoot), `Row`, `Cell`
**Figural:** `Figure`, `Image`, `Caption`, `Equation` (RPN-bearing,
Layer-3)
**Peripheral:** `Header`, `Footer`, `PageNumber`, `Footnote`,
`MarginNote`, `Hyperlink`, `Annotation`
**Style:** `StyleScope` (symlink hub for inheritance — not visible)

**Edge kinds:** `contains`, `reading_order_next|prev`, `style_inherits`,
`baseline_align`, `grid_align`, `references_ocr_bbox`,
`references_meaning_star`, `cross_refs`, `language_run`, `logical_equiv`.

**Per-node payload:** `bbox`, `reading_order`, `style_refs` (list of
StyleScope star_ids), `meaning_star_refs`, `language` (BCP-47),
`confidence`.

### 13.2 Opcode reservation — VIRTUAL_PAGE block

**Confirmed free range:** 0x1D0–0x1FF (48 slots — per opcode registry
audit 2026-04-18).
**Reservation target:** 0x1D0–0x1FF initially; right to extend into
0x200–0x20F if sub-families exceed 48 on registry-owner review.
**Reservation ACTION (Gate-0 in runbook — BLOCKS EVERYTHING):**
amend `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` §11 with a
VIRTUAL_PAGE row and the sub-family labels below — NO specific numbers
assigned yet (per opcode-range reservation protocol from
`memory/feedback_opcode_range_reservation_protocol.md`).

### 13.3 Opcode sub-families (arity + stack-effect)

Stack notation `(before -- after)`. `*` = opcode that carries a
meaning symlink (requires CanonicalLookup hit at ingestion time).

**Page / Frame (pure geometry):**
- `PAGE_BEGIN` `(page_id -- )` / `PAGE_END`
- `PAGE_SET_SIZE` `(w h unit -- )`
- `PAGE_SET_MARGINS` `(top right bottom left -- )`
- `FRAME_BEGIN` `(frame_id -- )` / `FRAME_END`
- `FRAME_POSITION` `(x y w h -- )`
- `FRAME_COLUMNS` `(n gap -- )`

**Table (pure structure):**
- `TABLE_BEGIN` `(table_id cols rows -- )` / `TABLE_END`
- `TABLE_ROW_BEGIN` / `TABLE_ROW_END`
- `TABLE_CELL_BEGIN` `(rowspan colspan -- )` / `TABLE_CELL_END`
- `TABLE_BORDER` `(edge_mask weight style -- )`
- `TABLE_HEADER_GROUP` / `TABLE_BODY_GROUP` / `TABLE_FOOT_GROUP`

**Paragraph / Line (layout):**
- `PARAGRAPH_BEGIN` / `PARAGRAPH_END`
- `PARAGRAPH_ALIGN` `(mode -- )` — left/right/center/justify/start/end
- `PARAGRAPH_INDENT` `(first rest -- )`
- `PARAGRAPH_LEAD` `(leading -- )`
- `LINE_BEGIN` / `LINE_END`
- `LINE_BASELINE` `(y -- )`
- `LINE_AXIS` `(horizontal|vertical -- )`  — CJK support
- `LINE_KERN` `(pair_table_ref -- )`

**Run (style, pure):**
- `RUN_BEGIN` `(style_scope_id -- )` / `RUN_END`
- `RUN_FONT*` `(font_star_id -- )` — Character Galaxy ref
- `RUN_SIZE` `(pt -- )`
- `RUN_WEIGHT` `(weight -- )`
- `RUN_STYLE` `(italic_flag -- )`
- `RUN_COLOR` `(rgb -- )`
- `RUN_LANG` `(lang_tag_id -- )`
- `RUN_FONT_FEATURES` `(feature_mask -- )` — ligatures, tabular nums

**Terminals (meaning-bearing — MUST resolve):**
- `WORD_EMIT*` `(word_star_id -- )` — Word Galaxy
- `GLYPH_EMIT*` `(glyph_star_id -- )` — Character Galaxy
- `NUMERAL_EMIT*` `(number_star_id -- )` — Number Galaxy, script-aware
- `EQUATION_EMIT*` `(equation_rpn_id -- )` — Math Galaxy RPN program
- `FIGURE_EMIT*` `(figure_star_id -- )` — Drawing Galaxy composition
- `SYMBOL_EMIT*` `(symbol_star_id -- )` — punctuation / operators

**Layout flow (pure):**
- `LAYOUT_LINE_BREAK`, `LAYOUT_COLUMN_BREAK`, `LAYOUT_PAGE_BREAK`
- `LAYOUT_KEEP_WITH_NEXT`, `LAYOUT_KEEP_TOGETHER`
- `LAYOUT_FLOAT` `(anchor_ref side -- )`

**Style scope:**
- `STYLE_PUSH` `(style_scope_id -- )` / `STYLE_POP`
- `STYLE_INHERIT` `(parent_scope_id -- new_scope_id)`

**BiDi / script:**
- `BIDI_ISOLATE_BEGIN` `(direction -- )` / `BIDI_ISOLATE_END` — UAX #9
- `SCRIPT_BEGIN` `(script_tag -- )` / `SCRIPT_END`

**Hyphenation (pattern-driven):**
- `HYPHEN_TRY*` `(word_star_id lang -- break_positions)` — consults
  per-language pattern star in Grammar Galaxy; terminal still emits
  the unsplit word_star.

Total ~58 opcodes across ~11 sub-families. Fits in 48 slots if we
prune `TABLE_HEADER_GROUP|BODY_GROUP|FOOT_GROUP` into a single
`TABLE_GROUP (kind -- )` and merge `LAYOUT_KEEP_*` into one
`LAYOUT_KEEP (mode -- )`. Otherwise extend reservation into 0x200.

### 13.4 Word-processing standards coverage

| Standard | Mechanism |
|---------|-----------|
| Line breaking (Knuth-Plass) | Grammar Galaxy RPN program consuming `WORD_EMIT*` stream, emitting `LAYOUT_LINE_BREAK` positions. No new opcode. |
| Justification | `PARAGRAPH_ALIGN justify` + per-line glue expansion. No new opcode. |
| Hyphenation | `HYPHEN_TRY*` family (4 opcodes). |
| Orphan/widow | `LAYOUT_KEEP_*` solved by LED-A* over layout graph. |
| Row/col span | `TABLE_CELL_BEGIN (rowspan colspan)`. |
| Column reading-order | `reading_order_next` edges + `FRAME_COLUMNS`. |
| BiDi (Arabic/Hebrew in Latin) | `BIDI_ISOLATE_BEGIN/END` nested scopes. |
| CJK vertical | `SCRIPT_BEGIN vertical` + `LINE_AXIS vertical`. |
| OpenType features | `RUN_FONT_FEATURES`. |

### 13.5 Ingestion contract (Agent B addendum)

Agent B's stitcher pipeline extends with a "Virtual Page Compile" step
between terminal resolution and star commit:

1. Classify OCR blocks into node kinds (heading/paragraph/table/etc.).
2. Build `contains` tree from bbox containment; build
   `reading_order_next` from OCR order + column detection.
3. Cluster runs by font/size/weight/color into `StyleScope` stars.
4. Resolve every terminal (`WORD_EMIT*`, `GLYPH_EMIT*`, …) via
   `CanonicalLookup`. Miss → synchronous proceduralization via Agent A
   (mint word_star / glyph_star) BEFORE this page commits. No fallback.
5. Walk graph in reading order, serialize opcode program.
6. Emit `virtual_page_<doc_sha>_<page_n>` star carrying:
   - `meaning_rpn` — byte-packed opcode program
   - `page_graph_refs` — star_ids for every graph node
   - `visual_refs` — drawing primitives + glyphs referenced
   - `surface_forms.<lang>` — flat text re-derivation (search/debug
     only; NOT authoritative)
   - `matryoshka_embeddings` — prefixes at {64, 128, 512, 2048}D of the
     page meaning; regenerable from `meaning_rpn`
   - `provenance` — doc_sha, page_n, OCR confidence histogram,
     stitcher version

**Invariant:** a page whose terminals cannot all be resolved is
rejected, not emitted with gaps. The manifest logs the miss and
releases the lock as `success=False` for human triage.

### 13.6 Proof-of-concept smoke target

One A4 page: `<h1>` heading, 2 body paragraphs (4–6 lines each), 1
small table (3×3 with header row), 1 figure + caption. Expected
artifact:

- 1 `virtual_page` star
- ~45 graph-node stars
- ~12 drawing-primitive refs
- ~40 unique glyph refs
- ~50 unique word refs
- ~350–450 opcodes in `meaning_rpn`
- 4 Matryoshka embedding tiers

Commit locus: `TEMP/virtual_page_poc/`. Deterministic: same OCR input
→ byte-identical RPN output (stitcher must be hash-stable).

### 13.7 Sovereignty invariant

Virtual page RPN **lives in the ingestion path**. The hot path only
executes the RPN via existing PTX dispatchers + new dispatcher stubs
reserved at the same time as the opcode range. **Zero new Python in
the hot path. Zero new fallbacks.** Any dispatcher stub that cannot
execute an opcode hard-fails and surfaces to sleep-time for triage —
the standard no-fallback contract.

### 13.8 Prior art to reference (do NOT duplicate)

- `knowledge3d/ingestion/atomic/drawing_grammar_builder.py` — hierarchical
  composition idiom (`component_refs` + `procedural_programs.composition`)
- `knowledge3d/ingestion/fonts/glyph_to_rpn.py` — glyph outline → RPN
- `docs/research/DRAWING_GRAMMAR_SPEC.md` — 7-level layer hierarchy
- `knowledge3d/cranium/bridges/procedural_drawing_bridge.py` — RPN
  drawing execution (MOVE/LINE/QUAD/CUBIC/ARC/CLOSE/STROKE/FILL)
- `docs/research/2d_engine_techniques_spec.md` — 2D canvas patterns
- `TEMP/RPN_DRAWING_INTEROP_DESIGN.md` — buffer handoff patterns
- `TEMP/PROCEDURAL_OCR_SOVEREIGN_PLAN.md` — prior procedural-OCR sketch
- `knowledge3d/tablet/wine/game2d_wine.py` — existing GAME_2D route

---

## 14. Python-Drift Pre-Flight (Hard Gate)

Per `memory/feedback_python_dispatch_is_not_a_line_item.md`: drift is
the partner, not the implementer. Before ANY Codex PR lands in this
rewire, the following grep must return zero hits:

```bash
# Forbidden in knowledge3d/cranium/, knowledge3d/knowledgeverse/,
# and any module imported by knowledgeverse at boot:
grep -RnE "if .*\.surface_kind ==|if .*\.kind ==|route_by_kind|dispatch_by_kind|galaxy_switch|mode_switch" \
    knowledge3d/cranium/ knowledge3d/knowledgeverse/
```

Zero hits = pass. Any hit = drift; Codex reports the violation,
does not work around. Runbook Gate-0 includes this check.

Additionally:
- No new `import numpy`, `import cupy`, `import scipy`, `import sympy`,
  `import torch` in modules loaded by `knowledgeverse.boot()`.
- No new `subprocess` calls on the hot path.
- No new `requests` / `httpx` / `urllib` calls outside
  `knowledge3d/ingestion/**`.

These are checked by an automated sovereignty test under
`tests/sovereignty/test_no_python_drift_on_hot_path.py` that Codex
adds in Gate 3.

---

## 15. Sub-Agent / Specialist Dispatch Policy

Per Daniel's standing directive: *"dispatch ollama specialists, sub-
claude-agents haiku and sonnet as needed so that no python drift is
again introduced"*.

Architecture (Claude) dispatches:
- **Explore subagent** — codebase audits that would burn main-context
  tokens (done this session for virtual-page prior art).
- **Plan subagent** — non-trivial implementation planning before
  writing runbooks.
- **k3d-architecture-partner subagent** — when a specific component
  needs its own deep architecture dive (e.g., graph-grammar dispatcher
  PTX kernels, eventually).

Implementation (Codex) dispatches:
- **ollama `plan_task`** before any PR touching more than 2 files.
- **ollama `ask_coder`** for idiomatic Python / Qdrant / Ollama-client
  patterns.
- **ollama `kimi_swarm`** for multi-angle review of symlink-graph
  invariants or lock-semantics proofs.
- **claude sub-agents (haiku for small edits, sonnet for
  multi-file refactors)** when a task is too small to justify a human-
  level Codex session but too big for a single-shot edit.

Main-context token economy: every research query that can be delegated
to a sub-agent MUST be. Main-context work = synthesis, decisions,
spec writing, and final review only.
