# Codex Spec — Batch 8: HS Math Foundation + Cluster 1 Ingestion

**Date:** 2026-04-14
**Author:** Claude (architecture partner)
**Status:** Approved — Daniel confirmed open questions (2026-04-14); Codex may proceed
**Role reminder:** Claude writes specs. Codex implements.
**Predecessor:** `TEMP/CODEX_BATCH7_REASONING_TAXONOMY_INGESTION_2026-04-14.md` (landed: 27 `meaning_star` + 46 `reasoning_taxonomy_symlink` + 73 `grammar_template` rows)

---

## 0. Executive Summary

Batch 7 proved the canonical-registry ingestion spine for one narrow source shape
(the pipe-delimited reasoning-taxonomy catalogue). Batch 8 begins the High-School
curriculum wave per master plan §9.4 item 2 (`CLAUSE: HS math clusters 1 → 2 → 3`).

HS Math source files diverge in three ways Batch 7 did not need to handle:

1. **Three different physical shapes** — Cluster 1 uses bullet-form key/value
   blocks, Cluster 2 uses fenced JSON blocks, Cluster 3 uses a hybrid of both
   plus catalogue tables. Batch 7's pipe-delimited parser cannot read any of
   them as-is.
2. **Three different canonical-id dialects** — `rule_order_of_operations_pemdas`
   (Cluster 1), `formula::triangle_area_base_height` (Cluster 2),
   `formula_population_variance` (Cluster 3). These must normalise to a single
   key/star_id form before `CanonicalLookup.register()` can dedupe them.
3. **Heavy symlinks to Phase 7.A.1 stars** — every HS math star references
   `star.letter.a` / `letter::a` / `star.symbol.fraction` / `constant::pi`.
   Batch 8 must resolve those references against the existing Phase 7.A.1 seed,
   not against a new catalogue.

Rather than solve all three clusters in one mega-batch, Batch 8 lands:

- **Shared infrastructure** (multi-format parser, canonical-id normaliser,
  symlink resolver, Phase 7.A.1 seed audit, RPN-sketch lexer) usable by
  Clusters 1/2/3 without rework.
- **Cluster 1 ingestion only** as proof-of-life for the infrastructure.
- **Documentation pins** for Cluster 2 and Cluster 3, which become Batch 9
  and Batch 10 as thin driver-only add-ons.

Scope is ingestion-path only. No hot-path changes, no new opcodes, no new
kernels, no persistent-swarm touches. The Batch 6 ABI stays untouched.

---

## 1. Scope Fence

### 1.1 What Batch 8 touches

- `knowledge3d/ingestion/hs_math_parser.py` — new, multi-format
- `knowledge3d/ingestion/math_canonical_id.py` — new, id normaliser
- `knowledge3d/ingestion/math_symlink_resolver.py` — new
- `knowledge3d/ingestion/math_semantic_aliases.py` — new, symbol alias table
- `knowledge3d/ingestion/math_symlink_allowlist.txt` — new, controlled escape hatch
- `knowledge3d/ingestion/rpn_sketch_lexer.py` — new, non-blocking validator
- `scripts/ingest_phase7a1_seed_audit.py` — new, audits + backfills pre-seed gaps
- `scripts/ingest_hs_math_cluster1.py` — new, Cluster 1 driver
- `tests/test_batch8_*.py` — new tests per slice

### 1.2 What Batch 8 MUST NOT touch

- `knowledge3d/cranium/cuda/k3d_swarm_persistent.cu`
- `knowledge3d/cranium/cuda/reasoning_tick_io.cuh`
- `knowledge3d/cranium/cuda/reasoning_tick_entrypoints.cuh`
- `knowledge3d/cranium/cuda/n_selector.cu`
- `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`
- any `knowledge3d/cranium/kernels/rpn_*.cu`
- `knowledge3d/ingestion/canonical_lookup.py` core API
- `knowledge3d/ingestion/reasoning_taxonomy_parser.py` (Batch 7 parser stays
  frozen; this batch does not refactor it)
- `tests/test_batch5_*`, `tests/test_batch6_*`, `tests/test_batch7_*`
- `TEMP/CLAUDE_REASONING_PARADIGMS_AND_N_SWARM_SPEC_2026-04-13.md` (master plan)

### 1.3 Allowed imports

Ingestion path only. Same rules as Batch 7:
- `qdrant_client`, `fastembed`, `uuid`, `hashlib`, `pathlib`, `dataclasses`,
  `typing`, stdlib `re` (ingestion regex is allowed; hot path regex is not).
- NOT allowed: `numpy`, `cupy`, `scipy`, `sympy`, `cuda`, any PTX machinery.

---

## 2. Source Shape Inventory

| File | Shape | Primary canonical-id dialect | Notes |
|---|---|---|---|
| `TEMP/KIMI_MATH_HS_CLUSTER1_ARITHMETIC_ALGEBRA_2026-04-13.md` | bullet key/value blocks under `####` headers | `rule_<snake>`, `formula_<snake>`, `identity_<snake>`, `theorem_<snake>`, `concept_<snake>`, `method_<snake>` | Symlink refs use `star.letter.X` / `star.symbol.X` / `star.constant.X` dotted form |
| `TEMP/KIMI_MATH_HS_CLUSTER2_GEOMETRY_TRIG_2026-04-13.md` | fenced ```json``` blocks | `formula::<snake>`, `theorem::<snake>` | Symlink refs use `letter::X` / `symbol::X` / `constant::X` double-colon form |
| `TEMP/KIMI_MATH_HS_CLUSTER3_STATS_DISCRETE_APPLIED_2026-04-13.md` | hybrid bullet + catalogue tables | same as Cluster 1 | Table rows may lack RPN sketches; catalogue is secondary to representative stars |

Batch 8 parses shape 1 only. Shapes 2 and 3 are defined at parser-module level
(so the module is future-proof) but their drivers land in Batches 9 and 10.

---

## 3. Canonical ID Normalisation

All three dialects collapse to one canonical form:

```
canonical_key  = <category>_<slug>
star_id        = math_<canonical_key>
kind           = meaning_star     # reuses Batch 7's kind
subkind        = math_hs_cluster<n>   # stored in metadata, for filtering
```

Where:

- `<category> ∈ {formula, identity, theorem, rule, concept, method}`
- `<slug>` is `canonical_slug()` applied to the leaf name (already in
  `canonical_lookup.py`).

### 3.1 Dialect → canonical mapping

| Input | Category | Slug | Canonical key | star_id |
|---|---|---|---|---|
| `rule_order_of_operations_pemdas` | `rule` | `order_of_operations_pemdas` | `rule_order_of_operations_pemdas` | `math_rule_order_of_operations_pemdas` |
| `formula::triangle_area_base_height` | `formula` | `triangle_area_base_height` | `formula_triangle_area_base_height` | `math_formula_triangle_area_base_height` |
| `formula_population_variance` | `formula` | `population_variance` | `formula_population_variance` | `math_formula_population_variance` |

Missing-category inputs (pure snake_case without a known category prefix)
MUST raise a normalisation error. They cannot default to `concept_`; ambiguity
becomes a dangling ref and goes through the allowlist, not through silent
category injection.

### 3.2 Category whitelist

`math_canonical_id.py` must hardcode the 6 categories above. Any input whose
prefix is not in the whitelist raises `MathCanonicalIdError` and the driver
fails fast (fail-and-fix, never silently promote unknown categories).

---

## 4. Symlink Resolution

### 4.1 Reference dialects observed

- `star.letter.a` / `letter::a` / `letter_a` → should resolve to `char_a`
- `star.letter.x` / `letter::x` / `letter_x` → `char_x`
- `star.symbol.<glyph>` — dotted glyph name (`pi`, `sqrt`, `fraction`)
- `symbol::<glyph>` — double-colon glyph name, may also carry a literal Unicode
  glyph (Cluster 2 uses `symbol::√` directly)
- `star.constant.pi`, `constant::pi`, `constant::e`, `constant::reciprocal`

### 4.1.1 Parse-time normalisation (Daniel-confirmed, 2026-04-14)

The parser normalises **all** symlink refs to a single canonical textual form
**before** the resolver ever sees them, so the resolver only has one dialect
to handle. Normalisation rules, applied in order:

1. Strip surrounding backticks and whitespace.
2. Collapse `star.letter.` / `letter::` / `letter_` → `letter::`
3. Collapse `star.symbol.` / `symbol::` / `symbol_` → `symbol::`
4. Collapse `star.constant.` / `constant::` / `constant_` → `constant::`
5. Collapse `star.concept.` / `concept::` / `concept_` → `concept::`
6. If the tail after `symbol::` or `constant::` is a **literal Unicode
   codepoint** (length 1 outside ASCII), translate it to a named alias by
   reverse lookup against `math_semantic_aliases.UNICODE_TO_NAME`:
   ```
   "\u221a" -> "sqrt"
   "\u03c0" -> "pi"
   "\u221e" -> "infinity"
   "\u2211" -> "sum"
   "\u220f" -> "product"
   "\u222b" -> "integral"
   "\u00d7" -> "times"
   "\u00f7" -> "divide"
   "\u00b1" -> "plus_minus"
   "\u2260" -> "neq"
   "\u2264" -> "leq"
   "\u2265" -> "geq"
   "\u2208" -> "in"
   "\u2209" -> "notin"
   "\u2282" -> "subset"
   "\u2286" -> "subseteq"
   ```
   If the glyph is not in the reverse table, the parser raises
   `MathSymlinkNormaliseError` with the offending codepoint and source line —
   no silent passthrough. Unknown glyphs get added to the reverse table
   deliberately, not by accident.
7. Slug the final tail with `canonical_slug` so case/whitespace variations
   collapse (`symbol::Pi` and `symbol::pi` become the same ref).

The parser attaches both forms to the row:

```python
@dataclass(frozen=True)
class MathMeaningStarRow:
    ...
    symlink_refs_raw:  tuple[str, ...]   # as written in the source file
    symlink_refs_norm: tuple[str, ...]   # after §4.1.1 normalisation
    ...
```

Tests pin both fields on the fixture rows so future shape adapters cannot
drop the normalised form without failing CI.

### 4.2 Resolution pipeline

```
resolve_symlink_ref(ref_norm: str) -> str  # returns canonical star_id or raises
```

**Input contract**: the resolver only ever receives the normalised form
produced by §4.1.1 (`letter::a`, `symbol::pi`, `constant::e`, `concept::foo`).
Raw dialects (`star.letter.a`, literal glyph tails) are never passed in —
the driver uses `row.symlink_refs_norm`, not `row.symlink_refs_raw`, when
calling the resolver. A ref that does not match one of the four namespace
prefixes raises `MathSymlinkResolveError` immediately.

Order of attempts (fail means fall through to next):

1. **Letter form** — `letter::<x>` where `<x>` is a single Latin letter →
   `canonical_char_star_id(letter)`.
2. **Constant semantic alias** — `constant::<name>` →
   `math_semantic_aliases.CONSTANT_ALIASES[name]`.
3. **Math symbol semantic alias** — `symbol::<name>` →
   `math_semantic_aliases.SYMBOL_ALIASES[name]`:
   ```
   "plus":     math_symbol_star_id("+")
   "minus":    math_symbol_star_id("-")
   "times":    math_symbol_star_id("\u00d7")
   "divide":   math_symbol_star_id("\u00f7")
   "equal":    math_symbol_star_id("=")
   "sqrt":     math_symbol_star_id("\u221a")
   "sum":      math_symbol_star_id("\u2211")
   "product":  math_symbol_star_id("\u220f")
   "integral": math_symbol_star_id("\u222b")
   "pi":       math_symbol_star_id("\u03c0")
   "infinity": math_symbol_star_id("\u221e")
   "leq":      math_symbol_star_id("\u2264")
   "geq":      math_symbol_star_id("\u2265")
   "neq":      math_symbol_star_id("\u2260")
   "in":       math_symbol_star_id("\u2208")
   "notin":    math_symbol_star_id("\u2209")
   "subset":   math_symbol_star_id("\u2282")
   "subseteq": math_symbol_star_id("\u2286")
   "plus_minus": math_symbol_star_id("\u00b1")
   ```
   Unicode literals for clarity — Codex please keep the actual source using
   `"\u221a"`-style escapes to keep the file ASCII-safe.

   The alias table is hand-maintained. Unknown aliases do NOT auto-coin new
   stars — they raise `MathSymlinkResolveError`.
4. **Semantic concept fallback** — `concept::<slug>` resolves via
   `CanonicalLookup.exists(kind="meaning_star", key=f"concept_{slug}")`
   and returns the registered star_id. Missing concepts raise
   `MathSymlinkResolveError`. Forward refs within the same batch are
   tolerated because the driver runs the symlink pass *after* the
   meaning-star pass (see §6, S42, three-pass driver).

All four resolvers hit an in-process cache keyed by the raw ref string so the
same symlink across thousands of stars pays one Qdrant round-trip.

### 4.3 Allowlist escape hatch

`knowledge3d/ingestion/math_symlink_allowlist.txt` follows the Batch 7 pattern:
one ref per line, `#` comments allowed, trimmed and matched literally against
the raw symlink ref before resolution is attempted. A ref on the allowlist is
recorded as `symlink_pending` (see §7) and never counted against the integrity
gate.

---

## 5. Phase 7.A.1 Seed Audit

### 5.1 Problem

Every HS math star symlinks into Phase 7.A.1 (math symbols, letters,
constants). If the canonical registry is missing any of these targets,
Cluster 1 ingestion fails fast. Today we do not know whether the Phase 7.A.1
seed covers every alias §4.2 references.

### 5.2 Audit slice

`scripts/ingest_phase7a1_seed_audit.py`:

- Reads `math_semantic_aliases.SYMBOL_ALIASES` and `CONSTANT_ALIASES`.
- For each `(alias_name, expected_star_id)` pair, calls
  `CanonicalLookup.star_id_exists(expected_star_id)`.
- Emits a report at `/K3D/Knowledge3D.local/reports/batch8_phase7a1_audit.json`:
  ```json
  {
    "checked": 37,
    "present": 33,
    "missing": [
      {"alias": "plus_minus", "expected": "math_sym_u00b1", "reason": "not_in_canonical"},
      ...
    ],
    "timestamp": "..."
  }
  ```
- If any entry is missing, the script exits non-zero AND prints the exact
  `scripts/seed_batch8_canonical_math_aliases.py` command line Daniel should
  rerun to backfill the gap. It does NOT auto-register Galaxy rows — the
  seed script only patches the canonical Qdrant alias overlay.

### 5.3 Backfill path (out of scope for Batch 8 code; in scope for Batch 8 runbook)

If the audit reports missing symbols, the sequence is:

```
python scripts/seed_batch8_canonical_math_aliases.py
python scripts/ingest_phase7a1_seed_audit.py   # must now exit zero
python scripts/ingest_hs_math_cluster1.py --dry-run
python scripts/ingest_hs_math_cluster1.py --write
```

Batch 8 ships the audit script and documents the runbook. It does not ship
backfill content.

---

## 6. Slices

### S39 — Phase 7.A.1 seed audit

**File:** `scripts/ingest_phase7a1_seed_audit.py`
**Depends on:** nothing outside Batch 7 infrastructure
**Test:** `tests/test_batch8_phase7a1_seed_audit.py`

- Defines the 37-entry alias expectation (from §4.2).
- Reads each via `CanonicalLookup.star_id_exists(...)`.
- Writes the report and exits non-zero on any miss.
- Test uses a fake `CanonicalLookup` that returns known-present / known-missing
  to exercise both branches without touching Qdrant.

### S40 — HS math parser module

**File:** `knowledge3d/ingestion/hs_math_parser.py`
**Depends on:** `canonical_lookup.canonical_slug`
**Test:** `tests/test_batch8_hs_math_parser.py`

- Immutable dataclass:
  ```python
  @dataclass(frozen=True)
  class MathMeaningStarRow:
      canonical_id_raw:   str                 # unnormalised source form
      category:           str                 # formula | identity | theorem | rule | concept | method
      slug:               str                 # canonical_slug of leaf
      is_a:               tuple[str, ...]     # parent refs (strings, unresolved — see §13.2)
      rpn_sketch_raw:     str                 # full RPN sketch as plain text
      symlink_refs_raw:   tuple[str, ...]     # as written in the source file
      symlink_refs_norm:  tuple[str, ...]     # after §4.1.1 normalisation; same length as _raw
      surface_forms:      dict[str, str]      # language → text, restricted to 9 canonical langs
      saudades:           bool
      source_file:        str                 # basename of TEMP/KIMI_MATH_HS_CLUSTER*.md
      source_line:        int                 # 1-indexed line of the opening header
  ```
  Invariant: `len(symlink_refs_raw) == len(symlink_refs_norm)`. Index `i`
  in one corresponds to index `i` in the other.
- Three shape adapters:
  - `parse_cluster1_bullets(text: str, *, source_file: str) -> list[MathMeaningStarRow]`
  - `parse_cluster2_json(text: str, *, source_file: str) -> list[MathMeaningStarRow]`
  - `parse_cluster3_hybrid(text: str, *, source_file: str) -> list[MathMeaningStarRow]`
- Dispatcher:
  - `parse_hs_math_file(path: Path, *, shape: str | None = None) -> list[MathMeaningStarRow]`
  - If `shape` is None, select by filename pattern
    (`CLUSTER1` → bullets, `CLUSTER2` → json, `CLUSTER3` → hybrid).
- Cluster 1 adapter must handle the bullet shape Sub-Agent B emits:
  ```
  #### rule_pemdas
  - **canonical_id**: `rule_order_of_operations_pemdas`
  - **is_a**: `concept_arithmetic_precedence`
  - **rpn_sketch**: `[GALAXY_LOOKUP star.symbol.parenthesis][...]`
  - **symlinks**: `star.symbol.parenthesis, star.letter.a, ...`
  - **surface_forms**:
    - en: "PEMDAS (...)"
    - pt: "..."
    - ...
  - **saudades**: `true`
  ```
  Parser rules:
  - `####` headers open a new row.
  - `**canonical_id**` value is stripped of backticks/trailing punctuation
    before category detection.
  - Missing `canonical_id` for a header block is a parse error.
  - `symlinks` is split on `,` and each token stripped; backticks removed.
  - `surface_forms` is parsed as a 9-key dict; unknown language keys raise.
  - `saudades` accepts `true` / `false` / missing (defaults to false).
- Parser never touches `CanonicalLookup`. Pure text → dataclass.
- Cluster 2 and Cluster 3 adapters exist as stubs that raise
  `NotImplementedError("shape deferred to Batch 9/10")`. Tests pin that the
  dispatcher picks the right adapter by filename.

### S41 — Canonical ID normaliser + symlink resolver

**Files:**
- `knowledge3d/ingestion/math_canonical_id.py`
- `knowledge3d/ingestion/math_symlink_resolver.py`
- `knowledge3d/ingestion/math_semantic_aliases.py`
- `knowledge3d/ingestion/math_symlink_allowlist.txt`

**Test:** `tests/test_batch8_math_canonical_id.py`, `tests/test_batch8_math_symlink_resolver.py`

- `math_canonical_id.py`:
  ```python
  CATEGORIES: frozenset[str] = frozenset({
      "formula", "identity", "theorem", "rule", "concept", "method"
  })

  class MathCanonicalIdError(ValueError): ...

  def normalise_canonical_id(raw: str) -> tuple[str, str]: ...
      # returns (category, canonical_key)
  ```
- Handles all three dialects: plain `formula_foo`, `formula::foo`, `formula_foo`.
- Raises on empty input, unknown category, non-ASCII after slugging.
- Round-trips: `normalise_canonical_id(normalise_canonical_id(x)[1])` is
  idempotent.
- `math_symlink_resolver.py` exposes:
  ```python
  class MathSymlinkResolveError(LookupError): ...

  class MathSymlinkResolver:
      def __init__(self, canonical_lookup: CanonicalLookup, allowlist_path: Path | None = None):
          ...
      def resolve(self, ref: str) -> str | None:
          """Returns star_id on success, None if allowlisted, raises on miss."""
  ```
- Allowlist load is explicit; passing `allowlist_path=None` means no
  allowlist (strict mode used in tests).
- `math_semantic_aliases.py` is pure Python constants. No I/O. Exports
  three tables:
  ```python
  SYMBOL_ALIASES:    dict[str, str]   # name -> canonical star_id
  CONSTANT_ALIASES:  dict[str, str]   # name -> canonical star_id
  UNICODE_TO_NAME:   dict[str, str]   # single-char glyph -> alias name used by parser §4.1.1
  ```
  `UNICODE_TO_NAME` is the inverse index used by the parser when a source
  file embeds a literal glyph inside `symbol::√` / `constant::π`. The
  three tables must round-trip: for every key `name` in `SYMBOL_ALIASES`,
  any glyph that maps to `name` in `UNICODE_TO_NAME` must reach the same
  star_id via the resolver. Tests pin this round-trip explicitly.

### S42 — HS Math Cluster 1 ingestion driver

**File:** `scripts/ingest_hs_math_cluster1.py`
**Depends on:** S39, S40, S41
**Test:** `tests/test_batch8_hs_math_cluster1_ingestion.py` (mock),
`tests/test_batch8_hs_math_cluster1_qdrant.py` (Qdrant-gated)

- CLI:
  ```
  python scripts/ingest_hs_math_cluster1.py [--dry-run] [--write] [--source PATH]
  ```
- Default source: `TEMP/KIMI_MATH_HS_CLUSTER1_ARITHMETIC_ALGEBRA_2026-04-13.md`.
- `--dry-run` is the default; `--write` opts into Qdrant writes and requires
  `K3D_QDRANT_INTEGRATION=1` in the environment (fail fast otherwise).
- **Three-pass pipeline (Daniel-confirmed, 2026-04-14):**

  **Pass 0 — Parse + normalise (in-memory only, no Qdrant writes):**
  1. Parse file via `parse_hs_math_file`.
  2. For every row, call `normalise_canonical_id(row.canonical_id_raw)` and
     cache `(canonical_key, star_id, row)` into an in-memory list.
  3. For every symlink in `row.symlink_refs_norm`, attempt
     `MathSymlinkResolver.resolve(ref_norm)` in a **dry probe** — catch
     `MathSymlinkResolveError` and collect the misses. Note: concept-fallback
     refs (§4.2 item 4) that point to HS math stars in *this same batch*
     will be dry-probe misses now and resolved successfully in Pass 2.
     The driver must distinguish these from real misses by checking
     `ref_norm.startswith("concept::")` and verifying the concept slug
     appears in the Pass 0 row list. Forward-ref-to-same-batch misses
     are OK; everything else is fail-fast.
  4. If any symlink is a hard miss (not allowlisted, not forward-ref),
     exit non-zero **before touching Qdrant**. Print the list.

  **Pass 1 — Meaning-star registration (Qdrant writes):**
  1. For every cached `(canonical_key, star_id, row)`, call
     ```
     CanonicalLookup.register(
         kind="meaning_star",
         key=canonical_key,
         star_id=f"math_{canonical_key}",
         metadata={
             "context_id": 0,
             "ethical_trit": 0,
             "subkind": "math_hs_cluster1",
             "category": category,
             "is_a": list(row.is_a),
             "rpn_sketch": row.rpn_sketch_raw,
             "surface_forms": row.surface_forms,
             "saudades": row.saudades,
             "source_file": row.source_file,
             "source_line": row.source_line,
         },
     )
     ```
  2. No symlink writes yet. This pass must complete cleanly before Pass 2.

  **Pass 2 — Symlink registration (Qdrant writes):**
  1. For every `(row, ref_raw, ref_norm)` triple, call
     `MathSymlinkResolver.resolve(ref_norm)` **for real** (not a dry probe).
     Forward refs into the Pass 1 batch now resolve because the target
     meaning_star rows exist in Qdrant.
  2. Register every resolved edge:
     ```
     CanonicalLookup.register(
         kind="math_symlink",
         key=f"{math_star_id}::{resolved_star_id}",
         star_id=f"math_symlink_{canonical_entry_id('math_symlink', key)}",
         metadata={
             "source_star_id": math_star_id,
             "target_star_id": resolved_star_id,
             "raw_ref": raw_ref,
             "norm_ref": ref_norm,
             "bidirectional": True,
         },
     )
     ```
     Bidirectional rendering is implicit in the resolver consumers; both
     endpoints in the payload make reverse lookup one filter query.

  **Pass 3 — Confirmation read-back (Daniel-added, 2026-04-14):**
  This is a paranoid integrity re-check that catches partial writes,
  Qdrant eventual-consistency surprises, and silent failures inside
  `register()` that would otherwise only surface much later.

  1. For every `(canonical_key, star_id)` written in Pass 1, call
     `CanonicalLookup.star_id_exists(star_id)`. Every one must return
     True. Any False is an immediate non-zero exit with the offending
     star_id.
  2. For every symlink edge written in Pass 2, call
     `CanonicalLookup.exists(kind="math_symlink", key=edge_key)`. Every
     one must return True.
  3. Additionally, for every edge, verify that the `target_star_id`
     recorded in the edge payload itself resolves:
     `CanonicalLookup.star_id_exists(target_star_id)` must be True.
     This catches the case where Pass 2 wrote an edge pointing at a
     target that was deleted mid-run or was never actually present.
  4. Emit a Pass-3 confirmation record into the summary report (see
     below) with counts: `meaning_star_confirmed`,
     `math_symlink_confirmed`, `target_star_id_confirmed`, and a list
     of any confirmation misses.
  5. If any confirmation check fails, the driver exits non-zero with
     exit code `3` (distinct from Pass 0/1/2 exit codes) so the runbook
     can distinguish a read-back failure from a parse or write failure.

  **Summary report:**
  `/K3D/Knowledge3D.local/reports/batch8_cluster1_ingest.json` contains
  counts per category, dangling/allowlisted symlinks, forward-ref
  counts from Pass 0, parser line numbers of any skipped blocks, the
  Pass-3 confirmation record, and a digest of `source_file` mtime.

- Sovereignty guardrails:
  - No `numpy` / `cupy` / `scipy` / `sympy` / `cuda` imports.
  - `re` is allowed but only inside the parser and resolver modules; the
    driver itself must not use `re`.
  - All writes go through `CanonicalLookup.register`. No direct
    `client.upsert` calls outside canonical_lookup.

### S43 — RPN sketch annotation lexer

**File:** `knowledge3d/ingestion/rpn_sketch_lexer.py`
**Test:** `tests/test_batch8_rpn_sketch_lexer.py`

- Tokenises `rpn_sketch` strings into `(opcode, args)` pairs. Grammar is
  permissive because the source files use a documentary, not executable,
  form:
  ```
  [GALAXY_LOOKUP star.symbol.parenthesis][TPACK 5][STORE precedence_stack]
  ```
  and also the Cluster 2 JSON form:
  ```
  "RECALL letter::a"
  "TMUL"
  ```
- Lexer returns:
  ```python
  @dataclass(frozen=True)
  class RpnSketchToken:
      opcode: str
      args: tuple[str, ...]
      raw: str
  ```
- The lexer classifies each opcode against three buckets:
  - **real** — matches a name in `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`
    `__all__` list (loaded once at module import time from the exported names).
  - **documentary** — matches a known documentation-only name the HS math
    catalogues use (`STORE`, `RECALL`, `GALAXY_LOOKUP`, `OP_BRANCH`). These are
    allowed but flagged.
  - **unknown** — everything else.
- A coverage-report helper writes
  `/K3D/Knowledge3D.local/reports/batch8_rpn_sketch_coverage.json`
  after Cluster 1 ingestion with:
  ```
  {
    "rows_scanned": int,
    "opcode_histogram": {"TADD": 142, "GALAXY_LOOKUP": 380, ...},
    "real_opcode_hits": int,
    "documentary_hits": int,
    "unknown_opcodes": {"TDIV": 4, ...}
  }
  ```
- The lexer is NON-BLOCKING. Unknown opcodes do not fail the ingest. This
  file is diagnostic data that feeds a later decision about whether to
  extend real opcodes or rewrite the catalogue sketches.
- Lexer MUST NOT import from `knowledge3d/cranium/cuda/*` or `kernels/*`.
  The opcode whitelist load is a plain `importlib` read of
  `knowledge3d.cranium.ptx_runtime.rpn_opcodes` — same module Python
  tooling already imports.

### S44 — Idempotency + Qdrant handshake

**File:** `tests/test_batch8_hs_math_cluster1_qdrant.py`
**Gate:** `K3D_QDRANT_INTEGRATION=1`

- Marker: `pytest.mark.integration`.
- Setup skips if gate env is not set.
- Test:
  1. Run `scripts/ingest_hs_math_cluster1.py --write`.
  2. Record `k3d_canonical` collection `points_count`.
  3. Run it again.
  4. Assert `points_count` unchanged (idempotency via uuid5).
  5. Assert at least one `meaning_star` row has `subkind == "math_hs_cluster1"`.
  6. Assert at least one `math_symlink` row has
     `raw_ref.startswith("star.letter.")` or `raw_ref.startswith("letter::")`.
- Must not run in the default pytest profile.

---

## 7. Integrity Invariants

Each invariant ships with an explicit assertion in tests.

1. **Category whitelist closed.** Every persisted `meaning_star` row must have
   `metadata.category ∈ {formula, identity, theorem, rule, concept, method}`.
2. **No silent category promotion.** Rows whose raw canonical_id lacks a
   recognised prefix fail the driver. The parser must not invent a category.
3. **Normalised key round-trip.** For every row, the driver must satisfy
   `normalise_canonical_id(row.canonical_id_raw)[1] == canonical_key`.
4. **Symlink resolution is total.** Every unresolved symlink ref must either
   resolve, be on the allowlist, or cause the driver to exit non-zero. No
   row with silently dropped symlinks.
5. **Pass-3 confirmation is total.** Every `meaning_star` row written in
   Pass 1 must read back True under `star_id_exists`; every `math_symlink`
   row written in Pass 2 must read back True under `exists(kind, key)`;
   every symlink's recorded `target_star_id` must also read back True.
   A single False anywhere in Pass 3 is a fail-fast non-zero exit (code 3)
   with no retry. "We fail and fix", not "we retry and hide".
6. **Parse-time normalisation is total.** Every symlink in
   `row.symlink_refs_norm` must match one of the four allowed namespace
   prefixes (`letter::`, `symbol::`, `constant::`, `concept::`). Any
   leaked raw dialect (`star.letter.`, literal glyph) is a parser bug
   and a test failure.
7. **Idempotent re-run.** A second invocation leaves `points_count` stable.
8. **No hot-path contamination.** `grep -r "import numpy\|import cupy\|import scipy\|import sympy"`
   over all Batch 8 files must return empty. The sovereignty grep must also
   fail the test if any Batch 8 file imports from `knowledge3d.cranium.cuda`
   or `knowledge3d.cranium.kernels` or from any PTX runtime module except
   the documented `rpn_opcodes` whitelist read.

---

## 8. Metadata Policy (Batch 8 specific)

| Field | Cluster 1 default | Rationale |
|---|---|---|
| `context_id` | `0` | HS math is curriculum-universal; no region/era scoping |
| `ethical_trit` | `0` | HS math stars have no ethical dimension |
| `subkind` | `"math_hs_cluster1"` | distinguishes from Batch 7 `meaning_star` rows |
| `category` | one of the 6 | mirrors `normalise_canonical_id` output |
| `is_a` | list of raw strings | resolved in a later batch; store raw for now |
| `rpn_sketch` | raw text | diagnostic; executable form is a separate future migration |
| `surface_forms` | 9-language dict | all 9 canonical languages required; missing raises |
| `saudades` | bool | flag for cultural-untranslatability pass |

Cluster 2 and Cluster 3 will flip `subkind` to `"math_hs_cluster2"` /
`"math_hs_cluster3"` in Batches 9/10 without any other metadata change.

---

## 9. Test Plan

| Test file | Scope | Qdrant? |
|---|---|---|
| `test_batch8_hs_math_parser.py` | Cluster 1 bullet parser, dispatcher filename routing, Cluster 2/3 NotImplementedError | no |
| `test_batch8_math_canonical_id.py` | Dialect normalisation, whitelist, round-trip | no |
| `test_batch8_math_symlink_resolver.py` | Letter / symbol / constant / concept resolver, allowlist, cache | no (fake CanonicalLookup) |
| `test_batch8_rpn_sketch_lexer.py` | Tokeniser, real/documentary/unknown classification, histogram helper | no |
| `test_batch8_phase7a1_seed_audit.py` | Audit script missing/present branches | no (fake CanonicalLookup) |
| `test_batch8_hs_math_cluster1_ingestion.py` | Driver against a synthetic Cluster 1 fixture, no Qdrant, exercises all three passes via a fake `CanonicalLookup` with write + read-back | no |
| `test_batch8_hs_math_driver_three_pass.py` | Dedicated three-pass isolation test: Pass 0 dry-probe distinguishes forward-refs from hard misses; Pass 1 writes without symlinks; Pass 2 resolves forward-refs post-Pass-1; Pass 3 catches a planted silent miss and exits with code 3 | no |
| `test_batch8_hs_math_parse_time_normalisation.py` | Parser normalises every observed dialect (`star.letter.a`, `letter::a`, literal glyph tails) into the canonical `letter::` / `symbol::` / `constant::` / `concept::` form; `UNICODE_TO_NAME` round-trip | no |
| `test_batch8_hs_math_cluster1_qdrant.py` | Real Qdrant three-pass write + idempotency + Pass-3 confirmation | yes, `K3D_QDRANT_INTEGRATION=1` |

Sovereignty grep: add a single test
`tests/test_batch8_sovereignty_grep.py` that globs all Batch 8 files and
asserts none import forbidden modules. Same pattern Batch 7 used.

Benchmark non-regression: Batch 8 is ingestion-path only and cannot affect
any Batch 5/6 benchmark. Codex need not re-run the full regression suite, but
must run `pytest -q tests/test_batch5_* tests/test_batch6_* tests/test_batch7_*`
to prove nothing regressed in prior batches, since the shared `CanonicalLookup`
metadata schema is touched by new subkinds.

---

## 10. Runbook

Fresh environment. Assume Batch 7 has landed.

```bash
# 1. Audit Phase 7.A.1 seed coverage.
python scripts/ingest_phase7a1_seed_audit.py
# -> /K3D/Knowledge3D.local/reports/batch8_phase7a1_audit.json

# 2. If audit exits non-zero, backfill and repeat.
python scripts/seed_batch8_canonical_math_aliases.py
python scripts/ingest_phase7a1_seed_audit.py   # must exit zero

# 3. Dry-run Cluster 1 ingest.
python scripts/ingest_hs_math_cluster1.py --dry-run
# -> console summary, no Qdrant writes

# 4. Actual ingest.
K3D_QDRANT_INTEGRATION=1 python scripts/ingest_hs_math_cluster1.py --write
# -> /K3D/Knowledge3D.local/reports/batch8_cluster1_ingest.json

# 5. Handshake test.
K3D_QDRANT_INTEGRATION=1 pytest -q tests/test_batch8_hs_math_cluster1_qdrant.py
```

---

## 11. §9.1 Reasoning Taxonomy Shortfall (Deferred)

Batch 7 landed only 27 `meaning_star` rows because two of the four source
files had incomplete canonical star tables:

- `TEMP/KIMI_KNOWLEDGE_AML_AND_SOLVERS_2026-04-13.md` — no parseable canonical
  star table in this checkout.
- `TEMP/KIMI_KNOWLEDGE_AUTOMATED_REASONING_2026-04-13.md` — truncated mid-table.

This does not block Batch 8. HS Math Cluster 1 is self-contained and does not
depend on reasoning-taxonomy stars being fully populated. However, before the
full §9.1 wave can be considered complete, these two source files need their
tables regenerated / completed. Suggested follow-up ticket:

> **Ticket: Batch 7.5 — §9.1 source file regeneration**
>
> Dispatch `mcp__ollama-specialists__kimi_swarm` (think=True) to regenerate
> the canonical star tables for `KIMI_KNOWLEDGE_AML_AND_SOLVERS` and
> `KIMI_KNOWLEDGE_AUTOMATED_REASONING`. Input: the relevant sections of the
> master plan (§9.1 rows + §2 paradigm table). Output: full pipe-delimited
> canonical star tables matching the shape Batch 7's parser already handles.
> Rerun `scripts/ingest_reasoning_taxonomy.py` — the existing integrity gate
> and idempotency will do the rest.

Batch 7.5 runs in parallel to Batch 8. It is documentation/content work, not
infrastructure, so it does not gate the HS curriculum waves.

---

## 12. Runway

| Batch | Scope | Reuses |
|---|---|---|
| **Batch 8** | HS Math Cluster 1 + shared infra | Batch 7 registry, new parser/resolver/normaliser |
| Batch 9 | HS Math Cluster 2 (JSON shape) driver only | Batch 8 parser adapter + resolver + normaliser |
| Batch 10 | HS Math Cluster 3 (hybrid shape) driver only | same |
| Batch 11 | HS Natural + Earth/Space sciences | new parser adapter, same resolver, `context_id` still 0 |
| Batch 12 | HS Languages + Linguistics | first `context_id` per-region setter |
| Batch 13 | HS History + Civics + Economics | per-era `context_id` setter |
| Batch 14 | HS Humanities / Philosophy / Ethics | first `ethical_trit=+1` setter |
| Batch 15 | HS Applied / CS / Health / Psych / Sociology | same pattern |
| Batch 16 | Cross-cultural glue (saudades, calendars, units, proverbs) | same pattern |

Each batch after Batch 8 adds one shape adapter (if the file shape is new) or
one driver (if the shape already exists). The canonical registry, integrity
gate, allowlist discipline, idempotency model, and sovereignty grep all stay
constant from Batch 8 forward.

---

## 13. Locked Decisions (Daniel, 2026-04-14)

1. **Parse-time normalisation is the single source of truth** — confirmed.
   Rationale: "we need to symlink", and a consistent normalised form means
   the resolver has one dialect to handle, not six. All symlink refs pass
   through the §4.1.1 normaliser before the resolver sees them.
   `UNICODE_TO_NAME` handles literal-glyph tails (`symbol::√` → `symbol::sqrt`).
   Unknown glyphs raise deliberately — no silent passthrough. See §4.1.1
   and S40 dataclass (`symlink_refs_raw` + `symlink_refs_norm`).

2. **`is_a` resolution deferred to Batch 9** — confirmed. Batch 8 stores
   `is_a` as raw string tuples in `meaning_star.metadata.is_a`. Batch 9
   adds a parent-resolution pass that walks the stored `is_a` lists and
   writes `kind="meaning_hierarchy"` edges. No retroactive Batch 8 writes;
   Batch 9 just reads Batch 8's metadata and adds edges on top.

3. **Three-pass driver with confirmation read-back** — confirmed and
   expanded. Pass 0 parses + dry-probes symlinks (distinguishes
   forward-refs-to-same-batch from hard misses). Pass 1 writes all
   `meaning_star` rows. Pass 2 resolves symlinks for real (forward refs
   now succeed) and writes `math_symlink` rows. **Pass 3** (Daniel's
   addition) reads every Pass-1 and Pass-2 write back from Qdrant and
   verifies each one exists, plus verifies each symlink edge's
   `target_star_id` still resolves. Any Pass-3 miss is a fail-fast
   non-zero exit with distinct code 3 — no retry, no auto-repair.
   See §6 S42 pipeline and §7 invariant 5.

All three decisions are folded into the slice specs above. Codex may
proceed with S39 → S40 → S41 → S42 → S43 → S44 in that order. No
further architect sign-off required for this spec.

---

**End of spec.**
