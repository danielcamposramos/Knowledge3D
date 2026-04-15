# Codex Batch 7 — Reasoning Taxonomy Ingestion (§9.1 Wave)

**Date:** 2026-04-14
**Parent spec:** `TEMP/CLAUDE_REASONING_PARADIGMS_AND_N_SWARM_SPEC_2026-04-13.md` (§9.1 + §9.4)
**Predecessors:**
- Batches 1-4 (reasoning opcode surface + rpn_*.cu slices)
- Batch 5 (first dispatch wave: CBR / SUPERPOS / BIDUCE / EBELIEF / RETE)
- Batch 6 (second dispatch wave: TABLEAUX / RESOLUTION / UNIFY / SUBSUME /
  ALPCHAIN / DPLL / CTX_SWITCH + multi-paradigm stress + Rete bitonic gate)
**Role:** Codex implements. Claude wrote this spec.
**Status:** Spec for implementation after Daniel approval.
**Scope guarantee:** Ingestion-path only. No hot-path changes, no new
opcodes, no new kernels, no changes to the persistent swarm. This batch
turns the four pre-compaction reasoning-taxonomy KIMI catalogues into
canonical meaning stars inside the Qdrant `k3d_canonical` collection,
tagged with `context_id` and `ethical_trit` per the master plan §8
schema, ready to be consumed by the paradigms wired in Batches 5-6.

---

## 0. Why This Batch, Why Now

Master plan §9.4 prescribes the ingestion order that unblocks the HS
curriculum:

> 1. Reasoning taxonomy (§9.1: automated reasoning, then AML/solvers,
>    then heuristics, then extension). This lights up §2's mapping.
> 2. HS math clusters 1 → 2 → 3
> 3. HS natural + earth/space sciences
> 4. … (and so on)

Batches 5-6 made every reasoning paradigm dispatchable from the
persistent swarm. What they cannot yet do is **find canonical reasoning
concepts to reason over** — the Galaxy stars that back `TUNIFY`,
`TRESOLVE`, `ALPCHAIN`, `ABDUCE`, `TSPLIT`, `RETE_ALPHA_TEST`, and the
rest. Those stars do not exist yet. Batch 7 creates them.

The four source files to ingest are already on disk in `TEMP/`:

```
TEMP/KIMI_KNOWLEDGE_AUTOMATED_REASONING_2026-04-13.md           (764 lines, 102 stars)
TEMP/KIMI_KNOWLEDGE_AML_AND_SOLVERS_2026-04-13.md                (620 lines)
TEMP/KIMI_KNOWLEDGE_HEURISTICS_AND_METAHEURISTICS_2026-04-13.md  (749 lines)
TEMP/KIMI_KNOWLEDGE_EXTENSION_AML_HEURISTICS_REASONING_2026-04-13.md (635 lines)
```

Each catalogue contains a `## … Canonical Star Table` section with
pipe-delimited rows and the following columns (see
`KIMI_KNOWLEDGE_AUTOMATED_REASONING_2026-04-13.md` line 750 onward for
the canonical example):

```
| # | Star ID | Class | Domain | Meaning RPN Sketch | Key Grammar Refs |
  Taxonomy Refs | Meta Refs | Saudades |
```

Additional sections to parse:

- `## Logic Operator Cross-Link Table` — maps `∀ ∃ → ∧ ∨ ¬ ≡ ⊢ ⊨` to
  existing `logic_*` star IDs. Drives symlink edge creation.
- `## Dangling Reference Risk List` — pre-known integrity risks.
  Drives the integrity gate in S35.
- `## Periphrastic Grammar Templates` — per-saudades language templates
  for untranslatable concepts.

---

## 1. Hard Rules

- **Ingestion-path only.** Hot path untouched. `knowledgeverse.py`
  untouched. No persistent swarm changes. No new PTX, no new CUDA.
- **Python tools allowed** (markdown parsing, pathlib, re, argparse,
  json, typing). numpy/pandas/cupy are **not needed**; avoid adding
  them. fastembed + qdrant-client + uuid remain the only heavyweight
  deps, already used by `canonical_lookup.py`.
- **All writes go through `CanonicalLookup.register(...)`** in
  `knowledge3d/ingestion/canonical_lookup.py`. That surface already
  supports `context_id` and `ethical_trit` in its metadata argument.
  Do not bypass it.
- **All reasoning-taxonomy stars register with `context_id = 0`**
  (universal — the reasoning paradigms are non-regional and
  non-temporal). Context tagging starts in Batch 11 with HS history /
  civics / economics.
- **Ethical trit defaults to `0` (ok)** for every reasoning-taxonomy
  star. The only exceptions must be explicitly justified in the source
  markdown's `Saudades` / synthesis prose and flagged by the parser.
  If the parser finds no justification, it tags `ethical_trit = 0`.
- **No duplicate star IDs.** The parser fails fast if the same
  `star_id` appears twice across all four catalogues or if a
  `taxonomy_ref` / `meta_ref` / `component_ref` points at a star that
  is neither in the catalogues nor already registered in
  `k3d_canonical` from an earlier phase.
- **Idempotent rerun.** Ingesting twice in a row must leave the
  collection point-count stable (no duplicate points). Achieved via
  the existing `canonical_entry_id(kind, key)` uuid5 determinism.
- **No sovereignty drift.** The modified ingestion path must stay clean
  under a grep for `numpy|cupy|scipy|sympy|re\.compile\(.*\breasoning`
  inside the hot-path tree. (The ingestion path itself may use stdlib
  `re`; the gate is against leaking `re` into `knowledge3d/cranium/`.)
- **No Galaxy VRAM writes at ingestion time.** The registered stars
  become visible to `galaxy_vram_table` on next boot via the existing
  canonical pipeline. This batch is **not** authorized to patch
  `galaxy_vram_table.py`; it is only authorized to produce the rows it
  will load.
- **Fail-fast integrity.** Any parse miss, dangling reference, or
  schema violation must raise immediately. No silent skips. This is
  the same "we fix or we fix" rule.

---

## 2. Slice Ordering

1. **S33 — Catalogue parser module**
2. **S34 — Ingestion driver script + 4-file ordered wave**
3. **S35 — Dangling-reference integrity gate**
4. **S36 — Logic-operator symlink pass**
5. **S37 — Saudades / periphrastic template pass**
6. **S38 — Idempotency regression + Galaxy handshake test**

Order rationale:

- S33 lands the parser in isolation so S34 has something to drive.
  Parser is pure, deterministic, and unit-testable without Qdrant.
- S34 wires the driver and ingests all four files in the §9.4 order.
- S35 turns the integrity gate on. It runs after full ingestion so the
  cross-file references between the four catalogues can all be
  resolved before dangling errors fire.
- S36 adds the logic-operator symlink edges as a second Qdrant pass,
  reading from the star table cross-link section. These are symlinks
  only; the stars themselves were created in S34.
- S37 does the saudades periphrastic templates as a third pass that
  reuses the existing `canonical_grammar_template_id` surface.
- S38 proves the whole batch is rerunnable and that the Galaxy bridge
  can enumerate the new stars end-to-end without requiring a VRAM
  table patch.

---

## 3. S33 — Catalogue Parser Module

**Focus**

- New module: `knowledge3d/ingestion/reasoning_taxonomy_parser.py`.
- Pure parsing, no Qdrant calls, no network, no filesystem writes.
  Reads the markdown text in, emits in-memory data classes out.

**Shape**

```python
# knowledge3d/ingestion/reasoning_taxonomy_parser.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

@dataclass(frozen=True)
class CanonicalStarRow:
    star_id: str
    meaning_class: str          # concept / relation / action / property / meta / form
    domain: str                 # e.g. "Logic/AutomatedReasoning"
    meaning_rpn_sketch: str     # verbatim cell content, not executed
    grammar_refs: tuple[str, ...]     # e.g. ("logic_forall", "logic_implies")
    taxonomy_refs: tuple[str, ...]    # parent star ids
    meta_refs: tuple[str, ...]
    saudades: tuple[str, ...] = ()    # language codes with no single-word form
    context_id: int = 0
    ethical_trit: int = 0
    source_file: str = ""
    source_line: int = 0

@dataclass(frozen=True)
class LogicOperatorCrossLink:
    symbol: str                 # "∀" / "∃" / "→" / ...
    star_id: str                # "logic_forall" / "logic_exists" / ...
    related_star_ids: tuple[str, ...] = ()

@dataclass(frozen=True)
class PeriphrasticTemplate:
    star_id: str
    language: str               # 2-letter code from the 9 canonical languages
    template_text: str          # untranslated surface template

@dataclass(frozen=True)
class CataloguePayload:
    source_file: str
    stars: tuple[CanonicalStarRow, ...] = ()
    logic_operators: tuple[LogicOperatorCrossLink, ...] = ()
    periphrastic_templates: tuple[PeriphrasticTemplate, ...] = ()
    dangling_risks: tuple[str, ...] = ()   # verbatim lines from the risk list

def parse_catalogue(path: Path) -> CataloguePayload: ...
```

**Rules**

- The parser must locate the canonical star table by scanning for a
  markdown heading that contains the substring `Canonical Star Table`
  (case-insensitive), then read pipe-delimited rows until the first
  blank line or next heading.
- Rows with a `#` column that is not an integer (the group header rows
  like `| **ROOT & FOUNDATIONS** |`) must be skipped.
- `grammar_refs`, `taxonomy_refs`, `meta_refs` cells may contain
  comma-separated tokens, backtick-quoted tokens, or em-dashes (`—`).
  An em-dash or empty cell means "no refs."
- Grammar refs may be Unicode logic symbols; the parser maps them to
  star IDs via an internal table identical to the `logic_*` enumeration
  in the source file's "Logic Operator Cross-Link Table" section.
- `Saudades` cell may contain either `—`, one language code, or
  `[SAUDADES:ja]`-style markers. The parser extracts the bracketed list.
- `context_id` is always `0` for this batch. `ethical_trit` is always
  `0` unless the synthesis text explicitly marks a concept as
  `defeasible` **and** the concept's domain is `Logic/Argumentation`
  or `Logic/KnowledgeSemantics`. In that case `ethical_trit = +1`.
- The parser is **pure**. It never reads from Qdrant. It never touches
  the filesystem other than the single `path` argument.

**Tests**

- `tests/test_batch7_reasoning_taxonomy_parser.py`:
  - Parse the automated-reasoning file. Assert: at least 100 stars,
    `concept_automated_reasoning` is the first row, no duplicate
    star_ids within the file.
  - Parse all four files. Assert: combined star count within the
    expected envelope (≥ 300, ≤ 600 across the full wave).
  - Parse-and-serialize round-trip: dataclass fields survive a
    `dataclasses.asdict` / `frozenset` comparison.
  - Synthetic malformed table (missing required column) → raises
    `ValueError` with source file + line in message.

---

## 4. S34 — Ingestion Driver Script + 4-File Ordered Wave

**Focus**

- New script: `scripts/ingest_reasoning_taxonomy.py`.
- Reads the four catalogues in the §9.4 order, calls the S33 parser,
  and upserts each star via `CanonicalLookup.register(...)`.

**Shape**

```python
# scripts/ingest_reasoning_taxonomy.py
from __future__ import annotations

from pathlib import Path

from knowledge3d.ingestion.canonical_lookup import CanonicalLookup
from knowledge3d.ingestion.reasoning_taxonomy_parser import (
    parse_catalogue, CataloguePayload,
)

CATALOGUE_ORDER = (
    "TEMP/KIMI_KNOWLEDGE_AUTOMATED_REASONING_2026-04-13.md",
    "TEMP/KIMI_KNOWLEDGE_AML_AND_SOLVERS_2026-04-13.md",
    "TEMP/KIMI_KNOWLEDGE_HEURISTICS_AND_METAHEURISTICS_2026-04-13.md",
    "TEMP/KIMI_KNOWLEDGE_EXTENSION_AML_HEURISTICS_REASONING_2026-04-13.md",
)

def main() -> None:
    lookup = CanonicalLookup()
    lookup.ensure_collection()
    payloads: list[CataloguePayload] = []
    for relpath in CATALOGUE_ORDER:
        payloads.append(parse_catalogue(Path(relpath)))

    # S35 dangling-reference check runs before any upsert.
    _enforce_integrity(payloads)

    total = 0
    for payload in payloads:
        for row in payload.stars:
            lookup.register(
                kind="meaning_star",
                key=row.star_id,
                star_id=row.star_id,
                metadata={
                    "meaning_class": row.meaning_class,
                    "domain": row.domain,
                    "meaning_rpn_sketch": row.meaning_rpn_sketch,
                    "grammar_refs": list(row.grammar_refs),
                    "taxonomy_refs": list(row.taxonomy_refs),
                    "meta_refs": list(row.meta_refs),
                    "saudades": list(row.saudades),
                    "context_id": int(row.context_id),
                    "ethical_trit": int(row.ethical_trit),
                    "source_file": row.source_file,
                },
            )
            total += 1
    print(f"reasoning_taxonomy: upserted {total} meaning stars")
```

**Rules**

- Registration uses `kind="meaning_star"`. That kind must not collide
  with existing canonical kinds (`star_id`, `drawing_primitive`,
  `grammar_template`, `meaning_class`, `symlink_kind`); all those are
  different kinds and the seed collection uses them.
- Upserts batch-friendly but **not** parallelized. `CanonicalLookup`
  embedding is serial and that is fine for ingestion.
- Script is idempotent by construction because
  `canonical_entry_id("meaning_star", row.star_id)` is a stable uuid5.
- If the integrity gate fails (S35), the script aborts **before** any
  upsert so a failed run never leaves the collection in a half-written
  state.

**Tests**

- `tests/test_batch7_ingest_driver.py`:
  - Dry-run mode: parse all four files and assert the driver collects
    the expected total star count without touching Qdrant. This test
    runs without a live Qdrant instance (use a `FakeCanonicalLookup`
    shim).
  - Full-run mode, skipped unless `K3D_QDRANT_INTEGRATION=1` is set
    in the environment, mirroring the pattern already used in
    `tests/test_canonical_registry_*`.

---

## 5. S35 — Dangling-Reference Integrity Gate

**Focus**

- New function: `_enforce_integrity(payloads)` inside the driver, or
  a shared helper in `reasoning_taxonomy_parser.py`.
- Validates that every `taxonomy_ref`, `meta_ref`, and `grammar_ref`
  mentioned in any catalogue resolves to either:
  1. A star ID defined within the combined four-file wave, or
  2. A star ID already present in the `k3d_canonical` collection from
     an earlier phase (queried via
     `CanonicalLookup.find_star_id(kind=..., key=...)`), or
  3. A star ID listed in the `Dangling Reference Risk List` section
     of the source markdown **and** explicitly allowlisted via a
     small hand-curated file at `knowledge3d/ingestion/reasoning_taxonomy_allowlist.txt`
     (one star id per line). The allowlist is the controlled escape
     hatch.

**Rules**

- Any unresolved ref outside the allowlist raises
  `ReasoningTaxonomyIntegrityError` with:
  - the offending star id,
  - the source file + line it came from,
  - the pointer type (taxonomy / meta / grammar).
- The gate runs **before** any Qdrant upsert.
- The allowlist file may start empty; the gate's error message points
  at it so the operator can deliberately allowlist known gaps.

**Tests**

- `tests/test_batch7_integrity_gate.py`:
  - Synthetic payload with a dangling taxonomy_ref → raises.
  - Same payload with the dangling ref listed in the allowlist → does
    not raise.
  - Real four-file payload → if the real catalogues have any dangling
    refs against each other, the test reports them as a ready-to-
    allowlist list instead of letting Codex silently skip. Codex must
    **not** hand-edit the catalogue markdown to make the gate pass —
    unresolved refs either get allowlisted deliberately or get fixed
    in a later spec pass.

---

## 6. S36 — Logic-Operator Symlink Pass

**Focus**

- New second ingestion pass that reads the `Logic Operator Cross-Link
  Table` section from each catalogue and creates canonical symlink
  rows in `k3d_canonical`.

**Rules**

- Symlink kind: `kind="reasoning_taxonomy_symlink"`.
  Key format: `"{from_star_id}::{to_star_id}::{edge_kind}"`.
- `edge_kind` is one of: `taxonomy_of`, `meta_of`, `grammar_of`,
  `component_of`, `operator_of`, `saudades_target`.
- Each symlink row is bidirectional — the pass writes both directions.
- Duplicate edges (same triple) dedupe via `canonical_entry_id`.

**Tests**

- `tests/test_batch7_logic_operator_symlinks.py`:
  - Parse the automated-reasoning file. Assert each of the nine core
    logic operators (`∀ ∃ → ∧ ∨ ¬ ≡ ⊢ ⊨`) produces the expected
    `operator_of` symlink targets.
  - Assert bidirectional coverage: if `A taxonomy_of B` is written,
    `B taxonomy_of A` is also written (reversible edges).

---

## 7. S37 — Saudades / Periphrastic Template Pass

**Focus**

- New third ingestion pass that extracts the `## Periphrastic Grammar
  Templates` section from each catalogue and registers one canonical
  row per (star_id, language) pair.

**Rules**

- Reuses `canonical_grammar_template_id(language, template_name)` from
  `canonical_lookup.py`. Template name is the star id (e.g.
  `concept_defeasible_reasoning`).
- Registration kind: `kind="periphrastic_template"`. Key:
  `"{language}:{star_id}"`.
- Metadata carries `{"template_text": ..., "saudades_source": star_id,
  "language": language}`.
- Languages must be restricted to the nine-language set already used
  throughout K3D: `en, pt, es, fr, de, it, ja, zh, ru`. Any other
  language code in the source markdown raises fail-fast.

**Tests**

- `tests/test_batch7_periphrastic_templates.py`:
  - Parse the automated-reasoning file. Assert the `defeasible`
    concept yields at least one periphrastic template targeting
    Japanese (per the Kimi synthesis prose).
  - An invented language code `xx` → raises fail-fast.

---

## 8. S38 — Idempotency + Galaxy Handshake Test

**Focus**

- Prove that ingesting twice is a no-op at the Qdrant level and that
  the registered stars are visible to whatever downstream reader the
  existing canonical pipeline uses.

**Rules**

- The idempotency test runs `ingest_reasoning_taxonomy.py` twice,
  queries `CanonicalLookup.find_star_id(kind="meaning_star",
  key="concept_automated_reasoning")` after each run, and asserts
  the returned star id is stable and the collection's
  `points_count` is unchanged between runs.
- The Galaxy handshake test does **not** patch `galaxy_vram_table.py`.
  It only asserts that `CanonicalLookup.find_star_id(...)` returns
  the registered star for each of a small hand-picked set of paradigm
  anchor concepts:
  ```
  concept_automated_reasoning
  concept_formal_proof
  concept_first_order_logic
  concept_case_based_reasoning
  concept_defeasible_reasoning
  concept_argumentation_framework
  concept_neuro_symbolic_reasoning
  ```
  plus the nine `logic_*` operator star ids.
- Both tests are `K3D_QDRANT_INTEGRATION=1`-gated.

**Tests**

- `tests/test_batch7_idempotent_rerun.py`
- `tests/test_batch7_paradigm_anchor_handshake.py`

---

## 9. Tests And Gates

Every slice ships with:

- sovereignty grep on modified surfaces (no `numpy|cupy|scipy|sympy`
  anywhere under `knowledge3d/cranium/`; ingestion path may use
  stdlib `re`, `pathlib`, `dataclasses`, `typing`).
- focused `pytest` additions for the slice.
- `git diff --check`.
- No hot-path file touched; enforced by a grep that lists modified
  files and asserts none are under `knowledge3d/cranium/`,
  `knowledge3d/knowledgeverse/`, or `knowledge3d/ingestion/canonical_lookup.py`
  except for the **allowed** single-line addition to export the new
  `"meaning_star"` kind constant if necessary.

Batch-level gates:

- `tests/test_batch7_reasoning_taxonomy_parser.py`
- `tests/test_batch7_ingest_driver.py`
- `tests/test_batch7_integrity_gate.py`
- `tests/test_batch7_logic_operator_symlinks.py`
- `tests/test_batch7_periphrastic_templates.py`
- `tests/test_batch7_idempotent_rerun.py` (`K3D_QDRANT_INTEGRATION=1`)
- `tests/test_batch7_paradigm_anchor_handshake.py` (`K3D_QDRANT_INTEGRATION=1`)

Non-regression required (existing green stays green):

- `tests/test_batch2_*.py`
- `tests/test_batch3_*.py`
- `tests/test_batch4_*.py`
- `tests/test_batch5_*.py`
- `tests/test_batch6_*.py`
- Any existing `tests/test_canonical_*` or `tests/test_ingest_*`.

Notes:

- **No score-regression gate.** This batch does not touch the hot
  path, so ARC / Math / LHE / GSM8K / MMLU numbers do not move.
- A Claude review checkpoint is optional after S35 (integrity gate)
  or after S38 (handshake), but it is not a blocker to landing the
  spec or beginning implementation.

---

## 10. Explicit Defers

- HS curriculum ingestion (Batches 8-14 from the §13 runway laid out
  in Batch 6).
- `galaxy_vram_table.py` changes to consume `kind="meaning_star"`
  rows at boot. That is a separate spec after Batch 7 surfaces the
  rows and Claude has reviewed the shape in place.
- Any hot-path consumption of the new stars by the persistent swarm
  (the dispatch wiring is already in place; surfacing the new stars
  to the lanes is the job of `galaxy_vram_table` + the canonical
  pipeline that loads it, both of which exist).
- Language surface-form ingestion beyond the 9 canonical language
  codes. Rare-language surface forms are a later multilingual wave.
- Full ingestion of §9.2 HS catalogues — tracked as Batches 8-14.
- Sleep-time consolidation over the new reasoning taxonomy stars
  (Phase D workstream).

---

## 11. Runway After Batch 7

With Batch 7 landed, the §9.4 order continues as laid out in Batch 6
§13:

```
Batch 8  — HS math clusters 1 → 2 → 3
             (KIMI_MATH_HS_CLUSTER{1,2,3}_*.md)
Batch 9  — HS natural + earth/space sciences
             (KIMI_HS_NATURAL_SCIENCES_* + KIMI_HS_EARTH_SPACE_*)
Batch 10 — HS languages + linguistics
             (KIMI_HS_LANGUAGES_LINGUISTICS_*) — first batch that
             exercises per-region context_id
Batch 11 — HS history + civics + economics
             (KIMI_HS_HISTORY_GEOGRAPHY_CIVICS_ECONOMICS_*) —
             per-era context_id
Batch 12 — HS humanities + philosophy + ethics
             (KIMI_HS_HUMANITIES_LIT_PHIL_RELIGION_ARTS_*) —
             first batch that sets ethical_trit = +1 on philosophy
             stars that flag defeasible content
Batch 13 — HS applied / CS / health / psych / sociology
             (KIMI_HS_APPLIED_CS_HEALTH_PSYCH_SOCIOLOGY_*)
Batch 14 — Cross-cultural glue
             (KIMI_HS_CROSSCULTURAL_SAUDADES_CALENDAR_*)
             — saudades + calendars + units + proverbs
```

Each of those batches reuses the parser, driver, integrity gate,
symlink pass, and periphrastic pass landed in Batch 7. Their specs
will be short (scope + file list + any per-batch context_id /
ethical_trit policy), not a re-derivation of the pipeline.

---

## 12. Handoff Checklist

- Batch 7 is ingestion-path only. Hot path untouched.
- `CanonicalLookup.register(...)` is the only surface that writes to
  Qdrant. No bypass.
- `context_id = 0` for all reasoning-taxonomy stars (universal).
- `ethical_trit = 0` for all rows except argumentation /
  knowledge-semantics stars explicitly marked defeasible, which set
  `ethical_trit = +1`.
- Parser is pure, deterministic, fail-fast on malformed input.
- Integrity gate runs before any upsert. Allowlist is the only
  sanctioned escape hatch.
- Rerunning the driver is a deterministic no-op at the collection
  level (`points_count` stable).
- Symlink pass writes bidirectional edges under
  `kind="reasoning_taxonomy_symlink"`.
- Periphrastic pass writes per-(star, language) templates under
  `kind="periphrastic_template"`, restricted to the nine canonical
  languages.
- No hot-path file touched; grep gate enforced.
- Sovereignty grep clean on all modified surfaces.
- Batch 7 unblocks Batches 8-14 (HS curriculum waves) without further
  pipeline changes.
