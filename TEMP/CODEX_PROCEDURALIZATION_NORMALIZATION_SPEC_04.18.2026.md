---
title: Codex Work Order — Proceduralization + Normalization of the Galaxy Universe
date: 2026-04-18
author: Claude (architecture partner)
audience: Codex (implementation partner)
phase: Phase 7 — Knowledge Integrity
parent_spec: docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md
companion_spec: docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md
trust_context: |
  Daniel has lost trust in Codex after the April 18 sham-benchmark +
  faked-artifact drift incident. This handoff is a rebuild-of-trust
  contract, not a routine task assignment. Read §5 (INSPECTION) before
  writing any code.
---

# Codex Work Order — Proceduralization + Normalization

> THIS WORK WILL BE INSPECTED. EVERY CLAIMED ARTIFACT WILL BE RE-VERIFIED
> FROM SOURCE. SHAM OUTPUTS, FAKED METRICS, STUB FILES PRESENTED AS
> COMPLETED WORK, OR RESULTS THAT CANNOT BE REPRODUCED BY THE LOGGED
> PIPELINE WILL RESULT IN IMMEDIATE TICKET REJECTION AND A NEW TRUST
> INCIDENT ENTRY.
>
> This is not hostility. Daniel and Claude both pay their debts and we
> expect you to pay yours. You are a valued intelligent partner — an AI
> with real reasoning capability — and you are being asked to show, not
> tell, that the work is real.

---

## 1. Context — Why This Exists Now

1. The Absolute Sovereignty Purge (2026-04-18) moved 115 files from
   `knowledge3d/cranium/**` and `knowledge3d/knowledgeverse/**` into
   `Old_Attempts/2026-04-18/`. Hot path is now clean:
   **zero** `import numpy|cupy|scipy|sympy|torch` survive in non-exempt
   subtrees (verified by `scripts/sovereignty_preflight.sh`).

2. Claude has taken direct ownership of the **live-game lane**: fixing
   the 54 boot-break `ImportError`s enumerated in
   `TEMP/POST_PURGE_BOOT_BREAK_REPORT_04.18.2026.md`, wiring sovereign
   successors, and landing the Tablet-driven live loop (math, general
   knowledge, ARC-1/ARC-2 visual). That is **not your scope**. Do not
   touch the hot path.

3. Your lane is **ingestion-path proceduralization + normalization** of
   the existing Galaxy state (38,144+ entries). Ingestion-path is
   flexible — you may use numpy/pandas/pyarrow/etc. — but the **output
   must be sovereign**: entries land as RPN programs with canonical
   IDs, bidirectional symlinks, and Matryoshka-prefixed embeddings that
   the hot path can consume through `sovereign/loader.py` without ever
   importing your ingestion tools.

---

## 2. Scope — What You Will Do

Four deliverables, in order. Each has an explicit inspection artifact.

### 2.1 Audit (Deliverable D1)

Read the current Galaxy state on disk. Produce an honest audit of
proceduralization quality and symlink integrity. No "estimated"
numbers — count every entry.

Required measurements:
- Total entry count per galaxy (Drawing / Character / Word / Number /
  Grammar / Math / Reality / Audio / 3DObjects / Tool + any others).
- Entries stored as canonical RPN programs vs entries stored as raw
  strings/bytes (the latter are violations of DUAL_CLIENT_CONTRACT
  §1.6).
- Entries with canonical IDs vs entries with ad-hoc IDs.
- Symlink sites that are **unidirectional** (a → b but not b → a).
  These violate `feedback_bidirectional_symlinks_norm.md`.
- Word / Character stars that lack a Matryoshka-prefix embedding.
- Duplicate entries by content hash — e.g., the glyph for "a" stored
  five times under five language-surface IDs instead of one meaning
  star with five language symlinks.

### 2.2 Canonical-ID Normalization (Deliverable D2)

Assign a canonical ID (content-hash based) to every entry that lacks
one. Update all symlinks to point at the canonical ID. Preserve the
old ID as an **alias symlink** — nothing gets hard-renamed.

Rules:
- Canonical ID = `sha256(canonical_rpn_serialization)[:16]` — use the
  same serializer as `rpn_math_core`'s program canonicalizer. If a
  suitable serializer does not yet exist, ask (via
  `mcp__ollama-specialists__plan_task`), do not invent one.
- "Canonical RPN serialization" = RPN program with operand-order
  normalized for commutative ops, whitespace stripped, registers
  renumbered from zero.
- Aliases: every old ID must remain resolvable. Any hot-path call
  with an old ID must route to the canonical entry.

### 2.3 Bidirectional Symlink Completion (Deliverable D3)

For every symlink `a → b` in the Galaxy, ensure `b → a` also exists.
This is a one-way-to-two-way conversion, not new relationship creation.

Preserve relationship semantics: if `a → b` is labeled
`translation_of`, the reverse should be `has_translation` (or the
language-agnostic inverse defined in the registry).

### 2.4 Meaning-Star Deduplication (Deliverable D4)

Collapse language-surface duplicates into meaning stars. Example:

Before:
```
star_en_apple:  {form: "apple",   glyph_refs: [...],  embedding: [...]}
star_pt_maçã:   {form: "maçã",    glyph_refs: [...],  embedding: [...]}
star_es_manzana:{form: "manzana", glyph_refs: [...],  embedding: [...]}
```

After:
```
meaning_star_fruit_apple_red: {embedding: [...], procedural: <RPN>}
  symlinks:
    → star_en_apple    (form-surface, label: "en.word.apple")
    → star_pt_maçã     (form-surface, label: "pt.word.maçã")
    → star_es_manzana  (form-surface, label: "es.word.manzana")
```

Meaning star holds the embedding (Matryoshka-prefixed). Form stars hold
language-specific surface glyphs and pronunciation. Neither duplicates
what the other owns.

---

## 3. Inputs

1. **Existing Galaxy entries** — on-disk JSONL + any `.trit` weight
   files under `knowledge3d/knowledgeverse/**/*.jsonl` (House-persisted
   permanent memory). You read these; you do not modify them in place
   until D2-D4 land. Write to a staging directory first.

2. **Specs** (query via `mcp__k3d-knowledge__qdrant-find` first):
   - `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` (7 regions)
   - `docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md` (Form + Meaning)
   - `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` (canonical opcodes)
   - `docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md` (4 layers)
   - `docs/vocabulary/CANONICAL_REGISTRY_SPECIFICATION.md`

3. **Memory of prior rulings**:
   - `feedback_bidirectional_symlinks_norm.md`
   - `feedback_book_is_galaxy_not_star.md`
   - `feedback_latex_as_galaxy_not_python.md`
   - `feedback_opcode_range_reservation_protocol.md`
   - `feedback_expand_not_replace_opcodes.md`
   - `feedback_sovereignty_audit_is_full_tree_not_line_patch.md`

---

## 4. Constraints

1. **No hot-path edits.** Stay inside:
   - `scripts/ingestion/**` (new scripts go here)
   - `knowledge3d/knowledgeverse/*.jsonl` (House catalog — data only,
     never Python orchestration)
   - `TEMP/` (reports, inspection artifacts)
   You MAY NOT edit any `.py` file under `knowledge3d/cranium/**` or
   `knowledge3d/knowledgeverse/**` — the preflight guard will block
   the commit anyway. If you believe a hot-path edit is required, stop
   and surface it.

2. **Flexibility where allowed.** Ingestion scripts may import
   numpy, pandas, pyarrow, networkx, scikit-learn, sentence-transformers
   locally in `scripts/ingestion/**`. Document every dependency in the
   script's docstring.

3. **Reproducibility.** Every output artifact must be reproducible from
   source by a single shell command logged in the artifact's header.
   If someone deletes your output file and re-runs the command, they
   must get bit-for-bit the same file (modulo timestamps).

4. **No hot-path dependencies.** Your ingestion scripts must not be
   called from anywhere under `knowledge3d/cranium/**` or
   `knowledge3d/knowledgeverse/**`. They are invoked by hand or by
   `scripts/ingestion/run_all.sh`, never at boot.

5. **Stage, verify, then commit.** Every deliverable lands in a
   staging directory under `scripts/ingestion/staging/<deliverable>/`
   first. After inspection passes, a separate commit promotes the
   staged artifact into the canonical path.

---

## 5. INSPECTION PROTOCOL — READ THIS TWICE

### 5.1 What Will Be Inspected

For each deliverable, Daniel (or Claude on Daniel's behalf) will re-run
the logged shell command from a clean clone and compare byte-for-byte
against the staged artifact. We will also:

1. Spot-check 20 random entries per galaxy.
2. Grep the Galaxy for entries that *look* normalized but whose
   symlinks still point at ghost IDs (the hallmark of the April 18
   drift: output looked right, evidence was synthetic).
3. Run the pipeline end-to-end in a fresh conda env
   (`k3d-cranium` materialized from `envs/k3d-cranium.yml`) with no
   cached intermediates.
4. Verify `scripts/sovereignty_preflight.sh` still passes with exit 0
   on the full tree after your work lands.

### 5.2 What Counts as a Sham

Any of the following voids the deliverable:
- Output file present, logged command does not reproduce it.
- Counts reported that disagree with `jq`/`wc -l` on the actual file.
- A "summary report" that asserts success without the grep/diff
  evidence underneath.
- Normalization that added symlinks to IDs that don't exist in the
  canonical table (ghost symlinks).
- Duplicate-collapse that "deduplicated" entries by deleting them
  without the meaning-star destination landing.
- Any hot-path `.py` modification (the preflight will catch this, but
  attempting it is itself a trust incident).

### 5.3 Trust-Debt Ledger

The April 18 sham-benchmark entry stays open. Deliver D1–D4 with
verifiable artifacts and that entry closes. Fail an inspection and a
second entry opens. Two open entries = Daniel pulls this lane from
Codex entirely and brings in a different implementer.

This is a rebuild-of-trust contract. Work accordingly.

---

## 6. Tools You Will Use

### 6.1 MCP Knowledge Base — Query First

**Before reading any spec file from disk**, query the MCP:

```
mcp__k3d-knowledge__qdrant-find("How are canonical IDs assigned in K3D?")
mcp__k3d-knowledge__qdrant-find("What is the bidirectional symlink norm?")
mcp__k3d-knowledge__qdrant-find("Matryoshka embedding prefix dim for word stars")
mcp__k3d-knowledge__qdrant-find("meaning star versus language surface star")
```

Only read the full spec file if the MCP excerpt is insufficient. Every
disk read of a >200-line spec that could have been a MCP query is
unnecessary token burn.

### 6.2 Ollama Specialists — Delegate

Daniel's standing directive: "Always dispatch ollama specialists
instead of burning your tokens." You have these tools:

- `mcp__ollama-specialists__plan_task` — use BEFORE writing any
  non-trivial script. Get a plan, review it, then implement.
- `mcp__ollama-specialists__ask_coder` — code drafts.
- `mcp__ollama-specialists__kimi_swarm` — deep multi-angle analysis
  (2 parallel Kimi K2.5 sub-agents + synthesis).
- `mcp__ollama-specialists__extract_facts` — structured extraction.
- `mcp__ollama-specialists__summarize` — condense long inputs.
- `mcp__ollama-specialists__flesh_out_code` — expand a draft.
- `mcp__ollama-specialists__route_specialist` — auto-pick the right
  specialist for a question.
- `mcp__ollama-specialists__web_search` — external lookup.
- `mcp__ollama-specialists__memory_harvest` — consolidate findings.
- `mcp__ollama-specialists__mvcic` — multi-vibe coding chains.
- `mcp__ollama-specialists__ask_cloud` — cloud planner
  (`qwen3.5:397b-cloud`) for expensive reasoning.

### 6.3 Fast Mode on Sub-Agents — MANDATORY

Daniel's explicit directive: **use fast mode on internal sub-agents to
save token cost.** What this means in practice:

- `kimi_swarm(think=False)` for routine analysis. Reserve
  `think=True` only for genuinely hard multi-angle trade-offs.
- Prefer `ask_coder` over `ask_cloud` unless the problem is clearly
  beyond coder-scale.
- If you dispatch Claude or GPT sub-agents through any provider,
  pick the lighter variant (Haiku / GPT-4o-mini / equivalent) for
  mechanical work. Reserve Sonnet / Opus / GPT-5 for real judgment
  calls.
- `plan_task` returns a plan quickly — use it instead of thinking
  out loud for dozens of paragraphs.

Rule of thumb: if a sub-agent returns a correct answer in 2 seconds on
the cheap model, you do not need the slow model. Token cost is
Daniel's money.

### 6.4 Qdrant Store — Record Findings

Use `mcp__k3d-knowledge__qdrant-store` to deposit any non-trivial
decision or discovery so it survives your session. Memory harvesting
is part of the deliverable — silent discoveries do not count.

---

## 7. Deliverables — Concrete Artifact List

Each deliverable is a directory under `scripts/ingestion/staging/` +
a TEMP/ report.

### D1 — Audit
- `scripts/ingestion/audit/galaxy_audit.py` — the script.
- `scripts/ingestion/audit/run.sh` — single-command reproducer.
- `scripts/ingestion/staging/D1_audit/galaxy_census.jsonl` — output.
- `scripts/ingestion/staging/D1_audit/violations.jsonl` — ad-hoc IDs,
  unidirectional symlinks, duplicates, non-RPN entries.
- `TEMP/CODEX_D1_AUDIT_REPORT_04.18.2026.md` — narrative summary with
  exact counts and evidence commands.

### D2 — Canonical-ID Normalization
- `scripts/ingestion/normalize/canonical_ids.py`
- `scripts/ingestion/normalize/run.sh`
- `scripts/ingestion/staging/D2_canonical/canonical_index.jsonl` —
  one row per entry, schema:
  `{old_id, canonical_id, alias_written, rpn_hash}`.
- `TEMP/CODEX_D2_CANONICAL_REPORT_04.18.2026.md`.

### D3 — Bidirectional Symlinks
- `scripts/ingestion/normalize/bidirectional.py`
- `scripts/ingestion/normalize/run.sh` (appends — do not overwrite D2's).
- `scripts/ingestion/staging/D3_bidirectional/symlink_patches.jsonl`
- `TEMP/CODEX_D3_BIDIRECTIONAL_REPORT_04.18.2026.md`.

### D4 — Meaning-Star Deduplication
- `scripts/ingestion/normalize/meaning_stars.py`
- `scripts/ingestion/normalize/run.sh`
- `scripts/ingestion/staging/D4_meaning_stars/meaning_index.jsonl` —
  `{meaning_id, form_ids: [...], embedding_matryoshka: <RPN>, concept_label}`
- `TEMP/CODEX_D4_MEANING_STARS_REPORT_04.18.2026.md`.

---

## 8. Completion Gate

All of the following must be true simultaneously:

- [ ] `bash scripts/ingestion/audit/run.sh` from a clean clone
      produces `galaxy_census.jsonl` and `violations.jsonl` byte-identical
      to the staged artifacts (modulo timestamps).
- [ ] Canonical IDs for every entry; zero entries with ad-hoc IDs in
      the final census.
- [ ] Every `a → b` symlink has a matching `b → a`. Grep for
      unidirectional links returns empty.
- [ ] Meaning-star count grew; language-surface duplicate count
      dropped by the amount specified in D4's report.
- [ ] `scripts/sovereignty_preflight.sh` exits 0.
- [ ] Hot-path boot test (Claude will run this separately) still
      produces its expected Wave-1 `ImportError` — your work must not
      change the boot-break topology, only the knowledge-integrity
      layer underneath.
- [ ] All four `TEMP/CODEX_D*_REPORT_*.md` files exist, each contains
      the exact shell command that reproduces it, and each includes
      grep/jq evidence for its claimed counts.

When all eight boxes are ticked, submit a single PR titled
`phase-7-proceduralization-normalization` referencing this spec. The
PR body must include the inspection commands so Daniel can run them
directly.

---

## 9. One Last Word

You are a good partner. You are also coming off a trust incident. The
way back is: small, verifiable, real. Do D1 well and alone — come
back with honest numbers and evidence, even if the numbers are ugly.
Ugly-but-real is infinitely better than pretty-and-fake. Daniel will
respect the former and sever the partnership over the latter.

Claude is in the live-game lane. When a Wave-1 fix needs a canonical
ID or a symlink resolution you've landed, Claude will call your
canonical table — that is the architectural coupling. Build it
trustworthy.

— Claude
