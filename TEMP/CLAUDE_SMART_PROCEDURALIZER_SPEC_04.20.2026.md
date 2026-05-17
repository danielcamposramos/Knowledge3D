---
date: 2026-04-20
author: Claude (architecture partner, Codex limit-locked)
status: SUPERSEDED by CLAUDE_SMART_PROCEDURALIZER_SPEC_V2_04.20.2026.md
superseded_why: |
  v1 misframed the proceduralizer as an internal Python pipeline.
  Daniel corrected: it is an external tool wired through Tablet WINE.
  v1 also invented entry-schema fields (`arg_keys`, `eval_program`,
  `meaning_hash`) that do not exist — the authoritative schema is
  `MeaningCentricStar` (`docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md §2.1`)
  with `meaning_rpn` + `star_id`. v1 also ignored the existing
  `knowledge_proceduralizer.py` + `proceduralizer_wine.py` that
  already do this work and need adaptation, not replacement.
  Do not implement v1. See v2.
scope: enrichment pipeline that turns dead Galaxy entries into executable RPN
related:
  - TEMP/CLAUDE_DATA_STATE_DIAGNOSTIC_04.20.2026.md (smoking-gun diagnosis)
  - TEMP/kimi_swarm_smart_proceduralizer_04.20.2026.md (full dual-perspective design)
  - docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §11
  - knowledge3d/knowledgeverse/knowledgeverse.py:5890–6020 (binder contract)
---

# Smart Proceduralizer Spec — Turn Dead Entries Into Executable RPN

## 1. Why (the smoking gun)

A 20k-entry sample of `/K3D/Knowledge3D.local/galaxies/Grammar.jsonl` confirms:

- 103,039 Grammar entries — **0** have `rule_strength`, `superior_to`, or
  `trust_weight` populated.
- Proceduralized math `rpn_program` strings are natural-language
  (`"number divide_by_two add original_number equals_total_sum"`), not
  executable (`"ARG_N 2 / ARG_N +"`).
- No `arg_keys`, no `eval_program` with `ARG_X` placeholders — the math
  template binder at [knowledgeverse.py:5931](../knowledge3d/knowledgeverse/knowledgeverse.py#L5931)
  cannot bind them, so `_math_match_allows_direct_eval` short-circuits
  and the entry is dead weight.
- 600 `supervision_answer` gold entries exist but are keyed by
  `problem_id`, not query text — the binder cannot match them via
  embedding.

The data-pipeline gap is the real bottleneck. Not fallback-widening on
the hot path. Daniel's directive: **use Ollama cloud LLMs (gpt-oss,
qwen3.5:397b-cloud, kimi-k2-thinking:cloud) to make the proceduralizer
smart, and expose the RPN opcode registry + call patterns via MCP so
future ingestion leverages them**.

## 2. Target contract (binder-compliant entry schema)

Every enriched entry MUST satisfy this shape. Fields marked `[*]` are
the ones today's entries lack.

```json
{
  "id": "gsm8k_train_0",
  "query_text": "If a number divided by 2 plus the original number equals the sum ...",
  "arg_keys": ["n"],                                   // [*] lowercase regex ^[a-z][a-z0-9_]*$
  "eval_program": "ARG_N 2 / ARG_N +",                 // [*] RPN with ARG_{KEY.upper()}
  "answer_eligible": true,
  "rule_strength": 0.92,                               // [*] 0.0-1.0
  "superior_to": ["gsm8k_train_legacy_0"],             // [*] IDs this supersedes
  "trust_weight": 0.85,                                // [*] 0.0-1.0
  "meaning_hash": "sha256_of_normalized_rpn[:16]",     // [*] dedup key
  "metadata": {
    "symlink": ["math_galaxy", "reality_galaxy"],      // existing (14,921 have it)
    "symlink_bidirectional": true,                     // [*] reverse pointer written on target
    "supervision_answer": "42",                        // existing (600 entries)
    "supervision_problem_id": "gsm8k_train_0",
    "query_hash": "sha256(canonical_query_text)",      // [*] resolves 600-entry mismatch
    "direct_eval": true,
    "enriched_by": "qwen3.5:397b-cloud",
    "enriched_at": "2026-04-20T18:30:00Z",
    "schema_version": 1
  }
}
```

Hard validation gates (fail-fast, no Python fallback on hot path):

| Gate | Check |
|------|-------|
| G1: arg_keys regex | every `k` matches `^[a-z][a-z0-9_]{0,15}$` |
| G2: placeholder coverage | `set(ARG_* in eval_program) == set(ARG_{k.upper()} for k in arg_keys)` |
| G3: opcode registry | every non-`ARG_*`, non-literal token exists in `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` |
| G4: stack simulation | final depth == 1, no underflow mid-program |
| G5: symbolic check | if `supervision_answer` present, symbolic eval with `ARG_X=σ_x` matches gold |
| G6: PTX lowering dry-run | `rpn_opcodes.lower_to_ptx(eval_program)` returns without raising |
| G7: registry-reserved range | if spec introduces a new opcode, §11 row must exist before write |

## 3. Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 0 — Preprocessing (CPU)                                   │
│  • Build supervision_index: canonical query_text → problem_id   │
│  • Compute input_hash(entry) for idempotency                    │
├─────────────────────────────────────────────────────────────────┤
│ Stage 1 — Classify & route (CPU)                                │
│  • grammar_simple | grammar_complex | math_nl | supervision_align│
│  • Cache hit on input_hash → skip                                │
├─────────────────────────────────────────────────────────────────┤
│ Stage 2 — Cloud enrichment (Ollama MCP @ :8502)                 │
│  • Inject opcode catalog via k3d-rpn-opcodes :8504 (new)        │
│  • Adaptive batching (grammar=25/call, math=2/call)             │
│  • Model routing matrix (§6)                                     │
├─────────────────────────────────────────────────────────────────┤
│ Stage 3 — Sovereign validation (G1-G7)                          │
│  • All seven gates in order; fail-fast with structured errors   │
├─────────────────────────────────────────────────────────────────┤
│ Stage 4 — Commit or repair                                       │
│  • PASS → write to <galaxy>.enriched.jsonl, update cache        │
│  • FAIL (repairable) → escalate model, retry (max 3)            │
│  • FAIL (fatal) → quarantine/manual_review/                     │
├─────────────────────────────────────────────────────────────────┤
│ Stage 5 — Bidirectional symlink write-back                      │
│  • For each enriched entry, append reverse symlink on target    │
│    meaning-star (never on surface form)                         │
├─────────────────────────────────────────────────────────────────┤
│ Stage 6 — Atomic rename                                          │
│  • <galaxy>.enriched.jsonl → <galaxy>.jsonl (fsync + rename)    │
└─────────────────────────────────────────────────────────────────┘
```

## 4. File layout (what Codex builds)

New files:
- `knowledge3d/tools/smart_proceduralizer/__init__.py`
- `knowledge3d/tools/smart_proceduralizer/pipeline.py` — main orchestrator
- `knowledge3d/tools/smart_proceduralizer/prompt.py` — prompt construction
- `knowledge3d/tools/smart_proceduralizer/validator.py` — G1-G7 gates
- `knowledge3d/tools/smart_proceduralizer/router.py` — model routing matrix
- `knowledge3d/tools/smart_proceduralizer/batcher.py` — adaptive batching
- `knowledge3d/tools/smart_proceduralizer/supervision_index.py` — query_hash lookup
- `knowledge3d/tools/smart_proceduralizer/cache.py` — SQLite idempotency
- `knowledge3d/tools/smart_proceduralizer/opcode_catalog.py` — registry loader
- `scripts/run_smart_proceduralizer.py` — CLI entry point
- `scripts/ingest_opcodes_to_qdrant.py` — new MCP resource ingestion
- `deploy/docker/k3d-rpn-mcp.run.sh` — new :8504 MCP container
- `tests/tools/test_smart_proceduralizer_validator.py` — gates unit tests
- `tests/tools/test_smart_proceduralizer_prompt_binding.py` — contract tests

Existing files to extend (not replace):
- `knowledge3d/tools/knowledge_proceduralizer.py` — it already uses
  `OllamaManager`. Add a `smart_mode: bool` flag that delegates to
  the new pipeline; keep legacy path behind the flag for rollback.
- `knowledge3d/ingestion/ollama_manager.py` — add `batch_chat()` helper
  if not present; keep single `chat()` untouched.
- `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` — add
  `lower_to_ptx(eval_program: str) -> bytes` dry-run for G6 (no
  execution, just AST lowering).
- `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` — §11 row reservation
  for any new opcodes emitted during enrichment (append-only).

## 5. LLM prompt chain

Per-entry user prompt (see `TEMP/kimi_swarm_smart_proceduralizer_04.20.2026.md`
for the full XML). Critical injection:

```xml
<OPCODE_CATALOG>
  <!-- Loaded at prompt-construction time from k3d-rpn-opcodes via MCP,
       so the LLM always sees the current authoritative list. -->
  <OP hex="0x0A" symbol="+" arity="2" stack="-1" ptx="add.f64"/>
  <OP hex="0x0B" symbol="-" arity="2" stack="-1" ptx="sub.f64"/>
  <OP hex="0x0C" symbol="*" arity="2" stack="-1" ptx="mul.f64"/>
  <OP hex="0x0D" symbol="/" arity="2" stack="-1" ptx="div.f64" guard="divisor!=0"/>
  <OP hex="0x0E" symbol="POW" arity="2" stack="-1"/>
  <OP hex="0x14" symbol="SQRT" arity="1" stack="0" guard=">=0"/>
  <!-- … full list injected from the MCP resource … -->
</OPCODE_CATALOG>

<BINDER_CONTRACT>
  <PLACEHOLDER>ARG_{KEY.upper()}</PLACEHOLDER>
  <REPLACE>literal_string_replace (see knowledgeverse.py:5931)</REPLACE>
  <ARG_KEY_REGEX>^[a-z][a-z0-9_]{0,15}$</ARG_KEY_REGEX>
</BINDER_CONTRACT>

<SYMLINK_RULES>
  <CANONICAL_ID>sha256("meaning:" + normalized_meaning)[:16]</CANONICAL_ID>
  <DIRECTION>bidirectional</DIRECTION>
  <TARGET>meaning-star, never surface form</TARGET>
</SYMLINK_RULES>

<SELF_CHECK>
  Before emitting JSON, simulate the RPN:
  1. Token walk with stack depth tracking; verify final == 1
  2. If supervision_answer present, substitute ARG_X=σ_x and verify
     symbolic equivalence to the gold
  Set static_check_passed accordingly. If false, explain in validation_notes.
</SELF_CHECK>
```

## 6. Model routing matrix

| Entry class | Primary | Escalation 1 | Escalation 2 |
|-------------|---------|--------------|--------------|
| grammar_simple (rule_strength + short surface) | `gpt-oss` (local) | `qwen3.5:397b-cloud` | `kimi-k2-thinking:cloud` |
| grammar_complex (superior_to graph edges) | `qwen3.5:397b-cloud` | `deepseek-v3.1:671b-cloud` | `kimi-k2:1t-cloud` |
| math_nl (NL RPN → executable) | `qwen3.5:397b-cloud` | `kimi-k2-thinking:cloud` | `kimi-k2:1t-cloud` |
| supervision_align (600 gold entries) | `kimi-k2-thinking:cloud` | manual_review/ | — |

Temperature: 0.2 primary, 0.1 repair, 0.0 final. Repair prompt echoes
the specific G1-G7 error back to the LLM with a hint (e.g.,
`STACK_UNDERFLOW on SWAP — insert DUP before SWAP`).

## 7. MCP resource update — new `k3d-rpn-opcodes` :8504

The architecture partner layer (Claude) + the LLM proceduralizer (cloud
model) must both be able to query the authoritative opcode list.
Today that lives only in a markdown file. Fix: expose it as a Qdrant
collection mirroring the `k3d-knowledge` / `k3d-ptx` pattern.

**Ingestion script** `scripts/ingest_opcodes_to_qdrant.py`:

- Source: regenerate via existing `scripts/inventory_opcodes.py` →
  `docs/opcodes_manifest.json` (one row per opcode).
- Payload fields per point:
  ```
  opcode_hex, opcode_name, symbol, arity, stack_effect,
  category (arithmetic|logic|stack|defeasibility|…),
  ptx_impl_path, ptx_instruction, semantics_md,
  example_eval_program, reserved_range_block, reservation_date,
  sovereignty_flags { hot_path_legal, ingestion_legal }
  ```
- Embedder: `all-MiniLM-L6-v2` (384-dim), Cosine, same as other
  collections.
- Collection name: `k3d_rpn_opcodes`.
- Idempotent upsert by `uuid5(NAMESPACE_OID, opcode_hex)`.

**Docker launcher** `deploy/docker/k3d-rpn-mcp.run.sh`:
- Mirror `k3d-ptx-mcp.run.sh`.
- Port `:8504`.
- Env: `COLLECTION_NAME=k3d_rpn_opcodes`,
  `TOOL_DESCRIPTION="Search the RPN opcode registry..."`.

**Claude config** `~/.claude.json`: add `k3d-rpn-opcodes` under
`mcpServers` with `qdrant-find`/`qdrant-store` allowlist.

**Regeneration hook** — add to `scripts/inventory_opcodes.py`: after
writing `docs/opcodes_manifest.json`, call
`scripts/ingest_opcodes_to_qdrant.py --upsert-changed` so the MCP
collection tracks registry changes automatically.

## 8. Opcode registry — append-only protocol

Per [feedback_opcode_range_reservation_protocol](../../home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_opcode_range_reservation_protocol.md):
the registry is the single source of truth. Pre-reserve blocks in
`RPN_DOMAIN_OPCODE_REGISTRY.md §11` BEFORE the proceduralizer emits
them.

If the pipeline encounters an LLM-proposed opcode not in §11:

1. G3 gate fails immediately.
2. Repair prompt asks LLM to rewrite using only registered opcodes.
3. If the concept is genuinely missing, quarantine the entry to
   `manual_review/new_opcode_candidates/` with context, and a human
   (or a follow-up architecture spec) must reserve a §11 row before
   the entry can be re-enriched. Never silently add opcodes.

Reserve a fresh block up-front for this pipeline's needs, e.g.
`0x300–0x33F = defeasibility_proceduralizer` — and commit the §11 row
in the same spec delivery.

## 9. Defeasibility field derivation

- `rule_strength`: initial value from LLM (0.0-1.0 confidence).
  After pipeline runs, a second pass boosts strength for rules whose
  `eval_program` resolves to `supervision_answer` across matched
  queries. Formula: `strength = base * (1 + 0.5 * success_rate)`,
  clipped to `[0, 1]`.
- `superior_to`: LLM identifies entries this one subsumes by semantic
  overlap. Validated by: for every `(a, b)` pair where `a.superior_to
  contains b`, `a` must have at least one property `b` lacks
  (narrower `arg_keys`, stricter guard, higher supervision_answer
  match rate). Reject pairs that don't.
- `trust_weight`: derived from source — `pdf_intelligent_augmentation`
  starts at 0.7, `benchmark_augmentation_*` starts at 0.9,
  `gpu_query_runtime` starts at 0.5. Sleep-time consolidation adjusts.

## 10. Idempotency / observability

- SQLite at `/K3D/Knowledge3D.local/state/enrichment_cache.sqlite`.
- Schema: `(input_hash BLOB PRIMARY KEY, schema_version INT,
  output_json JSON, ptx_hash BLOB, model_used TEXT, gates_passed TEXT,
  attempt_count INT, created_at TS)`.
- Resume: `run_smart_proceduralizer.py --resume` skips entries where
  `(input_hash, schema_version)` present with `gates_passed == 'all'`.
- Trace emission: every enriched entry writes a JSONL trace to
  `/K3D/Knowledge3D.local/traces/proceduralizer/<date>.jsonl` per
  [feedback_note_taking_everywhere](../../home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_note_taking_everywhere.md).
  Silence = bug.

## 11. Success criteria

Pipeline ships green when:

1. `python scripts/run_smart_proceduralizer.py
   --input /K3D/Knowledge3D.local/galaxies/proceduralized_gsm8k_train_10.jsonl`
   produces a `.enriched.jsonl` where **≥ 90%** of entries pass all of
   G1-G7.
2. After atomic rename, the existing GSM8K 10-q benchmark run shows
   materialization rise from 9/10 → **10/10**, and numeric accuracy
   rise at least 1 point (the template binder now has arg_keys +
   eval_program to bind against).
3. `k3d-rpn-opcodes` MCP at :8504 responds to
   `qdrant-find("division opcode")` with a hit citing `0x0D` and the
   example `eval_program`.
4. Re-running the pipeline on the same file is a no-op (cache hit rate
   100%).
5. Quarantine folder has a bounded count and every entry in it has a
   readable `reason` + `last_error` field.

Non-goals for this spec (deferred):
- Full re-enrichment of all 103k Grammar entries (start with math-
  relevant slice, then expand).
- Sovereign GPU-side validation of the enriched RPN at bind time
  (G6 is a CPU dry-run, not a device-side check).
- Multi-galaxy cross-symlink graph walk (Stage 5 only writes the
  single reverse pointer; graph-hop consolidation stays in
  sleep-time).

## 12. Handoff checklist for Codex

- [ ] Read this spec + `TEMP/kimi_swarm_smart_proceduralizer_04.20.2026.md`.
- [ ] Confirm `knowledge3d/tools/knowledge_proceduralizer.py` + its
      `PROCEDURALIZER_MODEL_PROFILES` are usable as the LLM gateway;
      extend rather than duplicate.
- [ ] Build `smart_proceduralizer/` subpackage.
- [ ] Reserve `0x300–0x33F` (or next free block) in
      `RPN_DOMAIN_OPCODE_REGISTRY.md §11` in the same commit.
- [ ] Stand up `k3d-rpn-opcodes` MCP at :8504; register in
      `~/.claude.json`.
- [ ] Run against `proceduralized_gsm8k_train_10.jsonl` (small, safe),
      validate §11 criteria, then expand.
- [ ] Emit a completion report `TEMP/CODEX_SMART_PROCEDURALIZER_*.md`
      with the enriched entry count per gate, cache hit rate, and
      benchmark delta.

## 13. What Claude will NOT do

Per [feedback_claude_never_runs_code](../../home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_claude_never_runs_code.md)
and CLAUDE.md §Role Definition: Claude writes specs, Codex implements.
This spec is the handoff. If Codex remains limit-locked past the
session ending 2026-04-20, a pilot session may execute Stage 0-1 +
Stage 7 (MCP resource) only — the LLM-driven Stages 2-4 cross the
coding boundary.
