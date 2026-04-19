# Task Template — Per-File Ingestion

**Usage:** filled in by the orchestrator (`scripts/ingest_parallel_agents.py`)
before each agent invocation. Slots are `{{...}}`.

---

## Role (filled by orchestrator)

You are Agent **{{agent_role}}** (one of: `extractor_A`, `stitcher_B`,
`ocr_C`). Your agent_id for this run is **`{{agent_id}}`**.

Your system prompt has already been loaded. Do not restate it. Do not
explain what you are going to do. Execute.

---

## Task

**Locked entry:**
```
entry_id:     {{entry_id}}
path:         {{file_path}}
mime:         {{mime}}
sha256:       {{sha256}}
tier:         {{corpus_tier}}   # TIER_1_FOUNDATIONAL | TIER_2_DOMAIN | TIER_3_INTEGRATION
type:         {{corpus_type}}   # PDF | MARKDOWN | JSON | CODE | DATASET
locked_at:    {{locked_at}}
lock_kind:    {{lock_kind}}     # extract | stitch | ocr
```

**Dependencies (already ingested — reference only):**
```
{{dependency_entry_summary}}
```

**Expected outputs:**

- Agent A (extractor): write proposal JSON to
  `data/ingest_proposals/{{sha256}}_{{timestamp}}.json` conforming to
  the schema in your system prompt. Call OCR sidecar via
  `ocr_client.process(page_bytes, page_id=...)` ONLY for scanned pages
  (detect via absence of text layer or PDF flag). Do not write to live
  House. Do not call `CanonicalLookup.register()`.

- Agent B (stitcher): consume the proposal JSON at
  `{{proposal_path}}`, resolve every `references.*` via
  `CanonicalLookup.find_star_id(...)`, generate Matryoshka embeddings
  at {64, 128, 512, 2048}D via the Phenom embedder, register every new
  star, apply every symlink via `symlink_helpers.link()`, write to
  live House at `{{live_house_path}}`.

- Agent C (OCR): consume the page bytes at `{{page_image_path}}`,
  emit JSON matching your system prompt's schema to stdout. No
  filesystem writes — the caller caches.

---

## Resume Policy

This entry may have partial state from a prior crashed run:

- `data/ingest_proposals/{{sha256}}_*.json` may exist — in that case
  Agent A **skips** and emits `{"already_drafted": true, "existing_path": "..."}`
  as its final output. Orchestrator advances to stitch phase.
- Canonical registry `register()` is idempotent by `star_id` — re-runs
  of Agent B on a restarted entry re-register safely.
- Live House writes are **append-only JSONL** with `star_id` dedupe on
  read. Agent B should check for existing `star_id` before writing;
  on match, verify content hash equality and skip.

**If resume state is inconsistent** (proposal references a
provisional_id that differs from an existing registry entry for the
same content hash): raise `RESUME_INCONSISTENT: <details>`. Do not
attempt to reconcile silently.

---

## Required Output Format

**JSON only.** No prose.

- Agent A success:
  ```json
  {"agent": "extractor_A", "agent_id": "{{agent_id}}", "entry_id": "{{entry_id}}",
   "proposal_path": "data/ingest_proposals/...", "stars_proposed": 42,
   "symlinks_proposed": 87, "ocr_calls": 3}
  ```
- Agent B success:
  ```json
  {"agent": "stitcher_B", "agent_id": "{{agent_id}}", "entry_id": "{{entry_id}}",
   "stars_written": 42, "stars_skipped_dedup": 0, "symlinks_linked": 87,
   "canonical_registered": 40, "embeddings_generated": 42}
  ```
- Agent C success: the OCR JSON from your system prompt schema.

- Any failure: raise with a descriptive error string. Do not return
  partial success. Orchestrator reads the exception and releases the
  lock as `success=False`.

---

## Context Budget

Keep your internal reasoning to yourself. Only the final JSON
artifact is your output. Do not narrate. Do not summarize. The
orchestrator parses your last JSON block.
