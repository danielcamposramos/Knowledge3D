# Knowledge Proceduralizer Specification

**Version**: 1.0  
**Date**: April 6, 2026  
**Status**: Active ingestion specification  
**Scope**: Canonical ingestion-time proceduralization for benchmark, PDF, manifest, and text sources

---

## 1. Purpose

The **Knowledge Proceduralizer** is the canonical ingestion-time transmuter for K3D.

It converts raw source chunks into **symlink-first knowledge packets** that follow the foundational four-layer contract:

1. **Form**
2. **Meaning**
3. **Rules**
4. **Meta-Rules**

This component lives in the **Ingestion Stargate (Region 7)**. It is **not** part of runtime reasoning, benchmark solving, or the sovereign hot path.

Grounding specifications:

- [FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md](./FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md)
- [KNOWLEDGEVERSE_SPECIFICATION.md](./KNOWLEDGEVERSE_SPECIFICATION.md)
- [MEMORY_TABLET_SPECIFICATION.md](./MEMORY_TABLET_SPECIFICATION.md)

---

## 2. Core Invariants

### 2.1 Layer Discipline

All emitted knowledge must follow:

`Form -> Meaning -> Rules -> Meta-Rules`

The proceduralizer must never flatten these layers into one mixed record.

### 2.2 Save Information Principle

The proceduralizer must preserve the **symlink pattern**:

- reference canonical symbols instead of duplicating them
- reference existing meanings instead of restating them
- emit new ids only when the concept is genuinely absent

### 2.3 Meaning-Named Knowledge

Benchmark, dataset, source-file, page, and chunk names are forbidden in:

- `star_id`
- `proposed_star_id`
- route-family identifiers
- anti-pattern identifiers
- category names intended to become resident knowledge

All persistent ids must be named by **meaning**, not by their motivating benchmark.

### 2.4 Ingestion-Only Authority

The proceduralizer may use host-side libraries and Ollama transport because it operates in Region 7. It must never leak this logic into:

- sovereign runtime routing
- benchmark scoring
- hot-path reasoning

### 2.5 Bounded Failure Semantics

The proceduralizer must explicitly emit one ingestion action:

- `skip`
- `augment`
- `needs_context`
- `reject`

It must never silently fabricate missing structure.

---

## 3. Canonical Contract

### 3.1 Request

`ProceduralizerRequest`

- `source_kind`
- `source_id`
- `source_path`
- `domain_hint`
- `content`
- `context_chunks`
- `existing_ref_menu`
- `quality_profile`
- `ingest_mode`

### 3.2 Receipt

`ProceduralizerReceipt`

- `status`
- `provider`
- `model`
- `latency_ms`
- `request_hash`
- `response_hash`
- `raw_response_path`
- `schema_ok`
- `failure_code`
- `retry_after_utc`
- `parsed_bundle`

### 3.3 Bundle

`ProceduralizerBundle`

- `ingest_action`
- `knowledge_packets`

Each packet is one layer-aligned unit with:

- `layer_kind`
- `star_id` or `proposed_star_id`
- `meaning_class`
- `meaning_rpn`
- `summary`
- `domain`
- `surface_forms`
- `symbol_refs`
- `word_refs`
- `taxonomy_refs`
- `grammar_refs`
- `reality_refs`
- `meta_refs`
- `relationships`
- `route_contract` only for true Layer 3/4 route-capable packets
- `confidence`
- `needs_review`

---

## 4. Transport and Capture Boundary

The proceduralizer uses a **WINE-like thin capture boundary**:

1. Build stable request envelope
2. Submit through Ollama transport
3. Capture raw request and raw response
4. Emit a structured receipt
5. Hand the parsed bundle to deterministic Stargate-side normalization

This boundary is intentionally thin:

- transport and outer JSON decoding may remain host-side
- semantic normalization, duplicate detection, symlink preservation, taxonomy placement, and route-contract derivation belong after the boundary

Full GPU JSON parsing is explicitly out of scope for v1.

### 4.1 Context Discipline

The proceduralizer must clear model context between distinct sources.

- one source entry = one fresh proceduralizer request chain
- no conversational carryover from one document, benchmark entry, or manifest item into the next

When a single source exceeds the working context budget:

- chunk the source
- preserve overlap between adjacent chunks
- continue with a fresh request per chunk
- emit `needs_context` only when the overlapped continuation is still insufficient

This keeps ingestion deterministic, resumable, and bounded while preserving continuity within one source.

---

## 5. Default Model Policy

Canonical transport: **Ollama**

Default model profiles:

- `quality` → `glm-5:cloud`
- `audit_reasoning` → `kimi-k2-thinking:cloud`
- `long_context_engineering` → `qwen3.5:397b-cloud`
- `balanced_fallback` → `deepseek-v3.2:cloud`

Verified via `ollama show` on 2026-04-06:

- `qwen3.5:397b-cloud` context length = `262144`
- `kimi-k2-thinking:cloud` context length = `262144`
- `glm-5:cloud` context length = `202752`
- `deepseek-v3.2:cloud` context length = `163840`

The proceduralizer defaults intentionally cap working `num_ctx` below those maxima, because ingestion prompts are bounded and lower working context is the better latency/cost default for batch augmentation.

Selection rationale:

- official Ollama pages position `glm-5:cloud` as a strong reasoning and agentic model for complex systems engineering and long-horizon tasks
- `kimi-k2-thinking:cloud` remains the audit model for deeper multi-step reasoning
- `qwen3.5:397b-cloud` keeps the long-context slot because it exposes the largest verified local context window in this set
- a bounded one-prompt proceduralizer smoke on 2026-04-06 returned schema-clean JSON only from `glm-5:cloud` under the shipped strict-JSON options and timeout budget

Default option policy:

- low temperature
- bounded `num_predict`
- bounded `num_ctx`
- `think=false` for strict JSON reliability

These defaults are intentional, not accidental:

- cloud models expose much larger maximum context windows, but the proceduralizer uses bounded per-source requests
- oversized sources are chunked with overlap
- lower working `num_ctx` reduces latency and plan burn for long ingestion runs
- the long-context profile is reserved for sources that justify it

---

## 6. Source-Specific Behavior

### 6.1 PDF

PDF ingestion must use **one proceduralizer pass per page/chunk**, not a separate classifier prompt plus augmenter prompt.

Expected outcomes:

- bibliography / metadata pages → `skip`
- incomplete fragments → `needs_context`
- knowledge pages → `augment`

The previous per-document and per-page resume behavior remains mandatory:

- stage each processed page
- allow interruption without losing prior work
- resume from the last completed page by default

### 6.2 Benchmark-Derived Knowledge

Benchmark augmentation must use the same canonical request/receipt path for text-based sources.

ARC/game-like sources may retain their specialized structural loader while text/math/question sources are proceduralized through the canonical engine.

### 6.3 Manifest/Text Sources

Manifest and plain text ingestion must use the same request contract and capture boundary as PDF and benchmark-derived sources.

All ingest surfaces must honor bounded stop behavior when Ollama plan usage is exhausted:

- stop after the current receipt is written
- keep all staged execution artifacts
- emit `retry_after_utc = now + 5 hours + 1 minute`
- return a non-success exit status reserved for retryable stop conditions

---

## 7. Acceptance Criteria

The proceduralizer is correct only if:

- packets respect the 4-layer contract
- lower-layer duplication is replaced by refs
- benchmark names do not leak into resident ids
- receipts are always written even on timeout or invalid JSON
- execution artifacts are written before cleanup
- plan-limit exhaustion stops cleanly with a 5h01m retry timestamp
- context is reset between sources and overlapped within oversized single-source chunk chains
- emitted payload rows ingest through the existing sovereign payload consumer

---

## 8. Implementation Notes

- Use the proceduralizer as the canonical augmentation engine
- keep existing scripts as thin callers
- keep Ollama as the canonical transport
- do not introduce runtime fallback logic
- do not bypass sovereign validation when converting bundles into payload rows
