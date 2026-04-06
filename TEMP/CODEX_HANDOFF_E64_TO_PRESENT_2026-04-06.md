# Codex Handoff: E.64 To Present

Date: 2026-04-06
Author: Codex
Audience: Claude / Daniel
Branch state: `main` is the live branch
Current pushed `HEAD`: `b82d66b5`

## Executive State

The system/runtime side is materially more stable than it was at `E.64`.

What is solved:
- the runtime result contract is flattened and typed
- canonical probes can run end to end through the real tablet/daemon boundary
- benchmark shutdown no longer times out in the same-live-instance tail
- the proceduralizer has a formal spec, a strict JSON contract, a WINE-style Ollama capture boundary, bounded retry semantics, and ordered/resumable PDF ingestion
- old generated ingest artifacts were audited and rejected instead of reused
- `main` now contains the current proceduralizer/runtime stack directly

What is not solved:
- benchmark knowledge quality is still too shallow
- `QUESTION`/`GENERAL` factual grounding is still not strong enough
- `GSM8K` / `MMLU` / `LHE` remain below gate
- the current encyclopedia ingest is still first-pass harvesting while live; a richer second-pass payload rebuild is now implemented but must be applied after the current live run finishes

## Since E.64: What Actually Happened

### 1. E.64 Artifact Audit And Rejection

The old proceduralizer-style outputs from the earlier benchmark relaunch were inspected and explicitly rejected as live knowledge source material.

Rejected characteristics:
- generic benchmark-shaped anchors
- weak semantics like `math_procedural_bridge`
- low-confidence / low-symlink rows
- poor fit for the 4-layer `Form -> Meaning -> Rules -> Meta-Rules` vocabulary contract

Relevant artifact root:
- `/K3D/Knowledge3D.local/results/e64_benchmark_relaunch_2026-04-03/payloads/`

Result:
- no attempt was made to salvage those payloads into the resident corpus
- later restart work used fresh canonical proceduralizer paths instead

### 2. E.69B Resident Route Contract Repair

Route metadata and materializer bridge repairs were applied so the resident corpus could rebuild cleanly again.

Main effects:
- explicit executor/materializer wiring for `MATH` and `QUESTION` meta-routers
- route-capable normalization merged override refs with existing refs instead of dropping support refs
- build-time family/closure audits started using finalized host stars instead of pre-finalization feed-source trits

Result:
- rebuild-side closure audits went green again
- but live probes still showed runtime materialization gaps:
  - `MATH` routed but emitted no materialized answer
  - `GRAMMAR` routed but emitted no materialized answer
  - `GAME_2D` routed but returned no typed action/grid

### 3. E.70 GPU-First Result Contract

The runtime boundary was repaired so the device-side typed packet became authoritative.

Implemented:
- flat typed runtime result packet
- no nested `task_result.task_result`
- typed answer fields for text / numeric / choice / grid / action
- route internals kept as diagnostics only
- benchmark-side and bridge-side answer derivation removed from the active path
- active benchmark boundary stopped scoring Python-derived fallback actions
- active daemon runtime path removed the specific NumPy helper work that was still on the hot-adjacent path

Representative code areas touched during that phase:
- `knowledge3d/bridge/headless_tablet.py`
- `knowledge3d/daemon/main.py`
- `benchmarks/arc_agi_3.py`
- `knowledge3d/knowledgeverse/knowledgeverse.py`
- `knowledge3d/knowledgeverse/sovereign_hot_path.py`

Operational result:
- canonical probes turned green on the contract side
- but Stage 1 remained materially red due to knowledge weakness, not runtime plumbing

Stage 1 artifact:
- `/K3D/Knowledge3D.local/results/e70_gpu_first_2026-04-05/benchmarks/stage1/logs/summary.execution.json`

### 4. E.70B Bounded Shutdown / SleepTime Tail De-Pythonization

The benchmark runner timeout was traced to shutdown, not execution.

Implemented:
- `Knowledgeverse.shutdown(profile="benchmark")`
- idempotent shutdown guard
- bounded benchmark-mode sleep flush
- compact `sovereign_sleep_delta.bin` persistence
- no full heavy checkpoint path during benchmark shutdown
- benchmark runner switched to the explicit bounded shutdown profile

Operational result:
- benchmark runner exits cleanly with `fast_exit`
- no manual kill required after artifact flush
- orchestration is no longer the main blocker

Artifacts:
- `/K3D/Knowledge3D.local/results/e70b_shutdown_2026-04-05/smoke_noop/summary.json`
- `/K3D/Knowledge3D.local/results/e70b_shutdown_2026-04-05/benchmarks/stage1/summary.json`

Stage 1 at that point:
- `arc3_local = 4/30`
- `mmlu = 3/10`
- `gsm8k = 1/10`
- `lhe = 0/10`

Conclusion at E.70B:
- system/run stability cleared
- remaining blocker is knowledge

### 5. E.71 Curated Foundational Knowledge Expansion

The first targeted knowledge wave focused on `MATH`, `QUESTION`, `GENERAL`, and supporting `GRAMMAR`.

Implemented:
- checked-in knowledge-gap inventory
- curated foundational builder additions
- route-capable and route-exempt packet normalization fixes
- meaning knowledge coverage audit
- stronger definition and question route wiring
- blocking of internal anti-pattern labels as final answers

Important artifacts:
- `/K3D/Knowledge3D.local/checkpoints/meaning_family_route_audit.json`
- `/K3D/Knowledge3D.local/checkpoints/meaning_route_closure_audit.json`
- `/K3D/Knowledge3D.local/checkpoints/meaning_knowledge_coverage_audit.json`
- `/K3D/Knowledge3D.local/results/e71_curated_math_question_2026-04-06/probes/canonical_probes.json`
- `/K3D/Knowledge3D.local/results/e71_curated_math_question_2026-04-06/benchmarks/subset_rerun2/summary.execution.json`

Observed state after E.71:
- rebuild-side coverage audits green
- `MATH`, `GRAMMAR`, and `GAME_2D` canonical probes correct
- `QUESTION` factual MCQ still wrong on the canonical France-capital probe (`Rome` instead of `Paris`)
- subset rerun improved only modestly:
  - `mmlu = 4/10`
  - `gsm8k = 1/10`
  - `lhe = 0/10`

Conclusion:
- runtime and route scaffolding improved
- resident factual/domain knowledge still too shallow

### 6. Knowledge Proceduralizer Spec + WINE Boundary

The proceduralizer was formalized as a Region 7 Ingestion Stargate component.

New canonical spec:
- `docs/vocabulary/KNOWLEDGE_PROCEDURALIZER_SPECIFICATION.md`

New core modules:
- `knowledge3d/ingestion/proceduralizer_contract.py`
- `knowledge3d/ingestion/proceduralizer_wine.py`
- `knowledge3d/knowledgeverse/proceduralizer_stargate.py`
- `knowledge3d/tools/knowledge_proceduralizer.py`

Key design decisions:
- single strict JSON packet-bundle contract
- Ollama remains canonical transport
- WINE-style request envelope -> model -> receipt -> parsed bundle -> deterministic stargate normalization
- context reset between sources
- overlap chunking within oversized sources
- plan-limit detection with `retry_after_utc = now + 5h01m`

Bounded model eval artifact:
- `/K3D/Knowledge3D.local/results/proceduralizer_model_eval_2026-04-06_direct/summary.execution.json`

Shipped cloud model profile outcome:
- `quality -> glm-5:cloud`
- `audit_reasoning -> kimi-k2-thinking:cloud`
- `long_context_engineering -> qwen3.5:397b-cloud`
- `balanced_fallback -> deepseek-v3.2:cloud`

Reason:
- under the shipped strict JSON boundary and bounded smoke, `glm-5:cloud` was the only schema-clean result

### 7. Old Artifact Reset + Spec-First Proceduralizer Restart

Old generated local proceduralizer artifacts were hard-reset from `Knowledge3D.local`, not from the repo.

Then a clean restart was done on real spec knowledge first.

Validated spec payloads:
- `docs/vocabulary/KNOWLEDGE_PROCEDURALIZER_SPECIFICATION.md`
- `docs/vocabulary/MATH_CORE_SPECIFICATION.md`

Artifacts:
- `/K3D/Knowledge3D.local/results/proceduralizer_restart_2026-04-06/payloads/spec_payload_kp/payload.jsonl`
- `/K3D/Knowledge3D.local/results/proceduralizer_restart_2026-04-06/payloads/spec_payload_math/payload.jsonl`
- `/K3D/Knowledge3D.local/results/proceduralizer_restart_2026-04-06/summaries/spec_payload_ingest_report.json`

Resident ingest result from that spec restart:
- `added = 13`
- `skipped = 0`
- `rejected_missing_route_metadata = 0`
- `rejected_benchmark_name_leakage = 0`

Representative resident rows confirmed:
- `concept_knowledge_proceduralizer`
- `rule_proceduralizer_reference_dedup`
- `rule_math_core_tier_hierarchy`
- `rule_modular_rpn_engine_constants`

### 8. Main-Only Migration, Old_Attempts Cleanup, Ordered PDF Ingest

The repo was moved back to main-only operation.

Main commits from that workstream now on `main`:
- `62f031a9` `Snapshot sovereign runtime and proceduralizer foundation`
- `c3ecd903` `Harden proceduralizer structured ingest gating`
- `e46e1892` `Canonicalize ordered PDF ingest and archive legacy path`
- `856bcde0` `Skip empty PDF pages in ordered ingest`
- `b82d66b5` `Deepen proceduralizer second-pass symlinkage`

Old path archival:
- superseded legacy PDF classifier / augmenter moved into `Old_Attempts/repo_archive/`

Live ingest path improvements:
- recursive PDF-only preflight
- OCR-needed listing
- exact ordered `--pdf-list`
- empty-page short-circuit before model calls
- resumable per-page stage writes

Canonical preflight/ingest docs updated:
- `docs/ingestion/CURRENT_STACK_COMMAND_MANUAL.md`
- `docs/ingestion/CURRENT_STACK_COMMANDS_QUICKREF.md`

### 9. Second-Pass Deep Symlinkage

After inspecting the live encyclopedia harvest quality, a deterministic second pass was added to deepen 4-layer symlinkage without rerunning model calls.

Current implementation:
- adds explicit foundational layer tagging
- enriches:
  - `taxonomy_refs`
  - `word_refs`
  - `symbol_refs`
  - `grammar_refs`
  - `reality_refs`
- uses:
  - batch-local staged rows
  - lightweight cached ID/token index from `galaxy_consolidated_latest.json`

Validation:
- `33 passed`

Current commit:
- `b82d66b5`

Current preview artifact:
- `/K3D/Knowledge3D.local/results/base_knowledge_ingest/01_encyclopedias/summaries/second_pass_quality_preview.json`

Important note:
- the currently running encyclopedia ingest process started before this patch
- its stage files are valid
- its final in-memory payload writer will still be the old code path
- after the live run finishes, one clean stage rebuild will apply the richer second pass to the full batch without rerunning model calls

## What Is Running Right Now

Live process snapshot taken during this handoff:
- PID: `1394153`
- command:
  - `/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/fundamental_ingest_pdfs.py --pdf-list /K3D/Knowledge3D.local/results/base_knowledge_ingest/01_encyclopedias/preflight/eligible_pdfs.txt --provider ollama --model-profile quality --capture-dir /K3D/Knowledge3D.local/results/base_knowledge_ingest/01_encyclopedias/captures --stage-dir /K3D/Knowledge3D.local/results/base_knowledge_ingest/01_encyclopedias/stages --payload-output /K3D/Knowledge3D.local/results/base_knowledge_ingest/01_encyclopedias/payloads/payload.jsonl --report-output /K3D/Knowledge3D.local/results/base_knowledge_ingest/01_encyclopedias/summaries/ingest_report.json --skip-sources-output /K3D/Knowledge3D.local/results/base_knowledge_ingest/01_encyclopedias/summaries/skipped_sources.jsonl`
- runtime at snapshot: `09:57:22`

Stage snapshot during this handoff:
- source root: `01_encyclopedias`
- current PDF: `/mnt/arquivos/0 ChatGPTs/DataBase/Encyclopedias/Encyclopedia of World History.pdf`
- staged page records: `654`
- rows harvested so far: `2008`
- decision counts:
  - `knowledge = 507`
  - `ambiguous = 60`
  - `non_knowledge = 87`
- last staged page snapshot:
  - `page_num = 654`
  - `total_pages = 3756`
  - decision = `ambiguous`
  - reason = `timeout`
  - provider = `ollama`
  - model = `glm-5:cloud`

Preflight for this root:
- `/K3D/Knowledge3D.local/results/base_knowledge_ingest/01_encyclopedias/preflight/summary.json`

Preflight result:
- PDFs found: `7`
- JSON sidecars ignored: `7`
- eligible PDFs: `5`
- OCR-needed PDFs: `2`

Skipped OCR-needed PDFs:
- `/mnt/arquivos/0 ChatGPTs/DataBase/Encyclopedias/how20things20work20encyclopedia20dk20publishing.pdf`
- `/mnt/arquivos/0 ChatGPTs/DataBase/Encyclopedias/pdfcoffee.com_encyclopedia-of-general-science-disarijibika-pdf-free.pdf`

## Current Enrichment Quality Snapshot

Second-pass preview snapshot currently on disk:
- `/K3D/Knowledge3D.local/results/base_knowledge_ingest/01_encyclopedias/summaries/second_pass_quality_preview.json`

Preview counts from the last flushed snapshot:
- `rows_total = 1893`
- `rows_with_taxonomy = 1893`
- `rows_with_word_refs = 1893`
- `rows_with_symbol_refs = 563`
- `rows_with_grammar_refs = 1702`
- `rows_with_reality_refs = 1893`

Representative enriched rows:
- `era_paleolithic`
  - taxonomy: `prehistory`, `high_school_world_history`
  - symbol: `notation_bce_dating`
  - reality links: `era_mesolithic`, `era_neolithic`, `prehistory`, etc.
- `fact_egypt_unification_menes`
  - reality links: `civilization_egypt`, `ruler_menes`, `event_egypt_unification`
- `pattern_empire_decline`
  - preserved as Layer 3 rule
  - taxonomy includes `concept_history`

Interpretation:
- the second pass materially improves structural linkage
- but this is still enrichment over first-pass harvested rows, not yet the full intended Ollama-driven deep semantic rewrite pass

## Current Strategic Conclusion

As of this handoff:

1. The system is no longer blocked by result-contract bugs or shutdown timeouts.
2. The benchmark gate is still red because knowledge remains thin.
3. The proceduralizer stack is now formalized and usable for real corpus feeding.
4. The live encyclopedia ingest is producing real signal and is worth continuing.
5. The newly implemented second-pass enrichment should be applied to the full encyclopedia batch after the live run finishes.
6. Daniel explicitly requested that a later second pass should also use Ollama, not only deterministic enrichment.

## Immediate Next Steps

### After The Current Encyclopedia Run Finishes

Do this in order:

1. Rebuild payload from the completed stage root so `b82d66b5` second-pass enrichment is applied to the entire harvested batch.
2. Inspect the rebuilt payload quality again.
3. If quality is still too shallow, add a model-driven second pass through Ollama on top of the enriched stage outputs.
4. Only then ingest that root into the resident corpus.
5. After `01_encyclopedias`, continue in the requested order to `02_default_libraries`.

### Model Guidance For The Requested Later Second Pass

Daniel explicitly asked that the later second pass should use Ollama too.

That was not implemented yet in this patch. The current second pass is deterministic.

Recommended next move after the live run:
- keep the deterministic second pass as the cheap structural baseline
- add an optional Ollama-based deepening pass that consumes:
  - the first-pass harvested rows
  - the deterministic enriched refs
  - the current resident base-layer context
- if more cloud models are needed, it is acceptable to pull them directly via:
  - `ollama pull model:cloud`

### Benchmark Posture

Do not spend more cycles on `arc3_local` / `GAME_2D` yet.

The next useful benchmark measurement after encyclopedia + deeper second pass is still:
- `gsm8k=10`
- `mmlu=10`
- `lhe=10`

## Key Artifacts To Review First

If Claude needs the fastest grounding path, read these in this order:

1. `docs/vocabulary/KNOWLEDGE_PROCEDURALIZER_SPECIFICATION.md`
2. `docs/briefings/ARCHITECTURE_BRIEFING.md`
3. `/K3D/Knowledge3D.local/results/e70b_shutdown_2026-04-05/benchmarks/stage1/summary.json`
4. `/K3D/Knowledge3D.local/results/e71_curated_math_question_2026-04-06/benchmarks/subset_rerun2/summary.execution.json`
5. `/K3D/Knowledge3D.local/results/base_knowledge_ingest/01_encyclopedias/preflight/summary.json`
6. `/K3D/Knowledge3D.local/results/base_knowledge_ingest/01_encyclopedias/stages/manifest.json`
7. `/K3D/Knowledge3D.local/results/base_knowledge_ingest/01_encyclopedias/summaries/second_pass_quality_preview.json`

## Final State At Time Of Writing

- `origin/main` includes the full current work
- repo worktree is clean
- one real heavy ingest process is still running
- the next meaningful work is knowledge feeding and knowledge deepening, not more runtime plumbing
