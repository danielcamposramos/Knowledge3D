# Codex -> Claude: Phase E.64 Benchmark Relaunch Report

Date: 2026-04-03

## Executive Summary
E.64 did not reach the benchmark gate. The bounded meaning-normalized expansion batch was generated and ingested successfully into the real resident corpus, but the required real sovereign rebuild failed before `meaning_family_route_audit.json` could be produced.

Status:
- Preflight captured
- Bounded payload batch generated
- Resident ingest completed with zero route-contract rejections and zero benchmark-name leakage rejections
- Real `--refresh-feed-source --refresh-build-feed --force-rebuild` run executed and failed
- `meaning_family_route_audit.json` was not produced
- Benchmark stages 1/2/3 were not started by design after rebuild failure

Primary blocker:
- Feed-source compilation now fails hard on resident procedural/object stars that still lack explicit sovereign route metadata
- Sample failure:
  - `reality_proc_lsystem_expand:missing_selection_role, missing_layer_id, missing_answer_eligible`
  - `obj3d_mesh_compute_normal:missing_selection_role, missing_layer_id, missing_answer_eligible`
  - `obj3d_mesh_grid_16x16:missing_selection_role, missing_layer_id, missing_answer_eligible`
  - `obj3d_xform_apply:missing_selection_role, missing_layer_id, missing_answer_eligible`
  - `... (+90 more)`

## Since E.58
- E.58:
  - forced sovereign device rebuild semantics landed
  - GPU star materializer and CSR build path became authoritative for rebuild
  - manifest provenance/backend tracking was added
- E.59:
  - reachable sovereign rebuild math moved off Python `math.*`
  - device finalization became the numeric shaping authority
  - fixed kernels stayed the bulk path; Math Core reuse stayed compact/validation-only
- E.60:
  - production boot/rebuild became build-feed only
  - boot-path source-entry translation and dict-driven ref resolution were removed from production
  - device decode/hash-resolve/reverse-symlink path landed
- E.61:
  - `feed_source_*` became the maintenance compiler authority
  - production boot stayed build-feed only
  - maintenance compile no longer depended on source-entry translation during build-feed refresh
- E.62:
  - explicit sovereign spines were added for `GRAMMAR`, `GENERAL`, and `CHAT`
  - per-family route audit surfaced meaning-family health
- E.63:
  - resident family depth expanded for `GRAMMAR`, `GENERAL`, `CHAT`, `QUESTION`, and `MATH`
  - anti-pattern families were deepened
  - benchmark-derived ingestion was normalized by meaning
  - `meaning_family_route_audit.json` became the benchmark relaunch gate artifact

## E.64 Work Completed
### Preflight
Artifacts:
- `../Knowledge3D.local/results/e64_benchmark_relaunch_2026-04-03/preflight/preflight_summary.json`
- `../Knowledge3D.local/results/e64_benchmark_relaunch_2026-04-03/preflight/galaxy_consolidated_latest.json`
- `../Knowledge3D.local/results/e64_benchmark_relaunch_2026-04-03/preflight/sovereign_runtime_manifest.json`

Preflight state:
- `galaxy_consolidated_latest.json` sha1: `41d471042e20ce5acbe12d18666e082b45d9addc`
- `sovereign_runtime_manifest.json` sha1: `c6ebdfc080cfc9585ae5319fecfdcfb384df5f1b`
- consolidated/default knowledge signature: `ba83b6c9`
- resident route audit file did not exist at preflight

### Payload Generation
Artifacts:
- `../Knowledge3D.local/results/e64_benchmark_relaunch_2026-04-03/payloads/benchmark_augmentation_payload.jsonl`
- `../Knowledge3D.local/results/e64_benchmark_relaunch_2026-04-03/payloads/benchmark_augmentation_report.json`
- `../Knowledge3D.local/results/e64_benchmark_relaunch_2026-04-03/payloads/proceduralizer_mmlu_val_payload.jsonl`
- `../Knowledge3D.local/results/e64_benchmark_relaunch_2026-04-03/payloads/proceduralizer_gsm8k_train_payload.jsonl`

Generated batch:
- benchmark augmentation: `4768` rows
- proceduralizer `mmlu_val`: `120`
- proceduralizer `gsm8k_train`: `120`

### Operational Blocker Fixes Made During E.64
These were required to make the operational plan runnable:

1. `scripts/fundamental_ingest_payloads.py`
- switched ingestion to a non-live `Knowledgeverse` construction path:
  - `eager_load_default_galaxies=False`
  - `start_live_loops=False`
- added explicit `save_consolidated_state()` so ingestion actually persists

2. `scripts/fundamental_ingest_payloads.py`
- benchmark-name leakage detection now checks token boundaries instead of naive substrings
- this removed false positives from ordinary text such as words containing `arc`

3. `scripts/fundamental_augment_benchmarks.py`
- benchmark-derived math entries no longer emit runtime-visible `amc_*` ids/names
- those math entries now emit explicit sovereign route metadata so they pass the ingestion contract

### Resident Ingest
Artifacts:
- `../Knowledge3D.local/results/e64_benchmark_relaunch_2026-04-03/ingest/ingest_report.json`

Final clean ingest result:
- `added=300`
- `skipped=4708`
- `rejected_missing_route_metadata=0`
- `rejected_benchmark_name_leakage=0`

Resident state after ingest:
- `galaxy_consolidated_latest.json` sha1 changed to `b748bc076400821d1c58501a2d4dc9a09635196f`
- `gpu_buffer_signature_base` moved to `2cad9efa`
- `default_knowledge_signature` remained `ba83b6c9`
- resident counts:
  - `Grammar=103908`
  - `Math=37667`
  - `Reality=10752`

Interpretation:
- the resident world changed and persisted
- the canonical default-knowledge signature did not change, which is consistent with the added entries landing outside the default-knowledge signature set

## Real Rebuild Outcome
Artifacts:
- `../Knowledge3D.local/results/e64_benchmark_relaunch_2026-04-03/rebuild/rebuild.log`
- `../Knowledge3D.local/results/e64_benchmark_relaunch_2026-04-03/rebuild/rebuild_summary.json`

Run:
- `scripts/rebuild_sovereign_artifact.py --storage-root ../Knowledge3D.local --refresh-feed-source --refresh-build-feed --force-rebuild --verbose`

Observed progress before failure:
- `knowledgeverse booted in 29.609s`
- default galaxy load: `0.152s`
- catalog flatten completed for `260879` rows in `26.327s`
- cache save: `54.234s`
- feed-source compile progressed through:
  - `25000/260879`
  - `50000/260879`
  - `75000/260879`
  - `100000/260879`
  - `125000/260879`
  - `150000/260879`
  - `175000/260879`
  - `200000/260879`
  - `225000/260879`
  - `250000/260879`
  - `260879/260879`

Failure:
- exit code: `1`
- failure site: `runtime.refresh_feed_source()`
- exception: `ValueError: sovereign_build_metadata_invalid: ...`

Meaning:
- the operational pipeline now reaches the full resident feed-source compile successfully
- the next blocker is not the compile pipeline itself
- the blocker is latent resident knowledge that now violates the stricter explicit sovereign metadata contract

## Why Benchmarks Did Not Run
Per the E.64 gate:
- rebuild must succeed
- `meaning_family_route_audit.json` must exist and pass

Neither condition was met:
- rebuild failed
- `../Knowledge3D.local/checkpoints/meaning_family_route_audit.json` still does not exist

Therefore:
- Stage 1 was not started
- Stage 2 was not started
- Stage 3 was not started

This was intentional and correct. Running benchmarks without a rebuilt resident artifact would have invalidated the relaunch gate.

## What Is Missing Before Benchmark Relaunch
The next work is no longer operational orchestration. It is resident metadata repair and expansion for non-family procedural/object knowledge that is being pulled into the sovereign feed-source authority.

Immediate required fixes:
- audit every resident star now tripping `sovereign_build_metadata_invalid`
- assign explicit:
  - `selection_role`
  - `layer_id`
  - `answer_eligible`
  - and, where appropriate, `route_family`, refs, and route policy
- verify whether these stars are truly route-capable or should instead be excluded from route-capable feed-source enforcement

Grounded failing examples:
- `reality_proc_lsystem_expand`
- `obj3d_mesh_compute_normal`
- `obj3d_mesh_grid_16x16`
- `obj3d_xform_apply`
- approximately `90+` more

Most likely next phase direction:
- classify resident procedural/object stars into:
  - route-capable knowledge that needs explicit sovereign metadata
  - non-route-capable utility/procedural stars that should remain resident but not be enforced as route-bearing reasoning entries
- rebuild again
- require successful `meaning_family_route_audit.json`
- only then run the staged benchmark gate

## Bottom Line
E.64 produced a real operational result:
- the bounded meaning-normalized expansion batch is working
- ingestion is working and persists cleanly
- the real rebuild runs far enough to expose the next true resident-corpus blocker

E.64 did not reach benchmark relaunch because the resident corpus still contains procedural/object stars that are incompatible with the strict sovereign feed-source metadata contract.
