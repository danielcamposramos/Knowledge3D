# TEMP Reconciliation Matrix

Date: 2026-04-08 15:14:27 -0300
Active log: `TEMP/CODEX_TRACK_RECONCILIATION_AND_ARC_EXECUTION_LOG_2026-04-08_1503-0300.md`

## Scan Summary

- TEMP files matching `CLAUDE|CODEX|KIMI`: `592`
- owner counts:
  - `CODEX`: `419`
  - `CLAUDE`: `156`
  - `KIMI`: `3`
  - `OTHER`: `14`

## Family Matrix

| Family | Disposition | Canonical Owner | Governing Specs | Live Code / Artifacts | Verified | Remaining Debt |
|---|---|---|---|---|---|---|
| `CODEX_REALITY_ENGINE_*` April 8 | `authoritative` | `CODEX.md` + this matrix | full `docs/vocabulary/` corpus, especially `REALITY_ENABLER_SPECIFICATION.md`, `MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md`, `AVATAR_EMBODIMENT_SPECIFICATION.md`, `RPN_DOMAIN_OPCODE_REGISTRY.md` | `knowledge3d/cranium/kernels/tex_*`, `entity_*`, `modular_rpn_kernel.cu`, `modular_rpn_engine.py` | physics/texture/entity slices green | step 4/5 launcher integration still deferred |
| `CODEX_SOVEREIGN_PHYSICS_SPEC_v2_2026-04-07.md` | `authoritative` | `CODEX.md` + `docs/ROADMAP.md` | `REALITY_ENABLER_SPECIFICATION.md`, `THREE_BRAIN_SYSTEM_SPECIFICATION.md`, `KNOWLEDGEVERSE_SPECIFICATION.md` | physics kernels, `sovereign_bridges.py`, rebuilt `modular_rpn_kernel.ptx` | sovereign physics suite green | full fused-step integration deferred |
| `KIMI_IMPLEMENTATION_CORRECTNESS_SPEC.md` | `partial` | `CODEX.md` + this matrix | full `docs/vocabulary/` corpus, `AGENTS.md` | zero-copy utility surface, backlog sync, transfer-yard default fix | reconciled infra slice green | deeper transfer-yard / drawing PTX bring-up still open |
| `KIMI_ZERO_COPY_MEMORY_ENHANCEMENT_SPEC.md` | `partial` | this matrix | `THREE_BRAIN_SYSTEM_SPECIFICATION.md`, `SOVEREIGN_TRAINING_SPECIFICATION.md` | `knowledge3d.cranium.kernels.*`, zero-copy tests | zero-copy control-plane tests green | raw updater kernels not yet promoted to benchmark-grade evidence |
| `KIMI_COLLABORATION_EVOLUTION_REPORT.md` | `historical` | this matrix | same as above | informs audit context only | N/A | no direct code work required |
| `CODEX_TRACK_*` March benchmark directives | `authoritative` | `CODEX.md` + `docs/ROADMAP.md` | foundational, three-brain, knowledgeverse, RPN registry, procedural memory specs | benchmark harnesses in `benchmarks/`, diagnostic runners in `scripts/` | existing benchmark suites remain part of canonical state | still need Track C + ARC score work after R0 |
| `CLAUDE_PHASE_E*` | `authoritative` | `docs/ROADMAP.md` | `ADAPTIVE_REASONING_BUDGET_SPECIFICATION.md`, `HYPER_PARALLEL_PROCESSING.md`, full vocabulary corpus | `knowledgeverse.py`, TRM/PTX stack, ARC benchmark harnesses | roadmap now reflects April baseline | hot-path migration still open |
| `CLAUDE_PHASE_D*` | `partial` | `CODEX.md` + `docs/ROADMAP.md` | `THREE_BRAIN_SYSTEM_SPECIFICATION.md`, `AVATAR_EMBODIMENT_SPECIFICATION.md` | `trm_launcher.py`, `trm_step_fused.cu` | deferred but tracked | launcher integration not yet live |
| `CLAUDE_TO_CODEX*` / `CODEX_TO_CLAUDE*` April reports | `implemented` | this matrix + dated log | governed by upstream directive families above | `TEMP/CODEX_TO_CLAUDE_*_2026-04-08.md` | used as evidence and handoff only | do not treat as canonical source over specs/backlog |
| ingestion / proceduralizer continuation | `authoritative` | `CODEX.md`, live log | `KNOWLEDGE_PROCEDURALIZER_SPECIFICATION.md`, `FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md`, `SLEEPTIME_PROTOCOL_SPECIFICATION.md` | `scripts/fundamental_ingest_pdfs.py`, `knowledge_proceduralizer.py`, `proceduralizer_*` | proceduralizer/ingest slice green; live run protected | wait for `01_encyclopedias` completion, second pass, resident ingest |
| older TEMP families outside active April + Phase E lanes | `historical` or `superseded` | `Old_Attempts/` or matrix notes | resolve by precedence chain | mixed | N/A | no equal execution priority in this track |

## Canonical Notes

- No April 6-8 directive family should be treated as “floating” outside the backlog anymore; `CODEX.md`, `docs/ROADMAP.md`, this matrix, and the dated execution log are now the canonical bridge.
- Unsupported or placeholder surfaces are not promoted by this matrix. They are either:
  - landed and verified
  - partial with explicit debt
  - historical/superseded
