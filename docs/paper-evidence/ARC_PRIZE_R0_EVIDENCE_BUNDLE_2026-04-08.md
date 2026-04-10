# ARC Prize R0 Evidence Bundle

Date: 2026-04-08 15:03:10 -0300
Status: Active scaffolding baseline

## Purpose

This bundle defines the evidence inputs for the first ARC Prize delivery lane:

1. ARC-AGI-2 submission infrastructure
2. Paper-track evidence assembly
3. ARC-AGI-3 continuity without reopening unsupported placeholder surfaces

R0 is intentionally about submitability and traceable evidence, not score inflation.

## Normative Architecture Inputs

The full `docs/vocabulary/` corpus is normative for paper-track framing and claim discipline, including:

- `FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md`
- `THREE_BRAIN_SYSTEM_SPECIFICATION.md`
- `KNOWLEDGEVERSE_SPECIFICATION.md`
- `MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md`
- `RPN_DOMAIN_OPCODE_REGISTRY.md`
- `MEMORY_TABLET_SPECIFICATION.md`
- `PROCEDURAL_MEMORY_KR_STANDARD_SPECIFICATION.md`
- `SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md`
- `AVATAR_EMBODIMENT_SPECIFICATION.md`
- `REALITY_ENABLER_SPECIFICATION.md`
- `SLEEPTIME_PROTOCOL_SPECIFICATION.md`
- all remaining vocabulary specs needed to support any claim made in the manuscript

## Evidence Categories

### Architecture Evidence

- briefings in `docs/briefings/`
- canonical backlog in `CODEX.md`
- roadmap state in `docs/ROADMAP.md`
- sovereignty contracts in the PTX/runtime bridge files

### Benchmark Evidence

- `benchmarks/arc_agi_2.py`
- `benchmarks/arc_agi_3.py`
- existing JSON and markdown artifacts in `docs/paper-evidence/`
- current benchmark-specific TEMP reports that remain active rather than historical

### Knowledge Feed Evidence

- proceduralizer specification and contract files
- `scripts/fundamental_ingest_pdfs.py`
- OCR retry and staged-ingest artifacts under `Knowledge3D.local/results/base_knowledge_ingest/`

### Sovereignty / Embodiment Evidence

- April 6-8 physics, texture, and entity implementation reports in `TEMP/`
- focused green test slices for sovereign physics, procedural texture, and entity hot-path projection

## R0 Deliverables

- `benchmarks/arc_submission_formatter.py`
- `scripts/run_arc2_submission.py`
- a competition-style submission JSON artifact
- a summary JSON artifact with the exact benchmark rows that fed the submission
- manuscript scaffold in `docs/reports/ARC_PRIZE_2026_MANUSCRIPT_SCAFFOLD_2026-04-08.md`

## Exclusions

- unsupported placeholder surfaces are not paper evidence
- benchmark-smoke results without reproducible artifact paths are not paper evidence
- OCR fallback gibberish pages are not evidence until repaired
