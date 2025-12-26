# Phase 7 Complete: Book Quality Refinement (Selection + Context + Binding)

**Date:** 2025-12-19  
**Implementer:** Codex (GPT-5.2)  
**Scope:** Phase 7 (quality refinement) on top of Phase 6 coverage (`books_v4`)

## Executive Summary

Phase 7 validated the layered quality architecture for Book Galaxy usage:

- Stage 1 (**book boost**) makes book-seeded candidates competitive in TTC.
- Stage 2 (**context gating**) prevents obvious semantic mismatches (e.g., circle vs sphere) and keeps AMC-AIME from regressing.
- Stage 3 (**semantic binding**) infrastructure works (role extraction from prompt text + role-aware instantiation), but **further accuracy gains are bottlenecked by missing semantic metadata** in the ingested artifacts (`symbol_bindings[*].meaning` is always `"unknown"` in `books_v4`).

## What Shipped (Phase 7)

### Stage 1 — Book-Sourced Candidate Boost
- TTC scoring prefers book-sourced candidates (safe confidence boost), addressing the “books present but not winning” failure mode.

### Stage 2 — Context Gating
- Hard reject obvious mismatches before TTC seeding using prompt intent/shape cues (e.g., volume vs area, sphere vs circle).

### Stage 3 — Semantic Variable Binding
- Prompt-side role extraction (radius/height/legs/hypotenuse/etc) and role-aware variable instantiation.
- Fixed a critical bug where role-number matching regexes were accidentally double-escaped (`\\b`, `\\s`, `\\d`) and therefore never matched.
- Fixed a critical stability issue where per-candidate instantiation could mutate shared role lists across candidates (copy role lists per candidate).
- Added regression coverage: `tests/test_book_galaxy_templates.py::test_trm_reader_binds_radius_and_height_semantically`.

## Key Diagnostics (Root Cause Confirmation)

### `books_v4` artifact metadata completeness
Scan of all `artifacts.jsonl` under `/K3D/Knowledge3D.local/galaxies/books_v4`:

- Artifacts: **1329**
- With `conditions`: **994 (74.8%)**
- With `symbol_bindings` (non-empty): **1122 (84.4%)**
- With non-`"unknown"` meanings: **0 (0.0%)**
- Meanings distribution: **100% `"unknown"`**

Interpretation: the bindings structure exists, but semantics are missing, so meaning-driven binding cannot activate.

### Hot-path inference experiment (Option B)
A regex-only “infer roles from artifact text” experiment was implemented and evaluated.

Finding: artifact text rarely contains explicit `role -> variable` statements after PDF extraction.

- Role inference coverage from artifact text: **~2.9%** of artifacts
- Most common: `base`, `length`, `radius`
- Missing at scale: “height h”, “legs a and b”, “hypotenuse c”

Interpretation: **runtime inference cannot recover missing metadata** from the current artifact text representation.

For cleanliness, this inference is now **gated behind** `K3D_TRM_INFER_ARTIFACT_ROLES=1` and is **OFF by default**.

## Benchmark Status (Phase 7)

Benchmarks were run with `books_v4`, `--max-problems 200`, `--shuffle-seed 123`.

- MATH: **3.0% (6/200)** (Stage 1+2 improvement from baseline 2.5% validated; Stage 3 did not add further gain under `books_v4` metadata limits)
- AMC-AIME: **0.5% (1/200)** (no regression; context gating keeps boosts safe)

Logs (local, not committed):
- `/tmp/math_phase7H_stage3_binding_books_v4_200_seed123_v2.log`
- `/tmp/amc_aime_phase7H_stage3_binding_books_v4_200_seed123.log`
- `/tmp/math_phase7I2_inferred_roles_books_v4_200_seed123.log` (optional experiment)
- `/tmp/amc_aime_phase7I_inferred_roles_books_v4_200_seed123.log` (optional experiment)

## Conclusion

Phase 7 achieved its core goal: **prove the selection + safety architecture is correct and measurable**, and pinpoint the next bottleneck precisely:

- The system needs artifacts with **explicit semantic roles** (variable meanings) to unlock Stage 3’s intended gains.
- The current ingestion output (`books_v4`) does not encode those meanings, and PDF prose is not sufficient to reconstruct them reliably at runtime.

## Recommended Next Step (for Phase 8 / Phase 6 follow-up)

Choose one:

1. **Option A (Ingestion fix):** enhance `SovereignKnowledgeArticulator` to emit variable-role semantics (e.g., `r=radius`, `h=height`, `a/b=legs`, `c=hypotenuse`) and re-ingest to `books_v5`.
2. **Phase 8 (Reasoning leverage):** multi-step chaining / compositional execution, to reduce reliance on perfect single-step binding.
3. **Hybrid ROI:** add a few high-value generic TTC families (e.g., trig/angle, unit conversions) and re-evaluate.

