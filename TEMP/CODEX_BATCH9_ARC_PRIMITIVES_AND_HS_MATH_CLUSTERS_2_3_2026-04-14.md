# CODEX BATCH 9 — HS Math Clusters 2 & 3 Bullet Ingestion Finish

**Date:** 2026-04-14
**Author:** Claude (architecture)
**Status:** Approved — ready for Codex
**Depends on:** Batch 8 (HS Math Cluster 1 + Phase 7.A.1 seed) — landed
**Unblocks:** full HS math ingestion runway after Cluster 1
**Scope:** ingestion path only (no hot-path, no PTX, no numpy/cupy/scipy)
**Sovereignty:** PTX + Galaxy + RPN only in hot path; ingestion path may use Qdrant + fastembed as already allowed

---

## 0. Goal

Finish the interrupted HS math ingestion work for Cluster 2 and Cluster 3 using the already-repaired bullet-dialect source files.

| Slice | Source                                                         | Format        | Target rows                | Status of source |
|-------|----------------------------------------------------------------|---------------|----------------------------|------------------|
| B     | `TEMP/KIMI_MATH_HS_CLUSTER2_GEOMETRY_TRIG_2026-04-13.md`       | bullet dialect | ~56 meaning_star           | Repaired and ready |
| C     | `TEMP/KIMI_MATH_HS_CLUSTER3_STATS_DISCRETE_APPLIED_2026-04-13.md` | bullet dialect | ~57 meaning_star | Repaired and ready |

Both files are bullet-shaped and route through the existing Batch 8 parser infrastructure. JSON/hybrid parser work is removed from this slice. ARC primitives and benchmark sweep are explicitly deferred.

---

## 1. Slice B — HS Math Cluster 2 (Geometry & Trigonometry, bullet shape)

### 1.1 Parser routing

Cluster 2 source file: `TEMP/KIMI_MATH_HS_CLUSTER2_GEOMETRY_TRIG_2026-04-13.md`. Format: bullet dialect, parsed by `parse_cluster1_bullets_with_diagnostics`.

Implementation:

- Route `cluster2` through the same bullet parser as Cluster 1.
- Trim the trailing human summary section beginning with `---` then `**Star count:**` before parsing.
- Keep strict language validation inside true `surface_forms` blocks.
- Expected parse result on the repaired file: `56` rows, `0` skipped blocks.

### 1.2 Driver: `scripts/ingest_hs_math_cluster2.py`

Same three-pass shape as `ingest_hs_math_cluster1.py`. Source tag: `"math_hs_cluster2_v1"`. Bootstrap tag: `"phase_7a2_hs_math_cluster2_v1"`.

### 1.3 Seed extensions

Cluster 2 introduces the following new canonical requirements:

- Greek mathematical letters: `alpha`, `beta`, `gamma`, `theta`
- Operator / relation surfaces: `sin`, `cos`, `tan`, `greater`

Each lands across the canonical surfaces where due:

- `char_*` for glyph-bearing letters / relations
- `math_symbol_*` for mathematical symbol/operator use
- `concept_*` for meaning-layer identity

The existing Batch 8.1 seed and audit scripts are extended rather than replaced.

### 1.4 Tests

- parser route test on the real repaired file
- summary-tail trim test
- named Greek `letter::` resolution test
- `symbol::sin`, `symbol::cos`, `symbol::tan`, `symbol::greater` resolution test
- Qdrant integration test for full `--write`

### 1.5 Acceptance — Slice B

- Parser handles the repaired Cluster 2 source without raising.
- `K3D_QDRANT_INTEGRATION=1 python3 scripts/ingest_hs_math_cluster2.py --write` writes the expected row count, Pass 3 `misses=[]`.
- Tests green.

---

## 2. Slice C — HS Math Cluster 3 (Statistics, Discrete, Applied, bullet shape)

### 2.1 Parser routing

Source file: `TEMP/KIMI_MATH_HS_CLUSTER3_STATS_DISCRETE_APPLIED_2026-04-13.md`. Format: bullet dialect, parsed by `parse_cluster1_bullets_with_diagnostics`.

Implementation:

- Route `cluster3` through the same bullet parser as Cluster 1.
- Trim the trailing human summary section beginning with `---` then `**Required canonical alias seeds` before parsing.
- Expected parse result on the repaired file: `57` rows, `0` skipped blocks.

### 2.2 Driver: `scripts/ingest_hs_math_cluster3.py`

Same three-pass shape. Source tag: `"math_hs_cluster3_v1"`. Bootstrap tag: `"phase_7a2_hs_math_cluster3_v1"`.

### 2.3 Seed extensions

Cluster 3 introduces these additional canonical requirements on top of Cluster 2:

- Greek mathematical letters: `mu`, `sigma`, `lambda`, `rho`
- relation surface reuse: `greater`

Each lands across the canonical surfaces where due:

- `char_*`
- `math_symbol_*`
- `concept_*`

### 2.4 Tests

- parser route test on the real repaired file
- summary-tail trim test
- named Greek `letter::` resolution test
- Qdrant integration test for full `--write`

### 2.5 Acceptance — Slice C

- Parser handles the repaired Cluster 3 source without raising.
- `K3D_QDRANT_INTEGRATION=1 python3 scripts/ingest_hs_math_cluster3.py --write` writes expected row count, Pass 3 `misses=[]`.
- Tests green.

---

## 3. Explicit defers

Deferred from this finish wave:

- ARC reasoning primitives ingestion
- benchmark sweep
- any new parser family beyond bullet-route reuse
K3D_QDRANT_INTEGRATION=1 python3 scripts/run_math_benchmark.py --output TEMP/CODEX_BATCH9_MATH_BENCHMARK_$(date +%F).json

# ARC-AGI 1 (curated 10 + expanded 50)
K3D_QDRANT_INTEGRATION=1 python3 scripts/run_arc_benchmark.py --output TEMP/CODEX_BATCH9_ARC_BENCHMARK_$(date +%F).json

# MMLU (sanity sample)
K3D_QDRANT_INTEGRATION=1 python3 scripts/run_mmlu_benchmark.py --limit 50 --output TEMP/CODEX_BATCH9_MMLU_BENCHMARK_$(date +%F).json
```

If those exact runner scripts have different names in the current tree, use the equivalents listed in `CODEX.md`. Do not invent new ones.

### 4.2 Reporting

Write `TEMP/CODEX_BATCH9_BENCHMARK_REPORT_2026-04-14.md` with:
- Per-suite pass/fail counts before vs after Batch 9.
- Star-recall counts (how many of the new Batch 8 + Batch 9 stars actually got hit by a query during the sweep).
- Any sovereignty violations grepped for in the reasoning path (`grep -rn 'import numpy\|import cupy\|import scipy\|import sympy' knowledge3d/cranium/ knowledge3d/knowledgeverse/sovereign_hot_path.py`).

### 4.3 Acceptance — Slice D

- All three benchmark runs complete without crashing the Qdrant connection.
- Report file exists and shows non-zero recall on Batch 8/9 stars.
- No new sovereignty violations introduced.

---

## 5. Order of Operations

Codex should land in this order (each step gated on the prior step's tests passing):

1. **Slice A.1–A.5** (ARC parser routing → seed → audit → driver → tests → real --write)
2. **Slice B.1–B.5** (Cluster 2 JSON parser → seed extension → driver → tests → real --write)
3. **Slice C.1–C.5** (Cluster 3 hybrid parser → seed extension → driver → tests → real --write)
4. **Slice D** (benchmark sweep + report)

A is parallelisable with B and C from a code standpoint (no shared files except `hs_math_parser.py` and `math_semantic_aliases.py`), but seed updates touch the shared aliases module so they must be merged carefully. Land sequentially to avoid merge conflicts.

---

## 6. Sovereignty Checklist (Hard Gates)

Before declaring Batch 9 complete, confirm:

- ✅ No `numpy`, `cupy`, `scipy`, `sympy`, `sklearn`, `torch` imports in `knowledge3d/cranium/`, `knowledge3d/knowledgeverse/sovereign_hot_path.py`, or any file under `knowledge3d/cranium/ptx_runtime/`.
- ✅ All ingestion-path additions live under `knowledge3d/ingestion/` and `scripts/`.
- ✅ No Python regex/string ops added to the hot path.
- ✅ No new Python fallbacks in any GPU code path.
- ✅ All new tests live under `tests/`.

---

## 7. Out of Scope

- Universal Knowledge waves (HS natural sciences, languages, history, humanities) — Batches 10+.
- TRM game-loop migration of these stars into the runtime tick — handled when sleep-time consolidation is wired up.
- ARC visual frontend integration — separate Memory Tablet workstream.
- Kernel work — none. This is pure ingestion.

---

## 8. Hand-off

Codex implements Slices A → D in order, each with its own commit. Report back per-slice with:
- Files touched.
- Test pass counts (`pytest -q` output).
- Real `--write` row counts.
- Any source-file repairs needed (truncated blocks, missing surface_forms, etc.) — flag them, do not block on them; ingest what parses and report what was skipped.

Post the final benchmark report inline so the next architecture spec can read it without spelunking through TEMP.

---

**Tooling note (from real-world feedback):** the 120s MCP timeout floor is too tight for Qdrant-backed runs and heavier specialist calls. The empirical floor from the latest Batch 8 Qdrant integration test was 113.52s; allow **180s minimum** for any tool that touches Qdrant or ingests >30 rows.
