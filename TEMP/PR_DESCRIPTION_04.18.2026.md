# PR — Sovereign BitNet b1.58 Attention Kernel + v4 Doctrine Landing

**Title:** `feat: sovereign BitNet b1.58 attention kernel + v4 doctrine landing (turns 1-6, 2026-04-18)`

**Branch:** `codex/batch11-knowledge-waves-observability-game2d-2026-04-15` → `main`

---

## Summary

- **Kernel landed:** `bitnet_attention.cu` — sovereign BitNet b1.58 attention (5 trits/byte packing, 1.6 bits/weight, multiplication-free add/sub/skip). Ternary weights + contrastive margin (no softmax in hot path). Value mixing lands in the Transfer Yard, not LIFO.
- **Opcodes expanded (append-only, per Expand-Not-Replace doctrine):** six attention opcodes reserved in range 0x1AA–0x1AF; IMAGE/SPARSE relocated to 0x1C0–0x1C5; `VEC_NORM_L2_INT8` at 0x1B0. Registry is append-only; 0x1A7 `BASE` kept, 0x1A8 `TERNARY` added as variant.
- **Old_Attempts migration:** 5 files archived to `Old_Attempts/2026-04-18/` under the migration manifest (replaced by sovereign equivalents; no in-place rewrites).
- **Doctrine formalized:** six new standing rules landed as v4 supersession patches in `TEMP/supersession_patches_04.18.2026_v{1..4}.md`, cross-referenced from agent memory.
- **Three-lane delivery:** this PR is Author (kernel + doctrine + runbook). Lane 1 wires the build/loader; Lane 2 produces the benchmark harness; Codex executes per the runbook in `TEMP/CODEX_RUNBOOK_SOVEREIGN_BITNET_BENCHMARK_04.18.2026.md`.

## Scope

- **194 modified entries** total (110 modified, 79 untracked, 5 renamed).
- High-level breakdown by top-level directory:
  - `knowledge3d/cranium/*` — 12 files (PTX runtime + kernel registration)
  - `knowledge3d/knowledgeverse/*` — 8 files (sovereign hot path, TRM game loop, navigator)
  - `knowledge3d/models/*` — 4 files (adapters, LoRA, policy, eval logs)
  - `knowledge3d/tablet/*` + `knowledge3d/daemon/*` + `knowledge3d/bridge/*` — 6 files
  - `docs/vocabulary/*` + `docs/ingestion/*` + `docs/research/*` — 8 files
  - `Old_Attempts/2026-04-18/*` — 5 archived files + manifest
  - `tests/*` — 12 files across bridge, daemon, benchmarks, integration, infra
  - `TEMP/*` — 24 dated design/doctrine/audit docs (04.18.2026)
  - `benchmarks/*`, `scripts/*`, `deploy/systemd/*`, top-level docs — remainder

## Sovereign additions

- **New kernel:** `bitnet_attention.cu` + compiled PTX
- **New opcodes (append-only):**
  - 0x1AA `ATTN_TERNARY_QK` — Q·K ternary inner product
  - 0x1AB `ATTN_CONTRASTIVE_MARGIN` — margin scoring (replaces softmax)
  - 0x1AC `ATTN_VALUE_MIX_YARD` — value mixing in Transfer Yard
  - 0x1AD `ATTN_HEAD_REDUCE` — multi-head reduction
  - 0x1AE `ATTN_WRITEBACK` — stream results back to Galaxy
  - 0x1AF `ATTN_LANE_SWITCH` — Path B dual-path lane switch
  - 0x1B0 `VEC_NORM_L2_INT8` — int8 L2 norm helper
  - 0x1C0–0x1C5 IMAGE/SPARSE (relocated from earlier draft range; old range vacated, not renumbered in place)
- **New PTX surface:** attention dispatch table entries registered in the loader (Lane 1).
- **Transfer Yard default** on tiers 1/2/3 (per Phase A spec) — value-mix stage writes through the yard, not LIFO.

## Doctrine additions (six v4 standing rules)

All land in `TEMP/supersession_patches_04.18.2026_v4.md` and agent memory:

1. **Expand-Not-Replace opcodes** — registry append-only; variants get new numbers.
2. **Old_Attempts protocol** — replaced files move whole under dated folder + MANIFEST entry.
3. **Bulk-lib purge hard gate** — numpy/cupy/scipy/sympy banned in sovereign hot path; audit is full-tree.
4. **W3C-only external alignment** — external spec surface goes through PM-KR / W3C CG, not ad-hoc integration.
5. **Hyper-modular symlink architecture** — phases stand on each other like symlinks; stubs are fallbacks in disguise.
6. **Core isolation + queue opcodes** — 46 cores / 414 instances on RTX 3070; `confidence_trit` field in star schema; queue opcodes 0x178–0x17A.

## Test plan

Codex executes the runbook in `TEMP/CODEX_RUNBOOK_SOVEREIGN_BITNET_BENCHMARK_04.18.2026.md` and checks:

- [ ] Lane 1: kernel compiles cleanly with `sm_86 -O3`; loader registers 0x1AA–0x1AF
- [ ] Lane 2: `benchmarks/sovereign_bitnet_attention.py` exists, is executable, emits the six success-criteria fields
- [ ] Preflight: env activates, `CUDA_VISIBLE_DEVICES=0`, `nvidia-smi` shows RTX 3070, PTX artifact present
- [ ] Smoke: `python benchmarks/sovereign_bitnet_attention.py --quick` passes
- [ ] Full: `sovereign_compliance=PASS`, `latency_ms_per_query.p50 ≤ 5.0`, `top_k_consistency=1.0`, `convergence_verified=true`, all six opcodes traced
- [ ] Artifacts committed (`data/benchmarks/*.json` + `logs/*.log`) with the canonical commit message

## Not in scope

- **Bulk-lib purge Phases 3–8** — top-5 plan (`TEMP/phase2_purge_top5_plan_04.18.2026.md`) is Phase 2 only; remaining 3,881 violations / 127 files land in follow-up PRs.
- **Tier 2 / Tier 3 yard kernel promotion** — Transfer Yard default is enforced on tiers 1/2/3 in Phase A; tier 2/3 PTX variants are scoped to a separate lane.
- **Phase B native embedding** — composable-basis RPN projection replaces trigram embedder per `TEMP/CLAUDE_CODEX_PHASE_B_NATIVE_EMBEDDING_04.18.2026.md`; implementation is a follow-up PR.
- **Qwen3 on Phenom for Qdrant** — deployment plan exists (`TEMP/qwen3_phenom_qdrant_deployment_plan_04.18.2026.md`) but belongs to the ingestion lane, not the sovereign hot path.
- **GPU game-loop closure** — `TEMP/CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md` scoped separately.
