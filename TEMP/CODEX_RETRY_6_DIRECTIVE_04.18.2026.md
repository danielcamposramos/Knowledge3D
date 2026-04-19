# Codex Retry Directive #6 — Smoke Verified, Full Run Handoff

**Date:** 2026-04-18 (after Claude verified smoke PASS at HEAD `5ef2bd83`)
**Role:** Codex — RUN-ONLY. Same contract. No edits.

---

## What Claude landed since retry #5

Three commits on top of `8a2b711e`. Each was proven in-process — `--quick`
smoke completes with `exit=0`, all gates green — before the next was
stacked on top.

```
5ef2bd83 fix(bench-bitnet): scale synthetic scores into realistic dp4a magnitude band
724fe514 fix(bitnet-kernel): hoist __shfl_sync out of divergent branch to avoid Ampere warp hang
60adf711 fix(sovereign-loader): extend cuda v2 binding to module/stream/primary-ctx-flags
8a2b711e fix(sovereign-loader): bind cuda v2 symbols consistently to fix cuMemGetInfo 201
3eaf2fc2 fix(sovereign-ctx): stop galaxy_buffer from releasing loader's primary ctx + drop redundant cuCtxSetCurrent
```

### 1. `60adf711` — loader v2 binding, second wave

Retry #5 died at `cuModuleGetGlobal` with code 201. Same v1/v2 bifurcation
as `8a2b711e`: `libcuda.so.1` exports `cuModuleGetGlobal` (v1, 0x2d67b0)
and `cuModuleGetGlobal_v2` (0x286d80) at distinct addresses; the bare
`nvcuda.cuModuleGetGlobal` resolved to v1 and rejected the v2-created
context. Extended `_bind_v2` helper to cover:

- `_cuModuleGetGlobal`          (used by `loader.get_global`)
- `_cuStreamDestroy`            (used by `loader.destroy_stream`)
- `_cuDevicePrimaryCtxSetFlags` (used by the primary-ctx fallback branch)

### 2. `724fe514` — Ampere warp hang in `bitnet_matmul_tile`

`loader.get_global` then worked; `cuLaunchKernel` returned 0 for
`k3d_bitnet_attention_proj`; but `cuCtxSynchronize` never returned
(confirmed via `CUDA_LAUNCH_BLOCKING=1` + `gdb` stack stuck in
libcuda). Root cause: inside the per-word loop, `__shfl_sync(0xFFFFFFFF, ...)`
sat inside `if (word_idx < n_words_total)`. For `K=64 → n_words_total=4`,
half the warp lanes skipped the `if` and never reached the shuffle.
Under sm_70+ independent thread scheduling this is UB and deadlocks the
warp (CUDA-C Programming Guide §5.2.3).

Fix: hoist the `__shfl_sync` call out of the guarded branch; keep the
load guarded. Downstream use is already gated by the per-trit bounds
check, so lanes with no valid word still contribute zero.

PTX rebuilt with `nvcc --ptx -arch=sm_86 -O3 -I. …/bitnet_attention.cu`.

### 3. `5ef2bd83` — synthetic score scale for Path A margin gate

Once kernels ran, the rank kernel wrote nothing: `convergence_verified
= false`, `min_sentinel_overwrites_per_query = 0`. Path A (0x1AE)
normalizes scores via `score >> 18` before the `3/4 × top` threshold;
the old fixture produced scores in `[11, 151536]`, and every normalized
value collapsed to zero. Scaled the monotone ascending pattern by 4096
so values land in the real dp4a band (±d·127² = ±1,032,256 for d=64).
Ordering is preserved; top-k is still stable.

## Smoke verified at HEAD `5ef2bd83`

```
exit=0
sovereign_compliance     = PASS
convergence_verified     = true
min_sentinel_overwrites_per_query = 320
top_k_consistency        = 1.0
latency_ms_per_query.p50 ≈ 0.062 ms
vram_peak_mb             = 119.125
```

`bash scripts/check_single_context_invariant.sh` → `Single-context invariant: CLEAN`.

## Your task

You're back in run-only mode. Preflight, smoke, full, commit artifacts, report.

```bash
git log --oneline -5            # top must be 5ef2bd83
git status --short              # ?? on TEMP/*.md and possibly the two stale
                                # data/benchmarks/*.json files Claude produced
                                # while verifying — do NOT add those, they
                                # predate HEAD. You'll generate your own below.
bash scripts/check_single_context_invariant.sh        # CLEAN
bash scripts/k3d_env.sh -e k3d-cranium \
     python benchmarks/sovereign_bitnet_attention.py --quick
```

If smoke passes, run the full benchmark:

```bash
mkdir -p logs data/benchmarks
TS=$(date +%Y%m%d_%H%M%S)
bash scripts/k3d_env.sh -e k3d-cranium \
     python benchmarks/sovereign_bitnet_attention.py \
     2>&1 | tee "logs/bitnet_attention_run_${TS}.log"
```

## Reporting — unchanged

### On PASS — paste values (not keys):

```
sovereign_compliance = <value>
convergence_verified = <value>
min_sentinel_overwrites_per_query = <value>
kernel_path_trace = <array>
vram_peak_mb = <value>
latency_ms_per_query.p50 = <value>
top_k_consistency = <value>
commit = <sha from git rev-parse HEAD>
```

Commit artifacts only:

```bash
git status --short   # MUST show only: data/benchmarks/sovereign_bitnet_attention_<ts>.json, logs/bitnet_attention_run_${TS}.log
git add "data/benchmarks/sovereign_bitnet_attention_<ts>.json" \
        "logs/bitnet_attention_run_${TS}.log"
git commit -m "bench(sovereign-bitnet): first honest sovereign attention run — ${TS}"
git rev-parse HEAD
```

No push. No amend. No `--no-verify`.

### On FAIL — paste exactly:

```
command: <exact command>
exit_code: <number>
git_status:
<output of git status --short>
stderr_tail_40:
<last 40 lines>
json_if_any:
<JSON contents or "none produced">
commit_head: <git rev-parse HEAD>
```

No prose, no speculation, no proposed fixes. Claude triages.

## Forbidden — unchanged

- No source edits. No try/except wrappers. No PYTHONPATH/LD_PRELOAD/K3D_FORCE_* overrides.
- Do NOT re-add the two `data/benchmarks/*.json` files Claude produced while
  verifying (timestamps 1776544243, 1776544325) — they predate HEAD and must
  not be passed off as the honest full-run artifact.
- No push, amend, or `--no-verify`.
- The faked-artifact rule from `memory/feedback_codex_cannot_silent_fix_to_unblock.md` still stands.

## Post-compaction reload protocol

1. Re-read this file.
2. Re-read `TEMP/CODEX_RERUN_NOTE_SOVEREIGN_BITNET_04.18.2026.md`,
   `TEMP/CODEX_HANDOFF_04.18.2026_VSCODE.md`,
   `TEMP/CODEX_RUNBOOK_SOVEREIGN_BITNET_BENCHMARK_04.18.2026.md`.
3. `git log --oneline -5` — top must be `5ef2bd83`.
4. `git status --short` — expect `??` on `TEMP/*.md` and the two stale
   JSON artifacts noted above (ignore those).

---

## One-sentence summary

Claude drove the loader past v1/v2, fixed an Ampere warp deadlock in
`bitnet_matmul_tile`, and rescaled the bench's synthetic scores into the
real dp4a band; smoke is green at `5ef2bd83` with all five gates satisfied
— your job is the honest full run and the usual seven-line report.
