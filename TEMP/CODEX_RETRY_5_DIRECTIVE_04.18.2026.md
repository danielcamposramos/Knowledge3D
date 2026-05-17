# Codex Retry Directive #5 — CUDA v1/v2 Symbol Interop Fix

**Date:** 2026-04-18 (after retry #4 still hit 201 at cuMemGetInfo_v2)
**Role:** Codex — RUN-ONLY. Same contract. No edits.

---

## What retry #4 showed

```
exit_code: 1
[loader] CUDA context materialized (pid=2413549, device=0, ctx=0x560593240a70)   ← ctx valid
RuntimeError: Sovereign loader error: invalid device context
  at loader.get_vram_usage() → cuMemGetInfo_v2 (code 201)
commit_head: 3eaf2fc2
```

Warmup alloc+free succeeded, `cuCtxSetCurrent` succeeded, but `cuMemGetInfo_v2` returned 201 — **on a context whose handle is valid**.

## Root cause (confirmed by `nm -D libcuda.so.1`)

`libcuda.so.1` exports v1 **and** v2 variants of the same API at **DISTINCT addresses**:

```
cuCtxCreate      0x2d6790   v1 (deprecated, pre-CUDA 3.2)
cuCtxCreate_v2   0x27d120   v2 (current, 64-bit size_t)
cuMemAlloc       0x2d67f0   v1
cuMemAlloc_v2    0x2870a0   v2
cuMemGetInfo     0x2d67d0   v1
cuMemGetInfo_v2  0x287080   v2
cuCtxDestroy     0x2e19c0   v1
cuCtxDestroy_v2  0x27d160   v2
```

The loader was mixing: **bare `cuCtxCreate` (v1)**, **bare `cuMemAlloc` (v1)**, **`cuMemGetInfo_v2` (v2)**. A v1-created context does not interoperate with v2 bookkeeping APIs — `cuMemGetInfo_v2` returns `CUDA_ERROR_INVALID_CONTEXT` even though the handle is otherwise live and `cuMemAlloc` (v1) + `cuCtxSetCurrent` (version-neutral) accept it.

## What landed (commit `8a2b711e`)

Single-file fix in `knowledge3d/cranium/sovereign/loader.py`:

- Added `_bind_v2(primary, fallback)` helper that resolves v2 symbol first, v1 only if v2 is absent.
- Rebound `_cuMemGetInfo`, `_cuMemAlloc`, `_cuMemFree`, `_cuMemsetD32` to prefer `_v2` (previously bare/v1).
- Replaced direct `nvcuda.cuCtxCreate` with new `_cuCtxCreate` (`cuCtxCreate_v2`).
- Replaced direct `nvcuda.cuCtxDestroy` with new `_cuCtxDestroy` (`cuCtxDestroy_v2`).

`bash scripts/check_single_context_invariant.sh` → `Single-context invariant: CLEAN`.

Top of branch:

```
8a2b711e fix(sovereign-loader): bind cuda v2 symbols consistently to fix cuMemGetInfo 201
3eaf2fc2 fix(sovereign-ctx): stop galaxy_buffer from releasing loader's primary ctx + drop redundant cuCtxSetCurrent
8a9bb30d fix(sovereign-loader): ctx.value not int(ctx) in boot diagnostic
40b98338 fix(sovereign-loader): unconditional 16-byte warmup + boot diagnostic
ae385716 refactor(sovereign): migrate 6 bindings to shared-context loader pattern
```

## Your task

Re-run smoke. Preflight must show `8a2b711e` at top.

```bash
git log --oneline -5            # top must be 8a2b711e
git status --short              # ?? on TEMP/*.md only
bash scripts/k3d_env.sh -e k3d-cranium python benchmarks/sovereign_bitnet_attention.py --quick
```

If smoke passes, proceed to full:

```bash
mkdir -p logs data/benchmarks
TS=$(date +%Y%m%d_%H%M%S)
bash scripts/k3d_env.sh -e k3d-cranium \
     python benchmarks/sovereign_bitnet_attention.py \
     2>&1 | tee "logs/bitnet_attention_run_${TS}.log"
```

## What to expect

- Exactly one `[loader] CUDA context materialized (pid=<N>, device=0, ctx=0x<hex>)` line.
- `get_vram_usage()` should now return real `used`/`total` bytes. This is the error that's blocked every prior run; it must clear.
- If it clears, the benchmark proceeds to PTX load, buffer allocation, kernel launches. That is the first real test of the PTX path and **any failure from here is new signal** — report it honestly.

## Three valid outcomes

1. **Smoke passes** → run full → report seven `key = value` lines + SHA per PASS format.
2. **Smoke fails at `get_vram_usage`** → v1/v2 interop was not the cause → paste traceback, we diagnose again.
3. **Smoke fails downstream** (PTX load / kernel launch / convergence / sentinel) → this is the goal: new, deeper signal → paste traceback.

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
git status --short   # MUST show only: data/benchmarks/*.json, logs/*.log
git add "data/benchmarks/sovereign_bitnet_attention_<ts>.json" \
        "logs/bitnet_attention_run_${TS}.log"
git commit -m "bench(sovereign-bitnet): honest sovereign attention run post v1/v2 fix — ${TS}"
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

No prose, no speculation, no proposed fixes.

## Forbidden — unchanged

- No source edits. No try/except wrappers. No env var overrides beyond `scripts/k3d_env.sh`.
- No JSON artifact reuse. No push, amend, `--no-verify`.

## Post-compaction reload

1. Re-read this file.
2. Re-read `TEMP/CODEX_RERUN_NOTE_SOVEREIGN_BITNET_04.18.2026.md`, `TEMP/CODEX_HANDOFF_04.18.2026_VSCODE.md`, `TEMP/CODEX_RUNBOOK_SOVEREIGN_BITNET_BENCHMARK_04.18.2026.md`.
3. `git log --oneline -5` — top must be `8a2b711e`.
4. `git status --short` — expect `??` on `TEMP/*.md` only.

---

## One-sentence summary

Loader was mixing v1 `cuCtxCreate` with v2 `cuMemGetInfo` — different functions at distinct addresses in libcuda — so a valid-looking context was rejected as invalid; `8a2b711e` rebinds all version-bifurcated symbols to v2 consistently.
