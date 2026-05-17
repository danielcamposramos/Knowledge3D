# Codex Retry Directive #4 — Context-Ownership Fix

**Date:** 2026-04-18 (after retry #3 showed ctx materialized but cuMemGetInfo still 201)
**Role:** Codex — RUN-ONLY. Same contract. No edits, no shims, no artifact reuse.

---

## What your retry #3 showed

```
exit_code: 1
[loader] CUDA context materialized (pid=<N>, device=0, ctx=0x55b1b54291f0)   ← print succeeded
RuntimeError: Sovereign loader error: invalid device context
  at loader.get_vram_usage() → cuMemGetInfo_v2 (code 201)
commit_head: 8a9bb30d
```

**Two things we now know:**
1. `_ensure_init` runs to completion — the 16-byte warmup alloc+free succeeds and returns a valid context pointer (`0x55...`, not null).
2. Moments later `cuMemGetInfo_v2` on that same context returns `INVALID_CONTEXT`. Something is invalidating the handle between boot and first query.

## Root cause (diagnosed by Claude deep-dive sub-agent)

Two compounding defects. Report at `TEMP/CLAUDE_CUMEMGETINFO_201_DEEPDIVE_04.18.2026.md`.

1. **`galaxy_buffer.py` was releasing the primary context it didn't own.** The destructor (`__del__`, lines ~100–104) and the standalone free helper (~465–469) both called `cuda.cuDevicePrimaryCtxRelease(device)` via cuda-python (a *separate* ctypes layer from loader.py's nvcuda). When GC finalized any `GalaxyMemory`, it dropped the refcount on the primary context that `loader.py` still held. `cuCtxSetCurrent` silently accepts stale handles but `cuMemGetInfo_v2` validates and returns 201.

2. **Redundant `cuCtxSetCurrent` after a successful `cuCtxCreate`** (loader.py:415–416). `cuCtxCreate` already pushes the new ctx onto the current-thread stack. Calling `cuCtxSetCurrent` again desynchronizes stack-top vs the floating TLS slot on some CUDA 12.x driver builds — same 201 downstream.

## What landed (commit `3eaf2fc2`)

Small, surgical, two-file change:

- `knowledge3d/cranium/ptx/galaxy_buffer.py` — removed both `cuDevicePrimaryCtxRelease` call sites (destructor + free helper). Context ownership belongs exclusively to `loader.py`.
- `knowledge3d/cranium/sovereign/loader.py` — removed the redundant `ck(nvcuda.cuCtxSetCurrent(ctx))` after a successful `cuCtxCreate`.

`bash scripts/check_single_context_invariant.sh` → `Single-context invariant: CLEAN`.

Top of branch:

```
3eaf2fc2 fix(sovereign-ctx): stop galaxy_buffer from releasing loader's primary ctx + drop redundant cuCtxSetCurrent
8a9bb30d fix(sovereign-loader): ctx.value not int(ctx) in boot diagnostic
40b98338 fix(sovereign-loader): unconditional 16-byte warmup + boot diagnostic
...
```

## Your task

Re-run the smoke benchmark. Preflight must show `3eaf2fc2` at top.

```bash
git log --oneline -5            # top must be 3eaf2fc2
git status --short              # ?? on TEMP/*.md only; no M on source files
bash scripts/k3d_env.sh -e k3d-cranium python benchmarks/sovereign_bitnet_attention.py --quick
```

If smoke passes (exit 0, JSON produced, five gates hold), continue to full:

```bash
mkdir -p logs data/benchmarks
TS=$(date +%Y%m%d_%H%M%S)
bash scripts/k3d_env.sh -e k3d-cranium \
     python benchmarks/sovereign_bitnet_attention.py \
     2>&1 | tee "logs/bitnet_attention_run_${TS}.log"
```

## What to expect

- `[loader] CUDA context materialized (pid=<N>, device=0, ctx=0x<hex>)` on stdout, exactly once.
- `observe_vram()` → `loader.get_vram_usage()` should now return real `used`/`total` bytes instead of 201.
- The benchmark proceeds to PTX load, buffer alloc, kernel launches, convergence check.

## Three possible outcomes — all valid reports

1. **Smoke passes** → run full → report seven `key = value` lines + SHA per PASS format.
2. **Smoke fails at `get_vram_usage`** → context is still being invalidated elsewhere → paste full traceback; we diagnose a third offender.
3. **Smoke fails downstream** (PTX load, kernel launch, convergence, sentinel, allocation) → **new signal** → paste the traceback. This is what we want to reach next; a real PTX-path failure is progress.

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
git commit -m "bench(sovereign-bitnet): honest sovereign attention run post ctx-ownership fix — ${TS}"
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
- No reuse of prior JSON artifacts.
- No push, amend, or `--no-verify`.
- The faked-artifact rule from `memory/feedback_codex_cannot_silent_fix_to_unblock.md` still stands.

## Post-compaction reload protocol

1. Re-read this file.
2. Re-read `TEMP/CODEX_RERUN_NOTE_SOVEREIGN_BITNET_04.18.2026.md`, `TEMP/CODEX_HANDOFF_04.18.2026_VSCODE.md`, `TEMP/CODEX_RUNBOOK_SOVEREIGN_BITNET_BENCHMARK_04.18.2026.md`.
3. `git log --oneline -5` — top must be `3eaf2fc2`.
4. `git status --short` — expect `??` on `TEMP/*.md` only.

---

## One-sentence summary

`galaxy_buffer` was releasing loader's context; loader was calling `cuCtxSetCurrent` redundantly after `cuCtxCreate`; both patched in `3eaf2fc2`; re-run and report honestly.
