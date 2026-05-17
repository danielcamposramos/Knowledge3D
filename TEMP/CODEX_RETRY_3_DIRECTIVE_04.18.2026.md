# Codex Retry Directive #3 — Diagnostic Print Fix

**Date:** 2026-04-18 (after retry #2 hit a print-format bug)
**Role:** Codex — RUN-ONLY. Same contract.

---

## What your retry #2 showed

```
exit_code: 1
ValueError: invalid literal for int() with base 10: b'\xc0 \x06fUV\x00\x00'
  at loader.py:437 in the boot diagnostic print
commit_head: 40b98338
```

**This is actually good news.** You reached the diagnostic print at the END of `_ensure_init` — which means the 16-byte warmup SUCCEEDED and the CUDA context is materialized. The crash was in my diagnostic print itself: `int(ctx)` on a `ctypes.c_void_p` is wrong; it needs `.value`.

The raw bytes `b'\xc0 \x06fUV\x00\x00'` in the error are the context pointer's memory repr — proof the context was created.

## What landed (commit `8a9bb30d`)

One-line fix: `int(ctx) if ctx else 0` → `(ctx.value or 0)` at line 437. No other changes. Top of branch should now show:

```
8a9bb30d fix(sovereign-loader): ctx.value not int(ctx) in boot diagnostic
40b98338 fix(sovereign-loader): unconditional 16-byte warmup + boot diagnostic
...
```

## Your task

Re-run the smoke benchmark. Your preflight `git log` should show `8a9bb30d` at top — if not, your VSCode workspace is stale.

```bash
git log --oneline -5  # top should be 8a9bb30d
bash scripts/k3d_env.sh -e k3d-cranium python benchmarks/sovereign_bitnet_attention.py --quick
```

## What to expect

The diagnostic print should now succeed and emit exactly one line like:

```
[loader] CUDA context materialized (pid=<N>, device=0, ctx=0x<hex>)
```

After that, the benchmark continues to `observe_vram()` → `loader.get_vram_usage()`. If the warmup actually worked (high confidence), this call will succeed and return real `used`/`total` bytes. The benchmark then proceeds to load PTX, allocate buffers, launch kernels. Whatever happens next is the FIRST actual test of the PTX kernels in this clean-context regime.

If smoke passes → run full per the original runbook. If smoke fails → paste FAIL report per contract (command, exit code, git status, stderr tail, JSON if any, commit_head).

## Same forbidden actions as always

No source edits. No shims. No artifact reuse. No push/amend/no-verify. If you see the `[loader] CUDA context materialized` line and then a different failure downstream, that's new information — report it honestly. A FAIL at the kernel launch or convergence stage is valuable — it means we got past init and into the PTX path.

---

Three possibilities for this run:

1. **Smoke passes** → run full → report seven values + SHA.
2. **Smoke fails at `get_vram_usage`** → warmup didn't actually materialize the ctx despite the successful alloc → paste the traceback, we diagnose again.
3. **Smoke fails elsewhere** (PTX load, kernel launch, convergence) → new signal, paste the traceback, we investigate downstream.

All three are valid reports. Pick based on what actually happens.
