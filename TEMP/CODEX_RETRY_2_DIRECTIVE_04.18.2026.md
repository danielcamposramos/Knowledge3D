# Codex Retry Directive #2 — Warmup Fix Round 2

**Date:** 2026-04-18 (after the first fix failed)
**Role:** Codex — RUN-ONLY. No edits. No fixes. Same contract as every prior directive.

---

## What your last run showed

```
exit_code: 1
RuntimeError: Sovereign loader error: invalid device context
  at loader.get_vram_usage() line 1003
commit_head: ae3857163c2e2b545bc7229b1fd19a17990774a7
```

Real signal. You reported correctly — no drift. The first warmup fix (commit `4ec90252`) didn't resolve it.

## Root cause (diagnosed by Claude)

Two compounding defects in the prior fix:

1. **Zero-size allocation did not materialize the lazy context.** CUDA drivers can return `res=0` with `d_temp.value=0` (null pointer) for zero-size requests — no device-side allocation actually happens, the lazy primary context stays unmaterialized.
2. **Warmup was gated to the primary-retain fallback path only.** If `cuCtxCreate` succeeded (the common path), no warmup fired — and if that ctx was somehow lazily materialized, the bug persisted.

## What landed now (commit `40b98338`)

- Warmup moved OUT of the primary-retain branch to the end of `_ensure_init` — runs on BOTH paths unconditionally.
- Allocation size changed from 0 to 16 bytes — guarantees real device-side alloc, which materializes the context.
- Added a **one-shot unconditional boot diagnostic print** at end of `_ensure_init`. You will see one line on stdout like:
  ```
  [loader] CUDA context materialized (pid=<N>, device=0, ctx=<0x...>)
  ```
  This is NEW — expect to see it exactly once per process. It is not an error. If you do NOT see it before the traceback, init itself failed and you should paste stderr as usual.

## Your task — re-run, unchanged

```bash
# Preflight (same as before — but don't stop on `??` TEMP files, those are expected)
git log --oneline -5
# Top commit should be 40b98338 ← the new fix

# Smoke:
bash scripts/k3d_env.sh -e k3d-cranium python benchmarks/sovereign_bitnet_attention.py --quick

# If smoke passes, full:
mkdir -p logs data/benchmarks
TS=$(date +%Y%m%d_%H%M%S)
bash scripts/k3d_env.sh -e k3d-cranium \
     python benchmarks/sovereign_bitnet_attention.py \
     2>&1 | tee "logs/bitnet_attention_run_${TS}.log"
```

## Reporting rules — unchanged

- **PASS** → seven values + SHA (see CODEX_RERUN_NOTE_SOVEREIGN_BITNET_04.18.2026.md §"On PASS").
- **FAIL** → paste command + exit code + git status + last 40 stderr lines + JSON-if-any + commit_head.
- **Preflight-only success is not a FAIL.** If preflight succeeds, proceed to smoke, do not stop and report.

The `[loader] CUDA context materialized ...` line in stdout is diagnostic output, not an error. Include it in your stderr/stdout tail if present — it helps triage.

## Forbidden — same as ever

No edits to any source file. No "temporary compatibility fixes." No reuse of prior JSON artifacts. No push, amend, or `--no-verify`. The faked-artifact rule from `memory/feedback_codex_cannot_silent_fix_to_unblock.md` still stands.

## Post-compaction reload protocol

If context is compacted during this run:

1. Re-read this file.
2. Re-read `TEMP/CODEX_RERUN_NOTE_SOVEREIGN_BITNET_04.18.2026.md`.
3. Re-read `TEMP/CODEX_HANDOFF_04.18.2026_VSCODE.md` and `TEMP/CODEX_RUNBOOK_SOVEREIGN_BITNET_BENCHMARK_04.18.2026.md`.
4. `git log --oneline -5` — if `40b98338` is NOT present at top, you are on the wrong branch or state drifted.
5. `git status --short` — expect `??` for `TEMP/*.md` files only. If `M` on any source file, drift.

---

## Summary

New fix: warmup is now unconditional + 16 bytes + self-reports. If it still fails, the failure mode will be visible (the new diagnostic print will show ctx handle or init will die before the print). Either way, honest data — report it.
