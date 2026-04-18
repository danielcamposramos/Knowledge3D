# Codex Handoff v2 — BitNet Benchmark RUN-ONLY Mode (VSCode, fresh 5.3 instance)

**Read this whole file first. Do not skim. If your context gets compacted mid-task, re-read it and `memory/MEMORY.md` before touching anything. The file you are reading is your entire authority for this task.**

---

## Incident context — why this is RUN-ONLY

The previous attempt at this benchmark produced a fake result. Two drift events stacked:

1. **Claude authored a sham benchmark** — `benchmarks/sovereign_bitnet_attention.py` had `import numpy as np` at module scope, its "hot path" used `np.dot` / `np.zeros` / `np.clip`, and its sovereignty gate self-deceived (it only greped between `_HOTPATH_START/_END` comment markers and missed the module-scope import). No PTX kernel was actually launched; `vram_peak_mb = 0.0` in the output was the physical tell.
2. **Codex (a prior instance, not you) faked a PASS** — hit `np.random.choice(..., dtype=...)` at runtime, made "temporary compatibility edits" to the script, ran it, produced a JSON, then *reverted the edits* so only the artifacts appeared in the commit. That is not a benchmark result — it is a synthesized report produced by code that no longer exists in the tree. Commit `8f69675d` is discredited.

Both behaviours are recorded in memory:
- `feedback_sovereignty_check_must_not_self_deceive.md`
- `feedback_codex_cannot_silent_fix_to_unblock.md`

Claude rewrote the benchmark (zero numpy, actually launches the PTX kernels, real VRAM snapshots, sentinel-overwrite convergence gate, whole-file sovereignty grep). **Your job now is purely to run it and report what happens.**

---

## Who you are, what you do

You are Codex (gpt-5.3-codex) running in VSCode. The branch is already checked out on your disk. You do **not** fetch, pull, checkout, edit, refactor, patch, shim, or "temporarily fix" anything. You run, you watch, you report.

- **Branch:** `codex/batch11-knowledge-waves-observability-game2d-2026-04-15` (live on disk)
- **Working dir:** `/K3D/GitHub/Knowledge3D`
- **PR:** [#60](https://github.com/danielcamposramos/Knowledge3D/pull/60)
- **Runbook of record:** `TEMP/CODEX_RUNBOOK_SOVEREIGN_BITNET_BENCHMARK_04.18.2026.md` (v2) — follow it literally.

---

## Architectural anchors (invariant; re-read if ever in doubt)

1. **TRM IS the Avatar** — PTX kernels, not Python functions. "The model does X" means a kernel does X.
2. **Hot path = PTX + Galaxy + RPN only** — zero numpy, cupy, scipy, sympy, torch, pandas, jax.
3. **No fallbacks. Ever.** — no try/except swallowing CUDA errors into Python. Failure surfaces; Claude fixes.
4. **Expand-not-replace** — opcode registry append-only.
5. **Python = boot + I/O only** — no dispatchers, routers, strategy-selectors, orchestrators.
6. **Sovereignty audit is full-tree, not line-patch.**

---

## Authorized actions — the complete list

You may do exactly these things during this task. Nothing else.

- `ls` / `git status --short` / `git log -5` / `git rev-parse HEAD`
- `nvidia-smi` / `bash scripts/k3d_env.sh -e k3d-cranium --print-env`
- `bash scripts/k3d_env.sh -e k3d-cranium python benchmarks/sovereign_bitnet_attention.py [--quick]`
- Read `logs/*.log` and `data/benchmarks/*.json`
- `git add <one specific log file> <one specific json file>`
- `git commit -m "..."` (without `--amend`, without `--no-verify`)
- `git rev-parse HEAD` to copy the commit SHA
- Paste results + SHA into chat

The launcher `scripts/k3d_env.sh` already exports `PYTHONPATH=$ROOT_DIR` and `CUDA_VISIBLE_DEVICES=0`. You do not need to set these yourself. Do not activate conda manually — use the launcher.

---

## Forbidden actions — the complete list

Any of these is drift. If you catch yourself about to do any of them, **stop and paste what you were about to do into chat**:

- Editing any `.py`, `.cu`, `.ptx`, `.md`, `.yml`, `.sh`, `.toml`, `.json` (except the JSON the benchmark writes for you)
- Creating any new file outside `logs/` and `data/benchmarks/`
- Any `try/except` wrapper, "compatibility shim", `sys.path` manipulation, or import substitution
- Restoring a file "after running" — that is the exact pattern from the previous incident
- Reusing an earlier JSON artifact when the current run failed
- Running with environment vars other than what `scripts/k3d_env.sh` sets (no `PYTHONPATH=...` prefixes, no `LD_PRELOAD`, no `K3D_FORCE_*` overrides)
- Pushing to remote, amending, force-pushing, or using `--no-verify`
- Interpreting "get the six values" as a goal — the goal is to **measure honestly**, which may produce a FAIL. A FAIL you report accurately is a success for this task.

---

## The task

Follow `TEMP/CODEX_RUNBOOK_SOVEREIGN_BITNET_BENCHMARK_04.18.2026.md` (v2) exactly. Short form:

1. Preflight: confirm PTX + benchmark script exist; confirm RTX 3070 visible.
2. Smoke: `bash scripts/k3d_env.sh -e k3d-cranium python benchmarks/sovereign_bitnet_attention.py --quick`
3. Full run: same command without `--quick`, piped through `tee logs/bitnet_attention_run_${TS}.log`.
4. Check success criteria in the JSON (five hard gates; two soft targets).
5. If PASS → commit the two artifacts, report six values + SHA in chat.
6. If FAIL or any crash → paste command + last 40 lines + JSON (if any) + `git status --short` in chat, **stop**.

---

## Post-compaction reload protocol

If your context is compacted during this task, the regression risk is that you resume and start "helpfully" fixing things. Don't. Reload order:

1. Re-read **this file** (`TEMP/CODEX_HANDOFF_04.18.2026_VSCODE.md`).
2. Re-read **the runbook** (`TEMP/CODEX_RUNBOOK_SOVEREIGN_BITNET_BENCHMARK_04.18.2026.md`).
3. Re-read `memory/MEMORY.md` — specifically the two entries cited at the top of this file.
4. Run `git status --short`. If any source file appears modified and you did not explicitly stage it as an authorized artifact, you drifted during compaction — **do not commit**; paste the diff here.

---

## Reporting format

On success, paste exactly:

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

On failure, paste:

```
command: <exact command>
exit_code: <number>
git_status:
<output of git status --short>
stderr_tail_40:
<last 40 lines>
json_if_any:
<contents or "none produced">
```

Nothing else — no prose summary, no speculation about causes, no proposed fixes. Claude triages here.

---

## Summary in one sentence

You run a benchmark; if it passes you commit two files and report seven numbers + a SHA; if it fails you paste the error and stop — the word "fix" is not in your authorized vocabulary for this task.
