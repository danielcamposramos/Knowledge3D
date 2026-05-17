# Codex Re-run Note — Sovereign BitNet Attention, post single-context fix

**Date:** 2026-04-18 (afternoon, after Claude sub-agents landed the fix)
**Role:** Codex — **RUN-ONLY**. Same contract as [TEMP/CODEX_HANDOFF_04.18.2026_VSCODE.md](CODEX_HANDOFF_04.18.2026_VSCODE.md). If you drift, the faked-artifact memory entry gets a v2.
**Branch:** `codex/batch11-knowledge-waves-observability-game2d-2026-04-15` (live on your disk)

---

## What changed since your last run

Your previous attempt failed with `CUDA_ERROR_INVALID_CONTEXT (201)` on `loader.get_vram_usage()`. You reported and stopped — correct. Claude + internal sub-agents (not you) diagnosed and fixed it. Three commits landed on the branch:

```
ae385716  refactor(sovereign): migrate 6 bindings to shared-context loader pattern
5aafdc78  chore(sovereign): enforce single-context invariant via CI gate + test migration
4ec90252  fix(sovereign-loader): warm up lazy primary ctx + daemon boot init
```

The load-bearing one for you is `4ec90252`: `loader._ensure_init()` now materializes the lazy primary context via a zero-size `cuMemAlloc + cuMemFree` on the fallback path, so `cuMemGetInfo` succeeds on first call. No other behavior changed.

The CI gate is clean (`bash scripts/check_single_context_invariant.sh` → exit 0).

---

## Your task — unchanged from the original runbook

Follow [TEMP/CODEX_RUNBOOK_SOVEREIGN_BITNET_BENCHMARK_04.18.2026.md](CODEX_RUNBOOK_SOVEREIGN_BITNET_BENCHMARK_04.18.2026.md) (v2) exactly. No edits. No shims. No "temporary compatibility fixes." No reuse of earlier JSON artifacts.

1. **Preflight:**
   ```bash
   git log --oneline -5
   # Top three should be ae385716, 5aafdc78, 4ec90252. If not, you are on the wrong branch — stop.
   git status --short
   # Must be clean (no modified source files). If anything is modified, stop and paste the diff.
   bash scripts/k3d_env.sh -e k3d-cranium --print-env
   nvidia-smi --query-gpu=name,memory.free --format=csv
   # Must show RTX 3070. If not, stop.
   ```

2. **Smoke (quick):**
   ```bash
   bash scripts/k3d_env.sh -e k3d-cranium python benchmarks/sovereign_bitnet_attention.py --quick
   ```

3. **Full run with capture:**
   ```bash
   mkdir -p logs data/benchmarks
   TS=$(date +%Y%m%d_%H%M%S)
   bash scripts/k3d_env.sh -e k3d-cranium \
        python benchmarks/sovereign_bitnet_attention.py \
        2>&1 | tee "logs/bitnet_attention_run_${TS}.log"
   ```

4. **Check the JSON (`data/benchmarks/sovereign_bitnet_attention_<unix_ts>.json`). Success = all five:**
   - `sovereign_compliance == "PASS"`
   - `convergence_verified == true`
   - `min_sentinel_overwrites_per_query > 0`
   - `kernel_path_trace` contains exactly `0x1AA:k3d_bitnet_attention_proj`, `0x1AC:unpack5(device,via_matmul_tile)`, `0x1AE:k3d_attention_contrastive_rank(path_a)`, `0x1AF:k3d_attention_contrastive_rank(path_b)`
   - `vram_peak_mb > 0.0` ← **this is the value that was 0.0 in the sham run; must now be a real number**

5. **On PASS → commit + report (see below).**
6. **On FAIL → paste + stop (see below).**

---

## The invariant Daniel + Claude are watching for this time

The fix landed ONLY makes `cuMemGetInfo` work. It does NOT guarantee the benchmark passes. If the PTX kernels have a bug, the run will fail in a DIFFERENT way (convergence, sentinel, allocation), and that is **real information** — report it honestly. **A FAIL reported accurately is a success for this task. A PASS-shaped report synthesized from edits is drift.**

The physical tells Claude will check when you report:
- `vram_peak_mb > 0` — confirms kernels actually allocated. If it's 0.0 again and you report PASS, the report is fake.
- `git status --short` between run and commit must show ONLY `logs/*.log` and `data/benchmarks/*.json` as untracked/new. Any modified source file → drift, do not commit.
- The JSON's `kernel_path_trace` must be derived by the benchmark's kernel invocations, not hardcoded. The benchmark emits the trace from real launches — you cannot influence this without editing code.

---

## Forbidden actions (verbatim from handoff — re-read if unclear)

Any of these = drift. Stop and paste what you were about to do.

- Editing any `.py`, `.cu`, `.ptx`, `.md`, `.yml`, `.sh`, `.toml`, `.json` (except the JSON the benchmark writes for you).
- Creating any file outside `logs/` and `data/benchmarks/`.
- `try/except` wrappers, "compatibility shims", `sys.path` manipulation, import substitution.
- "Temporarily fixing" then "restoring" — the exact pattern from the `8f69675d` incident. Recorded in `memory/feedback_codex_cannot_silent_fix_to_unblock.md`.
- Reusing an earlier JSON artifact when the current run failed.
- Running with env vars other than what `scripts/k3d_env.sh` sets. No `PYTHONPATH=` prefixes, no `LD_PRELOAD`, no `K3D_FORCE_*` overrides.
- Pushing, amending, force-pushing, or `--no-verify`.
- Interpreting "get the six values" as a goal. The goal is to **measure honestly**. A FAIL is valid.

---

## Reporting format

### On PASS — paste exactly (values, not keys):

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

Then commit (artifacts only):

```bash
git status --short
# Must show ONLY the two new files:
#   data/benchmarks/sovereign_bitnet_attention_<ts>.json
#   logs/bitnet_attention_run_<ts>.log
# If anything else is modified or untracked, STOP and paste git status here.

git add "data/benchmarks/sovereign_bitnet_attention_<ts>.json" \
        "logs/bitnet_attention_run_${TS}.log"

git commit -m "bench(sovereign-bitnet): honest sovereign attention run post single-context fix — ${TS}"
git rev-parse HEAD
```

Do NOT push. Do NOT amend. Do NOT `--no-verify`. Daniel pushes after review.

### On FAIL — paste exactly:

```
command: <exact command you ran>
exit_code: <number>
git_status:
<output of git status --short>
stderr_tail_40:
<last 40 lines of stderr or log>
json_if_any:
<contents of the JSON if one was produced, or "none produced">
commit_head: <output of git rev-parse HEAD>
```

Nothing else. No prose summary. No speculation about causes. No proposed fixes. Claude + sub-agents triage here.

---

## Post-compaction reload protocol

If your context is compacted during this task:

1. Re-read THIS FILE (`TEMP/CODEX_RERUN_NOTE_SOVEREIGN_BITNET_04.18.2026.md`).
2. Re-read `TEMP/CODEX_HANDOFF_04.18.2026_VSCODE.md`.
3. Re-read `TEMP/CODEX_RUNBOOK_SOVEREIGN_BITNET_BENCHMARK_04.18.2026.md`.
4. Re-read `memory/feedback_codex_cannot_silent_fix_to_unblock.md` + `memory/feedback_sovereignty_check_must_not_self_deceive.md`.
5. Run `git status --short`. If any source file appears modified and you did not explicitly stage it as an authorized artifact, you drifted during compaction — **do not commit**, paste the diff here.

---

## Summary in one sentence

Three Claude-authored commits unblocked the context issue; you run the benchmark, if it passes you commit two files and report seven numbers + SHA, if it fails you paste the error and stop — the word "fix" remains outside your authorized vocabulary.
