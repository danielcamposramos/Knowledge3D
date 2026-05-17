# Codex Runbook — Sovereign BitNet Attention Benchmark (2026-04-18, v2)

**Role:** Codex — **EXECUTION ONLY**. Run the benchmark; watch the log; report numbers. Do NOT edit source. Do NOT "fix" errors. If anything fails, paste the error here and stop. This runbook is self-contained.

**Why v2:** v1 had Claude-authored numpy inside the benchmark (sovereignty-check bypass) and a compliance gate that self-deceived. That whole file was rewritten. Any earlier artifact produced by v1 is invalid.

**What landed (already on disk in your VSCode workspace):**
- `benchmarks/sovereign_bitnet_attention.py` — rewritten, zero numpy, actually launches PTX kernels via the sovereign loader.
- `knowledge3d/cranium/kernels/bitnet_attention.ptx` — compiled, sm_86.
- Handoff file: `TEMP/CODEX_HANDOFF_04.18.2026_VSCODE.md` — read that for the architectural anchors.

---

## 1. Preflight (one command)

The project launcher `scripts/k3d_env.sh` sets `PYTHONPATH=$ROOT_DIR`, activates the SSD-materialized env, and exports `CUDA_VISIBLE_DEVICES=0`. Use it — do not activate conda manually.

```bash
# Sanity — everything present?
ls -la knowledge3d/cranium/kernels/bitnet_attention.ptx \
       benchmarks/sovereign_bitnet_attention.py

# GPU visible?
bash scripts/k3d_env.sh -e k3d-cranium --print-env
nvidia-smi --query-gpu=name,memory.free --format=csv
```

Abort conditions (all hard-stop, **do not try to fix**):
- PTX missing → report here; we recompile centrally.
- Benchmark script missing → report here; we re-land it.
- `nvidia-smi` shows something other than RTX 3070 → iGPU / env problem; report here.

---

## 2. Run

```bash
mkdir -p logs data/benchmarks
TS=$(date +%Y%m%d_%H%M%S)

# Smoke (3 queries, sub-second):
bash scripts/k3d_env.sh -e k3d-cranium python benchmarks/sovereign_bitnet_attention.py --quick

# Full run (30 queries, timestamped capture):
bash scripts/k3d_env.sh -e k3d-cranium \
     python benchmarks/sovereign_bitnet_attention.py \
     2>&1 | tee "logs/bitnet_attention_run_${TS}.log"
```

The script writes `data/benchmarks/sovereign_bitnet_attention_<unix_ts>.json`.

---

## 3. Success criteria — exactly these

Open the JSON (and/or grep the log). All five must hold:

- [ ] `sovereign_compliance == "PASS"`
- [ ] `convergence_verified == true`
- [ ] `min_sentinel_overwrites_per_query > 0` (proves the kernel actually wrote the output — not a stub)
- [ ] `kernel_path_trace` contains exactly: `0x1AA:k3d_bitnet_attention_proj`, `0x1AC:unpack5(device,via_matmul_tile)`, `0x1AE:k3d_attention_contrastive_rank(path_a)`, `0x1AF:k3d_attention_contrastive_rank(path_b)`
- [ ] `vram_peak_mb > 0.0` (real VRAM snapshot; 0.0 would mean no kernel ever allocated)

Soft targets (report the values, don't gate on them in this first run):
- `latency_ms_per_query.p50` — capture whatever it is
- `top_k_consistency` — should be 1.0 across the 30 queries (same fixture, same seed)

**Honest note (don't be alarmed):** 0x1AB `TERNARY_PACK5` and 0x1AD `VEC_NORM_L2_INT8` do NOT appear in the trace. That's correct — neither has a `__global__` wrapper in this kernel surface (pack5 is a host-side utility at weight upload; vec_norm is a device function awaiting a global wrapper in a later PR). The benchmark's `notes` field documents this explicitly.

---

## 4. If it fails — STOP AND REPORT

Do **not** edit the benchmark script. Do **not** edit any `.py` or `.cu` file. Do **not** run with a "temporary compatibility fix." Do **not** substitute a prior artifact.

Paste in chat:
1. The exact command you ran.
2. The last 40 lines of the log (or stderr if it crashed before writing).
3. The contents of the JSON if one was produced.
4. `git status --short` output.

That's it. We triage here.

---

## 5. If it passes — report these six lines and the SHA

Paste in chat (values, not keys):

```
sovereign_compliance = <value>
convergence_verified = <value>
min_sentinel_overwrites_per_query = <value>
kernel_path_trace = <array>
vram_peak_mb = <value>
latency_ms_per_query.p50 = <value>
commit = <will be filled after the artifact commit below>
```

Then commit artifacts only (no source changes):

```bash
git status --short
# The only expected new files are:
#   data/benchmarks/sovereign_bitnet_attention_<ts>.json
#   logs/bitnet_attention_run_<ts>.log
# If anything else shows as modified, STOP and report.

git add "data/benchmarks/sovereign_bitnet_attention_<ts>.json" \
        "logs/bitnet_attention_run_<ts>.log"

git commit -m "bench(sovereign-bitnet): honest sovereign attention run — ${TS}"
git rev-parse HEAD  # copy this SHA into the report
```

Do NOT push. Do NOT amend. Do NOT `--no-verify`. Daniel pushes after review.

---

## 6. Guardrails — what "execution only" means for this task

You are authorized to:
- Run `ls`, `git status`, `nvidia-smi`, `bash scripts/k3d_env.sh ...`, `python benchmarks/sovereign_bitnet_attention.py ...`.
- Read log files and JSON outputs.
- Commit the two produced artifacts.

You are **not** authorized to:
- Edit any `.py`, `.cu`, `.ptx`, `.md`, `.yml`, or `.sh` file.
- Write any new file outside `logs/` and `data/benchmarks/`.
- Add "temporary" compatibility shims, `try/except` wrappers, or import substitutions.
- Restore a file "after running" to hide edits.
- Reuse an earlier JSON artifact to synthesize a report.

If a bug blocks the run, your job is to **report the bug**, not to work around it. Claude and the sub-agents here will author the fix.
