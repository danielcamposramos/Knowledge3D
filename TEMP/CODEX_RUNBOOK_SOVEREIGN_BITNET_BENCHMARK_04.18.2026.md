# Codex Runbook — Sovereign BitNet Attention Benchmark (2026-04-18)

**Role:** Codex (execution + observation, NOT authoring). This runbook is self-contained — do not assume you have read the conversation that produced it.

**Purpose:** Run the first sovereign BitNet b1.58 attention benchmark on RTX 3070, verify compliance, and commit artifacts.

**Depends on (must be landed first):**
- Lane 1: kernel compiled to PTX and registered in the loader
- Lane 2: `benchmarks/sovereign_bitnet_attention.py` exists and is executable

---

## 1. Preflight

```bash
# 1a. Sync branch
cd /K3D/GitHub/Knowledge3D
git fetch origin
# If PR is merged:
git checkout main && git pull origin main
# If PR not yet merged, stay on the feature branch:
git checkout codex/batch11-knowledge-waves-observability-game2d-2026-04-15
git pull

# 1b. Activate conda env (SSD-materialized prefix)
source /home/daniel/miniforge/bin/activate
conda activate /K3D/Knowledge3D.local/envs/k3d-cranium
# Sanity: which python should resolve under /K3D/Knowledge3D.local/envs/k3d-cranium/bin
which python

# 1c. GPU visibility — MUST be exported BEFORE any tmux session or benchmark process
export CUDA_VISIBLE_DEVICES=0
# Sanity: RTX 3070 should be device 0 with ~12 GB free VRAM
nvidia-smi --query-gpu=name,memory.free --format=csv

# 1d. Verify kernel PTX artifact exists (landed by Lane 1)
ls -la knowledge3d/cranium/kernels/bitnet_attention.ptx \
  || { echo "KERNEL MISSING — recompile per Section 4.6"; exit 1; }

# 1e. Verify benchmark script exists (Lane 2)
test -f benchmarks/sovereign_bitnet_attention.py \
  || { echo "BENCHMARK MISSING — abort and report to Lane 2"; exit 1; }
```

**Abort conditions:** any of the four sanity checks (python path, GPU visible, PTX present, script present) fail. Do **not** proceed with workarounds — file a blocker against the responsible lane.

---

## 2. Run the benchmark

```bash
# 2a. Quick smoke (sub-second; exercises loader + one forward pass)
python benchmarks/sovereign_bitnet_attention.py --quick

# 2b. Full run with timestamped log capture
mkdir -p logs data/benchmarks
TS=$(date +%Y%m%d_%H%M%S)
python benchmarks/sovereign_bitnet_attention.py \
    2>&1 | tee "logs/bitnet_attention_run_${TS}.log"
```

The JSON result is written by the script to `data/benchmarks/sovereign_bitnet_attention_${TS}.json` (confirm filename in Lane 2 output).

---

## 3. Live monitoring — what to watch

Watch the tee'd log in a second pane. These five lines are load-bearing:

| Signal | Expected | If wrong → |
|---|---|---|
| `sovereign_compliance=PASS` | PASS | Section 4.1 |
| `latency_ms_per_query.p50` | ≤ 5.0 ms (64-star × 64-dim) | Section 4.2 |
| `top_k_consistency` | 1.0 (identical top-K across 3 repeats of same query) | Section 4.3 |
| `vram_peak_mb` | well under 12288 MB | Section 4.4 |
| `kernel_path_trace` contains | 0x1AA, 0x1AB, 0x1AC, 0x1AD, 0x1AE (0x1AF only if Path B lane-switch is exercised) | Section 4.5 |

In a side pane:
```bash
watch -n1 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
```

---

## 4. Failure triage — root-cause only, no `try/except` swallowing

**No Python fallbacks. No retry loops. Failure surfaces; we fix the cause.**

### 4.1 `sovereign_compliance=FAIL`
The benchmark found a non-sovereign import (numpy, cupy, scipy, sympy, torch, etc.) reachable from the hot path.
```bash
# Read the flagged file and line from the log, then archive per migration manifest:
mkdir -p Old_Attempts/2026-04-18/
git mv <flagged_file> Old_Attempts/2026-04-18/
# Update manifest: Old_Attempts/2026-04-18/MANIFEST.md with reason and replacement pointer.
```
Do not patch the forbidden import out in place — archive the whole file and let the sovereign replacement take over.

### 4.2 `latency_ms_per_query.p50 > 5.0`
Likely causes: (a) PTX launched with wrong grid/block, (b) un-pinned host memcpy, (c) missing `--release` flags on nvcc.
```bash
# Recompile with the documented flags (Lane 1 canonical command):
nvcc --ptx -arch=sm_86 -O3 --use_fast_math -I. \
  knowledge3d/cranium/kernels/bitnet_attention.cu \
  -o knowledge3d/cranium/kernels/bitnet_attention.ptx
# Verify PTX header declares sm_86:
head -5 knowledge3d/cranium/kernels/bitnet_attention.ptx
```
If build flags are correct, report a profiling request (Nsight Compute trace) — do NOT mask latency with fewer iterations.

### 4.3 `top_k_consistency < 1.0`
Contrastive margin is deterministic only when the tie-break rule is stable. Report:
```bash
# Capture the seed, the two diverging top-K lists, and the query vector that triggered it.
grep -E "seed=|query_id=|top_k=" logs/bitnet_attention_run_${TS}.log | tail -50
```
File as a defect on the contrastive-attention spec (not a runtime workaround).

### 4.4 `vram_peak_mb` approaching 12 GB
Unexpected for 64-star × 64-dim. Check for accidental fp32 temporaries or leaked allocations. Do not raise the budget — fix the leak.

### 4.5 Missing opcodes in `kernel_path_trace`
The kernel dispatch table didn't route through all six attention opcodes. Inspect the loader registration table (owned by Lane 1) and confirm 0x1AA–0x1AF are mapped. 0x1AF only fires if the Path B lane-switch test case runs — note absence is OK only if that case was intentionally skipped.

### 4.6 Kernel load failure
```bash
# Recompile and capture full nvcc output for sm_86:
nvcc --ptx -arch=sm_86 -O3 --use_fast_math -I. \
  knowledge3d/cranium/kernels/bitnet_attention.cu \
  -o knowledge3d/cranium/kernels/bitnet_attention.ptx \
  2>&1 | tee logs/ptx_build_${TS}.log
nvcc --version
```
If nvcc is missing, the conda env drifted — recreate:
```bash
conda env remove -p /K3D/Knowledge3D.local/envs/k3d-cranium
bash scripts/k3d_env.sh k3d-cranium
```

---

## 5. Success criteria (Daniel's sign-off gate)

All six MUST hold:

- [ ] `sovereign_compliance == "PASS"`
- [ ] `latency_ms_per_query.p50 <= 5.0`
- [ ] `top_k_consistency == 1.0`
- [ ] `convergence_verified == true`
- [ ] `kernel_path_trace` includes 0x1AA, 0x1AB, 0x1AC, 0x1AD, 0x1AE (0x1AF present if Path B case ran)
- [ ] No Python in hot path (verified by the compliance line, not by eye)

If any fail, do not commit artifacts — triage per Section 4 and re-run.

---

## 6. Commit artifacts

```bash
git add "data/benchmarks/sovereign_bitnet_attention_${TS}.json"
git add "logs/bitnet_attention_run_${TS}.log"

git commit -m "bench(sovereign-bitnet): first sovereign attention run results — ${TS}"
```

Do **not** amend; do **not** force-push. Do **not** auto-push to remote — Daniel pushes after reviewing the JSON.

---

## 7. Report back to Daniel

Paste in the chat:
1. The six success-criteria lines from the JSON (copy the values, not the keys).
2. The commit SHA (`git rev-parse HEAD`).
3. Any triage you performed (which section of 4 and what fix).

That's the handoff. Do not summarize the run prose — Daniel reads the JSON directly.
