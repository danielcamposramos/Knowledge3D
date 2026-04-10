# Codex Direction: ARC R0 — First Real Scores

**Date:** 2026-04-08
**Depends on:** R0 infrastructure ✅ (arc2_local_runner.py, arc3_sdk_agent.py, arc_verification.cu all green)
**Goal:** Get the first real ARC-2 score from the closed CAS/SAS loop, and get ARC-3 running
         on the live server. Both results feed directly into the paper's evidence bundle.
**Do not touch:** protected encyclopedia ingest (PID 101379)

---

## How secrets are stored in this project (read docs/ENV_POLICY.md)

API keys are NEVER stored in the repository. The canonical pattern:

```
/K3D/Knowledge3D.local/secrets/{service}_api_key.txt
```

This path is on the local SSD, physically outside the git working tree — it can never
be committed. Runtime code checks `os.environ.get("ARC_API_KEY")` first, then falls
back to reading the file directly. The `_resolve_api_key()` function in
`benchmarks/arc3_sdk_agent.py` already implements this correctly.

**Daniel:** If you haven't placed the ARC API key yet, run this once in your terminal:
```bash
mkdir -p /K3D/Knowledge3D.local/secrets
echo "your_arc_api_key_here" > /K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt
chmod 600 /K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt
```
Then pass this spec to Codex — Step 4 will confirm the key is present without printing it.

---

## Step 1 — ARC-2: Find or fetch the task data

`arc2_local_runner.py._resolve_tasks_dir()` tries several candidate paths. Check which exists:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python -c "
from benchmarks.arc2_local_runner import _resolve_tasks_dir
p = _resolve_tasks_dir(None)
print('found:', p, '| exists:', p.exists())
"
```

**If a path exists and is non-empty:** proceed directly to Step 2.

**If none exist:** download the public ARC-AGI evaluation set (MIT licensed, ~400 KB)
into the canonical local datasets root:

```bash
mkdir -p /K3D/K3D_llama_cpp/datasets
cd /K3D/K3D_llama_cpp/datasets
git clone --depth 1 https://github.com/fchollet/ARC-AGI.git ARC-AGI-master 2>/dev/null \
  || (cd ARC-AGI-master && git pull --depth 1)
```

Re-run the path check to confirm. The evaluation directory should be
`/K3D/K3D_llama_cpp/datasets/ARC-AGI-master/data/evaluation/` and contain ~400 JSON files.

---

## Step 2 — ARC-2: Run the closed-loop evaluation (20 tasks)

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python benchmarks/arc2_local_runner.py \
  --max-tasks 20 \
  --summary-output /tmp/arc2_r0_summary.json
```

This prints one JSON line per task as it runs, then the final summary.

**Expected honest result:** 0–15% exact match. The current pipeline:

- First tries exact `k3d_pattern_match` against each seeded training input
- Falls back to nearest-training-pair grid distance when no exact match

Most tasks will score via nearest-pair fallback — correct and expected for R0.
The score improves in R1 with compositional transformation rules.

Capture and include in the report: `tasks`, `correct`, `total_inputs`, `score`, and
the distribution of `match_type` values (exact_poly_match vs nearest_training_pair).

---

## Step 3 — ARC-2: Generate one Kaggle-format submission artifact

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python benchmarks/arc2_local_runner.py \
  --max-tasks 20 \
  --submission-output /tmp/arc2_r0_submission.json \
  --summary-output /tmp/arc2_r0_summary.json
```

Report whether `submission.csv` was generated and passes `validate_arc_submission()`.
Print the first 5 lines of the CSV so the format is visible in the report.

---

## Step 4 — ARC-3: Check the API key (safe check — never print the value)

```bash
# Check file exists and is non-empty — does NOT print the key value
test -s /K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt \
  && echo "KEY FILE: present and non-empty" \
  || echo "KEY FILE: missing or empty"

# Check env var (also safe — only prints SET or NOT SET)
bash scripts/k3d_env.sh run -e trmc_core python -c "
import os
k = os.environ.get('ARC_API_KEY', '').strip()
print('ARC_API_KEY env:', 'SET' if k else 'NOT SET')
"
```

**If the key exists (either source):** proceed to Step 5.

**If neither exists:** report clearly and skip Step 5. Do NOT fabricate, hardcode,
or guess a key. The remote compat layer is wired and ready — it just needs the credential.

---

## Step 5 — ARC-3: Run ls20 via remote compat (only if key is confirmed in Step 4)

```bash
bash scripts/k3d_env.sh run -e trmc_core python -c "
from benchmarks.arc3_sdk_agent import K3DAgent
import json
agent = K3DAgent('ls20', max_steps=60, allow_remote_compat=True)
try:
    result = agent.run_level()
finally:
    agent.close()
print(json.dumps(result, indent=2))
"
```

This uses `_RemoteArcCompatEnv` to talk to `https://three.arcprize.org` directly via
the same HTTP API that completed Level 1 on March 30, 2026 (13 actions, ls20-9607627b).

Report: `steps`, `levels_completed`, `score`, `transport`, `sdk_error`.

---

## Step 6 — Investigate the installed arc-agi package surface

Run regardless of Step 5 outcome — this documents what the installed package exposes
so the SDK loader can be updated if a newer version with `Arcade/make` is available:

```bash
bash scripts/k3d_env.sh run -e trmc_core python -c "
import arc_agi
print('arc_agi version:', getattr(arc_agi, '__version__', 'unknown'))
print('arc_agi API surface:', [x for x in dir(arc_agi) if not x.startswith('_')])
try:
    import arcengine
    print('arcengine surface:', [x for x in dir(arcengine) if not x.startswith('_')][:20])
except ImportError as e:
    print('arcengine not importable:', e)
try:
    import arc
    print('arc surface:', [x for x in dir(arc) if not x.startswith('_')])
except ImportError as e:
    print('arc not importable:', e)
"
```

Include the full output in the report.

---

## What NOT to do

- Do NOT store any key value in a file inside the repository working tree
- Do NOT print or log the key value anywhere — only print SET/NOT SET/present/missing
- Do NOT wait for the encyclopedia ingest to finish (Steps 1–6 are independent of it)
- Do NOT run more than 20 ARC-2 tasks (keeps R0 fast and readable)
- Do NOT claim a score higher than what the runner actually produces
- Do NOT change the hot path or sovereignty rules

---

## Report back

Write `TEMP/CODEX_TO_CLAUDE_ARC_R0_RUN_REPORT_2026-04-08.md` with:

1. ARC-2 task path found (or downloaded to)
2. ARC-2 score: `tasks=N, correct=K, total_inputs=M, score=X.XX%`
3. `match_type` distribution across the 20 tasks
4. `submission.csv` generated: yes/no, first 5 lines
5. ARC-3 API key: present/missing (do NOT include the key value)
6. ARC-3 result (if run): `steps, levels_completed, transport, sdk_error`
7. Installed `arc-agi` package surface (full output of Step 6)
8. Protected ingest PID 101379: still alive? current runtime?
