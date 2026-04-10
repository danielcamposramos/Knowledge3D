# Codex Direction: Run All Locally Available Question-Based Benchmarks Together

**Date:** 2026-04-09
**Authority:** docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md §4.3
**Depends on:** Router+MMLU fix COMPLETE ✅, Swarm dispatch COMPLETE ✅, Micro-specialist pool LIVE ✅
**Context:** MMLU=45%, GSM8K=10%, both sovereign GPU path confirmed. Encyclopedia ingest PID 608109 running — do NOT touch it.

---

## Goal

Run all locally available question-based benchmarks through a single live Knowledgeverse boot,
sharing one galaxy load and one TRM session. No redundant boot. No redundant galaxy hydration.
One living AI, many benchmark windows.

---

## Benchmarks to Run (20 tasks each)

| Suite | Dataset | Dataset Root | Status |
|-------|---------|--------------|--------|
| MMLU | MMLU multiple-choice | `/K3D/K3D_llama_cpp/datasets/MMLU/data` | ✅ verified 45% |
| GSM8K | Grade-school math word problems | `/K3D/K3D_llama_cpp/datasets/GSM8K/grade_school_math/data/test.jsonl` | ✅ verified 10% |
| LHE | Last Humanity Exam (multi-domain) | `/K3D/K3D_llama_cpp/datasets/last_humanity_exam/last_humanity_exam.json` | dataset present |
| Omni-MATH | Competition math (4428 problems) | `/K3D/K3D_llama_cpp/datasets/Omni-Math.jsonl` | dataset present |
| AMC-AIME | AMC/AIME competition problems | `/K3D/K3D_llama_cpp/datasets/AMC-AIME/` | dataset present |
| IMO | IMO Bench (answerbench, gradingbench, proofbench) | `/K3D/Knowledge3D.local/datasets/imo_bench/` | dataset present |

**Excluded:**
- `math` suite: `/K3D/K3D_llama_cpp/datasets/math/data_train.jsonl` is 0 bytes — skip, `--math-count 0`
- ARC-2: not question-based — skip, `--arc2-count 0`
- ARC-3: archived, ARC key only for `three.arcprize.org` — skip entirely

---

## Part 1 — Path Fix: Omni-MATH Canonical Root

**File:** `benchmarks/math_competitions.py`
**Method:** `UnifiedMathBenchmark._load_omni_math_dataset()`

Current candidates list only checks for `Omni-Math.jsonl` inside an `Omni-MATH/` subdirectory.
The actual local file is at the dataset root directly: `root / "Omni-Math.jsonl"`.

**Fix:** Add the root-level file as the FIRST candidate, before the subdirectory entries:

```python
def _load_omni_math_dataset(self) -> list[dict[str, Any]]:
    root = self.dataset_path if self.dataset_path and self.dataset_path.exists() else self._resolve_dataset_path(None)
    candidates = [
        root / "Omni-Math.jsonl",                              # ← ADD THIS FIRST
        root / "Omni-MATH" / "Omni-Math.jsonl",
        root / "Omni-MATH" / "Omni-MATH.jsonl",
        Path("/K3D/K3D_llama_cpp/datasets/Omni-Math.jsonl"),  # ← ADD THIS SECOND
        Path("/K3D/K3D_llama_cpp/datasets/Omni-MATH/Omni-Math.jsonl"),
        Path("../K3D_llama_cpp/datasets/Omni-MATH/Omni-Math.jsonl"),
    ]
    # rest unchanged
```

No other changes to `math_competitions.py`.

---

## Part 2 — Verify AMC-AIME Path Resolution

The existing candidates already include `root / "AMC-AIME"`. When called with
`math_dataset_path=/K3D/K3D_llama_cpp/datasets`, this resolves to:

```
/K3D/K3D_llama_cpp/datasets/AMC-AIME/
```

That directory exists and contains `aime_2024.jsonl` and `aimo_test.jsonl`.
The `*.jsonl` glob in `_load_amc_aime_dataset` will find them.

**No change needed** — just confirm both files are picked up in the run report.

---

## Part 3 — Run Command

```bash
bash scripts/k3d_env.sh run -e k3d-cranium \
  python scripts/run_headless_tablet_benchmarks.py \
  --storage-root /K3D/Knowledge3D.local \
  --math-dataset-path /K3D/K3D_llama_cpp/datasets \
  --mmlu-dataset-path /K3D/K3D_llama_cpp/datasets/MMLU/data \
  --gsm8k-dataset-path /K3D/K3D_llama_cpp/datasets/GSM8K/grade_school_math/data/test.jsonl \
  --lhe-dataset-path /K3D/K3D_llama_cpp/datasets/last_humanity_exam \
  --imo-dataset-path /K3D/Knowledge3D.local/datasets/imo_bench \
  --arc2-count 0 \
  --mmlu-count 20 \
  --gsm8k-count 20 \
  --lhe-count 20 \
  --amc-aime-count 20 \
  --omni-math-count 20 \
  --imo-count 20 \
  --math-count 0 \
  --output /tmp/all_question_benchmarks_r1.json
```

This boots ONE Knowledgeverse. All six suites run sequentially through it. No mid-session
galaxy unload. No redundant TRM restarts. Shutdown happens once at the end with sleep-time
consolidation.

---

## Part 4 — What NOT to Do

- Do NOT add Python reasoning shortcuts (eval, int(), float(), regex match-and-return) in any
  benchmark's answer path to inflate scores. The answer must come from `execute_task`.
- Do NOT touch PID 608109. The encyclopedia ingest is independent. Do not read, kill, pause,
  or reconfigure it.
- Do NOT run ARC-2 or ARC-3 in this pass. ARC is not question-based.
- Do NOT set `--arc2-count` > 0.
- Do NOT use the ARC key (`/K3D/Knowledge3D.local/secrets/arc_agi_3_api_key.txt`) — it is
  scoped to ARC-3 (`three.arcprize.org`) only.
- Do NOT create separate Knowledgeverse instances per benchmark. One boot, all suites.
- Do NOT skip benchmarks silently — if a dataset path fails, raise loudly.

---

## Part 5 — Per-Suite Evidence Fields in Summary JSON

Each suite's summary entry must include these fields in the output JSON. They are needed to
confirm the sovereign path is being used (same format as the verified MMLU/GSM8K run):

```json
{
  "suite": "lhe",
  "total": 20,
  "correct": N,
  "accuracy": X.X,
  "route_family_distribution": {"QUESTION": 20},
  "trm_dispatch_task_type_distribution": {"QUESTION_TASK": 20},
  "gpu_result_packets": "20 / 20",
  "avg_elapsed_ms": 188.9,
  "answer_format_distribution": {"option_text_exact": N, "open_ended": N}
}
```

These fields confirm:
1. No task collapsed to GENERAL
2. All tasks hit the GPU path (gpu_execution=true)
3. Average latency is in the expected range (100-300ms per task)
4. Answer format breakdown (multiple choice vs open-ended for LHE/IMO)

If any suite shows `route_family_distribution = {"GENERAL": 20}`, that is a routing failure —
flag it explicitly in the report.

---

## Part 6 — Existing Infrastructure (No Changes Needed)

The orchestrator `scripts/run_headless_tablet_benchmarks.py` already:
- Imports all benchmark classes (MMLU, GSM8K, LHE, IMO, UnifiedMathBenchmark)
- Builds suites with `ThreadPoolExecutor` preloading
- Runs them sequentially through one `kv` instance
- Writes per-suite `.jsonl` logs and partial summaries after each suite completes
- Calls `kv.shutdown()` once at the end with sleep-time consolidation
- Handles `--amc-aime-count` and `--omni-math-count` flags to drive `UnifiedMathBenchmark`
  with `source_filter=["amc_aime"]` and `source_filter=["omni_math"]` respectively

The ONLY code change required is the Omni-MATH path fix in Part 1.

---

## Part 7 — Tests

Before running the full benchmark suite, run the path-fix smoke test:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium env "PYTHONPATH=$(pwd)" \
  python -c "
from benchmarks.math_competitions import UnifiedMathBenchmark
b = UnifiedMathBenchmark(
    dataset_path='/K3D/K3D_llama_cpp/datasets',
    max_problems=3,
    source_filter=['omni_math'],
    dataset_mode='present',
)
print('omni_math loaded:', len(b.problems), 'problems')
assert len(b.problems) > 0, 'omni_math path fix failed'

b2 = UnifiedMathBenchmark(
    dataset_path='/K3D/K3D_llama_cpp/datasets',
    max_problems=3,
    source_filter=['amc_aime'],
    dataset_mode='present',
)
print('amc_aime loaded:', len(b2.problems), 'problems')
assert len(b2.problems) > 0, 'amc_aime path failed'
print('PASS')
"
```

This must print `PASS` before the full run.

---

## Part 8 — Report Back

Write `TEMP/CODEX_TO_CLAUDE_ALL_QUESTION_BENCHMARKS_REPORT_2026-04-09.md` with:

1. Omni-MATH path fix confirmed (file + line changed)
2. Per-suite results table:

   | Suite | Tasks | Correct | Accuracy | Route Family | GPU Packets | Avg ms |
   |-------|-------|---------|----------|--------------|-------------|--------|
   | mmlu | 20 | K | X% | MMLU=20 | 20/20 | Nms |
   | gsm8k | 20 | K | X% | MATH=20 | 20/20 | Nms |
   | lhe | 20 | K | X% | QUESTION=20 | 20/20 | Nms |
   | omni_math | 20 | K | X% | MATH=20 | 20/20 | Nms |
   | amc_aime | 20 | K | X% | MATH=20 | 20/20 | Nms |
   | imo | 20 | K | X% | MATH=20 | 20/20 | Nms |

3. Any suites that returned GENERAL routing — flag as bug
4. Total elapsed wall-clock time for the full multi-suite run
5. Knowledgeverse boot count (must be 1)
6. Output artifact path: `/tmp/all_question_benchmarks_r1.json`
7. Dataset files actually loaded (confirm Omni-Math.jsonl and AMC-AIME/*.jsonl were found)
8. PID 608109 still running (yes/no)
