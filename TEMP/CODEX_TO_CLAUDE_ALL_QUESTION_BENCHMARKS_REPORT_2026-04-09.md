# CODEX TO CLAUDE: All Question Benchmarks Run Report

**Date:** 2026-04-09  
**Completed at:** 2026-04-09 02:21:19 -0300  
**Spec executed:** `TEMP/CODEX_ALL_QUESTION_BENCHMARKS_RUN_SPEC_2026-04-09.md`

## 1. Required Omni-MATH path fix

Implemented in [benchmarks/math_competitions.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/benchmarks/math_competitions.py#L419):

- added `root / "Omni-Math.jsonl"` at [benchmarks/math_competitions.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/benchmarks/math_competitions.py#L422)
- added `Path("/K3D/K3D_llama_cpp/datasets/Omni-Math.jsonl")` at [benchmarks/math_competitions.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/benchmarks/math_competitions.py#L425)

Smoke test status:
- `omni_math loaded: 3 problems`
- `amc_aime loaded: 3 problems`
- `PASS`

Important environment truth:
- the root-level file `/K3D/K3D_llama_cpp/datasets/Omni-Math.jsonl` does **not** exist on this machine
- the live run therefore loaded Omni-MATH from `/K3D/K3D_llama_cpp/datasets/Omni-MATH/Omni-Math.jsonl`

## 2. Additional blockers fixed to complete the run

The spec’s single code-change assumption was not sufficient on this machine. Two real blockers surfaced during the one-boot execution path and were fixed honestly:

1. Sovereign route metadata blocker
- proceduralized resident star `rule_math_core_tier_hierarchy` was authored as a `router` without `route_policy.requires_executor`
- fixed in [resident_route_metadata.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/knowledgeverse/resident_route_metadata.py#L562)

2. IMO dataset root handling blocker
- the run command passes `--imo-dataset-path /K3D/Knowledge3D.local/datasets/imo_bench`
- [imo_bench.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/benchmarks/imo_bench.py#L63) previously treated that as a file path and never descended into `answerbench_v2.csv`
- fixed directory-root resolution in [imo_bench.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/benchmarks/imo_bench.py#L63)

I also fixed one summary serialization inconsistency so the final artifact reflects actual `correct / total` values when a benchmark emits `overall_accuracy` instead of `accuracy`:
- [run_headless_tablet_benchmarks.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/scripts/run_headless_tablet_benchmarks.py#L305)

## 3. Full one-boot run result

Command executed:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium python scripts/run_headless_tablet_benchmarks.py \
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

Knowledgeverse boot count:
- `1`

Wall-clock time:
- `125.79s`

Output artifact:
- `/tmp/all_question_benchmarks_r1.json`

Execution logs:
- `/K3D/Knowledge3D.local/logs/headless_tablet_20260409_021522/summary.execution.json`
- `/K3D/Knowledge3D.local/logs/headless_tablet_20260409_021522/full_results.execution.json`

## 4. Per-suite results

| Suite | Tasks | Correct | Accuracy | Route Family | GPU Packets | Avg ms |
|---|---:|---:|---:|---|---|---:|
| mmlu | 20 | 7 | 35.0% | `{'MMLU': 20}` | `20 / 20` | 301.15 |
| gsm8k | 20 | 1 | 5.0% | `{'MATH': 20}` | `20 / 20` | 1524.2 |
| lhe | 20 | 1 | 5.0% | `{'LHE': 13, 'MMLU': 7}` | `20 / 20` | 485.7 |
| amc_aime | 20 | 1 | 5.0% | `{'MATH': 20}` | `20 / 20` | 1118.15 |
| omni_math | 20 | 0 | 0.0% | `{'MATH': 20}` | `20 / 20` | 1137.35 |
| imo | 20 | 0 | 0.0% | `{'MATH': 20}` | `20 / 20` | 1710.15 |

## 5. Routing audit

`GENERAL` routing failures:
- none

Important routing note:
- LHE did **not** collapse to `GENERAL`, but it is mixed: `{'LHE': 13, 'MMLU': 7}`
- that is not the previous routing bug, but it is still a family-selection impurity worth tracking

GPU path evidence:
- every suite reported `gpu_result_packets = 20 / 20`
- no suite fell back to a CPU/non-sovereign answer path in this run

## 6. Dataset files actually loaded

MMLU:
- `/K3D/K3D_llama_cpp/datasets/MMLU/data`

GSM8K:
- `/K3D/K3D_llama_cpp/datasets/GSM8K/grade_school_math/data/test.jsonl`

LHE:
- `/K3D/K3D_llama_cpp/datasets/last_humanity_exam/last_humanity_exam.json`

AMC-AIME:
- dataset root used by the benchmark: `/K3D/K3D_llama_cpp/datasets`
- actual JSONL files present under the resolved source root:
  - `/K3D/K3D_llama_cpp/datasets/AMC-AIME/data/aime_2024.jsonl`
  - `/K3D/K3D_llama_cpp/datasets/AMC-AIME/data/aimo_test.jsonl`
  - `/K3D/K3D_llama_cpp/datasets/AMC-AIME/data/aimo_train.jsonl`

Omni-MATH:
- actual loaded source reported by the run:
  - `/K3D/K3D_llama_cpp/datasets/Omni-MATH/Omni-Math.jsonl`
- root-level spec path status on this host:
  - `/K3D/K3D_llama_cpp/datasets/Omni-Math.jsonl` does not exist

IMO:
- `/K3D/Knowledge3D.local/datasets/imo_bench/answerbench_v2.csv`

## 7. PID 608109 status

`PID 608109 still running`:
- `no`

I did not touch it. It was already gone before this run.

## 8. Outcome

The all-question orchestrator is now proven on the intended one-boot path:
- one live Knowledgeverse
- six suites
- no `GENERAL` collapse
- all suites returning GPU result packets

Honest current performance baseline from this run:
- MMLU remains the strongest of the question suites at `35.0%`
- GSM8K, LHE, and AMC-AIME are alive but still low
- Omni-MATH and IMO are routed correctly and executed on-GPU, but are still knowledge/grammar-limited at `0%`

