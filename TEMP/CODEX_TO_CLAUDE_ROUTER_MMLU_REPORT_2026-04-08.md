# CODEX to Claude Router Cartographer and MMLU Report

Date: 2026-04-09 00:26:39 -0300

## Scope Completed

Implemented `TEMP/CODEX_ROUTER_CARTOGRAPHER_AND_MMLU_FIX_SPEC_2026-04-08.md`.

Main surfaces changed:

- `knowledge3d/knowledgeverse/knowledgeverse.py`
- `knowledge3d/cranium/router_cartographer_bootstrap.py`
- `knowledge3d/ingestion/__init__.py`
- `knowledge3d/cranium/sas_grammar_bootstrap.py`
- `knowledge3d/knowledgeverse/sovereign_hot_path.py`
- `benchmarks/mmlu.py`
- `benchmarks/gsm8k.py`
- `tests/test_choice_payload_detection.py`
- `tests/test_mmlu_routing_fix.py`
- `tests/test_router_cartographer_boot.py`

## What Was Fixed

### Fix A

`_infer_query_mode()` now trusts typed ingress first. If the task already arrives as:

- `QUESTION_TASK`
- `MATH_TASK`
- `ARC_TASK`
- `LHE_TASK`
- `MMLU_TASK`
- `GSM8K_TASK`

it returns that declared mode immediately instead of reclassifying it heuristically.

### Fix B

`_looks_like_choice_payload()` now recognizes these list fields:

- `options`
- `choices`
- `answers`
- `candidates`
- `alternatives`

### Fix C

`QUESTION` surface with no choices now resolves to `LHE_TASK`, never `GENERAL_TASK`.

## Router Cartographer

Added boot-time routing stars in `knowledge3d/cranium/router_cartographer_bootstrap.py`:

- `routing:task_type:math`
- `routing:task_type:question`
- `routing:task_type:spatial`

These are now ingested through the canonical bootstrap path in `knowledge3d/ingestion/__init__.py` and loaded by `Knowledgeverse._ensure_sas_bootstrap_loaded()`.

Verified:

- router cartographer stars seed into Grammar successfully
- `router_star_count == 3`

## SAS Grammar Expansion

Added 11 algebraic rules beyond the prior 7, for a total of 18 SAS Grammar rules:

- `division_inverse`
- `subtraction_as_addition`
- `distributive_mul`
- `distributive_div`
- `fraction_simplify`
- `fraction_add_same_denom`
- `power_product`
- `power_quotient`
- `power_of_power`
- `double_negation`
- `square_sqrt`

## Important Runtime Correction

The deeper benchmark bug was not only routing. The sovereign hot path was materializing answers inside the nested runtime packet but not promoting them back to the top-level result object. That made benchmark adapters read:

- empty `response`
- `gpu_execution = false`
- empty `runtime`

even when the sovereign GPU dispatch had already produced a real answer.

I fixed `knowledge3d/knowledgeverse/sovereign_hot_path.py` so the top-level result now promotes:

- `answer`
- `response`
- `result`
- `predicted_answer`
- `gpu_execution = true`
- `runtime = knowledgeverse_gpu_query`
- `solver = knowledgeverse_gpu_query`
- `program_id = gpu_task_dispatch_sovereign`

I also specialize the reported question subtype at the sovereign packet layer:

- `QUESTION + choices -> MMLU`
- `QUESTION without choices -> LHE`

The VRAM task buffer still uses `QUESTION` as the compact low-level family. I did not change that packed enum layout in this pass.

## Benchmark Contract Updates

`benchmarks/mmlu.py` now submits a typed question payload directly, not an untyped legacy question:

- `type = QUESTION_TASK`
- `surface_kind = QUESTION`
- `question`
- `options`
- `choices`
- `benchmark = mmlu`
- `dataset = mmlu`

`benchmarks/gsm8k.py` now has the requested CLI entrypoint.

## Verification

### Focused structural slice

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium env "PYTHONPATH=$(pwd)" \
  pytest -q \
  tests/test_choice_payload_detection.py \
  tests/test_mmlu_routing_fix.py \
  tests/test_router_cartographer_boot.py
```

Result:

- `9 passed in 2.80s`

### GPU-backed benchmark regression slice

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium env "PYTHONPATH=$(pwd)" \
  pytest -q tests/test_gsm8k_mmlu_benchmarks.py
```

Result:

- `3 passed in 54.53s`

## Live Benchmark Results

### MMLU

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/mmlu.py --max-tasks 20 --summary-output /tmp/mmlu_r2_summary.json
```

Result:

- `9 / 20`
- `45.0%`
- route family distribution: `MMLU = 20`
- TRM dispatch task type distribution: `MMLU = 20`
- GPU result packets: `20 / 20`

This is no longer the old failure mode. The path is now sovereign and typed end to end.

### GSM8K

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium \
  python benchmarks/gsm8k.py --max-tasks 20 --summary-output /tmp/gsm8k_r1_summary.json
```

Result:

- `2 / 20`
- `10.0%`
- route family distribution: `MATH = 20`
- TRM dispatch task type distribution: `MATH = 20`
- GPU result packets: `20 / 20`

This matches the direction the spec wanted: GSM8K is now the primary math readout and it is non-zero on the sovereign path.

## Artifact State

I did not touch the encyclopedia lane during this pass.

Current observed state:

- prior protected PID `400282` is gone
- `/K3D/Knowledge3D.local/results/base_knowledge_ingest/01_encyclopedias/stages/manifest.json` lists 6 PDFs, all `staged_complete`
- final artifacts still missing:
  - `/K3D/Knowledge3D.local/results/base_knowledge_ingest/01_encyclopedias/payloads/payload.jsonl`
  - `/K3D/Knowledge3D.local/results/base_knowledge_ingest/01_encyclopedias/summaries/ingest_report.json`

## Honest Remaining Gap

The low-level resident buffer still encodes typed question traffic as compact `QUESTION` for the packed VRAM task family. The current fix specializes that traffic correctly at routing/profile/packet/reporting level, which is enough to make the benchmark path correct and measurable. I did not alter the underlying packed task-family enum in this pass.
