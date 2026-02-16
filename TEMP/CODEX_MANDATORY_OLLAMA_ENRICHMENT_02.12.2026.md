# Week 22 - Mandatory Ollama Enrichment Enforcement (Sovereign Workflow)

Date: 2026-02-12

## Goal
Align ingestion/sender workflow with the architecture directive:
- Ollama enrichment is central to benchmark-to-galaxy ingestion (not casual opt-in).
- Benchmark senders must run enriched mode by default with no `empty-mind` toggle.
- Keep sovereign hot path intact (specialist -> RPN -> PTX).

## Changes Applied

### 1) Augmentation pipeline: Ollama mandatory by default
File: `scripts/augment_benchmarks_to_galaxy.py`

- Removed opt-in flag:
  - `--enable-ollama` (deleted)
- Added emergency-only bypass:
  - `--skip-ollama-enrichment`
- Behavior:
  - Ollama is enabled by default.
  - If skipped, script prints explicit architectural warning.
- Internal state/reporting updated:
  - `report["ollama"]["enabled"]` now reflects `ollama_enabled` derived from skip flag.

### 2) Train script: Ollama augmentation defaults ON
File: `scripts/train_deterministic_foundation.py`

- Default changed:
  - `enable_ollama_augmentation: bool = True`
- CLI behavior:
  - Added explicit default true via parser defaults.
  - Added emergency-only disable flag:
    - `--disable-ollama-augmentation`
  - `--enable-ollama-augmentation` retained for explicitness/backward compatibility.
- Emits warning when disabled by override.

### 3) Ingestion wrapper script updated
File: `scripts/run_benchmark_augmentation_ingestion.sh`

- Removed `ENABLE_OLLAMA` toggle logic.
- Always forwards Ollama args (`--ollama-model`, `--ollama-stride`, `--max-ollama-calls`).
- Added emergency env override:
  - `SKIP_OLLAMA_ENRICHMENT=1` -> passes `--skip-ollama-enrichment` and prints warning.

### 4) Sender scripts: remove empty-mind mode
Files:
- `benchmarks/arc_sender.py`
- `benchmarks/math_sender.py`
- `benchmarks/lhe_sender.py`
- `benchmarks/mmlu_sender.py`

Changes:
- Removed CLI arg:
  - `--empty-mind`
- Set:
  - `use_enriched = True` (fixed)

This prevents accidental non-enriched sender runs while preserving sovereignty assertions.

## Validation

1. Compile checks
```bash
python3 -m py_compile \
  scripts/augment_benchmarks_to_galaxy.py \
  scripts/train_deterministic_foundation.py \
  benchmarks/arc_sender.py \
  benchmarks/math_sender.py \
  benchmarks/lhe_sender.py \
  benchmarks/mmlu_sender.py
```
Result: PASS

2. CLI checks (`PYTHONPATH=.`)
- `benchmarks/math_sender.py --help` PASS
- `benchmarks/mmlu_sender.py --help` PASS
- `scripts/augment_benchmarks_to_galaxy.py --help` PASS
- `scripts/train_deterministic_foundation.py --help` PASS

3. Pattern grep
- No remaining `empty-mind` toggles in sender targets.
- No remaining `--enable-ollama` in augmentation pipeline.

## Notes
- This patch enforces default enriched data flow and removes easy toggles that caused repeated drift.
- Emergency diagnostic bypass still exists but is explicit and noisy.
