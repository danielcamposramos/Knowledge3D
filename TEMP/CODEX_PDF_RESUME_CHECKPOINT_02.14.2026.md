# Codex: PDF Ingestion Resume/Checkpoint Hardening (2026-02-14)

## Implemented

### 1) Null-byte / control-char sanitization
- `knowledge3d/ingestion/ollama_manager.py`
  - Added `_sanitize_prompt()`
  - `query()` now passes sanitized prompt to `subprocess.run(["ollama","run", ...])`
- `knowledge3d/ingestion/pdf_classifier.py`
  - Added `_sanitize_text()`
  - `classify_page()` now sanitizes page text and context before prompt assembly

### 2) Resumable staged PDF ingestion
- `scripts/fundamental_ingest_pdfs.py`
  - Added per-page atomic staging (`.stage` dir near payload)
  - Resume strategy: reruns reprocess the last completed page (overwrite partial attempt)
  - Added stage manifest and deterministic page files (`page_00001.json` ...)
  - Added periodic payload rebuild checkpoints (`--payload-checkpoint-interval-pdfs`, default `25`)
  - Final payload is rebuilt from staged pages (crash-safe)
  - New CLI options:
    - `--stage-dir`
    - `--disable-resume-last-page`
    - `--payload-checkpoint-interval-pdfs`

### 3) Launcher robustness fixes
- `scripts/run_overnight_pdf_ingestion.sh`
  - `set -euo pipefail`
  - Correct exit-code handling for piped command (`PIPESTATUS`)
  - Stable output names by default (`full_pdf_payloads_overnight.jsonl` / `full_pdf_report_overnight.json`) to preserve resume path across reruns
  - Fresh log file each run (`: > /tmp/k3d_overnight_pdf_ingestion.log`)
- `scripts/fundamental_full_pdf_ingestion_overnight.sh`
  - `set -euo pipefail`
  - Correct `PIPESTATUS` usage in batch runner

## Tests
- Added: `tests/test_ollama_manager_sanitization.py`
- Added: `tests/test_fundamental_ingest_pdfs_resume.py`
- Updated: `tests/test_pdf_classifier.py`

Executed:
- `pytest -q tests/test_ollama_manager_sanitization.py tests/test_pdf_classifier.py tests/test_pdf_augmenter.py tests/test_fundamental_ingest_pdfs_resume.py`
- Result: `6 passed`

## Runtime state
- Restarted overnight ingestion in tmux session: `k3d_pdf_ingestion`
- Command now uses stable payload/report names for resumable reruns.
