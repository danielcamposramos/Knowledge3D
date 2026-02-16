# Codex -> Claude: Fundamental Pipeline Rename + Architecture Alignment (02.12.2026)

## Completed Renames (no new duplicate scripts)

Renamed canonical construction scripts:

1. `scripts/augment_benchmarks_to_galaxy.py` -> `scripts/fundamental_augment_benchmarks.py`
2. `scripts/ingest_pdf_intelligent.py` -> `scripts/fundamental_ingest_pdfs.py`
3. `scripts/ingest_payloads_single_world.py` -> `scripts/fundamental_ingest_payloads.py`
4. `scripts/run_benchmark_augmentation_ingestion.sh` -> `scripts/fundamental_construct_knowledge.sh`

## Updated Active References

- `scripts/fundamental_construct_knowledge.sh`
  - Calls `fundamental_augment_benchmarks.py`
  - Calls `fundamental_ingest_payloads.py`
- `scripts/run_parallel_external_ingestion.sh`
  - Calls `fundamental_ingest_payloads.py`
  - Calls `fundamental_construct_knowledge.sh`
- `scripts/fundamental_ingest_pdfs.py`
  - Optional ingestion step now targets `scripts/fundamental_ingest_payloads.py`
- `scripts/fundamental_augment_benchmarks.py`
  - Header reference updated to `fundamental_ingest_payloads.py`

## Script Semantics Clarified (headers updated)

- `fundamental_augment_benchmarks.py`
  - Explicitly documented as foundational knowledge construction only.
  - Not for hot-path runtime inference.
- `fundamental_ingest_pdfs.py`
  - Explicitly documented as foundational PDF construction/ingestion.
  - Not for runtime daemon routing.
- `fundamental_ingest_payloads.py`
  - Explicitly documented as mid-term ingestion with form->meaning symlink compression.
  - Not for PTX hot path.

## Your Key Requirements Confirmed in Code

### 1) Benchmark augmentation uses question + answer context
- Present in `scripts/fundamental_augment_benchmarks.py`
- Math/LHE/MMLU prompts include gold answers to construct stronger procedural K3D entries.
- `supervision_answer` metadata persisted.

### 2) Ingestion swaps text to procedural symlink references
- Present in `scripts/fundamental_ingest_payloads.py`
- Automatic `word_refs`, `char_refs`, `symbol_refs` generated during ingestion.
- Compression marker set: `symlink_compression=applied_v1`.

This keeps augmentation as non-hot-path, and ingestion as mid-term normalization/compression.

## Validation Performed

- `python3 -m py_compile` for renamed fundamental scripts: PASS
- `bash -n` for shell wrappers: PASS
- `--help` checks for:
  - `scripts/fundamental_augment_benchmarks.py`
  - `scripts/fundamental_ingest_pdfs.py`
  - `scripts/fundamental_ingest_payloads.py`
  all PASS

## Notes

- Historical TEMP reports still mention legacy names for chronology.
- Active scripts now use only the `fundamental_*` naming and call graph.
