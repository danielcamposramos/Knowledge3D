# Codex -> Claude: Week 22 Intelligent Ingestion Update (02.12.2026)

## What Was Completed

### 1) Benchmark augmentation now uses question+answer supervision context
- File: `scripts/fundamental_augment_benchmarks.py`
- Implemented:
  - Math prompt now includes `Problem + Gold answer`.
  - LHE/MMLU prompt now includes `Question + Options + Gold answer`.
  - Payload metadata now carries `supervision_answer`.
- Rationale:
  - This is augmentation-path supervision to produce stronger K3D procedural knowledge entries.
  - Hot path remains sovereign; no external dependency at inference.

### 2) Ingestion phase now performs text -> symlink reference compression
- File: `scripts/fundamental_ingest_payloads.py`
- Implemented:
  - Auto-derives `word_refs`, `char_refs`, and `symbol_refs` from entry text/metadata.
  - Writes `metadata["symlink_compression"] = "applied_v1"`.
  - Defaults `metadata["symlink"]` when absent.
- Rationale:
  - Preserves procedural compositional compression (form -> meaning references) during ingestion.
  - Matches Daniel's directive: enrichment text is swapped to references in mid-term ingestion.

### 3) Intelligent PDF pipeline scaffolded with mandatory Ollama classification + augmentation
- Added: `knowledge3d/ingestion/pdf_classifier.py`
  - LLM page classification: `knowledge | non_knowledge | ambiguous`.
  - Persistent per-PDF page-decision cache by SHA256.
  - Ambiguous-page second pass with neighbor context.
- Added: `knowledge3d/ingestion/pdf_augmenter.py`
  - LLM augmentation to structured payload (`summary`, `entities`, `relationships`, `cross_modal`, `embedding_text`, `galaxy_hints`).
  - Converts augmented page into payload rows (`target galaxy + Grammar bridge`).
- Added: `scripts/fundamental_ingest_pdfs.py`
  - End-to-end pipeline:
    1. Extract pages
    2. Classify pages (cached)
    3. Augment knowledge pages
    4. Emit payload JSONL + report
    5. Optional single-world ingestion via `scripts/fundamental_ingest_payloads.py`

## Mandatory Enrichment Contract Status

- Ollama remains central in augmentation flow.
- Emergency bypasses still exist only as explicit diagnostic overrides (`--skip-ollama-enrichment`), with warnings.
- No inference hot-path dependency added.

## Suggested Next Validation Commands

### A) Quick benchmark augmentation sanity
```bash
python scripts/fundamental_augment_benchmarks.py \
  --dataset-root ../Knowledge3D.local/datasets \
  --max-arc-tasks 5 \
  --max-math-problems 10 \
  --max-lhe-questions 10 \
  --max-mmlu-questions 10 \
  --max-word-entries 200
```

### B) Intelligent PDF ingest (single PDF, with ingestion)
```bash
python scripts/fundamental_ingest_pdfs.py \
  --pdf /path/to/paper.pdf \
  --max-pages-per-pdf 20 \
  --ingest \
  --storage-root ../Knowledge3D.local
```

### C) Intelligent PDF ingest (batch, payload only)
```bash
python scripts/fundamental_ingest_pdfs.py \
  --pdf-dir /path/to/pdfs \
  --pattern "**/*.pdf" \
  --limit-pdfs 5 \
  --max-pages-per-pdf 25
```

## Architectural Notes

- This update intentionally keeps augmentation/injection outside hot path.
- PTX sovereignty for solve-time path remains unchanged.
- Mid-term ingestion now enforces symlink compression to align with "form-to-meaning" memory structure.
