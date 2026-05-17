# Current Stack Command Manual

**Version**: 2.0  
**Date**: April 6, 2026  
**Scope**: Canonical preflight, ordered PDF ingestion, payload ingestion, daemon runtime, and benchmark senders.

## 1. Environment Baseline

From repo root:

```bash
cd "/K3D/GitHub/Knowledge3D"
export PYTHON_BIN="/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python"
export K3D_ROOT="/K3D/Knowledge3D.local"
```

All Python commands below assume:

```bash
PYTHONPATH=. "$PYTHON_BIN" ...
```

Use the managed env above. Do not use system Python for PDF preflight or ingestion.

## 2. Canonical Ingestion Stack

Use only the unified proceduralizer path:

- `scripts/analyze_pdf_types.py`
- `scripts/fundamental_ingest_pdfs.py`
- `scripts/fundamental_ingest_payloads.py`
- `knowledge3d/tools/knowledge_proceduralizer.py`

Legacy classifier/augmenter modules have been archived under `Old_Attempts/` and are not part of the live stack.

## 3. Ordered Base-Knowledge Restart

The current base-knowledge run order is fixed:

1. `/mnt/arquivos/0 ChatGPTs/DataBase/Encyclopedias`
2. `/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/`

The preflight is recursive and PDF-only. JSON and every other non-PDF sidecar are ignored before ingestion.

## 4. Canonical OCR / Eligibility Preflight

Run the preflight once for both roots, preserving the order above:

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/analyze_pdf_types.py \
  --root "/mnt/arquivos/0 ChatGPTs/DataBase/Encyclopedias" \
  --root "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/" \
  --results-root "$K3D_ROOT/results/base_knowledge_ingest"
```

Per-root artifacts are emitted under:

- `$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/preflight/`
- `$K3D_ROOT/results/base_knowledge_ingest/02_default_libraries/preflight/`

Each root emits:

- `all_pdf_inventory.json`
- `eligible_pdfs.txt`
- `ocr_needed_pdfs.txt`
- `extraction_errors.txt`
- `summary.json`

Eligibility rules for this wave:

- ingest: `vector`, `mixed`, `scanned_with_ocr`
- skip and list only: `scanned_no_text`
- skip and log separately: `error`

Eligible PDFs are ordered by page count descending, then absolute path ascending.

## 5. Ordered PDF Ingestion

Use the `eligible_pdfs.txt` artifact from preflight. `--pdf-list` is the preferred large-batch entrypoint because it preserves exact file order.

### Root 1: Encyclopedias

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/fundamental_ingest_pdfs.py \
  --pdf-list "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/preflight/eligible_pdfs.txt" \
  --provider ollama \
  --model-profile quality \
  --capture-dir "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/captures" \
  --stage-dir "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/stages" \
  --payload-output "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/payloads/payload.jsonl" \
  --report-output "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/summaries/ingest_report.json" \
  --skip-sources-output "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/summaries/skipped_sources.jsonl"
```

### Root 2: EchoSystems Default Libraries

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/fundamental_ingest_pdfs.py \
  --pdf-list "$K3D_ROOT/results/base_knowledge_ingest/02_default_libraries/preflight/eligible_pdfs.txt" \
  --provider ollama \
  --model-profile quality \
  --capture-dir "$K3D_ROOT/results/base_knowledge_ingest/02_default_libraries/captures" \
  --stage-dir "$K3D_ROOT/results/base_knowledge_ingest/02_default_libraries/stages" \
  --payload-output "$K3D_ROOT/results/base_knowledge_ingest/02_default_libraries/payloads/payload.jsonl" \
  --report-output "$K3D_ROOT/results/base_knowledge_ingest/02_default_libraries/summaries/ingest_report.json" \
  --skip-sources-output "$K3D_ROOT/results/base_knowledge_ingest/02_default_libraries/summaries/skipped_sources.jsonl"
```

Operational rules preserved by the canonical path:

- context clears between distinct sources
- oversized page/chunk processing preserves overlap
- per-document and per-page resume remains active
- plan-limit detection stops cleanly and writes `retry_after_utc = now + 5h01m`

## 6. Payload Ingestion into the Resident Corpus

After each root completes payload generation:

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/fundamental_ingest_payloads.py \
  --payload "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/payloads/payload.jsonl" \
  --storage-root "$K3D_ROOT" \
  --report "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/summaries/payload_ingest_report.json"
```

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/fundamental_ingest_payloads.py \
  --payload "$K3D_ROOT/results/base_knowledge_ingest/02_default_libraries/payloads/payload.jsonl" \
  --storage-root "$K3D_ROOT" \
  --report "$K3D_ROOT/results/base_knowledge_ingest/02_default_libraries/summaries/payload_ingest_report.json"
```

## 7. Progress and Safety Checks

Inspect current preflight summaries:

```bash
jq . "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/preflight/summary.json"
jq . "$K3D_ROOT/results/base_knowledge_ingest/02_default_libraries/preflight/summary.json"
```

Check ingestion progress:

```bash
ps -eo pid,ppid,etime,cmd | rg "scripts/fundamental_ingest_pdfs.py" | rg -v rg
```

Check resumable staging:

```bash
find "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/stages" -name 'page_*.json' | wc -l
find "$K3D_ROOT/results/base_knowledge_ingest/02_default_libraries/stages" -name 'page_*.json' | wc -l
```

Check OCR-needed PDFs:

```bash
wc -l "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/preflight/ocr_needed_pdfs.txt"
wc -l "$K3D_ROOT/results/base_knowledge_ingest/02_default_libraries/preflight/ocr_needed_pdfs.txt"
```

Check extraction/runtime skips:

```bash
tail -n 20 "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/summaries/skipped_sources.jsonl"
tail -n 20 "$K3D_ROOT/results/base_knowledge_ingest/02_default_libraries/summaries/skipped_sources.jsonl"
```

## 8. Post-Feed Validation

After both roots are ingested:

1. inspect representative resident rows in `galaxy_consolidated_latest.json`
2. rerun canonical probes
3. rerun the text-heavy benchmark slice only:
   - `gsm8k=10`
   - `mmlu=10`
   - `lhe=10`

`arc3_local` / `GAME_2D` knowledge work remains out of scope for this base-knowledge restart.
