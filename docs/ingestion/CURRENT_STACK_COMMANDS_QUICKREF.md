# Current Stack Commands Quick Reference

**Date**: April 6, 2026

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
export PYTHON_BIN="/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python"
export K3D_ROOT="/K3D/Knowledge3D.local"
```

## Preflight

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/analyze_pdf_types.py \
  --root "/mnt/arquivos/0 ChatGPTs/DataBase/Encyclopedias" \
  --root "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/" \
  --results-root "$K3D_ROOT/results/base_knowledge_ingest"
```

## Ordered PDF Ingest

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/fundamental_ingest_pdfs.py \
  --pdf-list "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/preflight/eligible_pdfs.txt" \
  --provider ollama \
  --model-profile quality \
  --capture-dir "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/captures" \
  --stage-dir "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/stages" \
  --payload-output "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/payloads/payload.jsonl" \
  --report-output "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/summaries/ingest_report.json"
```

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/fundamental_ingest_payloads.py \
  --payload "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/payloads/payload.jsonl" \
  --storage-root "$K3D_ROOT" \
  --report "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/summaries/payload_ingest_report.json"
```

## OCR Needed / Skips

```bash
wc -l "$K3D_ROOT/results/base_knowledge_ingest/01_encyclopedias/preflight/ocr_needed_pdfs.txt"
wc -l "$K3D_ROOT/results/base_knowledge_ingest/02_default_libraries/preflight/ocr_needed_pdfs.txt"
tail -n 20 "$K3D_ROOT/results/base_knowledge_ingest/02_default_libraries/summaries/skipped_sources.jsonl"
```

## Progress

```bash
ps -eo pid,ppid,etime,cmd | rg "scripts/fundamental_ingest_pdfs.py" | rg -v rg
find "$K3D_ROOT/results/base_knowledge_ingest/02_default_libraries/stages" -name 'page_*.json' | wc -l
```
