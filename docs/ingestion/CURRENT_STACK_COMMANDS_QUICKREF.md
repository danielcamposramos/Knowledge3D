# Current Stack Commands Quick Reference

**Date**: February 20, 2026  
**Env**:

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
export PYTHON_BIN="/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python"
export K3D_ROOT="../Knowledge3D.local"
```

---

## Fundamental Construction

```bash
bash scripts/fundamental_construct_knowledge.sh ../Knowledge3D.local ../Knowledge3D.local/datasets/external_payloads
```

---

## PDF Ingestion (Resumable)

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/fundamental_ingest_pdfs.py \
  --pdf-dir "/mnt/arquivos/0 ChatGPTs/DataBase" \
  --pattern "**/*.pdf" \
  --limit-pdfs 2000 \
  --classifier-model "deepseek-r1:14b" \
  --augmenter-model "qwen2.5:14b" \
  --cache-dir "$K3D_ROOT/pdf_cache" \
  --payload-output "$K3D_ROOT/fundamental_augmentation/full_pdf_payloads_overnight.jsonl" \
  --report-output "$K3D_ROOT/fundamental_augmentation/full_pdf_report_overnight.json"
```

---

## Overnight Run (tmux)

```bash
tmux new -s k3d_pdf_ingestion
bash scripts/run_overnight_pdf_ingestion.sh
```

```bash
tmux attach -t k3d_pdf_ingestion
tail -f /tmp/k3d_overnight_pdf_ingestion.log
```

---

## Ingest Payload into Knowledgeverse

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/fundamental_ingest_payloads.py \
  --payload "$K3D_ROOT/fundamental_augmentation/full_pdf_payloads_overnight.jsonl" \
  --storage-root "$K3D_ROOT" \
  --report "$K3D_ROOT/results/overnight_pdf_ingestion_report.json"
```

---

## Start Daemon (TCP)

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/k3d_daemon.py \
  --mode tcp \
  --host 127.0.0.1 \
  --port 54326 \
  --storage-root "$K3D_ROOT"
```

---

## Benchmark Senders

```bash
PYTHONPATH=. "$PYTHON_BIN" benchmarks/math_sender.py --host 127.0.0.1 --port 54326 --max-questions 400
PYTHONPATH=. "$PYTHON_BIN" benchmarks/arc_sender.py --host 127.0.0.1 --port 54326 --max-tasks 100
PYTHONPATH=. "$PYTHON_BIN" benchmarks/lhe_sender.py --host 127.0.0.1 --port 54326 --max-questions 100
PYTHONPATH=. "$PYTHON_BIN" benchmarks/mmlu_sender.py --host 127.0.0.1 --port 54326 --max-questions 100
```

---

## Progress and Safety

```bash
ps -eo pid,ppid,etime,cmd | rg "scripts/fundamental_ingest_pdfs.py --pdf-dir" | rg -v rg
wc -l "$K3D_ROOT/fundamental_augmentation/full_pdf_payloads_overnight_skipped_sources.jsonl"
```

```bash
tmux attach -t k3d_pdf_ingestion
# Ctrl+C to stop gracefully
```

Important: `scripts/run_overnight_pdf_ingestion.sh` has no `--help`; passing `--help` still launches ingestion.

