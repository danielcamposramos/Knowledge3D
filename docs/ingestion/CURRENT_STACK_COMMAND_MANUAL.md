# Current Stack Command Manual

**Version**: 1.0  
**Date**: February 20, 2026  
**Scope**: Canonical operational commands for fundamental augmentation, PDF ingestion, payload ingestion, daemon runtime, and benchmark senders.

---

## 1. Canonical Command Set

Use these scripts as the current source of truth:

- `scripts/fundamental_construct_knowledge.sh`
- `scripts/fundamental_augment_benchmarks.py`
- `scripts/fundamental_ingest_pdfs.py`
- `scripts/fundamental_ingest_payloads.py`
- `scripts/run_overnight_pdf_ingestion.sh`
- `scripts/k3d_daemon.py`
- `benchmarks/math_sender.py`
- `benchmarks/arc_sender.py`
- `benchmarks/lhe_sender.py`
- `benchmarks/mmlu_sender.py`

Avoid using wrappers that currently do not match `fundamental_ingest_pdfs.py` CLI (`--pdf-list`, `--skip-first`).

---

## 2. Environment Baseline

From repo root:

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
export PYTHON_BIN="/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python"
export K3D_ROOT="../Knowledge3D.local"
```

All Python commands below assume:

```bash
PYTHONPATH=. "$PYTHON_BIN" ...
```

---

## 3. Fundamental Benchmark Construction

One-command wrapper (augment then ingest):

```bash
bash scripts/fundamental_construct_knowledge.sh ../Knowledge3D.local ../Knowledge3D.local/datasets/external_payloads
```

Optional tuning:

```bash
OLLAMA_MODEL="qwen2.5:14b" \
OLLAMA_STRIDE=50 \
MAX_OLLAMA_CALLS=200 \
bash scripts/fundamental_construct_knowledge.sh
```

Outputs:

- Payload: `../Knowledge3D.local/datasets/external_payloads/benchmark_aug_*/benchmark_augmentation_payload.jsonl`
- Augment report: `.../benchmark_augmentation_report.json`
- Ingest report: `.../benchmark_augmentation_ingest_report.json`

---

## 4. Intelligent PDF Ingestion (Resumable)

Single PDF:

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/fundamental_ingest_pdfs.py \
  --pdf "/path/to/file.pdf" \
  --classifier-model "deepseek-r1:14b" \
  --augmenter-model "qwen2.5:14b" \
  --cache-dir "$K3D_ROOT/pdf_cache" \
  --payload-output "$K3D_ROOT/fundamental_augmentation/pdf_payload_single.jsonl" \
  --report-output "$K3D_ROOT/fundamental_augmentation/pdf_report_single.json"
```

Directory batch:

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

Resume behavior:

- Stage directory defaults to `.<payload_stem>_stage` next to payload.
- Rerun reprocesses the last staged page by default, then continues.
- This is intentional for power-loss safety.

---

## 5. Overnight Full Run (tmux)

Launch:

```bash
tmux new -s k3d_pdf_ingestion
bash scripts/run_overnight_pdf_ingestion.sh
```

Detach and reattach:

```bash
# detach: Ctrl+b, then d
tmux attach -t k3d_pdf_ingestion
```

Live log:

```bash
tail -f /tmp/k3d_overnight_pdf_ingestion.log
```

Important:

- `scripts/run_overnight_pdf_ingestion.sh` has no `--help` mode.
- Passing `--help` will still start ingestion.

---

## 6. Progress and Health Checks

Check active ingestion process:

```bash
ps -eo pid,ppid,etime,cmd | rg "scripts/fundamental_ingest_pdfs.py --pdf-dir" | rg -v rg
```

Check manifest-based resume/progress:

```bash
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
p = Path("../Knowledge3D.local/fundamental_augmentation/.full_pdf_payloads_overnight_stage/manifest.json")
if not p.exists():
    print("manifest_missing")
    raise SystemExit(0)
m = json.loads(p.read_text(encoding="utf-8"))
pdfs = m.get("pdfs", {})
total = len(pdfs)
done = sum(1 for v in pdfs.values() if isinstance(v, dict) and v.get("resume_from_page", 1) > 1)
print({"pdfs_tracked": total, "pdfs_with_progress": done})
PY
```

Check skipped/corrupt/encrypted sources:

```bash
wc -l ../Knowledge3D.local/fundamental_augmentation/full_pdf_payloads_overnight_skipped_sources.jsonl
tail -n 20 ../Knowledge3D.local/fundamental_augmentation/full_pdf_payloads_overnight_skipped_sources.jsonl
```

GPU monitor:

```bash
nvidia-smi
```

---

## 7. Payload Ingestion into Knowledgeverse

Ingest one or more payload files:

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/fundamental_ingest_payloads.py \
  --payload \
    "$K3D_ROOT/fundamental_augmentation/full_pdf_payloads_overnight.jsonl" \
    $K3D_ROOT/datasets/external_payloads/benchmark_aug_*/benchmark_augmentation_payload.jsonl \
  --storage-root "$K3D_ROOT" \
  --report "$K3D_ROOT/results/fundamental_ingestion_report.json"
```

Disable symlink compression only for diagnostics:

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/fundamental_ingest_payloads.py \
  --payload "$K3D_ROOT/fundamental_augmentation/full_pdf_payloads_overnight.jsonl" \
  --storage-root "$K3D_ROOT" \
  --report "$K3D_ROOT/results/fundamental_ingestion_report_debug.json" \
  --disable-symlink-compression
```

---

## 8. Daemon Runtime and Benchmark Senders

Start daemon (TCP mode):

```bash
PYTHONPATH=. "$PYTHON_BIN" scripts/k3d_daemon.py \
  --mode tcp \
  --host 127.0.0.1 \
  --port 54326 \
  --storage-root "$K3D_ROOT"
```

Run benchmark senders:

```bash
PYTHONPATH=. "$PYTHON_BIN" benchmarks/math_sender.py --host 127.0.0.1 --port 54326 --max-questions 400
PYTHONPATH=. "$PYTHON_BIN" benchmarks/arc_sender.py --host 127.0.0.1 --port 54326 --max-tasks 100
PYTHONPATH=. "$PYTHON_BIN" benchmarks/lhe_sender.py --host 127.0.0.1 --port 54326 --max-questions 100
PYTHONPATH=. "$PYTHON_BIN" benchmarks/mmlu_sender.py --host 127.0.0.1 --port 54326 --max-questions 100
```

Capture sender output to file:

```bash
PYTHONPATH=. "$PYTHON_BIN" benchmarks/math_sender.py --host 127.0.0.1 --port 54326 --max-questions 400 \
  | tee "$K3D_ROOT/results/math_sender_$(date +%Y%m%d_%H%M%S).json"
```

---

## 9. Graceful Stop and Resume

Stop overnight run cleanly:

```bash
tmux attach -t k3d_pdf_ingestion
# inside tmux, press Ctrl+C once
```

Restart later (resume from stage):

```bash
tmux new -s k3d_pdf_ingestion
bash scripts/run_overnight_pdf_ingestion.sh
```

The ingestion resumes from staged pages and rewrites the last page checkpoint to recover from partial writes.

---

## 10. Troubleshooting Quick Map

- `ModuleNotFoundError: knowledge3d`: missing `PYTHONPATH=.`
- Empty output payload with running GPU: check skip log JSONL and report JSON
- Slow or stalled progress: verify Ollama model availability and timeout (`--ollama-timeout`)
- Excess skipped PDFs: inspect skip log phases (`extract_pages`, classifier/augmenter failures)
