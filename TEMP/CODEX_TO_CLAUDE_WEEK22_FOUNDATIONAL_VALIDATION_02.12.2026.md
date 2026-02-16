# Codex -> Claude: Week22 Foundational Construction + Bounded Validation (02.12.2026)

## Scope Executed
1. Bounded intelligent PDF ingestion (3 PDFs, max 20 pages each)
2. Single-world payload ingestion (benchmark + PDF payloads)
3. Bounded daemon validation (20 math tasks, sovereign mode)
4. Critical runtime hardening:
   - daemon auto-configures CUDA include env for NVRTC (`cuda_fp16.h`)
   - math template selection hardened to trusted algebra templates

---

## 1) PDF Ingestion Result (bounded)

Command executed:

```bash
PYTHONPATH=. conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  python scripts/fundamental_ingest_pdfs.py \
  --pdf-dir "/mnt/arquivos/0 ChatGPTs/DataBase/OCR_TRAINING_SET" \
  --pattern "*.pdf" \
  --limit-pdfs 3 \
  --max-pages-per-pdf 20 \
  --classifier-model "deepseek-r1:14b" \
  --augmenter-model "qwen2.5:14b" \
  --ollama-timeout 180.0 \
  --cache-dir ../Knowledge3D.local/pdf_cache \
  --payload-output ../Knowledge3D.local/fundamental_augmentation/pdf_payloads.jsonl \
  --report-output ../Knowledge3D.local/fundamental_augmentation/pdf_ingestion_report.json
```

Observed:
- `pdfs=3`
- `rows=58`
- Galaxy distribution: `Grammar=31`, `Math=9`, `Reality=13`, `Word=5`

---

## 2) Payload Ingestion Result (single world)

Command executed:

```bash
PYTHONPATH=. conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  python scripts/fundamental_ingest_payloads.py \
  --payload ../Knowledge3D.local/fundamental_augmentation/full_benchmark_payloads.jsonl \
            ../Knowledge3D.local/fundamental_augmentation/pdf_payloads.jsonl \
  --storage-root ../Knowledge3D.local \
  --report ../Knowledge3D.local/fundamental_augmentation/ingestion_report.json
```

Observed:
- Added: `56`
- Skipped (already-present): `5844`
- `symlink_compression_enabled: true`
- Incremental galaxy deltas:
  - Grammar: `+29`
  - Math: `+9`
  - Reality: `+13`
  - Word: `+5`

---

## 3) Bounded Daemon Math Validation (20 tasks)

Command pattern:
- Start daemon (TCP, strict sovereign query)
- Run `benchmarks/math_sender.py --max-questions 20`
- Query daemon status + shutdown

Observed:
- Sender summary: `ok=7`, `failed=13` (`35%` solve-success rate by sender criteria)
- Daemon status during run:
  - `require_ptx_query: true`
  - `fallback_triggered: false`
  - `gpu_calls_total: 7`

Meaning:
- 7 solved tasks reached PTX execution (`gpu_calls_this_command > 0` on successful routes)
- 13 tasks failed at specialist capability layer (not fallback)

---

## Critical Fixes Applied During Validation

### A) NVRTC include-path blocker removed

Issue seen:
- `cannot open source file "cuda_fp16.h"` during sovereign query compile

Fix:
- `knowledge3d/daemon/main.py`
  - Added daemon startup CUDA include auto-configuration:
    - populates `CPATH` / `CPLUS_INCLUDE_PATH`
    - sets `CUDA_PATH` when missing
  - exposed in daemon `STATUS` as `cuda_env`

Result:
- Query compile blocker removed.

### B) Math template selection hardened

Issue seen:
- Invalid template selected from augmented rows (e.g., `A B C TRIANGLE_AREA`)
- caused `Unknown token: A`

Fix:
- `knowledge3d/knowledgeverse/specialists/math_specialist.py`
  - `_select_template` now prioritizes trusted bootstrap algebra templates
  - supports explicit placeholder templates only (`{a}`, `{b}`, `{c}`)
  - bootstrap fallback by `pattern_type`
  - `_select_pattern` similarly trusts equation-pattern bootstrap entries first

Result:
- canonical linear example routes to:
  - pattern: `grammar_linear_equation_ax_plus_b_eq_c_v1`
  - template: `math_template_linear_equation_solve_v1`
  - RPN: `11 3 - 2 /`
  - result: `4.0`
  - GPU call count increments.

---

## Current Position

- Fundamental construction pipeline is operational with real outputs.
- Sovereign constraints preserved (`require_ptx_query=true`, no CPU fallback flagging).
- Baseline moved from `0/20` to `7/20` on bounded math sender after runtime/template hardening.

---

## Suggested Next Steps

1. Expand trusted high-yield math templates (still Galaxy-first) for common AMC forms:
   - linear variants (`ax-b=c`, `b+ax=c`)
   - ratio/proportion
   - simple exponent/root forms
2. Add pattern-type-aware template filtering in `manager.query` metadata scoring (prefer `algebra_template` for math specialist).
3. Re-run bounded `--max-questions 50` before any full sweep.
