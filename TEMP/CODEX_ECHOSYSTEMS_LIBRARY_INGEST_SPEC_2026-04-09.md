# Codex Direction: Ingest EchoSystems Default Libraries into Galaxy

**Date:** 2026-04-09
**Authority:** docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md §3 (Reality Galaxy), §5 (Grammar Galaxy)
**Scope:** Payload generation only (Phase 1). Galaxy resident-ingest is Phase 2, after payload.jsonl is complete.

---

## Source

```
/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/
```

330 PDFs, 1.4 GB, across 30 subject domain folders:

| Domain Folder | Count | Target Galaxy |
|---------------|------:|---------------|
| Games | 44 | Drawing + Reality |
| Humans | 43 | Reality |
| Advanced Maths | 41 | Math + Grammar |
| CopyRight | 26 | Reality (legal/social) |
| Canada | 26 | Reality (geography/culture) |
| Eloquence | 17 | Word + Grammar |
| Understanding Typos | 16 | Grammar + Word + Character |
| Self Reflection | 14 | Reality (psychology) |
| Understand Time | 13 | Reality + Math |
| Numerology | 13 | Math + Number |
| Context | 13 | Word + Grammar |
| Story Telling | 8 | Word + Grammar |
| How to Teach | 8 | Reality (pedagogy) |
| Quotes | 6 | Word |
| How to Academic Research | 6 | Reality |
| FMEAI | 6 | Reality |
| Acting | 6 | Reality |
| How to think | 5 | Reality + Grammar |
| Apollo 11 | 5 | Reality (space/engineering) |
| Carthography | 3 | Reality + Drawing |
| Engineering | 2 | Reality |
| NASA SE | 1 | Reality (systems engineering) |
| Ethics | 1 | Reality |
| AI | 1 | Reality |
| (+ others) | | |

The proceduralizer determines galaxy assignment from content. Folder names above are
reference context only — do NOT hardcode routing by folder name.

---

## Output Directory

```
/K3D/Knowledge3D.local/results/base_knowledge_ingest/02_echosystems_libraries/
  payloads/payload.jsonl      ← cumulative payload output
  summaries/ingest_report.json ← final report
  stages/                      ← per-PDF per-page checkpoints (auto-created)
  ingest.log                   ← stdout+stderr captured
```

Create the directories before starting:

```bash
mkdir -p /K3D/Knowledge3D.local/results/base_knowledge_ingest/02_echosystems_libraries/payloads
mkdir -p /K3D/Knowledge3D.local/results/base_knowledge_ingest/02_echosystems_libraries/summaries
```

---

## Phase 1 — Payload Generation (run now, background tmux)

This is a long-running job (hours). Start it in a tmux session so it survives terminal disconnects.

```bash
tmux new-session -d -s echosys_ingest \
  "bash scripts/k3d_env.sh run -e k3d-cranium \
    python scripts/fundamental_ingest_pdfs.py \
    --pdf-dir '/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries' \
    --pattern '**/*.pdf' \
    --payload-output /K3D/Knowledge3D.local/results/base_knowledge_ingest/02_echosystems_libraries/payloads/payload.jsonl \
    --report-output /K3D/Knowledge3D.local/results/base_knowledge_ingest/02_echosystems_libraries/summaries/ingest_report.json \
    --stage-dir /K3D/Knowledge3D.local/results/base_knowledge_ingest/02_echosystems_libraries/stages \
    --storage-root /K3D/Knowledge3D.local \
    --provider ollama \
    --payload-checkpoint-interval-pdfs 10 \
    2>&1 | tee /K3D/Knowledge3D.local/results/base_knowledge_ingest/02_echosystems_libraries/ingest.log"
```

Do NOT use `--ingest` in this phase. Payload generation and Galaxy ingestion are
separate phases. This keeps them independently resumable.

**Resumability:** The stage checkpoint system (`stages/`) tracks per-PDF per-page
progress. If the process dies, restart with the same command and it skips already-staged
pages automatically.

**No limit:** Do NOT add `--limit-pdfs`. All 330 PDFs must be processed. Knowledge caps
are never acceptable — LOD + Frustum Culling handles scale at runtime.

---

## Phase 2 — Galaxy Resident Ingest (AFTER payload.jsonl is complete)

Do NOT run Phase 2 in this pass. Wait until `ingest_report.json` shows `status: complete`.
When ready, the command will be:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium \
  python scripts/fundamental_ingest_payloads.py \
  --payload /K3D/Knowledge3D.local/results/base_knowledge_ingest/02_echosystems_libraries/payloads/payload.jsonl \
  --storage-root /K3D/Knowledge3D.local
```

---

## What NOT to Do

- Do NOT use `--limit-pdfs` — all 330 PDFs, no cap
- Do NOT use `--ingest` flag in Phase 1 — Phase 2 is separate
- Do NOT run this without tmux — 330 PDFs will take hours
- Do NOT read or kill any existing running PDFs ingest processes (check first with `tmux ls`)
- Do NOT touch the benchmarks or Knowledgeverse runtime while this runs — it's a separate pipeline

---

## Report Back

Write `TEMP/CODEX_TO_CLAUDE_ECHOSYSTEMS_LIBRARY_INGEST_REPORT_2026-04-09.md` with:

1. tmux session name confirmed: `echosys_ingest` (yes/no)
2. Initial PDF discovery count (script logs this at startup)
3. First checkpoint written (after 10 PDFs): `stages/` dir exists (yes/no)
4. `ingest.log` tail (last 20 lines) after 5 minutes to confirm the process is alive
5. Estimated total PDFs found by the `**/*.pdf` glob
6. Any startup errors

This is Phase 1 only. Do not report Phase 2 results in this pass.
