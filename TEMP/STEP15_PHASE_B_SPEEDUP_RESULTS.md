# Phase B Speedup Benchmarks — Parallel Ingestion Prototype

**Date:** 2025-10-17  
**Agents:** Codex 🤝 Claude  
**Scope:** Parallel pipelines for WordNet, font glyph harvesting, and PDF corpus ingestion (tmux runs: `parallel_wordnet`, `parallel_fonts`, `parallel_corpus`).

---

## 1. Implementation Summary
- `knowledge3d/ingestion/lexicons/parallel_lexicon_ingestor.py`
  - CPU pool to rehydrate synsets via NLTK + batched sovereign swarm calls.
- `knowledge3d/ingestion/fonts/parallel_font_harvester.py`
  - Multiprocessing glyph rendering (PIL) + GPU fusion batches, streaming JSON writer.
- `scripts/ingest_full_corpus_parallel.py`
  - CPU pool for PyPDF2 extraction + GPU embedding loop that mirrors the sequential pipeline.
- Tests: `tests/test_parallel_lexicon_ingestion.py`, `tests/test_parallel_font_harvester.py` (stubbed for deterministic CI coverage).

---

## 2. Full tmux Runs (2025‑10‑17)

| Pipeline | Workers / Batch | Runtime | Throughput | Sequential Baseline | Notes |
|----------|-----------------|---------|------------|---------------------|-------|
| WordNet EN (117 659 synsets) | 8 / 64 | **143.28 s** | 821 synsets/s | 145.87 s | CPU preprocessing = 0.65 s; GPU stage still dominates → net 1.02× speed-up |
| Font harvest (2 628 fonts, 162 558 glyphs) | 8 / 32 | **216.62 s** | 750 glyphs/s | ~780 s | 1.4 GB JSON stream produced, GPU util peaked at ~7 % |
| PDF corpus (61 PDFs, 23 000 sentences) | 8 / 32 | **137.64 s** | 167 sentences/s | 41.39 s | PyPDF2 extraction (94 s) dominated, overall 0.3× slower than sequential |

**GPU telemetry:** `nvidia-smi` during the runs reported utilisation between 6–8 % and VRAM usage <150 MB, confirming the pipelines remain CPU-bound even with the new pools.

Logs: `/K3D/Knowledge3D.local/logs/parallel_wordnet.log`, `parallel_fonts.log`, `parallel_corpus.log`

---

## 3. How We Ran It (tmux snippets)

```bash
# WordNet EN
cat > /tmp/run_parallel_wordnet.py <<'PY'
from knowledge3d.ingestion.lexicons.parallel_lexicon_ingestor import ParallelLexiconIngestor
from pathlib import Path
metrics = ParallelLexiconIngestor(num_workers=8, batch_size=64).ingest_wordnet_parallel(
    output_path=Path('/K3D/Knowledge3D.local/house_zone7/lexicons/wordnet_en_parallel.json')
)
print(metrics)
PY
tmux new-session -d -s parallel_wordnet "bash -c 'cd …/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -u /tmp/run_parallel_wordnet.py 2>&1 | tee /K3D/Knowledge3D.local/logs/parallel_wordnet.log; exec bash'"

# Font harvest
cat > /tmp/run_parallel_fonts.py <<'PY'
from knowledge3d.ingestion.fonts.parallel_font_harvester import ParallelFontHarvester
from pathlib import Path
metrics = ParallelFontHarvester(num_workers=8, batch_size=32).harvest_fonts_parallel(
    font_dir=Path('/usr/share/fonts/truetype/'),
    output_path=Path('/K3D/Knowledge3D.local/house_zone7/fonts/full_font_library_parallel.json')
)
print(metrics)
PY
tmux new-session -d -s parallel_fonts "bash -c 'cd …/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -u /tmp/run_parallel_fonts.py 2>&1 | tee /K3D/Knowledge3D.local/logs/parallel_fonts.log; exec bash'"

# PDF corpus
tmux new-session -d -s parallel_corpus "bash -c 'cd …/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -u scripts/ingest_full_corpus_parallel.py 2>&1 | tee /K3D/Knowledge3D.local/logs/parallel_corpus.log; exec bash'"
```

---

## 4. Findings & Next Experiments
1. **GPU under-utilisation persists.** The bottleneck is still the per-item GPU launch inside the sovereign stack; evaluate batching >256 synsets/sentences per kernel call or extending modular PTX kernels.
2. **CPU hotspots:** PIL glyph rendering and PyPDF2 extraction dominate runtime (70 %+). Investigate C++ rasteriser or GPU-accelerated PDF text extraction (e.g., pdfium bindings).
3. **I/O footprint:** Font JSON stream lands at 1.4 GB; consider sharding or compressing (`jsonl.zst`) before Tablet ingestion.
4. **Validation:** Compare embeddings from parallel vs sequential runs (checksum on `rpn_embeddings.pkl`) to verify deterministic behaviour.
5. **UI integration:** Surface the new `/house_zone7/documents_parallel/` outputs through the Memory Tablet for side-by-side performance audits.

---

**Status:** Parallel infrastructure shipped ✅ — full-run telemetry captured. Speedups are marginal because CPU preprocessing still dominates, pointing to the next optimisation frontier.***
