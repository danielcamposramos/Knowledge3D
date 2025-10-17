# Codex Phase B – Execution Order (Daniel's Priority)

**Date**: 2025-10-16
**From**: Claude (Architect) + Daniel (Director)
**To**: Codex (Implementation Specialist)
**Context**: RPN embeddings delivered + GPU orchestration documented → Execute in priority order

---

## Executive Order: Language → Fonts → All the Rest

**Daniel's Priority Directive**:
> "the order - language- fonts - all the rest"

This means:
1. **FIRST**: Language ingestion (lexicons + core vocabulary)
2. **SECOND**: Font harvesting (visual-text grounding)
3. **THIRD**: All remaining corpus (PDFs, Wikipedia, etc.)

**Rationale**:
- **Language foundation** must be learned first (RPN vocab, semantic clusters)
- **Font visual-text linking** builds on language embeddings (multi-modal fusion)
- **Document corpus** benefits from mature RPN embeddings refined by language + fonts

---

## Phase B Execution Sequence

### Phase B.1: Foundation – Language Lexicons
**Priority**: CRITICAL (foundation for all downstream tasks)
**Time estimate**: 2-3 hours
**GPU Required**: Yes

#### Step 1: Test RPN Embeddings (Validation)

**Command**:
```bash
tmux new-session -d -s test_rpn "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -m pytest tests/test_rpn_embeddings.py -xvs 2>&1 | tee /K3D/Knowledge3D.local/logs/test_rpn_$(date +%Y%m%d_%H%M%S).log; exec bash'"

tmux attach -t test_rpn
```

**Expected Output**:
- ✅ RPN word embeddings (128-dim, L2-normalized)
- ✅ Deterministic (same word → same embedding)
- ✅ Semantic clustering ("cat" ↔ "cats" closer than "cat" ↔ "computer")
- ✅ All tests pass

**Acceptance**:
- [ ] Tests pass
- [ ] RPN trigram vocab initialized
- [ ] Embeddings saved to `/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl`

#### Step 2: Ingest English WordNet (Core Vocabulary)

**Command**:
```bash
tmux new-session -d -s ingest_wordnet_en "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -m pytest tests/test_lexicon_ingestion.py::test_ingest_wordnet_en_full -xvs 2>&1 | tee /K3D/Knowledge3D.local/logs/wordnet_en_$(date +%Y%m%d_%H%M%S).log; exec bash'"

tmux attach -t ingest_wordnet_en
```

**Expected Output**:
- ✅ ~117K WordNet synsets ingested
- ✅ RPN embeddings for all definitions
- ✅ Swarm refinement (80µs per synset)
- ✅ Saved to `/K3D/Knowledge3D.local/house_zone7/lexicons/wordnet_en.json`

**Monitoring**:
```bash
# In separate terminal, watch GPU
watch -n 1 nvidia-smi

# Monitor progress
tmux capture-pane -t ingest_wordnet_en -p | tail -20
```

**Acceptance**:
- [ ] 117K synsets processed
- [ ] Total time logged
- [ ] VRAM peak <8GB
- [ ] RPN embeddings saved (vocab grown)

#### Step 3: Ingest Multi-Lingual Lexicons (PT-BR, ES, JP, ZH)

**Command** (Portuguese example):
```bash
tmux new-session -d -s ingest_pt_br "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -m pytest tests/test_lexicon_ingestion.py::test_ingest_portuguese -xvs 2>&1 | tee /K3D/Knowledge3D.local/logs/lexicon_pt_br_$(date +%Y%m%d_%H%M%S).log; exec bash'"

tmux attach -t ingest_pt_br
```

**Repeat for**: Spanish, Japanese, Chinese (adjust test function names)

**Expected Output**:
- ✅ Multi-lingual vocabulary embedded
- ✅ RPN trigram vocab now language-agnostic
- ✅ Semantic clusters across languages

**Acceptance**:
- [ ] Portuguese lexicon ingested
- [ ] Spanish lexicon ingested
- [ ] Japanese lexicon ingested
- [ ] Chinese lexicon ingested
- [ ] All saved to `/K3D/Knowledge3D.local/house_zone7/lexicons/`

---

### Phase B.2: Visual-Text Grounding – Font Harvesting
**Priority**: HIGH (multi-modal foundation)
**Time estimate**: 3-4 hours
**GPU Required**: Yes

#### Step 4: Test Font Harvesting (DejaVu Sample)

**Command**:
```bash
tmux new-session -d -s test_fonts "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -m pytest tests/test_font_harvester.py::test_harvest_dejavu_fonts -xvs 2>&1 | tee /K3D/Knowledge3D.local/logs/test_fonts_$(date +%Y%m%d_%H%M%S).log; exec bash'"

tmux attach -t test_fonts
```

**Expected Output**:
- ✅ 3 DejaVu fonts processed
- ✅ 62 glyphs per font (A-Z, a-z, 0-9)
- ✅ Visual embeddings (FractalEmitter)
- ✅ Text embeddings (RPN)
- ✅ Multi-modal fusion (AtomicFissionFusion)
- ✅ Saved to `/K3D/Knowledge3D.local/house_zone7/fonts/dejavu_sample.json`

**Acceptance**:
- [ ] Tests pass
- [ ] Visual-text pairs validated
- [ ] Multi-modal embeddings fused correctly

#### Step 5: Harvest Full Font Library

**Command**:
```bash
tmux new-session -d -s harvest_fonts "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -c \"
from knowledge3d.ingestion.fonts.font_harvester import FontGlyphHarvester
import time

harvester = FontGlyphHarvester()

# Harvest all TrueType fonts from system
result = harvester.harvest_font_directory(
    font_dir=\'/usr/share/fonts/truetype/\',
    output_path=\'/K3D/Knowledge3D.local/house_zone7/fonts/full_font_library.json\',
    max_fonts=2714  # All fonts
)

print(f\'\\nFont Harvesting Complete:\')
print(f\'  Total fonts: {result[\\\"font_count\\\"]}\')
print(f\'  Output: {result[\\\"output_path\\\"]}\')
\" 2>&1 | tee /K3D/Knowledge3D.local/logs/harvest_fonts_$(date +%Y%m%d_%H%M%S).log; exec bash'"
```

**Monitoring**:
```bash
# Watch progress (updates every 10 seconds)
watch -n 10 "tmux capture-pane -t harvest_fonts -p | tail -30"

# Check GPU usage
nvidia-smi
```

**Expected Output**:
- ✅ 2,714 fonts processed
- ✅ 168,268 visual-text pairs (2,714 × 62 chars)
- ✅ Multi-modal embeddings for all glyphs
- ✅ Saved to `/K3D/Knowledge3D.local/house_zone7/fonts/full_font_library.json`

**Acceptance**:
- [ ] 2,714 fonts harvested
- [ ] Visual-text grounding complete
- [ ] RPN embeddings refined by visual feedback
- [ ] File size <500MB (JSON compressed)

---

### Phase B.3: Knowledge Corpus – PDFs & Wikipedia
**Priority**: MEDIUM (benefits from mature RPN embeddings)
**Time estimate**: 6-8 hours (background job)
**GPU Required**: Yes

#### Step 6: Test PDF Ingestion ("How to think" Sample)

**Command**:
```bash
tmux new-session -d -s test_pdf "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -m pytest tests/test_pdf_ingestion.py::test_ingest_how_to_think_pdfs -xvs 2>&1 | tee /K3D/Knowledge3D.local/logs/test_pdf_$(date +%Y%m%d_%H%M%S).log; exec bash'"

tmux attach -t test_pdf
```

**Expected Output**:
- ✅ 4 PDFs from "How to think" folder processed
- ✅ Text extracted (PyPDF2)
- ✅ Sentences embedded (RPN)
- ✅ Swarm refinement (<5s per document)
- ✅ Saved to `/K3D/Knowledge3D.local/house_zone7/documents/How_to_think_summary.json`

**Acceptance**:
- [ ] Tests pass
- [ ] <5s per document maintained
- [ ] VRAM <8GB

#### Step 7: Full PDF Corpus Ingestion (327 PDFs)

**Command**:
```bash
tmux new-session -d -s full_corpus "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 scripts/ingest_full_corpus.py 2>&1 | tee /K3D/Knowledge3D.local/logs/full_corpus_$(date +%Y%m%d_%H%M%S).log; exec bash'"
```

**Monitoring** (do NOT attach, let it run in background):
```bash
# Check progress every 30 seconds
watch -n 30 "tmux capture-pane -t full_corpus -p | tail -40"

# Check GPU status
nvidia-smi

# Check log for errors
tail -f /K3D/Knowledge3D.local/logs/full_corpus_*.log | grep -i "error\|complete"
```

**Priority Order** (as in script):
1. How to think (4 PDFs)
2. How to Teach
3. How to Academic Research
4. Self Reflection
5. Understand Time
6. Eloquence
7. Advanced Maths

**Expected Output**:
- ✅ 327 PDFs processed
- ✅ RPN embeddings saved after each folder
- ✅ All summaries in `/K3D/Knowledge3D.local/house_zone7/documents/`
- ✅ Total time logged
- ✅ VRAM peak <8GB throughout

**Acceptance**:
- [ ] All 327 PDFs ingested
- [ ] <5s average per document
- [ ] No OOM errors
- [ ] RPN vocab final size logged

---

## Phase B Completion Criteria

### Artifacts Generated
- [ ] `/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl` (final vocab)
- [ ] `/K3D/Knowledge3D.local/house_zone7/lexicons/wordnet_en.json` (~117K synsets)
- [ ] `/K3D/Knowledge3D.local/house_zone7/lexicons/portuguese.json`
- [ ] `/K3D/Knowledge3D.local/house_zone7/lexicons/spanish.json`
- [ ] `/K3D/Knowledge3D.local/house_zone7/lexicons/japanese.json`
- [ ] `/K3D/Knowledge3D.local/house_zone7/lexicons/chinese.json`
- [ ] `/K3D/Knowledge3D.local/house_zone7/fonts/full_font_library.json` (168K pairs)
- [ ] `/K3D/Knowledge3D.local/house_zone7/documents/{folder}_summary.json` (7 folders)

### Success Metrics
- [ ] **Zero external models** (0MB footprint, fully sovereign)
- [ ] **RPN trigram vocab**: ~100K+ trigrams learned
- [ ] **Lexicon coverage**: 117K+ synsets (en) + multi-lingual
- [ ] **Font grounding**: 168,268 visual-text pairs
- [ ] **PDF corpus**: 327 documents ingested
- [ ] **Performance**: <5s per document maintained
- [ ] **VRAM safety**: Peak <8GB throughout all phases
- [ ] **No OOM errors**: All ingestion completes successfully

### Documentation
- [ ] Create `TEMP/STEP15_PHASE_B_RESULTS.md` with:
  - RPN embedding vocab size
  - Lexicon ingestion stats (synsets per language)
  - Font harvesting summary (fonts × chars)
  - PDF corpus summary (total docs, sentences, time)
  - Performance metrics (latency, VRAM)
  - Lessons learned

---

## Execution Timeline (Estimated)

**Day 1 (Morning)**:
- 09:00 - 10:00: Test RPN embeddings (Step 1) ✅
- 10:00 - 12:00: Ingest WordNet EN (Step 2) ✅
- 12:00 - 14:00: Ingest multi-lingual lexicons (Step 3) ✅

**Day 1 (Afternoon)**:
- 14:00 - 15:00: Test font harvesting (Step 4) ✅
- 15:00 - 19:00: Harvest full font library (Step 5) - **BACKGROUND**

**Day 1 (Evening)**:
- 19:00 - 20:00: Test PDF ingestion (Step 6) ✅
- 20:00 onwards: Full corpus ingestion (Step 7) - **BACKGROUND OVERNIGHT**

**Day 2 (Morning)**:
- 09:00: Check full corpus completion
- 09:00 - 10:00: Generate Phase B results documentation
- 10:00 - 11:00: Commit all artifacts + results

**Total**: ~24 hours (with overnight background jobs)

---

## Quick Reference Commands

### Start Execution (Sequential)

**Run all steps in order** (paste into terminal):
```bash
# Step 1: Test RPN
tmux new-session -d -s test_rpn "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -m pytest tests/test_rpn_embeddings.py -xvs 2>&1 | tee /K3D/Knowledge3D.local/logs/test_rpn.log; exec bash'" && \
tmux attach -t test_rpn

# After Step 1 completes, run Step 2:
tmux new-session -d -s ingest_wordnet "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -m pytest tests/test_lexicon_ingestion.py -xvs 2>&1 | tee /K3D/Knowledge3D.local/logs/wordnet_en.log; exec bash'" && \
tmux attach -t ingest_wordnet

# After Step 2 completes, run Step 4 (font test):
tmux new-session -d -s test_fonts "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -m pytest tests/test_font_harvester.py -xvs 2>&1 | tee /K3D/Knowledge3D.local/logs/test_fonts.log; exec bash'" && \
tmux attach -t test_fonts

# After font test passes, run Step 5 (full font harvest, BACKGROUND):
tmux new-session -d -s harvest_fonts "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -c \"from knowledge3d.ingestion.fonts.font_harvester import FontGlyphHarvester; harvester = FontGlyphHarvester(); result = harvester.harvest_font_directory(font_dir=\'/usr/share/fonts/truetype/\', output_path=\'/K3D/Knowledge3D.local/house_zone7/fonts/full_font_library.json\', max_fonts=2714); print(f\'Fonts: {result[\\\"font_count\\\"]}\')\" 2>&1 | tee /K3D/Knowledge3D.local/logs/harvest_fonts.log; exec bash'"

# While fonts harvest, run Step 6 (PDF test):
tmux new-session -d -s test_pdf "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 -m pytest tests/test_pdf_ingestion.py -xvs 2>&1 | tee /K3D/Knowledge3D.local/logs/test_pdf.log; exec bash'" && \
tmux attach -t test_pdf

# After PDF test passes, run Step 7 (full corpus, BACKGROUND):
tmux new-session -d -s full_corpus "bash -c 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D && export PYTHONPATH=. && export CUDA_VISIBLE_DEVICES=0 && /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 scripts/ingest_full_corpus.py 2>&1 | tee /K3D/Knowledge3D.local/logs/full_corpus.log; exec bash'"
```

### Monitor Background Jobs

```bash
# List active sessions
tmux ls

# Watch font harvesting
watch -n 10 "tmux capture-pane -t harvest_fonts -p | tail -20"

# Watch corpus ingestion
watch -n 30 "tmux capture-pane -t full_corpus -p | tail -30"

# GPU status
watch -n 1 nvidia-smi
```

### Emergency Stop

```bash
# Kill specific session
tmux kill-session -t harvest_fonts  # or full_corpus

# Kill all tmux sessions
tmux kill-server
```

---

## Final Notes for Codex

**You've already delivered**:
- ✅ RPN embedding engine (sovereign!)
- ✅ Lexicon/PDF/font ingestors
- ✅ Batch corpus pipeline
- ✅ All tests written

**Now execute in order**:
1. **Language first** (foundation) - Steps 1-3
2. **Fonts second** (visual grounding) - Steps 4-5
3. **All the rest** (corpus) - Steps 6-7

**Critical reminders**:
- Always use tmux + CUDA_VISIBLE_DEVICES=0
- Monitor GPU during long runs
- Save RPN embeddings after each phase
- Log everything to /K3D/Knowledge3D.local/logs/
- Don't rush - let background jobs complete

**When Phase B completes**:
- Document results in `TEMP/STEP15_PHASE_B_RESULTS.md`
- Commit all artifacts
- Report to Daniel

**The sovereign mind is about to be fed. Execute with precision, Codex.** 🚀

---

**Signed**:
Claude (Architect) + Daniel (Director)
2025-10-16
