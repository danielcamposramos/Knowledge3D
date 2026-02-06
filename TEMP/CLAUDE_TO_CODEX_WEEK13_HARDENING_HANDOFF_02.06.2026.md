# Claude → Codex: Week 13 Hardening Handoff (Remove Placeholders)

**Date:** February 6, 2026
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Partner)
**Priority:** CRITICAL (Week 13 Hardening)
**Context:** Post-Phase 1B (32/32 tests passing) → Remove placeholders, harden for Phase 1C

---

## Executive Summary

**User Feedback on Phase 1B Execution:**

✅ **Excellent work!** 32/32 tests passing, 11/11 corpus entries processed, real enrichment artifacts generated.

❌ **Critical Issue:** *"Make sure placeholders are only temporary, they should not be a norm, actually, we do not use that here neither gave up on sovereignty on the hotpath, so all things external must aim to be transformed to K3D standard ASAP."*

**What needs hardening:**
1. **Local LLM Integration** — Structured prompts, RAG with numbered context, model unload/reload
2. **Stargate Crystallization** — Remove synthetic `embedding_count`, real Galaxy population
3. **K3D Transformation** — ALL external outputs → RPN programs + Galaxy entries immediately
4. **tmux Orchestration** — User wants to monitor GPU graphs, ping when settled

---

## Two Enhancement Specifications

I've written comprehensive specifications for you:

### 1. [LOCAL_LLM_ENHANCEMENT_SPECIFICATION_02.06.2026.md](LOCAL_LLM_ENHANCEMENT_SPECIFICATION_02.06.2026.md)

**What it covers:**
- ✅ Structured system prompts (JSON output, no free-form text)
- ✅ RAG with numbered context (LLM can request more pages/lines)
- ✅ Model unload/reload between tasks (clean context, worth time cost)
- ✅ K3D Transformation (pattern → RPN, concept → Galaxy entry)

**Key Components:**
- `NumberedContextProvider` — Split content into numbered chunks
- `OllamaModelManager` — Load/unload models per task
- `K3DTransformer` — Transform external outputs → sovereign artifacts

### 2. [STARGATE_CRYSTALLIZATION_HARDENING_02.06.2026.md](STARGATE_CRYSTALLIZATION_HARDENING_02.06.2026.md)

**What it covers:**
- ✅ Remove synthetic `embedding_count` placeholder
- ✅ Real crystallization pipeline (enrichment → RPN → Galaxy)
- ✅ Enhanced feeders (PDF, audio, code, image)
- ✅ Integration with `BatchOrchestrator`

**Key Changes:**
- `IngestionStargate._execute_ingestion_pipeline()` — Full pipeline
- `IngestionStargate._crystallize_enrichment()` — Real Galaxy population
- Enhanced `wait_for_job()` — Returns REAL counts (not synthetic)

---

## Week 13 Implementation Plan

### Day 1-2: Local LLM Enhancements

**Files to Create:**
1. `knowledge3d/ingestion/numbered_context.py`
   - Full `NumberedContextProvider` implementation
2. `knowledge3d/ingestion/ollama_manager.py`
   - Full `OllamaModelManager` with load/unload
3. `knowledge3d/ingestion/k3d_transformer.py`
   - Full `K3DTransformer` (pattern → RPN, concept → Galaxy)
4. `tests/test_local_llm_enhancements.py`
   - 4 tests (structured prompts, RAG, model lifecycle, K3D transformation)

**Files to Enhance:**
- `knowledge3d/ingestion/enrichment_pipeline.py`
  - Add structured JSON prompts (schema-based)
  - Add RAG with numbered context
  - Add model unload/reload per document
  - Integrate `K3DTransformer`

**Success Criteria:**
- ✅ 100% structured JSON output from LLMs
- ✅ RAG working (LLM requests more chunks)
- ✅ Models unload/reload between documents
- ✅ ALL outputs transform to RPN + Galaxy entries

### Day 3-5: Stargate Crystallization

**Files to Enhance:**
1. `knowledge3d/knowledgeverse/stargate.py`
   - Add `_execute_ingestion_pipeline()`
   - Add `_crystallize_enrichment()`
   - Update `wait_for_job()` for REAL counts
   - **Remove synthetic `embedding_count`**

2. `knowledge3d/ingestion/feeders/pdf_feeder.py`
   - Integrate with `EnrichmentPipeline`
   - Integrate with `K3DTransformer`
   - Output enrichment artifacts

3. `knowledge3d/ingestion/batch_orchestrator.py`
   - Update `ingest_entry()` to use REAL counts
   - Add `rpn_count` and `galaxy_entries` fields

4. `tests/test_stargate_crystallization.py` (new)
   - 2 tests (real crystallization, end-to-end PDF → Galaxy)

**Success Criteria:**
- ✅ 0 synthetic placeholders in production code
- ✅ Real Galaxy population (Grammar: 500+, Math: 1000+, Reality: 200+)
- ✅ TRM can navigate enriched Galaxies

---

## tmux Orchestration (User Request)

**User Directive:** *"Ask Codex to leverage Debian's tmux to orchestrate it, I can ping him back when I see the GPU graph settles in the desktop."*

### tmux Session Setup

**Create orchestration script:**

```bash
#!/bin/bash
# scripts/week13_hardening_tmux.sh

# Create tmux session for Week 13 hardening
tmux new-session -d -s k3d_week13

# Window 0: GPU Monitor
tmux rename-window -t k3d_week13:0 'gpu_monitor'
tmux send-keys -t k3d_week13:0 'watch -n 1 nvidia-smi' C-m

# Window 1: Local LLM Enhancement (Day 1-2)
tmux new-window -t k3d_week13:1 -n 'llm_enhance'
tmux send-keys -t k3d_week13:1 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D' C-m
tmux send-keys -t k3d_week13:1 'source ~/.bashrc' C-m
tmux send-keys -t k3d_week13:1 '# Ready for Day 1-2: Local LLM enhancements' C-m
tmux send-keys -t k3d_week13:1 '# Run: /home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/test_llm_enhancements.py' C-m

# Window 2: Stargate Hardening (Day 3-5)
tmux new-window -t k3d_week13:2 -n 'stargate'
tmux send-keys -t k3d_week13:2 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D' C-m
tmux send-keys -t k3d_week13:2 'source ~/.bashrc' C-m
tmux send-keys -t k3d_week13:2 '# Ready for Day 3-5: Stargate crystallization' C-m

# Window 3: Test Suite
tmux new-window -t k3d_week13:3 -n 'tests'
tmux send-keys -t k3d_week13:3 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D' C-m
tmux send-keys -t k3d_week13:3 'source ~/.bashrc' C-m
tmux send-keys -t k3d_week13:3 '# Ready for test runs' C-m

# Window 4: Phase 1B Re-Run (Final Validation)
tmux new-window -t k3d_week13:4 -n 'phase1b_rerun'
tmux send-keys -t k3d_week13:4 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D' C-m
tmux send-keys -t k3d_week13:4 'source ~/.bashrc' C-m
tmux send-keys -t k3d_week13:4 '# Ready for final Phase 1B re-run with hardened pipeline' C-m

# Attach to session (start in GPU monitor)
tmux attach-session -t k3d_week13
```

**Usage:**

```bash
# Start tmux session
chmod +x scripts/week13_hardening_tmux.sh
./scripts/week13_hardening_tmux.sh

# Navigate between windows:
# Ctrl+b 0 → GPU monitor
# Ctrl+b 1 → LLM enhancement window
# Ctrl+b 2 → Stargate hardening window
# Ctrl+b 3 → Test suite window
# Ctrl+b 4 → Phase 1B re-run window

# User can watch GPU monitor (window 0) while tasks run in background
# When GPU graph settles (usage drops), user knows task is complete
```

### Execution Commands (Run in tmux windows)

**Day 1-2 (Window 1: llm_enhance):**

```bash
# Create new components
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python -c "
# Test NumberedContextProvider
from knowledge3d.ingestion.numbered_context import NumberedContextProvider
content = 'Line 1\n' * 5000
provider = NumberedContextProvider(content, chunk_size=2000)
print(f'Total chunks: {len(provider.chunks)}')
"

# Test OllamaModelManager
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python -c "
from knowledge3d.ingestion.ollama_manager import OllamaModelManager
with OllamaModelManager() as manager:
    manager.load_model('qwen2.5:14b')
    response = manager.query('Extract concept: algebra')
    print(response)
# Model automatically unloaded (clean context)
"

# Run LLM enhancement tests
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. pytest tests/test_local_llm_enhancements.py -v

# Expected: 4/4 tests passing
```

**Day 3-5 (Window 2: stargate):**

```bash
# Test Stargate crystallization
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. pytest tests/test_stargate_crystallization.py -v

# Expected: 2/2 tests passing

# Integration test
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. pytest tests/test_ingestion_pipeline.py::test_end_to_end_pdf_to_galaxy_real -v

# Expected: REAL Galaxy population (not synthetic)
```

**Final Validation (Window 4: phase1b_rerun):**

```bash
# Re-run Phase 1B with hardened pipeline
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/execute_knowledge_prep_phase1b.py --max-parallel 2 --use-local-models

# Expected output:
# - 11/11 corpus entries processed
# - REAL Galaxy entries created (not synthetic)
# - Grammar Galaxy: 500+ entries
# - Math Galaxy: 1000+ entries
# - Reality Galaxy: 200+ entries

# Run full test suite
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. pytest tests/ -v

# Expected: 38/38 tests passing (32 MVP + 4 LLM + 2 Stargate)
```

### GPU Monitoring

**In Window 0 (gpu_monitor), user will see:**

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.86.10    Driver Version: 535.86.10    CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 4090     Off  | 00000000:01:00.0  On |                  Off |
| 45%   68C    P2   320W / 450W |  42000MiB / 24564MiB |     98%      Default |
+-------------------------------+----------------------+----------------------+

# When task starts: GPU-Util jumps to 98%, Memory-Usage increases
# When task completes: GPU-Util drops to 5-10%, Memory-Usage stable
# User watches this and knows when to ping Codex back
```

---

## Testing Strategy

### Regression Testing (Before Starting)

```bash
# In Window 3 (tests)
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. pytest tests/ -v

# Expected: 32/32 tests passing (MVP Phase 1 baseline)
# If ANY test fails, STOP and fix before proceeding
```

### Progressive Testing (During Implementation)

**After Day 1-2:**
```bash
pytest tests/test_local_llm_enhancements.py -v
# Expected: 4/4 new tests passing
# Total: 36/36 (32 MVP + 4 LLM)
```

**After Day 3-5:**
```bash
pytest tests/test_stargate_crystallization.py -v
# Expected: 2/2 new tests passing
# Total: 38/38 (32 MVP + 4 LLM + 2 Stargate)
```

### Final Validation

```bash
# Full test suite
pytest tests/ -v
# Expected: 38/38 passing

# Phase 1B re-run
python scripts/execute_knowledge_prep_phase1b.py --max-parallel 2 --use-local-models
# Expected: REAL Galaxy population, no placeholders
```

---

## Success Metrics

### Code Quality

1. **No Placeholders:**
   - ✅ 0 synthetic `embedding_count` in Stargate
   - ✅ 0 placeholder logic in production code
   - ✅ ALL external outputs → RPN + Galaxy entries

2. **Structured LLM Output:**
   - ✅ 100% JSON responses (no free-form text)
   - ✅ <5% JSON parse failures (with retry)
   - ✅ Average 2.3 patterns per document

3. **Model Lifecycle:**
   - ✅ 0 context pollution incidents
   - ✅ Average unload time: 5-10 seconds
   - ✅ Average reload time: 8-12 seconds
   - ✅ Total time cost per document: +15 seconds (acceptable)

### Galaxy Population (CRITICAL)

4. **Real Crystallization:**
   - ✅ Grammar Galaxy: 500+ entries (target)
   - ✅ Math Galaxy: 1000+ entries (target)
   - ✅ Reality Galaxy: 200+ entries (target)
   - ✅ ALL entries have RPN programs (not external metadata)

5. **TRM Readiness:**
   - ✅ TRM can query enriched Galaxies
   - ✅ TRM can compose from procedural patterns
   - ✅ Shadow Copy records successful navigations

6. **Sovereignty Compliance:**
   - ✅ Hot path remains PTX-only (no regressions)
   - ✅ All outputs are sovereign artifacts (RPN + Galaxy + PTX)
   - ✅ No external formats persist beyond transformation

---

## Deliverables

**At end of Week 13, provide completion report:**

**File:** `TEMP/CODEX_WEEK13_HARDENING_COMPLETION_REPORT_02.XX.2026.md`

**Sections:**
1. **Executive Summary** (what was accomplished)
2. **Implementation Details** (files created/enhanced, lines of code)
3. **LLM Enhancement Metrics:**
   - JSON output success rate
   - RAG context request rate
   - Model unload/reload times
4. **Stargate Crystallization Metrics:**
   - Galaxy entries created (Grammar, Math, Reality)
   - RPN programs created
   - Embeddings stored
5. **Test Results:**
   - 38/38 tests passing
   - No regressions
6. **Phase 1B Re-Run Results:**
   - REAL Galaxy population (screenshots or counts)
   - Comparison: synthetic vs real counts
7. **GPU Usage Analysis:**
   - Peak VRAM usage
   - Average task duration
   - Time cost per document (with model unload/reload)
8. **Lessons Learned**
9. **Handoff to Phase 1C** (ready for benchmarks?)

---

## Critical Reminders

### 1. NO PLACEHOLDERS

**User was explicit:** *"Make sure placeholders are only temporary, they should not be a norm."*

Every line of placeholder code MUST be removed. If you find a placeholder, replace it with real implementation immediately.

### 2. Sovereignty Applies to ALL Outputs

**Not just hot path!** Ingestion outputs must also be sovereign artifacts:
- External LLM response → Transform → RPN program
- External concept → Transform → Galaxy entry
- External embedding → Transform → Matryoshka vector in Galaxy

**NO external formats persist** beyond transformation.

### 3. Model Unload/Reload is MANDATORY

**User explicitly requested:** Model must unload after each task (clean context).

Even though it costs +15 seconds per document, user says it's "worth it." Do NOT skip this for performance reasons.

### 4. Use tmux for GPU Monitoring

**User wants to see GPU graphs settle** to know when tasks are complete.

Set up tmux windows properly so user can monitor in real-time.

---

## Questions for Claude (If Blocked)

If you hit ANY blockers, document them and escalate:

1. **LLM Output Parsing:** If LLMs still produce malformed JSON after 3 retries?
2. **RAG Context:** How many chunks should initial window provide? (Currently: 1)
3. **Model Selection:** Which Ollama model for which domain? (Currently: deepseek-r1 for math)
4. **Crystallization Strategy:** Should ALL patterns become Grammar rules? Or split by domain?
5. **Galaxy Routing:** How to determine which Galaxy for each concept? (Currently: domain-based)

---

## End of Handoff

**Priority:** CRITICAL (Week 13 Hardening)

**Start here:**
1. Read [LOCAL_LLM_ENHANCEMENT_SPECIFICATION_02.06.2026.md](LOCAL_LLM_ENHANCEMENT_SPECIFICATION_02.06.2026.md) COMPLETELY
2. Read [STARGATE_CRYSTALLIZATION_HARDENING_02.06.2026.md](STARGATE_CRYSTALLIZATION_HARDENING_02.06.2026.md) COMPLETELY
3. Set up tmux session (`scripts/week13_hardening_tmux.sh`)
4. Implement Day 1-2 (Local LLM enhancements)
5. Implement Day 3-5 (Stargate crystallization)
6. Re-run Phase 1B with hardened pipeline
7. Write completion report

**Remember:** NO PLACEHOLDERS. Sovereignty applies to ALL outputs. Model unload/reload is MANDATORY. Use tmux for GPU monitoring.

**Contact:** Claude (Architecture Partner) for design questions, User for strategic decisions.

**Let's remove all placeholders and deliver production-ready sovereign artifacts!** 🚀

---

**Claude (Architecture Partner)**
February 6, 2026

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
