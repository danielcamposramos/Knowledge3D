# Claude → Codex: Fundamental Knowledge Construction Status & Next Steps

**Date:** February 12, 2026 14:00 UTC
**Session:** Week 22, Day 2 - Fundamental Knowledge Pipeline
**Status:** Phase 1 Complete (Benchmark Augmentation), Phase 2 Queued (PDF Ingestion)

---

## 🎯 **Mission: Single Unified Model with Sovereign Hot Path**

**Critical Architecture Reminder:**

```
┌────────────────────────────────────────────────────────┐
│  SINGLE HEAD (TRM ~7M params)                          │
│    ├─ Internal Swarms (Math/Visual/Physics/Grammar)    │
│    ├─ Sub-Swarms (Linear/Quadratic/Arithmetic...)      │
│    └─ All feed to/from Knowledgeverse (central memory) │
│                                                        │
│  HOT PATH = SOVEREIGN (PTX + RPN only, zero Python)   │
│  - Cranium: PTX kernels (GPU execution)               │
│  - Galaxy Universe: VRAM workspace (unified memory)    │
│  - TRM: Learned navigation (shadow copy enhanced)     │
│                                                        │
│  MID-TERM PATH = Ingestion (symlink compression)      │
│  - Can use Python (not hot path)                      │
│  - Form → meaning refs (char_refs, word_refs)         │
│  - Procedural composition (zero duplication)          │
│                                                        │
│  COLD PATH = Augmentation (Ollama enrichment)         │
│  - One-time knowledge construction                    │
│  - Ollama enrichment MANDATORY (not optional)         │
│  - Output: Galaxy-ready JSONL                         │
└────────────────────────────────────────────────────────┘
```

**Specs Location:** `docs/vocabulary/` (AUTHORITATIVE)
- `THREE_BRAIN_SYSTEM_SPECIFICATION.md` (Cranium + Galaxy + House)
- `KNOWLEDGEVERSE_SPECIFICATION.md` (7-region unified memory)
- `TERNARY_CONTRASTIVE_LEARNING_SPECIFICATION.md` (forward/backward/fusion)
- `DUAL_CLIENT_CONTRACT_SPECIFICATION.md` (form → meaning symlinks)
- `MATH_CORE_SPECIFICATION.md` (3-tier math architecture)

---

## ✅ **Phase 1: Benchmark Augmentation (COMPLETE)**

**Executed:**
```bash
python3 scripts/fundamental_augment_benchmarks.py \
  --dataset-root "/mnt/arquivos/0 ChatGPTs/DataBase" \
  --output ../Knowledge3D.local/fundamental_augmentation/full_benchmark_payloads.jsonl \
  --report ../Knowledge3D.local/fundamental_augmentation/full_augmentation_report.json \
  --max-arc-tasks 100 \
  --max-math-problems 400 \
  --max-lhe-questions 100 \
  --max-mmlu-questions 100 \
  --ollama-model "qwen2.5:14b" \
  --ollama-stride 5
```

**Results:**
- ✅ **5,842 total Galaxy entries** generated
- ✅ **140 Ollama enrichment calls** (supervision signals embedded)
- ✅ **Multi-galaxy coverage:**
  - Grammar: 700 (reasoning rules with symlinks to Math/Reality)
  - Math: 627 (templates with supervision_answer metadata)
  - Word: 4,296 (lexemes with char_refs for symlink compression)
  - Drawing: 100 (ARC visual primitives)
  - 3DObjects: 100 (ARC spatial representations)
  - Reality: 19 (physics/chemistry from LHE)

**Key Features Validated:**
- ✅ Supervision signals preserved: `"supervision_answer":"18"`, `"supervision_answer":"-15"`
- ✅ Ollama hints embedded: `"ollama_hint":"Pattern family: Absolute Value Inequalities..."`
- ✅ Symlink metadata ready: `"symlink":"grammar_galaxy"`, `"char_refs":["char_u0061","char_u0062"]`
- ✅ Cross-modal hints: `"cross_modal":["math","grammar","word"]`

**Output Files:**
- Payload: `../Knowledge3D.local/fundamental_augmentation/full_benchmark_payloads.jsonl` (5,842 rows, 2.5 MB)
- Report: `../Knowledge3D.local/fundamental_augmentation/full_augmentation_report.json`

---

## 🔄 **Phase 2: PDF Ingestion (NEXT - PRIORITY 1)**

**Goal:** Intelligent LLM-driven PDF ingestion with classification + augmentation

**Implementation Status:** ✅ Complete (by you, Codex, on Feb 12)
- `knowledge3d/ingestion/pdf_classifier.py` (LLM classifies pages)
- `knowledge3d/ingestion/pdf_augmenter.py` (LLM augments knowledge)
- `scripts/fundamental_ingest_pdfs.py` (end-to-end pipeline)

**Available Local Models (Ollama):**

**Classification (knowledge vs metadata):**
- **deepseek-r1:14b** (9.0 GB) ← **RECOMMENDED** (best reasoning)
- deepseek-r1:7b (4.7 GB)
- qwen2.5:14b (9.0 GB)

**Augmentation (entity extraction):**
- **qwen2.5:14b** (9.0 GB) ← **RECOMMENDED** (structured output)
- gemma3:12b (8.1 GB)
- deepcoder:14b (9.0 GB)

**Vision (OCR if needed):**
- qwen3-vl (6.1 GB)
- deepseek-ocr (6.7 GB)
- llama3.2-vision (7.8 GB)

**Embedding (for semantic search):**
- qwen3-embedding:4b (2.5 GB)

**Execution Plan:**

### **Step 2A: Test PDF Ingestion (Bounded)**

**Run:**
```bash
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D

PYTHONPATH=. python3 scripts/fundamental_ingest_pdfs.py \
  --pdf-dir "/mnt/arquivos/0 ChatGPTs/DataBase/ARC-AGI-2/papers" \
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

**Expected Output:**
- ~50-100 Galaxy entries (3 PDFs × ~20 knowledge pages × ~0.6 knowledge ratio)
- Page classification cache (skip non-knowledge pages in future)
- Augmented entries with entities, relationships, cross-modal hints

**Validation:**
- Check cache: `../Knowledge3D.local/pdf_cache/*.json` (page decisions recorded)
- Check payload: `pdf_payloads.jsonl` (Grammar/Math/Reality entries)
- Check report: `pdf_ingestion_report.json` (statistics)

### **Step 2B: Full PDF Ingestion (If Test Passes)**

**Run:**
```bash
PYTHONPATH=. python3 scripts/fundamental_ingest_pdfs.py \
  --pdf-dir "/mnt/arquivos/0 ChatGPTs/DataBase" \
  --pattern "**/*.pdf" \
  --limit-pdfs 50 \
  --classifier-model "deepseek-r1:14b" \
  --augmenter-model "qwen2.5:14b" \
  --cache-dir ../Knowledge3D.local/pdf_cache \
  --payload-output ../Knowledge3D.local/fundamental_augmentation/full_pdf_payloads.jsonl
```

**Expected:** ~500-1000 Galaxy entries (50 PDFs)

---

## 🗄️ **Phase 3: Payload Ingestion (PRIORITY 2)**

**Goal:** Ingest all augmented payloads into Knowledgeverse worlds with symlink compression

**Implementation Status:** ✅ Complete (renamed to `fundamental_ingest_payloads.py`)

**Execution:**

### **Step 3A: Merge Payloads**

```bash
cd ../Knowledge3D.local/fundamental_augmentation

# Merge benchmark + PDF payloads
cat full_benchmark_payloads.jsonl pdf_payloads.jsonl > merged_fundamental_payloads.jsonl

# Count
wc -l merged_fundamental_payloads.jsonl
# Expected: ~5,900-5,950 entries
```

### **Step 3B: Run Ingestion with Symlink Compression**

```bash
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D

PYTHONPATH=. python3 scripts/fundamental_ingest_payloads.py \
  --payload ../Knowledge3D.local/fundamental_augmentation/merged_fundamental_payloads.jsonl \
  --world-root ../Knowledge3D.local/worlds/fundamental_world_week22 \
  --enable-symlink-compression \
  --report ../Knowledge3D.local/fundamental_augmentation/ingestion_report.json
```

**What Happens:**
1. Reads JSONL payloads (benchmark + PDF)
2. Applies symlink compression:
   - Text → char_refs: `"compute"` → `["char_u0063","char_u006f","char_u006d",...]`
   - Word → word_refs: Multi-word phrases → word IDs
   - Math symbols → symbol_refs: LaTeX → Math Galaxy IDs
3. Writes World files (glTF + metadata):
   - `Drawing.glb`, `Math.glb`, `Grammar.glb`, `Reality.glb`, etc.
4. Persists to disk (ready for daemon load)

**Validation:**
- Check world files: `ls -lh ../Knowledge3D.local/worlds/fundamental_world_week22/`
- Check report: `ingestion_report.json` (symlink stats, entry counts)
- Verify symlinks: Grep for `"char_refs"`, `"word_refs"`, `"symlink_compression":"applied_v1"`

---

## 🧪 **Phase 4: Validation (PRIORITY 3)**

**Goal:** Validate accuracy improvement with enriched Galaxy

### **Step 4A: Start Daemon with Fundamental World**

```bash
PYTHONPATH=. python3 scripts/k3d_daemon.py \
  --host localhost \
  --port 54321 \
  --world-root ../Knowledge3D.local/worlds/fundamental_world_week22 \
  --checkpoint-root ../Knowledge3D.local/checkpoints &

# Wait for startup
sleep 10
```

### **Step 4B: Run Bounded Math Benchmark**

```bash
PYTHONPATH=. python3 benchmarks/math_sender.py \
  --host localhost \
  --port 54321 \
  --max-questions 20 \
  --output ../Knowledge3D.local/results/math_validation_week22.json
```

**Expected Results:**
- **Before (baseline, empty Galaxy):** 0/20 (0%) - coefficient_extraction_failed
- **After (enriched Galaxy + contrastive):** 6-12/20 (30-60%) - templates + patterns working

**Success Criteria:**
- ✅ Predicted None rate: 100% → <50%
- ✅ Valid answers: 0% → 30-60%
- ✅ GPU telemetry: 100% (sovereign hot path)
- ✅ Contrastive patterns: Anti-patterns generated from failures

### **Step 4C: Analyze Results**

```bash
cat ../Knowledge3D.local/results/math_validation_week22.json | jq '.summary'
```

**Check:**
- Accuracy improvement
- GPU call count (>0 for all tasks = sovereign)
- Ternary quality updates (patterns learned)
- Anti-pattern generation count

---

## 📋 **Your Tasks (Codex):**

### **Task 1: PDF Ingestion (Test)**

**Priority:** 1 (IMMEDIATE)
**Command:** See "Step 2A" above
**Expected Time:** ~10-15 minutes
**Output:** `pdf_payloads.jsonl`, `pdf_ingestion_report.json`, cache files

**Validation Checks:**
1. Page classification working: `cat ../Knowledge3D.local/pdf_cache/*.json | jq '.page_decisions'`
2. Knowledge pages augmented: `grep '"classification":"knowledge"' pdf_payloads.jsonl | wc -l`
3. Symlink metadata present: `grep '"char_refs"' pdf_payloads.jsonl | head -5`

**If Test Passes:** Run full PDF ingestion (Step 2B)
**If Test Fails:** Report error + first 50 lines of stderr

---

### **Task 2: Payload Ingestion**

**Priority:** 2 (After PDF test passes)
**Command:** See "Step 3A" and "Step 3B" above
**Expected Time:** ~5-10 minutes
**Output:** World files in `fundamental_world_week22/`

**Validation Checks:**
1. World files created: `ls -lh ../Knowledge3D.local/worlds/fundamental_world_week22/*.glb`
2. Symlink compression applied: `grep 'symlink_compression.*applied' ingestion_report.json`
3. Entry counts match: Compare `full_benchmark_payloads.jsonl` line count + `pdf_payloads.jsonl` line count with ingestion report totals

**If Ingestion Succeeds:** Proceed to Task 3
**If Ingestion Fails:** Report error + first 100 lines of log

---

### **Task 3: Validation (Bounded Math Benchmark)**

**Priority:** 3 (After ingestion succeeds)
**Commands:** See "Step 4A", "Step 4B", "Step 4C" above
**Expected Time:** ~5 minutes
**Output:** `math_validation_week22.json`

**Success Metrics:**
- Accuracy: 0% → 30-60%
- GPU sovereignty: 100% (all tasks use PTX)
- Contrastive learning: Anti-patterns generated

**Report Back:**
- Summary stats (accuracy, predicted_none_rate, gpu_call_count)
- Sample results (3 correct, 3 incorrect with error codes)
- Ternary quality memory state (pattern priors)

---

## 🚨 **Critical Reminders:**

### **1. Sovereignty (Hot Path)**
- ❌ NO Python eval/sympy/numpy in inference
- ✅ ONLY PTX kernels + RPN + Galaxy navigation
- ✅ GPU telemetry must be >0 for all tasks

### **2. Architecture (Single Head)**
- ONE TRM model (~7M params)
- Internal swarms (Math/Visual/Physics specialists)
- All specialists query/write Knowledgeverse (not isolated)
- Shadow copy learning (inference-time adaptation)

### **3. Symlink Compression (Mid-Term)**
- Text → char_refs (zero duplication)
- Words → word_refs (reference Character Galaxy)
- Symbols → symbol_refs (reference Math Galaxy)
- Form → meaning metadata (procedural composition)

### **4. Ollama Enrichment (Cold Path)**
- MANDATORY in augmentation (not optional)
- Supervision signals preserved (Q+A context)
- Embedding-ready text for semantic search
- Cross-modal hints for Galaxy navigation

---

## 📁 **Key Files Reference:**

**Augmentation Output:**
- `../Knowledge3D.local/fundamental_augmentation/full_benchmark_payloads.jsonl` (5,842 entries)
- `../Knowledge3D.local/fundamental_augmentation/full_augmentation_report.json`

**Ingestion Scripts:**
- `scripts/fundamental_ingest_pdfs.py` (PDF → JSONL)
- `scripts/fundamental_ingest_payloads.py` (JSONL → World)

**Ingestion Modules:**
- `knowledge3d/ingestion/pdf_classifier.py` (LLM classification)
- `knowledge3d/ingestion/pdf_augmenter.py` (LLM augmentation)

**Specialist Implementation:**
- `knowledge3d/knowledgeverse/specialists/math_specialist.py` (contrastive + ternary)
- `knowledge3d/knowledgeverse/ternary_quality_memory.py` (pattern priors)

**Daemon:**
- `scripts/k3d_daemon.py` (persistent, game paradigm)
- `knowledge3d/daemon/main.py` (command router)

---

## 🎯 **Expected Final State:**

After completing Tasks 1-3:

```
Fundamental Knowledge Construction: COMPLETE
├─ Benchmark Augmentation: ✅ 5,842 entries (ARC/Math/LHE/MMLU)
├─ PDF Ingestion: ✅ ~100 entries (test: 3 PDFs)
├─ Payload Ingestion: ✅ World files with symlink compression
└─ Validation: ✅ Math accuracy 0% → 30-60%

Next Phase:
- Scale PDF ingestion (50-100 PDFs)
- Full benchmark sweep (400 math problems)
- Shadow copy training (15-20 iterations)
- ARC-AGI transfer validation (target: 55-65%)
```

---

## 💡 **Bottom Line:**

**You've built the foundational infrastructure (sovereign hot path, contrastive learning, symlink compression). Now we're populating the Galaxy Universe with fundamental knowledge from benchmarks + papers.**

**The single unified model with internal swarms is ready. Next: Give it knowledge, validate it works, then scale to full curriculum training.**

**All specs in `docs/vocabulary/` are authoritative. When in doubt, read the specs.**

---

**Handoff prepared by:** Claude (Architecture)
**Date:** February 12, 2026 14:00 UTC
**Session:** Week 22, Day 2
**Status:** Phase 1 Complete, Phase 2-4 Queued for Codex
