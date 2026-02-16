# LLM Integration + Dataset Migration Action Plan

**Date:** February 11, 2026
**Architects:** Claude + Daniel
**Implementers:** Claude (coding) + Codex (dataset enhancement)

---

## Executive Summary

**Discovery:** All needed infrastructure and data already exist in `/K3D/K3D_llama_cpp/datasets/`!

**Goals:**
1. ✅ Replace synthetic LHE with real MMLU (multiple-choice, 14,000+ questions)
2. ✅ Replace synthetic Math with real AMC-AIME (competition problems)
3. ✅ Create Chat Specialist using existing Ollama + LLM infrastructure
4. ✅ Enable multi-modal integration (Audio, 3D) using existing data

**Timeline:** 2-3 days (Claude codes architecture, Codex enhances datasets in parallel)

---

## Phase 1: Dataset Migration (Claude + Codex Parallel) — Day 1

### 1.1. MMLU Integration (Claude - 4 hours)

**Source:** `/K3D/K3D_llama_cpp/datasets/MMLU/data/`

**What exists:**
```bash
$ ls /K3D/K3D_llama_cpp/datasets/MMLU/data/
# 57 subjects: abstract_algebra, anatomy, astronomy, ... world_religions
# Each subject has: dev, test, val CSV files
# Total: ~14,000 multiple-choice questions
```

**Action (Claude):**
1. Create `benchmarks/mmlu.py` (replace LHE)
2. Adapter structure:
```python
class MMLUBenchmark:
    def __init__(self, dataset_path="/K3D/K3D_llama_cpp/datasets/MMLU/data", ...):
        self.subjects = self._load_subjects()  # 57 subjects
        self.questions = self._load_questions(max_questions)

    def _load_questions(self, max_questions):
        # Load from CSV: question, A, B, C, D, answer
        # Format compatible with K3D multiple-choice evaluation
        pass

    def run_benchmark(self, use_enriched=True):
        # Same structure as current LHE benchmark
        # Use TRMNavigator for question answering
        pass
```

3. Add to `scripts/run_all_benchmarks.py`:
```python
--max-mmlu-questions 1000  # Default 1000 (subset of 14k)
--mmlu-subjects "all"      # Or specific: "math,physics,chemistry"
--mmlu-require-real-dataset  # Fail-fast if synthetic fallback
```

**Expected Output:**
- MMLU benchmark ready to run on real 14k+ questions
- Can compare to SOTA (Gemini, Claude, GPT performance on MMLU is published)
- Scientific integrity: real dataset, established benchmark

**Deliverable:** `benchmarks/mmlu.py`, updated `run_all_benchmarks.py`

---

### 1.2. AMC-AIME Integration (Codex - 4 hours)

**Source:** `/K3D/K3D_llama_cpp/datasets/AMC-AIME/data/`

**What exists:**
```bash
$ ls /K3D/K3D_llama_cpp/datasets/AMC-AIME/data/
# AMC.zip (21 MB) - American Mathematics Competition problems
# Contains: AMC 8, AMC 10, AMC 12, AIME problems
```

**Action (Codex):**
1. Unzip and explore AMC-AIME data structure
2. Convert to K3D Math Competitions format:
```json
{
  "problems": [
    {
      "id": "amc12_2024_01",
      "competition": "AMC",
      "problem_text": "...",
      "answer": "..."
    }
  ]
}
```

3. Create converters:
```bash
scripts/prepare_amc_dataset.py --source /K3D/K3D_llama_cpp/datasets/AMC-AIME \
  --output ../Knowledge3D.local/datasets/math_competitions/amc_problems.json

scripts/prepare_aime_dataset.py --source /K3D/K3D_llama_cpp/datasets/AMC-AIME \
  --output ../Knowledge3D.local/datasets/math_competitions/aime_problems.json
```

4. Update `benchmarks/math_competitions.py` integrity check:
```python
if not amc_problems_file.exists():
    logger.error("AMC problems not found - run scripts/prepare_amc_dataset.py")
    if require_real_dataset:
        raise FileNotFoundError("Real AMC dataset required")
```

**Expected Output:**
- 1000+ real AMC/AIME problems in K3D format
- Math benchmark verified on competition data
- Can cite "AMC/AIME performance" in paper

**Deliverable:** `amc_problems.json`, `aime_problems.json`, converter scripts

---

## Phase 2: Chat Specialist (Claude - Day 1-2)

### 2.1. Create Chat Specialist Class

**Leverage existing:** `knowledge3d/ingestion/ollama_manager.py` + `knowledge3d/skills/llm.py`

**New file:** `knowledge3d/knowledgeverse/chat_specialist.py`

```python
from knowledge3d.ingestion.ollama_manager import OllamaModelManager
from knowledge3d.skills.llm import LLMSkill
from knowledge3d.knowledgeverse.specialist_base import SpecialistBase

class ChatSpecialist(SpecialistBase):
    """
    Specialist for open-ended conversational tasks using Ollama + LLM.

    Use cases:
    - Open-ended QA (real HLE if we adapt later)
    - Multi-turn dialogue
    - RAG-enhanced responses (query Galaxy, generate answer)
    """

    def __init__(self, knowledgeverse, model="llama3:8b", **kwargs):
        super().__init__(name="chat", domain="conversational", **kwargs)
        self.knowledgeverse = knowledgeverse
        self.ollama = OllamaModelManager()
        self.llm = LLMSkill()
        self.model = model
        self.conversation_history = []

    def answer_question(self, question: str, use_rag=True) -> str:
        """
        Answer open-ended question using RAG from Galaxy Universe.

        1. Query relevant galaxies (Drawing, Math, Grammar, Reality)
        2. Retrieve top-k relevant nodes
        3. Format as context for LLM
        4. Generate answer using Ollama
        """
        if use_rag:
            # Query Galaxy Universe for relevant knowledge
            contexts = self._query_galaxy_context(question)
            # Use LLM RAG skill
            answer = self.llm.answer_with_rag(
                query=question,
                contexts=contexts,
                max_tokens=512
            )
        else:
            # Direct Ollama query (no Galaxy context)
            result = self.ollama.query(
                model=self.model,
                prompt=question,
                timeout=60.0
            )
            answer = result.output

        # Record conversation
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "used_rag": use_rag
        })

        return answer

    def _query_galaxy_context(self, question: str, top_k=5):
        """
        Query Galaxy Universe for relevant knowledge nodes.

        Returns list of (label, content) tuples for RAG.
        """
        contexts = []

        # Query Drawing Galaxy for visual concepts
        drawing_results = self.knowledgeverse.query_galaxy(
            "Drawing",
            semantic_query=question,
            top_k=top_k
        )
        for entry in drawing_results:
            contexts.append((
                f"Drawing:{entry['id']}",
                entry.get('rpn_program', '')
            ))

        # Query Math Galaxy for mathematical concepts
        math_results = self.knowledgeverse.query_galaxy(
            "Math",
            semantic_query=question,
            top_k=top_k
        )
        for entry in math_results:
            contexts.append((
                f"Math:{entry['id']}",
                entry.get('rpn_program', '')
            ))

        # Query Grammar Galaxy for transformation rules
        grammar_results = self.knowledgeverse.query_galaxy(
            "Grammar",
            semantic_query=question,
            top_k=top_k
        )
        for entry in grammar_results:
            contexts.append((
                f"Grammar:{entry['id']}",
                entry.get('rpn_program', '')
            ))

        # Query Reality Galaxy for physics/chemistry/biology
        reality_results = self.knowledgeverse.query_galaxy(
            "Reality",
            semantic_query=question,
            top_k=top_k
        )
        for entry in reality_results:
            contexts.append((
                f"Reality:{entry['id']}",
                entry.get('rpn_program', '')
            ))

        return contexts[:top_k]  # Return top-k total across all galaxies
```

**Integration with TRM:**

```python
# In knowledge3d/knowledgeverse/trm_navigator.py
class TRMNavigator:
    def __init__(self, knowledgeverse):
        self.knowledgeverse = knowledgeverse
        self.chat_specialist = None  # Lazy init

    def get_chat_specialist(self):
        if self.chat_specialist is None:
            from knowledge3d.knowledgeverse.chat_specialist import ChatSpecialist
            self.chat_specialist = ChatSpecialist(
                knowledgeverse=self.knowledgeverse,
                model="llama3:8b"  # Or from config
            )
        return self.chat_specialist

    def answer_open_ended(self, question: str, use_rag=True):
        """Route to chat specialist for open-ended QA."""
        chat = self.get_chat_specialist()
        return chat.answer_question(question, use_rag=use_rag)
```

**Deliverable:** `knowledge3d/knowledgeverse/chat_specialist.py`, TRM integration

---

### 2.2. Adapt LHE Benchmark for Open-Ended (Optional - Day 2)

**If we want to run real HLE eventually:**

```python
# benchmarks/last_humanity_exam_openended.py
class LastHumanityExamOpenEnded:
    """
    Open-ended QA benchmark using real cais/hle dataset.

    Evaluation: Semantic matching (not exact match).
    """

    def __init__(self, knowledgeverse, dataset_path, max_questions=100):
        self.kv = knowledgeverse
        self.questions = self._load_hle_dataset(dataset_path)[:max_questions]

    def run_benchmark(self, use_rag=True):
        navigator = TRMNavigator(knowledgeverse=self.kv)
        results = []

        for q in self.questions:
            # Use chat specialist for open-ended answer
            predicted_answer = navigator.answer_open_ended(
                question=q["question"],
                use_rag=use_rag
            )

            # Semantic matching evaluation
            score = self._evaluate_answer(
                predicted=predicted_answer,
                ground_truth=q["answer"],
                answer_type=q["answer_type"]
            )

            results.append({
                "id": q["id"],
                "question": q["question"],
                "predicted": predicted_answer,
                "ground_truth": q["answer"],
                "score": score
            })

        return {
            "accuracy": sum(r["score"] for r in results) / len(results),
            "results": results
        }

    def _evaluate_answer(self, predicted, ground_truth, answer_type):
        """
        Semantic matching evaluation.

        - Numeric: tolerance-based matching
        - Text: semantic similarity (cosine, BLEU, or LLM judge)
        - Chess notation: exact match
        """
        if answer_type == "numeric":
            # Parse numbers and check tolerance
            return self._numeric_match(predicted, ground_truth)
        elif answer_type == "free_form":
            # Use semantic similarity (embeddings cosine similarity)
            return self._semantic_similarity(predicted, ground_truth)
        else:
            # Exact match for structured answers
            return 1.0 if predicted.strip() == ground_truth.strip() else 0.0
```

**Note:** This is lower priority. Focus on MMLU first (established benchmark).

---

## Phase 3: Multi-Modal Integration (Codex - Day 2-3)

### 3.1. Audio Data Integration

**Source:** `/K3D/K3D_llama_cpp/datasets/audio/` (phonemes for en_us, es_es, pt_br, pt_pt, zh_cn)

**Action (Codex):**
1. Convert phoneme data to Audio Galaxy format:
```python
# scripts/ingest_audio_phonemes.py
# Convert phoneme dictionaries to K3D Audio Galaxy nodes

audio_entry = {
    "id": f"phoneme_{language}_{sound}",
    "type": "audio_primitive",
    "rpn_program": f"PHONEME {sound_ipa} {duration_ms} {frequency_hz}",
    "domain": "audio",
    "provenance": {
        "source": "phoneme_external",
        "language": language,
        "confidence": 1.0
    }
}
```

2. Ingest into Audio Galaxy:
```bash
python scripts/ingest_audio_phonemes.py \
  --source /K3D/K3D_llama_cpp/datasets/audio/phoneme_external \
  --output ../Knowledge3D.local/galaxies/Audio.jsonl
```

**Expected:** +5000-10000 Audio Galaxy entries (phonemes across 5+ languages)

---

### 3.2. 3D Object Data Integration

**Source:** `/K3D/K3D_llama_cpp/datasets/galaxy_geometry/` (3D primitives, transformations)

**Action (Codex):**
1. Convert 3D geometry data to 3DObjects Galaxy format:
```python
# scripts/ingest_3d_geometry.py

object_entry = {
    "id": f"3d_{primitive_type}_{variant}",
    "type": "3d_primitive",
    "rpn_program": f"CUBE {width} {height} {depth} TRANSFORM {matrix}",
    "domain": "3dobjects",
    "provenance": {
        "source": "galaxy_geometry",
        "confidence": 1.0
    },
    "metadata": {
        "vertices": vertex_count,
        "faces": face_count,
        "volume": volume_m3
    }
}
```

2. Ingest into 3DObjects Galaxy:
```bash
python scripts/ingest_3d_geometry.py \
  --source /K3D/K3D_llama_cpp/datasets/galaxy_geometry \
  --output ../Knowledge3D.local/galaxies/3DObjects.jsonl
```

**Expected:** +1000-2000 3DObjects Galaxy entries (primitives, transformations, compositions)

---

### 3.3. Lexicon Data Integration (Word/Character Galaxies)

**Source:** `/K3D/K3D_llama_cpp/datasets/lexicons/`, `/K3D/K3D_llama_cpp/datasets/wordnet/`

**Action (Codex):**
1. Enhance Word Galaxy with WordNet definitions:
```python
# scripts/enrich_word_galaxy_wordnet.py

word_entry = {
    "id": f"word_{lemma}",
    "type": "word",
    "rpn_program": f"WORD {character_sequence}",  # Reference Character Galaxy
    "domain": "word",
    "metadata": {
        "definition": wordnet_definition,
        "synonyms": wordnet_synonyms,
        "pos": part_of_speech,  # noun, verb, adj, adv
        "frequency": corpus_frequency
    }
}
```

2. Ingest linguistic data:
```bash
python scripts/enrich_word_galaxy_wordnet.py \
  --source /K3D/K3D_llama_cpp/datasets/wordnet \
  --output ../Knowledge3D.local/galaxies/Word.jsonl
```

**Expected:** +50000-100000 Word Galaxy entries (definitions, synonyms, POS tags)

---

## Phase 4: Benchmark Validation (Claude + Codex - Day 3)

### 4.1. Run Full Benchmark Suite with Real Data

**Command:**
```bash
conda run -n k3d-cranium env PYTHONPATH=. python scripts/run_all_benchmarks.py \
  --model-persistence-mode unified \
  --max-arc-tasks 100 \
  --max-math-problems 100 \
  --max-mmlu-questions 1000 \
  --arc-enable-full-ptx \
  --arc-enable-contrastive-learning \
  --arc-enable-validity-gates \
  --arc-constraint-mode penalty \
  --arc-enable-figure-ground-reversal \
  --arc-enable-object-aware-generation \
  --arc-enable-rescue-lane --arc-rescue-lane-size 16 \
  --arc-oracle-search-lane-size 32 \
  --arc-enable-dual-track-oracle \
  --arc-enable-fuzzy-oracle --arc-fuzzy-oracle-threshold 0.95 \
  --arc-embedding-lazy-mode skip \
  --track-curriculum-coverage \
  --require-real-datasets \
  --output-dir ../Knowledge3D.local/results/week22_2_full_validation \
  --storage-root ../Knowledge3D.local
```

**Expected Results:**
- ARC: 6% (stable, verified)
- Math: 30-35% (on real AMC/AIME, not synthetic)
- MMLU: 15-25% (first run on real 14k+ benchmark)

**Validation:**
- All benchmarks use REAL datasets (no synthetic fallbacks)
- Integrity checks pass (fail-fast if synthetic)
- Results reproducible and citable

---

### 4.2. Update Paper Evidence

**Files to update:**
1. `docs/paper-evidence/EVIDENCE_MAP_K3D_PAPER.md`:
   - Add MMLU results section
   - Update Math section with AMC/AIME citation
   - Add benchmark integrity section

2. `docs/paper-evidence/PAPER_READY_SUMMARY.md`:
   - Replace LHE with MMLU in ready-to-cite content
   - Update Math performance with real dataset citation

3. `docs/paper-evidence/MANIFEST.md`:
   - Add MMLU results file + checksum
   - Update Math results file + checksum

4. Create `docs/paper-evidence/BENCHMARK_INTEGRITY.md`:
   - Document integrity validation process
   - List all datasets with sources and verification
   - Include SHA256 checksums for dataset files

---

## Phase 5: Parallel Operations Optimization (Claude - Day 3)

### 5.1. Enable Python Multiprocessing Where Possible

**Targets for parallelization:**

1. **Galaxy ingestion scripts:**
```python
# scripts/ingest_multi_modal_parallel.py
from multiprocessing import Pool

def ingest_galaxy_parallel(galaxy_name, data_source):
    # Ingest one galaxy (Audio, 3DObjects, Word, etc.)
    pass

if __name__ == "__main__":
    galaxies_to_ingest = [
        ("Audio", "/K3D/K3D_llama_cpp/datasets/audio/phoneme_external"),
        ("3DObjects", "/K3D/K3D_llama_cpp/datasets/galaxy_geometry"),
        ("Word", "/K3D/K3D_llama_cpp/datasets/wordnet"),
    ]

    # Parallel ingestion (3 processes)
    with Pool(processes=3) as pool:
        pool.starmap(ingest_galaxy_parallel, galaxies_to_ingest)
```

2. **Dataset preparation scripts:**
```python
# scripts/prepare_all_datasets_parallel.py
from concurrent.futures import ProcessPoolExecutor

tasks = [
    prepare_amc_dataset,
    prepare_aime_dataset,
    prepare_mmlu_dataset,
    prepare_audio_phonemes,
    prepare_3d_geometry,
]

with ProcessPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(task) for task in tasks]
    for future in futures:
        future.result()  # Wait for completion
```

3. **Benchmark evaluation (careful with GPU):**
```python
# For CPU-bound tasks (preprocessing, validation)
# NOT for GPU-bound tasks (PTX execution - GPU doesn't parallelize well across processes)

# Example: Parallel ARC task preprocessing
from multiprocessing import Pool

def preprocess_arc_task(task):
    # Extract features, compute constraints, etc. (CPU-bound)
    return preprocessed_task

with Pool(processes=8) as pool:
    preprocessed_tasks = pool.map(preprocess_arc_task, raw_tasks)
```

**Important:** GPU operations (PTX kernels) remain sequential (one unified Knowledgeverse instance).

---

## Summary Timeline

### Day 1 (Today/Tomorrow)
- **Morning (Claude):** Create `benchmarks/mmlu.py`, integrate MMLU dataset
- **Morning (Codex):** Extract and convert AMC-AIME data
- **Afternoon (Claude):** Create `chat_specialist.py`, integrate with TRM
- **Afternoon (Codex):** Prepare audio/3D/lexicon ingestion scripts

### Day 2
- **Morning (Claude):** Test MMLU benchmark, verify results
- **Morning (Codex):** Run audio/3D/lexicon ingestion (parallel)
- **Afternoon (Claude):** Implement parallel operations framework
- **Afternoon (Codex):** Validate ingested data (counts, format, integrity)

### Day 3
- **Morning (Both):** Run full benchmark suite (ARC 100, Math 100, MMLU 1000)
- **Afternoon (Claude):** Update paper evidence files
- **Afternoon (Codex):** Generate benchmark integrity report
- **Evening:** Review results, commit to repo

---

## Expected Outcomes

### Scientific Integrity ✅
- All benchmarks use REAL datasets (no synthetic fallbacks)
- MMLU: 14,000+ questions (established benchmark, SOTA comparison valid)
- Math: 1000+ AMC/AIME problems (competition data, citable)
- ARC: 400 tasks (verified)

### Architecture Enhancements ✅
- Chat Specialist: RAG-enhanced conversational QA
- Multi-modal: Audio, 3D, Lexicon galaxies populated
- Parallel operations: 2-3× faster ingestion/preparation

### Paper Evidence ✅
- Can cite MMLU performance vs SOTA (Gemini, Claude, GPT)
- Can cite AMC/AIME performance (math competitions)
- Benchmark integrity documented and verifiable

---

## Risk Mitigation

### Risk 1: MMLU Performance Lower Than Expected
- **Mitigation:** MMLU is harder than synthetic LHE (expected)
- **Action:** Document learning trajectory (expect improvement with multi-modal navigation)
- **Backup:** Still have ARC 6% as core validation

### Risk 2: Dataset Format Incompatibilities
- **Mitigation:** Test converters on small samples first
- **Action:** Add format validation in ingestion scripts
- **Backup:** Keep synthetic fallbacks for development (clearly flagged)

### Risk 3: Parallel Operations Break GPU Context
- **Mitigation:** Only parallelize CPU-bound tasks (preprocessing, ingestion)
- **Action:** Keep GPU operations (PTX, Knowledgeverse) sequential
- **Backup:** Run serially if issues arise (slower but safe)

---

## Next Steps (Immediate)

**For Claude (NOW):**
1. ✅ Create `benchmarks/mmlu.py`
2. ✅ Create `knowledge3d/knowledgeverse/chat_specialist.py`
3. ✅ Test MMLU benchmark on small sample (10 questions)

**For Codex (PARALLEL):**
1. ✅ Unzip and explore AMC-AIME data
2. ✅ Create `scripts/prepare_amc_dataset.py`
3. ✅ Create `scripts/prepare_aime_dataset.py`
4. ✅ Create `scripts/ingest_audio_phonemes.py`
5. ✅ Create `scripts/ingest_3d_geometry.py`

**Coordination:**
- Claude codes architecture (benchmarks, specialists)
- Codex prepares datasets (converters, ingestion)
- Both validate together (Day 3 full run)

---

**Ready to proceed?** 🚀

**For Daniel:** Review this plan and approve. Then:
1. Claude starts on MMLU + Chat Specialist (today)
2. Codex starts on dataset preparation (parallel)
3. Both sync on Day 3 for validation

**This fixes the benchmark integrity issue AND advances the architecture!** 💪
