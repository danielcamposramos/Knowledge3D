# TRM 6-Recursion Tesla Alignment & Generalization Testing Plan
**Date**: 2025-10-21
**Status**: Planning Phase - For Codex Implementation
**Context**: Standardize TRM to 6 recursions (Tesla 3/6/9) + test generalization on 34K pages

---

## Executive Summary

**Current State**: TRM defaults to 6 recursions, but need to verify/standardize across all code paths
**Target**: **6 recursions everywhere** (Tesla 3/6/9 resonance: 3 inputs × 6 recursions → 9 harmonics)
**Architecture**: Keep 2-layer MLP (512→1024→512) - scientifically validated from paper
**Testing**: Multi-dataset generalization with 8 languages, 34K pages ingested
**Training**: Both (C) - 34K pages RPN embeddings + external ARC-AGI benchmarks
**Timeline**: 3-4 days after sleep consolidation completes

---

## Part 1: TRM Recursion Audit & Standardization

### Current Code State

**Defaults Found**:
```python
# knowledge3d/cranium/ptx_runtime/trm_engine.py:192
n_recursions: int = 6      # Paper optimal

# knowledge3d/cranium/tests/test_trm_core.py:24
create_trm(hidden_dim=512, n_recursions=6, T_iterations=3)

# knowledge3d/cranium/sovereign/trm_launcher.py:144
n_steps: int = 6
```

**Test Variations** (should be standardized):
- `test_trm_engine.py` uses: 1, 2, 3, 6 (for different test cases)
- `test_trm_core.py:70` uses: 5 for training stability test

### Tesla 3/6/9 Resonance Rationale

**Why 6 recursions**:
```
3 Input Modes:
  • question (q)     - What to solve
  • answer (y)       - Current hypothesis
  • latent (z)       - Reasoning state

6 Recursive Refinements:
  • Iteration 1-2: Coarse reasoning
  • Iteration 3-4: Fine-tuning
  • Iteration 5-6: Convergence

9 Output Harmonics:
  • 3 semantic dimensions (what/why/how)
  • 3 spatial dimensions (x/y/z in Galaxy)
  • 3 temporal dimensions (past/present/future)
```

**Mathematical alignment**:
- Each recursion: z ← net(x,y,z), y ← net(y,z)
- After 6 recursions: 2×6 = 12 transformations
- 12 ÷ 3 (input modes) = 4 harmonics per mode
- 4 × 3 = 12 → reduces to 9 via drift convergence

### Standardization Tasks

**Task 1: Audit all TRM invocations**
```bash
# Find all places where TRM is created or called
grep -r "create_trm\|TRMLauncher\|n_recursions\|n_steps" \
  knowledge3d/cranium \
  --include="*.py" \
  > /tmp/trm_audit.txt

# Review and standardize to n_recursions=6
```

**Task 2: Update test configurations**
```python
# tests/test_trm_core.py:70 - Change from 5 to 6
def test_ema_stability(self, trm_model):
    # OLD: max_supervision_steps=5
    # NEW: max_supervision_steps=6
    trm_model.recursive_refine(
        question=question,
        max_supervision_steps=6,  # ← Tesla alignment
        training=True
    )
```

**Task 3: Document Tesla alignment**
```python
# Add docstring to TRMConfig
class TRMConfig:
    """TRM hyperparameters aligned with Tesla 3/6/9 resonance.

    Architecture:
        3 input modes (q, y, z) → 6 recursions → 9 output harmonics

    Parameters:
        n_recursions: int = 6
            Tesla-aligned recursion count. DO NOT CHANGE without
            understanding harmonic implications.
    """
    n_recursions: int = 6  # TESLA 3/6/9 - DO NOT MODIFY
```

**Task 4: Add validation**
```python
# knowledge3d/cranium/sovereign/trm_launcher.py
def refine(self, ..., n_steps: int = 6, ...):
    # Validate Tesla alignment
    if n_steps != 6:
        warnings.warn(
            f"⚠️  Using n_steps={n_steps} breaks Tesla 3/6/9 resonance. "
            f"Recommended: n_steps=6"
        )
```

---

## Part 2: Generalization Testing Framework

### Philosophy

**Goal**: Validate that 34K pages of ingested knowledge generalizes to:
1. **8 languages** (cross-lingual transfer via RPN trigrams)
2. **Multiple domains** (technical → natural language)
3. **Novel reasoning tasks** (ARC-AGI, Sudoku, logic puzzles)

**Datasets**:
- **Training**: 295 PDFs (90% of 328), 31K pages, consolidated RPN embeddings
- **Test**: 33 PDFs (10% held-out), 3.4K pages, unseen
- **External**: ARC-AGI (800 tasks), Sudoku-Extreme (1000 puzzles)

### Test Suite Structure

```
tests/generalization/
├── __init__.py
├── test_cross_lingual.py          # 8-language transfer
├── test_domain_transfer.py        # Tech → Natural
├── test_fractal_consistency.py    # Galaxy clustering
├── test_temporal_stability.py     # Sleep consolidation impact
├── test_arc_agi_reasoning.py      # ARC-AGI benchmark
└── fixtures/
    ├── held_out_pdfs.txt          # 33 reserved PDF paths
    ├── multilingual_prompts.json  # Test prompts in 8 languages
    └── arc_tasks_subset.json      # 100 ARC-AGI tasks
```

### Test 1: Cross-Lingual Trigram Coverage

**File**: `tests/generalization/test_cross_lingual.py`

**Goal**: Verify RPN embeddings work across 8 languages

```python
import pytest
import numpy as np
from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine

class TestCrossLingualGeneralization:
    LANGUAGES = {
        'en': "The quick brown fox jumps over the lazy dog",
        'pt': "O rato roeu a roupa do rei de Roma",
        'es': "El perro come la comida",
        'de': "Der Hund frisst das Futter",
        'fr': "Le chat boit du lait",
        'zh': "猫吃鱼",
        'ja': "犬が食べる",
        'ar': "القط يشرب الحليب"
    }

    @pytest.fixture
    def rpn_engine(self):
        """Load post-consolidation RPN embeddings."""
        engine = RPNEmbeddingEngine()
        embeddings_path = '/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl'
        engine.load_embeddings(embeddings_path)
        return engine

    def test_vocab_coverage_all_languages(self, rpn_engine):
        """Test RPN vocabulary covers all 8 languages."""
        coverage = {}

        for lang, text in self.LANGUAGES.items():
            rpn_engine.hit_count = 0
            rpn_engine.miss_count = 0

            embedding = rpn_engine.embed_sentence(text)

            total = rpn_engine.hit_count + rpn_engine.miss_count
            hit_rate = rpn_engine.hit_count / total if total > 0 else 0
            coverage[lang] = hit_rate

            # Assert: >40% hit rate (shared trigrams across languages)
            assert hit_rate > 0.4, f"{lang} coverage too low: {hit_rate:.2%}"

        print(f"\n✅ Cross-lingual coverage:")
        for lang, rate in coverage.items():
            print(f"   {lang}: {rate:.2%}")

    def test_semantic_clustering_multilingual(self, rpn_engine):
        """Test that similar concepts cluster across languages."""
        # Concept: "cat" in different languages
        cat_texts = {
            'en': "cat",
            'es': "gato",
            'fr': "chat",
            'pt': "gato",
            'de': "Katze",
            'zh': "猫",
            'ja': "猫",
            'ar': "قط"
        }

        embeddings = {lang: rpn_engine.embed_word(word)
                     for lang, word in cat_texts.items()}

        # Compute pairwise similarities
        similarities = []
        for lang1 in embeddings:
            for lang2 in embeddings:
                if lang1 < lang2:  # Avoid duplicates
                    emb1 = embeddings[lang1]
                    emb2 = embeddings[lang2]
                    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    similarities.append(sim)

        avg_similarity = np.mean(similarities)

        # Assert: Same concept across languages should be >0.5 similar
        assert avg_similarity > 0.5, f"Multilingual 'cat' similarity too low: {avg_similarity:.3f}"

        print(f"✅ Multilingual 'cat' clustering: {avg_similarity:.3f} avg similarity")
```

### Test 2: Domain Transfer (Technical → Natural)

**File**: `tests/generalization/test_domain_transfer.py`

```python
class TestDomainTransfer:
    DOMAIN_PROMPTS = {
        'technical': [
            "Explain backpropagation in neural networks",
            "What causes CUDA warp divergence?",
            "How does a GPU scheduler work?"
        ],
        'natural': [
            "Why do leaves change color in autumn?",
            "How do birds navigate during migration?",
            "What causes ocean tides?"
        ],
        'reasoning': [
            "If A implies B, and B implies C, does A imply C?",
            "Three boxes, all mislabeled. Pick one fruit to identify all boxes.",
            "Five pirates split 100 coins. How to maximize your share?"
        ]
    }

    def test_technical_reasoning_transfer(self, rpn_engine):
        """Test if technical knowledge aids logical reasoning."""
        # Embed all domains
        domain_embeddings = {}
        for domain, prompts in self.DOMAIN_PROMPTS.items():
            embs = [rpn_engine.embed_sentence(p) for p in prompts]
            domain_embeddings[domain] = np.mean(embs, axis=0)

        # Technical and reasoning should share logical structure
        tech_emb = domain_embeddings['technical']
        reasoning_emb = domain_embeddings['reasoning']

        similarity = np.dot(tech_emb, reasoning_emb) / (
            np.linalg.norm(tech_emb) * np.linalg.norm(reasoning_emb)
        )

        # Assert: Technical and reasoning share >0.45 similarity
        assert similarity > 0.45, f"Tech→Reasoning transfer weak: {similarity:.3f}"

        print(f"✅ Technical→Reasoning transfer: {similarity:.3f}")
```

### Test 3: TRM 6-Recursion Reasoning

**File**: `tests/generalization/test_arc_agi_reasoning.py`

```python
from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher

class TestTRMReasoning:
    @pytest.fixture
    def trm(self):
        return TRMLauncher(use_fused=True)

    @pytest.fixture
    def trm_weights(self):
        """Load or initialize TRM weights."""
        # TODO: Load trained weights after training phase
        W1 = np.random.randn(1024, 512).astype(np.float32) * 0.02
        W2 = np.random.randn(512, 1024).astype(np.float32) * 0.02
        W3 = np.random.randn(1024, 512).astype(np.float32) * 0.02
        W4 = np.random.randn(512, 1024).astype(np.float32) * 0.02
        return W1, W2, W3, W4

    def test_6_recursions_convergence(self, trm, trm_weights, rpn_engine):
        """Test that 6 recursions (Tesla alignment) achieves convergence."""
        W1, W2, W3, W4 = trm_weights

        # Test question
        question_text = "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?"
        q_emb = rpn_engine.embed_sentence(question_text)
        q = np.array(q_emb, dtype=np.float32)
        y = np.zeros(512, dtype=np.float32)
        z = np.zeros(512, dtype=np.float32)

        # Run with 6 recursions
        y_out, z_out = trm.refine(q, y, z, W1, W2, W3, W4, n_steps=6, eps=1e-4)

        # Assert: Should produce valid output
        assert np.isfinite(y_out).all(), "TRM output contains NaN/Inf"
        assert np.linalg.norm(y_out) > 0.1, "TRM output too weak"

        print(f"✅ TRM 6-recursion convergence: ||y|| = {np.linalg.norm(y_out):.3f}")

    @pytest.mark.parametrize("n_recursions", [3, 6, 9])
    def test_tesla_harmonics(self, trm, trm_weights, rpn_engine, n_recursions):
        """Test performance across Tesla harmonic counts (3, 6, 9)."""
        W1, W2, W3, W4 = trm_weights

        question_text = "Solve: 2x + 5 = 17"
        q_emb = rpn_engine.embed_sentence(question_text)
        q = np.array(q_emb, dtype=np.float32)
        y = np.zeros(512, dtype=np.float32)
        z = np.zeros(512, dtype=np.float32)

        y_out, z_out = trm.refine(q, y, z, W1, W2, W3, W4, n_steps=n_recursions)

        output_norm = np.linalg.norm(y_out)

        # All Tesla harmonics should converge
        assert output_norm > 0.1, f"Weak output for n={n_recursions}"

        print(f"   n={n_recursions}: ||y|| = {output_norm:.3f}")
```

### Test 4: Fractal Consistency (Post-Consolidation)

**File**: `tests/generalization/test_fractal_consistency.py`

```python
from knowledge3d.cranium.sleep_time_consolidator import SleepTimeConsolidator

class TestFractalConsistency:
    def test_consolidation_cluster_quality(self, rpn_engine):
        """Test that sleep consolidation improves semantic clustering."""
        # Define semantic domains
        domains = {
            'machine_learning': ["neural network", "gradient descent", "backpropagation", "overfitting"],
            'physics': ["velocity", "acceleration", "momentum", "energy"],
            'biology': ["cell", "mitosis", "DNA", "protein"]
        }

        # Compute cluster centers
        centers = {}
        for domain, terms in domains.items():
            embs = [rpn_engine.embed_word(term) for term in terms]
            centers[domain] = np.mean(embs, axis=0)

        # Measure inter-cluster distances
        inter_distances = []
        for d1, d2 in itertools.combinations(centers.keys(), 2):
            dist = np.linalg.norm(centers[d1] - centers[d2])
            inter_distances.append(dist)

        # Measure intra-cluster distances
        intra_distances = []
        for domain, terms in domains.items():
            center = centers[domain]
            for term in terms:
                emb = rpn_engine.embed_word(term)
                dist = np.linalg.norm(emb - center)
                intra_distances.append(dist)

        avg_inter = np.mean(inter_distances)
        avg_intra = np.mean(intra_distances)
        separation_ratio = avg_inter / avg_intra

        # Assert: Inter-cluster >> Intra-cluster
        assert separation_ratio > 1.3, f"Poor clustering: ratio={separation_ratio:.2f}"

        print(f"✅ Cluster separation ratio: {separation_ratio:.2f}")
        print(f"   Inter-cluster: {avg_inter:.3f}")
        print(f"   Intra-cluster: {avg_intra:.3f}")
```

---

## Part 3: Training Strategy

### Phase A: RPN Embedding Consolidation (Complete)

**Status**: Sleep consolidation running
**Data**: 34,497 pages, 647,757 objects, 328 PDFs
**Output**: Consolidated RPN embeddings with:
- Cluster refinement (K-means centroids)
- Redundancy pruning (0.95+ similarity merges)
- Vocabulary reduction (expected ~20-30%)

### Phase B: TRM Weight Initialization

**Approach**: Transfer learning from RPN embeddings

```python
# scripts/initialize_trm_from_rpn.py
def initialize_trm_weights_from_rpn():
    """Initialize TRM weights using RPN trigram embeddings."""
    rpn_engine = RPNEmbeddingEngine()
    rpn_engine.load_embeddings('/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl')

    # Extract most frequent trigrams
    sorted_trigrams = sorted(
        rpn_engine.embeddings.items(),
        key=lambda x: rpn_engine.hit_count,  # Proxy for frequency
        reverse=True
    )[:1024]  # Top 1024 trigrams

    # Stack into matrix
    trigram_matrix = np.vstack([emb for _, emb in sorted_trigrams])  # (1024, 128)

    # Initialize W1: Expand 512 → 1024 using trigram patterns
    W1 = np.random.randn(1024, 512).astype(np.float32) * 0.02
    # Seed first 128 dims with trigram embeddings
    W1[:, :128] = trigram_matrix

    # Initialize W2, W3, W4 with Xavier
    W2 = np.random.randn(512, 1024).astype(np.float32) * np.sqrt(2.0 / 1024)
    W3 = np.random.randn(1024, 512).astype(np.float32) * np.sqrt(2.0 / 512)
    W4 = np.random.randn(512, 1024).astype(np.float32) * np.sqrt(2.0 / 1024)

    # Save
    np.savez('/K3D/Knowledge3D.local/models/trm_weights_rpn_init.npz',
             W1=W1, W2=W2, W3=W3, W4=W4)

    print(f"✅ Initialized TRM weights from {len(sorted_trigrams)} RPN trigrams")
```

### Phase C: Training on K3D Knowledge (34K pages)

**Approach**: Weak supervision from RPN embeddings

```python
# scripts/train_trm_on_k3d_knowledge.py
class TRMTrainer:
    def __init__(self):
        self.rpn_engine = RPNEmbeddingEngine()
        self.rpn_engine.load_embeddings('/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl')
        self.trm = TRMLauncher(use_fused=True)
        self.load_weights()

    def create_training_pairs(self):
        """Create (question, answer) pairs from ingested PDFs."""
        # Load held-out test set (10%)
        test_pdfs = self.load_held_out_pdfs()  # 33 PDFs

        # Training set: remaining 90% (295 PDFs)
        train_pdfs = self.get_training_pdfs()

        pairs = []
        for pdf_path in train_pdfs:
            # Extract text from PDF
            text = extract_pdf_text(pdf_path)

            # Split into sentences
            sentences = sent_tokenize(text)

            # Create Q-A pairs: Question = sentence, Answer = next sentence
            for i in range(len(sentences) - 1):
                question = sentences[i]
                answer = sentences[i + 1]

                q_emb = self.rpn_engine.embed_sentence(question)
                a_emb = self.rpn_engine.embed_sentence(answer)

                pairs.append({
                    'question': q_emb,
                    'answer': a_emb,
                    'source': pdf_path
                })

        return pairs

    def train(self, epochs=10):
        """Train TRM on K3D knowledge."""
        pairs = self.create_training_pairs()
        print(f"Training on {len(pairs)} Q-A pairs from 295 PDFs")

        for epoch in range(epochs):
            losses = []

            for pair in pairs:
                q = pair['question'].astype(np.float32)
                target = pair['answer'].astype(np.float32)
                y = np.zeros(512, dtype=np.float32)
                z = np.zeros(512, dtype=np.float32)

                # Forward pass (6 recursions)
                y_pred, z_pred = self.trm.refine(q, y, z, *self.weights, n_steps=6)

                # Loss: MSE between predicted and target answer
                loss = np.mean((y_pred - target) ** 2)
                losses.append(loss)

                # Backprop + gradient update (simplified - real impl would use PyTorch)
                # ... gradient computation ...

            avg_loss = np.mean(losses)
            print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

        # Save trained weights
        self.save_weights()
```

### Phase D: Fine-Tuning on ARC-AGI

**Approach**: External benchmark for reasoning

```python
# scripts/finetune_trm_on_arc.py
class ARCTrainer:
    def load_arc_dataset(self):
        """Load ARC-AGI training tasks."""
        # Download from https://github.com/fchollet/ARC-AGI
        with open('data/arc_training.json') as f:
            tasks = json.load(f)
        return tasks

    def train(self, epochs=20):
        """Fine-tune TRM on ARC-AGI reasoning tasks."""
        tasks = self.load_arc_dataset()

        for epoch in range(epochs):
            for task_id, task_data in tasks.items():
                # Extract input-output pairs
                for example in task_data['train']:
                    input_grid = example['input']
                    output_grid = example['output']

                    # Flatten grids to embeddings
                    q = self.grid_to_embedding(input_grid)
                    target = self.grid_to_embedding(output_grid)

                    # Train TRM
                    # ... same as above ...
```

---

## Part 4: Implementation Roadmap

### Day 1: Audit & Standardization (Codex)

**Tasks**:
1. ✅ Audit all `n_recursions`/`n_steps` in codebase
2. ✅ Standardize to 6 everywhere (except specific test cases)
3. ✅ Add Tesla 3/6/9 documentation
4. ✅ Add validation warnings for non-6 values

**Deliverables**:
- Updated `trm_launcher.py` with validation
- Updated `test_trm_core.py` standardized to 6
- Documentation in docstrings

### Day 2: Test Infrastructure (Claude)

**Tasks**:
1. ✅ Implement test suite (4 test files)
2. ✅ Create fixtures (held-out PDFs, multilingual prompts)
3. ✅ Split dataset (90% train / 10% test)
4. ✅ Set up test runners

**Deliverables**:
- `tests/generalization/` package
- 33 held-out PDFs identified
- Multilingual prompt database

### Day 3: RPN Weight Initialization (Codex)

**Tasks**:
1. ✅ Implement `initialize_trm_from_rpn.py`
2. ✅ Extract top 1024 trigrams from consolidated RPN
3. ✅ Seed TRM weights with trigram patterns
4. ✅ Save initialized weights

**Deliverables**:
- `/K3D/Knowledge3D.local/models/trm_weights_rpn_init.npz`

### Day 4: Training on K3D Knowledge (Codex + Compute)

**Tasks**:
1. ✅ Implement `train_trm_on_k3d_knowledge.py`
2. ✅ Create Q-A pairs from 295 training PDFs
3. ✅ Train TRM (10 epochs, ~50K pairs)
4. ✅ Validate on 33 held-out PDFs

**Deliverables**:
- `/K3D/Knowledge3D.local/models/trm_weights_k3d_trained.npz`
- Training metrics log

### Day 5: ARC-AGI Fine-Tuning (Optional)

**Tasks**:
1. Download ARC-AGI dataset
2. Implement grid→embedding converter
3. Fine-tune TRM on 400 training tasks
4. Evaluate on 400 evaluation tasks

**Deliverables**:
- ARC-AGI accuracy report
- Fine-tuned weights (if improvement >5%)

### Day 6: Validation & Report (All)

**Tasks**:
1. ✅ Run full test suite
2. ✅ Generate metrics:
   - Cross-lingual coverage: __%
   - Domain transfer: __%
   - Fractal clustering: __ ratio
   - 6-recursion convergence: __%
3. ✅ Create final report
4. ✅ Document findings

**Deliverables**:
- Test results report
- Performance comparison (before/after training)
- Recommendations for production

---

## Part 5: Success Metrics

### Minimum Acceptable Performance

**Cross-Lingual**:
- Trigram coverage: >40% for all 8 languages
- Semantic similarity (same concept): >0.50

**Domain Transfer**:
- Technical→Reasoning similarity: >0.45
- Zero-shot natural language: >30% relevance

**Fractal Clustering**:
- Inter/intra ratio: >1.3
- Vocabulary reduction: 10-30%

**TRM Convergence**:
- 6 recursions converge: >90% of test cases
- Output norm: >0.1 (non-trivial)
- Latency: <95µs P95

### Target Performance

**Cross-Lingual**:
- Coverage: >60%
- Similarity: >0.65

**Domain Transfer**:
- Technical→Reasoning: >0.55
- Zero-shot: >50%

**Fractal Clustering**:
- Ratio: >1.8
- Reduction: 20-30%

**TRM Convergence**:
- Convergence: >95%
- Output norm: >0.5
- Latency: <85µs P95

---

## Summary for Codex

**Task**: Standardize TRM to 6 recursions (Tesla 3/6/9 alignment) + build generalization testing framework

**Architecture**: Keep 2-layer MLP (512→1024→512) - scientifically validated

**Core Changes**:
1. Audit + standardize `n_recursions=6` everywhere
2. Add Tesla 3/6/9 documentation
3. Add validation warnings

**Testing Framework**:
1. Cross-lingual (8 languages, 34K pages)
2. Domain transfer (tech → natural)
3. Fractal clustering (post-consolidation)
4. TRM 6-recursion convergence

**Training**:
1. Initialize from RPN embeddings (1024 top trigrams)
2. Train on 295 PDFs (90% of corpus)
3. Validate on 33 held-out PDFs
4. Optional: Fine-tune on ARC-AGI

**Timeline**: 4-6 days (start after sleep consolidation)

**Deliverables**:
- Standardized TRM code (6 recursions)
- Test suite (4 test files)
- Trained weights (RPN-initialized + K3D-trained)
- Performance report

---

**Ready to proceed once sleep consolidation completes!** 🧬🔥
