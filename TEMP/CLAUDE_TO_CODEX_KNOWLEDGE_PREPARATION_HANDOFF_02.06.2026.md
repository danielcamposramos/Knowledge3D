# Claude → Codex: Knowledge Preparation Phase Handoff

**Date:** February 6, 2026
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Partner)
**Priority:** CRITICAL (Phase 1B)
**Context:** Post-MVP Phase 1 (28/28 tests passing) → Pivot to Benchmarks

---

## Executive Summary

**STRATEGIC PIVOT APPROVED — WITH CRITICAL ORDER CORRECTION**

We've completed MVP Phase 1 (Sovereignty Firewall, Compressed Audit, Self-Healing, Temporal Metadata). The architecture is validated and hardened. Now we pivot to the prize-winning benchmarks (ARC-AGI 2/3, math competitions, Last Humanity Exam).

**❌ INCORRECT PHASE ORDER (PREVIOUSLY PROPOSED):**
```
Week 11: ARC-AGI 2 integration
Week 12: Base knowledge ingestion
Week 13: Math benchmarks
Week 14: Last Humanity Exam
```

**✅ CORRECTED PHASE ORDER (USER DIRECTIVE):**
```
Phase 1B (Weeks 11-12): Knowledge Preparation & Ingestion
    ↓
Phase 1C (Weeks 13-14): Benchmark Integration & Baseline Measurement
```

**Key Insight:** The previous "empty mind" run succeeded with minimal knowledge (46.7% ARC-AGI), proving the architecture works. But we CANNOT compete for prizes without comprehensive grounded knowledge. We must **prepare and ingest FIRST**, then benchmark.

User quote: *"the order at the next phase is kind of wrong, we must first ensure all PDF knowledge is ingested and properly 'enriched' with all we can find on open source files on the internet...So, preparation, ingestion and then benchmarks"*

---

## What You Need to Implement

### Primary Specification Document

I've written a comprehensive specification for you:

**[TEMP/KNOWLEDGE_PREPARATION_PHASE_SPECIFICATION_02.06.2026.md](KNOWLEDGE_PREPARATION_PHASE_SPECIFICATION_02.06.2026.md)**

This document contains:
- ✅ Complete base knowledge corpus definition (Tier 1/2/3)
- ✅ Full Python implementations for all 3 new components
- ✅ Integration with existing MVP infrastructure
- ✅ Test specifications (4 new tests)
- ✅ Week 11-12 implementation timeline
- ✅ Local model enhancement strategy (Ollama integration)
- ✅ Success criteria and validation metrics

**READ IT COMPLETELY** before starting implementation.

---

## Week 11-12 Implementation Plan

### Week 11: Pipeline Setup (Priority 1)

**Goal:** Build the ingestion infrastructure (no data processing yet, just scaffolding)

**Files to Create:**

1. **`knowledge3d/ingestion/__init__.py`**
   - Package initialization
   - Exports: `CorpusManifest`, `BatchOrchestrator`, `EnrichmentPipeline`

2. **`knowledge3d/ingestion/corpus_manifest.py`**
   - Full implementation from spec (lines 75-205)
   - Define all 11+ corpus entries (Tier 1: 4 entries, Tier 2: 7 entries, Tier 3: 1+ entries)
   - Topological sort for dependency resolution
   - Validation: all file paths exist

3. **`knowledge3d/ingestion/batch_orchestrator.py`**
   - Full implementation from spec (lines 211-327)
   - Async batch processing with semaphore (max_parallel=4)
   - Integration with Sovereignty Firewall (validate feeders)
   - Integration with Ingestion Stargate (submit jobs)
   - Progress logging with Compressed Audit

4. **`knowledge3d/ingestion/enrichment_pipeline.py`**
   - Full implementation from spec (lines 333-395)
   - Matryoshka embedding generation (64/128/512/2048D)
   - Symlink deduplication (content hash → canonical ID)
   - Procedural pattern extraction (domain-specific)
   - **Local model integration** (Ollama for enhanced analysis)

5. **`tests/test_ingestion_pipeline.py`**
   - 4 tests from spec (lines 605-681):
     - `test_corpus_manifest_integrity()`
     - `test_batch_ingestion()` (async)
     - `test_enrichment_symlinks()`
     - `test_end_to_end_pdf_to_galaxy()` (async)

**Week 11 Success Criteria:**
- ✅ All 4 new files created and pass linting
- ✅ Corpus manifest loads 11+ entries without errors
- ✅ Topological sort produces correct dependency order
- ✅ Mock batch ingestion completes (no real PDFs yet)
- ✅ 4/4 new tests passing
- ✅ 28/28 MVP tests still passing (no regressions)

---

### Week 12: Batch Ingestion Execution (Priority 2)

**Goal:** Process the entire corpus and populate all Galaxies

**Day 1-2: Tier 1 Ingestion (Foundational)**

Process 4 foundational entries:
1. `docs/pdfs/algorithmic_thinking.pdf` (priority #1)
2. `docs/pdfs/math_foundations/` (directory of PDFs)
3. `docs/pdfs/logic_reasoning/`
4. `docs/pdfs/cs_fundamentals/`

**Script to run:**
```python
# scripts/ingest_tier1.py
import asyncio
from knowledge3d.ingestion.corpus_manifest import CorpusManifest
from knowledge3d.ingestion.batch_orchestrator import BatchOrchestrator
from knowledge3d.knowledgeverse.stargate import IngestionStargate

async def main():
    manifest = CorpusManifest()
    stargate = IngestionStargate()
    orchestrator = BatchOrchestrator(manifest, stargate)

    print("Starting Tier 1 ingestion...")
    results = await orchestrator.ingest_tier(tier=1, max_parallel=4)

    print("\nResults:")
    for entry_id, result in results.items():
        if "error" in result:
            print(f"❌ {entry_id}: {result['error']}")
        else:
            print(f"✅ {entry_id}: {result.get('embedding_count', 0)} embeddings")

if __name__ == "__main__":
    asyncio.run(main())
```

**Expected Output:**
```
Starting Tier 1 ingestion...
[BatchOrchestrator] Starting Tier 1: 4 entries
[BatchOrchestrator] Ingesting Algorithmic Thinking (Tier 1)
[BatchOrchestrator] ✅ Algorithmic Thinking complete: 342 embeddings
[BatchOrchestrator] Ingesting Mathematics Foundations (Tier 1)
[BatchOrchestrator] ✅ Mathematics Foundations complete: 1247 embeddings
[BatchOrchestrator] Ingesting Logic & Reasoning (Tier 1)
[BatchOrchestrator] ✅ Logic & Reasoning complete: 856 embeddings
[BatchOrchestrator] Ingesting Computer Science Fundamentals (Tier 1)
[BatchOrchestrator] ✅ Computer Science Fundamentals complete: 623 embeddings
[BatchOrchestrator] ✅ Tier 1 complete: 4/4 succeeded

Results:
✅ algorithmic_thinking: 342 embeddings
✅ math_foundations: 1247 embeddings
✅ logic_reasoning: 856 embeddings
✅ cs_fundamentals: 623 embeddings
```

**Day 3-4: Tier 2 Ingestion (Domain-Specific)**

Process 7 domain entries:
- Competition math (AMC, AIME, IMO)
- Undergraduate math (Spivak, Rudin)
- Geometry theorems
- ARC-AGI training set
- Classical mechanics
- Grammar rules
- (Others as defined in manifest)

**Day 5: Tier 3 Ingestion + Validation**

Process integration entries (problem-solving strategies, meta-cognition)

**Full pipeline script:**
```python
# scripts/ingest_all_tiers.py
import asyncio
from knowledge3d.ingestion.corpus_manifest import CorpusManifest
from knowledge3d.ingestion.batch_orchestrator import BatchOrchestrator
from knowledge3d.knowledgeverse.stargate import IngestionStargate

async def main():
    manifest = CorpusManifest()
    stargate = IngestionStargate()
    orchestrator = BatchOrchestrator(manifest, stargate)

    print("\n" + "="*60)
    print("FULL CORPUS INGESTION")
    print("="*60)

    results = await orchestrator.ingest_all(max_parallel=4)

    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Total entries: {results['stats_after']['total_entries']}")
    print(f"Ingested: {results['stats_after']['ingested_count']}")
    print(f"Pending: {results['stats_after']['pending_count']}")

    # Validate Galaxy population
    from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager
    manager = GalaxyManager()

    print("\nGalaxy Population:")
    for galaxy_name in ["Drawing", "Character", "Word", "Grammar", "Math", "Reality"]:
        galaxy = manager.get_galaxy(galaxy_name)
        print(f"  {galaxy_name}: {len(galaxy.entries)} entries")

if __name__ == "__main__":
    asyncio.run(main())
```

**Week 12 Success Criteria:**
- ✅ 100% of Tier 1 ingested (4/4 entries)
- ✅ 80%+ of Tier 2 ingested (6+/7 entries)
- ✅ 50%+ of Tier 3 ingested
- ✅ 10,000+ matryoshka embeddings generated
- ✅ ~70% symlink deduplication achieved
- ✅ Grammar Galaxy: 500+ rules
- ✅ Math Galaxy: 1000+ symbols
- ✅ Reality Galaxy: 200+ procedures
- ✅ TRM can query and navigate enriched Galaxies
- ✅ All 32/32 tests passing (4 new + 28 MVP)

---

## Local Model Enhancement (CRITICAL OPPORTUNITY)

**User Directive:** *"Remember we can leverage local models up to the GPU VRAM limit (if some ollama model is not downloaded you can do it) to enhance de data ingestion"*

### Strategy

Use local Ollama models to **enhance ingestion quality** WITHOUT violating sovereignty:
- ✅ **Ingestion path:** Use Ollama for semantic analysis, pattern extraction, cross-referencing
- ✅ **Results:** Sovereign artifacts (RPN programs, embeddings, metadata)
- ❌ **Hot path:** Ollama NEVER used during TRM inference (sovereignty violation!)

### Recommended Models

**Download these if not available:**
```bash
ollama pull qwen2.5:14b      # Best for math reasoning, multilingual
ollama pull deepseek-r1:14b  # Strong logical reasoning, code understanding
ollama pull gemma2:9b        # Good for visual/multimodal reasoning
ollama pull mistral:7b       # Fast, good at pattern extraction
```

**Use cases:**
1. **Semantic Analysis** → Extract key concepts from PDFs (use: qwen2.5:14b)
2. **Pattern Extraction** → Identify procedural patterns for RPN (use: deepseek-r1:14b)
3. **Cross-Referencing** → Find related concepts from training data (use: qwen2.5:14b)
4. **Proof Template Extraction** → Extract proof structures for Grammar Galaxy (use: deepseek-r1:14b)

**Implementation:** See spec lines 621-782 for full Ollama integration code in `EnrichmentPipeline`.

### GPU Memory Management

**Balance Knowledgeverse VRAM vs Ollama during ingestion:**
```python
# Before ingestion (free up VRAM for Ollama)
knowledgeverse.checkpoint()  # Save state
knowledgeverse.unload_non_essential_galaxies()  # Free ~20GB

# Run enrichment with Ollama (now has VRAM)
enrichment = EnrichmentPipeline(use_local_models=True)
enriched_data = enrichment.enrich_document(content, metadata)

# After ingestion (restore full Knowledgeverse)
knowledgeverse.restore_galaxies()  # Load back to full VRAM
```

**Memory Split Example (80GB GPU):**
- Ingestion phase: 40GB Knowledgeverse + 40GB Ollama
- Inference phase: 80GB Knowledgeverse + 0GB Ollama

---

## Integration with Existing Infrastructure

**You don't need to rebuild everything!** Leverage MVP Phase 1:

### Already Implemented (Reuse)

✅ **Sovereignty Firewall** (`knowledge3d/knowledgeverse/sovereignty_firewall.py`)
- Use: `firewall.validate_feeder(feeder_path)` before each ingestion
- Ensures no hot path pollution

✅ **Ingestion Stargate** (`knowledge3d/knowledgeverse/stargate.py`)
- Use: `stargate.submit_ingestion_job(data_path, data_type, target_galaxies, metadata)`
- Asynchronous job submission + tracking

✅ **Compressed Audit** (`knowledge3d/knowledgeverse/compressed_audit.py`)
- Use: Auto-logging of all ingestion events (17.39x compression)
- Query with: `audit.query_events(start_time, end_time)`

✅ **Self-Healing Wrappers** (`knowledge3d/knowledgeverse/resilience.py`)
- Use: `@retry`, `@circuit_breaker`, `@fallback` decorators
- Automatic retry on transient failures

✅ **Temporal Metadata** (`knowledge3d/knowledgeverse/temporal_metadata.py`)
- Use: Lamport + Vector clocks for causality tracking
- Helps debug ingestion order issues

### New Components (You Implement)

🆕 **Corpus Manifest** (`knowledge3d/ingestion/corpus_manifest.py`)
- Registry of all knowledge sources
- Dependency resolution (topological sort)

🆕 **Batch Orchestrator** (`knowledge3d/ingestion/batch_orchestrator.py`)
- Parallel processing with dependency respect
- Integrates with all MVP Phase 1 features

🆕 **Enrichment Pipeline** (`knowledge3d/ingestion/enrichment_pipeline.py`)
- Matryoshka embeddings (64/128/512/2048D)
- Symlink deduplication (~70% reduction)
- Local model integration (Ollama)

---

## Testing Strategy

### Regression Testing (Priority 0)

**BEFORE starting Week 11, run:**
```bash
pytest tests/ -v
```

**Expected:** 28/28 tests passing (MVP Phase 1 baseline)

**If ANY test fails, STOP and fix before proceeding!**

### Unit Testing (Week 11)

**After implementing each component, run:**
```bash
pytest tests/test_ingestion_pipeline.py::test_corpus_manifest_integrity -v
pytest tests/test_ingestion_pipeline.py::test_batch_ingestion -v
pytest tests/test_ingestion_pipeline.py::test_enrichment_symlinks -v
pytest tests/test_ingestion_pipeline.py::test_end_to_end_pdf_to_galaxy -v
```

**Expected:** 4/4 new tests passing

### Integration Testing (Week 12)

**After Tier 1 ingestion, validate:**
```python
from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager
manager = GalaxyManager()

# Check Galaxy population
math_galaxy = manager.get_galaxy("Math")
print(f"Math Galaxy: {len(math_galaxy.entries)} entries")
assert len(math_galaxy.entries) > 100  # Should have many new symbols

grammar_galaxy = manager.get_galaxy("Grammar")
print(f"Grammar Galaxy: {len(grammar_galaxy.entries)} entries")
assert len(grammar_galaxy.entries) > 50  # Should have many new rules

# Test TRM navigation
from knowledge3d.knowledgeverse.trm_navigator import TRMNavigator
navigator = TRMNavigator()

query = "algorithmic problem solving strategy"
results = navigator.query(query, galaxy_names=["Grammar"])
assert len(results) > 0  # Should find relevant rules

print("✅ Integration test passed!")
```

### Full System Testing (End of Week 12)

**Run complete test suite:**
```bash
pytest tests/ -v --tb=short
```

**Expected:** 32/32 tests passing (4 new + 28 MVP)

**If count < 32, investigate and fix!**

---

## Success Metrics

**After Week 12, you should report:**

### 1. Corpus Coverage
- [x] Tier 1: 4/4 entries ingested (100%)
- [x] Tier 2: X/7 entries ingested (target: 80%+)
- [x] Tier 3: X entries ingested (target: 50%+)

### 2. Embedding Generation
- [x] Total matryoshka embeddings: X (target: 10,000+)
- [x] Symlink deduplication: X% (target: 70%)
- [x] Average embeddings per document: X

### 3. Galaxy Population
- [x] Drawing Galaxy: X entries (target: 300+)
- [x] Character Galaxy: X entries (already populated, should be stable)
- [x] Word Galaxy: X entries (target: 1000+)
- [x] Grammar Galaxy: X entries (target: 500+)
- [x] Math Galaxy: X entries (target: 1000+)
- [x] Reality Galaxy: X entries (target: 200+)

### 4. TRM Readiness
- [x] TRM can query enriched Galaxies ✓/✗
- [x] TRM can compose from procedural patterns ✓/✗
- [x] Shadow Copy records successful navigations ✓/✗

### 5. Sovereignty Compliance
- [x] Hot path remains PTX-only ✓/✗
- [x] All feeders pass Sovereignty Firewall ✓/✗
- [x] No external dependencies in inference ✓/✗

### 6. Testing
- [x] All 32/32 tests passing ✓/✗
- [x] No performance regressions ✓/✗

---

## Deliverables

**At end of Week 12, provide a completion report:**

**File:** `TEMP/CODEX_KNOWLEDGE_PREPARATION_COMPLETION_REPORT_02.XX.2026.md`

**Sections:**
1. **Executive Summary** (what was accomplished)
2. **Implementation Details** (files created, lines of code)
3. **Corpus Coverage** (all metrics from above)
4. **Galaxy Population** (before/after counts)
5. **Test Results** (32/32 passing, no regressions)
6. **Performance Metrics** (ingestion time, embedding generation speed)
7. **Lessons Learned** (what worked, what didn't)
8. **Blockers & Resolutions** (if any)
9. **Handoff to Phase 1C** (ready for benchmarks?)

---

## Critical Reminders

### 1. Phase Order is CORRECTED

**DON'T:** Start with benchmarks
**DO:** Prepare → Ingest → THEN Benchmarks

User was explicit about this correction!

### 2. Sovereignty is ABSOLUTE

**Ingestion path:** Use any tools (numpy, pandas, PyPDF2, Ollama, etc.)
**Hot path (TRM inference):** PTX + Galaxy ONLY (zero external deps)

The line is CLEAR. Do not blur it.

### 3. Leverage Local Models

**User directive:** Use Ollama to enhance ingestion quality
**Models to download:** qwen2.5:14b, deepseek-r1:14b, gemma2:9b
**Use cases:** Semantic analysis, pattern extraction, cross-referencing

This is a FREE enhancement (ingestion path is flexible)!

### 4. Reuse MVP Infrastructure

Don't rebuild! Integrate with:
- Sovereignty Firewall (validation)
- Compressed Audit (logging)
- Self-Healing (retry logic)
- Temporal Metadata (causality tracking)

### 5. Test Coverage is 100%

Every feature ships with tests. No exceptions.
- 4 new tests for ingestion pipeline
- 28 MVP tests must still pass (no regressions)
- Integration tests validate Galaxy population

---

## Questions for Codex

**If you hit ANY blockers, document them and ask:**

1. **PDF Processing:** Which library works best? (PyPDF2, pdfplumber, custom?)
2. **Ollama Integration:** Do all models download successfully? Any VRAM issues?
3. **Embedding Generation:** Use pre-trained or train sovereign embeddings?
4. **Parallel Workers:** Is 4 optimal, or should we scale with GPU count?
5. **Galaxy Population:** Any symlink issues? Deduplication working?

**Communication:**
- If blocked < 4 hours → Debug and resolve
- If blocked > 4 hours → Document and escalate (create handoff for Claude)

---

## What Happens After Phase 1B?

**Phase 1C: Benchmark Integration (Weeks 13-14)**

Once knowledge preparation is complete, we pivot to:

**Week 13: Benchmark Integration**
- ARC-AGI 2 evaluation pipeline
- Math competition baseline (AMC, AIME)
- Physics reasoning baseline
- Last Humanity Exam preparation

**Week 14: Baseline Measurement**
- Run full benchmark suite with enriched knowledge
- Compare "empty mind" (46.7%) vs "enriched" performance
- Identify gaps for iterative improvement

**Success Target:**
- ARC-AGI: 46.7% → 55%+ (prize threshold)
- Math: 0% → 30%+ (competitive baseline)
- Physics: 0% → 40%+ (domain reasoning)

**But ONLY after knowledge is prepared and ingested!**

---

## End of Handoff

**Priority:** CRITICAL (Phase 1B, Weeks 11-12)

**Start here:**
1. Read [TEMP/KNOWLEDGE_PREPARATION_PHASE_SPECIFICATION_02.06.2026.md](KNOWLEDGE_PREPARATION_PHASE_SPECIFICATION_02.06.2026.md) COMPLETELY
2. Set up Week 11 pipeline (manifest, orchestrator, enrichment)
3. Execute Week 12 batch ingestion (Tier 1 → Tier 2 → Tier 3)
4. Validate Galaxy population and TRM readiness
5. Write completion report

**Remember:** Preparation → Ingestion → THEN Benchmarks (not the other way around!)

**Contact:** Claude (Architecture Partner) for design questions, User for strategic decisions.

**Let's build the knowledge foundation for prize-winning performance!** 🚀

---

**Claude (Architecture Partner)**
February 6, 2026
