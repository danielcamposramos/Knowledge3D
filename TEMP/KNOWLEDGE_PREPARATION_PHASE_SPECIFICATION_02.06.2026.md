# Knowledge Preparation Phase — Base Corpus Ingestion & Enrichment

**Created:** February 6, 2026
**Author:** Claude (Architecture Partner)
**Target:** Codex (Implementation Partner)
**Status:** Phase 1B Specification (Post-MVP Hardening)

---

## Executive Summary

**CRITICAL ORDER CORRECTION:**

The next phase is **NOT** benchmarks first. The correct order is:

```
Phase 1B: Knowledge Preparation & Ingestion (Weeks 11-12)
    ↓
Phase 1C: Benchmark Integration & Baseline (Weeks 13-14)
```

**Rationale:** The previous "empty mind" run succeeded with minimal math/grammar rules (46.7% ARC-AGI), proving the architecture works. Now we need **comprehensive grounded knowledge** before running competitive benchmarks for prizes (ARC-AGI 2/3, math competitions, Last Humanity Exam).

**Key Insight:** OCR problems led us to matryoshka embeddings and procedural drawing/memory. Let's continue drawing inspiration from computer history and game industry to build a computational substrate that treats AI as a first-class user alongside humans.

---

## Phase 1B Goals

**Week 11: Knowledge Corpus Definition & Pipeline Setup**
- Define complete base knowledge corpus (PDFs + open source + foundational knowledge)
- Enhance Ingestion Stargate with batch processing capabilities
- Create corpus manifest with priority tiers
- Validate Sovereignty Firewall for all feeders

**Week 12: Batch Ingestion & Galaxy Population**
- Process all Tier 1 corpus (algorithmic thinking, CS fundamentals)
- Generate matryoshka embeddings (64/128/512/2048D) for all content
- Populate Drawing, Character, Word, Grammar, Math, Reality Galaxies
- Symlink common patterns (avoid duplication)
- Validate end-to-end: Raw PDF → RPN Programs → Galaxy Entries

---

## Base Knowledge Corpus Definition

### Tier 1: Foundational Knowledge (Already Identified)

**PDFs in Repository:**
1. **Algorithmic Thinking** (`docs/pdfs/algorithmic_thinking.pdf` — priority #1)
2. **Mathematics Foundations** (algebra, calculus, geometry, number theory)
3. **Logic & Reasoning** (propositional logic, predicate logic, proof techniques)
4. **Computer Science Fundamentals** (data structures, algorithms, complexity)

**Open Source Content (From Training Data):**
- Mathematical proofs (Lean, Coq theorem prover libraries)
- Physics simulations (classical mechanics, E&M, thermodynamics)
- Visual reasoning datasets (ARC-AGI training set, synthetic geometry)
- Grammar rules (natural language processing corpora)
- Programming patterns (open source algorithm implementations)

### Tier 2: Domain-Specific Extensions

**Math Domain:**
- Competition math problems (AMC, AIME, IMO archives)
- Undergraduate textbooks (Spivak, Rudin, Knuth TAOCP)
- Proof patterns (induction, contradiction, construction)

**Visual Domain:**
- Geometry theorems (Euclidean, projective, algebraic)
- Spatial reasoning patterns (rotations, reflections, symmetries)
- Procedural generation techniques (L-systems, fractals, cellular automata)

**Physics Domain:**
- Classical mechanics (Lagrangian, Hamiltonian formulations)
- Electromagnetism (Maxwell equations, boundary conditions)
- Thermodynamics (laws, statistical mechanics basics)
- Quantum mechanics (Schrödinger equation, observables)

**Language Domain:**
- Grammar transformation rules (context-free grammars, dependency parsing)
- Semantic networks (WordNet-style relations)
- Pragmatics (context-dependent meaning)

### Tier 3: Cross-Domain Integration

**Problem-Solving Strategies:**
- Polya's "How to Solve It" patterns
- Heuristic search techniques (A*, beam search, MCTS)
- Constraint satisfaction (backtracking, arc consistency)

**Meta-Cognition:**
- Self-monitoring patterns (confidence estimation, uncertainty quantification)
- Analogical reasoning (structure mapping, case-based reasoning)
- Transfer learning patterns (domain adaptation, curriculum learning)

---

## Ingestion Pipeline Architecture

### Current Infrastructure (Already Implemented)

✅ **Sovereignty Firewall** (`knowledge3d/knowledgeverse/sovereignty_firewall.py`)
- AST + runtime validation of all feeders
- Zero chance of hot path pollution

✅ **Ingestion Stargate** (`knowledge3d/knowledgeverse/stargate.py`)
- Asynchronous job submission
- Air-gapped from sovereign hot path

✅ **Compressed Audit** (`knowledge3d/knowledgeverse/compressed_audit.py`)
- 17.39x compression ratio
- 0.483ms query performance

✅ **Temporal Metadata** (`knowledge3d/knowledgeverse/temporal_metadata.py`)
- Lamport + Vector clocks for causality

✅ **Self-Healing Wrappers** (`knowledge3d/knowledgeverse/resilience.py`)
- Retry, circuit breaker, fallback patterns

### New Components Needed (Week 11)

#### 1. Corpus Manifest Manager

**Purpose:** Centralized registry of all knowledge sources with metadata

**File:** `knowledge3d/ingestion/corpus_manifest.py`

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class CorpusTier(Enum):
    TIER_1_FOUNDATIONAL = 1  # Must have (algorithmic thinking, CS fundamentals)
    TIER_2_DOMAIN = 2        # Domain-specific (math, physics, visual)
    TIER_3_INTEGRATION = 3   # Cross-domain (problem-solving, meta-cognition)

class CorpusType(Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"
    CODE = "code"
    DATASET = "dataset"

@dataclass
class CorpusEntry:
    """Single entry in the corpus manifest"""
    id: str                          # Unique identifier
    name: str                        # Human-readable name
    path: str                        # Path to source file
    tier: CorpusTier                 # Priority tier
    corpus_type: CorpusType          # Source type
    target_galaxies: List[str]       # Which galaxies to populate
    dependencies: List[str]          # IDs of required prerequisite entries
    metadata: Dict[str, any]         # Additional metadata
    ingested: bool = False           # Ingestion status
    embedding_count: Optional[int] = None  # Number of embeddings generated

class CorpusManifest:
    """Registry of all knowledge sources"""

    def __init__(self):
        self.entries: Dict[str, CorpusEntry] = {}
        self._load_default_corpus()

    def _load_default_corpus(self):
        """Populate manifest with base knowledge sources"""

        # TIER 1: FOUNDATIONAL KNOWLEDGE
        self.add_entry(CorpusEntry(
            id="algorithmic_thinking",
            name="Algorithmic Thinking",
            path="docs/pdfs/algorithmic_thinking.pdf",
            tier=CorpusTier.TIER_1_FOUNDATIONAL,
            corpus_type=CorpusType.PDF,
            target_galaxies=["Grammar", "Math", "Reality"],
            dependencies=[],
            metadata={
                "priority": 1,
                "description": "Foundational problem-solving strategies",
                "domains": ["algorithms", "logic", "reasoning"]
            }
        ))

        self.add_entry(CorpusEntry(
            id="math_foundations",
            name="Mathematics Foundations",
            path="docs/pdfs/math_foundations/",  # Directory of PDFs
            tier=CorpusTier.TIER_1_FOUNDATIONAL,
            corpus_type=CorpusType.PDF,
            target_galaxies=["Math", "Grammar"],
            dependencies=["algorithmic_thinking"],
            metadata={
                "priority": 2,
                "description": "Algebra, calculus, geometry, number theory",
                "domains": ["mathematics"]
            }
        ))

        self.add_entry(CorpusEntry(
            id="logic_reasoning",
            name="Logic & Reasoning",
            path="docs/pdfs/logic_reasoning/",
            tier=CorpusTier.TIER_1_FOUNDATIONAL,
            corpus_type=CorpusType.PDF,
            target_galaxies=["Grammar", "Math"],
            dependencies=["algorithmic_thinking"],
            metadata={
                "priority": 3,
                "description": "Propositional logic, predicate logic, proof techniques",
                "domains": ["logic", "mathematics"]
            }
        ))

        self.add_entry(CorpusEntry(
            id="cs_fundamentals",
            name="Computer Science Fundamentals",
            path="docs/pdfs/cs_fundamentals/",
            tier=CorpusTier.TIER_1_FOUNDATIONAL,
            corpus_type=CorpusType.PDF,
            target_galaxies=["Grammar", "Reality"],
            dependencies=["algorithmic_thinking"],
            metadata={
                "priority": 4,
                "description": "Data structures, algorithms, complexity",
                "domains": ["computer_science", "algorithms"]
            }
        ))

        # TIER 2: DOMAIN-SPECIFIC EXTENSIONS

        # Math Domain
        self.add_entry(CorpusEntry(
            id="competition_math",
            name="Competition Math Problems",
            path="docs/datasets/competition_math/",
            tier=CorpusTier.TIER_2_DOMAIN,
            corpus_type=CorpusType.DATASET,
            target_galaxies=["Math", "Grammar"],
            dependencies=["math_foundations", "logic_reasoning"],
            metadata={
                "priority": 5,
                "description": "AMC, AIME, IMO problem archives",
                "domains": ["mathematics", "problem_solving"]
            }
        ))

        self.add_entry(CorpusEntry(
            id="undergraduate_math",
            name="Undergraduate Mathematics",
            path="docs/pdfs/undergraduate_math/",
            tier=CorpusTier.TIER_2_DOMAIN,
            corpus_type=CorpusType.PDF,
            target_galaxies=["Math", "Grammar"],
            dependencies=["math_foundations"],
            metadata={
                "priority": 6,
                "description": "Spivak, Rudin, abstract algebra, real analysis",
                "domains": ["mathematics"]
            }
        ))

        # Visual Domain
        self.add_entry(CorpusEntry(
            id="geometry_theorems",
            name="Geometry Theorems",
            path="docs/pdfs/geometry/",
            tier=CorpusTier.TIER_2_DOMAIN,
            corpus_type=CorpusType.PDF,
            target_galaxies=["Drawing", "Math"],
            dependencies=["math_foundations"],
            metadata={
                "priority": 7,
                "description": "Euclidean, projective, algebraic geometry",
                "domains": ["mathematics", "visual"]
            }
        ))

        self.add_entry(CorpusEntry(
            id="arc_agi_training",
            name="ARC-AGI Training Set",
            path="docs/datasets/arc_agi/",
            tier=CorpusTier.TIER_2_DOMAIN,
            corpus_type=CorpusType.DATASET,
            target_galaxies=["Drawing", "Grammar"],
            dependencies=["geometry_theorems"],
            metadata={
                "priority": 8,
                "description": "Visual reasoning patterns from ARC-AGI",
                "domains": ["visual", "reasoning"]
            }
        ))

        # Physics Domain
        self.add_entry(CorpusEntry(
            id="classical_mechanics",
            name="Classical Mechanics",
            path="docs/pdfs/physics/classical_mechanics/",
            tier=CorpusTier.TIER_2_DOMAIN,
            corpus_type=CorpusType.PDF,
            target_galaxies=["Reality", "Math"],
            dependencies=["math_foundations", "cs_fundamentals"],
            metadata={
                "priority": 9,
                "description": "Lagrangian, Hamiltonian formulations",
                "domains": ["physics", "mathematics"]
            }
        ))

        # Language Domain
        self.add_entry(CorpusEntry(
            id="grammar_rules",
            name="Grammar Transformation Rules",
            path="docs/datasets/grammar/",
            tier=CorpusTier.TIER_2_DOMAIN,
            corpus_type=CorpusType.DATASET,
            target_galaxies=["Grammar", "Word"],
            dependencies=["logic_reasoning"],
            metadata={
                "priority": 10,
                "description": "Context-free grammars, dependency parsing",
                "domains": ["language", "logic"]
            }
        ))

        # TIER 3: CROSS-DOMAIN INTEGRATION

        self.add_entry(CorpusEntry(
            id="problem_solving_strategies",
            name="Problem-Solving Strategies",
            path="docs/pdfs/problem_solving/",
            tier=CorpusTier.TIER_3_INTEGRATION,
            corpus_type=CorpusType.PDF,
            target_galaxies=["Grammar", "Math", "Reality", "Drawing"],
            dependencies=["algorithmic_thinking", "competition_math"],
            metadata={
                "priority": 11,
                "description": "Polya, heuristic search, constraint satisfaction",
                "domains": ["reasoning", "meta_cognition"]
            }
        ))

    def add_entry(self, entry: CorpusEntry):
        """Add entry to manifest"""
        self.entries[entry.id] = entry

    def get_by_tier(self, tier: CorpusTier) -> List[CorpusEntry]:
        """Get all entries for a specific tier"""
        return [e for e in self.entries.values() if e.tier == tier]

    def get_ingestion_order(self) -> List[CorpusEntry]:
        """Return entries in dependency-respecting order"""
        # Topological sort based on dependencies
        result = []
        visited = set()

        def visit(entry_id: str):
            if entry_id in visited:
                return
            entry = self.entries[entry_id]
            for dep_id in entry.dependencies:
                visit(dep_id)
            visited.add(entry_id)
            result.append(entry)

        for entry_id in self.entries.keys():
            visit(entry_id)

        return result

    def get_stats(self) -> Dict[str, any]:
        """Get manifest statistics"""
        return {
            "total_entries": len(self.entries),
            "tier_1_count": len(self.get_by_tier(CorpusTier.TIER_1_FOUNDATIONAL)),
            "tier_2_count": len(self.get_by_tier(CorpusTier.TIER_2_DOMAIN)),
            "tier_3_count": len(self.get_by_tier(CorpusTier.TIER_3_INTEGRATION)),
            "ingested_count": sum(1 for e in self.entries.values() if e.ingested),
            "pending_count": sum(1 for e in self.entries.values() if not e.ingested)
        }
```

#### 2. Batch Ingestion Orchestrator

**Purpose:** Process corpus entries in parallel, respecting dependencies

**File:** `knowledge3d/ingestion/batch_orchestrator.py`

```python
import asyncio
from typing import List, Dict, Optional
from knowledge3d.ingestion.corpus_manifest import CorpusManifest, CorpusEntry, CorpusType
from knowledge3d.knowledgeverse.stargate import IngestionStargate
from knowledge3d.knowledgeverse.sovereignty_firewall import SovereigntyFirewall
from knowledge3d.knowledgeverse.resilience import retry, circuit_breaker, fallback

class BatchOrchestrator:
    """Orchestrate batch ingestion of corpus entries"""

    def __init__(self, manifest: CorpusManifest, stargate: IngestionStargate):
        self.manifest = manifest
        self.stargate = stargate
        self.firewall = SovereigntyFirewall()
        self.in_progress: Dict[str, asyncio.Task] = {}

    @retry(max_attempts=3, backoff_seconds=2.0)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30.0)
    async def ingest_entry(self, entry: CorpusEntry) -> Dict[str, any]:
        """Ingest a single corpus entry"""

        print(f"[BatchOrchestrator] Ingesting {entry.name} (Tier {entry.tier.value})")

        # 1. Validate feeder sovereignty (if code-based)
        if entry.corpus_type == CorpusType.CODE:
            feeder_path = f"knowledge3d/ingestion/feeders/{entry.id}_feeder.py"
            validation = self.firewall.validate_feeder(feeder_path)
            if not validation["is_sovereign"]:
                raise ValueError(f"Feeder {entry.id} failed sovereignty check: {validation['violations']}")

        # 2. Submit ingestion job
        job_id = await self.stargate.submit_ingestion_job(
            data_path=entry.path,
            data_type=entry.corpus_type.value,
            target_galaxies=entry.target_galaxies,
            metadata=entry.metadata
        )

        # 3. Wait for completion
        result = await self.stargate.wait_for_job(job_id, timeout=600.0)

        # 4. Update manifest
        entry.ingested = True
        entry.embedding_count = result.get("embedding_count", 0)

        print(f"[BatchOrchestrator] ✅ {entry.name} complete: {entry.embedding_count} embeddings")

        return result

    async def ingest_tier(self, tier: int, max_parallel: int = 4) -> Dict[str, any]:
        """Ingest all entries in a tier (respecting dependencies)"""

        from knowledge3d.ingestion.corpus_manifest import CorpusTier
        tier_enum = CorpusTier(tier)
        entries = self.manifest.get_by_tier(tier_enum)

        print(f"\n[BatchOrchestrator] Starting Tier {tier}: {len(entries)} entries")

        results = {}
        semaphore = asyncio.Semaphore(max_parallel)

        async def process_with_semaphore(entry: CorpusEntry):
            async with semaphore:
                return await self.ingest_entry(entry)

        # Process entries respecting dependencies
        completed = set()

        while len(completed) < len(entries):
            # Find entries ready to process (dependencies satisfied)
            ready = [
                e for e in entries
                if e.id not in completed
                and all(dep in completed for dep in e.dependencies)
            ]

            if not ready:
                # Check for circular dependencies
                pending = [e.id for e in entries if e.id not in completed]
                raise ValueError(f"Circular dependencies detected in tier {tier}: {pending}")

            # Process ready entries in parallel
            tasks = [process_with_semaphore(entry) for entry in ready]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for entry, result in zip(ready, batch_results):
                if isinstance(result, Exception):
                    print(f"[BatchOrchestrator] ❌ {entry.name} failed: {result}")
                    results[entry.id] = {"error": str(result)}
                else:
                    results[entry.id] = result
                completed.add(entry.id)

        print(f"[BatchOrchestrator] ✅ Tier {tier} complete: {len(completed)}/{len(entries)} succeeded")

        return results

    async def ingest_all(self, max_parallel: int = 4) -> Dict[str, any]:
        """Ingest entire corpus in order (Tier 1 → Tier 2 → Tier 3)"""

        print("\n" + "="*60)
        print("BATCH INGESTION STARTING")
        print("="*60)

        stats_before = self.manifest.get_stats()
        print(f"\nCorpus Manifest: {stats_before['total_entries']} entries")
        print(f"  Tier 1 (Foundational): {stats_before['tier_1_count']}")
        print(f"  Tier 2 (Domain): {stats_before['tier_2_count']}")
        print(f"  Tier 3 (Integration): {stats_before['tier_3_count']}")
        print(f"  Pending: {stats_before['pending_count']}")

        results = {}

        # Process each tier sequentially (tiers have dependencies between them)
        for tier in [1, 2, 3]:
            tier_results = await self.ingest_tier(tier, max_parallel)
            results[f"tier_{tier}"] = tier_results

        stats_after = self.manifest.get_stats()

        print("\n" + "="*60)
        print("BATCH INGESTION COMPLETE")
        print("="*60)
        print(f"Ingested: {stats_after['ingested_count']}/{stats_after['total_entries']}")

        return {
            "stats_before": stats_before,
            "stats_after": stats_after,
            "results": results
        }
```

#### 3. Enrichment Pipeline

**Purpose:** Cross-reference with training data knowledge, generate matryoshka embeddings

**File:** `knowledge3d/ingestion/enrichment_pipeline.py`

```python
import numpy as np
from typing import List, Dict, Optional
from knowledge3d.cranium.rpn_vm import RPNProgram

class EnrichmentPipeline:
    """Enrich raw content with embeddings, symlinks, and procedural programs"""

    def __init__(self):
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.symlink_registry: Dict[str, str] = {}  # Content hash → canonical ID

    def generate_matryoshka_embedding(self, content: str) -> Dict[int, np.ndarray]:
        """Generate multi-resolution embeddings (64/128/512/2048D)"""

        # TODO: Replace with actual embedding model (sovereign PTX kernel)
        # For now, simulate matryoshka structure

        base_embedding = np.random.randn(2048).astype(np.float32)
        base_embedding /= np.linalg.norm(base_embedding)

        return {
            64: base_embedding[:64],
            128: base_embedding[:128],
            512: base_embedding[:512],
            2048: base_embedding
        }

    def find_or_create_symlink(self, content: str) -> str:
        """Check if content already exists, return symlink or create new entry"""

        import hashlib
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        if content_hash in self.symlink_registry:
            canonical_id = self.symlink_registry[content_hash]
            print(f"[Enrichment] Found existing content, symlinking to {canonical_id}")
            return canonical_id
        else:
            # Create new entry
            new_id = f"entry_{content_hash}"
            self.symlink_registry[content_hash] = new_id
            return new_id

    def extract_procedural_patterns(self, content: str, domain: str) -> List[RPNProgram]:
        """Extract reusable procedural patterns from content"""

        # Domain-specific pattern extraction
        if domain == "math":
            return self._extract_math_patterns(content)
        elif domain == "visual":
            return self._extract_visual_patterns(content)
        elif domain == "logic":
            return self._extract_logic_patterns(content)
        else:
            return []

    def _extract_math_patterns(self, content: str) -> List[RPNProgram]:
        """Extract mathematical patterns (e.g., proof templates)"""
        # TODO: Implement pattern recognition
        # For now, return empty list
        return []

    def _extract_visual_patterns(self, content: str) -> List[RPNProgram]:
        """Extract visual patterns (e.g., geometric transformations)"""
        # TODO: Implement pattern recognition
        return []

    def _extract_logic_patterns(self, content: str) -> List[RPNProgram]:
        """Extract logical patterns (e.g., inference rules)"""
        # TODO: Implement pattern recognition
        return []

    def enrich_document(self, content: str, metadata: Dict[str, any]) -> Dict[str, any]:
        """Full enrichment pipeline for a document"""

        # 1. Check for duplicates (symlink if exists)
        entry_id = self.find_or_create_symlink(content)

        # 2. Generate matryoshka embeddings
        embeddings = self.generate_matryoshka_embedding(content)

        # 3. Extract procedural patterns
        domain = metadata.get("domain", "general")
        patterns = self.extract_procedural_patterns(content, domain)

        # 4. Cross-reference with training data (TODO: implement)
        related_concepts = self._find_related_concepts(content)

        return {
            "entry_id": entry_id,
            "embeddings": embeddings,
            "patterns": patterns,
            "related_concepts": related_concepts,
            "metadata": metadata
        }

    def _find_related_concepts(self, content: str) -> List[str]:
        """Find related concepts from training data knowledge"""
        # TODO: Implement semantic search against training data
        return []
```

---

## Implementation Timeline

### Week 11: Setup (5 days)

**Day 1-2: Corpus Manifest**
- Implement `corpus_manifest.py` with all 11+ entries
- Validate all file paths exist
- Test dependency resolution

**Day 3-4: Batch Orchestrator**
- Implement `batch_orchestrator.py`
- Test parallel processing with mock entries
- Validate sovereignty firewall integration

**Day 5: Enrichment Pipeline**
- Implement `enrichment_pipeline.py`
- Test matryoshka embedding generation
- Test symlink deduplication

**Success Criteria:**
- ✅ Manifest loads all entries
- ✅ Orchestrator processes mock entries in correct order
- ✅ Enrichment generates embeddings + symlinks

### Week 12: Batch Ingestion (5 days)

**Day 1-2: Tier 1 Ingestion**
- Process all 4 foundational entries:
  - Algorithmic thinking PDF
  - Math foundations
  - Logic & reasoning
  - CS fundamentals
- Validate Galaxy population (Grammar, Math, Reality)

**Day 3-4: Tier 2 Ingestion**
- Process all 7 domain entries:
  - Competition math
  - Undergraduate math
  - Geometry theorems
  - ARC-AGI training
  - Classical mechanics
  - Grammar rules
  - (Any additional domain sources)
- Validate cross-galaxy symlinks

**Day 5: Tier 3 Ingestion + Validation**
- Process integration entries (problem-solving strategies)
- Run end-to-end validation:
  - Query each Galaxy for new entries
  - Verify symlinks work correctly
  - Test TRM navigation with enriched knowledge

**Success Criteria:**
- ✅ All corpus entries ingested (100% coverage)
- ✅ Matryoshka embeddings generated for all content
- ✅ Symlinks reduce duplication (~70% as in Dual Client Contract)
- ✅ TRM can query and compose from enriched Galaxies
- ✅ Sovereignty maintained (hot path still PTX-only)

---

## Validation & Testing

### Test 1: Manifest Integrity

```python
def test_corpus_manifest_integrity():
    manifest = CorpusManifest()

    # All entries have valid paths
    for entry in manifest.entries.values():
        assert os.path.exists(entry.path), f"Missing: {entry.path}"

    # No circular dependencies
    order = manifest.get_ingestion_order()
    assert len(order) == len(manifest.entries)

    # Tier 1 comes before Tier 2
    tier_sequence = [e.tier.value for e in order]
    assert tier_sequence == sorted(tier_sequence)
```

### Test 2: Batch Ingestion

```python
async def test_batch_ingestion():
    manifest = CorpusManifest()
    stargate = IngestionStargate()
    orchestrator = BatchOrchestrator(manifest, stargate)

    # Ingest Tier 1
    results = await orchestrator.ingest_tier(1, max_parallel=2)

    # Verify all entries succeeded
    assert all(
        "error" not in r
        for r in results.values()
    )

    # Verify Galaxy population
    from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager
    manager = GalaxyManager()

    math_galaxy = manager.get_galaxy("Math")
    assert len(math_galaxy.entries) > 100  # Should have many new symbols
```

### Test 3: Enrichment & Symlinks

```python
def test_enrichment_symlinks():
    pipeline = EnrichmentPipeline()

    # Same content should get same ID
    content = "The derivative of x^2 is 2x"
    id1 = pipeline.find_or_create_symlink(content)
    id2 = pipeline.find_or_create_symlink(content)
    assert id1 == id2

    # Different content should get different IDs
    content2 = "The integral of 2x is x^2 + C"
    id3 = pipeline.find_or_create_symlink(content2)
    assert id3 != id1
```

### Test 4: End-to-End (Raw PDF → Galaxy Entry)

```python
async def test_end_to_end_pdf_to_galaxy():
    # 1. Submit PDF ingestion
    stargate = IngestionStargate()
    job_id = await stargate.submit_ingestion_job(
        data_path="docs/pdfs/algorithmic_thinking.pdf",
        data_type="pdf",
        target_galaxies=["Grammar", "Math"],
        metadata={"priority": 1}
    )

    # 2. Wait for completion
    result = await stargate.wait_for_job(job_id, timeout=300.0)
    assert result["status"] == "complete"

    # 3. Verify Galaxy entries
    from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager
    manager = GalaxyManager()
    grammar_galaxy = manager.get_galaxy("Grammar")

    # Should have new grammar rules from PDF
    assert len(grammar_galaxy.entries) > 0

    # 4. Verify TRM can navigate
    from knowledge3d.knowledgeverse.trm_navigator import TRMNavigator
    navigator = TRMNavigator()

    query = "algorithmic problem solving strategy"
    results = navigator.query(query, galaxy_names=["Grammar"])
    assert len(results) > 0
```

---

## Integration with Existing Infrastructure

### Leverage MVP Phase 1 Features

**Sovereignty Firewall** → Validate all feeders before ingestion
**Compressed Audit** → Log all ingestion events (17.39x compression)
**Self-Healing** → Retry failed ingestions automatically
**Temporal Metadata** → Track ingestion causality (Lamport clocks)

### Galaxy Manager Integration

All ingested content populates existing Galaxies:
- **Drawing Galaxy** → Visual patterns, geometry theorems
- **Character Galaxy** → Already has procedural fonts (reuse!)
- **Word Galaxy** → Symlinked character sequences
- **Grammar Galaxy** → Transformation rules, proof templates
- **Math Galaxy** → Symbols, theorems, solution patterns
- **Reality Galaxy** → Physics laws, simulation templates
- **Audio Galaxy** → (Future) Spectrograms, sonification patterns

### TRM Navigator Integration

The enriched Galaxies enable TRM to:
1. **Query** with richer semantic search (matryoshka embeddings)
2. **Compose** by combining procedural patterns
3. **Create** by synthesizing new entries (shadow copy learning)
4. **Route** by selecting appropriate specialist adapters

---

## Success Metrics

**After Week 12, we should have:**

1. **Corpus Coverage:**
   - ✅ 100% of Tier 1 entries ingested
   - ✅ 80%+ of Tier 2 entries ingested
   - ✅ 50%+ of Tier 3 entries ingested

2. **Galaxy Population:**
   - Grammar Galaxy: 500+ transformation rules
   - Math Galaxy: 1000+ symbols with RPN templates
   - Reality Galaxy: 200+ physics/chemistry procedures
   - Drawing Galaxy: 300+ geometric patterns

3. **Embedding Generation:**
   - 10,000+ matryoshka embeddings (64/128/512/2048D)
   - ~70% symlink deduplication (as per Dual Client Contract)

4. **Sovereignty:**
   - ✅ Hot path remains PTX-only (zero regressions)
   - ✅ All feeders pass Sovereignty Firewall validation
   - ✅ 28/28 MVP tests still passing

5. **TRM Readiness:**
   - ✅ TRM can query enriched Galaxies
   - ✅ TRM can compose from procedural patterns
   - ✅ Shadow Copy records successful navigations

---

## Handoff to Phase 1C (Benchmarks)

**After Knowledge Preparation is complete, we pivot to:**

**Week 13: Benchmark Integration**
- ARC-AGI 2 evaluation pipeline
- Math competition baseline (AMC, AIME)
- Physics reasoning baseline

**Week 14: Baseline Measurement & Analysis**
- Run full benchmark suite with enriched knowledge
- Compare "empty mind" (46.7%) vs "enriched" performance
- Identify remaining gaps for iterative improvement

**Success Target:** Beat "empty mind" baseline significantly
- ARC-AGI: 46.7% → 55%+ (prize threshold)
- Math: 0% → 30%+ (competitive baseline)
- Physics: 0% → 40%+ (domain reasoning)

---

## Codex Implementation Directive

**Priority:** CRITICAL (Week 11-12, Phase 1B)

**What Codex Should Implement:**

1. **Week 11: Setup**
   - `knowledge3d/ingestion/corpus_manifest.py` (full implementation)
   - `knowledge3d/ingestion/batch_orchestrator.py` (full implementation)
   - `knowledge3d/ingestion/enrichment_pipeline.py` (full implementation)
   - Test suite: `tests/test_ingestion_pipeline.py` (4 tests from above)

2. **Week 12: Execution**
   - Run batch ingestion script
   - Monitor progress (use Compressed Audit for logging)
   - Validate Galaxy population
   - Run end-to-end validation

**Sovereignty Requirement:** ABSOLUTE
- Ingestion path can use any tools (numpy, pandas, PyPDF2, etc.)
- BUT hot path (TRM navigation, PTX kernels) remains sovereign
- All feeders MUST pass Sovereignty Firewall validation

**Testing Requirement:** 100% Coverage
- All tests from "Validation & Testing" section
- Integration test: Raw PDF → Galaxy → TRM query
- Regression: 28/28 MVP tests still pass

---

## Local Model Enhancement Strategy

**CRITICAL OPPORTUNITY:** We can leverage local Ollama models (up to GPU VRAM limit) to enhance ingestion quality WITHOUT violating sovereignty (ingestion path is flexible!).

### Available Local Models for Enrichment

**Models from Previous Survey + New Releases:**
- `qwen2.5:14b` / `qwen2.5:32b` — Excellent math reasoning, multilingual
- `deepseek-r1:14b` / `deepseek-r1:32b` — Strong logical reasoning, code understanding
- `llama3.1:8b` / `llama3.1:70b` — General-purpose, good instruction following
- `phi3:14b` — Efficient, strong on technical content
- `gemma2:9b` / `gemma2:27b` — Google models, good multimodal understanding
- `mistral:7b` / `mixtral:8x7b` — Fast, good at pattern extraction

**If model not downloaded:**
```bash
ollama pull qwen2.5:14b  # Example
ollama pull deepseek-r1:14b
```

### Enrichment Use Cases

**1. Semantic Analysis (Use: Qwen/DeepSeek)**
```python
# Analyze PDF content for key concepts
prompt = f"""
Extract key concepts, definitions, and relationships from this text:
{pdf_content}

Output as JSON: {{"concepts": [...], "relations": [...]}}
"""
concepts = ollama_query("qwen2.5:14b", prompt)
```

**2. Pattern Extraction (Use: DeepSeek/Mistral)**
```python
# Identify procedural patterns for RPN conversion
prompt = f"""
Find reusable patterns in this algorithm description:
{algorithm_text}

Identify: input patterns, transformation steps, output patterns
"""
patterns = ollama_query("deepseek-r1:14b", prompt)
```

**3. Cross-Referencing (Use: Qwen/Gemma)**
```python
# Find related concepts from training data
prompt = f"""
Given this concept: {concept_description}
List 10 related concepts from mathematics, physics, or computer science.
"""
related = ollama_query("qwen2.5:14b", prompt)
```

**4. Proof Template Extraction (Use: DeepSeek)**
```python
# Extract proof structure for Grammar Galaxy
prompt = f"""
Analyze this mathematical proof and extract the template structure:
{proof_text}

Output: proof pattern (e.g., "induction", "contradiction", "construction")
"""
template = ollama_query("deepseek-r1:14b", prompt)
```

### Implementation in Enrichment Pipeline

**Update `enrichment_pipeline.py`:**

```python
class EnrichmentPipeline:
    def __init__(self, use_local_models: bool = True):
        self.use_local_models = use_local_models
        self.model_name = "qwen2.5:14b"  # Default, can switch per task

    def ollama_query(self, model: str, prompt: str) -> str:
        """Query local Ollama model"""
        import subprocess
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()

    def extract_procedural_patterns(self, content: str, domain: str) -> List[RPNProgram]:
        """Extract reusable procedural patterns from content"""

        if not self.use_local_models:
            return self._extract_patterns_heuristic(content, domain)

        # Use local model for enhanced extraction
        if domain == "math":
            model = "deepseek-r1:14b"  # Best for math reasoning
            prompt = f"""
            Extract mathematical patterns from this content that can be converted to procedural programs:
            {content[:2000]}  # Truncate to fit context

            Output each pattern with:
            1. Pattern name
            2. Input requirements
            3. Transformation steps
            4. Output format
            """
        elif domain == "visual":
            model = "gemma2:9b"  # Good for visual reasoning
            prompt = f"""
            Extract visual/geometric patterns from this content:
            {content[:2000]}

            Focus on: transformations, symmetries, spatial relationships
            """
        else:
            model = "qwen2.5:14b"  # General purpose
            prompt = f"""
            Extract reusable patterns from this content in the {domain} domain:
            {content[:2000]}
            """

        response = self.ollama_query(model, prompt)
        return self._parse_patterns_from_llm_response(response)

    def _find_related_concepts(self, content: str) -> List[str]:
        """Find related concepts from training data knowledge"""

        if not self.use_local_models:
            return []

        # Use local model to cross-reference
        prompt = f"""
        Given this concept/content:
        {content[:1000]}

        List 10 related concepts from mathematics, physics, computer science, or logic.
        Output as comma-separated list.
        """

        response = self.ollama_query("qwen2.5:14b", prompt)
        return [c.strip() for c in response.split(",")]
```

### GPU Memory Management

**Balance between Knowledgeverse VRAM and Ollama:**
- Reserve ~40GB for Knowledgeverse (Regions 1-7)
- Use remaining VRAM for Ollama models during ingestion
- After ingestion complete, unload Ollama models (full VRAM to Knowledgeverse)

**Example:**
```python
# Before ingestion
knowledgeverse.checkpoint()  # Save state
knowledgeverse.unload_non_essential_galaxies()  # Free VRAM temporarily

# Run enrichment with Ollama
enrichment = EnrichmentPipeline(use_local_models=True)
enriched_data = enrichment.enrich_document(content, metadata)

# After ingestion
knowledgeverse.restore_galaxies()  # Load back to full VRAM
```

### Sovereignty Compliance

**✅ SAFE:** Local models in ingestion path (air-gapped from hot path)
**✅ SAFE:** Results are RPN programs + embeddings (sovereign artifacts)
**❌ FORBIDDEN:** Using Ollama models during TRM inference (sovereignty violation)

The local models are **ingestion accelerators**, not runtime dependencies!

---

## Questions for Codex

1. **PDF Processing:** Should we use PyPDF2, pdfplumber, or custom parser?
2. **Embedding Model:** Use pre-trained model (sentence-transformers) or train sovereign embeddings?
3. **Local Model Selection:** Which Ollama models should we prioritize downloading?
4. **GPU Memory Split:** What ratio for Knowledgeverse vs Ollama during ingestion?
5. **Parallel Processing:** 4 workers optimal, or scale based on available GPUs?
6. **Storage:** Store embeddings in VRAM (Galaxy) or disk cache (House)?

---

## Appendix: Open Source Content from Training Data

**What Codex/Claude/Gemini know from training that can enrich the corpus:**

### Mathematical Knowledge
- Lean theorem prover library (formal proofs)
- Project Euler solutions (algorithmic problem-solving)
- OEIS (integer sequences and patterns)
- Mathematical proof patterns (induction, contradiction, construction)

### Visual Reasoning
- ARC-AGI training set (visual patterns, transformations)
- Synthetic geometry datasets (triangles, polygons, symmetries)
- Computer graphics algorithms (Bresenham, Cohen-Sutherland)

### Physics & Chemistry
- Classical mechanics examples (pendulum, projectile, orbits)
- Thermodynamics problems (ideal gas, Carnot cycle)
- Chemistry reaction databases (stoichiometry, equilibrium)

### Computer Science
- Algorithm implementations (sorting, searching, graph algorithms)
- Data structure patterns (trees, heaps, hash tables)
- Complexity analysis examples (Big-O, recurrence relations)

### Problem-Solving Strategies
- Polya's "How to Solve It" (4-step method)
- Heuristic search patterns (A*, beam search, MCTS)
- Constraint satisfaction (backtracking, arc consistency)

**Action:** Codex should create feeder scripts that extract this knowledge from:
1. Publicly available datasets (GitHub, arXiv, Kaggle)
2. Synthetic generation (using existing knowledge to create examples)
3. Cross-referencing (linking related concepts across domains)

---

## End of Specification

**Next Steps:**
1. Codex implements Week 11 setup (manifest, orchestrator, enrichment)
2. Codex runs Week 12 batch ingestion
3. Codex validates Galaxy population and TRM readiness
4. Claude receives completion report and approves Phase 1C (Benchmarks)

**Remember:** Preparation → Ingestion → THEN Benchmarks (not the other way around!)

**Contact:** Claude (Architecture Partner) for design questions, Codex (Implementation Partner) for execution.
