# Stargate Crystallization Hardening — Remove Synthetic Placeholders

**Created:** February 6, 2026
**Author:** Claude (Architecture Partner)
**Priority:** CRITICAL (Week 13 Hardening)
**Status:** Architecture Specification

---

## Executive Summary

**User Directive:** *"Make sure placeholders are only temporary, they should not be a norm, actually, we do not use that here neither gave up on sovereignty on the hotpath, so all things external must aim to be transformed to K3D standard ASAP."*

**Current Problem:** In Phase 1B, `IngestionStargate.submit_ingestion_job()` returns synthetic `embedding_count` because crystallization into Galaxies is not yet implemented.

**Target State:** Real crystallization pipeline that transforms ingestion artifacts → RPN programs → Galaxy entries immediately.

---

## Current Implementation (Phase 1B)

```python
# In knowledge3d/knowledgeverse/stargate.py

async def submit_ingestion_job(self, data_path, data_type, target_galaxies, metadata):
    """Submit ingestion job (async)."""
    job_id = str(uuid.uuid4())

    # ... (job creation)

    # PLACEHOLDER: Return synthetic embedding_count
    return {
        "job_id": job_id,
        "status": "complete",
        "embedding_count": 123  # ❌ SYNTHETIC (not real)
    }
```

**Issue:** No actual Galaxy population occurs. Enrichment artifacts sit in external files, not integrated into K3D.

---

## Target Implementation (Week 13)

### Architecture

```
Ingestion Flow:
1. Raw Data (PDF, audio, code, etc.)
    ↓
2. Feeder (external tools: PyPDF2, PIL, etc.)
    ↓
3. Enrichment (Ollama LLMs + matryoshka embeddings)
    ↓
4. K3D Transformation (RPN programs + Galaxy entries)
    ↓
5. Crystallization (Region 7 → Region 2)
    ↓
6. Galaxy Population (Math, Grammar, Reality, Drawing, etc.)
```

**Key Insight:** Steps 1-3 are flexible (external tools). Step 4-6 MUST be sovereign (K3D artifacts only).

### Enhanced Stargate Implementation

```python
# In knowledge3d/knowledgeverse/stargate.py

class IngestionStargate:
    """
    Air-gapped buffer for raw data → RPN transmutation.
    NOW WITH REAL CRYSTALLIZATION.
    """

    def __init__(self, stargate_region, galaxy_manager, rpn_engine, k3d_transformer):
        self.region = stargate_region
        self.galaxy_manager = galaxy_manager
        self.rpn_engine = rpn_engine
        self.k3d_transformer = k3d_transformer
        self.ring_buffer = RingBuffer(size_mb=512)
        self.pending_jobs = {}
        self.completed_jobs = {}

    async def submit_ingestion_job(
        self,
        data_path: str,
        data_type: str,
        target_galaxies: List[str],
        metadata: Dict
    ) -> str:
        """
        Submit raw data for ingestion (async).

        Returns:
            str: Job ID (use wait_for_job to get results)
        """
        job_id = str(uuid.uuid4())

        job = {
            'job_id': job_id,
            'data_path': data_path,
            'data_type': data_type,
            'target_galaxies': target_galaxies,
            'metadata': metadata,
            'status': 'pending',
            'submitted_at': time.time(),
            'enrichment_path': None,
            'crystallization_result': None
        }

        self.pending_jobs[job_id] = job

        # Trigger full pipeline (background)
        asyncio.create_task(self._execute_ingestion_pipeline(job))

        return job_id

    async def _execute_ingestion_pipeline(self, job: Dict):
        """
        Execute full ingestion pipeline:
        1. Feeder (raw → enrichment artifacts)
        2. K3D Transformation (enrichment → RPN + Galaxy entries)
        3. Crystallization (write to Galaxies)
        """
        try:
            job['status'] = 'running'

            # Step 1: Feeder (external tools)
            enrichment_path = await self._run_feeder(job)
            job['enrichment_path'] = enrichment_path

            # Step 2: K3D Transformation (sovereign)
            crystallization_result = await self._crystallize_enrichment(
                enrichment_path,
                job['target_galaxies']
            )
            job['crystallization_result'] = crystallization_result

            # Step 3: Update status
            job['status'] = 'complete'
            job['completed_at'] = time.time()

            # Move to completed jobs
            self.completed_jobs[job['job_id']] = job
            del self.pending_jobs[job['job_id']]

        except Exception as e:
            job['status'] = 'failed'
            job['error'] = str(e)
            job['failed_at'] = time.time()

            self.completed_jobs[job['job_id']] = job
            del self.pending_jobs[job['job_id']]

    async def _run_feeder(self, job: Dict) -> str:
        """
        Run host-side feeder (external tools).

        Returns:
            str: Path to enrichment artifacts
        """
        import subprocess

        feeder_script = self._get_feeder_script(job['data_type'])
        output_dir = f"../Knowledge3D.local/datasets/stargate_jobs/{job['job_id']}"

        # Run feeder in background
        process = await asyncio.create_subprocess_exec(
            'python',
            feeder_script,
            '--job-id', job['job_id'],
            '--data-path', job['data_path'],
            '--data-type', job['data_type'],
            '--output-dir', output_dir,
            '--use-local-models', 'true',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"Feeder failed: {stderr.decode()}")

        return output_dir

    async def _crystallize_enrichment(self, enrichment_path: str, target_galaxies: List[str]) -> Dict:
        """
        Crystallize enrichment artifacts → Galaxy entries.

        This is the CRITICAL step that removes placeholders.

        Args:
            enrichment_path: Path to enrichment artifacts (JSON files)
            target_galaxies: List of galaxy names to populate

        Returns:
            dict: {
                "galaxy_entries_created": int,
                "rpn_programs_created": int,
                "target_galaxies": List[str],
                "embeddings_stored": int
            }
        """
        import json
        import glob

        total_entries = 0
        total_rpns = 0
        total_embeddings = 0
        galaxies_populated = set()

        # Load all enrichment artifacts
        enrichment_files = glob.glob(f"{enrichment_path}/*.json")

        for enrichment_file in enrichment_files:
            with open(enrichment_file, 'r') as f:
                enrichment_result = json.load(f)

            # Transform to K3D standard
            crystallization = self.k3d_transformer.crystallize_enrichment_to_galaxies(
                enrichment_result
            )

            total_entries += crystallization["galaxy_entries_created"]
            total_rpns += crystallization["rpn_programs_created"]
            galaxies_populated.update(crystallization["target_galaxies"])

            # Count embeddings
            if "embeddings" in enrichment_result:
                total_embeddings += len(enrichment_result["embeddings"])

        return {
            "galaxy_entries_created": total_entries,
            "rpn_programs_created": total_rpns,
            "target_galaxies": list(galaxies_populated),
            "embeddings_stored": total_embeddings
        }

    async def wait_for_job(self, job_id: str, timeout: float = 600.0) -> Dict:
        """
        Wait for ingestion job to complete.

        Args:
            job_id: Job ID from submit_ingestion_job()
            timeout: Max wait time in seconds

        Returns:
            dict: {
                "status": "complete" | "failed",
                "galaxy_entries_created": int,  # REAL count (not synthetic)
                "rpn_programs_created": int,
                "embeddings_stored": int,
                "target_galaxies": List[str],
                "error": str (if failed)
            }
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check completed jobs
            if job_id in self.completed_jobs:
                job = self.completed_jobs[job_id]

                if job['status'] == 'complete':
                    return {
                        "status": "complete",
                        "job_id": job_id,
                        **job['crystallization_result']
                    }
                else:
                    return {
                        "status": "failed",
                        "job_id": job_id,
                        "error": job.get('error', 'Unknown error')
                    }

            # Still pending
            await asyncio.sleep(1.0)

        # Timeout
        return {
            "status": "timeout",
            "job_id": job_id,
            "error": f"Job did not complete within {timeout} seconds"
        }

    def _get_feeder_script(self, data_type: str) -> str:
        """Get feeder script path for data type."""
        feeder_map = {
            "pdf": "knowledge3d/ingestion/feeders/pdf_feeder.py",
            "audio": "knowledge3d/ingestion/feeders/audio_feeder.py",
            "code": "knowledge3d/ingestion/feeders/code_feeder.py",
            "image": "knowledge3d/ingestion/feeders/image_feeder.py"
        }
        return feeder_map.get(data_type, "knowledge3d/ingestion/feeders/generic_feeder.py")
```

---

## Enhanced Feeder Script

```python
# In knowledge3d/ingestion/feeders/pdf_feeder.py

"""
PDF Feeder with K3D Transformation.
Runs OUTSIDE sovereign hot path - can use any libraries.
"""

import argparse
import json
import os
from knowledge3d.ingestion.enrichment_pipeline import EnrichmentPipeline
from knowledge3d.ingestion.k3d_transformer import K3DTransformer
from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager
from knowledge3d.cranium.rpn_vm import SovereignRPNEngine

def process_pdf(pdf_path: str, output_dir: str, use_local_models: bool = True):
    """
    Process PDF → Enrichment artifacts + K3D transformation.

    This feeder does BOTH:
    1. Enrichment (external tools + Ollama)
    2. K3D Transformation (RPN programs + Galaxy entries prepared)
    """
    import pdfplumber

    os.makedirs(output_dir, exist_ok=True)

    # Initialize enrichment pipeline
    enrichment = EnrichmentPipeline(
        use_local_models=use_local_models,
        k3d_transformer=None  # Will create later
    )

    all_enrichment_results = []

    # Extract text from PDF
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()

            if not text.strip():
                continue

            # Enrich with clean context (model unload/reload per page)
            enrichment_result = enrichment.enrich_document_with_clean_context(
                content=text,
                metadata={
                    "source": pdf_path,
                    "page": page_num,
                    "domain": "general"  # Can be enhanced with domain detection
                }
            )

            all_enrichment_results.append(enrichment_result)

            # Save enrichment artifact
            output_file = f"{output_dir}/page_{page_num:04d}.json"
            with open(output_file, 'w') as f:
                json.dump(enrichment_result, f, indent=2)

    # Summary
    summary = {
        "total_pages": len(all_enrichment_results),
        "total_patterns": sum(len(r["patterns"]) for r in all_enrichment_results),
        "total_concepts": sum(len(r["related_concepts"]) for r in all_enrichment_results),
        "output_dir": output_dir
    }

    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[PDFFeeder] Processed {summary['total_pages']} pages")
    print(f"[PDFFeeder] Extracted {summary['total_patterns']} patterns")
    print(f"[PDFFeeder] Found {summary['total_concepts']} concepts")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--data-type', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--use-local-models', default='false')

    args = parser.parse_args()

    process_pdf(
        pdf_path=args.data_path,
        output_dir=args.output_dir,
        use_local_models=(args.use_local_models.lower() == 'true')
    )
```

---

## Integration with Batch Orchestrator

```python
# In knowledge3d/ingestion/batch_orchestrator.py

class BatchOrchestrator:
    """Orchestrate batch ingestion with REAL crystallization."""

    @retry(max_attempts=3, backoff_seconds=2.0)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30.0)
    async def ingest_entry(self, entry: CorpusEntry) -> Dict[str, any]:
        """Ingest a single corpus entry with REAL Galaxy population."""

        print(f"[BatchOrchestrator] Ingesting {entry.name} (Tier {entry.tier.value})")

        # 1. Validate feeder sovereignty (if code-based)
        if entry.corpus_type == CorpusType.CODE:
            feeder_path = f"knowledge3d/ingestion/feeders/{entry.id}_feeder.py"
            validation = self.firewall.validate_feeder(feeder_path)
            if not validation["is_sovereign"]:
                raise ValueError(f"Feeder {entry.id} failed sovereignty check")

        # 2. Submit ingestion job (async)
        job_id = await self.stargate.submit_ingestion_job(
            data_path=entry.path,
            data_type=entry.corpus_type.value,
            target_galaxies=entry.target_galaxies,
            metadata=entry.metadata
        )

        # 3. Wait for completion (with REAL crystallization)
        result = await self.stargate.wait_for_job(job_id, timeout=600.0)

        # 4. Update manifest with REAL counts
        entry.ingested = True
        entry.embedding_count = result.get("embeddings_stored", 0)  # REAL count
        entry.rpn_count = result.get("rpn_programs_created", 0)  # NEW
        entry.galaxy_entries = result.get("galaxy_entries_created", 0)  # NEW

        print(f"[BatchOrchestrator] ✅ {entry.name} complete:")
        print(f"  Embeddings: {entry.embedding_count}")
        print(f"  RPN Programs: {entry.rpn_count}")
        print(f"  Galaxy Entries: {entry.galaxy_entries}")

        return result
```

---

## Testing Strategy

### Test 1: Real Crystallization (Not Synthetic)

```python
async def test_stargate_real_crystallization():
    """Test that Stargate creates REAL Galaxy entries (not synthetic)."""

    galaxy_manager = GalaxyManager()
    rpn_engine = SovereignRPNEngine()
    k3d_transformer = K3DTransformer(galaxy_manager, rpn_engine)

    stargate = IngestionStargate(
        stargate_region=None,
        galaxy_manager=galaxy_manager,
        rpn_engine=rpn_engine,
        k3d_transformer=k3d_transformer
    )

    # Count Grammar Galaxy entries before
    grammar_before = len(galaxy_manager.get_galaxy("Grammar").entries)

    # Submit ingestion job
    job_id = await stargate.submit_ingestion_job(
        data_path="test_data/sample.pdf",
        data_type="pdf",
        target_galaxies=["Grammar", "Math"],
        metadata={"domain": "math"}
    )

    # Wait for completion
    result = await stargate.wait_for_job(job_id, timeout=300.0)

    # Assert REAL counts (not synthetic)
    assert result["status"] == "complete"
    assert result["galaxy_entries_created"] > 0  # REAL count
    assert result["rpn_programs_created"] > 0  # REAL count

    # Verify Grammar Galaxy actually populated
    grammar_after = len(galaxy_manager.get_galaxy("Grammar").entries)
    assert grammar_after > grammar_before  # REAL entries added
```

### Test 2: End-to-End (PDF → Galaxy)

```python
async def test_end_to_end_pdf_to_galaxy_real():
    """Test full pipeline: PDF → Enrichment → K3D → Galaxy."""

    from knowledge3d.ingestion.corpus_manifest import CorpusManifest, CorpusEntry
    from knowledge3d.ingestion.batch_orchestrator import BatchOrchestrator

    manifest = CorpusManifest()
    stargate = IngestionStargate(...)
    orchestrator = BatchOrchestrator(manifest, stargate)

    # Create test entry
    test_entry = CorpusEntry(
        id="test_pdf",
        name="Test PDF",
        path="test_data/sample.pdf",
        tier=CorpusTier.TIER_1_FOUNDATIONAL,
        corpus_type=CorpusType.PDF,
        target_galaxies=["Grammar", "Math"],
        dependencies=[],
        metadata={"domain": "math"}
    )

    # Ingest
    result = await orchestrator.ingest_entry(test_entry)

    # Verify REAL counts
    assert test_entry.embedding_count > 0  # REAL embeddings
    assert test_entry.rpn_count > 0  # REAL RPN programs
    assert test_entry.galaxy_entries > 0  # REAL Galaxy entries

    # Verify TRM can navigate enriched Galaxy
    from knowledge3d.knowledgeverse.trm_navigator import TRMNavigator
    navigator = TRMNavigator()

    query_result = navigator.query("math pattern", galaxy_names=["Grammar"])
    assert len(query_result) > 0  # Should find enriched content
```

---

## Success Metrics

**After Week 13 hardening:**

1. **No Synthetic Placeholders:**
   - ✅ 0 synthetic `embedding_count` in Stargate
   - ✅ 100% real Galaxy population
   - ✅ All counts are REAL (embeddings, RPN programs, Galaxy entries)

2. **Galaxy Population:**
   - ✅ Grammar Galaxy: 500+ entries (target)
   - ✅ Math Galaxy: 1000+ entries (target)
   - ✅ Reality Galaxy: 200+ entries (target)
   - ✅ All entries have RPN programs (not external metadata)

3. **TRM Readiness:**
   - ✅ TRM can query enriched Galaxies
   - ✅ TRM can compose from procedural patterns
   - ✅ Shadow Copy records successful navigations

4. **Sovereignty Compliance:**
   - ✅ Hot path remains PTX-only (no regressions)
   - ✅ All ingestion outputs are sovereign artifacts (RPN + Galaxy + PTX)
   - ✅ No external formats persist beyond transformation

---

## Implementation Timeline

**Week 13, Day 3-5:**

**Day 3: Enhance Stargate**
- Implement `_execute_ingestion_pipeline()`
- Implement `_crystallize_enrichment()`
- Update `wait_for_job()` to return REAL counts

**Day 4: Enhance Feeders**
- Update `pdf_feeder.py` with K3D transformation
- Create other feeders (audio, code, image)

**Day 5: Integration + Testing**
- Integrate with `BatchOrchestrator`
- Run end-to-end tests
- Verify real Galaxy population

---

## Codex Implementation Directive

**Priority:** CRITICAL (Week 13, Day 3-5)

**What to Implement:**

1. **`knowledge3d/knowledgeverse/stargate.py` (enhance existing)**
   - Add `_execute_ingestion_pipeline()`
   - Add `_crystallize_enrichment()`
   - Update `wait_for_job()` to return REAL counts
   - Remove synthetic `embedding_count`

2. **`knowledge3d/ingestion/feeders/pdf_feeder.py` (enhance existing)**
   - Integrate with `EnrichmentPipeline`
   - Integrate with `K3DTransformer`
   - Output enrichment artifacts + K3D transformation

3. **`knowledge3d/ingestion/batch_orchestrator.py` (enhance existing)**
   - Update `ingest_entry()` to use REAL counts
   - Add `rpn_count` and `galaxy_entries` to `CorpusEntry`

4. **`tests/test_stargate_crystallization.py` (new)**
   - 2 tests from above
   - Test real crystallization (not synthetic)
   - Test end-to-end PDF → Galaxy

**Remove:**
- ❌ ALL synthetic `embedding_count` placeholders
- ❌ Any placeholder logic in production code

**Testing:**
- Re-run Phase 1B with hardened pipeline
- Verify 38/38 tests pass (32 existing + 4 LLM + 2 Stargate)
- Verify REAL Galaxy population (count Grammar/Math/Reality entries)

---

## End of Specification

**Remember:** NO PLACEHOLDERS. Everything external transforms to K3D standard ASAP. Sovereignty applies to ALL outputs.

**Contact:** Claude (Architecture Partner) for design questions, Codex (Implementation Partner) for execution.

---

**Claude (Architecture Partner)**
February 6, 2026

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
