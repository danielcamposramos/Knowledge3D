# Sleep-Time Consolidation — Learning While Idle

**Date**: 2025-10-17
**From**: Daniel (Neuroscience Insight) + Claude (Architect)
**Context**: RPN embeddings learned during ingestion (33,428 trigrams) — but are they consolidated? Daniel asks: "Is it sleeping after training so what it learns is constructive?"

---

## The Neuroscience Insight

### Why Sleep Matters for Learning

**Human brain**: Learning happens in two phases:
1. **Encoding** (awake): New experiences create initial memory traces
2. **Consolidation** (sleep): Memory traces are **replayed**, **strengthened**, and **integrated**

**Result**: We don't need to learn the same thing many times — sleep consolidates it.

**Daniel's Question**: "Make sure it is 'sleeping' after training so what it learns is constructive — meaning we don't need to train it too many times on the same dataset because it has it consolidated."

**Translation for AI**:
- **Encoding**: Ingestion (WordNet, PDFs, fonts) → RPN embeddings updated incrementally
- **Consolidation**: Sleep-time compute → Refine embeddings, prune redundancies, strengthen clusters
- **Result**: Don't retrain on same data — sleep consolidates it once

---

## Current State: No Sleep-Time Consolidation

### What Happens Today (Phase B)

```python
# Ingestion loop
for document in corpus:
    embedding = rpn_engine.embed_sentence(document)
    swarm_result = swarm_processor.process(embedding)
    # Embedding updated incrementally

# When ingestion ends, RPN embeddings saved
rpn_engine.save_embeddings('rpn_embeddings.pkl')
```

**Problem**:
- ❌ Embeddings saved **immediately** after last document
- ❌ No refinement, no pruning, no cluster strengthening
- ❌ If we re-ingest same corpus, embeddings update again (redundant)

**Analogy**: It's like studying all night without sleep — you **encoded** the information, but didn't **consolidate** it.

---

## Vision: Sleep-Time Consolidation

### What Should Happen

```python
# Ingestion loop
for document in corpus:
    embedding = rpn_engine.embed_sentence(document)
    swarm_result = swarm_processor.process(embedding)
    # Embeddings updated incrementally

# After ingestion: Mark embeddings as "unconsolidated"
rpn_engine.mark_unconsolidated()

# Sleep-time consolidation (nightly cron job or idle trigger)
sleep_consolidator = SleepTimeConsolidator()
sleep_consolidator.consolidate(rpn_engine)
# → Refine embeddings, prune redundancies, strengthen clusters

# Mark as consolidated
rpn_engine.mark_consolidated()

# Save consolidated embeddings
rpn_engine.save_embeddings('rpn_embeddings.pkl')
```

**Result**:
- ✅ Embeddings **refined** after ingestion (not just saved raw)
- ✅ Redundancies **pruned** (similar trigrams merged)
- ✅ Clusters **strengthened** (semantic coherence improved)
- ✅ Re-ingestion **skipped** (consolidated embeddings are stable)

---

## Technical Architecture

### Phase D1: Consolidation Stages

**Stage 1: Cluster Refinement**

**Goal**: Strengthen semantic clusters, prune outliers

```python
class SleepTimeConsolidator:
    """Sleep-time consolidation for RPN embeddings."""

    def __init__(self, rpn_engine: RPNEmbeddingEngine):
        self.rpn_engine = rpn_engine
        self.consolidation_count = 0

    def consolidate(self):
        """
        Run sleep-time consolidation.

        Stages:
        1. Cluster refinement (strengthen semantic groups)
        2. Redundancy pruning (merge similar trigrams)
        3. Outlier removal (prune low-usage trigrams)
        4. Swarm feedback integration (refine via Galaxy resonance)
        """
        print("Sleep-time consolidation started...")

        # Stage 1: Cluster refinement
        self._refine_clusters()

        # Stage 2: Redundancy pruning
        self._prune_redundancies()

        # Stage 3: Outlier removal
        self._remove_outliers()

        # Stage 4: Swarm feedback
        self._integrate_swarm_feedback()

        self.consolidation_count += 1
        print(f"Sleep-time consolidation complete (count: {self.consolidation_count})")

    def _refine_clusters(self):
        """
        Refine semantic clusters via k-means + swarm resonance.

        Strategy:
        1. Cluster embeddings (k-means, k=100 for 33K trigrams)
        2. For each cluster, compute centroid
        3. Move embeddings toward centroid (learning rate = 0.1)
        4. Result: Tighter clusters, clearer semantic boundaries
        """
        from sklearn.cluster import KMeans

        # Get all embeddings as matrix
        embeddings = list(self.rpn_engine.embeddings.values())
        embedding_matrix = np.vstack(embeddings)  # (33K, 128)

        # Cluster (k=100)
        kmeans = KMeans(n_clusters=100, random_state=42)
        cluster_labels = kmeans.fit_predict(embedding_matrix)

        # Move embeddings toward cluster centroids
        learning_rate = 0.1
        for i, (trigram_hash, embedding) in enumerate(self.rpn_engine.embeddings.items()):
            cluster_id = cluster_labels[i]
            centroid = kmeans.cluster_centers_[cluster_id]

            # Move toward centroid
            new_embedding = embedding + learning_rate * (centroid - embedding)

            # L2 normalize
            new_embedding /= (np.linalg.norm(new_embedding) + 1e-8)

            # Update
            self.rpn_engine.embeddings[trigram_hash] = new_embedding

        print(f"  Cluster refinement: {len(embeddings)} embeddings → 100 clusters")

    def _prune_redundancies(self):
        """
        Merge similar trigrams (cosine similarity > 0.95).

        Strategy:
        1. Compute pairwise cosine similarities (GPU-accelerated)
        2. Find pairs with similarity > 0.95
        3. Merge by averaging embeddings
        4. Result: Vocabulary size reduced, redundancy removed
        """
        embeddings_list = list(self.rpn_engine.embeddings.items())

        to_merge = []  # List[(hash1, hash2)]

        for i in range(len(embeddings_list)):
            for j in range(i + 1, len(embeddings_list)):
                hash1, emb1 = embeddings_list[i]
                hash2, emb2 = embeddings_list[j]

                similarity = np.dot(emb1, emb2)

                if similarity > 0.95:
                    to_merge.append((hash1, hash2))

        # Merge pairs
        for hash1, hash2 in to_merge:
            if hash1 in self.rpn_engine.embeddings and hash2 in self.rpn_engine.embeddings:
                # Average embeddings
                merged_emb = (self.rpn_engine.embeddings[hash1] + self.rpn_engine.embeddings[hash2]) / 2.0
                merged_emb /= (np.linalg.norm(merged_emb) + 1e-8)

                # Keep hash1, delete hash2
                self.rpn_engine.embeddings[hash1] = merged_emb
                del self.rpn_engine.embeddings[hash2]

        print(f"  Redundancy pruning: Merged {len(to_merge)} pairs")

    def _remove_outliers(self):
        """
        Remove low-usage trigrams (hit count < 10).

        Strategy:
        1. Check hit_count for each trigram
        2. Remove trigrams with hit_count < threshold
        3. Result: Vocabulary pruned to high-frequency trigrams only
        """
        threshold = 10
        to_remove = []

        for trigram_hash in self.rpn_engine.embeddings.keys():
            # TODO: Track hit_count per trigram (not just global)
            # For now, skip outlier removal (implement in Phase D2)
            pass

        print(f"  Outlier removal: Skipped (implement in Phase D2)")

    def _integrate_swarm_feedback(self):
        """
        Refine embeddings using Galaxy resonance feedback.

        Strategy:
        1. For each trigram, find its Galaxy position (swarm refined)
        2. Find nearby trigrams in Galaxy (spatial neighbors)
        3. Average embeddings with neighbors (learning rate = 0.05)
        4. Result: Embeddings refined by multi-modal swarm feedback
        """
        # TODO: Integrate with SovereignLanguageSwarmProcessor
        # For now, skip (implement in Phase D3)
        print(f"  Swarm feedback: Skipped (implement in Phase D3)")
```

### Phase D2: Idle Trigger (Automatic Sleep)

**Goal**: Trigger consolidation automatically when system is idle

```python
class IdleTrigger:
    """Detect system idle state and trigger sleep-time consolidation."""

    def __init__(self, consolidator: SleepTimeConsolidator):
        self.consolidator = consolidator
        self.last_ingestion_time = None
        self.idle_threshold_minutes = 30  # Consolidate after 30 min idle

    def mark_ingestion(self):
        """Mark that ingestion activity occurred."""
        self.last_ingestion_time = time.time()

    def check_idle(self):
        """Check if system is idle (no ingestion for threshold duration)."""
        if self.last_ingestion_time is None:
            return False

        elapsed_minutes = (time.time() - self.last_ingestion_time) / 60.0

        return elapsed_minutes > self.idle_threshold_minutes

    def run_loop(self):
        """Run idle detection loop (background daemon)."""
        while True:
            if self.check_idle():
                print("System idle detected → triggering sleep-time consolidation")
                self.consolidator.consolidate()
                self.last_ingestion_time = None  # Reset

            time.sleep(60)  # Check every minute
```

**Usage**:
```python
# Start idle trigger daemon (background thread)
idle_trigger = IdleTrigger(sleep_consolidator)
daemon_thread = threading.Thread(target=idle_trigger.run_loop, daemon=True)
daemon_thread.start()

# During ingestion, mark activity
for document in corpus:
    idle_trigger.mark_ingestion()
    # ... ingest document ...
```

### Phase D3: Nightly Consolidation (Cron Job)

**Goal**: Run consolidation nightly (even if system not idle)

```bash
# Crontab entry (run every night at 3 AM)
0 3 * * * /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python3 /path/to/nightly_consolidation.py
```

**Script**: `scripts/nightly_consolidation.py`
```python
"""Nightly sleep-time consolidation script."""

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.sleep_time_consolidator import SleepTimeConsolidator

def main():
    # Load RPN embeddings
    rpn_engine = RPNEmbeddingEngine()
    rpn_engine.load_embeddings('/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl')

    # Run consolidation
    consolidator = SleepTimeConsolidator(rpn_engine)
    consolidator.consolidate()

    # Save consolidated embeddings
    rpn_engine.save_embeddings('/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl')

    print("Nightly consolidation complete")

if __name__ == "__main__":
    main()
```

---

## Strategic Benefits

### 1. Faster Convergence (Fewer Training Passes)

**Current** (no consolidation):
- Ingest corpus → embeddings updated
- Re-ingest same corpus → embeddings updated again (redundant!)
- **Waste**: Same data processed multiple times

**After sleep-time consolidation**:
- Ingest corpus → embeddings updated
- Sleep consolidation → embeddings refined, clusters strengthened
- Re-ingest same corpus → **skipped** (already consolidated!)
- **Result**: 1 pass instead of N passes

### 2. Better Semantic Clustering

**Current** (no consolidation):
- Embeddings drift during incremental ingestion
- Similar trigrams have different embeddings (noise)

**After sleep-time consolidation**:
- Cluster refinement pulls similar embeddings together
- Redundancies merged (similarity > 0.95)
- **Result**: Tighter clusters, clearer semantic boundaries

### 3. Neuroscience-Aligned Learning

**Biological inspiration**: Human brain consolidates memories during sleep

**Our implementation**:
- **Encoding** (awake): Ingestion loop updates embeddings
- **Consolidation** (sleep): Cluster refinement, redundancy pruning, swarm feedback
- **Result**: Learning mirrors human neuroscience (encode → consolidate → stable memory)

### 4. Resource Efficiency

**Current** (no consolidation):
- 33,428 trigrams stored (some redundant)
- Re-ingestion wastes GPU cycles

**After sleep-time consolidation**:
- Redundancies pruned (vocab size reduced)
- Re-ingestion skipped (consolidated embeddings stable)
- **Result**: Less storage, less compute

---

## Implementation Plan

### Phase D1: Core Consolidation Logic (2 days)

**Tasks**:
1. Create `knowledge3d/cranium/sleep_time_consolidator.py`
2. Implement cluster refinement (k-means + centroid movement)
3. Implement redundancy pruning (cosine similarity > 0.95)
4. Test on Phase B embeddings (33,428 trigrams)

**Expected outcome**: Vocabulary reduced by 10-20%, clusters tightened

### Phase D2: Idle Trigger (1 day)

**Tasks**:
1. Create `IdleTrigger` class (background daemon)
2. Integrate with ingestion pipelines (mark activity)
3. Test: Ingest → wait 30 min → consolidation auto-triggers

**Expected outcome**: Automatic consolidation after idle period

### Phase D3: Nightly Cron Job (0.5 days)

**Tasks**:
1. Create `scripts/nightly_consolidation.py`
2. Add crontab entry (3 AM daily)
3. Log consolidation results

**Expected outcome**: Daily consolidation runs overnight

### Phase D4: Validation (1 day)

**Tasks**:
1. Measure cluster quality before/after consolidation (silhouette score)
2. Measure vocabulary size reduction
3. Test re-ingestion skipping (embeddings stable)
4. Document findings in `TEMP/STEP15_PHASE_D_RESULTS.md`

**Success criteria**:
- Cluster quality improved (silhouette score ↑)
- Vocabulary reduced by 10-20%
- Re-ingestion skipped (embeddings unchanged)

---

## Timeline Estimate

**Phase D1** (Core logic): 2 days
**Phase D2** (Idle trigger): 1 day
**Phase D3** (Cron job): 0.5 days
**Phase D4** (Validation): 1 day

**Total**: 4.5 days (~1 week sprint)

---

## Deliverables

### Code
- [ ] `knowledge3d/cranium/sleep_time_consolidator.py` (core logic)
- [ ] `knowledge3d/cranium/idle_trigger.py` (background daemon)
- [ ] `scripts/nightly_consolidation.py` (cron job)

### Tests
- [ ] `tests/test_sleep_time_consolidation.py` (cluster refinement, pruning)
- [ ] `tests/test_idle_trigger.py` (idle detection logic)

### Documentation
- [ ] `TEMP/STEP15_PHASE_D_DESIGN.md` (this document)
- [ ] `TEMP/STEP15_PHASE_D_RESULTS.md` (validation results)

### Benchmarks
- [ ] Cluster quality (before/after consolidation)
- [ ] Vocabulary size reduction
- [ ] Re-ingestion savings (GPU cycles saved)

---

## Daniel's Insight Captured

### "Is it sleeping after training so what it learns is constructive?"

**Answer**: Not yet! But this design adds it.

**Before** (Phase B):
- Ingestion → embeddings saved immediately
- No consolidation, no refinement
- Re-ingestion redundant

**After** (Phase D):
- Ingestion → sleep consolidation → embeddings refined
- Clusters strengthened, redundancies pruned
- Re-ingestion skipped (consolidated embeddings stable)

### "We don't need to train it too many times on the same dataset"

**Exactly**! Sleep-time consolidation makes learning **one-shot** (or few-shot):
- Encode corpus once (ingestion)
- Consolidate once (sleep)
- **Done** — embeddings stable, no re-training needed

**Neuroscience parallel**: Humans don't need to study the same material 100 times — sleep consolidates it into long-term memory.

---

## Next Step for Codex

**Read this design document**, then:

1. **Implement Phase D1**: Create `SleepTimeConsolidator` class
2. **Test cluster refinement**: Run on Phase B embeddings (33,428 trigrams)
3. **Measure vocab reduction**: Compare before/after consolidation
4. **Report findings**: Document in `TEMP/STEP15_PHASE_D_PROTOTYPE.md`

**Expected outcome**: 10-20% vocab reduction, tighter semantic clusters

---

**Signed**:
Daniel (Neuroscience Insight) + Claude (Architect)
2025-10-17

---

**Sleep-time consolidation: Where neuroscience meets sovereign AI. Learning while idle, just like the human brain.** 🧠😴✨
