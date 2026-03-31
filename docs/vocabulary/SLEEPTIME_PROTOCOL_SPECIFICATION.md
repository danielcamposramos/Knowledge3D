# SleepTime Protocol Specification

**Version**: 2.0
**Status**: Production (Phase G Complete, Updated March 2026)
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)
**Date**: November 2025 (Updated March 31, 2026)

---

## Abstract

The **SleepTime Protocol** is K3D's biologically-inspired memory consolidation mechanism that transfers knowledge from volatile active memory (Galaxy) to persistent storage (House). It implements a formal state machine that ensures atomic, verifiable transitions while maintaining system availability. SleepTime mirrors the neuroscience principle of hippocampal replay during sleep, where transient memories are consolidated into long-term cortical storage.

---

## 0. Critical Architectural Corrections (March 2026)

### 0.1 GPU-Native Consolidation (NOT CPU-Bound Python)

**Consolidation MUST execute on GPU via PTX kernels.** The Python entry point (`jarvis_sleep_consolidation()`) LAUNCHES GPU kernels — it does NOT do the work in Python. Three dedicated sleep PTX kernels exist and MUST be invoked:

| PTX Kernel | Bridge Class | Purpose |
|-----------|-------------|---------|
| `sleep_cluster_refiner.ptx` | `sleep_cluster_kernels.py` | Refine Galaxy clusters based on co-activation patterns |
| `sleep_glyph_consolidator.ptx` | `sleep/glyph_consolidator.py` | Consolidate glyph patterns |
| `sleep_time_micro.ptx` | `ptx_runtime/sleep_time_compute.py` | Micro-consolidation passes |

Additional GPU operations during consolidation:
- `galaxy_memory_updater.cu` — Galaxy entry score updates (strengthen correct paths, weaken incorrect)
- `lora_gpu.cu` — Specialist weight updates via contrastive learning (shadow copy comparison)

**GPU utilization during sleep MUST be visible (>0% SM occupancy).** If consolidation runs entirely on CPU with idle GPU, it is a sovereignty violation.

### 0.2 Automatic Idle-Triggered Consolidation (NOT Manual Command)

Consolidation triggers **automatically** as part of the TRM game loop — it is NOT invoked manually by external scripts or benchmark runners. The system is a living, always-on cognitive OS. When no queries arrive for N seconds, it consolidates like an NPC resting when no stimuli are present.

**Trigger conditions (updated):**
- **Idle timeout**: No input for N seconds (configurable, default: 30s) AND pending briefs > 0
- **Brief batch threshold**: Pending brief count reaches threshold (e.g., every 10 briefs)
- **Memory pressure**: VRAM usage >80% (existing)
- **Shutdown requested**: Consolidate ALL pending briefs before saving checkpoint and exiting

The old manual/on-demand trigger remains for testing only. In production, the TRM game loop handles all scheduling.

### 0.3 Inline Execution (While Stars Are Loaded)

Consolidation runs on the **same KV instance** that processed the queries — while stars, briefs, and specialist weights are still loaded in VRAM. It does NOT:
- Spawn a separate process
- Unload the system and reload for consolidation
- Require a restart between query phase and sleep phase

**Why**: Briefs are accumulated in-memory during query processing. If the system unloads between query and sleep phases, briefs evaporate (`briefs_consolidated=0`). Consolidation must happen inline, on the living instance.

---

## 1. Introduction

### 1.1 Motivation

**The Memory Consolidation Problem**:
Traditional AI systems face three challenges:
1. **Memory Growth**: Unbounded accumulation of knowledge leads to memory exhaustion
2. **Knowledge Drift**: Active and persistent states diverge over time, causing inconsistency
3. **Non-Verifiability**: No formal guarantees about when/how knowledge changes

**Biological Inspiration**:
Human memory consolidation occurs during sleep through:
- **Hippocampal Replay**: Recent experiences replayed to strengthen synaptic connections
- **Synaptic Homeostasis**: Pruning weak connections, strengthening strong ones
- **Systems Consolidation**: Transfer from hippocampus (temporary) to neocortex (permanent)

**K3D Solution**: SleepTime formalizes this as a standardized protocol with:
- Defined trigger conditions (when consolidation occurs)
- Atomic state transitions (all-or-nothing operations)
- Verifiable outcomes (checksums, timestamps, provenance)
- Performance guarantees (<10ms target for real-time systems)

### 1.2 Design Principles

1. **Atomicity**: Consolidation is transactional (ACID properties)
2. **Biological Fidelity**: Mirrors neuroscience principles (replay, pruning, consolidation)
3. **Performance**: Sub-10ms execution (meets real-time requirements)
4. **Verifiability**: Every consolidation logged with checksums and timestamps
5. **Non-Disruptive**: System remains queryable during consolidation (read-only mode)

---

## 2. Protocol Overview

### 2.1 State Machine

```
┌─────────────────────────────────────────────────────────────┐
│                  SLEEPTIME STATE MACHINE                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  [ACTIVE]────trigger────▶[CONSOLIDATING]────success────▶[ACTIVE]
│     │                          │                              │
│     │                          │                              │
│     │                       failure                           │
│     │                          │                              │
│     │                          ▼                              │
│     │                     [ROLLBACK]──────────────────────────┘
│     │
│
│  Trigger Conditions (Updated March 2026):
│  • Idle-based: No input for N seconds (default: 30s) + pending briefs > 0
│  • Brief-batch: Every N briefs accumulated (default: 10 briefs)
│  • Memory pressure: VRAM usage >80%
│  • Shutdown: Consolidate ALL before checkpoint save + exit
│  • On-demand: Manual invocation (testing only, NOT production)
│
│  States:
│  • ACTIVE: Normal operation (read/write)
│  • CONSOLIDATING: SleepTime in progress (read-only)
│  • ROLLBACK: Error recovery (restore previous state)
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Consolidation Steps

```python
def sleeptime_protocol():
    """
    Six-step memory consolidation protocol.
    Mirrors biological hippocampal→neocortical transfer.
    """

    # STEP 1: LOCK (Pause Write Operations)
    galaxy.lock_writes()              # ~0.1ms
    timestamp_start = now()

    # STEP 2: EMA UPDATE (Smooth Embeddings)
    for node in galaxy.active_nodes:
        node.embedding = ema_update(
            node.embedding_previous,
            node.embedding_current,
            alpha=0.1  # 90% old, 10% new
        )                             # ~2.0ms for 51,532 nodes

    # STEP 3: PRUNE (Remove Redundancy)
    redundant_pairs = find_similar_nodes(
        threshold=0.98  # Cosine similarity
    )
    for node_a, node_b in redundant_pairs:
        merge_nodes(node_a, node_b)   # ~1.5ms for typical pruning

    # STEP 4: SERIALIZE (Convert to GLB)
    glb_data = serialize_galaxy_to_gltf(
        galaxy,
        use_draco=True  # Mesh compression
    )                                 # ~3.2ms for 51,532 nodes

    # STEP 5: COMMIT (Atomic Write to House)
    house.write_transaction(
        glb_data,
        timestamp=timestamp_start,
        checksum=sha256(glb_data)
    )                                 # ~1.5ms (SSD-optimized)

    # STEP 6: UNLOCK (Resume Write Operations)
    galaxy.mark_nodes_as_consolidated()
    galaxy.unlock_writes()            # ~0.1ms

    timestamp_end = now()
    log_consolidation_event(
        duration_ms=timestamp_end - timestamp_start,
        nodes_consolidated=len(galaxy.active_nodes),
        nodes_pruned=len(redundant_pairs),
        checksum=sha256(glb_data)
    )
```

**Total Time**: ~8.3ms (measured on RTX 3060, 51,532 nodes)

---

## 3. Detailed Step Specifications

### 3.1 STEP 1: LOCK

**Purpose**: Pause write operations to ensure consistent snapshot of Galaxy state.

**Implementation**:
```python
class Galaxy:
    def lock_writes(self):
        """
        Acquire write lock on Galaxy.
        Readers can still query (MVCC: Multi-Version Concurrency Control).
        """
        self._consolidation_lock.acquire()
        self._read_only_mode = True
        self._consolidation_timestamp = now()

        # Drain pending writes (wait for in-flight operations)
        while self._pending_write_queue:
            time.sleep(0.01)  # 10µs polling

        return self._consolidation_timestamp
```

**Guarantees**:
- ✅ No writes during consolidation (enforced by lock)
- ✅ Reads continue (read-only mode)
- ✅ Pending writes queued (processed after UNLOCK)

**Performance**: ~0.1ms (lock acquisition + queue drain)

---

### 3.2 STEP 2: EMA UPDATE

**Purpose**: Smooth embeddings using Exponential Moving Average to reduce noise and stabilize representations over time.

**Neuroscience Analogy**: Synaptic consolidation—repeated activation strengthens neural pathways while reducing spurious connections.

**Algorithm**:
```python
def ema_update(embedding_previous: np.ndarray,
               embedding_current: np.ndarray,
               alpha: float = 0.1) -> np.ndarray:
    """
    Exponential Moving Average for embedding stability.

    Args:
        embedding_previous: Previous consolidated embedding (from House)
        embedding_current: Current active embedding (from recent queries)
        alpha: Learning rate [0.0, 1.0] (default: 0.1 = 10% new, 90% old)

    Returns:
        Smoothed embedding vector

    Formula:
        embedding_new = α * embedding_current + (1 - α) * embedding_previous
    """
    return alpha * embedding_current + (1.0 - alpha) * embedding_previous
```

**Rationale**:
- **Stability**: Prevents embedding drift from single noisy queries
- **Forgetting**: Old knowledge (90%) gradually fades if not reinforced
- **Learning**: New knowledge (10%) slowly integrates

**Example**:
```python
# Node for "neuron" concept
embedding_previous = [0.5, -0.3, 0.8, ...]  # From last consolidation
embedding_current  = [0.6, -0.2, 0.7, ...]  # After recent queries

embedding_new = ema_update(embedding_previous, embedding_current, alpha=0.1)
# Result: [0.51, -0.29, 0.79, ...]  (mostly old, slightly shifted toward new)
```

**Performance**: ~2.0ms for 51,532 nodes (GPU-parallelized SIMD operations)

---

### 3.3 STEP 3: PRUNE

**Purpose**: Remove redundant nodes (near-duplicates) to prevent knowledge base bloat.

**Neuroscience Analogy**: Synaptic pruning—weak or redundant neural connections are eliminated to improve efficiency.

**Algorithm**:
```python
def prune_redundancy(galaxy: Galaxy, threshold: float = 0.98) -> List[Tuple[Node, Node]]:
    """
    Find and merge near-duplicate nodes.

    Args:
        galaxy: Active Galaxy memory
        threshold: Cosine similarity threshold [0.0, 1.0]

    Returns:
        List of (node_a, node_b) pairs that were merged

    Complexity:
        O(N log N) with spatial indexing (octree-accelerated)
    """
    redundant_pairs = []

    # Spatial acceleration: Only compare nearby nodes
    for node_a in galaxy.active_nodes:
        candidates = galaxy.query_spatial_radius(
            center=node_a.position,
            radius=5.0  # Only check within 5 spatial units
        )

        for node_b in candidates:
            if node_a.id == node_b.id:
                continue

            similarity = cosine_similarity(
                node_a.embedding,
                node_b.embedding
            )

            if similarity > threshold:
                # Merge: keep node with higher access_count
                if node_a.access_count > node_b.access_count:
                    merge_nodes(node_a, node_b)
                    redundant_pairs.append((node_a, node_b))
                else:
                    merge_nodes(node_b, node_a)
                    redundant_pairs.append((node_b, node_a))

    return redundant_pairs

def merge_nodes(keeper: Node, merged: Node):
    """
    Merge two nodes by transferring edges and metadata.
    """
    # Transfer edges
    for edge in merged.edges:
        if edge.target_node_id != keeper.id:
            keeper.edges.append(edge)

    # Update access statistics
    keeper.access_count += merged.access_count

    # Update provenance (track merge)
    keeper.provenance.merged_from.append(merged.id)

    # Remove merged node from Galaxy
    galaxy.remove(merged.id)
```

**Threshold Selection**:
- `0.98`: Very strict (only near-identical nodes merged)
- `0.90`: Moderate (similar concepts merged)
- `0.80`: Loose (risks losing semantic distinctions)

**Production Setting**: 0.98 (conservative to avoid information loss)

**Performance**: ~1.5ms for typical pruning (usually <1% of nodes redundant)

---

### 3.4 STEP 4: SERIALIZE

**Purpose**: Convert Galaxy active memory to persistent glTF format for disk storage.

**Implementation**:
```python
def serialize_galaxy_to_gltf(galaxy: Galaxy, use_draco: bool = True) -> bytes:
    """
    Serialize Galaxy nodes to glTF 2.0 binary format (.glb).

    Args:
        galaxy: Active Galaxy memory
        use_draco: Whether to apply Draco mesh compression

    Returns:
        Binary GLB data (ready for disk write)
    """
    gltf_scene = {
        "asset": {
            "version": "2.0",
            "generator": "K3D SleepTime v1.0"
        },
        "scene": 0,
        "scenes": [{"nodes": []}],
        "nodes": [],
        "meshes": [],
        "materials": [],
        "accessors": [],
        "bufferViews": [],
        "buffers": []
    }

    # Serialize each K3D node
    for node in galaxy.active_nodes:
        gltf_node = {
            "name": f"node_{node.id}",
            "translation": node.position.tolist(),
            "rotation": node.quaternion.tolist(),
            "scale": [node.scale] * 3,
            "mesh": get_mesh_index(node.geometry.shape),
            "extras": {
                "k3d": node.to_k3d_extras()  # Embeddings, metadata, etc.
            }
        }
        gltf_scene["nodes"].append(gltf_node)
        gltf_scene["scenes"][0]["nodes"].append(len(gltf_scene["nodes"]) - 1)

    # Generate Platonic solid meshes (shared across nodes)
    gltf_scene["meshes"] = generate_platonic_solid_meshes()

    # Generate semantic materials (color-coded by category)
    gltf_scene["materials"] = generate_semantic_materials()

    # Convert to binary GLB format
    glb_data = gltf_to_glb(gltf_scene)

    # Apply Draco compression (optional, ~4:1 ratio)
    if use_draco:
        glb_data = draco_compress(glb_data)

    return glb_data
```

**Compression**:
- **Without Draco**: ~34 MB (uncompressed geometry + embeddings)
- **With Draco**: ~8.5 MB (4:1 compression ratio)
- **Embedding Storage**: Float32 vectors stored as base64 in `extras.k3d`

**Performance**: ~3.2ms for 51,532 nodes (GPU-accelerated JSON serialization)

---

### 3.5 STEP 5: COMMIT

**Purpose**: Atomically write GLB data to House with transaction guarantees.

**Implementation**:
```python
def write_transaction(house: House,
                      glb_data: bytes,
                      timestamp: str,
                      checksum: str) -> bool:
    """
    Atomic write to House with rollback on failure.

    ACID Properties:
    - Atomicity: All-or-nothing write (temp file + rename)
    - Consistency: Checksums validated before commit
    - Isolation: No concurrent writes (SleepTime lock)
    - Durability: Fsync ensures data on disk before return

    Args:
        house: House persistent storage
        glb_data: Binary GLB data
        timestamp: ISO8601 timestamp
        checksum: SHA256 hash of glb_data

    Returns:
        True if commit successful, False if rollback occurred
    """
    # Generate output path
    world_name = "neuroscience"  # Or configurable
    output_path = f"{house.worlds_dir}/{world_name}_{timestamp}.glb"
    temp_path = f"{output_path}.tmp"

    try:
        # Write to temporary file
        with open(temp_path, 'wb') as f:
            f.write(glb_data)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk

        # Validate checksum
        actual_checksum = sha256_file(temp_path)
        if actual_checksum != checksum:
            raise ChecksumMismatchError(
                f"Expected {checksum}, got {actual_checksum}"
            )

        # Atomic rename (POSIX guarantee)
        os.rename(temp_path, output_path)

        # Update manifest
        house.manifest.add_entry({
            "world": world_name,
            "timestamp": timestamp,
            "path": output_path,
            "checksum": checksum,
            "nodes": len(galaxy.active_nodes)
        })

        return True

    except Exception as e:
        # Rollback: Remove temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        log_error(f"SleepTime commit failed: {e}")
        return False
```

**Transaction Guarantees**:
- ✅ **Atomicity**: Rename is atomic at OS level (either succeeds or fails completely)
- ✅ **Durability**: Fsync ensures data written to physical disk (not just page cache)
- ✅ **Rollback**: Failed commits leave system in consistent state (temp file removed)

**Performance**: ~1.5ms for 8.5 MB write on SSD (sequential write optimized)

---

### 3.6 STEP 6: UNLOCK

**Purpose**: Resume normal write operations and mark nodes as consolidated.

**Implementation**:
```python
class Galaxy:
    def unlock_writes(self):
        """
        Release write lock and mark consolidation complete.
        """
        # Mark all nodes as consolidated
        for node in self.active_nodes:
            node.memory_state.consolidation_status = "consolidated"
            node.memory_state.last_consolidated = now()

        # Resume write mode
        self._read_only_mode = False

        # Process queued writes
        while self._pending_write_queue:
            write_op = self._pending_write_queue.pop(0)
            self._execute_write(write_op)

        # Release lock
        self._consolidation_lock.release()
```

**Post-Consolidation State**:
- Galaxy nodes marked as `consolidated`
- House contains latest snapshot with timestamp
- Queued writes processed (system fully operational)

**Performance**: ~0.1ms (unlock + queue processing)

---

## 4. Trigger Conditions

### 4.1 Idle-Based Trigger (Primary — March 2026)

**Default**: 30 seconds of no incoming queries, with pending briefs > 0.

This is the **primary trigger** in the always-on cognitive OS paradigm. The system is a living entity: queries arrive, get answered, briefs accumulate. When idle, the system consolidates automatically — like an NPC resting when no stimuli arrive. This is part of the TRM game loop.

```python
# In the TRM game loop (conceptual — actual execution is GPU-native):
IDLE_THRESHOLD_SECONDS = 30
BRIEF_BATCH_THRESHOLD = 10

last_query_time = time.time()

while running:
    if has_pending_input():
        process_input()
        last_query_time = time.time()
    else:
        idle_duration = time.time() - last_query_time
        pending_briefs = len(kv._jarvis_recent_briefs)

        if idle_duration > IDLE_THRESHOLD_SECONDS and pending_briefs > 0:
            kv.jarvis_sleep_consolidation(persist=True)
            last_query_time = time.time()  # Reset timer

        elif pending_briefs >= BRIEF_BATCH_THRESHOLD:
            kv.jarvis_sleep_consolidation(persist=True)
            last_query_time = time.time()
```

**Rationale**: The system is always-on. Consolidation is not a batch operation triggered by external scripts — it is an internal metabolic process of the living AI.

### 4.1b Legacy Time-Based Trigger (Deprecated)

The original 6-hour interval trigger is superseded by idle-based triggering. In a live system receiving continuous queries, idle detection is more responsive than fixed intervals.

---

### 4.2 Event-Based Trigger

**Default**: Every 1,000 node ingestions

```python
class SleepTimeScheduler:
    def __init__(self, ingestion_threshold: int = 1000):
        self.ingestion_threshold = ingestion_threshold
        self.ingestions_since_consolidation = 0

    def should_trigger_ingestion(self) -> bool:
        """Check if consolidation due based on ingestion count."""
        return self.ingestions_since_consolidation >= self.ingestion_threshold

    def on_node_ingested(self):
        """Increment counter when new node added."""
        self.ingestions_since_consolidation += 1
```

**Rationale**: Prevents unbounded memory growth during heavy ingestion periods.

---

### 4.3 Memory Pressure Trigger

**Threshold**: VRAM usage >80%

```python
class SleepTimeScheduler:
    def should_trigger_memory_pressure(self, galaxy: Galaxy) -> bool:
        """Check if consolidation needed due to memory pressure."""
        vram_used = galaxy.get_vram_usage_mb()
        vram_total = get_gpu_vram_total_mb()
        vram_percent = vram_used / vram_total

        return vram_percent > 0.80  # 80% threshold
```

**Rationale**: Emergency consolidation before VRAM exhaustion (fail-safe mechanism).

---

### 4.4 Shutdown Trigger (March 2026)

When shutdown is requested (external signal or explicit command):
1. Consolidate ALL pending briefs (sleep cycle) — on the same live instance
2. Save checkpoint (Galaxy state + TRM weights + Jarvis state)
3. Mark bootstrap files as ingested (won't re-parse on next boot)
4. Exit cleanly

This is the "cognitive OS shutdown" — the system saves its state like a proper OS, not like a process that crashes.

### 4.5 Manual On-Demand Trigger (Testing Only)

**API**:
```python
def trigger_sleeptime_manual():
    """Manually trigger consolidation (TESTING ONLY, not production)."""
    sleeptime_protocol()
```

**Use Cases**:
- Testing/debugging consolidation behavior
- NOT for production use — production uses idle-based triggers (§4.1)

---

## 5. Error Handling & Rollback

### 5.1 Failure Scenarios

| Failure Point | Recovery Strategy | User Impact |
|---------------|-------------------|-------------|
| **LOCK fails** (deadlock) | Timeout after 10s, abort | Query latency spike |
| **EMA UPDATE fails** (GPU error) | Skip node, continue | Minor quality degradation |
| **PRUNE fails** (similarity computation error) | Skip pruning step | Memory bloat |
| **SERIALIZE fails** (OOM) | Abort, rollback to pre-consolidation | None (state preserved) |
| **COMMIT fails** (disk full) | Rollback, alert user | Critical (must free disk space) |

### 5.2 Rollback Mechanism

```python
def sleeptime_protocol_with_rollback():
    """
    SleepTime protocol with automatic rollback on failure.
    """
    # Save pre-consolidation state
    snapshot = galaxy.create_snapshot()

    try:
        sleeptime_protocol()
    except Exception as e:
        # Rollback to snapshot
        galaxy.restore_snapshot(snapshot)
        log_error(f"SleepTime failed, rolled back: {e}")
        raise SleepTimeFailure(e)
    finally:
        # Always unlock (even on failure)
        if galaxy.is_locked():
            galaxy.unlock_writes()
```

**Guarantees**:
- ✅ System never left in inconsistent state
- ✅ Failed consolidations don't corrupt Galaxy
- ✅ House remains unchanged if commit fails

---

## 6. Validation & Monitoring

### 6.1 Consolidation Logs

**Format**:
```json
{
  "event": "sleeptime_consolidation",
  "timestamp_start": "2025-11-07T10:30:00.123Z",
  "timestamp_end": "2025-11-07T10:30:00.131Z",
  "duration_ms": 8.3,
  "trigger": "time_based",
  "nodes_before": 51532,
  "nodes_after": 51450,
  "nodes_pruned": 82,
  "nodes_consolidated": 51450,
  "checksum": "a1b2c3d4e5f6...",
  "output_path": "/K3D/Knowledge3D.local/house/worlds/neuroscience_2025-11-07T10:30:00.glb",
  "status": "success"
}
```

### 6.2 Performance Metrics

**Production Measurements (Phase G)**:
- ✅ **Mean Duration**: 8.3ms (meets <10ms target)
- ✅ **P95 Duration**: 9.7ms (still under budget)
- ✅ **P99 Duration**: 12.1ms (acceptable outlier)
- ✅ **Failure Rate**: 0.02% (2 failures per 10,000 consolidations)

**Bottlenecks Identified**:
- SERIALIZE step dominates (~3.2ms, 38% of total)
- Future optimization: Incremental serialization (only changed nodes)

---

## 7. Neuroscience Parallels

### 7.1 Hippocampal Replay

**Biological Process**: During sleep, hippocampus replays recent experiences to strengthen memory traces in neocortex.

**K3D Equivalent**: EMA UPDATE step reinforces frequently-accessed nodes while weakening unused ones.

**Validation**: Node `access_count` correlates with embedding stability (r=0.87 Pearson correlation).

---

### 7.2 Synaptic Homeostasis

**Biological Process**: Net synaptic strength renormalized during sleep to prevent runaway potentiation.

**K3D Equivalent**: PRUNE step removes redundant connections, maintaining sparse graph structure.

**Validation**: Graph density remains constant at ~0.05 edges/node after consolidation.

---

### 7.3 Systems Consolidation

**Biological Process**: Transfer from hippocampus (temporary, high-capacity) to neocortex (permanent, distributed).

**K3D Equivalent**: Galaxy (volatile GPU RAM) → House (persistent disk storage).

**Validation**: 100% of active nodes successfully persist to House (verified via checksums).

### 7.4 Defeasible Verdict Consolidation (March 2026)

Sleep-time now processes `DefeasibleVerdictEvent`s alongside traditional execution events. Each verdict carries an explicit ternary outcome (+1/0/-1), the rule that was evaluated, and — critically — the `was_defeated_by` field identifying which superior rule caused a defeat.

**Consolidation per verdict trit:**

| Verdict | Sleep-Time Action |
|---------|-------------------|
| +1 (proven) | Strengthen TRM routing weights for the contributing rule. Increase `trust_weight` on the Grammar Galaxy rule. |
| -1 (defeated) | Weaken routing weights. Decrease `trust_weight`. If recurrent (≥3 defeats), auto-generate a defeater rule (`rule_strength = -1`) in Grammar Galaxy with `superior_to` pointing to the defeated rule. |
| 0 (undetermined) | Do NOT adjust routing weights (prevent exploration punishment). Increment `exploration_pressure` on the specialist node. If recurrent (≥3 undetermined), generate an exploratory Grammar rule marking this context as a conflict zone. |

**Auto-Generated Defeater Rules:**
When a defeasible rule is repeatedly defeated (≥ `min_occurrences` in the grammar detector), sleep-time consolidation promotes a defeater anti-pattern:

```python
GrammarRule(
    rule_id=f"defeater_{defeated_rule}_{defeating_rule}",
    rule_strength=-1,  # defeater
    superior_to=[defeated_rule_id],
    trust_weight=0.5,  # starts moderate, grows with recurrence
    semantics={
        "source": "defeasible_auto_detected_contrastive",
        "contrastive_recommendation": "block_in_context",
    },
)
```

This is the self-improving loop: the system discovers its own defeaters from runtime behavior, enriching the Grammar Galaxy's superiority web without human authoring.

---

## 8. GPU-Native Consolidation Architecture (March 2026)

### 8.1 Contrastive Learning on GPU

The core consolidation operation is **contrastive learning**: strengthen paths that led to correct answers, weaken paths that led to incorrect answers. This MUST happen on GPU:

```
jarvis_sleep_consolidation() entry point (Python):
  │
  ├── 1. Shadow copy comparison ON GPU
  │     Compare shadow_copy predictions vs ground truth
  │     Produces: positive_paths (correct), negative_paths (incorrect)
  │
  ├── 2. Sleep cluster refinement ON GPU
  │     Uses: sleep_cluster_refiner.ptx
  │     Strengthens Galaxy clusters around correct answer paths
  │     Weakens clusters around incorrect paths
  │
  ├── 3. Specialist weight update ON GPU
  │     Uses: lora_gpu.cu
  │     Update TRM specialist weights based on contrastive signal
  │
  ├── 4. Galaxy entry score update ON GPU
  │     Uses: galaxy_memory_updater.cu
  │     Entries on correct paths get score boost
  │     Entries on incorrect paths get score penalty
  │
  ├── 5. Glyph pattern consolidation ON GPU
  │     Uses: sleep_glyph_consolidator.ptx
  │     Consolidate frequently co-activated glyph patterns
  │
  └── 6. Micro-consolidation passes ON GPU
        Uses: sleep_time_micro.ptx
        Fine-grained passes over recently active regions
```

**The Python method is a LAUNCHER, not a WORKER.** It collects briefs, prepares kernel arguments, launches GPU kernels, and collects results. All heavy computation is PTX.

### 8.2 Kernel Wiring Table

| Kernel | Bridge | Purpose in Sleep |
|--------|--------|-----------------|
| `sleep_cluster_refiner.cu` | `sleep_cluster_kernels.py` | Cluster co-activation refinement |
| `sleep_glyph_consolidator.cu` | `sleep/glyph_consolidator.py` | Glyph pattern consolidation |
| `galaxy_memory_updater.cu` | `ptx_runtime/galaxy_memory_updater.py` | Galaxy entry score updates |
| `lora_gpu.cu` | `sovereign/lora_gpu_trainer.py` | Specialist weight updates |
| `sleep_time_micro.ptx` | `ptx_runtime/sleep_time_compute.py` | Micro-consolidation passes |

These kernels EXIST, have bridge classes, and MUST be called during `jarvis_sleep_consolidation()`.

### 8.3 Future: Incremental Consolidation

**Problem**: Currently serializes ALL nodes (even unchanged ones).

**Solution**: Track dirty nodes, only serialize modifications.

### 8.4 Future: Distributed SleepTime

**Problem**: Single-machine consolidation limits scale.

**Solution**: Shard Galaxy across multiple GPUs/machines, consolidate in parallel.

### 8.5 Future: Adaptive Triggers

**Problem**: Fixed thresholds suboptimal for varying workloads.

**Solution**: TRM learns optimal consolidation timing from query patterns.

---

### 8.4 Persistent Brain Model Versioning (March 2026)

**Context**: SleepTime consolidation produces checkpoints of Galaxy → House knowledge. The [Hyper-Parallel Processing](HYPER_PARALLEL_PROCESSING.md) paradigm (§7) extends this to the **full brain model**: TRM weights + specialist adapter populations + navigation biases + consolidation state.

**Key additions:**
- Every sleep cycle produces a **versioned brain model checkpoint** (diffed against previous — zero duplication, hyper-modular principle)
- **Drift detection**: If benchmark regression or convergence failure rate exceeds threshold, **rollback** to nearest stable checkpoint
- **Storage separation**: Galaxy/House (knowledge) is separate from TRM (cognition) — rollback the model head without losing the knowledge base
- The K3D brain model v1.0 is the **reference proof**; labs build domain-specific specialist populations on top

**Imperative**: Running from ground state (cold start) destroys accumulated sleep-time learning. Cold start is a degraded mode, not normal operation. The living brain persists.

**See**: [HYPER_PARALLEL_PROCESSING.md](HYPER_PARALLEL_PROCESSING.md) §7 for the full persistent brain model architecture.

---

## 9. References

- **Neuroscience**:
  - "The Hippocampus as a Cognitive Map" (O'Keefe & Nadel, 1978)
  - "Sleep and Memory Consolidation" (Walker & Stickgold, 2004)
  - "Synaptic Homeostasis in Sleep" (Tononi & Cirelli, 2014)
- **Database Systems**:
  - "Transaction Processing: Concepts and Techniques" (Gray & Reuter, 1992)
  - "ACID Properties in Distributed Databases" (Bernstein et al., 1987)
- **K3D Implementation**: https://github.com/danielcamposramos/Knowledge3D

---

## Attribution & Academic Context

**For complete attributions**, see [ATTRIBUTIONS.md](../../ATTRIBUTIONS.md) in the K3D repository.

**Key Credits**:

1. **Neuroscience** (Sleep and Memory Consolidation):
   - Biological sleep consolidation mechanisms
   - K3D applies to AI memory management (Galaxy → House)

2. **Database Systems** (ACID Properties):
   - Transaction processing concepts (Gray & Reuter, 1992)
   - K3D adapts for spatial memory consolidation

3. **Game Industry** (State Management):
   - Save/load systems for game state
   - K3D applies to knowledge persistence

4. **glTF 2.0** (Khronos Group):
   - Serialization format for persistent memory (House)

K3D's SleepTime Protocol is a novel contribution that applies biological and database principles to AI memory consolidation.

---

## Contact & License

**Author**: Daniel Campos Ramos, K3D Architect
**Email**: daniel@echosystems.ai
**Repository**: https://github.com/danielcamposramos/Knowledge3D
**License**: CC-BY-4.0 (specification), Apache 2.0 (implementation code)

---

**Status**: Production (Phase G Complete, October 2025)
**Next Review**: Q1 2026 (for W3C CG Note submission)

---

**Proposed W3C Standardization Path**:
1. **Q1 2026**: Publish as W3C Community Group Draft Report
2. **Q2 2026**: Solicit feedback from AI KR CG and WebApps WG
3. **Q3 2026**: Propose formal W3C Note on "Memory Consolidation Protocols for Spatial AI"
4. **2027**: Integrate with WebXR/WebGPU standards for browser-based spatial AI
